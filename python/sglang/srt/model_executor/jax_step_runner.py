"""JAX step runner — owns the main jax.jit(step_fun).

S3.4 first-pass. Wraps `torch.func.functional_call(model, params, kwargs,
tie_weights=False)` inside `torchax.default_env()` and runs it inside
`jax.jit(step_fun, donate_argnums=("kv_caches",))`. Mirrors tpu-inference's
`models/vllm/vllm_model_wrapper.py:277` (`jit_step_func`).

Validated by spike `spikes/s3_4_jit_probe.py` 2026-05-29 at full Qwen3-4B
(36 layers, 4 B params): compile + 1st call 6.28s, warm 4.3 ms.

NOT YET IMPLEMENTED (deferred to listed stages):
  * Shape bucketing (§S3.7). One trace per unique (num_tokens, num_reqs)
    pair today; production has many buckets.
  * Distribution-based extend/decode/mixed routing inside the JIT body.
    Currently we route by Python-side check on forward_batch.forward_mode
    and re-jit per mode → 2 traces, not 1.
  * Real KV pool integration (§S3.3). Currently the runner owns the
    `kv_caches` list as plain `list[jax.Array]`.
  * is_first_rank / is_last_rank static_argnames (PP support).
  * Sharding (§S5).

Hooked in by model_runner.forward_extend / forward_decode short-circuit
when self.device == "jax" (the in-tree device name post the
2026-05-30 rewrite). Lazily constructed on first call so non-TPU paths
don't import torchax.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class JaxStepRunner:
    """Single-JIT step runner."""

    def __init__(self, model_runner: "ModelRunner"):
        self.model_runner = model_runner
        self.model = model_runner.model
        # Cached JIT compiled step_fun per (forward_mode_str, num_tokens,
        # batch_size). Production needs proper bucketing; this is good
        # enough for S3.4 first-pass + spike runs.
        self._step_jits: dict = {}
        # Cached pytree of jax-view'd model params (built once).
        self._params_jax = None
        # Per-layer KV caches owned by the runner. Lifecycle: created on
        # first call; persists across steps; donated each call and
        # rebound from the returned new_kv list. Real KV pool integration
        # in §S3.3 takes over this lifecycle.
        self._kv_caches: Optional[list] = None
        # Attention backend (`JaxAttentionBackend`) — created lazily so we
        # don't pay the tpu_inference import cost unless a real step fires.
        self._attn_backend = None
        self._fwd_ctx = None

    # ------------------------------------------------------------------
    # Lazy setup
    # ------------------------------------------------------------------

    def _ensure_setup(self, forward_batch: "ForwardBatch") -> None:
        if self._params_jax is not None:
            return
        from torchax.interop import jax_view

        from sglang.srt.layers.attention.jax_backend import JaxAttentionBackend
        from sglang.srt.model_executor.forward_context import ForwardContext

        self._attn_backend = JaxAttentionBackend(model_runner=self.model_runner)
        self._fwd_ctx = ForwardContext(attn_backend=self._attn_backend)

        # Snapshot params as a name -> jax.Array pytree.
        # state_dict() walks all params + buffers under torchax env.
        sd = self.model.state_dict()
        self._params_jax = {k: jax_view(v) for k, v in sd.items()}
        logger.info("JaxStepRunner params snapshot: %d entries", len(sd))

        # Allocate KV caches sized to model_config.
        self._allocate_kv_caches()

    def _allocate_kv_caches(self) -> None:
        # Pull KV layout from the model config. S3.3 will move this into
        # JaxMHATokenToKVPool; here we own it directly.
        import jax.numpy as jnp

        from tpu_inference.kernels.ragged_paged_attention.v3.kernel import (
            get_kv_cache_shape,
        )

        cfg = self.model.config
        n_layers = cfg.num_hidden_layers
        n_kv = cfg.num_key_value_heads
        head_dim = getattr(cfg, "head_dim", None) or (
            cfg.hidden_size // cfg.num_attention_heads
        )

        # Conservative defaults; production needs --page-size tuning.
        num_pages = 256
        page_size = 16

        shape = get_kv_cache_shape(num_pages, page_size, n_kv, head_dim,
                                   jnp.bfloat16)
        self._kv_caches = [
            jnp.zeros(shape, dtype=jnp.bfloat16) for _ in range(n_layers)
        ]
        logger.info(
            "JaxStepRunner kv_caches: %d layers, per-layer shape %s",
            n_layers, shape,
        )

    # ------------------------------------------------------------------
    # Step entry
    # ------------------------------------------------------------------

    def step(self, forward_batch: "ForwardBatch") -> torch.Tensor:
        """Single entry point for decode + extend forward.

        Returns the model output (`LogitsProcessorOutput` for generation,
        embedding pooler output otherwise). Mutates internal kv_caches.
        """
        import jax
        from torchax.interop import jax_view, torch_view

        from sglang.srt.model_executor.forward_context import forward_context
        from sglang.srt.model_executor.jax_forward_context import (
            JaxForwardContext, set_jax_forward_context,
        )

        self._ensure_setup(forward_batch)

        # Build / fetch the JIT for this shape.
        mode = forward_batch.forward_mode
        mode_key = "extend" if mode.is_extend() else "decode"
        num_tokens = forward_batch.input_ids.shape[0]
        batch_size = forward_batch.batch_size
        cache_key = (mode_key, num_tokens, batch_size)

        if cache_key not in self._step_jits:
            self._step_jits[cache_key] = self._build_step_jit(
                forward_batch, cache_key
            )
        step_jit = self._step_jits[cache_key]

        # Pull inputs as jax arrays.
        ji = jax_view(forward_batch.input_ids)
        pj = jax_view(forward_batch.positions)

        logits_jax, new_kv = step_jit(
            self._params_jax, self._kv_caches, ji, pj
        )
        # Donation freed the prior list; rebind so next call sees the new state.
        self._kv_caches = list(new_kv)
        return torch_view(logits_jax)

    # ------------------------------------------------------------------
    # JIT compilation
    # ------------------------------------------------------------------

    def _build_step_jit(self, forward_batch: "ForwardBatch", cache_key):
        import jax

        # Closure captures: model + forward_batch + ctx. The ctx is rebuilt
        # inside the body so it lives in the traced graph.
        from torchax.interop import jax_view, torch_view

        from sglang.srt.model_executor.forward_context import forward_context
        from sglang.srt.model_executor.jax_forward_context import (
            JaxForwardContext, set_jax_forward_context,
        )

        model = self.model
        fb = forward_batch
        attn_mesh = self._attn_backend.mesh
        fwd_ctx = self._fwd_ctx

        def step_fun(params_jax, kv_caches, input_ids_jax, positions_jax):
            ctx = JaxForwardContext(kv_caches=list(kv_caches), mesh=attn_mesh)
            with set_jax_forward_context(ctx), forward_context(fwd_ctx):
                params_torch = {
                    k: torch_view(v) for k, v in params_jax.items()
                }
                kwargs = {
                    "input_ids": torch_view(input_ids_jax),
                    "positions": torch_view(positions_jax),
                    "forward_batch": fb,
                }
                out = torch.func.functional_call(
                    model,
                    params_torch,
                    args=(),
                    kwargs=kwargs,
                    strict=False,
                    tie_weights=False,
                )
            return jax_view(out.next_token_logits), ctx.kv_caches

        logger.info("JaxStepRunner compiling JIT for %s ...", cache_key)
        return jax.jit(step_fun, donate_argnums=(1,))
