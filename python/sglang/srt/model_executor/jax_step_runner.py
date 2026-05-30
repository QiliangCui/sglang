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

import bisect
import logging
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

# Token-padding buckets for extend mode. Mirrors tpu-inference's
# `runner.utils.get_token_paddings(min=16, max=...)` policy: powers of 2
# until 64, then 64-token gaps. Decode mode always uses num_tokens=1, so
# no bucket needed for it.
#
# Each unique bucket → one JIT compile (~6s for full Qwen3-4B). The list
# is intentionally short for MVP: small prompts trigger the lower
# buckets; production with longer contexts would extend up.
_EXTEND_TOKEN_BUCKETS: list[int] = [16, 32, 64, 128, 256, 512, 1024]


def _bucket_extend_tokens(n: int) -> int:
    """Round n up to the smallest bucket >= n. Raises if n is too big."""
    idx = bisect.bisect_left(_EXTEND_TOKEN_BUCKETS, n)
    if idx >= len(_EXTEND_TOKEN_BUCKETS):
        raise ValueError(
            f"extend num_tokens={n} exceeds max bucket "
            f"{_EXTEND_TOKEN_BUCKETS[-1]}; extend the bucket list."
        )
    return _EXTEND_TOKEN_BUCKETS[idx]


class JaxStepRunner:
    """Single-JIT step runner."""

    def __init__(self, model_runner: "ModelRunner"):
        self.model_runner = model_runner
        self.model = model_runner.model
        # Cached JIT compiled step_fun per (mode, bucket_num_tokens,
        # batch_size). After bucketing, all extend calls within the same
        # bucket reuse a single compile.
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
        # state_dict() walks all params + buffers; under torchax env so
        # the underlying jax arrays are reachable for jax_view.
        import torchax
        with torchax.default_env():
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

        Pads `input_ids` and `positions` up to the next bucket so the
        JIT cache is keyed by bucket size, not raw token count. The
        underlying `extend_seq_lens` / `seq_lens` keep the REAL token
        count — the RPA kernel uses those to mask out padded queries
        from KV cache writes and from attention output that gets
        sampled.

        Returns the model output (`LogitsProcessorOutput` for generation,
        embedding pooler output otherwise). Mutates internal kv_caches.
        """
        import jax
        import jax.numpy as jnp
        from torchax.interop import jax_view, torch_view

        self._ensure_setup(forward_batch)

        mode = forward_batch.forward_mode
        is_extend = mode.is_extend()
        mode_key = "extend" if is_extend else "decode"
        real_num_tokens = forward_batch.input_ids.shape[0]
        batch_size = forward_batch.batch_size

        # Bucket choice. Decode is always 1 token per seq → bucket = batch_size.
        if is_extend:
            bucket_num_tokens = _bucket_extend_tokens(real_num_tokens)
        else:
            bucket_num_tokens = real_num_tokens
        cache_key = (mode_key, bucket_num_tokens, batch_size)

        # Pad input_ids and positions to bucket_num_tokens, holding the
        # underlying forward_batch otherwise intact (extend_seq_lens etc.
        # still reflect real token counts → kernel writes only real K/V).
        # input_ids/positions arrive as plain CPU torch.Tensors (the schedule
        # path keeps them off-device on TPU). Lift to JAX via numpy bridge.
        import numpy as _np
        from torchax.tensor import Tensor as _TxTensor
        def _to_jax_i32(t):
            if isinstance(t, _TxTensor):
                return jax_view(t).astype(jnp.int32)
            return jnp.asarray(_np.asarray(t), dtype=jnp.int32)
        ji_jax = _to_jax_i32(forward_batch.input_ids)
        pj_jax = _to_jax_i32(forward_batch.positions)
        pad = bucket_num_tokens - real_num_tokens
        if pad > 0:
            ji_jax = jnp.pad(ji_jax, (0, pad))
            pj_jax = jnp.pad(pj_jax, (0, pad))

        if cache_key not in self._step_jits:
            self._step_jits[cache_key] = self._build_step_jit(
                forward_batch, cache_key
            )
        step_jit = self._step_jits[cache_key]

        logits_jax, new_kv = step_jit(
            self._params_jax, self._kv_caches, ji_jax, pj_jax
        )
        # Donation freed the prior list; rebind so next call sees the new state.
        self._kv_caches = list(new_kv)
        # Downstream sample() runs torch ops outside torchax.default_env, so
        # materialize logits as a plain CPU torch.Tensor.
        logits_np = _np.asarray(logits_jax)
        logits_cpu = torch.from_numpy(logits_np)
        # PROBE 0 / 2: per-step dump of input metadata + chosen token.
        import os as _probe_os
        if _probe_os.environ.get("SGLANG_PROBE0") or _probe_os.environ.get("SGLANG_PROBE2"):
            try:
                _top1 = int(logits_np.reshape(-1, logits_np.shape[-1])[-1].argmax())
                _fb = forward_batch
                _newkey = cache_key not in self._step_jits_keys_seen if hasattr(self, "_step_jits_keys_seen") else True
                if not hasattr(self, "_step_jits_keys_seen"):
                    self._step_jits_keys_seen = set()
                self._step_jits_keys_seen.add(cache_key)
                logger.warning(
                    "PROBE step mode=%s real_n=%s bucket_n=%s bs=%s "
                    "cache_key=%s jit_new=%s "
                    "input_ids=%s positions=%s seq_lens=%s "
                    "extend_seq_lens=%s extend_prefix_lens=%s out_cache_loc=%s "
                    "req_pool_indices=%s top1=%s",
                    mode_key, real_num_tokens, bucket_num_tokens, batch_size,
                    cache_key, _newkey,
                    _np.asarray(_fb.input_ids).tolist() if _fb.input_ids is not None else None,
                    _np.asarray(_fb.positions).tolist() if _fb.positions is not None else None,
                    _np.asarray(_fb.seq_lens).tolist() if _fb.seq_lens is not None else None,
                    _np.asarray(_fb.extend_seq_lens).tolist() if getattr(_fb, "extend_seq_lens", None) is not None else None,
                    _np.asarray(_fb.extend_prefix_lens).tolist() if getattr(_fb, "extend_prefix_lens", None) is not None else None,
                    _np.asarray(_fb.out_cache_loc).tolist() if getattr(_fb, "out_cache_loc", None) is not None else None,
                    _np.asarray(_fb.req_pool_indices).tolist() if getattr(_fb, "req_pool_indices", None) is not None else None,
                    _top1,
                )
            except Exception as _e_pr:
                logger.warning("PROBE dump failed: %s", _e_pr)
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        return LogitsProcessorOutput(
            next_token_logits=logits_cpu,
            hidden_states=None,
        )

    # ------------------------------------------------------------------
    # JIT compilation
    # ------------------------------------------------------------------

    def _build_step_jit(self, forward_batch: "ForwardBatch", cache_key):
        import jax
        import torchax

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
            with torchax.default_env(), set_jax_forward_context(ctx), forward_context(fwd_ctx):
                params_torch = {
                    k: torch_view(v) for k, v in params_jax.items()
                }
                # Mutate the closure-captured forward_batch so the attention
                # backend reads the same jax-staged input_ids / positions
                # the JIT trace was called with (not the CPU torch ones from
                # the time of compile).
                fb.input_ids = torch_view(input_ids_jax)
                fb.positions = torch_view(positions_jax)
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
