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

        # Real-sharding step 2: pre-shard qkv_proj.weight on every attention
        # block. MUST happen here (outside any JIT trace) so the sharded
        # outputs aren't traced values escaping the step_fun scope. The
        # patched Qwen3Attention.forward_prepare_native picks up the
        # pre-sharded weights via module attributes. At TP=1 this is a no-op
        # (the helper short-circuits on `is_sharding_active`).
        try:
            from sglang.srt.layers.jax_sharding_helpers import (
                pre_shard_qkv_weights,
            )
            _n_qkv = pre_shard_qkv_weights(self.model, self._attn_backend.mesh)
            if _n_qkv > 0:
                logger.info(
                    "JaxStepRunner pre-sharded %d qkv_proj weights along ATTN_HEAD.",
                    _n_qkv,
                )
        except Exception as _e_shard:
            logger.warning(
                "JaxStepRunner pre_shard_qkv_weights failed: %s "
                "(continuing with replicated qkv)",
                _e_shard,
            )

        # Real-sharding step 4 + step 3: shard the two row-parallel linears
        # together (o_proj from attention, down_proj from mlp). down_proj is
        # paired with gate_up below — sharding only down_proj while gate_up
        # is still replicated would add a collective without compute savings
        # (per step-4 analysis). Sharding both lets gate_up's sharded output
        # flow directly into down_proj's sharded input with no intermediate
        # all-gather.
        try:
            from sglang.srt.layers.jax_sharding_helpers import (
                pre_shard_row_parallel_weights,
            )
            _n_rp = pre_shard_row_parallel_weights(
                self.model,
                self._attn_backend.mesh,
                name_filter={"o_proj", "down_proj"},
            )
            if _n_rp > 0:
                logger.info(
                    "JaxStepRunner pre-sharded %d row-parallel weights "
                    "(o_proj + down_proj, col dim) along ATTN_HEAD.",
                    _n_rp,
                )
        except Exception as _e_shard_rp:
            logger.warning(
                "JaxStepRunner pre_shard_row_parallel_weights failed: %s "
                "(continuing with replicated row-parallel weights)",
                _e_shard_rp,
            )

        # Real-sharding step 3: pre-shard gate_up_proj (column-parallel,
        # combined gate||up layout). Each device ends up owning its
        # intermediate-dim slice of the SwiGLU pre-activation. Paired with
        # down_proj's row-parallel shard above so the sharded intermediate
        # flows in without an intermediate all-gather.
        try:
            from sglang.srt.layers.jax_sharding_helpers import (
                pre_shard_gate_up_weights,
            )
            _n_gu = pre_shard_gate_up_weights(
                self.model, self._attn_backend.mesh
            )
            if _n_gu > 0:
                logger.info(
                    "JaxStepRunner pre-sharded %d gate_up_proj weights "
                    "(col-parallel, intermediate dim) along ATTN_HEAD.",
                    _n_gu,
                )
        except Exception as _e_shard_gu:
            logger.warning(
                "JaxStepRunner pre_shard_gate_up_weights failed: %s "
                "(continuing with replicated gate_up_proj)",
                _e_shard_gu,
            )

        # Real-sharding step 6: log per-device memory after pre-shard so
        # the structural memory win is measurable. `memory_stats()` returns
        # None on non-TPU platforms, in which case we skip silently.
        try:
            import jax as _jax_mem
            for _i, _dev in enumerate(_jax_mem.devices()):
                _stats = _dev.memory_stats()
                if _stats is None:
                    continue
                _in_use_gb = _stats.get("bytes_in_use", 0) / (1024 ** 3)
                _peak_gb = _stats.get("peak_bytes_in_use", 0) / (1024 ** 3)
                _limit_gb = _stats.get("bytes_reservable_limit", 0) / (1024 ** 3)
                logger.info(
                    "POST_SHARD_MEM dev=%d in_use=%.3f GB peak=%.3f GB limit=%.3f GB",
                    _i, _in_use_gb, _peak_gb, _limit_gb,
                )
        except Exception as _e_mem:
            logger.warning("POST_SHARD_MEM probe failed: %s", _e_mem)

    def _allocate_kv_caches(self) -> None:
        # Pull KV layout from the model config. S3.3 will move this into
        # JaxMHATokenToKVPool; here we own it directly.
        import jax
        import jax.numpy as jnp
        from jax.sharding import NamedSharding, PartitionSpec as P

        from tpu_inference.kernels.ragged_paged_attention.v3.kernel import (
            get_kv_cache_shape,
        )
        from tpu_inference.layers.common.sharding import ShardingAxisName2D

        cfg = self.model.config
        n_layers = cfg.num_hidden_layers
        n_kv = cfg.num_key_value_heads
        head_dim = getattr(cfg, "head_dim", None) or (
            cfg.hidden_size // cfg.num_attention_heads
        )

        # Conservative defaults; production needs --page-size tuning.
        # S3.3 (this round): each single-request session "owns" pages
        # [0, 1, 2, ...] — block_tables in jax_backend is now arange-per-
        # request, so position p maps to physical slot p (page p//page_size,
        # offset p%page_size). Wrap-around no longer corrupts the prefix.
        num_pages = 256
        page_size = 16

        shape = get_kv_cache_shape(num_pages, page_size, n_kv, head_dim,
                                   jnp.bfloat16)
        # Real-sharding step 1: pin per-layer KV onto the same mesh the
        # RPA kernel uses, sharded along the head axis. At TP=1 the mesh
        # is 1x1 and this is a no-op; at TP>1 the per-decode-step
        # redistribute that drove the 11x/27x/51x cliff goes away. The
        # kernel's kv_cache_spec is P(ATTN_DATA, None, ATTN_HEAD, None,
        # None); ATTN_DATA is size 1 in our mesh shape so None vs
        # ATTN_DATA on dim 0 is equivalent here.
        mesh = self._attn_backend.mesh
        kv_spec = NamedSharding(
            mesh,
            P(None, None, ShardingAxisName2D.ATTN_HEAD, None, None),
        )
        self._kv_caches = [
            jax.device_put(jnp.zeros(shape, dtype=jnp.bfloat16), kv_spec)
            for _ in range(n_layers)
        ]
        # Hand pages_per_seq + page_size to the attention backend so its
        # _build_metadata builds a block_tables shaped/valued correctly.
        # At --max-running-requests 1 every request owns the entire pool.
        self._attn_backend._kv_pages_per_seq = num_pages
        self._attn_backend._kv_page_size = page_size
        # Also publish on the model_runner's KV pool so anything reading
        # via `model_runner.token_to_kv_pool` sees the layout. The pool's
        # ABI surface (`get_key_buffer`, `set_kv_buffer`, etc.) still
        # passes through to no-ops at single-request MVP.
        _pool = getattr(self.model_runner, "token_to_kv_pool", None)
        if _pool is not None and hasattr(_pool, "jax_kv_caches"):
            _pool.jax_kv_caches = self._kv_caches
            _pool.kv_num_pages = num_pages
            _pool.kv_page_size = page_size
            _pool.kv_pages_per_seq = num_pages
        logger.info(
            "JaxStepRunner kv_caches: %d layers, per-layer shape %s, "
            "pages_per_seq=%d page_size=%d",
            n_layers, shape, num_pages, page_size,
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

        Per-call dynamic forward_batch fields (extend_seq_lens, seq_lens,
        positions, etc.) are passed as **explicit JIT arguments** so JAX
        treats them as runtime inputs instead of trace-time constants.
        Earlier versions captured them in the step_fun closure, which
        baked the warmup's metadata into the JIT trace and produced
        wrong tokens on every subsequent request (see probes 0/2/3
        report 2026-05-30 19:25 UTC).

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

        # Materialize every per-call dynamic field as a JAX array, padding
        # token-axis fields up to bucket_num_tokens. These become explicit
        # JIT args — never closure-captured.
        import numpy as _np
        from torchax.tensor import Tensor as _TxTensor

        def _to_jax_i32(t, target_len=None):
            if isinstance(t, _TxTensor):
                arr = jax_view(t).astype(jnp.int32)
            else:
                arr = jnp.asarray(_np.asarray(t), dtype=jnp.int32)
            if target_len is not None and arr.shape[0] < target_len:
                arr = jnp.pad(arr, (0, target_len - arr.shape[0]))
            return arr

        ji_jax = _to_jax_i32(forward_batch.input_ids, bucket_num_tokens)
        pj_jax = _to_jax_i32(forward_batch.positions, bucket_num_tokens)
        sl_jax = _to_jax_i32(forward_batch.seq_lens)
        # out_cache_loc: extend uses sum-of-extend_seq_lens elements,
        # decode uses batch_size elements. Pad to bucket_num_tokens so the
        # JIT signature is shape-stable per cache_key.
        ocl_jax = _to_jax_i32(forward_batch.out_cache_loc, bucket_num_tokens)
        rpi_jax = _to_jax_i32(forward_batch.req_pool_indices)
        # extend_seq_lens / extend_prefix_lens: real only in extend mode.
        # For decode pass zeros of the right shape so the JIT signature
        # is identical across all decode calls — the closure-captured
        # `is_extend` flag in step_fun controls whether they get used.
        if is_extend:
            esl_jax = _to_jax_i32(forward_batch.extend_seq_lens)
            epl_jax = _to_jax_i32(forward_batch.extend_prefix_lens)
        else:
            esl_jax = jnp.zeros((batch_size,), dtype=jnp.int32)
            epl_jax = jnp.zeros((batch_size,), dtype=jnp.int32)

        if cache_key not in self._step_jits:
            self._step_jits[cache_key] = self._build_step_jit(
                forward_batch, cache_key, is_extend
            )
        step_jit = self._step_jits[cache_key]

        # step_fun mutates fb fields in-place so the model + attention
        # backend see the JIT-traced runtime arrays. Snapshot the
        # originals here and restore after so downstream code (sampler,
        # etc.) runs against the plain CPU torch tensors again.
        _saved = {
            "input_ids": forward_batch.input_ids,
            "positions": forward_batch.positions,
            "seq_lens": forward_batch.seq_lens,
            "out_cache_loc": forward_batch.out_cache_loc,
            "req_pool_indices": forward_batch.req_pool_indices,
            "extend_seq_lens": getattr(forward_batch, "extend_seq_lens", None),
            "extend_prefix_lens": getattr(forward_batch, "extend_prefix_lens", None),
        }
        try:
            logits_jax, new_kv = step_jit(
                self._params_jax, self._kv_caches,
                ji_jax, pj_jax, sl_jax, esl_jax, epl_jax, ocl_jax, rpi_jax,
            )
        finally:
            for _k, _v in _saved.items():
                setattr(forward_batch, _k, _v)
        # Donation freed the prior list; rebind so next call sees the new state.
        self._kv_caches = list(new_kv)
        # Refresh the pool's mirror so downstream readers via
        # model_runner.token_to_kv_pool see the freshest references.
        _pool = getattr(self.model_runner, "token_to_kv_pool", None)
        if _pool is not None and hasattr(_pool, "jax_kv_caches"):
            _pool.jax_kv_caches = self._kv_caches
        # Downstream sample() runs torch ops outside torchax.default_env, so
        # materialize logits as a plain CPU torch.Tensor.
        logits_np = _np.asarray(logits_jax)
        logits_cpu = torch.from_numpy(logits_np)
        # V2 LOGIT-DIFF probe: dump top-10 every step. Gate behind env var
        # so production / regression runs stay quiet.
        import os as _v2_os
        if _v2_os.environ.get("SGLANG_DUMP_TOP10"):
            try:
                _row = logits_np.reshape(-1, logits_np.shape[-1])[-1]
                _top = (-_row).argsort()[:10]
                _pairs = [(int(i), float(_row[i])) for i in _top]
                logger.warning(
                    "DUMP_TOP10 mode=%s positions=%s bs=%s top10=%s",
                    mode_key,
                    forward_batch.positions.tolist() if hasattr(forward_batch.positions, "tolist") else None,
                    batch_size, _pairs,
                )
            except Exception as _e_t10:
                logger.warning("DUMP_TOP10 failed: %s", _e_t10)
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

    def _build_step_jit(self, forward_batch: "ForwardBatch", cache_key,
                        is_extend: bool):
        import jax
        import torchax

        # Closure captures (these are static per cache_key):
        #   model, fb (mutable container), attn_mesh, fwd_ctx, is_extend
        # Per-call dynamic state is passed as explicit JIT args below — NEVER
        # closure-captured — so JAX traces them as runtime inputs, not as
        # trace-time constants. See probes 0/2/3 report in private-tool
        # sglang/_next_prompt.md 2026-05-30 19:25 UTC for why.
        from torchax.interop import jax_view, torch_view

        from sglang.srt.model_executor.forward_context import forward_context
        from sglang.srt.model_executor.jax_forward_context import (
            JaxForwardContext, set_jax_forward_context,
        )

        model = self.model
        fb = forward_batch
        attn_mesh = self._attn_backend.mesh
        fwd_ctx = self._fwd_ctx

        def step_fun(params_jax, kv_caches,
                     input_ids_jax, positions_jax,
                     seq_lens_jax, extend_seq_lens_jax, extend_prefix_lens_jax,
                     out_cache_loc_jax, req_pool_indices_jax):
            ctx = JaxForwardContext(kv_caches=list(kv_caches), mesh=attn_mesh)
            with torchax.default_env(), set_jax_forward_context(ctx), forward_context(fwd_ctx):
                params_torch = {
                    k: torch_view(v) for k, v in params_jax.items()
                }
                # Mutate fb so the model's attention backend and the
                # logits_processor see the JIT-traced runtime arrays, not
                # the values from JIT compile time. These mutations are
                # safe because each call into the JIT wraps fresh JAX
                # tracers (or fresh runtime arrays) into torchax tensors.
                fb.input_ids = torch_view(input_ids_jax)
                fb.positions = torch_view(positions_jax)
                fb.seq_lens = torch_view(seq_lens_jax)
                fb.out_cache_loc = torch_view(out_cache_loc_jax)
                fb.req_pool_indices = torch_view(req_pool_indices_jax)
                if is_extend:
                    fb.extend_seq_lens = torch_view(extend_seq_lens_jax)
                    fb.extend_prefix_lens = torch_view(extend_prefix_lens_jax)
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
