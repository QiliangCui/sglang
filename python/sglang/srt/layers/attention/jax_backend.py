"""JAX attention backend — wraps tpu-inference RPA-v3 for sglang.

S3.2 first-pass implementation. NOT yet feature-complete; explicitly
out of scope at this stage:
  * shape bucketing (S3.7)
  * KV cache writes/reads against a real `JaxMHATokenToKVPool` (S3.3) —
    we operate on a single-call cache passed via JaxForwardContext
  * sharding beyond (1, 1) (S5)
  * MLA / DSA / mamba paths

What it DOES do (verified by spike `spikes/s3_2_rpa_smoke.py` 2026-05-29):

  * Imports `tpu_inference.layers.common.attention_interface.attention`
    via the `_vllm_shim` module so the vllm-touching tpu_inference
    helpers (logger, envs, utils) load without pulling real vllm.
  * Pins `ShardingAxisName` to the 2D flavor via env-default fallback
    (matches plan §S3.6 requirement).
  * Builds a single-device Mesh on the first `__init__` and caches it
    as a class attribute so subsequent layers reuse it (avoids
    per-layer rebuild cost; the mesh itself is light, but recreating
    it would force JIT cache invalidations).
  * Translates sglang's `ForwardBatch.extend_seq_lens / seq_lens /
    extend_prefix_lens` into the RPA-v3 `AttentionMetadata`:
        - input_positions  = `positions` field
        - seq_lens         = `seq_lens + extend_prefix_lens`-equivalent
        - query_start_loc  = `cumsum(extend_seq_lens)` (padded with 0)
        - request_distribution = derived from forward_mode
        - block_tables = (placeholder; S3.3 supplies real one from
          JaxMHATokenToKVPool's allocator)
  * Calls `attention(kv_cache, q, k, v, md, mesh, sm_scale=layer.scaling)`
    and returns `[num_tokens, n_q_heads * v_head_dim]` — the shape
    sglang's RadixAttention.forward expects from a backend.

Until S3.3 wires a real KV pool, instances of this backend expect their
caller (the spike or future JaxStepRunner) to supply a kv_cache via
JaxForwardContext.kv_caches[layer_index]. If the context is missing or
has no kv_caches for the layer, forward_extend / forward_decode raise.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

# Install vllm shim BEFORE the first tpu_inference import below.
from sglang.srt.layers.attention import _vllm_shim  # noqa: F401

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.jax_forward_context import (
    get_jax_forward_context,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def _build_default_mesh() -> Mesh:
    """Single-device mesh (ATTN_DATA=1, ATTN_HEAD=1).

    S5 will replace this with a multi-chip mesh constructed against the
    server_args.tp_size etc. (`sglang_tpu.mesh.get_or_create_mesh` in
    plan §S3.6).
    """
    # Import inside the function so tpu_inference loads lazily — keeps
    # the cost off the sglang `import` cold path for non-TPU runs.
    from tpu_inference.layers.common.sharding import ShardingAxisName2D

    devs = np.array(jax.devices()[:1]).reshape(1, 1)
    return Mesh(
        devs,
        axis_names=(ShardingAxisName2D.ATTN_DATA, ShardingAxisName2D.ATTN_HEAD),
    )


class JaxAttentionBackend(AttentionBackend):
    """JAX/Pallas attention backend (S3.2 first-pass)."""

    # 14-method base contract: support_triton must be False on TPU.
    _mesh: Optional[Mesh] = None  # class-level singleton

    def __init__(self, model_runner=None, **_):
        super().__init__()
        # model_runner is optional in S3.2 — JaxStepRunner will pass it
        # in S3.4 so we can read tp_size etc. For now the spike passes
        # None and we fall back to single-device.
        self._model_runner = model_runner
        self.forward_metadata = None
        # Pools — populated by S3.3 wiring. None for now; the spike
        # supplies kv_cache through JaxForwardContext.
        self.req_to_token_pool = None
        self.token_to_kv_pool = None
        # JaxStepRunner sets _kv_pages_per_seq right after allocating the
        # kv_caches so _build_metadata can build a correctly-sized
        # block_tables. Default for the spike code path.
        self._kv_pages_per_seq = 4
        self._kv_page_size = 16

        # Build mesh once per process.
        if type(self)._mesh is None:
            type(self)._mesh = _build_default_mesh()
        self.mesh = type(self)._mesh

    @property
    def mesh_obj(self) -> Mesh:
        return self.mesh

    # ------------------------------------------------------------------
    # AttentionBackend contract
    # ------------------------------------------------------------------

    def init_forward_metadata(self, forward_batch: "ForwardBatch"):
        """Stash a cached AttentionMetadata for the upcoming forward.

        S3.2 builds it lazily inside _forward (the metadata depends on
        layer-specific KV layout). Override here is a no-op so the
        ModelRunner's bookkeeping is happy.
        """
        self.forward_metadata = None

    def support_triton(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_metadata(self, forward_batch: "ForwardBatch"):
        """sglang ForwardBatch -> tpu_inference AttentionMetadata.

        S3.2 simplifications:
          * No bucketing / padding (S3.7).
          * block_tables is a single page covering the whole batch — we
            don't have a real KV pool wired yet; S3.3 supplies a real one.
          * request_distribution: all-prefill -> (0, 1, 1); all-decode ->
            (1, 1, 1). Mixed batches will land later.
        """
        from tpu_inference.layers.common.attention_metadata import (
            AttentionMetadata,
        )

        positions = forward_batch.positions
        num_tokens = positions.shape[0]
        batch_size = forward_batch.batch_size

        # Translate ForwardBatch's torch tensors to JAX. The contract is
        # that callers (JaxStepRunner or the spike) have already moved the
        # tensors to the JAX device — but they may still be torchax-wrapped
        # torch tensors. Use the underlying jax array directly when so.
        def _to_jax(t):
            if t is None:
                return None
            # torchax: torch.Tensor whose data is jax.Array.
            if hasattr(t, "jax"):
                return t.jax()
            # plain torch.Tensor on cpu -> jnp.asarray (host->device).
            return jnp.asarray(t.numpy() if t.device.type == "cpu" else t)

        # input_positions
        input_positions = _to_jax(positions).astype(jnp.int32)

        # seq_lens — kernel contract is total tokens per seq AFTER this
        # step (kv_seq_len). For pure prefill with empty cache the kernel
        # uses query_start_loc to drive the causal attention; seq_lens
        # tells it the total context length each query attends to. So
        # seq_lens = extend_prefix_lens + extend_seq_lens for extend mode.
        # That equals forward_batch.seq_lens (which already includes prefix
        # + extend), so the original code was right. Leaving the explicit
        # construction here in case we need to deviate.
        seq_lens_jax = _to_jax(forward_batch.seq_lens).astype(jnp.int32)
        # Pad to batch_size if needed (no bucketing yet).
        if seq_lens_jax.shape[0] < batch_size:
            seq_lens_jax = jnp.pad(
                seq_lens_jax,
                (0, batch_size - seq_lens_jax.shape[0]),
            )

        # query_start_loc: cumsum of extend_seq_lens for extend, or
        # arange-like for decode.
        # request_distribution: per tpu_inference.runner.tpu_runner:1782-1785
        # the format is [num_decode, num_decode, num_total_reqs].
        # Decode seqs are those with num_scheduled_tokens == 1; everything
        # else (including extend / chunked-prefill) is "non-decode".
        # Plan §13 / kb-tpu §3.3 named this triple (decode_end, prefill_end,
        # mixed_end) — that's correct semantically, but prefill_end ==
        # decode_end in tpu-inference's runner because the kernel treats
        # any non-decode seq as "extend-style".
        if forward_batch.forward_mode.is_extend():
            ext = _to_jax(forward_batch.extend_seq_lens).astype(jnp.int32)
            qsl = jnp.concatenate(
                [jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(ext)]
            )
            # Pad to batch_size + 1.
            if qsl.shape[0] < batch_size + 1:
                qsl = jnp.pad(qsl, (0, batch_size + 1 - qsl.shape[0]),
                              constant_values=int(qsl[-1]))
            distribution = jnp.array([0, 0, batch_size], dtype=jnp.int32)
        else:
            # decode: each seq contributes exactly 1 token
            qsl = jnp.arange(batch_size + 1, dtype=jnp.int32)
            distribution = jnp.array(
                [batch_size, batch_size, batch_size], dtype=jnp.int32
            )

        # block_tables: S3.3 MVP — each request owns physical pages
        # [0, 1, 2, ..., pages_per_seq-1]. Position p of request r maps to
        # physical slot p (page p//page_size, offset p%page_size) — exactly
        # what the kernel expects when it does
        #   slot = block_tables[r][p // page_size] * page_size + (p % page_size).
        # At --max-running-requests 1 every prefill overwrites slots 0..N
        # before reading them, so the prior request's residue is harmless.
        # S3.3 followups: real `JaxMHATokenToKVPool` ABI + per-request
        # page allocation for max_running_requests>1.
        pages_per_seq = self._kv_pages_per_seq
        block_tables = jnp.tile(
            jnp.arange(pages_per_seq, dtype=jnp.int32), batch_size
        )

        # PROBE 3: dump the metadata the attention kernel will see.
        import os as _probe_os3
        import logging as _logging3
        if _probe_os3.environ.get("SGLANG_PROBE3"):
            _l3 = _logging3.getLogger(__name__)
            try:
                _l3.warning(
                    "PROBE3 _build_metadata: positions=%s seq_lens=%s qsl=%s "
                    "block_tables=%s distribution=%s padded_num_reqs=%s "
                    "fb.out_cache_loc=%s fb.req_pool_indices=%s "
                    "fb.seq_lens=%s fb.extend_seq_lens=%s",
                    np.asarray(input_positions).tolist(),
                    np.asarray(seq_lens_jax).tolist(),
                    np.asarray(qsl).tolist(),
                    np.asarray(block_tables).tolist(),
                    np.asarray(distribution).tolist(),
                    batch_size,
                    np.asarray(forward_batch.out_cache_loc).tolist() if getattr(forward_batch, "out_cache_loc", None) is not None else None,
                    np.asarray(forward_batch.req_pool_indices).tolist() if getattr(forward_batch, "req_pool_indices", None) is not None else None,
                    np.asarray(forward_batch.seq_lens).tolist() if forward_batch.seq_lens is not None else None,
                    np.asarray(forward_batch.extend_seq_lens).tolist() if getattr(forward_batch, "extend_seq_lens", None) is not None else None,
                )
            except Exception as _e3:
                _l3.warning("PROBE3 dump failed: %s", _e3)
        return AttentionMetadata(
            input_positions=input_positions,
            block_tables=block_tables,
            seq_lens=seq_lens_jax,
            query_start_loc=qsl,
            request_distribution=distribution,
            padded_num_reqs=batch_size,
        )

    def _forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
    ) -> torch.Tensor:
        from tpu_inference.layers.common.attention_interface import attention
        from torchax.interop import jax_view, torch_view

        # Shape contract (mirrors SdpaSpikeAttnBackend):
        #   q FLAT [num_tokens, n_q * qk_head_dim], k/v reshaped already.
        num_tokens = q.shape[0]
        n_q = layer.tp_q_head_num
        n_kv = layer.tp_k_head_num
        q = q.view(num_tokens, n_q, layer.qk_head_dim)
        # k, v come in already reshaped to [num_tokens, n_kv_heads, head_dim].

        # Pull KV cache for this layer out of the per-step forward context.
        ctx = get_jax_forward_context()
        if ctx is None or not ctx.kv_caches:
            raise RuntimeError(
                "JaxAttentionBackend requires a JaxForwardContext with "
                "kv_caches populated before forward. "
                "JaxStepRunner (S3.4) wires this; spikes must construct "
                "it manually."
            )
        layer_idx = layer.layer_id
        if layer_idx >= len(ctx.kv_caches):
            raise RuntimeError(
                f"JaxForwardContext.kv_caches has {len(ctx.kv_caches)} "
                f"entries, but layer_id={layer_idx} requested."
            )
        kv_cache = ctx.kv_caches[layer_idx]

        md = self._build_metadata(forward_batch)

        # jax_view: torch.Tensor (torchax) -> jax.Array (zero-copy).
        q_jax = jax_view(q)
        k_jax = jax_view(k)
        v_jax = jax_view(v)

        with self.mesh:
            new_kv, output = attention(
                kv_cache=kv_cache,
                q=q_jax,
                k=k_jax,
                v=v_jax,
                attention_metadata=md,
                mesh=self.mesh,
                sm_scale=layer.scaling,
            )

        # Mutate the KV cache list in-place — that's how the wrapper
        # context propagates the updated cache to the next layer.
        ctx.kv_caches[layer_idx] = new_kv

        # output: [num_tokens, n_q, v_head_dim] -> [num_tokens, n_q * v_head_dim]
        out_torch = torch_view(output)
        return out_torch.reshape(num_tokens, n_q * layer.v_head_dim)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        return self._forward(q, k, v, layer, forward_batch)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        return self._forward(q, k, v, layer, forward_batch)
