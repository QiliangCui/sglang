"""Reference SDPA attention backend for the S3.0.5 diff gate.

Uses `torch.nn.functional.scaled_dot_product_attention` to compute full
attention over each forward batch's (q, k, v) WITHOUT consulting a KV
cache. Single-pass only — not useful for multi-step generation; useful
as a known-good reference for:

  1. Validating sglang's Qwen3 forward produces correct numbers under
     torchax (compare CPU SDPA vs torchax SDPA).
  2. Generating per-layer activation references for S3.0.5 before
     swapping in the real JaxAttentionBackend in S3.2.

Inherits AttentionBackend but only implements the minimum surface the
ABC needs at S3.0.5 time:
  - init_forward_metadata: no-op (no KV cache, no graph).
  - forward_extend / forward_decode: same SDPA call; the dispatch on
    forward_mode is in the base class.
  - support_triton: False.

Single-host, single-seq batches only. Causal mask is forced on (LLM
prefill).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class SdpaSpikeAttnBackend(AttentionBackend):
    """Pure SDPA — no KV cache. Spike use only."""

    def __init__(self):
        super().__init__()
        self.forward_metadata = None
        # No pools — diff spike doesn't allocate KV.
        self.req_to_token_pool = None
        self.token_to_kv_pool = None

    def init_forward_metadata(self, forward_batch):
        return None

    def support_triton(self) -> bool:
        return False

    def _sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
    ) -> torch.Tensor:
        """Single-batch causal SDPA. Returns [num_tokens, n_q_heads * v_head_dim].

        Shape contract at backend entry:
          * q is FLAT [num_tokens, n_q_heads * qk_head_dim] — RadixAttention
            does NOT reshape q (only k, v at radix_attention.py:119-120).
          * k is [num_tokens, n_kv_heads, qk_head_dim].
          * v is [num_tokens, n_kv_heads, v_head_dim].
        SDPA wants [batch=1, n_heads, seq, head_dim] with matching head counts.
        """
        num_tokens = q.shape[0]
        n_q = layer.tp_q_head_num
        n_kv = layer.tp_k_head_num
        q = q.view(num_tokens, n_q, layer.qk_head_dim)

        # GQA: tile k,v so n_kv heads match n_q for SDPA.
        if n_q != n_kv:
            assert n_q % n_kv == 0, (n_q, n_kv)
            rep = n_q // n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # [num_tokens, n_q, head_dim] -> [1, n_q, num_tokens, head_dim]
        q4 = q.transpose(0, 1).unsqueeze(0)
        k4 = k.transpose(0, 1).unsqueeze(0)
        v4 = v.transpose(0, 1).unsqueeze(0)

        # Use the layer's scaling explicitly so we control determinism.
        scale = layer.scaling

        out4 = scaled_dot_product_attention(
            q4, k4, v4, scale=scale, is_causal=True
        )
        # [1, n_q, num_tokens, v_head_dim] -> [num_tokens, n_q * v_head_dim]
        out = out4.squeeze(0).transpose(0, 1).contiguous()
        return out.view(num_tokens, n_q * layer.v_head_dim)

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
        return self._sdpa(q, k, v, layer)

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
        # No KV cache: at decode time we only see the single new token.
        # SDPA-over-one-token is correct only for the spike use; real
        # generation needs the JaxAttentionBackend.
        return self._sdpa(q, k, v, layer)
