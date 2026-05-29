"""JAX attention backend (S3.2 placeholder).

Implements the 14-method `base_attn_backend.AttentionBackend` interface
on TPU by wrapping `tpu_inference.layers.common.attention_interface.attention`
(Pallas ragged-paged-attention v3) per kb_sglang_design.md §0.5 reuse
policy.

Real implementation lands in S3.2 (plan §13 row 6). For now this module
just declares the class so:
  - `TpuSRTPlatform.init_backend()` can register it in ATTENTION_BACKENDS
    without ImportError;
  - in-tree dispatch sites can `from sglang.srt.layers.attention.jax_backend
    import JaxAttentionBackend` without breaking the import surface.

Calling __init__ on the placeholder raises so we notice if something
instantiates it before S3.2 lands the real body.
"""
from __future__ import annotations

# The base class exists upstream; we subclass without overriding methods.
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend


class JaxAttentionBackend(AttentionBackend):
    """JAX/Pallas attention backend (S3.2 placeholder)."""

    support_triton = False

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "JaxAttentionBackend is a S3.2 placeholder. Implement "
            "init_forward_metadata, forward_extend, forward_decode, and "
            "the 11 other base_attn_backend methods using "
            "tpu_inference.layers.common.attention_interface.attention "
            "before instantiating. See plan_sglang_on_tpu.md §S3.2."
        )
