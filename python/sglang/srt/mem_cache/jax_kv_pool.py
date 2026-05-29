"""JAX KV pools (S3.3 placeholder).

Two classes declared from day one to satisfy the `model_runner_kv_cache_mixin`
dispatch table without circular imports:

  * JaxMHATokenToKVPool — MVP target for Qwen3-4B (dense decoder, MHA).
  * JaxMLATokenToKVPool — stub. MLA forward is deferred (plan §13 row 16,
    kb §18.5); the pool ABI must still exist so the mixin's "is this an
    MLA model?" branch can resolve to a class even if it's never
    instantiated in the MVP.

KV-layout strategy: kb_sglang_design.md §7.3 option A — sglang's flat
per-token layout stays in the allocator's CPU/host metadata; the
JaxStepRunner converts page-index slices into the RPA-v3 paged-block
layout at JIT entry. Implementation in S3.3.

Calling __init__ raises so instantiation pre-S3.3 surfaces loudly
instead of leaving a half-built pool around.
"""
from __future__ import annotations


class JaxMHATokenToKVPool:
    """MHA KV pool stored as a list of jax.Array per layer (S3.3 placeholder)."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "JaxMHATokenToKVPool is a S3.3 placeholder. Implement "
            "_create_buffers, set_kv_buffer, get_kv_buffer (override "
            "_create_buffers fully — see risk #38) before instantiating. "
            "See plan_sglang_on_tpu.md §S3.3."
        )


class JaxMLATokenToKVPool:
    """MLA KV pool (declared but deferred per kb §18.5, plan row 16)."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "JaxMLATokenToKVPool is post-MVP (deferred row 16). Forbidden "
            "by apply_server_args_defaults' multimodal/MLA arch check. "
            "If you hit this from a DeepSeek/MLA model, you forgot to "
            "reject the model arch upfront."
        )
