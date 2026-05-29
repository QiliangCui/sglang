"""Per-step context for the JAX/torchax forward path.

Mirrors `tpu_inference.models.vllm.vllm_model_wrapper_context` (kb-tpu §3.2):
a **module-level singleton** (not threading.local — workers are single-
threaded) holding the KV cache list, mesh, and layer→cache-index map for
the in-flight JIT step. Custom layers read it inside the traced body
without changing the model's PyTorch forward signature.

Implemented as a dataclass + context manager:

    with set_jax_forward_context(kv_caches=..., mesh=..., ...):
        out = step_fun(...)
    # context is cleared on exit; KV caches mutated in-place by attention.

Real implementation lands in S3 (plan §13 row 5).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # avoid jax import at module-load time
    import jax
    from jax.sharding import Mesh


@dataclass
class JaxForwardContext:
    """Per-step state threaded through the JIT body via module global."""

    kv_caches: list = field(default_factory=list)  # list[jax.Array]
    mesh: Optional["Mesh"] = None
    layer_name_to_kvcache_index: dict[str, int] = field(default_factory=dict)
    # Filled by S3.0 (eager dry-run) / S3.6 (sharding).


_current: Optional[JaxForwardContext] = None


def get_jax_forward_context() -> Optional[JaxForwardContext]:
    """Return the active context or None if no JIT step is in flight."""
    return _current


@contextmanager
def set_jax_forward_context(ctx: JaxForwardContext):
    """Install `ctx` as the active forward context for the body of the
    `with` block. Restores the previous context on exit (typically None).
    """
    global _current
    prev = _current
    _current = ctx
    try:
        yield ctx
    finally:
        _current = prev
