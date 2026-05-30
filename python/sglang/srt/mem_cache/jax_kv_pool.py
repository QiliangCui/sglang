"""JAX KV pools — minimal pass-through stubs for the MVP demo.

Per `~/private-tool/sglang/decisions/2026-05-29_kv-pool-in-runner-mvp.md`,
the real KV state is owned by `JaxStepRunner._kv_caches`. These pool
classes exist only so that sglang's `model_runner_kv_cache_mixin` and
`Scheduler` can construct/inspect a pool object without crashing at
`--disable-radix-cache --max-running-requests=1`.

Methods are intentionally minimal: enough surface for the Scheduler's
greedy single-sequence admission path; everything that would normally
read/write KV state is a no-op or a fixed value. Real KV reads/writes
go through `JaxStepRunner` → `JaxAttentionBackend` → tpu-inference RPA.

When S5 or RadixCache work picks up, replace these with proper pools
(plan §S3.3 Option A).
"""
from __future__ import annotations

import torch


class JaxMHATokenToKVPool:
    """MHA KV pool for the MVP demo.

    Owns config the Scheduler reads (`size`, `page_size`, `dtype`, etc.)
    plus a reference to the per-layer JAX KV cache arrays allocated by
    JaxStepRunner. The kv_caches themselves live in JAX device memory;
    JaxStepRunner mutates them per-step via the JIT donate/return pattern
    and updates the `jax_kv_caches` attribute below so anything reading
    via `model_runner.token_to_kv_pool` sees the freshest reference.

    S3.3 follow-up: convert the sglang flat `[max_total_tokens, n_heads,
    head_dim]` ABI on `get_key_buffer`/`set_kv_buffer` to/from the RPA-v3
    5-D paged layout. Today both pass-through to no-ops because the
    Scheduler at `--max-running-requests 1 --disable-radix-cache` never
    actually reads the buffers — the kernel writes/reads them in-place
    inside the JIT.
    """

    def __init__(
        self,
        size: int,
        page_size: int = 1,
        dtype=torch.bfloat16,
        head_num: int = 0,
        head_dim: int = 0,
        layer_num: int = 0,
        device: str = "jax",
        enable_memory_saver: bool = False,
        start_layer: int = 0,
        end_layer: int | None = None,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.device = device
        self.start_layer = start_layer
        self.end_layer = end_layer if end_layer is not None else layer_num
        # Some sglang call sites read .data_ptr() on internal buffers
        # (risk #38). Provide an empty CPU tensor for callers that just
        # want a stable object.
        self._k_buffer = [torch.empty(0) for _ in range(layer_num)]
        self._v_buffer = [torch.empty(0) for _ in range(layer_num)]
        # KV cache layout used by the RPA-v3 kernel. JaxStepRunner sets
        # these after _allocate_kv_caches and refreshes jax_kv_caches
        # after each step via the JIT donate/return pattern.
        # `jax_kv_caches`: list[jax.Array], shape
        #   (kv_num_pages, kv_page_size, n_kv, 2, head_dim) per layer.
        self.jax_kv_caches = None
        self.kv_num_pages = 0
        self.kv_page_size = page_size
        self.kv_pages_per_seq = 0

    # ---- common mixin API ------------------------------------------------
    @property
    def mem_usage(self) -> float:
        # GB used by jax_kv_caches across all layers. Probes (Scheduler
        # `get_internal_state`) call this on the bench / monitoring path.
        if self.jax_kv_caches is None:
            return 0.0
        try:
            per_layer_bytes = self.jax_kv_caches[0].nbytes
            return (per_layer_bytes * len(self.jax_kv_caches)) / (1024 ** 3)
        except Exception:
            return 0.0

    def get_kv_size_bytes(self) -> int:
        return 0

    def get_contiguous_buf_infos(self):
        return [], []

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self._k_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self._v_buffer[layer_id - self.start_layer]

    def get_kv_buffer(self, layer_id: int):
        return (self.get_key_buffer(layer_id), self.get_value_buffer(layer_id))

    def set_kv_buffer(self, layer, cache_loc, cache_k, cache_v, *args, **kwargs):
        # No-op: real KV state lives in JaxStepRunner._kv_caches via the
        # JaxAttentionBackend. The Scheduler may call this for bookkeeping
        # but the data flow on TPU bypasses the pool.
        return None

    def free(self, *args, **kwargs):
        return None

    def clear(self):
        return None

    # ---- required by some Scheduler probes -------------------------------
    @property
    def available_size(self) -> int:
        return self.size

    def get_num_token_used(self) -> int:
        return 0


class JaxMLATokenToKVPool:
    """MLA pool — declared but not in MVP scope (see kb §18.5, plan row 16)."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "JaxMLATokenToKVPool is post-MVP (deferred row 16). MLA model "
            "arch should have been rejected upfront in "
            "apply_server_args_defaults."
        )
