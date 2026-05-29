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
    """Minimal MHA KV pool for MVP demo.

    Stores config so the mixin / scheduler can read `size`, `page_size`,
    `dtype`, etc. Does NOT allocate any device buffers — those live in
    JaxStepRunner.
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

    # ---- common mixin API ------------------------------------------------
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
