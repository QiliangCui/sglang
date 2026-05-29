"""TPU defaults + forbidden-flag rejection.

Called from `TpuSRTPlatform.apply_server_args_defaults`, which is invoked
unconditionally from `ServerArgs.__post_init__` (`server_args.py:916`).

Forbidden-flag rules mirror kb_sglang_design.md §13 row 3.5. Two action
types:
  * "raise" — user explicitly opted in to a path the TPU MVP can't honor
    (PP, DP, DP-attention, two-batch overlap, disagg, memory-saver,
    deterministic-inference, fp8/fp4 KV cache, multimodal arch).
  * "override-with-log" — user left a default that conflicts with the TPU
    runtime (auto kv-cache-dtype, mismatched prefill/decode backends,
    flashinfer sampling backend, default 300 s watchdog, fp16/fp32 dtype).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


_FORBIDDEN_KV_CACHE_DTYPES = {"fp8_e5m2", "fp8_e4m3", "fp4_e2m1"}


def apply(server_args) -> None:
    """Mutate `server_args` in place: enforce TPU-required defaults and
    raise ValueError on flag combinations the MVP can't support.

    Reasoning for each branch points to kb_sglang_design.md §19 (risk
    register) and §13 row 3.5 (forbidden-flags table). Do not relax
    without updating both.
    """
    # --- raise-if-set: parallelism + advanced execution ----------------
    if server_args.pp_size != 1:
        raise ValueError(
            "TPU MVP supports pp_size=1 only (kb §13 row 3.5, risk #24). "
            "Pipeline parallelism requires PPProxyTensors as a JAX pytree "
            "and world_size>1 — not in scope."
        )

    if server_args.dp_size != 1:
        raise ValueError(
            "TPU MVP supports dp_size=1 only (kb §13 row 3.5). DP requires "
            "NCCL ranks; TPU MVP uses SPMD inside jax.jit."
        )

    if server_args.attn_cp_size != 1:
        raise ValueError(
            "TPU MVP supports attn_cp_size=1 only (kb §13 row 3.5). "
            "Context-parallel attention is not in scope."
        )

    if getattr(server_args, "enable_dp_attention", False):
        raise ValueError(
            "TPU MVP does not support --enable-dp-attention "
            "(kb §13 row 3.5). DP attention requires NCCL across ranks."
        )

    if getattr(server_args, "enable_two_batch_overlap", False):
        raise ValueError(
            "TPU MVP does not support --enable-two-batch-overlap "
            "(kb §13 row 3.5). TBO uses CUDA streams."
        )

    disagg = getattr(server_args, "disaggregation_mode", "null")
    if disagg not in (None, "null"):
        raise ValueError(
            f"TPU MVP does not support --disaggregation-mode={disagg!r} "
            "(kb §13 row 3.5, risk #29). Mooncake RDMA conflicts with "
            "XLA buffer donation."
        )

    if getattr(server_args, "enable_memory_saver", False):
        raise ValueError(
            "TPU MVP does not support --enable-memory-saver "
            "(kb §13 row 3.5, risk #39). LD_PRELOAD cudaMalloc interceptor "
            "is CUDA-only."
        )

    if getattr(server_args, "enable_deterministic_inference", False):
        raise ValueError(
            "TPU MVP does not support --enable-deterministic-inference "
            "(kb §13 row 3.5, risk #36). batch_invariant_ops pulls Triton "
            "CUDA kernels."
        )

    # --- kv-cache-dtype: bf16 only -------------------------------------
    if server_args.kv_cache_dtype in _FORBIDDEN_KV_CACHE_DTYPES:
        raise ValueError(
            f"TPU MVP forbids --kv-cache-dtype={server_args.kv_cache_dtype!r} "
            "(kb §13 row 3.5, risk #30). fp8/fp4 KV stored as "
            "uint8 + .view(fp8) bitcast is unsafe under torchax."
        )
    if server_args.kv_cache_dtype in ("auto", "bf16"):
        if server_args.kv_cache_dtype != "bfloat16":
            logger.info(
                "TPU: overriding kv_cache_dtype=%r -> 'bfloat16'.",
                server_args.kv_cache_dtype,
            )
            server_args.kv_cache_dtype = "bfloat16"

    # --- dtype: bf16 only ----------------------------------------------
    if server_args.dtype in ("float16", "fp16", "float32", "fp32"):
        raise ValueError(
            f"TPU MVP forbids --dtype={server_args.dtype!r} (kb §13 row 3.5, "
            "risk #35). TPU is bf16-optimised; fp16 sampler-bias risk."
        )
    if server_args.dtype == "auto":
        logger.info("TPU: overriding dtype='auto' -> 'bfloat16'.")
        server_args.dtype = "bfloat16"

    # --- attention backend: prefill == decode == 'jax' -----------------
    pf = server_args.prefill_attention_backend
    de = server_args.decode_attention_backend
    if pf not in (None, "jax") or de not in (None, "jax"):
        if pf != de:
            logger.warning(
                "TPU: forcing prefill_attention_backend=%r and "
                "decode_attention_backend=%r both to 'jax' (kb §13 row 3.5, "
                "risk #20). HybridAttnBackend would silently wrap mismatched "
                "backends.",
                pf, de,
            )
    server_args.prefill_attention_backend = "jax"
    server_args.decode_attention_backend = "jax"
    server_args.attention_backend = "jax"

    # --- sampler: pytorch only -----------------------------------------
    if server_args.sampling_backend not in (None, "pytorch"):
        logger.warning(
            "TPU: overriding sampling_backend=%r -> 'pytorch' (kb §13 row "
            "3.5). flashinfer sampler is CUDA-only.",
            server_args.sampling_backend,
        )
    server_args.sampling_backend = "pytorch"

    # --- watchdog: must tolerate JIT cold-compile (~30-100s per bucket) -
    if server_args.watchdog_timeout < 3600:
        logger.info(
            "TPU: bumping watchdog_timeout from %ss to 3600s "
            "(kb §13 row 3.5, risk #42).",
            server_args.watchdog_timeout,
        )
        server_args.watchdog_timeout = 3600.0

    # --- cuda graph / piecewise: forced off ----------------------------
    # ModelRunner consults current_platform.support_cuda_graph() which
    # already returns False for TPU; we set the explicit ServerArgs
    # toggles too so log lines don't lie.
    if getattr(server_args, "disable_cuda_graph", False) is False:
        # don't raise if user explicitly set it; just silently turn it on.
        server_args.disable_cuda_graph = True

    # --- multimodal arch rejection -------------------------------------
    # The model_config is built later, so we cannot inspect arch here.
    # Multimodal rejection lives in TpuSRTPlatform.check_and_update_config
    # if we add that hook; for now we rely on the runtime path failing
    # cleanly at vision-encoder import. (kb §13 row 3.5 / risk #23)

    # --- internal device-name rewrite: "tpu" -> "jax" ------------------
    # `torch.device("tpu")` raises on torch 2.11 (allow-list excludes it).
    # torchax has already claimed PrivateUse1 as "jax", so torch.device(
    # "jax") works. Rewrite once here so the 11 `torch.device(self.device)`
    # call sites in sglang core don't need per-site patches. User-facing
    # CLI flag stays --device tpu; internal name is "jax".
    # See decisions/2026-05-30_rewrite-device-to-jax-internally.md.
    if server_args.device == "tpu":
        logger.info("TPU: rewriting server_args.device 'tpu' -> 'jax' "
                    "(torch.device('tpu') not accepted on torch 2.11).")
        server_args.device = "jax"

    # Make the TPU choice sticky across spawned child processes — the
    # platform's auto-detect probe (jax.devices()) can fail in worker
    # processes where libtpu hasn't been initialized yet. Setting this
    # env var in the parent makes spawned workers force TPU.
    import os as _os
    _os.environ["SGLANG_FORCE_TPU"] = "1"
