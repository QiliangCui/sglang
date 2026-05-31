"""TPU SRT platform.

In-tree TPU backend that lifts unmodified sglang PyTorch models onto JAX
via torchax (mirrors vllm-project/tpu-inference's "vllm path"). Selected
when --device=tpu OR when jax.devices() reports a TpuDevice and no other
platform claims the host.

`_enum = PlatformEnum.TPU` so `is_tpu()` is True for free and
`is_out_of_tree()` is False; we go through the same hardcoded dispatch
dicts NPU uses (configs/device_config.py SUPPORTED_DEVICES,
model_runner.py, parallel_state.py, etc.) rather than the 15-factory
OOT contract.

See ~/private-tool/sglang/decisions/2026-05-29_in-tree-mvp.md for why
this is in-tree rather than an external plugin.
"""
from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.platforms.device_mixin import (
    DeviceCapability,
    DeviceMixin,
    PlatformEnum,
)
from sglang.srt.platforms.interface import SRTPlatform


def is_tpu_available() -> bool:
    """Best-effort probe — does this host have a TPU?

    Called from `platforms/__init__.py:_resolve_platform()` device-probe
    fallback. We MUST NOT initialise libtpu in this probe — the parent
    process touching libtpu locks it out for scheduler-worker subprocesses
    (`ABORTED: Internal error when accessing libtpu multi-process lockfile`).

    Instead we check `/dev/vfio/0` (TPU device file exposed by the kernel
    driver on GCP TPU VMs). That's a stat-only check; no JAX/libtpu init.
    """
    import os

    return os.path.exists("/dev/vfio/0")


def _libtpu_initialized() -> bool:
    """Has libtpu been initialized in THIS process?

    We must not import/call jax.devices() in the parent process — that
    claims TPU vfio devices the worker needs. `init_backend()` sets the
    sentinel on the platform instance when it runs (worker only).
    """
    return getattr(_TpuFlag, "ready", False)


class _TpuFlag:
    ready: bool = False


class TpuDeviceMixin(DeviceMixin):
    """TPU implementation of the shared device operations.

    All [Planned] methods that sglang core doesn't yet call through
    `current_platform.*` are left raising NotImplementedError so we
    notice the day core gets migrated. The [Active] ones — memory
    queries — return jax-sourced answers ONLY after init_backend has
    run in this process. Before init_backend, return conservative
    defaults — this keeps the parent process from initializing libtpu.
    """

    _enum: PlatformEnum = PlatformEnum.TPU
    device_name: str = "tpu"
    device_type: str = "tpu"

    # --- [Active] memory ----------------------------------------------
    def get_device_total_memory(self, device_id: int = 0) -> int:
        if not _libtpu_initialized():
            return 32 * 1024**3  # v6e default; won't init libtpu in parent.
        import jax

        try:
            # At TP>1 each JAX process sees only its own subset of devices,
            # all starting from index 0. The caller passes self.gpu_id which
            # is the global local_rank — at TP=2, rank 1 would IndexError on
            # `jax.devices()[1]` if the process only owns 1 device. Always
            # query [0] since memory-per-chip is uniform within a process.
            mem = jax.devices()[0].memory_stats()
            return int(mem.get("bytes_limit") or mem.get("bytes_in_use") or 0)
        except Exception:
            return 32 * 1024**3

    def get_current_memory_usage(
        self, device: Optional["torch.device"] = None
    ) -> float:
        if not _libtpu_initialized():
            return 0.0
        import jax

        try:
            return float(jax.devices()[0].memory_stats().get("bytes_in_use", 0))
        except Exception:
            return 0.0

    # --- [Planned] device handles -------------------------------------
    def get_device(self, local_rank: int) -> "torch.device":
        # PrivateUse1 is registered as "jax" by torchax; "tpu" is the
        # user-facing name. Return "jax:{rank}" so torch ops dispatch.
        return torch.device("jax", local_rank)

    def set_device(self, device: "torch.device") -> None:
        # No-op: torchax doesn't track a current device.
        return None

    def get_device_name(self, device_id: int = 0) -> str:
        if not _libtpu_initialized():
            return "tpu"
        import jax

        try:
            return str(jax.devices()[device_id].device_kind)
        except Exception:
            return "tpu"

    def get_device_uuid(self, device_id: int = 0) -> str:
        if not _libtpu_initialized():
            return f"tpu-{device_id}"
        import jax

        try:
            d = jax.devices()[device_id]
            return f"tpu-{d.process_index}-{d.id}"
        except Exception:
            return f"tpu-{device_id}"

    def get_device_capability(self, device_id: int = 0) -> DeviceCapability:
        # Not meaningful for TPU; return a sentinel.
        return DeviceCapability(0, 0)

    def empty_cache(self) -> None:
        if not _libtpu_initialized():
            return
        # XLA manages buffers; no public "empty cache" API.
        # Defragment per-backend on a best-effort basis.
        import jax

        try:
            for backend in jax.lib.xla_bridge.backends().values():
                try:
                    backend.defragment()
                except Exception:
                    pass
        except Exception:
            pass

    def synchronize(self) -> None:
        if not _libtpu_initialized():
            return
        import jax

        # block_until_ready on a 0-d array forces XLA to drain.
        jax.block_until_ready(jax.numpy.zeros(()))

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        if not _libtpu_initialized():
            total = self.get_device_total_memory(device_id)
            return (total, total)
        import jax

        try:
            s = jax.devices()[device_id].memory_stats()
            total = int(s.get("bytes_limit", 0))
            used = int(s.get("bytes_in_use", 0))
            return (max(total - used, 0), total)
        except Exception:
            total = self.get_device_total_memory(device_id)
            return (total, total)

    def get_torch_distributed_backend_str(self) -> str:
        # gloo: single-host MVP. SPMD collectives happen inside jax.jit.
        return "gloo"


class TpuSRTPlatform(TpuDeviceMixin, SRTPlatform):
    """In-tree TPU SRT platform.

    Capability flags forced off: cuda-graph capture, piecewise compile,
    fp8 KV. The `init_backend()` body registers the torch device-module
    alias and the "jax" attention backend in sglang's registry.
    """

    supported_quantization: list[str] = []  # populated as quant paths land

    def supports_fp8(self) -> bool:
        return False  # risk #30: bf16-only KV cache in MVP.

    def is_pin_memory_available(self) -> bool:
        return False  # TPU host pinning is unhelpful for XLA transfers.

    def support_cuda_graph(self) -> bool:
        return False  # JaxStepRunner replaces the cuda-graph path.

    def support_piecewise_cuda_graph(self) -> bool:
        return False  # outer jax.jit replaces piecewise compile.

    def get_compile_backend(self, mode: Optional[str] = None) -> str:
        return "eager"  # torch.compile becomes a no-op inside init_backend.

    def get_default_attention_backend(self) -> str:
        return "jax"

    def get_dispatch_key_name(self) -> str:
        return "tpu"

    def apply_server_args_defaults(self, server_args) -> None:
        from sglang.srt.platforms.tpu_server_args_defaults import apply

        apply(server_args)

    def init_backend(self) -> None:
        """One-time per-worker setup. Runs at the first call site that
        touches `current_platform.init_backend()` (model_runner import).
        """
        import sys as _sys
        import os as _os
        _sys.stderr.write(f"[TpuSRTPlatform.init_backend] firing pid={_os.getpid()} ppid={_os.getppid()}\n")
        _sys.stderr.flush()
        return self._init_backend_inner()

    def _init_backend_inner(self) -> None:
        # Mark THIS process as libtpu-initialized so the device-query
        # methods (get_device_total_memory etc) start using jax.devices().
        _TpuFlag.ready = True

        # TP=2+ pre-req: torch.distributed.barrier() calls
        # torch._C._get_accelerator() which fails on the torchax-registered
        # PrivateUse1 backend with "Please register PrivateUse1HooksInterface
        # by RegisterPrivateUse1HooksInterface first." Re-route the call to
        # report CPU so the gloo barrier uses CPU device (matches the
        # gloo backend choice from _DEVICE_TO_DISTRIBUTED_BACKEND["jax"]).
        # See s5_walls.md wall 1.
        import torch as _torch
        if not getattr(_torch._C, "_sglang_tpu_accel_patched", False):
            _torch._C._get_accelerator = lambda: _torch.device("cpu")
            _torch._C._sglang_tpu_accel_patched = True
        """Actual body. Wrapped so we can verify firing from logs.

        Order matters:
          1. Pin cache env vars (so child workers inherit them even if the
             parent shell did not source ~/sglang_tpu_env.sh).
          2. `torch.compile = no-op` BEFORE any sglang module that uses
             module-level @torch.compile imports — risk #32.
          3. Register torch device module under "tpu" so
             `torch.get_device_module("tpu")` works (model_runner.py uses
             it for synchronize / set_device).
          4. Register "jax" attention backend in ATTENTION_BACKENDS so
             init_attention_backend can find it.
          5. Best-effort unwrap any @torch.compile already bound to
             pre-imported sampler symbols (Appendix I safety net).
        """
        import os

        # 1. Cache anchors — defensive, no-op if shell already exports them.
        os.environ.setdefault("HF_HOME", "/mnt/disks/persist/hf")
        os.environ.setdefault("HF_HUB_CACHE", "/mnt/disks/persist/hf/hub")
        os.environ.setdefault(
            "HUGGINGFACE_HUB_CACHE", "/mnt/disks/persist/hf/hub"
        )
        os.environ.setdefault(
            "JAX_COMPILATION_CACHE_DIR",
            "/mnt/disks/persist/jax_cache_sglang",
        )
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

        # 2. torch.compile -> identity (risk #32). Subsequent @torch.compile
        #    decorations become no-ops; already-bound ones are unwrapped in (5).
        def _no_compile(fn=None, **kw):
            if fn is None:
                return lambda f: f
            return fn

        torch.compile = _no_compile

        # 2.5 Import torchax so PrivateUse1 gets renamed to "jax" (torchax
        #     __init__ does this on import). After this point, torch.device(
        #     "jax") returns a valid device, which the rest of the runtime
        #     (DeviceConfig, dp_attention, model_runner) relies on after
        #     apply_server_args_defaults rewrote 'tpu' -> 'jax'.
        #
        #     Per risk #34, doing this in the parent process is risky if the
        #     parent then forks workers — but the engine uses spawn (not
        #     fork) so worker processes get a clean torch state. In-process
        #     testing (S2, S3.0 spike) also calls init_backend exactly once.
        import torchax  # noqa: F401

        # 3. Device-module alias for "tpu". torch 2.11 won't let us call
        #    `torch._register_device_module("tpu", ...)` — its torch.device()
        #    constructor allow-list doesn't include "tpu", and torchax has
        #    already claimed privateuseone as "jax". So we monkey-patch
        #    `torch.get_device_module` to intercept the "tpu" call and
        #    return our shim; everything else falls through to upstream.
        #    (KB §12.5 option C — chosen 2026-05-29 after option A failed
        #    on torch 2.11; see progress.md.)
        from sglang.srt.platforms.tpu_device_module import TpuDeviceModule

        _tpu_module = TpuDeviceModule()
        _orig_get_device_module = torch.get_device_module

        def _get_device_module_with_tpu(device=None):
            if device in ("tpu", "jax"):
                return _tpu_module
            if isinstance(device, torch.device) and device.type in ("tpu", "jax", "privateuseone"):
                return _tpu_module
            if device is None:
                # No-arg form: sglang core uses this in module-level type
                # annotations (e.g. parallel_state.py GraphCaptureContext).
                # torch's default lookup hits torch._C._get_accelerator()
                # which raises on the partial PrivateUse1 registration
                # torchax leaves us with — return our shim instead.
                return _tpu_module
            return _orig_get_device_module(device)

        # Idempotent: don't double-wrap on second init_backend call.
        if not getattr(torch.get_device_module, "_tpu_aware", False):
            _get_device_module_with_tpu._tpu_aware = True
            torch.get_device_module = _get_device_module_with_tpu

        # 4. Register the JAX attention backend if sglang's attention
        #    registry is reachable. Done lazily — the backend module is
        #    imported only when this method runs (after S3 lands the body).
        try:
            from sglang.srt.layers.attention.attention_registry import (
                ATTENTION_BACKENDS,
            )
            from sglang.srt.layers.attention.jax_backend import (
                JaxAttentionBackend,
            )

            ATTENTION_BACKENDS.setdefault("jax", JaxAttentionBackend)
        except ImportError:
            # S3 hasn't landed the JAX backend module yet.
            pass

        # 4.5 Install Qwen3Attention.forward_prepare_native patch for real-TP
        #     sharding (real-sharding step 2). Glue is thin (~25 lines); all
        #     sharding logic lives in
        #     `sglang.srt.layers.jax_sharding_helpers`. At TP=1 the patch
        #     short-circuits to the original implementation, so no behavior
        #     change. Idempotent.
        try:
            from sglang.srt.models import qwen3 as _qwen3_mod
            if not getattr(_qwen3_mod.Qwen3Attention, "_sglang_tpu_qkv_shard_patched", False):
                _orig_prepare_native = _qwen3_mod.Qwen3Attention.forward_prepare_native

                def _patched_forward_prepare_native(self, positions, hidden_states):
                    # Lazy import keeps cold path off non-TPU runs.
                    from sglang.srt.layers.jax_sharding_helpers import (
                        is_sharding_active,
                        sharded_qkv_matmul,
                    )
                    # Two short-circuits to the original path:
                    #   (1) TP=1 (1x1 mesh) — sharding is a strict no-op.
                    #   (2) Pre-shard hasn't run yet (e.g. on a model class the
                    #       generic discovery rule didn't match). Falling back
                    #       keeps correctness; the warning in step-runner
                    #       captures the no-shard case.
                    if not is_sharding_active() or not hasattr(
                        self, "_sglang_q_weight_sharded"
                    ):
                        return _orig_prepare_native(self, positions, hidden_states)
                    from torchax.interop import jax_view
                    from torchax.tensor import Tensor as _TxTensor
                    h_jax = jax_view(hidden_states)
                    q_j, k_j, v_j = sharded_qkv_matmul(
                        h_jax,
                        self._sglang_q_weight_sharded,
                        self._sglang_k_weight_sharded,
                        self._sglang_v_weight_sharded,
                    )
                    # Wrap back as torchax tensors so apply_qk_norm + rotary_emb
                    # (which call torch ops) compose normally.
                    import torchax as _txa
                    env = _txa.default_env()
                    q = _TxTensor(q_j, env)
                    k = _TxTensor(k_j, env)
                    v = _TxTensor(v_j, env)
                    # Same downstream as the original: QK norm + rotary on q/k.
                    from sglang.srt.models.utils import apply_qk_norm
                    q, k = apply_qk_norm(
                        q=q,
                        k=k,
                        q_norm=self.q_norm,
                        k_norm=self.k_norm,
                        head_dim=self.head_dim,
                        alt_stream=self.alt_stream,
                    )
                    q, k = self.rotary_emb(positions, q, k)
                    return q, k, v

                _qwen3_mod.Qwen3Attention.forward_prepare_native = _patched_forward_prepare_native
                _qwen3_mod.Qwen3Attention._sglang_tpu_qkv_shard_patched = True
        except Exception:
            # Qwen3 module not importable on this build; nothing to patch.
            pass

        # 4.6 Real-sharding step 4: patch `RowParallelLinear.forward` to use
        #     the pre-sharded weight set by `pre_shard_row_parallel_weights`
        #     in `JaxStepRunner._ensure_setup`. Model-agnostic (works for
        #     o_proj on Qwen3, Llama, etc., and for down_proj later). Two
        #     short-circuits: (i) TP=1, no pre-shard; (ii) module has no
        #     `_sglang_w_sharded` (not in this step's name_filter).
        #     Idempotent.
        try:
            from sglang.srt.layers import linear as _linear_mod
            if not getattr(
                _linear_mod.RowParallelLinear, "_sglang_tpu_row_shard_patched", False
            ):
                _orig_rp_forward = _linear_mod.RowParallelLinear.forward

                def _patched_row_parallel_forward(
                    self, input_, skip_all_reduce=False, forward_batch=None
                ):
                    from sglang.srt.layers.jax_sharding_helpers import (
                        is_sharding_active,
                        sharded_row_parallel_matmul,
                    )
                    if not is_sharding_active() or not hasattr(
                        self, "_sglang_w_sharded"
                    ):
                        return _orig_rp_forward(
                            self,
                            input_,
                            skip_all_reduce=skip_all_reduce,
                            forward_batch=forward_batch,
                        )
                    from torchax.interop import jax_view
                    from torchax.tensor import Tensor as _TxTensor
                    import torchax as _txa
                    env = _txa.default_env()
                    inp_jax = jax_view(input_)
                    bias_jax = None
                    if self.bias is not None and not self.skip_bias_add:
                        bias_jax = jax_view(self.bias)
                    out_jax = sharded_row_parallel_matmul(
                        inp_jax, self._sglang_w_sharded, bias=bias_jax
                    )
                    out = _TxTensor(out_jax, env)
                    out_bias = self.bias if self.skip_bias_add else None
                    return out, out_bias

                _linear_mod.RowParallelLinear.forward = _patched_row_parallel_forward
                _linear_mod.RowParallelLinear._sglang_tpu_row_shard_patched = True
        except Exception:
            pass

        # 5. Unwrap @torch.compile that was bound at import time (before
        #    step 2 took effect). Currently known: sampler.multinomial_with_seed.
        try:
            from sglang.srt.layers import sampler  # noqa: WPS433

            fn = getattr(sampler, "multinomial_with_seed", None)
            if fn is not None:
                sampler.multinomial_with_seed = getattr(fn, "__wrapped__", fn)
        except Exception:
            pass

    @classmethod
    def seed_everything(cls, seed: int | None = None) -> None:
        if seed is None:
            return
        super().seed_everything(seed)
        # JAX uses explicit PRNGKey; nothing to seed at the device layer.
