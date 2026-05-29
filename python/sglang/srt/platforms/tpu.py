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
    """Best-effort probe — return True iff jax.devices() reports a TpuDevice.

    Called from `platforms/__init__.py:_resolve_platform()` device-probe
    fallback. We import jax lazily so non-TPU hosts don't pay the cost.
    """
    try:
        import jax
    except ImportError:
        return False
    try:
        return any(d.platform == "tpu" for d in jax.devices())
    except Exception:
        return False


class TpuDeviceMixin(DeviceMixin):
    """TPU implementation of the shared device operations.

    All [Planned] methods that sglang core doesn't yet call through
    `current_platform.*` are left raising NotImplementedError so we
    notice the day core gets migrated. The [Active] ones — memory
    queries — return jax-sourced answers.
    """

    _enum: PlatformEnum = PlatformEnum.TPU
    device_name: str = "tpu"
    device_type: str = "tpu"

    # --- [Active] memory ----------------------------------------------
    def get_device_total_memory(self, device_id: int = 0) -> int:
        import jax

        try:
            mem = jax.devices()[device_id].memory_stats()
            return int(mem.get("bytes_limit") or mem.get("bytes_in_use") or 0)
        except Exception:
            # v6e is ~32 GiB; conservative fallback so callers don't choke.
            return 32 * 1024**3

    def get_current_memory_usage(
        self, device: Optional["torch.device"] = None
    ) -> float:
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
        import jax

        try:
            return str(jax.devices()[device_id].device_kind)
        except Exception:
            return "tpu"

    def get_device_uuid(self, device_id: int = 0) -> str:
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
        import jax

        # block_until_ready on a 0-d array forces XLA to drain.
        jax.block_until_ready(jax.numpy.zeros(()))

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
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

        # 3. Device-module alias for "tpu". torchax claims PrivateUse1 as
        #    "jax"; we layer a "tpu" name on top so sglang core's
        #    `torch.get_device_module(self.device)` resolves cleanly.
        from sglang.srt.platforms.tpu_device_module import TpuDeviceModule

        torch._register_device_module("tpu", TpuDeviceModule())

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
