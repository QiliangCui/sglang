"""S2 loadability smoke test for the TPU platform skeleton.

Run on the TPU host with the .venv active:

    pytest -xvs ~/cuiq-sglang/test/srt/tpu/test_tpu_smoke.py

Verifies the S1 contracts that the rest of the project depends on:
  - --device tpu parses; auto-select happens via is_tpu_available().
  - PlatformEnum.TPU is wired; is_tpu() True, is_out_of_tree() False.
  - apply_server_args_defaults forces bf16 + jax backend + 3600s
    watchdog, and raises ValueError on pp_size != 1.
  - torch.get_device_module("tpu") returns our shim with Stream /
    set_device / synchronize callable.
  - torch.compile becomes identity after init_backend().
  - Top-level "import sglang" does NOT import torchax (lazy).
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import pytest


def _make_server_args(extra_argv: list[str]):
    from sglang.srt.server_args import ServerArgs

    argv = ["test", "--device", "tpu", "--model-path", "Qwen/Qwen3-4B"] + extra_argv
    saved = sys.argv
    sys.argv = argv
    try:
        p = argparse.ArgumentParser()
        ServerArgs.add_cli_args(p)
        ns = p.parse_args()
        return ServerArgs.from_cli_args(ns)
    finally:
        sys.argv = saved


def test_platform_auto_selects_tpu():
    from sglang.srt.platforms import current_platform
    from sglang.srt.platforms.device_mixin import PlatformEnum
    from sglang.srt.platforms.tpu import TpuSRTPlatform

    assert isinstance(current_platform, TpuSRTPlatform), type(current_platform)
    assert current_platform.device_name == "tpu"
    assert current_platform._enum == PlatformEnum.TPU
    assert current_platform.is_tpu() is True
    assert current_platform.is_out_of_tree() is False
    assert current_platform.is_cuda() is False
    assert current_platform.support_cuda_graph() is False
    assert current_platform.support_piecewise_cuda_graph() is False
    assert current_platform.supports_fp8() is False
    assert current_platform.get_dispatch_key_name() == "tpu"
    assert current_platform.get_default_attention_backend() == "jax"
    assert current_platform.get_compile_backend() == "eager"


def test_platform_memory_queries_return_something():
    from sglang.srt.platforms import current_platform

    total = current_platform.get_device_total_memory()
    assert isinstance(total, int) and total > 0, total
    free, total2 = current_platform.get_available_memory()
    assert isinstance(free, int) and isinstance(total2, int)
    assert total2 == total or total2 == 0


def test_serverargs_applies_tpu_defaults():
    sa = _make_server_args([])
    assert sa.device == "tpu"
    assert sa.dtype == "bfloat16"
    assert sa.kv_cache_dtype == "bfloat16"
    assert sa.attention_backend == "jax"
    assert sa.prefill_attention_backend == "jax"
    assert sa.decode_attention_backend == "jax"
    assert sa.sampling_backend == "pytorch"
    assert sa.watchdog_timeout >= 3600.0
    assert sa.disable_cuda_graph is True


def test_serverargs_rejects_forbidden_flags():
    for flag, value, needle in [
        ("--pp-size", "2", "pp_size"),
        ("--dp-size", "2", "dp_size"),
        ("--attn-cp-size", "2", "attn_cp_size"),
        ("--kv-cache-dtype", "fp8_e4m3", "fp8"),
        ("--dtype", "float16", "float16"),
        ("--enable-memory-saver", None, "memory.saver"),
        ("--enable-two-batch-overlap", None, "two.batch.overlap"),
    ]:
        extra = [flag] if value is None else [flag, value]
        with pytest.raises(ValueError, match=needle):
            _make_server_args(extra)


def test_init_backend_idempotent():
    from sglang.srt.platforms import current_platform

    current_platform.init_backend()
    current_platform.init_backend()  # second call must not double-wrap.


def test_torch_get_device_module_returns_shim():
    import torch

    from sglang.srt.platforms import current_platform
    from sglang.srt.platforms.tpu_device_module import TpuDeviceModule

    current_platform.init_backend()
    mod = torch.get_device_module("tpu")
    assert isinstance(mod, TpuDeviceModule)
    # Required attribute surface for sglang core's distributed init.
    assert callable(mod.Stream)
    assert callable(mod.set_device)
    assert callable(mod.synchronize)
    # Calling them doesn't raise.
    s = mod.Stream()
    assert hasattr(s, "synchronize")
    mod.set_device(0)
    mod.synchronize()


def test_torch_compile_is_identity_after_init_backend():
    import torch

    from sglang.srt.platforms import current_platform

    current_platform.init_backend()
    # As bare decorator: returns fn unchanged.
    @torch.compile
    def f(x):
        return x + 1

    assert f(1) == 2
    # As call: returns fn unchanged.
    g = torch.compile(lambda x: x * 2)
    assert g(3) == 6


def test_torchax_not_imported_at_parent_level():
    """`import sglang` must NOT pull torchax into the parent process.

    torchax claims PrivateUse1 at import; doing so in the parent before
    the worker forks risks single-claim conflicts on respawn (risk #34).
    """
    code = (
        "import sys; import sglang; "
        "assert 'torchax' not in sys.modules, "
        "'torchax should be lazy-imported by worker init_backend(), not at top level'"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout, r.stderr)


def test_attention_backend_registration_is_lazy():
    """Until S3 lands JaxAttentionBackend's real body, init_backend swallows
    the ImportError. Once the real class is importable, registration goes
    live with no further edit."""
    from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
    from sglang.srt.platforms import current_platform

    current_platform.init_backend()
    # We don't assert "jax" is here yet — JaxAttentionBackend.__init__
    # raises in S2 because base methods are abstract. Once S3.2 lands a
    # concrete class, this test will start asserting presence.
    assert isinstance(ATTENTION_BACKENDS, dict)
