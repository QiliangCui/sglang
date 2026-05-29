"""Minimal vllm shim — installs into sys.modules BEFORE tpu_inference imports.

`tpu_inference` declares no explicit vllm version dep, but its
`logger.py` (imported by every common helper) does
`from vllm.logger import init_logger`, and `utils.py` does
`from vllm import envs as vllm_envs`. Installing the real vllm pulls
flashinfer / cuda-python / etc — exactly what we stripped in S0.3.

We shim the smallest set of vllm modules that tpu_inference touches at
*module-import* time. Anything else (vllm.config, vllm.distributed,
vllm.model_executor) stays unimported because we don't pull MoE / LoRA
paths in MVP.

Usage: `import sglang.srt.layers.attention._vllm_shim` BEFORE any
`tpu_inference.*` import.
"""
from __future__ import annotations

import logging
import sys
import types


def _install() -> None:
    if "vllm" in sys.modules:
        return  # real vllm wins; don't shim over it.

    # vllm.logger.{init_logger, _VllmLogger}
    vllm = types.ModuleType("vllm")
    vllm_logger = types.ModuleType("vllm.logger")

    class _VllmLogger(logging.Logger):
        pass

    def init_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)

    vllm_logger.init_logger = init_logger
    vllm_logger._VllmLogger = _VllmLogger

    # vllm.envs — tpu_inference.utils reads VLLM_TPU_USING_PATHWAYS off it.
    # Provide a SimpleNamespace that returns False/None for any attr.
    class _Envs:
        def __getattr__(self, name):
            # Booleans default False; anything else None. Adjust if a
            # tpu_inference path branches on something specific.
            return False

    vllm_envs = _Envs()
    # vllm.utils — tpu_inference.utils does `from vllm import utils` at module
    # load but only USES it inside get_hash_fn_by_name() which our attention
    # path never reaches. An empty module is enough for the import to resolve.
    vllm_utils = types.ModuleType("vllm.utils")

    vllm.envs = vllm_envs
    vllm.logger = vllm_logger
    vllm.utils = vllm_utils

    sys.modules["vllm"] = vllm
    sys.modules["vllm.logger"] = vllm_logger
    sys.modules["vllm.envs"] = vllm_envs  # type: ignore[assignment]
    sys.modules["vllm.utils"] = vllm_utils


_install()
