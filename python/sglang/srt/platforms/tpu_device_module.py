"""TPU device-module shim.

torchax 0.0.11's `device_module` lacks Stream / set_device / synchronize.
sglang core (`model_runner.py:524, 1042`, etc.) calls
`torch.get_device_module(<device>)` and expects those attrs, so we register
this thin wrapper under the name "tpu" in `TpuSRTPlatform.init_backend()`.

KB §12.5: option (A) — alias "tpu" to torchax's "jax" device module, fill
in the missing surface with no-ops. We do not touch the PrivateUse1 slot
("jax") which torchax owns.
"""
from __future__ import annotations


class _NoOpStream:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def synchronize(self):
        pass

    def wait_stream(self, *args, **kwargs):
        pass


class TpuDeviceModule:
    def __init__(self):
        import torchax

        self._wrapped = torchax.device_module

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def Stream(self, *args, **kwargs):
        return _NoOpStream()

    def set_device(self, device_id):
        return None

    def synchronize(self, *args, **kwargs):
        import jax

        jax.block_until_ready(jax.numpy.zeros(()))

    def stream(self, ctx=None):
        return _NoOpStream()
