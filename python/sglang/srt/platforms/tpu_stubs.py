"""Stubs for TPU code paths that must never be instantiated in MVP."""


class DsaNotSupported:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "DSA (DeepSeek V3.2) KV pool is not supported on TPU in MVP. "
            "Reject --kv-cache-dtype=dsa in apply_server_args_defaults."
        )


class PiecewiseNotSupported:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Piecewise CUDA-graph backend is disabled on TPU "
            "(support_piecewise_cuda_graph() returns False)."
        )
