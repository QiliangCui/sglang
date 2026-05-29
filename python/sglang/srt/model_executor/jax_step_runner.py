"""JAX step runner — owns the main jax.jit(step_fun) (S3.4 placeholder).

ONE main JIT handles all of decode + extend + mixed via RPA-v3's
`distribution=(decode_end, prefill_end, mixed_end)` triple, mirroring
`tpu_inference.models.vllm.vllm_model_wrapper.jit_step_func` (kb-tpu
§3.3). Wraps `torch.func.functional_call(model, params, kwargs)` inside
`torchax.default_env()`. KV caches are donated argnames.

In-tree dispatch (plan §S1.5 row 9): `ModelRunner.forward_extend` and
`ModelRunner.forward_decode` short-circuit to `step(forward_batch)` when
`self.device == 'tpu'`. Hooking both is mandatory — without the
forward_extend hook, prefill runs eagerly under torchax → OOM (risk #31).

S3.4 fills in the jit body. The placeholder __init__ raises so we notice
if model_runner instantiates it before S3.4 lands.
"""
from __future__ import annotations


class JaxStepRunner:
    """Single-JIT step runner (S3.4 placeholder)."""

    def __init__(self, model_runner, *args, **kwargs):
        # model_runner: sglang.srt.model_executor.model_runner.ModelRunner
        raise NotImplementedError(
            "JaxStepRunner is a S3.4 placeholder. Construct a "
            "jax.jit(step_fun, donate_argnames=('kv_caches',), ...) wrapping "
            "torch.func.functional_call(self.model, params, kwargs) under "
            "torchax.default_env(). Bucket on (padded_num_tokens, "
            "padded_num_reqs, layer_name_to_kvcache_index, is_first_rank, "
            "is_last_rank). See plan_sglang_on_tpu.md §S3.4 and kb-tpu §3.3."
        )

    def step(self, forward_batch):
        """Single entry point for decode + extend + mixed batches.

        Routes via distribution=(decode_end, prefill_end, mixed_end) inside
        the JIT body. Returns the model output PyTorch tensor (torch_view of
        the JIT'd jax.Array).
        """
        raise NotImplementedError("JaxStepRunner.step — see S3.4")
