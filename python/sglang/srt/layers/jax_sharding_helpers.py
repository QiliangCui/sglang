"""Model-agnostic JAX sharding helpers for real-TP on TPU.

Per `~/private-tool/sglang/decisions/2026-05-31_real-sharding-design-deferred.md`
and REVIEWER 2026-05-31 08:00 UTC. All sharding LOGIC lives here; per-model
patches in `models/*.py` (or installed at `tpu.py` init time) are thin glue
that calls these helpers.

When a future model lands on TPU we add ~10-15 lines of glue per model — the
real work is reused. When/if we ever refactor to sglang's native
`attn_tp_size` path, the helpers lift up unchanged and the patches get
deleted.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple


def get_attn_backend_mesh():
    """Return the `JaxAttentionBackend._mesh` (class-level singleton built by
    `_build_default_mesh()` from `SGLANG_JAX_MESH_TP`). Built lazily by the
    backend's first `__init__`; this helper just reads the cached instance.
    """
    from sglang.srt.layers.attention.jax_backend import JaxAttentionBackend
    return JaxAttentionBackend._mesh


def _mesh_tp_size(mesh) -> int:
    """Number of devices along the ATTN_HEAD axis. Returns 1 if mesh is None."""
    if mesh is None:
        return int(os.environ.get("SGLANG_JAX_MESH_TP", "1"))
    return int(mesh.shape["model"])


def shard_qkv_weight(
    qkv_weight,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    mesh,
):
    """Slice a combined `[Q_all || K_all || V_all, hidden]` weight into Q, K,
    V; shard each on the head dim along ATTN_HEAD.

    Args:
      qkv_weight: `jax.Array` shape `[(num_heads + 2*num_kv_heads)*head_dim,
        hidden_size]`. The combined-qkv layout sglang produces at TP=1.
      num_heads: total attention heads (Q).
      num_kv_heads: total KV heads (K = V).
      head_dim: per-head dim.
      mesh: `jax.sharding.Mesh` with `model` axis = ATTN_HEAD.

    Returns:
      `(q, k, v)`, each `jax.Array` sharded with `P("model", None)`.
      Shapes: q `[num_heads*head_dim, hidden]`, k/v `[num_kv_heads*head_dim,
      hidden]`.

    At TP=1 (1x1 mesh) this is a strict no-op slice — the device_put is
    redundant but harmless. At TP>1 each device ends up with its own contiguous
    chunk of Q/K/V rows.
    """
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P

    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    q_repl = qkv_weight[:q_size, :]
    k_repl = qkv_weight[q_size:q_size + kv_size, :]
    v_repl = qkv_weight[q_size + kv_size:q_size + 2 * kv_size, :]
    spec = NamedSharding(mesh, P("model", None))
    return (
        jax.device_put(q_repl, spec),
        jax.device_put(k_repl, spec),
        jax.device_put(v_repl, spec),
    )


def shard_qkv_bias(
    qkv_bias,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    mesh,
):
    """Same Q/K/V split + ATTN_HEAD shard for a 1-D bias of length
    `(num_heads + 2*num_kv_heads)*head_dim`. Returns `(q_b, k_b, v_b)` or
    `(None, None, None)` if `qkv_bias is None`.
    """
    if qkv_bias is None:
        return None, None, None
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P

    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    spec = NamedSharding(mesh, P("model"))
    return (
        jax.device_put(qkv_bias[:q_size], spec),
        jax.device_put(qkv_bias[q_size:q_size + kv_size], spec),
        jax.device_put(qkv_bias[q_size + kv_size:q_size + 2 * kv_size], spec),
    )


def sharded_qkv_matmul(
    hidden_states,
    q_weight,
    k_weight,
    v_weight,
    q_bias=None,
    k_bias=None,
    v_bias=None,
):
    """Three separate matmuls against the already-sharded Q/K/V weights.

    Args:
      hidden_states: `jax.Array` shape `[batch, hidden]`. Replicated (or
        sharded on batch); the matmul output inherits the ATTN_HEAD sharding
        from the weight's first dim.
      q_weight / k_weight / v_weight: weights from `shard_qkv_weight`.
      q_bias / k_bias / v_bias: optional 1-D biases.

    Returns:
      `(q, k, v)`, each `jax.Array` with last-dim sharded on ATTN_HEAD.
    """
    q = hidden_states @ q_weight.T
    if q_bias is not None:
        q = q + q_bias
    k = hidden_states @ k_weight.T
    if k_bias is not None:
        k = k + k_bias
    v = hidden_states @ v_weight.T
    if v_bias is not None:
        v = v + v_bias
    return q, k, v


def is_sharding_active(mesh=None) -> bool:
    """True iff the ATTN_HEAD axis has more than one device. Used by glue
    patches to skip the sharded path at TP=1 (where it would be a strict no-op
    but adds compile / dispatch overhead)."""
    if mesh is None:
        mesh = get_attn_backend_mesh()
    return _mesh_tp_size(mesh) > 1


def pre_shard_qkv_weights(model, mesh) -> int:
    """Walk `model.modules()` once, pre-shard `qkv_proj.weight` on every module
    that looks like an attention block with combined-qkv layout.

    Generic discovery rule: a module is sharded iff it has `qkv_proj`,
    `num_heads`, `num_kv_heads`, `head_dim` attributes — the same surface
    sglang's attention classes expose for any model. No model-class import.

    Side-effect: sets `_sglang_{q,k,v}_weight_sharded` on each matched module.
    Returns the count of modules sharded.

    Idempotent — skip modules already sharded.

    MUST be called outside any `jax.jit` trace (typically once during
    `JaxStepRunner._ensure_setup` after the model loads). Calling it from
    inside a traced step_fun causes `UnexpectedTracerError` because the
    sharded outputs would become traced values leaking out of the JIT scope.
    """
    if not is_sharding_active(mesh):
        return 0
    import torchax
    from torchax.interop import jax_view

    count = 0
    with torchax.default_env():
        for module in model.modules():
            if getattr(module, "_sglang_q_weight_sharded", None) is not None:
                continue
            if not hasattr(module, "qkv_proj"):
                continue
            if not all(hasattr(module, a) for a in ("num_heads", "num_kv_heads", "head_dim")):
                continue
            qkv_w = module.qkv_proj.weight
            qkv_jax = jax_view(qkv_w)
            q_w, k_w, v_w = shard_qkv_weight(
                qkv_jax,
                num_heads=module.num_heads,
                num_kv_heads=module.num_kv_heads,
                head_dim=module.head_dim,
                mesh=mesh,
            )
            module._sglang_q_weight_sharded = q_w
            module._sglang_k_weight_sharded = k_w
            module._sglang_v_weight_sharded = v_w
            count += 1
    return count


def shard_row_parallel_weight(weight, mesh):
    """Shard a row-parallel linear's weight along its INPUT (column) dim along
    ATTN_HEAD.

    Args:
      weight: `jax.Array` shape `[out, in]` (sglang Linear convention).
      mesh: `Mesh` with `model` axis = ATTN_HEAD.

    Returns:
      `jax.Array` sharded with `P(None, "model")` — each device owns a
      contiguous column slice of size `in / N`.

    The row-parallel pattern: when the matmul's INPUT is also sharded on the
    same axis (e.g., RPA kernel output sharded by ATTN_HEAD on heads dim,
    which becomes the input cols dim of o_proj), each device computes a partial
    `[batch, out]` sum. JAX inserts an `all-reduce` along the `model` axis to
    sum the partials into the final result.
    """
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    return jax.device_put(weight, NamedSharding(mesh, P(None, "model")))


def sharded_row_parallel_matmul(input_arr, weight_col_sharded, bias=None):
    """Matmul for a row-parallel linear: `input @ weight.T + bias`.

    When `input_arr` is already sharded on its last dim by `model` axis (which
    matches `weight_col_sharded`'s col dim), each device computes a partial
    `[batch, out]`. JAX's auto-redistribute + the matching shardings cause an
    implicit `psum`/`all-reduce` to sum partials. If `input_arr` is replicated
    instead, JAX slices it on the fly — same end result, slightly less optimal.

    Returns `jax.Array` with replicated last dim (the summed result).
    """
    out = input_arr @ weight_col_sharded.T
    if bias is not None:
        out = out + bias
    return out


def shard_gate_up_weight(gate_up_weight, intermediate_size: int, mesh):
    """Slice a combined `[2*intermediate_size, hidden]` gate_up_proj weight
    into gate and up halves; shard each on the intermediate row dim along
    ATTN_HEAD.

    Args:
      gate_up_weight: `jax.Array` shape `[2*intermediate_size, hidden]`.
        sglang's `MergedColumnParallelLinear` lays out gate first, then up.
      intermediate_size: gate (and up) row count per half.
      mesh: `Mesh` with `model` axis = ATTN_HEAD.

    Returns:
      `(gate_weight, up_weight)` each `jax.Array` sharded with
      `P("model", None)`. Per device: `[intermediate_size/N, hidden]`.

    The matmul output of `hidden @ gate_w.T` (where hidden is replicated and
    gate_w is row-sharded along intermediate) emits a tensor sharded last-dim
    by ATTN_HEAD. The downstream `silu(gate) * up` is element-wise and
    preserves sharding. The result flows naturally into a `down_proj` that's
    col-sharded on intermediate — no all-gather between gate_up and down.
    """
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    gate_repl = gate_up_weight[:intermediate_size, :]
    up_repl = gate_up_weight[intermediate_size:2 * intermediate_size, :]
    spec = NamedSharding(mesh, P("model", None))
    return (
        jax.device_put(gate_repl, spec),
        jax.device_put(up_repl, spec),
    )


def sharded_mlp_silu_up(hidden, gate_weight_sharded, up_weight_sharded):
    """SwiGLU intermediate from sharded gate/up weights.

    `hidden`: `jax.Array` shape `[batch, hidden]`, replicated.
    Returns: `jax.Array` shape `[batch, intermediate_size]` sharded last-dim
    by ATTN_HEAD.
    """
    import jax
    gate_out = hidden @ gate_weight_sharded.T
    up_out = hidden @ up_weight_sharded.T
    return jax.nn.silu(gate_out) * up_out


def pre_shard_gate_up_weights(model, mesh) -> int:
    """Walk `model.modules()` and shard every `gate_up_proj` child via
    `shard_gate_up_weight`. Discovery rule: attribute `gate_up_proj` on a
    parent module, with a `.weight` whose first dim is even (so we can split
    into gate || up). Sets `_sglang_gate_w_sharded` and `_sglang_up_w_sharded`
    on the child. Idempotent. TP=1 no-op.

    MUST run outside any `jax.jit` trace.
    """
    if not is_sharding_active(mesh):
        return 0
    import torchax
    from torchax.interop import jax_view

    count = 0
    with torchax.default_env():
        for module in model.modules():
            for attr_name, child in module.named_children():
                if attr_name != "gate_up_proj":
                    continue
                if getattr(child, "_sglang_gate_w_sharded", None) is not None:
                    continue
                if not hasattr(child, "weight"):
                    continue
                w_jax = jax_view(child.weight)
                # sglang's MergedColumnParallelLinear lays gate||up as rows.
                # weight shape = [2 * intermediate_size, hidden].
                if w_jax.shape[0] % 2 != 0:
                    continue
                intermediate_size = w_jax.shape[0] // 2
                gate_w, up_w = shard_gate_up_weight(
                    w_jax, intermediate_size, mesh
                )
                child._sglang_gate_w_sharded = gate_w
                child._sglang_up_w_sharded = up_w
                count += 1
    return count


def pre_shard_row_parallel_weights(model, mesh, name_filter) -> int:
    """Walk `model.modules()` once and shard every linear submodule whose
    attribute name on its parent matches `name_filter` (a set or tuple of
    strings, e.g., `{"o_proj"}` for Step 4, `{"o_proj", "down_proj"}` later).

    The filter applies to ATTRIBUTE NAME — we walk parent modules and look at
    their named children to find e.g. `Qwen3Attention.o_proj`. The child's
    `.weight` is sharded in place via `shard_row_parallel_weight`. The child
    module gets `_sglang_w_sharded` set so the patched `RowParallelLinear.forward`
    in tpu.py can pick it up.

    Returns count of weights sharded. Idempotent. TP=1 no-op.

    MUST run outside any `jax.jit` trace.
    """
    if not is_sharding_active(mesh):
        return 0
    import torchax
    from torchax.interop import jax_view

    filter_set = set(name_filter)
    count = 0
    with torchax.default_env():
        for module in model.modules():
            for attr_name, child in module.named_children():
                if attr_name not in filter_set:
                    continue
                if getattr(child, "_sglang_w_sharded", None) is not None:
                    continue
                if not hasattr(child, "weight"):
                    continue
                w_jax = jax_view(child.weight)
                child._sglang_w_sharded = shard_row_parallel_weight(w_jax, mesh)
                count += 1
    return count
