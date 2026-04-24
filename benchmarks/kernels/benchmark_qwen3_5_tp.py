#!/usr/bin/env python
# Copyright (c) 2025, Machete Authors
"""Benchmark Qwen 3.5-0.8B tensor-parallel inference in same-device emulation.

Current scope:
- full-model prefill TP benchmark
- full-model decode TP benchmark
- correctness checks against full-model PyTorch reference
- timing against eager and torch.compile baselines

This benchmark uses two logical TP ranks inside one megakernel on a single GPU.
That exercises the framework's peer-communication path without requiring a
multi-GPU host. Real one-node multi-GPU end-to-end validation still requires a
2+ GPU system and multi-rank launch coordination.
"""

import contextlib
import io
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from machete.utils.benchmark import Benchmark
from benchmarks.kernels.benchmark_qwen3_5_decode import (
    DECODE_S,
    HEAD_DIM,
    HIDDEN,
    INTERMEDIATE,
    KV_DIM,
    KV_GROUP_SIZE,
    NUM_KV_HEADS,
    NUM_LAYERS,
    NUM_Q_HEADS,
    Q_DIM,
    VOCAB_SIZE,
    allocate_kv_cache,
    allocate_model_weights,
    is_sm90_or_newer,
    megakernel_decode_build,
    sequential_decode_step,
)
from benchmarks.kernels.benchmark_qwen3_5_layer import (
    _pick_single_layer_forward_mma_reg_count,
    _pick_single_layer_forward_tpb,
    maybe_compiled_forward,
)
from benchmarks.kernels.benchmark_qwen3_5_prefill import (
    PAGE_SIZES,
    sequential_prefill,
)
from benchmarks.kernels.benchmark_qwen3_5_decode import _apply_rope, _per_head_rmsnorm, _rmsnorm
import torch.nn.functional as F

try:
    import cutlass  # noqa: F401

    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False


TP_DEGREE = 2
LOCAL_Q_HEADS = NUM_Q_HEADS // TP_DEGREE
LOCAL_KV_HEADS = NUM_KV_HEADS // TP_DEGREE
LOCAL_Q_DIM = Q_DIM // TP_DEGREE
LOCAL_KV_DIM = KV_DIM // TP_DEGREE
LOCAL_INTERMEDIATE = INTERMEDIATE // TP_DEGREE
LOCAL_KV_GROUP_SIZE = LOCAL_Q_HEADS // LOCAL_KV_HEADS


def _rel_mean(a, b):
    return float((a.float() - b.float()).abs().mean() / (b.float().abs().mean() + 1e-12))


def sequential_decode_step_layers(x, residual, pos, k_caches, v_caches, weights, num_layers=NUM_LAYERS):
    """Reference decode step with configurable layer count."""
    batch, step, _ = x.shape
    cos, sin = weights["cos"], weights["sin"]

    for i in range(num_layers):
        pfx = f"layer.{i}"
        residual = x + residual
        h = _rmsnorm(residual, weights[f"{pfx}.attn_norm"])

        q = torch.matmul(h, weights[f"{pfx}.W_q"].t())
        k = torch.matmul(h, weights[f"{pfx}.W_k"].t())
        v = torch.matmul(h, weights[f"{pfx}.W_v"].t())

        q = _per_head_rmsnorm(q, weights[f"{pfx}.w_q_norm"], NUM_Q_HEADS, HEAD_DIM)
        k = _per_head_rmsnorm(k, weights[f"{pfx}.w_k_norm"], NUM_KV_HEADS, HEAD_DIM)

        q = _apply_rope(q, cos, sin, NUM_Q_HEADS, HEAD_DIM, 64, pos)
        k = _apply_rope(k, cos, sin, NUM_KV_HEADS, HEAD_DIM, 64, pos)

        k_caches[i][:, :, pos:pos + step, :] = k.view(batch, step, NUM_KV_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
        v_caches[i][:, :, pos:pos + step, :] = v.view(batch, step, NUM_KV_HEADS, HEAD_DIM).permute(0, 2, 1, 3)

        q_4d = q.view(batch, step, NUM_Q_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
        k_full = k_caches[i][:, :, :pos + step, :].repeat_interleave(KV_GROUP_SIZE, dim=1)
        v_full = v_caches[i][:, :, :pos + step, :].repeat_interleave(KV_GROUP_SIZE, dim=1)
        attn_out = F.scaled_dot_product_attention(q_4d, k_full, v_full, is_causal=False)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(batch, step, Q_DIM)

        proj = torch.matmul(attn_out, weights[f"{pfx}.W_o"].t())

        residual = proj + residual
        h2 = _rmsnorm(residual, weights[f"{pfx}.mlp_norm"])

        gate_up = torch.matmul(h2, weights[f"{pfx}.W_gate_up"].t())
        gate, up = gate_up.chunk(2, dim=-1)
        mlp_h = F.silu(gate) * up
        x = torch.matmul(mlp_h, weights[f"{pfx}.W_down"].t())

    residual = x + residual
    h_final = _rmsnorm(residual, weights["final_norm"])
    logits = torch.matmul(h_final, weights["lm_head"].t())
    return logits, residual


def _split_gate_up_weight(weight, tp_degree=TP_DEGREE):
    gate, up = weight.chunk(2, dim=0)
    gate_shards = gate.chunk(tp_degree, dim=0)
    up_shards = up.chunk(tp_degree, dim=0)
    return [torch.cat([gate_shards[i], up_shards[i]], dim=0).contiguous() for i in range(tp_degree)]


def _shard_tp_weights(weights, tp_degree=TP_DEGREE):
    shards = []
    for rank in range(tp_degree):
        shard = {"cos": weights["cos"], "sin": weights["sin"]}
        for i in range(NUM_LAYERS):
            pfx = f"layer.{i}"
            shard[f"{pfx}.W_q"] = weights[f"{pfx}.W_q"].chunk(tp_degree, dim=0)[rank].contiguous()
            shard[f"{pfx}.W_k"] = weights[f"{pfx}.W_k"].chunk(tp_degree, dim=0)[rank].contiguous()
            shard[f"{pfx}.W_v"] = weights[f"{pfx}.W_v"].chunk(tp_degree, dim=0)[rank].contiguous()
            shard[f"{pfx}.W_o"] = weights[f"{pfx}.W_o"].chunk(tp_degree, dim=1)[rank].contiguous()
            shard[f"{pfx}.W_gate_up"] = _split_gate_up_weight(weights[f"{pfx}.W_gate_up"], tp_degree)[rank]
            shard[f"{pfx}.W_down"] = weights[f"{pfx}.W_down"].chunk(tp_degree, dim=1)[rank].contiguous()
            shard[f"{pfx}.w_q_norm"] = weights[f"{pfx}.w_q_norm"]
            shard[f"{pfx}.w_k_norm"] = weights[f"{pfx}.w_k_norm"]
            shard[f"{pfx}.attn_norm"] = weights[f"{pfx}.attn_norm"]
            shard[f"{pfx}.mlp_norm"] = weights[f"{pfx}.mlp_norm"]
        shard["final_norm"] = weights["final_norm"]
        shard["lm_head"] = weights["lm_head"]
        shards.append(shard)
    return shards


def _alloc_peer_barriers(ops, device):
    n = sum(
        op.total_tiles
        for op in ops
        if getattr(op.op_cls, "peer_stores", set()) or getattr(op.op_cls, "peer_reduce_stores", set())
    )
    return torch.zeros(max(n, 1), dtype=torch.int32, device=device)


def _tensor_canonical_name(tensor_registry, tensor):
    ptr = tensor.data_ptr()
    for canonical, registered_tensor, _dtype in tensor_registry.tensors:
        if registered_tensor.data_ptr() == ptr:
            return canonical
    raise KeyError("Tensor not found in registry")


def _make_tp_peer_map(ops, mirrored_tensors):
    from machete.megakernel.registries import TensorRegistry

    tensor_registry = TensorRegistry.from_ops(ops)
    peer_map = {}
    for local_tensor, peer_tensor in mirrored_tensors:
        peer_map[_tensor_canonical_name(tensor_registry, local_tensor)] = [peer_tensor]
    return peer_map


def _schedule_tp_prefill_layer_ops(
    i,
    batch,
    seq_len,
    weights_tp,
    page_size,
    x_in,
    res_in,
    x_out,
    x_out_peer,
    res_mid,
    res_out,
    zero_buf,
    h_buf,
    h2_buf,
    q_bufs,
    k_bufs,
    v_bufs,
    attn_out_bufs,
    gate_up_bufs,
    mlp_h_bufs,
    lse_bufs,
):
    from machete.kernels.activation import AddOp
    from machete.kernels.attention import FlashAttentionSm120Op
    from machete.kernels.gemm import GemmOp
    from machete.kernels.glu import GLUOp
    from machete.kernels.qknorm_rope import QKNormRopeOp
    from machete.kernels.rms_norm import RMSNormOp

    pfx = f"layer.{i}"
    ops = []
    keep_alive = []
    max_fa_tpb = 0
    max_page_size = page_size

    ops += RMSNormOp.schedule(
        x=x_in,
        weight=weights_tp[0][f"{pfx}.attn_norm"],
        y=h_buf,
        residual_in=res_in,
        residual_out=res_mid,
        tile_sizes={"S": 16},
        page_size=page_size,
    )

    for rank in range(TP_DEGREE):
        ops += GemmOp.schedule_tp(tp_mode="column", a=h_buf, b=weights_tp[rank][f"{pfx}.W_q"], c=q_bufs[rank], page_size=page_size)
        ops += GemmOp.schedule_tp(tp_mode="column", a=h_buf, b=weights_tp[rank][f"{pfx}.W_k"], c=k_bufs[rank], page_size=page_size)
        ops += GemmOp.schedule_tp(tp_mode="column", a=h_buf, b=weights_tp[rank][f"{pfx}.W_v"], c=v_bufs[rank], page_size=page_size)

        q_4d = q_bufs[rank].view(batch * seq_len, LOCAL_Q_HEADS, HEAD_DIM)
        k_4d = k_bufs[rank].view(batch * seq_len, LOCAL_KV_HEADS, HEAD_DIM)
        q_fa = q_bufs[rank].view(batch, seq_len, LOCAL_Q_HEADS, HEAD_DIM)
        k_fa = k_bufs[rank].view(batch, seq_len, LOCAL_KV_HEADS, HEAD_DIM)
        v_fa = v_bufs[rank].view(batch, seq_len, LOCAL_KV_HEADS, HEAD_DIM)
        o_fa = attn_out_bufs[rank].view(batch, seq_len, LOCAL_Q_HEADS, HEAD_DIM)

        ops += QKNormRopeOp.schedule(
            q=q_4d,
            norm_weight=weights_tp[rank][f"{pfx}.w_q_norm"],
            cos=weights_tp[rank]["cos"],
            sin=weights_tp[rank]["sin"],
            page_size=page_size,
        )
        ops += QKNormRopeOp.schedule(
            q=k_4d,
            norm_weight=weights_tp[rank][f"{pfx}.w_k_norm"],
            cos=weights_tp[rank]["cos"],
            sin=weights_tp[rank]["sin"],
            page_size=page_size,
        )
        fa_ops, fa_config = flash_attention_schedule(
            q=q_fa,
            k=k_fa,
            v=v_fa,
            o=o_fa,
            lse=lse_bufs[rank],
            causal=True,
            kv_group_size=LOCAL_KV_GROUP_SIZE,
            page_size=page_size,
        )
        for op in fa_ops:
            op.dim_aliases["M"] = f"seq_tp_{i}_r{rank}"
        ops += fa_ops
        max_fa_tpb = max(max_fa_tpb, fa_config.threads_per_block)
        max_page_size = max(max_page_size, fa_config.page_size)
        keep_alive += [q_4d, k_4d, q_fa, k_fa, v_fa, o_fa]

    ops += AddOp.schedule(x=zero_buf, add=zero_buf, y=x_out, tile_sizes={"S": 16})
    for rank in range(TP_DEGREE):
        ops += GemmOp.schedule_tp(
            tp_mode="row",
            a=attn_out_bufs[rank],
            b=weights_tp[rank][f"{pfx}.W_o"],
            c=x_out,
            page_size=page_size,
        )

    ops += RMSNormOp.schedule(
        x=x_out,
        weight=weights_tp[0][f"{pfx}.mlp_norm"],
        y=h2_buf,
        residual_in=res_mid,
        residual_out=res_out,
        tile_sizes={"S": 16},
        page_size=page_size,
    )

    for rank in range(TP_DEGREE):
        ops += GemmOp.schedule_tp(
            tp_mode="column",
            a=h2_buf,
            b=weights_tp[rank][f"{pfx}.W_gate_up"],
            c=gate_up_bufs[rank],
            page_size=page_size,
        )
        ops += GLUOp.schedule(
            x=gate_up_bufs[rank],
            y=mlp_h_bufs[rank],
            activation="silu",
            page_size=page_size,
        )

    ops += AddOp.schedule(x=zero_buf, add=zero_buf, y=x_out, tile_sizes={"S": 16})
    for rank in range(TP_DEGREE):
        ops += GemmOp.schedule_tp(
            tp_mode="row",
            a=mlp_h_bufs[rank],
            b=weights_tp[rank][f"{pfx}.W_down"],
            c=x_out,
            page_size=page_size,
        )

    keep_alive += [x_out_peer]
    return ops, keep_alive, max_fa_tpb, max_page_size


def megakernel_tp_prefill_build(batch, seq_len, x, residual, weights, page_size=32768, num_layers=NUM_LAYERS):
    from machete.kernels.gemm import GemmOp
    from machete.kernels.rms_norm import RMSNormOp
    from machete.megakernel import Megakernel, MegakernelConfig

    dtype = x.dtype
    device = x.device
    weights_tp = _shard_tp_weights(weights)

    x_buf = [torch.empty(batch, seq_len, HIDDEN, dtype=dtype, device=device) for _ in range(2)]
    x_peer = [torch.zeros_like(x_buf[0]) for _ in range(2)]
    res_buf = [torch.empty(batch, seq_len, HIDDEN, dtype=dtype, device=device) for _ in range(2)]
    x_buf[0].copy_(x)
    res_buf[0].copy_(residual)

    zero_buf = torch.zeros(batch, seq_len, HIDDEN, dtype=dtype, device=device)
    res_mid = torch.empty(batch, seq_len, HIDDEN, dtype=dtype, device=device)
    h_buf = torch.empty(batch, seq_len, HIDDEN, dtype=dtype, device=device)
    h2_buf = torch.empty(batch, seq_len, HIDDEN, dtype=dtype, device=device)
    q_bufs = [torch.empty(batch, seq_len, LOCAL_Q_DIM, dtype=dtype, device=device) for _ in range(TP_DEGREE)]
    k_bufs = [torch.empty(batch, seq_len, LOCAL_KV_DIM, dtype=dtype, device=device) for _ in range(TP_DEGREE)]
    v_bufs = [torch.empty(batch, seq_len, LOCAL_KV_DIM, dtype=dtype, device=device) for _ in range(TP_DEGREE)]
    attn_out_bufs = [torch.empty(batch, seq_len, LOCAL_Q_DIM, dtype=dtype, device=device) for _ in range(TP_DEGREE)]
    gate_up_bufs = [torch.empty(batch, seq_len, 2 * LOCAL_INTERMEDIATE, dtype=dtype, device=device) for _ in range(TP_DEGREE)]
    mlp_h_bufs = [torch.empty(batch, seq_len, LOCAL_INTERMEDIATE, dtype=dtype, device=device) for _ in range(TP_DEGREE)]
    lse_bufs = [torch.empty(batch, LOCAL_Q_HEADS, seq_len, dtype=torch.float32, device=device) for _ in range(TP_DEGREE)]

    residual_final = torch.empty(batch, seq_len, HIDDEN, dtype=dtype, device=device)
    h_final = torch.empty(batch, seq_len, HIDDEN, dtype=dtype, device=device)
    logits = torch.empty(batch, seq_len, VOCAB_SIZE, dtype=dtype, device=device)

    keep_alive = list(weights.values()) + [x, residual, zero_buf, res_mid, h_buf, h2_buf, residual_final, h_final, logits]
    keep_alive += [*x_buf, *x_peer, *res_buf, *q_bufs, *k_bufs, *v_bufs, *attn_out_bufs, *gate_up_bufs, *mlp_h_bufs, *lse_bufs]

    all_ops = []
    max_fa_tpb = 0
    max_page_size = page_size
    for i in range(num_layers):
        cur_x = x_buf[i % 2]
        next_x = x_buf[(i + 1) % 2]
        next_x_peer = x_peer[(i + 1) % 2]
        cur_res = res_buf[i % 2]
        next_res = res_buf[(i + 1) % 2]
        layer_ops, extra_keep, fa_tpb, fa_page = _schedule_tp_prefill_layer_ops(
            i,
            batch,
            seq_len,
            weights_tp,
            page_size,
            cur_x,
            cur_res,
            next_x,
            next_x_peer,
            res_mid,
            next_res,
            zero_buf,
            h_buf,
            h2_buf,
            q_bufs,
            k_bufs,
            v_bufs,
            attn_out_bufs,
            gate_up_bufs,
            mlp_h_bufs,
            lse_bufs,
        )
        all_ops += layer_ops
        keep_alive += extra_keep
        max_fa_tpb = max(max_fa_tpb, fa_tpb)
        max_page_size = max(max_page_size, fa_page)

    final_x = x_buf[num_layers % 2]
    final_res = res_buf[num_layers % 2]
    all_ops += RMSNormOp.schedule(
        x=final_x,
        weight=weights["final_norm"],
        y=h_final,
        residual_in=final_res,
        residual_out=residual_final,
        tile_sizes={"S": 16},
        page_size=max_page_size,
    )
    all_ops += GemmOp.schedule(a=h_final, b=weights["lm_head"], c=logits, page_size=max_page_size)

    gemm_like_ops = [op for op in all_ops if op.tile_sizes.get("S") is not None]
    gemm_config = GemmOp.kernel_config(gemm_like_ops or all_ops)
    config = MegakernelConfig(
        threads_per_block=_pick_single_layer_forward_tpb(batch, seq_len, gemm_config.threads_per_block, max_fa_tpb),
        page_size=max_page_size,
        num_pages=1,
        mma_reg_count=_pick_single_layer_forward_mma_reg_count(batch, seq_len),
        peer_buffers=_make_tp_peer_map(all_ops, [(x_buf[0], x_peer[0]), (x_buf[1], x_peer[1])]),
        peer_barriers=_alloc_peer_barriers(all_ops, device=device),
        device_idx=0,
        num_devices=2,
    )

    kernel = Megakernel(all_ops, config=config)
    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()
    torch.cuda.synchronize()
    return kernel.bench_spec(keep_alive=keep_alive), logits, residual_final


def _split_tp_caches(k_caches, v_caches, tp_degree=TP_DEGREE):
    k_split = [[] for _ in range(tp_degree)]
    v_split = [[] for _ in range(tp_degree)]
    for i in range(NUM_LAYERS):
        k_chunks = k_caches[i].chunk(tp_degree, dim=1)
        v_chunks = v_caches[i].chunk(tp_degree, dim=1)
        for rank in range(tp_degree):
            k_split[rank].append(k_chunks[rank].contiguous())
            v_split[rank].append(v_chunks[rank].contiguous())
    return k_split, v_split


def _schedule_tp_decode_layer_ops(
    i,
    batch,
    pos,
    step,
    k_caches_tp,
    v_caches_tp,
    weights_tp,
    page_size,
    x_in,
    res_in,
    x_out,
    x_out_peer,
    zero_buf,
    res_out,
    h_buf,
    h2_buf,
    q_bufs,
    k_scratch_bufs,
    v_scratch_bufs,
    attn_out_bufs,
    gate_up_bufs,
    mlp_h_bufs,
    lse_bufs,
):
    from machete.kernels.activation import AddOp
    from machete.kernels.attention import FlashAttentionSm120Op
    from machete.kernels.gemm import GemmOp
    from machete.kernels.glu import GLUOp
    from machete.kernels.qknorm_rope import QKNormRopeOp
    from machete.kernels.rms_norm import RMSNormOp

    pfx = f"layer.{i}"
    ops = []
    keep_alive = []
    max_fa_tpb = 0
    max_page_size = page_size

    ops += RMSNormOp.schedule(
        x=x_in,
        weight=weights_tp[0][f"{pfx}.attn_norm"],
        y=h_buf,
        residual_in=res_in,
        residual_out=res_out,
        tile_sizes={"S": 1},
        page_size=page_size,
    )

    for rank in range(TP_DEGREE):
        ops += GemmOp.schedule_tp(tp_mode="column", a=h_buf, b=weights_tp[rank][f"{pfx}.W_q"], c=q_bufs[rank], page_size=page_size)
        ops += GemmOp.schedule_tp(tp_mode="column", a=h_buf, b=weights_tp[rank][f"{pfx}.W_k"], c=k_scratch_bufs[rank], page_size=page_size)
        ops += GemmOp.schedule_tp(tp_mode="column", a=h_buf, b=weights_tp[rank][f"{pfx}.W_v"], c=v_scratch_bufs[rank], page_size=page_size)

        q_4d = q_bufs[rank].view(batch * step, LOCAL_Q_HEADS, HEAD_DIM)
        k_4d = k_scratch_bufs[rank].view(batch * step, LOCAL_KV_HEADS, HEAD_DIM)
        q_fa = q_bufs[rank].view(batch, step, LOCAL_Q_HEADS, HEAD_DIM)
        k_full = k_caches_tp[rank][i][:, :, :pos + step, :].permute(0, 2, 1, 3).contiguous()
        v_full = v_caches_tp[rank][i][:, :, :pos + step, :].permute(0, 2, 1, 3).contiguous()
        o_fa = attn_out_bufs[rank].view(batch, step, LOCAL_Q_HEADS, HEAD_DIM)

        ops += QKNormRopeOp.schedule(
            q=q_4d,
            norm_weight=weights_tp[rank][f"{pfx}.w_q_norm"],
            cos=weights_tp[rank]["cos"],
            sin=weights_tp[rank]["sin"],
            page_size=page_size,
        )
        ops += QKNormRopeOp.schedule(
            q=k_4d,
            norm_weight=weights_tp[rank][f"{pfx}.w_k_norm"],
            cos=weights_tp[rank]["cos"],
            sin=weights_tp[rank]["sin"],
            page_size=page_size,
        )
        fa_ops = FlashAttentionSm120Op.schedule(
            q=q_fa,
            k=k_full,
            v=v_full,
            o=o_fa,
            lse=lse_bufs[rank],
            causal=False,
            kv_group_size=LOCAL_KV_GROUP_SIZE,
            page_size=page_size,
        )
        fa_config = FlashAttentionSm120Op.kernel_config(fa_ops)
        for op in fa_ops:
            op.dim_aliases["M"] = f"fa_tp_{i}_r{rank}"
        ops += fa_ops
        max_fa_tpb = max(max_fa_tpb, fa_config.threads_per_block)
        max_page_size = max(max_page_size, fa_config.page_size)
        keep_alive += [q_4d, k_4d, q_fa, k_full, v_full, o_fa, lse_bufs[rank]]

    ops += AddOp.schedule(x=zero_buf, add=zero_buf, y=x_out, tile_sizes={"S": 1})
    for rank in range(TP_DEGREE):
        ops += GemmOp.schedule_tp(tp_mode="row", a=attn_out_bufs[rank], b=weights_tp[rank][f"{pfx}.W_o"], c=x_out, page_size=page_size)

    ops += RMSNormOp.schedule(
        x=x_out,
        weight=weights_tp[0][f"{pfx}.mlp_norm"],
        y=h2_buf,
        residual_in=res_out,
        residual_out=res_out,
        tile_sizes={"S": 1},
        page_size=page_size,
    )

    for rank in range(TP_DEGREE):
        ops += GemmOp.schedule_tp(tp_mode="column", a=h2_buf, b=weights_tp[rank][f"{pfx}.W_gate_up"], c=gate_up_bufs[rank], page_size=page_size)
        ops += GLUOp.schedule(
            x=gate_up_bufs[rank],
            y=mlp_h_bufs[rank],
            activation="silu",
            tile_sizes={"S": 1},
            page_size=page_size,
        )

    ops += AddOp.schedule(x=zero_buf, add=zero_buf, y=x_out, tile_sizes={"S": 1})
    for rank in range(TP_DEGREE):
        ops += GemmOp.schedule_tp(tp_mode="row", a=mlp_h_bufs[rank], b=weights_tp[rank][f"{pfx}.W_down"], c=x_out, page_size=page_size)

    keep_alive += [x_out_peer]
    return ops, keep_alive, max_fa_tpb, max_page_size


def megakernel_tp_decode_build(batch, pos, k_caches, v_caches, weights, x_init, residual_init, page_size=32768, num_layers=NUM_LAYERS):
    from machete.kernels.gemm import GemmOp
    from machete.kernels.rms_norm import RMSNormOp
    from machete.megakernel import Megakernel, MegakernelConfig

    dtype = x_init.dtype
    device = x_init.device
    step = x_init.shape[1]
    weights_tp = _shard_tp_weights(weights)
    k_caches_tp, v_caches_tp = _split_tp_caches(k_caches, v_caches)

    x_buf = [torch.empty(batch, step, HIDDEN, dtype=dtype, device=device) for _ in range(2)]
    x_peer = [torch.zeros_like(x_buf[0]) for _ in range(2)]
    res_buf = [torch.empty(batch, step, HIDDEN, dtype=dtype, device=device) for _ in range(2)]
    x_buf[0].copy_(x_init)
    res_buf[0].copy_(residual_init)

    zero_buf = torch.zeros(batch, step, HIDDEN, dtype=dtype, device=device)
    res_mid = torch.empty(batch, step, HIDDEN, dtype=dtype, device=device)
    h_buf = torch.empty(batch, step, HIDDEN, dtype=dtype, device=device)
    h2_buf = torch.empty(batch, step, HIDDEN, dtype=dtype, device=device)
    q_bufs = [torch.empty(batch, step, LOCAL_Q_DIM, dtype=dtype, device=device) for _ in range(TP_DEGREE)]
    k_scratch_bufs = [torch.empty(batch, step, LOCAL_KV_DIM, dtype=dtype, device=device) for _ in range(TP_DEGREE)]
    v_scratch_bufs = [torch.empty(batch, step, LOCAL_KV_DIM, dtype=dtype, device=device) for _ in range(TP_DEGREE)]
    attn_out_bufs = [torch.empty(batch, step, LOCAL_Q_DIM, dtype=dtype, device=device) for _ in range(TP_DEGREE)]
    gate_up_bufs = [torch.empty(batch, step, 2 * LOCAL_INTERMEDIATE, dtype=dtype, device=device) for _ in range(TP_DEGREE)]
    mlp_h_bufs = [torch.empty(batch, step, LOCAL_INTERMEDIATE, dtype=dtype, device=device) for _ in range(TP_DEGREE)]
    lse_bufs = [torch.empty(batch, LOCAL_Q_HEADS, step, dtype=torch.float32, device=device) for _ in range(TP_DEGREE)]

    residual_final = torch.empty(batch, step, HIDDEN, dtype=dtype, device=device)
    h_final = torch.empty(batch, step, HIDDEN, dtype=dtype, device=device)
    logits = torch.empty(batch, step, VOCAB_SIZE, dtype=dtype, device=device)

    keep_alive = list(weights.values()) + k_caches + v_caches + [x_init, residual_init, zero_buf, res_mid, h_buf, h2_buf, residual_final, h_final, logits]
    keep_alive += [*x_buf, *x_peer, *res_buf, *q_bufs, *k_scratch_bufs, *v_scratch_bufs, *attn_out_bufs, *gate_up_bufs, *mlp_h_bufs, *lse_bufs]
    for rank in range(TP_DEGREE):
        keep_alive += k_caches_tp[rank] + v_caches_tp[rank]

    all_ops = []
    max_fa_tpb = 0
    max_page_size = page_size
    for i in range(num_layers):
        cur_x = x_buf[i % 2]
        next_x = x_buf[(i + 1) % 2]
        next_x_peer = x_peer[(i + 1) % 2]
        cur_res = res_buf[i % 2]
        next_res = res_buf[(i + 1) % 2]
        layer_ops, extra_keep, fa_tpb, fa_page = _schedule_tp_decode_layer_ops(
            i,
            batch,
            pos,
            step,
            k_caches_tp,
            v_caches_tp,
            weights_tp,
            page_size,
            cur_x,
            cur_res,
            next_x,
            next_x_peer,
            zero_buf,
            next_res,
            h_buf,
            h2_buf,
            q_bufs,
            k_scratch_bufs,
            v_scratch_bufs,
            attn_out_bufs,
            gate_up_bufs,
            mlp_h_bufs,
            lse_bufs,
        )
        all_ops += layer_ops
        keep_alive += extra_keep
        max_fa_tpb = max(max_fa_tpb, fa_tpb)
        max_page_size = max(max_page_size, fa_page)

    final_x = x_buf[num_layers % 2]
    final_res = res_buf[num_layers % 2]
    all_ops += RMSNormOp.schedule(
        x=final_x,
        weight=weights["final_norm"],
        y=h_final,
        residual_in=final_res,
        residual_out=residual_final,
        tile_sizes={"S": 1},
        page_size=max_page_size,
    )
    all_ops += GemmOp.schedule(a=h_final, b=weights["lm_head"], c=logits, page_size=max_page_size)

    gemm_like_ops = [op for op in all_ops if op.tile_sizes.get("S") is not None]
    gemm_config = GemmOp.kernel_config(gemm_like_ops or all_ops)
    config = MegakernelConfig(
        threads_per_block=max(gemm_config.threads_per_block, max_fa_tpb),
        page_size=max_page_size,
        num_pages=1,
        peer_buffers=_make_tp_peer_map(all_ops, [(x_buf[0], x_peer[0]), (x_buf[1], x_peer[1])]),
        peer_barriers=_alloc_peer_barriers(all_ops, device=device),
        device_idx=0,
        num_devices=2,
    )

    kernel = Megakernel(all_ops, config=config)
    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()
    torch.cuda.synchronize()
    return kernel.bench_spec(keep_alive=keep_alive), logits, residual_final


def verify_tp_prefill(batch=1, seq_len=128, num_layers=2, page_size=32768):
    torch.manual_seed(42)
    dtype = torch.bfloat16
    device = "cuda"
    weights = allocate_model_weights(dtype=dtype, device=device)
    x = torch.randn(batch, seq_len, HIDDEN, dtype=dtype, device=device)
    residual = torch.randn(batch, seq_len, HIDDEN, dtype=dtype, device=device)

    ref_logits, ref_residual = sequential_prefill(x, residual, weights, num_layers=num_layers)
    _spec, logits, residual_out = megakernel_tp_prefill_build(
        batch, seq_len, x, residual, weights, page_size=page_size, num_layers=num_layers
    )
    return {
        "logits": _rel_mean(logits, ref_logits),
        "residual": _rel_mean(residual_out, ref_residual),
    }


def verify_tp_decode(batch=1, context_len=128, num_layers=2, page_size=32768):
    torch.manual_seed(42)
    dtype = torch.bfloat16
    device = "cuda"
    weights = allocate_model_weights(dtype=dtype, device=device)
    k_caches, v_caches = allocate_kv_cache(batch, context_len + DECODE_S, dtype=dtype, device=device)
    for i in range(NUM_LAYERS):
        k_caches[i][:, :, :context_len, :].normal_()
        v_caches[i][:, :, :context_len, :].normal_()
    x = torch.randn(batch, DECODE_S, HIDDEN, dtype=dtype, device=device)
    residual = torch.zeros(batch, DECODE_S, HIDDEN, dtype=dtype, device=device)

    ref_k = [c.clone() for c in k_caches]
    ref_v = [c.clone() for c in v_caches]
    ref_logits, ref_residual = sequential_decode_step_layers(
        x, residual.clone(), context_len, ref_k, ref_v, weights, num_layers=num_layers
    )
    _spec, logits, residual_out = megakernel_tp_decode_build(
        batch,
        context_len,
        ref_k,
        ref_v,
        weights,
        x_init=x,
        residual_init=residual.clone(),
        page_size=page_size,
        num_layers=num_layers,
    )
    return {
        "logits": _rel_mean(logits, ref_logits),
        "residual": _rel_mean(residual_out, ref_residual),
    }


@Benchmark.parametrize("batch", [1])
@Benchmark.parametrize("seq_len", [128, 512])
@Benchmark.parametrize("page_size", PAGE_SIZES)
def bench_qwen35_prefill_tp(seq_len, batch, page_size):
    torch.manual_seed(42)
    dtype = torch.bfloat16
    device = "cuda"

    weights = allocate_model_weights(dtype=dtype, device=device)
    x = torch.randn(batch, seq_len, HIDDEN, dtype=dtype, device=device)
    residual = torch.randn(batch, seq_len, HIDDEN, dtype=dtype, device=device)

    funcs = {}
    args = (x, residual, weights)
    sequential_prefill(*args)
    torch.cuda.synchronize()
    funcs["sequential"] = lambda: sequential_prefill(*args)

    compiled_forward = maybe_compiled_forward(sequential_prefill, args)
    if compiled_forward is not None:
        funcs["torch_compile"] = lambda: compiled_forward(*args)

    if is_sm90_or_newer() and CUTLASS_AVAILABLE:
        spec, _logits, _residual = megakernel_tp_prefill_build(
            batch, seq_len, x, residual, weights, page_size=page_size
        )
        funcs["megakernel_tp"] = spec

    return funcs


@Benchmark.parametrize("batch", [1])
@Benchmark.parametrize("context_len", [128, 512, 1024])
@Benchmark.parametrize("page_size", PAGE_SIZES)
def bench_qwen35_decode_tp(context_len, batch, page_size):
    torch.manual_seed(42)
    dtype = torch.bfloat16
    device = "cuda"

    weights = allocate_model_weights(dtype=dtype, device=device)
    k_caches, v_caches = allocate_kv_cache(batch, context_len + DECODE_S, dtype=dtype, device=device)
    pos = context_len

    x = torch.randn(batch, DECODE_S, HIDDEN, dtype=dtype, device=device)
    residual = torch.zeros(batch, DECODE_S, HIDDEN, dtype=dtype, device=device)
    for i in range(NUM_LAYERS):
        k_caches[i][:, :, :pos, :].normal_()
        v_caches[i][:, :, :pos, :].normal_()

    funcs = {}
    funcs["sequential"] = lambda: sequential_decode_step(
        x,
        residual.clone(),
        pos,
        [c.clone() for c in k_caches],
        [c.clone() for c in v_caches],
        weights,
    )

    def decode_reference(x_arg, residual_arg):
        return sequential_decode_step(
            x_arg,
            residual_arg,
            pos,
            [c.clone() for c in k_caches],
            [c.clone() for c in v_caches],
            weights,
        )

    compiled_forward = maybe_compiled_forward(decode_reference, (x, residual.clone()))
    if compiled_forward is not None:
        funcs["torch_compile"] = lambda: compiled_forward(x, residual.clone())

    if is_sm90_or_newer() and CUTLASS_AVAILABLE:
        ref_k = [c.clone() for c in k_caches]
        ref_v = [c.clone() for c in v_caches]
        sequential_decode_step(x, residual.clone(), pos, ref_k, ref_v, weights)
        spec, _logits, _residual = megakernel_tp_decode_build(
            batch,
            pos,
            ref_k,
            ref_v,
            weights,
            x_init=x,
            residual_init=residual.clone(),
            page_size=page_size,
        )
        funcs["megakernel_tp"] = spec

    return funcs


if __name__ == "__main__":
    print("=" * 100)
    print("Qwen 3.5-0.8B Tensor-Parallel Benchmark (Same-Device Emulation)")
    print(f"  tp_degree={TP_DEGREE}, layers={NUM_LAYERS}, hidden={HIDDEN}, intermediate={INTERMEDIATE}")
    print("=" * 100)

    if is_sm90_or_newer() and CUTLASS_AVAILABLE:
        print("verify tp prefill:", verify_tp_prefill())
        print("verify tp decode:", verify_tp_decode())
    else:
        print("Requires Hopper+ GPU with CUTLASS.")

    print()
    print("-" * 80)
    print("Full-Model Prefill TP: Eager vs torch.compile vs Megakernel TP")
    print("-" * 80)
    bench_qwen35_prefill_tp._benchmark.run(mode="kernel", warmup=3, rep=10)

    print()
    print("-" * 80)
    print("Full-Model Decode TP: Eager vs torch.compile vs Megakernel TP")
    print("-" * 80)
    bench_qwen35_decode_tp._benchmark.run(mode="kernel", warmup=3, rep=10)
