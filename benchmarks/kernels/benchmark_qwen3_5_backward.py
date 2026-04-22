#!/usr/bin/env python
# Copyright (c) 2025, Machete Authors
"""Benchmark and validate full-model Qwen 3.5-0.8B backward megakernel.

Current scope is activation backward:
- full reverse graph over all layers
- gradients wrt model activations / residual stream
- no parameter-gradient coverage yet
"""

import contextlib
import io

import torch
import torch.nn.functional as F

from machete.utils.benchmark import Benchmark
from benchmarks.kernels.benchmark_qwen3_5_decode import (
    D2,
    EPS,
    HEAD_DIM,
    HIDDEN,
    INTERMEDIATE,
    KV_DIM,
    KV_GROUP_SIZE,
    NUM_KV_HEADS,
    NUM_LAYERS,
    NUM_Q_HEADS,
    Q_DIM,
    ROTARY_DIM,
    VOCAB_SIZE,
    allocate_model_weights,
    is_sm90_or_newer,
)
from benchmarks.kernels.benchmark_qwen3_5_layer import (
    _apply_rope,
    _per_head_rmsnorm,
    _pick_single_layer_backward_mma_reg_count,
    _pick_single_layer_backward_tpb,
    _rmsnorm,
)


PAGE_SIZES = [32768]


try:
    import cutlass  # noqa: F401

    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False


try:
    from machete.kernels.gemm import GemmOp as _BaseGemmOp
except ImportError:
    _BaseGemmOp = None


if _BaseGemmOp is not None:
    class LmHeadGemmOp(_BaseGemmOp):
        """Separate handler family for the final lm_head backward GEMM."""


else:
    class LmHeadGemmOp:
        pass


def sequential_prefill_backward(x, residual, weights, d_logits, num_layers=NUM_LAYERS):
    """Autograd reference for full-model activation backward."""
    x_ = x.detach().requires_grad_(True)
    residual_ = residual.detach().requires_grad_(True)
    x_cur, res_cur = x_, residual_

    for i in range(num_layers):
        pfx = f"layer.{i}"
        res_add = x_cur + res_cur
        h = _rmsnorm(res_add, weights[f"{pfx}.attn_norm"])
        q = torch.matmul(h, weights[f"{pfx}.W_q"].t())
        k = torch.matmul(h, weights[f"{pfx}.W_k"].t())
        v = torch.matmul(h, weights[f"{pfx}.W_v"].t())
        q = _per_head_rmsnorm(q, weights[f"{pfx}.w_q_norm"], NUM_Q_HEADS, HEAD_DIM)
        k = _per_head_rmsnorm(k, weights[f"{pfx}.w_k_norm"], NUM_KV_HEADS, HEAD_DIM)
        q = _apply_rope(q, weights["cos"], weights["sin"], NUM_Q_HEADS, HEAD_DIM, ROTARY_DIM)
        k = _apply_rope(k, weights["cos"], weights["sin"], NUM_KV_HEADS, HEAD_DIM, ROTARY_DIM)
        q4 = q.view(x.shape[0], x.shape[1], NUM_Q_HEADS, HEAD_DIM).transpose(1, 2)
        k4 = k.view(x.shape[0], x.shape[1], NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v4 = v.view(x.shape[0], x.shape[1], NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        k4 = k4.repeat_interleave(KV_GROUP_SIZE, dim=1)
        v4 = v4.repeat_interleave(KV_GROUP_SIZE, dim=1)
        attn_out = F.scaled_dot_product_attention(q4, k4, v4, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], Q_DIM)
        proj = torch.matmul(attn_out, weights[f"{pfx}.W_o"].t())
        res_out = proj + res_add
        h2 = _rmsnorm(res_out, weights[f"{pfx}.mlp_norm"])
        gate_up = torch.matmul(h2, weights[f"{pfx}.W_gate_up"].t())
        gate, up = gate_up.chunk(2, dim=-1)
        mlp_h = F.silu(gate) * up
        x_cur = torch.matmul(mlp_h, weights[f"{pfx}.W_down"].t())
        res_cur = res_out

    residual_final = x_cur + res_cur
    h_final = _rmsnorm(residual_final, weights["final_norm"])
    logits = torch.matmul(h_final, weights["lm_head"].t())
    logits.backward(d_logits)
    return x_.grad, residual_.grad


def _capture_forward_activations(batch, seq_len, x, residual, weights, page_size, num_layers):
    """Run sequential forward and store every activation needed by backward."""
    from machete.megakernel import Megakernel
    from machete.kernels.attention import FlashAttentionSm120Op

    dtype = x.dtype
    device = x.device
    activations = []
    x_cur = x
    res_cur = residual
    keep_alive = []

    for i in range(num_layers):
        pfx = f"layer.{i}"
        residual_out = (x_cur + res_cur).to(dtype)
        h = _rmsnorm(residual_out, weights[f"{pfx}.attn_norm"]).to(dtype)
        q_prenorm = torch.matmul(h.float(), weights[f"{pfx}.W_q"].float().t()).to(dtype)
        k_prenorm = torch.matmul(h.float(), weights[f"{pfx}.W_k"].float().t()).to(dtype)
        v_3d = torch.matmul(h.float(), weights[f"{pfx}.W_v"].float().t()).to(dtype)
        q_normed = _per_head_rmsnorm(q_prenorm, weights[f"{pfx}.w_q_norm"], NUM_Q_HEADS, HEAD_DIM)
        k_normed = _per_head_rmsnorm(k_prenorm, weights[f"{pfx}.w_k_norm"], NUM_KV_HEADS, HEAD_DIM)
        q_roped = _apply_rope(q_normed, weights["cos"], weights["sin"], NUM_Q_HEADS, HEAD_DIM, ROTARY_DIM)
        k_roped = _apply_rope(k_normed, weights["cos"], weights["sin"], NUM_KV_HEADS, HEAD_DIM, ROTARY_DIM)

        q_fa = q_roped.view(batch, seq_len, NUM_Q_HEADS, HEAD_DIM)
        k_fa = k_roped.view(batch, seq_len, NUM_KV_HEADS, HEAD_DIM)
        v_fa = v_3d.view(batch, seq_len, NUM_KV_HEADS, HEAD_DIM)
        attn_out_3d = torch.empty(batch, seq_len, Q_DIM, dtype=dtype, device=device)
        o_fa = attn_out_3d.view(batch, seq_len, NUM_Q_HEADS, HEAD_DIM)
        lse = torch.empty(batch, NUM_Q_HEADS, seq_len, dtype=torch.float32, device=device)

        fwd_ops = FlashAttentionSm120Op.schedule(
            q=q_fa, k=k_fa, v=v_fa, o=o_fa, lse=lse,
            causal=True, kv_group_size=KV_GROUP_SIZE, page_size=page_size,
        )
        fwd_kernel = Megakernel(fwd_ops, config=FlashAttentionSm120Op.kernel_config(fwd_ops))
        with contextlib.redirect_stdout(io.StringIO()):
            fwd_kernel.run()
        torch.cuda.synchronize()

        proj = torch.matmul(attn_out_3d.float(), weights[f"{pfx}.W_o"].float().t()).to(dtype)
        residual_out2 = (proj + residual_out).to(dtype)
        h2 = _rmsnorm(residual_out2, weights[f"{pfx}.mlp_norm"]).to(dtype)
        gate_up = torch.matmul(h2.float(), weights[f"{pfx}.W_gate_up"].float().t()).to(dtype)
        gate, up = gate_up.chunk(2, dim=-1)
        mlp_h = (F.silu(gate.float()) * up.float()).to(dtype)
        x_out = torch.matmul(mlp_h.float(), weights[f"{pfx}.W_down"].float().t()).to(dtype)

        act = {
            "x_in": x_cur,
            "res_in": res_cur,
            "residual_out": residual_out,
            "h": h,
            "q_prenorm": q_prenorm,
            "k_prenorm": k_prenorm,
            "q_roped": q_roped,
            "k_roped": k_roped,
            "v_3d": v_3d,
            "attn_out_3d": attn_out_3d,
            "lse": lse,
            "proj": proj,
            "residual_out2": residual_out2,
            "h2": h2,
            "gate_up": gate_up,
            "mlp_h": mlp_h,
        }
        activations.append(act)
        keep_alive.extend(act.values())
        x_cur = x_out
        res_cur = residual_out2

    residual_final = (x_cur + res_cur).to(dtype)
    h_final = _rmsnorm(residual_final, weights["final_norm"]).to(dtype)
    logits = torch.matmul(h_final.float(), weights["lm_head"].float().t()).to(dtype)
    tail = {
        "x_last": x_cur,
        "res_last": res_cur,
        "residual_final": residual_final,
        "h_final": h_final,
        "logits": logits,
    }
    keep_alive.extend(tail.values())
    return activations, tail, keep_alive


def megakernel_backward_build(batch, seq_len, x, residual, weights, d_logits, page_size=32768, num_layers=NUM_LAYERS):
    """Build one full-model activation-backward megakernel."""
    from machete.megakernel import Megakernel, MegakernelConfig
    from machete.kernels.activation import AddOp
    from machete.kernels.attention import AttentionDPSumOp, FlashAttentionSm120BwdOp
    from machete.kernels.gemm import GemmOp, GemmRowParallelOp
    from machete.kernels.glu import GLUBwdOp
    from machete.kernels.qknorm_rope import QKNormRopeBwdOp
    from machete.kernels.rms_norm.rms_norm import RMSNormBwdOp

    dtype = x.dtype
    device = x.device
    activations, tail, keep_alive = _capture_forward_activations(batch, seq_len, x, residual, weights, page_size, num_layers)
    keep_alive += list(weights.values()) + [x, residual, d_logits]

    all_ops = []
    max_fa_tpb = 0
    max_page_size = page_size

    # Final lm_head + norm tail.
    d_h_final = torch.empty_like(tail["h_final"])
    dx_out = torch.empty_like(tail["x_last"])
    dres_out = torch.empty_like(tail["res_last"])
    all_ops += LmHeadGemmOp.schedule_backward(
        dout=d_logits, a=tail["h_final"], b=weights["lm_head"], da=d_h_final, page_size=page_size
    )
    all_ops += RMSNormBwdOp.schedule(
        dout=d_h_final, x=tail["residual_final"], weight=weights["final_norm"],
        dx=dx_out, d_residual=dres_out, tile_sizes={"S": 16}, page_size=page_size,
    )
    keep_alive += [d_h_final, dx_out, dres_out]

    for i in range(num_layers - 1, -1, -1):
        pfx = f"layer.{i}"
        act = activations[i]
        d_mlp_h = torch.empty_like(act["mlp_h"])
        d_gate_up = torch.empty_like(act["gate_up"])
        d_h2 = torch.empty_like(act["h2"])
        d_proj = torch.empty_like(act["proj"])
        d_res2_local = torch.empty_like(act["residual_out"])
        d_res2_total = torch.empty_like(act["residual_out"])
        d_attn_out_3d = torch.empty_like(act["attn_out_3d"])
        d_attn_fa = d_attn_out_3d.view(batch, seq_len, NUM_Q_HEADS, HEAD_DIM)
        dpsum = torch.empty(batch, NUM_Q_HEADS, seq_len, dtype=torch.float32, device=device)
        dq_fa = torch.zeros(batch, seq_len, NUM_Q_HEADS, HEAD_DIM, dtype=torch.float32, device=device)
        dk_fa = torch.zeros(batch, seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=dtype, device=device)
        dv_fa = torch.zeros(batch, seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=dtype, device=device)
        q_fa = act["q_roped"].view(batch, seq_len, NUM_Q_HEADS, HEAD_DIM)
        k_fa = act["k_roped"].view(batch, seq_len, NUM_KV_HEADS, HEAD_DIM)
        v_fa = act["v_3d"].view(batch, seq_len, NUM_KV_HEADS, HEAD_DIM)
        o_fa = act["attn_out_3d"].view(batch, seq_len, NUM_Q_HEADS, HEAD_DIM)
        q_prenorm_4d = act["q_prenorm"].view(batch * seq_len, NUM_Q_HEADS, HEAD_DIM).contiguous()
        k_prenorm_4d = act["k_prenorm"].view(batch * seq_len, NUM_KV_HEADS, HEAD_DIM).contiguous()
        dq_fa_flat = dq_fa.view(batch * seq_len, NUM_Q_HEADS, HEAD_DIM).contiguous()
        dk_fa_flat = dk_fa.view(batch * seq_len, NUM_KV_HEADS, HEAD_DIM).contiguous()
        dq_qnr_q_flat = torch.empty_like(q_prenorm_4d)
        dq_qnr_k_flat = torch.empty_like(k_prenorm_4d)
        dq_qnr_q = dq_qnr_q_flat.view(batch, seq_len, Q_DIM)
        dq_qnr_k = dq_qnr_k_flat.view(batch, seq_len, KV_DIM)
        dv_3d = dv_fa.view(batch, seq_len, KV_DIM)
        d_h1 = torch.zeros(batch, seq_len, HIDDEN, dtype=dtype, device=device)
        dx_prev = torch.empty_like(act["x_in"])
        dres_prev = torch.empty_like(act["res_in"])
        W_q_t = weights[f"{pfx}.W_q"].t().contiguous()
        W_k_t = weights[f"{pfx}.W_k"].t().contiguous()
        W_v_t = weights[f"{pfx}.W_v"].t().contiguous()

        layer_ops = []
        layer_ops += GemmOp.schedule_backward(
            dout=dx_out, a=act["mlp_h"], b=weights[f"{pfx}.W_down"], da=d_mlp_h, page_size=page_size,
        )
        layer_ops += GLUBwdOp.schedule(
            dy=d_mlp_h, x=act["gate_up"], dx=d_gate_up, activation="silu", page_size=page_size,
        )
        layer_ops += GemmOp.schedule_backward(
            dout=d_gate_up, a=act["h2"], b=weights[f"{pfx}.W_gate_up"], da=d_h2, page_size=page_size,
        )
        layer_ops += RMSNormBwdOp.schedule(
            dout=d_h2, x=act["residual_out2"], weight=weights[f"{pfx}.mlp_norm"],
            dx=d_proj, d_residual=d_res2_total, add=dres_out,
            tile_sizes={"S": 16}, page_size=page_size,
        )
        layer_ops += GemmOp.schedule_backward(
            dout=d_proj, a=act["attn_out_3d"], b=weights[f"{pfx}.W_o"], da=d_attn_out_3d, page_size=page_size,
        )
        layer_ops += AttentionDPSumOp.schedule(
            dout=d_attn_fa, o=o_fa, dpsum=dpsum,
            tile_sizes={"S": 16, "H": 1}, page_size=page_size,
        )
        fa_bwd_ops = FlashAttentionSm120BwdOp.schedule(
            k=k_fa, v=v_fa, q=q_fa,
            dout=d_attn_fa, lse=act["lse"], dpsum=dpsum,
            dq=dq_fa, dk=dk_fa, dv=dv_fa,
            causal=True, kv_group_size=KV_GROUP_SIZE,
        )
        for op in fa_bwd_ops:
            op.dim_aliases["M"] = f"seq_bwd_{i}"
        layer_ops += fa_bwd_ops
        fa_bwd_config = FlashAttentionSm120BwdOp.kernel_config(fa_bwd_ops)
        max_fa_tpb = max(max_fa_tpb, fa_bwd_config.threads_per_block)
        max_page_size = max(max_page_size, fa_bwd_config.page_size)
        q_ops = QKNormRopeBwdOp.schedule(
            q=q_prenorm_4d, dout=dq_fa_flat, dq=dq_qnr_q_flat,
            norm_weight=weights[f"{pfx}.w_q_norm"], cos=weights["cos"], sin=weights["sin"], page_size=page_size,
        )
        k_ops = QKNormRopeBwdOp.schedule(
            q=k_prenorm_4d, dout=dk_fa_flat, dq=dq_qnr_k_flat,
            norm_weight=weights[f"{pfx}.w_k_norm"], cos=weights["cos"], sin=weights["sin"], page_size=page_size,
        )
        for op in q_ops + k_ops:
            op.dim_aliases["M"] = f"seq_bwd_{i}"
        layer_ops += q_ops + k_ops
        layer_ops += GemmRowParallelOp.schedule(a=dq_qnr_q, b=W_q_t, c=d_h1, page_size=page_size)
        layer_ops += GemmRowParallelOp.schedule(a=dq_qnr_k, b=W_k_t, c=d_h1, page_size=page_size)
        layer_ops += GemmRowParallelOp.schedule(a=dv_3d, b=W_v_t, c=d_h1, page_size=page_size)
        layer_ops += RMSNormBwdOp.schedule(
            dout=d_h1, x=act["residual_out"], weight=weights[f"{pfx}.attn_norm"],
            dx=dx_prev, d_residual=dres_prev, add=d_res2_total,
            tile_sizes={"S": 16}, page_size=page_size,
        )

        all_ops += layer_ops
        keep_alive += [
            d_mlp_h, d_gate_up, d_h2, d_proj, d_res2_local, d_res2_total,
            d_attn_out_3d, d_attn_fa, dpsum, dq_fa, dk_fa, dv_fa,
            q_prenorm_4d, k_prenorm_4d, dq_fa_flat, dk_fa_flat,
            dq_qnr_q_flat, dq_qnr_k_flat, dq_qnr_q, dq_qnr_k, dv_3d,
            d_h1, dx_prev, dres_prev, W_q_t, W_k_t, W_v_t,
        ]

        dx_out = dx_prev
        dres_out = dres_prev

    gemm_like_ops = [op for op in all_ops if op.tile_sizes.get("S") is not None]
    gemm_config = GemmOp.kernel_config(gemm_like_ops or all_ops)
    config = MegakernelConfig(
        threads_per_block=_pick_single_layer_backward_tpb(
            batch, seq_len, gemm_config.threads_per_block, max_fa_tpb or gemm_config.threads_per_block
        ),
        page_size=max_page_size,
        num_pages=1,
        mma_reg_count=_pick_single_layer_backward_mma_reg_count(batch, seq_len),
    )
    kernel = Megakernel(all_ops, config=config)
    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()
    torch.cuda.synchronize()
    return kernel.bench_spec(keep_alive=keep_alive), dx_out, dres_out


def verify_full_backward(batch, seq_len, num_layers=2, page_size=32768):
    torch.manual_seed(42)
    dtype = torch.bfloat16
    device = "cuda"
    weights = allocate_model_weights(dtype=dtype, device=device)
    x = torch.randn(batch, seq_len, HIDDEN, dtype=dtype, device=device)
    residual = torch.randn(batch, seq_len, HIDDEN, dtype=dtype, device=device)
    d_logits = torch.randn(batch, seq_len, VOCAB_SIZE, dtype=dtype, device=device)
    _, dx, dres = megakernel_backward_build(batch, seq_len, x, residual, weights, d_logits, page_size=page_size, num_layers=num_layers)
    ref_dx, ref_dres = sequential_prefill_backward(x, residual, weights, d_logits, num_layers=num_layers)
    rel = lambda a, b: float((a.float() - b.float()).abs().mean() / (b.float().abs().mean() + 1e-12))
    return {"dx": rel(dx, ref_dx), "d_res": rel(dres, ref_dres)}


@Benchmark.parametrize("batch", [1])
@Benchmark.parametrize("seq_len", [128])
@Benchmark.parametrize("page_size", PAGE_SIZES)
def bench_qwen35_backward(seq_len, batch, page_size):
    torch.manual_seed(42)
    dtype = torch.bfloat16
    device = "cuda"
    weights = allocate_model_weights(dtype=dtype, device=device)
    x = torch.randn(batch, seq_len, HIDDEN, dtype=dtype, device=device)
    residual = torch.randn(batch, seq_len, HIDDEN, dtype=dtype, device=device)
    d_logits = torch.randn(batch, seq_len, VOCAB_SIZE, dtype=dtype, device=device)
    spec, _dx, _dres = megakernel_backward_build(batch, seq_len, x, residual, weights, d_logits, page_size=page_size)
    funcs = {"sequential": lambda: sequential_prefill_backward(x, residual, weights, d_logits)}
    funcs["megakernel"] = spec
    return funcs


if __name__ == "__main__":
    if is_sm90_or_newer() and CUTLASS_AVAILABLE:
        print("verify 1 layer:", verify_full_backward(1, 128, num_layers=1))
        print("verify 2 layers:", verify_full_backward(1, 128, num_layers=2))
    else:
        print("Requires Hopper+ GPU with CUTLASS.")
