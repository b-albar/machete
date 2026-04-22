#!/usr/bin/env python
# Copyright (c) 2025, Machete Authors
"""Benchmark full-model Qwen 3.5-0.8B prefill.

This complements the dedicated full-model decode benchmark:
- decode: one full-model decode step as a single megakernel
- prefill: one full prompt pass through all 36 layers as a single megakernel
"""

import contextlib
import io

import torch

from machete.utils.benchmark import Benchmark
from benchmarks.kernels.benchmark_qwen3_5_decode import (
    HIDDEN,
    NUM_LAYERS,
    NUM_Q_HEADS,
    NUM_KV_HEADS,
    HEAD_DIM,
    KV_GROUP_SIZE,
    Q_DIM,
    KV_DIM,
    INTERMEDIATE,
    allocate_model_weights,
    is_sm90_or_newer,
)
from benchmarks.kernels.benchmark_qwen3_5_layer import (
    sequential_forward,
    _pick_single_layer_forward_mma_reg_count,
    _pick_single_layer_forward_tpb,
)

try:
    import cutlass  # noqa: F401

    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False


PAGE_SIZES = [32768, 49152]


try:
    from machete.kernels.gemm import GemmOp as _BaseGemmOp
except ImportError:
    _BaseGemmOp = None


if _BaseGemmOp is not None:
    class LmHeadGemmOp(_BaseGemmOp):
        """Separate GEMM handler family for the final lm_head projection."""


else:
    class LmHeadGemmOp:
        """Import-safe placeholder used only when Machete is not importable."""



def sequential_prefill(x, residual, weights, num_layers=NUM_LAYERS):
    """Run the full Qwen 3.5 prefill path sequentially."""
    for i in range(num_layers):
        pfx = f"layer.{i}"
        x, residual = sequential_forward(
            x,
            residual,
            weights[f"{pfx}.attn_norm"],
            weights[f"{pfx}.W_q"],
            weights[f"{pfx}.W_k"],
            weights[f"{pfx}.W_v"],
            weights[f"{pfx}.w_q_norm"],
            weights[f"{pfx}.w_k_norm"],
            weights["cos"],
            weights["sin"],
            weights[f"{pfx}.W_o"],
            weights[f"{pfx}.mlp_norm"],
            weights[f"{pfx}.W_gate_up"],
            weights[f"{pfx}.W_down"],
        )

    residual = x + residual
    variance = residual.float().pow(2).mean(-1, keepdim=True)
    h_final = (residual.float() * torch.rsqrt(variance + 1e-6) * weights["final_norm"].float()).to(residual.dtype)
    logits = torch.matmul(h_final, weights["lm_head"].t())
    return logits, residual


def _schedule_prefill_layer_ops(
    i,
    batch,
    seq_len,
    weights,
    page_size,
    x_in,
    res_in,
    x_out,
    res_mid,
    res_out,
    h_buf,
    q_buf,
    k_buf,
    v_buf,
    attn_out_buf,
    proj_buf,
    h2_buf,
    gate_up_buf,
    mlp_h_buf,
    lse_buf,
):
    """Schedule one full prefill layer into the shared full-model op stream."""
    from machete.kernels.attention import flash_attention_schedule
    from machete.kernels.gemm import GemmOp
    from machete.kernels.glu import GLUOp
    from machete.kernels.qknorm_rope import QKNormRopeOp
    from machete.kernels.rms_norm import RMSNormOp

    pfx = f"layer.{i}"
    cos, sin = weights["cos"], weights["sin"]

    q_4d = q_buf.view(batch * seq_len, NUM_Q_HEADS, HEAD_DIM)
    k_4d = k_buf.view(batch * seq_len, NUM_KV_HEADS, HEAD_DIM)

    q_fa = q_buf.view(batch, seq_len, NUM_Q_HEADS, HEAD_DIM)
    k_fa = k_buf.view(batch, seq_len, NUM_KV_HEADS, HEAD_DIM)
    v_fa = v_buf.view(batch, seq_len, NUM_KV_HEADS, HEAD_DIM)
    o_fa = attn_out_buf.view(batch, seq_len, NUM_Q_HEADS, HEAD_DIM)

    fa_ops, fa_config = flash_attention_schedule(
        q=q_fa,
        k=k_fa,
        v=v_fa,
        o=o_fa,
        lse=lse_buf,
        causal=True,
        kv_group_size=KV_GROUP_SIZE,
        page_size=page_size,
    )
    for op in fa_ops:
        op.dim_aliases["M"] = f"seq_{i}"

    fa_page_size = fa_config.page_size

    ops = []
    ops += RMSNormOp.schedule(
        x=x_in,
        weight=weights[f"{pfx}.attn_norm"],
        y=h_buf,
        residual_in=res_in,
        residual_out=res_mid,
        tile_sizes={"S": 16},
        page_size=fa_page_size,
    )
    ops += GemmOp.schedule(a=h_buf, b=weights[f"{pfx}.W_q"], c=q_buf, page_size=fa_page_size)
    ops += GemmOp.schedule(a=h_buf, b=weights[f"{pfx}.W_k"], c=k_buf, page_size=fa_page_size)
    ops += GemmOp.schedule(a=h_buf, b=weights[f"{pfx}.W_v"], c=v_buf, page_size=fa_page_size)
    ops += QKNormRopeOp.schedule(q=q_4d, norm_weight=weights[f"{pfx}.w_q_norm"], cos=cos, sin=sin, page_size=fa_page_size)
    ops += QKNormRopeOp.schedule(q=k_4d, norm_weight=weights[f"{pfx}.w_k_norm"], cos=cos, sin=sin, page_size=fa_page_size)
    ops += fa_ops
    ops += GemmOp.schedule(a=attn_out_buf, b=weights[f"{pfx}.W_o"], c=proj_buf, page_size=fa_page_size)
    ops += RMSNormOp.schedule(
        x=proj_buf,
        weight=weights[f"{pfx}.mlp_norm"],
        y=h2_buf,
        residual_in=res_mid,
        residual_out=res_out,
        tile_sizes={"S": 16},
        page_size=fa_page_size,
    )
    ops += GemmOp.schedule(a=h2_buf, b=weights[f"{pfx}.W_gate_up"], c=gate_up_buf, page_size=fa_page_size)
    ops += GLUOp.schedule(x=gate_up_buf, y=mlp_h_buf, activation="silu", tile_sizes={"S": 2}, page_size=fa_page_size)
    ops += GemmOp.schedule(a=mlp_h_buf, b=weights[f"{pfx}.W_down"], c=x_out, page_size=fa_page_size)

    extra_keep = [q_4d, k_4d, q_fa, k_fa, v_fa, o_fa]
    return ops, fa_config, extra_keep


def megakernel_prefill_build(batch, seq_len, x, residual, weights, page_size=32768, num_layers=NUM_LAYERS):
    """Build a single full-model prefill megakernel."""
    from machete.megakernel import Megakernel, MegakernelConfig
    from machete.kernels.gemm import GemmOp
    from machete.kernels.rms_norm import RMSNormOp

    dtype = x.dtype
    device = x.device

    x_buf = [torch.empty(batch, seq_len, HIDDEN, dtype=dtype, device=device) for _ in range(2)]
    res_buf = [torch.empty(batch, seq_len, HIDDEN, dtype=dtype, device=device) for _ in range(2)]
    x_buf[0].copy_(x)
    res_buf[0].copy_(residual)

    res_mid_bufs = [
        torch.empty(batch, seq_len, HIDDEN, dtype=dtype, device=device)
        for _ in range(max(1, num_layers))
    ]
    h_buf = torch.empty(batch, seq_len, HIDDEN, dtype=dtype, device=device)
    q_buf = torch.empty(batch, seq_len, Q_DIM, dtype=dtype, device=device)
    k_buf = torch.empty(batch, seq_len, KV_DIM, dtype=dtype, device=device)
    v_buf = torch.empty(batch, seq_len, KV_DIM, dtype=dtype, device=device)
    attn_out_buf = torch.empty(batch, seq_len, Q_DIM, dtype=dtype, device=device)
    lse_buf = torch.empty(batch, NUM_Q_HEADS, seq_len, dtype=torch.float32, device=device)
    proj_buf = torch.empty(batch, seq_len, HIDDEN, dtype=dtype, device=device)
    h2_buf = torch.empty(batch, seq_len, HIDDEN, dtype=dtype, device=device)
    gate_up_buf = torch.empty(batch, seq_len, 2 * INTERMEDIATE, dtype=dtype, device=device)
    mlp_h_buf = torch.empty(batch, seq_len, INTERMEDIATE, dtype=dtype, device=device)

    residual_final = torch.empty(batch, seq_len, HIDDEN, dtype=x.dtype, device=x.device)
    h_final = torch.empty(batch, seq_len, HIDDEN, dtype=x.dtype, device=x.device)
    logits = torch.empty(batch, seq_len, weights["lm_head"].shape[0], dtype=x.dtype, device=x.device)

    keep_alive = list(weights.values()) + [
        x,
        residual,
        *x_buf,
        *res_buf,
        *res_mid_bufs,
        h_buf,
        q_buf,
        k_buf,
        v_buf,
        attn_out_buf,
        lse_buf,
        proj_buf,
        h2_buf,
        gate_up_buf,
        mlp_h_buf,
        residual_final,
        h_final,
        logits,
    ]

    all_ops = []
    max_fa_tpb = 0
    fa_page_size = page_size

    for i in range(num_layers):
        cur_x = x_buf[i % 2]
        cur_res = res_buf[i % 2]
        next_x = x_buf[(i + 1) % 2]
        next_res = res_buf[(i + 1) % 2]
        layer_ops, fa_config, extra_keep = _schedule_prefill_layer_ops(
            i,
            batch,
            seq_len,
            weights,
            page_size,
            cur_x,
            cur_res,
            next_x,
            res_mid_bufs[i],
            next_res,
            h_buf,
            q_buf,
            k_buf,
            v_buf,
            attn_out_buf,
            proj_buf,
            h2_buf,
            gate_up_buf,
            mlp_h_buf,
            lse_buf,
        )
        all_ops += layer_ops
        max_fa_tpb = max(max_fa_tpb, fa_config.threads_per_block)
        fa_page_size = max(fa_page_size, fa_config.page_size)
        keep_alive += extra_keep

    final_x = x_buf[num_layers % 2]
    final_res = res_buf[num_layers % 2]
    all_ops += RMSNormOp.schedule(
        x=final_x,
        weight=weights["final_norm"],
        y=h_final,
        residual_in=final_res,
        residual_out=residual_final,
        tile_sizes={"S": 16},
        page_size=fa_page_size,
    )
    all_ops += LmHeadGemmOp.schedule(a=h_final, b=weights["lm_head"], c=logits, page_size=fa_page_size)

    gemm_like_ops = [op for op in all_ops if op.tile_sizes.get("S") is not None]
    gemm_config = GemmOp.kernel_config(gemm_like_ops or all_ops)
    config = MegakernelConfig(
        threads_per_block=_pick_single_layer_forward_tpb(
            batch,
            seq_len,
            gemm_config.threads_per_block,
            max_fa_tpb,
        ),
        page_size=fa_page_size,
        num_pages=1,
        mma_reg_count=_pick_single_layer_forward_mma_reg_count(batch, seq_len),
    )

    print(
        f"  Megakernel: {len(all_ops)} ops ({num_layers} layers), "
        f"{config.threads_per_block} threads, page={fa_page_size}"
    )

    kernel = Megakernel(all_ops, config=config)
    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()
    torch.cuda.synchronize()
    return kernel.bench_spec(keep_alive=keep_alive), logits, residual_final


@Benchmark.parametrize("batch", [1])
@Benchmark.parametrize("seq_len", [128, 512, 1024])
@Benchmark.parametrize("page_size", PAGE_SIZES)
def bench_qwen35_prefill(seq_len, batch, page_size):
    """Benchmark full-model Qwen 3.5 prefill."""
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

    if is_sm90_or_newer() and CUTLASS_AVAILABLE:
        try:
            spec, _, _ = megakernel_prefill_build(
                batch,
                seq_len,
                x,
                residual,
                weights,
                page_size=page_size,
            )
            funcs["megakernel"] = spec
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"  megakernel build failed: {e}")

    return funcs


if __name__ == "__main__":
    print("=" * 100)
    print("Qwen 3.5-0.8B Full-Model Prefill Benchmark")
    print(f"  {NUM_LAYERS} layers, hidden={HIDDEN}, intermediate={INTERMEDIATE}")
    print("=" * 100)
    bench_qwen35_prefill._benchmark.run(
        mode="kernel",
        warmup=3,
        rep=10,
    )
