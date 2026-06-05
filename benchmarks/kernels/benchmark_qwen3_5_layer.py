#!/usr/bin/env python
# Copyright (c) 2025, Machete Authors
"""Benchmark one full Qwen 3.5 full-attention layer as one megakernel.

This is the source-backed layer benchmark used by the Qwen graphics and trace
scripts.  It intentionally models one full-attention layer, not the old MLP-only
block benchmark.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from machete.kernels.attention import AttentionDPSumOp, FlashAttentionSm120BwdOp, FlashAttentionSm120Op
from machete.kernels.decode_matvec import ResidualAddSm120Op
from machete.kernels.gemm import GemmOp, ProjectionDaReduceGemmOp
from machete.kernels.glu import GLUBwdOp, GLUOp
from machete.kernels.qknorm_rope import PackedQKNormRopeOp, QKNormRopeBwdOp, QKNormRopeOp
from machete.kernels.qwen_3_5.qwen_3_5_forward import schedule_qwen3_5_forward_ops
from machete.kernels.qwen_3_5.sm120 import QWEN3_5_EPS
from machete.kernels.rms_norm import RMSNormBwdOp, RMSNormOp
from machete.megakernel import Megakernel, MegakernelConfig, OverlapTileScheduler
from machete.megakernel.dim_windows import iter_dim_windows
from machete.utils.benchmark import Benchmark
from machete.utils.benchmark_utils import KernelBenchSpec


DEFAULT_PAGE_SIZE = 32768
HIDDEN = 1024
INTERMEDIATE = 3584
NUM_Q_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = 256
Q_DIM = NUM_Q_HEADS * HEAD_DIM
KV_DIM = NUM_KV_HEADS * HEAD_DIM
KV_GROUP_SIZE = NUM_Q_HEADS // NUM_KV_HEADS
D2 = 32

CONFIGS = [
    (128, 1, DEFAULT_PAGE_SIZE),
    (512, 1, DEFAULT_PAGE_SIZE),
]

BENCH_USE_PACKED_QKV_PROJECTION = True
BENCH_USE_FUSED_RMS_PROJ = False
BENCH_USE_QKNORM_4D = True
BENCH_USE_TMA_ATTENTION = False
BENCH_USE_SPLIT_ATTENTION = False
BENCH_SPLIT_ATTENTION_SPLITS = 0
BENCH_ATTENTION_TILE_M = None


def _configs_with_fetch_stride(fetch_stride: int):
    return [
        (seq_len, batch, page_size, int(fetch_stride))
        for seq_len, batch, page_size in CONFIGS
    ]


def _config_for(
    ops,
    page_size: int,
    *,
    tracing: bool = False,
    num_pages: int | None = None,
    page_free_extra_slots: int = 0,
) -> MegakernelConfig:
    gemm_ops = [op for op in ops if issubclass(op.op_cls, GemmOp)]
    gemm_config = GemmOp.kernel_config(gemm_ops or ops)
    threads_per_block = max(256, gemm_config.threads_per_block)
    return MegakernelConfig(
        threads_per_block=threads_per_block,
        page_size=max(page_size, *(op.static_dims.get("page_size", page_size) for op in ops)),
        num_pages=num_pages,
        page_free_extra_slots=page_free_extra_slots,
        tracing=tracing,
    )


def _scheduler(variant: str, fetch_stride: int | None = None):
    if variant == "default":
        return None
    if variant in ("overlap", "overlap-adaptive"):
        resolved_fetch_stride = (
            int(fetch_stride)
            if fetch_stride is not None and int(fetch_stride) > 0
            else None
        )
        return OverlapTileScheduler(
            fetch_stride=resolved_fetch_stride,
            adaptive_fetch_stride=variant == "overlap-adaptive",
        )
    raise ValueError(f"unknown scheduler variant: {variant}")


def _rope_tables(seq_len: int, dtype: torch.dtype, device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    dims = torch.arange(D2, device=device, dtype=torch.float32)
    inv_freq = torch.pow(torch.tensor(10000000.0, device=device, dtype=torch.float32), -dims / D2)
    angles = pos[:, None] * inv_freq[None, :]
    return torch.cos(angles).to(dtype).contiguous(), torch.sin(angles).to(dtype).contiguous()

def megakernel_forward_build(
    batch: int,
    seq_len: int,
    x: torch.Tensor,
    residual: torch.Tensor,
    attn_norm: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    w_o: torch.Tensor,
    mlp_norm: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
    scheduler=None,
    tracing: bool = False,
    num_pages: int | None = None,
    page_free_extra_slots: int = 0,
    gemm_tile_sizes: dict[str, dict[str, int]] | None = None,
    rms_tile_s: int | None = None,
    adaptive_gemm_tiling: bool = False,
    use_packed_qk: bool = True,
    use_qwen_projection: bool = False,
    use_packed_qkv_projection: bool = True,
    use_fused_rms_proj: bool = False,
    use_qknorm_4d: bool = True,
    use_cpasync_projection: bool = False,
    use_tma_attention: bool = False,
    use_split_attention: bool = False,
    split_attention_splits: int = 0,
    attention_tile_m: int | None = None,
    forward_op_limit: int | None = None,
):
    forward = schedule_qwen3_5_forward_ops(
        batch,
        seq_len,
        x,
        residual,
        attn_norm,
        w_q,
        w_k,
        w_v,
        q_norm,
        k_norm,
        cos,
        sin,
        w_o,
        mlp_norm,
        w_gate_up,
        w_down,
        page_size=page_size,
        scheduler=scheduler,
        gemm_tile_sizes=gemm_tile_sizes,
        rms_tile_s=rms_tile_s,
        adaptive_gemm_tiling=adaptive_gemm_tiling,
        use_packed_qk=use_packed_qk,
        use_qwen_projection=use_qwen_projection,
        use_packed_qkv_projection=use_packed_qkv_projection,
        use_fused_rms_proj=use_fused_rms_proj,
        use_qknorm_4d=use_qknorm_4d,
        use_cpasync_projection=use_cpasync_projection,
        use_tma_attention=use_tma_attention,
        use_split_attention=use_split_attention,
        split_attention_splits=split_attention_splits,
        attention_tile_m=attention_tile_m,
        op_limit=forward_op_limit,
    )
    ops = forward.ops

    kernel = Megakernel(
        ops,
        config=_config_for(
            ops,
            page_size,
            tracing=tracing,
            num_pages=num_pages,
            page_free_extra_slots=page_free_extra_slots,
        ),
        scheduler=scheduler,
    )
    spec = kernel.bench_spec(
        keep_alive=[
            x, residual, attn_norm, w_q, w_k, w_v, q_norm, k_norm, cos, sin,
            w_o, mlp_norm, w_gate_up, w_down, *forward.keep_alive, kernel,
        ]
    )
    return spec, forward.output, forward.residual


def megakernel_layer_bwd_build(
    batch: int,
    seq_len: int,
    x: torch.Tensor,
    residual: torch.Tensor,
    attn_norm: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    w_o: torch.Tensor,
    mlp_norm: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
    scheduler=None,
    tracing: bool = False,
    num_pages: int | None = None,
    page_free_extra_slots: int = 0,
    attention_bwd_batch_window: int = 4,
    projection_bwd_input_tile_s: int | None = None,
    projection_bwd_reduce_tile_n: int | None = None,
):
    dtype = x.dtype
    device = x.device
    cos_flat = cos.repeat(batch, 1).contiguous()
    sin_flat = sin.repeat(batch, 1).contiguous()

    dy = torch.randn_like(x)
    x0 = torch.randn_like(x)
    q_pre = torch.randn(batch, seq_len, Q_DIM, dtype=dtype, device=device)
    k_pre = torch.randn(batch, seq_len, KV_DIM, dtype=dtype, device=device)
    v = torch.randn(batch, seq_len, KV_DIM, dtype=dtype, device=device)
    q = torch.randn_like(q_pre)
    k = torch.randn_like(k_pre)
    attn = torch.randn(batch, seq_len, Q_DIM, dtype=dtype, device=device)
    lse = torch.randn(batch, seq_len, NUM_Q_HEADS, dtype=torch.float32, device=device)
    residual1 = torch.randn_like(x)
    x1 = torch.randn_like(x)
    gate_up = torch.randn(batch, seq_len, 2 * INTERMEDIATE, dtype=dtype, device=device)
    mlp = torch.randn(batch, seq_len, INTERMEDIATE, dtype=dtype, device=device)

    d_mlp = torch.empty_like(mlp)
    d_gate_up = torch.empty_like(gate_up)
    d_x1 = torch.empty_like(x1)
    d_residual1 = torch.empty_like(residual1)
    d_attn = torch.empty_like(attn)
    dpsum = torch.empty(batch, seq_len, NUM_Q_HEADS, dtype=torch.float32, device=device)
    d_q = torch.zeros(batch, seq_len, NUM_Q_HEADS, HEAD_DIM, dtype=torch.float32, device=device)
    d_k = torch.zeros(batch, seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=dtype, device=device)
    d_v = torch.zeros_like(d_k)
    d_q_pre = torch.empty_like(q_pre)
    d_k_pre = torch.empty_like(k_pre)
    use_projection_bwd_reduce_split = projection_bwd_reduce_tile_n is not None
    d_x0_q = torch.zeros_like(x0) if use_projection_bwd_reduce_split else torch.empty_like(x0)
    d_x0_k = torch.zeros_like(x0) if use_projection_bwd_reduce_split else torch.empty_like(x0)
    d_x0_v = torch.zeros_like(x0) if use_projection_bwd_reduce_split else torch.empty_like(x0)
    d_x0_qk = torch.empty_like(x0)
    d_x0 = torch.empty_like(x0)
    dx = torch.empty_like(x)

    dw_down = torch.empty_like(w_down).unsqueeze(0)
    dw_gate_up = torch.empty_like(w_gate_up).unsqueeze(0)
    dw_o = torch.empty_like(w_o).unsqueeze(0)
    dw_q = torch.empty_like(w_q).unsqueeze(0)
    dw_k = torch.empty_like(w_k).unsqueeze(0)
    dw_v = torch.empty_like(w_v).unsqueeze(0)

    qh_pre = _unflatten_heads(q_pre, batch, seq_len, NUM_Q_HEADS)
    kh_pre = _unflatten_heads(k_pre, batch, seq_len, NUM_KV_HEADS)
    qh = _unflatten_heads(q, batch, seq_len, NUM_Q_HEADS)
    kh = _unflatten_heads(k, batch, seq_len, NUM_KV_HEADS)
    vh = _unflatten_heads(v, batch, seq_len, NUM_KV_HEADS)
    attn_h = _unflatten_heads(attn, batch, seq_len, NUM_Q_HEADS)
    d_attn_h = _unflatten_heads(d_attn, batch, seq_len, NUM_Q_HEADS)

    ops = []
    ops += GemmOp.schedule_backward(dout=dy, a=mlp, b=w_down, da=d_mlp, db=dw_down, page_size=page_size)
    ops += GLUBwdOp.schedule(dy=d_mlp, x=gate_up, dx=d_gate_up, activation="silu", page_size=page_size)
    ops += GemmOp.schedule_backward(dout=d_gate_up, a=x1, b=w_gate_up, da=d_x1, db=dw_gate_up, page_size=page_size)
    # Fuse `d_residual1 = dy + rmsnorm_bwd(d_x1, residual1)` into the RMSNorm
    # backward via its has_add path (dx = rmsnorm_grad + add). Removes a
    # separate add op from the critical path.
    ops += RMSNormBwdOp.schedule(
        dout=d_x1, x=residual1, weight=mlp_norm,
        add=dy, dx=d_residual1,
        page_size=page_size, eps=QWEN3_5_EPS,
    )

    ops += GemmOp.schedule_backward(dout=d_residual1, a=attn, b=w_o, da=d_attn, db=dw_o, page_size=page_size)
    for dim_window in iter_dim_windows("B", batch, attention_bwd_batch_window):
        ops += AttentionDPSumOp.schedule(
            dout=d_attn_h,
            o=attn_h,
            dpsum=dpsum,
            page_size=page_size,
            dim_windows=dim_window,
        )
        ops += FlashAttentionSm120BwdOp.schedule(
            k=kh,
            v=vh,
            q=qh,
            dout=d_attn_h,
            lse=lse,
            dpsum=dpsum,
            dq=d_q,
            dk=d_k,
            dv=d_v,
            causal=True,
            kv_group_size=KV_GROUP_SIZE,
            page_size=page_size,
            dim_windows=dim_window,
        )
    def _schedule_projection_bwd(dout, weight, da, db):
        if projection_bwd_reduce_tile_n is not None:
            split_ops = []
            weight_t = weight.t().contiguous()
            split_ops += ProjectionDaReduceGemmOp.schedule(
                a=dout,
                b=weight_t,
                c=da,
                page_size=page_size,
                reduce_tile_n=projection_bwd_reduce_tile_n,
                tile_sizes={"S": 128, "N": 64, "K": 32},
            )
            split_ops += GemmOp.schedule_backward(
                dout=dout,
                a=x0,
                b=weight,
                db=db,
                page_size=page_size,
            )
            return split_ops

        if projection_bwd_input_tile_s is None:
            return GemmOp.schedule_backward(
                dout=dout,
                a=x0,
                b=weight,
                da=da,
                db=db,
                page_size=page_size,
            )
        split_ops = []
        split_ops += GemmOp.schedule_backward(
            dout=dout,
            a=x0,
            b=weight,
            da=da,
            page_size=page_size,
            tile_sizes={"S": projection_bwd_input_tile_s, "K": 64, "N": 32},
        )
        split_ops += GemmOp.schedule_backward(
            dout=dout,
            a=x0,
            b=weight,
            db=db,
            page_size=page_size,
        )
        return split_ops

    ops += QKNormRopeBwdOp.schedule(
        q=qh_pre,
        dout=d_q,
        norm_weight=q_norm,
        cos=cos,
        sin=sin,
        dq=_unflatten_heads(d_q_pre, batch, seq_len, NUM_Q_HEADS),
        page_size=page_size,
        eps=QWEN3_5_EPS,
    )
    ops += QKNormRopeBwdOp.schedule(
        q=kh_pre,
        dout=d_k,
        norm_weight=k_norm,
        cos=cos,
        sin=sin,
        dq=_unflatten_heads(d_k_pre, batch, seq_len, NUM_KV_HEADS),
        page_size=page_size,
        eps=QWEN3_5_EPS,
    )
    ops += _schedule_projection_bwd(d_q_pre, w_q, d_x0_q, dw_q)
    ops += _schedule_projection_bwd(d_k_pre, w_k, d_x0_k, dw_k)
    ops += _schedule_projection_bwd(d_v.reshape(batch, seq_len, KV_DIM), w_v, d_x0_v, dw_v)
    ops += ResidualAddSm120Op.schedule(
        x=d_x0_q,
        residual_in=d_x0_k,
        residual_out=d_x0_qk,
        tile_sizes={"S": 16, "K": 256},
        page_size=page_size,
    )
    ops += ResidualAddSm120Op.schedule(
        x=d_x0_qk,
        residual_in=d_x0_v,
        residual_out=d_x0,
        tile_sizes={"S": 16, "K": 256},
        page_size=page_size,
    )
    # Fuse the final `dx = d_residual1 + rmsnorm_bwd(d_x0, x)` into the
    # attn-norm RMSNorm backward via its has_add path. Removes the final add op.
    ops += RMSNormBwdOp.schedule(
        dout=d_x0, x=x, weight=attn_norm,
        add=d_residual1, dx=dx,
        page_size=page_size, eps=QWEN3_5_EPS,
    )

    kernel = Megakernel(
        ops,
        config=_config_for(
            ops,
            page_size,
            tracing=tracing,
            num_pages=num_pages,
            page_free_extra_slots=page_free_extra_slots,
        ),
        scheduler=scheduler,
    )
    spec = kernel.bench_spec(
        keep_alive=[
            x, residual, attn_norm, w_q, w_k, w_v, q_norm, k_norm, cos, sin,
            w_o, mlp_norm, w_gate_up, w_down, cos_flat, sin_flat, dy, x0,
            q_pre, k_pre, v, q, k, attn, lse, residual1, x1, gate_up, mlp,
            d_mlp, d_gate_up, d_x1, d_residual1, d_attn,
            dpsum, d_q, d_k, d_v, d_q_pre, d_k_pre, d_x0_q, d_x0_k, d_x0_v,
            d_x0_qk, d_x0, dx, dw_down, dw_gate_up, dw_o,
            dw_q, dw_k, dw_v, qh_pre, kh_pre, qh, kh, vh, attn_h, d_attn_h,
            kernel,
        ]
    )
    return spec, dx


def _alloc_layer(batch: int, seq_len: int):
    torch.manual_seed(2026)
    dtype = torch.bfloat16
    device = "cuda"
    cos, sin = _rope_tables(seq_len, dtype, device)
    return (
        batch,
        seq_len,
        torch.randn(batch, seq_len, HIDDEN, dtype=dtype, device=device),
        torch.randn(batch, seq_len, HIDDEN, dtype=dtype, device=device),
        torch.ones(HIDDEN, dtype=dtype, device=device),
        torch.randn(Q_DIM, HIDDEN, dtype=dtype, device=device) * 0.02,
        torch.randn(KV_DIM, HIDDEN, dtype=dtype, device=device) * 0.02,
        torch.randn(KV_DIM, HIDDEN, dtype=dtype, device=device) * 0.02,
        torch.ones(HEAD_DIM, dtype=dtype, device=device),
        torch.ones(HEAD_DIM, dtype=dtype, device=device),
        cos,
        sin,
        torch.randn(HIDDEN, Q_DIM, dtype=dtype, device=device) * 0.02,
        torch.ones(HIDDEN, dtype=dtype, device=device),
        torch.randn(2 * INTERMEDIATE, HIDDEN, dtype=dtype, device=device) * 0.02,
        torch.randn(HIDDEN, INTERMEDIATE, dtype=dtype, device=device) * 0.02,
    )


def sequential_layer_fwd(
    x, residual, attn_norm, w_q, w_k, w_v, q_norm, k_norm, cos, sin, w_o, mlp_norm, w_gate_up, w_down
):
    x0 = torch.nn.functional.rms_norm(x.float(), (HIDDEN,), attn_norm.float(), eps=QWEN3_5_EPS).to(x.dtype)
    q = torch.matmul(x0, w_q.t()).view(x.shape[0], x.shape[1], NUM_Q_HEADS, HEAD_DIM)
    k = torch.matmul(x0, w_k.t()).view(x.shape[0], x.shape[1], NUM_KV_HEADS, HEAD_DIM)
    v = torch.matmul(x0, w_v.t()).view(x.shape[0], x.shape[1], NUM_KV_HEADS, HEAD_DIM)
    q = torch.nn.functional.rms_norm(q.float(), (HEAD_DIM,), q_norm.float(), eps=QWEN3_5_EPS).to(x.dtype)
    k = torch.nn.functional.rms_norm(k.float(), (HEAD_DIM,), k_norm.float(), eps=QWEN3_5_EPS).to(x.dtype)
    q1, q2 = q[..., :D2], q[..., D2 : 2 * D2]
    k1, k2 = k[..., :D2], k[..., D2 : 2 * D2]
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    q = torch.cat((q1 * c - q2 * s, q2 * c + q1 * s, q[..., 2 * D2 :]), dim=-1)
    k = torch.cat((k1 * c - k2 * s, k2 * c + k1 * s, k[..., 2 * D2 :]), dim=-1)
    k_rep = k.repeat_interleave(KV_GROUP_SIZE, dim=2)
    v_rep = v.repeat_interleave(KV_GROUP_SIZE, dim=2)
    q_t = q.transpose(1, 2)
    k_t = k_rep.transpose(1, 2)
    v_t = v_rep.transpose(1, 2)
    attn = torch.nn.functional.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True).transpose(1, 2)
    attn = attn.reshape(x.shape[0], x.shape[1], Q_DIM)
    residual1 = residual + torch.matmul(attn, w_o.t())
    x1 = torch.nn.functional.rms_norm(residual1.float(), (HIDDEN,), mlp_norm.float(), eps=QWEN3_5_EPS).to(x.dtype)
    gate_up = torch.matmul(x1, w_gate_up.t())
    gate, up = gate_up.chunk(2, dim=-1)
    mlp = torch.nn.functional.silu(gate) * up
    return residual1 + torch.matmul(mlp, w_down.t())


def sequential_layer_bwd(*args):
    tensors = [arg.detach().clone().requires_grad_(arg.is_floating_point()) for arg in args[:-1]]
    dy = args[-1]
    out = sequential_layer_fwd(*tensors)
    out.backward(dy)
    return tuple(t.grad for t in tensors if t.requires_grad)


def _torch_forward_spec(batch: int, seq_len: int, page_size: int):
    args = _alloc_layer(batch, seq_len)
    compiled = torch.compile(sequential_layer_fwd, mode="max-autotune")
    sink = {"out": compiled(*args[2:])}
    torch.cuda.synchronize()

    def _launch():
        sink["out"] = compiled(*args[2:])

    return KernelBenchSpec(launch_fn=_launch, stream=(torch.cuda.current_stream(), None), _keep_alive=[*args, compiled, sink])


def _torch_backward_spec(batch: int, seq_len: int, page_size: int):
    args = _alloc_layer(batch, seq_len)
    dy = torch.randn(batch, seq_len, HIDDEN, dtype=torch.bfloat16, device="cuda")
    compiled = torch.compile(sequential_layer_bwd, mode="max-autotune")
    sink = {"out": compiled(*args[2:], dy)}
    torch.cuda.synchronize()

    def _launch():
        sink["out"] = compiled(*args[2:], dy)

    return KernelBenchSpec(launch_fn=_launch, stream=(torch.cuda.current_stream(), None), _keep_alive=[*args, dy, compiled, sink])


# =============================================================================
# Incremental forward harness: build the megakernel one op at a time, and at
# each step compare to the equivalent torch.compile partial (speed) and to an
# eager fp32 reference (accuracy), plus export a perfetto trace.
# =============================================================================

_INCR_TRACE_DIR = _REPO_ROOT / "traces" / "qwen3_5_forward"


def _rmsn(t: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.rms_norm(
        t.float(), (t.shape[-1],), w.float(), eps=QWEN3_5_EPS
    ).to(t.dtype)


def _qk_norm_rope(q, k, q_norm, k_norm, cos, sin):
    b, s = q.shape[0], q.shape[1]
    qh = _rmsn(q.view(b, s, NUM_Q_HEADS, HEAD_DIM), q_norm)
    kh = _rmsn(k.view(b, s, NUM_KV_HEADS, HEAD_DIM), k_norm)
    c = cos[None, :, None, :]
    s_ = sin[None, :, None, :]
    q1, q2 = qh[..., :D2], qh[..., D2 : 2 * D2]
    k1, k2 = kh[..., :D2], kh[..., D2 : 2 * D2]
    qh = torch.cat((q1 * c - q2 * s_, q2 * c + q1 * s_, qh[..., 2 * D2 :]), dim=-1)
    kh = torch.cat((k1 * c - k2 * s_, k2 * c + k1 * s_, kh[..., 2 * D2 :]), dim=-1)
    return qh.reshape(b, s, Q_DIM), kh.reshape(b, s, KV_DIM)


def _attn(q, k, v):
    b, s = q.shape[0], q.shape[1]
    qh = q.view(b, s, NUM_Q_HEADS, HEAD_DIM)
    kh = k.view(b, s, NUM_KV_HEADS, HEAD_DIM)
    vh = v.view(b, s, NUM_KV_HEADS, HEAD_DIM)
    k_rep = kh.repeat_interleave(KV_GROUP_SIZE, dim=2)
    v_rep = vh.repeat_interleave(KV_GROUP_SIZE, dim=2)
    out = torch.nn.functional.scaled_dot_product_attention(
        qh.transpose(1, 2), k_rep.transpose(1, 2), v_rep.transpose(1, 2), is_causal=True
    ).transpose(1, 2)
    return out.reshape(b, s, Q_DIM)


def _stage_fn(op_limit: int, *, use_packed_qkv_projection: bool = False):
    """Return f(*args[2:]) -> tuple of tensors the megakernel materializes at op_limit.

    Element 0 is the tensor to accuracy-check (the megakernel's `output`); the
    rest are extra outputs the megakernel also produces at this prefix. Returning
    ALL of them prevents torch.compile from dead-code-eliminating work the
    megakernel actually performs (e.g. q/k when only v is the step-3 output) so
    the speed comparison covers the same work.
    """

    def f(x, residual, attn_norm, w_q, w_k, w_v, q_norm, k_norm, cos, sin,
          w_o, mlp_norm, w_gate_up, w_down):
        x0 = _rmsn(x, attn_norm)
        if op_limit == 1:
            return (x0,)
        q = x0 @ w_q.t()
        k = x0 @ w_k.t()
        qk = torch.cat((q, k), dim=-1)
        v = x0 @ w_v.t()
        if use_packed_qkv_projection:
            if op_limit == 2:
                return (qk, v)
        else:
            if op_limit == 2:
                return (qk,)
            if op_limit == 3:
                return (v, qk)
        qn, kn = _qk_norm_rope(q, k, q_norm, k_norm, cos, sin)
        qknorm_step = 3 if use_packed_qkv_projection else 4
        if op_limit == qknorm_step:
            return (torch.cat((qn, kn), dim=-1), v)
        attn = _attn(qn, kn, v)
        if op_limit == qknorm_step + 1:
            return (attn,)
        attn_proj = attn @ w_o.t()
        if op_limit == qknorm_step + 2:
            return (attn_proj,)
        residual1 = residual + attn_proj
        x1 = _rmsn(residual1, mlp_norm)
        if op_limit == qknorm_step + 3:
            return (x1,)
        gate_up = x1 @ w_gate_up.t()
        if op_limit == qknorm_step + 4:
            return (gate_up,)
        gate, up = gate_up.chunk(2, dim=-1)
        mlp = torch.nn.functional.silu(gate.float()).to(up.dtype) * up
        if op_limit == qknorm_step + 5:
            return (mlp,)
        mlp_out = mlp @ w_down.t()
        if op_limit == qknorm_step + 6:
            if use_packed_qkv_projection:
                return (residual1 + mlp_out,)
            return (mlp_out,)
        return (residual1 + mlp_out,)

    return f


_STEP_NAMES = {
    1: "rmsnorm", 2: "qk_proj", 3: "v_proj", 4: "qk_norm_rope", 5: "attention",
    6: "o_proj", 7: "post_rmsnorm", 8: "gate_up", 9: "glu", 10: "down", 11: "residual",
}

_PACKED_QKV_STEP_NAMES = {
    1: "rmsnorm", 2: "qkv_proj", 3: "qk_norm_rope", 4: "attention",
    5: "o_proj", 6: "post_rmsnorm", 7: "gate_up", 8: "glu", 9: "down_residual",
}


def _time_kernel_spec(spec, warmup: int, rep: int) -> float:
    import statistics
    for _ in range(warmup):
        spec.setup_fn(); spec.launch_fn()
    torch.cuda.synchronize()
    bench_stream = spec.stream[0]
    times = []
    for _ in range(rep):
        spec.setup_fn()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(bench_stream):
            s.record(); spec.launch_fn(); e.record()
        e.synchronize(); times.append(s.elapsed_time(e) * 1000.0)
    return statistics.median(times)


def _time_torch(fn, args, warmup: int, rep: int) -> float:
    import statistics
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    times = []
    for _ in range(rep):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(*args); e.record()
        e.synchronize(); times.append(s.elapsed_time(e) * 1000.0)
    return statistics.median(times)


def run_incremental(batch, seqs, steps, *, page_size=DEFAULT_PAGE_SIZE,
                    do_torch=True, do_trace=True, torch_mode="max-autotune",
                    warmup=8, rep=30, use_packed_qkv_projection=False,
                    use_fused_rms_proj=False, use_qknorm_4d=True,
                    use_tma_attention=False, use_split_attention=False,
                    split_attention_splits=0):
    import contextlib, io
    _INCR_TRACE_DIR.mkdir(parents=True, exist_ok=True)
    for seq_len in seqs:
        args = _alloc_layer(batch, seq_len)
        tensors = args[2:]
        print(f"\n=== B={batch} S={seq_len} pg={page_size//1024}K ===")
        print(f"{'step':>4} {'op':14} {'mk_us':>9} {'torch_us':>9} {'speedup':>8} "
              f"{'max_abs':>10} {'max_rel':>10}")
        for op_limit in steps:
            with contextlib.redirect_stdout(io.StringIO()):
                spec, mk_out, _ = megakernel_forward_build(
                    *args, page_size=page_size, scheduler=_scheduler("overlap"),
                    use_packed_qkv_projection=use_packed_qkv_projection,
                    use_fused_rms_proj=use_fused_rms_proj,
                    use_qknorm_4d=use_qknorm_4d,
                    use_tma_attention=use_tma_attention,
                    use_split_attention=use_split_attention,
                    split_attention_splits=split_attention_splits,
                    forward_op_limit=op_limit)
            spec.setup_fn(); spec.launch_fn(); torch.cuda.synchronize()
            stage_fn = _stage_fn(op_limit, use_packed_qkv_projection=use_packed_qkv_projection)
            ref = stage_fn(*tensors)[0].float()
            got = mk_out.float()
            max_abs = (got - ref).abs().max().item()
            denom = ref.abs().max().item() + 1e-6
            max_rel = max_abs / denom
            mk_us = _time_kernel_spec(spec, warmup, rep)
            torch_us = float("nan"); speed = float("nan")
            if do_torch:
                fn = torch.compile(stage_fn, mode=torch_mode)
                fn(*tensors); torch.cuda.synchronize()
                torch_us = _time_torch(fn, tensors, warmup, rep)
                speed = torch_us / mk_us
            names = _PACKED_QKV_STEP_NAMES if use_packed_qkv_projection else _STEP_NAMES
            name = names.get(op_limit, f"op{op_limit}")
            print(f"{op_limit:>4} {name:14} {mk_us:9.2f} {torch_us:9.2f} {speed:7.2f}x "
                  f"{max_abs:10.2e} {max_rel:10.2e}")
            if do_trace:
                with contextlib.redirect_stdout(io.StringIO()):
                    tspec, tk, _ = _build_traced_forward(
                        args,
                        page_size,
                        op_limit,
                        use_packed_qkv_projection=use_packed_qkv_projection,
                        use_fused_rms_proj=use_fused_rms_proj,
                        use_qknorm_4d=use_qknorm_4d,
                    )
                tspec.setup_fn(); tspec.launch_fn(); torch.cuda.synchronize()
                suffix = ""
                if use_packed_qkv_projection:
                    suffix += "_packed_qkv"
                if use_fused_rms_proj:
                    suffix += "_fused_rms_proj"
                path = _INCR_TRACE_DIR / f"step{op_limit:02d}_{name}_b{batch}_s{seq_len}{suffix}.perfetto.json"
                tk.write_trace_perfetto(str(path))


def _build_traced_forward(
    args,
    page_size,
    op_limit,
    *,
    use_packed_qkv_projection=False,
    use_fused_rms_proj=False,
    use_qknorm_4d=True,
):
    forward = schedule_qwen3_5_forward_ops(
        *args, page_size=page_size, scheduler=_scheduler("overlap"),
        use_packed_qk=True,
        use_packed_qkv_projection=use_packed_qkv_projection,
        use_fused_rms_proj=use_fused_rms_proj,
        use_qknorm_4d=use_qknorm_4d,
        op_limit=op_limit)
    kernel = Megakernel(forward.ops, config=_config_for(forward.ops, page_size, tracing=True),
                        scheduler=_scheduler("overlap"))
    spec = kernel.bench_spec(keep_alive=[*args, *forward.keep_alive, kernel])
    return spec, kernel, forward.output


@Benchmark.configs(["seq_len", "batch", "page_size", "fetch_stride"], _configs_with_fetch_stride(0))
def bench_qwen35_layer_fwd(seq_len: int, batch: int, page_size: int, fetch_stride: int = 0):
    args = _alloc_layer(batch, seq_len)
    return {
        "torch_compile": _torch_forward_spec(batch, seq_len, page_size),
        "megakernel": megakernel_forward_build(
            *args,
            page_size=page_size,
            scheduler=_scheduler("overlap", fetch_stride),
            use_packed_qkv_projection=BENCH_USE_PACKED_QKV_PROJECTION,
            use_fused_rms_proj=BENCH_USE_FUSED_RMS_PROJ,
            use_qknorm_4d=BENCH_USE_QKNORM_4D,
            use_tma_attention=BENCH_USE_TMA_ATTENTION,
            use_split_attention=BENCH_USE_SPLIT_ATTENTION,
            split_attention_splits=BENCH_SPLIT_ATTENTION_SPLITS,
            attention_tile_m=BENCH_ATTENTION_TILE_M,
        )[0],
    }


@Benchmark.configs(["seq_len", "batch", "page_size", "fetch_stride"], _configs_with_fetch_stride(0))
def bench_qwen35_layer_bwd(seq_len: int, batch: int, page_size: int, fetch_stride: int = 0):
    args = _alloc_layer(batch, seq_len)
    return {
        "torch_compile": _torch_backward_spec(batch, seq_len, page_size),
        "megakernel": megakernel_layer_bwd_build(
            *args,
            page_size=page_size,
            scheduler=_scheduler("overlap", fetch_stride),
        )[0],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, nargs="+", default=[1])
    parser.add_argument("--seq-len", type=int, nargs="+", default=[128])
    parser.add_argument("--page-size", type=int, nargs="+", default=[DEFAULT_PAGE_SIZE])
    parser.add_argument("--direction", choices=["forward", "backward"], nargs="+", default=["forward", "backward"])
    parser.add_argument(
        "--fetch-stride",
        type=int,
        default=0,
        help="Overlap scheduler fetch stride; 0 uses the scheduler default.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=20)
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Build the forward megakernel one op at a time; compare each step to "
        "torch.compile (speed) and an eager fp32 reference (accuracy), and write a "
        "perfetto trace per step into traces/qwen3_5_forward/.",
    )
    parser.add_argument("--steps", type=int, nargs="+", default=list(range(1, 12)),
                        help="Which op_limit steps to run in --incremental mode.")
    parser.add_argument("--no-torch", dest="incr_torch", action="store_false", default=True,
                        help="Skip torch.compile timing in --incremental mode.")
    parser.add_argument("--no-trace", dest="incr_trace", action="store_false", default=True,
                        help="Skip perfetto trace export in --incremental mode.")
    parser.add_argument("--torch-mode", default="max-autotune",
                        help="torch.compile mode used for the per-step comparison.")
    qkv_group = parser.add_mutually_exclusive_group()
    qkv_group.add_argument(
        "--packed-qkv-projection",
        dest="packed_qkv_projection",
        action="store_true",
        default=True,
        help="Use one Qwen-local QKV projection buffer/op instead of separate packed QK + V projection.",
    )
    qkv_group.add_argument(
        "--separate-v-projection",
        dest="packed_qkv_projection",
        action="store_false",
        help="Use the older packed-QK plus separate V projection path.",
    )
    parser.add_argument(
        "--fused-rms-proj",
        action="store_true",
        help="Fuse the first RMSNorm into the first projection op in the Qwen-local forward path.",
    )
    parser.add_argument(
        "--attention-tile-m",
        type=int,
        default=None,
        help="Override Qwen forward attention M tile for overlap experiments.",
    )
    attn_group = parser.add_mutually_exclusive_group()
    attn_group.add_argument(
        "--tma-attention",
        action="store_true",
        help="Use Qwen-local full attention with compute-issued TMA K/V loads.",
    )
    attn_group.add_argument(
        "--split-attention",
        action="store_true",
        help="Use split-KV attention experiment. Currently decode-shaped and not valid for large M.",
    )
    parser.add_argument(
        "--split-attention-splits",
        type=int,
        default=0,
        help="Number of split-KV chunks for --split-attention; 0 auto-selects.",
    )
    qknorm_group = parser.add_mutually_exclusive_group()
    qknorm_group.add_argument(
        "--qknorm-4d",
        dest="qknorm_4d",
        action="store_true",
        default=True,
        help="Use the Qwen-local 4D packed QKNorm dependency path.",
    )
    qknorm_group.add_argument(
        "--flat-qknorm",
        dest="qknorm_4d",
        action="store_false",
        help="Use the older flattened packed QKNorm path.",
    )
    args = parser.parse_args()

    BENCH_USE_PACKED_QKV_PROJECTION = args.packed_qkv_projection
    BENCH_USE_FUSED_RMS_PROJ = args.fused_rms_proj
    BENCH_USE_QKNORM_4D = args.qknorm_4d
    BENCH_USE_TMA_ATTENTION = args.tma_attention
    BENCH_USE_SPLIT_ATTENTION = args.split_attention
    BENCH_SPLIT_ATTENTION_SPLITS = args.split_attention_splits
    BENCH_ATTENTION_TILE_M = args.attention_tile_m

    if args.incremental:
        for batch in args.batch:
            for page_size in args.page_size:
                run_incremental(
                    batch, args.seq_len, args.steps, page_size=page_size,
                    do_torch=args.incr_torch, do_trace=args.incr_trace,
                    torch_mode=args.torch_mode, warmup=args.warmup, rep=args.rep,
            use_packed_qkv_projection=args.packed_qkv_projection,
            use_fused_rms_proj=args.fused_rms_proj,
            use_qknorm_4d=args.qknorm_4d,
            use_tma_attention=args.tma_attention,
            use_split_attention=args.split_attention,
            split_attention_splits=args.split_attention_splits,
        )
        sys.exit(0)

    if "forward" in args.direction:
        bench_qwen35_layer_fwd._benchmark._configs = [
            (seq_len, batch, page_size, int(args.fetch_stride))
            for seq_len in args.seq_len
            for batch in args.batch
            for page_size in args.page_size
        ]
        bench_qwen35_layer_fwd._benchmark.run(mode="kernel", warmup=args.warmup, rep=args.rep, export_csv=False)
    if "backward" in args.direction:
        bench_qwen35_layer_bwd._benchmark._configs = [
            (seq_len, batch, page_size, int(args.fetch_stride))
            for seq_len in args.seq_len
            for batch in args.batch
            for page_size in args.page_size
        ]
        bench_qwen35_layer_bwd._benchmark.run(mode="kernel", warmup=args.warmup, rep=args.rep, export_csv=False)
