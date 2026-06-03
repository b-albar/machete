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
    return MegakernelConfig(
        threads_per_block=max(256, gemm_config.threads_per_block),
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


def _rms_tile_sizes_for_overlap(scheduler, rms_tile_s: int | None) -> dict[str, int] | None:
    if rms_tile_s is not None:
        return {"S": int(rms_tile_s)}
    if scheduler is None:
        return None
    return {"S": 8}


def _rope_tables(seq_len: int, dtype: torch.dtype, device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    dims = torch.arange(D2, device=device, dtype=torch.float32)
    inv_freq = torch.pow(torch.tensor(10000000.0, device=device, dtype=torch.float32), -dims / D2)
    angles = pos[:, None] * inv_freq[None, :]
    return torch.cos(angles).to(dtype).contiguous(), torch.sin(angles).to(dtype).contiguous()


def _flatten_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    return x.reshape(x.shape[0] * x.shape[1], heads, HEAD_DIM)


def _unflatten_heads(x: torch.Tensor, batch: int, seq_len: int, heads: int) -> torch.Tensor:
    return x.reshape(batch, seq_len, heads, HEAD_DIM)


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
    use_packed_qk: bool = True,
):
    dtype = x.dtype
    device = x.device
    cos_flat = cos.repeat(batch, 1).contiguous()
    sin_flat = sin.repeat(batch, 1).contiguous()

    x0 = torch.empty_like(x)
    if use_packed_qk:
        qk = torch.empty(batch, seq_len, Q_DIM + KV_DIM, dtype=dtype, device=device)
        q = qk[..., :Q_DIM]
        k = qk[..., Q_DIM:]
        w_qk = torch.cat((w_q, w_k), dim=0).contiguous()
    else:
        qk = None
        w_qk = None
        q = torch.empty(batch, seq_len, Q_DIM, dtype=dtype, device=device)
        k = torch.empty(batch, seq_len, KV_DIM, dtype=dtype, device=device)
    v = torch.empty(batch, seq_len, KV_DIM, dtype=dtype, device=device)
    attn = torch.empty(batch, seq_len, Q_DIM, dtype=dtype, device=device)
    lse = torch.empty(batch, seq_len, NUM_Q_HEADS, dtype=torch.float32, device=device)
    attn_proj = torch.empty_like(x)
    residual1 = torch.empty_like(x)
    x1 = torch.empty_like(x)
    gate_up = torch.empty(batch, seq_len, 2 * INTERMEDIATE, dtype=dtype, device=device)
    mlp = torch.empty(batch, seq_len, INTERMEDIATE, dtype=dtype, device=device)
    mlp_out = torch.empty_like(x)
    y = torch.empty_like(x)

    qh = _flatten_heads(q, NUM_Q_HEADS)
    kh = _flatten_heads(k, NUM_KV_HEADS)
    qkh = (
        qk.reshape(batch * seq_len, NUM_Q_HEADS + NUM_KV_HEADS, HEAD_DIM)
        if qk is not None
        else None
    )
    kh4 = _unflatten_heads(k, batch, seq_len, NUM_KV_HEADS)
    vh = _unflatten_heads(v, batch, seq_len, NUM_KV_HEADS)
    attn_h = _unflatten_heads(attn, batch, seq_len, NUM_Q_HEADS)

    gemm_tile_sizes = gemm_tile_sizes or {}
    rms_tile_sizes = _rms_tile_sizes_for_overlap(scheduler, rms_tile_s)

    def _gemm(name: str, *, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
        return GemmOp.schedule(
            a=a,
            b=b,
            c=c,
            page_size=page_size,
            tile_sizes=gemm_tile_sizes.get(name),
        )

    ops = []
    ops += RMSNormOp.schedule(
        x=x,
        weight=attn_norm,
        y=x0,
        page_size=page_size,
        eps=QWEN3_5_EPS,
        tile_sizes=rms_tile_sizes,
    )
    if use_packed_qk:
        ops += _gemm("qk", a=x0, b=w_qk, c=qk)
    else:
        ops += _gemm("q", a=x0, b=w_q, c=q)
        ops += _gemm("k", a=x0, b=w_k, c=k)
    ops += _gemm("v", a=x0, b=w_v, c=v)
    if use_packed_qk:
        ops += PackedQKNormRopeOp.schedule(
            qk=qkh,
            q_norm_weight=q_norm,
            k_norm_weight=k_norm,
            cos=cos_flat,
            sin=sin_flat,
            page_size=page_size,
            eps=QWEN3_5_EPS,
            num_q_heads=NUM_Q_HEADS,
            num_k_heads=NUM_KV_HEADS,
        )
    else:
        ops += QKNormRopeOp.schedule(q=qh, norm_weight=q_norm, cos=cos_flat, sin=sin_flat, page_size=page_size, eps=QWEN3_5_EPS)
        ops += QKNormRopeOp.schedule(q=kh, norm_weight=k_norm, cos=cos_flat, sin=sin_flat, page_size=page_size, eps=QWEN3_5_EPS)
    ops += FlashAttentionSm120Op.schedule(
        q=_unflatten_heads(q, batch, seq_len, NUM_Q_HEADS),
        k=kh4,
        v=vh,
        o=attn_h,
        lse=lse,
        causal=True,
        kv_group_size=KV_GROUP_SIZE,
        page_size=page_size,
        write_lse=True,
    )
    ops += _gemm("o", a=attn, b=w_o, c=attn_proj)
    # Fuse the post-attention residual add into the MLP-norm RMSNorm via the
    # pre-norm fused-add path (residual_out = x + residual_in; y = rmsnorm(...)).
    #   residual1 = residual + attn_proj;  x1 = rmsnorm(residual1)
    # This removes a separate add op from the critical path (the MLP RMSNorm
    # otherwise stalls in dep_wait on the Add) and avoids an extra global
    # round-trip of residual1. residual1 is still emitted (residual_out) for the
    # final residual add. Note: `residual_in` alone drives the pre-norm add;
    # the `residual=True` flag is a different (post-norm) op and must NOT be set.
    ops += RMSNormOp.schedule(
        x=attn_proj,
        residual_in=residual,
        residual_out=residual1,
        weight=mlp_norm,
        y=x1,
        page_size=page_size,
        eps=QWEN3_5_EPS,
        tile_sizes=rms_tile_sizes,
    )
    ops += _gemm("gate_up", a=x1, b=w_gate_up, c=gate_up)
    ops += GLUOp.schedule(x=gate_up, y=mlp, activation="silu", page_size=page_size)
    ops += _gemm("down", a=mlp, b=w_down, c=mlp_out)
    ops += ResidualAddSm120Op.schedule(
        x=mlp_out,
        residual_in=residual1,
        residual_out=y,
        tile_sizes={"S": 16, "K": 256},
        page_size=page_size,
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
            w_o, mlp_norm, w_gate_up, w_down, cos_flat, sin_flat, x0, q, k, v,
            qk, w_qk, qh, kh, qkh, kh4, vh, attn, attn_h, lse, attn_proj,
            residual1, x1, gate_up, mlp, mlp_out, y, kernel,
        ]
    )
    return spec, y, residual1


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


@Benchmark.configs(["seq_len", "batch", "page_size", "fetch_stride"], _configs_with_fetch_stride(0))
def bench_qwen35_layer_fwd(seq_len: int, batch: int, page_size: int, fetch_stride: int = 0):
    args = _alloc_layer(batch, seq_len)
    return {
        "torch_compile": _torch_forward_spec(batch, seq_len, page_size),
        "megakernel": megakernel_forward_build(
            *args,
            page_size=page_size,
            scheduler=_scheduler("overlap", fetch_stride),
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
    args = parser.parse_args()

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
