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

from machete.kernels.gemm import GemmOp
from machete.kernels.qwen_3_5.qwen_3_5_forward import (
    Qwen3_5ForwardCausalRankScheduler,
    Qwen3_5ForwardOverlapScheduler,
    schedule_qwen3_5_backward_ops,
    schedule_qwen3_5_forward_ops,
)
from machete.kernels.qwen_3_5.sm120 import QWEN3_5_EPS
from machete.megakernel import Megakernel, MegakernelConfig, OverlapTileScheduler
from machete.utils.benchmark import Benchmark
from machete.utils.benchmark_utils import KernelBenchSpec
from machete.utils.output import suppress_stdout_stderr, suppress_torch_compile_logs


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
    (1024, 1, DEFAULT_PAGE_SIZE),
]

BENCH_USE_PACKED_QKV_PROJECTION = True
BENCH_USE_FUSED_RMS_PROJ = True
BENCH_USE_QKNORM_4D = True
BENCH_USE_TMA_ATTENTION = True
BENCH_USE_TWO_PAGE_ATTENTION = False
BENCH_USE_THREE_PAGE_ATTENTION = True
BENCH_USE_SPLIT_ATTENTION = False
BENCH_SPLIT_ATTENTION_SPLITS = 1
BENCH_ATTENTION_TILE_M = None
BENCH_BWD_SCHEDULER = "overlap-adaptive"
BENCH_BWD_FETCH_STRIDE = 16
BENCH_PROJECTION_BWD_INPUT_TILE_S = 0
BENCH_PROJECTION_BWD_REDUCE_TILE_N = 0


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
    if variant in (
        "overlap",
        "overlap-controller",
        "overlap-adaptive",
        "overlap-adaptive-controller",
        "qwen-forward",
        "qwen-causal-rank",
    ):
        resolved_fetch_stride = (
            int(fetch_stride)
            if fetch_stride is not None and int(fetch_stride) > 0
            else None
        )
        if variant == "qwen-forward":
            return Qwen3_5ForwardOverlapScheduler(fetch_stride=resolved_fetch_stride)
        if variant == "qwen-causal-rank":
            return Qwen3_5ForwardCausalRankScheduler(fetch_stride=resolved_fetch_stride)
        return OverlapTileScheduler(
            fetch_stride=resolved_fetch_stride,
            adaptive_fetch_stride=variant in ("overlap-adaptive", "overlap-adaptive-controller"),
            readiness_wait_phase=(
                "controller"
                if variant in ("overlap-controller", "overlap-adaptive-controller")
                else "all"
            ),
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
    use_fused_rms_proj: bool = True,
    use_qknorm_4d: bool = True,
    use_cpasync_projection: bool = False,
    use_tma_attention: bool = True,
    use_two_page_attention: bool = False,
    use_three_page_attention: bool = True,
    use_split_attention: bool = False,
    split_attention_splits: int = 1,
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
        use_two_page_attention=use_two_page_attention,
        use_three_page_attention=use_three_page_attention,
        use_split_attention=use_split_attention,
        split_attention_splits=split_attention_splits,
        attention_tile_m=attention_tile_m,
        op_limit=forward_op_limit,
    )
    ops = forward.ops

    with suppress_stdout_stderr():
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
    backward = schedule_qwen3_5_backward_ops(
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
        attention_bwd_batch_window=attention_bwd_batch_window,
        projection_bwd_input_tile_s=projection_bwd_input_tile_s,
        projection_bwd_reduce_tile_n=projection_bwd_reduce_tile_n,
    )
    ops = backward.ops

    with suppress_stdout_stderr():
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
                w_o, mlp_norm, w_gate_up, w_down, *backward.keep_alive, kernel,
            ]
        )
    return spec, backward.output


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
    sink = {}
    with suppress_stdout_stderr(suppress_torch_compile_logs()):
        compiled = torch.compile(sequential_layer_fwd, mode="max-autotune")
        sink["out"] = compiled(*args[2:])
        torch.cuda.synchronize()

    def _launch():
        sink["out"] = compiled(*args[2:])

    return KernelBenchSpec(launch_fn=_launch, stream=(torch.cuda.current_stream(), None), _keep_alive=[*args, compiled, sink])


def _torch_backward_spec(batch: int, seq_len: int, page_size: int):
    args = _alloc_layer(batch, seq_len)
    dy = torch.randn(batch, seq_len, HIDDEN, dtype=torch.bfloat16, device="cuda")
    sink = {}
    with suppress_stdout_stderr(suppress_torch_compile_logs()):
        compiled = torch.compile(sequential_layer_bwd, mode="max-autotune")
        sink["out"] = compiled(*args[2:], dy)
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
        if spec.setup_fn is not None:
            spec.setup_fn()
        spec.launch_fn()
    torch.cuda.synchronize()
    bench_stream = spec.stream[0]
    times = []
    for _ in range(rep):
        if spec.setup_fn is not None:
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
                    use_tma_attention=True, use_split_attention=True,
                    split_attention_splits=1, use_two_page_attention=False,
                    use_three_page_attention=False):
    import contextlib, io
    if use_tma_attention and not use_split_attention:
        use_packed_qkv_projection = False
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
                    use_two_page_attention=use_two_page_attention,
                    use_three_page_attention=use_three_page_attention,
                    use_split_attention=use_split_attention,
                    split_attention_splits=split_attention_splits,
                    forward_op_limit=op_limit)
            if spec.setup_fn is not None:
                spec.setup_fn()
            spec.launch_fn(); torch.cuda.synchronize()
            stage_fn = _stage_fn(op_limit, use_packed_qkv_projection=use_packed_qkv_projection)
            ref = stage_fn(*tensors)[0].float()
            got = mk_out.float()
            max_abs = (got - ref).abs().max().item()
            denom = ref.abs().max().item() + 1e-6
            max_rel = max_abs / denom
            mk_us = _time_kernel_spec(spec, warmup, rep)
            torch_us = float("nan"); speed = float("nan")
            if do_torch:
                with suppress_stdout_stderr(suppress_torch_compile_logs()):
                    fn = torch.compile(stage_fn, mode=torch_mode)
                    fn(*tensors)
                    torch.cuda.synchronize()
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
                if tspec.setup_fn is not None:
                    tspec.setup_fn()
                tspec.launch_fn(); torch.cuda.synchronize()
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
            scheduler=_scheduler("qwen-causal-rank", fetch_stride),
            use_packed_qkv_projection=BENCH_USE_PACKED_QKV_PROJECTION,
            use_fused_rms_proj=BENCH_USE_FUSED_RMS_PROJ,
            use_qknorm_4d=BENCH_USE_QKNORM_4D,
            use_tma_attention=BENCH_USE_TMA_ATTENTION,
            use_two_page_attention=BENCH_USE_TWO_PAGE_ATTENTION,
            use_three_page_attention=BENCH_USE_THREE_PAGE_ATTENTION,
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
            scheduler=_scheduler(BENCH_BWD_SCHEDULER, fetch_stride),
            projection_bwd_input_tile_s=(
                None
                if BENCH_PROJECTION_BWD_INPUT_TILE_S <= 0
                else BENCH_PROJECTION_BWD_INPUT_TILE_S
            ),
            projection_bwd_reduce_tile_n=(
                None
                if BENCH_PROJECTION_BWD_REDUCE_TILE_N <= 0
                else BENCH_PROJECTION_BWD_REDUCE_TILE_N
            ),
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
    parser.add_argument(
        "--bwd-scheduler",
        choices=["default", "overlap", "overlap-controller", "overlap-adaptive", "overlap-adaptive-controller"],
        default=BENCH_BWD_SCHEDULER,
        help="Scheduler used by the backward megakernel benchmark.",
    )
    parser.add_argument(
        "--projection-bwd-input-tile-s",
        type=int,
        default=0,
        help="Split projection backward input-gradient tiles along S; 0 uses the default GEMM backward schedule.",
    )
    parser.add_argument(
        "--projection-bwd-reduce-tile-n",
        type=int,
        default=0,
        help="Use the experimental projection backward reduce-split path; 0 disables it.",
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
    rms_proj_group = parser.add_mutually_exclusive_group()
    rms_proj_group.add_argument(
        "--fused-rms-proj",
        dest="fused_rms_proj",
        action="store_true",
        default=True,
        help="Fuse the first RMSNorm into the first projection op in the Qwen-local forward path.",
    )
    rms_proj_group.add_argument(
        "--no-fused-rms-proj",
        dest="fused_rms_proj",
        action="store_false",
        help="Keep the first RMSNorm as a separate op.",
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
        dest="attention_path",
        action="store_const",
        const="tma",
        default="tma",
        help="Use Qwen-local full attention with compute-issued TMA K/V loads.",
    )
    attn_group.add_argument(
        "--split-attention",
        dest="attention_path",
        action="store_const",
        const="split",
        help="Use split-MMA attention followed by the tuned Qwen o_proj path.",
    )
    parser.add_argument(
        "--two-page-attention",
        action="store_true",
        help="With --tma-attention, use separate Q and KV/O shared-memory pages.",
    )
    parser.add_argument(
        "--three-page-attention",
        action="store_true",
        default=True,
        help="With --tma-attention, use separate Q/O, K, and V shared-memory pages.",
    )
    parser.add_argument(
        "--split-attention-splits",
        type=int,
        default=1,
        help="Number of split-KV chunks for --split-attention.",
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
    args.split_attention = args.attention_path == "split"
    args.tma_attention = args.attention_path == "tma"
    if args.two_page_attention:
        args.three_page_attention = False

    BENCH_USE_PACKED_QKV_PROJECTION = args.packed_qkv_projection
    BENCH_USE_FUSED_RMS_PROJ = args.fused_rms_proj
    BENCH_USE_QKNORM_4D = args.qknorm_4d
    BENCH_USE_TMA_ATTENTION = args.tma_attention
    BENCH_USE_TWO_PAGE_ATTENTION = args.two_page_attention
    BENCH_USE_THREE_PAGE_ATTENTION = args.three_page_attention
    BENCH_USE_SPLIT_ATTENTION = args.split_attention
    BENCH_SPLIT_ATTENTION_SPLITS = args.split_attention_splits
    BENCH_ATTENTION_TILE_M = args.attention_tile_m
    BENCH_BWD_SCHEDULER = args.bwd_scheduler
    BENCH_PROJECTION_BWD_INPUT_TILE_S = args.projection_bwd_input_tile_s
    BENCH_PROJECTION_BWD_REDUCE_TILE_N = args.projection_bwd_reduce_tile_n

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
            use_two_page_attention=args.two_page_attention,
            use_three_page_attention=args.three_page_attention,
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
        bwd_fetch_stride = (
            int(args.fetch_stride)
            if int(args.fetch_stride) > 0
            else int(BENCH_BWD_FETCH_STRIDE)
        )
        bench_qwen35_layer_bwd._benchmark._configs = [
            (seq_len, batch, page_size, bwd_fetch_stride)
            for seq_len in args.seq_len
            for batch in args.batch
            for page_size in args.page_size
        ]
        bench_qwen35_layer_bwd._benchmark.run(mode="kernel", warmup=args.warmup, rep=args.rep, export_csv=False)
