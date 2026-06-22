#!/usr/bin/env python
"""Benchmark Qwen 3.5 attention followed by output projection.

This isolates the attention -> o_proj boundary from the full forward layer:

    attn = causal_gqa_sdpa(q, k, v)
    out = attn @ w_o.T

Machete runs the same work as a two-op megakernel using the Qwen TMA attention
op and Qwen forward projection op.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from machete.kernels.gemm import GemmOp
from machete.kernels.qwen_3_5.qwen_3_5_forward import (
    DEFAULT_PAGE_SIZE,
    HEAD_DIM,
    HIDDEN,
    KV_GROUP_SIZE,
    NUM_KV_HEADS,
    NUM_Q_HEADS,
    Q_DIM,
    Qwen3_5ForwardCausalRankScheduler,
    Qwen3_5ForwardProjectionOp,
    Qwen3_5ForwardThreePageTmaAttentionOp,
    Qwen3_5ForwardTmaAttentionOp,
    Qwen3_5ForwardTwoPageTmaAttentionOp,
    schedule_qwen3_5_forward_fused_split_mma_attention,
)
from machete.megakernel import Megakernel, MegakernelConfig, OverlapTileScheduler
from machete.megakernel.megakernel import NUM_DMA_WARPS
from machete.utils.output import suppress_stdout_stderr, suppress_torch_compile_logs


def _make_config(ops, page_size: int, tracing: bool = False) -> MegakernelConfig:
    gemm_ops = [op for op in ops if issubclass(op.op_cls, GemmOp)]
    gemm_config = GemmOp.kernel_config(gemm_ops or ops)
    requested_compute_warps = max(
        (int(op.static_dims.get("num_mma_warps", 1)) for op in ops),
        default=1,
    )
    requested_threads = (requested_compute_warps + NUM_DMA_WARPS) * 32
    return MegakernelConfig(
        threads_per_block=max(256, gemm_config.threads_per_block, requested_threads),
        page_size=max(page_size, *(op.static_dims.get("page_size", page_size) for op in ops)),
        tracing=tracing,
    )


def _time_callable(fn, *, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(times)


def _time_spec(spec, *, warmup: int, rep: int) -> float:
    def launch():
        if spec.setup_fn is not None:
            spec.setup_fn()
        spec.launch_fn()

    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()

    stream = spec.stream[0]
    times = []
    for _ in range(rep):
        if spec.setup_fn is not None:
            spec.setup_fn()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start.record()
            spec.launch_fn()
            end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(times)


def _sdpa_attention_o_proj(q, k, v, w_o):
    q_t = q.permute(0, 2, 1, 3).contiguous()
    k_t = k.permute(0, 2, 1, 3).contiguous()
    v_t = v.permute(0, 2, 1, 3).contiguous()
    try:
        attn = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            is_causal=True,
            enable_gqa=True,
        )
    except TypeError:
        k_rep = k_t.repeat_interleave(KV_GROUP_SIZE, dim=1).contiguous()
        v_rep = v_t.repeat_interleave(KV_GROUP_SIZE, dim=1).contiguous()
        attn = F.scaled_dot_product_attention(q_t, k_rep, v_rep, is_causal=True)
    attn = attn.permute(0, 2, 1, 3).reshape(q.shape[0], q.shape[1], Q_DIM)
    return torch.matmul(attn, w_o.t())


def _build_machete_spec(
    q,
    k,
    v,
    w_o,
    *,
    page_size: int,
    attention_mode: str,
    two_page_attention: bool,
    three_page_attention: bool,
    attention_tile_m: int,
    attention_mma_warps: int | None,
    attention_n_block: int | None,
    attention_load_d_block: int | None,
    split_attention_splits: int,
    split_combine_tile_m: int,
    scheduler,
    tracing: bool = False,
):
    batch, seq_len = q.shape[:2]
    attn = torch.empty(batch, seq_len, Q_DIM, device=q.device, dtype=q.dtype)
    attn_h = attn.reshape(batch, seq_len, NUM_Q_HEADS, HEAD_DIM)
    out = torch.empty(batch, seq_len, HIDDEN, device=q.device, dtype=q.dtype)
    lse = torch.empty(batch, seq_len, NUM_Q_HEADS, device=q.device, dtype=torch.float32)

    if attention_mode == "split-mma":
        ops, attention_keep_alive = schedule_qwen3_5_forward_fused_split_mma_attention(
            q=q,
            k=k,
            v=v,
            o=attn_h,
            lse=lse,
            tile_m=attention_tile_m,
            combine_tile_m=split_combine_tile_m,
            num_splits=split_attention_splits,
            page_size=page_size,
            causal=True,
            kv_group_size=KV_GROUP_SIZE,
        )
    else:
        attention_keep_alive = []
        attention_cls = (
            Qwen3_5ForwardThreePageTmaAttentionOp
            if three_page_attention
            else Qwen3_5ForwardTwoPageTmaAttentionOp
            if two_page_attention
            else Qwen3_5ForwardTmaAttentionOp
        )
        ops = attention_cls.schedule(
            q=q,
            k=k,
            v=v,
            o=attn_h,
            lse=lse,
            causal=True,
            kv_group_size=KV_GROUP_SIZE,
            page_size=page_size,
            tile_sizes={"M": attention_tile_m},
            num_mma_warps=attention_mma_warps,
            write_lse=False,
        )
        if attention_n_block is not None:
            for op in ops:
                op.static_dims["tma_n_block"] = int(attention_n_block)
                if attention_cls is Qwen3_5ForwardThreePageTmaAttentionOp:
                    op.static_dims["tma_scratch_after_k"] = (
                        0 if int(attention_n_block) >= 64 else 1
                    )
        if attention_load_d_block is not None:
            for op in ops:
                op.static_dims["tma_load_d_block"] = int(attention_load_d_block)
                op.static_dims["tma_chunked_kv_swizzle"] = 1
    o_proj_tile_k = 32 if page_size <= 32 * 1024 else 64
    ops += Qwen3_5ForwardProjectionOp.schedule(
        a=attn,
        b=w_o,
        c=out,
        page_size=page_size,
        tile_sizes={"S": 64, "N": 64, "K": o_proj_tile_k},
    )

    kernel = Megakernel(
        ops,
        config=_make_config(ops, page_size, tracing=tracing),
        scheduler=scheduler,
    )
    spec = kernel.bench_spec(
        keep_alive=[
            q, k, v, w_o, attn, attn_h, out, lse,
            kernel, *ops, *attention_keep_alive,
        ]
    )
    return spec, out, kernel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--attention-tile-m", type=int, default=32)
    parser.add_argument(
        "--attention-mode",
        choices=("tma", "split-mma"),
        default="split-mma",
        help="Attention implementation to benchmark.",
    )
    parser.add_argument(
        "--attention-mma-warps",
        type=int,
        default=None,
        help="Override Qwen TMA attention MMA warp count.",
    )
    parser.add_argument(
        "--attention-n-block",
        type=int,
        default=None,
        help="Override Qwen TMA attention K/V block size.",
    )
    parser.add_argument(
        "--attention-load-d-block",
        type=int,
        default=None,
        help="Override chunked K/V TMA load D block while keeping 64-D MMA chunks.",
    )
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--two-page-attention", action="store_true")
    parser.add_argument("--three-page-attention", action="store_true")
    parser.add_argument("--split-attention-splits", type=int, default=1)
    parser.add_argument("--split-combine-tile-m", type=int, default=32)
    parser.add_argument("--fetch-stride", type=int, default=None)
    parser.add_argument("--prefer-ready-consumers", action="store_true")
    parser.add_argument(
        "--causal-rank-scheduler",
        action="store_true",
        help="Balance causal attention M tiles while keeping generic ready-consumer interleaving.",
    )
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="Skip torch timing and run exactly one CUDA-profiler-delimited Machete launch.",
    )
    parser.add_argument("--trace", default=None, help="Optional Perfetto JSON output path for the Machete kernel.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(2026)
    dtype = torch.bfloat16
    device = "cuda"
    q = torch.randn(args.batch, args.seq_len, NUM_Q_HEADS, HEAD_DIM, device=device, dtype=dtype)
    k = torch.randn(args.batch, args.seq_len, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    v = torch.randn(args.batch, args.seq_len, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    w_o = torch.randn(HIDDEN, Q_DIM, device=device, dtype=dtype) * 0.02

    scheduler_cls = Qwen3_5ForwardCausalRankScheduler if args.causal_rank_scheduler else OverlapTileScheduler
    scheduler = scheduler_cls(
        fetch_stride=args.fetch_stride,
        prefer_ready_consumers=args.prefer_ready_consumers,
    )
    spec, machete_out, kernel = _build_machete_spec(
        q,
        k,
        v,
        w_o,
        page_size=args.page_size,
        attention_mode=args.attention_mode,
        two_page_attention=args.two_page_attention,
        three_page_attention=args.three_page_attention,
        attention_tile_m=args.attention_tile_m,
        attention_mma_warps=args.attention_mma_warps,
        attention_n_block=args.attention_n_block,
        attention_load_d_block=args.attention_load_d_block,
        split_attention_splits=args.split_attention_splits,
        split_combine_tile_m=args.split_combine_tile_m,
        scheduler=scheduler,
        tracing=args.trace is not None,
    )

    def launch_machete():
        if spec.setup_fn is not None:
            spec.setup_fn()
        spec.launch_fn()

    if args.profile_only:
        launch_machete()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        launch_machete()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        print("profile_launches=1")
        return

    def torch_fn():
        return _sdpa_attention_o_proj(q, k, v, w_o)

    with suppress_stdout_stderr(suppress_torch_compile_logs()):
        compiled_torch_fn = torch.compile(torch_fn, mode="max-autotune")
        compiled_torch_fn()
        torch.cuda.synchronize()

    if spec.setup_fn is not None:
        spec.setup_fn()
    spec.launch_fn()
    torch.cuda.synchronize()

    ref = compiled_torch_fn()
    torch.cuda.synchronize()
    diff = (machete_out.float() - ref.float()).abs()

    torch_us = _time_callable(torch_fn, warmup=args.warmup, rep=args.rep)
    torch_compile_us = _time_callable(compiled_torch_fn, warmup=args.warmup, rep=args.rep)
    machete_us = _time_spec(spec, warmup=args.warmup, rep=args.rep)

    print(
        f"B={args.batch} S={args.seq_len} page={args.page_size // 1024}K "
        f"attention={args.attention_mode}"
        f"{'/3-page' if args.three_page_attention and args.attention_mode == 'tma' else '/2-page' if args.two_page_attention and args.attention_mode == 'tma' else ''} "
        f"tile_M={args.attention_tile_m} mma_warps={args.attention_mma_warps or 'auto'} "
        f"n_block={args.attention_n_block or 'auto'} load_D={args.attention_load_d_block or 'auto'} "
        f"split_combine_M={args.split_combine_tile_m} "
        f"fetch_stride={args.fetch_stride or 'auto'} prefer_consumers={args.prefer_ready_consumers} "
        f"causal_rank={args.causal_rank_scheduler}"
    )
    print(f"accuracy max_abs={diff.max().item():.6g} mean_abs={diff.mean().item():.6g}")
    print("name,us,gap_vs_torch_compile_us,speedup")
    for name, us in (
        ("torch.sdpa_o_proj", torch_us),
        ("torch.compile.sdpa_o_proj", torch_compile_us),
        ("machete.attn_o_proj", machete_us),
    ):
        print(f"{name},{us:.3f},{us - torch_compile_us:+.3f},{torch_compile_us / us:.3f}x")

    if args.trace:
        Path(args.trace).parent.mkdir(parents=True, exist_ok=True)
        kernel.write_trace_perfetto(args.trace)
        print(f"wrote {args.trace}")


if __name__ == "__main__":
    main()
