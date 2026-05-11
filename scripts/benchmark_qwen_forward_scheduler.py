#!/usr/bin/env python
"""Compare Qwen 3.5 layer-forward instruction schedulers.

This is a focused A/B helper for the megakernel tile scheduler. It times the
same full Qwen 3.5 single-layer forward megakernel with the default instruction
order and with stride-aware overlap scheduler variants.
"""

import argparse
import contextlib
import gc
import io

import torch

from machete.megakernel import OverlapTileScheduler
from machete.megakernel.ops import TileRange
from machete.utils.benchmark import Benchmark
from benchmarks.kernels.benchmark_qwen3_5_layer import (
    D2,
    HEAD_DIM,
    HIDDEN,
    INTERMEDIATE,
    KV_DIM,
    Q_DIM,
    megakernel_forward_build,
)


def _alloc_qwen_layer(batch: int, seq_len: int):
    torch.manual_seed(1234)
    dtype = torch.bfloat16
    device = "cuda"
    return (
        batch,
        seq_len,
        torch.randn(batch, seq_len, HIDDEN, dtype=dtype, device=device),
        torch.randn(batch, seq_len, HIDDEN, dtype=dtype, device=device),
        torch.randn(HIDDEN, dtype=dtype, device=device),
        torch.randn(Q_DIM, HIDDEN, dtype=dtype, device=device) * 0.02,
        torch.randn(KV_DIM, HIDDEN, dtype=dtype, device=device) * 0.02,
        torch.randn(KV_DIM, HIDDEN, dtype=dtype, device=device) * 0.02,
        torch.ones(HEAD_DIM, dtype=dtype, device=device),
        torch.ones(HEAD_DIM, dtype=dtype, device=device),
        torch.randn(seq_len, D2, dtype=dtype, device=device),
        torch.randn(seq_len, D2, dtype=dtype, device=device),
        torch.randn(HIDDEN, Q_DIM, dtype=dtype, device=device) * 0.02,
        torch.randn(HIDDEN, dtype=dtype, device=device),
        torch.randn(2 * INTERMEDIATE, HIDDEN, dtype=dtype, device=device) * 0.02,
        torch.randn(HIDDEN, INTERMEDIATE, dtype=dtype, device=device) * 0.02,
    )


def _time_variant(
    bench: Benchmark,
    args,
    page_size: int,
    scheduler,
    gemm_tile_range,
    warmup: int,
    rep: int,
):
    with contextlib.redirect_stdout(io.StringIO()):
        spec, out, residual = megakernel_forward_build(
            *args,
            page_size=page_size,
            scheduler=scheduler,
            gemm_tile_range=gemm_tile_range,
        )
    torch.cuda.synchronize()
    ms = bench._bench_kernel_func(spec, warmup=warmup, rep=rep)
    return ms, out, residual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--page-size", type=int, nargs="+", default=[32768, 49152])
    parser.add_argument("--range-blocks", type=int, nargs="+", default=[0, 2, 4])
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--rep", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    bench = Benchmark()
    variants = [("default", None), ("overlap", OverlapTileScheduler())]
    range_variants = [("single", None)]
    for blocks in args.range_blocks:
        if blocks > 0:
            range_variants.append((f"N{blocks}", TileRange.coalesced("N", block_size=blocks)))

    print("seq_len,page_size,range,scheduler,time_ms,speedup_vs_default")
    for seq_len in args.seq_len:
        for page_size in args.page_size:
            baseline_ms = None
            for range_name, gemm_tile_range in range_variants:
                qwen_args = _alloc_qwen_layer(args.batch, seq_len)
                timings = {}
                for name, scheduler in variants:
                    try:
                        timings[name], _, _ = _time_variant(
                            bench,
                            qwen_args,
                            page_size,
                            scheduler,
                            gemm_tile_range,
                            args.warmup,
                            args.rep,
                        )
                    except Exception as exc:
                        timings[name] = float("nan")
                        print(
                            f"# {name}/{range_name} failed for "
                            f"seq_len={seq_len} page_size={page_size}: {exc}"
                        )

                if baseline_ms is None:
                    baseline_ms = timings["default"]
                for name, ms in timings.items():
                    speedup = baseline_ms / ms if ms == ms and baseline_ms == baseline_ms else float("nan")
                    print(f"{seq_len},{page_size},{range_name},{name},{ms:.6f},{speedup:.6f}")

                del qwen_args
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
