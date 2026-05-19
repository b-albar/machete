#!/usr/bin/env python
"""Compare Qwen 3.5 layer-backward instruction schedulers."""

import argparse
import contextlib
import gc
import io

import torch

from benchmarks.kernels.benchmark_qwen3_5_layer import (
    D2,
    HEAD_DIM,
    HIDDEN,
    INTERMEDIATE,
    KV_DIM,
    Q_DIM,
    megakernel_layer_bwd_build,
)
from machete.megakernel import OverlapTileScheduler
from machete.utils.benchmark import Benchmark


def _alloc_qwen_layer(batch: int, seq_len: int):
    torch.manual_seed(4321)
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


def _time_variant(bench: Benchmark, args, page_size: int, scheduler, warmup: int, rep: int):
    with contextlib.redirect_stdout(io.StringIO()):
        spec, _ = megakernel_layer_bwd_build(
            *args,
            page_size=page_size,
            scheduler=scheduler,
        )
    torch.cuda.synchronize()
    return bench._bench_kernel_func(spec, warmup=warmup, rep=rep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--page-size", type=int, nargs="+", default=[32768, 49152])
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--rep", type=int, default=12)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    bench = Benchmark()
    variants = [("default", None), ("overlap", OverlapTileScheduler())]

    print("seq_len,page_size,scheduler,time_ms,speedup_vs_default")
    for seq_len in args.seq_len:
        for page_size in args.page_size:
            qwen_args = _alloc_qwen_layer(args.batch, seq_len)
            timings = {}
            for name, scheduler in variants:
                try:
                    timings[name] = _time_variant(
                        bench,
                        qwen_args,
                        page_size,
                        scheduler,
                        args.warmup,
                        args.rep,
                    )
                except Exception as exc:
                    timings[name] = float("nan")
                    print(f"# {name} failed for seq_len={seq_len} page_size={page_size}: {exc}")

            default_ms = timings["default"]
            for name, ms in timings.items():
                speedup = default_ms / ms if ms == ms and default_ms == default_ms else float("nan")
                print(f"{seq_len},{page_size},{name},{ms:.6f},{speedup:.6f}")

            del qwen_args
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
