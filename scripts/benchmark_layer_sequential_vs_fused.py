#!/usr/bin/env python
"""Benchmark sequential 1-layer megakernel work vs one fused all-layer megakernel.

Uses a synthetic chained TMA op so the comparison isolates framework/runtime
overhead and scaling behavior with repeated layers.
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import time

import torch

if importlib.util.find_spec("cutlass") is None:
    raise SystemExit("Requires CUTLASS")

from machete.megakernel.megakernel import Megakernel, MegakernelConfig
from tests.megakernel.support_tma import (
    SYNTHETIC_TMA_N,
    SYNTHETIC_TMA_TILE_M,
    SyntheticTMAAddOneOp,
)


def make_chain_tensors(num_layers: int, *, device: str):
    tensors = [
        torch.zeros(
            SYNTHETIC_TMA_TILE_M,
            SYNTHETIC_TMA_N,
            dtype=torch.float16,
            device=device,
        )
        for _ in range(num_layers + 1)
    ]
    tensors[0].fill_(1.0)
    return tensors


def make_ops_from_chain(tensors):
    ops = []
    for i in range(len(tensors) - 1):
        ops.extend(
            SyntheticTMAAddOneOp.schedule(
                x=tensors[i],
                y=tensors[i + 1],
                tile_sizes={"M": SYNTHETIC_TMA_TILE_M},
            )
        )
    return ops


def build_kernels(num_layers: int, device: str, *, compile_kernels: bool):
    seq_tensors = make_chain_tensors(num_layers, device=device)
    fused_tensors = make_chain_tensors(num_layers, device=device)

    seq_kernels = []
    for i in range(num_layers):
        ops = SyntheticTMAAddOneOp.schedule(
            x=seq_tensors[i],
            y=seq_tensors[i + 1],
            tile_sizes={"M": SYNTHETIC_TMA_TILE_M},
        )
        kernel = Megakernel(
            ops,
            config=MegakernelConfig(num_sms=1),
            device=device,
        )
        if compile_kernels:
            kernel.compile()
        seq_kernels.append(kernel)

    fused_kernel = Megakernel(
        make_ops_from_chain(fused_tensors),
        config=MegakernelConfig(num_sms=1),
        device=device,
    )
    if compile_kernels:
        fused_kernel.compile()
    return seq_kernels, fused_kernel, seq_tensors, fused_tensors


def run_seq(seq_kernels):
    for kernel in seq_kernels:
        kernel.run(validate=False)


def run_fused(fused_kernel):
    fused_kernel.run(validate=False)


def bench_run_ms(fn, warmup: int, rep: int):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(rep):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return sum(samples) / len(samples), min(samples), max(samples)


def bench_prepare_ms(seq_kernels, fused_kernel, warmup: int, rep: int):
    def clone_seq():
        return [
            Megakernel(k.ops, config=k.config, device=k.device)
            for k in seq_kernels
        ]

    def clone_fused():
        return Megakernel(fused_kernel.ops, config=fused_kernel.config, device=fused_kernel.device)

    for _ in range(warmup):
        warm_seq = clone_seq()
        for kernel in warm_seq:
            kernel._prepare_tensors()
        warm_fused = clone_fused()
        warm_fused._prepare_tensors()

    seq_samples = []
    fused_samples = []
    for _ in range(rep):
        curr_seq = clone_seq()
        t0 = time.perf_counter()
        for kernel in curr_seq:
            kernel._prepare_tensors()
        t1 = time.perf_counter()
        curr_fused = clone_fused()
        t2 = time.perf_counter()
        curr_fused._prepare_tensors()
        t3 = time.perf_counter()
        seq_samples.append((t1 - t0) * 1000.0)
        fused_samples.append((t3 - t2) * 1000.0)
    return (
        statistics.mean(seq_samples),
        min(seq_samples),
        max(seq_samples),
        statistics.mean(fused_samples),
        min(fused_samples),
        max(fused_samples),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--metric", choices=("run", "prepare"), default="run")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.metric == "run" and not torch.cuda.is_available():
        raise SystemExit("CUDA required for --metric run")

    print("layers,sequential_ms,fused_ms,speedup_x,seq_min_ms,fused_min_ms,seq_max_ms,fused_max_ms")
    for layers in args.layers:
        try:
            seq_kernels, fused_kernel, _seq_tensors, _fused_tensors = build_kernels(
                layers,
                args.device,
                compile_kernels=args.metric == "run",
            )
            if args.metric == "run":
                seq_mean, seq_min, seq_max = bench_run_ms(
                    lambda: run_seq(seq_kernels), args.warmup, args.rep
                )
                fused_mean, fused_min, fused_max = bench_run_ms(
                    lambda: run_fused(fused_kernel), args.warmup, args.rep
                )
            else:
                seq_mean, seq_min, seq_max, fused_mean, fused_min, fused_max = bench_prepare_ms(
                    seq_kernels, fused_kernel, args.warmup, args.rep
                )
            speedup = seq_mean / fused_mean if fused_mean else float("inf")
            print(
                f"{layers},{seq_mean:.4f},{fused_mean:.4f},{speedup:.3f},"
                f"{seq_min:.4f},{fused_min:.4f},{seq_max:.4f},{fused_max:.4f}"
            )
        except Exception as exc:
            print(f"{layers},ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR  # {exc}")


if __name__ == "__main__":
    main()
