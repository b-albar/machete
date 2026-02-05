#!/usr/bin/env python
# Copyright (c) 2025, Machete Authors
"""Benchmark GEMM: Megakernel vs PyTorch vs cuBLAS.

Compares GPU kernel execution time and compute throughput of GEMM
implementations across various matrix sizes.

Implementations:
- PyTorch: torch.mm (backed by cuBLAS)
- Megakernel: Machete megakernel framework GemmOp

Usage:
    python benchmarks/kernels/benchmark_gemm.py
"""

import contextlib
import io

import torch

from machete.megakernel import Megakernel, MegakernelConfig
from machete.kernels.gemm import GemmOp
from machete.utils.benchmark import Benchmark
from machete.utils.testing import is_hopper_available

try:
    import cutlass  # noqa: F401
    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False


def gemm_flops(m, n, k):
    """Compute FLOPs for GEMM: 2 * M * N * K (multiply-add)."""
    return 2 * m * n * k


def gemm_bytes(m, n, k, dtype_bytes=2):
    """Total bytes read + written for GEMM.

    Reads: A (M*K) + B (N*K)
    Writes: C (M*N)
    """
    return (m * k + n * k + m * n) * dtype_bytes


# =============================================================================
# Benchmark Setup
# =============================================================================


@Benchmark.parametrize("m", [512, 1024, 2048, 4096])
@Benchmark.parametrize("n", [512, 1024, 2048, 4096])
@Benchmark.parametrize("k", [512, 1024, 2048])
def bench_gemm(m, n, k):
    """Setup GEMM benchmark functions for each implementation."""
    torch.manual_seed(42)
    dtype = torch.float16

    a = torch.randn(m, k, dtype=dtype, device="cuda")
    b = torch.randn(n, k, dtype=dtype, device="cuda")

    funcs = {}

    # PyTorch (cuBLAS)
    def pytorch_gemm():
        return torch.mm(a, b.t())

    funcs["pytorch"] = pytorch_gemm

    # Megakernel (requires Hopper+)
    if is_hopper_available() and CUTLASS_AVAILABLE:
        c = torch.empty(m, n, dtype=dtype, device="cuda")

        # Schedule GEMM operation
        scheduled_op = GemmOp.schedule(a=a, b=b, c=c)

        ops = [scheduled_op]
        config = MegakernelConfig(threads_per_block=256)

        try:
            kernel = Megakernel(ops, config=config)

            # Trigger compilation + first run
            with contextlib.redirect_stdout(io.StringIO()):
                kernel.run()
            torch.cuda.synchronize()

            funcs["megakernel"] = kernel.bench_spec(keep_alive=[a, b, c])
        except Exception as e:
            print(f"Megakernel compilation failed for M={m}, N={n}, K={k}: {e}")

    return funcs


@Benchmark.parametrize("size", [512, 1024, 2048, 4096])
def bench_gemm_square(size):
    """Benchmark square matrices (M=N=K)."""
    torch.manual_seed(42)
    dtype = torch.float16
    m = n = k = size

    a = torch.randn(m, k, dtype=dtype, device="cuda")
    b = torch.randn(n, k, dtype=dtype, device="cuda")

    funcs = {}

    # PyTorch (cuBLAS)
    funcs["pytorch"] = lambda: torch.mm(a, b.t())

    # Megakernel
    if is_hopper_available() and CUTLASS_AVAILABLE:
        c = torch.empty(m, n, dtype=dtype, device="cuda")
        scheduled_op = GemmOp.schedule(a=a, b=b, c=c)

        ops = [scheduled_op]
        config = MegakernelConfig(threads_per_block=256)

        try:
            kernel = Megakernel(ops, config=config)
            with contextlib.redirect_stdout(io.StringIO()):
                kernel.run()
            torch.cuda.synchronize()

            funcs["megakernel"] = kernel.bench_spec(keep_alive=[a, b, c])
        except Exception as e:
            print(f"Megakernel compilation failed for size={size}: {e}")

    return funcs


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    print("=" * 100)
    print("GEMM Benchmark: Megakernel vs PyTorch (cuBLAS)")
    print("=" * 100)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.multi_processor_count} SMs)")
        print(f"Hopper+: {is_hopper_available()}")
    print(f"CUTLASS (megakernel): {CUTLASS_AVAILABLE}")
    print()

    print("-" * 100)
    print("Square Matrix Benchmark (M=N=K)")
    print("-" * 100)

    bench_gemm_square._benchmark.run(
        mode="kernel",
        bytes_fn=lambda size: gemm_bytes(size, size, size),
        warmup=10,
        rep=50,
    )

    print()
    print("-" * 100)
    print("General Matrix Benchmark")
    print("-" * 100)

    # Run only a subset to keep benchmark time reasonable
    bench_gemm._benchmark.parameters["m"] = [1024, 2048]
    bench_gemm._benchmark.parameters["n"] = [1024, 2048]
    bench_gemm._benchmark.parameters["k"] = [1024]

    bench_gemm._benchmark.run(
        mode="kernel",
        bytes_fn=lambda m, n, k: gemm_bytes(m, n, k),
        warmup=10,
        rep=50,
    )
