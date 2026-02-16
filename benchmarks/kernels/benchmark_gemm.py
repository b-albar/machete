#!/usr/bin/env python
# Copyright (c) 2025, Machete Authors
"""Benchmark GEMM: Megakernel vs PyTorch.

Compares GPU kernel execution time of the megakernel GemmOp (tensor core MMA
with TMA store_add K-reduction) against PyTorch's torch.matmul across
LLM-scale matrix shapes. Both implementations are measured using CUDA event
timing for fair comparison.

Usage:
    python benchmarks/kernels/benchmark_gemm.py
"""

import contextlib
import io

import torch

from machete.megakernel import Megakernel, MegakernelConfig
from machete.kernels.gemm import GemmOp
from machete.megakernel.paged_memory import PAGE_SIZE
from machete.utils.benchmark import Benchmark

try:
    import cutlass  # noqa: F401

    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False


def is_sm90_or_newer():
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor >= 90


def gemm_bytes(M, K, N, dtype):
    """Total bytes read + written for GEMM.

    Reads: A (M*K) + B (N*K)
    Writes: C (M*N)
    """
    elem_bytes = 2 if dtype in ("float16", "bfloat16") else 4
    return (M * K + N * K + M * N) * elem_bytes


def gemm_flops(M, K, N, dtype):
    """FLOPs for GEMM: 2*M*N*K (multiply-add)."""
    return 2 * M * N * K


def compute_tile_sizes(M, K, N, elem_bytes=2):
    """Compute tile sizes that fit in PAGE_SIZE (16KB).

    Constraints:
      - (tile_M + tile_N) * tile_K * elem_bytes <= PAGE_SIZE  (A+B during load)
      - tile_M * tile_N * elem_bytes <= PAGE_SIZE              (C during store)
      - tile_K must be a multiple of 16 (MMA instruction size)
    """
    tile_k = 32

    # Start with large tile_M, tile_N and shrink if needed
    for tile_m, tile_n in [(128, 64), (64, 64), (64, 32), (32, 32)]:
        ab_bytes = (tile_m + tile_n) * tile_k * elem_bytes
        c_bytes = tile_m * tile_n * elem_bytes
        if ab_bytes <= PAGE_SIZE and c_bytes <= PAGE_SIZE:
            return tile_m, tile_n, tile_k

    return 32, 32, 16


# =============================================================================
# Benchmark Setup
# =============================================================================

# LLM-scale shapes: (batch*seq, hidden, proj)
# Typical hidden=4096, FFN=4*hidden=16384, batch*seq=1-8192
@Benchmark.parametrize("dtype", ["float16", "bfloat16"])
@Benchmark.parametrize("N", [4096, 8192, 16384])
@Benchmark.parametrize("K", [4096, 8192])
@Benchmark.parametrize("M", [128, 512, 2048, 4096])
def bench_gemm(M, K, N, dtype):
    """Setup GEMM benchmark functions for each implementation."""
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
    elem_bytes = 2

    torch.manual_seed(42)
    a = torch.randn(M, K, dtype=torch_dtype, device="cuda")
    b = torch.randn(K, N, dtype=torch_dtype, device="cuda")
    b_t = b.t().contiguous()  # (N, K) layout for GemmOp

    funcs = {}

    # PyTorch (cuBLAS)
    funcs["pytorch"] = lambda: torch.matmul(a, b)

    # Megakernel GemmOp
    if is_sm90_or_newer() and CUTLASS_AVAILABLE:
        tile_m, tile_n, tile_k = compute_tile_sizes(M, K, N, elem_bytes)
        c = torch.zeros(M, N, dtype=torch_dtype, device="cuda")

        ops = [GemmOp.schedule(
            a=a, b=b_t, c=c,
            tile_sizes={"M": tile_m, "N": tile_n, "K": tile_k},
        )]
        # 5 warps: 4 MMA + 1 DMA = 160 threads
        config = MegakernelConfig(threads_per_block=160)
        kernel = Megakernel(ops, config=config)

        # Trigger compilation + first run
        with contextlib.redirect_stdout(io.StringIO()):
            kernel.run()
        torch.cuda.synchronize()

        funcs["megakernel"] = kernel.bench_spec(
            setup_fn=lambda: c.zero_(),
            keep_alive=[a, b_t, c],
        )

    return funcs


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    print("=" * 100)
    print("GEMM Benchmark: Megakernel vs PyTorch")
    print("=" * 100)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.multi_processor_count} SMs)")
        print(f"SM_90+: {is_sm90_or_newer()}")
    print(f"CUTLASS: {CUTLASS_AVAILABLE}")
    print()

    bench_gemm._benchmark.run(
        mode="kernel",
        flops=gemm_flops,
        bytes_fn=gemm_bytes,
        warmup=25,
        rep=100,
    )
