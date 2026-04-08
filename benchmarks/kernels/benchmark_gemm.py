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

from machete.megakernel import Megakernel
from machete.kernels.gemm import GemmOp
from machete.utils.benchmark import Benchmark

try:
    import cutlass  # noqa: F401

    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False

PAGE_SIZES = [16384, 32768, 49152]


def is_sm90_or_newer():
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor >= 90


def gemm_bytes(M, K, N, dtype, page_size):
    """Total bytes read + written for GEMM.

    Reads: A (M*K) + B (N*K)
    Writes: C (M*N)
    """
    elem_bytes = 2 if dtype in ("float16", "bfloat16") else 4
    return (M * K + N * K + M * N) * elem_bytes


def gemm_flops(M, K, N, dtype, page_size):
    """FLOPs for GEMM: 2*M*N*K (multiply-add)."""
    return 2 * M * N * K


# =============================================================================
# Benchmark Setup
# =============================================================================

# LLM-scale shapes: (batch*seq, hidden, proj)
# Typical hidden=4096, FFN=4*hidden=16384, batch*seq=1-8192
@Benchmark.parametrize("page_size", PAGE_SIZES)
@Benchmark.parametrize("dtype", ["bfloat16"])
@Benchmark.parametrize("N", [4096, 8192, 16384])
@Benchmark.parametrize("K", [4096, 8192])
@Benchmark.parametrize("M", [128, 512, 2048, 4096])
def bench_gemm(M, K, N, dtype, page_size):
    """Setup GEMM benchmark functions for each implementation."""
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

    torch.manual_seed(42)
    a = torch.randn(1, M, K, dtype=torch_dtype, device="cuda")
    b = torch.randn(K, N, dtype=torch_dtype, device="cuda")
    b_t = b.t().contiguous()  # (N, K) layout for GemmOp

    funcs = {}

    # PyTorch (cuBLAS) — squeeze to 2D for matmul
    a_2d = a.squeeze(0)
    funcs["pytorch"] = lambda: torch.matmul(a_2d, b)

    # Megakernel GemmOp
    if is_sm90_or_newer() and CUTLASS_AVAILABLE:
        try:
            c = torch.zeros(1, M, N, dtype=torch_dtype, device="cuda")
            ops = GemmOp.schedule(a=a, b=b_t, c=c, page_size=page_size)
            config = GemmOp.kernel_config(ops)
            kernel = Megakernel(ops, config=config)
            with contextlib.redirect_stdout(io.StringIO()):
                kernel.run()
            torch.cuda.synchronize()
            funcs["megakernel"] = kernel.bench_spec(
                setup_fn=lambda c=c: c.zero_(),
                keep_alive=[a, b_t, c],
            )
        except Exception:
            pass

    return funcs


# =============================================================================
# Backward Benchmark
# =============================================================================

@Benchmark.parametrize("page_size", PAGE_SIZES)
@Benchmark.parametrize("dtype", ["bfloat16"])
@Benchmark.parametrize("N", [4096, 8192])
@Benchmark.parametrize("K", [4096, 8192])
@Benchmark.parametrize("M", [128, 512, 2048, 4096])
def bench_gemm_bwd(M, K, N, dtype, page_size):
    """Setup GEMM backward benchmark (dA and dB gradients)."""
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

    torch.manual_seed(42)
    a = torch.randn(1, M, K, dtype=torch_dtype, device="cuda")
    b = torch.randn(K, N, dtype=torch_dtype, device="cuda")
    b_t = b.t().contiguous()  # (N, K) layout for GemmOp
    dout = torch.randn(1, M, N, dtype=torch_dtype, device="cuda")

    funcs = {}

    # PyTorch backward: dA = dout @ B^T, dB = A^T @ dout (squeeze to 2D)
    a_2d = a.squeeze(0)
    dout_2d = dout.squeeze(0)
    def pytorch_bwd():
        da = torch.matmul(dout_2d, b_t)
        db = torch.matmul(a_2d.t(), dout_2d)
        return da, db

    funcs["pytorch"] = pytorch_bwd

    # Megakernel backward
    if is_sm90_or_newer() and CUTLASS_AVAILABLE:
        try:
            da = torch.zeros(1, M, K, dtype=torch_dtype, device="cuda")
            db = torch.zeros(1, N, K, dtype=torch_dtype, device="cuda")
            ops = GemmOp.schedule_backward(
                dout=dout, a=a, b=b_t, da=da, db=db, page_size=page_size,
            )
            if ops:
                config = GemmOp.kernel_config(ops)
                kernel = Megakernel(ops, config=config)
                with contextlib.redirect_stdout(io.StringIO()):
                    kernel.run()
                torch.cuda.synchronize()
                funcs["megakernel"] = kernel.bench_spec(
                    setup_fn=lambda da=da, db=db: (da.zero_(), db.zero_()),
                    keep_alive=[a, b_t, dout, da, db],
                )
        except Exception:
            pass

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

    print()
    print("=" * 100)
    print("GEMM Backward Benchmark: Megakernel vs PyTorch")
    print("=" * 100)
    print()

    bench_gemm_bwd._benchmark.run(
        mode="kernel",
        warmup=25,
        rep=100,
    )
