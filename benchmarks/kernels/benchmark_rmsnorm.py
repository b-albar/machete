#!/usr/bin/env python
# Copyright (c) 2025, Machete Authors
"""Benchmark RMSNorm: Megakernel vs PyTorch vs Triton.

Compares GPU kernel execution time and memory throughput of three RMSNorm
implementations across realistic transformer shapes.

Usage:
    python benchmarks/kernels/benchmark_rmsnorm.py
"""

import contextlib
import io

import torch

from machete.megakernel import Megakernel, MegakernelConfig
from machete.kernels.rms_norm import RMSNormOp
from machete.kernels.rms_norm.ref import rmsnorm_pytorch, HAS_TRITON
from machete.utils.benchmark import Benchmark

if HAS_TRITON:
    from machete.kernels.rms_norm.ref import rmsnorm_triton
else:
    print("WARNING: Triton not available — Triton benchmark will be skipped.")

try:
    import cutlass  # noqa: F401

    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False


def is_hopper_or_newer():
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


def rmsnorm_bytes(batch, seq_len, hidden_dim):
    """Total bytes read + written for RMSNorm.

    Reads: x (B*S*D) + weight (D)
    Writes: y (B*S*D)
    All float32 (4 bytes).
    """
    x_elems = batch * seq_len * hidden_dim
    w_elems = hidden_dim
    return (2 * x_elems + w_elems) * 4


# =============================================================================
# Benchmark Setup
# =============================================================================


@Benchmark.parametrize("batch", [1, 4, 8])
@Benchmark.parametrize("seq_len", [128, 512, 2048])
@Benchmark.parametrize("hidden_dim", [1024, 2048, 4096])
def bench_rmsnorm(batch, seq_len, hidden_dim):
    """Setup RMSNorm benchmark functions for each implementation."""
    torch.manual_seed(42)
    M = batch * seq_len
    D = hidden_dim
    x = torch.randn(M, D, dtype=torch.float32, device="cuda")
    weight = torch.randn(D, dtype=torch.float32, device="cuda")

    funcs = {}

    # PyTorch (out-of-place)
    funcs["pytorch"] = lambda: rmsnorm_pytorch(x, weight)

    # Triton (out-of-place)
    if HAS_TRITON:
        # Warmup Triton JIT
        rmsnorm_triton(x, weight)
        torch.cuda.synchronize()
        funcs["triton"] = lambda: rmsnorm_triton(x, weight)

    # Megakernel (out-of-place, via bench_spec for raw kernel timing)
    if is_hopper_or_newer() and CUTLASS_AVAILABLE:
        y = torch.empty_like(x)
        ops = [RMSNormOp.schedule(x=x, weight=weight, y=y)]
        kernel = Megakernel(ops, config=MegakernelConfig())

        # Trigger compilation + first run
        with contextlib.redirect_stdout(io.StringIO()):
            kernel.run()
        torch.cuda.synchronize()

        funcs["megakernel"] = kernel.bench_spec(keep_alive=[x, weight, y])

    return funcs


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    print("=" * 100)
    print("RMSNorm Benchmark: Megakernel vs PyTorch vs Triton")
    print("=" * 100)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.multi_processor_count} SMs)")
        print(f"Hopper+: {is_hopper_or_newer()}")
    print(f"Triton: {HAS_TRITON}")
    print(f"CUTLASS: {CUTLASS_AVAILABLE}")
    print()

    bench_rmsnorm._benchmark.run(
        mode="kernel",
        bytes_fn=rmsnorm_bytes,
        warmup=25,
        rep=100,
    )
