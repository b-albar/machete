#!/usr/bin/env python
# Copyright (c) 2025, Machete Authors
"""Benchmark RMSNorm: Megakernel vs PyTorch vs Triton vs CUTLASS.

Compares GPU kernel execution time and memory throughput of RMSNorm
implementations across realistic transformer shapes.

Implementations:
- PyTorch: Pure PyTorch reference
- Triton: Triton JIT-compiled kernel
- CUTLASS: NVIDIA CUTLASS standalone kernel (SM90+ only, cluster-based reduction)
- Megakernel: Machete megakernel framework

Usage:
    python benchmarks/kernels/benchmark_rmsnorm.py
"""

import contextlib
import io

import torch

from machete.megakernel import Megakernel, MegakernelConfig
from machete.kernels.rms_norm import RMSNormOp
from machete.kernels.rms_norm.ref import rmsnorm_pytorch, HAS_TRITON, HAS_CUTLASS_RMSNORM
from machete.utils.benchmark import Benchmark

if HAS_TRITON:
    from machete.kernels.rms_norm.ref import rmsnorm_triton
else:
    print("WARNING: Triton not available — Triton benchmark will be skipped.")

if HAS_CUTLASS_RMSNORM:
    from machete.kernels.rms_norm.ref import rmsnorm_cutlass
else:
    print("WARNING: CUTLASS RMSNorm not available (requires SM90+) — CUTLASS benchmark will be skipped.")

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


def _calculate_threads(hidden_dim):
    """Calculate optimal threads per block based on hidden dimension.

    Matches Triton's configuration:
        block_size = next_power_of_2(hidden_dim), clamped to [32, 65536]
        num_warps = max(block_size // 256, 1), clamped to [1, 32]
        threads = num_warps * 32
    """
    # Next power of 2, clamped
    block_size = 1
    while block_size < hidden_dim:
        block_size *= 2
    block_size = max(block_size, 32)
    block_size = min(block_size, 65536)

    # Warps and threads
    num_warps = max(block_size // 256, 1)
    num_warps = min(num_warps, 32)
    return num_warps * 32


# =============================================================================
# Benchmark Setup
# =============================================================================


@Benchmark.parametrize("batch", [1, 4])
@Benchmark.parametrize("seq_len", [512, 2048, 8192, 32768])
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

    # CUTLASS (out-of-place, SM90+ only)
    # Note: CUTLASS standalone kernel may fail to compile due to dynamic layout
    # issues with the CuTe DSL. This is gracefully skipped if it fails.
    if HAS_CUTLASS_RMSNORM:
        try:
            # Warmup CUTLASS JIT
            rmsnorm_cutlass(x, weight)
            torch.cuda.synchronize()
            funcs["cutlass"] = lambda: rmsnorm_cutlass(x, weight)
        except Exception:
            pass  # Skip CUTLASS standalone if compilation fails

    # Megakernel (out-of-place, via bench_spec for raw kernel timing)
    if is_hopper_or_newer() and CUTLASS_AVAILABLE:
        y = torch.empty_like(x)
        tile_m = min(4, max(1, 16384 // (D * 4)))
        ops = [RMSNormOp.schedule(x=x, weight=weight, y=y, tile_sizes={"M": tile_m})]

        # Optimal thread config: 128 threads (4 warps) for warp-parallel rows
        # Each warp handles one row independently with warp-level reduction only
        config = MegakernelConfig(threads_per_block=128)
        kernel = Megakernel(ops, config=config)

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
    print("RMSNorm Benchmark: Megakernel vs PyTorch vs Triton vs CUTLASS")
    print("=" * 100)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.multi_processor_count} SMs)")
        print(f"Hopper+: {is_hopper_or_newer()}")
    print(f"Triton: {HAS_TRITON}")
    print(f"CUTLASS (megakernel): {CUTLASS_AVAILABLE}")
    print(f"CUTLASS (standalone): {HAS_CUTLASS_RMSNORM}")
    print()

    bench_rmsnorm._benchmark.run(
        mode="kernel",
        bytes_fn=rmsnorm_bytes,
        warmup=10,
        rep=50,
    )
