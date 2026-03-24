#!/usr/bin/env python
# Copyright (c) 2025, Machete Authors
"""Benchmark RMSNorm: Megakernel vs PyTorch vs Triton vs CUTLASS.

Compares GPU kernel execution time and memory throughput of RMSNorm
implementations across realistic transformer shapes.

Implementations:
- PyTorch: Pure PyTorch reference
- Triton: Triton JIT-compiled kernel
- CUTLASS: NVIDIA CUTLASS standalone kernel (SM90+ only, cluster-based reduction)
- Megakernel: Machete megakernel framework (direct global access)

Usage:
    python benchmarks/kernels/benchmark_rmsnorm.py
"""

import contextlib
import io

import torch

from machete.megakernel import Megakernel
from machete.kernels.rms_norm import RMSNormOp, RMSNormBwdOp
from machete.kernels.rms_norm.ref import rmsnorm_pytorch, rmsnorm_backward_pytorch, HAS_TRITON, HAS_CUTLASS_RMSNORM
from machete.kernels.utils import SingleOpKernel
from machete.utils.benchmark import Benchmark

if HAS_TRITON:
    from machete.kernels.rms_norm.ref import rmsnorm_triton, rmsnorm_backward_triton
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


def rmsnorm_bytes_fwd(batch, seq_len, hidden_dim):
    """Total bytes read + written for RMSNorm forward."""
    x_elems = batch * seq_len * hidden_dim
    w_elems = hidden_dim
    return (2 * x_elems + w_elems) * 2


# =============================================================================
# Benchmark Setup
# =============================================================================


@Benchmark.parametrize("batch", [1, 4])
@Benchmark.parametrize("seq_len", [512, 2048, 8192, 32768])
@Benchmark.parametrize("hidden_dim", [1024, 2048, 4096])
def bench_rmsnorm(hidden_dim, seq_len, batch):
    """Setup RMSNorm benchmark functions for each implementation."""
    torch.manual_seed(42)
    M = batch * seq_len
    D = hidden_dim
    x = torch.randn(M, D, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(D, dtype=torch.bfloat16, device="cuda")

    funcs = {}

    # PyTorch (out-of-place)
    funcs["pytorch"] = lambda: rmsnorm_pytorch(x, weight)

    # Triton (out-of-place)
    if HAS_TRITON:
        rmsnorm_triton(x, weight)
        torch.cuda.synchronize()
        funcs["triton"] = lambda: rmsnorm_triton(x, weight)

    # CUTLASS (out-of-place, SM90+ only)
    if HAS_CUTLASS_RMSNORM:
        try:
            rmsnorm_cutlass(x, weight)
            torch.cuda.synchronize()
            funcs["cutlass"] = lambda: rmsnorm_cutlass(x, weight)
        except Exception:
            pass

    # Megakernel + SingleOp forward
    if is_hopper_or_newer() and CUTLASS_AVAILABLE:
        x_3d = x.view(batch, seq_len, D)
        try:
            y_3d = torch.empty_like(x_3d)
            ops = RMSNormOp.schedule(x=x_3d, weight=weight, y=y_3d)
            config = RMSNormOp.kernel_config(ops)
            kernel = Megakernel(ops, config=config)
            with contextlib.redirect_stdout(io.StringIO()):
                kernel.run()
            torch.cuda.synchronize()
            funcs["megakernel"] = kernel.bench_spec(keep_alive=[x_3d, weight, y_3d])
        except Exception:
            pass

        try:
            y_so_3d = torch.empty_like(x_3d)
            so_ops = RMSNormOp.schedule(x=x_3d, weight=weight, y=y_so_3d)
            so_config = RMSNormOp.kernel_config(so_ops)
            so_kernel = SingleOpKernel(so_ops, config=so_config)
            with contextlib.redirect_stdout(io.StringIO()):
                so_kernel.run()
            torch.cuda.synchronize()
            funcs["single_op"] = so_kernel.bench_spec(keep_alive=[x_3d, weight, y_so_3d])
        except Exception:
            pass

    return funcs


# =============================================================================
# Backward Benchmark
# =============================================================================


def rmsnorm_bytes_bwd(batch, seq_len, hidden_dim):
    """Total bytes read + written for RMSNorm backward."""
    x_elems = batch * seq_len * hidden_dim
    w_elems = hidden_dim
    return (3 * x_elems + w_elems) * 2


@Benchmark.parametrize("batch", [1, 4])
@Benchmark.parametrize("seq_len", [512, 2048, 8192, 32768])
@Benchmark.parametrize("hidden_dim", [1024, 2048, 4096])
def bench_rmsnorm_bwd(hidden_dim, seq_len, batch):
    """Setup RMSNorm backward benchmark functions."""
    torch.manual_seed(42)
    M = batch * seq_len
    D = hidden_dim
    x = torch.randn(M, D, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(D, dtype=torch.bfloat16, device="cuda")
    dout = torch.randn(M, D, dtype=torch.bfloat16, device="cuda")
    funcs = {}

    # PyTorch backward
    funcs["pytorch"] = lambda: rmsnorm_backward_pytorch(dout, x, weight)

    # Triton backward
    if HAS_TRITON:
        rmsnorm_backward_triton(dout, x, weight)
        torch.cuda.synchronize()
        funcs["triton"] = lambda: rmsnorm_backward_triton(dout, x, weight)

    if is_hopper_or_newer() and CUTLASS_AVAILABLE:
        x_3d = x.view(batch, seq_len, D)
        dout_3d = dout.view(batch, seq_len, D)
        try:
            dx_3d = torch.empty_like(x_3d)
            bwd_ops = RMSNormBwdOp.schedule(
                dout=dout_3d, x=x_3d, weight=weight, dx=dx_3d)
            bwd_config = RMSNormBwdOp.kernel_config(bwd_ops)
            bwd_kernel = Megakernel(bwd_ops, config=bwd_config)
            with contextlib.redirect_stdout(io.StringIO()):
                bwd_kernel.run()
            torch.cuda.synchronize()
            funcs["megakernel"] = bwd_kernel.bench_spec(
                keep_alive=[dout_3d, x_3d, weight, dx_3d])
        except Exception:
            pass

        try:
            dx_so_3d = torch.empty_like(x_3d)
            so_bwd_ops = RMSNormBwdOp.schedule(
                dout=dout_3d, x=x_3d, weight=weight, dx=dx_so_3d)
            so_bwd_config = RMSNormBwdOp.kernel_config(so_bwd_ops)
            so_bwd_kernel = SingleOpKernel(so_bwd_ops, config=so_bwd_config)
            with contextlib.redirect_stdout(io.StringIO()):
                so_bwd_kernel.run()
            torch.cuda.synchronize()
            funcs["single_op"] = so_bwd_kernel.bench_spec(
                keep_alive=[dout_3d, x_3d, weight, dx_so_3d])
        except Exception:
            pass

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
        bytes_fn=rmsnorm_bytes_fwd,
        warmup=10,
        rep=50,
    )

    print()
    print("=" * 100)
    print("RMSNorm Backward Benchmark")
    print("=" * 100)
    print()

    bench_rmsnorm_bwd._benchmark.run(
        mode="kernel",
        bytes_fn=rmsnorm_bytes_bwd,
        warmup=10,
        rep=50,
    )
