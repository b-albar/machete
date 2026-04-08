#!/usr/bin/env python
# Copyright (c) 2025, Machete Authors
"""Benchmark fused GEMM+RMSNorm: fusion vs sequential.

Compares strategies for computing y = RMSNorm(x @ W, weight):
  1. fused:      Megakernel(GEMM + RMSNormOp)   — single persistent kernel
  2. sequential: Megakernel(GEMM) + Megakernel(RMSNormOp)  — 2 kernel launches

Usage:
    python benchmarks/kernels/benchmark_fusion_rmsnorm.py
"""

import contextlib
import io

import torch

from machete.utils.benchmark import Benchmark
from machete.utils.benchmark_utils import KernelBenchSpec

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


def _build_and_run(ops, config):
    """Build megakernel, warm up, return kernel."""
    from machete.megakernel import Megakernel
    kernel = Megakernel(ops, config=config)
    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()
    torch.cuda.synchronize()
    return kernel


def _merged_config(*configs):
    """Merge multiple MegakernelConfigs by taking max threads and page_size."""
    from machete.megakernel import MegakernelConfig
    return MegakernelConfig(
        threads_per_block=max(c.threads_per_block for c in configs),
        page_size=max(c.page_size for c in configs),
    )


# =============================================================================
# Benchmark: GEMM + RMSNorm fusion
# =============================================================================

PAGE_SIZE = 49152  # 48KB — default for Hopper


@Benchmark.parametrize("D", [1024, 2048, 4096])
@Benchmark.parametrize("M", [128, 512, 2048, 4096, 8192])
def bench_gemm_rmsnorm(M, D):
    """GEMM(x @ W) + RMSNorm: fused vs sequential."""
    if not (is_hopper_or_newer() and CUTLASS_AVAILABLE):
        return {}

    from machete.kernels.gemm import GemmOp
    from machete.kernels.rms_norm import RMSNormOp
    import cuda.bindings.driver as cuda

    dtype = torch.bfloat16
    torch.manual_seed(42)

    B = 1
    x = torch.randn(B, M, D, dtype=dtype, device="cuda") * 0.02
    W = torch.randn(D, D, dtype=dtype, device="cuda") * 0.02  # (N, K) for GemmOp
    rms_weight = torch.randn(D, dtype=dtype, device="cuda")

    funcs = {}

    # --- 1. fused: single Megakernel(GEMM + RMSNormOp) ---
    try:
        x_f = x.clone()
        h_f = torch.zeros(B, M, D, dtype=dtype, device="cuda")  # GEMM output
        y_f = torch.zeros(B, M, D, dtype=dtype, device="cuda")  # RMSNorm output

        gemm_ops = GemmOp.schedule(a=x_f, b=W, c=h_f, page_size=PAGE_SIZE)
        rms_ops = RMSNormOp.schedule(x=h_f, weight=rms_weight, y=y_f)

        all_ops = gemm_ops + rms_ops
        config = _merged_config(
            GemmOp.kernel_config(gemm_ops),
            RMSNormOp.kernel_config(rms_ops),
        )
        fused_kern = _build_and_run(all_ops, config)

        f_stream = torch.cuda.Stream()
        f_cu = cuda.CUstream(f_stream.cuda_stream)

        funcs["fused"] = KernelBenchSpec(
            launch_fn=lambda fk=fused_kern, fc=f_cu: fk.run(stream=fc, sync=False),
            setup_fn=lambda hf=h_f, yf=y_f: (hf.zero_(), yf.zero_()),
            stream=(f_stream, f_cu),
            _keep_alive=[fused_kern, x_f, W, rms_weight, h_f, y_f],
        )
    except Exception as e:
        print(f"  fused failed: {e}")

    # --- 2. sequential: Megakernel(GEMM) + Megakernel(RMSNormOp) ---
    try:
        x_sd = x.clone()
        h_sd = torch.zeros(B, M, D, dtype=dtype, device="cuda")
        y_sd = torch.zeros(B, M, D, dtype=dtype, device="cuda")

        gemm_ops_sd = GemmOp.schedule(a=x_sd, b=W, c=h_sd, page_size=PAGE_SIZE)
        gemm_kern_d = _build_and_run(gemm_ops_sd, GemmOp.kernel_config(gemm_ops_sd))

        dir_ops = RMSNormOp.schedule(x=h_sd, weight=rms_weight, y=y_sd)
        dir_config = RMSNormOp.kernel_config(dir_ops)
        dir_kern = Megakernel(dir_ops, config=dir_config)
        with contextlib.redirect_stdout(io.StringIO()):
            dir_kern.run()
        torch.cuda.synchronize()

        sd_stream = torch.cuda.Stream()
        sd_cu = cuda.CUstream(sd_stream.cuda_stream)

        funcs["sequential"] = KernelBenchSpec(
            launch_fn=lambda gk=gemm_kern_d, dk=dir_kern, sc=sd_cu: (
                gk.run(stream=sc, sync=False), dk.run(stream=sc, sync=False)),
            setup_fn=lambda hs=h_sd, ys=y_sd: (hs.zero_(), ys.zero_()),
            stream=(sd_stream, sd_cu),
            _keep_alive=[gemm_kern_d, dir_kern, x_sd, W, rms_weight, h_sd, y_sd],
        )
    except Exception as e:
        print(f"  sequential failed: {e}")

    return funcs


def total_bytes(M, D):
    """Total bytes: GEMM (A + B + C) + RMSNorm (input + weight + output), all bf16."""
    elem = 2  # bfloat16
    gemm_bytes = (M * D + D * D + M * D) * elem
    rms_bytes = (2 * M * D + D) * elem
    return gemm_bytes + rms_bytes


# =============================================================================
# Standalone RMSNorm comparison (vs Triton)
# =============================================================================


@Benchmark.parametrize("D", [1024, 2048, 4096])
@Benchmark.parametrize("M", [128, 512, 2048, 4096, 8192, 32768])
def bench_rmsnorm_standalone(M, D):
    """Standalone RMSNorm: Megakernel vs Triton."""
    if not (is_hopper_or_newer() and CUTLASS_AVAILABLE):
        return {}

    from machete.kernels.rms_norm import RMSNormOp
    from machete.kernels.rms_norm.ref import rmsnorm_pytorch, HAS_TRITON

    dtype = torch.bfloat16
    torch.manual_seed(42)

    B = 1
    x = torch.randn(B, M, D, dtype=dtype, device="cuda")
    weight = torch.randn(D, dtype=dtype, device="cuda")
    funcs = {}

    # PyTorch
    x_2d = x.view(M, D)
    funcs["pytorch"] = lambda: rmsnorm_pytorch(x_2d, weight)

    # Triton
    if HAS_TRITON:
        from machete.kernels.rms_norm.ref import rmsnorm_triton
        rmsnorm_triton(x_2d, weight)
        torch.cuda.synchronize()
        funcs["triton"] = lambda: rmsnorm_triton(x_2d, weight)

    # Single-op megakernel
    try:
        y_dir = torch.empty_like(x)
        dir_ops = RMSNormOp.schedule(x=x, weight=weight, y=y_dir)
        dir_config = RMSNormOp.kernel_config(dir_ops)
        dir_kern = Megakernel(dir_ops, config=dir_config)
        with contextlib.redirect_stdout(io.StringIO()):
            dir_kern.run()
        torch.cuda.synchronize()
        funcs["megakernel"] = dir_kern.bench_spec(keep_alive=[x, weight, y_dir])
    except Exception:
        pass

    return funcs


def rmsnorm_bytes(M, D):
    """RMSNorm: read x + weight, write y."""
    return (2 * M * D + D) * 2


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 100)
    print("GEMM + RMSNorm Fusion Benchmark")
    print("=" * 100)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.multi_processor_count} SMs)")
        print(f"Hopper+: {is_hopper_or_newer()}")
    print(f"CUTLASS: {CUTLASS_AVAILABLE}")
    print()

    print("-" * 100)
    print("1. Standalone RMSNorm: Megakernel vs Triton")
    print("-" * 100)
    bench_rmsnorm_standalone._benchmark.run(
        mode="kernel",
        bytes_fn=rmsnorm_bytes,
        warmup=10,
        rep=50,
        columns=["pytorch", "triton", "megakernel"],
    )

    print()
    print("-" * 100)
    print("2. GEMM + RMSNorm: Fused vs Sequential")
    print("   fused      = single Megakernel(GEMM + RMSNormOp)")
    print("   sequential = Megakernel(GEMM) + Megakernel(RMSNormOp)")
    print("-" * 100)
    bench_gemm_rmsnorm._benchmark.run(
        mode="kernel",
        warmup=10,
        rep=50,
        columns=["fused", "sequential"],
    )
