#!/usr/bin/env python
# Copyright (c) 2025, Machete Authors
"""Benchmark SM100 GEMM: Quack vs Machete Megakernel vs SingleOp vs cuBLAS.

Compares GPU kernel execution time of:
  - cuBLAS (torch.matmul) — baseline
  - Quack GemmSm100 (persistent UMMA GEMM)
  - Machete GemmSm100Op (megakernel UMMA GEMM)
  - Machete GemmSm100Op (single-op mode)

Requires SM100+ (Blackwell) GPU and CUTLASS.

Usage:
    python benchmarks/kernels/benchmark_gemm_sm100.py
"""

import contextlib
import io

import torch

from machete.megakernel import Megakernel
from machete.kernels.gemm import GemmSm100Op
from machete.kernels.utils import SingleOpKernel
from machete.utils.benchmark import Benchmark

try:
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Float32
    from cutlass.cute.runtime import from_dlpack, make_ptr
    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False

try:
    import cuda.bindings.driver as cuda
    from quack.gemm_sm100 import GemmSm100
    from quack.tile_scheduler import TileSchedulerOptions
    QUACK_AVAILABLE = True
except ImportError:
    QUACK_AVAILABLE = False

PAGE_SIZE = 65536  # 64KB — fits well on B200 (228KB smem)


def is_blackwell_available():
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 12


def gemm_bytes(M, K, N, dtype, page_size):
    elem_bytes = 2 if dtype in ("float16", "bfloat16") else 4
    return (M * K + N * K + M * N) * elem_bytes


def gemm_flops(M, K, N, dtype, page_size):
    return 2 * M * N * K


# =============================================================================
# Quack helper
# =============================================================================


def _make_quack_gemm(M, N, K, torch_dtype, a_2d, b_t, d_torch):
    """Compile quack GemmSm100 and return a callable for benchmarking."""
    if not QUACK_AVAILABLE:
        return None

    ab_dtype = cutlass.BFloat16 if torch_dtype == torch.bfloat16 else cutlass.Float16
    d_dtype = ab_dtype

    # A: (M, K, L=1) K-contiguous → torch (1, M, K) permuted to (M, K, 1)
    a_3d = a_2d.unsqueeze(-1).contiguous()  # (M, K, 1) with K stride=1
    # B: (N, K, L=1) K-contiguous → torch (1, N, K) permuted to (N, K, 1)
    b_3d = b_t.unsqueeze(-1).contiguous()  # (N, K, 1) with K stride=1
    # D: (M, N, L=1) N-contiguous → torch (1, M, N) permuted to (M, N, 1)
    d_3d = d_torch.unsqueeze(-1).contiguous()  # (M, N, 1) with N stride=1

    mA = from_dlpack(a_3d, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    mA.element_type = ab_dtype
    mB = from_dlpack(b_3d, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    mB.element_type = ab_dtype
    mD = from_dlpack(d_3d, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    mD.element_type = d_dtype

    # Pick MMA tiler: use 128x128 (no clustering for fair comparison)
    mma_tiler_mn = (128, 128)
    cluster_shape_mnk = (1, 1, 1)

    if not GemmSm100.can_implement(
        ab_dtype, Float32, d_dtype, mma_tiler_mn,
        cluster_shape_mnk[:2], M, N, K, 1, "k", "k", "n",
    ):
        return None

    gemm = GemmSm100(Float32, ab_dtype, mma_tiler_mn, cluster_shape_mnk)

    hardware_info = cutlass.utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(1)
    scheduler_args = TileSchedulerOptions(Int32(max_active_clusters))
    epi_args = gemm.EpilogueArguments()

    torch_stream = torch.cuda.current_stream()
    cu_stream = cuda.CUstream(torch_stream.cuda_stream)

    compiled = cute.compile(
        gemm, mA, mB, mD, None, epi_args, scheduler_args, None, cu_stream,
    )

    def run_quack():
        nonlocal cu_stream
        cu_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled(mA, mB, mD, None, epi_args, scheduler_args, None, cu_stream)

    return run_quack


# =============================================================================
# Benchmark
# =============================================================================


@Benchmark.parametrize("page_size", [PAGE_SIZE])
@Benchmark.parametrize("dtype", ["bfloat16"])
@Benchmark.parametrize("N", [4096, 8192, 16384])
@Benchmark.parametrize("K", [4096, 8192])
@Benchmark.parametrize("M", [128, 512, 2048, 4096])
def bench_gemm_sm100(M, K, N, dtype, page_size):
    """Setup SM100 GEMM benchmark: cuBLAS vs Quack vs Machete."""
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

    torch.manual_seed(42)
    a = torch.randn(1, M, K, dtype=torch_dtype, device="cuda")
    b = torch.randn(K, N, dtype=torch_dtype, device="cuda")
    b_t = b.t().contiguous()  # (N, K) layout

    funcs = {}

    # cuBLAS baseline
    a_2d = a.squeeze(0)
    funcs["cublas"] = lambda: torch.matmul(a_2d, b)

    if not (is_blackwell_available() and CUTLASS_AVAILABLE):
        return funcs

    # Quack GemmSm100
    try:
        d_quack = torch.zeros(M, N, dtype=torch_dtype, device="cuda")
        quack_fn = _make_quack_gemm(M, N, K, torch_dtype, a_2d, b_t, d_quack)
        if quack_fn is not None:
            quack_fn()
            torch.cuda.synchronize()
            funcs["quack"] = quack_fn
    except Exception:
        pass

    # Machete Megakernel GemmSm100Op
    try:
        c = torch.zeros(1, M, N, dtype=torch_dtype, device="cuda")
        ops = GemmSm100Op.schedule(a=a, b=b_t, c=c, page_size=page_size)
        config = GemmSm100Op.kernel_config(ops)
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

    # Machete SingleOp GemmSm100Op
    try:
        c_so = torch.zeros(1, M, N, dtype=torch_dtype, device="cuda")
        so_ops = GemmSm100Op.schedule(a=a, b=b_t, c=c_so, page_size=page_size)
        so_kernel = SingleOpKernel(so_ops)
        with contextlib.redirect_stdout(io.StringIO()):
            so_kernel.run()
        torch.cuda.synchronize()
        funcs["single_op"] = so_kernel.bench_spec(
            setup_fn=lambda c_so=c_so: c_so.zero_(),
            keep_alive=[a, b_t, c_so],
        )
    except Exception:
        pass

    return funcs


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    print("=" * 100)
    print("SM100 GEMM Benchmark: Quack vs Machete Megakernel vs SingleOp vs cuBLAS")
    print("=" * 100)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.multi_processor_count} SMs)")
        major, minor = torch.cuda.get_device_capability()
        print(f"SM: {major}.{minor}")
    print(f"CUTLASS: {CUTLASS_AVAILABLE}")
    print(f"Quack: {QUACK_AVAILABLE}")
    print()

    bench_gemm_sm100._benchmark.run(
        mode="kernel",
        flops=gemm_flops,
        bytes_fn=gemm_bytes,
        warmup=25,
        rep=100,
    )
