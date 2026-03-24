#!/usr/bin/env python
# Copyright (c) 2025, Machete Authors
"""Benchmark MoE Grouped GEMM: Megakernel vs PyTorch loop.

Compares GPU kernel execution time of the megakernel MoeGemmOp against
a naive PyTorch loop over experts (torch.matmul per expert). Both
implementations are measured using CUDA event timing for fair comparison.

Usage:
    python benchmarks/kernels/benchmark_moe_gemm.py
"""

import contextlib
import io

import torch

from machete.megakernel import Megakernel
from machete.kernels.moe import MoeGemmOp, moe_align_sort
from machete.kernels.utils import SingleOpKernel
from machete.utils.benchmark import Benchmark
from machete.utils.benchmark_utils import KernelBenchSpec

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


def moe_gemm_bytes(num_tokens, num_experts, topk, K, N, dtype, page_size):
    """Total bytes read + written for grouped GEMM."""
    elem_bytes = 2
    total_tokens = num_tokens * topk
    return (total_tokens * K + num_experts * N * K + total_tokens * N) * elem_bytes


def moe_gemm_flops(num_tokens, num_experts, topk, K, N, dtype, page_size):
    """FLOPs for grouped GEMM: 2 * total_tokens * N * K."""
    return 2 * num_tokens * topk * N * K


# =============================================================================
# Benchmark Setup
# =============================================================================

# Practical MoE configurations:
#   (num_tokens, topk, num_experts, K, N, dtype)
# Covers small/medium/large with varying expert counts.
_BASE_CONFIGS = [
    # Small: few tokens, few experts
    (64, 2, 8, 256, 256, "bfloat16"),
    (128, 2, 8, 256, 512, "bfloat16"),
    (256, 2, 8, 512, 512, "bfloat16"),
    # Medium: moderate scale
    (256, 2, 8, 512, 1024, "bfloat16"),
    (512, 2, 8, 1024, 1024, "bfloat16"),
    (512, 2, 16, 512, 1024, "bfloat16"),
    # Larger: more tokens and experts
    (512, 2, 16, 1024, 1024, "bfloat16"),
    (1024, 2, 8, 512, 1024, "bfloat16"),
    (1024, 2, 16, 1024, 1024, "bfloat16"),
    # Many experts (MoE-heavy)
    (256, 2, 64, 256, 256, "bfloat16"),
    (512, 2, 64, 256, 512, "bfloat16"),
    (1024, 2, 64, 512, 512, "bfloat16"),
]

CONFIGS = [c + (ps,) for c in _BASE_CONFIGS for ps in PAGE_SIZES]


@Benchmark.configs(
    ["num_tokens", "num_experts", "topk", "K", "N", "dtype", "page_size"],
    CONFIGS,
)
def bench_moe_gemm(num_tokens, num_experts, topk, K, N, dtype, page_size):
    """Setup MoE grouped GEMM benchmark."""
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

    torch.manual_seed(42)
    device = "cuda"

    x = torch.randn(num_tokens, K, dtype=torch_dtype, device=device)
    w = torch.randn(num_experts, N, K, dtype=torch_dtype, device=device)

    topk_ids = torch.randint(0, num_experts, (num_tokens, topk),
                             dtype=torch.int32, device=device)
    topk_weights = torch.randn(num_tokens, topk, dtype=torch.float32,
                               device=device).softmax(dim=-1)

    # Use _auto_tiles tile_M as block_size for align_sort
    tile_m_initial = 128  # _auto_tiles default for 48KB pages
    sorted_token_ids, expert_ids, sorted_weights, _ = (
        moe_align_sort(topk_ids, topk_weights, num_experts,
                       block_size=tile_m_initial)
    )
    total_padded = sorted_token_ids.shape[0]
    clamped = sorted_token_ids.clamp(max=num_tokens - 1).long()
    sorted_x = x[clamped]

    funcs = {}

    # --- PyTorch loop over experts ---
    # Wrapped as KernelBenchSpec to avoid CUDA graph capture (dynamic masking
    # is not graph-capturable, and failed capture corrupts allocator state).
    c_ref = torch.zeros(total_padded, N, dtype=torch_dtype, device=device)

    def pytorch_loop():
        c_ref.zero_()
        for e in range(num_experts):
            mask = expert_ids == e
            if not mask.any():
                continue
            x_e = sorted_x[mask]
            c_ref[mask] = (x_e.float() @ w[e].float().t()).to(torch_dtype)

    torch_stream = torch.cuda.Stream()
    cu_stream = torch_stream.cuda_stream
    funcs["pytorch_loop"] = KernelBenchSpec(
        launch_fn=pytorch_loop,
        stream=(torch_stream, cu_stream),
        _keep_alive=[sorted_x, w, expert_ids, c_ref],
    )

    # --- Megakernel + SingleOp MoeGemmOp ---
    if is_sm90_or_newer() and CUTLASS_AVAILABLE:
        try:
            c = torch.zeros(total_padded, N, dtype=torch_dtype, device=device)
            ops = MoeGemmOp.schedule(
                sorted_x=sorted_x, w=w, expert_ids=expert_ids, c=c, page_size=page_size,
            )
            config = MoeGemmOp.kernel_config(ops)
            kernel = Megakernel(ops, config=config)
            with contextlib.redirect_stdout(io.StringIO()):
                kernel.run()
            torch.cuda.synchronize()
            funcs["megakernel"] = kernel.bench_spec(
                setup_fn=lambda c=c: c.zero_(),
                keep_alive=[sorted_x, w, expert_ids, c],
            )
        except Exception:
            pass

        try:
            c_so = torch.zeros(total_padded, N, dtype=torch_dtype, device=device)
            so_ops = MoeGemmOp.schedule(
                sorted_x=sorted_x, w=w, expert_ids=expert_ids, c=c_so, page_size=page_size,
            )
            so_kernel = SingleOpKernel(so_ops)
            with contextlib.redirect_stdout(io.StringIO()):
                so_kernel.run()
            torch.cuda.synchronize()
            funcs["single_op"] = so_kernel.bench_spec(
                setup_fn=lambda c_so=c_so: c_so.zero_(),
                keep_alive=[sorted_x, w, expert_ids, c_so],
            )
        except Exception:
            pass

    return funcs


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    print("=" * 100)
    print("MoE Grouped GEMM Benchmark: Megakernel vs PyTorch")
    print("=" * 100)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.multi_processor_count} SMs)")
        print(f"SM_90+: {is_sm90_or_newer()}")
    print(f"CUTLASS: {CUTLASS_AVAILABLE}")
    print()

    bench_moe_gemm._benchmark.run(
        mode="kernel",
        flops=moe_gemm_flops,
        bytes_fn=moe_gemm_bytes,
        warmup=25,
        rep=100,
    )
