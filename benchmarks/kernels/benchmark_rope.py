#!/usr/bin/env python
# Copyright (c) 2025, Machete Authors
"""Benchmark RoPE: Megakernel vs PyTorch vs Triton.

Compares GPU kernel execution time and memory throughput of three RoPE
implementations across realistic transformer shapes. All implementations
are measured using CUDA graphs + CUDA event timing for fair comparison.

Usage:
    python benchmarks/kernels/benchmark_rope.py
"""

import contextlib
import io

import torch

from machete.megakernel import Megakernel
from machete.kernels.rope import RopeOp, RopeBwdOp
from machete.kernels.rope.ref import rope_pytorch, HAS_TRITON
from machete.kernels.utils import SingleOpKernel
from machete.utils.benchmark import Benchmark

if HAS_TRITON:
    from machete.kernels.rope.ref import rope_triton
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


def rope_bytes(batch, seq_len, n_heads, head_dim):
    """Total bytes read + written for RoPE.

    Reads: q (B*S*H*D) + cos (S*D/2) + sin (S*D/2)
    Writes: q (B*S*H*D)
    All bfloat16 (2 bytes).
    """
    q_elems = batch * seq_len * n_heads * head_dim
    cs_elems = seq_len * (head_dim // 2)
    return (2 * q_elems + 2 * cs_elems) * 2


# =============================================================================
# Forward Benchmark
# =============================================================================


@Benchmark.parametrize("batch", [1, 4, 8])
@Benchmark.parametrize("seq_len", [128, 512, 2048])
@Benchmark.parametrize("n_heads", [32])
@Benchmark.parametrize("head_dim", [64, 128])
def bench_rope_fwd(batch, seq_len, n_heads, head_dim):
    """Setup RoPE forward benchmark functions."""
    torch.manual_seed(42)
    q = torch.randn(batch, seq_len, n_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    cos = torch.randn(seq_len, head_dim // 2, dtype=torch.bfloat16, device="cuda")
    sin = torch.randn(seq_len, head_dim // 2, dtype=torch.bfloat16, device="cuda")

    funcs = {}

    # PyTorch (out-of-place)
    funcs["pytorch"] = lambda: rope_pytorch(q, cos, sin)

    # Triton (in-place)
    if HAS_TRITON:
        q_tri = q.clone()
        rope_triton(q_tri, cos, sin)
        torch.cuda.synchronize()

        def triton_fn():
            q_tri.copy_(q)
            rope_triton(q_tri, cos, sin)

        funcs["triton"] = triton_fn

    # Megakernel (in-place)
    if is_hopper_or_newer() and CUTLASS_AVAILABLE:
        q_mk = q.clone()

        ops = RopeOp.schedule(q=q_mk, cos=cos, sin=sin)
        config = RopeOp.kernel_config(ops)
        kernel = Megakernel(ops, config=config)

        with contextlib.redirect_stdout(io.StringIO()):
            kernel.run()
        torch.cuda.synchronize()

        funcs["megakernel"] = kernel.bench_spec(
            setup_fn=lambda: q_mk.copy_(q),
            keep_alive=[q_mk, cos, sin],
        )

    # SingleOpKernel (in-place)
    if is_hopper_or_newer() and CUTLASS_AVAILABLE:
        q_so = q.clone()
        so_ops = RopeOp.schedule(q=q_so, cos=cos, sin=sin)
        so_kernel = SingleOpKernel(so_ops)
        with contextlib.redirect_stdout(io.StringIO()):
            so_kernel.run()
        torch.cuda.synchronize()

        funcs["single_op"] = so_kernel.bench_spec(
            setup_fn=lambda: q_so.copy_(q),
            keep_alive=[q_so, cos, sin],
        )

    return funcs


# =============================================================================
# Backward Benchmark
# =============================================================================


@Benchmark.parametrize("batch", [1, 4, 8])
@Benchmark.parametrize("seq_len", [128, 512, 2048])
@Benchmark.parametrize("n_heads", [32])
@Benchmark.parametrize("head_dim", [64, 128])
def bench_rope_bwd(batch, seq_len, n_heads, head_dim):
    """Setup RoPE backward benchmark functions."""
    torch.manual_seed(42)
    q = torch.randn(batch, seq_len, n_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    cos = torch.randn(seq_len, head_dim // 2, dtype=torch.bfloat16, device="cuda")
    sin = torch.randn(seq_len, head_dim // 2, dtype=torch.bfloat16, device="cuda")

    funcs = {}

    # PyTorch (out-of-place, inverse rotation)
    funcs["pytorch"] = lambda: rope_pytorch(q, cos, -sin)

    # Megakernel backward (inverse rotation)
    if is_hopper_or_newer() and CUTLASS_AVAILABLE:
        q_bwd = q.clone()

        bwd_ops = RopeBwdOp.schedule(q=q_bwd, cos=cos, sin=sin)
        bwd_config = RopeBwdOp.kernel_config(bwd_ops)
        bwd_kernel = Megakernel(bwd_ops, config=bwd_config)

        with contextlib.redirect_stdout(io.StringIO()):
            bwd_kernel.run()
        torch.cuda.synchronize()

        funcs["megakernel"] = bwd_kernel.bench_spec(
            setup_fn=lambda: q_bwd.copy_(q),
            keep_alive=[q_bwd, cos, sin],
        )

    # SingleOpKernel backward
    if is_hopper_or_newer() and CUTLASS_AVAILABLE:
        q_so_bwd = q.clone()
        so_bwd_ops = RopeBwdOp.schedule(q=q_so_bwd, cos=cos, sin=sin)
        so_bwd_kernel = SingleOpKernel(so_bwd_ops)
        with contextlib.redirect_stdout(io.StringIO()):
            so_bwd_kernel.run()
        torch.cuda.synchronize()

        funcs["single_op"] = so_bwd_kernel.bench_spec(
            setup_fn=lambda: q_so_bwd.copy_(q),
            keep_alive=[q_so_bwd, cos, sin],
        )

    return funcs


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.multi_processor_count} SMs)")
        print(f"Hopper+: {is_hopper_or_newer()}")
    print(f"Triton: {HAS_TRITON}")
    print(f"CUTLASS: {CUTLASS_AVAILABLE}")
    print()

    print("=" * 100)
    print("RoPE Forward Benchmark: Megakernel vs PyTorch vs Triton")
    print("=" * 100)
    print()

    bench_rope_fwd._benchmark.run(
        mode="kernel",
        bytes_fn=rope_bytes,
        warmup=25,
        rep=100,
    )

    print()
    print("=" * 100)
    print("RoPE Backward Benchmark: Megakernel vs PyTorch")
    print("=" * 100)
    print()

    bench_rope_bwd._benchmark.run(
        mode="kernel",
        bytes_fn=rope_bytes,
        warmup=25,
        rep=100,
    )
