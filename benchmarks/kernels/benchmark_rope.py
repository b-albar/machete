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

from machete.megakernel import Megakernel, MegakernelConfig
from machete.kernels.rope import RopeOp
from machete.kernels.rope.ref import rope_pytorch, HAS_TRITON
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
    All float32 (4 bytes).
    """
    q_elems = batch * seq_len * n_heads * head_dim
    cs_elems = seq_len * (head_dim // 2)
    return (2 * q_elems + 2 * cs_elems) * 4


# =============================================================================
# Benchmark Setup
# =============================================================================


@Benchmark.parametrize("batch", [1, 4, 8])
@Benchmark.parametrize("seq_len", [128, 512, 2048])
@Benchmark.parametrize("n_heads", [32])
@Benchmark.parametrize("head_dim", [64, 128])
def bench_rope(batch, seq_len, n_heads, head_dim):
    """Setup RoPE benchmark functions for each implementation."""
    torch.manual_seed(42)
    q = torch.randn(batch, seq_len, n_heads, head_dim, dtype=torch.float32, device="cuda")
    cos = torch.randn(seq_len, head_dim // 2, dtype=torch.float32, device="cuda")
    sin = torch.randn(seq_len, head_dim // 2, dtype=torch.float32, device="cuda")

    funcs = {}

    # PyTorch (out-of-place)
    funcs["pytorch"] = lambda: rope_pytorch(q, cos, sin)

    # Triton (in-place)
    if HAS_TRITON:
        q_tri = q.clone()
        # Warmup Triton JIT compilation before graph capture
        rope_triton(q_tri, cos, sin)
        torch.cuda.synchronize()

        def triton_fn():
            q_tri.copy_(q)
            rope_triton(q_tri, cos, sin)

        funcs["triton"] = triton_fn

    # Megakernel (in-place, via bench_spec for raw kernel timing)
    if is_hopper_or_newer() and CUTLASS_AVAILABLE:
        b, s, h, d = batch, seq_len, n_heads, head_dim
        q_mk = q.clone()
        q_flat = q_mk.view(b * s, h, d)

        ops = [RopeOp.schedule(q=q_flat, cos=cos, sin=sin, tile_sizes={"M": 2, "H": 8})]
        kernel = Megakernel(ops, config=MegakernelConfig())

        # Trigger compilation + first run
        with contextlib.redirect_stdout(io.StringIO()):
            kernel.run()
        torch.cuda.synchronize()

        funcs["megakernel"] = kernel.bench_spec(
            setup_fn=lambda: q_mk.copy_(q),
            keep_alive=[q_flat, cos, sin],
        )

    return funcs


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    print("=" * 100)
    print("RoPE Benchmark: Megakernel vs PyTorch vs Triton")
    print("=" * 100)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.multi_processor_count} SMs)")
        print(f"Hopper+: {is_hopper_or_newer()}")
    print(f"Triton: {HAS_TRITON}")
    print(f"CUTLASS: {CUTLASS_AVAILABLE}")
    print()

    bench_rope._benchmark.run(
        mode="kernel",
        bytes_fn=rope_bytes,
        warmup=25,
        rep=100,
    )
