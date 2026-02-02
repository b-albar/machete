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
from machete.utils.benchmark_utils import KernelBenchSpec

if HAS_TRITON:
    from machete.kernels.rope.ref import rope_triton
else:
    print("WARNING: Triton not available — Triton benchmark will be skipped.")

try:
    from cutlass import Int32, Int64
    from cutlass.cute.testing import JitArguments
    import cuda.bindings.driver as cuda

    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False


def is_blackwell_available():
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 10


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

    # Megakernel (in-place, via KernelBenchSpec)
    if is_blackwell_available() and CUTLASS_AVAILABLE:
        b, s, h, d = batch, seq_len, n_heads, head_dim
        q_mk = q.clone()
        q_flat = q_mk.view(b * s, h, d)

        ops = [RopeOp.schedule(q=q_flat, cos=cos, sin=sin)]
        mk_config = MegakernelConfig()
        kernel = Megakernel(ops, config=mk_config)

        # Trigger compilation + first run
        with contextlib.redirect_stdout(io.StringIO()):
            kernel.run()
        torch.cuda.synchronize()

        compiled = kernel._compiled_kernel
        barriers = kernel._barriers_tensor

        # Create a dedicated stream for the workspace generator
        bench_stream = torch.cuda.Stream()
        cu_stream = cuda.CUstream(bench_stream.cuda_stream)

        def gen_workspace():
            q_mk.copy_(q)
            barriers.zero_()
            return JitArguments(
                Int64(kernel._instructions_tensor.data_ptr()),
                Int64(barriers.data_ptr()),
                Int64(kernel._op_configs_tensor.data_ptr()),
                Int64(0),  # trace_buffer_ptr (tracing disabled)
                Int32(kernel._num_instructions),
                cu_stream,
            )

        funcs["megakernel"] = KernelBenchSpec(
            compiled_kernel=compiled,
            workspace_generator=gen_workspace,
            stream=(bench_stream, cu_stream),
            workspace_count=1,
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
        print(f"Blackwell: {is_blackwell_available()}")
    print(f"Triton: {HAS_TRITON}")
    print(f"CUTLASS: {CUTLASS_AVAILABLE}")
    print()

    bench_rope._benchmark.run(
        mode="kernel",
        bytes_fn=rope_bytes,
        warmup=25,
        rep=100,
    )
