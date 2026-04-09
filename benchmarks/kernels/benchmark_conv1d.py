#!/usr/bin/env python
# Copyright (c) 2025, Machete Authors
"""Benchmark Conv1d: Megakernel vs PyTorch.

Compares GPU kernel execution time and memory throughput of causal
depthwise Conv1d implementations across realistic transformer shapes.

Implementations:
- PyTorch: Pure PyTorch reference (F.conv1d with causal padding)
- Megakernel: Machete megakernel framework

Usage:
    python benchmarks/kernels/benchmark_conv1d.py
"""

import contextlib
import io

import torch

from machete.megakernel import Megakernel
from machete.kernels.conv1d import Conv1dOp, Conv1dBwdOp
from machete.kernels.conv1d.ref import causal_conv1d_ref, causal_conv1d_bwd_ref
from machete.utils.benchmark import Benchmark

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


def conv1d_bytes_fwd(batch, seq_len, hidden_dim, kernel_size):
    """Total bytes read + written for Conv1d forward.

    Reads x (B*S*D) + w (D*K), writes y (B*S*D) → ~2*B*S*D + D*K elements.
    """
    D = hidden_dim
    K = kernel_size
    x_elems = batch * seq_len * D
    w_elems = D * K
    return (2 * x_elems + w_elems) * 2  # bf16 = 2 bytes


def conv1d_bytes_bwd(batch, seq_len, hidden_dim, kernel_size):
    """Total bytes read + written for Conv1d backward.

    Reads dy (B*S*D) + w (D*K), writes dx (B*S*D) → ~2*B*S*D + D*K elements.
    """
    D = hidden_dim
    K = kernel_size
    x_elems = batch * seq_len * D
    w_elems = D * K
    return (2 * x_elems + w_elems) * 2


# =============================================================================
# Forward Benchmark
# =============================================================================


@Benchmark.parametrize("batch", [1, 4])
@Benchmark.parametrize("seq_len", [512, 2048, 8192, 32768])
@Benchmark.parametrize("hidden_dim", [1024, 2048, 4096])
@Benchmark.parametrize("kernel_size", [4])
def bench_conv1d_fwd(kernel_size, hidden_dim, seq_len, batch):
    """Setup Conv1d forward benchmark functions."""
    torch.manual_seed(42)
    B, S, D, K = batch, seq_len, hidden_dim, kernel_size
    x = torch.randn(B, S, D, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(D, K, dtype=torch.bfloat16, device="cuda") * 0.1

    funcs = {}

    # PyTorch baseline
    def pytorch_conv1d():
        return causal_conv1d_ref(x, w)

    pytorch_conv1d()
    torch.cuda.synchronize()
    funcs["pytorch"] = pytorch_conv1d

    # Megakernel
    if is_hopper_or_newer() and CUTLASS_AVAILABLE:
        try:
            y = torch.empty_like(x)
            ops = Conv1dOp.schedule(x=x, w=w, y=y)
            config = Conv1dOp.kernel_config(ops)
            with contextlib.redirect_stdout(io.StringIO()):
                kernel = Megakernel(ops, config=config)
                kernel.run()
            torch.cuda.synchronize()
            funcs["megakernel"] = kernel.bench_spec(keep_alive=[x, w, y])
        except Exception:
            pass

    return funcs


# =============================================================================
# Backward Benchmark
# =============================================================================


@Benchmark.parametrize("batch", [1, 4])
@Benchmark.parametrize("seq_len", [512, 2048, 8192, 32768])
@Benchmark.parametrize("hidden_dim", [1024, 2048, 4096])
@Benchmark.parametrize("kernel_size", [4])
def bench_conv1d_bwd(kernel_size, hidden_dim, seq_len, batch):
    """Setup Conv1d backward benchmark functions."""
    torch.manual_seed(42)
    B, S, D, K = batch, seq_len, hidden_dim, kernel_size
    x = torch.randn(B, S, D, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(D, K, dtype=torch.bfloat16, device="cuda") * 0.1
    dy = torch.randn(B, S, D, dtype=torch.bfloat16, device="cuda")

    funcs = {}

    # PyTorch baseline
    def pytorch_conv1d_bwd():
        return causal_conv1d_bwd_ref(dy, x, w)

    pytorch_conv1d_bwd()
    torch.cuda.synchronize()
    funcs["pytorch"] = pytorch_conv1d_bwd

    if is_hopper_or_newer() and CUTLASS_AVAILABLE:
        try:
            dx = torch.empty_like(dy)
            ops = Conv1dBwdOp.schedule(dy=dy, w=w, dx=dx)
            config = Conv1dBwdOp.kernel_config(ops)
            with contextlib.redirect_stdout(io.StringIO()):
                kernel = Megakernel(ops, config=config)
                kernel.run()
            torch.cuda.synchronize()
            funcs["megakernel"] = kernel.bench_spec(
                keep_alive=[dy, w, dx])
        except Exception:
            pass

    return funcs


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    print("=" * 100)
    print("Conv1d Benchmark: Megakernel vs PyTorch")
    print("=" * 100)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.multi_processor_count} SMs)")
        print(f"Hopper+: {is_hopper_or_newer()}")
    print(f"CUTLASS (megakernel): {CUTLASS_AVAILABLE}")
    print()

    bench_conv1d_fwd._benchmark.run(
        mode="kernel",
        bytes_fn=conv1d_bytes_fwd,
        warmup=10,
        rep=50,
    )

    print()
    print("=" * 100)
    print("Conv1d Backward Benchmark")
    print("=" * 100)
    print()

    bench_conv1d_bwd._benchmark.run(
        mode="kernel",
        bytes_fn=conv1d_bytes_bwd,
        warmup=10,
        rep=50,
    )
