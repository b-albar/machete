#!/usr/bin/env python
# Copyright (c) 2025, Machete Authors
"""Benchmark GLU: Megakernel vs PyTorch.

Compares GPU kernel execution time and memory throughput of GLU
(Gated Linear Unit) implementations across realistic transformer shapes.

Implementations:
- PyTorch: Pure PyTorch reference (F.silu(gate) * up)
- Megakernel: Machete megakernel framework (direct global access)
- SingleOp: Single-op kernel (no megakernel overhead)

Usage:
    python benchmarks/kernels/benchmark_glu.py
"""

import contextlib
import io

import torch
import torch.nn.functional as F

from machete.megakernel import Megakernel
from machete.kernels.glu import GLUOp, GLUBwdOp
from machete.kernels.utils import SingleOpKernel
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


def glu_bytes_fwd(batch, seq_len, hidden_dim):
    """Total bytes read + written for GLU forward.

    Reads x (B*S*2D), writes y (B*S*D) → 3*B*S*D elements.
    """
    D = hidden_dim
    elems = batch * seq_len * D
    return 3 * elems * 2  # bf16 = 2 bytes


def glu_bytes_bwd(batch, seq_len, hidden_dim):
    """Total bytes read + written for GLU backward.

    Reads dy (B*S*D) + x (B*S*2D), writes dx (B*S*2D) → 5*B*S*D elements.
    """
    D = hidden_dim
    elems = batch * seq_len * D
    return 5 * elems * 2


# =============================================================================
# Forward Benchmark
# =============================================================================


@Benchmark.parametrize("batch", [1, 4])
@Benchmark.parametrize("seq_len", [512, 2048, 8192, 32768])
@Benchmark.parametrize("hidden_dim", [1024, 2048, 4096])
def bench_glu_fwd(hidden_dim, seq_len, batch):
    """Setup GLU forward benchmark functions."""
    torch.manual_seed(42)
    M = batch * seq_len
    D = hidden_dim
    N = 2 * D
    x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    funcs = {}

    # PyTorch baseline
    def pytorch_glu():
        gate = x[:, :D]
        up = x[:, D:]
        return F.silu(gate) * up

    pytorch_glu()
    torch.cuda.synchronize()
    funcs["pytorch"] = pytorch_glu

    # Megakernel + SingleOp
    if is_hopper_or_newer() and CUTLASS_AVAILABLE:
        x_3d = x.view(batch, seq_len, N)

        try:
            y_3d = torch.empty(batch, seq_len, D, dtype=x.dtype, device="cuda")
            ops = GLUOp.schedule_forward(x=x_3d, y=y_3d)
            config = GLUOp.kernel_config(ops)
            with contextlib.redirect_stdout(io.StringIO()):
                kernel = Megakernel(ops, config=config)
                kernel.run()
            torch.cuda.synchronize()
            funcs["megakernel"] = kernel.bench_spec(keep_alive=[x_3d, y_3d])
        except Exception:
            pass

        try:
            y_so = torch.empty(batch, seq_len, D, dtype=x.dtype, device="cuda")
            so_ops = GLUOp.schedule_forward(x=x_3d, y=y_so)
            so_config = GLUOp.kernel_config(so_ops)
            so_kernel = SingleOpKernel(so_ops, config=so_config)
            with contextlib.redirect_stdout(io.StringIO()):
                so_kernel.run()
            torch.cuda.synchronize()
            funcs["single_op"] = so_kernel.bench_spec(keep_alive=[x_3d, y_so])
        except Exception:
            pass

    return funcs


# =============================================================================
# Backward Benchmark
# =============================================================================


@Benchmark.parametrize("batch", [1, 4])
@Benchmark.parametrize("seq_len", [512, 2048, 8192, 32768])
@Benchmark.parametrize("hidden_dim", [1024, 2048, 4096])
def bench_glu_bwd(hidden_dim, seq_len, batch):
    """Setup GLU backward benchmark functions."""
    torch.manual_seed(42)
    M = batch * seq_len
    D = hidden_dim
    N = 2 * D
    x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    dy = torch.randn(M, D, dtype=torch.bfloat16, device="cuda")

    funcs = {}

    # PyTorch baseline
    def pytorch_glu_bwd():
        gate = x[:, :D].float()
        up = x[:, D:].float()
        dy_f = dy.float()
        sig = torch.sigmoid(gate)
        silu_val = gate * sig
        silu_grad = sig * (1.0 + gate * (1.0 - sig))
        d_up = dy_f * silu_val
        d_gate = dy_f * up * silu_grad
        return torch.cat([d_gate.bfloat16(), d_up.bfloat16()], dim=-1)

    pytorch_glu_bwd()
    torch.cuda.synchronize()
    funcs["pytorch"] = pytorch_glu_bwd

    if is_hopper_or_newer() and CUTLASS_AVAILABLE:
        x_3d = x.view(batch, seq_len, N)
        dy_3d = dy.view(batch, seq_len, D)

        try:
            dx_3d = torch.empty_like(x_3d)
            ops = GLUBwdOp.schedule_forward(
                dy=dy_3d, x=x_3d, dx=dx_3d)
            config = GLUBwdOp.kernel_config(ops)
            with contextlib.redirect_stdout(io.StringIO()):
                kernel = Megakernel(ops, config=config)
                kernel.run()
            torch.cuda.synchronize()
            funcs["megakernel"] = kernel.bench_spec(
                keep_alive=[dy_3d, x_3d, dx_3d])
        except Exception:
            pass

        try:
            dx_so = torch.empty_like(x_3d)
            so_ops = GLUBwdOp.schedule_forward(
                dy=dy_3d, x=x_3d, dx=dx_so)
            so_config = GLUBwdOp.kernel_config(so_ops)
            so_kernel = SingleOpKernel(so_ops, config=so_config)
            with contextlib.redirect_stdout(io.StringIO()):
                so_kernel.run()
            torch.cuda.synchronize()
            funcs["single_op"] = so_kernel.bench_spec(
                keep_alive=[dy_3d, x_3d, dx_so])
        except Exception:
            pass

    return funcs


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    print("=" * 100)
    print("GLU Benchmark: Megakernel vs PyTorch")
    print("=" * 100)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.multi_processor_count} SMs)")
        print(f"Hopper+: {is_hopper_or_newer()}")
    print(f"CUTLASS (megakernel): {CUTLASS_AVAILABLE}")
    print()

    bench_glu_fwd._benchmark.run(
        mode="kernel",
        bytes_fn=glu_bytes_fwd,
        warmup=10,
        rep=50,
    )

    print()
    print("=" * 100)
    print("GLU Backward Benchmark")
    print("=" * 100)
    print()

    bench_glu_bwd._benchmark.run(
        mode="kernel",
        bytes_fn=glu_bytes_bwd,
        warmup=10,
        rep=50,
    )
