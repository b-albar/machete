# Copyright (c) 2025, Machete Authors
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import math
from typing import Tuple

from machete.kernels import Rope, geglu_func, swiglu_func, silu_func, gelu_func
from machete.kernels.rope_sm90 import RopeSM90
from machete.kernels.gated_linear_sm90 import gated_linear_sm90

# --- Triton Reference Kernels ---


# From Liger-Kernel
@triton.jit
def _geglu_tanh_forward_kernel(a, b, c, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0).to(tl.int64)
    a += program_id * stride
    b += program_id * stride
    c += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b + col_offsets, mask=mask, other=0)

    sqrt_2_over_pi = 0.7978845608028654
    a_cubed = a_row * a_row * a_row
    tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
    tanh_result = (
        tl.extra.cuda.libdevice.tanh(tanh_arg) if hasattr(tl.extra.cuda, "libdevice") else tl.math.tanh(tanh_arg)
    )
    geglu_a = 0.5 * a_row * (1 + tanh_result)
    c_row = geglu_a.to(b_row.dtype) * b_row
    tl.store(c + col_offsets, c_row, mask=mask)


def geglu_triton(a, b):
    n_rows, n_cols = a.shape[0], a.shape[1]
    c = torch.empty_like(a)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    _geglu_tanh_forward_kernel[(n_rows,)](a, b, c, a.stride(0), n_cols, BLOCK_SIZE)
    return c


# Unsloth RoPE (simplified for benchmark)
@triton.jit
def _rope_embedding_triton(
    Q,
    q_stride,
    cos,
    cos_stride,
    sin,
    sin_stride,
    seqlen,
    head_dim: tl.constexpr,
    n_heads: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_position = tl.program_id(0)
    group_head_position = tl.program_id(1)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    half_head_dim = head_dim // 2
    mask = col_offsets < half_head_dim

    # Simplified load for benchmark
    s_idx = row_position % seqlen
    sin1 = tl.load(sin + s_idx * sin_stride + col_offsets, mask=mask, other=0)
    cos1 = tl.load(cos + s_idx * cos_stride + col_offsets, mask=mask, other=0)

    head_idx = group_head_position
    offs_q1 = row_position * q_stride + head_idx * head_dim + col_offsets
    offs_q2 = row_position * q_stride + head_idx * head_dim + col_offsets + half_head_dim

    Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(tl.float32)
    Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(tl.float32)

    tl.store(Q + offs_q1, (Q1 * cos1 - Q2 * sin1).to(Q.dtype.element_ty), mask=mask)
    tl.store(Q + offs_q2, (Q2 * cos1 + Q1 * sin1).to(Q.dtype.element_ty), mask=mask)


def rope_triton(Q, cos, sin):
    B, S, H, D = Q.shape
    Q_view = Q.view(B * S, H * D)
    BLOCK_SIZE = triton.next_power_of_2(D // 2)
    _rope_embedding_triton[(B * S, H)](
        Q_view, Q_view.stride(0), cos, cos.stride(0), sin, sin.stride(0), S, D, H, BLOCK_SIZE
    )
    return Q


# --- Benchmarking Logic ---


def benchmark_op(name, configs, op_map):
    print(f"\n{'=' * 20} {name} {'=' * 20}")
    print(f"{'Config':<20} | {'Provider':<15} | {'Speed (GB/s)':<15} | {'Time (ms)':<10} | {'Peak Mem (MB)':<12}")
    print("-" * 85)

    for config_name, args in configs.items():
        for provider, func in op_map.items():
            # Estimate throughput (GB/s)
            numel = args[0].numel()
            if "SwiGLU" in name or "Gated" in name:
                numel *= 3  # (A, B, C)
            elif "RoPE" in name:
                numel = args[0].numel() + args[1].numel() * 2
            else:
                numel *= 2  # (X, Y)

            bytes_transferred = numel * args[0].element_size()

            # Time benchmark
            ms = triton.testing.do_bench(lambda: func(*args))
            gbps = bytes_transferred / (ms * 1e-3) / 1e9

            # Memory benchmark
            torch.cuda.reset_peak_memory_stats()
            base_mem = torch.cuda.memory_allocated()
            _ = func(*args)
            peak_mem = torch.cuda.max_memory_allocated()
            peak_delta_mb = (peak_mem - base_mem) / (1024 * 1024)

            print(f"{config_name:<20} | {provider:<15} | {gbps:<15.2f} | {ms:<10.4f} | {peak_delta_mb:<12.2f}")


def main():
    device = "cuda"
    dtype = torch.float16

    # --- RoPE Benchmark ---
    H, D = 32, 128
    rope_configs = {
        "BS=1, S=2048": (
            torch.randn(1, 2048, H, D, device=device, dtype=dtype),
            torch.randn(2048, D, device=device, dtype=dtype),
            torch.randn(2048, D, device=device, dtype=dtype),
        ),
        "BS=4, S=4096": (
            torch.randn(4, 4096, H, D, device=device, dtype=dtype),
            torch.randn(4096, D, device=device, dtype=dtype),
            torch.randn(4096, D, device=device, dtype=dtype),
        ),
    }

    rope_inst = Rope(dtype=dtype, head_dim=D)

    def rope_pytorch(Q, cos, sin):
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        half = Q.shape[-1] // 2
        RH_Q = torch.cat((-Q[..., half:], Q[..., :half]), dim=-1)
        return Q * cos + RH_Q * sin

    benchmark_op(
        "RoPE Forward",
        rope_configs,
        {
            "PyTorch": rope_pytorch,
            "Triton": rope_triton,
            "cuteDSL": lambda Q, c, s: rope_inst(Q.clone(), c, s),
            "cuteDSL_SM90": lambda Q, c, s: RopeSM90.apply(Q.clone(), c, s),
        },
    )

    # --- Gated Benchmark (SwiGLU) ---
    gated_configs = {
        "4k x 4k": (
            torch.randn(4096, 4096, device=device, dtype=dtype),
            torch.randn(4096, 4096, device=device, dtype=dtype),
        ),
        "8k x 8k": (
            torch.randn(8192, 8192, device=device, dtype=dtype),
            torch.randn(8192, 8192, device=device, dtype=dtype),
        ),
    }

    benchmark_op(
        "SwiGLU Forward",
        gated_configs,
        {
            "PyTorch": lambda a, b: F.silu(a) * b,
            "Triton": geglu_triton,  # Using same kernel structure
            "cuteDSL": swiglu_func,
            "cuteDSL_SM90": lambda a, b: gated_linear_sm90(a, b, "silu"),
        },
    )

    # --- Activation Benchmark (SiLU) ---
    act_configs = {
        "16M elements": (torch.randn(16 * 1024 * 1024, device=device, dtype=dtype),),
        "64M elements": (torch.randn(64 * 1024 * 1024, device=device, dtype=dtype),),
    }

    benchmark_op("SiLU Forward", act_configs, {"PyTorch": F.silu, "cuteDSL": silu_func})


if __name__ == "__main__":
    main()
