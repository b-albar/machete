# Copyright (c) 2025, Machete Authors
import torch
import triton
import triton.language as tl
from machete.kernels.rope import Rope
from machete.utils.testing import benchmark_op

# --- Triton Reference Kernels ---


# Unsloth RoPE (simplified for benchmark)
@triton.jit
def _rope_embedding_triton(
    q,
    q_stride,
    cos,
    cos_stride,
    sin,
    sin_stride,
    seqlen,
    head_dim: tl.constexpr,
    n_heads: tl.constexpr,
    block_size: tl.constexpr,
):
    row_position = tl.program_id(0)
    group_head_position = tl.program_id(1)
    col_offsets = tl.arange(0, block_size)
    half_head_dim = head_dim // 2
    mask = col_offsets < half_head_dim

    # Simplified load for benchmark
    s_idx = row_position % seqlen
    sin1 = tl.load(sin + s_idx * sin_stride + col_offsets, mask=mask, other=0)
    cos1 = tl.load(cos + s_idx * cos_stride + col_offsets, mask=mask, other=0)

    head_idx = group_head_position
    offs_q1 = row_position * q_stride + head_idx * head_dim + col_offsets
    offs_q2 = row_position * q_stride + head_idx * head_dim + col_offsets + half_head_dim

    q1 = tl.load(q + offs_q1, mask=mask, other=0).to(tl.float32)
    q2 = tl.load(q + offs_q2, mask=mask, other=0).to(tl.float32)

    tl.store(q + offs_q1, (q1 * cos1 - q2 * sin1).to(q.dtype.element_ty), mask=mask)
    tl.store(q + offs_q2, (q2 * cos1 + q1 * sin1).to(q.dtype.element_ty), mask=mask)


def rope_triton(q, cos, sin):
    b, s, h, d = q.shape
    q_view = q.view(b * s, h * d)
    block_size = triton.next_power_of_2(d // 2)
    _rope_embedding_triton[(b * s, h)](
        q_view, q_view.stride(0), cos, cos.stride(0), sin, sin.stride(0), s, d, h, block_size
    )
    return q


# --- Benchmarking Logic ---


def main():
    device = "cuda"
    dtype = torch.float16

    # --- RoPE Benchmark ---
    h, d = 32, 128
    rope_configs = {
        "BS=1, S=2048": (
            torch.randn(1, 2048, h, d, device=device, dtype=dtype),
            torch.randn(2048, d, device=device, dtype=dtype),
            torch.randn(2048, d, device=device, dtype=dtype),
        ),
        "BS=4, S=4096": (
            torch.randn(4, 4096, h, d, device=device, dtype=dtype),
            torch.randn(4096, d, device=device, dtype=dtype),
            torch.randn(4096, d, device=device, dtype=dtype),
        ),
    }

    rope_inst = Rope(dtype=dtype, head_dim=d)

    def rope_pytorch(q, cos, sin):
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        half = q.shape[-1] // 2
        rh_q = torch.cat((-q[..., half:], q[..., :half]), dim=-1)
        return q * cos + rh_q * sin

    def rope_numel_provider(args):
        # RoPE transfers: q (in/out) + cos (in) + sin (in)
        # However, many implementations are in-place or fused.
        # q: b*s*h*d (read) + b*s*h*d (write) = 2 * args[0].numel()
        # cos: s*d (read) = args[1].numel()
        # sin: s*d (read) = args[2].numel()
        return 2 * args[0].numel() + args[1].numel() + args[2].numel()

    benchmark_op(
        "RoPE Forward",
        rope_configs,
        {
            "PyTorch": rope_pytorch,
            "Triton": rope_triton,
            "cuteDSL": lambda q, c, s: rope_inst(q.clone(), c, s),
        },
        rope_numel_provider,
    )


if __name__ == "__main__":
    main()
