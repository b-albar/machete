# Copyright (c) 2025, Machete Authors
"""
Test No Bubbles scheduling with paged shared memory.
"""

import torch
import pytest
from machete.megakernel.core import Megakernel
from machete.kernels.gated_linear.sm80 import GatedLinearSM80
from machete.kernels.rope.sm80 import RopeSM80


def test_no_bubbles_basic():
    """Test basic No Bubbles mode with paged pool."""
    device = "cuda"
    dtype = torch.float16

    # Create tensors
    batch_size = 4
    seq_len = 128
    n_heads = 8
    head_dim = 64

    q = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device, dtype=dtype)
    cos = torch.randn(seq_len, head_dim, device=device, dtype=dtype)
    sin = torch.randn(seq_len, head_dim, device=device, dtype=dtype)

    # Create megakernel with paged pool
    rope = RopeSM80(dtype, head_dim)
    mk = Megakernel("test_no_bubbles", mode="forward", num_pages=4, page_size=16384)

    q_flat = q.view(batch_size * seq_len, n_heads, head_dim)
    mk.add(rope, q_flat, cos, sin, seq_len)

    grid = [batch_size * seq_len, 1, 1]
    block = [256, 1, 1]

    # Copy for reference
    q_ref = q.clone()

    # Launch
    mk.launch(grid[0], grid, block)
    torch.cuda.synchronize()

    # Verify result matches reference
    rope_ref = RopeSM80(dtype, head_dim)
    rope_ref(q_ref, cos, sin)

    torch.testing.assert_close(q.view_as(q_ref), q_ref, rtol=1e-3, atol=1e-3)


def test_no_bubbles_multi_op():
    """Test No Bubbles with multiple operations (RoPE + GatedLinear)."""
    device = "cuda"
    dtype = torch.float16

    batch_size = 2
    seq_len = 512
    n_heads = 32
    head_dim = 128
    hidden_dim = n_heads * head_dim

    # Tensors
    q = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device, dtype=dtype)
    cos = torch.randn(seq_len, head_dim, device=device, dtype=dtype)
    sin = torch.randn(seq_len, head_dim, device=device, dtype=dtype)
    gate = torch.randn(batch_size * seq_len, hidden_dim, device=device, dtype=dtype)
    out = torch.empty(batch_size * seq_len, hidden_dim, device=device, dtype=dtype)

    # Reference
    q_ref = q.clone()

    # Create megakernel with paged pool for pipelining
    rope = RopeSM80(dtype, head_dim)
    gl = GatedLinearSM80(dtype, "silu")

    # 4 pages × 16KB = 64KB paged pool
    mk = Megakernel("no_bubbles_multi", mode="forward", num_pages=4)

    q_flat = q.view(batch_size * seq_len, n_heads, head_dim)
    mk.add(rope, q_flat, cos, sin, seq_len)
    mk.add(gl, q.view(batch_size * seq_len, hidden_dim), gate, out, hidden_dim)

    grid = [batch_size * seq_len, 1, 1]
    block = [256, 1, 1]

    mk.launch(grid[0], grid, block)
    torch.cuda.synchronize()

    # Verify RoPE was applied
    rope_ref = RopeSM80(dtype, head_dim)
    rope_ref(q_ref, cos, sin)

    # Verify q was modified (RoPE applied)
    torch.testing.assert_close(q.view_as(q_ref), q_ref, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_no_bubbles_basic()
    test_no_bubbles_multi_op()
    print("All No Bubbles tests passed!")
