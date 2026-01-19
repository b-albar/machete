# Copyright (c) 2025, Machete Authors
"""
RoPE kernel with proper L/C/S (Load/Compute/Store) separation for No Bubbles.

The L/C/S pattern:
- load():    Global memory → Shared memory (can overlap with prev op's store)
- compute(): Pure compute using shared memory (no global loads)
- store():   Shared memory → Global memory (can overlap with next op's load)
"""

import torch
from torch import Tensor
from typing import List
import cutlass.cute as cute
from cutlass import Float32, const_expr

from quack.cute_dsl_utils import torch2cute_dtype_map
from machete.kernels.utils.smem import MacheteSmemAllocator
from machete.megakernel.interface import FusableKernel, TensorParam
from machete.megakernel.single import SingleKernel


class RopeSM80(SingleKernel, FusableKernel):
    """
    RoPE kernel with proper L/C/S decomposition for No Bubbles pipelining.

    Shared memory layout per page:
    - [0, half_d): cos values
    - [half_d, head_dim): sin values
    - [head_dim, head_dim + n_heads * half_d): q0 values (first half of each head)
    - [head_dim + n_heads * half_d, ...): q1 values (second half)
    """

    NUM_THREADS = 256

    def __init__(self, dtype: torch.dtype, head_dim: int, n_heads: int):
        FusableKernel.__init__(self)
        self.torch_dtype = dtype
        self.cute_dtype = torch2cute_dtype_map[dtype]
        self.head_dim = head_dim
        self.half_head_dim = head_dim // 2
        self.n_heads = n_heads
        SingleKernel.__init__(self, self, self.grid_fn, self.block_fn)

    @property
    def tensor_params(self) -> List[TensorParam]:
        """Tensor parameters using index-based shape/stride.

        Args are: q (3D), cos (2D), sin (2D), seq_len (scalar), n_tokens (scalar)
        """
        return [
            TensorParam("q", shape=(0, 1, 2), stride=(0, 1, 2)),    # (n_tokens, n_heads, head_dim)
            TensorParam("cos", shape=(0, 1), stride=(0, 1)),        # (seq_len, half_d)
            TensorParam("sin", shape=(0, 1), stride=(0, 1)),        # (seq_len, half_d)
        ]

    @property
    def smem_size_fwd(self) -> int:
        """Shared memory for cos/sin values: 2 * half_head_dim * element_size."""
        return self.half_head_dim * 2 * (self.cute_dtype.width // 8)

    @property
    def smem_size_bwd(self) -> int:
        """Shared memory for backward (same as forward for RoPE)."""
        return self.smem_size_fwd

    def get_logical_grid_size(self, *args) -> int:
        """Return total work units (one per token).

        Args are: m_q, m_cos, m_sin, seq_len, n_tokens
        """
        # n_tokens is the 5th argument (index 4)
        return args[4]

    # ========== Forward Pass L/C/S ==========

    @cute.jit
    def load_forward(self, logical_idx, smem, m_q, m_cos, m_sin, seq_len, n_tokens):
        """Load cos/sin from global memory into shared memory."""
        tidx, _, _ = cute.arch.thread_idx()
        num_threads = const_expr(self.NUM_THREADS)
        half_d = const_expr(self.half_head_dim)
        s = logical_idx % seq_len

        # Coalesced load of cos/sin
        alloc = MacheteSmemAllocator(smem)
        s_cos = alloc.allocate_array(self.cute_dtype, half_d, byte_alignment=1)
        s_sin = alloc.allocate_array(self.cute_dtype, half_d, byte_alignment=1)

        for i in range(tidx, half_d, num_threads):
            s_cos[i] = m_cos[s, i]
            s_sin[i] = m_sin[s, i]

    @cute.jit
    def compute_forward(self, logical_idx, smem, m_q, m_cos, m_sin, seq_len, n_tokens):
        """
        Apply RoPE rotation using cos/sin from shared memory.
        Note: smem already contains cos/sin loaded by load_forward.
        """
        tidx, _, _ = cute.arch.thread_idx()
        num_threads = const_expr(self.NUM_THREADS)
        m = logical_idx  # Token index
        half_d = const_expr(self.half_head_dim)
        n_heads = const_expr(self.n_heads)

        alloc = MacheteSmemAllocator(smem)
        s_cos = alloc.allocate_array(self.cute_dtype, half_d, byte_alignment=1)
        s_sin = alloc.allocate_array(self.cute_dtype, half_d, byte_alignment=1)

        # Compute RoPE rotation in-place
        total_work = n_heads * half_d
        for work_idx in range(tidx, total_work, num_threads):
            h_idx = work_idx // half_d
            i_idx = work_idx % half_d

            # Read cos/sin from smem (fast)
            cos_val = s_cos[i_idx].to(Float32)
            sin_val = s_sin[i_idx].to(Float32)

            # Read q values from global
            q0 = m_q[m, h_idx, i_idx].to(Float32)
            q1 = m_q[m, h_idx, i_idx + half_d].to(Float32)

            # RoPE rotation
            r0 = q0 * cos_val - q1 * sin_val
            r1 = q1 * cos_val + q0 * sin_val

            # Write back to global (in-place)
            m_q[m, h_idx, i_idx] = r0.to(self.cute_dtype)
            m_q[m, h_idx, i_idx + half_d] = r1.to(self.cute_dtype)

    @cute.jit
    def store_forward(self, logical_idx, smem, m_q, m_cos, m_sin, seq_len, n_tokens):
        """No-op: RoPE writes directly to global in compute phase."""
        pass

    # ========== Backward Pass L/C/S ==========

    @cute.jit
    def load_backward(self, logical_idx, smem, m_q, m_cos, m_sin, seq_len, n_tokens):
        """Load cos/sin for backward pass."""
        tidx, _, _ = cute.arch.thread_idx()
        num_threads = const_expr(self.NUM_THREADS)
        half_d = const_expr(self.half_head_dim)
        s = logical_idx % seq_len

        alloc = MacheteSmemAllocator(smem)
        s_cos = alloc.allocate_array(self.cute_dtype, half_d, byte_alignment=1)
        s_sin = alloc.allocate_array(self.cute_dtype, half_d, byte_alignment=1)

        for i in range(tidx, half_d, num_threads):
            s_cos[i] = m_cos[s, i]
            s_sin[i] = m_sin[s, i]

    @cute.jit
    def compute_backward(self, logical_idx, smem, m_q, m_cos, m_sin, seq_len, n_tokens):
        """Backward RoPE: inverse rotation (negated sin)."""
        tidx, _, _ = cute.arch.thread_idx()
        num_threads = const_expr(self.NUM_THREADS)
        m = logical_idx  # Token index
        half_d = const_expr(self.half_head_dim)
        n_heads = const_expr(self.n_heads)

        alloc = MacheteSmemAllocator(smem)
        s_cos = alloc.allocate_array(self.cute_dtype, half_d, byte_alignment=1)
        s_sin = alloc.allocate_array(self.cute_dtype, half_d, byte_alignment=1)

        total_work = n_heads * half_d
        for work_idx in range(tidx, total_work, num_threads):
            h_idx = work_idx // half_d
            i_idx = work_idx % half_d

            cos_val = s_cos[i_idx].to(Float32)
            sin_val = -s_sin[i_idx].to(Float32)  # Negated for inverse

            q0 = m_q[m, h_idx, i_idx].to(Float32)
            q1 = m_q[m, h_idx, i_idx + half_d].to(Float32)

            r0 = q0 * cos_val - q1 * sin_val
            r1 = q1 * cos_val + q0 * sin_val

            m_q[m, h_idx, i_idx] = r0.to(self.cute_dtype)
            m_q[m, h_idx, i_idx + half_d] = r1.to(self.cute_dtype)

    @cute.jit
    def store_backward(self, logical_idx, smem, m_q, m_cos, m_sin, seq_len, n_tokens):
        """No-op: backward RoPE writes directly to global."""
        pass

    # ========== Launch Helpers ==========

    def grid_fn(self, q, cos, sin, seq_len, n_tokens):
        return [n_tokens, 1, 1]

    def block_fn(self, q, cos, sin, seq_len, n_tokens):
        return [self.NUM_THREADS, 1, 1]

    def __call__(self, q: Tensor, cos: Tensor, sin: Tensor):
        b, s, h, d = q.shape
        q_flat = q.view(b * s, h, d)
        half_d = d // 2
        # cos/sin have shape (s, d) but we only need the first half since they're symmetric
        cos_flat = cos[:, :half_d].contiguous().view(-1, half_d)
        sin_flat = sin[:, :half_d].contiguous().view(-1, half_d)
        n_tokens = b * s
        # s passed as seq_len parameter to kernel, n_tokens for tensor shapes
        out_flat = self.apply_autograd(q_flat, cos_flat, sin_flat, s, n_tokens)
        return out_flat.view(b, s, h, d)
