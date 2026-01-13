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
import cutlass.cute as cute
from cutlass import Float32, const_expr

from quack.cute_dsl_utils import torch2cute_dtype_map
from machete.megakernel.interface import machete_op, FusableKernel
from machete.megakernel.single import SingleKernel
from machete.megakernel.autograd import MegakernelAutograd


class RopeSM80(FusableKernel):
    """
    RoPE kernel with proper L/C/S decomposition for No Bubbles pipelining.

    Shared memory layout per page:
    - [0, half_d): cos values
    - [half_d, head_dim): sin values
    - [head_dim, head_dim + n_heads * half_d): q0 values (first half of each head)
    - [head_dim + n_heads * half_d, ...): q1 values (second half)
    """

    def __init__(self, dtype: torch.dtype, head_dim: int):
        self.torch_dtype = dtype
        self.cute_dtype = torch2cute_dtype_map[dtype]
        self.head_dim = head_dim
        self.half_head_dim = head_dim // 2
        self.runner = SingleKernel(self, self.grid_fn, self.block_fn)

    @property
    def smem_per_page(self):
        # We need to store cos/sin in shared memory
        # 2 elements * half_head_dim * 2 bytes (half)
        return self.half_head_dim * 2 * 2

    @property
    def num_pages(self) -> int:
        return 1

    @property
    def needs_block_sync(self):
        # Thread-local operation when fused with element-wise ops
        return False

    # ========== Forward Pass L/C/S ==========

    @cute.jit
    def load_forward(self, paged_pool, page_idx, smem, mq, m_cos, m_sin, seq_len):
        """Load cos/sin from global memory into shared memory."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        num_threads, _, _ = cute.arch.block_dim()
        half_d = const_expr(self.half_head_dim)
        s = bidx % seq_len

        # Coalesced load of cos/sin
        for i in range(tidx, half_d, num_threads):
            smem[i] = m_cos[s, i]
            smem[half_d + i] = m_sin[s, i]

    @cute.jit
    def compute_forward(self, smem, mq, m_cos, m_sin, seq_len):
        """
        Apply RoPE rotation using cos/sin from shared memory.
        Note: smem already contains cos/sin loaded by load_forward.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        num_threads, _, _ = cute.arch.block_dim()
        m = bidx
        half_d = const_expr(self.half_head_dim)
        n_heads = mq.shape[1]

        # Compute RoPE rotation in-place
        total_work = n_heads * half_d
        for work_idx in range(tidx, total_work, num_threads):
            h_idx = work_idx // half_d
            i_idx = work_idx % half_d

            # Read cos/sin from smem (fast)
            cos_val = smem[i_idx].to(Float32)
            sin_val = smem[half_d + i_idx].to(Float32)

            # Read q values from global
            q0 = mq[m, h_idx, i_idx].to(Float32)
            q1 = mq[m, h_idx, i_idx + half_d].to(Float32)

            # RoPE rotation
            r0 = q0 * cos_val - q1 * sin_val
            r1 = q1 * cos_val + q0 * sin_val

            # Write back to global (in-place)
            mq[m, h_idx, i_idx] = r0.to(mq.element_type)
            mq[m, h_idx, i_idx + half_d] = r1.to(mq.element_type)

    @cute.jit
    def store_forward(self, paged_pool, page_idx, smem, mq, m_cos, m_sin, seq_len):
        """No-op: RoPE writes directly to global in compute phase."""
        pass

    # ========== Backward Pass L/C/S ==========

    @cute.jit
    def load_backward(self, paged_pool, page_idx, smem, mq, m_cos, m_sin, seq_len):
        """Load cos/sin for backward pass."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        num_threads, _, _ = cute.arch.block_dim()
        half_d = const_expr(self.half_head_dim)
        s = bidx % seq_len

        for i in range(tidx, half_d, num_threads):
            smem[i] = m_cos[s, i]
            smem[half_d + i] = m_sin[s, i]

    @cute.jit
    def compute_backward(self, smem, mq, m_cos, m_sin, seq_len):
        """Backward RoPE: inverse rotation (negated sin)."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        num_threads, _, _ = cute.arch.block_dim()
        m = bidx
        half_d = const_expr(self.half_head_dim)
        n_heads = mq.shape[1]

        total_work = n_heads * half_d
        for work_idx in range(tidx, total_work, num_threads):
            h_idx = work_idx // half_d
            i_idx = work_idx % half_d

            cos_val = smem[i_idx].to(Float32)
            sin_val = -smem[half_d + i_idx].to(Float32)  # Negated for inverse

            q0 = mq[m, h_idx, i_idx].to(Float32)
            q1 = mq[m, h_idx, i_idx + half_d].to(Float32)

            r0 = q0 * cos_val - q1 * sin_val
            r1 = q1 * cos_val + q0 * sin_val

            mq[m, h_idx, i_idx] = r0.to(mq.element_type)
            mq[m, h_idx, i_idx + half_d] = r1.to(mq.element_type)

    @cute.jit
    def store_backward(self, paged_pool, page_idx, smem, mq, m_cos, m_sin, seq_len):
        """No-op: backward RoPE writes directly to global."""
        pass

    # ========== Launch Helpers ==========

    def grid_fn(self, q, cos, sin, seq_len):
        return [q.shape[0], 1, 1]

    def block_fn(self, q, cos, sin, seq_len):
        return [256, 1, 1]

    def __call__(self, q: Tensor, cos: Tensor, sin: Tensor):
        b, s, h, d = q.shape
        q_flat = q.view(b * s, h, d)
        cos_flat = cos.view(-1, d)
        sin_flat = sin.view(-1, d)
        out_flat = MegakernelAutograd.apply(self.runner, q_flat, cos_flat, sin_flat, s)
        return out_flat.view(b, s, h, d)
