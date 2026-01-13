# Copyright (c) 2025, Machete Authors

import torch
from torch import Tensor
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

from quack.cute_dsl_utils import torch2cute_dtype_map
from machete.megakernel.interface import machete_op, FusableKernel
from machete.megakernel.single import SingleKernel
from machete.megakernel.autograd import MegakernelAutograd


class RopeSM80(FusableKernel):
    """
    Optimized RoPE kernel for SM80 using L/C/S decomposition for No Bubbles pipelining.
    """

    def __init__(self, dtype: torch.dtype, head_dim: int):
        self.torch_dtype = dtype
        self.cute_dtype = torch2cute_dtype_map[dtype]
        self.head_dim = head_dim
        self.half_head_dim = head_dim // 2
        self.runner = SingleKernel(self, self.grid_fn, self.block_fn)

    @property
    def smem_per_page(self) -> int:
        return 2 * self.half_head_dim * (self.cute_dtype.width // 8)

    @property
    def num_pages(self) -> int:
        return 1

    # ========== Forward Pass ==========

    @cute.jit
    def load_forward(self, paged_pool, page_idx, smem_page, mq, m_cos, m_sin, seq_len):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        num_threads, _, _ = cute.arch.block_dim()
        half_d = const_expr(self.half_head_dim)
        s = bidx % seq_len
        for i in range(tidx, half_d, num_threads):
            smem_page[i] = m_cos[s, i]
            smem_page[half_d + i] = m_sin[s, i]

    @machete_op(num_tensors=3, smem_per_page=1, num_pages=1)
    @cute.jit
    def compute_forward(self, smem_page, mq, m_cos, m_sin, seq_len):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        num_threads, _, _ = cute.arch.block_dim()
        m = bidx
        half_d = const_expr(self.half_head_dim)
        n_heads = mq.shape[1]

        # Load cos/sin
        s = m % seq_len
        for i in range(tidx, half_d, num_threads):
            smem_page[i] = m_cos[s, i]
            smem_page[half_d + i] = m_sin[s, i]
        cute.arch.sync_threads()

        # Compute
        total_work = n_heads * half_d
        for work_idx in range(tidx, total_work, num_threads):
            h_idx = work_idx // half_d
            i_idx = work_idx % half_d
            cos_val = smem_page[i_idx].to(Float32)
            sin_val = smem_page[half_d + i_idx].to(Float32)
            q0 = mq[m, h_idx, i_idx].to(Float32)
            q1 = mq[m, h_idx, i_idx + half_d].to(Float32)
            r0 = q0 * cos_val - q1 * sin_val
            r1 = q1 * cos_val + q0 * sin_val
            mq[m, h_idx, i_idx] = r0.to(mq.element_type)
            mq[m, h_idx, i_idx + half_d] = r1.to(mq.element_type)

    @cute.jit
    def store_forward(self, paged_pool, page_idx, smem_page, mq, m_cos, m_sin, seq_len):
        pass  # In-place

    # ========== Backward Pass ==========

    @cute.jit
    def load_backward(self, paged_pool, page_idx, smem_page, mq, m_cos, m_sin, seq_len):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        num_threads, _, _ = cute.arch.block_dim()
        half_d = const_expr(self.half_head_dim)
        s = bidx % seq_len
        for i in range(tidx, half_d, num_threads):
            smem_page[i] = m_cos[s, i]
            smem_page[half_d + i] = m_sin[s, i]

    @machete_op(num_tensors=3, smem_per_page=1, num_pages=1)
    @cute.jit
    def compute_backward(self, smem_page, mq, m_cos, m_sin, seq_len):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        num_threads, _, _ = cute.arch.block_dim()
        m = bidx
        half_d = const_expr(self.half_head_dim)
        n_heads = mq.shape[1]

        # Load cos/sin
        s = m % seq_len
        for i in range(tidx, half_d, num_threads):
            smem_page[i] = m_cos[s, i]
            smem_page[half_d + i] = m_sin[s, i]
        cute.arch.sync_threads()

        # Compute (negated sin for backward)
        total_work = n_heads * half_d
        for work_idx in range(tidx, total_work, num_threads):
            h_idx = work_idx // half_d
            i_idx = work_idx % half_d
            cos_val = smem_page[i_idx].to(Float32)
            sin_val = -smem_page[half_d + i_idx].to(Float32)
            q0 = mq[m, h_idx, i_idx].to(Float32)
            q1 = mq[m, h_idx, i_idx + half_d].to(Float32)
            r0 = q0 * cos_val - q1 * sin_val
            r1 = q1 * cos_val + q0 * sin_val
            mq[m, h_idx, i_idx] = r0.to(mq.element_type)
            mq[m, h_idx, i_idx + half_d] = r1.to(mq.element_type)

    @cute.jit
    def store_backward(self, paged_pool, page_idx, smem_page, mq, m_cos, m_sin, seq_len):
        pass  # In-place

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
