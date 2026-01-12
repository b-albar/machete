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
    Optimized RoPE kernel for SM80 using Managed Shared Memory.
    """

    def __init__(self, dtype: torch.dtype, head_dim: int):
        self.torch_dtype = dtype
        self.cute_dtype = torch2cute_dtype_map[dtype]
        self.head_dim = head_dim
        self.half_head_dim = head_dim // 2

        # Initialize SingleKernel runner
        self.runner = SingleKernel(self, self.grid_fn, self.block_fn)

    @property
    def smem_per_page(self) -> int:
        # Space for cos and sin: 2 * half_head_dim * element_size
        return 2 * self.half_head_dim * (self.cute_dtype.width // 8)

    @property
    def num_pages(self) -> int:
        return 1

    @machete_op(num_tensors=3, smem_per_page=1, num_pages=1)
    @cute.jit
    def compute_forward(
        self, smem_page: cute.Tensor, mq: cute.Tensor, m_cos: cute.Tensor, m_sin: cute.Tensor, seq_len: Int32
    ):
        self._compute_logic(smem_page, mq, m_cos, m_sin, seq_len, False)

    @machete_op(num_tensors=3, smem_per_page=1, num_pages=1)
    @cute.jit
    def compute_backward(
        self, smem_page: cute.Tensor, mq: cute.Tensor, m_cos: cute.Tensor, m_sin: cute.Tensor, seq_len: Int32
    ):
        self._compute_logic(smem_page, mq, m_cos, m_sin, seq_len, True)

    @cute.jit
    def _compute_logic(
        self,
        smem_page: cute.Tensor,
        mq: cute.Tensor,
        m_cos: cute.Tensor,
        m_sin: cute.Tensor,
        seq_len: Int32,
        backward: bool,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        num_threads, _, _ = cute.arch.block_dim()

        m = bidx
        s = m % seq_len

        half_d = const_expr(self.half_head_dim)
        n_heads = mq.shape[1]

        # Cooperatively load cos/sin into shared memory
        for i in range(tidx, half_d, num_threads):
            smem_page[i] = m_cos[s, i]
            smem_page[half_d + i] = m_sin[s, i]

        cute.arch.sync_threads()

        # Process all heads
        total_work = n_heads * half_d
        for work_idx in range(tidx, total_work, num_threads):
            h_idx = work_idx // half_d
            i_idx = work_idx % half_d

            cos_val = smem_page[i_idx].to(Float32)
            sin_val = smem_page[half_d + i_idx].to(Float32)

            if const_expr(backward):
                sin_val = -sin_val

            q0 = mq[m, h_idx, i_idx].to(Float32)
            q1 = mq[m, h_idx, i_idx + half_d].to(Float32)

            r0 = q0 * cos_val - q1 * sin_val
            r1 = q1 * cos_val + q0 * sin_val

            mq[m, h_idx, i_idx] = r0.to(mq.element_type)
            mq[m, h_idx, i_idx + half_d] = r1.to(mq.element_type)

    def grid_fn(self, q, cos, sin, seq_len):
        return [q.shape[0], 1, 1]

    def block_fn(self, q, cos, sin, seq_len):
        return [256, 1, 1]

    def __call__(self, q: Tensor, cos: Tensor, sin: Tensor):
        b, s, h, d = q.shape
        q_flat = q.view(b * s, h, d)
        cos_flat = cos.view(-1, d)
        sin_flat = sin.view(-1, d)

        # Inject seq_len required by kernel logic
        seq_len = s

        out_flat = MegakernelAutograd.apply(self.runner, q_flat, cos_flat, sin_flat, seq_len)
        return out_flat.view(b, s, h, d)
