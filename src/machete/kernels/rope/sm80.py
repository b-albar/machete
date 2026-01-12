# Copyright (c) 2025, Machete Authors
from typing import Dict, Any

import torch
from torch import Tensor
import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

from quack.cute_dsl_utils import torch2cute_dtype_map
from quack.compile_utils import make_fake_tensor as fake_tensor
from machete.megakernel.interface import machete_op, FusableKernel


class RopeSM80(FusableKernel):
    """
    Optimized RoPE kernel for SM80 using Managed Shared Memory.
    """

    def __init__(self, dtype: torch.dtype, head_dim: int):
        self.torch_dtype = dtype
        self.cute_dtype = torch2cute_dtype_map[dtype]
        self.head_dim = head_dim
        self.half_head_dim = head_dim // 2
        self._compile_cache: Dict[bool, Any] = {}

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

    def __call__(self, q: Tensor, cos: Tensor, sin: Tensor, backward: bool = False):
        b, s, h, d = q.shape
        q_flat = q.view(b * s, h, d)
        cos = cos.view(-1, d)
        sin = sin.view(-1, d)

        if backward not in self._compile_cache:
            m_sym = cute.sym_int()
            s_sym = cute.sym_int()

            num_smem_elements = 2 * self.half_head_dim
            smem_fake = fake_tensor(self.cute_dtype, (num_smem_elements,))
            mq_fake = fake_tensor(self.cute_dtype, (m_sym, h, d))
            m_cos_fake = fake_tensor(self.cute_dtype, (s_sym, d))
            m_sin_fake = fake_tensor(self.cute_dtype, (s_sym, d))

            entry_point = self.backward if backward else self.forward

            self._compile_cache[backward] = cute.compile(
                entry_point,
                smem_fake,
                mq_fake,
                m_cos_fake,
                m_sin_fake,
                Int32(0),
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )

        num_smem_elements = 2 * self.half_head_dim
        # Pass a typed tensor for smem in standalone
        smem_dummy = torch.empty(num_smem_elements, dtype=self.torch_dtype, device="cuda")
        self._compile_cache[backward](smem_dummy, q_flat, cos, sin, s)
        return q

    @cute.jit
    def forward(
        self,
        smem_page: cute.Tensor,
        mq: cute.Tensor,
        m_cos: cute.Tensor,
        m_sin: cute.Tensor,
        seq_len: Int32,
        stream: cuda.CUstream,
    ):
        grid = [mq.shape[0], 1, 1]
        block = [256, 1, 1]
        self.forward_kernel(smem_page, mq, m_cos, m_sin, seq_len).launch(grid=grid, block=block, stream=stream)

    @cute.jit
    def backward(
        self,
        smem_page: cute.Tensor,
        mq: cute.Tensor,
        m_cos: cute.Tensor,
        m_sin: cute.Tensor,
        seq_len: Int32,
        stream: cuda.CUstream,
    ):
        grid = [mq.shape[0], 1, 1]
        block = [256, 1, 1]
        self.backward_kernel(smem_page, mq, m_cos, m_sin, seq_len).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def forward_kernel(
        self, smem_page: cute.Tensor, mq: cute.Tensor, m_cos: cute.Tensor, m_sin: cute.Tensor, seq_len: Int32
    ):
        self.compute_forward(smem_page, mq, m_cos, m_sin, seq_len)

    @cute.kernel
    def backward_kernel(
        self, smem_page: cute.Tensor, mq: cute.Tensor, m_cos: cute.Tensor, m_sin: cute.Tensor, seq_len: Int32
    ):
        self.compute_backward(smem_page, mq, m_cos, m_sin, seq_len)
