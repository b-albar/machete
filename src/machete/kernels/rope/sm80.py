# Copyright (c) 2025, Machete Authors
import math
from typing import Optional, Type, Dict, Tuple

import torch
from torch import Tensor
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Float16, BFloat16, Int32, Int64, const_expr

from quack.cute_dsl_utils import torch2cute_dtype_map
from quack.compile_utils import make_fake_tensor as fake_tensor
import quack.copy_utils as copy_utils
import quack.layout_utils as layout_utils


class RopeImpl:
    """
    Optimized RoPE kernel for SM80.

    Key optimizations:
    1. Vectorized loads/stores (128-bit = 8 fp16 elements)
    2. Each thread block handles one (batch*seq) position
    3. All heads processed by the thread block with good coalescing
    4. Cos/Sin loaded once and reused across all heads
    """

    def __init__(self, dtype: Type[cutlass.Numeric], head_dim: int, backward: bool):
        self.dtype = dtype
        self.head_dim = head_dim
        self.half_head_dim = head_dim // 2
        self.backward = backward

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (M, H, D)
        mCos: cute.Tensor,  # (S, D//2) - only first half
        mSin: cute.Tensor,  # (S, D//2) - only first half
        seq_len: Int32,
        stream: cuda.CUstream,
    ):
        # Each block handles one M position, all heads
        # 256 threads to cover half_head_dim elements with good occupancy
        num_threads = 256

        # Grid: one block per (batch * seq) position
        grid = [mQ.shape[0], 1, 1]
        block = [num_threads, 1, 1]

        # Shared memory for cos/sin (half_head_dim elements each)
        smem_size = 2 * self.half_head_dim * (self.dtype.width // 8)

        self.kernel(mQ, mCos, mSin, seq_len).launch(grid=grid, block=block, smem=smem_size, stream=stream)

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mCos: cute.Tensor,
        mSin: cute.Tensor,
        seq_len: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        num_threads, _, _ = cute.arch.block_dim()

        m = bidx
        s = m % seq_len

        half_D = const_expr(self.half_head_dim)
        n_heads = mQ.shape[1]

        # Allocate shared memory for cos and sin
        smem = cutlass.utils.SmemAllocator()
        cos_smem = smem.allocate_tensor(self.dtype, cute.make_layout(half_D))
        sin_smem = smem.allocate_tensor(self.dtype, cute.make_layout(half_D))

        # Cooperatively load cos/sin into shared memory
        for i in range(tidx, half_D, num_threads):
            cos_smem[i] = mCos[s, i]
            sin_smem[i] = mSin[s, i]

        cute.arch.sync_threads()

        # Process all heads - each thread handles multiple (i, h) pairs
        # Total work = n_heads * half_D
        # Thread i handles indices: i, i + num_threads, i + 2*num_threads, ...

        total_work = n_heads * half_D

        for work_idx in range(tidx, total_work, num_threads):
            h = work_idx // half_D
            i = work_idx % half_D

            # Load cos/sin from shared memory
            cos_val = cos_smem[i].to(Float32)
            sin_val = sin_smem[i].to(Float32)

            if const_expr(self.backward):
                sin_val = -sin_val

            # Load q values
            q0 = mQ[m, h, i].to(Float32)
            q1 = mQ[m, h, i + half_D].to(Float32)

            # Apply RoPE rotation
            r0 = q0 * cos_val - q1 * sin_val
            r1 = q1 * cos_val + q0 * sin_val

            # Store results in-place
            mQ[m, h, i] = r0.to(mQ.element_type)
            mQ[m, h, i + half_D] = r1.to(mQ.element_type)


class RopeSM80:
    def __init__(self, dtype: torch.dtype, head_dim: int):
        self.torch_dtype = dtype
        self.cute_dtype = torch2cute_dtype_map[dtype]
        self.head_dim = head_dim
        self.compile_cache = {}

    def __call__(self, Q: Tensor, Cos: Tensor, Sin: Tensor, backward: bool = False):
        """Apply Rotary Positional Embedding to Q.

        Args:
            Q: Input tensor of shape (B, S, H, D)
            Cos: Cosine embeddings of shape (S, D) - uses first D//2
            Sin: Sine embeddings of shape (S, D) - uses first D//2
            backward: If True, apply inverse rotation (for gradient computation)
        """
        B, S, H, D = Q.shape
        assert D == self.head_dim, f"Input head_dim {D} must match Rope head_dim {self.head_dim}"

        # Reshape for kernel: (Batch * Seq, Heads, HeadDim)
        Q_flat = Q.view(B * S, H, D)

        # Canonicalize Cos/Sin shapes - kernel will only access first D//2 elements
        Cos = Cos.view(-1, D)
        Sin = Sin.view(-1, D)
        assert Cos.shape[0] >= S, "Cos must have at least S rows"

        compile_key = (self.torch_dtype, H, D, backward)
        if compile_key not in self.compile_cache:
            m_sym = cute.sym_int()
            s_sym = cute.sym_int()

            # Create fake tensors for compilation
            mQ_fake = fake_tensor(self.cute_dtype, (m_sym, H, D))
            mCos_fake = fake_tensor(self.cute_dtype, (s_sym, D))
            mSin_fake = fake_tensor(self.cute_dtype, (s_sym, D))

            impl = RopeImpl(self.cute_dtype, self.head_dim, bool(backward))

            self.compile_cache[compile_key] = cute.compile(
                impl,
                mQ_fake,
                mCos_fake,
                mSin_fake,
                Int32(0),  # seq_len (placeholder)
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )

        self.compile_cache[compile_key](Q_flat, Cos, Sin, S)
        return Q
