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
    def __init__(self, dtype: Type[cutlass.Numeric], head_dim: int, backward: bool):
        self.dtype = dtype
        self.head_dim = head_dim
        self.half_head_dim = head_dim // 2
        self.backward = backward

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (M, H, D)
        mCos: cute.Tensor,  # (S, D)
        mSin: cute.Tensor,  # (S, D)
        seq_len: Int32,
        stream: cuda.CUstream,
    ):
        heads_per_cta = const_expr(4)
        n_heads = mQ.shape[1]

        # Grid: (M, ceil(H / heads_per_cta))
        grid = [mQ.shape[0], cute.ceil_div(n_heads, heads_per_cta), 1]
        # Use a reasonable number of threads to cover half_head_dim
        # If head_dim=128, half_head_dim=64. 64 or 128 threads is fine.
        block = [128, 1, 1]

        self.kernel(mQ, mCos, mSin, seq_len).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mCos: cute.Tensor,
        mSin: cute.Tensor,
        seq_len: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        num_threads, _, _ = cute.arch.block_dim()

        m = bidx
        s = m % seq_len

        half_D = const_expr(self.half_head_dim)
        heads_per_cta = const_expr(4)
        n_heads = mQ.shape[1]
        h_start = bidy * heads_per_cta

        # Pre-load cos and sin for this thread into fragments
        # Since they are shared across all heads in the CTA
        # We only need them once per thread.

        # Vectorized loading would be better, but let's start with this.
        # We use a loop for now, but in a real Cute kernel we'd use TiledCopy.

        for i in range(tidx, half_D, num_threads):
            # Load from the first half of Cos/Sin rows
            cos_val = mCos[s, i].to(Float32)
            sin_val = mSin[s, i].to(Float32)

            if const_expr(self.backward):
                sin_val = -sin_val

            # Unswapped RoPE mixing
            # q_rot[i]          = q[i] * cos[i] - q[i + half_D] * sin[i]
            # q_rot[i + half_D] = q[i + half_D] * cos[i] + q[i] * sin[i]

            for h_offset in range(heads_per_cta):
                h = h_start + h_offset
                if h < n_heads:
                    # Accessing memory in a coalesced way (mostly)
                    # For better performance, we'd want to load multiple heads at once.
                    q0 = mQ[m, h, i].to(Float32)
                    q1 = mQ[m, h, i + half_D].to(Float32)

                    r0 = q0 * cos_val - q1 * sin_val
                    r1 = q1 * cos_val + q0 * sin_val

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
            Cos: Cosine embeddings of shape (S, D)
            Sin: Sine embeddings of shape (S, D)
            backward: If True, apply inverse rotation (for gradient computation)
        """
        B, S, H, D = Q.shape
        assert D == self.head_dim, f"Input head_dim {D} must match Rope head_dim {self.head_dim}"

        # Reshape for kernel: (Batch * Seq, Heads, HeadDim)
        # We use view() to keep it zero-copy if possible.
        # Requires Q to be contiguous in (H, D) at least.
        Q_flat = Q.view(B * S, H, D)

        # Canonicalize Cos/Sin shapes (often they have extra dims)
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
