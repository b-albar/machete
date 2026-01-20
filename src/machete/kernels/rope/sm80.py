# Copyright (c) 2025, Machete Authors
"""
RoPE kernel with L/C/S (Load/Compute/Store) interface for megakernel fusion.

This kernel uses a 2D logical grid (n_tokens, n_heads) for better GPU occupancy,
matching the performance characteristics of optimized Triton implementations.

Grid: Each logical block handles one (token, head) pair.
Threads: Each thread handles elements within a single head dimension.
"""

import torch
from torch import Tensor
from typing import List
import cutlass.cute as cute
from cutlass import Float32, Int64, const_expr

from machete.megakernel.interface import FusableKernel
from machete.megakernel.single import SingleKernel

# Local dtype mapping (torch dtype -> CuTe dtype)
TORCH_TO_CUTE_DTYPE = {
    torch.float16: cute.Float16,
    torch.bfloat16: cute.BFloat16,
    torch.float32: cute.Float32,
    torch.float64: cute.Float64,
}


class RopeSM80(SingleKernel, FusableKernel):
    """
    RoPE kernel with L/C/S interface for megakernel fusion.

    Uses a 2D logical grid: (n_tokens, n_heads) for optimal GPU occupancy.
    Each block processes one (token, head) pair.

    Slot assignments:
    - t0: q_ptr (n_tokens, n_heads, head_dim)
    - t1: cos_ptr (seq_len, half_d)
    - t2: sin_ptr (seq_len, half_d)
    """

    # Use smaller block size since work per block is now smaller (just one head)
    NUM_THREADS = 128

    def __init__(self, dtype: torch.dtype, head_dim: int, n_heads: int = None):
        FusableKernel.__init__(self)
        self.torch_dtype = dtype
        self.cute_dtype = TORCH_TO_CUTE_DTYPE[dtype]
        self.head_dim = head_dim
        self.half_head_dim = head_dim // 2
        # n_heads will be determined at runtime from tensor shape
        self._n_heads = n_heads
        # shapes will be populated by _extract_shapes before kernel launch
        self.shapes = {}
        SingleKernel.__init__(self, self, self.grid_fn, self.block_fn)

    def _extract_shapes(self, args: List) -> dict:
        """Extract shape/stride info from tensor arguments for JIT compilation.

        Args are: q (3D), cos (2D), sin (2D), seq_len (int), n_tokens (int)
        """
        q, cos, sin, seq_len, n_tokens = args
        n_heads = q.shape[1]
        return {
            "q_shape": tuple(q.shape),      # (n_tokens, n_heads, head_dim)
            "q_stride": tuple(q.stride()),  # strides for q
            "cos_shape": tuple(cos.shape),  # (seq_len, half_d)
            "cos_stride": tuple(cos.stride()),
            "sin_shape": tuple(sin.shape),  # (seq_len, half_d)
            "sin_stride": tuple(sin.stride()),
            "seq_len": seq_len,
            "n_tokens": n_tokens,
            "n_heads": n_heads,
        }

    @property
    def smem_size_fwd(self) -> int:
        """No shared memory needed for this version."""
        return 0

    @property
    def smem_size_bwd(self) -> int:
        """No shared memory needed for this version."""
        return 0

    def get_logical_grid_size(self, *args) -> int:
        """Return total work units: n_tokens * n_heads.

        Args are: q, cos, sin, seq_len, n_tokens
        """
        q = args[0]
        n_tokens = args[4]
        n_heads = q.shape[1]
        return n_tokens * n_heads

    # ========== Forward Pass L/C/S ==========

    @cute.jit
    def load_forward(
        self, logical_idx, smem,
        t0: Int64, t1: Int64, t2: Int64, t3: Int64,
        t4: Int64, t5: Int64, t6: Int64, t7: Int64,
        t8: Int64, t9: Int64, t10: Int64, t11: Int64,
        t12: Int64, t13: Int64, t14: Int64, t15: Int64,
        t16: Int64, t17: Int64, t18: Int64, t19: Int64,
        t20: Int64, t21: Int64, t22: Int64, t23: Int64,
        t24: Int64, t25: Int64, t26: Int64, t27: Int64,
        t28: Int64, t29: Int64, t30: Int64, t31: Int64,
    ):
        """No-op: This version reads cos/sin directly from global memory."""
        pass

    @cute.jit
    def compute_forward(
        self, logical_idx, smem,
        t0: Int64, t1: Int64, t2: Int64, t3: Int64,
        t4: Int64, t5: Int64, t6: Int64, t7: Int64,
        t8: Int64, t9: Int64, t10: Int64, t11: Int64,
        t12: Int64, t13: Int64, t14: Int64, t15: Int64,
        t16: Int64, t17: Int64, t18: Int64, t19: Int64,
        t20: Int64, t21: Int64, t22: Int64, t23: Int64,
        t24: Int64, t25: Int64, t26: Int64, t27: Int64,
        t28: Int64, t29: Int64, t30: Int64, t31: Int64,
    ):
        """Apply RoPE rotation for one (token, head) pair."""
        tidx, _, _ = cute.arch.thread_idx()
        num_threads = const_expr(self.NUM_THREADS)
        half_d = const_expr(self.half_head_dim)
        n_heads = const_expr(self.shapes["n_heads"])
        seq_len = const_expr(self.shapes["seq_len"])

        # Decode logical_idx to (token, head)
        m = logical_idx // n_heads  # Token index
        h_idx = logical_idx % n_heads  # Head index

        # Get strides from shapes dict
        q_stride_0 = const_expr(self.shapes["q_stride"][0])  # stride for token dim
        q_stride_1 = const_expr(self.shapes["q_stride"][1])  # stride for head dim
        q_stride_2 = const_expr(self.shapes["q_stride"][2])  # stride for d dim (usually 1)
        cos_stride_0 = const_expr(self.shapes["cos_stride"][0])  # stride for seq dim
        cos_stride_1 = const_expr(self.shapes["cos_stride"][1])  # stride for d dim (usually 1)

        # Create 1D tensors using total element count
        n_q_elems = const_expr(self.shapes["q_shape"][0] * self.shapes["q_shape"][1] * self.shapes["q_shape"][2])
        n_cos_elems = const_expr(self.shapes["cos_shape"][0] * self.shapes["cos_shape"][1])

        q_ptr = cute.make_ptr(self.cute_dtype, t0, cute.AddressSpace.gmem)
        cos_ptr = cute.make_ptr(self.cute_dtype, t1, cute.AddressSpace.gmem)
        sin_ptr = cute.make_ptr(self.cute_dtype, t2, cute.AddressSpace.gmem)

        m_q = cute.make_tensor(q_ptr, cute.make_layout((n_q_elems,)))
        m_cos = cute.make_tensor(cos_ptr, cute.make_layout((n_cos_elems,)))
        m_sin = cute.make_tensor(sin_ptr, cute.make_layout((n_cos_elems,)))

        # Position in sequence for cos/sin lookup
        s = m % seq_len

        # Base offset for this (token, head) in q
        q_base = m * q_stride_0 + h_idx * q_stride_1
        # Base offset for this position in cos/sin
        cos_base = s * cos_stride_0

        # Each thread processes elements within the head dimension
        # Thread i processes element i, i+num_threads, i+2*num_threads, etc.
        for i_idx in range(tidx, half_d, num_threads):
            # Linear indices
            cos_idx = cos_base + i_idx * cos_stride_1
            q_idx_0 = q_base + i_idx * q_stride_2
            q_idx_1 = q_base + (i_idx + half_d) * q_stride_2

            # Read cos/sin from global memory
            cos_val = m_cos[cos_idx].to(Float32)
            sin_val = m_sin[cos_idx].to(Float32)

            # Read q values from global
            q0 = m_q[q_idx_0].to(Float32)
            q1 = m_q[q_idx_1].to(Float32)

            # RoPE rotation
            r0 = q0 * cos_val - q1 * sin_val
            r1 = q1 * cos_val + q0 * sin_val

            # Write back to global (in-place)
            m_q[q_idx_0] = r0.to(self.cute_dtype)
            m_q[q_idx_1] = r1.to(self.cute_dtype)

    @cute.jit
    def store_forward(
        self, logical_idx, smem,
        t0: Int64, t1: Int64, t2: Int64, t3: Int64,
        t4: Int64, t5: Int64, t6: Int64, t7: Int64,
        t8: Int64, t9: Int64, t10: Int64, t11: Int64,
        t12: Int64, t13: Int64, t14: Int64, t15: Int64,
        t16: Int64, t17: Int64, t18: Int64, t19: Int64,
        t20: Int64, t21: Int64, t22: Int64, t23: Int64,
        t24: Int64, t25: Int64, t26: Int64, t27: Int64,
        t28: Int64, t29: Int64, t30: Int64, t31: Int64,
    ):
        """No-op: RoPE writes directly to global in compute phase."""
        pass

    # ========== Backward Pass L/C/S ==========

    @cute.jit
    def load_backward(
        self, logical_idx, smem,
        t0: Int64, t1: Int64, t2: Int64, t3: Int64,
        t4: Int64, t5: Int64, t6: Int64, t7: Int64,
        t8: Int64, t9: Int64, t10: Int64, t11: Int64,
        t12: Int64, t13: Int64, t14: Int64, t15: Int64,
        t16: Int64, t17: Int64, t18: Int64, t19: Int64,
        t20: Int64, t21: Int64, t22: Int64, t23: Int64,
        t24: Int64, t25: Int64, t26: Int64, t27: Int64,
        t28: Int64, t29: Int64, t30: Int64, t31: Int64,
    ):
        """No-op: This version reads cos/sin directly from global memory."""
        pass

    @cute.jit
    def compute_backward(
        self, logical_idx, smem,
        t0: Int64, t1: Int64, t2: Int64, t3: Int64,
        t4: Int64, t5: Int64, t6: Int64, t7: Int64,
        t8: Int64, t9: Int64, t10: Int64, t11: Int64,
        t12: Int64, t13: Int64, t14: Int64, t15: Int64,
        t16: Int64, t17: Int64, t18: Int64, t19: Int64,
        t20: Int64, t21: Int64, t22: Int64, t23: Int64,
        t24: Int64, t25: Int64, t26: Int64, t27: Int64,
        t28: Int64, t29: Int64, t30: Int64, t31: Int64,
    ):
        """Backward RoPE: inverse rotation (negated sin)."""
        tidx, _, _ = cute.arch.thread_idx()
        num_threads = const_expr(self.NUM_THREADS)
        half_d = const_expr(self.half_head_dim)
        n_heads = const_expr(self.shapes["n_heads"])
        seq_len = const_expr(self.shapes["seq_len"])

        # Decode logical_idx to (token, head)
        m = logical_idx // n_heads  # Token index
        h_idx = logical_idx % n_heads  # Head index

        # Get strides from shapes dict
        q_stride_0 = const_expr(self.shapes["q_stride"][0])
        q_stride_1 = const_expr(self.shapes["q_stride"][1])
        q_stride_2 = const_expr(self.shapes["q_stride"][2])
        cos_stride_0 = const_expr(self.shapes["cos_stride"][0])
        cos_stride_1 = const_expr(self.shapes["cos_stride"][1])

        # Create 1D tensors
        n_q_elems = const_expr(self.shapes["q_shape"][0] * self.shapes["q_shape"][1] * self.shapes["q_shape"][2])
        n_cos_elems = const_expr(self.shapes["cos_shape"][0] * self.shapes["cos_shape"][1])

        q_ptr = cute.make_ptr(self.cute_dtype, t0, cute.AddressSpace.gmem)
        cos_ptr = cute.make_ptr(self.cute_dtype, t1, cute.AddressSpace.gmem)
        sin_ptr = cute.make_ptr(self.cute_dtype, t2, cute.AddressSpace.gmem)

        m_q = cute.make_tensor(q_ptr, cute.make_layout((n_q_elems,)))
        m_cos = cute.make_tensor(cos_ptr, cute.make_layout((n_cos_elems,)))
        m_sin = cute.make_tensor(sin_ptr, cute.make_layout((n_cos_elems,)))

        # Position in sequence for cos/sin lookup
        s = m % seq_len

        # Base offsets
        q_base = m * q_stride_0 + h_idx * q_stride_1
        cos_base = s * cos_stride_0

        for i_idx in range(tidx, half_d, num_threads):
            cos_idx = cos_base + i_idx * cos_stride_1
            q_idx_0 = q_base + i_idx * q_stride_2
            q_idx_1 = q_base + (i_idx + half_d) * q_stride_2

            cos_val = m_cos[cos_idx].to(Float32)
            sin_val = -m_sin[cos_idx].to(Float32)  # Negated for inverse

            q0 = m_q[q_idx_0].to(Float32)
            q1 = m_q[q_idx_1].to(Float32)

            r0 = q0 * cos_val - q1 * sin_val
            r1 = q1 * cos_val + q0 * sin_val

            m_q[q_idx_0] = r0.to(self.cute_dtype)
            m_q[q_idx_1] = r1.to(self.cute_dtype)

    @cute.jit
    def store_backward(
        self, logical_idx, smem,
        t0: Int64, t1: Int64, t2: Int64, t3: Int64,
        t4: Int64, t5: Int64, t6: Int64, t7: Int64,
        t8: Int64, t9: Int64, t10: Int64, t11: Int64,
        t12: Int64, t13: Int64, t14: Int64, t15: Int64,
        t16: Int64, t17: Int64, t18: Int64, t19: Int64,
        t20: Int64, t21: Int64, t22: Int64, t23: Int64,
        t24: Int64, t25: Int64, t26: Int64, t27: Int64,
        t28: Int64, t29: Int64, t30: Int64, t31: Int64,
    ):
        """No-op: backward RoPE writes directly to global."""
        pass

    # ========== Launch Helpers ==========

    def grid_fn(self, q, cos, sin, seq_len, n_tokens):
        n_heads = q.shape[1]
        return [n_tokens * n_heads, 1, 1]

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
