# Copyright (c) 2025, Machete Authors
"""
RoPE kernel using the MacheteKernel interface.

This kernel demonstrates the MacheteKernel interface where:
- Tensors are declared via declare_tensors() with symbolic dimensions
- L/C/S methods receive CuTe tensors directly (not pointers)
- setup_kernel() runs per logical_idx for shared state and smem allocation
- No @cute.jit decorators - code is extracted and inlined
"""

import torch
from torch import Tensor
from typing import Dict, Tuple
import cutlass.cute as cute
from cutlass import Float32, const_expr

from quack.cute_dsl_utils import torch2cute_dtype_map
from machete.kernels.utils.smem import MacheteSmemAllocator
from machete.megakernel.interface import MacheteKernel, TensorSpec
from machete.megakernel.single import SingleKernel


class RopeSM80(SingleKernel, MacheteKernel):
    """
    RoPE kernel using the MacheteKernel interface.

    Key features:
    1. declare_tensors() specifies tensor shapes with symbolic dimensions
    2. setup_kernel() allocates shared memory and sets up per-block state
    3. L/C/S methods receive CuTe tensors, not pointers
    4. No @cute.jit decorators on L/C/S methods - code is inlined

    Shared memory layout:
    - cos_cache[half_d]: Cached cos values for current sequence position
    - sin_cache[half_d]: Cached sin values for current sequence position
    """

    NUM_THREADS = 256

    def __init__(self, dtype: torch.dtype, head_dim: int, n_heads: int):
        self.torch_dtype = dtype
        self.cute_dtype = torch2cute_dtype_map[dtype]
        self.head_dim = head_dim
        self.half_head_dim = head_dim // 2
        self.n_heads = n_heads
        SingleKernel.__init__(self, self, self.grid_fn, self.block_fn)

    # ========== Tensor Declarations ==========

    def declare_tensors(self) -> Dict[str, TensorSpec]:
        """Declare all tensors with symbolic dimensions.

        Dimension names (n_tokens, n_heads, etc.) resolve from scalar args.
        """
        return {
            "q": TensorSpec(
                name="q",
                dtype=self.cute_dtype,
                shape_expr=("n_tokens", "n_heads", "head_dim"),
                stride_expr=("n_heads * head_dim", "head_dim", "1"),
                is_input=True,
                is_output=True,  # In-place operation
            ),
            "cos": TensorSpec(
                name="cos",
                dtype=self.cute_dtype,
                shape_expr=("seq_len", "half_d"),
                stride_expr=("half_d", "1"),
                is_input=True,
            ),
            "sin": TensorSpec(
                name="sin",
                dtype=self.cute_dtype,
                shape_expr=("seq_len", "half_d"),
                stride_expr=("half_d", "1"),
                is_input=True,
            ),
        }

    def declare_scalars(self) -> Tuple[str, ...]:
        """Declare scalar parameters."""
        return ("seq_len", "n_tokens", "n_heads", "head_dim", "half_d")

    # ========== Shared Memory Size ==========

    @property
    def smem_size_fwd(self) -> int:
        """Shared memory for cos/sin values: 2 * half_head_dim * element_size."""
        return self.half_head_dim * 2 * (self.cute_dtype.width // 8)

    @property
    def smem_size_bwd(self) -> int:
        """Shared memory for backward (same as forward for RoPE)."""
        return self.smem_size_fwd

    # ========== Logical Blocks API ==========

    def get_logical_grid_size(self, q, cos, sin, seq_len, n_tokens) -> int:
        """One logical block per token."""
        return n_tokens

    def get_logical_coord_names(self) -> Tuple[str, ...]:
        """Coordinate name for debugging."""
        return ("token_idx",)

    # ========== Per-Block Setup ==========

    def setup_kernel(self, logical_idx, smem, q, cos, sin, seq_len, n_tokens):
        """Per-logical_idx setup - allocate smem and compute sequence position.

        This runs ONCE per logical_idx before L/C/S phases.
        Sets up shared state accessible by all three phases.
        """
        half_d = const_expr(self.half_head_dim)

        # Compute sequence position from logical index
        self.m = logical_idx  # Token index for q tensor
        self.s = logical_idx % seq_len  # Sequence position for cos/sin

        # Allocate shared memory regions
        alloc = MacheteSmemAllocator(smem)
        self.s_cos = alloc.allocate_array(self.cute_dtype, half_d, byte_alignment=1)
        self.s_sin = alloc.allocate_array(self.cute_dtype, half_d, byte_alignment=1)

    # ========== Forward Pass L/C/S ==========

    def load_forward(self, logical_idx, smem, q, cos, sin, seq_len, n_tokens):
        """Load cos/sin into shared memory for current sequence position."""
        tidx, _, _ = cute.arch.thread_idx()
        num_threads = const_expr(self.NUM_THREADS)
        half_d = const_expr(self.half_head_dim)

        # Get sequence position and smem from setup_kernel
        s = self.s
        s_cos = self.s_cos
        s_sin = self.s_sin

        # Coalesced load of cos/sin for this sequence position
        for i in range(tidx, half_d, num_threads):
            s_cos[i] = cos[s, i]
            s_sin[i] = sin[s, i]

    def compute_forward(self, logical_idx, smem, q, cos, sin, seq_len, n_tokens):
        """Apply RoPE rotation using cos/sin from shared memory."""
        tidx, _, _ = cute.arch.thread_idx()
        num_threads = const_expr(self.NUM_THREADS)
        half_d = const_expr(self.half_head_dim)
        n_heads = const_expr(self.n_heads)

        # Get state from setup_kernel
        m = self.m
        s_cos = self.s_cos
        s_sin = self.s_sin

        # Compute RoPE rotation in-place
        total_work = n_heads * half_d
        for work_idx in range(tidx, total_work, num_threads):
            h_idx = work_idx // half_d
            i_idx = work_idx % half_d

            # Read cos/sin from smem (fast)
            cos_val = s_cos[i_idx].to(Float32)
            sin_val = s_sin[i_idx].to(Float32)

            # Read q values from global
            q0 = q[m, h_idx, i_idx].to(Float32)
            q1 = q[m, h_idx, i_idx + half_d].to(Float32)

            # RoPE rotation
            r0 = q0 * cos_val - q1 * sin_val
            r1 = q1 * cos_val + q0 * sin_val

            # Write back to global (in-place)
            q[m, h_idx, i_idx] = r0.to(self.cute_dtype)
            q[m, h_idx, i_idx + half_d] = r1.to(self.cute_dtype)

    def store_forward(self, logical_idx, smem, q, cos, sin, seq_len, n_tokens):
        """No-op: RoPE writes directly to global in compute phase."""
        pass

    # ========== Backward Pass L/C/S ==========

    def load_backward(self, logical_idx, smem, q, cos, sin, seq_len, n_tokens):
        """Load cos/sin for backward pass (same as forward)."""
        tidx, _, _ = cute.arch.thread_idx()
        num_threads = const_expr(self.NUM_THREADS)
        half_d = const_expr(self.half_head_dim)

        s = self.s
        s_cos = self.s_cos
        s_sin = self.s_sin

        for i in range(tidx, half_d, num_threads):
            s_cos[i] = cos[s, i]
            s_sin[i] = sin[s, i]

    def compute_backward(self, logical_idx, smem, q, cos, sin, seq_len, n_tokens):
        """Backward RoPE: inverse rotation (negated sin)."""
        tidx, _, _ = cute.arch.thread_idx()
        num_threads = const_expr(self.NUM_THREADS)
        half_d = const_expr(self.half_head_dim)
        n_heads = const_expr(self.n_heads)

        m = self.m
        s_cos = self.s_cos
        s_sin = self.s_sin

        total_work = n_heads * half_d
        for work_idx in range(tidx, total_work, num_threads):
            h_idx = work_idx // half_d
            i_idx = work_idx % half_d

            cos_val = s_cos[i_idx].to(Float32)
            sin_val = -s_sin[i_idx].to(Float32)  # Negated for inverse

            q0 = q[m, h_idx, i_idx].to(Float32)
            q1 = q[m, h_idx, i_idx + half_d].to(Float32)

            r0 = q0 * cos_val - q1 * sin_val
            r1 = q1 * cos_val + q0 * sin_val

            q[m, h_idx, i_idx] = r0.to(self.cute_dtype)
            q[m, h_idx, i_idx + half_d] = r1.to(self.cute_dtype)

    def store_backward(self, logical_idx, smem, q, cos, sin, seq_len, n_tokens):
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
        cos_flat = cos.view(-1, half_d)
        sin_flat = sin.view(-1, half_d)
        n_tokens = b * s
        out_flat = self.apply_autograd(q_flat, cos_flat, sin_flat, s, n_tokens)
        return out_flat.view(b, s, h, d)
