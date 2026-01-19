# Copyright (c) 2025, Machete Authors
"""
Gated Linear kernel with L/C/S decomposition for No Bubbles pipelining.

For element-wise operations like GatedLinear:
- load(): No-op (element-wise, nothing to preload)
- compute(): Do the gated activation
- store(): No-op (writes directly to global)
"""

import torch
from torch import Tensor
from typing import List
import cutlass.cute as cute
from cutlass import Float32, const_expr

from quack.cute_dsl_utils import torch2cute_dtype_map
import quack.activation as qact
from machete.megakernel.interface import FusableKernel, TensorParam
from machete.megakernel.single import SingleKernel


class GatedLinearSM80(SingleKernel, FusableKernel):
    """
    Gated Linear kernel with activation (SiLU, GELU, ReLU).
    Uses shared memory tiling to maximize global memory bandwidth.
    """

    TILE_N = 2048  # Tile size for columns (multiple of 256*8=2048)

    def __init__(self, dtype: torch.dtype, act_type: str, n_rows: int, n_cols: int):
        FusableKernel.__init__(self)
        self.torch_dtype = dtype
        self.cute_dtype = torch2cute_dtype_map[dtype]
        self.act_type = act_type
        self.n_rows = n_rows
        self.n_cols = n_cols
        SingleKernel.__init__(self, self, self.grid_fn, self.block_fn)

        # Determine vector size based on dtype
        # 128 bit / 16 bit = 8
        # 128 bit / 32 bit = 4
        self.vec_size = 8 if dtype == torch.float16 else 4

    @property
    def tensor_params_fwd(self) -> List[TensorParam]:
        """Tensor parameters for forward pass: a, b, c.

        Uses index-based shape/stride: (0, 1) means tensor.shape[0], tensor.shape[1]
        """
        return [
            TensorParam("a", shape=(0, 1), stride=(0, 1)),
            TensorParam("b", shape=(0, 1), stride=(0, 1)),
            TensorParam("c", shape=(0, 1), stride=(0, 1)),
        ]

    @property
    def tensor_params_bwd(self) -> List[TensorParam]:
        """Tensor parameters for backward pass: dc, a, b, da, db."""
        return [
            TensorParam("dc", shape=(0, 1), stride=(0, 1)),
            TensorParam("a", shape=(0, 1), stride=(0, 1)),
            TensorParam("b", shape=(0, 1), stride=(0, 1)),
            TensorParam("da", shape=(0, 1), stride=(0, 1)),
            TensorParam("db", shape=(0, 1), stride=(0, 1)),
        ]

    @property
    def smem_size_fwd(self) -> int:
        """Shared memory for forward: A + B tiles (2 tiles)."""
        element_size = 2 if self.torch_dtype in [torch.float16, torch.bfloat16] else 4
        return self.TILE_N * 2 * element_size

    @property
    def smem_size_bwd(self) -> int:
        """Shared memory for backward: dc + a + b tiles (3 tiles)."""
        element_size = 2 if self.torch_dtype in [torch.float16, torch.bfloat16] else 4
        return self.TILE_N * 3 * element_size

    # ========== Forward Pass L/C/S ==========

    @cute.jit
    def load_forward(self, logical_idx, smem, m_a, m_b, m_c):
        """Load A and B tiles global memory to shared memory."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx_x, bidx_y, _ = cute.arch.block_idx()
        num_threads, _, _ = cute.arch.block_dim()

        n_cols = const_expr(self.n_cols)
        TILE_N = const_expr(self.TILE_N)

        row_idx = bidx_x
        col_start = bidx_y * TILE_N

        unroll_factor = 4
        stride = num_threads * unroll_factor

        for k in range(tidx * unroll_factor, TILE_N, stride):
            # Manual Unroll
            base = col_start + k
            if base + 3 < n_cols:
                smem[k] = m_a[row_idx, base]
                smem[k + 1] = m_a[row_idx, base + 1]
                smem[k + 2] = m_a[row_idx, base + 2]
                smem[k + 3] = m_a[row_idx, base + 3]

                smem[TILE_N + k] = m_b[row_idx, base]
                smem[TILE_N + k + 1] = m_b[row_idx, base + 1]
                smem[TILE_N + k + 2] = m_b[row_idx, base + 2]
                smem[TILE_N + k + 3] = m_b[row_idx, base + 3]
            else:
                for u in range(unroll_factor):
                    if base + u < n_cols:
                        smem[k + u] = m_a[row_idx, base + u]
                        smem[TILE_N + k + u] = m_b[row_idx, base + u]

    @cute.jit
    def compute_forward(self, logical_idx, smem, m_a, m_b, m_c):
        """Compute."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx_x, bidx_y, _ = cute.arch.block_idx()
        num_threads, _, _ = cute.arch.block_dim()

        n_cols = const_expr(self.n_cols)
        TILE_N = const_expr(self.TILE_N)

        row_idx = bidx_x
        col_start = bidx_y * TILE_N
        b_offset = TILE_N

        unroll_factor = 4
        stride = num_threads * unroll_factor

        for k in range(tidx * unroll_factor, TILE_N, stride):
            base = col_start + k

            if base + 3 < n_cols:
                # Unroll 4
                for u in range(unroll_factor):
                    idx = k + u
                    a_val = smem[idx].to(Float32)
                    b_val = smem[b_offset + idx].to(Float32)

                    if const_expr(self.act_type == "gelu"):
                        gate = qact.geglu(a_val, Float32(1.0))
                    elif const_expr(self.act_type == "silu"):
                        gate = qact.silu(a_val)
                    elif const_expr(self.act_type == "relu"):
                        gate = qact.relu(a_val)
                    else:
                        gate = a_val

                    res = gate * b_val
                    m_c[row_idx, base + u] = res.to(self.cute_dtype)
            else:
                for u in range(unroll_factor):
                    if base + u < n_cols:
                        idx = k + u
                        a_val = smem[idx].to(Float32)
                        b_val = smem[b_offset + idx].to(Float32)

                        if const_expr(self.act_type == "gelu"):
                            gate = qact.geglu(a_val, Float32(1.0))
                        elif const_expr(self.act_type == "silu"):
                            gate = qact.silu(a_val)
                        elif const_expr(self.act_type == "relu"):
                            gate = qact.relu(a_val)
                        else:
                            gate = a_val

                        m_c[row_idx, base + u] = (gate * b_val).to(self.cute_dtype)

    @cute.jit
    def store_forward(self, logical_idx, smem, m_a, m_b, m_c):
        pass

    # ========== Backward Pass L/C/S ==========
    # Skip vectorization for backward for now as we focus on Forward bench.
    # But keep method signature consistent.

    @cute.jit
    def load_backward(self, logical_idx, smem, m_dc, m_a, m_b, m_da, m_db):
        # Load dc, a, b into smem
        tidx, _, _ = cute.arch.thread_idx()
        bidx_x, bidx_y, _ = cute.arch.block_idx()
        num_threads, _, _ = cute.arch.block_dim()

        n_cols = const_expr(self.n_cols)
        TILE_N = const_expr(self.TILE_N)

        row_idx = bidx_x
        col_start = bidx_y * TILE_N

        off_a = TILE_N
        off_b = 2 * TILE_N

        unroll_factor = 4
        stride = num_threads * unroll_factor

        for k in range(tidx * unroll_factor, TILE_N, stride):
            base = col_start + k
            if base + 3 < n_cols:
                smem[k] = m_dc[row_idx, base]
                smem[k + 1] = m_dc[row_idx, base + 1]
                smem[k + 2] = m_dc[row_idx, base + 2]
                smem[k + 3] = m_dc[row_idx, base + 3]

                smem[off_a + k] = m_a[row_idx, base]
                smem[off_a + k + 1] = m_a[row_idx, base + 1]
                smem[off_a + k + 2] = m_a[row_idx, base + 2]
                smem[off_a + k + 3] = m_a[row_idx, base + 3]

                smem[off_b + k] = m_b[row_idx, base]
                smem[off_b + k + 1] = m_b[row_idx, base + 1]
                smem[off_b + k + 2] = m_b[row_idx, base + 2]
                smem[off_b + k + 3] = m_b[row_idx, base + 3]
            else:
                for u in range(unroll_factor):
                    if base + u < n_cols:
                        smem[k + u] = m_dc[row_idx, base + u]
                        smem[off_a + k + u] = m_a[row_idx, base + u]
                        smem[off_b + k + u] = m_b[row_idx, base + u]

    @cute.jit
    def compute_backward(self, logical_idx, smem, m_dc, m_a, m_b, m_da, m_db):
        tidx, _, _ = cute.arch.thread_idx()
        bidx_x, bidx_y, _ = cute.arch.block_idx()
        num_threads, _, _ = cute.arch.block_dim()

        n_cols = const_expr(self.n_cols)
        TILE_N = const_expr(self.TILE_N)

        row_idx = bidx_x
        col_start = bidx_y * TILE_N
        off_a = TILE_N
        off_b = 2 * TILE_N

        unroll_factor = 4
        stride = num_threads * unroll_factor

        for k in range(tidx * unroll_factor, TILE_N, stride):
            base = col_start + k

            if base + 3 < n_cols:
                for u in range(unroll_factor):
                    idx = k + u
                    dc_val = smem[idx].to(Float32)
                    a_val = smem[off_a + idx].to(Float32)
                    b_val = smem[off_b + idx].to(Float32)

                    if const_expr(self.act_type == "gelu"):
                        da_val, db_val, _ = qact.dgeglu(a_val, b_val, dc_val)
                    elif const_expr(self.act_type == "silu"):
                        da_val, db_val, _ = qact.dswiglu(a_val, b_val, dc_val)
                    elif const_expr(self.act_type == "relu"):
                        da_val, db_val, _ = qact.dreglu(a_val, b_val, dc_val)
                    else:
                        da_val = dc_val * b_val
                        db_val = dc_val * a_val

                    m_da[row_idx, base + u] = Float32(da_val).to(self.cute_dtype)
                    m_db[row_idx, base + u] = Float32(db_val).to(self.cute_dtype)
            else:
                for u in range(unroll_factor):
                    if base + u < n_cols:
                        idx = k + u
                        dc_val = smem[idx].to(Float32)
                        a_val = smem[off_a + idx].to(Float32)
                        b_val = smem[off_b + idx].to(Float32)

                        if const_expr(self.act_type == "gelu"):
                            da_val, db_val, _ = qact.dgeglu(a_val, b_val, dc_val)
                        elif const_expr(self.act_type == "silu"):
                            da_val, db_val, _ = qact.dswiglu(a_val, b_val, dc_val)
                        elif const_expr(self.act_type == "relu"):
                            da_val, db_val, _ = qact.dreglu(a_val, b_val, dc_val)
                        else:
                            da_val = dc_val * b_val
                            db_val = dc_val * a_val

                        m_da[row_idx, base + u] = Float32(da_val).to(self.cute_dtype)
                        m_db[row_idx, base + u] = Float32(db_val).to(self.cute_dtype)

    @cute.jit
    def store_backward(self, logical_idx, smem, m_dc, m_a, m_b, m_da, m_db):
        pass

    # ========== Launch Helpers ==========

    def grid_fn(self, *args):
        # args[0] is 'a' tensor
        m_rows = args[0].shape[0]
        n_cols = args[0].shape[-1]
        n_tiles = (n_cols + self.TILE_N - 1) // self.TILE_N
        return [m_rows, n_tiles, 1]

    def block_fn(self, *args):
        return [256, 1, 1]

    def run_forward(self, ctx, a, b):
        ctx.save_for_backward(a, b)
        c = torch.empty_like(a)
        args = (a, b, c)
        grid = self.grid_fn(*args)
        block = self.block_fn(*args)
        n_blocks = grid[0] * grid[1] * grid[2]
        self._update_or_add(self.mk_fwd, args)
        self.mk_fwd.launch(n_blocks, grid, block)
        return c

    def run_backward(self, ctx, dc):
        a, b = ctx.saved_tensors
        da = torch.empty_like(a)
        db = torch.empty_like(b)
        args = (dc, a, b, da, db)
        grid = self.grid_fn(*args)
        block = self.block_fn(*args)
        n_blocks = grid[0] * grid[1] * grid[2]
        self._update_or_add(self.mk_bwd, args)
        self.mk_bwd.launch(n_blocks, grid, block)
        return da, db

    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return self.apply_autograd(a, b)
