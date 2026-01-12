# Copyright (c) 2025, Machete Authors

import torch
from torch import Tensor
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

from quack.cute_dsl_utils import torch2cute_dtype_map
import quack.activation as qact
from machete.megakernel.interface import machete_op, FusableKernel
from machete.megakernel.single import SingleKernel


class GatedLinearSM80(SingleKernel, FusableKernel):
    def __init__(self, dtype: torch.dtype, act_type: str):
        self.torch_dtype = dtype
        self.cute_dtype = torch2cute_dtype_map[dtype]
        self.act_type = act_type
        # Inherit from SingleKernel, passing 'self' as the op provider.
        SingleKernel.__init__(self, self, self.grid_fn, self.block_fn)

    @property
    def smem_per_page(self) -> int:
        return 0

    @property
    def num_pages(self) -> int:
        return 1

    @machete_op(num_tensors=3)
    @cute.jit
    def compute_forward(self, m_a: cute.Tensor, m_b: cute.Tensor, m_c: cute.Tensor, n_cols: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        num_threads, _, _ = cute.arch.block_dim()

        row = bidx
        for col in range(tidx, n_cols, num_threads):
            a_val = m_a[row, col].to(Float32)
            b_val = m_b[row, col].to(Float32)

            if const_expr(self.act_type == "gelu"):
                gate = qact.geglu(a_val, Float32(1.0))  # geglu(a, 1) = gelu(a)
            elif const_expr(self.act_type == "silu"):
                gate = qact.silu(a_val)
            elif const_expr(self.act_type == "relu"):
                gate = qact.relu(a_val)
            else:
                gate = a_val  # Fallback

            m_c[row, col] = (gate * b_val).to(m_c.element_type)

    @machete_op(num_tensors=5)
    @cute.jit
    def compute_backward(
        self,
        m_dc: cute.Tensor,
        m_a: cute.Tensor,
        m_b: cute.Tensor,
        m_da: cute.Tensor,
        m_db: cute.Tensor,
        n_cols: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        num_threads, _, _ = cute.arch.block_dim()

        row = bidx
        for col in range(tidx, n_cols, num_threads):
            dc_val = m_dc[row, col].to(Float32)
            a_val = m_a[row, col].to(Float32)
            b_val = m_b[row, col].to(Float32)

            if const_expr(self.act_type == "gelu"):
                da_val, db_val, _ = qact.dgeglu(a_val, b_val, dc_val)
            elif const_expr(self.act_type == "silu"):
                da_val, db_val, _ = qact.dswiglu(a_val, b_val, dc_val)
            elif const_expr(self.act_type == "relu"):
                da_val, db_val, _ = qact.dreglu(a_val, b_val, dc_val)
            else:
                da_val = dc_val * b_val
                db_val = dc_val * a_val

            m_da[row, col] = Float32(da_val).to(m_da.element_type)
            m_db[row, col] = Float32(db_val).to(m_db.element_type)

    def grid_fn(self, *args):
        # args[0] is the first tensor (a or dc), shape [batch*seq_len, n_cols] (flattened info)
        return [args[0].shape[0], 1, 1]

    def block_fn(self, *args):
        return [128, 1, 1]

    def run_forward(self, ctx, a, b, n_cols):
        # Save for backward
        ctx.save_for_backward(a, b)
        ctx.n_cols = n_cols

        # Allocate output
        c = torch.empty_like(a)

        # Args for kernel: a, b, c, n_cols
        args = (a, b, c, n_cols)

        grid = self.grid_fn(*args)
        block = self.block_fn(*args)
        n_blocks = grid[0] * grid[1] * grid[2]

        barrier = torch.zeros(1, dtype=torch.int32, device=a.device)

        self._update_or_add(self.mk_fwd, args)
        self.mk_fwd.launch(barrier, n_blocks, grid, block)

        return c

    def run_backward(self, ctx, dc):
        a, b = ctx.saved_tensors
        n_cols = ctx.n_cols

        da = torch.empty_like(a)
        db = torch.empty_like(b)

        # Args for kernel: dc, a, b, da, db, n_cols
        # Kernel signature: compute_backward(m_dc, m_a, m_b, m_da, m_db, n_cols)
        args = (dc, a, b, da, db, n_cols)

        grid = self.grid_fn(*args)
        block = self.block_fn(*args)
        n_blocks = grid[0] * grid[1] * grid[2]

        barrier = torch.zeros(1, dtype=torch.int32, device=dc.device)

        self._update_or_add(self.mk_bwd, args)
        self.mk_bwd.launch(barrier, n_blocks, grid, block)

        return da, db, None

    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        ori_shape = a.shape
        n_cols = ori_shape[-1]
        a_flat = a.view(-1, n_cols)
        b_flat = b.view(-1, n_cols)

        # Use SingleKernel.apply (which uses MegakernelAutograd.apply with `self` as runner)
        c_flat = self.apply(a_flat, b_flat, n_cols)

        return c_flat.view(*ori_shape)
