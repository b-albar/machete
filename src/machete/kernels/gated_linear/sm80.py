# Copyright (c) 2025, Machete Authors
from typing import Dict, Tuple, Any

import torch
from torch import Tensor
import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

from quack.cute_dsl_utils import torch2cute_dtype_map
from quack.compile_utils import make_fake_tensor as fake_tensor
import quack.activation as qact
from machete.megakernel.interface import machete_op, FusableKernel


class GatedLinearSM80(FusableKernel):
    def __init__(self, dtype: torch.dtype, act_type: str):
        self.torch_dtype = dtype
        self.cute_dtype = torch2cute_dtype_map[dtype]
        self.act_type = act_type
        self._compile_cache: Dict[str, Any] = {}

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

    def forward(self, a, b):
        ori_shape = a.shape
        n_cols = ori_shape[-1]
        a_flat = a.view(-1, n_cols)
        b_flat = b.view(-1, n_cols)
        c_flat = torch.empty_like(a_flat)

        key = f"forward_{n_cols}"
        if key not in self._compile_cache:
            m_sym = cute.sym_int()
            self._compile_cache[key] = cute.compile(
                self.forward_kernel,
                fake_tensor(self.cute_dtype, (m_sym, n_cols)),
                fake_tensor(self.cute_dtype, (m_sym, n_cols)),
                fake_tensor(self.cute_dtype, (m_sym, n_cols)),
                Int32(n_cols),
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )

        self._compile_cache[key](a_flat, b_flat, c_flat, n_cols)
        return c_flat.view(*ori_shape)

    def backward(self, dc, a, b):
        n_cols = a.shape[-1]
        dc_flat, a_flat, b_flat = dc.view(-1, n_cols), a.view(-1, n_cols), b.view(-1, n_cols)
        da_flat, db_flat = torch.empty_like(a_flat), torch.empty_like(b_flat)

        key = f"backward_{n_cols}"
        if key not in self._compile_cache:
            m_sym = cute.sym_int()
            self._compile_cache[key] = cute.compile(
                self.backward_kernel,
                fake_tensor(self.cute_dtype, (m_sym, n_cols)),
                fake_tensor(self.cute_dtype, (m_sym, n_cols)),
                fake_tensor(self.cute_dtype, (m_sym, n_cols)),
                fake_tensor(self.cute_dtype, (m_sym, n_cols)),
                fake_tensor(self.cute_dtype, (m_sym, n_cols)),
                Int32(n_cols),
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )

        self._compile_cache[key](dc_flat, a_flat, b_flat, da_flat, db_flat, n_cols)
        return da_flat.view_as(a), db_flat.view_as(b)

    @cute.jit
    def forward_kernel(
        self, m_a: cute.Tensor, m_b: cute.Tensor, m_c: cute.Tensor, n_cols: Int32, stream: cuda.CUstream
    ):
        grid = [m_a.shape[0], 1, 1]
        block = [128, 1, 1]
        self._fwd_entry(m_a, m_b, m_c, n_cols).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def _fwd_entry(self, m_a: cute.Tensor, m_b: cute.Tensor, m_c: cute.Tensor, n_cols: Int32):
        self.compute_forward(m_a, m_b, m_c, n_cols)

    @cute.jit
    def backward_kernel(
        self,
        m_dc: cute.Tensor,
        m_a: cute.Tensor,
        m_b: cute.Tensor,
        m_da: cute.Tensor,
        m_db: cute.Tensor,
        n_cols: Int32,
        stream: cuda.CUstream,
    ):
        grid = [m_a.shape[0], 1, 1]
        block = [128, 1, 1]
        self._bwd_entry(m_dc, m_a, m_b, m_da, m_db, n_cols).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def _bwd_entry(
        self, m_dc: cute.Tensor, m_a: cute.Tensor, m_b: cute.Tensor, m_da: cute.Tensor, m_db: cute.Tensor, n_cols: Int32
    ):
        self.compute_backward(m_dc, m_a, m_b, m_da, m_db, n_cols)


class GatedLinear(torch.autograd.Function):
    _instances: Dict[Tuple[torch.dtype, str], GatedLinearSM80] = {}

    @staticmethod
    def _get_instance(dtype, act_type):
        key = (dtype, act_type)
        if key not in GatedLinear._instances:
            GatedLinear._instances[key] = GatedLinearSM80(dtype, act_type)
        return GatedLinear._instances[key]

    @staticmethod
    def forward(ctx, a, b, act_type="gelu"):
        ctx.save_for_backward(a, b)
        ctx.act_type = act_type
        ctx.dtype = a.dtype
        instance = GatedLinear._get_instance(a.dtype, act_type)
        return instance.forward(a, b)

    @staticmethod
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        instance = GatedLinear._get_instance(ctx.dtype, ctx.act_type)
        da, db = instance.backward(dc, a, b)
        return da, db, None


def geglu_func(a: Tensor, b: Tensor) -> Tensor:
    return GatedLinear.apply(a, b, "gelu")


def swiglu_func(a: Tensor, b: Tensor) -> Tensor:
    return GatedLinear.apply(a, b, "silu")


def reglu_func(a: Tensor, b: Tensor) -> Tensor:
    return GatedLinear.apply(a, b, "relu")
