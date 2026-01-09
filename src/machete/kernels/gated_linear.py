# Copyright (c) 2025, Machete Authors
import math
from typing import Optional, Type, Dict, Tuple, Any

import torch
from torch import Tensor
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

from quack.cute_dsl_utils import torch2cute_dtype_map
from quack.compile_utils import make_fake_tensor as fake_tensor
import quack.activation as quack_act


class GatedLinearImpl:
    def __init__(self, dtype: Type[cutlass.Numeric], act_type: str):
        self.dtype = dtype
        self.act_type = act_type

    @cute.jit
    def forward(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        n_cols: Int32,
        stream: cuda.CUstream,
    ):
        grid = [mA.shape[0], 1, 1]
        block = [128, 1, 1]
        self.forward_kernel(mA, mB, mC, n_cols).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def forward_kernel(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, n_cols: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        num_threads, _, _ = cute.arch.block_dim()

        row = bidx
        for col in range(tidx, n_cols, num_threads):
            a_val = mA[row, col].to(Float32)
            b_val = mB[row, col].to(Float32)

            if const_expr(self.act_type == "gelu"):
                gate = quack_act.geglu(a_val, Float32(1.0))  # geglu(a, 1) = gelu(a)
            elif const_expr(self.act_type == "silu"):
                gate = quack_act.silu(a_val)
            elif const_expr(self.act_type == "relu"):
                gate = quack_act.relu(a_val)
            else:
                gate = a_val  # Fallback

            mC[row, col] = (gate * b_val).to(mC.element_type)

    @cute.jit
    def backward(
        self,
        mdC: cute.Tensor,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mdA: cute.Tensor,
        mdB: cute.Tensor,
        n_cols: Int32,
        stream: cuda.CUstream,
    ):
        grid = [mA.shape[0], 1, 1]
        block = [128, 1, 1]
        self.backward_kernel(mdC, mA, mB, mdA, mdB, n_cols).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def backward_kernel(
        self,
        mdC: cute.Tensor,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mdA: cute.Tensor,
        mdB: cute.Tensor,
        n_cols: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        num_threads, _, _ = cute.arch.block_dim()

        row = bidx
        for col in range(tidx, n_cols, num_threads):
            dc_val = mdC[row, col].to(Float32)
            a_val = mA[row, col].to(Float32)
            b_val = mB[row, col].to(Float32)

            if const_expr(self.act_type == "gelu"):
                da_val, db_val, _ = quack_act.dgeglu(a_val, b_val, dc_val)
            elif const_expr(self.act_type == "silu"):
                da_val, db_val, _ = quack_act.dswiglu(a_val, b_val, dc_val)
            elif const_expr(self.act_type == "relu"):
                da_val, db_val, _ = quack_act.dreglu(a_val, b_val, dc_val)
            else:
                da_val = dc_val * b_val
                db_val = dc_val * a_val

            mdA[row, col] = da_val.to(mdA.element_type)
            mdB[row, col] = db_val.to(mdB.element_type)


class GatedLinear(torch.autograd.Function):
    _compile_cache: Dict[Tuple, Any] = {}

    @staticmethod
    def forward(ctx, a, b, act_type="gelu"):
        ori_shape = a.shape
        n_cols = ori_shape[-1]
        a_flat = a.view(-1, n_cols)
        b_flat = b.view(-1, n_cols)
        c_flat = torch.empty_like(a_flat)

        dtype = a.dtype
        cute_dtype = torch2cute_dtype_map[dtype]

        compile_key = (dtype, n_cols, act_type, "forward")
        if compile_key not in GatedLinear._compile_cache:
            m_sym = cute.sym_int()
            impl = GatedLinearImpl(cute_dtype, act_type)
            GatedLinear._compile_cache[compile_key] = cute.compile(
                impl.forward,
                fake_tensor(cute_dtype, (m_sym, n_cols)),
                fake_tensor(cute_dtype, (m_sym, n_cols)),
                fake_tensor(cute_dtype, (m_sym, n_cols)),
                Int32(n_cols),
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )

        GatedLinear._compile_cache[compile_key](a_flat, b_flat, c_flat, n_cols)

        ctx.save_for_backward(a, b)
        ctx.n_cols = n_cols
        ctx.dtype = dtype
        ctx.act_type = act_type

        return c_flat.view(*ori_shape)

    @staticmethod
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        n_cols = ctx.n_cols
        dtype = ctx.dtype
        act_type = ctx.act_type
        cute_dtype = torch2cute_dtype_map[dtype]

        dc_flat, a_flat, b_flat = dc.view(-1, n_cols), a.view(-1, n_cols), b.view(-1, n_cols)
        da_flat, db_flat = torch.empty_like(a_flat), torch.empty_like(b_flat)

        compile_key = (dtype, n_cols, act_type, "backward")
        if compile_key not in GatedLinear._compile_cache:
            m_sym = cute.sym_int()
            impl = GatedLinearImpl(cute_dtype, act_type)
            GatedLinear._compile_cache[compile_key] = cute.compile(
                impl.backward,
                fake_tensor(cute_dtype, (m_sym, n_cols)),
                fake_tensor(cute_dtype, (m_sym, n_cols)),
                fake_tensor(cute_dtype, (m_sym, n_cols)),
                fake_tensor(cute_dtype, (m_sym, n_cols)),
                fake_tensor(cute_dtype, (m_sym, n_cols)),
                Int32(n_cols),
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )

        GatedLinear._compile_cache[compile_key](dc_flat, a_flat, b_flat, da_flat, db_flat, n_cols)
        return da_flat.view_as(a), db_flat.view_as(b), None


def geglu_func(a: Tensor, b: Tensor) -> Tensor:
    return GatedLinear.apply(a, b, "gelu")


def swiglu_func(a: Tensor, b: Tensor) -> Tensor:
    return GatedLinear.apply(a, b, "silu")


def reglu_func(a: Tensor, b: Tensor) -> Tensor:
    return GatedLinear.apply(a, b, "relu")
