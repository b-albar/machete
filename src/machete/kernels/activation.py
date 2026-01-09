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


class ActivationImpl:
    def __init__(self, dtype: Type[cutlass.Numeric], act_type: str):
        self.dtype = dtype
        self.act_type = act_type

    @cute.jit
    def forward(
        self,
        mX: cute.Tensor,
        mY: cute.Tensor,
        n_elements: Int32,
        stream: cuda.CUstream,
    ):
        num_threads = 256
        num_blocks = (n_elements + num_threads - 1) // num_threads
        # Cap blocks to avoid overly large grid, though 1D grid is usually fine up to 2^31
        # But we'll use a grid-stride loop in the kernel anyway
        grid = [min(num_blocks, 65535), 1, 1]
        block = [num_threads, 1, 1]
        self.forward_kernel(mX, mY, n_elements).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def forward_kernel(self, mX: cute.Tensor, mY: cute.Tensor, n_elements: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        num_threads, _, _ = cute.arch.block_dim()
        num_blocks, _, _ = cute.arch.grid_dim()

        start = bidx * num_threads + tidx
        stride = num_blocks * num_threads

        for i in range(start, n_elements, stride):
            x_val = mX[i].to(Float32)

            if const_expr(self.act_type == "silu"):
                y_val = quack_act.silu(x_val)
            elif const_expr(self.act_type == "gelu"):
                y_val = quack_act.gelu_tanh_approx(x_val)
            elif const_expr(self.act_type == "relu"):
                y_val = quack_act.relu(x_val)
            else:
                y_val = x_val

            mY[i] = y_val.to(mY.element_type)

    @cute.jit
    def backward(
        self,
        mdY: cute.Tensor,
        mX: cute.Tensor,
        mdX: cute.Tensor,
        n_elements: Int32,
        stream: cuda.CUstream,
    ):
        num_threads = 256
        num_blocks = (n_elements + num_threads - 1) // num_threads
        grid = [min(num_blocks, 65535), 1, 1]
        block = [num_threads, 1, 1]
        self.backward_kernel(mdY, mX, mdX, n_elements).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def backward_kernel(self, mdY: cute.Tensor, mX: cute.Tensor, mdX: cute.Tensor, n_elements: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        num_threads, _, _ = cute.arch.block_dim()
        num_blocks, _, _ = cute.arch.grid_dim()

        start = bidx * num_threads + tidx
        stride = num_blocks * num_threads

        for i in range(start, n_elements, stride):
            dy_val = mdY[i].to(Float32)
            x_val = mX[i].to(Float32)

            if const_expr(self.act_type == "silu"):
                sig_x = quack_act.sigmoid(x_val)
                deriv = sig_x * (1.0 + x_val * (1.0 - sig_x))
            elif const_expr(self.act_type == "gelu"):
                deriv, _ = quack_act.dgelu_tanh_approx(x_val, Float32(1.0))
            elif const_expr(self.act_type == "relu"):
                x_pos = quack_act.Boolean(x_val > 0)
                deriv = Float32(1.0) if x_pos else Float32(0.0)
            else:
                deriv = Float32(1.0)

            mdX[i] = (dy_val * deriv).to(mdX.element_type)


class Activation(torch.autograd.Function):
    _compile_cache: Dict[Tuple, Any] = {}

    @staticmethod
    def forward(ctx, x, act_type="silu"):
        ori_shape = x.shape
        n_elements = x.numel()
        x_flat = x.view(-1)
        y_flat = torch.empty_like(x_flat)

        dtype = x.dtype
        cute_dtype = torch2cute_dtype_map[dtype]

        compile_key = (dtype, "forward", act_type)
        if compile_key not in Activation._compile_cache:
            num_sym = cute.sym_int()
            impl = ActivationImpl(cute_dtype, act_type)
            Activation._compile_cache[compile_key] = cute.compile(
                impl.forward,
                fake_tensor(cute_dtype, (num_sym,)),
                fake_tensor(cute_dtype, (num_sym,)),
                Int32(n_elements),
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )

        Activation._compile_cache[compile_key](x_flat, y_flat, n_elements)

        ctx.save_for_backward(x)
        ctx.n_elements = n_elements
        ctx.dtype = dtype
        ctx.act_type = act_type

        return y_flat.view(*ori_shape)

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        n_elements = ctx.n_elements
        dtype = ctx.dtype
        act_type = ctx.act_type
        cute_dtype = torch2cute_dtype_map[dtype]

        dy_flat = dy.view(-1)
        x_flat = x.view(-1)
        dx_flat = torch.empty_like(x_flat)

        compile_key = (dtype, "backward", act_type)
        if compile_key not in Activation._compile_cache:
            num_sym = cute.sym_int()
            impl = ActivationImpl(cute_dtype, act_type)
            Activation._compile_cache[compile_key] = cute.compile(
                impl.backward,
                fake_tensor(cute_dtype, (num_sym,)),
                fake_tensor(cute_dtype, (num_sym,)),
                fake_tensor(cute_dtype, (num_sym,)),
                Int32(n_elements),
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )

        Activation._compile_cache[compile_key](dy_flat, x_flat, dx_flat, n_elements)
        return dx_flat.view_as(x), None


def silu_func(x: Tensor) -> Tensor:
    return Activation.apply(x, "silu")


def gelu_func(x: Tensor) -> Tensor:
    return Activation.apply(x, "gelu")


def relu_func(x: Tensor) -> Tensor:
    return Activation.apply(x, "relu")
