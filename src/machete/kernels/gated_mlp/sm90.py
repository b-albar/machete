# Copyright (c) 2025, Machete Authors
import torch
from cutlass.cutlass_dsl import dsl_user_op
import os
import quack.activation as qact
import quack.gemm_act


# Helper to adapter tuple input for SwiGLU
# The GemmActMixin in Quack passes `(tRS_rD[2 * i], tRS_rD[2 * i + 1])`.
# We need to return a tuple to match the assignment interactions in Quack's gemm_act.
@dsl_user_op
def swiglu_tuple_dupe(args, *, loc=None, ip=None):
    res = qact.swiglu(args[0], args[1])
    return res, res


@dsl_user_op
def geglu_tuple_dupe(args, *, loc=None, ip=None):
    res = qact.geglu(args[0], args[1])
    return res, res


# Patch the activation map in quack.gemm_act to use our tuple-aware ops
quack.gemm_act.act_fn_map["swiglu"] = swiglu_tuple_dupe
quack.gemm_act.act_fn_map["geglu"] = geglu_tuple_dupe


def gated_mlp_sm90_forward(x, weight, bias=None, act_type="swiglu"):
    """
    Forward pass for Gated MLP on SM90 using Quack kernels.

    Args:
        x: Input tensor of shape (..., K)
        weight: Weight tensor of shape (K, 2N), expected to be interleaved [Gate, Up, ...]
        bias: Optional bias (currently unused/supported by kernel signature but not fully plumbed?)
        act_type: Activation type ("swiglu" or "geglu")

    Returns:
        Tensor of shape (..., N)
    """
    # x shape handling
    x_2d = x.reshape(-1, x.shape[-1])
    M, K = x_2d.shape
    K2, N2 = weight.shape

    assert K == K2
    assert N2 % 2 == 0

    # Check device capability
    if not x.is_cuda:
        raise ValueError("Input must be on CUDA")

    # We output to a buffer of size (M, 2N) because the kernel expects to write back
    # the same number of elements it read for the activation (2 inputs -> 2 outputs).
    # We will slice this buffer later.
    post_act = torch.empty((M, N2), dtype=x.dtype, device=x.device)

    # Launch kernel
    # args: A, B, D, C, PostAct, tile_count, activation, ...
    # We treat W as B input.
    quack.gemm_act.gemm_act(
        x_2d.reshape(1, M, K),  # A: (l, m, k)
        weight.reshape(1, N2, K),  # B: (l, n, k) - Quack expects N-major weight?
        # Previously code said `weight.reshape(1, N2, K)`.
        # If weight is (K, 2N), reshape(N2, K) implies it's laid out as (N2, K)?
        # Standard Linear layer weight is (Out, In) = (2N, K).
        # So weight.reshape(1, N2, K) works if weight is contiguous row-major (2N, K).
        # If weight is (K, 2N), we might need transpose.
        # The previous code assumed input weight was (K, 2N).
        None,  # D
        None,  # C
        post_act.reshape(1, M, N2),  # PostAct
        None,  # tile_count_semaphore
        act_type,
        tile_M=128,
        tile_N=128,
        cluster_M=1,
        cluster_N=1,
    )

    # Result is in post_act (M, 2N) where columns 2i and 2i+1 are identical.
    # We return the first of the pairs.
    return post_act[:, ::2]


class GatedMLPSM90Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, act_type="swiglu"):
        ctx.save_for_backward(x, weight)
        ctx.act_type = act_type
        return gated_mlp_sm90_forward(x, weight, act_type=act_type)

    @staticmethod
    def backward(ctx, dout):
        # Placeholder for backward pass
        # Providing proper gradients would require custom kernel or re-implementation using torch
        # similar to the SM80 implementation.
        x, weight = ctx.saved_tensors
        return None, None, None


def gated_mlp_sm90(x, weight, act_type="swiglu"):
    return GatedMLPSM90Func.apply(x, weight, act_type)
