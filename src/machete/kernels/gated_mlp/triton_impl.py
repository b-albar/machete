# Copyright (c) 2025, Machete Authors
import torch
import triton
import triton.language as tl


@triton.jit
def _gated_mlp_fwd_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    # Fused GEMM + Gated Activation
    # X (M, K) @ W (K, 2*N) -> Y (M, N)

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # -----------------------------------------------------------
    # Map program ids to blocks
    # -----------------------------------------------------------
    # (Optional) Swizzle for better L2 cache reuse

    # pointers to blocks
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = x_ptr + (offs_am[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    # We load gate and up portions from W.
    # Assume W is stored as [W_gate, W_up] concatenated along N dimension
    # W_gate is at w_ptr, W_up is at w_ptr + N * stride_wn
    b_gate_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_bn[None, :] * stride_wn)
    b_up_ptrs = w_ptr + N * stride_wn + (offs_k[:, None] * stride_wk + offs_bn[None, :] * stride_wn)

    # -----------------------------------------------------------
    # Mainloop
    # -----------------------------------------------------------
    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k_idx * BLOCK_K, other=0.0)
        b_gate = tl.load(b_gate_ptrs, mask=offs_k[:, None] < K - k_idx * BLOCK_K, other=0.0)
        b_up = tl.load(b_up_ptrs, mask=offs_k[:, None] < K - k_idx * BLOCK_K, other=0.0)

        acc_gate += tl.dot(a, b_gate)
        acc_up += tl.dot(a, b_up)

        a_ptrs += BLOCK_K * stride_xk
        b_gate_ptrs += BLOCK_K * stride_wk
        b_up_ptrs += BLOCK_K * stride_wk

    # -----------------------------------------------------------
    # Epilogue: Fused Activation
    # -----------------------------------------------------------
    if ACTIVATION == "silu":
        # silu(gate) * up
        y = (tl.sigmoid(acc_gate) * acc_gate) * acc_up
    elif ACTIVATION == "gelu":
        # simplified gelu approx
        y = 0.5 * acc_gate * (1.0 + tl.tanh(0.79788 * (acc_gate + 0.044715 * acc_gate * acc_gate * acc_gate))) * acc_up
    else:
        y = acc_gate * acc_up

    # Store result
    offs_ym = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_yn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    y_ptrs = y_ptr + stride_ym * offs_ym[:, None] + stride_yn * offs_yn[None, :]
    mask = (offs_ym[:, None] < M) & (offs_yn[None, :] < N)
    tl.store(y_ptrs, y.to(y_ptr.dtype.element_ty), mask=mask)


def gated_mlp_triton(x, weight, activation="silu"):
    # Reshape x to (M, K) to handle arbitrary input shapes (e.g. B, S, D)
    original_shape = x.shape
    x_2d = x.reshape(-1, original_shape[-1])

    M, K = x_2d.shape
    K2, N2 = weight.shape
    N = N2 // 2

    y_2d = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))

    _gated_mlp_fwd_kernel[grid](
        x_2d,
        weight,
        y_2d,
        M,
        N,
        K,
        x_2d.stride(0),
        x_2d.stride(1),
        weight.stride(0),
        weight.stride(1),
        y_2d.stride(0),
        y_2d.stride(1),
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
        GROUP_SIZE_M=8,
        ACTIVATION=activation,
    )

    # Reshape output back to matches input structure but with last dim N
    return y_2d.reshape(original_shape[:-1] + (N,))
