# Copyright (c) 2025, Machete Authors
"""Reference RMSNorm implementations for testing and benchmarking.

Provides PyTorch and Triton reference implementations of RMSNorm for
correctness verification and performance comparison against the
megakernel RMSNormOp.
"""

import torch


# =============================================================================
# PyTorch Reference
# =============================================================================


def rmsnorm_pytorch(x, weight, eps=1e-6, residual=False, gemma=False):
    """Pure PyTorch RMSNorm forward reference.

    Args:
        x: (..., D) float32
        weight: (D,) or (M, D) float32 — per-row weight supported via broadcasting
        eps: float
        residual: if True, compute y = rmsnorm(x) + x
        gemma: if True, use w_eff = 1 + weight (Gemma-style)

    Returns:
        (..., D) float32
    """
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x_normed = x.float() * torch.rsqrt(variance + eps)
    w = weight.float()
    if gemma:
        w = 1.0 + w
    y = x_normed * w
    if residual:
        y = y + x.float()
    return y.to(x.dtype)


def rmsnorm_backward_pytorch(dout, x, weight, eps=1e-6, residual=False, gemma=False):
    """Pure PyTorch RMSNorm backward reference.

    Args:
        dout: (..., D) float32 — grad_output
        x: (..., D) float32 — saved input
        weight: (D,) or (M, D) float32 — per-row weight supported via broadcasting
        eps: float
        residual: if True, dx includes identity gradient (dx += dout)
        gemma: if True, use w_eff = 1 + weight (Gemma-style)

    Returns:
        dx: (..., D) float32 — grad_input
    """
    x_f = x.float()
    dout_f = dout.float()
    w_f = weight.float()
    if gemma:
        w_f = 1.0 + w_f

    variance = x_f.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + eps)
    x_hat = x_f * rstd  # normalized x

    # dx = (dout * w - x_hat * mean(dout * w * x_hat)) * rstd
    dout_w = dout_f * w_f
    grad_sum = (dout_w * x_hat).mean(-1, keepdim=True)
    dx = (dout_w - x_hat * grad_sum) * rstd
    if residual:
        dx = dx + dout_f

    return dx.to(x.dtype)


# =============================================================================
# Fused Add + RMSNorm Reference
# =============================================================================


def fused_add_rmsnorm_pytorch(x, residual_in, weight, eps=1e-6, gemma=False):
    """Fused add + RMSNorm forward: residual_out = x + residual, y = rmsnorm(residual_out).

    Returns:
        (y, residual_out)
    """
    residual_out = (x.float() + residual_in.float()).to(x.dtype)
    y = rmsnorm_pytorch(residual_out, weight, eps=eps, gemma=gemma)
    return y, residual_out


def fused_add_rmsnorm_backward_pytorch(dout, residual_out, weight, eps=1e-6, gemma=False):
    """Fused add + RMSNorm backward.

    Returns:
        (dx, d_residual) — both equal to d_rmsnorm(dout, residual_out, weight)
    """
    dx = rmsnorm_backward_pytorch(dout, residual_out, weight, eps=eps, gemma=gemma)
    d_residual = dx.clone()
    return dx, d_residual


# =============================================================================
# Gated RMSNorm Reference
# =============================================================================


def rmsnorm_gated_pytorch(x, gate, weight, eps=1e-6, gemma=False):
    """Gated RMSNorm forward: y = rmsnorm(x, weight) * silu(gate).

    Returns:
        y: (..., D)
    """
    normed = rmsnorm_pytorch(x, weight, eps=eps, gemma=gemma)
    silu_gate = gate.float() * torch.sigmoid(gate.float())
    y = normed.float() * silu_gate
    return y.to(x.dtype)


def rmsnorm_gated_backward_pytorch(dout, x, gate, weight, eps=1e-6, gemma=False):
    """Gated RMSNorm backward.

    Returns:
        (dx, dgate)
    """
    x_f = x.float()
    gate_f = gate.float()
    dout_f = dout.float()
    w_f = weight.float()
    if gemma:
        w_f = 1.0 + w_f

    variance = x_f.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + eps)
    normed = x_f * rstd * w_f  # rmsnorm(x, w)

    sig = torch.sigmoid(gate_f)
    silu_gate = gate_f * sig
    silu_grad = sig * (1.0 + gate_f * (1.0 - sig))

    # dgate = dout * normed * silu'(gate)
    dgate = (dout_f * normed * silu_grad).to(x.dtype)

    # dy_norm = dout * silu(gate), then standard rmsnorm backward
    dy_norm = dout_f * silu_gate
    x_hat = x_f * rstd
    dout_w = dy_norm * w_f
    grad_sum = (dout_w * x_hat).mean(-1, keepdim=True)
    dx = ((dout_w - x_hat * grad_sum) * rstd).to(x.dtype)

    return dx, dgate


# =============================================================================
# Triton Reference (optional)
# =============================================================================

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def _calculate_settings(n_cols):
    """Calculate block size and num_warps for RMSNorm Triton kernel."""
    block_size = triton.next_power_of_2(n_cols)
    block_size = max(block_size, 32)
    block_size = min(block_size, 65536)
    num_warps = max(block_size // 256, 1)
    num_warps = min(num_warps, 32)
    return block_size, num_warps


if HAS_TRITON:

    @triton.jit
    def _rms_layernorm_forward_kernel(
        Y,
        Y_row_stride: tl.constexpr,
        X,
        X_row_stride: tl.constexpr,
        W,
        W_row_stride: tl.constexpr,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        Y += row_idx * Y_row_stride
        X += row_idx * X_row_stride

        X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
        W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)

        row_var = tl.sum(X_row * X_row, axis=0) / n_cols
        inv_var = tl.math.rsqrt(row_var + eps)
        output = X_row * inv_var * W_row
        tl.store(Y + col_offsets, output.to(X_row.dtype), mask=mask)

    @triton.jit
    def _rms_layernorm_backward_kernel(
        dY,
        dY_row_stride: tl.constexpr,
        dX,
        dX_row_stride: tl.constexpr,
        X,
        X_row_stride: tl.constexpr,
        W,
        W_row_stride: tl.constexpr,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        dY += row_idx * dY_row_stride
        dX += row_idx * dX_row_stride
        X += row_idx * X_row_stride

        dY_row = tl.load(dY + col_offsets, mask=mask, other=0).to(tl.float32)
        X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
        W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)

        row_var = tl.sum(X_row * X_row, axis=0) / n_cols
        inv_var = tl.math.rsqrt(row_var + eps)
        normed = X_row * inv_var

        dY_W = dY_row * W_row
        rowsum_dY_normed = tl.sum(dY_W * normed, axis=0)
        output = inv_var / n_cols * (n_cols * dY_W - normed * rowsum_dY_normed)
        tl.store(dX + col_offsets, output.to(dY_row.dtype), mask=mask)

    def rmsnorm_triton(x, weight, eps=1e-6):
        """Triton RMSNorm forward (out-of-place).

        Args:
            x: (M, D) float32, CUDA.
            weight: (D,) float32, CUDA.
            eps: float

        Returns:
            y: (M, D) float32
        """
        M, D = x.shape
        y = torch.empty_like(x)
        BLOCK_SIZE, num_warps = _calculate_settings(D)
        _rms_layernorm_forward_kernel[(M,)](
            y, y.stride(0),
            x, x.stride(0),
            weight, weight.stride(0),
            D, eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return y

    def rmsnorm_backward_triton(dout, x, weight, eps=1e-6):
        """Triton RMSNorm backward.

        Args:
            dout: (M, D) float32
            x: (M, D) float32
            weight: (D,) float32
            eps: float

        Returns:
            dx: (M, D) float32
        """
        M, D = x.shape
        dx = torch.empty_like(x)
        BLOCK_SIZE, num_warps = _calculate_settings(D)
        _rms_layernorm_backward_kernel[(M,)](
            dout, dout.stride(0),
            dx, dx.stride(0),
            x, x.stride(0),
            weight, weight.stride(0),
            D, eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return dx


# =============================================================================
# CUTLASS Reference (optional, requires SM90+)
# =============================================================================

# Import from the dedicated CUTLASS reference file
try:
    from machete.kernels.rms_norm.rmsnorm_cutlass import (
        rmsnorm_cutlass,
        HAS_CUTLASS_RMSNORM,
    )
except ImportError:
    HAS_CUTLASS_RMSNORM = False


__all__ = [
    "rmsnorm_pytorch",
    "rmsnorm_backward_pytorch",
    "fused_add_rmsnorm_pytorch",
    "fused_add_rmsnorm_backward_pytorch",
    "rmsnorm_gated_pytorch",
    "rmsnorm_gated_backward_pytorch",
    "HAS_TRITON",
    "HAS_CUTLASS_RMSNORM",
]

if HAS_TRITON:
    __all__.extend(["rmsnorm_triton", "rmsnorm_backward_triton"])

if HAS_CUTLASS_RMSNORM:
    __all__.append("rmsnorm_cutlass")
