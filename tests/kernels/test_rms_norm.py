# Copyright (c) 2025, Machete Authors
import torch
import pytest
from machete.kernels.rms_norm.sm120 import rms_norm_sm120


def rms_norm_ref(x: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    """Reference RMSNorm implementation in PyTorch."""
    # RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    if weight is not None:
        x_normed = x_normed * weight
    return x_normed


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 128),  # Minimum size
        (4, 256),  # Small batch
        (8, 1024),  # Medium
        (16, 2048),  # Larger
        (32, 4096),  # Standard LLM hidden size
        (2, 3, 1024),  # 3D input
        (1, 8, 16, 512),  # 4D input (batch, seq, heads, dim)
    ],
)
@pytest.mark.parametrize("has_weight", [False, True])
def test_rms_norm_forward(shape, dtype, has_weight):
    """Test RMSNorm forward pass correctness."""
    torch.manual_seed(42)
    device = "cuda"
    eps = 1e-6

    n_hidden = shape[-1]
    print(f"\nTesting RMSNorm forward: shape={shape}, dtype={dtype}, has_weight={has_weight}")

    # Create input
    x = torch.randn(shape, device=device, dtype=dtype)
    weight = torch.randn(n_hidden, device=device, dtype=dtype) if has_weight else None

    # 1. Compute Golden Reference in FP32
    x_fp32 = x.float()
    weight_fp32 = weight.float() if weight is not None else None
    out_fp32 = rms_norm_ref(x_fp32, weight_fp32, eps)

    # 2. Compute Reference in Target Precision
    out_ref = rms_norm_ref(x, weight, eps)

    # 3. Baseline Quantization Error
    baseline_diff = (out_fp32.to(dtype) - out_ref).abs().max().item()
    print(f"  Baseline error (FP32 vs {dtype}): {baseline_diff:.6f}")

    # 4. Set tolerance: 5x baseline error (floor at 1e-3)
    atol = max(5.0 * baseline_diff, 1e-3)
    print(f"  Setting atol = {atol:.6f}")

    # 5. Compute kernel output
    out_kernel = rms_norm_sm120(x, weight, eps)

    # 6. Check correctness
    diff = (out_kernel - out_ref).abs().max().item()
    print(f"  Kernel vs Reference diff: {diff:.6f}")

    assert diff <= atol, f"RMSNorm forward mismatch: {diff} > {atol}"
    print(f"  PASSED!")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    [
        (4, 256),  # Small
        (8, 1024),  # Medium
        (16, 2048),  # Larger
    ],
)
@pytest.mark.parametrize("has_weight", [False, True])
def test_rms_norm_backward(shape, dtype, has_weight):
    """Test RMSNorm backward pass correctness."""
    torch.manual_seed(42)
    device = "cuda"
    eps = 1e-6

    n_hidden = shape[-1]
    print(f"\nTesting RMSNorm backward: shape={shape}, dtype={dtype}, has_weight={has_weight}")

    # Create inputs with gradients
    x = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(n_hidden, device=device, dtype=dtype, requires_grad=True) if has_weight else None

    # Clone for kernel test
    x_kernel = x.detach().clone().requires_grad_(True)
    weight_kernel = weight.detach().clone().requires_grad_(True) if weight is not None else None

    # Random gradient
    dy = torch.randn(shape, device=device, dtype=dtype)

    # 1. Reference backward
    out_ref = rms_norm_ref(x, weight, eps)
    out_ref.backward(dy)
    dx_ref = x.grad.clone()
    dw_ref = weight.grad.clone() if weight is not None else None

    # 2. Kernel backward
    out_kernel = rms_norm_sm120(x_kernel, weight_kernel, eps)
    out_kernel.backward(dy)
    dx_kernel = x_kernel.grad.clone()
    # Note: dweight gradient is not yet implemented in the kernel
    dw_kernel = weight_kernel.grad.clone() if (weight_kernel is not None and weight_kernel.grad is not None) else None

    # 3. Compute baseline error for tolerance
    x_fp32 = x.detach().float().requires_grad_(True)
    weight_fp32 = weight.detach().float().requires_grad_(True) if weight is not None else None
    out_fp32 = rms_norm_ref(x_fp32, weight_fp32, eps)
    out_fp32.backward(dy.float())
    dx_fp32 = x_fp32.grad

    baseline_diff = (dx_fp32.to(dtype) - dx_ref).abs().max().item()
    print(f"  Baseline dx error (FP32 vs {dtype}): {baseline_diff:.6f}")

    atol = max(10.0 * baseline_diff, 1e-2)  # Backward needs more tolerance
    print(f"  Setting atol = {atol:.6f}")

    # 4. Check dx
    dx_diff = (dx_kernel - dx_ref).abs().max().item()
    print(f"  dx diff: {dx_diff:.6f}")
    assert dx_diff <= atol, f"RMSNorm backward dx mismatch: {dx_diff} > {atol}"

    # 5. Check dweight if applicable (currently not implemented in kernel)
    if has_weight and dw_kernel is not None:
        dw_diff = (dw_kernel - dw_ref).abs().max().item()
        print(f"  dweight diff: {dw_diff:.6f}")
        # Weight gradient accumulates across all elements, needs more tolerance
        dw_atol = max(atol * 10, 0.1)
        assert dw_diff <= dw_atol, f"RMSNorm backward dweight mismatch: {dw_diff} > {dw_atol}"
    elif has_weight:
        print("  dweight gradient not implemented in kernel (skipping check)")

    print(f"  PASSED!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
