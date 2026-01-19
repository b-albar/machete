# Copyright (c) 2025, Machete Authors
"""Tests for SM120 GEMM kernel."""

import torch
import pytest


def gemm_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.0,
    c: torch.Tensor = None,
) -> torch.Tensor:
    """
    Reference GEMM implementation using PyTorch.

    Computes: C = alpha * A @ B + beta * C
    Where:
    - A is (L, M, K) or (M, K)
    - B is (L, K, N) or (K, N)
    - C is (L, M, N) or (M, N)

    Args:
        a: Input tensor (L, M, K) or (M, K)
        b: Weight tensor (L, K, N) or (K, N)
        alpha: Scalar for A @ B result
        beta: Scalar for existing C
        c: Optional output tensor for beta scaling

    Returns:
        Output tensor (L, M, N) or (M, N)
    """
    # Handle batched vs non-batched
    was_2d = a.dim() == 2
    if was_2d:
        a = a.unsqueeze(0)
    if b.dim() == 2:
        b = b.unsqueeze(0)

    # Compute in FP32 for reference accuracy
    a_f32 = a.float()
    b_f32 = b.float()

    # A @ B where A is (L, M, K) and B is (L, K, N)
    result = torch.bmm(a_f32, b_f32)  # (L, M, N)
    result = alpha * result

    if beta != 0.0 and c is not None:
        c_f32 = c.float()
        if c_f32.dim() == 2:
            c_f32 = c_f32.unsqueeze(0)
        result = result + beta * c_f32

    result = result.to(a.dtype)

    if was_2d:
        result = result.squeeze(0)

    return result


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "m_dim, n_dim, k_dim",
    [
        (64, 64, 64),  # Small square
        (128, 128, 128),  # Medium square
        (256, 256, 256),  # Larger square
        (128, 256, 64),  # Rectangular
        (256, 128, 512),  # Rectangular (different)
    ],
)
def test_gemm_forward_basic(m_dim, n_dim, k_dim, dtype):
    """Test GEMM forward pass correctness for basic shapes."""
    from machete.kernels.gemm.sm120 import gemm_sm120

    torch.manual_seed(42)
    device = "cuda"

    print(f"\nTesting GEMM M={m_dim}, N={n_dim}, K={k_dim}, dtype={dtype}")

    # Create inputs: A is (M, K), B is (K, N)
    a = torch.randn(m_dim, k_dim, device=device, dtype=dtype) * 0.1
    b = torch.randn(k_dim, n_dim, device=device, dtype=dtype) * 0.1

    # Reference
    out_ref = gemm_ref(a, b)

    # Kernel
    out_kernel = gemm_sm120(a, b)

    # Check shapes
    expected_shape = (m_dim, n_dim)
    assert out_kernel.shape == expected_shape, f"Shape mismatch: {out_kernel.shape} vs {expected_shape}"

    # Check values with tolerance scaled for accumulation
    base_atol = 0.1 if dtype == torch.bfloat16 else 0.05
    atol = base_atol * (1 + k_dim / 128)  # Scale with K for accumulation error

    diff = (out_kernel - out_ref).abs().max().item()
    print(f"  Max diff: {diff:.6f}, atol: {atol:.6f}")

    assert diff <= atol, f"GEMM forward mismatch: {diff} > {atol}"
    print("  PASSED!")


def test_gemm_single_kernel():
    """Test using Sm120GemmWS (SingleKernel) directly."""
    from machete.kernels.gemm.sm120 import Sm120GemmWS
    from cutlass import Float32

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float16
    m, n, k = 128, 128, 64

    print(f"\nTesting Sm120GemmWS SingleKernel direct use")

    a = torch.randn(1, m, k, device=device, dtype=dtype)
    b = torch.randn(1, k, n, device=device, dtype=dtype)
    c = torch.zeros(1, m, n, device=device, dtype=dtype)

    kernel = Sm120GemmWS(dtype, dtype, dtype, Float32, (128, 128, 64))

    # Grid/Block functions are built-in for Sm120GemmWS via SingleKernel base
    out = kernel(a, b)  # Calls __call__ -> apply_autograd

    ref = torch.matmul(a[0].float(), b[0].float()).to(dtype)
    diff = (out[0] - ref).abs().max().item()
    print(f"  Max diff: {diff:.6f}")
    assert diff < 0.1
    print("  PASSED!")


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize(
    "batch_size, m_dim, n_dim, k_dim",
    [
        (2, 64, 64, 64),  # Small batched
        (4, 128, 128, 128),  # Medium batched
        (8, 64, 128, 64),  # Larger batch, smaller dims
    ],
)
def test_gemm_forward_batched(batch_size, m_dim, n_dim, k_dim, dtype):
    """Test batched GEMM forward pass correctness."""
    from machete.kernels.gemm.sm120 import gemm_sm120

    torch.manual_seed(42)
    device = "cuda"

    print(f"\nTesting Batched GEMM L={batch_size}, M={m_dim}, N={n_dim}, K={k_dim}, dtype={dtype}")

    # Create batched inputs: A is (L, M, K), B is (L, K, N)
    a = torch.randn(batch_size, m_dim, k_dim, device=device, dtype=dtype) * 0.1
    b = torch.randn(batch_size, k_dim, n_dim, device=device, dtype=dtype) * 0.1

    # Reference
    out_ref = gemm_ref(a, b)

    # Kernel
    out_kernel = gemm_sm120(a, b)

    # Check shapes
    expected_shape = (batch_size, m_dim, n_dim)
    assert out_kernel.shape == expected_shape, f"Shape mismatch: {out_kernel.shape} vs {expected_shape}"

    # Check values
    base_atol = 0.1 if dtype == torch.bfloat16 else 0.05
    atol = base_atol * (1 + k_dim / 128)

    diff = (out_kernel - out_ref).abs().max().item()
    print(f"  Max diff: {diff:.6f}, atol: {atol:.6f}")

    assert diff <= atol, f"Batched GEMM forward mismatch: {diff} > {atol}"
    print("  PASSED!")


@pytest.mark.parametrize("dtype", [torch.float16])
def test_gemm_alpha_scaling(dtype):
    """Test GEMM with alpha scaling."""
    from machete.kernels.gemm.sm120 import gemm_sm120

    torch.manual_seed(42)
    device = "cuda"
    m_dim, n_dim, k_dim = 128, 128, 128
    alpha = 2.5

    print(f"\nTesting GEMM with alpha={alpha}")

    a = torch.randn(m_dim, k_dim, device=device, dtype=dtype) * 0.1
    b = torch.randn(k_dim, n_dim, device=device, dtype=dtype) * 0.1

    # Reference
    out_ref = gemm_ref(a, b, alpha=alpha)

    # Kernel
    out_kernel = gemm_sm120(a, b, alpha=alpha)

    diff = (out_kernel - out_ref).abs().max().item()
    atol = 0.2
    print(f"  Max diff: {diff:.6f}, atol: {atol:.6f}")

    assert diff <= atol, f"GEMM alpha scaling mismatch: {diff} > {atol}"
    print("  PASSED!")


@pytest.mark.parametrize("dtype", [torch.float16])
def test_gemm_convenience_function(dtype):
    """Test the gemm_sm120 convenience function."""
    from machete.kernels.gemm.sm120 import gemm_sm120

    torch.manual_seed(42)
    device = "cuda"
    m_dim, n_dim, k_dim = 128, 128, 128

    print(f"\nTesting gemm_sm120 convenience function")

    a = torch.randn(m_dim, k_dim, device=device, dtype=dtype) * 0.1
    b = torch.randn(k_dim, n_dim, device=device, dtype=dtype) * 0.1

    # Reference
    out_ref = gemm_ref(a, b)

    # Convenience function
    out_kernel = gemm_sm120(a, b)

    diff = (out_kernel - out_ref).abs().max().item()
    atol = 0.1
    print(f"  Max diff: {diff:.6f}, atol: {atol:.6f}")

    assert diff <= atol, f"gemm_sm120 mismatch: {diff} > {atol}"
    print("  PASSED!")


@pytest.mark.parametrize(
    "tile_m, tile_n, tile_k",
    [
        (64, 64, 32),
        (128, 128, 32),
        (128, 64, 32),
    ],
)
def test_gemm_tile_configs(tile_m, tile_n, tile_k):
    """Test GEMM with different tile configurations."""
    from machete.kernels.gemm.sm120 import gemm_sm120

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float16
    m_dim, n_dim, k_dim = 256, 256, 256

    print(f"\nTesting GEMM tiles=({tile_m}, {tile_n}, {tile_k})")

    a = torch.randn(m_dim, k_dim, device=device, dtype=dtype) * 0.1
    b = torch.randn(k_dim, n_dim, device=device, dtype=dtype) * 0.1

    # Reference
    out_ref = gemm_ref(a, b)

    # Kernel with custom tiles
    out_kernel = gemm_sm120(a, b, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k)

    diff = (out_kernel - out_ref).abs().max().item()
    atol = 0.15
    print(f"  Max diff: {diff:.6f}, atol: {atol:.6f}")

    assert diff <= atol, f"GEMM tile config mismatch: {diff} > {atol}"
    print("  PASSED!")


def test_gemm_cache_clearing():
    """Test that kernel cache clearing works."""
    from machete.kernels.gemm.sm120 import clear_kernel_cache, _kernel_cache, _compile_cache

    # Trigger some kernel compilations
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float16

    from machete.kernels.gemm.sm120 import gemm_sm120

    a = torch.randn(64, 64, device=device, dtype=dtype)
    b = torch.randn(64, 64, device=device, dtype=dtype)
    _ = gemm_sm120(a, b)

    # Clear cache
    clear_kernel_cache()

    assert len(_kernel_cache) == 0, "Kernel cache should be empty after clearing"
    assert len(_compile_cache) == 0, "Compile cache should be empty after clearing"
    print("  Cache clearing PASSED!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
