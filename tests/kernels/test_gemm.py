# Copyright (c) 2025, Machete Authors
"""Tests for GEMM megakernel operation.

Verifies correctness of GemmOp against PyTorch reference implementation.
Tests TMA-based warp-specialized execution.
"""

import pytest
import torch

from machete.megakernel import Megakernel, MegakernelConfig
from machete.kernels.gemm import GemmOp
from machete.utils.testing import is_hopper_available


def get_tolerances(dtype):
    """Get appropriate rtol/atol for dtype.

    bf16/fp16 have less precision than float32, so we use relaxed tolerances.
    """
    if dtype == torch.bfloat16:
        return {"rtol": 5e-2, "atol": 5e-2}
    elif dtype == torch.float16:
        return {"rtol": 1e-2, "atol": 1e-2}
    return {"rtol": 1e-4, "atol": 1e-4}


def gemm_pytorch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch reference GEMM: C = A @ B^T.

    Args:
        a: Input tensor with shape (M, K)
        b: Weight tensor with shape (N, K)

    Returns:
        Output tensor with shape (M, N)
    """
    # B is (N, K), we want A @ B^T = (M, K) @ (K, N) = (M, N)
    return torch.mm(a, b.t())


@pytest.mark.skipif(not is_hopper_available(), reason="Hopper (SM90+) GPU required")
class TestGemmOp:
    """Tests for GemmOp with TMA and warp specialization."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("m,n,k", [
        (64, 64, 64),      # Single tile
        (128, 128, 64),    # 2x2 tiles
        (128, 128, 128),   # 2x2 tiles, larger K
        (256, 256, 128),   # 4x4 tiles
    ])
    def test_gemm_basic(self, dtype, m, n, k):
        """Test basic GEMM: C = A @ B^T."""
        torch.manual_seed(42)

        # Create input tensors
        a = torch.randn(m, k, dtype=dtype, device="cuda")
        b = torch.randn(n, k, dtype=dtype, device="cuda")
        c = torch.empty(m, n, dtype=dtype, device="cuda")

        # PyTorch reference (compute in float32 for accuracy)
        c_ref = gemm_pytorch(a.float(), b.float()).to(dtype)

        # Schedule GEMM operation
        ops = [GemmOp.schedule(a=a, b=b, c=c)]
        # 4 warps for compute (simplified version without TMA)
        # Need ~16KB smem: A(8KB) + B(8KB)
        config = MegakernelConfig(threads_per_block=128)
        kernel = Megakernel(ops, config=config)
        kernel.run()

        # Compare
        torch.testing.assert_close(
            c, c_ref,
            **get_tolerances(dtype),
            msg=f"GEMM mismatch for M={m}, N={n}, K={k}, dtype={dtype}"
        )

    @pytest.mark.parametrize("dtype", [torch.float16])
    def test_gemm_multiple_runs(self, dtype):
        """Test barrier reset by running kernel multiple times."""
        torch.manual_seed(42)
        m, n, k = 64, 64, 64

        a = torch.randn(m, k, dtype=dtype, device="cuda")
        b = torch.randn(n, k, dtype=dtype, device="cuda")
        c = torch.empty(m, n, dtype=dtype, device="cuda")

        ops = [GemmOp.schedule(a=a, b=b, c=c)]
        config = MegakernelConfig(threads_per_block=128)
        kernel = Megakernel(ops, config=config)

        # Run multiple times
        for i in range(3):
            a.copy_(torch.randn_like(a))
            b.copy_(torch.randn_like(b))
            c.zero_()

            c_ref = gemm_pytorch(a.float(), b.float()).to(dtype)
            kernel.run()

            torch.testing.assert_close(
                c, c_ref,
                **get_tolerances(dtype),
                msg=f"GEMM mismatch on iteration {i}"
            )

    @pytest.mark.parametrize("dtype", [torch.float16])
    @pytest.mark.skip(reason="Non-power-of-2 requires bounds checking (not yet implemented)")
    def test_gemm_non_power_of_2(self, dtype):
        """Test with non-power-of-2 dimensions (requires bounds checking)."""
        torch.manual_seed(42)
        m, n, k = 100, 100, 100

        a = torch.randn(m, k, dtype=dtype, device="cuda")
        b = torch.randn(n, k, dtype=dtype, device="cuda")
        c = torch.empty(m, n, dtype=dtype, device="cuda")

        c_ref = gemm_pytorch(a.float(), b.float()).to(dtype)

        ops = [GemmOp.schedule(a=a, b=b, c=c)]
        config = MegakernelConfig(threads_per_block=128)
        kernel = Megakernel(ops, config=config)
        kernel.run()

        torch.testing.assert_close(
            c, c_ref,
            **get_tolerances(dtype),
            msg="GEMM mismatch for non-power-of-2 dims"
        )


@pytest.mark.skipif(not is_hopper_available(), reason="Hopper (SM90+) GPU required")
class TestGemmMegakernelAPI:
    """Tests for the high-level gemm_megakernel API."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_gemm_api(self, dtype):
        """Test gemm_megakernel convenience function."""
        from machete.kernels.gemm import gemm_megakernel

        torch.manual_seed(42)
        m, n, k = 128, 128, 64

        a = torch.randn(m, k, dtype=dtype, device="cuda")
        b = torch.randn(n, k, dtype=dtype, device="cuda")

        c_ref = gemm_pytorch(a.float(), b.float()).to(dtype)
        c = gemm_megakernel(a, b)

        torch.testing.assert_close(
            c, c_ref,
            **get_tolerances(dtype),
            msg=f"gemm_megakernel mismatch for dtype={dtype}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
