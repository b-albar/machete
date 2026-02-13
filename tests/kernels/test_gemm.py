# Copyright (c) 2025, Machete Authors
"""Tests for GemmOp — matmul correctness for fp16 and bf16.

Tests run on GPU (SM_120+) and compare the megakernel GemmOp against
torch.matmul with fp32 accumulation as reference.
"""

import contextlib
import io

import pytest
import torch


def is_sm120_or_newer():
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor >= 120


try:
    import cutlass  # noqa: F401

    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False


requires_gpu = pytest.mark.skipif(
    not (is_sm120_or_newer() and CUTLASS_AVAILABLE),
    reason="Requires SM_120+ GPU with CUTLASS",
)


# =============================================================================
# Helpers
# =============================================================================


def _gemm_reference(a, b_t):
    """Reference GEMM: C = A @ B_T^T = A @ B, computed in fp32."""
    # a: (M, K), b_t: (N, K) → C = a @ b_t.T → (M, N)
    return (a.float() @ b_t.float().t()).to(a.dtype)


def _run_gemm(a, b_t, tile_m=64, tile_n=32):
    """Run GemmOp and return output tensor C."""
    from machete.megakernel import Megakernel, MegakernelConfig
    from machete.kernels.gemm import GemmOp

    M, K = a.shape
    N = b_t.shape[0]
    c = torch.zeros(M, N, dtype=a.dtype, device=a.device)

    ops = [GemmOp.schedule(
        a=a, b=b_t, c=c,
        tile_sizes={"M": tile_m, "N": tile_n},
    )]
    # 5 warps: 4 MMA + 1 DMA = 160 threads
    config = MegakernelConfig(threads_per_block=160)
    kernel = Megakernel(ops, config=config)

    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()

    return c


# =============================================================================
# Tests
# =============================================================================


@requires_gpu
class TestGemmForward:
    """GEMM forward pass correctness tests."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_small_single_k_step(self, dtype):
        """Small GEMM where K fits in a single tile_K step."""
        M, K, N = 64, 32, 32
        torch.manual_seed(42)
        a = torch.randn(M, K, dtype=dtype, device="cuda")
        b = torch.randn(K, N, dtype=dtype, device="cuda")
        b_t = b.t().contiguous()  # (N, K)

        c = _run_gemm(a, b_t, tile_m=64, tile_n=32)
        ref = _gemm_reference(a, b_t)

        torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_medium_multi_k_steps(self, dtype):
        """Medium GEMM requiring multiple K-loop iterations."""
        M, K, N = 128, 128, 64
        torch.manual_seed(42)
        a = torch.randn(M, K, dtype=dtype, device="cuda")
        b = torch.randn(K, N, dtype=dtype, device="cuda")
        b_t = b.t().contiguous()  # (N, K)

        c = _run_gemm(a, b_t, tile_m=64, tile_n=32)
        ref = _gemm_reference(a, b_t)

        torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_non_divisible_k(self, dtype):
        """K not divisible by tile_K — tests partial last K step."""
        M, K, N = 64, 48, 32
        torch.manual_seed(42)
        a = torch.randn(M, K, dtype=dtype, device="cuda")
        b = torch.randn(K, N, dtype=dtype, device="cuda")
        b_t = b.t().contiguous()  # (N, K)

        c = _run_gemm(a, b_t, tile_m=64, tile_n=32)
        ref = _gemm_reference(a, b_t)

        torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_non_divisible_m_n(self, dtype):
        """M and N not divisible by tile sizes — tests boundary handling.

        N must be a multiple of 8 (fp16/bf16) for TMA S2G 16-byte stride alignment.
        """
        M, K, N = 100, 64, 48
        torch.manual_seed(42)
        a = torch.randn(M, K, dtype=dtype, device="cuda")
        b = torch.randn(K, N, dtype=dtype, device="cuda")
        b_t = b.t().contiguous()  # (N, K)

        c = _run_gemm(a, b_t, tile_m=64, tile_n=32)
        ref = _gemm_reference(a, b_t)

        torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)
