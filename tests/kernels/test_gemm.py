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


def _gemm_case(M, K, N, dtype, tile_m=64, tile_n=32, atol=1e-1, rtol=1e-2):
    """Run a single GEMM test case and assert correctness."""
    torch.manual_seed(42)
    a = torch.randn(M, K, dtype=dtype, device="cuda")
    b = torch.randn(K, N, dtype=dtype, device="cuda")
    b_t = b.t().contiguous()

    c = _run_gemm(a, b_t, tile_m=tile_m, tile_n=tile_n)
    ref = _gemm_reference(a, b_t)

    torch.testing.assert_close(c, ref, atol=atol, rtol=rtol)


@requires_gpu
class TestGemmForward:
    """GEMM forward pass correctness tests."""

    # ----- Basic sizes -----

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_small_single_k_step(self, dtype):
        """Small GEMM where K fits in a single tile_K step."""
        _gemm_case(64, 32, 32, dtype)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_medium_multi_k_steps(self, dtype):
        """Medium GEMM requiring multiple K-loop iterations (pipeline active)."""
        _gemm_case(128, 128, 64, dtype)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_large_k(self, dtype):
        """Large K with many pipeline stages (K=256 → 8 steps with tile_K=32)."""
        _gemm_case(64, 256, 32, dtype)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_large_k_512(self, dtype):
        """Very large K (K=512 → 16 pipeline steps)."""
        _gemm_case(64, 512, 32, dtype)

    # ----- Non-divisible dimensions -----

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_non_divisible_k(self, dtype):
        """K not divisible by tile_K — tests partial last K step."""
        _gemm_case(64, 48, 32, dtype)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_non_divisible_k_large(self, dtype):
        """Large non-divisible K (K=80 → not a multiple of tile_K=32)."""
        _gemm_case(64, 80, 32, dtype)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_non_divisible_m_n(self, dtype):
        """M and N not divisible by tile sizes.

        N must be a multiple of 8 (fp16/bf16) for TMA S2G 16-byte stride alignment.
        """
        _gemm_case(100, 64, 48, dtype)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_non_divisible_all(self, dtype):
        """M, K, N all non-divisible by their respective tile sizes."""
        _gemm_case(100, 80, 48, dtype)

    # ----- Multiple M/N tiles -----

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_multi_m_tiles(self, dtype):
        """Multiple tiles along M dimension (M=256 → 4 M-tiles)."""
        _gemm_case(256, 64, 32, dtype)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_multi_n_tiles(self, dtype):
        """Multiple tiles along N dimension (N=128 → 4 N-tiles)."""
        _gemm_case(64, 64, 128, dtype)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_multi_mn_tiles(self, dtype):
        """Multiple tiles along both M and N (4x4 tile grid)."""
        _gemm_case(256, 128, 128, dtype)

    # ----- Edge cases -----

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_k_equals_16(self, dtype):
        """Minimum K (K=16 = one MMA K-block, single step, no pipeline)."""
        _gemm_case(64, 16, 32, dtype)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_exactly_two_k_steps(self, dtype):
        """K chosen so exactly 2 K steps (pipeline prologue + 1 loop iter)."""
        _gemm_case(64, 64, 32, dtype)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_small_m(self, dtype):
        """M smaller than tile_M — tests M boundary predication."""
        _gemm_case(16, 64, 32, dtype)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_small_n(self, dtype):
        """N smaller than tile_N — tests N boundary predication."""
        _gemm_case(64, 64, 16, dtype)
