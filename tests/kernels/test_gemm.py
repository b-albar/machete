# Copyright (c) 2025, Machete Authors
"""Tests for GemmOp — matmul correctness for fp16 and bf16.

Tests run on GPU (SM_90+) and compare the megakernel GemmOp against
torch.matmul with fp32 accumulation as reference.

GemmOp tiles on (M, N) with an inner K loop in compute. K reduction
uses fp32 accumulation across all K blocks, followed by a single
regular TMA store of the final C tile. No output pre-zeroing needed.
"""

import contextlib
import io

import pytest
import torch


def is_sm90_or_newer():
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor >= 90


try:
    import cutlass  # noqa: F401

    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False


requires_gpu = pytest.mark.skipif(
    not (is_sm90_or_newer() and CUTLASS_AVAILABLE),
    reason="Requires SM_90+ GPU with CUTLASS",
)


# =============================================================================
# Helpers
# =============================================================================


def _gemm_reference(a, b_t):
    """Reference GEMM: C = A @ B_T^T = A @ B, computed in fp32."""
    # a: (M, K), b_t: (N, K) → C = a @ b_t.T → (M, N)
    return (a.float() @ b_t.float().t()).to(a.dtype)


def _run_gemm(a, b_t, tile_m=64, tile_n=32, tile_k=32):
    """Run GemmOp and return output tensor C."""
    from machete.megakernel import Megakernel, MegakernelConfig
    from machete.kernels.gemm import GemmOp

    M, K = a.shape
    N = b_t.shape[0]
    c = torch.zeros(M, N, dtype=a.dtype, device=a.device)

    ops = GemmOp.schedule(
        a=a, b=b_t, c=c,
        tile_sizes={"M": tile_m, "N": tile_n, "K": tile_k},
    )
    # 5 warps: 4 MMA + 1 DMA = 160 threads
    config = MegakernelConfig(threads_per_block=160)
    kernel = Megakernel(ops, config=config)

    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()

    return c


# =============================================================================
# Tests
# =============================================================================


def _gemm_case(M, K, N, dtype, tile_m=64, tile_n=32, tile_k=32,
               atol=1e-1, rtol=1e-2):
    """Run a single GEMM test case and assert correctness."""
    torch.manual_seed(42)
    a = torch.randn(M, K, dtype=dtype, device="cuda")
    b = torch.randn(K, N, dtype=dtype, device="cuda")
    b_t = b.t().contiguous()

    c = _run_gemm(a, b_t, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k)
    ref = _gemm_reference(a, b_t)

    # fp32 accumulation across all K blocks — precision is constant
    # regardless of number of K tiles.
    torch.testing.assert_close(c, ref, atol=atol, rtol=rtol)


@requires_gpu
class TestGemmForward:
    """GEMM forward pass correctness tests."""

    # ----- Basic sizes -----

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_small_single_k_tile(self, dtype):
        """Small GEMM where K fits in a single tile_K."""
        _gemm_case(64, 32, 32, dtype)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_medium_multi_k_tiles(self, dtype):
        """Medium GEMM with multiple K tiles (K=128, tile_K=32 → 4 K tiles)."""
        _gemm_case(128, 128, 64, dtype)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_large_k(self, dtype):
        """Large K with many K tiles (K=256, tile_K=32 → 8 K tiles)."""
        _gemm_case(64, 256, 32, dtype)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_large_k_512(self, dtype):
        """Very large K (K=512, tile_K=32 → 16 K tiles)."""
        _gemm_case(64, 512, 32, dtype)

    # ----- Non-divisible dimensions -----

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_non_divisible_k(self, dtype):
        """K not divisible by tile_K — tests partial last K tile."""
        _gemm_case(64, 48, 32, dtype)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_non_divisible_k_large(self, dtype):
        """Large non-divisible K (K=80, tile_K=32 → 3 tiles, last partial)."""
        _gemm_case(64, 80, 32, dtype)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_non_divisible_m_n(self, dtype):
        """M and N not divisible by tile sizes."""
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
        """Minimum K (K=16 = one MMA K-block, single K tile)."""
        _gemm_case(64, 16, 32, dtype, tile_k=16)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_exactly_two_k_tiles(self, dtype):
        """K chosen so exactly 2 K tiles (K=64, tile_K=32)."""
        _gemm_case(64, 64, 32, dtype)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_small_m(self, dtype):
        """M smaller than tile_M — tests M boundary handling."""
        _gemm_case(16, 64, 32, dtype)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_small_n(self, dtype):
        """N smaller than tile_N — tests N boundary handling."""
        _gemm_case(64, 64, 16, dtype)

    # ----- Different tile_K sizes -----

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_tile_k_64(self, dtype):
        """Larger tile_K=64 (fewer K tiles, more work per tile)."""
        _gemm_case(64, 128, 32, dtype, tile_k=64)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_tile_k_16(self, dtype):
        """Minimum tile_K=16 (many K tiles, tests atomic add convergence)."""
        _gemm_case(64, 64, 32, dtype, tile_k=16)


# =============================================================================
# Backward Tests
# =============================================================================


def _run_gemm_backward(dout, a, b, da=None, db=None,
                       tile_m=64, tile_n=32, tile_k=32):
    """Run GemmOp backward and return (da, db).

    Both dA and dB ops go into the same megakernel. They both produce
    buffer 'c' but with different data_ptr, so the framework treats
    them as independent.
    """
    from machete.megakernel import Megakernel, MegakernelConfig
    from machete.kernels.gemm import GemmOp

    config = MegakernelConfig(threads_per_block=160)
    ts = {"M": tile_m, "N": tile_n, "K": tile_k}

    ops = GemmOp.schedule(backward=True, dout=dout, a=a, b=b, da=da, db=db,
                          tile_sizes=ts)
    if ops:
        with contextlib.redirect_stdout(io.StringIO()):
            Megakernel(ops, config=config).run()

    return da, db


def _gemm_backward_case(M, K, N, dtype, tile_m=64, tile_n=32, tile_k=32,
                        compute_da=True, compute_db=True,
                        atol=1e-1, rtol=1e-2):
    """Run a single GEMM backward test case and assert correctness."""
    torch.manual_seed(42)
    a = torch.randn(M, K, dtype=dtype, device="cuda")
    b = torch.randn(N, K, dtype=dtype, device="cuda")  # (N, K) layout
    dout = torch.randn(M, N, dtype=dtype, device="cuda")

    da = torch.zeros(M, K, dtype=dtype, device="cuda") if compute_da else None
    db = torch.zeros(N, K, dtype=dtype, device="cuda") if compute_db else None

    _run_gemm_backward(dout, a, b, da=da, db=db,
                       tile_m=tile_m, tile_n=tile_n, tile_k=tile_k)

    # fp32 accumulation — precision is constant regardless of K tiles.
    if da is not None:
        da_ref = (dout.float() @ b.float()).to(dtype)
        torch.testing.assert_close(da, da_ref, atol=atol, rtol=rtol)

    if db is not None:
        db_ref = (dout.float().t() @ a.float()).to(dtype)
        torch.testing.assert_close(db, db_ref, atol=atol, rtol=rtol)


@requires_gpu
class TestGemmBackward:
    """GEMM backward pass correctness tests."""

    # ----- dA only -----

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_backward_da_basic(self, dtype):
        """dA only, single K tile (N=32, tile_N=32)."""
        _gemm_backward_case(64, 32, 32, dtype, compute_db=False)

    # ----- dB only -----

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_backward_db_basic(self, dtype):
        """dB only, single K tile (M=64, tile_M=64)."""
        _gemm_backward_case(64, 32, 32, dtype, compute_da=False)

    # ----- Both dA + dB -----

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_backward_both(self, dtype):
        """Both dA and dB (separate megakernels, same buffer name 'c')."""
        _gemm_backward_case(64, 32, 32, dtype)

    # ----- Multiple tiles -----

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_backward_multi_tiles(self, dtype):
        """Larger shapes with multiple tiles along all dims."""
        _gemm_backward_case(128, 128, 64, dtype)

    # ----- Non-divisible -----

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_backward_non_divisible(self, dtype):
        """M, K, N all non-divisible by tile sizes.

        All dims must be multiples of 8 (TMA alignment for fp16/bf16),
        since backward remaps M→K and N→K for the two gradient GEMMs.
        """
        _gemm_backward_case(104, 80, 48, dtype)

    # ----- Large K (many K tiles) -----

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_backward_large_k(self, dtype):
        """Large K with many K tiles."""
        _gemm_backward_case(64, 256, 32, dtype)
