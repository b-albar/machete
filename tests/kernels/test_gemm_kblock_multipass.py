# Copyright (c) 2025, Machete Authors
"""Tests for GEMM K-block iteration with multi-pass execution.

When a CTA processes more tiles than ring buffer pages (multi-pass),
the double-buffered K-block pipeline must correctly handle:
  - TMA loading of first 2 K-blocks
  - cpasync loading of remaining K-blocks in compute
  - work_notify_mbar phase tracking across slot reuse

These tests use num_sms=1 to force all tiles through a single CTA,
guaranteeing multi-pass execution regardless of GPU SM count.
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
    return (a.float() @ b_t.float().t()).to(a.dtype)


def _run_gemm(a, b_t, tile_m=64, tile_n=32, tile_k=32, num_sms=1,
              page_size=None):
    """Run GemmOp and return output tensor C.

    Inputs are 2D (M, K)/(N, K), auto-wrapped to 3D (1, M, K) for GemmOp.
    Uses num_sms=1 by default to force multi-pass execution.
    """
    from machete.megakernel import Megakernel, MegakernelConfig
    from machete.kernels.gemm import GemmOp

    M, K = a.shape
    N = b_t.shape[0]
    c = torch.zeros(1, M, N, dtype=a.dtype, device=a.device)

    kw = {}
    if page_size is not None:
        kw["page_size"] = page_size

    ops = GemmOp.schedule(
        a=a.unsqueeze(0), b=b_t, c=c,
        tile_sizes={"S": tile_m, "N": tile_n, "K": tile_k},
        **kw,
    )
    config = GemmOp.kernel_config(ops)
    config = MegakernelConfig(
        threads_per_block=config.threads_per_block,
        page_size=config.page_size,
        num_sms=num_sms,
    )
    kernel = Megakernel(ops, config=config)

    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()

    return c.squeeze(0)


def _gemm_case(M, K, N, dtype, tile_m=64, tile_n=32, tile_k=32,
               num_sms=1, atol=1e-1, rtol=1e-2, page_size=None):
    """Run a single GEMM test case with correctness check."""
    torch.manual_seed(42)
    a = torch.randn(M, K, dtype=dtype, device="cuda")
    b = torch.randn(K, N, dtype=dtype, device="cuda")
    b_t = b.t().contiguous()

    c = _run_gemm(a, b_t, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
                  num_sms=num_sms, page_size=page_size)
    ref = _gemm_reference(a, b_t)
    torch.testing.assert_close(c, ref, atol=atol, rtol=rtol)


# =============================================================================
# Single-pass baseline (1 tile, multiple K-blocks)
# =============================================================================

@requires_gpu
class TestGemmKBlockSinglePass:
    """K-block iterations with a single (M,N) tile — no multi-pass.

    These should always work since the DMA loop runs once.
    """

    def test_2_k_blocks(self):
        """K=64, tile_K=32 → 2 K-blocks, 1 tile."""
        _gemm_case(64, 64, 32, torch.float16)

    def test_4_k_blocks(self):
        """K=128, tile_K=32 → 4 K-blocks, 1 tile."""
        _gemm_case(64, 128, 32, torch.float16)

    def test_8_k_blocks(self):
        """K=256, tile_K=32 → 8 K-blocks, 1 tile."""
        _gemm_case(64, 256, 32, torch.float16)

    def test_3_k_blocks_odd(self):
        """K=96, tile_K=32 → 3 K-blocks (odd), 1 tile."""
        _gemm_case(64, 96, 32, torch.float16)


# =============================================================================
# Multi-pass with K-blocks (the critical test cases)
# =============================================================================

@requires_gpu
class TestGemmKBlockMultiPass:
    """K-block iterations with multiple (M,N) tiles on num_sms=1.

    This is the critical case: DMA processes tiles sequentially,
    each tile has TMA load of first 2 K-blocks + cpasync for the rest.
    """

    # ----- 2 tiles, varying K-blocks -----

    def test_2_tiles_2_kblocks(self):
        """2 M-tiles × 2 K-blocks (even K). Simplest multi-pass."""
        _gemm_case(128, 64, 32, torch.float16)

    def test_2_tiles_3_kblocks(self):
        """2 M-tiles × 3 K-blocks (odd K). Phase tracking differs for odd K."""
        _gemm_case(128, 96, 32, torch.float16)

    def test_2_tiles_4_kblocks(self):
        """2 M-tiles × 4 K-blocks (even K)."""
        _gemm_case(128, 128, 32, torch.float16)

    def test_2_tiles_8_kblocks(self):
        """2 M-tiles × 8 K-blocks (even K, many iterations)."""
        _gemm_case(128, 256, 32, torch.float16)

    # ----- 4+ tiles with slot reuse -----

    def test_4_tiles_2_kblocks(self):
        """4 M-tiles × 2 K-blocks. With 2 pages, each slot used twice."""
        _gemm_case(256, 64, 32, torch.float16)

    def test_4_tiles_3_kblocks(self):
        """4 M-tiles × 3 K-blocks (odd K). Slot reuse + odd phase."""
        _gemm_case(256, 96, 32, torch.float16)

    def test_4_tiles_4_kblocks(self):
        """4 M-tiles × 4 K-blocks. Heavy multi-pass."""
        _gemm_case(256, 128, 32, torch.float16)

    # ----- Multi-tile in both M and N -----

    def test_mn_tiles_2_kblocks(self):
        """2×2 tile grid × 2 K-blocks. 4 tiles total."""
        _gemm_case(128, 64, 64, torch.float16)

    def test_mn_tiles_4_kblocks(self):
        """2×2 tile grid × 4 K-blocks. 4 tiles, heavy K-loop."""
        _gemm_case(128, 128, 64, torch.float16)

    # ----- Stress: many tiles × many K-blocks -----

    def test_8_tiles_4_kblocks(self):
        """4×2 tile grid × 4 K-blocks. 8 tiles, each slot reused 4x."""
        _gemm_case(256, 128, 64, torch.float16)

    def test_8_tiles_2_kblocks_bf16(self):
        """Same as above but bf16. Verify dtype doesn't affect framework."""
        _gemm_case(256, 64, 64, torch.bfloat16)

    # ----- Different tile_K sizes -----

    def test_tile_k_64_multipass(self):
        """tile_K=64, K=128 → 2 K-blocks. Larger tile, fewer iters."""
        _gemm_case(128, 128, 32, torch.float16, tile_k=64,
                   page_size=32768)

    def test_tile_k_16_multipass(self):
        """tile_K=16, K=64 → 4 K-blocks. Smallest tile, most iters."""
        _gemm_case(128, 64, 32, torch.float16, tile_k=16)

    # ----- Single K-block (no cpasync), multi-pass baseline -----

    def test_multi_pass_no_cpasync(self):
        """2 tiles × 1 K-block. Multi-pass WITHOUT cpasync — must work."""
        _gemm_case(128, 32, 32, torch.float16)

    def test_4_tiles_no_cpasync(self):
        """4 tiles × 1 K-block. Slot reuse without cpasync."""
        _gemm_case(256, 32, 32, torch.float16)


# =============================================================================
# Double-buffered K-blocks (always 2-stage now)
# =============================================================================

@requires_gpu
class TestGemmDoubleBufferSinglePass:
    """2-stage double buffer with a single (M,N) tile — no multi-pass.

    DMA TMA-loads first 2 K-blocks, then MMA warps use cpasync for
    K-blocks 2+. Smem budget: 2 * (A_tile + B_tile) <= page_size.
    """

    def test_2_k_blocks_depth2(self):
        """K=64, tile_K=32 → 2 K-blocks. Exactly fills both buffers."""
        _gemm_case(64, 64, 32, torch.float16)

    def test_3_k_blocks_depth2(self):
        """K=96, tile_K=32 → 3 K-blocks. Two TMA + one cpasync."""
        _gemm_case(64, 96, 32, torch.float16)

    def test_4_k_blocks_depth2(self):
        """K=128, tile_K=32 → 4 K-blocks. Full double-buffer pipeline."""
        _gemm_case(64, 128, 32, torch.float16)

    def test_8_k_blocks_depth2(self):
        """K=256, tile_K=32 → 8 K-blocks. Many overlapped iterations."""
        _gemm_case(64, 256, 32, torch.float16)

    def test_tile_k16_depth2(self):
        """tile_K=16, K=64 → 4 K-blocks. Smallest tile + double buffer."""
        _gemm_case(64, 64, 32, torch.float16, tile_k=16)


@requires_gpu
class TestGemmDoubleBufferMultiPass:
    """2-stage double buffer with multiple (M,N) tiles on num_sms=1.

    This tests the full pipeline: TMA prefill, cpasync overlap, and
    correct phase tracking across tile boundaries.
    """

    def test_2_tiles_2_kblocks_depth2(self):
        """2 M-tiles × 2 K-blocks."""
        _gemm_case(128, 64, 32, torch.float16)

    def test_2_tiles_4_kblocks_depth2(self):
        """2 M-tiles × 4 K-blocks."""
        _gemm_case(128, 128, 32, torch.float16)

    def test_4_tiles_4_kblocks_depth2(self):
        """4 M-tiles × 4 K-blocks. Heavy multi-pass."""
        _gemm_case(256, 128, 32, torch.float16)

    def test_mn_tiles_4_kblocks_depth2(self):
        """2×2 tile grid × 4 K-blocks."""
        _gemm_case(128, 128, 64, torch.float16)

    def test_8_tiles_4_kblocks_depth2(self):
        """4×2 tile grid × 4 K-blocks. 8 tiles, each slot reused."""
        _gemm_case(256, 128, 64, torch.float16)

    def test_8_tiles_depth2_bf16(self):
        """8 tiles × 4 K-blocks, bf16."""
        _gemm_case(256, 128, 64, torch.bfloat16)
