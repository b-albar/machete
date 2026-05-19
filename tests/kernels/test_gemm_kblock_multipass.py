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
from tests.kernels.support import requires_sm90_cutlass


requires_gpu = requires_sm90_cutlass

SINGLE_PASS_CASES = [
    (64, 64, 32, torch.float16, 64, 32, 32, None),
    (64, 96, 32, torch.float16, 64, 32, 32, None),
    (64, 128, 32, torch.float16, 64, 32, 32, None),
    (64, 256, 32, torch.float16, 64, 32, 32, None),
]

MULTIPASS_CASES = [
    (128, 64, 32, torch.float16, 64, 32, 32, None),
    (128, 96, 32, torch.float16, 64, 32, 32, None),
    (128, 128, 32, torch.float16, 64, 32, 32, None),
    (128, 256, 32, torch.float16, 64, 32, 32, None),
    (256, 64, 32, torch.float16, 64, 32, 32, None),
    (256, 128, 32, torch.float16, 64, 32, 32, None),
    (128, 128, 64, torch.float16, 64, 32, 32, None),
    (256, 128, 64, torch.float16, 64, 32, 32, None),
    (256, 64, 64, torch.bfloat16, 64, 32, 32, None),
    (128, 128, 32, torch.float16, 64, 32, 64, 32768),
    (128, 64, 32, torch.float16, 64, 32, 16, None),
    (128, 32, 32, torch.float16, 64, 32, 32, None),
]

DOUBLE_BUFFER_CASES = [
    (64, 64, 32, torch.float16, 64, 32, 32, None),
    (64, 96, 32, torch.float16, 64, 32, 32, None),
    (64, 128, 32, torch.float16, 64, 32, 32, None),
    (64, 256, 32, torch.float16, 64, 32, 32, None),
    (64, 64, 32, torch.float16, 64, 32, 16, None),
    (128, 64, 32, torch.float16, 64, 32, 32, None),
    (128, 128, 32, torch.float16, 64, 32, 32, None),
    (256, 128, 32, torch.float16, 64, 32, 32, None),
    (128, 128, 64, torch.float16, 64, 32, 32, None),
    (256, 128, 64, torch.float16, 64, 32, 32, None),
    (256, 128, 64, torch.bfloat16, 64, 32, 32, None),
]


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

    @pytest.mark.parametrize("M,K,N,dtype,tile_m,tile_n,tile_k,page_size", SINGLE_PASS_CASES)
    def test_single_pass_matrix(self, M, K, N, dtype, tile_m, tile_n, tile_k, page_size):
        """Representative single-pass K-block cases."""
        _gemm_case(M, K, N, dtype, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, page_size=page_size)


# =============================================================================
# Multi-pass with K-blocks (the critical test cases)
# =============================================================================

@requires_gpu
class TestGemmKBlockMultiPass:
    """K-block iterations with multiple (M,N) tiles on num_sms=1.

    This is the critical case: DMA processes tiles sequentially,
    each tile has TMA load of first 2 K-blocks + cpasync for the rest.
    """

    @pytest.mark.parametrize("M,K,N,dtype,tile_m,tile_n,tile_k,page_size", MULTIPASS_CASES)
    def test_multipass_matrix(self, M, K, N, dtype, tile_m, tile_n, tile_k, page_size):
        """Representative multi-pass K-block matrix."""
        _gemm_case(M, K, N, dtype, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, page_size=page_size)


# =============================================================================
# Double-buffered K-blocks (always 2-stage now)
# =============================================================================

@requires_gpu
class TestGemmDoubleBufferSinglePass:
    """2-stage double buffer with a single (M,N) tile — no multi-pass.

    DMA TMA-loads first 2 K-blocks, then MMA warps use cpasync for
    K-blocks 2+. Smem budget: 2 * (A_tile + B_tile) <= page_size.
    """

    @pytest.mark.parametrize("M,K,N,dtype,tile_m,tile_n,tile_k,page_size", SINGLE_PASS_CASES[:4] + [DOUBLE_BUFFER_CASES[4]])
    def test_double_buffer_single_pass_matrix(self, M, K, N, dtype, tile_m, tile_n, tile_k, page_size):
        """Representative double-buffer single-pass cases."""
        _gemm_case(M, K, N, dtype, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, page_size=page_size)


@requires_gpu
class TestGemmDoubleBufferMultiPass:
    """2-stage double buffer with multiple (M,N) tiles on num_sms=1.

    This tests the full pipeline: TMA prefill, cpasync overlap, and
    correct phase tracking across tile boundaries.
    """

    @pytest.mark.parametrize("M,K,N,dtype,tile_m,tile_n,tile_k,page_size", DOUBLE_BUFFER_CASES[5:])
    def test_double_buffer_multipass_matrix(self, M, K, N, dtype, tile_m, tile_n, tile_k, page_size):
        """Representative double-buffer multi-pass cases."""
        _gemm_case(M, K, N, dtype, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, page_size=page_size)
