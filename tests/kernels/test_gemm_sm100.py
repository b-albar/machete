# Copyright (c) 2025, Machete Authors
"""Tests for GemmSm100Op — SM100 UMMA GEMM correctness for fp16 and bf16.

Tests run on Blackwell (SM_120+) GPUs and compare GemmSm100Op against
torch.matmul with fp32 accumulation as reference.
"""

import contextlib
import io

import pytest
import torch


def _has_sm100_smem():
    """SM100 GEMM needs 2×64KB pages; check smem budget is sufficient."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    if major < 9 or major > 10:
        return False
    props = torch.cuda.get_device_properties(0)
    return props.max_shared_memory_per_block_optin >= 2 * 65536


try:
    import cutlass  # noqa: F401
    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False


requires_blackwell = pytest.mark.skipif(
    not (_has_sm100_smem() and CUTLASS_AVAILABLE),
    reason="Requires Hopper (SM100) GPU with >=128KB smem and CUTLASS",
)


# =============================================================================
# Helpers
# =============================================================================


def _gemm_reference(a, b_t):
    """Reference GEMM: C = A @ B_T^T, computed in fp32."""
    return (a.float() @ b_t.float().t()).to(a.dtype)


def _run_gemm_sm100(a, b_t, tile_m=128, tile_n=128, tile_k=64, page_size=65536):
    """Run GemmSm100Op and return output tensor C."""
    from machete.megakernel import Megakernel
    from machete.kernels.gemm import GemmSm100Op

    M, K = a.shape
    N = b_t.shape[0]
    c = torch.zeros(1, M, N, dtype=a.dtype, device=a.device)

    ops = GemmSm100Op.schedule(
        a=a.unsqueeze(0), b=b_t, c=c,
        tile_sizes={"S": tile_m, "N": tile_n, "K": tile_k},
        page_size=page_size,
    )
    config = GemmSm100Op.kernel_config(ops)
    kernel = Megakernel(ops, config=config)

    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()

    return c.squeeze(0)


# =============================================================================
# Tests
# =============================================================================


def _gemm_case(M, K, N, dtype, tile_m=128, tile_n=128, tile_k=64,
               page_size=65536, atol=1e-1, rtol=1e-2):
    """Run a single GEMM test case and assert correctness."""
    torch.manual_seed(42)
    a = torch.randn(M, K, dtype=dtype, device="cuda")
    b = torch.randn(K, N, dtype=dtype, device="cuda")
    b_t = b.t().contiguous()

    c = _run_gemm_sm100(a, b_t, tile_m=tile_m, tile_n=tile_n,
                        tile_k=tile_k, page_size=page_size)
    ref = _gemm_reference(a, b_t)

    torch.testing.assert_close(c, ref, atol=atol, rtol=rtol)


@requires_blackwell
class TestGemmSm100Forward:
    """GemmSm100Op forward pass correctness tests."""

    # --- Basic shapes ---

    def test_basic_128x128(self):
        _gemm_case(128, 128, 128, torch.bfloat16)

    def test_basic_128x64(self):
        _gemm_case(128, 64, 128, torch.bfloat16)

    def test_basic_256x256(self):
        _gemm_case(256, 256, 256, torch.bfloat16)

    # --- Larger shapes ---

    def test_256x512x1024(self):
        _gemm_case(256, 512, 1024, torch.bfloat16)

    def test_512x1024x512(self):
        _gemm_case(512, 1024, 512, torch.bfloat16)

    def test_1024x1024x1024(self):
        _gemm_case(1024, 1024, 1024, torch.bfloat16)

    # --- Multiple K-blocks ---

    def test_multi_k_blocks(self):
        """K=256 with tile_K=64 → 4 K-blocks."""
        _gemm_case(128, 256, 128, torch.bfloat16, tile_k=64)

    def test_large_k(self):
        """K=1024 → 16 K-blocks."""
        _gemm_case(128, 1024, 128, torch.bfloat16, tile_k=64)

    # --- Smaller tile configs ---

    def test_tile_64x64(self):
        _gemm_case(128, 128, 128, torch.bfloat16,
                   tile_m=64, tile_n=64, tile_k=64, page_size=32768)

    def test_tile_128x64(self):
        _gemm_case(256, 256, 256, torch.bfloat16,
                   tile_m=128, tile_n=64, tile_k=64, page_size=49152)

    # --- fp16 ---

    def test_fp16_basic(self):
        _gemm_case(128, 128, 128, torch.float16)

    def test_fp16_large(self):
        _gemm_case(256, 512, 256, torch.float16)

    # --- Multi-tile (output larger than one tile) ---

    def test_multi_tile_m(self):
        """M=512 with tile_S=128 → 4 M-tiles."""
        _gemm_case(512, 128, 128, torch.bfloat16,
                   tile_m=128, tile_n=128, tile_k=64)

    def test_multi_tile_n(self):
        """N=512 with tile_N=128 → 4 N-tiles."""
        _gemm_case(128, 128, 512, torch.bfloat16,
                   tile_m=128, tile_n=128, tile_k=64)

    def test_multi_tile_both(self):
        """M=256, N=256 with 128×128 tiles → 2×2 tiles."""
        _gemm_case(256, 256, 256, torch.bfloat16,
                   tile_m=128, tile_n=128, tile_k=64)


@requires_blackwell
class TestGemmSm100Activation:
    """GemmSm100Op with fused activation."""

    def test_relu(self):
        torch.manual_seed(42)
        M, K, N = 128, 128, 128
        dtype = torch.bfloat16
        a = torch.randn(M, K, dtype=dtype, device="cuda")
        b = torch.randn(K, N, dtype=dtype, device="cuda")
        b_t = b.t().contiguous()

        from machete.megakernel import Megakernel
        from machete.kernels.gemm import GemmSm100Op

        c = torch.zeros(1, M, N, dtype=dtype, device="cuda")
        ops = GemmSm100Op.schedule(
            a=a.unsqueeze(0), b=b_t, c=c,
            activation='relu', page_size=65536,
        )
        config = GemmSm100Op.kernel_config(ops)
        kernel = Megakernel(ops, config=config)
        with contextlib.redirect_stdout(io.StringIO()):
            kernel.run()

        ref = _gemm_reference(a, b_t)
        ref = torch.relu(ref)
        torch.testing.assert_close(c.squeeze(0), ref, atol=1e-1, rtol=1e-2)

    def test_silu(self):
        torch.manual_seed(42)
        M, K, N = 128, 128, 128
        dtype = torch.bfloat16
        a = torch.randn(M, K, dtype=dtype, device="cuda")
        b = torch.randn(K, N, dtype=dtype, device="cuda")
        b_t = b.t().contiguous()

        from machete.megakernel import Megakernel
        from machete.kernels.gemm import GemmSm100Op

        c = torch.zeros(1, M, N, dtype=dtype, device="cuda")
        ops = GemmSm100Op.schedule(
            a=a.unsqueeze(0), b=b_t, c=c,
            activation='silu', page_size=65536,
        )
        config = GemmSm100Op.kernel_config(ops)
        kernel = Megakernel(ops, config=config)
        with contextlib.redirect_stdout(io.StringIO()):
            kernel.run()

        ref = _gemm_reference(a, b_t)
        ref = torch.nn.functional.silu(ref)
        torch.testing.assert_close(c.squeeze(0), ref, atol=1e-1, rtol=1e-2)


@requires_blackwell
class TestGemmSm100AutoTile:
    """Test auto-tiling at different page sizes."""

    def test_auto_tile_65536(self):
        """64KB page → should pick 128×128×64."""
        _gemm_case(256, 256, 256, torch.bfloat16, page_size=65536,
                   tile_m=128, tile_n=128, tile_k=64)

    def test_auto_schedule(self):
        """Let schedule() pick tiles automatically."""
        torch.manual_seed(42)
        M, K, N = 256, 256, 256
        dtype = torch.bfloat16
        a = torch.randn(M, K, dtype=dtype, device="cuda")
        b_t = torch.randn(N, K, dtype=dtype, device="cuda")

        from machete.megakernel import Megakernel
        from machete.kernels.gemm import GemmSm100Op

        c = torch.zeros(1, M, N, dtype=dtype, device="cuda")
        ops = GemmSm100Op.schedule(
            a=a.unsqueeze(0), b=b_t, c=c, page_size=65536)
        config = GemmSm100Op.kernel_config(ops)
        kernel = Megakernel(ops, config=config)
        with contextlib.redirect_stdout(io.StringIO()):
            kernel.run()

        ref = _gemm_reference(a, b_t)
        torch.testing.assert_close(c.squeeze(0), ref, atol=1e-1, rtol=1e-2)
