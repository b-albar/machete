# Copyright (c) 2025, Machete Authors
"""Tests for RMSNormGemmOp — fused RMSNorm + GEMM correctness."""

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

RMSNORM_EPS = 1e-6


def _rmsnorm_ref(x, weight, gemma=False):
    """Reference RMSNorm in fp32."""
    x_f = x.float()
    rms = (x_f ** 2).mean(dim=-1, keepdim=True).add(RMSNORM_EPS).rsqrt()
    w = weight.float()
    if gemma:
        w = 1.0 + w
    return (x_f * rms * w.unsqueeze(0)).to(x.dtype)


def _rmsnorm_gemm_ref(a, b_t, weight, gemma=False):
    """Reference: rmsnorm(A, w) @ B_T^T in fp32."""
    # a: (M, K), b_t: (N, K), weight: (K,)
    normed = _rmsnorm_ref(a, weight, gemma=gemma)
    return (normed.float() @ b_t.float().t()).to(a.dtype)


def _run_fused(a, b_t, weight, gemma=False):
    """Run RMSNormGemmOp and return output."""
    from machete.megakernel import Megakernel
    from machete.kernels.rmsnorm_gemm import RMSNormGemmOp

    M, K = a.shape
    N = b_t.shape[0]
    c = torch.zeros(1, M, N, dtype=a.dtype, device=a.device)

    ops = RMSNormGemmOp.schedule(
        a=a.unsqueeze(0), b=b_t, c=c,
        rmsnorm_weight=weight, gemma=gemma,
    )
    config = RMSNormGemmOp.kernel_config(ops)
    kernel = Megakernel(ops, config=config)

    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()

    return c.squeeze(0)


@requires_gpu
class TestRMSNormGemmForward:
    """Fused RMSNorm + GEMM correctness tests."""

    def test_basic_small(self):
        """Basic small case: M=64, K=64, N=32."""
        torch.manual_seed(42)
        M, K, N = 64, 64, 32
        a = torch.randn(M, K, dtype=torch.float16, device="cuda")
        b = torch.randn(K, N, dtype=torch.float16, device="cuda")
        b_t = b.t().contiguous()
        w = torch.randn(K, dtype=torch.float16, device="cuda")

        c = _run_fused(a, b_t, w)
        ref = _rmsnorm_gemm_ref(a, b_t, w)
        torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)

    @pytest.mark.parametrize("M,K,N", [
        (64, 128, 64),
        (128, 256, 64),
        (128, 256, 128),
        (256, 512, 64),
    ])
    def test_shapes(self, M, K, N):
        """Test various shapes."""
        torch.manual_seed(42)
        a = torch.randn(M, K, dtype=torch.float16, device="cuda")
        b = torch.randn(K, N, dtype=torch.float16, device="cuda")
        b_t = b.t().contiguous()
        w = torch.randn(K, dtype=torch.float16, device="cuda")

        c = _run_fused(a, b_t, w)
        ref = _rmsnorm_gemm_ref(a, b_t, w)
        torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)

    def test_bf16(self):
        """Test bf16 dtype."""
        torch.manual_seed(42)
        M, K, N = 128, 128, 64
        a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
        b_t = b.t().contiguous()
        w = torch.randn(K, dtype=torch.bfloat16, device="cuda")

        c = _run_fused(a, b_t, w)
        ref = _rmsnorm_gemm_ref(a, b_t, w)
        torch.testing.assert_close(c, ref, atol=2e-1, rtol=2e-2)

    def test_gemma_mode(self):
        """Test Gemma variant: weight = 1 + w."""
        torch.manual_seed(42)
        M, K, N = 128, 128, 64
        a = torch.randn(M, K, dtype=torch.float16, device="cuda")
        b = torch.randn(K, N, dtype=torch.float16, device="cuda")
        b_t = b.t().contiguous()
        w = torch.randn(K, dtype=torch.float16, device="cuda")

        c = _run_fused(a, b_t, w, gemma=True)
        ref = _rmsnorm_gemm_ref(a, b_t, w, gemma=True)
        torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)

    def test_ones_weight(self):
        """With weight=ones, should equal regular GEMM with rstd scaling."""
        torch.manual_seed(42)
        M, K, N = 64, 64, 32
        a = torch.randn(M, K, dtype=torch.float16, device="cuda")
        b = torch.randn(K, N, dtype=torch.float16, device="cuda")
        b_t = b.t().contiguous()
        w = torch.ones(K, dtype=torch.float16, device="cuda")

        c = _run_fused(a, b_t, w)
        ref = _rmsnorm_gemm_ref(a, b_t, w)
        torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)

    def test_multi_k_blocks(self):
        """K > tile_K forces multiple K-block iterations."""
        torch.manual_seed(42)
        M, K, N = 128, 512, 64
        a = torch.randn(M, K, dtype=torch.float16, device="cuda")
        b = torch.randn(K, N, dtype=torch.float16, device="cuda")
        b_t = b.t().contiguous()
        w = torch.randn(K, dtype=torch.float16, device="cuda")

        c = _run_fused(a, b_t, w)
        ref = _rmsnorm_gemm_ref(a, b_t, w)
        torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)

    def test_large_k(self):
        """Large K=4096 (typical hidden dim)."""
        torch.manual_seed(42)
        M, K, N = 128, 4096, 64
        a = torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.1
        b = torch.randn(K, N, dtype=torch.float16, device="cuda") * 0.01
        b_t = b.t().contiguous()
        w = torch.randn(K, dtype=torch.float16, device="cuda")

        c = _run_fused(a, b_t, w)
        ref = _rmsnorm_gemm_ref(a, b_t, w)
        torch.testing.assert_close(c, ref, atol=2e-1, rtol=5e-2)
