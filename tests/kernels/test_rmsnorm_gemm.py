# Copyright (c) 2025, Machete Authors
"""Tests for RMSNormGemmOp — fused RMSNorm + GEMM correctness."""

import contextlib
import io
import importlib.util

import pytest
import torch

if importlib.util.find_spec("cutlass") is None:
    pytest.skip("Requires CUTLASS", allow_module_level=True)
from tests.kernels.support import requires_sm90_cutlass


requires_gpu = requires_sm90_cutlass

FORWARD_CASES = [
    (64, 64, 32),
    (64, 128, 64),
    (128, 256, 128),
    (128, 512, 64),
    (128, 4096, 64),
]

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

    @pytest.mark.parametrize("M,K,N", FORWARD_CASES)
    def test_forward_matrix(self, M, K, N):
        """Representative fused RMSNorm+GEMM shape matrix."""
        torch.manual_seed(42)
        scale_a = 0.1 if K >= 4096 else 1.0
        scale_b = 0.01 if K >= 4096 else 1.0
        a = torch.randn(M, K, dtype=torch.float16, device="cuda") * scale_a
        b = torch.randn(K, N, dtype=torch.float16, device="cuda") * scale_b
        b_t = b.t().contiguous()
        w = torch.randn(K, dtype=torch.float16, device="cuda")

        c = _run_fused(a, b_t, w)
        ref = _rmsnorm_gemm_ref(a, b_t, w)
        atol, rtol = ((2e-1, 5e-2) if K >= 4096 else (1e-1, 1e-2))
        torch.testing.assert_close(c, ref, atol=atol, rtol=rtol)

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
