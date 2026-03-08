# Copyright (c) 2025, Machete Authors
"""Tests for fused QKNorm+RoPE Op correctness.

Verifies the megakernel QKNormRopeOp matches a PyTorch reference that
applies per-head RMSNorm then partial RoPE.

Tests cover:
1. Full RoPE (D2 = D//2) — Path A (D2 >= 32)
2. Partial RoPE with D2 >= 32 — Path A
3. Partial RoPE with D2 < 32 (Qwen3.5-style) — Path B
4. Pass-through dimensions are unchanged by RoPE
5. Normalization weight effect
"""

import pytest
import torch

from machete.megakernel import Megakernel, MegakernelConfig
from machete.kernels.qknorm_rope import QKNormRopeOp
from machete.kernels.qknorm_rope.ref import qknorm_rope_pytorch
from machete.utils.testing import verify_kernel


def is_hopper_available():
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


# =============================================================================
# Helpers
# =============================================================================


def ref_atol(q, norm_weight, cos, sin, eps=1e-6):
    """2x the error of the reference at bf16 vs fp32."""
    out_f32 = qknorm_rope_pytorch(q.float(), norm_weight.float(),
                                  cos.float(), sin.float(), eps)
    out_bf16 = qknorm_rope_pytorch(q.bfloat16(), norm_weight.bfloat16(),
                                   cos.bfloat16(), sin.bfloat16(), eps)
    err = (out_bf16.float() - out_f32).abs().max().item()
    return max(2 * err, 1e-5)


def run_qknorm_rope(q_4d, norm_weight, cos, sin, eps=1e-6, num_sms=2):
    """Run QKNormRopeOp via megakernel on (B, S, H, D) tensor."""
    b, s, h, d = q_4d.shape
    q_flat = q_4d.float().view(b * s, h, d).contiguous()
    cos_f32 = cos.float().contiguous()
    sin_f32 = sin.float().contiguous()
    nw_f32 = norm_weight.float().contiguous()

    ops = QKNormRopeOp.schedule(q=q_flat, norm_weight=nw_f32,
                                cos=cos_f32, sin=sin_f32, eps=eps)
    mk_config = MegakernelConfig(num_sms=num_sms)
    kernel = Megakernel(ops, config=mk_config)
    kernel.run()

    q_4d.copy_(q_flat.view(b, s, h, d).to(q_4d.dtype))


def mk_qknorm_rope(q, norm_weight, cos, sin):
    """Megakernel QKNorm+RoPE returning a new tensor."""
    q_out = q.clone()
    run_qknorm_rope(q_out, norm_weight, cos, sin)
    return q_out


def ref_qknorm_rope(q, norm_weight, cos, sin):
    """PyTorch reference wrapper with same signature."""
    return qknorm_rope_pytorch(q, norm_weight, cos, sin)


# =============================================================================
# GPU Tests
# =============================================================================


@pytest.mark.skipif(not is_hopper_available(),
                    reason="Hopper (SM90+) GPU required")
class TestQKNormRopeGPU:
    """Test QKNormRopeOp correctness against PyTorch reference."""

    @pytest.mark.parametrize(
        "b,s,h,d,d2",
        [
            # Path A: D2 >= 32 — fused in-register
            (1, 1, 1, 64, 32),        # minimal
            (1, 4, 2, 64, 32),         # small
            (2, 16, 4, 64, 32),        # multi-batch, full rope
            (1, 128, 32, 64, 32),      # long seq, many heads
            (1, 16, 4, 128, 64),       # D=128, full rope
            (2, 64, 8, 128, 64),       # D=128, multi-batch
            (4, 64, 8, 128, 32),       # D=128, 50% partial
            (1, 16, 4, 256, 128),      # D=256, full rope
            (2, 32, 8, 256, 32),       # Qwen3.5: D=256, D2=32
            (1, 16, 4, 256, 64),       # D=256, 50% partial
            # Path B: D2 < 32 — two-pass with barrier
            (1, 16, 4, 128, 16),       # Qwen3.5: D=128, 25% partial
            (2, 64, 8, 128, 16),       # multi-batch, D2=16
            (4, 32, 32, 64, 16),       # many heads, D2=16
        ],
        ids=[
            "min_d64",
            "small_d64",
            "b2_d64_full",
            "long_h32_d64",
            "d128_full",
            "b2_d128_full",
            "d128_half",
            "d256_full",
            "qwen_d256_d2_32",
            "d256_half",
            "qwen_d128_d2_16",
            "b2_d128_d2_16",
            "h32_d64_d2_16",
        ],
    )
    def test_forward_matches_pytorch(self, b, s, h, d, d2):
        """Megakernel QKNorm+RoPE matches PyTorch reference."""
        torch.manual_seed(42)
        q = torch.randn(b, s, h, d, dtype=torch.float32, device="cuda")
        norm_weight = torch.randn(d, dtype=torch.float32, device="cuda")
        cos = torch.randn(s, d2, dtype=torch.float32, device="cuda")
        sin = torch.randn(s, d2, dtype=torch.float32, device="cuda")

        atol = ref_atol(q, norm_weight, cos, sin)
        verify_kernel(
            "QKNorm+RoPE",
            func=mk_qknorm_rope,
            ref_func=ref_qknorm_rope,
            inputs=(q, norm_weight, cos, sin),
            dtype=torch.float32,
            atol=atol,
            check_grad=False,
        )

    def test_passthrough_dims_unchanged(self):
        """Dimensions beyond 2*D2 should only be RMSNorm'd, not rotated."""
        b, s, h, d, d2 = 1, 16, 4, 256, 32
        torch.manual_seed(42)
        q = torch.randn(b, s, h, d, dtype=torch.float32, device="cuda")
        # Use ones weight and zero sin to isolate normalization effect
        norm_weight = torch.ones(d, dtype=torch.float32, device="cuda")
        cos = torch.ones(s, d2, dtype=torch.float32, device="cuda")
        sin = torch.zeros(s, d2, dtype=torch.float32, device="cuda")

        mk_out = mk_qknorm_rope(q, norm_weight, cos, sin)
        ref_out = qknorm_rope_pytorch(q, norm_weight, cos, sin)

        # With sin=0 and cos=1, RoPE is identity → output = RMSNorm(q)
        # Passthrough dims should equal normalized values
        torch.testing.assert_close(
            mk_out[..., 2 * d2:], ref_out[..., 2 * d2:],
            atol=1e-5, rtol=1e-5,
        )

    def test_norm_weight_effect(self):
        """Scaling the norm weight should scale the output proportionally."""
        b, s, h, d, d2 = 1, 16, 4, 128, 64
        torch.manual_seed(42)
        q = torch.randn(b, s, h, d, dtype=torch.float32, device="cuda")
        cos = torch.randn(s, d2, dtype=torch.float32, device="cuda")
        sin = torch.randn(s, d2, dtype=torch.float32, device="cuda")
        w1 = torch.ones(d, dtype=torch.float32, device="cuda")
        w2 = 2.0 * torch.ones(d, dtype=torch.float32, device="cuda")

        out1 = mk_qknorm_rope(q, w1, cos, sin)
        out2 = mk_qknorm_rope(q, w2, cos, sin)

        # RoPE is linear → output scales with weight
        torch.testing.assert_close(out2, 2.0 * out1, atol=1e-4, rtol=1e-4)

    def test_eps_effect(self):
        """Very small q values should not cause NaN with proper eps."""
        b, s, h, d, d2 = 1, 4, 2, 64, 32
        torch.manual_seed(42)
        q = torch.randn(b, s, h, d, dtype=torch.float32, device="cuda") * 1e-10
        norm_weight = torch.ones(d, dtype=torch.float32, device="cuda")
        cos = torch.randn(s, d2, dtype=torch.float32, device="cuda")
        sin = torch.randn(s, d2, dtype=torch.float32, device="cuda")

        out = mk_qknorm_rope(q, norm_weight, cos, sin)
        assert not torch.isnan(out).any(), "NaN in output with tiny inputs"
        assert not torch.isinf(out).any(), "Inf in output with tiny inputs"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
