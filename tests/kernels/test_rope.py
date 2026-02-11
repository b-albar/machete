# Copyright (c) 2025, Machete Authors
"""Tests for RoPE Op correctness against PyTorch and Triton references.

Tolerance is set to 2x the error of the PyTorch reference at bf16 vs fp32,
measured per test case. This gives a principled bound: if the megakernel is
as accurate as the reference (up to bf16 precision), the test passes.

Tests cover:
1. Forward RoPE matches PyTorch reference
2. Forward RoPE matches Triton (Unsloth) reference
3. Backward (inverse) RoPE roundtrip: backward(forward(q)) ≈ q
4. Backward RoPE matches PyTorch inverse reference
5. Multi-batch correctness (position wrapping via tile_M % seq_len)
6. Autograd forward + backward through megakernel_apply
"""

import pytest
import torch

from machete.megakernel import Megakernel, MegakernelConfig
from machete.kernels.rope import RopeOp
from machete.kernels.rope.ref import (
    rope_pytorch,
    rope_pytorch_backward,
    HAS_TRITON,
)
from machete.utils.testing import verify_kernel
from machete.megakernel.functional import megakernel_apply
from machete.kernels.rope.autograd import RopeAutogradOp

if HAS_TRITON:
    from machete.kernels.rope.ref import rope_triton


def is_hopper_available():
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


# =============================================================================
# Helpers
# =============================================================================


def ref_atol(ref_func, q, cos, sin):
    """2x the error of the PyTorch reference at bf16 vs fp32.

    Measures how much precision the reference loses when computed in bf16
    compared to fp32, then doubles it as tolerance headroom.
    """
    out_f32 = ref_func(q.float(), cos.float(), sin.float())
    out_bf16 = ref_func(q.bfloat16(), cos.bfloat16(), sin.bfloat16())
    err = (out_bf16.float() - out_f32).abs().max().item()
    return max(2 * err, 1e-7)


def run_rope_megakernel(q_4d, cos, sin, backward=False, num_sms=2):
    """Run RopeOp via megakernel on a (B, S, H, D) tensor. Modifies q_4d in-place.

    The kernel operates in fp32 internally. Input tensors are cast to fp32
    before launching and the result is written back in-place.
    """
    b, s, h, d = q_4d.shape
    q_flat = q_4d.float().view(b * s, h, d).contiguous()
    cos_f32 = cos.float().contiguous()
    sin_f32 = sin.float().contiguous()

    ops = [RopeOp.schedule(q=q_flat, cos=cos_f32, sin=sin_f32, tile_sizes={"M": 1})]
    mk_config = MegakernelConfig(num_sms=num_sms)
    kernel = Megakernel(ops, config=mk_config, backward=backward)
    kernel.run()

    q_4d.copy_(q_flat.view(b, s, h, d).to(q_4d.dtype))


def mk_rope_forward(q, cos, sin):
    """Megakernel RoPE forward returning a new tensor."""
    q_out = q.clone()
    run_rope_megakernel(q_out, cos, sin, backward=False)
    return q_out


def mk_rope_backward(q, cos, sin):
    """Megakernel inverse RoPE returning a new tensor."""
    q_out = q.clone()
    run_rope_megakernel(q_out, cos, sin, backward=True)
    return q_out


# =============================================================================
# GPU Tests — Forward
# =============================================================================


@pytest.mark.skipif(not is_hopper_available(), reason="Hopper (SM90+) GPU required")
class TestRopeForwardGPU:
    """Test RopeOp forward pass correctness."""

    @pytest.mark.parametrize("b,s,h,d", [
        # Edge cases
        (1, 1, 1, 4),      # minimal
        (1, 4, 1, 4),      # single batch/head, small dim
        # Typical LLM shapes
        (1, 128, 32, 64),   # single-batch, long seq, GQA-style heads
        (1, 512, 32, 128),  # single-batch, long seq, large head_dim
        (2, 16, 4, 64),     # small multi-batch
        (4, 64, 8, 128),    # multi-batch, medium seq
        (8, 32, 32, 64),    # large batch, many heads
        # Stress: position wrapping (batch * seq >> seq)
        (16, 8, 8, 128),    # 128 positions, wraps every 8
        (4, 256, 4, 64),    # 1024 positions
        # Odd head dims (must be even for half_d split)
        (1, 16, 4, 32),     # small head_dim
        (2, 16, 4, 256),    # large head_dim
    ], ids=[
        "minimal", "small",
        "long_seq_h32_d64", "long_seq_h32_d128",
        "b2_s16_h4_d64", "b4_s64_h8_d128", "b8_s32_h32_d64",
        "wrap_b16_s8", "wrap_b4_s256",
        "head_dim_32", "head_dim_256",
    ])
    def test_forward_matches_pytorch(self, b, s, h, d):
        """Megakernel RoPE forward matches PyTorch reference."""
        torch.manual_seed(42)
        q = torch.randn(b, s, h, d, dtype=torch.float32, device="cuda")
        cos = torch.randn(s, d // 2, dtype=torch.float32, device="cuda")
        sin = torch.randn(s, d // 2, dtype=torch.float32, device="cuda")

        atol = ref_atol(rope_pytorch, q, cos, sin)
        verify_kernel(
            "RoPE forward",
            func=mk_rope_forward,
            ref_func=rope_pytorch,
            inputs=(q, cos, sin),
            dtype=torch.float32,
            atol=atol,
            check_grad=False,
        )

    @pytest.mark.skipif(not HAS_TRITON, reason="Triton not installed")
    def test_forward_matches_triton(self):
        """Megakernel RoPE forward matches Triton (Unsloth) reference."""
        b, s, h, d = 2, 16, 4, 64
        torch.manual_seed(42)
        q = torch.randn(b, s, h, d, dtype=torch.float32, device="cuda")
        cos = torch.randn(s, d // 2, dtype=torch.float32, device="cuda")
        sin = torch.randn(s, d // 2, dtype=torch.float32, device="cuda")

        def triton_rope(q, cos, sin):
            q_out = q.clone()
            rope_triton(q_out, cos, sin)
            return q_out

        atol = ref_atol(rope_pytorch, q, cos, sin)
        verify_kernel(
            "RoPE forward (vs Triton)",
            func=mk_rope_forward,
            ref_func=triton_rope,
            inputs=(q, cos, sin),
            dtype=torch.float32,
            atol=atol,
            check_grad=False,
        )


# =============================================================================
# GPU Tests — Backward (inverse rotation)
# =============================================================================


@pytest.mark.skipif(not is_hopper_available(), reason="Hopper (SM90+) GPU required")
class TestRopeBackwardGPU:
    """Test RopeOp backward pass correctness."""

    def test_backward_roundtrip(self):
        """forward then backward should recover original q (RoPE is orthogonal)."""
        b, s, h, d = 2, 16, 4, 64
        torch.manual_seed(42)
        q = torch.randn(b, s, h, d, dtype=torch.float32, device="cuda")
        angles = torch.randn(s, d // 2, dtype=torch.float32, device="cuda")
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        q_original = q.clone()
        atol = ref_atol(rope_pytorch, q, cos, sin)

        run_rope_megakernel(q, cos, sin, backward=False)
        assert not torch.allclose(q, q_original, atol=atol)

        run_rope_megakernel(q, cos, sin, backward=True)
        assert torch.allclose(q, q_original, atol=atol), (
            f"Roundtrip max diff: {(q - q_original).abs().max().item()}"
        )

    @pytest.mark.parametrize("b,s,h,d", [
        (1, 1, 1, 4),
        (1, 4, 1, 4),
        (1, 128, 32, 64),
        (1, 512, 32, 128),
        (2, 16, 4, 64),
        (4, 64, 8, 128),
        (8, 32, 32, 64),
        (16, 8, 8, 128),
        (4, 256, 4, 64),
        (1, 16, 4, 32),
        (2, 16, 4, 256),
    ], ids=[
        "minimal", "small",
        "long_seq_h32_d64", "long_seq_h32_d128",
        "b2_s16_h4_d64", "b4_s64_h8_d128", "b8_s32_h32_d64",
        "wrap_b16_s8", "wrap_b4_s256",
        "head_dim_32", "head_dim_256",
    ])
    def test_backward_matches_pytorch(self, b, s, h, d):
        """Megakernel backward matches PyTorch inverse reference."""
        torch.manual_seed(42)
        q = torch.randn(b, s, h, d, dtype=torch.float32, device="cuda")
        cos = torch.randn(s, d // 2, dtype=torch.float32, device="cuda")
        sin = torch.randn(s, d // 2, dtype=torch.float32, device="cuda")

        atol = ref_atol(rope_pytorch_backward, q, cos, sin)
        verify_kernel(
            "RoPE backward",
            func=mk_rope_backward,
            ref_func=rope_pytorch_backward,
            inputs=(q, cos, sin),
            dtype=torch.float32,
            atol=atol,
            check_grad=False,
        )


# =============================================================================
# GPU Tests — Autograd (forward + backward through megakernel_apply)
# =============================================================================

SHAPE_PARAMS = [
    (1, 1, 1, 4),
    (1, 4, 1, 4),
    (1, 128, 32, 64),
    (1, 512, 32, 128),
    (2, 16, 4, 64),
    (4, 64, 8, 128),
    (8, 32, 32, 64),
    (16, 8, 8, 128),
    (4, 256, 4, 64),
    (1, 16, 4, 32),
    (2, 16, 4, 256),
]

SHAPE_IDS = [
    "minimal", "small",
    "long_seq_h32_d64", "long_seq_h32_d128",
    "b2_s16_h4_d64", "b4_s64_h8_d128", "b8_s32_h32_d64",
    "wrap_b16_s8", "wrap_b4_s256",
    "head_dim_32", "head_dim_256",
]


@pytest.mark.skipif(not is_hopper_available(), reason="Hopper (SM90+) GPU required")
class TestRopeAutogradGPU:
    """Test RoPE autograd bridge: forward output + backward gradients."""

    @pytest.mark.parametrize("b,s,h,d", SHAPE_PARAMS, ids=SHAPE_IDS)
    def test_autograd_forward_and_grad(self, b, s, h, d):
        """megakernel_apply forward + backward matches differentiable PyTorch ref."""
        torch.manual_seed(42)
        q = torch.randn(b, s, h, d, dtype=torch.float32, device="cuda",
                         requires_grad=True)
        cos = torch.randn(s, d // 2, dtype=torch.float32, device="cuda")
        sin = torch.randn(s, d // 2, dtype=torch.float32, device="cuda")

        def mk_autograd_rope(q, cos, sin):
            q_in = q.clone()
            return megakernel_apply(RopeAutogradOp(), q=q_in, cos=cos, sin=sin)

        atol = ref_atol(rope_pytorch, q, cos, sin)
        verify_kernel(
            "RoPE autograd",
            func=mk_autograd_rope,
            ref_func=rope_pytorch,
            inputs=(q, cos, sin),
            dtype=torch.float32,
            atol=atol,
            check_grad=True,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
