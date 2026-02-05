# Copyright (c) 2025, Machete Authors
"""Tests for MacheteRoPE rotary position embedding wrapper."""

import pytest
import torch

from machete.patching.ops.rope import (
    MacheteRoPE,
    apply_rope,
    HAS_MEGAKERNEL_ROPE,
)


def is_hopper_available():
    """Check if Hopper (SM90+) GPU is available."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


def rope_pytorch_ref(q, cos, sin):
    """PyTorch reference implementation of RoPE."""
    d = q.shape[-1]
    q1 = q[..., : d // 2]
    q2 = q[..., d // 2 :]

    # Broadcast cos/sin to match q shape
    cos_b = cos.unsqueeze(0).unsqueeze(2)  # (1, S, 1, D/2)
    sin_b = sin.unsqueeze(0).unsqueeze(2)

    out1 = q1 * cos_b - q2 * sin_b
    out2 = q2 * cos_b + q1 * sin_b

    return torch.cat([out1, out2], dim=-1)


# =============================================================================
# Unit Tests - MacheteRoPE Module
# =============================================================================


class TestMacheteRoPECPU:
    """Test MacheteRoPE on CPU (always uses PyTorch fallback)."""

    def test_forward_matches_reference(self):
        """MacheteRoPE forward matches reference on CPU."""
        torch.manual_seed(42)
        dim, max_seq_len = 64, 128
        b, s, h = 2, 16, 4

        rope = MacheteRoPE(dim=dim, max_seq_len=max_seq_len)
        q = torch.randn(b, s, h, dim)

        # Get cos/sin from the module
        cos = rope.cos_cached[:s]
        sin = rope.sin_cached[:s]

        q_rotated, _ = rope(q)
        q_ref = rope_pytorch_ref(q, cos, sin)

        assert torch.allclose(q_rotated, q_ref, atol=1e-5)

    def test_forward_with_keys(self):
        """MacheteRoPE can rotate both q and k."""
        torch.manual_seed(42)
        dim, max_seq_len = 64, 128
        b, s, h = 2, 16, 4

        rope = MacheteRoPE(dim=dim, max_seq_len=max_seq_len)
        q = torch.randn(b, s, h, dim)
        k = torch.randn(b, s, h, dim)

        q_rotated, k_rotated = rope(q, k=k)

        assert q_rotated.shape == q.shape
        assert k_rotated.shape == k.shape
        assert k_rotated is not None

    def test_cache_extension(self):
        """MacheteRoPE extends cache when sequence is longer."""
        dim = 64
        rope = MacheteRoPE(dim=dim, max_seq_len=16)

        assert rope.max_seq_len_cached == 16

        # Request longer sequence
        q = torch.randn(1, 32, 1, dim)
        rope(q)

        assert rope.max_seq_len_cached >= 32

    def test_different_bases(self):
        """MacheteRoPE with different base values."""
        dim = 64

        rope_10k = MacheteRoPE(dim=dim, base=10000.0)
        rope_50k = MacheteRoPE(dim=dim, base=50000.0)

        # Different bases should give different inv_freq
        assert not torch.allclose(rope_10k.inv_freq, rope_50k.inv_freq)


@pytest.mark.skipif(not is_hopper_available(), reason="Hopper (SM90+) GPU required")
@pytest.mark.skipif(not HAS_MEGAKERNEL_ROPE, reason="megakernel RoPE not available")
class TestMacheteRoPEGPU:
    """Test MacheteRoPE on GPU with megakernel backend."""

    @pytest.mark.parametrize("b,s,h,d", [
        (1, 16, 4, 64),
        (2, 32, 8, 128),
        (4, 64, 16, 64),
        (1, 128, 32, 128),
    ])
    def test_forward_matches_reference(self, b, s, h, d):
        """MacheteRoPE forward matches reference on GPU."""
        torch.manual_seed(42)

        rope = MacheteRoPE(dim=d, max_seq_len=s, device="cuda", dtype=torch.float32)
        q = torch.randn(b, s, h, d, device="cuda", dtype=torch.float32)

        cos = rope.cos_cached[:s]
        sin = rope.sin_cached[:s]

        # Clone q before rotation since megakernel modifies in-place
        q_clone = q.clone()
        q_ref = rope_pytorch_ref(q_clone, cos, sin)

        q_rotated, _ = rope(q)

        assert torch.allclose(q_rotated, q_ref, atol=1e-4)

    def test_backward_gradient(self):
        """MacheteRoPE supports autograd backward."""
        torch.manual_seed(42)
        b, s, h, d = 2, 16, 4, 64

        rope = MacheteRoPE(dim=d, max_seq_len=s, device="cuda", dtype=torch.float32)
        q = torch.randn(b, s, h, d, device="cuda", dtype=torch.float32, requires_grad=True)

        q_rotated, _ = rope(q)
        loss = q_rotated.sum()
        loss.backward()

        assert q.grad is not None
        assert q.grad.shape == q.shape


# =============================================================================
# Unit Tests - Functional API
# =============================================================================


class TestApplyRopeFunctional:
    """Test apply_rope functional API."""

    def test_apply_rope_cpu(self):
        """apply_rope works on CPU."""
        torch.manual_seed(42)
        b, s, h, d = 2, 16, 4, 64

        q = torch.randn(b, s, h, d)
        cos = torch.randn(s, d // 2)
        sin = torch.randn(s, d // 2)

        q_rotated, k_rotated = apply_rope(q, cos, sin)

        assert q_rotated.shape == q.shape
        assert k_rotated is None

    def test_apply_rope_with_keys(self):
        """apply_rope with both q and k."""
        torch.manual_seed(42)
        b, s, h, d = 2, 16, 4, 64

        q = torch.randn(b, s, h, d)
        k = torch.randn(b, s, h, d)
        cos = torch.randn(s, d // 2)
        sin = torch.randn(s, d // 2)

        q_rotated, k_rotated = apply_rope(q, cos, sin, k=k)

        assert q_rotated.shape == q.shape
        assert k_rotated.shape == k.shape

    @pytest.mark.skipif(not is_hopper_available(), reason="Hopper (SM90+) GPU required")
    @pytest.mark.skipif(not HAS_MEGAKERNEL_ROPE, reason="megakernel RoPE not available")
    def test_apply_rope_gpu(self):
        """apply_rope uses megakernel on GPU."""
        torch.manual_seed(42)
        b, s, h, d = 2, 16, 4, 64

        q = torch.randn(b, s, h, d, device="cuda", dtype=torch.float32)
        cos = torch.randn(s, d // 2, device="cuda", dtype=torch.float32)
        sin = torch.randn(s, d // 2, device="cuda", dtype=torch.float32)

        # Clone q before rotation since megakernel modifies in-place
        q_clone = q.clone()
        q_ref = rope_pytorch_ref(q_clone, cos, sin)

        q_rotated, _ = apply_rope(q, cos, sin)

        assert torch.allclose(q_rotated, q_ref, atol=1e-4)


# =============================================================================
# Edge Cases
# =============================================================================


class TestMacheteRoPEEdgeCases:
    """Test edge cases for MacheteRoPE."""

    def test_single_position(self):
        """RoPE with single position."""
        rope = MacheteRoPE(dim=64, max_seq_len=1)
        q = torch.randn(1, 1, 1, 64)

        q_rotated, _ = rope(q)
        assert q_rotated.shape == q.shape

    def test_large_head_dim(self):
        """RoPE with large head dimension."""
        rope = MacheteRoPE(dim=256, max_seq_len=16)
        q = torch.randn(1, 16, 4, 256)

        q_rotated, _ = rope(q)
        assert q_rotated.shape == q.shape

    def test_many_heads(self):
        """RoPE with many attention heads."""
        rope = MacheteRoPE(dim=64, max_seq_len=16)
        q = torch.randn(1, 16, 128, 64)

        q_rotated, _ = rope(q)
        assert q_rotated.shape == q.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
