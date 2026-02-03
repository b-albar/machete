# Copyright (c) 2025, Machete Authors
"""Tests for MacheteLinear optimized GEMM layer."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from machete.patching.ops.linear import (
    MacheteLinear,
    patch_linear,
    unpatch_linear,
    HAS_QUACK_LINEAR,
)


def is_sm90_available():
    """Check if SM90+ GPU is available."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


# =============================================================================
# Unit Tests - MacheteLinear Module
# =============================================================================


class TestMacheteLinearCPU:
    """Test MacheteLinear on CPU (always falls back to PyTorch)."""

    def test_forward_matches_pytorch(self):
        """MacheteLinear forward matches nn.Linear on CPU."""
        torch.manual_seed(42)
        in_features, out_features = 64, 128
        batch_size, seq_len = 2, 16

        machete_linear = MacheteLinear(in_features, out_features, bias=False)
        pytorch_linear = nn.Linear(in_features, out_features, bias=False)
        pytorch_linear.weight.data.copy_(machete_linear.weight.data)

        x = torch.randn(batch_size, seq_len, in_features)

        out_machete = machete_linear(x)
        out_pytorch = pytorch_linear(x)

        assert torch.allclose(out_machete, out_pytorch, atol=1e-6)

    def test_forward_with_bias(self):
        """MacheteLinear with bias matches nn.Linear."""
        torch.manual_seed(42)
        in_features, out_features = 64, 128

        machete_linear = MacheteLinear(in_features, out_features, bias=True)
        pytorch_linear = nn.Linear(in_features, out_features, bias=True)
        pytorch_linear.weight.data.copy_(machete_linear.weight.data)
        pytorch_linear.bias.data.copy_(machete_linear.bias.data)

        x = torch.randn(2, 16, in_features)

        out_machete = machete_linear(x)
        out_pytorch = pytorch_linear(x)

        assert torch.allclose(out_machete, out_pytorch, atol=1e-6)


@pytest.mark.skipif(not is_sm90_available(), reason="SM90+ GPU required")
@pytest.mark.skipif(not HAS_QUACK_LINEAR, reason="quack not available")
class TestMacheteLinearGPU:
    """Test MacheteLinear on SM90+ GPU with quack GEMM."""

    @pytest.mark.parametrize("in_features,out_features", [
        (64, 128),
        (128, 256),
        (256, 512),
        (512, 1024),
        (1024, 4096),
    ])
    def test_forward_matches_pytorch(self, in_features, out_features):
        """MacheteLinear forward matches nn.Linear on GPU."""
        torch.manual_seed(42)
        batch_size, seq_len = 2, 16

        machete_linear = MacheteLinear(in_features, out_features, bias=False).cuda()
        pytorch_linear = nn.Linear(in_features, out_features, bias=False).cuda()
        pytorch_linear.weight.data.copy_(machete_linear.weight.data)

        x = torch.randn(batch_size, seq_len, in_features, device="cuda")

        out_machete = machete_linear(x)
        out_pytorch = pytorch_linear(x)

        # Allow some tolerance for different numerical paths
        assert torch.allclose(out_machete, out_pytorch, atol=1e-4, rtol=1e-3)

    def test_backward_matches_pytorch(self):
        """MacheteLinear backward matches nn.Linear."""
        torch.manual_seed(42)
        in_features, out_features = 128, 256

        machete_linear = MacheteLinear(in_features, out_features, bias=False).cuda()
        pytorch_linear = nn.Linear(in_features, out_features, bias=False).cuda()
        pytorch_linear.weight.data.copy_(machete_linear.weight.data)

        x1 = torch.randn(2, 16, in_features, device="cuda", requires_grad=True)
        x2 = x1.clone().detach().requires_grad_(True)

        out_machete = machete_linear(x1)
        out_pytorch = pytorch_linear(x2)

        grad_out = torch.randn_like(out_machete)
        out_machete.backward(grad_out)
        out_pytorch.backward(grad_out)

        assert torch.allclose(x1.grad, x2.grad, atol=1e-4, rtol=1e-3)

    def test_half_precision(self):
        """MacheteLinear works with float16."""
        torch.manual_seed(42)
        in_features, out_features = 128, 256

        machete_linear = MacheteLinear(in_features, out_features, bias=False).cuda().half()
        x = torch.randn(2, 16, in_features, device="cuda", dtype=torch.float16)

        out = machete_linear(x)
        assert out.dtype == torch.float16
        assert out.shape == (2, 16, out_features)


# =============================================================================
# Unit Tests - Patching Functions
# =============================================================================


class TestPatchLinear:
    """Test patch_linear and unpatch_linear functions."""

    def test_patch_unpatch(self):
        """Patching and unpatching preserves module state."""
        linear = nn.Linear(64, 128, bias=False)
        original_forward = linear.forward

        patch_linear(linear)
        if HAS_QUACK_LINEAR:
            assert hasattr(linear, "_machete_original_forward")

        unpatch_linear(linear)
        assert not hasattr(linear, "_machete_original_forward")

    @pytest.mark.skipif(not is_sm90_available(), reason="SM90+ GPU required")
    @pytest.mark.skipif(not HAS_QUACK_LINEAR, reason="quack not available")
    def test_patched_forward_matches(self):
        """Patched linear produces same results as unpatched."""
        torch.manual_seed(42)
        linear = nn.Linear(128, 256, bias=False).cuda()
        x = torch.randn(2, 16, 128, device="cuda")

        out_original = linear(x).clone()

        patch_linear(linear)
        out_patched = linear(x)

        assert torch.allclose(out_original, out_patched, atol=1e-4, rtol=1e-3)

    def test_double_patch_noop(self):
        """Patching twice doesn't break the module."""
        linear = nn.Linear(64, 128, bias=False)

        patch_linear(linear)
        patch_linear(linear)  # Second patch should be no-op

        if HAS_QUACK_LINEAR:
            assert hasattr(linear, "_machete_original_forward")

        unpatch_linear(linear)
        assert not hasattr(linear, "_machete_original_forward")


# =============================================================================
# Edge Cases
# =============================================================================


class TestMacheteLinearEdgeCases:
    """Test edge cases and fallback behavior."""

    def test_unaligned_dimensions_fallback(self):
        """Unaligned dimensions should fall back to PyTorch."""
        # Dimensions not divisible by 8
        machete_linear = MacheteLinear(65, 129, bias=False)
        x = torch.randn(2, 16, 65)

        # Should not raise, falls back to F.linear
        out = machete_linear(x)
        assert out.shape == (2, 16, 129)

    def test_with_bias_fallback(self):
        """With bias should fall back to PyTorch on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        machete_linear = MacheteLinear(64, 128, bias=True).cuda()
        x = torch.randn(2, 16, 64, device="cuda")

        # Should work (falls back due to bias)
        out = machete_linear(x)
        assert out.shape == (2, 16, 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
