# Copyright (c) 2025, Machete Authors
"""Tests for MacheteMLP and MacheteGatedMLP optimized layers."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from machete.patching.ops.mlp import (
    MacheteMLP,
    MacheteGatedMLP,
    patch_gated_mlp,
    unpatch_gated_mlp,
    HAS_QUACK_MLP,
)


def is_sm90_available():
    """Check if SM90+ GPU is available."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


def mlp_pytorch_ref(x, fc1_weight, fc2_weight, activation="gelu"):
    """PyTorch reference MLP."""
    h = F.linear(x, fc1_weight)
    if activation == "gelu":
        h = F.gelu(h)
    elif activation == "silu":
        h = F.silu(h)
    elif activation == "relu":
        h = F.relu(h)
    return F.linear(h, fc2_weight)


def gated_mlp_pytorch_ref(x, gate_weight, up_weight, down_weight, activation="silu"):
    """PyTorch reference gated MLP (SwiGLU-style)."""
    gate = F.linear(x, gate_weight)
    if activation == "silu":
        gate = F.silu(gate)
    elif activation == "gelu":
        gate = F.gelu(gate)
    up = F.linear(x, up_weight)
    return F.linear(gate * up, down_weight)


# =============================================================================
# Unit Tests - MacheteMLP Module
# =============================================================================


class TestMachetemLPCPU:
    """Test MacheteMLP on CPU (always falls back to PyTorch)."""

    def test_forward_matches_pytorch(self):
        """MacheteMLP forward matches reference on CPU."""
        torch.manual_seed(42)
        in_features, hidden_features, out_features = 64, 256, 64
        batch_size, seq_len = 2, 16

        mlp = MacheteMLP(in_features, hidden_features, out_features, activation="gelu")
        x = torch.randn(batch_size, seq_len, in_features)

        out = mlp(x)
        out_ref = mlp_pytorch_ref(x, mlp.fc1.weight, mlp.fc2.weight, activation="gelu")

        assert torch.allclose(out, out_ref, atol=1e-5)

    @pytest.mark.parametrize("activation", ["gelu", "silu", "relu"])
    def test_different_activations(self, activation):
        """MacheteMLP with different activation functions."""
        mlp = MacheteMLP(64, 256, 64, activation=activation)
        x = torch.randn(2, 16, 64)

        out = mlp(x)
        assert out.shape == (2, 16, 64)

    def test_default_dimensions(self):
        """MacheteMLP with default hidden/out dimensions."""
        mlp = MacheteMLP(64)  # hidden = 4*64 = 256, out = 64
        x = torch.randn(2, 16, 64)

        out = mlp(x)
        assert out.shape == (2, 16, 64)
        assert mlp.hidden_features == 256


@pytest.mark.skipif(not is_sm90_available(), reason="SM90+ GPU required")
@pytest.mark.skipif(not HAS_QUACK_MLP, reason="quack MLP not available")
class TestMachetemLPGPU:
    """Test MacheteMLP on SM90+ GPU with quack."""

    @pytest.mark.parametrize("in_f,hidden_f,out_f", [
        (64, 256, 64),
        (128, 512, 128),
        (256, 1024, 256),
    ])
    def test_forward_matches_pytorch(self, in_f, hidden_f, out_f):
        """MacheteMLP forward matches reference on GPU."""
        torch.manual_seed(42)

        mlp = MacheteMLP(in_f, hidden_f, out_f, activation="gelu", bias=False).cuda()
        x = torch.randn(2, 16, in_f, device="cuda")

        out = mlp(x)
        out_ref = mlp_pytorch_ref(x, mlp.fc1.weight, mlp.fc2.weight, activation="gelu")

        assert torch.allclose(out, out_ref, atol=1e-3, rtol=1e-2)

    def test_backward_gradient(self):
        """MacheteMLP supports autograd backward."""
        torch.manual_seed(42)

        mlp = MacheteMLP(64, 256, 64, bias=False).cuda()
        x = torch.randn(2, 16, 64, device="cuda", requires_grad=True)

        out = mlp(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


# =============================================================================
# Unit Tests - MacheteGatedMLP Module
# =============================================================================


class TestMacheteGatedMLPCPU:
    """Test MacheteGatedMLP on CPU."""

    def test_forward_matches_pytorch(self):
        """MacheteGatedMLP forward matches reference on CPU."""
        torch.manual_seed(42)
        in_features, hidden_features, out_features = 64, 256, 64

        mlp = MacheteGatedMLP(in_features, hidden_features, out_features, activation="silu")
        x = torch.randn(2, 16, in_features)

        out = mlp(x)
        out_ref = gated_mlp_pytorch_ref(
            x, mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight, activation="silu"
        )

        assert torch.allclose(out, out_ref, atol=1e-5)

    def test_different_activations(self):
        """MacheteGatedMLP with different activations."""
        for activation in ["silu", "gelu"]:
            mlp = MacheteGatedMLP(64, 256, 64, activation=activation)
            x = torch.randn(2, 16, 64)
            out = mlp(x)
            assert out.shape == (2, 16, 64)


@pytest.mark.skipif(not is_sm90_available(), reason="SM90+ GPU required")
@pytest.mark.skipif(not HAS_QUACK_MLP, reason="quack MLP not available")
class TestMacheteGatedMLPGPU:
    """Test MacheteGatedMLP on SM90+ GPU."""

    @pytest.mark.parametrize("in_f,hidden_f,out_f", [
        (64, 256, 64),
        (128, 512, 128),
        (256, 1024, 256),
    ])
    def test_forward_matches_pytorch(self, in_f, hidden_f, out_f):
        """MacheteGatedMLP forward matches reference on GPU."""
        torch.manual_seed(42)

        mlp = MacheteGatedMLP(in_f, hidden_f, out_f, activation="silu", bias=False).cuda()
        x = torch.randn(2, 16, in_f, device="cuda")

        out = mlp(x)
        out_ref = gated_mlp_pytorch_ref(
            x, mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight, activation="silu"
        )

        assert torch.allclose(out, out_ref, atol=1e-3, rtol=1e-2)

    def test_backward_gradient(self):
        """MacheteGatedMLP supports autograd backward."""
        torch.manual_seed(42)

        mlp = MacheteGatedMLP(64, 256, 64, bias=False).cuda()
        x = torch.randn(2, 16, 64, device="cuda", requires_grad=True)

        out = mlp(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None


# =============================================================================
# Unit Tests - Patching Functions
# =============================================================================


class LlamaStyleMLP(nn.Module):
    """Simplified LLaMA-style gated MLP for testing patches."""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TestPatchGatedMLP:
    """Test patch_gated_mlp and unpatch_gated_mlp functions."""

    def test_patch_unpatch(self):
        """Patching and unpatching preserves module state."""
        mlp = LlamaStyleMLP(64, 256)

        patch_gated_mlp(mlp)
        if HAS_QUACK_MLP:
            assert hasattr(mlp, "_machete_original_forward")

        unpatch_gated_mlp(mlp)
        assert not hasattr(mlp, "_machete_original_forward")

    @pytest.mark.skipif(not is_sm90_available(), reason="SM90+ GPU required")
    @pytest.mark.skipif(not HAS_QUACK_MLP, reason="quack MLP not available")
    def test_patched_forward_matches(self):
        """Patched MLP produces same results as unpatched."""
        torch.manual_seed(42)
        mlp = LlamaStyleMLP(128, 512).cuda()
        x = torch.randn(2, 16, 128, device="cuda")

        out_original = mlp(x).clone()

        patch_gated_mlp(mlp)
        out_patched = mlp(x)

        assert torch.allclose(out_original, out_patched, atol=1e-3, rtol=1e-2)

    def test_double_patch_noop(self):
        """Patching twice doesn't break the module."""
        mlp = LlamaStyleMLP(64, 256)

        patch_gated_mlp(mlp)
        patch_gated_mlp(mlp)  # Second patch should be no-op

        if HAS_QUACK_MLP:
            assert hasattr(mlp, "_machete_original_forward")

        unpatch_gated_mlp(mlp)
        assert not hasattr(mlp, "_machete_original_forward")

    def test_custom_attribute_names(self):
        """Patching with custom attribute names."""

        class CustomMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.w_gate = nn.Linear(64, 256, bias=False)
                self.w_up = nn.Linear(64, 256, bias=False)
                self.w_down = nn.Linear(256, 64, bias=False)

            def forward(self, x):
                return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

        mlp = CustomMLP()
        x = torch.randn(2, 16, 64)

        out_original = mlp(x).clone()

        patch_gated_mlp(mlp, gate_proj_attr="w_gate", up_proj_attr="w_up", down_proj_attr="w_down")
        out_patched = mlp(x)

        assert torch.allclose(out_original, out_patched, atol=1e-5)


# =============================================================================
# Edge Cases
# =============================================================================


class TestMLPEdgeCases:
    """Test edge cases for MLP modules."""

    def test_unaligned_dimensions_fallback(self):
        """Unaligned dimensions should fall back to PyTorch."""
        mlp = MacheteMLP(65, 257, 65, bias=False)
        x = torch.randn(2, 16, 65)

        out = mlp(x)
        assert out.shape == (2, 16, 65)

    def test_with_bias_fallback(self):
        """MLP with bias should fall back to PyTorch."""
        mlp = MacheteMLP(64, 256, 64, bias=True)
        x = torch.randn(2, 16, 64)

        out = mlp(x)
        assert out.shape == (2, 16, 64)

    def test_single_token(self):
        """MLP with single token input."""
        mlp = MacheteMLP(64, 256, 64)
        x = torch.randn(1, 1, 64)

        out = mlp(x)
        assert out.shape == (1, 1, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
