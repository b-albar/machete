# Copyright (c) 2025, Machete Authors
"""Tests for MacheteLinearCrossEntropy fused layer."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from machete.patching.ops.linear_cross_entropy import (
    MacheteLinearCrossEntropy,
    fused_linear_cross_entropy,
    HAS_QUACK_LINEAR_CE,
)


def is_sm90_available():
    """Check if SM90+ GPU is available."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


def linear_cross_entropy_ref(x, weight, target, ignore_index=-100, reduction="mean"):
    """PyTorch reference for linear + cross entropy."""
    logits = F.linear(x, weight)
    return F.cross_entropy(logits, target, ignore_index=ignore_index, reduction=reduction)


# =============================================================================
# Unit Tests - MacheteLinearCrossEntropy Module
# =============================================================================


class TestMacheteLinearCrossEntropyCPU:
    """Test MacheteLinearCrossEntropy on CPU (always falls back to PyTorch)."""

    def test_forward_matches_pytorch(self):
        """MacheteLinearCrossEntropy forward matches reference on CPU."""
        torch.manual_seed(42)
        in_features, vocab_size = 64, 1000
        batch_size, seq_len = 2, 16

        lce = MacheteLinearCrossEntropy(in_features, vocab_size, bias=False)
        x = torch.randn(batch_size, seq_len, in_features)
        target = torch.randint(0, vocab_size, (batch_size, seq_len))

        loss = lce(x, target)
        loss_ref = linear_cross_entropy_ref(
            x.view(-1, in_features), lce.weight, target.view(-1)
        )

        assert torch.allclose(loss, loss_ref, atol=1e-5)

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_different_reductions(self, reduction):
        """MacheteLinearCrossEntropy with different reduction types."""
        torch.manual_seed(42)
        in_features, vocab_size = 64, 1000

        lce = MacheteLinearCrossEntropy(in_features, vocab_size, reduction=reduction)
        x = torch.randn(2, 16, in_features)
        target = torch.randint(0, vocab_size, (2, 16))

        loss = lce(x, target)
        loss_ref = linear_cross_entropy_ref(
            x.view(-1, in_features), lce.weight, target.view(-1), reduction=reduction
        )

        assert torch.allclose(loss, loss_ref, atol=1e-5)

    def test_ignore_index(self):
        """MacheteLinearCrossEntropy with ignore_index."""
        torch.manual_seed(42)
        in_features, vocab_size = 64, 1000
        ignore_index = -100

        lce = MacheteLinearCrossEntropy(
            in_features, vocab_size, ignore_index=ignore_index
        )
        x = torch.randn(2, 16, in_features)
        target = torch.randint(0, vocab_size, (2, 16))
        # Set some targets to ignore_index
        target[0, :4] = ignore_index

        loss = lce(x, target)
        loss_ref = linear_cross_entropy_ref(
            x.view(-1, in_features),
            lce.weight,
            target.view(-1),
            ignore_index=ignore_index,
        )

        assert torch.allclose(loss, loss_ref, atol=1e-5)


@pytest.mark.skipif(not is_sm90_available(), reason="SM90+ GPU required")
@pytest.mark.skipif(not HAS_QUACK_LINEAR_CE, reason="quack linear CE not available")
class TestMacheteLinearCrossEntropyGPU:
    """Test MacheteLinearCrossEntropy on SM90+ GPU."""

    @pytest.mark.parametrize("in_f,vocab_size", [
        (64, 1024),
        (128, 4096),
        (256, 8192),
    ])
    def test_forward_matches_pytorch(self, in_f, vocab_size):
        """MacheteLinearCrossEntropy forward matches reference on GPU."""
        torch.manual_seed(42)
        batch_size, seq_len = 2, 16

        lce = MacheteLinearCrossEntropy(
            in_f, vocab_size, bias=False, chunk_size=4096
        ).cuda()
        x = torch.randn(batch_size, seq_len, in_f, device="cuda")
        target = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")

        loss = lce(x, target)
        loss_ref = linear_cross_entropy_ref(
            x.view(-1, in_f), lce.weight, target.view(-1)
        )

        # Allow some tolerance for fused kernel
        assert torch.allclose(loss, loss_ref, atol=1e-3, rtol=1e-2)

    def test_backward_gradient(self):
        """MacheteLinearCrossEntropy supports autograd backward."""
        torch.manual_seed(42)
        in_features, vocab_size = 128, 4096

        lce = MacheteLinearCrossEntropy(
            in_features, vocab_size, bias=False, chunk_size=4096
        ).cuda()
        x = torch.randn(2, 16, in_features, device="cuda", requires_grad=True)
        target = torch.randint(0, vocab_size, (2, 16), device="cuda")

        loss = lce(x, target)
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_large_vocabulary(self):
        """MacheteLinearCrossEntropy with large vocabulary (typical LLM)."""
        torch.manual_seed(42)
        in_features = 256
        vocab_size = 32000  # Typical LLM vocab

        # Use small batch to fit in memory
        lce = MacheteLinearCrossEntropy(
            in_features, vocab_size, bias=False, chunk_size=4096
        ).cuda()
        x = torch.randn(1, 8, in_features, device="cuda")
        target = torch.randint(0, vocab_size, (1, 8), device="cuda")

        loss = lce(x, target)
        assert loss.dim() == 0  # Scalar loss
        assert not torch.isnan(loss)


# =============================================================================
# Unit Tests - Functional API
# =============================================================================


class TestFusedLinearCrossEntropyFunctional:
    """Test fused_linear_cross_entropy functional API."""

    def test_functional_cpu(self):
        """fused_linear_cross_entropy works on CPU."""
        torch.manual_seed(42)
        in_features, vocab_size = 64, 1000

        x = torch.randn(32, in_features)
        weight = torch.randn(vocab_size, in_features)
        target = torch.randint(0, vocab_size, (32,))

        loss = fused_linear_cross_entropy(x, weight, target)
        loss_ref = linear_cross_entropy_ref(x, weight, target)

        assert torch.allclose(loss, loss_ref, atol=1e-5)

    @pytest.mark.skipif(not is_sm90_available(), reason="SM90+ GPU required")
    @pytest.mark.skipif(not HAS_QUACK_LINEAR_CE, reason="quack linear CE not available")
    def test_functional_gpu(self):
        """fused_linear_cross_entropy uses quack on GPU."""
        torch.manual_seed(42)
        in_features, vocab_size = 128, 4096

        x = torch.randn(32, in_features, device="cuda")
        weight = torch.randn(vocab_size, in_features, device="cuda")
        target = torch.randint(0, vocab_size, (32,), device="cuda")

        loss = fused_linear_cross_entropy(x, weight, target, chunk_size=4096)
        loss_ref = linear_cross_entropy_ref(x, weight, target)

        assert torch.allclose(loss, loss_ref, atol=1e-3, rtol=1e-2)

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_functional_reductions(self, reduction):
        """fused_linear_cross_entropy with different reductions."""
        torch.manual_seed(42)
        in_features, vocab_size = 64, 1000

        x = torch.randn(32, in_features)
        weight = torch.randn(vocab_size, in_features)
        target = torch.randint(0, vocab_size, (32,))

        loss = fused_linear_cross_entropy(x, weight, target, reduction=reduction)
        loss_ref = linear_cross_entropy_ref(x, weight, target, reduction=reduction)

        assert torch.allclose(loss, loss_ref, atol=1e-5)


# =============================================================================
# Edge Cases
# =============================================================================


class TestLinearCrossEntropyEdgeCases:
    """Test edge cases for linear cross entropy."""

    def test_unaligned_dimensions_fallback(self):
        """Unaligned dimensions should fall back to PyTorch."""
        in_features, vocab_size = 65, 1001  # Not divisible by 8

        lce = MacheteLinearCrossEntropy(in_features, vocab_size)
        x = torch.randn(2, 16, in_features)
        target = torch.randint(0, vocab_size, (2, 16))

        loss = lce(x, target)
        assert loss.dim() == 0

    def test_single_token(self):
        """Linear cross entropy with single token."""
        lce = MacheteLinearCrossEntropy(64, 1000)
        x = torch.randn(1, 1, 64)
        target = torch.randint(0, 1000, (1, 1))

        loss = lce(x, target)
        assert loss.dim() == 0

    def test_all_ignored(self):
        """Linear cross entropy with all tokens ignored."""
        ignore_index = -100
        lce = MacheteLinearCrossEntropy(64, 1000, ignore_index=ignore_index)
        x = torch.randn(2, 16, 64)
        target = torch.full((2, 16), ignore_index, dtype=torch.long)

        loss = lce(x, target)
        # With mean reduction and all ignored, loss should be 0 or nan
        # depending on implementation
        assert loss.dim() == 0

    def test_with_bias(self):
        """Linear cross entropy with bias."""
        lce = MacheteLinearCrossEntropy(64, 1000, bias=True)
        x = torch.randn(2, 16, 64)
        target = torch.randint(0, 1000, (2, 16))

        loss = lce(x, target)
        assert loss.dim() == 0

    @pytest.mark.skipif(not is_sm90_available(), reason="SM90+ GPU required")
    @pytest.mark.skipif(not HAS_QUACK_LINEAR_CE, reason="quack linear CE not available")
    def test_different_chunk_sizes(self):
        """Linear cross entropy with different chunk sizes."""
        torch.manual_seed(42)
        in_features, vocab_size = 128, 4096

        x = torch.randn(4, 32, in_features, device="cuda")
        target = torch.randint(0, vocab_size, (4, 32), device="cuda")

        for chunk_size in [1024, 2048, 4096]:
            lce = MacheteLinearCrossEntropy(
                in_features, vocab_size, chunk_size=chunk_size, bias=False
            ).cuda()
            loss = lce(x, target)
            assert loss.dim() == 0
            assert not torch.isnan(loss)


# =============================================================================
# Integration Tests
# =============================================================================


class TestLinearCrossEntropyIntegration:
    """Integration tests for linear cross entropy in LM training context."""

    def test_lm_head_replacement(self):
        """Test as drop-in replacement for LM head + loss."""
        torch.manual_seed(42)
        hidden_size, vocab_size = 128, 1000
        batch_size, seq_len = 2, 16

        # Standard approach
        lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        loss_fn = nn.CrossEntropyLoss()

        # Machete approach
        machete_lm = MacheteLinearCrossEntropy(hidden_size, vocab_size, bias=False)
        machete_lm.weight.data.copy_(lm_head.weight.data)

        x = torch.randn(batch_size, seq_len, hidden_size)
        target = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Standard loss
        logits = lm_head(x)
        loss_std = loss_fn(logits.view(-1, vocab_size), target.view(-1))

        # Machete loss
        loss_machete = machete_lm(x, target)

        assert torch.allclose(loss_std, loss_machete, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
