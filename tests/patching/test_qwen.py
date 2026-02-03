# Copyright (c) 2025, Machete Authors
"""Tests for Qwen model patching."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from machete.patching.qwen import (
    ATTENTION_CLASSES,
    RMSNORM_CLASSES,
    MLP_CLASSES,
    HAS_QUACK_MLP,
    HAS_QUACK_LINEAR_CE,
    patch_attention,
    unpatch_attention,
    patch_mlp,
    unpatch_mlp,
    patch_causal_lm,
    unpatch_causal_lm,
)
from machete.patch import patch, unpatch, is_patched


def is_sm90_available():
    """Check if SM90+ GPU is available."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


# =============================================================================
# Unit Tests - MLP Patching
# =============================================================================


class MockQwenMLP(nn.Module):
    """Mock Qwen MLP for testing."""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TestQwenMLPPatching:
    """Test Qwen MLP patching."""

    def test_mlp_classes_defined(self):
        """MLP_CLASSES should include Qwen2MLP and Qwen3MLP."""
        assert "Qwen2MLP" in MLP_CLASSES
        assert "Qwen3MLP" in MLP_CLASSES

    def test_patch_unpatch_mlp(self):
        """Patching and unpatching MLP preserves behavior on CPU."""
        torch.manual_seed(42)
        mlp = MockQwenMLP(64, 256)
        x = torch.randn(2, 16, 64)

        # Original output
        out_original = mlp(x).clone()

        # Patch (will be no-op on CPU without quack)
        patch_mlp(mlp)

        # Patched output should match
        out_patched = mlp(x)
        assert torch.allclose(out_original, out_patched, atol=1e-5)

        # Unpatch
        unpatch_mlp(mlp)

        # Unpatched output should match
        out_unpatched = mlp(x)
        assert torch.allclose(out_original, out_unpatched, atol=1e-5)

    def test_double_patch_noop(self):
        """Patching twice should be a no-op."""
        mlp = MockQwenMLP(64, 256)

        patch_mlp(mlp)
        has_original_1 = hasattr(mlp, "_machete_original_forward")

        patch_mlp(mlp)  # Second patch
        has_original_2 = hasattr(mlp, "_machete_original_forward")

        assert has_original_1 == has_original_2

        unpatch_mlp(mlp)

    @pytest.mark.skipif(not is_sm90_available(), reason="SM90+ GPU required")
    @pytest.mark.skipif(not HAS_QUACK_MLP, reason="quack MLP not available")
    def test_patched_mlp_gpu(self):
        """Patched MLP works on GPU."""
        torch.manual_seed(42)
        mlp = MockQwenMLP(128, 512).cuda()
        x = torch.randn(2, 16, 128, device="cuda", dtype=torch.float16)

        out_original = mlp(x).clone()

        patch_mlp(mlp)
        out_patched = mlp(x)

        # Allow some tolerance for optimized kernel
        assert torch.allclose(out_original, out_patched, atol=1e-3, rtol=1e-2)

        unpatch_mlp(mlp)


# =============================================================================
# Unit Tests - Attention Patching
# =============================================================================


class TestQwenAttentionPatching:
    """Test Qwen attention patching."""

    def test_attention_classes_defined(self):
        """ATTENTION_CLASSES should include Qwen2 and Qwen3 attention."""
        assert "Qwen2Attention" in ATTENTION_CLASSES
        assert "Qwen3Attention" in ATTENTION_CLASSES

    def test_rmsnorm_classes_defined(self):
        """RMSNORM_CLASSES should include Qwen2 and Qwen3 RMSNorm."""
        assert "Qwen2RMSNorm" in RMSNORM_CLASSES
        assert "Qwen3RMSNorm" in RMSNORM_CLASSES


# =============================================================================
# Unit Tests - CausalLM Patching
# =============================================================================


class MockQwenModel(nn.Module):
    """Mock Qwen base model."""

    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Simple mock: return embeddings as hidden states
        class Output:
            def __init__(self, h):
                self.last_hidden_state = h
                self.past_key_values = None
                self.hidden_states = None
                self.attentions = None

        return Output(inputs_embeds)


class MockQwenForCausalLM(nn.Module):
    """Mock QwenForCausalLM for testing."""

    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.model = MockQwenModel(hidden_size, vocab_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.vocab_size = vocab_size

    def forward(
        self,
        input_ids=None,
        labels=None,
        return_dict=True,
        **kwargs,
    ):
        outputs = self.model(input_ids=input_ids, **kwargs)
        hidden_states = outputs.last_hidden_state

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)

        class Output:
            def __init__(self, l, log, pkv, hs, att):
                self.loss = l
                self.logits = log
                self.past_key_values = pkv
                self.hidden_states = hs
                self.attentions = att

        return Output(loss, logits, None, None, None)


class TestQwenCausalLMPatching:
    """Test Qwen CausalLM patching."""

    def test_patch_unpatch_causal_lm(self):
        """Patching and unpatching CausalLM preserves behavior on CPU."""
        torch.manual_seed(42)
        model = MockQwenForCausalLM(64, 1000)
        input_ids = torch.randint(0, 1000, (2, 16))
        labels = torch.randint(0, 1000, (2, 16))

        # Original loss
        out_original = model(input_ids, labels=labels, return_dict=True)
        loss_original = out_original.loss.clone()

        # Patch (will use fallback on CPU)
        patch_causal_lm(model)
        assert hasattr(model, "_machete_original_forward") or not HAS_QUACK_LINEAR_CE

        # Patched output should match (falls back on CPU)
        out_patched = model(input_ids, labels=labels, return_dict=True)
        assert torch.allclose(loss_original, out_patched.loss, atol=1e-5)

        # Unpatch
        unpatch_causal_lm(model)
        assert not hasattr(model, "_machete_original_forward")

        # Unpatched output should match
        out_unpatched = model(input_ids, labels=labels, return_dict=True)
        assert torch.allclose(loss_original, out_unpatched.loss, atol=1e-5)

    def test_causal_lm_without_labels(self):
        """CausalLM without labels uses original forward."""
        model = MockQwenForCausalLM(64, 1000)
        input_ids = torch.randint(0, 1000, (2, 16))

        patch_causal_lm(model)

        # Without labels, should use original forward
        out = model(input_ids, return_dict=True)
        assert out.loss is None
        assert out.logits is not None

        unpatch_causal_lm(model)

    @pytest.mark.skipif(not is_sm90_available(), reason="SM90+ GPU required")
    @pytest.mark.skipif(not HAS_QUACK_LINEAR_CE, reason="quack linear CE not available")
    def test_patched_causal_lm_gpu(self):
        """Patched CausalLM uses fused kernel on GPU."""
        torch.manual_seed(42)
        hidden_size = 128
        vocab_size = 4096  # Must be divisible by 8

        model = MockQwenForCausalLM(hidden_size, vocab_size).cuda()
        input_ids = torch.randint(0, vocab_size, (2, 16), device="cuda")
        labels = torch.randint(0, vocab_size, (2, 16), device="cuda")

        # Need to use float16/bfloat16 for fused kernel
        model = model.half()

        out_original = model(input_ids, labels=labels, return_dict=True)
        loss_original = out_original.loss.clone()

        patch_causal_lm(model)
        out_patched = model(input_ids, labels=labels, return_dict=True)

        # Allow some tolerance for fused kernel
        assert torch.allclose(loss_original, out_patched.loss, atol=1e-2, rtol=1e-2)

        unpatch_causal_lm(model)


# =============================================================================
# Integration Tests - Full Model Patching
# =============================================================================


class TestQwenModelPatching:
    """Test full Qwen model patching via machete.patch()."""

    def test_patch_function_signature(self):
        """Patch function accepts all new parameters."""
        from machete.patch import patch

        import inspect
        sig = inspect.signature(patch)
        params = list(sig.parameters.keys())

        assert "patch_attention" in params
        assert "patch_rmsnorm" in params
        assert "patch_mlp" in params
        assert "patch_cross_entropy" in params
        assert "patch_fused_lm_head" in params


# =============================================================================
# Edge Cases
# =============================================================================


class TestQwenPatchingEdgeCases:
    """Test edge cases for Qwen patching."""

    def test_mlp_without_required_attrs_skipped(self):
        """MLP without gate/up/down proj attrs is skipped."""

        class IncompleteMLPModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(64, 256)
                self.fc2 = nn.Linear(256, 64)

            def forward(self, x):
                return self.fc2(F.silu(self.fc1(x)))

        mlp = IncompleteMLPModule()
        patch_mlp(mlp)

        # Should not be patched
        assert not hasattr(mlp, "_machete_original_forward")

    def test_causal_lm_without_lm_head_skipped(self):
        """CausalLM without lm_head is skipped."""

        class IncompleteCausalLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Linear(64, 64)

            def forward(self, x):
                return self.model(x)

        model = IncompleteCausalLM()
        patch_causal_lm(model)

        # Should not be patched
        assert not hasattr(model, "_machete_original_forward")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
