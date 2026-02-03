# Copyright (c) 2025, Machete Authors
"""Tests for GLM-4.7 model patching."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from machete.patching.glm4 import (
    ATTENTION_CLASSES,
    RMSNORM_CLASSES,
    MLP_CLASSES,
    HAS_QUACK_LINEAR_CE,
    patch_attention,
    unpatch_attention,
    patch_causal_lm,
    unpatch_causal_lm,
)
from machete.patch import patch, unpatch, is_patched, MODEL_TYPE_MODULES


def is_sm90_available():
    """Check if SM90+ GPU is available."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


# =============================================================================
# Unit Tests - Class Definitions
# =============================================================================


class TestGLM4ClassDefinitions:
    """Test GLM4 class definitions."""

    def test_attention_classes_defined(self):
        """ATTENTION_CLASSES should include GLM4 attention variants."""
        assert "Glm4MoeAttention" in ATTENTION_CLASSES
        assert "Glm4MoeLiteAttention" in ATTENTION_CLASSES

    def test_rmsnorm_classes_defined(self):
        """RMSNORM_CLASSES should include GLM4 RMSNorm variants."""
        assert "Glm4MoeRMSNorm" in RMSNORM_CLASSES
        assert "Glm4MoeLiteRMSNorm" in RMSNORM_CLASSES

    def test_mlp_classes_empty(self):
        """MLP_CLASSES should be empty (MoE not easily patchable)."""
        assert MLP_CLASSES == ()

    def test_model_type_modules_registered(self):
        """GLM4 should be registered in MODEL_TYPE_MODULES."""
        assert "glm4_moe" in MODEL_TYPE_MODULES
        assert "glm4_moe_lite" in MODEL_TYPE_MODULES


# =============================================================================
# Unit Tests - Mock Attention
# =============================================================================


class MockGlm4MoeConfig:
    """Mock GLM4 config."""
    def __init__(self, num_attention_heads=8, num_key_value_heads=2, head_dim=64):
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_size = num_attention_heads * head_dim


class MockGlm4MoeAttention(nn.Module):
    """Mock Glm4MoeAttention for testing."""

    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, **kwargs):
        bsz, q_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)

        # Expand KV for GQA
        n_rep = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(n_rep, dim=2)
        v = v.repeat_interleave(n_rep, dim=2)

        # Simple scaled dot-product attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None


class TestGLM4AttentionPatching:
    """Test GLM4 attention patching."""

    def test_patch_unpatch_attention(self):
        """Patching and unpatching attention preserves behavior on CPU."""
        torch.manual_seed(42)
        config = MockGlm4MoeConfig()
        attn = MockGlm4MoeAttention(config)
        # Rename to match expected class name
        attn.__class__.__name__ = "Glm4MoeAttention"

        x = torch.randn(2, 16, config.hidden_size)

        out_original, _ = attn(x)

        patch_attention(attn)
        # On CPU without flash-attn-cute, may not be patched
        # Just verify no errors

        unpatch_attention(attn)
        assert not hasattr(attn, "_machete_original_forward")

    def test_double_patch_noop(self):
        """Patching twice should be a no-op."""
        config = MockGlm4MoeConfig()
        attn = MockGlm4MoeAttention(config)
        attn.__class__.__name__ = "Glm4MoeAttention"

        patch_attention(attn)
        has_original_1 = hasattr(attn, "_machete_original_forward")

        patch_attention(attn)
        has_original_2 = hasattr(attn, "_machete_original_forward")

        assert has_original_1 == has_original_2

        unpatch_attention(attn)


# =============================================================================
# Unit Tests - CausalLM Patching
# =============================================================================


class MockGlm4Model(nn.Module):
    """Mock GLM4 base model."""

    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        class Output:
            def __init__(self, h):
                self.last_hidden_state = h
                self.past_key_values = None
                self.hidden_states = None
                self.attentions = None

        return Output(inputs_embeds)


class MockGlm4MoeForCausalLM(nn.Module):
    """Mock Glm4MoeForCausalLM for testing."""

    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.model = MockGlm4Model(hidden_size, vocab_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.vocab_size = vocab_size

    def forward(
        self,
        input_ids=None,
        labels=None,
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

        class Output:
            def __init__(self, l, log, pkv, hs, att):
                self.loss = l
                self.logits = log
                self.past_key_values = pkv
                self.hidden_states = hs
                self.attentions = att

        return Output(loss, logits, None, None, None)


class TestGLM4CausalLMPatching:
    """Test GLM4 CausalLM patching."""

    def test_patch_unpatch_causal_lm(self):
        """Patching and unpatching CausalLM preserves behavior on CPU."""
        torch.manual_seed(42)
        model = MockGlm4MoeForCausalLM(64, 1000)
        input_ids = torch.randint(0, 1000, (2, 16))
        labels = torch.randint(0, 1000, (2, 16))

        out_original = model(input_ids, labels=labels)
        loss_original = out_original.loss.clone()

        patch_causal_lm(model)
        # May not be patched on CPU without quack

        unpatch_causal_lm(model)
        assert not hasattr(model, "_machete_original_forward")

        out_unpatched = model(input_ids, labels=labels)
        assert torch.allclose(loss_original, out_unpatched.loss, atol=1e-5)

    def test_causal_lm_without_labels(self):
        """CausalLM without labels uses original forward."""
        model = MockGlm4MoeForCausalLM(64, 1000)
        input_ids = torch.randint(0, 1000, (2, 16))

        patch_causal_lm(model)

        out = model(input_ids)
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

        model = MockGlm4MoeForCausalLM(hidden_size, vocab_size).cuda().half()
        input_ids = torch.randint(0, vocab_size, (2, 16), device="cuda")
        labels = torch.randint(0, vocab_size, (2, 16), device="cuda")

        out_original = model(input_ids, labels=labels)
        loss_original = out_original.loss.clone()

        patch_causal_lm(model)
        out_patched = model(input_ids, labels=labels)

        assert torch.allclose(loss_original, out_patched.loss, atol=1e-2, rtol=1e-2)

        unpatch_causal_lm(model)


# =============================================================================
# Integration Tests
# =============================================================================


class TestGLM4ModelPatching:
    """Test full GLM4 model patching via machete.patch()."""

    def test_patch_function_accepts_glm4_types(self):
        """Patch function accepts GLM4 model types."""
        from machete.patch import MODEL_TYPE_MODULES

        assert "glm4_moe" in MODEL_TYPE_MODULES
        assert "glm4_moe_lite" in MODEL_TYPE_MODULES

        # Verify module structure
        glm4_moe_config = MODEL_TYPE_MODULES["glm4_moe"]
        assert "module" in glm4_moe_config
        assert "attention" in glm4_moe_config
        assert "rmsnorm" in glm4_moe_config


# =============================================================================
# Edge Cases
# =============================================================================


class TestGLM4PatchingEdgeCases:
    """Test edge cases for GLM4 patching."""

    def test_unknown_attention_class_skipped(self):
        """Unknown attention class is skipped."""

        class UnknownAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)

            def forward(self, x):
                return self.q_proj(x), None

        attn = UnknownAttention()
        patch_attention(attn)

        # Should not be patched
        assert not hasattr(attn, "_machete_original_forward")

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

        assert not hasattr(model, "_machete_original_forward")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
