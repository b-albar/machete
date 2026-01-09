# Copyright (c) 2025, Machete Authors
"""Tests for Machete patching system."""

import pytest
import torch


def test_patch_unpatch_llama():
    """Test that patching and unpatching works correctly."""
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM
    import machete

    # Create a small model for testing
    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=1000,
    )
    model = LlamaForCausalLM(config)

    # Test patch
    assert not machete.is_patched(model)
    machete.patch(model)
    assert machete.is_patched(model)

    # Verify modules have _machete_original_forward
    for name, module in model.named_modules():
        if "LlamaAttention" in module.__class__.__name__:
            assert hasattr(module, "_machete_original_forward")

    # Test unpatch
    machete.unpatch(model)
    assert not machete.is_patched(model)

    # Verify _machete_original_forward is removed
    for name, module in model.named_modules():
        assert not hasattr(module, "_machete_original_forward")


def test_patched_model_runs():
    """Test that a patched model can run forward pass."""
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM
    import machete

    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=1000,
    )
    model = LlamaForCausalLM(config)

    if torch.cuda.is_available():
        model = model.cuda()

    machete.patch(model)

    # Create dummy input
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    # Forward pass should work
    with torch.no_grad():
        outputs = model(input_ids)

    assert outputs.logits.shape == (batch_size, seq_len, 1000)


def test_lora_compatibility():
    """Test that patched model works with LoRA."""
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from transformers import LlamaConfig, LlamaForCausalLM
    from peft import get_peft_model, LoraConfig
    import machete

    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=1000,
    )
    model = LlamaForCausalLM(config)

    # Patch first
    machete.patch(model)

    # Apply LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
    )
    model = get_peft_model(model, lora_config)

    if torch.cuda.is_available():
        model = model.cuda()

    # Forward pass should work with LoRA
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    with torch.no_grad():
        outputs = model(input_ids)

    assert outputs.logits.shape == (batch_size, seq_len, 1000)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
