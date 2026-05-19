# Copyright (c) 2025, Machete Authors
"""Llama patch/unpatch coverage."""

import pytest
import torch


def _make_small_llama():
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=1000,
    )
    return LlamaForCausalLM(config)


def test_patch_unpatch_llama():
    """Patching records original attention forwards and unpatch removes them."""
    import machete

    model = _make_small_llama()

    assert not machete.is_patched(model)
    machete.patch(model)
    assert machete.is_patched(model)

    for module in model.modules():
        if "LlamaAttention" in module.__class__.__name__:
            assert hasattr(module, "_machete_original_forward")

    machete.unpatch(model)
    assert not machete.is_patched(model)

    for module in model.modules():
        assert not hasattr(module, "_machete_original_forward")


def test_patched_llama_forward_runs():
    """Patched Llama keeps the standard forward path usable."""
    import machete

    model = _make_small_llama()
    if torch.cuda.is_available():
        model = model.cuda().half()

    machete.patch(model)

    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    try:
        with torch.no_grad():
            outputs = model(input_ids)
    except Exception as exc:
        if "OpError" in type(exc).__name__ or "arch" in str(exc).lower():
            pytest.skip(f"flash-attn-cute unsupported on this GPU arch: {exc}")
        raise

    assert outputs.logits.shape == (batch_size, seq_len, 1000)


def test_lora_compatibility():
    """Patched Llama remains compatible with PEFT LoRA wrappers."""
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model
    import machete

    model = _make_small_llama()
    machete.patch(model)

    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
        ),
    )
    if torch.cuda.is_available():
        model = model.cuda()

    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    with torch.no_grad():
        outputs = model(input_ids)

    assert outputs.logits.shape == (batch_size, seq_len, 1000)
