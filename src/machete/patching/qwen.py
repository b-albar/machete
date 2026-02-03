# Copyright (c) 2025, Machete Authors
"""Qwen model patching using flash-attn-cute and quack."""

import types
from typing import Optional, Tuple, Dict
import torch
import torch.nn.functional as F
import transformers
from machete.patching.llama import flash_attn_func

# Module class names for Qwen models
ATTENTION_CLASSES = ("Qwen2Attention", "Qwen2SdpaAttention", "Qwen2FlashAttention2", "Qwen3Attention")
RMSNORM_CLASSES = ("Qwen2RMSNorm", "Qwen3RMSNorm")
MLP_CLASSES = ("Qwen2MLP", "Qwen3MLP")

# Import MLP patching utilities
try:
    from machete.patching.ops.mlp import patch_gated_mlp, unpatch_gated_mlp, HAS_QUACK_MLP
except ImportError:
    HAS_QUACK_MLP = False
    patch_gated_mlp = None
    unpatch_gated_mlp = None

# Import linear cross entropy utilities
try:
    from machete.patching.ops.linear_cross_entropy import (
        fused_linear_cross_entropy,
        HAS_QUACK_LINEAR_CE,
    )
except ImportError:
    HAS_QUACK_LINEAR_CE = False
    fused_linear_cross_entropy = None


def make_qwen_attention_forward():
    """Create optimized forward for QwenAttention (supports Qwen2 and Qwen3)."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[object] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.shape

        # Qwen3Attention uses config for num_heads check, Qwen2 might have attributes
        if hasattr(self, "num_heads"):
            num_heads = self.num_heads
            num_kv_heads = self.num_key_value_heads
        else:
            num_heads = self.config.num_attention_heads
            num_kv_heads = self.config.num_key_value_heads

        head_dim = self.head_dim

        # 1. Projections and Reshape
        # Qwen2/3 applies norm on (bsz, seqlen, num_heads, head_dim)
        query_states = self.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim)

        # Apply QK Norm if available
        if hasattr(self, "q_norm") and self.q_norm is not None:
            query_states = self.q_norm(query_states)
        if hasattr(self, "k_norm") and self.k_norm is not None:
            key_states = self.k_norm(key_states)

        # 2. RoPE
        # Transpose to (bsz, num_heads, seqlen, head_dim) for apply_rotary_pos_emb
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Get apply_rotary_pos_emb. Try qwen3, then qwen2, then fallback
        # Ideally we'd import this at top level but it depends on transformers version
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

        # position_embeddings passed as (cos, sin) usually
        # Qwen3 signature: position_embeddings: tuple[torch.Tensor, torch.Tensor]
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # 3. KV Cache
        if past_key_values is not None:
            # Qwen3 uses cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            # We need to adapt based on what update expects.
            # Assuming standard DynamicCache/Cache interface

            # Note: Qwen3 passes sin/cos to update!?
            cache_kwargs = {}
            if cache_position is not None:
                cache_kwargs["cache_position"] = cache_position
            if position_embeddings is not None:
                cos, sin = position_embeddings
                cache_kwargs["sin"] = sin
                cache_kwargs["cos"] = cos

            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # 4. Flash Attention using flash_attn_func
        # flash_attn_func expects (bsz, seqlen, num_heads, head_dim)
        query_states = query_states.transpose(1, 2).contiguous()
        key_states = key_states.transpose(1, 2).contiguous()
        value_states = value_states.transpose(1, 2).contiguous()

        # Qwen3 logic for causal: "input_shape = hidden_states.shape[:-1]; is_causal = True"
        # We can pass causal=True for generation/prefill usually

        attn_output = flash_attn_func(query_states, key_states, value_states, causal=True)
        # flash_attn_func returns (output, lse, ...), we only need the output
        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]

        # 5. Output Projection
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None

    return forward


def patch_attention(module):
    """Patch Qwen attention module with optimized forward.

    No-op if flash-attn-cute is not available.
    """
    if flash_attn_func is None:
        return  # flash-attn-cute not available, skip patching

    if hasattr(module, "_machete_original_forward"):
        return  # Already patched

    module._machete_original_forward = module.forward
    module.forward = types.MethodType(make_qwen_attention_forward(), module)


def unpatch_attention(module):
    """Restore original Qwen attention forward."""
    if hasattr(module, "_machete_original_forward"):
        module.forward = module._machete_original_forward
        del module._machete_original_forward


def make_qwen_mlp_forward():
    """Create optimized forward for Qwen MLP (SwiGLU-style).

    Qwen uses a gated MLP with gate_proj, up_proj, down_proj:
        output = down_proj(act(gate_proj(x)) * up_proj(x))
    """
    try:
        from quack import linear_act, linear
    except ImportError:
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check if we can use quack kernels
        use_quack = (
            HAS_QUACK_MLP
            and x.is_cuda
            and x.dtype in (torch.float16, torch.bfloat16)
            and x.shape[-1] % 8 == 0
            and self.gate_proj.weight.shape[0] % 8 == 0
            and not self.gate_proj.bias
        )

        if use_quack:
            # Fused gate projection with SiLU activation
            gate = linear_act(x, self.gate_proj.weight, activation="silu")
            # Up projection
            up = linear(x, self.up_proj.weight)
            # Element-wise multiply and down projection
            return linear(gate * up, self.down_proj.weight)
        else:
            # Fallback to original forward
            return self._machete_original_forward(x)

    return forward


def patch_mlp(module):
    """Patch Qwen MLP module with optimized forward.

    No-op if quack is not available or already patched.
    """
    if not HAS_QUACK_MLP:
        return  # quack not available, skip patching

    if hasattr(module, "_machete_original_forward"):
        return  # Already patched

    # Verify this is a Qwen-style gated MLP
    if not all(hasattr(module, attr) for attr in ("gate_proj", "up_proj", "down_proj")):
        return  # Not a gated MLP

    forward_fn = make_qwen_mlp_forward()
    if forward_fn is None:
        return

    module._machete_original_forward = module.forward
    module.forward = types.MethodType(forward_fn, module)


def unpatch_mlp(module):
    """Restore original Qwen MLP forward."""
    if hasattr(module, "_machete_original_forward"):
        module.forward = module._machete_original_forward
        del module._machete_original_forward


def make_qwen_causal_lm_forward():
    """Create optimized forward for QwenForCausalLM using fused linear CE.

    This patches the model's forward to use fused linear + cross entropy
    for computing the language modeling loss, avoiding the full logits
    materialization.
    """

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[object] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        # If no labels, use original forward (no loss computation)
        if labels is None:
            return self._machete_original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                **kwargs,
            )

        # Get model outputs (without loss)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        # Check if we can use fused linear cross entropy
        use_fused = (
            HAS_QUACK_LINEAR_CE
            and hidden_states.is_cuda
            and hidden_states.dtype in (torch.float16, torch.bfloat16)
            and self.lm_head.weight.shape[0] % 8 == 0
            and self.lm_head.weight.shape[1] % 8 == 0
            and not getattr(self.lm_head, "bias", None)
        )

        if use_fused:
            # Shift labels for next token prediction
            shift_hidden = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Use fused linear cross entropy
            loss = fused_linear_cross_entropy(
                shift_hidden,
                self.lm_head.weight,
                shift_labels,
                ignore_index=-100,
                reduction="mean",
            )

            # Still compute logits for output (optional, can be skipped)
            logits = self.lm_head(hidden_states)
        else:
            # Fallback to standard computation
            logits = self.lm_head(hidden_states)

            # Shift for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # Return in expected format
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        from transformers.modeling_outputs import CausalLMOutputWithPast

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    return forward


def patch_causal_lm(model):
    """Patch QwenForCausalLM with fused linear cross entropy.

    Only patches if quack linear_cross_entropy is available.
    """
    if not HAS_QUACK_LINEAR_CE:
        return  # Not available, skip

    if hasattr(model, "_machete_original_forward"):
        return  # Already patched

    # Verify this is a CausalLM model
    if not hasattr(model, "lm_head") or not hasattr(model, "model"):
        return

    model._machete_original_forward = model.forward
    model.forward = types.MethodType(make_qwen_causal_lm_forward(), model)


def unpatch_causal_lm(model):
    """Restore original CausalLM forward."""
    if hasattr(model, "_machete_original_forward"):
        model.forward = model._machete_original_forward
        del model._machete_original_forward
