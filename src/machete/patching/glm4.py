# Copyright (c) 2025, Machete Authors
"""GLM-4.7 model patching using flash-attn-cute and quack.

Supports both GLM-4.7 (glm4_moe) and GLM-4.7-Flash (glm4_moe_lite).
"""

import types
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

from machete.patching.llama import flash_attn_func

# Module class names for GLM4 models
# GLM-4.7 (glm4_moe)
ATTENTION_CLASSES = (
    "Glm4MoeAttention",
    "Glm4MoeLiteAttention",
)
RMSNORM_CLASSES = (
    "Glm4MoeRMSNorm",
    "Glm4MoeLiteRMSNorm",
)
# MoE MLPs - not easily patchable due to routing complexity
MLP_CLASSES = ()

# Import linear cross entropy utilities
try:
    from machete.patching.ops.linear_cross_entropy import (
        fused_linear_cross_entropy,
        HAS_QUACK_LINEAR_CE,
    )
except ImportError:
    HAS_QUACK_LINEAR_CE = False
    fused_linear_cross_entropy = None


def make_glm4_moe_attention_forward():
    """Create optimized forward for Glm4MoeAttention using flash-attn-cute.

    GLM-4.7 uses standard attention with:
    - q_proj, k_proj, v_proj, o_proj projections
    - GQA with 96 attention heads and 8 KV heads
    - Optional QK normalization (q_norm, k_norm)
    - Partial rotary embeddings (partial_rotary_factor)
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[object] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.shape

        # Get head configuration
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.head_dim

        # 1. Projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to (bsz, seq_len, num_heads, head_dim)
        query_states = query_states.view(bsz, q_len, num_heads, head_dim)
        key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim)
        value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim)

        # Apply QK normalization if available
        if hasattr(self, "q_norm") and self.q_norm is not None:
            query_states = self.q_norm(query_states)
        if hasattr(self, "k_norm") and self.k_norm is not None:
            key_states = self.k_norm(key_states)

        # 2. RoPE - transpose to (bsz, num_heads, seq_len, head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            # Apply rotary embeddings
            from transformers.models.glm4_moe.modeling_glm4_moe import apply_rotary_pos_emb
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # 3. KV Cache
        if past_key_value is not None:
            cache_kwargs = {}
            if cache_position is not None:
                cache_kwargs["cache_position"] = cache_position
            if position_embeddings is not None:
                cos, sin = position_embeddings
                cache_kwargs["sin"] = sin
                cache_kwargs["cos"] = cos
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # 4. Flash Attention
        # Transpose to (bsz, seq_len, num_heads, head_dim) for flash_attn_func
        query_states = query_states.transpose(1, 2).contiguous()
        key_states = key_states.transpose(1, 2).contiguous()
        value_states = value_states.transpose(1, 2).contiguous()

        attn_output = flash_attn_func(query_states, key_states, value_states, causal=True)
        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]

        # 5. Output projection
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None

    return forward


def make_glm4_moe_lite_attention_forward():
    """Create optimized forward for Glm4MoeLiteAttention using flash-attn-cute.

    GLM-4.7-Flash uses LoRA-style attention with:
    - q_a_proj/q_b_proj for query (when q_lora_rank is set)
    - kv_a_proj_with_mqa/kv_b_proj for key-value
    - kv_a_layernorm for normalization
    - Interleaved rotary embeddings
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[object] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.shape

        # Get configuration
        num_heads = self.num_heads
        num_kv_heads = self.num_key_value_heads

        # Query projection (LoRA-style or direct)
        if hasattr(self, "q_a_proj") and self.q_a_proj is not None:
            # LoRA-style: q_a_proj -> q_a_layernorm -> q_b_proj
            q_compressed = self.q_a_proj(hidden_states)
            if hasattr(self, "q_a_layernorm") and self.q_a_layernorm is not None:
                q_compressed = self.q_a_layernorm(q_compressed)
            query_states = self.q_b_proj(q_compressed)
        else:
            query_states = self.q_proj(hidden_states)

        # KV projection (LoRA-style)
        if hasattr(self, "kv_a_proj_with_mqa"):
            # LoRA-style KV: kv_a_proj_with_mqa -> kv_a_layernorm -> kv_b_proj
            kv_compressed = self.kv_a_proj_with_mqa(hidden_states)
            if hasattr(self, "kv_a_layernorm") and self.kv_a_layernorm is not None:
                kv_compressed = self.kv_a_layernorm(kv_compressed)
            kv_states = self.kv_b_proj(kv_compressed)
            # Split KV
            kv_seq_len = q_len
            key_states = kv_states[..., : num_kv_heads * self.qk_head_dim]
            value_states = kv_states[..., num_kv_heads * self.qk_head_dim :]
        else:
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        # Reshape
        head_dim = self.qk_head_dim if hasattr(self, "qk_head_dim") else self.head_dim
        v_head_dim = self.v_head_dim if hasattr(self, "v_head_dim") else head_dim

        query_states = query_states.view(bsz, q_len, num_heads, head_dim)
        key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim)
        value_states = value_states.view(bsz, q_len, num_kv_heads, v_head_dim)

        # Transpose to (bsz, num_heads, seq_len, head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Apply RoPE
        if position_embeddings is not None:
            cos, sin = position_embeddings
            from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import apply_rotary_pos_emb
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # KV Cache
        if past_key_value is not None:
            cache_kwargs = {}
            if cache_position is not None:
                cache_kwargs["cache_position"] = cache_position
            if position_embeddings is not None:
                cos, sin = position_embeddings
                cache_kwargs["sin"] = sin
                cache_kwargs["cos"] = cos
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # Flash Attention
        query_states = query_states.transpose(1, 2).contiguous()
        key_states = key_states.transpose(1, 2).contiguous()
        value_states = value_states.transpose(1, 2).contiguous()

        attn_output = flash_attn_func(query_states, key_states, value_states, causal=True)
        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]

        # Output projection
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None

    return forward


def patch_attention(module):
    """Patch GLM4 attention module with optimized forward.

    No-op if flash-attn-cute is not available.
    """
    if flash_attn_func is None:
        return  # flash-attn-cute not available

    if hasattr(module, "_machete_original_forward"):
        return  # Already patched

    class_name = module.__class__.__name__

    if class_name == "Glm4MoeAttention":
        forward_fn = make_glm4_moe_attention_forward()
    elif class_name == "Glm4MoeLiteAttention":
        forward_fn = make_glm4_moe_lite_attention_forward()
    else:
        return  # Unknown attention class

    module._machete_original_forward = module.forward
    module.forward = types.MethodType(forward_fn, module)


def unpatch_attention(module):
    """Restore original GLM4 attention forward."""
    if hasattr(module, "_machete_original_forward"):
        module.forward = module._machete_original_forward
        del module._machete_original_forward


def make_glm4_causal_lm_forward():
    """Create optimized forward for GLM4ForCausalLM using fused linear CE.

    Works for both Glm4MoeForCausalLM and Glm4MoeLiteForCausalLM.
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
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: int = 0,
        **kwargs,
    ):
        # If no labels, use original forward
        if labels is None:
            return self._machete_original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

        # Get model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
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
            # Shift for next token prediction
            shift_hidden = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = fused_linear_cross_entropy(
                shift_hidden,
                self.lm_head.weight,
                shift_labels,
                ignore_index=-100,
                reduction="mean",
            )

            # Compute logits for output
            logits = self.lm_head(hidden_states)
        else:
            logits = self.lm_head(hidden_states)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

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
    """Patch GLM4ForCausalLM with fused linear cross entropy."""
    if not HAS_QUACK_LINEAR_CE:
        return

    if hasattr(model, "_machete_original_forward"):
        return  # Already patched

    if not hasattr(model, "lm_head") or not hasattr(model, "model"):
        return

    model._machete_original_forward = model.forward
    model.forward = types.MethodType(make_glm4_causal_lm_forward(), model)


def unpatch_causal_lm(model):
    """Restore original CausalLM forward."""
    if hasattr(model, "_machete_original_forward"):
        model.forward = model._machete_original_forward
        del model._machete_original_forward
