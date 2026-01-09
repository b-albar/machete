# Copyright (c) 2025, Machete Authors
"""Main patching logic for HuggingFace models."""

from typing import Optional
import logging

from machete.patching import llama, qwen
from machete.patching.ops import unpatch_rmsnorm, patch_cross_entropy_loss

logger = logging.getLogger(__name__)


# Mapping of model types to their patching modules and class names
MODEL_TYPE_MODULES = {
    "llama": {
        "module": llama,
        "attention": llama.ATTENTION_CLASSES,
        "rmsnorm": llama.RMSNORM_CLASSES,
    },
    "qwen2": {
        "module": qwen,
        "attention": qwen.ATTENTION_CLASSES,
        "rmsnorm": qwen.RMSNORM_CLASSES,
    },
}


def _detect_model_type(model) -> Optional[str]:
    """Detect the model type from the model's class name or config."""
    # Try from config
    if hasattr(model, "config"):
        model_type = getattr(model.config, "model_type", None)
        if model_type in MODEL_TYPE_MODULES:
            return model_type
        # Handle variations
        if model_type in ("llama", "llama3"):
            return "llama"
        if model_type in ("qwen2", "qwen3"):
            return "qwen2"

    # Try from class name
    class_name = model.__class__.__name__.lower()
    if "llama" in class_name:
        return "llama"
    if "qwen" in class_name:
        return "qwen2"

    return None


def _get_module_class_name(module) -> str:
    """Get the class name of a module."""
    return module.__class__.__name__


def patch(
    model,
    model_type: Optional[str] = None,
    patch_attention: bool = True,
    patch_rmsnorm: bool = True,
    patch_cross_entropy: bool = False,
) -> None:
    """
    Patch a HuggingFace model with Machete optimizations.

    This function modifies the model in-place, replacing forward methods
    with optimized versions using flash-attn-cute and quack.

    Args:
        model: A HuggingFace model (e.g., from AutoModelForCausalLM)
        model_type: Model type ("llama", "qwen2"). Auto-detected if None.
        patch_attention: Whether to patch attention layers
        patch_rmsnorm: Whether to patch RMSNorm layers
        patch_cross_entropy: Whether to patch CrossEntropyLoss

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> import machete
        >>>
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
        >>> machete.patch(model)
        >>>
        >>> # Works with LoRA
        >>> from peft import get_peft_model, LoraConfig
        >>> model = get_peft_model(model, LoraConfig(...))
    """
    if model_type is None:
        model_type = _detect_model_type(model)
        if model_type is None:
            raise ValueError(
                "Could not detect model type. Please specify model_type explicitly. Supported types: llama, qwen2"
            )

    if model_type not in MODEL_TYPE_MODULES:
        raise ValueError(f"Unsupported model type: {model_type}. Supported: {list(MODEL_TYPE_MODULES.keys())}")

    type_config = MODEL_TYPE_MODULES[model_type]
    model_module = type_config["module"]
    patched_counts = {"attention": 0, "rmsnorm": 0}

    for name, module in model.named_modules():
        class_name = _get_module_class_name(module)

        if patch_attention and class_name in type_config["attention"]:
            model_module.patch_attention(module)
            patched_counts["attention"] += 1

        elif patch_rmsnorm and class_name in type_config["rmsnorm"]:
            from machete.patching.ops import patch_rmsnorm as _patch_rmsnorm

            _patch_rmsnorm(module)
            patched_counts["rmsnorm"] += 1

    if patch_cross_entropy:
        patch_cross_entropy_loss(model)

    logger.info(
        f"Machete patched {model_type} model: "
        f"{patched_counts['attention']} attention, "
        f"{patched_counts['rmsnorm']} RMSNorm layers"
    )

    # Mark model as patched
    model._machete_patched = True
    model._machete_model_type = model_type


def unpatch(model) -> None:
    """
    Remove Machete patches from a model, restoring original forward methods.

    Args:
        model: A Machete-patched HuggingFace model
    """
    if not getattr(model, "_machete_patched", False):
        logger.warning("Model does not appear to be patched by Machete")
        return

    model_type = getattr(model, "_machete_model_type", None)
    if model_type is None:
        model_type = _detect_model_type(model)

    if model_type not in MODEL_TYPE_MODULES:
        logger.warning(f"Unknown model type: {model_type}, attempting generic unpatch")
        type_config = {"attention": (), "rmsnorm": (), "module": None}
    else:
        type_config = MODEL_TYPE_MODULES[model_type]

    model_module = type_config.get("module")
    unpatched_counts = {"attention": 0, "rmsnorm": 0}

    for name, module in model.named_modules():
        if hasattr(module, "_machete_original_forward"):
            class_name = _get_module_class_name(module)

            if class_name in type_config.get("attention", ()):
                if model_module:
                    model_module.unpatch_attention(module)
                else:
                    module.forward = module._machete_original_forward
                    del module._machete_original_forward
                unpatched_counts["attention"] += 1
            elif class_name in type_config.get("rmsnorm", ()):
                unpatch_rmsnorm(module)
                unpatched_counts["rmsnorm"] += 1
            else:
                # Generic unpatch
                module.forward = module._machete_original_forward
                del module._machete_original_forward

    logger.info(
        f"Machete unpatched model: "
        f"{unpatched_counts['attention']} attention, "
        f"{unpatched_counts['rmsnorm']} RMSNorm layers"
    )

    del model._machete_patched
    if hasattr(model, "_machete_model_type"):
        del model._machete_model_type


def is_patched(model) -> bool:
    """Check if a model has been patched by Machete."""
    return getattr(model, "_machete_patched", False)
