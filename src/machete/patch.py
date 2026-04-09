# Copyright (c) 2025, Machete Authors
"""Patch and unpatch supported HuggingFace model families.

This module centralizes model-family detection and the host-side traversal
that applies per-module Machete patches. Family-specific implementations
live under ``machete.patching``.
"""

import logging
from typing import Any, Dict, Optional

from machete.patching import llama, qwen, glm4
from machete.patching.ops import patch_cross_entropy_loss, patch_rmsnorm, unpatch_rmsnorm

logger = logging.getLogger(__name__)


# Mapping of model types to their patching modules and class names.
MODEL_TYPE_MODULES: Dict[str, Dict[str, Any]] = {
    "llama": {
        "module": llama,
        "attention": llama.ATTENTION_CLASSES,
        "rmsnorm": llama.RMSNORM_CLASSES,
        "mlp": getattr(llama, "MLP_CLASSES", ()),
    },
    "qwen2": {
        "module": qwen,
        "attention": qwen.ATTENTION_CLASSES,
        "rmsnorm": qwen.RMSNORM_CLASSES,
        "mlp": qwen.MLP_CLASSES,
    },
    "glm4_moe": {
        "module": glm4,
        "attention": glm4.ATTENTION_CLASSES,
        "rmsnorm": glm4.RMSNORM_CLASSES,
        "mlp": glm4.MLP_CLASSES,
    },
    "glm4_moe_lite": {
        "module": glm4,
        "attention": glm4.ATTENTION_CLASSES,
        "rmsnorm": glm4.RMSNORM_CLASSES,
        "mlp": glm4.MLP_CLASSES,
    },
}

SUPPORTED_MODEL_TYPES = tuple(MODEL_TYPE_MODULES)


def _normalize_model_type(model_type: Optional[str]) -> Optional[str]:
    """Map known aliases onto canonical model type keys."""
    aliases = {
        "llama3": "llama",
        "qwen3": "qwen2",
    }
    if model_type in MODEL_TYPE_MODULES:
        return model_type
    return aliases.get(model_type)


def _detect_model_type(model) -> Optional[str]:
    """Detect the canonical Machete model type from config or class name."""
    if hasattr(model, "config"):
        normalized = _normalize_model_type(getattr(model.config, "model_type", None))
        if normalized is not None:
            return normalized

    class_name = type(model).__name__.lower()
    if "llama" in class_name:
        return "llama"
    if "qwen" in class_name:
        return "qwen2"
    if "glm4moelite" in class_name:
        return "glm4_moe_lite"
    if "glm4moe" in class_name:
        return "glm4_moe"

    return None


def _model_type_config(model_type: str) -> Dict[str, Any]:
    """Return patch metadata for a supported model type."""
    try:
        return MODEL_TYPE_MODULES[model_type]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported model type: {model_type}. Supported: {list(SUPPORTED_MODEL_TYPES)}"
        ) from exc


def _iter_patchable_modules(model, type_config: Dict[str, Any]):
    """Yield ``(module, class_name)`` pairs for modules relevant to a family."""
    target_classes = (
        set(type_config["attention"])
        | set(type_config["rmsnorm"])
        | set(type_config.get("mlp", ()))
    )
    for module in model.modules():
        class_name = type(module).__name__
        if class_name in target_classes:
            yield module, class_name


def _restore_original_forward(module) -> None:
    """Restore ``module.forward`` from the Machete patch marker."""
    module.forward = module._machete_original_forward
    del module._machete_original_forward


def _apply_module_patches(
    model,
    type_config: Dict[str, Any],
    *,
    patch_attention_layers: bool,
    patch_rmsnorm_layers: bool,
    patch_mlp_layers: bool,
) -> Dict[str, int]:
    """Apply enabled module patches and return patch counts by category."""
    model_module = type_config["module"]
    patched_counts = {"attention": 0, "rmsnorm": 0, "mlp": 0}

    for module, class_name in _iter_patchable_modules(model, type_config):
        if patch_attention_layers and class_name in type_config["attention"]:
            model_module.patch_attention(module)
            patched_counts["attention"] += 1
            continue

        if patch_rmsnorm_layers and class_name in type_config["rmsnorm"]:
            patch_rmsnorm(module)
            patched_counts["rmsnorm"] += 1
            continue

        if patch_mlp_layers and class_name in type_config.get("mlp", ()) and hasattr(model_module, "patch_mlp"):
            model_module.patch_mlp(module)
            patched_counts["mlp"] += 1

    return patched_counts


def _remove_module_patches(model, type_config: Dict[str, Any]) -> Dict[str, int]:
    """Remove module-level patches and return unpatch counts by category."""
    model_module = type_config.get("module")
    unpatched_counts = {"attention": 0, "rmsnorm": 0, "mlp": 0}

    for module, class_name in _iter_patchable_modules(model, type_config):
        if not hasattr(module, "_machete_original_forward"):
            continue

        if class_name in type_config.get("attention", ()):
            if model_module is not None:
                model_module.unpatch_attention(module)
            else:
                _restore_original_forward(module)
            unpatched_counts["attention"] += 1
            continue

        if class_name in type_config.get("rmsnorm", ()):
            unpatch_rmsnorm(module)
            unpatched_counts["rmsnorm"] += 1
            continue

        if class_name in type_config.get("mlp", ()):
            if model_module is not None and hasattr(model_module, "unpatch_mlp"):
                model_module.unpatch_mlp(module)
            else:
                _restore_original_forward(module)
            unpatched_counts["mlp"] += 1

    return unpatched_counts


def patch(
    model,
    model_type: Optional[str] = None,
    patch_attention: bool = True,
    patch_rmsnorm: bool = True,
    patch_mlp: bool = False,
    patch_cross_entropy: bool = False,
    patch_fused_lm_head: bool = False,
) -> None:
    """Patch a supported HuggingFace model in place.

    The patch is reversible via :func:`unpatch`. Individual feature families
    can be enabled or disabled independently.

    Args:
        model: HuggingFace model instance to patch.
        model_type: Canonical model family key. If omitted, Machete inspects
            ``model.config.model_type`` and then the model class name.
        patch_attention: Patch attention modules for the detected family.
        patch_rmsnorm: Patch RMSNorm layers.
        patch_mlp: Patch family-specific MLP implementations when available.
        patch_cross_entropy: Patch ``CrossEntropyLoss`` usage.
        patch_fused_lm_head: Patch the model forward for fused LM head loss.

    Raises:
        ValueError: If the model family cannot be detected or is unsupported.
    """
    if model_type is None:
        model_type = _detect_model_type(model)
        if model_type is None:
            raise ValueError(
                "Could not detect model type. Please specify model_type explicitly. "
                "Supported types: llama, qwen2, glm4_moe, glm4_moe_lite"
            )

    model_type = _normalize_model_type(model_type) or model_type
    type_config = _model_type_config(model_type)
    model_module = type_config["module"]
    patched_counts = _apply_module_patches(
        model,
        type_config,
        patch_attention_layers=patch_attention,
        patch_rmsnorm_layers=patch_rmsnorm,
        patch_mlp_layers=patch_mlp,
    )

    if patch_cross_entropy:
        patch_cross_entropy_loss(model)

    if patch_fused_lm_head:
        if hasattr(model_module, "patch_causal_lm"):
            model_module.patch_causal_lm(model)
            patched_counts["fused_lm_head"] = 1
        else:
            patched_counts["fused_lm_head"] = 0

    logger.info(
        f"Machete patched {model_type} model: "
        f"{patched_counts['attention']} attention, "
        f"{patched_counts['rmsnorm']} RMSNorm, "
        f"{patched_counts['mlp']} MLP layers"
        + (", fused LM head" if patched_counts.get("fused_lm_head") else "")
    )

    # Mark model as patched
    model._machete_patched = True
    model._machete_model_type = model_type


def unpatch(model) -> None:
    """Remove Machete patches from a model in place.

    Args:
        model: Model previously patched by :func:`patch`.
    """
    if not getattr(model, "_machete_patched", False):
        logger.warning("Model does not appear to be patched by Machete")
        return

    model_type = getattr(model, "_machete_model_type", None)
    if model_type is None:
        model_type = _detect_model_type(model)

    if model_type not in MODEL_TYPE_MODULES:
        logger.warning(f"Unknown model type: {model_type}, attempting generic unpatch")
        type_config = {"attention": (), "rmsnorm": (), "mlp": (), "module": None}
    else:
        type_config = MODEL_TYPE_MODULES[model_type]

    model_module = type_config.get("module")
    unpatched_counts = _remove_module_patches(model, type_config)

    # Unpatch fused LM head if present
    if hasattr(model, "_machete_original_forward"):
        if model_module and hasattr(model_module, "unpatch_causal_lm"):
            model_module.unpatch_causal_lm(model)
        else:
            model.forward = model._machete_original_forward
            del model._machete_original_forward
        unpatched_counts["fused_lm_head"] = 1

    logger.info(
        f"Machete unpatched model: "
        f"{unpatched_counts['attention']} attention, "
        f"{unpatched_counts['rmsnorm']} RMSNorm, "
        f"{unpatched_counts['mlp']} MLP layers"
        + (", fused LM head" if unpatched_counts.get("fused_lm_head") else "")
    )

    del model._machete_patched
    if hasattr(model, "_machete_model_type"):
        del model._machete_model_type


def is_patched(model) -> bool:
    """Check if a model has been patched by Machete."""
    return getattr(model, "_machete_patched", False)
