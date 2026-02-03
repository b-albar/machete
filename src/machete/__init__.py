# Copyright (c) 2025, Machete Authors
"""Machete: Optimized kernels for HuggingFace Transformer models.

Machete patches existing HuggingFace models with optimized implementations
using flash-attn-cute and quack, while maintaining full compatibility with
Transformers, TRL, and LoRA/PEFT.

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
    >>>
    >>> # Works with TRL
    >>> from trl import SFTTrainer
    >>> trainer = SFTTrainer(model=model, ...)
"""

__version__ = "0.1.0"

from machete.patch import patch, unpatch, is_patched

__all__ = [
    "patch",
    "unpatch",
    "is_patched",
    "__version__",
]
