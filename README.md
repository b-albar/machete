# Machete

Optimized kernels for HuggingFace Transformer models using [flash-attn-cute](https://github.com/b-albar/flash-attention/tree/main/flash_attn/cute) and [quack](https://github.com/b-albar/quack).

Machete patches existing HuggingFace models in-place, replacing forward methods with optimized implementations while maintaining full compatibility with:
- **Transformers** - Works with `AutoModelForCausalLM`
- **TRL** - Works with `SFTTrainer`, `DPOTrainer`, etc.
- **PEFT/LoRA** - LoRA adapters

## Installation

```bash
pip install machete
```

### Dependencies

- [flash-attn-cute](https://github.com/b-albar/flash-attention/tree/main/flash_attn/cute)
- [quack](https://github.com/b-albar/quack)

## Usage

```python
from transformers import AutoModelForCausalLM
import machete

# Load model normally
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# Apply Machete optimizations
machete.patch(model)

# Use with LoRA
from peft import get_peft_model, LoraConfig
model = get_peft_model(model, LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
))

# Use with TRL
from trl import SFTTrainer
trainer = SFTTrainer(model=model, ...)
```

## Supported Models

- **Llama** (Llama 2, Llama 3, Llama 3.2)
- **Qwen** (Qwen2, Qwen3)

## What Gets Patched

| Component | Optimization |
|-----------|-------------|
| Attention | `flash-attn-cute` for faster attention computation |
| RMSNorm | `quack.rmsnorm` for fused normalization |

## API

### `machete.patch(model, **options)`

Patch a model with optimizations.

```python
machete.patch(
    model,
    model_type=None,        # Auto-detected, or "llama"/"qwen2"
    patch_attention=True,   # Patch attention layers
    patch_mlp=True,         # Patch MLP layers
    patch_rmsnorm=True,     # Patch RMSNorm layers
    patch_cross_entropy=False,  # Patch CrossEntropyLoss
)
```

### `machete.unpatch(model)`

Remove patches and restore original forward methods.

### `machete.is_patched(model)`

Check if a model has been patched.
