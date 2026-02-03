# Machete

High-performance GPU kernel library for deep learning, built on NVIDIA CuteDSL.

## Features

- **Megakernel** - Persistent kernel architecture for fusing multiple operations
  - Instruction stream execution with automatic tile scheduling
  - Dependency resolution via named buffers
  - Multiple scheduling strategies (level-batched, backward)
  - Per-SM tracing for performance analysis

- **Optimized Kernels** - High-performance implementations of common operations

- **Model Patching** - Optional drop-in optimization for transformer models
  - Works with Transformers, TRL, PEFT/LoRA

## Installation

```bash
pip install machete
```

### Dependencies

- NVIDIA CUTLASS
- [flash-attn-cute](https://github.com/b-albar/flash-attention/tree/main/flash_attn/cute)
- [quack](https://github.com/b-albar/quack)
- [nvidia-cutlass-dsl](https://github.com/NVIDIA/cutlass)

## Quick Start

### Megakernel

```python
from machete.megakernel import Megakernel, MegakernelConfig
from machete.kernels.rope import RopeOp

# Schedule operations
ops = [RopeOp.schedule(q=q_tensor, cos=cos, sin=sin)]

# Create and run kernel
config = MegakernelConfig(tracing=True)
kernel = Megakernel(ops, config=config)
kernel.run()

# Export trace for performance analysis
kernel.write_trace("trace.nanotrace")  # or .perfetto
```

### Model Patching

```python
from transformers import AutoModelForCausalLM
import machete

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
machete.patch(model)
```

## Megakernel Architecture

The megakernel system fuses multiple operations into a single persistent kernel:

```
┌─────────────────────────────────────────────────────┐
│                   Megakernel                        │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐          │
│  │  Op A   │──▶│  Op B   │──▶│  Op C   │          │
│  └─────────┘   └─────────┘   └─────────┘          │
│       │             │             │                │
│       ▼             ▼             ▼                │
│  [Instruction Stream: A0,A1,B0,A2,B1,C0,...]      │
│                                                    │
│  SM0: fetches A0, B0, C0, ...  (strided)          │
│  SM1: fetches A1, B1, C1, ...                     │
└─────────────────────────────────────────────────────┘
```

### Tile Schedulers

- **LevelBatchedScheduler** (default): Emits all source tiles first for better SM load balancing
- **BackwardScheduler**: Depth-based priority for latency optimization

```python
from machete.megakernel.ops import BackwardScheduler, set_default_scheduler

set_default_scheduler(BackwardScheduler())
```

### Tracing

Export execution traces for performance analysis:

```python
config = MegakernelConfig(tracing=True)
kernel = Megakernel(ops, config=config)
kernel.run()

# Export formats
kernel.write_trace("trace.nanotrace")  # Native format
kernel.write_trace("trace.perfetto")   # Open at https://ui.perfetto.dev/
```

## Defining Custom Operations

```python
from machete.megakernel.ops import Op
from cutlass import Float32

class MyOp(Op):
    reads  = {"x": (Float32, "M, D")}
    writes = {"y": (Float32, "M, D")}
    tile   = ("M",)

    INPUTS = ["x"]
    OUTPUTS = ["y"]

    @staticmethod
    def compute_forward(smem_base, config_ptr, page_ids,
                        tile_m, tile_n, tile_l, op_config_ptr):
        # M is dynamic (tile dim), D is static (compile-time constant)
        ...

ops = [MyOp.schedule(x=input_tensor, y=output_tensor)]
```

## License

Apache 2.0
