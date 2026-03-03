# Machete

High-performance GPU kernel library for deep learning, built on NVIDIA CuTe DSL.

## Features

- **Megakernel framework** -- Persistent kernel that fuses multiple operations into a single launch
  - Pipelined load/compute/store with paged shared memory
  - Automatic tile scheduling, dependency resolution, and SM work distribution
  - Dedicated DMA warps (TMA load + store) and MMA compute warps
  - Per-SM tracing (Perfetto / nanotrace export)

- **Built-in kernels** -- Flash Attention (SM100 Hopper + SM120 Blackwell), GEMM, RoPE, RMSNorm, Activation

- **Model patching** -- Drop-in optimization for HuggingFace Transformers models

## Installation

```bash
pip install machete
```

Requires [nvidia-cutlass-dsl](https://github.com/NVIDIA/cutlass) and a Hopper (SM90+) or Blackwell (SM120+) GPU.

## Quick Start

### Using built-in kernels

```python
import torch
from machete.megakernel import Megakernel
from machete.kernels.attention import FlashAttentionOp

q = torch.randn(1, 128, 64, dtype=torch.float16, device="cuda")  # (BH, M, D)
k = torch.randn(1, 128, 64, dtype=torch.float16, device="cuda")  # (BH, N, D)
v = torch.randn(1, 128, 64, dtype=torch.float16, device="cuda")  # (BH, N, D)
o = torch.zeros_like(q)

ops = FlashAttentionOp.schedule_forward(q=q, k=k, v=v, o=o)
config = FlashAttentionOp.kernel_config(ops)
kernel = Megakernel(ops, config=config)
kernel.run()
```

### Fusing operations

```python
from machete.kernels.gemm import GemmOp
from machete.kernels.activation import ActivationOp

c = torch.zeros(M, N, dtype=torch.float16, device="cuda")
ops  = GemmOp.schedule_forward(a=a, b=b, c=c)
ops += ActivationOp.schedule_forward(x=c, activation='silu')

kernel = Megakernel(ops)
kernel.run()
# GEMM and activation run fused -- the framework auto-detects the
# dependency on tensor c and pipelines them in a single kernel launch.
```

### Model patching

```python
from transformers import AutoModelForCausalLM
import machete

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
machete.patch(model)
```

## Architecture

The megakernel compiles all scheduled ops into a single persistent CUDA kernel. Each SM fetches work tiles from a shared instruction stream:

```
┌──────────────────────────────────────────────────────┐
│                    Megakernel                         │
│                                                      │
│  Instruction stream: [A0, A1, B0, A2, B1, C0, ...]  │
│                                                      │
│  SM 0 ──▶ A0 ──▶ B0 ──▶ C0 ──▶ ...  (strided)      │
│  SM 1 ──▶ A1 ──▶ B1 ──▶ C1 ──▶ ...                  │
│  SM 2 ──▶ A2 ──▶ B2 ──▶ C2 ──▶ ...                  │
│                                                      │
│  Per SM:                                             │
│    Load warp  ── TMA G→S ──┐                         │
│                             ├── paged shared memory  │
│    MMA warps  ── compute ──┤                         │
│                             │                        │
│    Store warp ── TMA S→G ──┘                         │
└──────────────────────────────────────────────────────┘
```

### Configuration

```python
from machete.megakernel import Megakernel, MegakernelConfig

config = MegakernelConfig(
    threads_per_block=192,  # (4 MMA + 2 DMA) warps * 32
    page_size=32768,        # 32KB shared memory per page
    tracing=True,           # enable per-SM trace recording
)
kernel = Megakernel(ops, config=config)
kernel.run()

# Most ops provide a kernel_config() classmethod that returns a
# recommended config based on tile sizes and tensor shapes:
config = MyOp.kernel_config(ops)
```

### Tracing

```python
config = MegakernelConfig(tracing=True)
kernel = Megakernel(ops, config=config)
kernel.run()

kernel.write_trace("trace.nanotrace")
kernel.write_trace("trace.perfetto")   # open at https://ui.perfetto.dev/
```

### Tile schedulers

- **LevelBatchedScheduler** (default) -- emits all source tiles first for better SM load balancing
- **BackwardScheduler** -- depth-based priority for latency optimization

```python
from machete.megakernel.ops import BackwardScheduler, set_default_scheduler
set_default_scheduler(BackwardScheduler())
```

## Defining Custom Operations

An op declares its tensor interface, tiling strategy, and implements three phases: `load`, `compute`, `store`.

```python
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32

from machete.megakernel.ops import Op, DEFAULT_PAGE_SIZE
from machete.megakernel.interpreter import mbarrier_arrive_expect_tx


class ScaleOp(Op):
    """Multiply every element by a scalar. Demonstrates the op interface."""

    # Tensor declarations: {name: (dtype, dim_names)}
    # dtype=None infers from the tensor passed at schedule time.
    reads  = {"x": (None, ("M", "D"))}
    writes = {"y": (None, ("M", "D"))}

    # Dimensions to tile over -- these define the grid of work tiles.
    # Non-tiled dimensions (here D) are processed in full within each tile.
    tile = ("M",)

    # Tensors transferred via TMA (async bulk copy).
    tma_loads  = {"x"}
    tma_stores = {"y"}

    def __init__(self, **config):
        # The framework injects: M, D, tile_size_M, threads_per_row,
        # x_dtype, y_dtype, and any static_dims set during scheduling.
        super().__init__(**config)
        self.scale = getattr(self, 'scale', 1.0)
        self.elem_bytes = 2 if self.x_dtype in (cutlass.Float16, cutlass.BFloat16) else 4
        self.tile_bytes = self.tile_size_M * self.D * self.elem_bytes

    @classmethod
    def schedule_forward(cls, scale=1.0, page_size=DEFAULT_PAGE_SIZE, **tensors):
        x = tensors["x"]
        D = x.shape[1]
        tile_M = max(1, page_size // (D * x.element_size()))
        if "y" not in tensors:
            tensors["y"] = tensors["x"]  # in-place

        ops = [cls._schedule_single(tile_sizes={"M": tile_M}, **tensors)]
        ops[0].static_dims["scale"] = scale
        ops[0].static_dims["page_size"] = page_size
        return ops

    @classmethod
    def kernel_config(cls, ops):
        from machete.megakernel import MegakernelConfig
        page_size = ops[0].static_dims.get("page_size", DEFAULT_PAGE_SIZE)
        return MegakernelConfig(page_size=page_size)

    # ----- DMA warp: load x tile from global → shared memory -----
    @cute.jit
    def load(self, page_ptr, tile_M, x_tma, x_tma_gmem, work_mbar):
        nbytes = Int32(self.tile_bytes)
        mbar_ptr = cute.get_mbarrier_ptr(work_mbar)
        sX = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.tile_size_M, self.D)),
        )
        gX = cute.local_tile(x_tma_gmem, (self.tile_size_M, self.D), (0, 0))
        tXsX, tXgX = cute.tma_partition(x_tma, mbar_ptr, sX, gX)
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, nbytes)
        cute.copy(x_tma, tXgX[0], tXsX, tma_bar_ptr=mbar_ptr)

    # ----- MMA warps: compute in shared memory -----
    @cute.jit
    def compute(self, page_ptr, tile_M):
        tidx = cute.arch.thread_idx()
        n_elems = Int32(self.tile_size_M * self.D)
        smem_ptr = cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem)
        for i in range(tidx, n_elems, Int32(self.threads_per_row)):
            val = Float32(smem_ptr[i])
            smem_ptr[i] = val * Float32(self.scale)

    # ----- DMA warp: store result from shared → global memory -----
    @cute.jit
    def store(self, page_ptr, tile_M, y_tma, y_tma_gmem):
        sY = cute.make_tensor(
            cute.make_ptr(self.y_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.tile_size_M, self.D)),
        )
        gY = cute.local_tile(y_tma_gmem, (self.tile_size_M, self.D), (0, 0))
        tYsY, tYgY = cute.tma_partition(y_tma, None, sY, gY)
        cute.copy(y_tma, tYsY, tYgY[0])
```

## Built-in Kernels

| Kernel | Op Class | Description |
|--------|----------|-------------|
| Flash Attention | `FlashAttentionOp` | Auto-selects SM100 (Hopper) or SM120 (Blackwell) |
| GEMM | `GemmOp` | Tiled matrix multiply with TMA and smem swizzle |
| RoPE | `RopeOp` | Rotary position embedding (forward + backward) |
| RMSNorm | `RMSNormOp` | Root mean square normalization (forward + backward) |
| Activation | `ActivationOp` | Element-wise ReLU / SiLU (fuses with GEMM) |

## License

Apache 2.0
