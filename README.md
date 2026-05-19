# Machete

Machete is a CuTe DSL kernel framework for building persistent GPU kernels from small, composable ops. An op declares tensors, tiling, and optional `load` / `compute` / `store` phases; the framework builds one instruction stream, schedules tiles across SMs, and manages shared-memory pages, barriers, TMA descriptors, and launch plumbing.

## Core Ideas

- **Ops stay local.** Each op owns its math and tensor interface.
- **The framework owns replay.** Tile scheduling, dependencies, page allocation, barriers, and TMA metadata are prepared once for the whole megakernel.
- **Streaming is explicit.** `PipelineSpec.streaming(...)` declares a load/compute/store page ring for ops that can overlap data movement with compute.

## Install

```bash
pip install machete
```

Requires NVIDIA CuTe DSL / CUTLASS and a Hopper or Blackwell GPU.

## Basic Use

```python
import torch
from machete.megakernel import Megakernel
from machete.kernels.gemm import GemmOp

x = torch.randn(16, 1024, device="cuda", dtype=torch.bfloat16)
w = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
y = torch.empty(16, 1024, device="cuda", dtype=torch.bfloat16)

ops = GemmOp.schedule(a=x, b=w, c=y)
kernel = Megakernel(ops)
kernel.run()
```

## Defining An Op

```python
import cutlass.cute as cute
from cutlass import Float32, Int32
from machete.megakernel import Op

class ScaleOp(Op):
    reads = {"x": (None, ("N",))}
    writes = {"y": (None, ("N",))}
    tile = ("N",)

    @classmethod
    def schedule(cls, *, x, y, tile_n=256):
        return [cls._schedule_single(tile_sizes={"N": tile_n}, x=x, y=y)]

    @cute.jit
    def compute(self, page_ptr, tile_N, x, y):
        tidx = cute.arch.thread_idx()[0]
        start = tile_N * Int32(self.tile_size_N)
        i = tidx
        while i < Int32(self.tile_size_N) and start + i < Int32(self.N):
            y.iterator[start + i] = (x.iterator[start + i].to(Float32) * Float32(2.0)).to(self.y_dtype)
            i = i + Int32(self.threads_per_row)
```

## Streaming Ops

For ops that benefit from load/compute/store overlap, declare a streaming pipeline and implement the phases. The op describes the resources; the framework decides how to replay phases safely.

```python
from machete.megakernel import Op, PipelineSpec

class StreamingOp(Op):
    pipeline = PipelineSpec.streaming(
        input_stages=3,
        output_stages=3,
        stage_pages=4,
        page_bytes=16 * 1024,
    )

    # Optional phase methods:
    # def load(...):    issue async loads into the current page/stage
    # def compute(...): consume staged data
    # def store(...):   write staged output
```

## Built-In Areas

- Attention kernels for SM100/SM120.
- GEMM, RMSNorm, RoPE, GLU, MoE, and cross entropy ops.
- Decode-oriented Qwen 3.5 SM120/NVFP4 kernels.
- Autograd helpers for composing megakernel-backed modules.
- Trace export helpers for profiling persistent-kernel replay.

## Development

Run tests through the project environment:

```bash
env PYTHONPATH=. venvmachete/bin/python -m pytest tests/megakernel -q
```

## License

Apache 2.0
