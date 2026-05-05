# Machete

Machete is a CuTe DSL kernel framework for building persistent GPU kernels from small, composable ops. An op declares tensors, tiling, and optional `load` / `compute` / `store` phases; the framework builds one instruction stream, schedules tiles across SMs, and manages shared-memory pages, barriers, TMA descriptors, and launch plumbing.

## Core Ideas

- **Ops stay local.** Each op owns its math and tensor interface.
- **The framework owns replay.** Tile scheduling, dependencies, page allocation, barriers, and TMA metadata are generated once for the whole megakernel.
- **Regions express fast paths.** A `region(...)` groups semantic ops. It can replay those ops, or lower to one generated op that owns a fused persistent loop.
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

`inner_iter_idx` is intentionally not part of the op API. Long ranges should be expressed inside the op body or by a generated region so the load/compute/store protocol remains owned by one implementation.

## Persistent Regions

Use `region(...)` when several ops should be treated as one instruction boundary. Without `generated_op`, the region replays the child ops. With `generated_op`, it lowers to that single specialized op while keeping the child ops as the semantic schedule.

```python
from machete.megakernel import PipelineSpec, region

semantic_ops = op_a + op_b + op_c
generated = FusedRegionOp.schedule(...)[0]

items = [
    region(
        "fused_block",
        semantic_ops,
        generated_op=generated,
        pipeline=PipelineSpec.streaming(page_bytes=16 * 1024),
    )
]

kernel = Megakernel(items)
```

This is the intended path for decode kernels that need a custom persistent loop: normalize once, stream weights or cache blocks through pages, compute, and store/reduce without returning to a generic per-op replay loop.

## Built-In Areas

- Attention kernels for SM100/SM120.
- GEMM, RMSNorm, RoPE, GLU, MoE, and cross entropy ops.
- Decode-oriented Qwen 3.5 SM120/NVFP4 experiments.
- Autograd helpers for composing megakernel-backed modules.
- Trace export helpers for profiling persistent-kernel replay.

## Development

Run tests through the project environment:

```bash
env PYTHONPATH=. venvmachete/bin/python -m pytest tests/megakernel -q
```

## License

Apache 2.0
