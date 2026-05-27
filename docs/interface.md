# Machete Interface

This document describes the public programming model for building Machete
megakernels. The README stays focused on what the project is and the current
benchmark results.

## Core Model

Machete builds a persistent GPU kernel from a list of scheduled ops. Each op is
a small CuTe DSL component with a static tensor interface, tile shape, and one or
more execution phases. The host side lowers the op list into an instruction
stream, infers tile dependencies from named tensor reads and writes, allocates
shared-memory pages, and launches one persistent kernel.

Typical use:

```python
import torch

from machete.megakernel import Megakernel, MegakernelConfig
from machete.ops.gemm import GemmOp

a = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
b = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
c = torch.empty(128, 128, device="cuda", dtype=torch.bfloat16)

ops = GemmOp.schedule(a=a, b=b, c=c, tile_sizes={"M": 64, "N": 64, "K": 64})
kernel = Megakernel(ops, config=MegakernelConfig(page_size=32768, num_pages=3))
kernel.run()
```

## Op Contract

An op subclasses `Op` and declares named tensor reads, named tensor writes, and
tile dimensions. The declaration is compile-time structure; the implementation
is CuTe DSL code.

```python
import cutlass.cute as cute
from cutlass import Float32

from machete.megakernel import Op, config_flat_tensor


class ScaleOp(Op):
    reads = {"x": (Float32, ("M", "D"))}
    writes = {"y": (Float32, ("M", "D"))}
    tile = ("M", "D")

    @cute.jit
    def compute(self, page_ptr, tile_M: cute.Int32, tile_D: cute.Int32, op_config_ptr):
        x = config_flat_tensor(op_config_ptr, "x", self.x_dtype, self.M * self.D, type(self))
        y = config_flat_tensor(op_config_ptr, "y", self.y_dtype, self.M * self.D, type(self))
        tid = cute.arch.thread_idx()[0]
        base = tile_M * self.tile_size_M * self.D + tile_D * self.tile_size_D
        for i in cute.range(tid, self.tile_size_M * self.tile_size_D, cute.arch.block_dim()[0]):
            row = i // self.tile_size_D
            col = i - row * self.tile_size_D
            y[base + row * self.D + col] = x[base + row * self.D + col] * 2.0
```

The framework injects static attributes such as dimensions, tile sizes, tensor
dtypes, strides, and page offsets into the op instance at compile time.

## Scheduling

Most ops use `Op.schedule(...)`, which returns one or more `ScheduledOp`
instances. Tensor arguments bind declared tensor names to runtime tensors, and
`tile_sizes` defines the tile extent for named tile dimensions.

```python
ops = ScaleOp.schedule(x=x, y=y, tile_sizes={"M": 1, "D": 256})
kernel = Megakernel(ops)
kernel.compile()
kernel.run()
```

Ops with custom tiling or multiple generated instructions can override
`schedule`. External callers should express problem size and tensor bindings at
this level. Internal iteration details, such as a long-context inner loop, should
stay inside the op body so the load/compute/store protocol remains owned by one
implementation.

## Dependencies

Machete infers dependencies from declared tensor names:

- A later op that reads a tensor waits for earlier ops that write the same
  tensor.
- Named dimensions and tile sizes define tile-level dependency formulas.
- Producer/consumer ratios are supported when tile sizes differ.
- Controller waits include both DMA and compute-side waits, so scheduling keeps
  full dependency formulas rather than reducing them to only load/store waits.

For unusual dataflow, use explicit `ScheduledOp` construction or a custom
`schedule` method in the op.

## Phases And Pages

An op can implement any combination of `load`, `compute`, `store`, and
`communicate`. Compute-only ops can put all work in `compute`. Staged ops declare
a `PipelineSpec` so the megakernel reserves op-local pages, semaphores, and
scratch.

```python
from machete.megakernel import PipelineSpec


class StagedOp(Op):
    pipeline = PipelineSpec.streaming(
        input_stages=3,
        output_stages=3,
        stage_pages=4,
        page_bytes=32768,
    )
```

The kernel-wide shared-memory ring is configured with `MegakernelConfig`:

```python
config = MegakernelConfig(
    threads_per_block=256,
    page_size=32768,
    num_pages=3,
    tracing=False,
)
```

For Qwen 3.5 full-layer benchmarks on Blackwell, the current reference setting
is `page_size=32768` and `num_pages=3`.

## Tracing

Enable tracing when creating the kernel, run it, then export either the raw trace
or a Perfetto-compatible file.

```python
kernel = Megakernel(ops, config=MegakernelConfig(tracing=True))
kernel.run()
kernel.write_trace("trace.nanotrace")
kernel.write_trace_perfetto("trace.json")
```

Tracing is stripped from generated source when `tracing=False`.

## Benchmark Interfaces

The Qwen graphics are generated with:

```bash
venvmachete/bin/python scripts/generate_qwen35_benchmark_graphics.py \
  --group all \
  --out-dir benchmark_results/qwen35_graphics \
  --llamacpp-model /tmp/llama-models/qwen35/Qwen3.5-0.8B-Q8_0.gguf \
  --luce-dir /home/elentir/Projets/lucebox-hub \
  --luce-backend nvfp4 \
  --decode-context-len 128 256 512 1024 \
  --decode-page-size 32768 \
  --decode-num-pages 3 \
  --machete-decode-bench nvfp4 \
  --machete-scheduler overlap-adaptive \
  --no-machete-dummy-weights \
  --training-seq-len 128 256 512 1024 \
  --training-page-size 32768 \
  --num-pages 3
```

The training graphics intentionally require a source-backed full Qwen layer
benchmark at `benchmarks/kernels/benchmark_qwen3_5_layer.py`. The old
MLP-block benchmark is not used for these graphs.

Machete decode should be run with `--machete-decode-bench nvfp4` and
`--no-machete-dummy-weights` for published graphics. Token equivalence with the
Luce/llama reference path is checked by
`scripts/check_qwen35_machete_decode_equivalence.py`.
