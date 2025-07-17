import torch
from machete.utils.benchmark import Benchmark
import random
import math

from machete.utils.references.attention.triton_flash_attention import triton_flash_attention
from machete.utils.references.attention.attention_ref import attn_ref


@Benchmark.parametrize("batch_size", [16])
@Benchmark.parametrize("sequence_length", [512, 1024])
@Benchmark.parametrize("dtype", [torch.float16, torch.bfloat16])
@Benchmark.parametrize("logits", [32768])
@Benchmark.parametrize("z_loss_factor", [0.0, 1.0])
def benchmark_fa3_h100(batch_size: int, sequence_length: int, dtype: torch.dtype, logits: int, z_loss_factor: float):
    """
    Benchmarks the triton cross entropy loss implementation
    """
    values = []
    shape = (batch_size, sequence_length)
    for _ in range(math.prod(shape)):
        values.append(random.randint(0, 4 - 1))

    labels = torch.tensor(data=values, dtype=torch.long).view(shape).contiguous().cuda()
    input = torch.randn((batch_size, sequence_length, logits), dtype=dtype).cuda().requires_grad_()

    return {"Torch": attn_ref, "Triton": triton_flash_attention, "Machete": flash_attention}

print(benchmark_fa3_h100._benchmark.run(
    mode="fwd_bwd",
    memory=True,
    export_graphics=True,
    key_split="sequence_length",
    path_graphics="bench_cross_entropy")
)