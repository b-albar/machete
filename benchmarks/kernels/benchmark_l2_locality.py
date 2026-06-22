#!/usr/bin/env python
# Copyright (c) 2026, Machete Authors
"""Probe kernel-side SM-domain work placement and L2 behavior.

This benchmark is intentionally standalone: it validates whether a `%smid`
driven domain queue changes L2/DRAM counters before wiring the idea into the
generated megakernel replay loop.

Useful Nsight Compute run:

    ncu --csv --target-processes all --log-file ncu_l2_locality.csv \
      --metrics lts__t_sector_hit_rate.pct,lts__t_bytes.sum,dram__bytes.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active,\
sm__throughput.avg.pct_of_peak_sustained_elapsed \
      python benchmarks/kernels/benchmark_l2_locality.py --mode domain
"""

from __future__ import annotations

import argparse
import os
import statistics
from dataclasses import dataclass

import torch
from torch.utils.cpp_extension import load_inline


CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ unsigned read_smid() {
    unsigned r;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(r));
    return r;
}

__global__ void smid_probe_kernel(int* block_smids, int* smid_counts) {
    unsigned smid = read_smid();
    if (threadIdx.x == 0) {
        block_smids[blockIdx.x] = static_cast<int>(smid);
        atomicAdd(&smid_counts[smid], 1);
    }
}

__global__ void locality_stream_kernel(
    const float* __restrict__ x,
    float* __restrict__ block_sums,
    const int* __restrict__ work_items,
    const int* __restrict__ domain_offsets,
    const int* __restrict__ domain_lengths,
    int* __restrict__ domain_heads,
    const int* __restrict__ smid_to_domain,
    int* __restrict__ smid_counts,
    int* __restrict__ domain_tile_counts,
    int num_work,
    int chunk_elems,
    int inner_iters,
    int mode,
    int num_domains
) {
    extern __shared__ float smem[];
    __shared__ int sh_work;
    __shared__ int sh_domain;

    float thread_sum = 0.0f;

    unsigned smid = read_smid();
    int domain = smid_to_domain[smid];
    if (domain < 0) {
        domain = 0;
    }
    if (domain >= num_domains) {
        domain = num_domains - 1;
    }

    if (threadIdx.x == 0) {
        atomicAdd(&smid_counts[smid], 1);
    }

    int loop_idx = blockIdx.x;
    while (true) {
        int work = -1;
        if (mode == 0) {
            // Baseline: current persistent-kernel style block-strided stream.
            if (loop_idx < num_work) {
                work = work_items[loop_idx];
                loop_idx += gridDim.x;
            }
            if (threadIdx.x == 0) {
                sh_domain = domain;
            }
        } else if (mode == 1) {
            // Global queue control: measures atomic scheduling overhead without
            // locality partitioning.
            if (threadIdx.x == 0) {
                int local = atomicAdd(&domain_heads[0], 1);
                sh_work = local < num_work ? work_items[local] : -1;
                sh_domain = domain;
            }
            __syncthreads();
            work = sh_work;
        } else {
            // Domain queue: CTA discovers where it landed and pulls only from
            // that SM-domain's queue.
            if (threadIdx.x == 0) {
                int local = atomicAdd(&domain_heads[domain], 1);
                int len = domain_lengths[domain];
                if (local < len) {
                    sh_work = work_items[domain_offsets[domain] + local];
                } else {
                    sh_work = -1;
                }
                sh_domain = domain;
            }
            __syncthreads();
            work = sh_work;
        }

        if (work < 0) {
            break;
        }

        if (threadIdx.x == 0) {
            atomicAdd(&domain_tile_counts[sh_domain], 1);
        }

        const int base = work * chunk_elems;
        float acc = 0.0f;
        for (int r = 0; r < inner_iters; ++r) {
            for (int i = threadIdx.x; i < chunk_elems; i += blockDim.x) {
                // Volatile prevents the compiler from collapsing repeated
                // reads while still using normal global loads.
                const volatile float* vx = x + base + i;
                acc += *vx;
            }
        }
        thread_sum += acc;
        __syncthreads();
    }

    smem[threadIdx.x] = thread_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        block_sums[blockIdx.x] = smem[0];
    }
}

void smid_probe(torch::Tensor block_smids, torch::Tensor smid_counts, int num_blocks, int threads) {
    smid_probe_kernel<<<num_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        block_smids.data_ptr<int>(),
        smid_counts.data_ptr<int>());
}

void locality_stream(
    torch::Tensor x,
    torch::Tensor block_sums,
    torch::Tensor work_items,
    torch::Tensor domain_offsets,
    torch::Tensor domain_lengths,
    torch::Tensor domain_heads,
    torch::Tensor smid_to_domain,
    torch::Tensor smid_counts,
    torch::Tensor domain_tile_counts,
    int chunk_elems,
    int inner_iters,
    int mode,
    int num_domains,
    int num_blocks,
    int threads
) {
    int num_work = work_items.numel();
    size_t smem = static_cast<size_t>(threads) * sizeof(float);
    locality_stream_kernel<<<num_blocks, threads, smem, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        block_sums.data_ptr<float>(),
        work_items.data_ptr<int>(),
        domain_offsets.data_ptr<int>(),
        domain_lengths.data_ptr<int>(),
        domain_heads.data_ptr<int>(),
        smid_to_domain.data_ptr<int>(),
        smid_counts.data_ptr<int>(),
        domain_tile_counts.data_ptr<int>(),
        num_work,
        chunk_elems,
        inner_iters,
        mode,
        num_domains);
}
"""


CPP_SRC = r"""
#include <torch/extension.h>

void smid_probe(torch::Tensor block_smids, torch::Tensor smid_counts, int num_blocks, int threads);
void locality_stream(
    torch::Tensor x,
    torch::Tensor block_sums,
    torch::Tensor work_items,
    torch::Tensor domain_offsets,
    torch::Tensor domain_lengths,
    torch::Tensor domain_heads,
    torch::Tensor smid_to_domain,
    torch::Tensor smid_counts,
    torch::Tensor domain_tile_counts,
    int chunk_elems,
    int inner_iters,
    int mode,
    int num_domains,
    int num_blocks,
    int threads);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("smid_probe", &smid_probe, "record block -> smid placement");
    m.def("locality_stream", &locality_stream, "stream chunks using baseline/global/domain queues");
}
"""


@dataclass(frozen=True)
class WorkQueues:
    work_items: torch.Tensor
    offsets: torch.Tensor
    lengths: torch.Tensor


def _load_ext():
    extra_cflags = ["-DGLOG_USE_GLOG_EXPORT"]
    extra_cuda_cflags = ["-O3", "--use_fast_math", *extra_cflags]
    return load_inline(
        name="machete_l2_locality_ext",
        cpp_sources=[CPP_SRC],
        cuda_sources=[CUDA_SRC],
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=bool(int(os.environ.get("MACHETE_LOCALITY_VERBOSE_BUILD", "0"))),
    )


def _device_sms() -> int:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def _default_smid_to_domain(num_domains: int, max_smids: int = 512) -> torch.Tensor:
    # Do not assume SM ID numbering is physically meaningful. This default is a
    # starting split; the smid probe output tells you whether it needs replacing.
    sms = _device_sms()
    host = torch.empty(max_smids, dtype=torch.int32)
    for smid in range(max_smids):
        if smid >= sms:
            host[smid] = num_domains - 1
        else:
            host[smid] = min(smid * num_domains // max(1, sms), num_domains - 1)
    return host.cuda()


def _make_work_queues(num_chunks: int, repeats: int, num_domains: int, owner: str) -> WorkQueues:
    buckets: list[list[int]] = [[] for _ in range(num_domains)]
    g = torch.Generator().manual_seed(123)
    random_owner = torch.randint(0, num_domains, (num_chunks,), generator=g).tolist()
    for _ in range(repeats):
        for chunk in range(num_chunks):
            if owner == "chunk":
                domain = min(chunk * num_domains // max(1, num_chunks), num_domains - 1)
            elif owner == "mod":
                domain = chunk % num_domains
            elif owner == "random":
                domain = int(random_owner[chunk])
            else:
                raise ValueError(f"unknown owner {owner!r}")
            buckets[domain].append(chunk)

    offsets = []
    lengths = []
    flat = []
    for bucket in buckets:
        offsets.append(len(flat))
        lengths.append(len(bucket))
        flat.extend(bucket)
    return WorkQueues(
        work_items=torch.tensor(flat, dtype=torch.int32, device="cuda"),
        offsets=torch.tensor(offsets, dtype=torch.int32, device="cuda"),
        lengths=torch.tensor(lengths, dtype=torch.int32, device="cuda"),
    )


def _mode_id(mode: str) -> int:
    return {"baseline": 0, "global": 1, "domain": 2}[mode]


def _time_ms(fn, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        out.append(start.elapsed_time(end))
    return out


def _print_ncu_hint(args: argparse.Namespace) -> None:
    metrics = ",".join(
        [
            "lts__t_sector_hit_rate.pct",
            "lts__t_bytes.sum",
            "dram__bytes.sum",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "smsp__warp_issue_stalled_long_scoreboard_per_warp_active",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        ]
    )
    cmd = (
        f"ncu --csv --target-processes all --log-file ncu_l2_locality_{args.mode}_{args.owner}.csv "
        f"--kernel-name regex:locality_stream_kernel "
        f"--metrics {metrics} "
        f"python benchmarks/kernels/benchmark_l2_locality.py --mode {args.mode} "
        f"--owner {args.owner} --chunks {args.chunks} --chunk-mb {args.chunk_mb} "
        f"--repeats {args.repeats} --inner-iters {args.inner_iters} "
        f"--num-domains {args.num_domains} --num-blocks {args.num_blocks} --threads {args.threads}"
    )
    print("\nNCU command:")
    print(cmd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "global", "domain"], default="baseline")
    parser.add_argument("--owner", choices=["chunk", "mod", "random"], default="chunk")
    parser.add_argument("--num-domains", type=int, default=2)
    parser.add_argument("--num-blocks", type=int, default=0, help="0 means one persistent CTA per SM")
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--chunks", type=int, default=256)
    parser.add_argument("--chunk-mb", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=64)
    parser.add_argument("--inner-iters", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--probe-smids", action="store_true")
    parser.add_argument("--ncu-hint", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    ext = _load_ext()
    sms = _device_sms()
    num_blocks = args.num_blocks or sms
    chunk_elems = int(args.chunk_mb * 1024 * 1024 // 4)
    if chunk_elems <= 0:
        raise ValueError("chunk-mb is too small")

    smid_to_domain = _default_smid_to_domain(args.num_domains)
    smid_counts = torch.zeros(512, dtype=torch.int32, device="cuda")
    block_smids = torch.empty(num_blocks, dtype=torch.int32, device="cuda")

    if args.probe_smids:
        ext.smid_probe(block_smids, smid_counts, num_blocks, args.threads)
        torch.cuda.synchronize()
        counts = smid_counts.cpu()
        observed = [(idx, int(v)) for idx, v in enumerate(counts.tolist()) if v]
        print(f"Observed {len(observed)} SM IDs with {num_blocks} CTAs:")
        print(observed)

    queues = _make_work_queues(args.chunks, args.repeats, args.num_domains, args.owner)
    torch.manual_seed(1234)
    x = torch.randn(args.chunks * chunk_elems, dtype=torch.float32, device="cuda")
    block_sums = torch.empty(num_blocks, dtype=torch.float32, device="cuda")
    domain_heads = torch.zeros(max(args.num_domains, 1), dtype=torch.int32, device="cuda")
    domain_tile_counts = torch.zeros(max(args.num_domains, 1), dtype=torch.int32, device="cuda")
    smid_counts.zero_()

    total_bytes = args.chunks * args.repeats * args.inner_iters * chunk_elems * 4

    def run_once() -> None:
        domain_heads.zero_()
        domain_tile_counts.zero_()
        smid_counts.zero_()
        ext.locality_stream(
            x,
            block_sums,
            queues.work_items,
            queues.offsets,
            queues.lengths,
            domain_heads,
            smid_to_domain,
            smid_counts,
            domain_tile_counts,
            chunk_elems,
            args.inner_iters,
            _mode_id(args.mode),
            args.num_domains,
            num_blocks,
            args.threads,
        )

    times = _time_ms(run_once, args.warmup, args.iters)
    checksum = float(block_sums.sum().item())
    tile_counts = domain_tile_counts.cpu().tolist()
    observed_smids = [(i, int(v)) for i, v in enumerate(smid_counts.cpu().tolist()) if v]

    mean_ms = statistics.mean(times)
    print(f"mode={args.mode} owner={args.owner} domains={args.num_domains}")
    print(f"sms={sms} blocks={num_blocks} threads={args.threads}")
    print(f"chunks={args.chunks} chunk_mb={args.chunk_mb} repeats={args.repeats} inner_iters={args.inner_iters}")
    print(f"time_ms mean={mean_ms:.3f} min={min(times):.3f} max={max(times):.3f}")
    print(f"effective_stream_GBps={total_bytes / (mean_ms * 1e6):.2f}")
    print(f"domain_tile_counts={tile_counts}")
    print(f"observed_smids={observed_smids[:32]}{' ...' if len(observed_smids) > 32 else ''}")
    print(f"checksum={checksum:.6e}")

    if args.ncu_hint:
        _print_ncu_hint(args)


if __name__ == "__main__":
    main()
