#!/usr/bin/env python
"""Benchmark default vs overlap tile scheduling on a synthetic dependency chain."""

from __future__ import annotations

import argparse
import statistics

import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32

from machete.megakernel import Megakernel, MegakernelConfig, Op, OverlapTileScheduler
from machete.megakernel.interpreter import mbarrier_arrive_expect_tx


ELEM_BYTES = 2


class _TmaScaleProducer(Op):
    reads = {"x": (None, ("M", "N"))}
    writes = {"y": (None, ("M", "N"))}
    tile = ("M",)
    tma_loads = {"x"}
    tma_stores = {"y"}

    @classmethod
    def schedule(cls, *, x, y, tile_m: int, page_size: int):
        op = cls._schedule_single(tile_sizes={"M": tile_m}, x=x, y=y)
        op.static_dims["page_size"] = page_size
        return [op]

    @cute.jit
    def load(self, page_ptr, tile_M, x_tma, x_tma_gmem, work_mbar):
        s = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.N, self.tile_size_M)),
        )
        g = cute.local_tile(x_tma_gmem, (self.N, self.tile_size_M), (None, None))
        tS, tG = cute.nvgpu.cpasync.tma_partition(
            x_tma,
            Int32(0),
            cute.make_layout(1),
            cute.group_modes(s, 0, 2),
            cute.group_modes(g, 0, 2),
        )
        nbytes = Int32(self.N * self.tile_size_M * ELEM_BYTES)
        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, nbytes)
        cute.copy(x_tma, tG[(None, Int32(0), tile_M)], tS, tma_bar_ptr=mbar_ptr)

    @cute.jit
    def compute(self, page_ptr, tile_M, x, y):
        tidx = cute.arch.thread_idx()[0]
        total = Int32(self.N * self.tile_size_M)
        s = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout(total),
        )
        for i in range(tidx, total, self.threads_per_row):
            s[i] = (s[i].to(Float32) * Float32(1.001)).to(self.x_dtype)

    @cute.jit
    def store(self, page_ptr, tile_M, y_tma, y_tma_gmem):
        s = cute.make_tensor(
            cute.make_ptr(self.y_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.N, self.tile_size_M)),
        )
        g = cute.local_tile(y_tma_gmem, (self.N, self.tile_size_M), (None, None))
        tS, tG = cute.nvgpu.cpasync.tma_partition(
            y_tma,
            Int32(0),
            cute.make_layout(1),
            cute.group_modes(s, 0, 2),
            cute.group_modes(g, 0, 2),
        )
        with cute.arch.elect_one():
            cute.copy(y_tma, tS, tG[(None, Int32(0), tile_M)])


class _DirectComputeConsumer(Op):
    reads = {"y": (None, ("M", "N"))}
    writes = {"z": (None, ("M", "N"))}
    tile = ("M",)

    @classmethod
    def schedule(cls, *, y, z, tile_m: int, compute_iters: int, page_size: int):
        op = cls._schedule_single(tile_sizes={"M": tile_m}, y=y, z=z)
        op.static_dims["compute_iters"] = compute_iters
        op.static_dims["page_size"] = page_size
        return [op]

    @cute.jit
    def compute(self, page_ptr, tile_M, y, z):
        tidx = cute.arch.thread_idx()[0]
        base = tile_M * Int32(self.tile_size_M * self.N)
        total = Int32(self.tile_size_M * self.N)
        y_flat = cute.make_tensor(y.iterator + base, cute.make_layout(total))
        z_flat = cute.make_tensor(z.iterator + base, cute.make_layout(total))
        for i in range(tidx, total, self.threads_per_row):
            v = y_flat[i].to(Float32)
            for _ in range(self.compute_iters):
                v = v * Float32(1.0001) + Float32(0.0001)
            z_flat[i] = v.to(self.z_dtype)


def _time(kernel: Megakernel, *, warmup: int, rep: int) -> tuple[float, float]:
    kernel.compile()
    kernel.run(validate=False)
    torch.cuda.synchronize()
    for _ in range(warmup):
        kernel.run(validate=False)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(rep):
        start.record()
        kernel.run(validate=False)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.mean(times), min(times)


def _build_kernel(args, scheduler):
    tile_m = args.tile_bytes // (args.n * ELEM_BYTES)
    total_m = tile_m * args.tiles
    x = torch.randn(total_m, args.n, device="cuda", dtype=torch.float16)
    y = torch.empty_like(x)
    z = torch.empty_like(x)
    ops = []
    ops += _TmaScaleProducer.schedule(
        x=x,
        y=y,
        tile_m=tile_m,
        page_size=args.page_size,
    )
    ops += _DirectComputeConsumer.schedule(
        y=y,
        z=z,
        tile_m=tile_m,
        compute_iters=args.compute_iters,
        page_size=args.page_size,
    )
    kernel = Megakernel(
        ops,
        config=MegakernelConfig(
            num_sms=args.sms,
            threads_per_block=args.threads,
            page_size=args.page_size,
            num_pages=args.pages,
        ),
        scheduler=scheduler,
    )
    return kernel, x, y, z


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-bytes", type=int, default=32768)
    parser.add_argument("--tiles", type=int, default=2048)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--compute-iters", type=int, default=64)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--sms", type=int, default=None)
    parser.add_argument("--page-size", type=int, default=32768)
    parser.add_argument("--pages", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=30)
    args = parser.parse_args()

    if args.tile_bytes % (args.n * ELEM_BYTES) != 0:
        raise SystemExit("tile-bytes must be divisible by n * element_size")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    if args.sms is None:
        args.sms = torch.cuda.get_device_properties(0).multi_processor_count

    print(f"GPU: {torch.cuda.get_device_name()}, sm={torch.cuda.get_device_capability()}")
    print("scheduler,mean_ms,min_ms,instructions,tiles,tile_bytes,compute_iters,pages,sms")
    for name, scheduler in (
        ("backward", None),
        ("overlap", OverlapTileScheduler()),
    ):
        kernel, _x, _y, _z = _build_kernel(args, scheduler)
        mean_ms, min_ms = _time(kernel, warmup=args.warmup, rep=args.rep)
        print(
            f"{name},{mean_ms:.4f},{min_ms:.4f},{kernel._num_instructions},"
            f"{kernel.total_tiles},{args.tile_bytes},{args.compute_iters},"
            f"{args.pages},{args.sms}",
            flush=True,
        )


if __name__ == "__main__":
    main()
