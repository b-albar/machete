#!/usr/bin/env python
"""Sweep staged TMA vs bulk cp.async copy/compute overlap.

This is intentionally synthetic.  Each case launches one Machete persistent
kernel with the same op shape:

    load global tile -> shared memory
    compute in shared memory
    store shared memory -> global tile

Only the transport changes:
    - TMA uses framework TMA descriptors and cute.nvgpu.cpasync.tma_partition.
    - cpbulk uses CopyBulkG2SOp/CopyBulkS2GOp inside the op load/store.
    - cpcoop uses cooperative 128-bit CopyG2SOp plus universal stores.

The compute loop is configurable so we can see where copy overhead dominates
and where data movement gets hidden by useful work.
"""

from __future__ import annotations

import argparse
import statistics

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass.cute.nvgpu.cpasync import (
    CopyBulkG2SOp,
    CopyBulkS2GOp,
    group_bulk_copy_modes,
)

from machete.megakernel.interpreter import mbarrier_arrive, mbarrier_arrive_expect_tx
from machete.megakernel.megakernel import Megakernel, MegakernelConfig
from machete.megakernel.ops import Op


ELEM_BYTES = 2


def _page_size(tile_bytes: int) -> int:
    return max(32768, 1 << (tile_bytes - 1).bit_length())


class _OverlapTmaOp(Op):
    reads = {"x": (None, ("M", "N"))}
    writes = {"y": (None, ("M", "N"))}
    tile = ("M",)

    tma_loads = {"x"}
    tma_stores = {"y"}

    @classmethod
    def schedule(cls, *, x, y, tile_m: int, compute_iters: int, page_size: int):
        ops = cls._schedule_single(tile_sizes={"M": tile_m}, x=x, y=y)
        ops.static_dims["compute_iters"] = compute_iters
        ops.static_dims["page_size"] = page_size
        return [ops]

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
        nbytes = Int32(self.tile_size_M * self.N * ELEM_BYTES)
        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, nbytes)
        cute.copy(x_tma, tG[(None, Int32(0), tile_M)], tS, tma_bar_ptr=mbar_ptr)

    @cute.jit
    def compute(self, page_ptr, tile_M, x, y):
        tidx = cute.arch.thread_idx()[0]
        total = Int32(self.tile_size_M * self.N)
        s = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout(total),
        )
        for i in range(tidx, total, self.threads_per_row):
            v = s[i].to(Float32)
            for _ in range(self.compute_iters):
                v = v * Float32(1.0001) + Float32(0.0001)
            s[i] = v.to(self.x_dtype)

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


class _OverlapCpAsyncBulkOp(Op):
    reads = {"x": (None, ("M", "N"))}
    writes = {"y": (None, ("M", "N"))}
    tile = ("M",)

    @classmethod
    def schedule(cls, *, x, y, tile_m: int, compute_iters: int, page_size: int):
        ops = cls._schedule_single(tile_sizes={"M": tile_m}, x=x, y=y)
        ops.static_dims["compute_iters"] = compute_iters
        ops.static_dims["copy_bits"] = tile_m * int(x.shape[1]) * ELEM_BYTES * 8
        ops.static_dims["page_size"] = page_size
        return [ops]

    @cute.jit
    def load(self, page_ptr, tile_M, x, y, work_mbar):
        g2s = cute.make_copy_atom(
            CopyBulkG2SOp(),
            self.x_dtype,
            num_bits_per_copy=self.copy_bits,
        )
        s = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout(self.tile_size_M * self.N),
        )
        g = cute.make_tensor(
            x.iterator + tile_M * Int32(self.tile_size_M * self.N),
            cute.make_layout(self.tile_size_M * self.N),
        )
        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
        mbarrier_arrive_expect_tx(work_mbar, Int32(self.tile_size_M * self.N * ELEM_BYTES))
        gsrc, sdst = group_bulk_copy_modes(g, s)
        cute.copy(g2s, gsrc, sdst, mbar_ptr=mbar_ptr)

    @cute.jit
    def compute(self, page_ptr, tile_M, x, y):
        tidx = cute.arch.thread_idx()[0]
        total = Int32(self.tile_size_M * self.N)
        s = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout(total),
        )
        for i in range(tidx, total, self.threads_per_row):
            v = s[i].to(Float32)
            for _ in range(self.compute_iters):
                v = v * Float32(1.0001) + Float32(0.0001)
            s[i] = v.to(self.x_dtype)

    @cute.jit
    def store(self, page_ptr, tile_M, x, y):
        s2g = cute.make_copy_atom(
            CopyBulkS2GOp(),
            self.y_dtype,
            num_bits_per_copy=self.copy_bits,
        )
        s = cute.make_tensor(
            cute.make_ptr(self.y_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout(self.tile_size_M * self.N),
        )
        g = cute.make_tensor(
            y.iterator + tile_M * Int32(self.tile_size_M * self.N),
            cute.make_layout(self.tile_size_M * self.N),
        )
        ssrc, gdst = group_bulk_copy_modes(s, g)
        cute.copy(s2g, ssrc, gdst)


class _OverlapCpAsyncCoopOp(Op):
    reads = {"x": (None, ("M", "N"))}
    writes = {"y": (None, ("M", "N"))}
    tile = ("M",)
    collective_non_tma_load = True

    @classmethod
    def schedule(cls, *, x, y, tile_m: int, compute_iters: int, page_size: int):
        if int(x.shape[1]) % 8 != 0:
            raise ValueError("cooperative cp.async path requires N divisible by 8")
        ops = cls._schedule_single(tile_sizes={"M": tile_m}, x=x, y=y)
        ops.static_dims["compute_iters"] = compute_iters
        ops.static_dims["copy_elems"] = 8
        ops.static_dims["copy_dim1"] = int(x.shape[1]) // 8
        ops.static_dims["copy_dim0"] = max(1, 32 // (int(x.shape[1]) // 8))
        ops.static_dims["page_size"] = page_size
        return [ops]

    @cute.jit
    def _copy_layout(self):
        return (
            cute.make_layout((self.copy_dim0, self.copy_dim1), stride=(self.copy_dim1, 1)),
            cute.make_layout((1, self.copy_elems)),
        )

    @cute.jit
    def load(self, page_ptr, tile_M, x, y, work_mbar):
        lane = cute.arch.lane_idx()
        g2s = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            self.x_dtype,
            num_bits_per_copy=128,
        )
        copy_thread_layout, copy_value_layout = self._copy_layout()
        tiled_copy = cute.make_tiled_copy_tv(g2s, copy_thread_layout, copy_value_layout)
        thr_copy = tiled_copy.get_slice(lane)
        s = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=16),
            cute.make_layout((self.tile_size_M, self.N), stride=(self.N, 1)),
        )
        g = cute.make_tensor(
            (x.iterator + tile_M * Int32(self.tile_size_M * self.N)).align(16),
            cute.make_layout((self.tile_size_M, self.N), stride=(self.N, 1)),
        )
        tG = thr_copy.partition_S(g)
        tS = thr_copy.partition_D(s)
        cute.copy(tiled_copy, tG, tS)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        with cute.arch.elect_one():
            mbarrier_arrive(work_mbar)

    @cute.jit
    def compute(self, page_ptr, tile_M, x, y):
        tidx = cute.arch.thread_idx()[0]
        total = Int32(self.tile_size_M * self.N)
        s = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout(total),
        )
        for i in range(tidx, total, self.threads_per_row):
            v = s[i].to(Float32)
            for _ in range(self.compute_iters):
                v = v * Float32(1.0001) + Float32(0.0001)
            s[i] = v.to(self.x_dtype)

    @cute.jit
    def store(self, page_ptr, tile_M, x, y):
        lane = cute.arch.lane_idx()
        s2g = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.y_dtype,
            num_bits_per_copy=128,
        )
        copy_thread_layout, copy_value_layout = self._copy_layout()
        tiled_copy = cute.make_tiled_copy_tv(s2g, copy_thread_layout, copy_value_layout)
        thr_copy = tiled_copy.get_slice(lane)
        s = cute.make_tensor(
            cute.make_ptr(self.y_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=16),
            cute.make_layout((self.tile_size_M, self.N), stride=(self.N, 1)),
        )
        g = cute.make_tensor(
            (y.iterator + tile_M * Int32(self.tile_size_M * self.N)).align(16),
            cute.make_layout((self.tile_size_M, self.N), stride=(self.N, 1)),
        )
        tS = thr_copy.partition_S(s)
        tG = thr_copy.partition_D(g)
        cute.copy(tiled_copy, tS, tG)


def _time_kernel(kernel: Megakernel, *, warmup: int, rep: int) -> tuple[float, float]:
    times = []
    kernel.run()
    torch.cuda.synchronize()
    for _ in range(warmup):
        kernel.run()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(rep):
        start.record()
        kernel.run()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.mean(times), min(times)


def _bench_one(kind: str, tile_bytes: int, compute_iters: int, args) -> dict[str, float]:
    n = args.n
    tile_m = tile_bytes // (n * ELEM_BYTES)
    if tile_m < 1:
        raise ValueError(f"tile_bytes={tile_bytes} is too small for N={n}")
    total_m = tile_m * args.tiles
    x = torch.randn(total_m, n, device="cuda", dtype=torch.float16)
    y = torch.empty_like(x)
    page_size = _page_size(tile_bytes)
    op_cls = {
        "tma": _OverlapTmaOp,
        "cpbulk": _OverlapCpAsyncBulkOp,
        "cpcoop": _OverlapCpAsyncCoopOp,
    }[kind]
    ops = op_cls.schedule(
        x=x,
        y=y,
        tile_m=tile_m,
        compute_iters=compute_iters,
        page_size=page_size,
    )
    kernel = Megakernel(
        ops,
        config=MegakernelConfig(
            threads_per_block=args.threads,
            num_pages=args.pages,
            page_size=page_size,
        ),
    )
    mean_ms, min_ms = _time_kernel(kernel, warmup=args.warmup, rep=args.rep)
    if args.verify:
        ref = x.float()
        for _ in range(compute_iters):
            ref = ref * 1.0001 + 0.0001
        torch.testing.assert_close(y.float(), ref, atol=2e-2, rtol=2e-2)
    bytes_total = 2 * total_m * n * ELEM_BYTES
    return {
        "mean_ms": mean_ms,
        "min_ms": min_ms,
        "gb_s": bytes_total / (min_ms * 1.0e-3) / 1.0e9,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-bytes", type=int, nargs="+", default=[2048, 4096, 8192, 16384, 32768, 65536])
    parser.add_argument("--compute-iters", type=int, nargs="+", default=[0, 4, 16, 64])
    parser.add_argument("--tiles", type=int, default=4096)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--pages", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=30)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--include-coop", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    name = torch.cuda.get_device_name()
    major, minor = torch.cuda.get_device_capability()
    print(f"GPU: {name}, capability=sm_{major}{minor}")
    if args.include_coop:
        print(
            "tile_bytes,compute_iters,"
            "pages,"
            "tma_min_ms,tma_GBps,"
            "cpbulk_min_ms,cpbulk_GBps,tma_speedup_vs_cpbulk,"
            "cpcoop_min_ms,cpcoop_GBps,tma_speedup_vs_cpcoop"
        )
    else:
        print(
            "tile_bytes,compute_iters,"
            "pages,"
            "tma_min_ms,tma_GBps,"
            "cpbulk_min_ms,cpbulk_GBps,tma_speedup_vs_cpbulk"
        )
    for tile_bytes in args.tile_bytes:
        if tile_bytes % (args.n * ELEM_BYTES) != 0:
            print(f"# skip tile_bytes={tile_bytes}: not divisible by N*elem_bytes={args.n * ELEM_BYTES}")
            continue
        for compute_iters in args.compute_iters:
            tma = _bench_one("tma", tile_bytes, compute_iters, args)
            cpbulk = _bench_one("cpbulk", tile_bytes, compute_iters, args)
            speedup_bulk = cpbulk["min_ms"] / tma["min_ms"]
            row = (
                f"{tile_bytes},{compute_iters},{args.pages},"
                f"{tma['min_ms']:.4f},{tma['gb_s']:.1f},"
                f"{cpbulk['min_ms']:.4f},{cpbulk['gb_s']:.1f},{speedup_bulk:.3f}"
            )
            if args.include_coop:
                cpcoop = _bench_one("cpcoop", tile_bytes, compute_iters, args)
                speedup_coop = cpcoop["min_ms"] / tma["min_ms"]
                row += f",{cpcoop['min_ms']:.4f},{cpcoop['gb_s']:.1f},{speedup_coop:.3f}"
            print(row)


if __name__ == "__main__":
    main()
