#!/usr/bin/env python
"""Measure decode-style activation staging overhead.

This isolates the specific question behind the rejected staged NVFP4 matvec:
does copying the small per-token activation row into shared memory help when
many output tiles reuse it?

The benchmark uses a simple BF16 dot product so the transport/framework effect
is visible without NVFP4 decode details:

    direct:       each output tile reads activation directly from global
    staged:       each output tile stages the same activation row
    ranged_stage: one staged activation serves a coalesced range of O tiles
"""

from __future__ import annotations

import argparse
import operator

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass.cute.nvgpu.cpasync import CopyBulkG2SOp, group_bulk_copy_modes

from machete.megakernel import Megakernel, MegakernelConfig, TileRange
from machete.megakernel.interpreter import mbarrier_arrive_expect_tx
from machete.megakernel.ops import DEFAULT_PAGE_SIZE, Op, PipelineSpec


class _DirectActivationDotOp(Op):
    reads = {
        "a": (None, ("B", "K")),
        "w": (None, ("O", "K")),
    }
    writes = {"y": (None, ("B", "O"))}
    tile = ("B", "O")
    dynamic_dims = ("B",)

    @classmethod
    def schedule(cls, *, a, w, y, tile_o: int):
        return [cls._schedule_single(tile_sizes={"B": 1, "O": tile_o}, a=a, w=w, y=y)]

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_O, a, w, y):
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        out_start = tile_O * Int32(self.tile_size_O)
        a_row = cute.make_tensor(
            a.iterator + tile_B * Int32(self.a_stride_B),
            cute.make_layout(self.K),
        )
        for local_o in range(warp, self.tile_size_O, self.threads_per_row // 32):
            out_idx = out_start + Int32(local_o)
            if out_idx < Int32(self.O):
                w_row = cute.make_tensor(
                    w.iterator + out_idx * Int32(self.w_stride_O),
                    cute.make_layout(self.K),
                )
                acc = Float32(0.0)
                k = lane
                while k < Int32(self.K):
                    acc = acc + a_row[k].to(Float32) * w_row[k].to(Float32)
                    k = k + Int32(32)
                total = cute.arch.warp_reduction(acc, operator.add)
                if lane == Int32(0):
                    y_row = cute.make_tensor(
                        y.iterator + tile_B * Int32(self.y_stride_B) + out_start,
                        cute.make_layout(self.tile_size_O),
                    )
                    y_row[local_o] = total.to(self.y_dtype)


class _StagedActivationDotOp(_DirectActivationDotOp):
    pipeline = PipelineSpec.range_capable(range_axis=1, range_end_axis=2)

    @classmethod
    def schedule(cls, *, a, w, y, tile_o: int, page_size: int, tile_range=None):
        ops = cls._schedule_single(
            tile_sizes={"B": 1, "O": tile_o},
            tile_range=tile_range,
            a=a,
            w=w,
            y=y,
        )
        activation_bytes = int(a.shape[1]) * a.element_size()
        if activation_bytes > page_size:
            raise ValueError(f"activation tile needs {activation_bytes}B, page_size={page_size}")
        ops.static_dims["activation_bytes"] = activation_bytes
        ops.static_dims["activation_copy_bits"] = activation_bytes * 8
        ops.static_dims["page_size"] = page_size
        return [ops]

    @cute.jit
    def load(self, page_ptr, tile_B, tile_O, a, w, y, work_mbar):
        g2s = cute.make_copy_atom(
            CopyBulkG2SOp(),
            self.a_dtype,
            num_bits_per_copy=self.activation_copy_bits,
        )
        s = cute.make_tensor(
            cute.make_ptr(self.a_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout(self.K),
        )
        g = cute.make_tensor(
            a.iterator + tile_B * Int32(self.a_stride_B),
            cute.make_layout(self.K),
        )
        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
        mbarrier_arrive_expect_tx(work_mbar, Int32(self.activation_bytes))
        gsrc, sdst = group_bulk_copy_modes(g, s)
        cute.copy(g2s, gsrc, sdst, mbar_ptr=mbar_ptr)

    @cute.jit
    def _compute_one_o_block(self, page_ptr, tile_B, block_o, w, y):
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        out_start = block_o * Int32(self.tile_size_O)
        a_row = cute.make_tensor(
            cute.make_ptr(self.a_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout(self.K),
        )
        for local_o in range(warp, self.tile_size_O, self.threads_per_row // 32):
            out_idx = out_start + Int32(local_o)
            if out_idx < Int32(self.O):
                w_row = cute.make_tensor(
                    w.iterator + out_idx * Int32(self.w_stride_O),
                    cute.make_layout(self.K),
                )
                acc = Float32(0.0)
                k = lane
                while k < Int32(self.K):
                    acc = acc + a_row[k].to(Float32) * w_row[k].to(Float32)
                    k = k + Int32(32)
                total = cute.arch.warp_reduction(acc, operator.add)
                if lane == Int32(0):
                    y_row = cute.make_tensor(
                        y.iterator + tile_B * Int32(self.y_stride_B) + out_start,
                        cute.make_layout(self.tile_size_O),
                    )
                    y_row[local_o] = total.to(self.y_dtype)

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_O, a, w, y):
        self._compute_one_o_block(page_ptr, tile_B, tile_O, w, y)


class _RangeOwnedStagedActivationDotOp(_StagedActivationDotOp):
    @cute.jit
    def load(self, page_ptr, tile_B, tile_O, tile_2, a, w, y, work_mbar):
        g2s = cute.make_copy_atom(
            CopyBulkG2SOp(),
            self.a_dtype,
            num_bits_per_copy=self.activation_copy_bits,
        )
        s = cute.make_tensor(
            cute.make_ptr(self.a_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout(self.K),
        )
        g = cute.make_tensor(
            a.iterator + tile_B * Int32(self.a_stride_B),
            cute.make_layout(self.K),
        )
        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
        mbarrier_arrive_expect_tx(work_mbar, Int32(self.activation_bytes))
        gsrc, sdst = group_bulk_copy_modes(g, s)
        cute.copy(g2s, gsrc, sdst, mbar_ptr=mbar_ptr)

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_O, tile_2, a, w, y):
        block_o = tile_O
        block_end = tile_2
        if block_end <= block_o:
            block_end = block_o + Int32(1)
        while block_o < block_end:
            self._compute_one_o_block(page_ptr, tile_B, block_o, w, y)
            block_o = block_o + Int32(1)


def _time_kernel(kernel, warmup: int, rep: int) -> float:
    kernel.compile()
    for _ in range(warmup):
        kernel.run(validate=False)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        kernel.run(validate=False)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep


def _make_kernel(kind: str, a, w, y, args):
    if kind == "direct":
        ops = _DirectActivationDotOp.schedule(a=a, w=w, y=y, tile_o=args.tile_o)
    elif kind == "staged":
        ops = _StagedActivationDotOp.schedule(
            a=a,
            w=w,
            y=y,
            tile_o=args.tile_o,
            page_size=args.page_size,
        )
    elif kind == "range_staged":
        ops = _RangeOwnedStagedActivationDotOp.schedule(
            a=a,
            w=w,
            y=y,
            tile_o=args.tile_o,
            page_size=args.page_size,
            tile_range=TileRange.coalesced("O", block_size=args.range_blocks),
        )
    else:
        raise ValueError(kind)
    return Megakernel(
        ops,
        config=MegakernelConfig(
            num_sms=args.sms,
            threads_per_block=args.threads,
            page_size=args.page_size,
            num_pages=args.pages,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=128)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--out", type=int, default=1024)
    parser.add_argument("--tile-o", type=int, default=16)
    parser.add_argument("--range-blocks", type=int, default=4)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--sms", type=int, default=None)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--pages", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=30)
    args = parser.parse_args()

    props = torch.cuda.get_device_properties(0)
    if args.sms is None:
        args.sms = props.multi_processor_count

    torch.manual_seed(123)
    a = torch.randn(args.batches, args.k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(args.out, args.k, device="cuda", dtype=torch.bfloat16)
    y = torch.empty(args.batches, args.out, device="cuda", dtype=torch.bfloat16)

    print(
        "kind,ms,tiles,instructions,batches,k,out,tile_o,range_blocks,pages",
        flush=True,
    )
    for kind in ("direct", "staged", "range_staged"):
        kernel = _make_kernel(kind, a, w, y, args)
        ms = _time_kernel(kernel, args.warmup, args.rep)
        print(
            f"{kind},{ms:.4f},{kernel.total_tiles},{kernel._num_instructions},"
            f"{args.batches},{args.k},{args.out},{args.tile_o},"
            f"{args.range_blocks},{args.pages}",
            flush=True,
        )


if __name__ == "__main__":
    main()
