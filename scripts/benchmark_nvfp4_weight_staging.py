#!/usr/bin/env python
"""Benchmark Hazy-style staged NVFP4 weight/scale streaming for decode matvec.

This is an experiment, not a production schedule.  It compares the existing
direct NVFP4 replay matvec against an op-owned streaming variant:

    load warp:   copy packed weights + scales for consecutive O tiles
    compute:     consume staged weights from a double buffer while loader
                 prefetches the next O tile

The activation stays in global memory because prior measurements showed
activation staging is slower than direct cached reads for decode.
"""

from __future__ import annotations

import argparse
import operator

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass.cute.nvgpu.cpasync import CopyBulkG2SOp, group_bulk_copy_modes

from machete.kernels.decode_matvec.sm120 import (
    MatvecNvfp4Sm120Op,
    _MatvecNvfp4Sm120Base,
)
from machete.megakernel import Megakernel, MegakernelConfig, TileRange
from machete.megakernel.interpreter import (
    mbarrier_arrive,
    mbarrier_arrive_expect_tx,
    mbarrier_init,
    mbarrier_init_fence,
    mbarrier_inval,
    mbarrier_wait,
    named_barrier_sync,
)
from machete.megakernel.ops import DEFAULT_PAGE_SIZE, PipelineSpec
from machete.quantization import quantize_nvfp4_weight


class StagedWeightMatvecNvfp4Sm120Op(_MatvecNvfp4Sm120Base):
    """NVFP4 matvec with op-owned packed weight/scale streaming."""

    pipeline = PipelineSpec.streaming(
        range_axis=2,
        range_end_axis=3,
        range_block_size=1,
        coalesce_ranges=True,
    )
    writes = {"y": (None, ("B", "S", "O"))}

    @classmethod
    def schedule(
        cls,
        tile_sizes=None,
        page_size=DEFAULT_PAGE_SIZE,
        group_size=32,
        range_blocks=4,
        range_stride=1,
        **tensors,
    ):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", 1)
        tile_sizes.setdefault("O", 16)
        if tensors["weight_packed"].shape[1] * 2 != tensors["a"].shape[-1]:
            raise ValueError("weight_packed K2 must equal a.K / 2")
        if tensors["weight_scales"].shape[1] != tensors["a"].shape[-1] // group_size:
            raise ValueError("weight_scales G must equal a.K / group_size")
        tile_range = (
            TileRange.strided("O", stride=range_stride, block_size=range_blocks)
            if range_stride > 1
            else TileRange.coalesced("O", block_size=range_blocks)
        )
        op = cls._schedule_single(
            tile_sizes=tile_sizes,
            tile_range=tile_range,
            **tensors,
        )
        packed_bytes = tile_sizes["O"] * int(tensors["weight_packed"].shape[1])
        scale_offset = ((packed_bytes + 127) // 128) * 128
        scale_bytes = (
            tile_sizes["O"]
            * int(tensors["weight_scales"].shape[1])
            * tensors["weight_scales"].element_size()
        )
        buf_stride = ((scale_offset + scale_bytes + 127) // 128) * 128
        required = max(1, range_blocks) * buf_stride
        page_size = max(page_size, required)
        op.static_dims["page_size"] = page_size
        op.static_dims["group_size"] = group_size
        op.static_dims["reduction_tile_K"] = min(512, tensors["a"].shape[-1])
        op.static_dims["packed_bytes"] = packed_bytes
        op.static_dims["scale_offset"] = scale_offset
        op.static_dims["scale_bytes"] = scale_bytes
        op.static_dims["buf_stride"] = buf_stride
        op.static_dims["range_blocks"] = range_blocks
        op.static_dims["range_stride"] = range_stride
        op.static_dims["mbar_offset"] = required
        op.static_dims["packed_copy_bits"] = packed_bytes * 8
        op.static_dims["scale_copy_bits"] = scale_bytes * 8
        return [op]

    @cute.jit
    def _copy_weight_block(
        self,
        page_ptr,
        buf_idx,
        block_o,
        weight_packed,
        weight_scales,
        mbar_ptr,
    ):
        buf_base = page_ptr + buf_idx * Int32(self.buf_stride)
        packed_copy = cute.make_copy_atom(
            CopyBulkG2SOp(),
            self.weight_packed_dtype,
            num_bits_per_copy=self.packed_copy_bits,
        )
        scale_copy = cute.make_copy_atom(
            CopyBulkG2SOp(),
            self.weight_scales_dtype,
            num_bits_per_copy=self.scale_copy_bits,
        )
        s_packed = cute.make_tensor(
            cute.make_ptr(self.weight_packed_dtype, buf_base, cute.AddressSpace.smem),
            cute.make_layout(self.tile_size_O * self.K2),
        )
        g_packed = cute.make_tensor(
            weight_packed.iterator + block_o * Int32(self.tile_size_O * self.K2),
            cute.make_layout(self.tile_size_O * self.K2),
        )
        s_scales = cute.make_tensor(
            cute.make_ptr(
                self.weight_scales_dtype,
                buf_base + Int32(self.scale_offset),
                cute.AddressSpace.smem,
            ),
            cute.make_layout(self.tile_size_O * self.G),
        )
        g_scales = cute.make_tensor(
            weight_scales.iterator + block_o * Int32(self.tile_size_O * self.G),
            cute.make_layout(self.tile_size_O * self.G),
        )
        gps, sps = group_bulk_copy_modes(g_packed, s_packed)
        gss, sss = group_bulk_copy_modes(g_scales, s_scales)
        cute.copy(packed_copy, gps, sps, mbar_ptr=mbar_ptr)
        cute.copy(scale_copy, gss, sss, mbar_ptr=mbar_ptr)

    @cute.jit
    def load(self, page_ptr, tile_B, tile_S, tile_O, tile_3, tile_4,
             a, weight_packed, weight_scales, y, work_mbar):
        block_o = tile_O
        range_stride = tile_4
        if range_stride < Int32(1):
            range_stride = Int32(1)
        block_end = tile_3
        if range_stride == Int32(1):
            block_end = tile_3
            if block_end <= block_o:
                block_end = block_o + Int32(1)
        else:
            range_count = tile_3
            if range_count <= Int32(0):
                range_count = Int32(1)
            block_end = block_o + range_count * range_stride
        iter_idx = Int32(0)
        total_bytes = Int32(0)
        probe_o = block_o
        while probe_o < block_end:
            total_bytes = total_bytes + Int32(self.packed_bytes + self.scale_bytes)
            probe_o = probe_o + range_stride
        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, total_bytes)
        while block_o < block_end:
            self._copy_weight_block(
                page_ptr,
                iter_idx,
                block_o,
                weight_packed,
                weight_scales,
                mbar_ptr,
            )
            block_o = block_o + range_stride
            iter_idx = iter_idx + Int32(1)

    @cute.jit
    def _dot_staged(self, tile_B, row_idx, local_o, a, packed_row, scale_row):
        a_base = tile_B * Int32(self.a_stride_B) + row_idx * Int32(self.a_stride_S)
        a_row = cute.make_tensor(a.iterator + a_base, cute.make_layout(self.K))
        return self._dot_nvfp4(a_row, packed_row, scale_row)

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_O, tile_3, tile_4,
                a, weight_packed, weight_scales, y):
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.threads_per_row // 32
        row_start = tile_S * Int32(self.tile_size_S)

        block_o = tile_O
        range_stride = tile_4
        if range_stride < Int32(1):
            range_stride = Int32(1)
        block_end = tile_3
        if range_stride == Int32(1):
            block_end = tile_3
            if block_end <= block_o:
                block_end = block_o + Int32(1)
        else:
            range_count = tile_3
            if range_count <= Int32(0):
                range_count = Int32(1)
            block_end = block_o + range_count * range_stride
        iter_idx = Int32(0)
        while block_o < block_end:
            buf_base = page_ptr + iter_idx * Int32(self.buf_stride)
            s_packed = cute.make_tensor(
                cute.make_ptr(self.weight_packed_dtype, buf_base, cute.AddressSpace.smem),
                cute.make_layout((self.tile_size_O, self.K2), stride=(self.K2, 1)),
            )
            s_scales = cute.make_tensor(
                cute.make_ptr(
                    self.weight_scales_dtype,
                    buf_base + Int32(self.scale_offset),
                    cute.AddressSpace.smem,
                ),
                cute.make_layout((self.tile_size_O, self.G), stride=(self.G, 1)),
            )
            out_start = block_o * Int32(self.tile_size_O)
            for local_work in range(warp_idx, self.tile_size_S * self.tile_size_O, num_warps):
                local_row = local_work // self.tile_size_O
                local_o = local_work - local_row * self.tile_size_O
                row_idx = row_start + Int32(local_row)
                if row_idx < Int32(self.S):
                    out_idx = out_start + Int32(local_o)
                    if out_idx < Int32(self.O):
                        total = self._dot_staged(
                            tile_B,
                            row_idx,
                            Int32(local_o),
                            a,
                            s_packed[(Int32(local_o), None)],
                            s_scales[(Int32(local_o), None)],
                        )
                        if lane_idx == Int32(0):
                            y_base = tile_B * Int32(self.y_stride_B) + row_idx * Int32(self.y_stride_S)
                            y_tile = cute.make_tensor(
                                y.iterator + y_base + out_start,
                                cute.make_layout(self.tile_size_O),
                            )
                            y_tile[local_o] = total.to(self.y_dtype)
            block_o = block_o + range_stride
            iter_idx = iter_idx + Int32(1)


class TmaStagedWeightMatvecNvfp4Sm120Op(StagedWeightMatvecNvfp4Sm120Op):
    """Same experiment as StagedWeightMatvecNvfp4Sm120Op, but uses TMA loads."""

    tma_loads = {"weight_packed", "weight_scales"}

    @classmethod
    def get_tma_tile_shape(cls, tensor_name, tile_sizes, static_dims):
        if tensor_name == "weight_packed":
            return (tile_sizes["O"], static_dims["K2"])
        if tensor_name == "weight_scales":
            return (tile_sizes["O"], static_dims["G"])
        return None

    @classmethod
    def get_tma_smem_layout_src(cls, tensor_name, tma_tile_shape, tile_sizes, static_dims):
        if tensor_name == "weight_packed":
            k2, tile_o = tma_tile_shape
            return f"cute.make_layout(({k2}, {tile_o}), stride=(1, {k2}))"
        if tensor_name == "weight_scales":
            g, tile_o = tma_tile_shape
            return f"cute.make_layout(({g}, {tile_o}), stride=(1, {g}))"
        return None

    @cute.jit
    def _copy_weight_block_tma(
        self,
        page_ptr,
        buf_idx,
        block_o,
        weight_packed_tma,
        weight_packed_tma_gmem,
        weight_scales_tma,
        weight_scales_tma_gmem,
        mbar_ptr,
    ):
        buf_base = page_ptr + buf_idx * Int32(self.buf_stride)
        s_packed = cute.make_tensor(
            cute.make_ptr(self.weight_packed_dtype, buf_base, cute.AddressSpace.smem),
            cute.make_layout((self.K2, self.tile_size_O), stride=(1, self.K2)),
        )
        g_packed = cute.local_tile(
            weight_packed_tma_gmem,
            (self.K2, self.tile_size_O),
            (None, None),
        )
        tPsP, tPgP = cute.nvgpu.cpasync.tma_partition(
            weight_packed_tma,
            Int32(0),
            cute.make_layout(1),
            cute.group_modes(s_packed, 0, 2),
            cute.group_modes(g_packed, 0, 2),
        )
        s_scales = cute.make_tensor(
            cute.make_ptr(
                self.weight_scales_dtype,
                buf_base + Int32(self.scale_offset),
                cute.AddressSpace.smem,
            ),
            cute.make_layout((self.G, self.tile_size_O), stride=(1, self.G)),
        )
        g_scales = cute.local_tile(
            weight_scales_tma_gmem,
            (self.G, self.tile_size_O),
            (None, None),
        )
        tSsS, tSgS = cute.nvgpu.cpasync.tma_partition(
            weight_scales_tma,
            Int32(0),
            cute.make_layout(1),
            cute.group_modes(s_scales, 0, 2),
            cute.group_modes(g_scales, 0, 2),
        )
        cute.copy(
            weight_packed_tma,
            tPgP[(None, Int32(0), block_o)],
            tPsP,
            tma_bar_ptr=mbar_ptr,
        )
        cute.copy(
            weight_scales_tma,
            tSgS[(None, Int32(0), block_o)],
            tSsS,
            tma_bar_ptr=mbar_ptr,
        )

    @cute.jit
    def load(self, page_ptr, tile_B, tile_S, tile_O, tile_3, tile_4,
             a, y,
             weight_packed_tma, weight_packed_tma_gmem,
             weight_scales_tma, weight_scales_tma_gmem,
             work_mbar):
        block_o = tile_O
        range_stride = tile_4
        if range_stride < Int32(1):
            range_stride = Int32(1)
        block_end = tile_3
        if range_stride == Int32(1):
            block_end = tile_3
            if block_end <= block_o:
                block_end = block_o + Int32(1)
        else:
            range_count = tile_3
            if range_count <= Int32(0):
                range_count = Int32(1)
            block_end = block_o + range_count * range_stride
        iter_idx = Int32(0)
        total_bytes = Int32(0)
        probe_o = block_o
        while probe_o < block_end:
            total_bytes = total_bytes + Int32(self.packed_bytes + self.scale_bytes)
            probe_o = probe_o + range_stride
        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, total_bytes)
        while block_o < block_end:
            self._copy_weight_block_tma(
                page_ptr,
                iter_idx,
                block_o,
                weight_packed_tma,
                weight_packed_tma_gmem,
                weight_scales_tma,
                weight_scales_tma_gmem,
                mbar_ptr,
            )
            block_o = block_o + range_stride
            iter_idx = iter_idx + Int32(1)


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


def _kernel_for(kind: str, a, qweight, y, args):
    if kind == "direct":
        ops = MatvecNvfp4Sm120Op.schedule(
            a=a,
            weight_packed=qweight.packed,
            weight_scales=qweight.scales,
            y=y,
            tile_sizes={"S": args.seq, "O": args.tile_o},
            page_size=args.page_size,
            group_size=args.group_size,
        )
    elif kind == "staged_weight":
        ops = StagedWeightMatvecNvfp4Sm120Op.schedule(
            a=a,
            weight_packed=qweight.packed,
            weight_scales=qweight.scales,
            y=y,
            tile_sizes={"S": args.seq, "O": args.tile_o},
            page_size=args.page_size,
            group_size=args.group_size,
            range_blocks=args.range_blocks,
            range_stride=args.range_stride,
        )
    elif kind == "tma_staged_weight":
        ops = TmaStagedWeightMatvecNvfp4Sm120Op.schedule(
            a=a,
            weight_packed=qweight.packed,
            weight_scales=qweight.scales,
            y=y,
            tile_sizes={"S": args.seq, "O": args.tile_o},
            page_size=args.page_size,
            group_size=args.group_size,
            range_blocks=args.range_blocks,
            range_stride=args.range_stride,
        )
    else:
        raise ValueError(kind)
    kernel_page_size = max(args.page_size, *(int(op.static_dims.get("page_size", 0)) for op in ops))
    return Megakernel(
        ops,
        config=MegakernelConfig(
            num_sms=args.sms,
            threads_per_block=args.threads,
            page_size=kernel_page_size,
            num_pages=args.pages,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=1)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--out", type=int, default=1024)
    parser.add_argument("--tile-o", type=int, default=16)
    parser.add_argument("--range-blocks", type=int, default=4)
    parser.add_argument("--range-stride", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--sms", type=int, default=None)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--pages", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=30)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--kinds",
        nargs="+",
        default=["direct", "staged_weight", "tma_staged_weight"],
        choices=["direct", "staged_weight", "tma_staged_weight"],
    )
    args = parser.parse_args()

    props = torch.cuda.get_device_properties(0)
    if args.sms is None:
        args.sms = props.multi_processor_count

    torch.manual_seed(11)
    a = torch.randn(args.batch, args.seq, args.k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(args.out, args.k, device="cuda", dtype=torch.bfloat16) * 0.05
    qweight = quantize_nvfp4_weight(weight, group_size=args.group_size)
    y_direct = torch.empty(args.batch, args.seq, args.out, device="cuda", dtype=torch.bfloat16)
    y_staged = torch.empty_like(y_direct)

    print("kind,ms,tiles,instructions,batch,seq,k,out,tile_o,range_blocks,pages", flush=True)
    results = {}
    y_by_kind = {
        "direct": y_direct,
        "staged_weight": y_staged,
        "tma_staged_weight": torch.empty_like(y_direct),
    }
    for kind in args.kinds:
        y = y_by_kind[kind]
        print(f"# running {kind}", flush=True)
        kernel = _kernel_for(kind, a, qweight, y, args)
        ms = _time_kernel(kernel, args.warmup, args.rep)
        results[kind] = ms
        print(
            f"{kind},{ms:.4f},{kernel.total_tiles},{kernel._num_instructions},"
            f"{args.batch},{args.seq},{args.k},{args.out},{args.tile_o},"
            f"{args.range_blocks},{args.pages}",
            flush=True,
        )

    if args.verify:
        if "staged_weight" in args.kinds:
            torch.testing.assert_close(y_staged.float(), y_direct.float(), atol=0, rtol=0)
        if "tma_staged_weight" in args.kinds:
            torch.testing.assert_close(y_by_kind["tma_staged_weight"].float(), y_direct.float(), atol=0, rtol=0)
    if "direct" in results and "staged_weight" in results:
        print(f"speedup_staged_vs_direct,{results['direct'] / results['staged_weight']:.3f}", flush=True)
    if "direct" in results and "tma_staged_weight" in results:
        print(f"speedup_tma_staged_vs_direct,{results['direct'] / results['tma_staged_weight']:.3f}", flush=True)


if __name__ == "__main__":
    main()
