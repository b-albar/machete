#!/usr/bin/env python
"""Minimal SASS repro for TMA S2G store serialization.

The script compiles a tiny Megakernel that loads a tile into shared memory,
touches it, and stores it back.  It keeps CuTe DSL artifacts, disassembles the
generated cubin, and reports whether PTXAS emitted the suspicious
UTMASTG/PLOP3/BRA.U.ANY serialization loop.

Useful variants:
  tma     - tensor TMA load + tensor TMA store
  tma3d   - 3D tensor TMA store shaped like Qwen/RMS/GEMM outputs
  chunked - tensor TMA load + several tensor TMA stores in one elected region
  two_tma - two tensor-TMA ops in one megakernel, exercising store dispatch
  many_tma - many tensor-TMA ops in one megakernel, stressing dispatch shape
  many_distinct - many distinct tensor-TMA op classes, stressing local dispatch
  postwork - tensor TMA store followed by all-lane scalar work in store()
  lane0   - same as tma, but store guarded by lane_idx() == 0
  cpbulk  - bulk G2S/S2G copy, no tensor TMA store
  cpcoop  - cooperative cp.async load + universal stores
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("CUTE_DSL_ARCH", "sm_120a")
os.environ.setdefault("CUTE_DSL_KEEP", "all")
os.environ.setdefault("CUTE_DSL_LINEINFO", "1")
os.environ.setdefault("CUTE_DSL_NO_CACHE", "1")

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32

from cutlass.cute.nvgpu.cpasync import (
    CopyBulkG2SOp,
    CopyBulkS2GOp,
)
from machete.megakernel import Megakernel, MegakernelConfig
from machete.megakernel.interpreter import (
    mbarrier_arrive,
    mbarrier_arrive_expect_tx,
)
from machete.megakernel.ops import Op


ELEM_BYTES = 2


def _page_size(tile_bytes: int) -> int:
    return max(32768, 1 << (tile_bytes - 1).bit_length())


class _TmaStoreOp(Op):
    reads = {"x": (None, ("M", "N"))}
    writes = {"y": (None, ("M", "N"))}
    tile = ("M",)
    tma_loads = {"x"}
    tma_stores = {"y"}

    @classmethod
    def schedule(cls, *, x, y, tile_m: int, page_size: int, lane0_store: bool = False):
        op = cls._schedule_single(tile_sizes={"M": tile_m}, x=x, y=y)
        op.static_dims["page_size"] = page_size
        op.static_dims["lane0_store"] = lane0_store
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
            s[i] = (s[i].to(Float32) + Float32(1.0)).to(self.x_dtype)

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
        if self.lane0_store:
            if cute.arch.lane_idx() == Int32(0):
                cute.copy(y_tma, tS, tG[(None, Int32(0), tile_M)])
        else:
            with cute.arch.elect_one():
                cute.copy(y_tma, tS, tG[(None, Int32(0), tile_M)])


class _Tma3DStoreOp(Op):
    reads = {"x": (None, ("B", "M", "N"))}
    writes = {"y": (None, ("B", "M", "N"))}
    tile = ("B", "M")
    tma_loads = {"x"}
    tma_stores = {"y"}

    @classmethod
    def schedule(cls, *, x, y, tile_m: int, page_size: int):
        op = cls._schedule_single(tile_sizes={"B": 1, "M": tile_m}, x=x, y=y)
        op.static_dims["page_size"] = page_size
        return [op]

    @cute.jit
    def load(self, page_ptr, tile_B, tile_M, x_tma, x_tma_gmem, work_mbar):
        s = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.N, self.tile_size_M, 1)),
        )
        g = cute.local_tile(
            x_tma_gmem,
            (self.N, self.tile_size_M, 1),
            (None, None, None),
        )
        tS, tG = cute.nvgpu.cpasync.tma_partition(
            x_tma,
            Int32(0),
            cute.make_layout(1),
            cute.group_modes(s, 0, 3),
            cute.group_modes(g, 0, 3),
        )
        nbytes = Int32(self.tile_size_M * self.N * ELEM_BYTES)
        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, nbytes)
        cute.copy(x_tma, tG[(None, Int32(0), tile_M, tile_B)], tS, tma_bar_ptr=mbar_ptr)

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_M, x, y):
        tidx = cute.arch.thread_idx()[0]
        total = Int32(self.tile_size_M * self.N)
        s = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout(total),
        )
        for i in range(tidx, total, self.threads_per_row):
            s[i] = (s[i].to(Float32) + Float32(1.0)).to(self.x_dtype)

    @cute.jit
    def store(self, page_ptr, tile_B, tile_M, y_tma, y_tma_gmem):
        s = cute.make_tensor(
            cute.make_ptr(self.y_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.N, self.tile_size_M, 1)),
        )
        g = cute.local_tile(
            y_tma_gmem,
            (self.N, self.tile_size_M, 1),
            (None, None, None),
        )
        tS, tG = cute.nvgpu.cpasync.tma_partition(
            y_tma,
            Int32(0),
            cute.make_layout(1),
            cute.group_modes(s, 0, 3),
            cute.group_modes(g, 0, 3),
        )
        with cute.arch.elect_one():
            cute.copy(y_tma, tS, tG[(None, Int32(0), tile_M, tile_B)])


class _TmaStorePostWorkOp(_Tma3DStoreOp):
    scratch = (cutlass.Float32, ("B", "M", "N"))

    reads = {"x": (None, ("B", "M", "N"))}
    writes = {
        "y": (None, ("B", "M", "N")),
        "scratch": (cutlass.Float32, ("B", "M", "N")),
    }
    tile = ("B", "M")
    tma_loads = {"x"}
    tma_stores = {"y"}

    @classmethod
    def schedule(cls, *, x, y, scratch, tile_m: int, page_size: int):
        op = cls._schedule_single(tile_sizes={"B": 1, "M": tile_m}, x=x, y=y, scratch=scratch)
        op.static_dims["page_size"] = page_size
        return [op]

    @cute.jit
    def store(self, page_ptr, tile_B, tile_M, scratch, y_tma, y_tma_gmem):
        s = cute.make_tensor(
            cute.make_ptr(self.y_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.N, self.tile_size_M, 1)),
        )
        g = cute.local_tile(
            y_tma_gmem,
            (self.N, self.tile_size_M, 1),
            (None, None, None),
        )
        tS, tG = cute.nvgpu.cpasync.tma_partition(
            y_tma,
            Int32(0),
            cute.make_layout(1),
            cute.group_modes(s, 0, 3),
            cute.group_modes(g, 0, 3),
        )
        with cute.arch.elect_one():
            cute.copy(y_tma, tS, tG[(None, Int32(0), tile_M, tile_B)])

        tidx = cute.arch.thread_idx()[0]
        total = Int32(self.tile_size_M * self.N)
        g_scratch = cute.make_tensor(
            scratch.iterator + tile_M * Int32(self.tile_size_M * self.N),
            cute.make_layout(total),
        )
        for i in range(tidx, total, self.threads_per_row):
            g_scratch[i] = Float32(0.0)


_DISTINCT_TMA_OPS = tuple(
    type(f"_DistinctTmaStoreOp{i}", (_TmaStoreOp,), {})
    for i in range(32)
)


class _ChunkedTmaStoreOp(Op):
    reads = {"x": (None, ("M", "N"))}
    writes = {"y": (None, ("M", "N"))}
    tile = ("M",)
    tma_loads = {"x"}
    tma_stores = {"y"}

    @classmethod
    def schedule(cls, *, x, y, tile_m: int, page_size: int, chunk_n: int):
        if int(x.shape[1]) % chunk_n != 0:
            raise ValueError("N must be divisible by chunk_n")
        op = cls._schedule_single(tile_sizes={"M": tile_m}, x=x, y=y)
        op.static_dims["page_size"] = page_size
        op.static_dims["chunk_n"] = chunk_n
        op.static_dims["num_chunks"] = int(x.shape[1]) // chunk_n
        op.static_dims["chunk_stride_bytes"] = tile_m * chunk_n * ELEM_BYTES
        return [op]

    @cute.jit
    def load(self, page_ptr, tile_M, x_tma, x_tma_gmem, work_mbar):
        nbytes = Int32(self.tile_size_M * self.N * ELEM_BYTES)
        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, nbytes)
        for wi in range(self.num_chunks):
            s = cute.make_tensor(
                cute.make_ptr(
                    self.x_dtype,
                    page_ptr + Int32(wi * self.chunk_stride_bytes),
                    cute.AddressSpace.smem,
                ),
                cute.make_layout((self.chunk_n, self.tile_size_M)),
            )
            g = cute.local_tile(
                x_tma_gmem,
                (self.chunk_n, self.tile_size_M),
                (Int32(wi), tile_M),
            )
            tS, tG = cute.nvgpu.cpasync.tma_partition(
                x_tma,
                Int32(0),
                cute.make_layout(1),
                cute.group_modes(s, 0, 2),
                cute.group_modes(g, 0, 2),
            )
            cute.copy(x_tma, tG, tS, tma_bar_ptr=mbar_ptr)

    @cute.jit
    def compute(self, page_ptr, tile_M, x, y):
        tidx = cute.arch.thread_idx()[0]
        total = Int32(self.tile_size_M * self.N)
        s = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout(total),
        )
        for i in range(tidx, total, self.threads_per_row):
            s[i] = (s[i].to(Float32) + Float32(1.0)).to(self.x_dtype)

    @cute.jit
    def store(self, page_ptr, tile_M, y_tma, y_tma_gmem):
        with cute.arch.elect_one():
            for wi in range(self.num_chunks):
                s = cute.make_tensor(
                    cute.make_ptr(
                        self.y_dtype,
                        page_ptr + Int32(wi * self.chunk_stride_bytes),
                        cute.AddressSpace.smem,
                    ),
                    cute.make_layout((self.chunk_n, self.tile_size_M)),
                )
                g = cute.local_tile(
                    y_tma_gmem,
                    (self.chunk_n, self.tile_size_M),
                    (Int32(wi), tile_M),
                )
                tS, tG = cute.nvgpu.cpasync.tma_partition(
                    y_tma,
                    Int32(0),
                    cute.make_layout(1),
                    cute.group_modes(s, 0, 2),
                    cute.group_modes(g, 0, 2),
                )
                cute.copy(y_tma, tS, tG)


class _CpBulkStoreOp(Op):
    reads = {"x": (None, ("M", "N"))}
    writes = {"y": (None, ("M", "N"))}
    tile = ("M",)

    @classmethod
    def schedule(cls, *, x, y, tile_m: int, page_size: int):
        op = cls._schedule_single(tile_sizes={"M": tile_m}, x=x, y=y)
        op.static_dims["page_size"] = page_size
        op.static_dims["copy_bits"] = tile_m * int(x.shape[1]) * ELEM_BYTES * 8
        return [op]

    @cute.jit
    def load(self, page_ptr, tile_M, x, y, work_mbar):
        atom = cute.make_copy_atom(CopyBulkG2SOp(), self.x_dtype, num_bits_per_copy=self.copy_bits)
        s = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout(self.tile_size_M * self.N),
        )
        g = cute.make_tensor(
            x.iterator + tile_M * Int32(self.tile_size_M * self.N),
            cute.make_layout(self.tile_size_M * self.N),
        )
        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, Int32(self.tile_size_M * self.N * ELEM_BYTES))
        gsrc = cute.group_modes(g, 0, 1)
        sdst = cute.group_modes(s, 0, 1)
        cute.copy(atom, gsrc, sdst, mbar_ptr=mbar_ptr)

    @cute.jit
    def compute(self, page_ptr, tile_M, x, y):
        tidx = cute.arch.thread_idx()[0]
        total = Int32(self.tile_size_M * self.N)
        s = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout(total),
        )
        for i in range(tidx, total, self.threads_per_row):
            s[i] = (s[i].to(Float32) + Float32(1.0)).to(self.x_dtype)

    @cute.jit
    def store(self, page_ptr, tile_M, x, y):
        atom = cute.make_copy_atom(CopyBulkS2GOp(), self.y_dtype, num_bits_per_copy=self.copy_bits)
        s = cute.make_tensor(
            cute.make_ptr(self.y_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout(self.tile_size_M * self.N),
        )
        g = cute.make_tensor(
            y.iterator + tile_M * Int32(self.tile_size_M * self.N),
            cute.make_layout(self.tile_size_M * self.N),
        )
        ssrc = cute.group_modes(s, 0, 1)
        gdst = cute.group_modes(g, 0, 1)
        cute.copy(atom, ssrc, gdst)


class _CpCoopStoreOp(_CpBulkStoreOp):
    collective_non_tma_load = True

    @cute.jit
    def _copy_layout(self):
        return (
            cute.make_layout((self.copy_dim0, self.copy_dim1), stride=(self.copy_dim1, 1)),
            cute.make_layout((1, self.copy_elems)),
        )

    @classmethod
    def schedule(cls, *, x, y, tile_m: int, page_size: int):
        op = cls._schedule_single(tile_sizes={"M": tile_m}, x=x, y=y)
        op.static_dims["page_size"] = page_size
        op.static_dims["copy_elems"] = 8
        op.static_dims["copy_dim1"] = int(x.shape[1]) // 8
        op.static_dims["copy_dim0"] = max(1, 32 // (int(x.shape[1]) // 8))
        return [op]

    @cute.jit
    def load(self, page_ptr, tile_M, x, y, work_mbar):
        lane = cute.arch.lane_idx()
        atom = cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(), self.x_dtype, num_bits_per_copy=128)
        thread_layout, value_layout = self._copy_layout()
        tiled = cute.make_tiled_copy_tv(atom, thread_layout, value_layout)
        thr = tiled.get_slice(lane)
        s = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=16),
            cute.make_layout((self.tile_size_M, self.N), stride=(self.N, 1)),
        )
        g = cute.make_tensor(
            (x.iterator + tile_M * Int32(self.tile_size_M * self.N)).align(16),
            cute.make_layout((self.tile_size_M, self.N), stride=(self.N, 1)),
        )
        cute.copy(tiled, thr.partition_S(g), thr.partition_D(s))
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        with cute.arch.elect_one():
            mbarrier_arrive(work_mbar)

    @cute.jit
    def store(self, page_ptr, tile_M, x, y):
        lane = cute.arch.lane_idx()
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.y_dtype, num_bits_per_copy=128)
        thread_layout, value_layout = self._copy_layout()
        tiled = cute.make_tiled_copy_tv(atom, thread_layout, value_layout)
        thr = tiled.get_slice(lane)
        s = cute.make_tensor(
            cute.make_ptr(self.y_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=16),
            cute.make_layout((self.tile_size_M, self.N), stride=(self.N, 1)),
        )
        g = cute.make_tensor(
            (y.iterator + tile_M * Int32(self.tile_size_M * self.N)).align(16),
            cute.make_layout((self.tile_size_M, self.N), stride=(self.N, 1)),
        )
        cute.copy(tiled, thr.partition_S(s), thr.partition_D(g))


def _scan_sass(path: Path) -> dict[str, object]:
    lines = path.read_text(errors="ignore").splitlines()
    bad = []
    counts = {"UTMALDG": 0, "UTMASTG": 0, "PLOP3": 0, "BRA.U.ANY": 0, "ELECT": 0}
    for i, line in enumerate(lines):
        for key in counts:
            if key in line:
                counts[key] += 1
        if "UTMASTG" in line and any(
            ("PLOP3" in x or "BRA.U.ANY" in x)
            for x in lines[max(0, i - 8) : min(len(lines), i + 8)]
        ):
            src = "unknown"
            for j in range(i, max(-1, i - 30), -1):
                if "//## File" in lines[j]:
                    src = lines[j].strip()
                    break
            bad.append((i + 1, line.strip(), src))
    return {"counts": counts, "bad": bad}


def _compile(kind: str, args) -> tuple[Path, Path]:
    dump_dir = Path(args.dump_dir) / kind
    if dump_dir.exists():
        shutil.rmtree(dump_dir)
    dump_dir.mkdir(parents=True)

    tile_m = args.tile_bytes // (args.n * ELEM_BYTES)
    total_m = tile_m * args.tiles
    if kind == "tma3d":
        x = torch.randn(1, total_m, args.n, device="cuda", dtype=torch.float16)
    elif kind == "postwork":
        x = torch.randn(1, total_m, args.n, device="cuda", dtype=torch.float16)
    else:
        x = torch.randn(total_m, args.n, device="cuda", dtype=torch.float16)
    y = torch.empty_like(x)
    y2 = torch.empty_like(x)
    scratch = torch.empty(x.shape, device="cuda", dtype=torch.float32)
    page_size = _page_size(args.tile_bytes)
    if kind == "tma":
        ops = _TmaStoreOp.schedule(x=x, y=y, tile_m=tile_m, page_size=page_size)
    elif kind == "tma3d":
        ops = _Tma3DStoreOp.schedule(x=x, y=y, tile_m=tile_m, page_size=page_size)
    elif kind == "postwork":
        ops = _TmaStorePostWorkOp.schedule(x=x, y=y, scratch=scratch, tile_m=tile_m, page_size=page_size)
    elif kind == "chunked":
        ops = _ChunkedTmaStoreOp.schedule(
            x=x,
            y=y,
            tile_m=tile_m,
            page_size=page_size,
            chunk_n=args.chunk_n,
        )
    elif kind == "two_tma":
        ops = _TmaStoreOp.schedule(x=x, y=y, tile_m=tile_m, page_size=page_size)
        ops += _ChunkedTmaStoreOp.schedule(
            x=x,
            y=y2,
            tile_m=tile_m,
            page_size=page_size,
            chunk_n=args.chunk_n,
        )
    elif kind == "many_tma":
        ops = []
        for i in range(args.many_ops):
            yi = torch.empty_like(x)
            if i % 2 == 0:
                ops += _TmaStoreOp.schedule(x=x, y=yi, tile_m=tile_m, page_size=page_size)
            else:
                ops += _ChunkedTmaStoreOp.schedule(
                    x=x,
                    y=yi,
                    tile_m=tile_m,
                    page_size=page_size,
                    chunk_n=args.chunk_n,
                )
    elif kind == "many_distinct":
        ops = []
        for i in range(args.many_ops):
            yi = torch.empty_like(x)
            op_cls = _DISTINCT_TMA_OPS[i % len(_DISTINCT_TMA_OPS)]
            ops += op_cls.schedule(x=x, y=yi, tile_m=tile_m, page_size=page_size)
    elif kind == "lane0":
        ops = _TmaStoreOp.schedule(x=x, y=y, tile_m=tile_m, page_size=page_size, lane0_store=True)
    elif kind == "cpbulk":
        ops = _CpBulkStoreOp.schedule(x=x, y=y, tile_m=tile_m, page_size=page_size)
    elif kind == "cpcoop":
        ops = _CpCoopStoreOp.schedule(x=x, y=y, tile_m=tile_m, page_size=page_size)
    else:
        raise ValueError(kind)

    kernel = Megakernel(
        ops,
        config=MegakernelConfig(
            threads_per_block=args.threads,
            num_pages=args.pages,
            page_size=page_size,
        ),
    )
    kernel.compile()

    can_verify = kind in {"tma", "tma3d", "lane0", "cpbulk"}
    if args.verify and can_verify:
        kernel.run()
        torch.cuda.synchronize()
        torch.testing.assert_close(y.float(), x.float() + 1.0, atol=2e-2, rtol=2e-2)
    elif args.verify:
        print(f"{kind}: verification skipped for comparison-only variant")

    cubins = sorted(dump_dir.rglob("*.cubin"))
    if not cubins:
        raise RuntimeError(f"no cubin dumped under {dump_dir}")
    sass = dump_dir / f"{kind}.sass"
    subprocess.run([args.nvdisasm, "-g", str(cubins[0])], check=True, stdout=sass.open("w"))
    return cubins[0], sass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", nargs="+", default=["tma", "tma3d", "chunked", "two_tma", "many_tma", "many_distinct", "postwork", "lane0", "cpbulk", "cpcoop"])
    parser.add_argument("--tile-bytes", type=int, default=16384)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--chunk-n", type=int, default=128)
    parser.add_argument("--tiles", type=int, default=4)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--pages", type=int, default=3)
    parser.add_argument("--many-ops", type=int, default=11)
    parser.add_argument("--dump-dir", default="traces/tma_store_sass_repro")
    parser.add_argument("--nvdisasm", default="/opt/cuda/bin/nvdisasm")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.tile_bytes % (args.n * ELEM_BYTES) != 0:
        raise SystemExit("--tile-bytes must be divisible by n * elem_bytes")

    if not args.child:
        base = Path(args.dump_dir)
        if base.exists():
            shutil.rmtree(base)
        for kind in args.kind:
            env = os.environ.copy()
            env["CUTE_DSL_DUMP_DIR"] = str(base / kind)
            cmd = [
                sys.executable,
                __file__,
                "--child",
                "--kind",
                kind,
                "--tile-bytes",
                str(args.tile_bytes),
                "--n",
                str(args.n),
                "--chunk-n",
                str(args.chunk_n),
                "--tiles",
                str(args.tiles),
                "--threads",
                str(args.threads),
                "--pages",
                str(args.pages),
                "--many-ops",
                str(args.many_ops),
                "--dump-dir",
                args.dump_dir,
                "--nvdisasm",
                args.nvdisasm,
            ]
            if args.verify:
                cmd.append("--verify")
            subprocess.run(cmd, check=True, env=env)
        return

    if len(args.kind) != 1:
        raise SystemExit("--child expects exactly one --kind")

    for kind in args.kind:
        cubin, sass = _compile(kind, args)
        result = _scan_sass(sass)
        counts = result["counts"]
        bad = result["bad"]
        print(
            f"{kind}: cubin={cubin} sass={sass} "
            f"UTMALDG={counts['UTMALDG']} UTMASTG={counts['UTMASTG']} "
            f"ELECT={counts['ELECT']} PLOP3={counts['PLOP3']} "
            f"BRA.U.ANY={counts['BRA.U.ANY']} bad_UTMASTG={len(bad)}"
        )
        for line_no, instr, src in bad[:8]:
            print(f"  bad line {line_no}: {instr}")
            print(f"    {src}")


if __name__ == "__main__":
    main()
