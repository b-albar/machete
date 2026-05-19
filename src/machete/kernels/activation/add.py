# Copyright (c) 2025, Machete Authors
"""Elementwise add op for megakernel pipelines."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32

from machete.megakernel.ops import Op


class AddOp(Op):
    """Compute ``y = x + add`` for 3D tensors shaped ``(B, S, D)``."""

    reads = {
        "x": (None, ("B", "S", "D")),
        "add": (None, ("B", "S", "D")),
    }
    writes = {
        "y": (None, ("B", "S", "D")),
    }
    tile = ("B", "S", "D")
    dynamic_dims = ("B",)

    @classmethod
    def schedule(cls, tile_sizes=None, **tensors):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", 16)
        if "y" not in tensors:
            tensors["y"] = tensors["x"]
        return [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_D, x, add, y):
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.threads_per_row // 32
        thr_layout = cute.make_layout(32)
        row_start = tile_S * self.tile_size_S

        for local_row in range(warp_idx, self.tile_size_S, num_warps):
            row_idx = row_start + local_row
            if row_idx < self.S:
                global_offset = tile_B * (self.S * self.D) + row_idx * self.D
                x_row = cute.make_tensor(x.iterator + global_offset, cute.make_layout(self.D))
                add_row = cute.make_tensor(add.iterator + global_offset, cute.make_layout(self.D))
                y_row = cute.make_tensor(y.iterator + global_offset, cute.make_layout(self.D))

                x_part = cute.local_partition(x_row, thr_layout, lane_idx)
                add_part = cute.local_partition(add_row, thr_layout, lane_idx)
                y_part = cute.local_partition(y_row, thr_layout, lane_idx)

                x_reg = cute.make_fragment_like(x_part)
                add_reg = cute.make_fragment_like(add_part)
                y_reg = cute.make_fragment_like(y_part)
                cute.autovec_copy(x_part, x_reg)
                cute.autovec_copy(add_part, add_reg)
                for i in range(cute.size(y_reg)):
                    y_reg[i] = (x_reg[i].to(Float32) + add_reg[i].to(Float32)).to(self.y_dtype)
                cute.autovec_copy(y_reg, y_part)


__all__ = ["AddOp"]
