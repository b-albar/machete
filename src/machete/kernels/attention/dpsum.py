# Copyright (c) 2025, Machete Authors
"""Row-sum helper for FlashAttention backward.

Computes ``dpsum = rowsum(dout * o)`` for BSHD tensors:

    dout:  (B, S, H, D)
    o:     (B, S, H, D)
    dpsum: (B, S, H)

This is the missing in-graph reduction needed to feed FlashAttention backward
inside a single megakernel, where ``dout`` is itself produced by an earlier op.
"""

import operator

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32

from machete.megakernel.ops import (
    DEFAULT_PAGE_SIZE,
    Op,
    config_dim_i32,
    config_flat_tensor,
)


class AttentionDPSumOp(Op):
    """Compute ``rowsum(dout * o)`` in native BSH layout."""

    reads = {
        "dout": (None, ("B", "S", "H", "D")),
        "o": (None, ("B", "S", "H", "D")),
    }
    writes = {
        "dpsum": (cutlass.Float32, ("B", "S", "H")),
    }
    tile = ("B", "S", "H")
    dynamic_dims = ("B", "S")

    def __init__(self, **config):
        super().__init__(**config)
        self.page_size = getattr(self, "page_size", DEFAULT_PAGE_SIZE)
        self.num_mma_warps = 1

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, dim_windows=None, **tensors):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", 32)
        tile_sizes.setdefault("H", 1)
        ops = [cls._schedule_single(tile_sizes=tile_sizes, dim_windows=dim_windows, **tensors)]
        ops[0].static_dims["page_size"] = page_size
        dout = tensors.get("dout")
        if dout is not None:
            ops[0].static_dims["S"] = dout.shape[1]
        return ops

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_H, op_config_ptr):
        runtime_B = config_dim_i32(op_config_ptr, "B", type(self))
        dout = config_flat_tensor(
            op_config_ptr,
            "dout",
            self.dout_dtype,
            runtime_B * Int32(self.S * self.H * self.D),
            type(self),
        )
        o = config_flat_tensor(
            op_config_ptr,
            "o",
            self.o_dtype,
            runtime_B * Int32(self.S * self.H * self.D),
            type(self),
        )
        dpsum = config_flat_tensor(
            op_config_ptr,
            "dpsum",
            self.dpsum_dtype,
            runtime_B * Int32(self.S * self.H),
            type(self),
        )

        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.threads_per_row // 32
        thr_layout = cute.make_layout(32)

        b = tile_B
        s_start = tile_S * self.tile_size_S
        h_start = tile_H * self.tile_size_H

        if b < runtime_B:
            for local_s in range(self.tile_size_S):
                s = s_start + local_s
                if s < self.S:
                    for local_h in range(warp_idx, self.tile_size_H, num_warps):
                        h = h_start + local_h
                        if h < self.H:
                            row_base = (
                                b * Int32(self.S * self.H * self.D)
                                + s * Int32(self.H * self.D)
                                + h * Int32(self.D)
                            )
                            dout_row = cute.make_tensor(
                                dout.iterator + row_base,
                                cute.make_layout(self.D),
                            )
                            o_row = cute.make_tensor(
                                o.iterator + row_base,
                                cute.make_layout(self.D),
                            )
                            dout_part = cute.local_partition(dout_row, thr_layout, lane_idx)
                            o_part = cute.local_partition(o_row, thr_layout, lane_idx)
                            dout_reg = cute.make_fragment_like(dout_part)
                            o_reg = cute.make_fragment_like(o_part)
                            cute.autovec_copy(dout_part, dout_reg)
                            cute.autovec_copy(o_part, o_reg)

                            partial = Float32(0.0)
                            for i in range(cute.size(dout_reg)):
                                partial = partial + dout_reg[i].to(Float32) * o_reg[i].to(Float32)
                            row_sum = cute.arch.warp_reduction(partial, operator.add)

                            if lane_idx == 0:
                                dpsum_idx = b * Int32(self.S * self.H) + s * Int32(self.H) + h
                                dpsum[dpsum_idx] = row_sum


__all__ = ["AttentionDPSumOp"]
