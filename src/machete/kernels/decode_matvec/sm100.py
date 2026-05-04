# Copyright (c) 2025, Machete Authors
"""Instruction-shaped decode decode ops for Blackwell/B200.

These ops intentionally mirror the coarse instructions in
HazyResearch/Megakernels' low-latency Llama C++ demo rather than composing the
generic RMSNorm/GEMM/RoPE/GLU kernels:

* RMS + Q matvec + RoPE
* RMS + K matvec + RoPE + KV-cache append
* RMS + V matvec + KV-cache append
* O/down matvec variants with residual semantics
* RMS + gate/up matvec + SiLU
* final RMS + LM-head matvec

The first version is a direct global-memory, warp-reduced matvec design.  It is
deliberately decode-specialized: one warp computes one token row and one
16-wide output block at a time, matching the C++ demo's 16-element matvec block
granularity.  This gives the benchmark a real fused-kernel path to evaluate on
B200 while leaving room for later TMA page pipelining and tcgen05-specific
micro-optimizations.
"""

import operator
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32

from machete.megakernel.ops import Op, DEFAULT_PAGE_SIZE


SM100_DECODE_HIDDEN_DEFAULT = 2048
SM100_DECODE_HEAD_DIM_DEFAULT = 64
SM100_DECODE_ROTARY_D2_DEFAULT = 32
SM100_DECODE_Q_DIM_DEFAULT = 2048
SM100_DECODE_KV_DIM_DEFAULT = 512
SM100_DECODE_INTERMEDIATE_DEFAULT = 8192
SM100_DECODE_VOCAB_DEFAULT = 128256
SM100_DECODE_MATVEC_BLOCK = 16
SM100_DECODE_REDUCTION_DIM_PER_WARP = 512
SM100_DECODE_CONSUMER_WARPS_DEFAULT = SM100_DECODE_HIDDEN_DEFAULT // SM100_DECODE_REDUCTION_DIM_PER_WARP
SM100_DECODE_NUM_Q_HEADS_DEFAULT = SM100_DECODE_Q_DIM_DEFAULT // SM100_DECODE_HEAD_DIM_DEFAULT
SM100_DECODE_NUM_KV_HEADS_DEFAULT = SM100_DECODE_KV_DIM_DEFAULT // SM100_DECODE_HEAD_DIM_DEFAULT
SM100_DECODE_KV_GROUP_SIZE_DEFAULT = SM100_DECODE_NUM_Q_HEADS_DEFAULT // SM100_DECODE_NUM_KV_HEADS_DEFAULT


@cute.jit
def _silu(x):
    neg = Float32(0.0) - x
    exp_neg = cute.math.exp(neg, fastmath=True)
    return x / (Float32(1.0) + exp_neg)


class _DecodeMatvecSm100Base(Op):
    """Shared schedule/config helpers for decode matvec ops."""

    @classmethod
    def kernel_config(cls, ops):
        from machete.megakernel import MegakernelConfig
        from machete.megakernel.megakernel import NUM_DMA_WARPS

        page_size = max(op.static_dims.get("page_size", DEFAULT_PAGE_SIZE) for op in ops)
        reduction_tile = ops[0].static_dims.get(
            "reduction_tile_K", SM100_DECODE_REDUCTION_DIM_PER_WARP
        )
        k_dim = ops[0].static_dims.get("K", SM100_DECODE_HIDDEN_DEFAULT)
        consumer_warps = max(1, (k_dim + reduction_tile - 1) // reduction_tile)
        return MegakernelConfig(
            threads_per_block=(consumer_warps + NUM_DMA_WARPS) * 32,
            page_size=page_size,
            mma_reg_count=96,
        )


class _RmsProjectionSm100Base(_DecodeMatvecSm100Base):
    """RMS(input+residual) followed by one projection matrix."""

    reads = {
        "x": (None, ("B", "S", "K")),
        "residual_in": (None, ("B", "S", "K")),
        "norm_weight": (None, ("K",)),
        "weight": (None, ("O", "K")),
        "cos": (None, ("S", "D2")),
        "sin": (None, ("S", "D2")),
    }
    writes = {
        "residual_out": (None, ("B", "S", "K")),
    }
    tile = ("B", "S", "O")
    dynamic_dims = ("B",)

    apply_rope = 0
    write_q = 0
    write_cache = 0

    def __init__(self, **config):
        super().__init__(**config)
        if self.x_dtype not in (cutlass.Float16, cutlass.BFloat16):
            raise ValueError("Decode SM100 decode ops require fp16/bf16 activations")
        self.eps = getattr(self, "eps", 1e-5)
        self.cache_pos = getattr(self, "cache_pos", 0)
        self.page_size = getattr(self, "page_size", DEFAULT_PAGE_SIZE)
        self.reduction_tile_K = getattr(self, "reduction_tile_K", SM100_DECODE_REDUCTION_DIM_PER_WARP)
        self.head_dim = getattr(self, "head_dim", self.D2 * 2)
        assert self.tile_size_O == 16

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, eps=1e-5, cache_pos=0, **tensors):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", 16)
        tile_sizes.setdefault("O", 16)
        op = cls._schedule_single(tile_sizes=tile_sizes, **tensors)
        op.static_dims["page_size"] = page_size
        op.static_dims["eps"] = eps
        op.static_dims["cache_pos"] = cache_pos
        op.static_dims["head_dim"] = tensors["cos"].shape[1] * 2
        op.static_dims["reduction_tile_K"] = min(
            SM100_DECODE_REDUCTION_DIM_PER_WARP, tensors["x"].shape[-1]
        )
        return [op]

    @cute.jit
    def _row_rstd(self, tile_B, row_idx, x, residual_in):
        lane_idx = cute.arch.lane_idx()
        x_base = tile_B * Int32(self.x_stride_B) + row_idx * Int32(self.x_stride_S)
        res_base = tile_B * Int32(self.residual_in_stride_B) + row_idx * Int32(self.residual_in_stride_S)

        sum_sq = Float32(0.0)
        k_base = Int32(0)
        while k_base < Int32(self.K):
            x_row = cute.make_tensor(x.iterator + x_base + k_base, cute.make_layout(self.reduction_tile_K))
            res_row = cute.make_tensor(residual_in.iterator + res_base + k_base, cute.make_layout(self.reduction_tile_K))
            elem = lane_idx
            while elem < Int32(self.reduction_tile_K):
                k = k_base + elem
                if k < Int32(self.K):
                    xv = x_row[elem].to(Float32)
                    rv = res_row[elem].to(Float32)
                    val = xv + rv
                    sum_sq = sum_sq + val * val
                elem = elem + Int32(32)
            k_base = k_base + Int32(self.reduction_tile_K)
        total = cute.arch.warp_reduction(sum_sq, operator.add)
        return cute.math.rsqrt(total * Float32(1.0 / self.K) + Float32(self.eps), fastmath=True)

    @cute.jit
    def _row_rstd_and_store_residual(self, tile_B, row_idx, x, residual_in, residual_out):
        lane_idx = cute.arch.lane_idx()
        x_base = tile_B * Int32(self.x_stride_B) + row_idx * Int32(self.x_stride_S)
        res_base = tile_B * Int32(self.residual_in_stride_B) + row_idx * Int32(self.residual_in_stride_S)
        out_base = tile_B * Int32(self.residual_out_stride_B) + row_idx * Int32(self.residual_out_stride_S)

        sum_sq = Float32(0.0)
        k_base = Int32(0)
        while k_base < Int32(self.K):
            x_row = cute.make_tensor(x.iterator + x_base + k_base, cute.make_layout(self.reduction_tile_K))
            res_row = cute.make_tensor(residual_in.iterator + res_base + k_base, cute.make_layout(self.reduction_tile_K))
            out_row = cute.make_tensor(residual_out.iterator + out_base + k_base, cute.make_layout(self.reduction_tile_K))
            elem = lane_idx
            while elem < Int32(self.reduction_tile_K):
                k = k_base + elem
                if k < Int32(self.K):
                    xv = x_row[elem].to(Float32)
                    rv = res_row[elem].to(Float32)
                    val = xv + rv
                    sum_sq = sum_sq + val * val
                    out_row[elem] = val.to(self.residual_out_dtype)
                elem = elem + Int32(32)
            k_base = k_base + Int32(self.reduction_tile_K)
        total = cute.arch.warp_reduction(sum_sq, operator.add)
        return cute.math.rsqrt(total * Float32(1.0 / self.K) + Float32(self.eps), fastmath=True)

    @cute.jit
    def _dot_output(self, tile_B, row_idx, out_idx, rstd, x, residual_in, norm_weight, weight):
        lane_idx = cute.arch.lane_idx()
        x_base = tile_B * Int32(self.x_stride_B) + row_idx * Int32(self.x_stride_S)
        res_base = tile_B * Int32(self.residual_in_stride_B) + row_idx * Int32(self.residual_in_stride_S)
        w_base = out_idx * Int32(self.weight_stride_O)
        acc = Float32(0.0)
        k_base = Int32(0)
        while k_base < Int32(self.K):
            x_row = cute.make_tensor(x.iterator + x_base + k_base, cute.make_layout(self.reduction_tile_K))
            res_row = cute.make_tensor(residual_in.iterator + res_base + k_base, cute.make_layout(self.reduction_tile_K))
            weight_row = cute.make_tensor(weight.iterator + w_base + k_base, cute.make_layout(self.reduction_tile_K))
            norm_row = cute.make_tensor(norm_weight.iterator + k_base, cute.make_layout(self.reduction_tile_K))
            elem = lane_idx
            while elem < Int32(self.reduction_tile_K):
                k = k_base + elem
                if k < Int32(self.K):
                    xv = x_row[elem].to(Float32)
                    rv = res_row[elem].to(Float32)
                    nv = (xv + rv) * rstd * norm_row[elem].to(Float32)
                    wv = weight_row[elem].to(Float32)
                    acc = acc + nv * wv
                elem = elem + Int32(32)
            k_base = k_base + Int32(self.reduction_tile_K)
        return cute.arch.warp_reduction(acc, operator.add)

    @cute.jit
    def _rope_value(self, tile_B, row_idx, out_idx, value, rstd, x, residual_in, norm_weight, weight, cos, sin):
        dim = out_idx % Int32(self.head_dim)
        out = value
        if dim < Int32(self.D2):
            pair_idx = out_idx + Int32(self.D2)
            pair = self._dot_output(tile_B, row_idx, pair_idx, rstd, x, residual_in, norm_weight, weight)
            cos_row = cute.make_tensor(cos.iterator + row_idx * Int32(self.cos_stride_S), cute.make_layout(self.D2))
            sin_row = cute.make_tensor(sin.iterator + row_idx * Int32(self.sin_stride_S), cute.make_layout(self.D2))
            c = cos_row[dim].to(Float32)
            s = sin_row[dim].to(Float32)
            out = value * c - pair * s
        elif dim < Int32(self.head_dim):
            pair_dim = dim - Int32(self.D2)
            pair_idx = out_idx - Int32(self.D2)
            pair = self._dot_output(tile_B, row_idx, pair_idx, rstd, x, residual_in, norm_weight, weight)
            cos_row = cute.make_tensor(cos.iterator + row_idx * Int32(self.cos_stride_S), cute.make_layout(self.D2))
            sin_row = cute.make_tensor(sin.iterator + row_idx * Int32(self.sin_stride_S), cute.make_layout(self.D2))
            c = cos_row[pair_dim].to(Float32)
            s = sin_row[pair_dim].to(Float32)
            out = value * c + pair * s
        return out


class RmsQMatvecRopeSm100Op(_RmsProjectionSm100Base):
    """Fused RMS(input+residual) + Q matvec + RoPE."""

    writes = {
        "residual_out": (None, ("B", "S", "K")),
        "q": (None, ("B", "S", "O")),
    }
    apply_rope = 1
    write_q = 1

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_O, x, residual_in, norm_weight, weight, cos, sin, residual_out, q):
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.threads_per_row // 32
        row_start = tile_S * Int32(self.tile_size_S)
        out_start = tile_O * Int32(self.tile_size_O)
        for local_row in range(warp_idx, self.tile_size_S, num_warps):
            row_idx = row_start + Int32(local_row)
            if row_idx < Int32(self.S):
                rstd = self._row_rstd_and_store_residual(tile_B, row_idx, x, residual_in, residual_out)
                for local_o in range(self.tile_size_O):
                    out_idx = out_start + Int32(local_o)
                    if out_idx < Int32(self.O):
                        val = self._dot_output(tile_B, row_idx, out_idx, rstd, x, residual_in, norm_weight, weight)
                        val = self._rope_value(tile_B, row_idx, out_idx, val, rstd, x, residual_in, norm_weight, weight, cos, sin)
                        if lane_idx == Int32(0):
                            q_base = tile_B * Int32(self.q_stride_B) + row_idx * Int32(self.q_stride_S)
                            q_tile = cute.make_tensor(q.iterator + q_base + out_start, cute.make_layout(self.tile_size_O))
                            q_tile[local_o] = val.to(self.q_dtype)


class RmsKMatvecRopeCacheSm100Op(_RmsProjectionSm100Base):
    """Fused RMS(input+residual) + K matvec + RoPE + cache append."""

    writes = {
        "dst_cache": (None, ("B", "T", "H", "HD")),
    }
    apply_rope = 1
    write_cache = 1

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_O, x, residual_in, norm_weight, weight, cos, sin, dst_cache):
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.threads_per_row // 32
        row_start = tile_S * Int32(self.tile_size_S)
        out_start = tile_O * Int32(self.tile_size_O)
        for local_row in range(warp_idx, self.tile_size_S, num_warps):
            row_idx = row_start + Int32(local_row)
            if row_idx < Int32(self.S):
                rstd = self._row_rstd(tile_B, row_idx, x, residual_in)
                for local_o in range(self.tile_size_O):
                    out_idx = out_start + Int32(local_o)
                    if out_idx < Int32(self.O):
                        val = self._dot_output(tile_B, row_idx, out_idx, rstd, x, residual_in, norm_weight, weight)
                        val = self._rope_value(tile_B, row_idx, out_idx, val, rstd, x, residual_in, norm_weight, weight, cos, sin)
                        if lane_idx == Int32(0):
                            head = out_idx // Int32(self.head_dim)
                            dim = out_idx % Int32(self.head_dim)
                            dst_base = (
                                tile_B * Int32(self.dst_cache_stride_B)
                                + (row_idx + Int32(self.cache_pos)) * Int32(self.dst_cache_stride_T)
                                + head * Int32(self.dst_cache_stride_H)
                            )
                            dst_row = cute.make_tensor(dst_cache.iterator + dst_base, cute.make_layout(self.HD))
                            dst_row[dim] = val.to(self.dst_cache_dtype)


class RmsVMatvecCacheSm100Op(_RmsProjectionSm100Base):
    """Fused RMS(input+residual) + V matvec + cache append."""

    writes = {
        "dst_cache": (None, ("B", "T", "H", "HD")),
    }

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_O, x, residual_in, norm_weight, weight, cos, sin, dst_cache):
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.threads_per_row // 32
        row_start = tile_S * Int32(self.tile_size_S)
        out_start = tile_O * Int32(self.tile_size_O)
        for local_row in range(warp_idx, self.tile_size_S, num_warps):
            row_idx = row_start + Int32(local_row)
            if row_idx < Int32(self.S):
                rstd = self._row_rstd(tile_B, row_idx, x, residual_in)
                for local_o in range(self.tile_size_O):
                    out_idx = out_start + Int32(local_o)
                    if out_idx < Int32(self.O):
                        val = self._dot_output(tile_B, row_idx, out_idx, rstd, x, residual_in, norm_weight, weight)
                        if lane_idx == Int32(0):
                            head = out_idx // Int32(self.head_dim)
                            dim = out_idx % Int32(self.head_dim)
                            dst_base = (
                                tile_B * Int32(self.dst_cache_stride_B)
                                + (row_idx + Int32(self.cache_pos)) * Int32(self.dst_cache_stride_T)
                                + head * Int32(self.dst_cache_stride_H)
                            )
                            dst_row = cute.make_tensor(dst_cache.iterator + dst_base, cute.make_layout(self.HD))
                            dst_row[dim] = val.to(self.dst_cache_dtype)


class MatvecResidualSm100Op(_DecodeMatvecSm100Base):
    """Matvec projection with residual add: residual_out = a @ W.T + residual_in."""

    reads = {
        "a": (None, ("B", "S", "K")),
        "weight": (None, ("O", "K")),
        "residual_in": (None, ("B", "S", "O")),
    }
    writes = {"residual_out": (None, ("B", "S", "O"))}
    tile = ("B", "S", "O")
    dynamic_dims = ("B",)

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, **tensors):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", 16)
        tile_sizes.setdefault("O", 16)
        op = cls._schedule_single(tile_sizes=tile_sizes, **tensors)
        op.static_dims["page_size"] = page_size
        return [op]

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_O, a, weight, residual_in, residual_out):
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.threads_per_row // 32
        row_start = tile_S * Int32(self.tile_size_S)
        out_start = tile_O * Int32(self.tile_size_O)
        for local_row in range(warp_idx, self.tile_size_S, num_warps):
            row_idx = row_start + Int32(local_row)
            if row_idx < Int32(self.S):
                a_base = tile_B * Int32(self.a_stride_B) + row_idx * Int32(self.a_stride_S)
                for local_o in range(self.tile_size_O):
                    out_idx = out_start + Int32(local_o)
                    if out_idx < Int32(self.O):
                        acc = Float32(0.0)
                        w_base = out_idx * Int32(self.weight_stride_O)
                        k_base = Int32(0)
                        while k_base < Int32(self.K):
                            a_row = cute.make_tensor(a.iterator + a_base + k_base, cute.make_layout(SM100_DECODE_REDUCTION_DIM_PER_WARP))
                            weight_row = cute.make_tensor(weight.iterator + w_base + k_base, cute.make_layout(SM100_DECODE_REDUCTION_DIM_PER_WARP))
                            elem = lane_idx
                            while elem < Int32(SM100_DECODE_REDUCTION_DIM_PER_WARP):
                                k = k_base + elem
                                if k < Int32(self.K):
                                    acc = acc + a_row[elem].to(Float32) * weight_row[elem].to(Float32)
                                elem = elem + Int32(32)
                            k_base = k_base + Int32(SM100_DECODE_REDUCTION_DIM_PER_WARP)
                        total = cute.arch.warp_reduction(acc, operator.add)
                        if lane_idx == Int32(0):
                            out_base = tile_B * Int32(self.residual_out_stride_B) + row_idx * Int32(self.residual_out_stride_S)
                            res_base = tile_B * Int32(self.residual_in_stride_B) + row_idx * Int32(self.residual_in_stride_S)
                            res_tile = cute.make_tensor(residual_in.iterator + res_base + out_start, cute.make_layout(self.tile_size_O))
                            out_tile = cute.make_tensor(residual_out.iterator + out_base + out_start, cute.make_layout(self.tile_size_O))
                            val = total + res_tile[local_o].to(Float32)
                            out_tile[local_o] = val.to(self.residual_out_dtype)


class MatvecSm100Op(_DecodeMatvecSm100Base):
    """Down projection matvec that writes the layer MLP delta."""

    reads = {
        "a": (None, ("B", "S", "K")),
        "weight": (None, ("O", "K")),
    }
    writes = {"y": (None, ("B", "S", "O"))}
    tile = ("B", "S", "O")
    dynamic_dims = ("B",)

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, **tensors):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", 16)
        tile_sizes.setdefault("O", 16)
        op = cls._schedule_single(tile_sizes=tile_sizes, **tensors)
        op.static_dims["page_size"] = page_size
        return [op]

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_O, a, weight, y):
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.threads_per_row // 32
        row_start = tile_S * Int32(self.tile_size_S)
        out_start = tile_O * Int32(self.tile_size_O)
        for local_row in range(warp_idx, self.tile_size_S, num_warps):
            row_idx = row_start + Int32(local_row)
            if row_idx < Int32(self.S):
                a_base = tile_B * Int32(self.a_stride_B) + row_idx * Int32(self.a_stride_S)
                for local_o in range(self.tile_size_O):
                    out_idx = out_start + Int32(local_o)
                    if out_idx < Int32(self.O):
                        acc = Float32(0.0)
                        w_base = out_idx * Int32(self.weight_stride_O)
                        k_base = Int32(0)
                        while k_base < Int32(self.K):
                            a_row = cute.make_tensor(a.iterator + a_base + k_base, cute.make_layout(SM100_DECODE_REDUCTION_DIM_PER_WARP))
                            weight_row = cute.make_tensor(weight.iterator + w_base + k_base, cute.make_layout(SM100_DECODE_REDUCTION_DIM_PER_WARP))
                            elem = lane_idx
                            while elem < Int32(SM100_DECODE_REDUCTION_DIM_PER_WARP):
                                k = k_base + elem
                                if k < Int32(self.K):
                                    acc = acc + a_row[elem].to(Float32) * weight_row[elem].to(Float32)
                                elem = elem + Int32(32)
                            k_base = k_base + Int32(SM100_DECODE_REDUCTION_DIM_PER_WARP)
                        total = cute.arch.warp_reduction(acc, operator.add)
                        if lane_idx == Int32(0):
                            y_base = tile_B * Int32(self.y_stride_B) + row_idx * Int32(self.y_stride_S)
                            y_tile = cute.make_tensor(y.iterator + y_base + out_start, cute.make_layout(self.tile_size_O))
                            y_tile[local_o] = total.to(self.y_dtype)


class ResidualAddSm100Op(_DecodeMatvecSm100Base):
    """Vector add for final residual materialization inside the megakernel."""

    reads = {
        "x": (None, ("B", "S", "K")),
        "residual_in": (None, ("B", "S", "K")),
    }
    writes = {"residual_out": (None, ("B", "S", "K"))}
    tile = ("B", "S", "K")
    dynamic_dims = ("B",)

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, **tensors):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", 16)
        tile_sizes.setdefault("K", 256)
        op = cls._schedule_single(tile_sizes=tile_sizes, **tensors)
        op.static_dims["page_size"] = page_size
        return [op]

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_K, x, residual_in, residual_out):
        tidx = cute.arch.thread_idx()[0]
        row_start = tile_S * Int32(self.tile_size_S)
        k_start = tile_K * Int32(self.tile_size_K)
        for local_row in range(self.tile_size_S):
            row_idx = row_start + Int32(local_row)
            if row_idx < Int32(self.S):
                x_base = tile_B * Int32(self.x_stride_B) + row_idx * Int32(self.x_stride_S)
                res_base = tile_B * Int32(self.residual_in_stride_B) + row_idx * Int32(self.residual_in_stride_S)
                out_base = tile_B * Int32(self.residual_out_stride_B) + row_idx * Int32(self.residual_out_stride_S)
                x_row = cute.make_tensor(x.iterator + x_base + k_start, cute.make_layout(self.tile_size_K))
                res_row = cute.make_tensor(residual_in.iterator + res_base + k_start, cute.make_layout(self.tile_size_K))
                out_row = cute.make_tensor(residual_out.iterator + out_base + k_start, cute.make_layout(self.tile_size_K))
                elem = tidx
                while elem < Int32(self.tile_size_K):
                    k = k_start + elem
                    if k < Int32(self.K):
                        val = x_row[elem].to(Float32) + res_row[elem].to(Float32)
                        out_row[elem] = val.to(self.residual_out_dtype)
                    elem = elem + Int32(self.threads_per_row)


class RmsGateUpSiluSm100Op(_DecodeMatvecSm100Base):
    """Fused RMS + gate/up matvec + SiLU, producing the MLP hidden vector."""

    reads = {
        "x": (None, ("B", "S", "K")),
        "norm_weight": (None, ("K",)),
        "gate_weight": (None, ("D", "K")),
        "up_weight": (None, ("D", "K")),
    }
    writes = {"y": (None, ("B", "S", "D"))}
    tile = ("B", "S", "D")
    dynamic_dims = ("B",)

    def __init__(self, **config):
        super().__init__(**config)
        self.eps = getattr(self, "eps", 1e-5)
        self.reduction_tile_K = getattr(self, "reduction_tile_K", min(SM100_DECODE_REDUCTION_DIM_PER_WARP, self.K))
        assert self.tile_size_D == 16

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, eps=1e-5, **tensors):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", 16)
        tile_sizes.setdefault("D", 16)
        op = cls._schedule_single(tile_sizes=tile_sizes, **tensors)
        op.static_dims["page_size"] = page_size
        op.static_dims["eps"] = eps
        op.static_dims["reduction_tile_K"] = min(
            SM100_DECODE_REDUCTION_DIM_PER_WARP, tensors["x"].shape[-1]
        )
        return [op]

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_D, x, norm_weight, gate_weight, up_weight, y):
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.threads_per_row // 32
        row_start = tile_S * Int32(self.tile_size_S)
        d_start = tile_D * Int32(self.tile_size_D)
        for local_row in range(warp_idx, self.tile_size_S, num_warps):
            row_idx = row_start + Int32(local_row)
            if row_idx < Int32(self.S):
                x_base = tile_B * Int32(self.x_stride_B) + row_idx * Int32(self.x_stride_S)
                sum_sq = Float32(0.0)
                k_base = Int32(0)
                while k_base < Int32(self.K):
                    x_row = cute.make_tensor(x.iterator + x_base + k_base, cute.make_layout(SM100_DECODE_REDUCTION_DIM_PER_WARP))
                    elem = lane_idx
                    while elem < Int32(SM100_DECODE_REDUCTION_DIM_PER_WARP):
                        k = k_base + elem
                        if k < Int32(self.K):
                            xv = x_row[elem].to(Float32)
                            sum_sq = sum_sq + xv * xv
                        elem = elem + Int32(32)
                    k_base = k_base + Int32(SM100_DECODE_REDUCTION_DIM_PER_WARP)
                total_sq = cute.arch.warp_reduction(sum_sq, operator.add)
                rstd = cute.math.rsqrt(total_sq * Float32(1.0 / self.K) + Float32(self.eps), fastmath=True)

                for local_d in range(self.tile_size_D):
                    d = d_start + Int32(local_d)
                    if d < Int32(self.D):
                        gate_acc = Float32(0.0)
                        up_acc = Float32(0.0)
                        gate_base = d * Int32(self.gate_weight_stride_D)
                        up_base = d * Int32(self.up_weight_stride_D)
                        k2_base = Int32(0)
                        while k2_base < Int32(self.K):
                            x_row = cute.make_tensor(x.iterator + x_base + k2_base, cute.make_layout(SM100_DECODE_REDUCTION_DIM_PER_WARP))
                            norm_row = cute.make_tensor(norm_weight.iterator + k2_base, cute.make_layout(SM100_DECODE_REDUCTION_DIM_PER_WARP))
                            gate_row = cute.make_tensor(gate_weight.iterator + gate_base + k2_base, cute.make_layout(SM100_DECODE_REDUCTION_DIM_PER_WARP))
                            up_row = cute.make_tensor(up_weight.iterator + up_base + k2_base, cute.make_layout(SM100_DECODE_REDUCTION_DIM_PER_WARP))
                            elem2 = lane_idx
                            while elem2 < Int32(SM100_DECODE_REDUCTION_DIM_PER_WARP):
                                k2 = k2_base + elem2
                                if k2 < Int32(self.K):
                                    nv = x_row[elem2].to(Float32) * rstd * norm_row[elem2].to(Float32)
                                    gate_acc = gate_acc + nv * gate_row[elem2].to(Float32)
                                    up_acc = up_acc + nv * up_row[elem2].to(Float32)
                                elem2 = elem2 + Int32(32)
                            k2_base = k2_base + Int32(SM100_DECODE_REDUCTION_DIM_PER_WARP)
                        gate_total = cute.arch.warp_reduction(gate_acc, operator.add)
                        up_total = cute.arch.warp_reduction(up_acc, operator.add)
                        if lane_idx == Int32(0):
                            y_base = tile_B * Int32(self.y_stride_B) + row_idx * Int32(self.y_stride_S)
                            y_tile = cute.make_tensor(y.iterator + y_base + d_start, cute.make_layout(self.tile_size_D))
                            y_tile[local_d] = (_silu(gate_total) * up_total).to(self.y_dtype)


class FinalRmsLmHeadSm100Op(RmsGateUpSiluSm100Op):
    """Fused final RMS + LM-head matvec."""

    reads = {
        "x": (None, ("B", "S", "K")),
        "norm_weight": (None, ("K",)),
        "weight": (None, ("V", "K")),
    }
    writes = {"logits": (None, ("B", "S", "V"))}
    tile = ("B", "S", "V")
    dynamic_dims = ("B",)

    def __init__(self, **config):
        Op.__init__(self, **config)
        self.eps = getattr(self, "eps", 1e-5)
        self.reduction_tile_K = getattr(self, "reduction_tile_K", min(SM100_DECODE_REDUCTION_DIM_PER_WARP, self.K))
        assert self.tile_size_V == 16

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, eps=1e-5, **tensors):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", 16)
        tile_sizes.setdefault("V", 16)
        op = cls._schedule_single(tile_sizes=tile_sizes, **tensors)
        op.static_dims["page_size"] = page_size
        op.static_dims["eps"] = eps
        op.static_dims["reduction_tile_K"] = min(
            SM100_DECODE_REDUCTION_DIM_PER_WARP, tensors["x"].shape[-1]
        )
        return [op]

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_V, x, norm_weight, weight, logits):
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.threads_per_row // 32
        row_start = tile_S * Int32(self.tile_size_S)
        v_start = tile_V * Int32(self.tile_size_V)
        for local_row in range(warp_idx, self.tile_size_S, num_warps):
            row_idx = row_start + Int32(local_row)
            if row_idx < Int32(self.S):
                x_base = tile_B * Int32(self.x_stride_B) + row_idx * Int32(self.x_stride_S)
                sum_sq = Float32(0.0)
                k_base = Int32(0)
                while k_base < Int32(self.K):
                    x_row = cute.make_tensor(x.iterator + x_base + k_base, cute.make_layout(SM100_DECODE_REDUCTION_DIM_PER_WARP))
                    elem = lane_idx
                    while elem < Int32(SM100_DECODE_REDUCTION_DIM_PER_WARP):
                        k = k_base + elem
                        if k < Int32(self.K):
                            xv = x_row[elem].to(Float32)
                            sum_sq = sum_sq + xv * xv
                        elem = elem + Int32(32)
                    k_base = k_base + Int32(SM100_DECODE_REDUCTION_DIM_PER_WARP)
                total_sq = cute.arch.warp_reduction(sum_sq, operator.add)
                rstd = cute.math.rsqrt(total_sq * Float32(1.0 / self.K) + Float32(self.eps), fastmath=True)

                for local_v in range(self.tile_size_V):
                    v = v_start + Int32(local_v)
                    if v < Int32(self.V):
                        acc = Float32(0.0)
                        w_base = v * Int32(self.weight_stride_V)
                        k2_base = Int32(0)
                        while k2_base < Int32(self.K):
                            x_row = cute.make_tensor(x.iterator + x_base + k2_base, cute.make_layout(SM100_DECODE_REDUCTION_DIM_PER_WARP))
                            norm_row = cute.make_tensor(norm_weight.iterator + k2_base, cute.make_layout(SM100_DECODE_REDUCTION_DIM_PER_WARP))
                            weight_row = cute.make_tensor(weight.iterator + w_base + k2_base, cute.make_layout(SM100_DECODE_REDUCTION_DIM_PER_WARP))
                            elem2 = lane_idx
                            while elem2 < Int32(SM100_DECODE_REDUCTION_DIM_PER_WARP):
                                k2 = k2_base + elem2
                                if k2 < Int32(self.K):
                                    nv = x_row[elem2].to(Float32) * rstd * norm_row[elem2].to(Float32)
                                    acc = acc + nv * weight_row[elem2].to(Float32)
                                elem2 = elem2 + Int32(32)
                            k2_base = k2_base + Int32(SM100_DECODE_REDUCTION_DIM_PER_WARP)
                        total = cute.arch.warp_reduction(acc, operator.add)
                        if lane_idx == Int32(0):
                            out_base = tile_B * Int32(self.logits_stride_B) + row_idx * Int32(self.logits_stride_S)
                            out_tile = cute.make_tensor(logits.iterator + out_base + v_start, cute.make_layout(self.tile_size_V))
                            out_tile[local_v] = total.to(self.logits_dtype)


@dataclass
class DecodeLayerSchedule:
    """Scheduled ops and view keep-alives for one SM100 Llama decoder layer."""

    ops: list
    attention_config: object
    keep_alive: list


def schedule_decode_layer_sm100(
    *,
    layer_idx,
    batch,
    seq_len,
    cache_pos,
    weights,
    k_cache,
    v_cache,
    x_in,
    residual_in,
    x_out,
    residual_out,
    q_buf,
    attn_out_buf,
    mlp_h_buf,
    page_size=DEFAULT_PAGE_SIZE,
    eps=1e-5,
    fa_num_splits=0,
    hidden_size=SM100_DECODE_HIDDEN_DEFAULT,
    intermediate_size=SM100_DECODE_INTERMEDIATE_DEFAULT,
    num_q_heads=SM100_DECODE_NUM_Q_HEADS_DEFAULT,
    num_kv_heads=SM100_DECODE_NUM_KV_HEADS_DEFAULT,
    head_dim=SM100_DECODE_HEAD_DIM_DEFAULT,
    kv_group_size=SM100_DECODE_KV_GROUP_SIZE_DEFAULT,
    matvec_block=SM100_DECODE_MATVEC_BLOCK,
):
    """Schedule one decode decode layer using instruction-shaped SM100 ops.

    This is the public convenience layer for the low-level classes above.  It
    follows the Hazy C++ instruction structure:
    RMS+Q/RoPE, RMS+K/RoPE/cache, RMS+V/cache, attention, O+residual,
    RMS+gate/up+SiLU, and four 2048-wide down-proj reduction blocks.
    """
    from machete.kernels.attention import FlashAttentionOp
    from machete.kernels.attention.flash_decoding import flash_decoding_schedule

    pfx = f"layer.{layer_idx}"
    cos = weights["cos"][cache_pos : cache_pos + seq_len]
    sin = weights["sin"][cache_pos : cache_pos + seq_len]

    q_4d = q_buf.view(batch, seq_len, num_q_heads, head_dim)
    k_window = k_cache[:, : cache_pos + seq_len]
    v_window = v_cache[:, : cache_pos + seq_len]
    o_4d = attn_out_buf.view(batch, seq_len, num_q_heads, head_dim)

    ops = []
    ops += RmsQMatvecRopeSm100Op.schedule(
        x=x_in,
        residual_in=residual_in,
        norm_weight=weights[f"{pfx}.attn_norm"],
        weight=weights[f"{pfx}.W_q"],
        cos=cos,
        sin=sin,
        residual_out=residual_out,
        q=q_buf,
        tile_sizes={"S": seq_len, "O": matvec_block},
        page_size=page_size,
        eps=eps,
    )
    ops += RmsKMatvecRopeCacheSm100Op.schedule(
        x=x_in,
        residual_in=residual_in,
        norm_weight=weights[f"{pfx}.attn_norm"],
        weight=weights[f"{pfx}.W_k"],
        cos=cos,
        sin=sin,
        dst_cache=k_window,
        cache_pos=cache_pos,
        tile_sizes={"S": seq_len, "O": matvec_block},
        page_size=page_size,
        eps=eps,
    )
    ops += RmsVMatvecCacheSm100Op.schedule(
        x=x_in,
        residual_in=residual_in,
        norm_weight=weights[f"{pfx}.attn_norm"],
        weight=weights[f"{pfx}.W_v"],
        cos=cos,
        sin=sin,
        dst_cache=v_window,
        cache_pos=cache_pos,
        tile_sizes={"S": seq_len, "O": matvec_block},
        page_size=page_size,
        eps=eps,
    )

    if k_window.shape[1] <= 256:
        fa_ops = FlashAttentionOp.schedule(
            q=q_4d,
            k=k_window,
            v=v_window,
            o=o_4d,
            kv_group_size=kv_group_size,
            page_size=page_size,
        )
        fa_config = FlashAttentionOp.kernel_config(fa_ops)
    else:
        fa_ops, fa_config = flash_decoding_schedule(
            q=q_4d,
            k=k_window,
            v=v_window,
            o=o_4d,
            kv_group_size=kv_group_size,
            page_size=page_size,
            num_splits=fa_num_splits,
        )
    ops += fa_ops

    ops += MatvecResidualSm100Op.schedule(
        a=attn_out_buf,
        weight=weights[f"{pfx}.W_o"],
        residual_in=residual_out,
        residual_out=residual_out,
        tile_sizes={"S": seq_len, "O": matvec_block},
        page_size=page_size,
    )
    ops += RmsGateUpSiluSm100Op.schedule(
        x=residual_out,
        norm_weight=weights[f"{pfx}.mlp_norm"],
        gate_weight=weights[f"{pfx}.W_gate"],
        up_weight=weights[f"{pfx}.W_up"],
        y=mlp_h_buf,
        tile_sizes={"S": seq_len, "D": matvec_block},
        page_size=page_size,
        eps=eps,
    )

    down_keep = []
    for reduction_block in range(intermediate_size // hidden_size):
        start = reduction_block * hidden_size
        stop = start + hidden_size
        a_block = mlp_h_buf[:, :, start:stop]
        w_block = weights[f"{pfx}.W_down"][:, start:stop]
        down_keep += [a_block, w_block]
        if reduction_block == 0:
            ops += MatvecSm100Op.schedule(
                a=a_block,
                weight=w_block,
                y=x_out,
                tile_sizes={"S": seq_len, "O": matvec_block},
                page_size=page_size,
            )
        else:
            ops += MatvecResidualSm100Op.schedule(
                a=a_block,
                weight=w_block,
                residual_in=x_out,
                residual_out=x_out,
                tile_sizes={"S": seq_len, "O": matvec_block},
                page_size=page_size,
            )

    return DecodeLayerSchedule(
        ops=ops,
        attention_config=fa_config,
        keep_alive=[cos, sin, q_4d, k_window, v_window, o_4d, *down_keep],
    )


def schedule_final_sm100(
    *,
    x,
    residual_in,
    residual_out,
    final_norm,
    lm_head=None,
    logits=None,
    seq_len,
    page_size=DEFAULT_PAGE_SIZE,
    eps=1e-5,
):
    """Schedule final residual and, optionally, fused final RMS+LM-head."""
    ops = ResidualAddSm100Op.schedule(
        x=x,
        residual_in=residual_in,
        residual_out=residual_out,
        tile_sizes={"S": seq_len, "K": SM100_DECODE_REDUCTION_DIM_PER_WARP},
        page_size=page_size,
    )
    if lm_head is not None:
        if logits is None:
            raise ValueError("logits must be provided when lm_head is scheduled")
        ops += FinalRmsLmHeadSm100Op.schedule(
            x=residual_out,
            norm_weight=final_norm,
            weight=lm_head,
            logits=logits,
            tile_sizes={"S": seq_len, "V": SM100_DECODE_MATVEC_BLOCK},
            page_size=page_size,
            eps=eps,
        )
    return ops


__all__ = [
    "MatvecSm100Op",
    "FinalRmsLmHeadSm100Op",
    "MatvecResidualSm100Op",
    "ResidualAddSm100Op",
    "RmsGateUpSiluSm100Op",
    "RmsKMatvecRopeCacheSm100Op",
    "RmsQMatvecRopeSm100Op",
    "RmsVMatvecCacheSm100Op",
    "DecodeLayerSchedule",
    "schedule_decode_layer_sm100",
    "schedule_final_sm100",
]
