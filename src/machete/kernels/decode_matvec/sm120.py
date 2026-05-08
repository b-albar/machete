# Copyright (c) 2025, Machete Authors
"""Instruction-shaped decode matvec ops for SM120 Blackwell GPUs.

These ops intentionally use coarse decode instructions rather than composing
the generic RMSNorm/GEMM/RoPE/GLU kernels:

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
SM120 while leaving room for later TMA page pipelining and tcgen05-specific
micro-optimizations.
"""

import operator
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from machete.megakernel.interpreter import mbarrier_arrive_expect_tx, named_barrier_sync
from machete.megakernel.ops import Op, DEFAULT_PAGE_SIZE, PipelineSpec, TileRange


SM120_DECODE_HIDDEN_DEFAULT = 2048
SM120_DECODE_HEAD_DIM_DEFAULT = 64
SM120_DECODE_ROTARY_D2_DEFAULT = 32
SM120_DECODE_Q_DIM_DEFAULT = 2048
SM120_DECODE_KV_DIM_DEFAULT = 512
SM120_DECODE_INTERMEDIATE_DEFAULT = 8192
SM120_DECODE_VOCAB_DEFAULT = 128256
SM120_DECODE_MATVEC_BLOCK = 16
SM120_DECODE_REDUCTION_DIM_PER_WARP = 512
SM120_DECODE_CONSUMER_WARPS_DEFAULT = SM120_DECODE_HIDDEN_DEFAULT // SM120_DECODE_REDUCTION_DIM_PER_WARP
SM120_DECODE_NUM_Q_HEADS_DEFAULT = SM120_DECODE_Q_DIM_DEFAULT // SM120_DECODE_HEAD_DIM_DEFAULT
SM120_DECODE_NUM_KV_HEADS_DEFAULT = SM120_DECODE_KV_DIM_DEFAULT // SM120_DECODE_HEAD_DIM_DEFAULT
SM120_DECODE_KV_GROUP_SIZE_DEFAULT = SM120_DECODE_NUM_Q_HEADS_DEFAULT // SM120_DECODE_NUM_KV_HEADS_DEFAULT
SM120_OUTPUT_RANGE_PIPELINE = PipelineSpec.range_capable(
    range_axis=2,
    range_end_axis=3,
)


def _finalize_nvfp4_matvec_schedule(op, *, page_size, group_size, k_dim):
    op.static_dims["page_size"] = page_size
    op.static_dims["group_size"] = group_size
    op.static_dims["reduction_tile_K"] = min(SM120_DECODE_REDUCTION_DIM_PER_WARP, k_dim)
    return [op]


@cute.jit
def _silu(x):
    neg = Float32(0.0) - x
    exp_neg = cute.math.exp(neg, fastmath=True)
    return x / (Float32(1.0) + exp_neg)


@cute.jit
def _fp4_e2m1_value(code):
    mag = code & Int32(7)
    out = mag.to(Float32) * Float32(0.5)
    out = out + cute.arch.fmax((mag - Int32(4)).to(Float32), Float32(0.0)) * Float32(0.5)
    out = out + cute.arch.fmax((mag - Int32(6)).to(Float32), Float32(0.0))
    sign = Float32(1.0) - (code >> Int32(3)).to(Float32) * Float32(2.0)
    return out * sign


@dsl_user_op
def _nvfp4_byte_ptr_dot2(ptr, offset: Int32, v0: Float32, v1: Float32, scale: Float32, *, loc=None, ip=None) -> Float32:
    from cutlass._mlir import ir

    result = llvm.inline_asm(
        ir.F32Type.get(),
        [
            ptr.llvm_ptr,
            Int32(offset).ir_value(loc=loc, ip=ip),
            Float32(v0).ir_value(loc=loc, ip=ip),
            Float32(v1).ir_value(loc=loc, ip=ip),
            Float32(scale).ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        ".reg .u64 addr;\n"
        ".reg .b8 byte0;\n"
        ".reg .f16x2 fp4_f16x2;\n"
        ".reg .f16 weight0_f16, weight1_f16;\n"
        ".reg .f32 weight0, weight1, acc;\n"
        "cvt.u64.u32 addr, $2;\n"
        "add.u64 addr, addr, $1;\n"
        "ld.global.u8 byte0, [addr];\n"
        "cvt.rn.f16x2.e2m1x2 fp4_f16x2, byte0;\n"
        "mov.b32 {weight0_f16, weight1_f16}, fp4_f16x2;\n"
        "cvt.f32.f16 weight0, weight0_f16;\n"
        "cvt.f32.f16 weight1, weight1_f16;\n"
        "mul.rn.f32 acc, weight0, $3;\n"
        "fma.rn.f32 acc, weight1, $4, acc;\n"
        "mul.rn.f32 $0, acc, $5;\n"
        "}\n",
        "=f,l,r,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Float32(result)


class _DecodeMatvecSm120Base(Op):
    """Shared schedule/config helpers for decode matvec ops."""

    @classmethod
    def kernel_config(cls, ops):
        from machete.megakernel import MegakernelConfig
        from machete.megakernel.megakernel import NUM_DMA_WARPS

        page_size = max(op.static_dims.get("page_size", DEFAULT_PAGE_SIZE) for op in ops)
        reduction_tile = ops[0].static_dims.get(
            "reduction_tile_K", SM120_DECODE_REDUCTION_DIM_PER_WARP
        )
        k_dim = ops[0].static_dims.get("K", SM120_DECODE_HIDDEN_DEFAULT)
        consumer_warps = max(1, (k_dim + reduction_tile - 1) // reduction_tile)
        return MegakernelConfig(
            threads_per_block=(consumer_warps + NUM_DMA_WARPS) * 32,
            page_size=page_size,
            mma_reg_count=96,
        )


class _Nvfp4WeightMixin:
    """Shared packed E2M1 NVFP4 decode helpers."""

    def __init__(self, **config):
        super().__init__(**config)
        self.group_size = getattr(self, "group_size", 32)
        self.reduction_tile_K = getattr(
            self,
            "reduction_tile_K",
            min(SM120_DECODE_REDUCTION_DIM_PER_WARP, self.K),
        )

    @cute.jit
    def _nvfp4_weight_value(self, packed_row, scale_row, k):
        byte = packed_row[k >> Int32(1)].to(Int32) & Int32(255)
        code = byte & Int32(15)
        if (k & Int32(1)) != Int32(0):
            code = byte >> Int32(4)
        if const_expr(self.group_size == 32):
            scale_idx = k >> Int32(5)
        else:
            scale_idx = k // Int32(self.group_size)
        scale = scale_row[scale_idx].to(Float32)
        return _fp4_e2m1_value(code) * scale

    @cute.jit
    def _dot8_nvfp4_values(self, packed_row, scale_row, k, v0, v1, v2, v3, v4, v5, v6, v7):
        byte_idx = k >> Int32(1)
        if const_expr(self.group_size == 32):
            scale_idx = k >> Int32(5)
        else:
            scale_idx = k // Int32(self.group_size)
        scale = scale_row[scale_idx].to(Float32)
        acc = _nvfp4_byte_ptr_dot2(packed_row.iterator, byte_idx, v0, v1, scale)
        acc = acc + _nvfp4_byte_ptr_dot2(packed_row.iterator, byte_idx + Int32(1), v2, v3, scale)
        acc = acc + _nvfp4_byte_ptr_dot2(packed_row.iterator, byte_idx + Int32(2), v4, v5, scale)
        acc = acc + _nvfp4_byte_ptr_dot2(packed_row.iterator, byte_idx + Int32(3), v6, v7, scale)
        return acc

    @cute.jit
    def _dot8_nvfp4(self, a_row, packed_row, scale_row, k):
        return self._dot8_nvfp4_values(
            packed_row,
            scale_row,
            k,
            a_row[k].to(Float32),
            a_row[k + Int32(1)].to(Float32),
            a_row[k + Int32(2)].to(Float32),
            a_row[k + Int32(3)].to(Float32),
            a_row[k + Int32(4)].to(Float32),
            a_row[k + Int32(5)].to(Float32),
            a_row[k + Int32(6)].to(Float32),
            a_row[k + Int32(7)].to(Float32),
        )

    @cute.jit
    def _dot_nvfp4(self, a_row, packed_row, scale_row):
        lane_idx = cute.arch.lane_idx()
        acc = Float32(0.0)
        full_k = Int32((self.K // 8) * 8)
        k = lane_idx * Int32(8)
        while k < full_k:
            acc = acc + self._dot8_nvfp4(a_row, packed_row, scale_row, k)
            k = k + Int32(256)
        k_tail = full_k + lane_idx
        while k_tail < Int32(self.K):
            acc = acc + a_row[k_tail].to(Float32) * self._nvfp4_weight_value(packed_row, scale_row, k_tail)
            k_tail = k_tail + Int32(32)
        return cute.arch.warp_reduction(acc, operator.add)


class _RmsProjectionSm120Base(_DecodeMatvecSm120Base):
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
            raise ValueError("Decode SM120 decode ops require fp16/bf16 activations")
        self.eps = getattr(self, "eps", 1e-5)
        self.cache_pos = getattr(self, "cache_pos", 0)
        self.page_size = getattr(self, "page_size", DEFAULT_PAGE_SIZE)
        self.reduction_tile_K = getattr(self, "reduction_tile_K", SM120_DECODE_REDUCTION_DIM_PER_WARP)
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
            SM120_DECODE_REDUCTION_DIM_PER_WARP, tensors["x"].shape[-1]
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


class RmsQMatvecRopeSm120Op(_RmsProjectionSm120Base):
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


class RmsKMatvecRopeCacheSm120Op(_RmsProjectionSm120Base):
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


class RmsVMatvecCacheSm120Op(_RmsProjectionSm120Base):
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


class _RmsProjectionNvfp4Sm120Base(_Nvfp4WeightMixin, _DecodeMatvecSm120Base):
    """RMS(input+residual) followed by one packed NVFP4 projection."""

    reads = {
        "x": (None, ("B", "S", "K")),
        "residual_in": (None, ("B", "S", "K")),
        "norm_weight": (None, ("K",)),
        "weight_packed": (cutlass.Uint8, ("O", "K2")),
        "weight_scales": (cutlass.Float16, ("O", "G")),
        "cos": (None, ("S", "D2")),
        "sin": (None, ("S", "D2")),
    }
    writes = {
        "residual_out": (None, ("B", "S", "K")),
    }
    tile = ("B", "S", "O")
    dynamic_dims = ("B",)

    def __init__(self, **config):
        super().__init__(**config)
        if self.x_dtype not in (cutlass.Float16, cutlass.BFloat16):
            raise ValueError("Decode SM120 NVFP4 ops require fp16/bf16 activations")
        self.eps = getattr(self, "eps", 1e-5)
        self.cache_pos = getattr(self, "cache_pos", 0)
        self.head_dim = getattr(self, "head_dim", self.D2 * 2)
        assert self.tile_size_O == 16

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, eps=1e-5,
                 cache_pos=0, group_size=32, head_dim=None, **tensors):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", 16)
        tile_sizes.setdefault("O", 16)
        if tensors["weight_packed"].shape[1] * 2 != tensors["x"].shape[-1]:
            raise ValueError("weight_packed K2 must equal x.K / 2")
        if tensors["weight_scales"].shape[1] != tensors["x"].shape[-1] // group_size:
            raise ValueError("weight_scales G must equal x.K / group_size")
        op = cls._schedule_single(tile_sizes=tile_sizes, **tensors)
        op.static_dims["page_size"] = page_size
        op.static_dims["eps"] = eps
        op.static_dims["cache_pos"] = cache_pos
        op.static_dims["head_dim"] = int(head_dim or tensors["cos"].shape[1] * 2)
        op.static_dims["group_size"] = group_size
        op.static_dims["reduction_tile_K"] = min(
            SM120_DECODE_REDUCTION_DIM_PER_WARP, tensors["x"].shape[-1]
        )
        return [op]

    @cute.jit
    def _row_rstd(self, tile_B, row_idx, x, residual_in):
        lane_idx = cute.arch.lane_idx()
        x_base = tile_B * Int32(self.x_stride_B) + row_idx * Int32(self.x_stride_S)
        res_base = tile_B * Int32(self.residual_in_stride_B) + row_idx * Int32(self.residual_in_stride_S)

        sum_sq = Float32(0.0)
        k = lane_idx
        while k < Int32(self.K):
            x_row = cute.make_tensor(x.iterator + x_base, cute.make_layout(self.K))
            res_row = cute.make_tensor(residual_in.iterator + res_base, cute.make_layout(self.K))
            val = x_row[k].to(Float32) + res_row[k].to(Float32)
            sum_sq = sum_sq + val * val
            k = k + Int32(32)
        total = cute.arch.warp_reduction(sum_sq, operator.add)
        return cute.math.rsqrt(total * Float32(1.0 / self.K) + Float32(self.eps), fastmath=True)

    @cute.jit
    def _row_rstd_and_store_residual(self, tile_B, row_idx, x, residual_in, residual_out):
        lane_idx = cute.arch.lane_idx()
        x_base = tile_B * Int32(self.x_stride_B) + row_idx * Int32(self.x_stride_S)
        res_base = tile_B * Int32(self.residual_in_stride_B) + row_idx * Int32(self.residual_in_stride_S)
        out_base = tile_B * Int32(self.residual_out_stride_B) + row_idx * Int32(self.residual_out_stride_S)

        sum_sq = Float32(0.0)
        k = lane_idx
        while k < Int32(self.K):
            x_row = cute.make_tensor(x.iterator + x_base, cute.make_layout(self.K))
            res_row = cute.make_tensor(residual_in.iterator + res_base, cute.make_layout(self.K))
            out_row = cute.make_tensor(residual_out.iterator + out_base, cute.make_layout(self.K))
            val = x_row[k].to(Float32) + res_row[k].to(Float32)
            sum_sq = sum_sq + val * val
            out_row[k] = val.to(self.residual_out_dtype)
            k = k + Int32(32)
        total = cute.arch.warp_reduction(sum_sq, operator.add)
        return cute.math.rsqrt(total * Float32(1.0 / self.K) + Float32(self.eps), fastmath=True)

    @cute.jit
    def _dot_output(self, tile_B, row_idx, out_idx, rstd, x, residual_in,
                    norm_weight, weight_packed, weight_scales):
        lane_idx = cute.arch.lane_idx()
        x_base = tile_B * Int32(self.x_stride_B) + row_idx * Int32(self.x_stride_S)
        res_base = tile_B * Int32(self.residual_in_stride_B) + row_idx * Int32(self.residual_in_stride_S)
        packed_base = out_idx * Int32(self.weight_packed_stride_O)
        scale_base = out_idx * Int32(self.weight_scales_stride_O)
        x_row = cute.make_tensor(x.iterator + x_base, cute.make_layout(self.K))
        res_row = cute.make_tensor(residual_in.iterator + res_base, cute.make_layout(self.K))
        norm_row = cute.make_tensor(norm_weight.iterator, cute.make_layout(self.K))
        packed_row = cute.make_tensor(weight_packed.iterator + packed_base, cute.make_layout(self.K2))
        scale_row = cute.make_tensor(weight_scales.iterator + scale_base, cute.make_layout(self.G))
        acc = Float32(0.0)
        full_k = Int32((self.K // 8) * 8)
        k = lane_idx * Int32(8)
        while k < full_k:
            v0 = (x_row[k].to(Float32) + res_row[k].to(Float32)) * rstd * norm_row[k].to(Float32)
            v1 = (x_row[k + Int32(1)].to(Float32) + res_row[k + Int32(1)].to(Float32)) * rstd * norm_row[k + Int32(1)].to(Float32)
            v2 = (x_row[k + Int32(2)].to(Float32) + res_row[k + Int32(2)].to(Float32)) * rstd * norm_row[k + Int32(2)].to(Float32)
            v3 = (x_row[k + Int32(3)].to(Float32) + res_row[k + Int32(3)].to(Float32)) * rstd * norm_row[k + Int32(3)].to(Float32)
            v4 = (x_row[k + Int32(4)].to(Float32) + res_row[k + Int32(4)].to(Float32)) * rstd * norm_row[k + Int32(4)].to(Float32)
            v5 = (x_row[k + Int32(5)].to(Float32) + res_row[k + Int32(5)].to(Float32)) * rstd * norm_row[k + Int32(5)].to(Float32)
            v6 = (x_row[k + Int32(6)].to(Float32) + res_row[k + Int32(6)].to(Float32)) * rstd * norm_row[k + Int32(6)].to(Float32)
            v7 = (x_row[k + Int32(7)].to(Float32) + res_row[k + Int32(7)].to(Float32)) * rstd * norm_row[k + Int32(7)].to(Float32)
            acc = acc + self._dot8_nvfp4_values(
                packed_row, scale_row, k,
                v0, v1, v2, v3, v4, v5, v6, v7,
            )
            k = k + Int32(256)
        k = full_k + lane_idx
        while k < Int32(self.K):
            nv = (x_row[k].to(Float32) + res_row[k].to(Float32)) * rstd * norm_row[k].to(Float32)
            acc = acc + nv * self._nvfp4_weight_value(packed_row, scale_row, k)
            k = k + Int32(32)
        return cute.arch.warp_reduction(acc, operator.add)

    @cute.jit
    def _rope_value(self, tile_B, row_idx, out_idx, value, rstd, x, residual_in,
                    norm_weight, weight_packed, weight_scales, cos, sin):
        dim = out_idx % Int32(self.head_dim)
        out = value
        if dim < Int32(self.D2):
            pair_idx = out_idx + Int32(self.D2)
            pair = self._dot_output(
                tile_B, row_idx, pair_idx, rstd, x, residual_in,
                norm_weight, weight_packed, weight_scales,
            )
            cos_row = cute.make_tensor(cos.iterator + row_idx * Int32(self.cos_stride_S), cute.make_layout(self.D2))
            sin_row = cute.make_tensor(sin.iterator + row_idx * Int32(self.sin_stride_S), cute.make_layout(self.D2))
            out = value * cos_row[dim].to(Float32) - pair * sin_row[dim].to(Float32)
        elif dim < Int32(self.head_dim):
            pair_dim = dim - Int32(self.D2)
            pair_idx = out_idx - Int32(self.D2)
            pair = self._dot_output(
                tile_B, row_idx, pair_idx, rstd, x, residual_in,
                norm_weight, weight_packed, weight_scales,
            )
            cos_row = cute.make_tensor(cos.iterator + row_idx * Int32(self.cos_stride_S), cute.make_layout(self.D2))
            sin_row = cute.make_tensor(sin.iterator + row_idx * Int32(self.sin_stride_S), cute.make_layout(self.D2))
            out = value * cos_row[pair_dim].to(Float32) + pair * sin_row[pair_dim].to(Float32)
        return out


class RmsQMatvecRopeNvfp4Sm120Op(_RmsProjectionNvfp4Sm120Base):
    """Fused RMS(input+residual) + packed NVFP4 Q matvec + RoPE."""

    writes = {
        "residual_out": (None, ("B", "S", "K")),
        "q": (None, ("B", "S", "O")),
    }

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_O, x, residual_in, norm_weight,
                weight_packed, weight_scales, cos, sin, residual_out, q):
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
                        val = self._dot_output(
                            tile_B, row_idx, out_idx, rstd, x, residual_in,
                            norm_weight, weight_packed, weight_scales,
                        )
                        val = self._rope_value(
                            tile_B, row_idx, out_idx, val, rstd, x, residual_in,
                            norm_weight, weight_packed, weight_scales, cos, sin,
                        )
                        if lane_idx == Int32(0):
                            q_base = tile_B * Int32(self.q_stride_B) + row_idx * Int32(self.q_stride_S)
                            q_tile = cute.make_tensor(q.iterator + q_base + out_start, cute.make_layout(self.tile_size_O))
                            q_tile[local_o] = val.to(self.q_dtype)


class RmsKMatvecRopeCacheNvfp4Sm120Op(_RmsProjectionNvfp4Sm120Base):
    """Fused RMS(input+residual) + packed NVFP4 K matvec + RoPE + cache append."""

    writes = {"dst_cache": (None, ("B", "T", "H", "HD"))}

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_O, x, residual_in, norm_weight,
                weight_packed, weight_scales, cos, sin, dst_cache):
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
                        val = self._dot_output(
                            tile_B, row_idx, out_idx, rstd, x, residual_in,
                            norm_weight, weight_packed, weight_scales,
                        )
                        val = self._rope_value(
                            tile_B, row_idx, out_idx, val, rstd, x, residual_in,
                            norm_weight, weight_packed, weight_scales, cos, sin,
                        )
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


class RmsVMatvecCacheNvfp4Sm120Op(_RmsProjectionNvfp4Sm120Base):
    """Fused RMS(input+residual) + packed NVFP4 V matvec + cache append."""

    writes = {"dst_cache": (None, ("B", "T", "H", "HD"))}

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_O, x, residual_in, norm_weight,
                weight_packed, weight_scales, cos, sin, dst_cache):
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
                        val = self._dot_output(
                            tile_B, row_idx, out_idx, rstd, x, residual_in,
                            norm_weight, weight_packed, weight_scales,
                        )
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


class RmsMatvecNvfp4Sm120Op(_RmsProjectionNvfp4Sm120Base):
    """Fused RMS(input+residual) + packed NVFP4 projection."""

    writes = {
        "residual_out": (None, ("B", "S", "K")),
        "y": (None, ("B", "S", "O")),
    }

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, eps=1e-5,
                 group_size=32, **tensors):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", 16)
        tile_sizes.setdefault("O", 16)
        if tensors["weight_packed"].shape[1] * 2 != tensors["x"].shape[-1]:
            raise ValueError("weight_packed K2 must equal x.K / 2")
        if tensors["weight_scales"].shape[1] != tensors["x"].shape[-1] // group_size:
            raise ValueError("weight_scales G must equal x.K / group_size")
        op = cls._schedule_single(tile_sizes=tile_sizes, **tensors)
        op.static_dims["page_size"] = page_size
        op.static_dims["eps"] = eps
        op.static_dims["cache_pos"] = 0
        op.static_dims["head_dim"] = tensors["x"].shape[-1]
        op.static_dims["group_size"] = group_size
        op.static_dims["reduction_tile_K"] = min(
            SM120_DECODE_REDUCTION_DIM_PER_WARP, tensors["x"].shape[-1]
        )
        return [op]

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_O, x, residual_in, norm_weight,
                weight_packed, weight_scales, cos, sin, residual_out, y):
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
                        val = self._dot_output(
                            tile_B, row_idx, out_idx, rstd, x, residual_in,
                            norm_weight, weight_packed, weight_scales,
                        )
                        if lane_idx == Int32(0):
                            y_base = tile_B * Int32(self.y_stride_B) + row_idx * Int32(self.y_stride_S)
                            y_tile = cute.make_tensor(y.iterator + y_base + out_start, cute.make_layout(self.tile_size_O))
                            y_tile[local_o] = val.to(self.y_dtype)


class RmsReadMatvecNvfp4Sm120Op(RmsMatvecNvfp4Sm120Op):
    """Fused RMS(input+residual) + packed NVFP4 projection without residual write."""

    writes = {"y": (None, ("B", "S", "O"))}

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_O, x, residual_in, norm_weight,
                weight_packed, weight_scales, cos, sin, y):
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
                        val = self._dot_output(
                            tile_B, row_idx, out_idx, rstd, x, residual_in,
                            norm_weight, weight_packed, weight_scales,
                        )
                        if lane_idx == Int32(0):
                            y_base = tile_B * Int32(self.y_stride_B) + row_idx * Int32(self.y_stride_S)
                            y_tile = cute.make_tensor(y.iterator + y_base + out_start, cute.make_layout(self.tile_size_O))
                            y_tile[local_o] = val.to(self.y_dtype)


class MatvecResidualSm120Op(_DecodeMatvecSm120Base):
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
                            a_row = cute.make_tensor(a.iterator + a_base + k_base, cute.make_layout(SM120_DECODE_REDUCTION_DIM_PER_WARP))
                            weight_row = cute.make_tensor(weight.iterator + w_base + k_base, cute.make_layout(SM120_DECODE_REDUCTION_DIM_PER_WARP))
                            elem = lane_idx
                            while elem < Int32(SM120_DECODE_REDUCTION_DIM_PER_WARP):
                                k = k_base + elem
                                if k < Int32(self.K):
                                    acc = acc + a_row[elem].to(Float32) * weight_row[elem].to(Float32)
                                elem = elem + Int32(32)
                            k_base = k_base + Int32(SM120_DECODE_REDUCTION_DIM_PER_WARP)
                        total = cute.arch.warp_reduction(acc, operator.add)
                        if lane_idx == Int32(0):
                            out_base = tile_B * Int32(self.residual_out_stride_B) + row_idx * Int32(self.residual_out_stride_S)
                            res_base = tile_B * Int32(self.residual_in_stride_B) + row_idx * Int32(self.residual_in_stride_S)
                            res_tile = cute.make_tensor(residual_in.iterator + res_base + out_start, cute.make_layout(self.tile_size_O))
                            out_tile = cute.make_tensor(residual_out.iterator + out_base + out_start, cute.make_layout(self.tile_size_O))
                            val = total + res_tile[local_o].to(Float32)
                            out_tile[local_o] = val.to(self.residual_out_dtype)


class MatvecSm120Op(_DecodeMatvecSm120Base):
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
                            a_row = cute.make_tensor(a.iterator + a_base + k_base, cute.make_layout(SM120_DECODE_REDUCTION_DIM_PER_WARP))
                            weight_row = cute.make_tensor(weight.iterator + w_base + k_base, cute.make_layout(SM120_DECODE_REDUCTION_DIM_PER_WARP))
                            elem = lane_idx
                            while elem < Int32(SM120_DECODE_REDUCTION_DIM_PER_WARP):
                                k = k_base + elem
                                if k < Int32(self.K):
                                    acc = acc + a_row[elem].to(Float32) * weight_row[elem].to(Float32)
                                elem = elem + Int32(32)
                            k_base = k_base + Int32(SM120_DECODE_REDUCTION_DIM_PER_WARP)
                        total = cute.arch.warp_reduction(acc, operator.add)
                        if lane_idx == Int32(0):
                            y_base = tile_B * Int32(self.y_stride_B) + row_idx * Int32(self.y_stride_S)
                            y_tile = cute.make_tensor(y.iterator + y_base + out_start, cute.make_layout(self.tile_size_O))
                            y_tile[local_o] = total.to(self.y_dtype)


class _MatvecNvfp4Sm120Base(_Nvfp4WeightMixin, _DecodeMatvecSm120Base):
    """Packed NVFP4 projection matvec base."""

    pipeline = SM120_OUTPUT_RANGE_PIPELINE
    reads = {
        "a": (None, ("B", "S", "K")),
        "weight_packed": (cutlass.Uint8, ("O", "K2")),
        "weight_scales": (cutlass.Float16, ("O", "G")),
    }
    tile = ("B", "S", "O")
    dynamic_dims = ("B",)

    @classmethod
    def schedule(
        cls,
        tile_sizes=None,
        page_size=DEFAULT_PAGE_SIZE,
        group_size=32,
        tile_range=None,
        **tensors,
    ):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", 16)
        tile_sizes.setdefault("O", 16)
        if tensors["weight_packed"].shape[1] * 2 != tensors["a"].shape[-1]:
            raise ValueError("weight_packed K2 must equal a.K / 2")
        if tensors["weight_scales"].shape[1] != tensors["a"].shape[-1] // group_size:
            raise ValueError("weight_scales G must equal a.K / group_size")
        op = cls._schedule_single(tile_sizes=tile_sizes, tile_range=tile_range, **tensors)
        return _finalize_nvfp4_matvec_schedule(
            op,
            page_size=page_size,
            group_size=group_size,
            k_dim=tensors["a"].shape[-1],
        )

    @cute.jit
    def _dot_output(self, tile_B, row_idx, out_idx, a, weight_packed, weight_scales):
        a_base = tile_B * Int32(self.a_stride_B) + row_idx * Int32(self.a_stride_S)
        packed_base = out_idx * Int32(self.weight_packed_stride_O)
        scale_base = out_idx * Int32(self.weight_scales_stride_O)
        a_row = cute.make_tensor(a.iterator + a_base, cute.make_layout(self.K))
        packed_row = cute.make_tensor(weight_packed.iterator + packed_base, cute.make_layout(self.K2))
        scale_row = cute.make_tensor(weight_scales.iterator + scale_base, cute.make_layout(self.G))
        return self._dot_nvfp4(a_row, packed_row, scale_row)


class MatvecNvfp4Sm120Op(_MatvecNvfp4Sm120Base):
    """Packed NVFP4 down/projection matvec."""

    writes = {"y": (None, ("B", "S", "O"))}

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_O, a, weight_packed, weight_scales, y):
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.threads_per_row // 32
        row_start = tile_S * Int32(self.tile_size_S)
        out_start = tile_O * Int32(self.tile_size_O)
        for local_work in range(warp_idx, self.tile_size_S * self.tile_size_O, num_warps):
            local_row = local_work // self.tile_size_O
            local_o = local_work - local_row * self.tile_size_O
            row_idx = row_start + Int32(local_row)
            if row_idx < Int32(self.S):
                out_idx = out_start + Int32(local_o)
                if out_idx < Int32(self.O):
                    total = self._dot_output(tile_B, row_idx, out_idx, a, weight_packed, weight_scales)
                    if lane_idx == Int32(0):
                        y_base = tile_B * Int32(self.y_stride_B) + row_idx * Int32(self.y_stride_S)
                        y_tile = cute.make_tensor(y.iterator + y_base + out_start, cute.make_layout(self.tile_size_O))
                        y_tile[local_o] = total.to(self.y_dtype)


class MatvecPairNvfp4Sm120Op(_Nvfp4WeightMixin, _DecodeMatvecSm120Base):
    """Packed NVFP4 matvec for two same-sized projections sharing one input."""

    pipeline = SM120_OUTPUT_RANGE_PIPELINE
    reads = {
        "a": (None, ("B", "S", "K")),
        "weight0_packed": (cutlass.Uint8, ("O", "K2")),
        "weight0_scales": (cutlass.Float16, ("O", "G")),
        "weight1_packed": (cutlass.Uint8, ("O", "K2")),
        "weight1_scales": (cutlass.Float16, ("O", "G")),
    }
    writes = {
        "y0": (None, ("B", "S", "O")),
        "y1": (None, ("B", "S", "O")),
    }
    tile = ("B", "S", "O")
    dynamic_dims = ("B",)

    @classmethod
    def schedule(
        cls,
        tile_sizes=None,
        page_size=DEFAULT_PAGE_SIZE,
        group_size=32,
        tile_range=None,
        **tensors,
    ):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", 16)
        tile_sizes.setdefault("O", 16)
        if tensors["weight0_packed"].shape != tensors["weight1_packed"].shape:
            raise ValueError("paired NVFP4 projections must have matching packed shapes")
        if tensors["weight0_scales"].shape != tensors["weight1_scales"].shape:
            raise ValueError("paired NVFP4 projections must have matching scale shapes")
        if tensors["weight0_packed"].shape[1] * 2 != tensors["a"].shape[-1]:
            raise ValueError("weight K2 must equal a.K / 2")
        if tensors["weight0_scales"].shape[1] != tensors["a"].shape[-1] // group_size:
            raise ValueError("weight scales G must equal a.K / group_size")
        op = cls._schedule_single(tile_sizes=tile_sizes, tile_range=tile_range, **tensors)
        return _finalize_nvfp4_matvec_schedule(
            op,
            page_size=page_size,
            group_size=group_size,
            k_dim=tensors["a"].shape[-1],
        )

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_O, a,
                weight0_packed, weight0_scales, weight1_packed, weight1_scales,
                y0, y1):
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.threads_per_row // 32
        row_start = tile_S * Int32(self.tile_size_S)
        out_start = tile_O * Int32(self.tile_size_O)
        for local_work in range(warp_idx, self.tile_size_S * self.tile_size_O, num_warps):
            local_row = local_work // self.tile_size_O
            local_o = local_work - local_row * self.tile_size_O
            row_idx = row_start + Int32(local_row)
            if row_idx < Int32(self.S):
                out_idx = out_start + Int32(local_o)
                if out_idx < Int32(self.O):
                    a_base = tile_B * Int32(self.a_stride_B) + row_idx * Int32(self.a_stride_S)
                    a_row = cute.make_tensor(a.iterator + a_base, cute.make_layout(self.K))
                    packed0_row = cute.make_tensor(
                        weight0_packed.iterator + out_idx * Int32(self.weight0_packed_stride_O),
                        cute.make_layout(self.K2),
                    )
                    scale0_row = cute.make_tensor(
                        weight0_scales.iterator + out_idx * Int32(self.weight0_scales_stride_O),
                        cute.make_layout(self.G),
                    )
                    packed1_row = cute.make_tensor(
                        weight1_packed.iterator + out_idx * Int32(self.weight1_packed_stride_O),
                        cute.make_layout(self.K2),
                    )
                    scale1_row = cute.make_tensor(
                        weight1_scales.iterator + out_idx * Int32(self.weight1_scales_stride_O),
                        cute.make_layout(self.G),
                    )
                    acc0 = Float32(0.0)
                    acc1 = Float32(0.0)
                    full_k = Int32((self.K // 8) * 8)
                    k = lane_idx * Int32(8)
                    while k < full_k:
                        v0 = a_row[k].to(Float32)
                        v1 = a_row[k + Int32(1)].to(Float32)
                        v2 = a_row[k + Int32(2)].to(Float32)
                        v3 = a_row[k + Int32(3)].to(Float32)
                        v4 = a_row[k + Int32(4)].to(Float32)
                        v5 = a_row[k + Int32(5)].to(Float32)
                        v6 = a_row[k + Int32(6)].to(Float32)
                        v7 = a_row[k + Int32(7)].to(Float32)
                        acc0 = acc0 + self._dot8_nvfp4_values(
                            packed0_row, scale0_row, k,
                            v0, v1, v2, v3, v4, v5, v6, v7,
                        )
                        acc1 = acc1 + self._dot8_nvfp4_values(
                            packed1_row, scale1_row, k,
                            v0, v1, v2, v3, v4, v5, v6, v7,
                        )
                        k = k + Int32(256)
                    k = full_k + lane_idx
                    while k < Int32(self.K):
                        av = a_row[k].to(Float32)
                        acc0 = acc0 + av * self._nvfp4_weight_value(packed0_row, scale0_row, k)
                        acc1 = acc1 + av * self._nvfp4_weight_value(packed1_row, scale1_row, k)
                        k = k + Int32(32)
                    total0 = cute.arch.warp_reduction(acc0, operator.add)
                    total1 = cute.arch.warp_reduction(acc1, operator.add)
                    if lane_idx == Int32(0):
                        y0_base = tile_B * Int32(self.y0_stride_B) + row_idx * Int32(self.y0_stride_S)
                        y1_base = tile_B * Int32(self.y1_stride_B) + row_idx * Int32(self.y1_stride_S)
                        y0_tile = cute.make_tensor(y0.iterator + y0_base + out_start, cute.make_layout(self.tile_size_O))
                        y1_tile = cute.make_tensor(y1.iterator + y1_base + out_start, cute.make_layout(self.tile_size_O))
                        y0_tile[local_o] = total0.to(self.y0_dtype)
                        y1_tile[local_o] = total1.to(self.y1_dtype)


class MatvecQuadNvfp4Sm120Op(_Nvfp4WeightMixin, _DecodeMatvecSm120Base):
    """Packed NVFP4 matvec for four same-sized projections sharing one input."""

    pipeline = SM120_OUTPUT_RANGE_PIPELINE
    reads = {
        "a": (None, ("B", "S", "K")),
        "weight0_packed": (cutlass.Uint8, ("O", "K2")),
        "weight0_scales": (cutlass.Float16, ("O", "G")),
        "weight1_packed": (cutlass.Uint8, ("O", "K2")),
        "weight1_scales": (cutlass.Float16, ("O", "G")),
        "weight2_packed": (cutlass.Uint8, ("O", "K2")),
        "weight2_scales": (cutlass.Float16, ("O", "G")),
        "weight3_packed": (cutlass.Uint8, ("O", "K2")),
        "weight3_scales": (cutlass.Float16, ("O", "G")),
    }
    writes = {
        "y0": (None, ("B", "S", "O")),
        "y1": (None, ("B", "S", "O")),
        "y2": (None, ("B", "S", "O")),
        "y3": (None, ("B", "S", "O")),
    }
    tile = ("B", "S", "O")
    dynamic_dims = ("B",)

    @classmethod
    def schedule(
        cls,
        tile_sizes=None,
        page_size=DEFAULT_PAGE_SIZE,
        group_size=32,
        tile_range=None,
        **tensors,
    ):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", 16)
        tile_sizes.setdefault("O", 16)
        packed_shape = tensors["weight0_packed"].shape
        scale_shape = tensors["weight0_scales"].shape
        for idx in range(1, 4):
            if tensors[f"weight{idx}_packed"].shape != packed_shape:
                raise ValueError("quad NVFP4 projections must have matching packed shapes")
            if tensors[f"weight{idx}_scales"].shape != scale_shape:
                raise ValueError("quad NVFP4 projections must have matching scale shapes")
        if packed_shape[1] * 2 != tensors["a"].shape[-1]:
            raise ValueError("weight K2 must equal a.K / 2")
        if scale_shape[1] != tensors["a"].shape[-1] // group_size:
            raise ValueError("weight scales G must equal a.K / group_size")
        op = cls._schedule_single(tile_sizes=tile_sizes, tile_range=tile_range, **tensors)
        return _finalize_nvfp4_matvec_schedule(
            op,
            page_size=page_size,
            group_size=group_size,
            k_dim=tensors["a"].shape[-1],
        )

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_O, a,
                weight0_packed, weight0_scales, weight1_packed, weight1_scales,
                weight2_packed, weight2_scales, weight3_packed, weight3_scales,
                y0, y1, y2, y3):
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.threads_per_row // 32
        row_start = tile_S * Int32(self.tile_size_S)
        out_start = tile_O * Int32(self.tile_size_O)
        for local_work in range(warp_idx, self.tile_size_S * self.tile_size_O, num_warps):
            local_row = local_work // self.tile_size_O
            local_o = local_work - local_row * self.tile_size_O
            row_idx = row_start + Int32(local_row)
            if row_idx < Int32(self.S):
                out_idx = out_start + Int32(local_o)
                if out_idx < Int32(self.O):
                    a_base = tile_B * Int32(self.a_stride_B) + row_idx * Int32(self.a_stride_S)
                    a_row = cute.make_tensor(a.iterator + a_base, cute.make_layout(self.K))
                    packed0_row = cute.make_tensor(weight0_packed.iterator + out_idx * Int32(self.weight0_packed_stride_O), cute.make_layout(self.K2))
                    scale0_row = cute.make_tensor(weight0_scales.iterator + out_idx * Int32(self.weight0_scales_stride_O), cute.make_layout(self.G))
                    packed1_row = cute.make_tensor(weight1_packed.iterator + out_idx * Int32(self.weight1_packed_stride_O), cute.make_layout(self.K2))
                    scale1_row = cute.make_tensor(weight1_scales.iterator + out_idx * Int32(self.weight1_scales_stride_O), cute.make_layout(self.G))
                    packed2_row = cute.make_tensor(weight2_packed.iterator + out_idx * Int32(self.weight2_packed_stride_O), cute.make_layout(self.K2))
                    scale2_row = cute.make_tensor(weight2_scales.iterator + out_idx * Int32(self.weight2_scales_stride_O), cute.make_layout(self.G))
                    packed3_row = cute.make_tensor(weight3_packed.iterator + out_idx * Int32(self.weight3_packed_stride_O), cute.make_layout(self.K2))
                    scale3_row = cute.make_tensor(weight3_scales.iterator + out_idx * Int32(self.weight3_scales_stride_O), cute.make_layout(self.G))
                    acc0 = Float32(0.0)
                    acc1 = Float32(0.0)
                    acc2 = Float32(0.0)
                    acc3 = Float32(0.0)
                    full_k = Int32((self.K // 8) * 8)
                    k = lane_idx * Int32(8)
                    while k < full_k:
                        v0 = a_row[k].to(Float32)
                        v1 = a_row[k + Int32(1)].to(Float32)
                        v2 = a_row[k + Int32(2)].to(Float32)
                        v3 = a_row[k + Int32(3)].to(Float32)
                        v4 = a_row[k + Int32(4)].to(Float32)
                        v5 = a_row[k + Int32(5)].to(Float32)
                        v6 = a_row[k + Int32(6)].to(Float32)
                        v7 = a_row[k + Int32(7)].to(Float32)
                        acc0 = acc0 + self._dot8_nvfp4_values(packed0_row, scale0_row, k, v0, v1, v2, v3, v4, v5, v6, v7)
                        acc1 = acc1 + self._dot8_nvfp4_values(packed1_row, scale1_row, k, v0, v1, v2, v3, v4, v5, v6, v7)
                        acc2 = acc2 + self._dot8_nvfp4_values(packed2_row, scale2_row, k, v0, v1, v2, v3, v4, v5, v6, v7)
                        acc3 = acc3 + self._dot8_nvfp4_values(packed3_row, scale3_row, k, v0, v1, v2, v3, v4, v5, v6, v7)
                        k = k + Int32(256)
                    k = full_k + lane_idx
                    while k < Int32(self.K):
                        av = a_row[k].to(Float32)
                        acc0 = acc0 + av * self._nvfp4_weight_value(packed0_row, scale0_row, k)
                        acc1 = acc1 + av * self._nvfp4_weight_value(packed1_row, scale1_row, k)
                        acc2 = acc2 + av * self._nvfp4_weight_value(packed2_row, scale2_row, k)
                        acc3 = acc3 + av * self._nvfp4_weight_value(packed3_row, scale3_row, k)
                        k = k + Int32(32)
                    total0 = cute.arch.warp_reduction(acc0, operator.add)
                    total1 = cute.arch.warp_reduction(acc1, operator.add)
                    total2 = cute.arch.warp_reduction(acc2, operator.add)
                    total3 = cute.arch.warp_reduction(acc3, operator.add)
                    if lane_idx == Int32(0):
                        y0_base = tile_B * Int32(self.y0_stride_B) + row_idx * Int32(self.y0_stride_S)
                        y1_base = tile_B * Int32(self.y1_stride_B) + row_idx * Int32(self.y1_stride_S)
                        y2_base = tile_B * Int32(self.y2_stride_B) + row_idx * Int32(self.y2_stride_S)
                        y3_base = tile_B * Int32(self.y3_stride_B) + row_idx * Int32(self.y3_stride_S)
                        y0_tile = cute.make_tensor(y0.iterator + y0_base + out_start, cute.make_layout(self.tile_size_O))
                        y1_tile = cute.make_tensor(y1.iterator + y1_base + out_start, cute.make_layout(self.tile_size_O))
                        y2_tile = cute.make_tensor(y2.iterator + y2_base + out_start, cute.make_layout(self.tile_size_O))
                        y3_tile = cute.make_tensor(y3.iterator + y3_base + out_start, cute.make_layout(self.tile_size_O))
                        y0_tile[local_o] = total0.to(self.y0_dtype)
                        y1_tile[local_o] = total1.to(self.y1_dtype)
                        y2_tile[local_o] = total2.to(self.y2_dtype)
                        y3_tile[local_o] = total3.to(self.y3_dtype)


class MatvecResidualNvfp4Sm120Op(_MatvecNvfp4Sm120Base):
    """Packed NVFP4 matvec with residual add."""

    reads = {
        "a": (None, ("B", "S", "K")),
        "weight_packed": (cutlass.Uint8, ("O", "K2")),
        "weight_scales": (cutlass.Float16, ("O", "G")),
        "residual_in": (None, ("B", "S", "O")),
    }
    writes = {"residual_out": (None, ("B", "S", "O"))}

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_O, a, weight_packed,
                weight_scales, residual_in, residual_out):
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.threads_per_row // 32
        row_start = tile_S * Int32(self.tile_size_S)
        out_start = tile_O * Int32(self.tile_size_O)
        for local_work in range(warp_idx, self.tile_size_S * self.tile_size_O, num_warps):
            local_row = local_work // self.tile_size_O
            local_o = local_work - local_row * self.tile_size_O
            row_idx = row_start + Int32(local_row)
            if row_idx < Int32(self.S):
                out_idx = out_start + Int32(local_o)
                if out_idx < Int32(self.O):
                    total = self._dot_output(tile_B, row_idx, out_idx, a, weight_packed, weight_scales)
                    if lane_idx == Int32(0):
                        out_base = tile_B * Int32(self.residual_out_stride_B) + row_idx * Int32(self.residual_out_stride_S)
                        res_base = tile_B * Int32(self.residual_in_stride_B) + row_idx * Int32(self.residual_in_stride_S)
                        res_tile = cute.make_tensor(residual_in.iterator + res_base + out_start, cute.make_layout(self.tile_size_O))
                        out_tile = cute.make_tensor(residual_out.iterator + out_base + out_start, cute.make_layout(self.tile_size_O))
                        out_tile[local_o] = (total + res_tile[local_o].to(Float32)).to(self.residual_out_dtype)


class ResidualAddSm120Op(_DecodeMatvecSm120Base):
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


class RmsAddNormSm120Op(_DecodeMatvecSm120Base):
    """Fused residual add + RMSNorm for decode projection reuse."""

    reads = {
        "x": (None, ("B", "S", "K")),
        "residual_in": (None, ("B", "S", "K")),
        "norm_weight": (None, ("K",)),
    }
    writes = {
        "residual_out": (None, ("B", "S", "K")),
        "y": (None, ("B", "S", "K")),
    }
    tile = ("B", "S")
    dynamic_dims = ("B",)

    def __init__(self, **config):
        super().__init__(**config)
        self.eps = getattr(self, "eps", 1e-5)

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, eps=1e-5, **tensors):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", 16)
        op = cls._schedule_single(tile_sizes=tile_sizes, **tensors)
        op.static_dims["page_size"] = page_size
        op.static_dims["eps"] = eps
        return [op]

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_2, x, residual_in, norm_weight, residual_out, y):
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.threads_per_row // 32
        row_start = tile_S * Int32(self.tile_size_S)
        for local_row in range(warp_idx, self.tile_size_S, num_warps):
            row_idx = row_start + Int32(local_row)
            if row_idx < Int32(self.S):
                x_base = tile_B * Int32(self.x_stride_B) + row_idx * Int32(self.x_stride_S)
                res_base = tile_B * Int32(self.residual_in_stride_B) + row_idx * Int32(self.residual_in_stride_S)
                out_base = tile_B * Int32(self.residual_out_stride_B) + row_idx * Int32(self.residual_out_stride_S)
                y_base = tile_B * Int32(self.y_stride_B) + row_idx * Int32(self.y_stride_S)
                x_row = cute.make_tensor(x.iterator + x_base, cute.make_layout(self.K))
                res_row = cute.make_tensor(residual_in.iterator + res_base, cute.make_layout(self.K))
                out_row = cute.make_tensor(residual_out.iterator + out_base, cute.make_layout(self.K))
                y_row = cute.make_tensor(y.iterator + y_base, cute.make_layout(self.K))
                norm_row = cute.make_tensor(norm_weight.iterator, cute.make_layout(self.K))

                sum_sq = Float32(0.0)
                k = lane_idx
                while k < Int32(self.K):
                    val = x_row[k].to(Float32) + res_row[k].to(Float32)
                    sum_sq = sum_sq + val * val
                    k = k + Int32(32)
                total = cute.arch.warp_reduction(sum_sq, operator.add)
                rstd = cute.math.rsqrt(total * Float32(1.0 / self.K) + Float32(self.eps), fastmath=True)

                k2 = lane_idx
                while k2 < Int32(self.K):
                    val = x_row[k2].to(Float32) + res_row[k2].to(Float32)
                    out_row[k2] = val.to(self.residual_out_dtype)
                    y_row[k2] = (val * rstd * norm_row[k2].to(Float32)).to(self.y_dtype)
                    k2 = k2 + Int32(32)


class RmsGateUpSiluSm120Op(_DecodeMatvecSm120Base):
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
        self.reduction_tile_K = getattr(self, "reduction_tile_K", min(SM120_DECODE_REDUCTION_DIM_PER_WARP, self.K))
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
            SM120_DECODE_REDUCTION_DIM_PER_WARP, tensors["x"].shape[-1]
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
                    x_row = cute.make_tensor(x.iterator + x_base + k_base, cute.make_layout(SM120_DECODE_REDUCTION_DIM_PER_WARP))
                    elem = lane_idx
                    while elem < Int32(SM120_DECODE_REDUCTION_DIM_PER_WARP):
                        k = k_base + elem
                        if k < Int32(self.K):
                            xv = x_row[elem].to(Float32)
                            sum_sq = sum_sq + xv * xv
                        elem = elem + Int32(32)
                    k_base = k_base + Int32(SM120_DECODE_REDUCTION_DIM_PER_WARP)
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
                            x_row = cute.make_tensor(x.iterator + x_base + k2_base, cute.make_layout(SM120_DECODE_REDUCTION_DIM_PER_WARP))
                            norm_row = cute.make_tensor(norm_weight.iterator + k2_base, cute.make_layout(SM120_DECODE_REDUCTION_DIM_PER_WARP))
                            gate_row = cute.make_tensor(gate_weight.iterator + gate_base + k2_base, cute.make_layout(SM120_DECODE_REDUCTION_DIM_PER_WARP))
                            up_row = cute.make_tensor(up_weight.iterator + up_base + k2_base, cute.make_layout(SM120_DECODE_REDUCTION_DIM_PER_WARP))
                            elem2 = lane_idx
                            while elem2 < Int32(SM120_DECODE_REDUCTION_DIM_PER_WARP):
                                k2 = k2_base + elem2
                                if k2 < Int32(self.K):
                                    nv = x_row[elem2].to(Float32) * rstd * norm_row[elem2].to(Float32)
                                    gate_acc = gate_acc + nv * gate_row[elem2].to(Float32)
                                    up_acc = up_acc + nv * up_row[elem2].to(Float32)
                                elem2 = elem2 + Int32(32)
                            k2_base = k2_base + Int32(SM120_DECODE_REDUCTION_DIM_PER_WARP)
                        gate_total = cute.arch.warp_reduction(gate_acc, operator.add)
                        up_total = cute.arch.warp_reduction(up_acc, operator.add)
                        if lane_idx == Int32(0):
                            y_base = tile_B * Int32(self.y_stride_B) + row_idx * Int32(self.y_stride_S)
                            y_tile = cute.make_tensor(y.iterator + y_base + d_start, cute.make_layout(self.tile_size_D))
                            y_tile[local_d] = (_silu(gate_total) * up_total).to(self.y_dtype)


class RmsGateUpSiluNvfp4Sm120Op(_Nvfp4WeightMixin, _DecodeMatvecSm120Base):
    """Fused RMS + packed NVFP4 gate/up matvec + SiLU."""

    pipeline = SM120_OUTPUT_RANGE_PIPELINE
    reads = {
        "x": (None, ("B", "S", "K")),
        "norm_weight": (None, ("K",)),
        "gate_packed": (cutlass.Uint8, ("D", "K2")),
        "gate_scales": (cutlass.Float16, ("D", "G")),
        "up_packed": (cutlass.Uint8, ("D", "K2")),
        "up_scales": (cutlass.Float16, ("D", "G")),
    }
    writes = {"y": (None, ("B", "S", "D"))}
    tile = ("B", "S", "D")
    dynamic_dims = ("B",)

    def __init__(self, **config):
        super().__init__(**config)
        self.eps = getattr(self, "eps", 1e-5)

    @classmethod
    def schedule(
        cls,
        tile_sizes=None,
        page_size=DEFAULT_PAGE_SIZE,
        eps=1e-5,
        group_size=32,
        tile_range=None,
        **tensors,
    ):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", 16)
        tile_sizes.setdefault("D", 16)
        if tensors["gate_packed"].shape[1] * 2 != tensors["x"].shape[-1]:
            raise ValueError("gate_packed K2 must equal x.K / 2")
        if tensors["up_packed"].shape[1] * 2 != tensors["x"].shape[-1]:
            raise ValueError("up_packed K2 must equal x.K / 2")
        expected_g = tensors["x"].shape[-1] // group_size
        if tensors["gate_scales"].shape[1] != expected_g or tensors["up_scales"].shape[1] != expected_g:
            raise ValueError("gate/up scales G must equal x.K / group_size")
        op = cls._schedule_single(tile_sizes=tile_sizes, tile_range=tile_range, **tensors)
        op.static_dims["eps"] = eps
        return _finalize_nvfp4_matvec_schedule(
            op,
            page_size=page_size,
            group_size=group_size,
            k_dim=tensors["x"].shape[-1],
        )

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_D, x, norm_weight,
                gate_packed, gate_scales, up_packed, up_scales, y):
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.threads_per_row // 32
        row_start = tile_S * Int32(self.tile_size_S)
        rstd_smem = cute.make_tensor(
            cute.make_ptr(cutlass.Float32, page_ptr, cute.AddressSpace.smem),
            cute.make_layout(self.tile_size_S),
        )
        for local_row in range(warp_idx, self.tile_size_S, num_warps):
            row_idx = row_start + Int32(local_row)
            if row_idx < Int32(self.S):
                x_base = tile_B * Int32(self.x_stride_B) + row_idx * Int32(self.x_stride_S)
                x_row = cute.make_tensor(x.iterator + x_base, cute.make_layout(self.K))
                sum_sq = Float32(0.0)
                k = lane_idx
                while k < Int32(self.K):
                    xv = x_row[k].to(Float32)
                    sum_sq = sum_sq + xv * xv
                    k = k + Int32(32)
                total_sq = cute.arch.warp_reduction(sum_sq, operator.add)
                rstd = cute.math.rsqrt(total_sq * Float32(1.0 / self.K) + Float32(self.eps), fastmath=True)
                if lane_idx == Int32(0):
                    rstd_smem[local_row] = rstd

        named_barrier_sync(Int32(2), Int32(self.threads_per_row))

        d_start = tile_D * Int32(self.tile_size_D)
        for local_work in range(warp_idx, self.tile_size_S * self.tile_size_D, num_warps):
            local_row = local_work // self.tile_size_D
            local_d = local_work - local_row * self.tile_size_D
            row_idx = row_start + Int32(local_row)
            if row_idx < Int32(self.S):
                x_base = tile_B * Int32(self.x_stride_B) + row_idx * Int32(self.x_stride_S)
                x_row = cute.make_tensor(x.iterator + x_base, cute.make_layout(self.K))
                d = d_start + Int32(local_d)
                if d < Int32(self.D):
                    rstd = rstd_smem[local_row]
                    gate_packed_row = cute.make_tensor(
                        gate_packed.iterator + d * Int32(self.gate_packed_stride_D),
                        cute.make_layout(self.K2),
                    )
                    gate_scale_row = cute.make_tensor(
                        gate_scales.iterator + d * Int32(self.gate_scales_stride_D),
                        cute.make_layout(self.G),
                    )
                    up_packed_row = cute.make_tensor(
                        up_packed.iterator + d * Int32(self.up_packed_stride_D),
                        cute.make_layout(self.K2),
                    )
                    up_scale_row = cute.make_tensor(
                        up_scales.iterator + d * Int32(self.up_scales_stride_D),
                        cute.make_layout(self.G),
                    )
                    norm_row = cute.make_tensor(norm_weight.iterator, cute.make_layout(self.K))
                    gate_acc = Float32(0.0)
                    up_acc = Float32(0.0)
                    full_k = Int32((self.K // 8) * 8)
                    k2 = lane_idx * Int32(8)
                    while k2 < full_k:
                        v0 = x_row[k2].to(Float32) * rstd * norm_row[k2].to(Float32)
                        v1 = x_row[k2 + Int32(1)].to(Float32) * rstd * norm_row[k2 + Int32(1)].to(Float32)
                        v2 = x_row[k2 + Int32(2)].to(Float32) * rstd * norm_row[k2 + Int32(2)].to(Float32)
                        v3 = x_row[k2 + Int32(3)].to(Float32) * rstd * norm_row[k2 + Int32(3)].to(Float32)
                        v4 = x_row[k2 + Int32(4)].to(Float32) * rstd * norm_row[k2 + Int32(4)].to(Float32)
                        v5 = x_row[k2 + Int32(5)].to(Float32) * rstd * norm_row[k2 + Int32(5)].to(Float32)
                        v6 = x_row[k2 + Int32(6)].to(Float32) * rstd * norm_row[k2 + Int32(6)].to(Float32)
                        v7 = x_row[k2 + Int32(7)].to(Float32) * rstd * norm_row[k2 + Int32(7)].to(Float32)
                        gate_acc = gate_acc + self._dot8_nvfp4_values(
                            gate_packed_row, gate_scale_row, k2,
                            v0, v1, v2, v3, v4, v5, v6, v7,
                        )
                        up_acc = up_acc + self._dot8_nvfp4_values(
                            up_packed_row, up_scale_row, k2,
                            v0, v1, v2, v3, v4, v5, v6, v7,
                        )
                        k2 = k2 + Int32(256)
                    k2 = full_k + lane_idx
                    while k2 < Int32(self.K):
                        nv = x_row[k2].to(Float32) * rstd * norm_row[k2].to(Float32)
                        gate_acc = gate_acc + nv * self._nvfp4_weight_value(gate_packed_row, gate_scale_row, k2)
                        up_acc = up_acc + nv * self._nvfp4_weight_value(up_packed_row, up_scale_row, k2)
                        k2 = k2 + Int32(32)
                    gate_total = cute.arch.warp_reduction(gate_acc, operator.add)
                    up_total = cute.arch.warp_reduction(up_acc, operator.add)
                    if lane_idx == Int32(0):
                        y_base = tile_B * Int32(self.y_stride_B) + row_idx * Int32(self.y_stride_S)
                        y_tile = cute.make_tensor(y.iterator + y_base + d_start, cute.make_layout(self.tile_size_D))
                        y_tile[local_d] = (_silu(gate_total) * up_total).to(self.y_dtype)


class FinalRmsLmHeadSm120Op(RmsGateUpSiluSm120Op):
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
        self.reduction_tile_K = getattr(self, "reduction_tile_K", min(SM120_DECODE_REDUCTION_DIM_PER_WARP, self.K))
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
            SM120_DECODE_REDUCTION_DIM_PER_WARP, tensors["x"].shape[-1]
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
                    x_row = cute.make_tensor(x.iterator + x_base + k_base, cute.make_layout(SM120_DECODE_REDUCTION_DIM_PER_WARP))
                    elem = lane_idx
                    while elem < Int32(SM120_DECODE_REDUCTION_DIM_PER_WARP):
                        k = k_base + elem
                        if k < Int32(self.K):
                            xv = x_row[elem].to(Float32)
                            sum_sq = sum_sq + xv * xv
                        elem = elem + Int32(32)
                    k_base = k_base + Int32(SM120_DECODE_REDUCTION_DIM_PER_WARP)
                total_sq = cute.arch.warp_reduction(sum_sq, operator.add)
                rstd = cute.math.rsqrt(total_sq * Float32(1.0 / self.K) + Float32(self.eps), fastmath=True)

                for local_v in range(self.tile_size_V):
                    v = v_start + Int32(local_v)
                    if v < Int32(self.V):
                        acc = Float32(0.0)
                        w_base = v * Int32(self.weight_stride_V)
                        k2_base = Int32(0)
                        while k2_base < Int32(self.K):
                            x_row = cute.make_tensor(x.iterator + x_base + k2_base, cute.make_layout(SM120_DECODE_REDUCTION_DIM_PER_WARP))
                            norm_row = cute.make_tensor(norm_weight.iterator + k2_base, cute.make_layout(SM120_DECODE_REDUCTION_DIM_PER_WARP))
                            weight_row = cute.make_tensor(weight.iterator + w_base + k2_base, cute.make_layout(SM120_DECODE_REDUCTION_DIM_PER_WARP))
                            elem2 = lane_idx
                            while elem2 < Int32(SM120_DECODE_REDUCTION_DIM_PER_WARP):
                                k2 = k2_base + elem2
                                if k2 < Int32(self.K):
                                    nv = x_row[elem2].to(Float32) * rstd * norm_row[elem2].to(Float32)
                                    acc = acc + nv * weight_row[elem2].to(Float32)
                                elem2 = elem2 + Int32(32)
                            k2_base = k2_base + Int32(SM120_DECODE_REDUCTION_DIM_PER_WARP)
                        total = cute.arch.warp_reduction(acc, operator.add)
                        if lane_idx == Int32(0):
                            out_base = tile_B * Int32(self.logits_stride_B) + row_idx * Int32(self.logits_stride_S)
                            out_tile = cute.make_tensor(logits.iterator + out_base + v_start, cute.make_layout(self.tile_size_V))
                            out_tile[local_v] = total.to(self.logits_dtype)


class FinalRmsLmHeadNvfp4Sm120Op(_Nvfp4WeightMixin, _DecodeMatvecSm120Base):
    """Fused final RMS + packed NVFP4 LM-head matvec."""

    reads = {
        "x": (None, ("B", "S", "K")),
        "norm_weight": (None, ("K",)),
        "weight_packed": (cutlass.Uint8, ("V", "K2")),
        "weight_scales": (cutlass.Float16, ("V", "G")),
    }
    writes = {"logits": (None, ("B", "S", "V"))}
    tile = ("B", "S", "V")
    dynamic_dims = ("B",)

    def __init__(self, **config):
        super().__init__(**config)
        self.eps = getattr(self, "eps", 1e-5)
        assert self.tile_size_V == 16

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, eps=1e-5,
                 group_size=32, **tensors):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", 16)
        tile_sizes.setdefault("V", 16)
        if tensors["weight_packed"].shape[1] * 2 != tensors["x"].shape[-1]:
            raise ValueError("weight_packed K2 must equal x.K / 2")
        if tensors["weight_scales"].shape[1] != tensors["x"].shape[-1] // group_size:
            raise ValueError("weight_scales G must equal x.K / group_size")
        op = cls._schedule_single(tile_sizes=tile_sizes, **tensors)
        op.static_dims["page_size"] = page_size
        op.static_dims["eps"] = eps
        op.static_dims["group_size"] = group_size
        op.static_dims["reduction_tile_K"] = min(
            SM120_DECODE_REDUCTION_DIM_PER_WARP, tensors["x"].shape[-1]
        )
        return [op]

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_V, x, norm_weight,
                weight_packed, weight_scales, logits):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.threads_per_row // 32
        row_start = tile_S * Int32(self.tile_size_S)
        v_start = tile_V * Int32(self.tile_size_V)
        rstd_smem = cute.make_tensor(
            cute.make_ptr(cutlass.Float32, page_ptr, cute.AddressSpace.smem),
            cute.make_layout(self.tile_size_S),
        )
        norm_smem = cute.make_tensor(
            cute.make_ptr(
                cutlass.Float32,
                page_ptr + Int32(self.tile_size_S * 4),
                cute.AddressSpace.smem,
            ),
            cute.make_layout(Int32(self.tile_size_S * self.K)),
        )
        for local_row in range(warp_idx, self.tile_size_S, num_warps):
            row_idx = row_start + Int32(local_row)
            if row_idx < Int32(self.S):
                x_base = tile_B * Int32(self.x_stride_B) + row_idx * Int32(self.x_stride_S)
                x_row = cute.make_tensor(x.iterator + x_base, cute.make_layout(self.K))
                sum_sq = Float32(0.0)
                k = lane_idx
                while k < Int32(self.K):
                    xv = x_row[k].to(Float32)
                    sum_sq = sum_sq + xv * xv
                    k = k + Int32(32)
                total_sq = cute.arch.warp_reduction(sum_sq, operator.add)
                rstd = cute.math.rsqrt(total_sq * Float32(1.0 / self.K) + Float32(self.eps), fastmath=True)
                if lane_idx == Int32(0):
                    rstd_smem[local_row] = rstd

        named_barrier_sync(Int32(2), Int32(self.threads_per_row))

        norm_row = cute.make_tensor(norm_weight.iterator, cute.make_layout(self.K))
        elem = tidx
        while elem < Int32(self.tile_size_S * self.K):
            local_row = elem // Int32(self.K)
            k = elem - local_row * Int32(self.K)
            row_idx = row_start + local_row
            if row_idx < Int32(self.S):
                x_base = tile_B * Int32(self.x_stride_B) + row_idx * Int32(self.x_stride_S)
                x_row = cute.make_tensor(x.iterator + x_base, cute.make_layout(self.K))
                norm_smem[elem] = x_row[k].to(Float32) * rstd_smem[local_row] * norm_row[k].to(Float32)
            elem = elem + Int32(self.threads_per_row)

        named_barrier_sync(Int32(2), Int32(self.threads_per_row))

        for local_work in range(warp_idx, self.tile_size_S * self.tile_size_V, num_warps):
            local_row = local_work // self.tile_size_V
            local_v = local_work - local_row * self.tile_size_V
            row_idx = row_start + Int32(local_row)
            if row_idx < Int32(self.S):
                v = v_start + Int32(local_v)
                if v < Int32(self.V):
                    packed_row = cute.make_tensor(
                        weight_packed.iterator + v * Int32(self.weight_packed_stride_V),
                        cute.make_layout(self.K2),
                    )
                    scale_row = cute.make_tensor(
                        weight_scales.iterator + v * Int32(self.weight_scales_stride_V),
                        cute.make_layout(self.G),
                    )
                    normed_row = cute.make_tensor(
                        norm_smem.iterator + local_row * Int32(self.K),
                        cute.make_layout(self.K),
                    )
                    acc = Float32(0.0)
                    full_k = Int32((self.K // 8) * 8)
                    k2 = lane_idx * Int32(8)
                    while k2 < full_k:
                        v0 = normed_row[k2]
                        v1 = normed_row[k2 + Int32(1)]
                        v2 = normed_row[k2 + Int32(2)]
                        v3 = normed_row[k2 + Int32(3)]
                        v4 = normed_row[k2 + Int32(4)]
                        v5 = normed_row[k2 + Int32(5)]
                        v6 = normed_row[k2 + Int32(6)]
                        v7 = normed_row[k2 + Int32(7)]
                        acc = acc + self._dot8_nvfp4_values(
                            packed_row, scale_row, k2,
                            v0, v1, v2, v3, v4, v5, v6, v7,
                        )
                        k2 = k2 + Int32(256)
                    k2 = full_k + lane_idx
                    while k2 < Int32(self.K):
                        acc = acc + normed_row[k2] * self._nvfp4_weight_value(packed_row, scale_row, k2)
                        k2 = k2 + Int32(32)
                    total = cute.arch.warp_reduction(acc, operator.add)
                    if lane_idx == Int32(0):
                        out_base = tile_B * Int32(self.logits_stride_B) + row_idx * Int32(self.logits_stride_S)
                        out_tile = cute.make_tensor(logits.iterator + out_base + v_start, cute.make_layout(self.tile_size_V))
                        out_tile[local_v] = total.to(self.logits_dtype)


class FinalRmsTop1LmHeadNvfp4Sm120Op(_Nvfp4WeightMixin, _DecodeMatvecSm120Base):
    """Fused final RMS + packed NVFP4 LM head that writes only top-1."""

    pipeline = PipelineSpec(page_count=1)
    reads = {
        "x": (None, ("B", "S", "K")),
        "norm_weight": (None, ("K",)),
        "weight_packed": (cutlass.Uint8, ("V", "K2")),
        "weight_scales": (cutlass.Float16, ("V", "G")),
    }
    writes = {
        "top_values": (None, ("B", "S")),
        "top_indices": (cutlass.Int32, ("B", "S")),
    }
    tile = ("B", "S", "V")
    dynamic_dims = ("B",)

    def __init__(self, **config):
        super().__init__(**config)
        self.eps = getattr(self, "eps", 1e-5)
        assert self.tile_size_V == 16

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, eps=1e-5,
                 group_size=32, **tensors):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", 16)
        tile_sizes.setdefault("V", 16)
        if tensors["weight_packed"].shape[1] * 2 != tensors["x"].shape[-1]:
            raise ValueError("weight_packed K2 must equal x.K / 2")
        if tensors["weight_scales"].shape[1] != tensors["x"].shape[-1] // group_size:
            raise ValueError("weight_scales G must equal x.K / group_size")
        op = cls._schedule_single(tile_sizes=tile_sizes, **tensors)
        op.static_dims["page_size"] = page_size
        op.static_dims["eps"] = eps
        op.static_dims["group_size"] = group_size
        op.static_dims["reduction_tile_K"] = min(
            SM120_DECODE_REDUCTION_DIM_PER_WARP, tensors["x"].shape[-1]
        )
        return [op]

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_V, tile_3, x, norm_weight,
                weight_packed, weight_scales, top_values, top_indices):
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.threads_per_row // 32
        row_start = tile_S * Int32(self.tile_size_S)
        for local_row in range(warp_idx, self.tile_size_S, num_warps):
            row_idx = row_start + Int32(local_row)
            if row_idx < Int32(self.S):
                x_base = tile_B * Int32(self.x_stride_B) + row_idx * Int32(self.x_stride_S)
                x_row = cute.make_tensor(x.iterator + x_base, cute.make_layout(self.K))
                sum_sq = Float32(0.0)
                k = lane_idx
                while k < Int32(self.K):
                    xv = x_row[k].to(Float32)
                    sum_sq = sum_sq + xv * xv
                    k = k + Int32(32)
                total_sq = cute.arch.warp_reduction(sum_sq, operator.add)
                rstd = cute.math.rsqrt(total_sq * Float32(1.0 / self.K) + Float32(self.eps), fastmath=True)
                norm_row = cute.make_tensor(norm_weight.iterator, cute.make_layout(self.K))

                best_val = Float32(-3.4028234663852886e38)
                best_idx = Int32(0)
                block_v = tile_V
                block_end = tile_3
                if block_end <= block_v:
                    block_v = Int32(0)
                    block_end = Int32((self.V + self.tile_size_V - 1) // self.tile_size_V)
                while block_v < block_end:
                    v_start = block_v * Int32(self.tile_size_V)
                    for local_v in range(self.tile_size_V):
                        v = v_start + Int32(local_v)
                        if v < Int32(self.V):
                            packed_row = cute.make_tensor(
                                weight_packed.iterator + v * Int32(self.weight_packed_stride_V),
                                cute.make_layout(self.K2),
                            )
                            scale_row = cute.make_tensor(
                                weight_scales.iterator + v * Int32(self.weight_scales_stride_V),
                                cute.make_layout(self.G),
                            )
                            acc = Float32(0.0)
                            k2 = lane_idx
                            while k2 < Int32(self.K):
                                nv = x_row[k2].to(Float32) * rstd * norm_row[k2].to(Float32)
                                acc = acc + nv * self._nvfp4_weight_value(packed_row, scale_row, k2)
                                k2 = k2 + Int32(32)
                            total = cute.arch.warp_reduction(acc, operator.add)
                            if total > best_val:
                                best_val = total
                                best_idx = v
                    block_v = block_v + Int32(1)

                if lane_idx == Int32(0):
                    out_base = tile_B * Int32(self.top_values_stride_B) + row_idx
                    val_row = cute.make_tensor(top_values.iterator + out_base, cute.make_layout(1))
                    idx_row = cute.make_tensor(
                        top_indices.iterator + tile_B * Int32(self.top_indices_stride_B) + row_idx,
                        cute.make_layout(1),
                    )
                    val_row[Int32(0)] = best_val.to(self.top_values_dtype)
                    idx_row[Int32(0)] = best_idx


class FinalRmsTop1PartialLmHeadNvfp4Sm120Op(_Nvfp4WeightMixin, _DecodeMatvecSm120Base):
    """Per-partition final RMS + packed NVFP4 LM-head top-1 partial."""

    pipeline = PipelineSpec.range_capable(
        range_axis=2,
        range_end_axis=3,
    )
    reads = {
        "x": (None, ("B", "S", "K")),
        "norm_weight": (None, ("K",)),
        "weight_packed": (cutlass.Uint8, ("V", "K2")),
        "weight_scales": (cutlass.Float16, ("V", "G")),
    }
    writes = {
        "partial_values": (cutlass.Float32, ("B", "S", "P")),
        "partial_indices": (cutlass.Int32, ("B", "S", "P")),
    }
    tile = ("B", "S", "P")
    dynamic_dims = ("B",)

    def __init__(self, **config):
        super().__init__(**config)
        self.eps = getattr(self, "eps", 1e-5)
        self.partitions = getattr(self, "partitions", self.P)

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, eps=1e-5,
                 group_size=32, tile_range=None, **tensors):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", 1)
        tile_sizes.setdefault("P", 1)
        if tensors["weight_packed"].shape[1] * 2 != tensors["x"].shape[-1]:
            raise ValueError("weight_packed K2 must equal x.K / 2")
        if tensors["weight_scales"].shape[1] != tensors["x"].shape[-1] // group_size:
            raise ValueError("weight_scales G must equal x.K / group_size")
        if tensors["partial_values"].shape != tensors["partial_indices"].shape:
            raise ValueError("partial_values and partial_indices must have the same shape")
        op = cls._schedule_single(tile_sizes=tile_sizes, tile_range=tile_range, **tensors)
        op.static_dims["page_size"] = page_size
        op.static_dims["eps"] = eps
        op.static_dims["group_size"] = group_size
        op.static_dims["partitions"] = int(tensors["partial_values"].shape[-1])
        op.static_dims["reduction_tile_K"] = min(
            SM120_DECODE_REDUCTION_DIM_PER_WARP, tensors["x"].shape[-1]
        )
        return [op]

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_P, x, norm_weight,
                weight_packed, weight_scales, partial_values, partial_indices):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.threads_per_row // 32
        row_idx = tile_S * Int32(self.tile_size_S)
        partition = tile_P

        value_smem = cute.make_tensor(
            cute.make_ptr(cutlass.Float32, page_ptr, cute.AddressSpace.smem),
            cute.make_layout(self.threads_per_row),
        )
        index_smem = cute.make_tensor(
            cute.make_ptr(cutlass.Int32, page_ptr + Int32(self.threads_per_row * 4), cute.AddressSpace.smem),
            cute.make_layout(self.threads_per_row),
        )
        norm_smem = cute.make_tensor(
            cute.make_ptr(
                self.x_dtype,
                page_ptr + Int32(self.threads_per_row * 8),
                cute.AddressSpace.smem,
            ),
            cute.make_layout(self.K),
        )

        if row_idx < Int32(self.S) and partition < Int32(self.partitions):
            x_base = tile_B * Int32(self.x_stride_B) + row_idx * Int32(self.x_stride_S)
            x_row = cute.make_tensor(x.iterator + x_base, cute.make_layout(self.K))
            sum_sq = Float32(0.0)
            k = lane_idx
            while k < Int32(self.K):
                xv = x_row[k].to(Float32)
                sum_sq = sum_sq + xv * xv
                k = k + Int32(32)
            total_sq = cute.arch.warp_reduction(sum_sq, operator.add)
            rstd = cute.math.rsqrt(total_sq * Float32(1.0 / self.K) + Float32(self.eps), fastmath=True)
            norm_row = cute.make_tensor(norm_weight.iterator, cute.make_layout(self.K))
            nk = tidx
            while nk < Int32(self.K):
                norm_smem[nk] = (x_row[nk].to(Float32) * rstd * norm_row[nk].to(Float32)).to(self.x_dtype)
                nk = nk + Int32(self.threads_per_row)
        named_barrier_sync(Int32(2), Int32(self.threads_per_row))

        if row_idx < Int32(self.S) and partition < Int32(self.partitions):
            vocab_start = (Int32(self.V) * partition) // Int32(self.partitions)
            vocab_end = (Int32(self.V) * (partition + Int32(1))) // Int32(self.partitions)
            best_val = Float32(-3.4028234663852886e38)
            best_idx = Int32(-1)
            v = vocab_start + warp_idx
            while v < vocab_end:
                packed_row = cute.make_tensor(
                    weight_packed.iterator + v * Int32(self.weight_packed_stride_V),
                    cute.make_layout(self.K2),
                )
                scale_row = cute.make_tensor(
                    weight_scales.iterator + v * Int32(self.weight_scales_stride_V),
                    cute.make_layout(self.G),
                )
                acc = Float32(0.0)
                full_k = Int32((self.K // 8) * 8)
                k2 = lane_idx * Int32(8)
                while k2 < full_k:
                    v0 = norm_smem[k2].to(Float32)
                    v1 = norm_smem[k2 + Int32(1)].to(Float32)
                    v2 = norm_smem[k2 + Int32(2)].to(Float32)
                    v3 = norm_smem[k2 + Int32(3)].to(Float32)
                    v4 = norm_smem[k2 + Int32(4)].to(Float32)
                    v5 = norm_smem[k2 + Int32(5)].to(Float32)
                    v6 = norm_smem[k2 + Int32(6)].to(Float32)
                    v7 = norm_smem[k2 + Int32(7)].to(Float32)
                    acc = acc + self._dot8_nvfp4_values(
                        packed_row, scale_row, k2,
                        v0, v1, v2, v3, v4, v5, v6, v7,
                    )
                    k2 = k2 + Int32(256)
                k2 = full_k + lane_idx
                while k2 < Int32(self.K):
                    acc = acc + norm_smem[k2].to(Float32) * self._nvfp4_weight_value(packed_row, scale_row, k2)
                    k2 = k2 + Int32(32)
                total = cute.arch.warp_reduction(acc, operator.add)
                if total > best_val:
                    best_val = total
                    best_idx = v
                v = v + num_warps

            if lane_idx == Int32(0):
                value_smem[warp_idx] = best_val
                index_smem[warp_idx] = best_idx
        named_barrier_sync(Int32(2), Int32(self.threads_per_row))

        if row_idx < Int32(self.S) and partition < Int32(self.partitions) and cute.arch.thread_idx()[0] == Int32(0):
            part_val = Float32(-3.4028234663852886e38)
            part_idx = Int32(-1)
            wi = Int32(0)
            while wi < num_warps:
                other_val = value_smem[wi]
                other_idx = index_smem[wi]
                if other_val > part_val:
                    part_val = other_val
                    part_idx = other_idx
                wi = wi + Int32(1)
            out_base = (
                tile_B * Int32(self.partial_values_stride_B)
                + row_idx * Int32(self.partial_values_stride_S)
            )
            val_row = cute.make_tensor(
                partial_values.iterator + out_base,
                cute.make_layout(self.P),
            )
            idx_row = cute.make_tensor(
                partial_indices.iterator
                + tile_B * Int32(self.partial_indices_stride_B)
                + row_idx * Int32(self.partial_indices_stride_S),
                cute.make_layout(self.P),
            )
            val_row[partition] = part_val
            idx_row[partition] = part_idx


class FinalAddRmsTop1PartialLmHeadNvfp4Sm120Op(FinalRmsTop1PartialLmHeadNvfp4Sm120Op):
    """Per-partition final residual add + RMS + packed NVFP4 LM-head top-1."""

    reads = {
        "x": (None, ("B", "S", "K")),
        "residual_in": (None, ("B", "S", "K")),
        "norm_weight": (None, ("K",)),
        "weight_packed": (cutlass.Uint8, ("V", "K2")),
        "weight_scales": (cutlass.Float16, ("V", "G")),
    }

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_P, x, residual_in, norm_weight,
                weight_packed, weight_scales, partial_values, partial_indices):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.threads_per_row // 32
        row_idx = tile_S * Int32(self.tile_size_S)
        partition = tile_P

        value_smem = cute.make_tensor(
            cute.make_ptr(cutlass.Float32, page_ptr, cute.AddressSpace.smem),
            cute.make_layout(self.threads_per_row),
        )
        index_smem = cute.make_tensor(
            cute.make_ptr(cutlass.Int32, page_ptr + Int32(self.threads_per_row * 4), cute.AddressSpace.smem),
            cute.make_layout(self.threads_per_row),
        )
        norm_smem = cute.make_tensor(
            cute.make_ptr(
                self.x_dtype,
                page_ptr + Int32(self.threads_per_row * 8),
                cute.AddressSpace.smem,
            ),
            cute.make_layout(self.K),
        )

        if row_idx < Int32(self.S) and partition < Int32(self.partitions):
            x_base = tile_B * Int32(self.x_stride_B) + row_idx * Int32(self.x_stride_S)
            residual_base = (
                tile_B * Int32(self.residual_in_stride_B)
                + row_idx * Int32(self.residual_in_stride_S)
            )
            x_row = cute.make_tensor(x.iterator + x_base, cute.make_layout(self.K))
            residual_row = cute.make_tensor(residual_in.iterator + residual_base, cute.make_layout(self.K))
            sum_sq = Float32(0.0)
            k = lane_idx
            while k < Int32(self.K):
                xv = x_row[k].to(Float32) + residual_row[k].to(Float32)
                sum_sq = sum_sq + xv * xv
                k = k + Int32(32)
            total_sq = cute.arch.warp_reduction(sum_sq, operator.add)
            rstd = cute.math.rsqrt(total_sq * Float32(1.0 / self.K) + Float32(self.eps), fastmath=True)
            norm_row = cute.make_tensor(norm_weight.iterator, cute.make_layout(self.K))
            nk = tidx
            while nk < Int32(self.K):
                xv = x_row[nk].to(Float32) + residual_row[nk].to(Float32)
                norm_smem[nk] = (xv * rstd * norm_row[nk].to(Float32)).to(self.x_dtype)
                nk = nk + Int32(self.threads_per_row)
        named_barrier_sync(Int32(2), Int32(self.threads_per_row))

        if row_idx < Int32(self.S) and partition < Int32(self.partitions):
            vocab_start = (Int32(self.V) * partition) // Int32(self.partitions)
            vocab_end = (Int32(self.V) * (partition + Int32(1))) // Int32(self.partitions)
            best_val = Float32(-3.4028234663852886e38)
            best_idx = Int32(-1)
            v = vocab_start + warp_idx
            while v < vocab_end:
                packed_row = cute.make_tensor(
                    weight_packed.iterator + v * Int32(self.weight_packed_stride_V),
                    cute.make_layout(self.K2),
                )
                scale_row = cute.make_tensor(
                    weight_scales.iterator + v * Int32(self.weight_scales_stride_V),
                    cute.make_layout(self.G),
                )
                acc = Float32(0.0)
                full_k = Int32((self.K // 8) * 8)
                k2 = lane_idx * Int32(8)
                while k2 < full_k:
                    v0 = norm_smem[k2].to(Float32)
                    v1 = norm_smem[k2 + Int32(1)].to(Float32)
                    v2 = norm_smem[k2 + Int32(2)].to(Float32)
                    v3 = norm_smem[k2 + Int32(3)].to(Float32)
                    v4 = norm_smem[k2 + Int32(4)].to(Float32)
                    v5 = norm_smem[k2 + Int32(5)].to(Float32)
                    v6 = norm_smem[k2 + Int32(6)].to(Float32)
                    v7 = norm_smem[k2 + Int32(7)].to(Float32)
                    acc = acc + self._dot8_nvfp4_values(
                        packed_row, scale_row, k2,
                        v0, v1, v2, v3, v4, v5, v6, v7,
                    )
                    k2 = k2 + Int32(256)
                k2 = full_k + lane_idx
                while k2 < Int32(self.K):
                    acc = acc + norm_smem[k2].to(Float32) * self._nvfp4_weight_value(packed_row, scale_row, k2)
                    k2 = k2 + Int32(32)
                total = cute.arch.warp_reduction(acc, operator.add)
                if total > best_val:
                    best_val = total
                    best_idx = v
                v = v + num_warps

            if lane_idx == Int32(0):
                value_smem[warp_idx] = best_val
                index_smem[warp_idx] = best_idx
        named_barrier_sync(Int32(2), Int32(self.threads_per_row))

        if row_idx < Int32(self.S) and partition < Int32(self.partitions) and cute.arch.thread_idx()[0] == Int32(0):
            part_val = Float32(-3.4028234663852886e38)
            part_idx = Int32(-1)
            wi = Int32(0)
            while wi < num_warps:
                other_val = value_smem[wi]
                other_idx = index_smem[wi]
                if other_val > part_val:
                    part_val = other_val
                    part_idx = other_idx
                wi = wi + Int32(1)
            out_base = (
                tile_B * Int32(self.partial_values_stride_B)
                + row_idx * Int32(self.partial_values_stride_S)
            )
            val_row = cute.make_tensor(
                partial_values.iterator + out_base,
                cute.make_layout(self.P),
            )
            idx_row = cute.make_tensor(
                partial_indices.iterator
                + tile_B * Int32(self.partial_indices_stride_B)
                + row_idx * Int32(self.partial_indices_stride_S),
                cute.make_layout(self.P),
            )
            val_row[partition] = part_val
            idx_row[partition] = part_idx


class ReduceTop1PartialsSm120Op(_DecodeMatvecSm120Base):
    """Reduce per-partition top-1 values to one token result."""

    reads = {
        "partial_values": (cutlass.Float32, ("B", "S", "P")),
        "partial_indices": (cutlass.Int32, ("B", "S", "P")),
    }
    writes = {
        "top_values": (cutlass.Float32, ("B", "S")),
        "top_indices": (cutlass.Int32, ("B", "S")),
    }
    tile = ("B", "S")
    dynamic_dims = ("B",)

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, **tensors):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", 1)
        op = cls._schedule_single(tile_sizes=tile_sizes, **tensors)
        op.static_dims["page_size"] = page_size
        return [op]

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, partial_values, partial_indices, top_values, top_indices):
        tidx = cute.arch.thread_idx()[0]
        row_idx = tile_S * Int32(self.tile_size_S)

        if row_idx < Int32(self.S) and tidx == Int32(0):
            val_base = tile_B * Int32(self.partial_values_stride_B) + row_idx * Int32(self.partial_values_stride_S)
            idx_base = tile_B * Int32(self.partial_indices_stride_B) + row_idx * Int32(self.partial_indices_stride_S)
            val_row = cute.make_tensor(partial_values.iterator + val_base, cute.make_layout(self.P))
            idx_row = cute.make_tensor(partial_indices.iterator + idx_base, cute.make_layout(self.P))
            final_val = Float32(-3.4028234663852886e38)
            final_idx = Int32(-1)
            p = Int32(0)
            while p < Int32(self.P):
                val = val_row[p]
                idx = idx_row[p]
                if val > final_val:
                    final_val = val
                    final_idx = idx
                p = p + Int32(1)
            out_base = tile_B * Int32(self.top_values_stride_B) + row_idx
            val_out = cute.make_tensor(top_values.iterator + out_base, cute.make_layout(1))
            idx_out = cute.make_tensor(
                top_indices.iterator + tile_B * Int32(self.top_indices_stride_B) + row_idx,
                cute.make_layout(1),
            )
            val_out[Int32(0)] = final_val
            idx_out[Int32(0)] = final_idx


@dataclass
class DecodeLayerSchedule:
    """Scheduled ops and view keep-alives for one SM120 Llama decoder layer."""

    ops: list
    attention_config: object
    keep_alive: list


def schedule_decode_layer_sm120(
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
    hidden_size=SM120_DECODE_HIDDEN_DEFAULT,
    intermediate_size=SM120_DECODE_INTERMEDIATE_DEFAULT,
    num_q_heads=SM120_DECODE_NUM_Q_HEADS_DEFAULT,
    num_kv_heads=SM120_DECODE_NUM_KV_HEADS_DEFAULT,
    head_dim=SM120_DECODE_HEAD_DIM_DEFAULT,
    kv_group_size=SM120_DECODE_KV_GROUP_SIZE_DEFAULT,
    matvec_block=SM120_DECODE_MATVEC_BLOCK,
):
    """Schedule one decode layer using instruction-shaped SM120 ops.

    This is the public convenience layer for the low-level classes above.  It
    follows the fused decode instruction structure:
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
    ops += RmsQMatvecRopeSm120Op.schedule(
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
    ops += RmsKMatvecRopeCacheSm120Op.schedule(
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
    ops += RmsVMatvecCacheSm120Op.schedule(
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

    ops += MatvecResidualSm120Op.schedule(
        a=attn_out_buf,
        weight=weights[f"{pfx}.W_o"],
        residual_in=residual_out,
        residual_out=residual_out,
        tile_sizes={"S": seq_len, "O": matvec_block},
        page_size=page_size,
    )
    ops += RmsGateUpSiluSm120Op.schedule(
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
            ops += MatvecSm120Op.schedule(
                a=a_block,
                weight=w_block,
                y=x_out,
                tile_sizes={"S": seq_len, "O": matvec_block},
                page_size=page_size,
            )
        else:
            ops += MatvecResidualSm120Op.schedule(
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


def schedule_decode_layer_nvfp4_sm120(
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
    group_size=32,
    fa_num_splits=0,
    hidden_size=SM120_DECODE_HIDDEN_DEFAULT,
    intermediate_size=SM120_DECODE_INTERMEDIATE_DEFAULT,
    num_q_heads=SM120_DECODE_NUM_Q_HEADS_DEFAULT,
    num_kv_heads=SM120_DECODE_NUM_KV_HEADS_DEFAULT,
    head_dim=SM120_DECODE_HEAD_DIM_DEFAULT,
    kv_group_size=SM120_DECODE_KV_GROUP_SIZE_DEFAULT,
    matvec_block=SM120_DECODE_MATVEC_BLOCK,
):
    """Schedule one decode layer using packed NVFP4 projection weights."""
    from machete.kernels.attention import FlashAttentionOp
    from machete.kernels.attention.flash_decoding import flash_decoding_schedule

    pfx = f"layer.{layer_idx}"
    cos = weights["cos"][cache_pos : cache_pos + seq_len]
    sin = weights["sin"][cache_pos : cache_pos + seq_len]

    q_4d = q_buf.view(batch, seq_len, num_q_heads, head_dim)
    k_window = k_cache[:, : cache_pos + seq_len]
    v_window = v_cache[:, : cache_pos + seq_len]
    o_4d = attn_out_buf.view(batch, seq_len, num_q_heads, head_dim)

    def qparts(name):
        qweight = weights[f"{pfx}.{name}_nvfp4"]
        return qweight.packed, qweight.scales

    q_packed, q_scales = qparts("W_q")
    k_packed, k_scales = qparts("W_k")
    v_packed, v_scales = qparts("W_v")
    o_packed, o_scales = qparts("W_o")
    gate_packed, gate_scales = qparts("W_gate")
    up_packed, up_scales = qparts("W_up")
    down_packed, down_scales = qparts("W_down")

    ops = []
    ops += RmsQMatvecRopeNvfp4Sm120Op.schedule(
        x=x_in,
        residual_in=residual_in,
        norm_weight=weights[f"{pfx}.attn_norm"],
        weight_packed=q_packed,
        weight_scales=q_scales,
        cos=cos,
        sin=sin,
        residual_out=residual_out,
        q=q_buf,
        tile_sizes={"S": seq_len, "O": matvec_block},
        page_size=page_size,
        eps=eps,
        group_size=group_size,
        head_dim=head_dim,
    )
    ops += RmsKMatvecRopeCacheNvfp4Sm120Op.schedule(
        x=x_in,
        residual_in=residual_in,
        norm_weight=weights[f"{pfx}.attn_norm"],
        weight_packed=k_packed,
        weight_scales=k_scales,
        cos=cos,
        sin=sin,
        dst_cache=k_window,
        cache_pos=cache_pos,
        tile_sizes={"S": seq_len, "O": matvec_block},
        page_size=page_size,
        eps=eps,
        group_size=group_size,
        head_dim=head_dim,
    )
    ops += RmsVMatvecCacheNvfp4Sm120Op.schedule(
        x=x_in,
        residual_in=residual_in,
        norm_weight=weights[f"{pfx}.attn_norm"],
        weight_packed=v_packed,
        weight_scales=v_scales,
        cos=cos,
        sin=sin,
        dst_cache=v_window,
        cache_pos=cache_pos,
        tile_sizes={"S": seq_len, "O": matvec_block},
        page_size=page_size,
        eps=eps,
        group_size=group_size,
        head_dim=head_dim,
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

    ops += MatvecResidualNvfp4Sm120Op.schedule(
        a=attn_out_buf,
        weight_packed=o_packed,
        weight_scales=o_scales,
        residual_in=residual_out,
        residual_out=residual_out,
        tile_sizes={"S": seq_len, "O": matvec_block},
        page_size=page_size,
        group_size=group_size,
    )
    ops += RmsGateUpSiluNvfp4Sm120Op.schedule(
        x=residual_out,
        norm_weight=weights[f"{pfx}.mlp_norm"],
        gate_packed=gate_packed,
        gate_scales=gate_scales,
        up_packed=up_packed,
        up_scales=up_scales,
        y=mlp_h_buf,
        tile_sizes={"S": seq_len, "D": matvec_block},
        page_size=page_size,
        eps=eps,
        group_size=group_size,
    )
    ops += MatvecNvfp4Sm120Op.schedule(
        a=mlp_h_buf,
        weight_packed=down_packed,
        weight_scales=down_scales,
        y=x_out,
        tile_sizes={"S": seq_len, "O": matvec_block},
        page_size=page_size,
        group_size=group_size,
    )

    keep = [
        cos, sin, q_4d, k_window, v_window, o_4d,
        q_packed, q_scales, k_packed, k_scales, v_packed, v_scales,
        o_packed, o_scales, gate_packed, gate_scales, up_packed, up_scales,
        down_packed, down_scales,
    ]
    return DecodeLayerSchedule(ops=ops, attention_config=fa_config, keep_alive=keep)


def schedule_final_sm120(
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
    ops = ResidualAddSm120Op.schedule(
        x=x,
        residual_in=residual_in,
        residual_out=residual_out,
        tile_sizes={"S": seq_len, "K": SM120_DECODE_REDUCTION_DIM_PER_WARP},
        page_size=page_size,
    )
    if lm_head is not None:
        if logits is None:
            raise ValueError("logits must be provided when lm_head is scheduled")
        ops += FinalRmsLmHeadSm120Op.schedule(
            x=residual_out,
            norm_weight=final_norm,
            weight=lm_head,
            logits=logits,
            tile_sizes={"S": seq_len, "V": SM120_DECODE_MATVEC_BLOCK},
            page_size=page_size,
            eps=eps,
        )
    return ops


def schedule_final_nvfp4_sm120(
    *,
    x,
    residual_in,
    residual_out,
    final_norm,
    lm_head_nvfp4=None,
    logits=None,
    top_values=None,
    top_indices=None,
    top_partial_values=None,
    top_partial_indices=None,
    seq_len,
    page_size=DEFAULT_PAGE_SIZE,
    eps=1e-5,
    group_size=32,
    final_head_range_block=1,
):
    """Schedule final residual plus optional packed NVFP4 LM-head."""
    ops = []
    if lm_head_nvfp4 is not None:
        if top_values is not None or top_indices is not None:
            if top_values is None or top_indices is None:
                raise ValueError("top_values and top_indices must be provided together")
            if top_partial_values is not None or top_partial_indices is not None:
                if top_partial_values is None or top_partial_indices is None:
                    raise ValueError("top_partial_values and top_partial_indices must be provided together")
                ops += FinalAddRmsTop1PartialLmHeadNvfp4Sm120Op.schedule(
                    x=x,
                    residual_in=residual_in,
                    norm_weight=final_norm,
                    weight_packed=lm_head_nvfp4.packed,
                    weight_scales=lm_head_nvfp4.scales,
                    partial_values=top_partial_values,
                    partial_indices=top_partial_indices,
                    tile_sizes={"S": seq_len, "P": 1},
                    tile_range=(
                        TileRange.coalesced("P", block_size=final_head_range_block)
                        if final_head_range_block > 1
                        else None
                    ),
                    page_size=page_size,
                    eps=eps,
                    group_size=group_size,
                )
                ops += ReduceTop1PartialsSm120Op.schedule(
                    partial_values=top_partial_values,
                    partial_indices=top_partial_indices,
                    top_values=top_values,
                    top_indices=top_indices,
                    tile_sizes={"S": seq_len},
                    page_size=page_size,
                )
            else:
                ops += ResidualAddSm120Op.schedule(
                    x=x,
                    residual_in=residual_in,
                    residual_out=residual_out,
                    tile_sizes={"S": seq_len, "K": SM120_DECODE_REDUCTION_DIM_PER_WARP},
                    page_size=page_size,
                )
                ops += FinalRmsTop1LmHeadNvfp4Sm120Op.schedule(
                    x=residual_out,
                    norm_weight=final_norm,
                    weight_packed=lm_head_nvfp4.packed,
                    weight_scales=lm_head_nvfp4.scales,
                    top_values=top_values,
                    top_indices=top_indices,
                    tile_sizes={"S": seq_len, "V": SM120_DECODE_MATVEC_BLOCK},
                    page_size=page_size,
                    eps=eps,
                    group_size=group_size,
                )
        else:
            if logits is None:
                raise ValueError(
                    "logits or top_values/top_indices must be provided when lm_head_nvfp4 is scheduled"
                )
            ops += ResidualAddSm120Op.schedule(
                x=x,
                residual_in=residual_in,
                residual_out=residual_out,
                tile_sizes={"S": seq_len, "K": SM120_DECODE_REDUCTION_DIM_PER_WARP},
                page_size=page_size,
            )
            ops += FinalRmsLmHeadNvfp4Sm120Op.schedule(
                x=residual_out,
                norm_weight=final_norm,
                weight_packed=lm_head_nvfp4.packed,
                weight_scales=lm_head_nvfp4.scales,
                logits=logits,
                tile_sizes={"S": seq_len, "V": SM120_DECODE_MATVEC_BLOCK},
                page_size=page_size,
                eps=eps,
                group_size=group_size,
            )
    else:
        ops += ResidualAddSm120Op.schedule(
            x=x,
            residual_in=residual_in,
            residual_out=residual_out,
            tile_sizes={"S": seq_len, "K": SM120_DECODE_REDUCTION_DIM_PER_WARP},
            page_size=page_size,
        )
    return ops


__all__ = [
    "FinalRmsLmHeadSm120Op",
    "FinalRmsLmHeadNvfp4Sm120Op",
    "FinalAddRmsTop1PartialLmHeadNvfp4Sm120Op",
    "FinalRmsTop1PartialLmHeadNvfp4Sm120Op",
    "FinalRmsTop1LmHeadNvfp4Sm120Op",
    "MatvecPairNvfp4Sm120Op",
    "MatvecQuadNvfp4Sm120Op",
    "MatvecResidualSm120Op",
    "MatvecResidualNvfp4Sm120Op",
    "MatvecSm120Op",
    "MatvecNvfp4Sm120Op",
    "ResidualAddSm120Op",
    "RmsAddNormSm120Op",
    "RmsGateUpSiluSm120Op",
    "RmsGateUpSiluNvfp4Sm120Op",
    "RmsMatvecNvfp4Sm120Op",
    "RmsReadMatvecNvfp4Sm120Op",
    "RmsKMatvecRopeCacheSm120Op",
    "RmsKMatvecRopeCacheNvfp4Sm120Op",
    "RmsQMatvecRopeSm120Op",
    "RmsQMatvecRopeNvfp4Sm120Op",
    "RmsVMatvecCacheSm120Op",
    "RmsVMatvecCacheNvfp4Sm120Op",
    "ReduceTop1PartialsSm120Op",
    "DecodeLayerSchedule",
    "schedule_decode_layer_nvfp4_sm120",
    "schedule_decode_layer_sm120",
    "schedule_final_nvfp4_sm120",
    "schedule_final_sm120",
]
