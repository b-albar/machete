# Copyright (c) 2025, Machete Authors
"""Native CuTe DSL MXFP4 SIMT decode schedules for Qwen3.5-0.8B."""

from __future__ import annotations

from dataclasses import dataclass
import operator

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass._mlir.dialects import llvm
import torch

from machete.kernels.decode_matvec import (
    DecodeLayerScheduleSm120,
    FinalAddRmsTop1AtomicLmHeadNvfp4Sm120Op,
    FinalAddRmsTop1PartialLmHeadNvfp4Sm120Op,
    FinalRmsLmHeadNvfp4Sm120Op,
    FinalRmsTop1LmHeadNvfp4Sm120Op,
    FinalTop1AtomicInitSm120Op,
    MatvecPairSm120Op,
    MatvecResidualNvfp4Sm120Op,
    MatvecNvfp4Sm120Op,
    MatvecPairNvfp4Sm120Op,
    MatvecQuadNvfp4Sm120Op,
    RmsAddNormSm120Op,
    RmsCopyNormSm120Op,
    RmsGateUpSiluNvfp4Sm120Op,
    ReduceTop1PartialsSm120Op,
    ResidualAddSm120Op,
)
from machete.megakernel.interpreter import named_barrier_sync
from machete.megakernel.ops import DEFAULT_PAGE_SIZE, Op, PipelineSpec
from machete.quantization.mxfp4 import quantize_mxfp4_weight


QWEN3_5_MXFP4_NUM_LAYERS = 24
QWEN3_5_MXFP4_HIDDEN = 1024
QWEN3_5_MXFP4_INTERMEDIATE = 3584
QWEN3_5_MXFP4_VOCAB = 248320
QWEN3_5_MXFP4_NUM_Q_HEADS = 8
QWEN3_5_MXFP4_NUM_KV_HEADS = 2
QWEN3_5_MXFP4_HEAD_DIM = 256
QWEN3_5_MXFP4_ROTARY_D2 = 32
QWEN3_5_MXFP4_Q_DIM = QWEN3_5_MXFP4_NUM_Q_HEADS * QWEN3_5_MXFP4_HEAD_DIM
QWEN3_5_MXFP4_Q_RAW_DIM = 2 * QWEN3_5_MXFP4_Q_DIM
QWEN3_5_MXFP4_KV_DIM = QWEN3_5_MXFP4_NUM_KV_HEADS * QWEN3_5_MXFP4_HEAD_DIM
QWEN3_5_MXFP4_KV_GROUP_SIZE = QWEN3_5_MXFP4_NUM_Q_HEADS // QWEN3_5_MXFP4_NUM_KV_HEADS
QWEN3_5_MXFP4_EPS = 1e-6
QWEN3_5_MXFP4_GROUP_SIZE = 32
QWEN3_5_MXFP4_DN_NUM_HEADS = 16
QWEN3_5_MXFP4_DN_KEY_DIM = 128
QWEN3_5_MXFP4_DN_VALUE_DIM = 128
QWEN3_5_MXFP4_DN_QK_SIZE = QWEN3_5_MXFP4_DN_NUM_HEADS * QWEN3_5_MXFP4_DN_KEY_DIM
QWEN3_5_MXFP4_DN_V_SIZE = QWEN3_5_MXFP4_DN_NUM_HEADS * QWEN3_5_MXFP4_DN_VALUE_DIM
QWEN3_5_MXFP4_DN_CONV_CHANNELS = 2 * QWEN3_5_MXFP4_DN_QK_SIZE + QWEN3_5_MXFP4_DN_V_SIZE
QWEN3_5_MXFP4_DN_CONV_KERNEL = 4
QWEN3_5_MXFP4_MATVEC_BLOCK = 16
QWEN3_5_MXFP4_GATE_UP_BLOCK = 64
QWEN3_5_MXFP4_SIMT_MATVEC_BLOCK = 15
QWEN3_5_MXFP4_FP32_NEG_INF = -3.4028234663852886e38
QWEN3_5_MXFP4_ATTN_SCALE = 0.0625
QWEN3_5_LAYER_TYPES = ("linear_attention", "linear_attention", "linear_attention", "full_attention") * 6


@dataclass(frozen=True)
class Qwen3_5Fp4OpSet:
    matvec: type
    matvec_pair: type
    matvec_quad: type
    matvec_residual: type
    rms_gate_up_silu: type
    final_rms_lm_head: type
    final_rms_top1_lm_head: type
    final_add_rms_top1_partial_lm_head: type
    final_add_rms_top1_atomic_lm_head: type


@dataclass(frozen=True)
class MXFP4SimtWeight:
    packed: torch.Tensor
    scales: torch.Tensor
    group_size: int
    rows: int
    cols: int


def empty_mxfp4_simt_weight(rows: int, cols: int, group_size: int = 32, device: str = "cuda") -> MXFP4SimtWeight:
    return MXFP4SimtWeight(
        packed=torch.empty(rows, cols // 2, device=device, dtype=torch.uint8),
        scales=torch.empty(rows, cols // group_size, device=device, dtype=torch.uint8),
        group_size=group_size,
        rows=rows,
        cols=cols,
    )


def quantize_mxfp4_simt_weight(weight: torch.Tensor, group_size: int = 32) -> MXFP4SimtWeight:
    if group_size != 32:
        raise ValueError("MXFP4 SIMT decode currently requires group_size=32")
    q = quantize_mxfp4_weight(weight.contiguous())
    return MXFP4SimtWeight(q.packed, q.scales_e8m0, group_size=32, rows=q.rows, cols=q.cols)


def _e8m0_bits_to_f32(bits: Int32, *, loc=None, ip=None) -> Float32:
    from cutlass._mlir import ir

    r = llvm.inline_asm(
        ir.F32Type.get(),
        [Int32(bits).ir_value(loc=loc, ip=ip)],
        "mov.b32 $0, $1;",
        "=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Float32(r)


def _e8m0_byte_to_f32(byte: Int32) -> Float32:
    return _e8m0_bits_to_f32(byte << Int32(23))


@cute.jit
def _e8m0_decode_scale(self, scale_row, scale_idx):
    return _e8m0_byte_to_f32(scale_row[scale_idx].to(Int32))


def _mxfp4_reads(nvfp4_cls):
    reads = dict(nvfp4_cls.reads)
    for name, spec in list(reads.items()):
        if name.endswith("scales"):
            reads[name] = (cutlass.Uint8, spec[1])
    return reads


class MatvecMxfp4SimtSm120Op(MatvecNvfp4Sm120Op):
    reads = _mxfp4_reads(MatvecNvfp4Sm120Op)
    _decode_scale = _e8m0_decode_scale


class MatvecPairMxfp4SimtSm120Op(MatvecPairNvfp4Sm120Op):
    reads = _mxfp4_reads(MatvecPairNvfp4Sm120Op)
    _decode_scale = _e8m0_decode_scale


class MatvecQuadMxfp4SimtSm120Op(MatvecQuadNvfp4Sm120Op):
    reads = _mxfp4_reads(MatvecQuadNvfp4Sm120Op)
    _decode_scale = _e8m0_decode_scale


class MatvecResidualMxfp4SimtSm120Op(MatvecResidualNvfp4Sm120Op):
    reads = _mxfp4_reads(MatvecResidualNvfp4Sm120Op)
    _decode_scale = _e8m0_decode_scale


class RmsGateUpSiluMxfp4SimtSm120Op(RmsGateUpSiluNvfp4Sm120Op):
    reads = _mxfp4_reads(RmsGateUpSiluNvfp4Sm120Op)
    _decode_scale = _e8m0_decode_scale


class FinalRmsLmHeadMxfp4SimtSm120Op(FinalRmsLmHeadNvfp4Sm120Op):
    reads = _mxfp4_reads(FinalRmsLmHeadNvfp4Sm120Op)
    _decode_scale = _e8m0_decode_scale


class FinalRmsTop1LmHeadMxfp4SimtSm120Op(FinalRmsTop1LmHeadNvfp4Sm120Op):
    reads = _mxfp4_reads(FinalRmsTop1LmHeadNvfp4Sm120Op)
    _decode_scale = _e8m0_decode_scale


class FinalAddRmsTop1PartialLmHeadMxfp4SimtSm120Op(FinalAddRmsTop1PartialLmHeadNvfp4Sm120Op):
    reads = _mxfp4_reads(FinalAddRmsTop1PartialLmHeadNvfp4Sm120Op)
    _decode_scale = _e8m0_decode_scale


class FinalAddRmsTop1AtomicLmHeadMxfp4SimtSm120Op(FinalAddRmsTop1AtomicLmHeadNvfp4Sm120Op):
    reads = _mxfp4_reads(FinalAddRmsTop1AtomicLmHeadNvfp4Sm120Op)
    _decode_scale = _e8m0_decode_scale


QWEN3_5_MXFP4_SIMT_OPS = Qwen3_5Fp4OpSet(
    matvec=MatvecMxfp4SimtSm120Op,
    matvec_pair=MatvecPairMxfp4SimtSm120Op,
    matvec_quad=MatvecQuadMxfp4SimtSm120Op,
    matvec_residual=MatvecResidualMxfp4SimtSm120Op,
    rms_gate_up_silu=RmsGateUpSiluMxfp4SimtSm120Op,
    final_rms_lm_head=FinalRmsLmHeadMxfp4SimtSm120Op,
    final_rms_top1_lm_head=FinalRmsTop1LmHeadMxfp4SimtSm120Op,
    final_add_rms_top1_partial_lm_head=FinalAddRmsTop1PartialLmHeadMxfp4SimtSm120Op,
    final_add_rms_top1_atomic_lm_head=FinalAddRmsTop1AtomicLmHeadMxfp4SimtSm120Op,
)


@cute.jit
def _silu(x):
    neg = Float32(0.0) - x
    return x / (Float32(1.0) + cute.math.exp(neg, fastmath=True))


def _row_slice(tensor, start: int, stop: int):
    return tensor[start:stop]


def _last_dim_slice(tensor, start: int, stop: int):
    return tensor[:, :, start:stop]


def _schedule_mxfp4_pair_projection(
    *,
    x,
    weights0,
    weights1,
    y0,
    y1,
    seq_len,
    matvec_block,
    page_size,
    group_size,
    fp4_ops=QWEN3_5_MXFP4_SIMT_OPS,
):
    fp4_ops = fp4_ops or QWEN3_5_MXFP4_SIMT_OPS
    packed0, scales0 = weights0
    packed1, scales1 = weights1
    return fp4_ops.matvec_pair.schedule(
        a=x,
        weight0_packed=packed0,
        weight0_scales=scales0,
        weight1_packed=packed1,
        weight1_scales=scales1,
        y0=y0,
        y1=y1,
        tile_sizes={"S": seq_len, "O": matvec_block},
        page_size=page_size,
        group_size=group_size,
    )


def _schedule_mxfp4_quad_projection(
    *,
    x,
    weights0,
    weights1,
    weights2,
    weights3,
    y0,
    y1,
    y2,
    y3,
    seq_len,
    matvec_block,
    page_size,
    group_size,
    fp4_ops=QWEN3_5_MXFP4_SIMT_OPS,
):
    fp4_ops = fp4_ops or QWEN3_5_MXFP4_SIMT_OPS
    packed0, scales0 = weights0
    packed1, scales1 = weights1
    packed2, scales2 = weights2
    packed3, scales3 = weights3
    return fp4_ops.matvec_quad.schedule(
        a=x,
        weight0_packed=packed0,
        weight0_scales=scales0,
        weight1_packed=packed1,
        weight1_scales=scales1,
        weight2_packed=packed2,
        weight2_scales=scales2,
        weight3_packed=packed3,
        weight3_scales=scales3,
        y0=y0,
        y1=y1,
        y2=y2,
        y3=y3,
        tile_sizes={"S": seq_len, "O": matvec_block},
        page_size=page_size,
        group_size=group_size,
    )


class Qwen3_5DeltaNetCoreSm120Op(Op):
    """Native Qwen3.5 DeltaNet decode recurrence core.

    Inputs are the already-projected decode vectors for one token:
    ``qkv`` contains raw Q, K, V convolution channels; ``z`` is the output
    gate; ``beta`` and ``alpha`` are scalar per-head projections. The op
    updates ``conv_buf`` and ``dn_state`` in place and writes ``y`` with shape
    ``(B, S, 2048)`` as fp32/bf16 depending on the scheduled output buffer.
    """

    reads = {
        "qkv": (None, ("B", "S", "C")),
        "z": (None, ("B", "S", "V")),
        "beta": (None, ("B", "S", "H")),
        "alpha": (None, ("B", "S", "H")),
        "conv_weight": (None, ("C", "W")),
        "a_log": (None, ("H",)),
        "dt_bias": (None, ("H",)),
        "norm_weight": (None, ("D",)),
        "dn_state": (cutlass.Float32, ("B", "H", "D", "K")),
        "conv_buf": (cutlass.Float32, ("B", "C", "W")),
    }
    writes = {
        "dn_state": (cutlass.Float32, ("B", "H", "D", "K")),
        "conv_buf": (cutlass.Float32, ("B", "C", "W")),
        "y": (None, ("B", "S", "V")),
    }
    tile = ("B", "S", "H")
    dynamic_dims = ("B",)

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, **tensors):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", 1)
        tile_sizes.setdefault("H", 1)
        op = cls._schedule_single(tile_sizes=tile_sizes, **tensors)
        op.static_dims["page_size"] = page_size
        op.static_dims["barrier_signal_y_alias_H"] = "V"
        op.static_dims["barrier_signal_y_tile_size_H"] = QWEN3_5_MXFP4_DN_VALUE_DIM
        return [op]

    @cute.jit
    def _q_ptr(self, page_ptr):
        return cute.make_ptr(cutlass.Float32, page_ptr, cute.AddressSpace.smem)

    @cute.jit
    def _k_ptr(self, page_ptr):
        return cute.make_ptr(
            cutlass.Float32,
            page_ptr + Int32(QWEN3_5_MXFP4_DN_KEY_DIM * 4),
            cute.AddressSpace.smem,
        )

    @cute.jit
    def _v_ptr(self, page_ptr):
        return cute.make_ptr(
            cutlass.Float32,
            page_ptr + Int32(2 * QWEN3_5_MXFP4_DN_KEY_DIM * 4),
            cute.AddressSpace.smem,
        )

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_H,
                qkv, z, beta, alpha, conv_weight, a_log, dt_bias,
                norm_weight, dn_state, conv_buf, y):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        row_idx = tile_S * Int32(self.tile_size_S)
        head = tile_H

        q_s = cute.make_tensor(self._q_ptr(page_ptr), cute.make_layout(QWEN3_5_MXFP4_DN_KEY_DIM))
        k_s = cute.make_tensor(self._k_ptr(page_ptr), cute.make_layout(QWEN3_5_MXFP4_DN_KEY_DIM))
        v_s = cute.make_tensor(self._v_ptr(page_ptr), cute.make_layout(QWEN3_5_MXFP4_DN_VALUE_DIM))

        if row_idx < Int32(self.S):
            qkv_base = tile_B * Int32(self.qkv_stride_B) + row_idx * Int32(self.qkv_stride_S)
            conv_base = tile_B * Int32(self.conv_buf_stride_B)
            regions = Int32(0)
            while regions < Int32(3):
                ch_count = Int32(QWEN3_5_MXFP4_DN_KEY_DIM)
                head_offset = head * Int32(QWEN3_5_MXFP4_DN_KEY_DIM)
                if regions == Int32(1):
                    head_offset = Int32(QWEN3_5_MXFP4_DN_QK_SIZE) + head * Int32(QWEN3_5_MXFP4_DN_KEY_DIM)
                if regions == Int32(2):
                    head_offset = Int32(2 * QWEN3_5_MXFP4_DN_QK_SIZE) + head * Int32(QWEN3_5_MXFP4_DN_VALUE_DIM)
                    ch_count = Int32(QWEN3_5_MXFP4_DN_VALUE_DIM)
                elem = tidx
                while elem < ch_count:
                    ch = head_offset + elem
                    qkv_row = cute.make_tensor(qkv.iterator + qkv_base, cute.make_layout(self.C))
                    cw_row = cute.make_tensor(
                        conv_weight.iterator + ch * Int32(self.conv_weight_stride_C),
                        cute.make_layout(self.W),
                    )
                    cb_row = cute.make_tensor(
                        conv_buf.iterator + conv_base + ch * Int32(self.conv_buf_stride_C),
                        cute.make_layout(self.W),
                    )
                    h0 = cb_row[Int32(1)]
                    h1 = cb_row[Int32(2)]
                    h2 = cb_row[Int32(3)]
                    new_val = qkv_row[ch].to(Float32)
                    cb_row[Int32(0)] = h0
                    cb_row[Int32(1)] = h1
                    cb_row[Int32(2)] = h2
                    cb_row[Int32(3)] = new_val
                    co = (
                        h0 * cw_row[Int32(0)].to(Float32)
                        + h1 * cw_row[Int32(1)].to(Float32)
                        + h2 * cw_row[Int32(2)].to(Float32)
                        + new_val * cw_row[Int32(3)].to(Float32)
                    )
                    val = _silu(co)
                    if regions == Int32(0):
                        q_s[elem] = val
                    elif regions == Int32(1):
                        k_s[elem] = val
                    else:
                        v_s[elem] = val
                    elem = elem + Int32(self.threads_per_row)
                regions = regions + Int32(1)
            named_barrier_sync(Int32(2), Int32(self.threads_per_row))

            if warp_idx == Int32(0):
                ss = Float32(0.0)
                i = lane_idx
                while i < Int32(QWEN3_5_MXFP4_DN_KEY_DIM):
                    qv = q_s[i]
                    ss = ss + qv * qv
                    i = i + Int32(32)
                ss = cute.arch.warp_reduction(ss, operator.add)
                n = cute.math.rsqrt(ss + Float32(1.0e-6), fastmath=True) * Float32(0.08838834764831845)
                i2 = lane_idx
                while i2 < Int32(QWEN3_5_MXFP4_DN_KEY_DIM):
                    q_s[i2] = q_s[i2] * n
                    i2 = i2 + Int32(32)
            if warp_idx == Int32(1):
                ss_k = Float32(0.0)
                ik = lane_idx
                while ik < Int32(QWEN3_5_MXFP4_DN_KEY_DIM):
                    kv = k_s[ik]
                    ss_k = ss_k + kv * kv
                    ik = ik + Int32(32)
                ss_k = cute.arch.warp_reduction(ss_k, operator.add)
                nk = cute.math.rsqrt(ss_k + Float32(1.0e-6), fastmath=True)
                ik2 = lane_idx
                while ik2 < Int32(QWEN3_5_MXFP4_DN_KEY_DIM):
                    k_s[ik2] = k_s[ik2] * nk
                    ik2 = ik2 + Int32(32)
            named_barrier_sync(Int32(2), Int32(self.threads_per_row))

            beta_row = cute.make_tensor(
                beta.iterator + tile_B * Int32(self.beta_stride_B) + row_idx * Int32(self.beta_stride_S),
                cute.make_layout(self.H),
            )
            alpha_row = cute.make_tensor(
                alpha.iterator + tile_B * Int32(self.alpha_stride_B) + row_idx * Int32(self.alpha_stride_S),
                cute.make_layout(self.H),
            )
            beta_h = Float32(1.0) / (Float32(1.0) + cute.math.exp(Float32(0.0) - beta_row[head].to(Float32), fastmath=True))
            a_log_row = cute.make_tensor(a_log.iterator, cute.make_layout(self.H))
            dt_row = cute.make_tensor(dt_bias.iterator, cute.make_layout(self.H))
            ax = alpha_row[head].to(Float32) + dt_row[head].to(Float32)
            softplus = cute.math.log(Float32(1.0) + cute.math.exp(ax, fastmath=True), fastmath=True)
            if ax > Float32(20.0):
                softplus = ax
            decay = cute.math.exp(
                Float32(0.0) - cute.math.exp(a_log_row[head].to(Float32), fastmath=True) * softplus,
                fastmath=True,
            )

            kq = Float32(0.0)
            kk = lane_idx
            while kk < Int32(QWEN3_5_MXFP4_DN_KEY_DIM):
                kq = kq + k_s[kk] * q_s[kk]
                kk = kk + Int32(32)
            kq = cute.arch.warp_reduction(kq, operator.add)

            state_base = (
                tile_B * Int32(self.dn_state_stride_B)
                + head * Int32(self.dn_state_stride_H)
            )
            state_head = cute.make_tensor(
                dn_state.iterator + state_base,
                cute.make_layout(
                    (self.D, self.K),
                    stride=(self.dn_state_stride_D, self.dn_state_stride_K),
                ),
            )
            out_base = (
                tile_B * Int32(self.y_stride_B)
                + row_idx * Int32(self.y_stride_S)
                + head * Int32(QWEN3_5_MXFP4_DN_VALUE_DIM)
            )
            y_row = cute.make_tensor(y.iterator + out_base, cute.make_layout(self.D))
            # KEY_DIM (128) is a multiple of the 32-lane warp, so each lane owns
            # KPER state elements per value row. Cache them in registers on the
            # read for stk/sqv and reuse on the update, halving state reads (the
            # state is the memory-bound term: D*K fp32 per head).
            KPER = QWEN3_5_MXFP4_DN_KEY_DIM // 32
            j = warp_idx
            while j < Int32(QWEN3_5_MXFP4_DN_VALUE_DIM):
                stk = Float32(0.0)
                sqv = Float32(0.0)
                st_vals = [None] * KPER
                for c in cutlass.range_constexpr(KPER):
                    i3 = lane_idx + Int32(c * 32)
                    st = state_head[(Int32(j), i3)]
                    st_vals[c] = st
                    stk = stk + st * k_s[i3]
                    sqv = sqv + st * q_s[i3]
                stk = cute.arch.warp_reduction(stk, operator.add)
                sqv = cute.arch.warp_reduction(sqv, operator.add)
                err = (v_s[Int32(j)] - stk) * beta_h
                o_j = decay * sqv + err * kq
                for c in cutlass.range_constexpr(KPER):
                    i4 = lane_idx + Int32(c * 32)
                    state_head[(Int32(j), i4)] = st_vals[c] * decay + k_s[i4] * err
                if lane_idx == Int32(0):
                    y_row[Int32(j)] = o_j.to(self.y_dtype)
                j = j + Int32(self.threads_per_row // 32)
            named_barrier_sync(Int32(2), Int32(self.threads_per_row))

            # Output RMS + z gate in-place on this head's value vector.
            sq_out = Float32(0.0)
            jj = tidx
            while jj < Int32(QWEN3_5_MXFP4_DN_VALUE_DIM):
                ov = y_row[jj].to(Float32)
                sq_out = sq_out + ov * ov
                jj = jj + Int32(self.threads_per_row)
            sq_out = cute.arch.warp_reduction(sq_out, operator.add)
            if lane_idx == Int32(0):
                q_s[warp_idx] = sq_out
            named_barrier_sync(Int32(2), Int32(self.threads_per_row))
            if warp_idx == Int32(0):
                total = Float32(0.0)
                ww = lane_idx
                while ww < Int32(self.threads_per_row // 32):
                    total = total + q_s[ww]
                    ww = ww + Int32(32)
                total = cute.arch.warp_reduction(total, operator.add)
                if lane_idx == Int32(0):
                    q_s[Int32(0)] = cute.math.rsqrt(
                        total * Float32(1.0 / QWEN3_5_MXFP4_DN_VALUE_DIM) + Float32(QWEN3_5_MXFP4_EPS),
                        fastmath=True,
                    )
            named_barrier_sync(Int32(2), Int32(self.threads_per_row))
            rstd = q_s[Int32(0)]
            z_row = cute.make_tensor(
                z.iterator + tile_B * Int32(self.z_stride_B) + row_idx * Int32(self.z_stride_S),
                cute.make_layout(self.V),
            )
            norm_row = cute.make_tensor(norm_weight.iterator, cute.make_layout(self.D))
            j2 = tidx
            while j2 < Int32(QWEN3_5_MXFP4_DN_VALUE_DIM):
                idx = head * Int32(QWEN3_5_MXFP4_DN_VALUE_DIM) + j2
                ov = y_row[j2].to(Float32)
                gate = _silu(z_row[idx].to(Float32))
                y_row[j2] = (ov * rstd * norm_row[j2].to(Float32) * gate).to(self.y_dtype)
                j2 = j2 + Int32(self.threads_per_row)


class Qwen3_5QGateRopeCacheSm120Op(Op):
    """Normalize/split Q projection, materialize the attention gate, and cache K/V."""

    reads = {
        "q_raw": (None, ("B", "S", "QR")),
        "k_raw": (None, ("B", "S", "KV")),
        "v_raw": (None, ("B", "S", "KV")),
        "cos": (None, ("S", "D2")),
        "sin": (None, ("S", "D2")),
        "q_norm_weight": (None, ("HD",)),
        "k_norm_weight": (None, ("HD",)),
    }
    writes = {
        "q": (None, ("B", "S", "Q")),
        "gate": (None, ("B", "S", "Q")),
        "k_cache": (None, ("B", "T", "KVH", "HD")),
        "v_cache": (None, ("B", "T", "KVH", "HD")),
    }
    tile = ("B", "S")
    dynamic_dims = ("B",)

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, cache_pos=0, **tensors):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", 1)
        op = cls._schedule_single(tile_sizes=tile_sizes, **tensors)
        op.static_dims["page_size"] = page_size
        op.static_dims["cache_pos"] = cache_pos
        op.static_dims["q_heads"] = tensors["q"].shape[2] // tensors["k_cache"].shape[3]
        op.static_dims["kv_heads"] = tensors["k_cache"].shape[2]
        op.static_dims["head_dim"] = tensors["k_cache"].shape[3]
        return [op]

    @cute.jit
    def _norm_scratch(self, page_ptr):
        return cute.make_tensor(
            cute.make_ptr(cutlass.Float32, page_ptr, cute.AddressSpace.smem),
            cute.make_layout(32),
        )

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_2,
                q_raw, k_raw, v_raw, cos, sin, q_norm_weight, k_norm_weight,
                q, gate, k_cache, v_cache):
        tidx = cute.arch.thread_idx()[0]
        row_idx = tile_S * Int32(self.tile_size_S)
        q_raw_base = tile_B * Int32(self.q_raw_stride_B) + row_idx * Int32(self.q_raw_stride_S)
        k_raw_base = tile_B * Int32(self.k_raw_stride_B) + row_idx * Int32(self.k_raw_stride_S)
        v_raw_base = tile_B * Int32(self.v_raw_stride_B) + row_idx * Int32(self.v_raw_stride_S)
        q_base = tile_B * Int32(self.q_stride_B) + row_idx * Int32(self.q_stride_S)
        gate_base = tile_B * Int32(self.gate_stride_B) + row_idx * Int32(self.gate_stride_S)
        cos_row = cute.make_tensor(cos.iterator + row_idx * Int32(self.cos_stride_S), cute.make_layout(self.D2))
        sin_row = cute.make_tensor(sin.iterator + row_idx * Int32(self.sin_stride_S), cute.make_layout(self.D2))
        q_raw_row = cute.make_tensor(q_raw.iterator + q_raw_base, cute.make_layout(self.QR))
        k_raw_row = cute.make_tensor(k_raw.iterator + k_raw_base, cute.make_layout(self.KV))
        v_raw_row = cute.make_tensor(v_raw.iterator + v_raw_base, cute.make_layout(self.KV))
        q_row = cute.make_tensor(q.iterator + q_base, cute.make_layout(self.Q))
        gate_row = cute.make_tensor(gate.iterator + gate_base, cute.make_layout(self.Q))
        q_norm = cute.make_tensor(q_norm_weight.iterator, cute.make_layout(self.HD))
        k_norm = cute.make_tensor(k_norm_weight.iterator, cute.make_layout(self.HD))
        scratch = self._norm_scratch(page_ptr)

        if tidx < Int32(self.q_heads):
            ss_q = Float32(0.0)
            d = Int32(0)
            raw_head = tidx * Int32(2 * self.head_dim)
            while d < Int32(self.head_dim):
                qv = q_raw_row[raw_head + d].to(Float32)
                ss_q = ss_q + qv * qv
                d = d + Int32(1)
            scratch[tidx] = cute.math.rsqrt(
                ss_q * Float32(1.0 / self.head_dim) + Float32(QWEN3_5_MXFP4_EPS),
                fastmath=True,
            )
        if tidx < Int32(self.kv_heads):
            ss_k = Float32(0.0)
            kd = Int32(0)
            k_head = tidx * Int32(self.head_dim)
            while kd < Int32(self.head_dim):
                kv = k_raw_row[k_head + kd].to(Float32)
                ss_k = ss_k + kv * kv
                kd = kd + Int32(1)
            scratch[Int32(self.q_heads) + tidx] = cute.math.rsqrt(
                ss_k * Float32(1.0 / self.head_dim) + Float32(QWEN3_5_MXFP4_EPS),
                fastmath=True,
            )
        named_barrier_sync(Int32(2), Int32(self.threads_per_row))

        elem = tidx
        while elem < Int32(self.Q):
            head = elem // Int32(self.head_dim)
            dim = elem % Int32(self.head_dim)
            raw_head = head * Int32(2 * self.head_dim)
            raw_q = q_raw_row[raw_head + dim].to(Float32) * scratch[head] * q_norm[dim].to(Float32)
            gate_row[elem] = q_raw_row[raw_head + Int32(self.head_dim) + dim]
            if dim < Int32(self.D2):
                low = raw_q
                high = q_raw_row[raw_head + dim + Int32(self.D2)].to(Float32)
                high = high * scratch[head] * q_norm[dim + Int32(self.D2)].to(Float32)
                c = cos_row[dim].to(Float32)
                s = sin_row[dim].to(Float32)
                q_row[head * Int32(self.head_dim) + dim] = (low * c - high * s).to(self.q_dtype)
                q_row[head * Int32(self.head_dim) + dim + Int32(self.D2)] = (high * c + low * s).to(self.q_dtype)
            elif dim >= Int32(2 * self.D2):
                q_row[elem] = raw_q.to(self.q_dtype)
            elem = elem + Int32(self.threads_per_row)

        kv_dim = Int32(self.kv_heads * self.head_dim)
        kv = tidx
        while kv < kv_dim:
            head = kv // Int32(self.head_dim)
            dim = kv % Int32(self.head_dim)
            head_base = head * Int32(self.head_dim)
            k_out = k_raw_row[kv].to(Float32) * scratch[Int32(self.q_heads) + head] * k_norm[dim].to(Float32)
            k_base = (
                tile_B * Int32(self.k_cache_stride_B)
                + (row_idx + Int32(self.cache_pos)) * Int32(self.k_cache_stride_T)
                + head * Int32(self.k_cache_stride_KVH)
            )
            k_row = cute.make_tensor(k_cache.iterator + k_base, cute.make_layout(self.HD))
            if dim < Int32(self.D2):
                low = k_out
                high = k_raw_row[head_base + dim + Int32(self.D2)].to(Float32)
                high = high * scratch[Int32(self.q_heads) + head] * k_norm[dim + Int32(self.D2)].to(Float32)
                c = cos_row[dim].to(Float32)
                s = sin_row[dim].to(Float32)
                k_row[dim] = (low * c - high * s).to(self.k_cache_dtype)
                k_row[dim + Int32(self.D2)] = (high * c + low * s).to(self.k_cache_dtype)
            elif dim >= Int32(2 * self.D2):
                k_row[dim] = k_out.to(self.k_cache_dtype)

            v_base = (
                tile_B * Int32(self.v_cache_stride_B)
                + (row_idx + Int32(self.cache_pos)) * Int32(self.v_cache_stride_T)
                + head * Int32(self.v_cache_stride_KVH)
            )
            v_row = cute.make_tensor(v_cache.iterator + v_base, cute.make_layout(self.HD))
            v_row[dim] = v_raw_row[kv].to(self.v_cache_dtype)
            kv = kv + Int32(self.threads_per_row)


def schedule_qwen3_5_deltanet_mxfp4_sm120(
    *,
    layer_idx,
    batch,
    seq_len,
    weights,
    x_in,
    residual_in,
    x_out,
    residual_out,
    norm_buf,
    qkv_buf,
    z_buf,
    beta_buf,
    alpha_buf,
    dn_out_buf,
    mlp_h_buf,
    dn_state,
    conv_buf,
    page_size=DEFAULT_PAGE_SIZE,
    group_size=QWEN3_5_MXFP4_GROUP_SIZE,
    matvec_block=QWEN3_5_MXFP4_MATVEC_BLOCK,
    gate_up_block=None,
    prefetch_gate_up=False,
    pre_added_input=False,
    preadd_mlp_output=False,
    fp4_ops=QWEN3_5_MXFP4_SIMT_OPS,
) -> DecodeLayerScheduleSm120:
    """Schedule one Qwen3.5 DeltaNet layer with native packed NVFP4 ops.

    This is the linear-attention layer path. All dense projections are scheduled
    as Machete ops; the recurrent DeltaNet body is a single CuTe DSL op that
    updates convolution and recurrent state in place.
    """

    fp4_ops = fp4_ops or QWEN3_5_MXFP4_SIMT_OPS
    pfx = f"layer.{layer_idx}"
    cos = weights.get("cos")
    sin = weights.get("sin")
    if cos is None or sin is None:
        raise KeyError("weights must include cos/sin scratch tensors for RMS projection metadata")
    cos = cos[:seq_len]
    sin = sin[:seq_len]

    def qparts(name):
        qweight = weights[f"{pfx}.{name}_mxfp4"]
        return qweight.packed, qweight.scales

    qkv_packed, qkv_scales = qparts("W_qkv")
    z_packed, z_scales = qparts("W_z")
    out_packed, out_scales = qparts("W_out")
    gate_packed, gate_scales = qparts("W_gate")
    up_packed, up_scales = qparts("W_up")
    down_packed, down_scales = qparts("W_down")
    gate_up_block = int(gate_up_block or QWEN3_5_MXFP4_GATE_UP_BLOCK)

    ops = []
    if pre_added_input:
        ops += RmsCopyNormSm120Op.schedule(
            x=residual_in,
            norm_weight=weights[f"{pfx}.attn_norm"],
            residual_out=residual_out,
            y=norm_buf,
            tile_sizes={"S": seq_len},
            page_size=page_size,
            eps=QWEN3_5_MXFP4_EPS,
        )
    else:
        ops += RmsAddNormSm120Op.schedule(
            x=x_in,
            residual_in=residual_in,
            norm_weight=weights[f"{pfx}.attn_norm"],
            residual_out=residual_out,
            y=norm_buf,
            tile_sizes={"S": seq_len},
            page_size=page_size,
            eps=QWEN3_5_MXFP4_EPS,
        )
    q0 = 0
    k0 = QWEN3_5_MXFP4_DN_QK_SIZE
    v0 = 2 * QWEN3_5_MXFP4_DN_QK_SIZE
    end = QWEN3_5_MXFP4_DN_CONV_CHANNELS
    qkv_q_buf = _last_dim_slice(qkv_buf, q0, k0)
    qkv_k_buf = _last_dim_slice(qkv_buf, k0, v0)
    qkv_v_buf = _last_dim_slice(qkv_buf, v0, end)
    q_packed, q_scales = _row_slice(qkv_packed, q0, k0), _row_slice(qkv_scales, q0, k0)
    k_packed, k_scales = _row_slice(qkv_packed, k0, v0), _row_slice(qkv_scales, k0, v0)
    v_packed, v_scales = _row_slice(qkv_packed, v0, end), _row_slice(qkv_scales, v0, end)

    ops += _schedule_mxfp4_quad_projection(
        x=norm_buf,
        weights0=(q_packed, q_scales),
        weights1=(k_packed, k_scales),
        weights2=(v_packed, v_scales),
        weights3=(z_packed, z_scales),
        y0=qkv_q_buf,
        y1=qkv_k_buf,
        y2=qkv_v_buf,
        y3=z_buf,
        seq_len=seq_len,
        matvec_block=matvec_block,
        page_size=page_size,
        group_size=group_size,
        fp4_ops=fp4_ops,
    )
    ops += MatvecPairSm120Op.schedule(
        a=norm_buf,
        weight0=weights[f"{pfx}.W_beta"],
        weight1=weights[f"{pfx}.W_alpha"],
        y0=beta_buf,
        y1=alpha_buf,
        tile_sizes={"S": seq_len, "O": 16},
        page_size=page_size,
    )
    ops += Qwen3_5DeltaNetCoreSm120Op.schedule(
        qkv=qkv_buf,
        z=z_buf,
        beta=beta_buf,
        alpha=alpha_buf,
        conv_weight=weights[f"{pfx}.conv_weight"],
        a_log=weights[f"{pfx}.a_log"],
        dt_bias=weights[f"{pfx}.dt_bias"],
        norm_weight=weights[f"{pfx}.linear_norm"],
        dn_state=dn_state,
        conv_buf=conv_buf,
        y=dn_out_buf,
        tile_sizes={"S": seq_len, "H": 1},
        page_size=page_size,
    )
    ops += fp4_ops.matvec_residual.schedule(
        a=dn_out_buf,
        weight_packed=out_packed,
        weight_scales=out_scales,
        residual_in=residual_out,
        residual_out=residual_out,
        tile_sizes={"S": seq_len, "O": matvec_block},
        page_size=page_size,
        group_size=group_size,
    )
    ops += fp4_ops.rms_gate_up_silu.schedule(
        x=residual_out,
        norm_weight=weights[f"{pfx}.mlp_norm"],
        gate_packed=gate_packed,
        gate_scales=gate_scales,
        up_packed=up_packed,
        up_scales=up_scales,
        y=mlp_h_buf,
        tile_sizes={"S": seq_len, "D": gate_up_block},
        page_size=page_size,
        eps=QWEN3_5_MXFP4_EPS,
        group_size=group_size,
        prefetch_nvfp4=prefetch_gate_up,
    )
    if preadd_mlp_output:
        ops += fp4_ops.matvec_residual.schedule(
            a=mlp_h_buf,
            weight_packed=down_packed,
            weight_scales=down_scales,
            residual_in=residual_out,
            residual_out=residual_out,
            tile_sizes={"S": seq_len, "O": matvec_block},
            page_size=page_size,
            group_size=group_size,
        )
    else:
        ops += fp4_ops.matvec.schedule(
            a=mlp_h_buf,
            weight_packed=down_packed,
            weight_scales=down_scales,
            y=x_out,
            tile_sizes={"S": seq_len, "O": matvec_block},
            page_size=page_size,
            group_size=group_size,
        )

    keep = [
        qkv_packed, qkv_scales, z_packed, z_scales,
        qkv_q_buf, qkv_k_buf, qkv_v_buf,
        q_packed, q_scales, k_packed, k_scales, v_packed, v_scales,
        out_packed, out_scales, gate_packed, gate_scales,
        up_packed, up_scales, down_packed, down_scales,
    ]
    return DecodeLayerScheduleSm120(ops=ops, attention_config=None, keep_alive=keep)


def schedule_qwen3_5_full_attention_mxfp4_sm120(
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
    norm_buf,
    q_buf,
    q_raw_buf=None,
    kv_raw_buf=None,
    q_gate_buf=None,
    attn_out_buf,
    mlp_h_buf,
    page_size=DEFAULT_PAGE_SIZE,
    group_size=QWEN3_5_MXFP4_GROUP_SIZE,
    fa_num_splits=0,
    matvec_block=QWEN3_5_MXFP4_MATVEC_BLOCK,
    gate_up_block=None,
    prefetch_gate_up=False,
    pre_added_input=False,
    preadd_mlp_output=False,
    fp4_ops=QWEN3_5_MXFP4_SIMT_OPS,
) -> DecodeLayerScheduleSm120:
    """Schedule one Qwen3.5 full-attention layer with packed MXFP4 weights.

    Expected packed-weight keys for ``layer_idx``:
    ``W_q_mxfp4``, ``W_k_mxfp4``, ``W_v_mxfp4``, ``W_o_mxfp4``,
    ``W_gate_mxfp4``, ``W_up_mxfp4``, and ``W_down_mxfp4``.
    """

    fp4_ops = fp4_ops or QWEN3_5_MXFP4_SIMT_OPS
    if seq_len == 1:
        import torch

        pfx = f"layer.{layer_idx}"
        cos = weights["cos"][cache_pos : cache_pos + seq_len]
        sin = weights["sin"][cache_pos : cache_pos + seq_len]
        if q_raw_buf is None:
            q_raw_buf = torch.empty(batch, seq_len, QWEN3_5_MXFP4_Q_RAW_DIM, dtype=q_buf.dtype, device=q_buf.device)
        if kv_raw_buf is None:
            kv_raw_buf = torch.empty(batch, seq_len, 2 * QWEN3_5_MXFP4_KV_DIM, dtype=q_buf.dtype, device=q_buf.device)
        if q_gate_buf is None:
            q_gate_buf = torch.empty_like(q_buf)
        q_4d = q_buf.view(batch, seq_len, QWEN3_5_MXFP4_NUM_Q_HEADS, QWEN3_5_MXFP4_HEAD_DIM)
        k_window = k_cache[:, : cache_pos + seq_len]
        v_window = v_cache[:, : cache_pos + seq_len]
        o_4d = attn_out_buf.view(batch, seq_len, QWEN3_5_MXFP4_NUM_Q_HEADS, QWEN3_5_MXFP4_HEAD_DIM)

        def qparts(name):
            qweight = weights[f"{pfx}.{name}_mxfp4"]
            return qweight.packed, qweight.scales

        q_packed, q_scales = qparts("W_q")
        k_packed, k_scales = qparts("W_k")
        v_packed, v_scales = qparts("W_v")
        o_packed, o_scales = qparts("W_o")
        gate_packed, gate_scales = qparts("W_gate")
        up_packed, up_scales = qparts("W_up")
        down_packed, down_scales = qparts("W_down")
        gate_up_block = int(gate_up_block or QWEN3_5_MXFP4_GATE_UP_BLOCK)

        ops = []
        k_raw_buf = kv_raw_buf[:, :, :QWEN3_5_MXFP4_KV_DIM]
        v_raw_buf = kv_raw_buf[:, :, QWEN3_5_MXFP4_KV_DIM : 2 * QWEN3_5_MXFP4_KV_DIM]

        if pre_added_input:
            ops += RmsCopyNormSm120Op.schedule(
                x=residual_in,
                norm_weight=weights[f"{pfx}.attn_norm"],
                residual_out=residual_out,
                y=norm_buf,
                tile_sizes={"S": seq_len},
                page_size=page_size,
                eps=QWEN3_5_MXFP4_EPS,
            )
        else:
            ops += RmsAddNormSm120Op.schedule(
                x=x_in,
                residual_in=residual_in,
                norm_weight=weights[f"{pfx}.attn_norm"],
                residual_out=residual_out,
                y=norm_buf,
                tile_sizes={"S": seq_len},
                page_size=page_size,
                eps=QWEN3_5_MXFP4_EPS,
            )
        ops += fp4_ops.matvec.schedule(
            a=norm_buf,
            weight_packed=q_packed,
            weight_scales=q_scales,
            y=q_raw_buf,
            tile_sizes={"S": seq_len, "O": matvec_block},
            page_size=page_size,
            group_size=group_size,
        )
        ops += _schedule_mxfp4_pair_projection(
            x=norm_buf,
            weights0=(k_packed, k_scales),
            weights1=(v_packed, v_scales),
            y0=k_raw_buf,
            y1=v_raw_buf,
            seq_len=seq_len,
            matvec_block=16,
            page_size=page_size,
            group_size=group_size,
            fp4_ops=fp4_ops,
        )
        ops += Qwen3_5QGateRopeCacheSm120Op.schedule(
            q_raw=q_raw_buf,
            k_raw=k_raw_buf,
            v_raw=v_raw_buf,
            cos=cos,
            sin=sin,
            q_norm_weight=weights[f"{pfx}.q_norm"],
            k_norm_weight=weights[f"{pfx}.k_norm"],
            q=q_buf,
            gate=q_gate_buf,
            k_cache=k_window,
            v_cache=v_window,
            cache_pos=cache_pos,
            tile_sizes={"S": seq_len},
            page_size=page_size,
        )
        # GQA-grouped flash-decoding: K/V loaded once per KV head and the KV
        # sequence split across CTAs so the op fills the GPU (the per-q-head
        # decode attention only used QH=8 SMs). Gate applied inside the combine.
        from machete.kernels.qwen_3_5.gqa_decode_attention import schedule_qwen3_5_gqa_attention
        attention_ops, attention_keep = schedule_qwen3_5_gqa_attention(
            q=q_4d,
            k=k_window,
            v=v_window,
            gate=q_gate_buf,
            o=o_4d,
            kv_group_size=QWEN3_5_MXFP4_KV_GROUP_SIZE,
            page_size=page_size,
            num_splits=fa_num_splits,
        )
        ops += attention_ops
        ops += fp4_ops.matvec_residual.schedule(
            a=attn_out_buf,
            weight_packed=o_packed,
            weight_scales=o_scales,
            residual_in=residual_out,
            residual_out=residual_out,
            tile_sizes={"S": seq_len, "O": matvec_block},
            page_size=page_size,
            group_size=group_size,
        )
        ops += fp4_ops.rms_gate_up_silu.schedule(
            x=residual_out,
            norm_weight=weights[f"{pfx}.mlp_norm"],
            gate_packed=gate_packed,
            gate_scales=gate_scales,
            up_packed=up_packed,
            up_scales=up_scales,
            y=mlp_h_buf,
            tile_sizes={"S": seq_len, "D": gate_up_block},
            page_size=page_size,
            eps=QWEN3_5_MXFP4_EPS,
            group_size=group_size,
            prefetch_nvfp4=prefetch_gate_up,
        )
        if preadd_mlp_output:
            ops += fp4_ops.matvec_residual.schedule(
                a=mlp_h_buf,
                weight_packed=down_packed,
                weight_scales=down_scales,
                residual_in=residual_out,
                residual_out=residual_out,
                tile_sizes={"S": seq_len, "O": matvec_block},
                page_size=page_size,
                group_size=group_size,
            )
        else:
            ops += fp4_ops.matvec.schedule(
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
            q_raw_buf, kv_raw_buf, q_gate_buf, k_raw_buf, v_raw_buf, *attention_keep,
            q_packed, q_scales, k_packed, k_scales, v_packed, v_scales,
            o_packed, o_scales, gate_packed, gate_scales, up_packed, up_scales,
            down_packed, down_scales,
        ]
        return DecodeLayerScheduleSm120(ops=ops, attention_config=None, keep_alive=keep)

    raise NotImplementedError("Qwen3.5 MXFP4 decode currently supports seq_len=1")


def schedule_qwen3_5_final_mxfp4_sm120(
    *,
    x,
    residual_in,
    residual_out,
    final_norm,
    lm_head_mxfp4=None,
    logits=None,
    top_values=None,
    top_indices=None,
    top_partial_values=None,
    top_partial_indices=None,
    top_atomic_winner=None,
    top_atomic_counter=None,
    top_atomic_partitions=None,
    top_atomic_skip_init=False,
    seq_len,
    page_size=DEFAULT_PAGE_SIZE,
    group_size=QWEN3_5_MXFP4_GROUP_SIZE,
    fp4_ops=QWEN3_5_MXFP4_SIMT_OPS,
):
    """Schedule final residual plus packed NVFP4 LM head for Qwen3.5."""
    fp4_ops = fp4_ops or QWEN3_5_MXFP4_SIMT_OPS
    ops = []
    if lm_head_mxfp4 is not None:
        if top_values is not None or top_indices is not None:
            if top_values is None or top_indices is None:
                raise ValueError("top_values and top_indices must be provided together")
            if top_atomic_winner is not None or top_atomic_counter is not None:
                if top_atomic_winner is None or top_atomic_counter is None:
                    raise ValueError("top_atomic_winner and top_atomic_counter must be provided together")
                if top_atomic_partitions is None:
                    raise ValueError("top_atomic_partitions must be provided for atomic top-1")
                if not top_atomic_skip_init:
                    ops += FinalTop1AtomicInitSm120Op.schedule(
                        atomic_winner=top_atomic_winner,
                        atomic_counter=top_atomic_counter,
                        tile_sizes={"S": seq_len},
                        page_size=page_size,
                    )
                ops += fp4_ops.final_add_rms_top1_atomic_lm_head.schedule(
                    x=x,
                    residual_in=residual_in,
                    norm_weight=final_norm,
                    weight_packed=lm_head_mxfp4.packed,
                    weight_scales=lm_head_mxfp4.scales,
                    atomic_winner=top_atomic_winner,
                    atomic_counter=top_atomic_counter,
                    top_values=top_values,
                    top_indices=top_indices,
                    tile_sizes={"S": seq_len, "P": 1},
                    page_size=page_size,
                    eps=QWEN3_5_MXFP4_EPS,
                    group_size=group_size,
                    partitions=top_atomic_partitions,
                    reset_scratch=top_atomic_skip_init,
                )
            elif top_partial_values is not None or top_partial_indices is not None:
                if top_partial_values is None or top_partial_indices is None:
                    raise ValueError("top_partial_values and top_partial_indices must be provided together")
                ops += fp4_ops.final_add_rms_top1_partial_lm_head.schedule(
                    x=x,
                    residual_in=residual_in,
                    norm_weight=final_norm,
                    weight_packed=lm_head_mxfp4.packed,
                    weight_scales=lm_head_mxfp4.scales,
                    partial_values=top_partial_values,
                    partial_indices=top_partial_indices,
                    tile_sizes={"S": seq_len, "P": 1},
                    page_size=page_size,
                    eps=QWEN3_5_MXFP4_EPS,
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
                    tile_sizes={"S": seq_len},
                    page_size=page_size,
                )
                ops += fp4_ops.final_rms_top1_lm_head.schedule(
                    x=residual_out,
                    norm_weight=final_norm,
                    weight_packed=lm_head_mxfp4.packed,
                    weight_scales=lm_head_mxfp4.scales,
                    top_values=top_values,
                    top_indices=top_indices,
                    tile_sizes={"S": seq_len, "V": 16},
                    page_size=page_size,
                    eps=QWEN3_5_MXFP4_EPS,
                    group_size=group_size,
                )
        else:
            if logits is None:
                raise ValueError(
                    "logits or top_values/top_indices must be provided when lm_head_mxfp4 is scheduled"
                )
            ops += ResidualAddSm120Op.schedule(
                x=x,
                residual_in=residual_in,
                residual_out=residual_out,
                tile_sizes={"S": seq_len},
                page_size=page_size,
            )
            ops += fp4_ops.final_rms_lm_head.schedule(
                x=residual_out,
                norm_weight=final_norm,
                weight_packed=lm_head_mxfp4.packed,
                weight_scales=lm_head_mxfp4.scales,
                logits=logits,
                tile_sizes={"S": seq_len, "V": 16},
                page_size=page_size,
                eps=QWEN3_5_MXFP4_EPS,
                group_size=group_size,
            )
    else:
        ops += ResidualAddSm120Op.schedule(
            x=x,
            residual_in=residual_in,
            residual_out=residual_out,
            tile_sizes={"S": seq_len},
            page_size=page_size,
        )
    return ops


def _layer_resource(resource, layer_idx, slot=None):
    if isinstance(resource, (list, tuple)):
        return resource[layer_idx if slot is None else slot]
    if hasattr(resource, "dim") and resource.dim() > 0:
        if slot is not None and resource.shape[0] == QWEN3_5_LAYER_TYPES.count("linear_attention"):
            return resource[slot]
        if resource.shape[0] == QWEN3_5_MXFP4_NUM_LAYERS:
            return resource[layer_idx]
    return resource


def schedule_qwen3_5_mxfp4_decode_sm120(
    *,
    batch,
    seq_len,
    cache_pos,
    weights,
    x_buffers,
    residual_buffers,
    k_cache,
    v_cache,
    q_buf,
    q_raw_buf=None,
    kv_raw_buf=None,
    q_gate_buf=None,
    attn_out_buf,
    norm_buf,
    qkv_buf,
    z_buf,
    beta_buf,
    alpha_buf,
    dn_out_buf,
    mlp_h_buf,
    dn_state,
    conv_buf,
    final_norm=None,
    lm_head_mxfp4=None,
    logits=None,
    top_values=None,
    top_indices=None,
    top_partial_values=None,
    top_partial_indices=None,
    top_atomic_winner=None,
    top_atomic_counter=None,
    top_atomic_partitions=None,
    top_atomic_skip_init=False,
    page_size=DEFAULT_PAGE_SIZE,
    group_size=QWEN3_5_MXFP4_GROUP_SIZE,
    fa_num_splits=0,
    matvec_block=QWEN3_5_MXFP4_MATVEC_BLOCK,
    gate_up_block=None,
    prefetch_gate_up=False,
    max_layers=None,
    fuse_down_next_norm=False,
    fp4_ops=QWEN3_5_MXFP4_SIMT_OPS,
):
    """Build the full 24-layer Qwen3.5 NVFP4 decode schedule.

    ``x_buffers`` and ``residual_buffers`` must contain one entry per layer
    boundary: index 0 is model input state and index 24 is final layer output.
    Scratch tensors may be single reusable tensors or per-layer/per-linear-layer
    lists/tensors on their leading dimension.
    """

    fp4_ops = fp4_ops or QWEN3_5_MXFP4_SIMT_OPS
    if len(x_buffers) < QWEN3_5_MXFP4_NUM_LAYERS + 1:
        raise ValueError("x_buffers must contain 25 layer-boundary tensors")
    if len(residual_buffers) < QWEN3_5_MXFP4_NUM_LAYERS + 1:
        raise ValueError("residual_buffers must contain 25 layer-boundary tensors")

    ops = []
    keep = []
    attention_configs = []
    linear_slot = 0
    layer_limit = QWEN3_5_MXFP4_NUM_LAYERS if max_layers is None else int(max_layers)
    for layer_idx, layer_type in enumerate(QWEN3_5_LAYER_TYPES[:layer_limit]):
        pre_added_input = bool(fuse_down_next_norm and layer_idx > 0)
        preadd_mlp_output = bool(
            fuse_down_next_norm and layer_idx + 1 < layer_limit
        )
        if layer_type == "full_attention":
            layer = schedule_qwen3_5_full_attention_mxfp4_sm120(
                layer_idx=layer_idx,
                batch=batch,
                seq_len=seq_len,
                cache_pos=cache_pos,
                weights=weights,
                k_cache=_layer_resource(k_cache, layer_idx),
                v_cache=_layer_resource(v_cache, layer_idx),
                x_in=x_buffers[layer_idx],
                residual_in=residual_buffers[layer_idx],
                x_out=x_buffers[layer_idx + 1],
                residual_out=residual_buffers[layer_idx + 1],
                norm_buf=_layer_resource(norm_buf, layer_idx),
                q_raw_buf=None if q_raw_buf is None else _layer_resource(q_raw_buf, layer_idx),
                kv_raw_buf=None if kv_raw_buf is None else _layer_resource(kv_raw_buf, layer_idx),
                q_gate_buf=None if q_gate_buf is None else _layer_resource(q_gate_buf, layer_idx),
                q_buf=_layer_resource(q_buf, layer_idx),
                attn_out_buf=_layer_resource(attn_out_buf, layer_idx),
                mlp_h_buf=_layer_resource(mlp_h_buf, layer_idx),
                page_size=page_size,
                group_size=group_size,
                fa_num_splits=fa_num_splits,
                matvec_block=matvec_block,
                gate_up_block=gate_up_block,
                prefetch_gate_up=prefetch_gate_up,
                pre_added_input=pre_added_input,
                preadd_mlp_output=preadd_mlp_output,
                fp4_ops=fp4_ops,
            )
        elif layer_type == "linear_attention":
            layer = schedule_qwen3_5_deltanet_mxfp4_sm120(
                layer_idx=layer_idx,
                batch=batch,
                seq_len=seq_len,
                weights=weights,
                x_in=x_buffers[layer_idx],
                residual_in=residual_buffers[layer_idx],
                x_out=x_buffers[layer_idx + 1],
                residual_out=residual_buffers[layer_idx + 1],
                norm_buf=_layer_resource(norm_buf, layer_idx, linear_slot),
                qkv_buf=_layer_resource(qkv_buf, layer_idx, linear_slot),
                z_buf=_layer_resource(z_buf, layer_idx, linear_slot),
                beta_buf=_layer_resource(beta_buf, layer_idx, linear_slot),
                alpha_buf=_layer_resource(alpha_buf, layer_idx, linear_slot),
                dn_out_buf=_layer_resource(dn_out_buf, layer_idx, linear_slot),
                mlp_h_buf=_layer_resource(mlp_h_buf, layer_idx),
                dn_state=_layer_resource(dn_state, layer_idx, linear_slot),
                conv_buf=_layer_resource(conv_buf, layer_idx, linear_slot),
                page_size=page_size,
                group_size=group_size,
                matvec_block=matvec_block,
                gate_up_block=gate_up_block,
                prefetch_gate_up=prefetch_gate_up,
                pre_added_input=pre_added_input,
                preadd_mlp_output=preadd_mlp_output,
                fp4_ops=fp4_ops,
            )
            linear_slot += 1
        else:
            raise ValueError(f"unknown Qwen3.5 layer type {layer_type!r}")
        ops.extend(layer.ops)
        keep.extend(layer.keep_alive)
        attention_configs.append(layer.attention_config)

    if final_norm is not None and layer_limit == QWEN3_5_MXFP4_NUM_LAYERS:
        final_ops = schedule_qwen3_5_final_mxfp4_sm120(
            x=x_buffers[QWEN3_5_MXFP4_NUM_LAYERS],
            residual_in=residual_buffers[QWEN3_5_MXFP4_NUM_LAYERS],
            residual_out=residual_buffers[QWEN3_5_MXFP4_NUM_LAYERS],
            final_norm=final_norm,
            lm_head_mxfp4=lm_head_mxfp4,
            logits=logits,
            top_values=top_values,
            top_indices=top_indices,
            top_partial_values=top_partial_values,
            top_partial_indices=top_partial_indices,
            top_atomic_winner=top_atomic_winner,
            top_atomic_counter=top_atomic_counter,
            top_atomic_partitions=top_atomic_partitions,
            top_atomic_skip_init=top_atomic_skip_init,
            seq_len=seq_len,
            page_size=page_size,
            group_size=group_size,
            fp4_ops=fp4_ops,
        )
        ops.extend(final_ops)

    return DecodeLayerScheduleSm120(
        ops=ops,
        attention_config=attention_configs,
        keep_alive=keep,
    )


def schedule_qwen3_5_mxfp4_simt_decode_sm120(*args, **kwargs):
    kwargs.setdefault("matvec_block", QWEN3_5_MXFP4_SIMT_MATVEC_BLOCK)
    kwargs.setdefault("fp4_ops", QWEN3_5_MXFP4_SIMT_OPS)
    return schedule_qwen3_5_mxfp4_decode_sm120(*args, **kwargs)


__all__ = [
    "Qwen3_5DeltaNetCoreSm120Op",
    "Qwen3_5Fp4OpSet",
    "MXFP4SimtWeight",
    "empty_mxfp4_simt_weight",
    "quantize_mxfp4_simt_weight",
    "QWEN3_5_MXFP4_SIMT_MATVEC_BLOCK",
    "QWEN3_5_MXFP4_SIMT_OPS",
    "MatvecMxfp4SimtSm120Op",
    "MatvecPairMxfp4SimtSm120Op",
    "MatvecQuadMxfp4SimtSm120Op",
    "MatvecResidualMxfp4SimtSm120Op",
    "RmsGateUpSiluMxfp4SimtSm120Op",
    "FinalRmsLmHeadMxfp4SimtSm120Op",
    "FinalRmsTop1LmHeadMxfp4SimtSm120Op",
    "FinalAddRmsTop1PartialLmHeadMxfp4SimtSm120Op",
    "FinalAddRmsTop1AtomicLmHeadMxfp4SimtSm120Op",
    "Qwen3_5QGateRopeCacheSm120Op",
    "QWEN3_5_MXFP4_DN_CONV_CHANNELS",
    "QWEN3_5_MXFP4_DN_CONV_KERNEL",
    "QWEN3_5_MXFP4_DN_NUM_HEADS",
    "QWEN3_5_MXFP4_DN_VALUE_DIM",
    "QWEN3_5_MXFP4_DN_V_SIZE",
    "QWEN3_5_MXFP4_EPS",
    "QWEN3_5_MXFP4_HEAD_DIM",
    "QWEN3_5_MXFP4_HIDDEN",
    "QWEN3_5_MXFP4_INTERMEDIATE",
    "QWEN3_5_MXFP4_KV_DIM",
    "QWEN3_5_MXFP4_KV_GROUP_SIZE",
    "QWEN3_5_LAYER_TYPES",
    "QWEN3_5_MXFP4_NUM_KV_HEADS",
    "QWEN3_5_MXFP4_NUM_LAYERS",
    "QWEN3_5_MXFP4_NUM_Q_HEADS",
    "QWEN3_5_MXFP4_Q_DIM",
    "QWEN3_5_MXFP4_Q_RAW_DIM",
    "QWEN3_5_MXFP4_ROTARY_D2",
    "QWEN3_5_MXFP4_VOCAB",
    "schedule_qwen3_5_final_mxfp4_sm120",
    "schedule_qwen3_5_deltanet_mxfp4_sm120",
    "schedule_qwen3_5_full_attention_mxfp4_sm120",
    "schedule_qwen3_5_mxfp4_simt_decode_sm120",
    "schedule_qwen3_5_mxfp4_decode_sm120",
]
