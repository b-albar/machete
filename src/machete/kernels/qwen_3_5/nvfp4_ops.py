# Copyright (c) 2025, Machete Authors
"""Native CuTe DSL decode schedules for Qwen3.5-0.8B NVFP4.

This file contains Machete megakernel-op schedules only. It does not import or
delegate to external CUDA extensions.
"""

from __future__ import annotations

import operator

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32

from machete.kernels.decode_matvec import (
    DecodeLayerScheduleSm120,
    MatvecResidualNvfp4Sm120Op,
    MatvecNvfp4Sm120Op,
    MatvecPairNvfp4Sm120Op,
    MatvecQuadNvfp4Sm120Op,
    RmsAddNormSm120Op,
    RmsGateUpSiluNvfp4Sm120Op,
    RmsKMatvecRopeCacheNvfp4Sm120Op,
    RmsMatvecNvfp4Sm120Op,
    RmsQMatvecRopeNvfp4Sm120Op,
    RmsReadMatvecNvfp4Sm120Op,
    RmsVMatvecCacheNvfp4Sm120Op,
    schedule_decode_layer_nvfp4_sm120,
    schedule_final_nvfp4_sm120,
)
from machete.kernels.attention.flash_decoding import FlashDecodingCombineBSHDOp, flash_decoding_schedule
from machete.megakernel.interpreter import named_barrier_sync
from machete.megakernel.ops import DEFAULT_PAGE_SIZE, Op, PipelineSpec


QWEN3_5_NVFP4_NUM_LAYERS = 24
QWEN3_5_NVFP4_HIDDEN = 1024
QWEN3_5_NVFP4_INTERMEDIATE = 3584
QWEN3_5_NVFP4_VOCAB = 248320
QWEN3_5_NVFP4_NUM_Q_HEADS = 8
QWEN3_5_NVFP4_NUM_KV_HEADS = 2
QWEN3_5_NVFP4_HEAD_DIM = 256
QWEN3_5_NVFP4_ROTARY_D2 = 32
QWEN3_5_NVFP4_Q_DIM = QWEN3_5_NVFP4_NUM_Q_HEADS * QWEN3_5_NVFP4_HEAD_DIM
QWEN3_5_NVFP4_Q_RAW_DIM = 2 * QWEN3_5_NVFP4_Q_DIM
QWEN3_5_NVFP4_KV_DIM = QWEN3_5_NVFP4_NUM_KV_HEADS * QWEN3_5_NVFP4_HEAD_DIM
QWEN3_5_NVFP4_KV_GROUP_SIZE = QWEN3_5_NVFP4_NUM_Q_HEADS // QWEN3_5_NVFP4_NUM_KV_HEADS
QWEN3_5_NVFP4_DECODE_S = 1
QWEN3_5_NVFP4_EPS = 1e-6
QWEN3_5_NVFP4_GROUP_SIZE = 32
QWEN3_5_NVFP4_DN_NUM_HEADS = 16
QWEN3_5_NVFP4_DN_KEY_DIM = 128
QWEN3_5_NVFP4_DN_VALUE_DIM = 128
QWEN3_5_NVFP4_DN_QK_SIZE = QWEN3_5_NVFP4_DN_NUM_HEADS * QWEN3_5_NVFP4_DN_KEY_DIM
QWEN3_5_NVFP4_DN_V_SIZE = QWEN3_5_NVFP4_DN_NUM_HEADS * QWEN3_5_NVFP4_DN_VALUE_DIM
QWEN3_5_NVFP4_DN_CONV_CHANNELS = 2 * QWEN3_5_NVFP4_DN_QK_SIZE + QWEN3_5_NVFP4_DN_V_SIZE
QWEN3_5_NVFP4_DN_CONV_KERNEL = 4
QWEN3_5_NVFP4_MATVEC_BLOCK = 16
QWEN3_5_NVFP4_FLASH_ATTN_MIN_CONTEXT = 4096
QWEN3_5_NVFP4_FP32_NEG_INF = -3.4028234663852886e38
QWEN3_5_NVFP4_ATTN_SCALE = 0.0625
QWEN3_5_LAYER_TYPES = ("linear_attention", "linear_attention", "linear_attention", "full_attention") * 6


@cute.jit
def _silu(x):
    neg = Float32(0.0) - x
    return x / (Float32(1.0) + cute.math.exp(neg, fastmath=True))


def _row_slice(tensor, start: int, stop: int):
    return tensor[start:stop]


def _last_dim_slice(tensor, start: int, stop: int):
    return tensor[:, :, start:stop]


def _schedule_nvfp4_pair_projection(
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
):
    packed0, scales0 = weights0
    packed1, scales1 = weights1
    return MatvecPairNvfp4Sm120Op.schedule(
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


def _schedule_nvfp4_quad_projection(
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
):
    packed0, scales0 = weights0
    packed1, scales1 = weights1
    packed2, scales2 = weights2
    packed3, scales3 = weights3
    return MatvecQuadNvfp4Sm120Op.schedule(
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
        op.static_dims["barrier_signal_y_tile_size_H"] = QWEN3_5_NVFP4_DN_VALUE_DIM
        return [op]

    @cute.jit
    def _q_ptr(self, page_ptr):
        return cute.make_ptr(cutlass.Float32, page_ptr, cute.AddressSpace.smem)

    @cute.jit
    def _k_ptr(self, page_ptr):
        return cute.make_ptr(
            cutlass.Float32,
            page_ptr + Int32(QWEN3_5_NVFP4_DN_KEY_DIM * 4),
            cute.AddressSpace.smem,
        )

    @cute.jit
    def _v_ptr(self, page_ptr):
        return cute.make_ptr(
            cutlass.Float32,
            page_ptr + Int32(2 * QWEN3_5_NVFP4_DN_KEY_DIM * 4),
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

        q_s = cute.make_tensor(self._q_ptr(page_ptr), cute.make_layout(QWEN3_5_NVFP4_DN_KEY_DIM))
        k_s = cute.make_tensor(self._k_ptr(page_ptr), cute.make_layout(QWEN3_5_NVFP4_DN_KEY_DIM))
        v_s = cute.make_tensor(self._v_ptr(page_ptr), cute.make_layout(QWEN3_5_NVFP4_DN_VALUE_DIM))

        if row_idx < Int32(self.S):
            qkv_base = tile_B * Int32(self.qkv_stride_B) + row_idx * Int32(self.qkv_stride_S)
            conv_base = tile_B * Int32(self.conv_buf_stride_B)
            regions = Int32(0)
            while regions < Int32(3):
                ch_count = Int32(QWEN3_5_NVFP4_DN_KEY_DIM)
                head_offset = head * Int32(QWEN3_5_NVFP4_DN_KEY_DIM)
                if regions == Int32(1):
                    head_offset = Int32(QWEN3_5_NVFP4_DN_QK_SIZE) + head * Int32(QWEN3_5_NVFP4_DN_KEY_DIM)
                if regions == Int32(2):
                    head_offset = Int32(2 * QWEN3_5_NVFP4_DN_QK_SIZE) + head * Int32(QWEN3_5_NVFP4_DN_VALUE_DIM)
                    ch_count = Int32(QWEN3_5_NVFP4_DN_VALUE_DIM)
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
                while i < Int32(QWEN3_5_NVFP4_DN_KEY_DIM):
                    qv = q_s[i]
                    ss = ss + qv * qv
                    i = i + Int32(32)
                ss = cute.arch.warp_reduction(ss, operator.add)
                n = cute.math.rsqrt(ss + Float32(1.0e-6), fastmath=True) * Float32(0.08838834764831845)
                i2 = lane_idx
                while i2 < Int32(QWEN3_5_NVFP4_DN_KEY_DIM):
                    q_s[i2] = q_s[i2] * n
                    i2 = i2 + Int32(32)
            if warp_idx == Int32(1):
                ss_k = Float32(0.0)
                ik = lane_idx
                while ik < Int32(QWEN3_5_NVFP4_DN_KEY_DIM):
                    kv = k_s[ik]
                    ss_k = ss_k + kv * kv
                    ik = ik + Int32(32)
                ss_k = cute.arch.warp_reduction(ss_k, operator.add)
                nk = cute.math.rsqrt(ss_k + Float32(1.0e-6), fastmath=True)
                ik2 = lane_idx
                while ik2 < Int32(QWEN3_5_NVFP4_DN_KEY_DIM):
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
            while kk < Int32(QWEN3_5_NVFP4_DN_KEY_DIM):
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
                + head * Int32(QWEN3_5_NVFP4_DN_VALUE_DIM)
            )
            y_row = cute.make_tensor(y.iterator + out_base, cute.make_layout(self.D))
            j = warp_idx
            while j < Int32(QWEN3_5_NVFP4_DN_VALUE_DIM):
                stk = Float32(0.0)
                sqv = Float32(0.0)
                i3 = lane_idx
                while i3 < Int32(QWEN3_5_NVFP4_DN_KEY_DIM):
                    st = state_head[(Int32(j), i3)]
                    stk = stk + st * k_s[i3]
                    sqv = sqv + st * q_s[i3]
                    i3 = i3 + Int32(32)
                stk = cute.arch.warp_reduction(stk, operator.add)
                sqv = cute.arch.warp_reduction(sqv, operator.add)
                err = (v_s[Int32(j)] - stk) * beta_h
                o_j = decay * sqv + err * kq
                i4 = lane_idx
                while i4 < Int32(QWEN3_5_NVFP4_DN_KEY_DIM):
                    st_old = state_head[(Int32(j), i4)]
                    state_head[(Int32(j), i4)] = st_old * decay + k_s[i4] * err
                    i4 = i4 + Int32(32)
                if lane_idx == Int32(0):
                    y_row[Int32(j)] = o_j.to(self.y_dtype)
                j = j + Int32(self.threads_per_row // 32)
            named_barrier_sync(Int32(2), Int32(self.threads_per_row))

            # Output RMS + z gate in-place on this head's value vector.
            sq_out = Float32(0.0)
            jj = tidx
            while jj < Int32(QWEN3_5_NVFP4_DN_VALUE_DIM):
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
                        total * Float32(1.0 / QWEN3_5_NVFP4_DN_VALUE_DIM) + Float32(QWEN3_5_NVFP4_EPS),
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
            while j2 < Int32(QWEN3_5_NVFP4_DN_VALUE_DIM):
                idx = head * Int32(QWEN3_5_NVFP4_DN_VALUE_DIM) + j2
                ov = y_row[j2].to(Float32)
                gate = _silu(z_row[idx].to(Float32))
                y_row[j2] = (ov * rstd * norm_row[j2].to(Float32) * gate).to(self.y_dtype)
                j2 = j2 + Int32(self.threads_per_row)


class Qwen3_5SingleTokenAttentionSm120Op(Op):
    """Native S=1 decode attention for Qwen3.5 full-attention layers."""

    framework_owned_ranges = True
    reads = {
        "q": (None, ("B", "S", "QH", "HD")),
        "k": (None, ("B", "T", "KVH", "HD")),
        "v": (None, ("B", "T", "KVH", "HD")),
    }
    writes = {"o": (None, ("B", "S", "QH", "HD"))}
    tile = ("B", "QH")
    dynamic_dims = ("B", "T")

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, kv_group_size=1, **tensors):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("QH", 1)
        required_page_size = (int(tensors["k"].shape[1]) + 32) * 4
        if page_size < required_page_size:
            raise ValueError(
                "Qwen3_5SingleTokenAttentionSm120Op page_size is too small for "
                f"T={tensors['k'].shape[1]}: need at least {required_page_size} bytes"
            )
        op = cls._schedule_single(tile_sizes=tile_sizes, **tensors)
        op.static_dims["page_size"] = page_size
        op.static_dims["kv_group_size"] = kv_group_size
        op.static_dims["T"] = tensors["k"].shape[1]
        op.static_dims["HD"] = tensors["k"].shape[3]
        op.static_dims["QH"] = tensors["q"].shape[2]
        op.static_dims["KVH"] = tensors["k"].shape[2]
        return [op]

    @cute.jit
    def _scores_ptr(self, page_ptr):
        return cute.make_ptr(cutlass.Float32, page_ptr, cute.AddressSpace.smem)

    @cute.jit
    def _scratch_ptr(self, page_ptr):
        return cute.make_ptr(
            cutlass.Float32,
            page_ptr + Int32(self.T * 4),
            cute.AddressSpace.smem,
        )

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_QH, tile_2, q, k, v, o):
        tidx = cute.arch.thread_idx()[0]
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()
        num_warps = self.threads_per_row // 32
        scores = cute.make_tensor(self._scores_ptr(page_ptr), cute.make_layout(self.T))
        scratch = cute.make_tensor(
            self._scratch_ptr(page_ptr),
            cute.make_layout(Int32(num_warps)),
        )

        kv_head = tile_QH // Int32(self.kv_group_size)
        q_base = (
            tile_B * Int32(self.q_stride_B)
            + tile_QH * Int32(self.q_stride_QH)
        )
        q_row = cute.make_tensor(q.iterator + q_base, cute.make_layout(self.HD))

        row_max = Float32(QWEN3_5_NVFP4_FP32_NEG_INF)
        n = warp_idx
        while n < Int32(self.T):
            k_base = (
                tile_B * Int32(self.k_stride_B)
                + n * Int32(self.k_stride_T)
                + kv_head * Int32(self.k_stride_KVH)
            )
            k_row = cute.make_tensor(k.iterator + k_base, cute.make_layout(self.HD))
            acc = Float32(0.0)
            d = lane_idx
            while d < Int32(self.HD):
                acc = acc + q_row[d].to(Float32) * k_row[d].to(Float32)
                d = d + Int32(32)
            score = cute.arch.warp_reduction(acc, operator.add) * Float32(QWEN3_5_NVFP4_ATTN_SCALE)
            if lane_idx == Int32(0):
                scores[n] = score
                if score > row_max:
                    row_max = score
            n = n + Int32(num_warps)
        if lane_idx == Int32(0):
            scratch[warp_idx] = row_max
        named_barrier_sync(Int32(2), Int32(self.threads_per_row))

        if warp_idx == Int32(0):
            partial_max = Float32(QWEN3_5_NVFP4_FP32_NEG_INF)
            w = lane_idx
            while w < Int32(num_warps):
                val = scratch[w]
                if val > partial_max:
                    partial_max = val
                w = w + Int32(32)
            row_max = cute.arch.warp_reduction(partial_max, cute.arch.fmax)
            if lane_idx == Int32(0):
                scratch[Int32(0)] = row_max
        named_barrier_sync(Int32(2), Int32(self.threads_per_row))

        rmax = scratch[Int32(0)]
        row_sum_part = Float32(0.0)
        n2 = warp_idx * Int32(32) + lane_idx
        while n2 < Int32(self.T):
            row_sum_part = row_sum_part + cute.math.exp(scores[n2] - rmax, fastmath=True)
            n2 = n2 + Int32(self.threads_per_row)
        row_sum_part = cute.arch.warp_reduction(row_sum_part, operator.add)
        if lane_idx == Int32(0):
            scratch[warp_idx] = row_sum_part
        named_barrier_sync(Int32(2), Int32(self.threads_per_row))

        if warp_idx == Int32(0):
            row_sum = Float32(0.0)
            w2 = lane_idx
            while w2 < Int32(num_warps):
                row_sum = row_sum + scratch[w2]
                w2 = w2 + Int32(32)
            row_sum = cute.arch.warp_reduction(row_sum, operator.add)
            if lane_idx == Int32(0):
                scratch[Int32(1)] = Float32(1.0) / row_sum
        named_barrier_sync(Int32(2), Int32(self.threads_per_row))

        rinv = scratch[Int32(1)]
        n_prob = tidx
        while n_prob < Int32(self.T):
            scores[n_prob] = cute.math.exp(scores[n_prob] - rmax, fastmath=True) * rinv
            n_prob = n_prob + Int32(self.threads_per_row)
        named_barrier_sync(Int32(2), Int32(self.threads_per_row))

        out_base = (
            tile_B * Int32(self.o_stride_B)
            + tile_QH * Int32(self.o_stride_QH)
        )
        out_row = cute.make_tensor(o.iterator + out_base, cute.make_layout(self.HD))
        d_out = tidx
        while d_out < Int32(self.HD):
            acc_o = Float32(0.0)
            n3 = Int32(0)
            while n3 < Int32(self.T):
                v_base = (
                    tile_B * Int32(self.v_stride_B)
                    + n3 * Int32(self.v_stride_T)
                    + kv_head * Int32(self.v_stride_KVH)
                )
                v_row = cute.make_tensor(v.iterator + v_base, cute.make_layout(self.HD))
                acc_o = acc_o + scores[n3] * v_row[d_out].to(Float32)
                n3 = n3 + Int32(1)
            out_row[d_out] = acc_o.to(self.o_dtype)
            d_out = d_out + Int32(self.threads_per_row)


class Qwen3_5GatedSingleTokenAttentionSm120Op(Qwen3_5SingleTokenAttentionSm120Op):
    """S=1 decode attention that applies Qwen3.5's output gate before storing."""

    reads = {
        "q": (None, ("B", "S", "QH", "HD")),
        "k": (None, ("B", "T", "KVH", "HD")),
        "v": (None, ("B", "T", "KVH", "HD")),
        "gate": (None, ("B", "S", "Q")),
    }
    writes = {"o": (None, ("B", "S", "QH", "HD"))}

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_QH, tile_2, q, k, v, gate, o):
        tidx = cute.arch.thread_idx()[0]
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()
        num_warps = self.threads_per_row // 32
        scores = cute.make_tensor(self._scores_ptr(page_ptr), cute.make_layout(self.T))
        scratch = cute.make_tensor(
            self._scratch_ptr(page_ptr),
            cute.make_layout(Int32(num_warps)),
        )

        kv_head = tile_QH // Int32(self.kv_group_size)
        q_base = (
            tile_B * Int32(self.q_stride_B)
            + tile_QH * Int32(self.q_stride_QH)
        )
        q_row = cute.make_tensor(q.iterator + q_base, cute.make_layout(self.HD))

        # Phase 1: compute QK scores and each warp's local softmax max.
        row_max = Float32(QWEN3_5_NVFP4_FP32_NEG_INF)
        n = warp_idx
        while n < Int32(self.T):
            k_base = (
                tile_B * Int32(self.k_stride_B)
                + n * Int32(self.k_stride_T)
                + kv_head * Int32(self.k_stride_KVH)
            )
            k_row = cute.make_tensor(k.iterator + k_base, cute.make_layout(self.HD))
            acc = Float32(0.0)
            d = lane_idx
            while d < Int32(self.HD):
                acc = acc + q_row[d].to(Float32) * k_row[d].to(Float32)
                d = d + Int32(32)
            score = cute.arch.warp_reduction(acc, operator.add) * Float32(QWEN3_5_NVFP4_ATTN_SCALE)
            if lane_idx == Int32(0):
                scores[n] = score
                if score > row_max:
                    row_max = score
            n = n + Int32(num_warps)
        if lane_idx == Int32(0):
            scratch[warp_idx] = row_max
        named_barrier_sync(Int32(2), Int32(self.threads_per_row))

        # Phase 2: reduce warp-local maxima to a row max for stable softmax.
        if warp_idx == Int32(0):
            partial_max = Float32(QWEN3_5_NVFP4_FP32_NEG_INF)
            w = lane_idx
            while w < Int32(num_warps):
                val = scratch[w]
                if val > partial_max:
                    partial_max = val
                w = w + Int32(32)
            row_max = cute.arch.warp_reduction(partial_max, cute.arch.fmax)
            if lane_idx == Int32(0):
                scratch[Int32(0)] = row_max
        named_barrier_sync(Int32(2), Int32(self.threads_per_row))

        # Phase 3: compute exp(score - row_max) partial sums per warp.
        rmax = scratch[Int32(0)]
        row_sum_part = Float32(0.0)
        n2 = warp_idx * Int32(32) + lane_idx
        while n2 < Int32(self.T):
            row_sum_part = row_sum_part + cute.math.exp(scores[n2] - rmax, fastmath=True)
            n2 = n2 + Int32(self.threads_per_row)
        row_sum_part = cute.arch.warp_reduction(row_sum_part, operator.add)
        if lane_idx == Int32(0):
            scratch[warp_idx] = row_sum_part
        named_barrier_sync(Int32(2), Int32(self.threads_per_row))

        # Phase 4: reduce the softmax denominator and store its reciprocal.
        if warp_idx == Int32(0):
            row_sum = Float32(0.0)
            w2 = lane_idx
            while w2 < Int32(num_warps):
                row_sum = row_sum + scratch[w2]
                w2 = w2 + Int32(32)
            row_sum = cute.arch.warp_reduction(row_sum, operator.add)
            if lane_idx == Int32(0):
                scratch[Int32(1)] = Float32(1.0) / row_sum
        named_barrier_sync(Int32(2), Int32(self.threads_per_row))

        # Phase 5: normalize scores in shared memory to probabilities.
        rinv = scratch[Int32(1)]
        n_prob = tidx
        while n_prob < Int32(self.T):
            scores[n_prob] = cute.math.exp(scores[n_prob] - rmax, fastmath=True) * rinv
            n_prob = n_prob + Int32(self.threads_per_row)
        named_barrier_sync(Int32(2), Int32(self.threads_per_row))

        # Phase 6: accumulate softmax(QK) V, apply sigmoid gate, and store.
        out_base = (
            tile_B * Int32(self.o_stride_B)
            + tile_QH * Int32(self.o_stride_QH)
        )
        gate_base = tile_B * Int32(self.gate_stride_B) + tile_QH * Int32(self.HD)
        out_row = cute.make_tensor(o.iterator + out_base, cute.make_layout(self.HD))
        gate_row = cute.make_tensor(gate.iterator + gate_base, cute.make_layout(self.HD))
        d_out = tidx
        while d_out < Int32(self.HD):
            acc_o = Float32(0.0)
            n3 = Int32(0)
            while n3 < Int32(self.T):
                v_base = (
                    tile_B * Int32(self.v_stride_B)
                    + n3 * Int32(self.v_stride_T)
                    + kv_head * Int32(self.v_stride_KVH)
                )
                v_row = cute.make_tensor(v.iterator + v_base, cute.make_layout(self.HD))
                acc_o = acc_o + scores[n3] * v_row[d_out].to(Float32)
                n3 = n3 + Int32(1)
            g = gate_row[d_out].to(Float32)
            sigmoid = Float32(1.0) / (Float32(1.0) + cute.math.exp(Float32(0.0) - g, fastmath=True))
            out_row[d_out] = (acc_o * sigmoid).to(self.o_dtype)
            d_out = d_out + Int32(self.threads_per_row)


class Qwen3_5StagedPartialAttentionSm120Op(Op):
    """Split-context S=1 attention with staged K/V cache blocks.

    Each tile owns one query head and one contiguous context block. The load
    phase copies the K/V block into the instruction page, then compute performs
    a numerically stable block softmax and writes fp32 partial O/LSE. A separate
    combine op reduces partials across blocks.
    """

    pipeline = PipelineSpec(page_count=1)
    reads = {
        "q": (None, ("B", "S", "QH", "HD")),
        "k": (None, ("B", "T", "KVH", "HD")),
        "v": (None, ("B", "T", "KVH", "HD")),
    }
    writes = {
        "o_partial": (cutlass.Float32, ("B", "QH", "SPLIT", "S", "HD")),
        "lse_partial": (cutlass.Float32, ("B", "QH", "SPLIT", "S")),
    }
    tile = ("B", "QH", "T")
    dynamic_dims = ("B", "T")

    @classmethod
    def schedule(
        cls,
        tile_sizes=None,
        page_size=DEFAULT_PAGE_SIZE,
        kv_group_size=1,
        context_block=16,
        **tensors,
    ):
        import torch

        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("QH", 1)
        tile_sizes.setdefault("T", context_block)
        q = tensors["q"]
        k = tensors["k"]
        B, S, QH, HD = q.shape
        T = k.shape[1]
        split = (T + tile_sizes["T"] - 1) // tile_sizes["T"]
        o_partial = torch.empty(B, QH, split, S, HD, dtype=torch.float32, device=q.device)
        lse_partial = torch.empty(B, QH, split, S, dtype=torch.float32, device=q.device)
        tensors["o_partial"] = o_partial
        tensors["lse_partial"] = lse_partial
        op = cls._schedule_single(tile_sizes=tile_sizes, **tensors)
        elem_bytes = q.element_size()
        kv_bytes = tile_sizes["T"] * HD * elem_bytes
        scores_offset = 2 * kv_bytes
        scratch_offset = scores_offset + tile_sizes["T"] * 4
        required = scratch_offset + 32 * 4
        if page_size < required:
            raise ValueError(
                "Qwen3_5StagedPartialAttentionSm120Op page_size is too small: "
                f"need at least {required} bytes for block={tile_sizes['T']} HD={HD}"
            )
        op.static_dims["page_size"] = page_size
        op.static_dims["kv_group_size"] = kv_group_size
        op.static_dims["context_block"] = tile_sizes["T"]
        op.static_dims["T"] = T
        op.static_dims["SPLIT"] = split
        op.static_dims["HD"] = HD
        op.static_dims["QH"] = QH
        op.static_dims["KVH"] = k.shape[2]
        op.static_dims["kv_bytes"] = kv_bytes
        op.static_dims["v_offset"] = kv_bytes
        op.static_dims["scores_offset"] = scores_offset
        op.static_dims["scratch_offset"] = scratch_offset
        return [op], o_partial, lse_partial

    @cute.jit
    def _k_smem(self, page_ptr):
        return cute.make_tensor(
            cute.make_ptr(self.k_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=128),
            cute.make_layout((self.tile_size_T, self.HD), stride=(self.HD, 1)),
        )

    @cute.jit
    def _v_smem(self, page_ptr):
        return cute.make_tensor(
            cute.make_ptr(
                self.v_dtype,
                page_ptr + Int32(self.v_offset),
                cute.AddressSpace.smem,
                assumed_align=128,
            ),
            cute.make_layout((self.tile_size_T, self.HD), stride=(self.HD, 1)),
        )

    @cute.jit
    def _scores_smem(self, page_ptr):
        return cute.make_tensor(
            cute.make_ptr(cutlass.Float32, page_ptr + Int32(self.scores_offset), cute.AddressSpace.smem),
            cute.make_layout(self.tile_size_T),
        )

    @cute.jit
    def _scratch_smem(self, page_ptr):
        return cute.make_tensor(
            cute.make_ptr(cutlass.Float32, page_ptr + Int32(self.scratch_offset), cute.AddressSpace.smem),
            cute.make_layout(32),
        )

    @cute.jit
    def load(self, page_ptr):
        pass

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_QH, tile_T, q, k, v, o_partial, lse_partial):
        tidx = cute.arch.thread_idx()[0]
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()
        num_warps = self.threads_per_row // 32
        start_t = tile_T * Int32(self.tile_size_T)
        scores = self._scores_smem(page_ptr)
        scratch = self._scratch_smem(page_ptr)
        kv_head = tile_QH // Int32(self.kv_group_size)
        q_base = tile_B * Int32(self.q_stride_B) + tile_QH * Int32(self.q_stride_QH)
        q_row = cute.make_tensor(q.iterator + q_base, cute.make_layout(self.HD))

        local_max = Float32(QWEN3_5_NVFP4_FP32_NEG_INF)
        local_t = warp_idx
        while local_t < Int32(self.tile_size_T):
            global_t = start_t + local_t
            acc = Float32(0.0)
            k_base = (
                tile_B * Int32(self.k_stride_B)
                + global_t * Int32(self.k_stride_T)
                + kv_head * Int32(self.k_stride_KVH)
            )
            k_row = cute.make_tensor(k.iterator + k_base, cute.make_layout(self.HD))
            d = lane_idx
            while d < Int32(self.HD):
                kval = Float32(0.0)
                if global_t < Int32(self.T):
                    kval = k_row[d].to(Float32)
                acc = acc + q_row[d].to(Float32) * kval
                d = d + Int32(32)
            score = cute.arch.warp_reduction(acc, operator.add) * Float32(QWEN3_5_NVFP4_ATTN_SCALE)
            if global_t >= Int32(self.T):
                score = Float32(QWEN3_5_NVFP4_FP32_NEG_INF)
            if lane_idx == Int32(0):
                scores[local_t] = score
                if score > local_max:
                    local_max = score
            local_t = local_t + Int32(num_warps)
        if lane_idx == Int32(0):
            scratch[warp_idx] = local_max
        named_barrier_sync(Int32(2), Int32(self.threads_per_row))

        if warp_idx == Int32(0):
            max_part = Float32(QWEN3_5_NVFP4_FP32_NEG_INF)
            w = lane_idx
            while w < Int32(num_warps):
                val = scratch[w]
                if val > max_part:
                    max_part = val
                w = w + Int32(32)
            block_max = cute.arch.warp_reduction(max_part, cute.arch.fmax)
            if lane_idx == Int32(0):
                scratch[Int32(0)] = block_max
        named_barrier_sync(Int32(2), Int32(self.threads_per_row))

        block_max = scratch[Int32(0)]
        sum_part = Float32(0.0)
        idx = tidx
        while idx < Int32(self.tile_size_T):
            global_t = start_t + idx
            prob = Float32(0.0)
            if global_t < Int32(self.T):
                prob = cute.math.exp(scores[idx] - block_max, fastmath=True)
            scores[idx] = prob
            sum_part = sum_part + prob
            idx = idx + Int32(self.threads_per_row)
        sum_part = cute.arch.warp_reduction(sum_part, operator.add)
        if lane_idx == Int32(0):
            scratch[warp_idx] = sum_part
        named_barrier_sync(Int32(2), Int32(self.threads_per_row))

        if warp_idx == Int32(0):
            block_sum = Float32(0.0)
            w2 = lane_idx
            while w2 < Int32(num_warps):
                block_sum = block_sum + scratch[w2]
                w2 = w2 + Int32(32)
            block_sum = cute.arch.warp_reduction(block_sum, operator.add)
            if lane_idx == Int32(0):
                scratch[Int32(1)] = block_sum
        named_barrier_sync(Int32(2), Int32(self.threads_per_row))

        block_sum = scratch[Int32(1)]
        inv_sum = Float32(1.0) / block_sum
        out_base = (
            tile_B * Int32(self.o_partial_stride_B)
            + tile_QH * Int32(self.o_partial_stride_QH)
            + tile_T * Int32(self.o_partial_stride_SPLIT)
        )
        out_row = cute.make_tensor(o_partial.iterator + out_base, cute.make_layout((self.S, self.HD), stride=(self.HD, 1)))
        dim_out = tidx
        while dim_out < Int32(self.HD):
            acc_o = Float32(0.0)
            n = Int32(0)
            while n < Int32(self.tile_size_T):
                global_t = start_t + n
                vval = Float32(0.0)
                if global_t < Int32(self.T):
                    v_base = (
                        tile_B * Int32(self.v_stride_B)
                        + global_t * Int32(self.v_stride_T)
                        + kv_head * Int32(self.v_stride_KVH)
                    )
                    v_row = cute.make_tensor(v.iterator + v_base, cute.make_layout(self.HD))
                    vval = v_row[dim_out].to(Float32)
                acc_o = acc_o + scores[n] * vval
                n = n + Int32(1)
            out_row[Int32(0), dim_out] = acc_o * inv_sum
            dim_out = dim_out + Int32(self.threads_per_row)

        if tidx == Int32(0):
            lse_base = (
                tile_B * Int32(self.lse_partial_stride_B)
                + tile_QH * Int32(self.lse_partial_stride_QH)
                + tile_T * Int32(self.lse_partial_stride_SPLIT)
            )
            lse_row = cute.make_tensor(lse_partial.iterator + lse_base, cute.make_layout(self.S))
            lse_row[Int32(0)] = block_max + cute.math.log(block_sum)


def schedule_qwen3_5_staged_attention_sm120(
    *,
    q,
    k,
    v,
    o,
    kv_group_size=QWEN3_5_NVFP4_KV_GROUP_SIZE,
    context_block=16,
    page_size=DEFAULT_PAGE_SIZE,
):
    partial_ops, o_partial, lse_partial = Qwen3_5StagedPartialAttentionSm120Op.schedule(
        q=q,
        k=k,
        v=v,
        kv_group_size=kv_group_size,
        context_block=context_block,
        page_size=page_size,
    )
    import torch

    lse = torch.empty(q.shape[0], q.shape[2], q.shape[1], dtype=torch.float32, device=q.device)
    combine_ops = FlashDecodingCombineBSHDOp.schedule(
        o_partial=o_partial,
        lse_partial=lse_partial,
        o=o,
        lse=lse,
    )
    return partial_ops + combine_ops, (o_partial, lse_partial, lse)


class Qwen3_5PadDecodeQueryForFlashSm120Op(Op):
    """Pad one decode query row to the 16-row MMA tile used by split attention."""

    reads = {"q": (None, ("B", "S", "QH", "HD"))}
    writes = {"q_pad": (None, ("B", "M", "QH", "HD"))}
    tile = ("B", "QH")
    dynamic_dims = ("B",)

    @classmethod
    def schedule(cls, tile_sizes=None, **tensors):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("QH", 1)
        op = cls._schedule_single(tile_sizes=tile_sizes, **tensors)
        op.static_dims["M"] = tensors["q_pad"].shape[1]
        return [op]

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_QH, q, q_pad):
        tidx = cute.arch.thread_idx()[0]
        base_q = tile_B * Int32(self.q_stride_B) + tile_QH * Int32(self.q_stride_QH)
        q_row = cute.make_tensor(q.iterator + base_q, cute.make_layout(self.HD))
        base_pad = tile_B * Int32(self.q_pad_stride_B) + tile_QH * Int32(self.q_pad_stride_QH)
        q_pad_head = cute.make_tensor(
            q_pad.iterator + base_pad,
            cute.make_layout((self.M, self.HD), stride=(self.q_pad_stride_M, 1)),
        )
        elem = tidx
        total = Int32(self.M * self.HD)
        while elem < total:
            row = elem // Int32(self.HD)
            dim = elem % Int32(self.HD)
            val = Float32(0.0).to(self.q_pad_dtype)
            if row == Int32(0):
                val = q_row[dim]
            q_pad_head[row, dim] = val
            elem = elem + Int32(self.threads_per_row)


class Qwen3_5CopyFlashDecodeRowSm120Op(Op):
    """Copy row 0 from padded split-attention output back to BSHD decode output."""

    reads = {"o_pad": (None, ("B", "M", "QH", "HD"))}
    writes = {"o": (None, ("B", "S", "QH", "HD"))}
    tile = ("B", "QH")
    dynamic_dims = ("B",)

    @classmethod
    def schedule(cls, tile_sizes=None, **tensors):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("QH", 1)
        return [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_QH, o_pad, o):
        tidx = cute.arch.thread_idx()[0]
        src_base = tile_B * Int32(self.o_pad_stride_B) + tile_QH * Int32(self.o_pad_stride_QH)
        dst_base = tile_B * Int32(self.o_stride_B) + tile_QH * Int32(self.o_stride_QH)
        src = cute.make_tensor(o_pad.iterator + src_base, cute.make_layout(self.HD))
        dst = cute.make_tensor(o.iterator + dst_base, cute.make_layout(self.HD))
        dim = tidx
        while dim < Int32(self.HD):
            dst[dim] = src[dim]
            dim = dim + Int32(self.threads_per_row)


def schedule_qwen3_5_flash_decode_attention_sm120(
    *,
    q,
    k,
    v,
    o,
    kv_group_size=QWEN3_5_NVFP4_KV_GROUP_SIZE,
    page_size=DEFAULT_PAGE_SIZE,
    num_splits=0,
):
    import torch

    q_pad = torch.empty(q.shape[0], 16, q.shape[2], q.shape[3], dtype=q.dtype, device=q.device)
    o_pad = torch.empty_like(q_pad)
    ops = []
    ops += Qwen3_5PadDecodeQueryForFlashSm120Op.schedule(q=q, q_pad=q_pad)
    flash_ops, _flash_config = flash_decoding_schedule(
        q_pad,
        k,
        v,
        o_pad,
        num_splits=num_splits,
        page_size=page_size,
        kv_group_size=kv_group_size,
    )
    ops += flash_ops
    ops += Qwen3_5CopyFlashDecodeRowSm120Op.schedule(o_pad=o_pad, o=o)
    return ops, (q_pad, o_pad)


class Qwen3_5QkvRopeCacheSm120Op(Op):
    """Postprocess normalized Q/K/V projections for one decode token."""

    reads = {
        "q_raw": (None, ("B", "S", "Q")),
        "kv_raw": (None, ("B", "S", "Q")),
        "cos": (None, ("S", "D2")),
        "sin": (None, ("S", "D2")),
    }
    writes = {
        "q": (None, ("B", "S", "Q")),
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
    def compute(self, page_ptr, tile_B, tile_S, tile_2, q_raw, kv_raw, cos, sin,
                q, k_cache, v_cache):
        tidx = cute.arch.thread_idx()[0]
        row_idx = tile_S * Int32(self.tile_size_S)
        q_raw_base = tile_B * Int32(self.q_raw_stride_B) + row_idx * Int32(self.q_raw_stride_S)
        kv_raw_base = tile_B * Int32(self.kv_raw_stride_B) + row_idx * Int32(self.kv_raw_stride_S)
        q_base = tile_B * Int32(self.q_stride_B) + row_idx * Int32(self.q_stride_S)
        cos_row = cute.make_tensor(cos.iterator + row_idx * Int32(self.cos_stride_S), cute.make_layout(self.D2))
        sin_row = cute.make_tensor(sin.iterator + row_idx * Int32(self.sin_stride_S), cute.make_layout(self.D2))
        q_raw_row = cute.make_tensor(q_raw.iterator + q_raw_base, cute.make_layout(self.Q))
        kv_raw_row = cute.make_tensor(kv_raw.iterator + kv_raw_base, cute.make_layout(self.Q))
        q_row = cute.make_tensor(q.iterator + q_base, cute.make_layout(self.Q))

        elem = tidx
        while elem < Int32(self.Q):
            head = elem // Int32(self.head_dim)
            dim = elem % Int32(self.head_dim)
            head_base = head * Int32(self.head_dim)
            if head < Int32(self.q_heads):
                if dim < Int32(self.D2):
                    low = q_raw_row[head_base + dim].to(Float32)
                    high = q_raw_row[head_base + dim + Int32(self.D2)].to(Float32)
                    c = cos_row[dim].to(Float32)
                    s = sin_row[dim].to(Float32)
                    q_row[head_base + dim] = (low * c - high * s).to(self.q_dtype)
                    q_row[head_base + dim + Int32(self.D2)] = (high * c + low * s).to(self.q_dtype)
                elif dim >= Int32(2 * self.D2):
                    q_row[elem] = q_raw_row[elem]
            elem = elem + Int32(self.threads_per_row)

        kv_dim = Int32(self.kv_heads * self.head_dim)
        kv = tidx
        while kv < kv_dim:
            head = kv // Int32(self.head_dim)
            dim = kv % Int32(self.head_dim)
            head_base = head * Int32(self.head_dim)
            k_out = kv_raw_row[kv].to(Float32)
            write_k = Int32(1)
            if dim < Int32(self.D2):
                low = kv_raw_row[head_base + dim].to(Float32)
                high = kv_raw_row[head_base + dim + Int32(self.D2)].to(Float32)
                c = cos_row[dim].to(Float32)
                s = sin_row[dim].to(Float32)
                k_base = (
                    tile_B * Int32(self.k_cache_stride_B)
                    + (row_idx + Int32(self.cache_pos)) * Int32(self.k_cache_stride_T)
                    + head * Int32(self.k_cache_stride_KVH)
                )
                k_row = cute.make_tensor(k_cache.iterator + k_base, cute.make_layout(self.HD))
                k_row[dim] = (low * c - high * s).to(self.k_cache_dtype)
                k_row[dim + Int32(self.D2)] = (high * c + low * s).to(self.k_cache_dtype)
                write_k = Int32(0)
            elif dim < Int32(2 * self.D2):
                write_k = Int32(0)
            if write_k != Int32(0):
                k_base = (
                    tile_B * Int32(self.k_cache_stride_B)
                    + (row_idx + Int32(self.cache_pos)) * Int32(self.k_cache_stride_T)
                    + head * Int32(self.k_cache_stride_KVH)
                )
                k_row = cute.make_tensor(k_cache.iterator + k_base, cute.make_layout(self.HD))
                k_row[dim] = k_out.to(self.k_cache_dtype)

            v_base = (
                tile_B * Int32(self.v_cache_stride_B)
                + (row_idx + Int32(self.cache_pos)) * Int32(self.v_cache_stride_T)
                + head * Int32(self.v_cache_stride_KVH)
            )
            v_row = cute.make_tensor(v_cache.iterator + v_base, cute.make_layout(self.HD))
            v_row[dim] = kv_raw_row[kv_dim + kv].to(self.v_cache_dtype)
            kv = kv + Int32(self.threads_per_row)


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
                ss_q * Float32(1.0 / self.head_dim) + Float32(QWEN3_5_NVFP4_EPS),
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
                ss_k * Float32(1.0 / self.head_dim) + Float32(QWEN3_5_NVFP4_EPS),
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


class Qwen3_5ApplyAttentionGateSm120Op(Op):
    """Apply the full-attention output gate before the output projection."""

    reads = {
        "gate": (None, ("B", "S", "Q")),
        "attn_out": (None, ("B", "S", "Q")),
    }
    writes = {"attn_out": (None, ("B", "S", "Q"))}
    tile = ("B", "S")
    dynamic_dims = ("B",)

    @classmethod
    def schedule(cls, tile_sizes=None, **tensors):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", 1)
        return [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_2, gate, attn_out):
        tidx = cute.arch.thread_idx()[0]
        row_idx = tile_S * Int32(self.tile_size_S)
        gate_base = tile_B * Int32(self.gate_stride_B) + row_idx * Int32(self.gate_stride_S)
        out_base = tile_B * Int32(self.attn_out_stride_B) + row_idx * Int32(self.attn_out_stride_S)
        gate_row = cute.make_tensor(gate.iterator + gate_base, cute.make_layout(self.Q))
        out_row = cute.make_tensor(attn_out.iterator + out_base, cute.make_layout(self.Q))
        elem = tidx
        while elem < Int32(self.Q):
            g = gate_row[elem].to(Float32)
            sigmoid = Float32(1.0) / (Float32(1.0) + cute.math.exp(Float32(0.0) - g, fastmath=True))
            out_row[elem] = (out_row[elem].to(Float32) * sigmoid).to(self.attn_out_dtype)
            elem = elem + Int32(self.threads_per_row)


def schedule_qwen3_5_deltanet_nvfp4_sm120(
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
    group_size=QWEN3_5_NVFP4_GROUP_SIZE,
    matvec_block=QWEN3_5_NVFP4_MATVEC_BLOCK,
    prefetch_gate_up=True,
) -> DecodeLayerScheduleSm120:
    """Schedule one Qwen3.5 DeltaNet layer with native packed NVFP4 ops.

    This is the linear-attention layer path. All dense projections are scheduled
    as Machete ops; the recurrent DeltaNet body is a single CuTe DSL op that
    updates convolution and recurrent state in place.
    """

    pfx = f"layer.{layer_idx}"
    cos = weights.get("cos")
    sin = weights.get("sin")
    if cos is None or sin is None:
        raise KeyError("weights must include cos/sin scratch tensors for RMS projection metadata")
    cos = cos[:seq_len]
    sin = sin[:seq_len]

    def qparts(name):
        qweight = weights[f"{pfx}.{name}_nvfp4"]
        return qweight.packed, qweight.scales

    qkv_packed, qkv_scales = qparts("W_qkv")
    z_packed, z_scales = qparts("W_z")
    beta_packed, beta_scales = qparts("W_beta")
    alpha_packed, alpha_scales = qparts("W_alpha")
    out_packed, out_scales = qparts("W_out")
    gate_packed, gate_scales = qparts("W_gate")
    up_packed, up_scales = qparts("W_up")
    down_packed, down_scales = qparts("W_down")

    ops = []
    ops += RmsAddNormSm120Op.schedule(
        x=x_in,
        residual_in=residual_in,
        norm_weight=weights[f"{pfx}.attn_norm"],
        residual_out=residual_out,
        y=norm_buf,
        tile_sizes={"S": seq_len},
        page_size=page_size,
        eps=QWEN3_5_NVFP4_EPS,
    )
    q0 = 0
    k0 = QWEN3_5_NVFP4_DN_QK_SIZE
    v0 = 2 * QWEN3_5_NVFP4_DN_QK_SIZE
    end = QWEN3_5_NVFP4_DN_CONV_CHANNELS
    qkv_q_buf = _last_dim_slice(qkv_buf, q0, k0)
    qkv_k_buf = _last_dim_slice(qkv_buf, k0, v0)
    qkv_v_buf = _last_dim_slice(qkv_buf, v0, end)
    q_packed, q_scales = _row_slice(qkv_packed, q0, k0), _row_slice(qkv_scales, q0, k0)
    k_packed, k_scales = _row_slice(qkv_packed, k0, v0), _row_slice(qkv_scales, k0, v0)
    v_packed, v_scales = _row_slice(qkv_packed, v0, end), _row_slice(qkv_scales, v0, end)

    ops += _schedule_nvfp4_quad_projection(
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
    )
    ops += _schedule_nvfp4_pair_projection(
        x=norm_buf,
        weights0=(beta_packed, beta_scales),
        weights1=(alpha_packed, alpha_scales),
        y0=beta_buf,
        y1=alpha_buf,
        seq_len=seq_len,
        matvec_block=matvec_block,
        page_size=page_size,
        group_size=group_size,
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
    ops += MatvecResidualNvfp4Sm120Op.schedule(
        a=dn_out_buf,
        weight_packed=out_packed,
        weight_scales=out_scales,
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
        eps=QWEN3_5_NVFP4_EPS,
        group_size=group_size,
        prefetch_nvfp4=prefetch_gate_up,
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
        qkv_packed, qkv_scales, z_packed, z_scales,
        qkv_q_buf, qkv_k_buf, qkv_v_buf,
        q_packed, q_scales, k_packed, k_scales, v_packed, v_scales,
        beta_packed, beta_scales, alpha_packed, alpha_scales,
        out_packed, out_scales, gate_packed, gate_scales,
        up_packed, up_scales, down_packed, down_scales,
    ]
    return DecodeLayerScheduleSm120(ops=ops, attention_config=None, keep_alive=keep)


def schedule_qwen3_5_full_attention_nvfp4_sm120(
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
    group_size=QWEN3_5_NVFP4_GROUP_SIZE,
    fa_num_splits=0,
    use_flash_attention=False,
    matvec_block=QWEN3_5_NVFP4_MATVEC_BLOCK,
    prefetch_gate_up=True,
) -> DecodeLayerScheduleSm120:
    """Schedule one Qwen3.5 full-attention layer with packed NVFP4 weights.

    Expected packed-weight keys for ``layer_idx``:
    ``W_q_nvfp4``, ``W_k_nvfp4``, ``W_v_nvfp4``, ``W_o_nvfp4``,
    ``W_gate_nvfp4``, ``W_up_nvfp4``, and ``W_down_nvfp4``.
    """

    if seq_len == 1:
        import torch

        pfx = f"layer.{layer_idx}"
        cos = weights["cos"][cache_pos : cache_pos + seq_len]
        sin = weights["sin"][cache_pos : cache_pos + seq_len]
        if q_raw_buf is None:
            q_raw_buf = torch.empty(batch, seq_len, QWEN3_5_NVFP4_Q_RAW_DIM, dtype=q_buf.dtype, device=q_buf.device)
        if kv_raw_buf is None:
            kv_raw_buf = torch.empty(batch, seq_len, 2 * QWEN3_5_NVFP4_KV_DIM, dtype=q_buf.dtype, device=q_buf.device)
        if q_gate_buf is None:
            q_gate_buf = torch.empty_like(q_buf)
        q_4d = q_buf.view(batch, seq_len, QWEN3_5_NVFP4_NUM_Q_HEADS, QWEN3_5_NVFP4_HEAD_DIM)
        k_window = k_cache[:, : cache_pos + seq_len]
        v_window = v_cache[:, : cache_pos + seq_len]
        o_4d = attn_out_buf.view(batch, seq_len, QWEN3_5_NVFP4_NUM_Q_HEADS, QWEN3_5_NVFP4_HEAD_DIM)

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
        k_raw_buf = kv_raw_buf[:, :, :QWEN3_5_NVFP4_KV_DIM]
        v_raw_buf = kv_raw_buf[:, :, QWEN3_5_NVFP4_KV_DIM : 2 * QWEN3_5_NVFP4_KV_DIM]

        ops += RmsAddNormSm120Op.schedule(
            x=x_in,
            residual_in=residual_in,
            norm_weight=weights[f"{pfx}.attn_norm"],
            residual_out=residual_out,
            y=norm_buf,
            tile_sizes={"S": seq_len},
            page_size=page_size,
            eps=QWEN3_5_NVFP4_EPS,
        )
        ops += MatvecNvfp4Sm120Op.schedule(
            a=norm_buf,
            weight_packed=q_packed,
            weight_scales=q_scales,
            y=q_raw_buf,
            tile_sizes={"S": seq_len, "O": matvec_block},
            page_size=page_size,
            group_size=group_size,
        )
        ops += MatvecNvfp4Sm120Op.schedule(
            a=norm_buf,
            weight_packed=k_packed,
            weight_scales=k_scales,
            y=k_raw_buf,
            tile_sizes={"S": seq_len, "O": 16},
            page_size=page_size,
            group_size=group_size,
        )
        ops += MatvecNvfp4Sm120Op.schedule(
            a=norm_buf,
            weight_packed=v_packed,
            weight_scales=v_scales,
            y=v_raw_buf,
            tile_sizes={"S": seq_len, "O": 16},
            page_size=page_size,
            group_size=group_size,
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
        attention_keep = []
        if use_flash_attention:
            attention_ops, attention_keep = schedule_qwen3_5_flash_decode_attention_sm120(
                q=q_4d,
                k=k_window,
                v=v_window,
                o=o_4d,
                kv_group_size=QWEN3_5_NVFP4_KV_GROUP_SIZE,
                page_size=page_size,
                num_splits=fa_num_splits,
            )
            ops += attention_ops
        else:
            ops += Qwen3_5GatedSingleTokenAttentionSm120Op.schedule(
                q=q_4d,
                k=k_window,
                v=v_window,
                gate=q_gate_buf,
                o=o_4d,
                kv_group_size=QWEN3_5_NVFP4_KV_GROUP_SIZE,
                page_size=page_size,
            )
        if use_flash_attention:
            ops += Qwen3_5ApplyAttentionGateSm120Op.schedule(
                gate=q_gate_buf,
                attn_out=attn_out_buf,
                tile_sizes={"S": seq_len},
            )
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
            eps=QWEN3_5_NVFP4_EPS,
            group_size=group_size,
            prefetch_nvfp4=prefetch_gate_up,
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
            q_raw_buf, kv_raw_buf, q_gate_buf, k_raw_buf, v_raw_buf, *attention_keep,
            q_packed, q_scales, k_packed, k_scales, v_packed, v_scales,
            o_packed, o_scales, gate_packed, gate_scales, up_packed, up_scales,
            down_packed, down_scales,
        ]
        return DecodeLayerScheduleSm120(ops=ops, attention_config=None, keep_alive=keep)

    return schedule_decode_layer_nvfp4_sm120(
        layer_idx=layer_idx,
        batch=batch,
        seq_len=seq_len,
        cache_pos=cache_pos,
        weights=weights,
        k_cache=k_cache,
        v_cache=v_cache,
        x_in=x_in,
        residual_in=residual_in,
        x_out=x_out,
        residual_out=residual_out,
        q_buf=q_buf,
        attn_out_buf=attn_out_buf,
        mlp_h_buf=mlp_h_buf,
        page_size=page_size,
        eps=QWEN3_5_NVFP4_EPS,
        group_size=group_size,
        fa_num_splits=fa_num_splits,
        hidden_size=QWEN3_5_NVFP4_HIDDEN,
        intermediate_size=QWEN3_5_NVFP4_INTERMEDIATE,
        num_q_heads=QWEN3_5_NVFP4_NUM_Q_HEADS,
        num_kv_heads=QWEN3_5_NVFP4_NUM_KV_HEADS,
        head_dim=QWEN3_5_NVFP4_HEAD_DIM,
        kv_group_size=QWEN3_5_NVFP4_KV_GROUP_SIZE,
        matvec_block=matvec_block,
    )


def schedule_qwen3_5_final_nvfp4_sm120(
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
    group_size=QWEN3_5_NVFP4_GROUP_SIZE,
):
    """Schedule final residual plus packed NVFP4 LM head for Qwen3.5."""

    return schedule_final_nvfp4_sm120(
        x=x,
        residual_in=residual_in,
        residual_out=residual_out,
        final_norm=final_norm,
        lm_head_nvfp4=lm_head_nvfp4,
        logits=logits,
        top_values=top_values,
        top_indices=top_indices,
        top_partial_values=top_partial_values,
        top_partial_indices=top_partial_indices,
        seq_len=seq_len,
        page_size=page_size,
        eps=QWEN3_5_NVFP4_EPS,
        group_size=group_size,
    )


def _layer_resource(resource, layer_idx, slot=None):
    if isinstance(resource, (list, tuple)):
        return resource[layer_idx if slot is None else slot]
    if hasattr(resource, "dim") and resource.dim() > 0:
        if slot is not None and resource.shape[0] == QWEN3_5_LAYER_TYPES.count("linear_attention"):
            return resource[slot]
        if resource.shape[0] == QWEN3_5_NVFP4_NUM_LAYERS:
            return resource[layer_idx]
    return resource


def schedule_qwen3_5_nvfp4_decode_sm120(
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
    lm_head_nvfp4=None,
    logits=None,
    top_values=None,
    top_indices=None,
    top_partial_values=None,
    top_partial_indices=None,
    page_size=DEFAULT_PAGE_SIZE,
    group_size=QWEN3_5_NVFP4_GROUP_SIZE,
    fa_num_splits=0,
    use_flash_attention=False,
    matvec_block=QWEN3_5_NVFP4_MATVEC_BLOCK,
    prefetch_gate_up=True,
):
    """Build the full 24-layer Qwen3.5 NVFP4 decode schedule.

    ``x_buffers`` and ``residual_buffers`` must contain one entry per layer
    boundary: index 0 is model input state and index 24 is final layer output.
    Scratch tensors may be single reusable tensors or per-layer/per-linear-layer
    lists/tensors on their leading dimension.
    """

    if len(x_buffers) < QWEN3_5_NVFP4_NUM_LAYERS + 1:
        raise ValueError("x_buffers must contain 25 layer-boundary tensors")
    if len(residual_buffers) < QWEN3_5_NVFP4_NUM_LAYERS + 1:
        raise ValueError("residual_buffers must contain 25 layer-boundary tensors")

    ops = []
    keep = []
    attention_configs = []
    linear_slot = 0
    for layer_idx, layer_type in enumerate(QWEN3_5_LAYER_TYPES):
        if layer_type == "full_attention":
            layer = schedule_qwen3_5_full_attention_nvfp4_sm120(
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
                use_flash_attention=use_flash_attention,
                matvec_block=matvec_block,
                prefetch_gate_up=prefetch_gate_up,
            )
        elif layer_type == "linear_attention":
            layer = schedule_qwen3_5_deltanet_nvfp4_sm120(
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
                prefetch_gate_up=prefetch_gate_up,
            )
            linear_slot += 1
        else:
            raise ValueError(f"unknown Qwen3.5 layer type {layer_type!r}")
        ops.extend(layer.ops)
        keep.extend(layer.keep_alive)
        attention_configs.append(layer.attention_config)

    if final_norm is not None:
        final_ops = schedule_qwen3_5_final_nvfp4_sm120(
            x=x_buffers[QWEN3_5_NVFP4_NUM_LAYERS],
            residual_in=residual_buffers[QWEN3_5_NVFP4_NUM_LAYERS],
            residual_out=residual_buffers[QWEN3_5_NVFP4_NUM_LAYERS],
            final_norm=final_norm,
            lm_head_nvfp4=lm_head_nvfp4,
            logits=logits,
            top_values=top_values,
            top_indices=top_indices,
            top_partial_values=top_partial_values,
            top_partial_indices=top_partial_indices,
            seq_len=seq_len,
            page_size=page_size,
            group_size=group_size,
        )
        ops.extend(final_ops)

    return DecodeLayerScheduleSm120(
        ops=ops,
        attention_config=attention_configs,
        keep_alive=keep,
    )


__all__ = [
    "Qwen3_5DeltaNetCoreSm120Op",
    "Qwen3_5ApplyAttentionGateSm120Op",
    "Qwen3_5QGateRopeCacheSm120Op",
    "Qwen3_5GatedSingleTokenAttentionSm120Op",
    "Qwen3_5QkvRopeCacheSm120Op",
    "Qwen3_5SingleTokenAttentionSm120Op",
    "QWEN3_5_NVFP4_DECODE_S",
    "QWEN3_5_NVFP4_DN_CONV_CHANNELS",
    "QWEN3_5_NVFP4_DN_CONV_KERNEL",
    "QWEN3_5_NVFP4_DN_NUM_HEADS",
    "QWEN3_5_NVFP4_DN_VALUE_DIM",
    "QWEN3_5_NVFP4_DN_V_SIZE",
    "QWEN3_5_NVFP4_EPS",
    "QWEN3_5_NVFP4_HEAD_DIM",
    "QWEN3_5_NVFP4_HIDDEN",
    "QWEN3_5_NVFP4_INTERMEDIATE",
    "QWEN3_5_NVFP4_KV_DIM",
    "QWEN3_5_NVFP4_KV_GROUP_SIZE",
    "QWEN3_5_LAYER_TYPES",
    "QWEN3_5_NVFP4_NUM_KV_HEADS",
    "QWEN3_5_NVFP4_NUM_LAYERS",
    "QWEN3_5_NVFP4_NUM_Q_HEADS",
    "QWEN3_5_NVFP4_Q_DIM",
    "QWEN3_5_NVFP4_Q_RAW_DIM",
    "QWEN3_5_NVFP4_ROTARY_D2",
    "QWEN3_5_NVFP4_VOCAB",
    "schedule_qwen3_5_final_nvfp4_sm120",
    "schedule_qwen3_5_deltanet_nvfp4_sm120",
    "schedule_qwen3_5_full_attention_nvfp4_sm120",
    "schedule_qwen3_5_nvfp4_decode_sm120",
]
