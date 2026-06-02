# Copyright (c) 2025, Machete Authors
"""GQA flash-decoding attention for Qwen3.5 MXFP4 single-token decode (BMHD).

The decode path treats the gqa_ratio query heads of a KV head as the kernel's
effective batch and also splits the KV sequence across CTAs (flash-decoding), so

  * K/V for a KV head are streamed ONCE and reused by all gqa_ratio query heads
    (gqa_ratio=4x less KV traffic), and
  * tiles = KVH * num_splits fills the GPU (occupancy), with a cheap combine.

All tensors are BMHD: q/o ``(B, M, QH, HD)``, k/v cache ``(B, T, KVH, HD)``.

Two ops:
  * :class:`Qwen3_5GqaAttnSplitSm120Op` (B x KVH x SPLIT tiles) -> per-(qhead,split)
    partial max / exp-sum / unnormalized output.
  * :class:`Qwen3_5GqaAttnCombineSm120Op` (B x QH tiles) -> merge splits, sigmoid
    gate, write o.
"""
from __future__ import annotations

import math
import operator

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32

from machete.megakernel.interpreter import named_barrier_sync
from machete.megakernel.ops import DEFAULT_PAGE_SIZE, Op
from machete.kernels.qwen_3_5.mxfp4_ops import (
    QWEN3_5_MXFP4_ATTN_SCALE,
    QWEN3_5_MXFP4_FP32_NEG_INF,
    QWEN3_5_MXFP4_KV_GROUP_SIZE,
)


class Qwen3_5GqaAttnSplitSm120Op(Op):
    """Split-KV, GQA-grouped QK^T·softmax·V producing per-split partials.

    One tile = (KV head, KV-sequence split). It streams that KV chunk once and
    computes all ``gqa_ratio`` query heads of the group against it (two-pass
    softmax within the chunk), writing the chunk max, exp-sum and the
    *unnormalized* weighted V for each query head / split."""

    framework_owned_ranges = True
    reads = {
        "q": (None, ("B", "M", "QH", "HD")),
        "k": (None, ("B", "T", "KVH", "HD")),
        "v": (None, ("B", "T", "KVH", "HD")),
    }
    writes = {
        "m_part": (cutlass.Float32, ("B", "QH", "SPLIT")),
        "l_part": (cutlass.Float32, ("B", "QH", "SPLIT")),
        "o_part": (cutlass.Float32, ("B", "QH", "SPLIT", "HD")),
    }
    tile = ("B", "KVH", "SPLIT")
    dynamic_dims = ("B",)

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE,
                 kv_group_size=QWEN3_5_MXFP4_KV_GROUP_SIZE, num_splits=0, **tensors):
        T = int(tensors["k"].shape[1])
        gqa = int(kv_group_size)
        chunk = (T + num_splits - 1) // num_splits if num_splits > 0 else 0
        if num_splits <= 0:  # should be resolved by the schedule helper
            num_splits = 1
            chunk = T
        ts = dict(tile_sizes or {})
        ts.setdefault("B", 1)
        ts.setdefault("KVH", 1)
        ts.setdefault("SPLIT", 1)
        op = cls._schedule_single(tile_sizes=ts, **tensors)
        op.static_dims["page_size"] = page_size
        op.static_dims["T"] = T
        op.static_dims["HD"] = int(tensors["k"].shape[3])
        op.static_dims["QH"] = int(tensors["q"].shape[2])
        op.static_dims["KVH"] = int(tensors["k"].shape[2])
        op.static_dims["SPLIT"] = num_splits
        op.static_dims["kv_group_size"] = gqa
        op.static_dims["chunk"] = chunk
        return [op]

    # ---- smem layout: scores[gqa, chunk] | scratch[gqa, 32] | cmax[gqa] | lsum[gqa]
    @cute.jit
    def _scores_ptr(self, page_ptr):
        return cute.make_ptr(cutlass.Float32, page_ptr, cute.AddressSpace.smem)

    @cute.jit
    def _scratch_ptr(self, page_ptr):
        off = Int32(self.kv_group_size * self.chunk * 4)
        return cute.make_ptr(cutlass.Float32, page_ptr + off, cute.AddressSpace.smem)

    @cute.jit
    def _cmax_ptr(self, page_ptr):
        off = Int32(self.kv_group_size * self.chunk * 4 + self.kv_group_size * 32 * 4)
        return cute.make_ptr(cutlass.Float32, page_ptr + off, cute.AddressSpace.smem)

    @cute.jit
    def _lsum_ptr(self, page_ptr):
        off = Int32(self.kv_group_size * self.chunk * 4 + self.kv_group_size * 32 * 4
                    + self.kv_group_size * 4)
        return cute.make_ptr(cutlass.Float32, page_ptr + off, cute.AddressSpace.smem)



    @cute.jit
    def compute(self, page_ptr, tile_B, tile_KVH, tile_SPLIT, q, k, v, m_part, l_part, o_part):
        tidx = cute.arch.thread_idx()[0]
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()
        num_warps = self.threads_per_row // 32
        GQA = self.kv_group_size
        HD = self.HD
        chunk = self.chunk
        kvh = tile_KVH
        split = tile_SPLIT

        scores = cute.make_tensor(self._scores_ptr(page_ptr),
                                  cute.make_layout((GQA, chunk), stride=(chunk, 1)))
        scratch = cute.make_tensor(self._scratch_ptr(page_ptr),
                                   cute.make_layout((GQA, 32), stride=(32, 1)))
        cmax = cute.make_tensor(self._cmax_ptr(page_ptr), cute.make_layout(GQA))
        lsum = cute.make_tensor(self._lsum_ptr(page_ptr), cute.make_layout(GQA))

        n_start = split * Int32(chunk)
        n_end = n_start + Int32(chunk)
        if n_end > Int32(self.T):
            n_end = Int32(self.T)

        # --- Phase 1: QK scores for the chunk; every warp does all GQA heads ---
        local_max = [Float32(QWEN3_5_MXFP4_FP32_NEG_INF) for _ in range(GQA)]
        n = n_start + warp_idx
        while n < n_end:
            k_base = tile_B * Int32(self.k_stride_B) + n * Int32(self.k_stride_T) + kvh * Int32(self.k_stride_KVH)
            k_row = cute.make_tensor(k.iterator + k_base, cute.make_layout(HD))
            for g in cutlass.range_constexpr(GQA):
                qh = kvh * Int32(GQA) + Int32(g)
                q_base = tile_B * Int32(self.q_stride_B) + qh * Int32(self.q_stride_QH)
                q_row = cute.make_tensor(q.iterator + q_base, cute.make_layout(HD))
                acc = Float32(0.0)
                d = lane_idx
                while d < Int32(HD):
                    acc = acc + q_row[d].to(Float32) * k_row[d].to(Float32)
                    d = d + Int32(32)
                score = cute.arch.warp_reduction(acc, operator.add) * Float32(QWEN3_5_MXFP4_ATTN_SCALE)
                if lane_idx == Int32(0):
                    scores[Int32(g), n - n_start] = score
                local_max[g] = cute.arch.fmax(local_max[g], score)
            n = n + Int32(num_warps)
        for g in cutlass.range_constexpr(GQA):
            if lane_idx == Int32(0):
                scratch[Int32(g), warp_idx] = local_max[g]
        named_barrier_sync(Int32(2), Int32(self.threads_per_row))

        # --- Phase 2: per-head chunk max (warp 0) ---
        if warp_idx == Int32(0):
            for g in cutlass.range_constexpr(GQA):
                pm = Float32(QWEN3_5_MXFP4_FP32_NEG_INF)
                w = lane_idx
                while w < Int32(num_warps):
                    pm = cute.arch.fmax(pm, scratch[Int32(g), w])
                    w = w + Int32(32)
                pm = cute.arch.warp_reduction(pm, cute.arch.fmax)
                if lane_idx == Int32(0):
                    cmax[Int32(g)] = pm
        named_barrier_sync(Int32(2), Int32(self.threads_per_row))

        # --- Phase 3: exp(score - max), store prob back, partial sums ---
        chunk_len = n_end - n_start
        psum = [Float32(0.0) for _ in range(GQA)]
        key = tidx
        while key < chunk_len:
            for g in cutlass.range_constexpr(GQA):
                prob = cute.math.exp(scores[Int32(g), key] - cmax[Int32(g)], fastmath=True)
                scores[Int32(g), key] = prob
                psum[g] = psum[g] + prob
            key = key + Int32(self.threads_per_row)
        for g in cutlass.range_constexpr(GQA):
            ps = cute.arch.warp_reduction(psum[g], operator.add)
            if lane_idx == Int32(0):
                scratch[Int32(g), warp_idx] = ps
        named_barrier_sync(Int32(2), Int32(self.threads_per_row))
        if warp_idx == Int32(0):
            for g in cutlass.range_constexpr(GQA):
                s = Float32(0.0)
                w = lane_idx
                while w < Int32(num_warps):
                    s = s + scratch[Int32(g), w]
                    w = w + Int32(32)
                s = cute.arch.warp_reduction(s, operator.add)
                if lane_idx == Int32(0):
                    lsum[Int32(g)] = s
        named_barrier_sync(Int32(2), Int32(self.threads_per_row))

        # --- Phase 4: PV (threads own HD positions; v read once / key, reused by GQA) ---
        d_out = tidx
        while d_out < Int32(HD):
            acc_o = [Float32(0.0) for _ in range(GQA)]
            key = Int32(0)
            while key < chunk_len:
                n3 = n_start + key
                v_base = tile_B * Int32(self.v_stride_B) + n3 * Int32(self.v_stride_T) + kvh * Int32(self.v_stride_KVH)
                v_row = cute.make_tensor(v.iterator + v_base, cute.make_layout(HD))
                vd = v_row[d_out].to(Float32)
                for g in cutlass.range_constexpr(GQA):
                    acc_o[g] = acc_o[g] + scores[Int32(g), key] * vd
                key = key + Int32(1)
            for g in cutlass.range_constexpr(GQA):
                qh = kvh * Int32(GQA) + Int32(g)
                op_base = (tile_B * Int32(self.o_part_stride_B) + qh * Int32(self.o_part_stride_QH)
                           + split * Int32(self.o_part_stride_SPLIT))
                o_row = cute.make_tensor(o_part.iterator + op_base, cute.make_layout(HD))
                o_row[d_out] = acc_o[g]
            d_out = d_out + Int32(self.threads_per_row)

        # --- write m_part / l_part (one thread per head) ---
        if tidx < Int32(GQA):
            qh = kvh * Int32(GQA) + tidx
            base = tile_B * Int32(self.m_part_stride_B) + qh * Int32(self.m_part_stride_QH) + split * Int32(self.m_part_stride_SPLIT)
            cute.make_tensor(m_part.iterator + base, cute.make_layout(1))[Int32(0)] = cmax[tidx]
            baseL = tile_B * Int32(self.l_part_stride_B) + qh * Int32(self.l_part_stride_QH) + split * Int32(self.l_part_stride_SPLIT)
            cute.make_tensor(l_part.iterator + baseL, cute.make_layout(1))[Int32(0)] = lsum[tidx]


class Qwen3_5GqaAttnCombineSm120Op(Op):
    """Merge the per-split partials for each query head, sigmoid-gate, write o."""

    framework_owned_ranges = True
    reads = {
        "m_part": (cutlass.Float32, ("B", "QH", "SPLIT")),
        "l_part": (cutlass.Float32, ("B", "QH", "SPLIT")),
        "o_part": (cutlass.Float32, ("B", "QH", "SPLIT", "HD")),
        "gate": (None, ("B", "M", "Q")),
    }
    writes = {"o": (None, ("B", "M", "QH", "HD"))}
    tile = ("B", "QH")
    dynamic_dims = ("B",)

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, num_splits=1, **tensors):
        ts = dict(tile_sizes or {})
        ts.setdefault("B", 1)
        ts.setdefault("QH", 1)
        op = cls._schedule_single(tile_sizes=ts, **tensors)
        op.static_dims["page_size"] = page_size
        op.static_dims["SPLIT"] = num_splits
        op.static_dims["HD"] = int(tensors["o"].shape[3])
        op.static_dims["QH"] = int(tensors["o"].shape[2])
        return [op]

    @cute.jit
    def _corr_ptr(self, page_ptr):
        return cute.make_ptr(cutlass.Float32, page_ptr, cute.AddressSpace.smem)

    @cute.jit
    def _ginv_ptr(self, page_ptr):
        return cute.make_ptr(cutlass.Float32, page_ptr + Int32(self.SPLIT * 4), cute.AddressSpace.smem)

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_QH, m_part, l_part, o_part, gate, o):
        tidx = cute.arch.thread_idx()[0]
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()
        HD = self.HD
        S = self.SPLIT
        qh = tile_QH

        corr = cute.make_tensor(self._corr_ptr(page_ptr), cute.make_layout(S))
        ginv = cute.make_tensor(self._ginv_ptr(page_ptr), cute.make_layout(1))

        m_base = tile_B * Int32(self.m_part_stride_B) + qh * Int32(self.m_part_stride_QH)
        m_row = cute.make_tensor(m_part.iterator + m_base, cute.make_layout(S))
        l_base = tile_B * Int32(self.l_part_stride_B) + qh * Int32(self.l_part_stride_QH)
        l_row = cute.make_tensor(l_part.iterator + l_base, cute.make_layout(S))

        # --- Phase A: global max, correction factors, denominator (warp 0) ---
        if warp_idx == Int32(0):
            gmax = Float32(QWEN3_5_MXFP4_FP32_NEG_INF)
            s = lane_idx
            while s < Int32(S):
                gmax = cute.arch.fmax(gmax, m_row[s])
                s = s + Int32(32)
            gmax = cute.arch.warp_reduction(gmax, cute.arch.fmax)
            gl = Float32(0.0)
            s = lane_idx
            while s < Int32(S):
                c = cute.math.exp(m_row[s] - gmax, fastmath=True)
                corr[s] = c
                gl = gl + l_row[s] * c
                s = s + Int32(32)
            gl = cute.arch.warp_reduction(gl, operator.add)
            if lane_idx == Int32(0):
                ginv[Int32(0)] = Float32(1.0) / gl
        named_barrier_sync(Int32(2), Int32(self.threads_per_row))

        rinv = ginv[Int32(0)]
        gate_base = tile_B * Int32(self.gate_stride_B) + qh * Int32(HD)
        gate_row = cute.make_tensor(gate.iterator + gate_base, cute.make_layout(HD))
        out_base = tile_B * Int32(self.o_stride_B) + qh * Int32(self.o_stride_QH)
        out_row = cute.make_tensor(o.iterator + out_base, cute.make_layout(HD))

        # --- Phase B: weighted-sum over splits, sigmoid gate, write ---
        d = tidx
        while d < Int32(HD):
            acc = Float32(0.0)
            s = Int32(0)
            while s < Int32(S):
                op_base = (tile_B * Int32(self.o_part_stride_B) + qh * Int32(self.o_part_stride_QH)
                           + s * Int32(self.o_part_stride_SPLIT))
                o_row = cute.make_tensor(o_part.iterator + op_base, cute.make_layout(HD))
                acc = acc + o_row[d] * corr[s]
                s = s + Int32(1)
            gval = gate_row[d].to(Float32)
            sig = Float32(1.0) / (Float32(1.0) + cute.math.exp(Float32(0.0) - gval, fastmath=True))
            out_row[d] = (acc * rinv * sig).to(self.o_dtype)
            d = d + Int32(self.threads_per_row)


def _auto_num_splits(T, kvh, device=None, min_chunk=None, max_chunk=128, target_tiles=None):
    """Pick num_splits so the split op runs in ~one full SM wave (tiles == KVH*splits
    ≈ num_SMs). Over-splitting into a 2nd partial wave was measured ~24% slower at
    T=2048 (wave-quantization tail); under-splitting starves the GPU. ``max_chunk``
    keeps each tile's KV chunk small enough to stage into a 32KB smem page."""
    import torch
    if min_chunk is None:
        min_chunk = 8 if T <= 256 else 16
    if target_tiles is None:
        target_tiles = torch.cuda.get_device_properties(device or 0).multi_processor_count  # ~1 wave
    per_kvh = max(1, target_tiles // max(1, kvh))
    by_min = max(1, T // min_chunk)                # don't make chunks tinier than min_chunk
    by_max = (T + max_chunk - 1) // max_chunk      # don't make chunks bigger than max_chunk (smem fit)
    return max(by_max, min(per_kvh, by_min))


def schedule_qwen3_5_gqa_attention(q, k, v, gate, o, *, page_size=DEFAULT_PAGE_SIZE,
                                   kv_group_size=QWEN3_5_MXFP4_KV_GROUP_SIZE, num_splits=0):
    """Schedule the split + combine ops, allocating partial buffers.

    Returns ``(ops, keep_alive)``. All tensors are BMHD.
    """
    import torch
    B, M, QH, HD = q.shape
    T = k.shape[1]
    KVH = k.shape[2]
    if num_splits <= 0:
        num_splits = _auto_num_splits(T, KVH, device=q.device)
    m_part = torch.empty(B, QH, num_splits, device=q.device, dtype=torch.float32)
    l_part = torch.empty(B, QH, num_splits, device=q.device, dtype=torch.float32)
    o_part = torch.empty(B, QH, num_splits, HD, device=q.device, dtype=torch.float32)

    split_ops = Qwen3_5GqaAttnSplitSm120Op.schedule(
        q=q, k=k, v=v, m_part=m_part, l_part=l_part, o_part=o_part,
        page_size=page_size, kv_group_size=kv_group_size, num_splits=num_splits)
    combine_ops = Qwen3_5GqaAttnCombineSm120Op.schedule(
        m_part=m_part, l_part=l_part, o_part=o_part, gate=gate, o=o,
        page_size=page_size, num_splits=num_splits)
    return split_ops + combine_ops, [m_part, l_part, o_part]
