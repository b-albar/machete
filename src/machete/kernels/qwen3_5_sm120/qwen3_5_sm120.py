# Copyright (c) 2025, Machete Authors
"""Qwen 3.5 decode-oriented SM120 megakernel ops.

This module is intentionally Qwen-shaped instead of reusing the Llama-1B
decode matvec schedule directly.  Qwen 3.5 applies attention RMS before the
Q/K/V projections, then per-head Q/K RMSNorm and partial RoPE after projection.
The K/V cache is native BSHD, with K normalized/rotated before cache storage and
V copied from the packed KV scratch buffer.
"""

import operator
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32

from machete.kernels.gemm import GemmOp
from machete.kernels.qknorm_rope import QKNormRopeOp as _BaseQKNormRopeOp
from machete.kernels.qknorm_rope.qknorm_rope import CopyBulkS2GOp, group_bulk_copy_modes
from machete.megakernel.ops import DEFAULT_PAGE_SIZE, Op, config_dim_i32


QWEN3_5_NUM_LAYERS = 36
QWEN3_5_HIDDEN = 1024
QWEN3_5_INTERMEDIATE = 3584
QWEN3_5_NUM_Q_HEADS = 8
QWEN3_5_NUM_KV_HEADS = 2
QWEN3_5_HEAD_DIM = 256
QWEN3_5_ROTARY_DIM = 64
QWEN3_5_ROTARY_D2 = QWEN3_5_ROTARY_DIM // 2
QWEN3_5_Q_DIM = QWEN3_5_NUM_Q_HEADS * QWEN3_5_HEAD_DIM
QWEN3_5_KV_DIM = QWEN3_5_NUM_KV_HEADS * QWEN3_5_HEAD_DIM
QWEN3_5_KV_GROUP_SIZE = QWEN3_5_NUM_Q_HEADS // QWEN3_5_NUM_KV_HEADS
QWEN3_5_EPS = 1e-6
QWEN3_5_VOCAB = 151936
QWEN3_5_DECODE_S = 16
QWEN3_5_GATE_UP_TILE = {"S": 16, "N": 128, "K": 32}
QWEN3_5_LM_HEAD_TILE_V = 128
QWEN3_5_REDUCTION_DIM_PER_WARP = 512


class Qwen3_5LmHeadGemmSm120Op(GemmOp):
    """Separate GEMM handler family for the Qwen final lm_head projection."""


class Qwen3_5FinalRmsLmHeadSm120Op(Op):
    """Qwen final residual-add RMSNorm + LM-head decode projection.

    This is the coarse final instruction used by the Qwen SM120 decode path.
    It mirrors Hazy's ``rms_lm_head`` shape: the final hidden state is
    materialized, RMS-normalized, and projected to logits in one scheduled op.
    The current CuTe implementation uses a direct warp-reduced matvec over a
    larger vocab tile; the public scheduling contract is intentionally the
    Hazy-style single final instruction so the internals can be upgraded to a
    TMA weight-streaming pipeline without changing the Qwen decode schedule.
    """

    reads = {
        "x": (None, ("B", "S", "K")),
        "residual_in": (None, ("B", "S", "K")),
        "norm_weight": (None, ("K",)),
        "weight": (None, ("V", "K")),
    }
    writes = {
        "residual_out": (None, ("B", "S", "K")),
        "logits": (None, ("B", "S", "V")),
    }
    tile = ("B", "S", "V")
    dynamic_dims = ("B",)

    def __init__(self, **config):
        super().__init__(**config)
        if self.x_dtype not in (cutlass.Float16, cutlass.BFloat16):
            raise ValueError("Qwen final decode op requires fp16/bf16 activations")
        self.eps = getattr(self, "eps", QWEN3_5_EPS)
        self.reduction_tile_K = getattr(
            self, "reduction_tile_K", min(QWEN3_5_REDUCTION_DIM_PER_WARP, self.K)
        )

    @classmethod
    def kernel_config(cls, ops):
        from machete.megakernel import MegakernelConfig
        from machete.megakernel.megakernel import NUM_DMA_WARPS

        page_size = max(op.static_dims.get("page_size", DEFAULT_PAGE_SIZE) for op in ops)
        reduction_tile = ops[0].static_dims.get(
            "reduction_tile_K", QWEN3_5_REDUCTION_DIM_PER_WARP
        )
        k_dim = ops[0].static_dims.get("K", QWEN3_5_HIDDEN)
        consumer_warps = max(1, (k_dim + reduction_tile - 1) // reduction_tile)
        return MegakernelConfig(
            threads_per_block=(consumer_warps + NUM_DMA_WARPS) * 32,
            page_size=page_size,
            mma_reg_count=96,
        )

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, eps=QWEN3_5_EPS, **tensors):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", QWEN3_5_DECODE_S)
        tile_sizes.setdefault("V", QWEN3_5_LM_HEAD_TILE_V)
        op = cls._schedule_single(tile_sizes=tile_sizes, **tensors)
        op.static_dims["page_size"] = page_size
        op.static_dims["eps"] = eps
        op.static_dims["reduction_tile_K"] = min(
            QWEN3_5_REDUCTION_DIM_PER_WARP, tensors["x"].shape[-1]
        )
        return [op]

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
                    val = x_row[elem].to(Float32) + res_row[elem].to(Float32)
                    sum_sq = sum_sq + val * val
                    out_row[elem] = val.to(self.residual_out_dtype)
                elem = elem + Int32(32)
            k_base = k_base + Int32(self.reduction_tile_K)
        total = cute.arch.warp_reduction(sum_sq, operator.add)
        return cute.math.rsqrt(total * Float32(1.0 / self.K) + Float32(self.eps), fastmath=True)

    @cute.jit
    def _dot_vocab(self, tile_B, row_idx, vocab_idx, rstd, residual_out, norm_weight, weight):
        lane_idx = cute.arch.lane_idx()
        out_base = tile_B * Int32(self.residual_out_stride_B) + row_idx * Int32(self.residual_out_stride_S)
        w_base = vocab_idx * Int32(self.weight_stride_V)
        acc = Float32(0.0)
        k_base = Int32(0)
        while k_base < Int32(self.K):
            residual_row = cute.make_tensor(
                residual_out.iterator + out_base + k_base,
                cute.make_layout(self.reduction_tile_K),
            )
            norm_row = cute.make_tensor(norm_weight.iterator + k_base, cute.make_layout(self.reduction_tile_K))
            weight_row = cute.make_tensor(weight.iterator + w_base + k_base, cute.make_layout(self.reduction_tile_K))
            elem = lane_idx
            while elem < Int32(self.reduction_tile_K):
                k = k_base + elem
                if k < Int32(self.K):
                    nv = residual_row[elem].to(Float32) * rstd * norm_row[elem].to(Float32)
                    acc = acc + nv * weight_row[elem].to(Float32)
                elem = elem + Int32(32)
            k_base = k_base + Int32(self.reduction_tile_K)
        return cute.arch.warp_reduction(acc, operator.add)

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_V, x, residual_in, norm_weight, weight, residual_out, logits):
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.threads_per_row // 32
        row_start = tile_S * Int32(self.tile_size_S)
        vocab_start = tile_V * Int32(self.tile_size_V)

        for local_row in range(warp_idx, self.tile_size_S, num_warps):
            row_idx = row_start + Int32(local_row)
            if row_idx < Int32(self.S):
                rstd = self._row_rstd_and_store_residual(tile_B, row_idx, x, residual_in, residual_out)
                for local_v in range(self.tile_size_V):
                    vocab_idx = vocab_start + Int32(local_v)
                    if vocab_idx < Int32(self.V):
                        total = self._dot_vocab(tile_B, row_idx, vocab_idx, rstd, residual_out, norm_weight, weight)
                        if lane_idx == Int32(0):
                            logits_base = tile_B * Int32(self.logits_stride_B) + row_idx * Int32(self.logits_stride_S)
                            logits_tile = cute.make_tensor(
                                logits.iterator + logits_base + vocab_start,
                                cute.make_layout(self.tile_size_V),
                            )
                            logits_tile[local_v] = total.to(self.logits_dtype)


class Qwen3_5QKNormRopeKCacheStoreSm120Op(_BaseQKNormRopeOp):
    """QKNorm+partial RoPE for K, then store directly to native BSHD cache."""

    reads = {
        "q": (None, ("M", "H", "D")),
        "norm_weight": (None, ("D",)),
        "cos": (None, ("S", "D2")),
        "sin": (None, ("S", "D2")),
    }
    writes = {
        "dst_k": (None, ("B", "N", "H", "D")),
    }
    tile = ("M", "H")
    dynamic_dims = ("M",)

    @classmethod
    def schedule(
        cls,
        tile_sizes=None,
        page_size=DEFAULT_PAGE_SIZE,
        eps=QWEN3_5_EPS,
        cache_pos=0,
        **tensors,
    ):
        tile_sizes = dict(tile_sizes or {})
        auto = cls._auto_tiles(page_size, **tensors)
        for key, value in auto.items():
            tile_sizes.setdefault(key, value)
        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        ops[0].static_dims["page_size"] = page_size
        ops[0].static_dims["eps"] = eps
        ops[0].static_dims["cache_pos"] = cache_pos
        return ops

    @cute.jit
    def store(self, page_ptr, tile_M, tile_H, q, norm_weight, cos, sin, dst_k, op_config_ptr):
        runtime_M = config_dim_i32(op_config_ptr, "M", type(self))
        s2g = cute.make_copy_atom(
            CopyBulkS2GOp(),
            self.q_dtype,
            num_bits_per_copy=self.q_nbits_per_row,
        )
        pos_start = tile_M * self.tile_size_M
        head_start = tile_H * self.tile_size_H
        for local_pos in range(self.tile_size_M):
            pos = pos_start + local_pos
            if pos < runtime_M:
                seq = pos % Int32(self.S)
                batch_idx = pos // Int32(self.S)
                s_tile = cute.make_tensor(
                    cute.make_ptr(
                        self.q_dtype,
                        page_ptr + Int32(local_pos * self.q_row_elems * self.elem_bytes),
                        cute.AddressSpace.smem,
                    ),
                    cute.make_layout((self.q_row_elems,)),
                )
                dst_k_tile = cute.make_tensor(
                    dst_k.iterator
                    + batch_idx * Int32(self.dst_k_stride_B)
                    + (seq + Int32(self.cache_pos)) * Int32(self.dst_k_stride_N)
                    + head_start * Int32(self.dst_k_stride_H),
                    cute.make_layout((self.q_row_elems,)),
                )
                ssrc_k, kdst = group_bulk_copy_modes(s_tile, dst_k_tile)
                cute.copy(s2g, ssrc_k, kdst)


class Qwen3_5VCacheStoreSm120Op(Op):
    """Copy BSHD V scratch blocks into their BSHD KV-cache windows."""

    reads = {"src_v": (None, ("B", "S", "H", "D"))}
    writes = {"dst_v": (None, ("B", "N", "H", "D"))}
    tile = ("B", "S", "H")
    dynamic_dims = ("B", "S", "N")

    @classmethod
    def schedule(cls, tile_sizes=None, cache_pos=0, **tensors):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", tensors["src_v"].shape[1])
        tile_sizes.setdefault("H", tensors["src_v"].shape[2])
        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        ops[0].static_dims["cache_pos"] = cache_pos
        return ops

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_H, src_v, dst_v, op_config_ptr):
        tidx = cute.arch.thread_idx()[0]
        runtime_S = config_dim_i32(op_config_ptr, "S", type(self))
        seq_start = tile_S * Int32(self.tile_size_S)
        head_start = tile_H * Int32(self.tile_size_H)
        for local_s in range(self.tile_size_S):
            seq = seq_start + Int32(local_s)
            if seq < runtime_S:
                for local_h in range(self.tile_size_H):
                    h = head_start + Int32(local_h)
                    if h < Int32(self.H):
                        src_v_base = (
                            tile_B * Int32(self.src_v_stride_B)
                            + seq * Int32(self.src_v_stride_S)
                            + h * Int32(self.src_v_stride_H)
                        )
                        dst_v_base = (
                            tile_B * Int32(self.dst_v_stride_B)
                            + (seq + Int32(self.cache_pos)) * Int32(self.dst_v_stride_N)
                            + h * Int32(self.dst_v_stride_H)
                        )
                        src_v_row = cute.make_tensor(src_v.iterator + src_v_base, cute.make_layout(self.D))
                        dst_v_row = cute.make_tensor(dst_v.iterator + dst_v_base, cute.make_layout(self.D))
                        elem = tidx
                        while elem < Int32(self.D):
                            dst_v_row[elem] = src_v_row[elem]
                            elem = elem + Int32(self.threads_per_row)


@dataclass
class Qwen3_5DecodeLayerSchedule:
    """Scheduled ops and view keep-alives for one Qwen 3.5 SM120 decode layer."""

    ops: list
    attention_config: object
    keep_alive: list


def schedule_decode_layer_qwen3_5_sm120(
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
    h_buf,
    q_buf,
    kv_buf,
    attn_out_buf,
    proj_buf,
    h2_buf,
    gate_up_buf,
    mlp_h_buf,
    qkv_buf=None,
    page_size=DEFAULT_PAGE_SIZE,
    eps=QWEN3_5_EPS,
    fa_num_splits=0,
    rms_tile_s=1,
    gate_up_tile=None,
    packed_qkv=False,
    chunked_attention_barriers=False,
):
    """Schedule one Qwen 3.5 decode layer for an SM120 megakernel.

    The scheduled layer matches the full-model decode benchmark layout:
    fused-add RMS, Q plus packed-KV projection, Q/K norm+RoPE, native BSHD
    cache update, decode attention, O projection, pre-MLP RMS, gate/up GLU,
    and down projection.
    """
    from machete.kernels.attention import FlashAttentionOp
    from machete.kernels.attention.flash_decoding import flash_decoding_schedule
    from machete.kernels.glu import GLUOp
    from machete.kernels.qknorm_rope import QKNormRopeOp
    from machete.kernels.rms_norm import RMSNormOp

    pfx = f"layer.{layer_idx}"
    cos = weights["cos"][cache_pos : cache_pos + seq_len]
    sin = weights["sin"][cache_pos : cache_pos + seq_len]
    gate_up_tile = dict(gate_up_tile or QWEN3_5_GATE_UP_TILE)
    packed_qkv = packed_qkv or qkv_buf is not None
    ops = []

    ops += RMSNormOp.schedule(
        x=x_in,
        weight=weights[f"{pfx}.attn_norm"],
        y=h_buf,
        residual_in=residual_in,
        residual_out=residual_out,
        tile_sizes={"S": rms_tile_s},
        page_size=page_size,
    )

    if packed_qkv:
        if qkv_buf is None:
            raise ValueError("qkv_buf must be provided when packed_qkv is enabled")
        qkv_flat = qkv_buf.view(
            batch,
            seq_len,
            (QWEN3_5_NUM_Q_HEADS + 2 * QWEN3_5_NUM_KV_HEADS) * QWEN3_5_HEAD_DIM,
        )
        qkv_ops = GemmOp.schedule(a=h_buf, b=weights[f"{pfx}.W_qkv"], c=qkv_flat, page_size=page_size)
        for op in qkv_ops:
            op.dim_aliases["N"] = f"qkv_chunk_{layer_idx}"
            op.static_dims["barrier_group_count_N"] = QWEN3_5_NUM_Q_HEADS + 2 * QWEN3_5_NUM_KV_HEADS
        ops += qkv_ops
    elif not packed_qkv:
        q_ops = GemmOp.schedule(a=h_buf, b=weights[f"{pfx}.W_q"], c=q_buf, page_size=page_size)
        for op in q_ops:
            op.dim_aliases["N"] = f"q_head_{layer_idx}"
        ops += q_ops

        kv_flat = kv_buf.view(batch, seq_len, 2 * QWEN3_5_KV_DIM)
        kv_ops = GemmOp.schedule(a=h_buf, b=weights[f"{pfx}.W_kv"], c=kv_flat, page_size=page_size)
        for op in kv_ops:
            op.dim_aliases["N"] = f"kv_chunk_{layer_idx}"
            op.static_dims["barrier_group_count_N"] = 2 * QWEN3_5_NUM_KV_HEADS
        ops += kv_ops

    if packed_qkv:
        q_block = qkv_buf[:, :, :QWEN3_5_NUM_Q_HEADS, :]
        q_4d = q_block.as_strided(
            (batch * seq_len, QWEN3_5_NUM_Q_HEADS, QWEN3_5_HEAD_DIM),
            (q_block.stride(1), q_block.stride(2), q_block.stride(3)),
        )
        k_block = qkv_buf[:, :, QWEN3_5_NUM_Q_HEADS : QWEN3_5_NUM_Q_HEADS + QWEN3_5_NUM_KV_HEADS, :]
        q_fd = qkv_buf[:, :, :QWEN3_5_NUM_Q_HEADS, :]
        v_block = qkv_buf[:, :, QWEN3_5_NUM_Q_HEADS + QWEN3_5_NUM_KV_HEADS :, :]
    else:
        q_4d = q_buf.view(batch * seq_len, QWEN3_5_NUM_Q_HEADS, QWEN3_5_HEAD_DIM)
        k_block = kv_buf[:, :, :QWEN3_5_NUM_KV_HEADS, :]
        q_fd = q_buf.view(batch, seq_len, QWEN3_5_NUM_Q_HEADS, QWEN3_5_HEAD_DIM)
        v_block = kv_buf[:, :, QWEN3_5_NUM_KV_HEADS :, :]

    k_4d = k_block.as_strided(
        (batch * seq_len, QWEN3_5_NUM_KV_HEADS, QWEN3_5_HEAD_DIM),
        (k_block.stride(1), k_block.stride(2), k_block.stride(3)),
    )
    k_window = k_cache[:, : cache_pos + seq_len, :, :]
    v_window = v_cache[:, : cache_pos + seq_len, :, :]
    o_fd = attn_out_buf.view(batch, seq_len, QWEN3_5_NUM_Q_HEADS, QWEN3_5_HEAD_DIM)

    q_norm_ops = QKNormRopeOp.schedule(
        q=q_4d,
        norm_weight=weights[f"{pfx}.w_q_norm"],
        cos=cos,
        sin=sin,
        tile_sizes={"M": 16, "H": 1},
        page_size=page_size,
        eps=eps,
    )
    for op in q_norm_ops:
        op.dim_aliases["H"] = f"q_head_{layer_idx}"
        if packed_qkv:
            op.static_dims["barrier_wait_alias_H"] = f"qkv_chunk_{layer_idx}"
            op.static_dims["barrier_wait_tile_size_H"] = QWEN3_5_HEAD_DIM
            op.static_dims["barrier_signal_alias_H"] = f"q_head_{layer_idx}"
            op.static_dims["barrier_signal_tile_size_H"] = QWEN3_5_HEAD_DIM
            op.static_dims["barrier_wait_acquire"] = 1
        else:
            op.static_dims["barrier_tile_size_H"] = QWEN3_5_HEAD_DIM
    ops += q_norm_ops

    k_cache_ops = Qwen3_5QKNormRopeKCacheStoreSm120Op.schedule(
        q=k_4d,
        norm_weight=weights[f"{pfx}.w_k_norm"],
        cos=cos,
        sin=sin,
        dst_k=k_window,
        cache_pos=cache_pos,
        tile_sizes={"M": 16, "H": 1},
        page_size=page_size,
        eps=eps,
    )
    for op in k_cache_ops:
        op.dim_aliases["H"] = f"kv_chunk_{layer_idx}"
        if packed_qkv:
            op.static_dims["barrier_wait_alias_H"] = f"qkv_chunk_{layer_idx}"
            op.static_dims["barrier_wait_tile_size_H"] = QWEN3_5_HEAD_DIM
            op.static_dims["barrier_wait_index_offset_H"] = QWEN3_5_NUM_Q_HEADS
            op.static_dims["barrier_wait_acquire"] = 1
        else:
            op.static_dims["barrier_tile_size_H"] = QWEN3_5_HEAD_DIM
        if chunked_attention_barriers:
            op.static_dims["barrier_signal_alias_H"] = f"q_head_{layer_idx}"
            op.static_dims["barrier_signal_tile_size_H"] = QWEN3_5_HEAD_DIM * QWEN3_5_KV_GROUP_SIZE
    ops += k_cache_ops

    v_cache_ops = Qwen3_5VCacheStoreSm120Op.schedule(
        src_v=v_block,
        dst_v=v_window,
        cache_pos=cache_pos,
        tile_sizes={"S": seq_len, "H": 1},
    )
    for op in v_cache_ops:
        op.dim_aliases["H"] = f"kv_chunk_{layer_idx}"
        op.static_dims["barrier_wait_alias_H"] = f"qkv_chunk_{layer_idx}" if packed_qkv else f"kv_chunk_{layer_idx}"
        op.static_dims["barrier_wait_tile_size_H"] = QWEN3_5_HEAD_DIM
        op.static_dims["barrier_wait_index_offset_H"] = (
            QWEN3_5_NUM_Q_HEADS + QWEN3_5_NUM_KV_HEADS if packed_qkv else QWEN3_5_NUM_KV_HEADS
        )
        if packed_qkv:
            op.static_dims["barrier_wait_acquire"] = 1
        if chunked_attention_barriers:
            op.static_dims["barrier_signal_alias_H"] = f"q_head_{layer_idx}"
            op.static_dims["barrier_signal_tile_size_H"] = QWEN3_5_HEAD_DIM * QWEN3_5_KV_GROUP_SIZE
    ops += v_cache_ops

    if k_window.shape[1] <= 256:
        fa_ops = FlashAttentionOp.schedule(
            q=q_fd,
            k=k_window,
            v=v_window,
            o=o_fd,
            kv_group_size=QWEN3_5_KV_GROUP_SIZE,
            page_size=page_size,
        )
        fa_config = FlashAttentionOp.kernel_config(fa_ops)
    else:
        fa_ops, fa_config = flash_decoding_schedule(
            q=q_fd,
            k=k_window,
            v=v_window,
            o=o_fd,
            kv_group_size=QWEN3_5_KV_GROUP_SIZE,
            page_size=page_size,
            num_splits=fa_num_splits,
        )
    for op in fa_ops:
        op.dim_aliases["M"] = f"fa_M_{layer_idx}"
        op.dim_aliases["H"] = f"q_head_{layer_idx}"
        op.static_dims["barrier_tile_size_H"] = QWEN3_5_HEAD_DIM
    ops += fa_ops

    ops += GemmOp.schedule(a=attn_out_buf, b=weights[f"{pfx}.W_o"], c=proj_buf, page_size=page_size)
    ops += RMSNormOp.schedule(
        x=proj_buf,
        weight=weights[f"{pfx}.mlp_norm"],
        y=h2_buf,
        residual_in=residual_out,
        residual_out=residual_out,
        tile_sizes={"S": rms_tile_s},
        page_size=page_size,
    )
    ops += GemmOp.schedule(
        a=h2_buf,
        b=weights[f"{pfx}.W_gate_up"],
        c=gate_up_buf,
        page_size=page_size,
        tile_sizes=gate_up_tile,
    )
    ops += GLUOp.schedule(
        x=gate_up_buf,
        y=mlp_h_buf,
        activation="silu",
        tile_sizes={"S": 1},
        page_size=page_size,
    )
    ops += GemmOp.schedule(a=mlp_h_buf, b=weights[f"{pfx}.W_down"], c=x_out, page_size=page_size)

    keep_alive = [cos, sin, q_4d, k_block, k_4d, q_fd, v_block, k_window, v_window, o_fd]
    if not packed_qkv:
        keep_alive.append(kv_buf.view(batch, seq_len, 2 * QWEN3_5_KV_DIM))
    return Qwen3_5DecodeLayerSchedule(ops=ops, attention_config=fa_config, keep_alive=keep_alive)


def schedule_final_qwen3_5_sm120(
    *,
    x,
    residual_in,
    residual_out,
    h_final,
    final_norm,
    lm_head=None,
    logits=None,
    seq_len,
    page_size=DEFAULT_PAGE_SIZE,
    eps=QWEN3_5_EPS,
    rms_tile_s=1,
):
    """Schedule Qwen final fused-add RMSNorm and optional LM head projection."""
    from machete.kernels.rms_norm import RMSNormOp

    ops = RMSNormOp.schedule(
        x=x,
        weight=final_norm,
        y=h_final,
        residual_in=residual_in,
        residual_out=residual_out,
        tile_sizes={"S": rms_tile_s},
        page_size=page_size,
    )
    if lm_head is not None:
        if logits is None:
            raise ValueError("logits must be provided when lm_head is scheduled")
        ops += Qwen3_5LmHeadGemmSm120Op.schedule(a=h_final, b=lm_head, c=logits, page_size=page_size)
    return ops


__all__ = [
    "QWEN3_5_NUM_LAYERS",
    "QWEN3_5_HIDDEN",
    "QWEN3_5_INTERMEDIATE",
    "QWEN3_5_NUM_Q_HEADS",
    "QWEN3_5_NUM_KV_HEADS",
    "QWEN3_5_HEAD_DIM",
    "QWEN3_5_ROTARY_DIM",
    "QWEN3_5_ROTARY_D2",
    "QWEN3_5_Q_DIM",
    "QWEN3_5_KV_DIM",
    "QWEN3_5_KV_GROUP_SIZE",
    "QWEN3_5_EPS",
    "QWEN3_5_VOCAB",
    "QWEN3_5_DECODE_S",
    "QWEN3_5_GATE_UP_TILE",
    "QWEN3_5_LM_HEAD_TILE_V",
    "Qwen3_5DecodeLayerSchedule",
    "Qwen3_5FinalRmsLmHeadSm120Op",
    "Qwen3_5LmHeadGemmSm120Op",
    "Qwen3_5QKNormRopeKCacheStoreSm120Op",
    "Qwen3_5VCacheStoreSm120Op",
    "schedule_decode_layer_qwen3_5_sm120",
    "schedule_final_qwen3_5_sm120",
]
