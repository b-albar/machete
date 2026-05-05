#!/usr/bin/env python
# Copyright (c) 2025, Machete Authors
"""Benchmark full-model Qwen 3.5-0.8B autoregressive decode as a single megakernel.

Autoregressive decode is deeply memory-bound:
    - Every GEMM is a matrix-vector multiply (M=1 per batch): 1 FLOP/byte
    - Attention reads the entire KV cache: ~1 FLOP/byte
    - ~31 MB of weights per layer × 36 layers = ~1.1 GB per token
    - H100 (3.35 TB/s): ~0.33ms pure bandwidth, ~2ms kernel launch overhead
    - Megakernel eliminates ~400 kernel launches → single launch

Architecture (single megakernel, all 36 layers):
    Per layer:
        RMSNorm(fused-add) → GEMM(Q) → GEMM(KV packed BSHD)
        → QKNormRope(Q,K) → KV cache update → FlashDecoding(Q, KV_cache[layer])
        → GEMM(O) → RMSNorm(fused-add) → GEMM(gate_up)
        → GLU → GEMM(down)
    Final: RMSNorm → GEMM(lm_head)

KV cache handling:
    K/V caches use native BSHD layout. The decode path writes current K/V into
    cache in-kernel; K and V are consumed directly from packed BSHD views.

Usage:
    python benchmarks/kernels/benchmark_qwen3_5_decode.py
"""

import argparse
import contextlib
import io
import os

import torch
import torch.nn.functional as F

from machete.utils.benchmark import Benchmark

try:
    import cutlass  # noqa: F401
    import cutlass.cute as cute
    from cutlass import Int32
    from machete.megakernel.ops import Op

    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False


# =============================================================================
# Qwen 3.5-0.8B Model Config
# =============================================================================

NUM_LAYERS = 36
HIDDEN = 1024
INTERMEDIATE = 3584
NUM_Q_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = 256
ROTARY_DIM = 64
D2 = ROTARY_DIM // 2
Q_DIM = NUM_Q_HEADS * HEAD_DIM   # 2048
KV_DIM = NUM_KV_HEADS * HEAD_DIM  # 512
KV_GROUP_SIZE = NUM_Q_HEADS // NUM_KV_HEADS  # 4
EPS = 1e-6
VOCAB_SIZE = 151936


try:
    from machete.kernels.gemm import GemmOp as _BaseGemmOp
except ImportError:
    _BaseGemmOp = None


if _BaseGemmOp is not None:
    class LmHeadGemmOp(_BaseGemmOp):
        """Separate GEMM handler family for the final lm_head projection."""


else:
    class LmHeadGemmOp:
        """Import-safe placeholder used only when Machete is not importable."""


def maybe_compiled_forward(fn, example_args):
    """Return a warmed ``torch.compile`` version of ``fn`` when available."""
    try:
        compiled = torch.compile(fn, mode="reduce-overhead", fullgraph=True)
        compiled(*example_args)
        compiled(*example_args)
        torch.cuda.synchronize()
        return compiled
    except Exception as exc:
        print(f"  torch.compile failed: {exc}")
        return None


def _correctness_status(actual, expected, rtol=2e-2, atol=2e-2):
    """Compact correctness status for benchmark tables without aborting timing."""
    with torch.no_grad():
        actual_f = actual.float()
        expected_f = expected.float()
        if not torch.isfinite(actual_f).all():
            bad_count = int((~torch.isfinite(actual_f)).sum().item())
            bad_pct = 100.0 * bad_count / max(1, actual.numel())
            return f"BAD {bad_pct:.1f}%/nonfinite"
        if not torch.isfinite(expected_f).all():
            bad_count = int((~torch.isfinite(expected_f)).sum().item())
            bad_pct = 100.0 * bad_count / max(1, expected.numel())
            return f"REFBAD {bad_pct:.1f}%/nonfinite"
        diff = (actual_f - expected_f).abs()
        max_abs = float(diff.max().item())
        tol = atol + rtol * expected_f.abs()
        bad = diff > tol
        bad_count = int(bad.sum().item())
        if bad_count == 0:
            return f"OK/{max_abs:.3g}"
        bad_pct = 100.0 * bad_count / max(1, actual.numel())
        return f"BAD {bad_pct:.1f}%/{max_abs:.3g}"


if CUTLASS_AVAILABLE:
    from machete.kernels.qknorm_rope import QKNormRopeOp as _BaseQKNormRopeOp
    from machete.kernels.qknorm_rope.qknorm_rope import CopyBulkS2GOp, group_bulk_copy_modes
    from machete.megakernel.ops import config_dim_i32


    class QKNormRopeKCacheStoreOp(_BaseQKNormRopeOp):
        """QKNorm+RoPE for K that also writes K into the BSHD KV cache."""

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
        def schedule(cls, tile_sizes=None, page_size=32768, eps=1e-6, cache_pos=0, **tensors):
            tile_sizes = dict(tile_sizes or {})
            auto = cls._auto_tiles(page_size, **tensors)
            for k, v in auto.items():
                tile_sizes.setdefault(k, v)
            ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
            ops[0].static_dims["page_size"] = page_size
            ops[0].static_dims["eps"] = eps
            ops[0].static_dims["cache_pos"] = cache_pos
            return ops

        @cute.jit
        def store(self, page_ptr, tile_M, tile_H, q, norm_weight, cos, sin, dst_k,
                  op_config_ptr):
            """Store normalized K directly to cache."""
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

    class VCacheStoreOp(Op):
        """Copy BSHD V blocks into their BSHD KV-cache windows."""

        reads = {"src_v": (None, ("B", "S", "H", "D"))}
        writes = {"dst_v": (None, ("B", "N", "H", "D"))}
        tile = ("B", "S", "H")
        dynamic_dims = ("B", "S", "N")

        @classmethod
        def schedule(cls, tile_sizes=None, cache_pos=0, **tensors):
            ts = dict(tile_sizes or {})
            ts.setdefault("B", 1)
            ts.setdefault("S", tensors["src_v"].shape[1])
            ts.setdefault("H", tensors["src_v"].shape[2])
            ops = [cls._schedule_single(tile_sizes=ts, **tensors)]
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

else:
    QKNormRopeKCacheStoreOp = None
    VCacheStoreOp = None


def is_sm90_or_newer():
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


# =============================================================================
# Reference ops (PyTorch)
# =============================================================================

def _rmsnorm(x, weight, eps=EPS):
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)


def _per_head_rmsnorm(x, weight, num_heads, head_dim, eps=EPS):
    B, S, _ = x.shape
    x = x.view(B, S, num_heads, head_dim)
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x = (x.float() * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)
    return x.view(B, S, num_heads * head_dim)


def _apply_rope(x, cos, sin, num_heads, head_dim, rotary_dim, pos):
    B, S, _ = x.shape
    x = x.view(B, S, num_heads, head_dim)
    d2 = rotary_dim // 2
    x1 = x[..., :d2]
    x2 = x[..., d2:rotary_dim]
    cos_s = cos[pos:pos + S].unsqueeze(0).unsqueeze(2)
    sin_s = sin[pos:pos + S].unsqueeze(0).unsqueeze(2)
    r1 = x1.float() * cos_s.float() - x2.float() * sin_s.float()
    r2 = x2.float() * cos_s.float() + x1.float() * sin_s.float()
    out = torch.cat([r1.to(x.dtype), r2.to(x.dtype), x[..., rotary_dim:]], dim=-1)
    return out.reshape(B, S, num_heads * head_dim)


# =============================================================================
# Model weights + KV cache
# =============================================================================

def allocate_model_weights(dtype=torch.bfloat16, device="cuda"):
    """Allocate all Qwen 3.5-0.8B weights."""
    w = {}
    for i in range(NUM_LAYERS):
        pfx = f"layer.{i}"
        w[f"{pfx}.attn_norm"] = torch.randn(HIDDEN, dtype=dtype, device=device)
        w[f"{pfx}.W_q"] = torch.randn(Q_DIM, HIDDEN, dtype=dtype, device=device) * 0.02
        w[f"{pfx}.W_k"] = torch.randn(KV_DIM, HIDDEN, dtype=dtype, device=device) * 0.02
        w[f"{pfx}.W_v"] = torch.randn(KV_DIM, HIDDEN, dtype=dtype, device=device) * 0.02
        w[f"{pfx}.W_kv"] = torch.cat(
            [w[f"{pfx}.W_k"], w[f"{pfx}.W_v"]],
            dim=0,
        ).contiguous()
        w[f"{pfx}.w_q_norm"] = torch.ones(HEAD_DIM, dtype=dtype, device=device)
        w[f"{pfx}.w_k_norm"] = torch.ones(HEAD_DIM, dtype=dtype, device=device)
        w[f"{pfx}.W_o"] = torch.randn(HIDDEN, Q_DIM, dtype=dtype, device=device) * 0.02
        w[f"{pfx}.mlp_norm"] = torch.randn(HIDDEN, dtype=dtype, device=device)
        w[f"{pfx}.W_gate_up"] = torch.randn(2 * INTERMEDIATE, HIDDEN, dtype=dtype, device=device) * 0.02
        w[f"{pfx}.W_down"] = torch.randn(HIDDEN, INTERMEDIATE, dtype=dtype, device=device) * 0.02
    w["final_norm"] = torch.randn(HIDDEN, dtype=dtype, device=device)
    w["lm_head"] = torch.randn(VOCAB_SIZE, HIDDEN, dtype=dtype, device=device) * 0.02
    w["cos"] = torch.randn(8192, D2, dtype=dtype, device=device)
    w["sin"] = torch.randn(8192, D2, dtype=dtype, device=device)
    return w


def allocate_kv_cache(batch, max_seq_len, dtype=torch.bfloat16, device="cuda"):
    """Per-layer K and V caches as native BSHD (B, max_seq, H_kv, D)."""
    k_caches = [
        torch.zeros(batch, max_seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=dtype, device=device)
        for _ in range(NUM_LAYERS)
    ]
    v_caches = [
        torch.zeros(batch, max_seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=dtype, device=device)
        for _ in range(NUM_LAYERS)
    ]
    return k_caches, v_caches


# =============================================================================
# Sequential decode (PyTorch baseline)
# =============================================================================

def sequential_decode_step(x, residual, pos, k_caches, v_caches, weights,
                           num_layers=NUM_LAYERS):
    """Full-model decode step: 36 layers + final norm + LM head.

    Args:
        x: (B, S, HIDDEN) — current token embeddings (S=16 for MMA alignment)
        residual: (B, S, HIDDEN) — residual stream
        pos: int — starting sequence position
        k_caches, v_caches: lists of native BSHD (B, max_seq, H_kv, D) per layer
        weights: dict

    Returns:
        logits: (B, S, VOCAB_SIZE), updated residual
    """
    B, S, _ = x.shape
    cos, sin = weights["cos"], weights["sin"]

    for i in range(num_layers):
        pfx = f"layer.{i}"
        residual = x + residual
        h = _rmsnorm(residual, weights[f"{pfx}.attn_norm"])

        q = torch.matmul(h, weights[f"{pfx}.W_q"].t())
        k = torch.matmul(h, weights[f"{pfx}.W_k"].t())
        v = torch.matmul(h, weights[f"{pfx}.W_v"].t())

        q = _per_head_rmsnorm(q, weights[f"{pfx}.w_q_norm"], NUM_Q_HEADS, HEAD_DIM)
        k = _per_head_rmsnorm(k, weights[f"{pfx}.w_k_norm"], NUM_KV_HEADS, HEAD_DIM)

        q = _apply_rope(q, cos, sin, NUM_Q_HEADS, HEAD_DIM, ROTARY_DIM, pos)
        k = _apply_rope(k, cos, sin, NUM_KV_HEADS, HEAD_DIM, ROTARY_DIM, pos)

        # Update KV cache
        k_caches[i][:, pos:pos + S, :, :] = k.view(B, S, NUM_KV_HEADS, HEAD_DIM)
        v_caches[i][:, pos:pos + S, :, :] = v.view(B, S, NUM_KV_HEADS, HEAD_DIM)

        # Attention over full context
        q_4d = q.view(B, S, NUM_Q_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
        k_full = (
            k_caches[i][:, :pos + S, :, :]
            .permute(0, 2, 1, 3)
            .repeat_interleave(KV_GROUP_SIZE, dim=1)
        )
        v_full = (
            v_caches[i][:, :pos + S, :, :]
            .permute(0, 2, 1, 3)
            .repeat_interleave(KV_GROUP_SIZE, dim=1)
        )
        attn_out = F.scaled_dot_product_attention(q_4d, k_full, v_full, is_causal=False)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, S, Q_DIM)

        proj = torch.matmul(attn_out, weights[f"{pfx}.W_o"].t())

        residual = proj + residual
        h2 = _rmsnorm(residual, weights[f"{pfx}.mlp_norm"])

        gate_up = torch.matmul(h2, weights[f"{pfx}.W_gate_up"].t())
        gate, up = gate_up.chunk(2, dim=-1)
        mlp_h = F.silu(gate) * up
        x = torch.matmul(mlp_h, weights[f"{pfx}.W_down"].t())

    residual = x + residual
    h_final = _rmsnorm(residual, weights["final_norm"])
    logits = torch.matmul(h_final, weights["lm_head"].t())
    return logits, residual


# =============================================================================
# Multi-kernel helper
# =============================================================================

# =============================================================================
# Single-megakernel decode builder
# =============================================================================

def _schedule_layer_ops(
    i, batch, pos, S, k_caches, v_caches, weights, page_size,
    x_in, res_in, x_out, res_out,
    h_buf, q_buf, kv_buf, attn_out_buf, proj_buf,
    h2_buf, gate_up_buf, mlp_h_buf,
    qkv_buf=None,
):
    """Schedule all ops for one decoder layer. Returns (ops, fa_config, extra_keep_alive)."""
    from machete.kernels.gemm import GemmOp
    from machete.kernels.rms_norm import RMSNormOp
    from machete.kernels.glu import GLUOp
    from machete.kernels.qknorm_rope import QKNormRopeOp
    from machete.kernels.attention import FlashAttentionOp
    from machete.kernels.attention.flash_decoding import flash_decoding_schedule

    cos = weights["cos"][pos:pos + S]
    sin = weights["sin"][pos:pos + S]
    pfx = f"layer.{i}"
    ops = []
    gate_up_tile = {"S": 16, "N": 128, "K": 32}
    rms_tile_s = int(os.environ.get("QWEN_DECODE_RMS_TILE_S", "1"))

    # 1. RMSNorm (fused-add residual)
    ops += RMSNormOp.schedule(
        x=x_in, weight=weights[f"{pfx}.attn_norm"], y=h_buf,
        residual_in=res_in, residual_out=res_out,
        tile_sizes={"S": rms_tile_s}, page_size=page_size,
    )

    # 2-4. Q projection plus packed KV projection. Optional packed-QKV keeps
    # the BSHD scratch layout but remains opt-in because it benchmarks slower.
    packed_qkv = qkv_buf is not None
    if packed_qkv:
        qkv_flat = qkv_buf.view(batch, S, (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM)
        qkv_proj_ops = GemmOp.schedule(
            a=h_buf, b=weights[f"{pfx}.W_qkv"], c=qkv_flat,
            page_size=page_size,
        )
        for op in qkv_proj_ops:
            op.dim_aliases["N"] = f"qkv_chunk_{i}"
            op.static_dims["barrier_group_count_N"] = NUM_Q_HEADS + 2 * NUM_KV_HEADS
        ops += qkv_proj_ops
    else:
        q_proj_ops = GemmOp.schedule(
            a=h_buf, b=weights[f"{pfx}.W_q"], c=q_buf,
            page_size=page_size,
        )
        for op in q_proj_ops:
            op.dim_aliases["N"] = f"q_head_{i}"
        ops += q_proj_ops

        kv_flat = kv_buf.view(batch, S, 2 * KV_DIM)
        kv_proj_ops = GemmOp.schedule(
            a=h_buf, b=weights[f"{pfx}.W_kv"], c=kv_flat,
            page_size=page_size,
        )
        for op in kv_proj_ops:
            op.dim_aliases["N"] = f"kv_chunk_{i}"
            op.static_dims["barrier_group_count_N"] = 2 * NUM_KV_HEADS
        ops += kv_proj_ops

    # 5. QKNormRope (per-head norm + partial RoPE, in-place)
    if packed_qkv:
        q_block = qkv_buf[:, :, :NUM_Q_HEADS, :]
        q_4d = q_block.as_strided(
            (batch * S, NUM_Q_HEADS, HEAD_DIM),
            (q_block.stride(1), q_block.stride(2), q_block.stride(3)),
        )
        k_block = qkv_buf[:, :, NUM_Q_HEADS:NUM_Q_HEADS + NUM_KV_HEADS, :]
    else:
        q_4d = q_buf.view(batch * S, NUM_Q_HEADS, HEAD_DIM)
        k_block = kv_buf[:, :, :NUM_KV_HEADS, :]
    k_4d = k_block.as_strided(
        (batch * S, NUM_KV_HEADS, HEAD_DIM),
        (k_block.stride(1), k_block.stride(2), k_block.stride(3)),
    )
    q_norm_ops = QKNormRopeOp.schedule(
        q=q_4d, norm_weight=weights[f"{pfx}.w_q_norm"],
        cos=cos, sin=sin, tile_sizes={"M": 16, "H": 1}, page_size=page_size,
    )
    for op in q_norm_ops:
        op.dim_aliases["H"] = f"q_head_{i}"
        if packed_qkv:
            op.static_dims["barrier_wait_alias_H"] = f"qkv_chunk_{i}"
            op.static_dims["barrier_wait_tile_size_H"] = HEAD_DIM
            op.static_dims["barrier_signal_alias_H"] = f"q_head_{i}"
            op.static_dims["barrier_signal_tile_size_H"] = HEAD_DIM
            op.static_dims["barrier_wait_acquire"] = 1
        else:
            op.static_dims["barrier_tile_size_H"] = HEAD_DIM
    ops += q_norm_ops
    # 6. Normalize/rope K and write current K/V into cache, then decode over
    # the full cache window. K is already in shared memory here, so store it
    # directly to cache instead of launching a separate K-cache copy op.
    if packed_qkv:
        q_fd = qkv_buf[:, :, :NUM_Q_HEADS, :]
        v_block = qkv_buf[:, :, NUM_Q_HEADS + NUM_KV_HEADS:, :]
    else:
        q_fd = q_buf.view(batch, S, NUM_Q_HEADS, HEAD_DIM)
        v_block = kv_buf[:, :, NUM_KV_HEADS:, :]
    k_fd = k_caches[i][:, :pos + S, :, :]
    v_fd = v_caches[i][:, :pos + S, :, :]
    o_fd = attn_out_buf.view(batch, S, NUM_Q_HEADS, HEAD_DIM)

    chunked_fa = os.environ.get("QWEN_DECODE_CHUNKED_FA", "0") == "1"
    k_cache_ops = QKNormRopeKCacheStoreOp.schedule(
        q=k_4d, norm_weight=weights[f"{pfx}.w_k_norm"],
        cos=cos, sin=sin, dst_k=k_fd, cache_pos=pos,
        tile_sizes={"M": 16, "H": 1}, page_size=page_size,
    )
    for op in k_cache_ops:
        op.dim_aliases["H"] = f"kv_chunk_{i}"
        if packed_qkv:
            op.static_dims["barrier_wait_alias_H"] = f"qkv_chunk_{i}"
            op.static_dims["barrier_wait_tile_size_H"] = HEAD_DIM
            op.static_dims["barrier_wait_index_offset_H"] = NUM_Q_HEADS
            op.static_dims["barrier_wait_acquire"] = 1
        else:
            op.static_dims["barrier_tile_size_H"] = HEAD_DIM
        if chunked_fa:
            op.static_dims["barrier_signal_alias_H"] = f"q_head_{i}"
            op.static_dims["barrier_signal_tile_size_H"] = HEAD_DIM * KV_GROUP_SIZE
    ops += k_cache_ops

    v_cache_ops = VCacheStoreOp.schedule(
        src_v=v_block, dst_v=v_fd, cache_pos=pos,
        tile_sizes={"S": S, "H": 1},
    )
    for op in v_cache_ops:
        op.dim_aliases["H"] = f"kv_chunk_{i}"
        op.static_dims["barrier_wait_alias_H"] = f"qkv_chunk_{i}" if packed_qkv else f"kv_chunk_{i}"
        op.static_dims["barrier_wait_tile_size_H"] = HEAD_DIM
        op.static_dims["barrier_wait_index_offset_H"] = (
            NUM_Q_HEADS + NUM_KV_HEADS if packed_qkv else NUM_KV_HEADS
        )
        if packed_qkv:
            op.static_dims["barrier_wait_acquire"] = 1
        if chunked_fa:
            op.static_dims["barrier_signal_alias_H"] = f"q_head_{i}"
            op.static_dims["barrier_signal_tile_size_H"] = HEAD_DIM * KV_GROUP_SIZE
    ops += v_cache_ops

    if k_fd.shape[1] <= 256 or page_size <= 16 * 1024:
        fa_ops = FlashAttentionOp.schedule(
            q=q_fd,
            k=k_fd,
            v=v_fd,
            o=o_fd,
            kv_group_size=KV_GROUP_SIZE,
            page_size=page_size,
        )
        fa_config = FlashAttentionOp.kernel_config(fa_ops)
    else:
        fa_ops, fa_config = flash_decoding_schedule(
            q=q_fd,
            k=k_fd,
            v=v_fd,
            o=o_fd,
            kv_group_size=KV_GROUP_SIZE,
            page_size=page_size,
            num_splits=int(os.environ.get("QWEN_DECODE_FA_SPLITS", "0")),
        )
    for op in fa_ops:
        op.dim_aliases["M"] = f"fa_M_{i}"
        op.dim_aliases["H"] = f"q_head_{i}"
        op.static_dims["barrier_tile_size_H"] = HEAD_DIM
    ops += fa_ops

    # 7. O projection
    ops += GemmOp.schedule(
        a=attn_out_buf, b=weights[f"{pfx}.W_o"], c=proj_buf,
        page_size=page_size,
    )

    # 8. Pre-MLP RMSNorm (fused-add)
    ops += RMSNormOp.schedule(
        x=proj_buf, weight=weights[f"{pfx}.mlp_norm"], y=h2_buf,
        residual_in=res_out, residual_out=res_out,
        tile_sizes={"S": rms_tile_s}, page_size=page_size,
    )

    # 9-11. MLP
    ops += GemmOp.schedule(
        a=h2_buf, b=weights[f"{pfx}.W_gate_up"], c=gate_up_buf,
        page_size=page_size, tile_sizes=gate_up_tile,
    )
    ops += GLUOp.schedule(x=gate_up_buf, y=mlp_h_buf, activation="silu", tile_sizes={"S": 1}, page_size=page_size)
    ops += GemmOp.schedule(
        a=mlp_h_buf, b=weights[f"{pfx}.W_down"], c=x_out,
        page_size=page_size,
    )

    extra_keep = [cos, sin, q_4d, k_block, k_4d, q_fd, v_block, k_fd, v_fd, o_fd]
    if not packed_qkv:
        extra_keep.append(kv_flat)
    return ops, fa_config, extra_keep


def megakernel_decode_build(batch, pos, k_caches, v_caches, weights,
                            x_init, residual_init,
                            page_size=32768, num_pages=1,
                            torch_lm_head=False,
                            num_layers=NUM_LAYERS,
                            use_qwen_sm120_ops=False):
    """Build megakernel(s) for full-model decode.

    All layers are fused into a single megakernel instruction stream.
    Each layer uses the same op types, so the framework compiles each
    unique (op_cls, static_dims) once and reuses across layers.

    KV cache: pre-filled before launch. K/V GEMMs write to scratch buffers
    (same compute cost); FlashDecoding reads pre-filled caches.

    Args:
        x_init: (B, S, HIDDEN) — initial input to copy into ping-pong buffer 0
        residual_init: (B, S, HIDDEN) — initial residual

    Returns:
        (spec, logits_buf, keep_alive)
    """
    from machete.megakernel import Megakernel, MegakernelConfig
    from machete.kernels.gemm import GemmOp
    from machete.kernels.rms_norm import RMSNormOp
    if use_qwen_sm120_ops:
        from machete.kernels.qwen3_5_sm120 import (
            schedule_decode_layer_qwen3_5_sm120,
            schedule_final_qwen3_5_sm120,
        )

    dtype = torch.bfloat16
    device = "cuda"
    # FA/FD require M >= 16 (MMA tile). Use S=16 for all ops — GEMMs are still
    # weight-dominated (16 rows × 2KB << 31MB weights), so overhead is negligible.
    S = 16
    # --- Shared activation buffers (ping-pong across layers) ---
    x_buf = [torch.empty(batch, S, HIDDEN, dtype=dtype, device=device) for _ in range(2)]
    res_buf = [torch.empty(batch, S, HIDDEN, dtype=dtype, device=device) for _ in range(2)]

    # Initialize buffer 0 with input data
    x_buf[0].copy_(x_init)
    res_buf[0].copy_(residual_init)

    # Shared per-op intermediates (reused every layer)
    h_buf = torch.empty(batch, S, HIDDEN, dtype=dtype, device=device)
    q_buf = torch.empty(batch, S, Q_DIM, dtype=dtype, device=device)
    packed_qkv = use_qwen_sm120_ops or os.environ.get("QWEN_DECODE_PACKED_QKV", "0") == "1"
    qkv_buf = None
    qk_sumsq_buf = None
    if packed_qkv:
        qkv_buf = torch.empty(
            batch, S, NUM_Q_HEADS + 2 * NUM_KV_HEADS, HEAD_DIM,
            dtype=dtype, device=device,
        )
        qk_sumsq_buf = torch.empty(
            batch,
            S,
            NUM_Q_HEADS + NUM_KV_HEADS,
            HEAD_DIM // 64,
            dtype=torch.float32,
            device=device,
        )
        for i in range(num_layers):
            pfx = f"layer.{i}"
            weights[f"{pfx}.W_qkv"] = torch.cat(
                [weights[f"{pfx}.W_q"], weights[f"{pfx}.W_kv"]],
                dim=0,
            ).contiguous()
    kv_buf = torch.empty(batch, S, 2 * NUM_KV_HEADS, HEAD_DIM, dtype=dtype, device=device)
    attn_out_buf = torch.empty(batch, S, Q_DIM, dtype=dtype, device=device)
    proj_buf = torch.empty(batch, S, HIDDEN, dtype=dtype, device=device)
    h2_buf = torch.empty(batch, S, HIDDEN, dtype=dtype, device=device)
    gate_up_buf = torch.empty(batch, S, 2 * INTERMEDIATE, dtype=dtype, device=device)
    mlp_h_buf = torch.empty(batch, S, INTERMEDIATE, dtype=dtype, device=device)

    # Final outputs
    h_final_buf = torch.empty(batch, S, HIDDEN, dtype=dtype, device=device)
    res_final = torch.empty(batch, S, HIDDEN, dtype=dtype, device=device)
    logits_buf = torch.empty(batch, S, VOCAB_SIZE, dtype=dtype, device=device)

    keep_alive = list(weights.values()) + k_caches + v_caches + [
        *x_buf, *res_buf, h_buf, q_buf, kv_buf,
        attn_out_buf, proj_buf, h2_buf, gate_up_buf, mlp_h_buf,
        h_final_buf, res_final, logits_buf,
    ]
    if qkv_buf is not None:
        keep_alive.append(qkv_buf)
    if qk_sumsq_buf is not None:
        keep_alive.append(qk_sumsq_buf)

    all_ops = []
    max_fa_tpb = 0

    for i in range(num_layers):
        cur_x = x_buf[i % 2]
        cur_res = res_buf[i % 2]
        next_x = x_buf[(i + 1) % 2]
        next_res = res_buf[(i + 1) % 2]

        if use_qwen_sm120_ops:
            layer = schedule_decode_layer_qwen3_5_sm120(
                layer_idx=i,
                batch=batch,
                seq_len=S,
                cache_pos=pos,
                weights=weights,
                k_cache=k_caches[i],
                v_cache=v_caches[i],
                x_in=cur_x,
                residual_in=cur_res,
                x_out=next_x,
                residual_out=next_res,
                h_buf=h_buf,
                q_buf=q_buf,
                kv_buf=kv_buf,
                attn_out_buf=attn_out_buf,
                proj_buf=proj_buf,
                h2_buf=h2_buf,
                gate_up_buf=gate_up_buf,
                mlp_h_buf=mlp_h_buf,
                qkv_buf=qkv_buf,
                qk_sumsq_buf=qk_sumsq_buf,
                page_size=page_size,
                fa_num_splits=int(os.environ.get("QWEN_DECODE_FA_SPLITS", "0")),
                rms_tile_s=int(os.environ.get("QWEN_DECODE_RMS_TILE_S", "1")),
                packed_qkv=packed_qkv,
                chunked_attention_barriers=os.environ.get("QWEN_DECODE_CHUNKED_FA", "0") == "1",
            )
            layer_ops = layer.ops
            fa_config = layer.attention_config
            extra_keep = layer.keep_alive
        else:
            layer_ops, fa_config, extra_keep = _schedule_layer_ops(
                i, batch, pos, S, k_caches, v_caches, weights, page_size,
                cur_x, cur_res, next_x, next_res,
                h_buf, q_buf, kv_buf, attn_out_buf, proj_buf,
                h2_buf, gate_up_buf, mlp_h_buf,
                qkv_buf=qkv_buf,
            )
        all_ops += layer_ops
        max_fa_tpb = max(max_fa_tpb, fa_config.threads_per_block)
        keep_alive += extra_keep

    # --- Final: RMSNorm + LM head ---
    final_x = x_buf[num_layers % 2]
    final_res = res_buf[num_layers % 2]
    if use_qwen_sm120_ops:
        all_ops += schedule_final_qwen3_5_sm120(
            x=final_x,
            residual_in=final_res,
            residual_out=res_final,
            h_final=h_final_buf,
            final_norm=weights["final_norm"],
            lm_head=None if torch_lm_head else weights["lm_head"],
            logits=None if torch_lm_head else logits_buf,
            seq_len=S,
            page_size=page_size,
            rms_tile_s=int(os.environ.get("QWEN_DECODE_RMS_TILE_S", "1")),
        )
    else:
        all_ops += RMSNormOp.schedule(
            x=final_x, weight=weights["final_norm"], y=h_final_buf,
            residual_in=final_res, residual_out=res_final,
            tile_sizes={"S": int(os.environ.get("QWEN_DECODE_RMS_TILE_S", "1"))},
            page_size=page_size,
        )
        if not torch_lm_head:
            all_ops += LmHeadGemmOp.schedule(
                a=h_final_buf, b=weights["lm_head"], c=logits_buf,
                page_size=page_size,
            )

    # --- Build megakernel ---
    gemm_like_ops = [op for op in all_ops if op.tile_sizes.get('S') is not None]
    gemm_config = GemmOp.kernel_config(gemm_like_ops or all_ops)
    effective_page_size = max(
        op.static_dims.get('page_size', page_size) for op in all_ops
    )
    qwen_min_tpb = 256 if use_qwen_sm120_ops else 224
    default_tpb = max(qwen_min_tpb, gemm_config.threads_per_block, max_fa_tpb)
    threads_per_block = int(os.environ.get("QWEN_DECODE_THREADS", str(default_tpb)))
    default_noinline = "1" if use_qwen_sm120_ops else "0"
    config = MegakernelConfig(
        num_sms=int(os.environ["QWEN_DECODE_NUM_SMS"]) if "QWEN_DECODE_NUM_SMS" in os.environ else None,
        threads_per_block=threads_per_block,
        page_size=effective_page_size,
        num_pages=num_pages,
        sync_compute_warps_after_tile=False,
        per_sm_instruction_queues=os.environ.get("QWEN_DECODE_PER_SM_QUEUES", "1") != "0",
        noinline=os.environ.get("QWEN_DECODE_NOINLINE", default_noinline) != "0",
        inline_thin_phases=os.environ.get("QWEN_DECODE_INLINE_THIN", "1") != "0",
        loader_idle_sleep_ns=int(os.environ.get("QWEN_DECODE_LOADER_SLEEP", "0")),
        relaxed_global_barriers=os.environ.get("QWEN_DECODE_RELAXED_BARRIER", "1") == "1",
        global_barrier_sleep_ns=int(os.environ.get("QWEN_DECODE_BARRIER_SLEEP", "0")),
        opt_level=int(os.environ.get("QWEN_DECODE_OPT_LEVEL", "2")),
    )

    kernel = Megakernel(all_ops, config=config)

    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()
    torch.cuda.synchronize()
    if torch_lm_head:
        torch.matmul(h_final_buf, weights["lm_head"].t(), out=logits_buf)
        torch.cuda.synchronize()

    spec = kernel.bench_spec(keep_alive=keep_alive)
    if torch_lm_head:
        from machete.utils.benchmark_utils import KernelBenchSpec

        core_spec = spec
        bench_stream, cu_stream = core_spec.stream

        def _setup():
            if core_spec.setup_fn is not None:
                core_spec.setup_fn()

        def _launch():
            core_spec.launch_fn()
            with torch.cuda.stream(bench_stream):
                torch.matmul(h_final_buf, weights["lm_head"].t(), out=logits_buf)

        spec = KernelBenchSpec(
            launch_fn=_launch,
            setup_fn=_setup,
            stream=(bench_stream, cu_stream),
            use_host_timer=core_spec.use_host_timer,
            _keep_alive=(core_spec, keep_alive, h_final_buf, logits_buf, weights["lm_head"]),
        )
    return spec, logits_buf, res_final

# =============================================================================
# Bandwidth / FLOPs analysis
# =============================================================================

def _total_weight_bytes():
    """Total model weight bytes (bf16)."""
    elem = 2
    per_layer = (
        Q_DIM * HIDDEN           # W_q
        + KV_DIM * HIDDEN * 2    # W_k, W_v
        + HIDDEN * Q_DIM         # W_o
        + 2 * INTERMEDIATE * HIDDEN  # W_gate_up
        + HIDDEN * INTERMEDIATE  # W_down
        + HIDDEN * 2             # norms
        + HEAD_DIM * 2           # qk norms
    )
    return (NUM_LAYERS * per_layer + VOCAB_SIZE * HIDDEN + HIDDEN) * elem


def _total_bytes_decode(batch, context_len):
    """Total bytes for one decode step."""
    elem = 2
    weight_bytes = _total_weight_bytes()
    # KV cache reads (per layer): 2 tensors × H_kv × (context_len+1) × D
    kv_bytes = NUM_LAYERS * 2 * NUM_KV_HEADS * (context_len + 1) * HEAD_DIM * elem
    # Activation reads/writes negligible for M=1
    return weight_bytes + kv_bytes


def _total_flops_decode(batch, context_len):
    """Total FLOPs for one decode step."""
    M = batch  # S=1
    per_layer = (
        2 * M * Q_DIM * HIDDEN
        + 2 * M * KV_DIM * HIDDEN * 2    # K, V
        + 2 * M * HIDDEN * Q_DIM         # O
        + 2 * M * 2 * INTERMEDIATE * HIDDEN  # gate_up
        + 2 * M * HIDDEN * INTERMEDIATE  # down
    )
    attn = batch * NUM_Q_HEADS * 2 * 1 * (context_len + 1) * HEAD_DIM
    lm_head = 2 * M * VOCAB_SIZE * HIDDEN
    return NUM_LAYERS * (per_layer + attn) + lm_head


def _kernel_launch_count():
    """Approx kernel launches for sequential decode (1 per op)."""
    per_layer = 11  # rmsnorm, 3×gemm, 2×qknorm, attn, gemm_o, rmsnorm, gemm_gu, glu, gemm_down
    return NUM_LAYERS * per_layer + 2  # + final_norm + lm_head


# =============================================================================
# Benchmarks
# =============================================================================

# Megakernel uses S=16 (FA/FD MMA minimum). For fair comparison, sequential
# also uses S=16 (padding). GEMMs are weight-dominated so overhead is negligible.
DECODE_S = 16


@Benchmark.parametrize("batch", [1])
@Benchmark.parametrize("context_len", [128, 256, 512, 1024, 2048, 4096])
@Benchmark.parametrize("num_pages", [1, 2, 3])
@Benchmark.parametrize("page_size", [32768, 49152])
def bench_qwen35_decode(context_len, batch, num_pages, page_size):
    """Benchmark full-model Qwen 3.5 decode step (S=16 for MMA alignment)."""
    torch.manual_seed(42)
    dtype = torch.bfloat16
    device = "cuda"
    num_layers = int(os.environ.get("QWEN_DECODE_NUM_LAYERS", str(NUM_LAYERS)))

    weights = allocate_model_weights(dtype=dtype, device=device)
    k_caches, v_caches = allocate_kv_cache(batch, context_len + DECODE_S, dtype=dtype, device=device)
    pos = context_len

    x = torch.randn(batch, DECODE_S, HIDDEN, dtype=dtype, device=device)
    residual = torch.zeros(batch, DECODE_S, HIDDEN, dtype=dtype, device=device)
    for i in range(NUM_LAYERS):
        k_caches[i][:, :pos, :, :].normal_()
        v_caches[i][:, :pos, :, :].normal_()

    # --- Sequential baseline ---
    sequential_decode_step(
        x, residual.clone(), pos, k_caches, v_caches, weights,
        num_layers=num_layers,
    )
    torch.cuda.synchronize()

    funcs = {}
    funcs["sequential"] = lambda: sequential_decode_step(
        x, residual.clone(), pos, k_caches, v_caches, weights,
        num_layers=num_layers,
    )

    def decode_timed(x_arg, residual_arg):
        return sequential_decode_step(
            x_arg, residual_arg, pos, k_caches, v_caches, weights,
            num_layers=num_layers,
        )

    compiled_forward = maybe_compiled_forward(decode_timed, (x, residual.clone()))
    if compiled_forward is not None:
        funcs["torch_compile"] = lambda: compiled_forward(x, residual.clone())

    # --- Megakernel ---
    if is_sm90_or_newer() and CUTLASS_AVAILABLE:
        ref_logits, ref_residual = sequential_decode_step(
            x,
            residual.clone(),
            pos,
            [c.clone() for c in k_caches],
            [c.clone() for c in v_caches],
            weights,
            num_layers=num_layers,
        )
        torch.cuda.synchronize()
        kc_pre = [c.clone() for c in k_caches]
        vc_pre = [c.clone() for c in v_caches]

        try:
            spec_1k, logits_mk, residual_mk = megakernel_decode_build(
                batch, pos, kc_pre, vc_pre, weights,
                x_init=x, residual_init=residual.clone(),
                page_size=page_size,
                num_pages=num_pages,
                torch_lm_head=False,
                num_layers=num_layers,
            )
            logits_check = _correctness_status(
                logits_mk, ref_logits, rtol=2e-2, atol=2e-1
            )
            residual_check = _correctness_status(
                residual_mk, ref_residual, rtol=2e-2, atol=3.5e-1
            )
            spec_1k.metadata = (
                f"OK L={logits_check.split('/', 1)[1]} R={residual_check.split('/', 1)[1]}"
                if logits_check.startswith("OK") and residual_check.startswith("OK")
                else (
                    f"L {logits_check}"
                    if not logits_check.startswith("OK")
                    else f"R {residual_check}"
                )
            )
            funcs["mega"] = spec_1k
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  megakernel build failed: {e}")

        kc_pre = [c.clone() for c in k_caches]
        vc_pre = [c.clone() for c in v_caches]
        try:
            spec_qwen, logits_qwen, residual_qwen = megakernel_decode_build(
                batch, pos, kc_pre, vc_pre, weights,
                x_init=x, residual_init=residual.clone(),
                page_size=page_size,
                num_pages=num_pages,
                torch_lm_head=False,
                num_layers=num_layers,
                use_qwen_sm120_ops=True,
            )
            logits_check = _correctness_status(
                logits_qwen, ref_logits, rtol=2e-2, atol=2e-1
            )
            residual_check = _correctness_status(
                residual_qwen, ref_residual, rtol=2e-2, atol=3.5e-1
            )
            spec_qwen.metadata = (
                f"OK L={logits_check.split('/', 1)[1]} R={residual_check.split('/', 1)[1]}"
                if logits_check.startswith("OK") and residual_check.startswith("OK")
                else (
                    f"L {logits_check}"
                    if not logits_check.startswith("OK")
                    else f"R {residual_check}"
                )
            )
            funcs["mega_qwen_sm120"] = spec_qwen
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  qwen sm120 megakernel build failed: {e}")

    return funcs


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-len", type=int, action="append")
    parser.add_argument("--batch", type=int, action="append")
    parser.add_argument("--page-size", type=int, action="append")
    parser.add_argument("--num-pages", type=int, action="append")
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=20)
    args = parser.parse_args()

    params = bench_qwen35_decode._benchmark.parameters
    if args.context_len is not None:
        params["context_len"] = args.context_len
    if args.batch is not None:
        params["batch"] = args.batch
    if args.page_size is not None:
        params["page_size"] = args.page_size
    if args.num_pages is not None:
        params["num_pages"] = args.num_pages
    if args.num_layers is not None:
        os.environ["QWEN_DECODE_NUM_LAYERS"] = str(args.num_layers)

    print("=" * 100)
    print("Qwen 3.5-0.8B Full-Model Decode Benchmark (Single Megakernel)")
    print(f"  {NUM_LAYERS} layers, hidden={HIDDEN}, intermediate={INTERMEDIATE}")
    if "QWEN_DECODE_NUM_LAYERS" in os.environ:
        print(f"  benchmark layers={os.environ['QWEN_DECODE_NUM_LAYERS']}")
    print(f"  Q heads={NUM_Q_HEADS}, KV heads={NUM_KV_HEADS}, head_dim={HEAD_DIM}")
    print(f"  vocab={VOCAB_SIZE}")
    print("=" * 100)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.multi_processor_count} SMs)")
        print(f"SM90+: {is_sm90_or_newer()}")
        hbm_gb = props.total_memory / 1e9
        print(f"HBM: {hbm_gb:.1f} GB")
    print(f"CUTLASS: {CUTLASS_AVAILABLE}")

    # --- Bandwidth analysis ---
    print()
    print("-" * 80)
    print("Memory-Bound Analysis (B=1, bf16)")
    print("-" * 80)

    weight_bytes = _total_weight_bytes()
    print(f"  Model weights: {weight_bytes / 1e9:.2f} GB")

    for ctx in [128, 512, 1024, 2048, 4096]:
        total_bytes = _total_bytes_decode(1, ctx)
        total_flops = _total_flops_decode(1, ctx)
        ai = total_flops / total_bytes
        kv_bytes = total_bytes - weight_bytes
        print(f"  ctx={ctx:5d}: total={total_bytes / 1e9:.2f} GB "
              f"(weights={weight_bytes / 1e9:.2f} + KV={kv_bytes / 1e6:.0f} MB), "
              f"AI={ai:.2f} FLOP/byte")

    num_launches = _kernel_launch_count()
    launch_overhead_ms = num_launches * 5 / 1000  # ~5μs per launch
    print(f"\n  Sequential: ~{num_launches} kernel launches → ~{launch_overhead_ms:.1f} ms overhead")
    print(f"  Megakernel: 1 launch → ~0.005 ms overhead")

    # Theoretical decode time at different bandwidths
    print(f"\n  Theoretical decode time (weights only):")
    for name, bw_gbs in [("A100", 2039), ("H100", 3350), ("B200", 8000)]:
        t_ms = weight_bytes / (bw_gbs * 1e9) * 1000
        speedup_pct = launch_overhead_ms / (t_ms + launch_overhead_ms) * 100
        print(f"    {name} ({bw_gbs} GB/s): {t_ms:.2f} ms compute + "
              f"{launch_overhead_ms:.1f} ms launches = {t_ms + launch_overhead_ms:.2f} ms → "
              f"megakernel saves ~{speedup_pct:.0f}%")

    print()
    print("-" * 80)
    print("Full-Model Decode: Sequential vs Single Megakernel")
    print("-" * 80)
    bench_qwen35_decode._benchmark.run(
        mode="kernel",
        warmup=args.warmup,
        rep=args.rep,
    )
