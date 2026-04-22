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
        RMSNorm(fused-add) → GEMM(Q) → GEMM(K) → GEMM(V)
        → QKNormRope(Q,K) → FlashDecoding(Q, KV_cache[layer])
        → GEMM(O) → RMSNorm(fused-add) → GEMM(gate_up)
        → GLU → GEMM(down)
    Final: RMSNorm → GEMM(lm_head)

KV cache handling:
    For the benchmark, caches are pre-filled via sequential forward so
    FlashDecoding reads correct data. K/V GEMMs still execute (same timing)
    but write to scratch buffers. For production, a CopyOp would scatter
    K/V into cache between GEMM and FlashDecoding.

Usage:
    python benchmarks/kernels/benchmark_qwen3_5_decode.py
"""

import contextlib
import io

import torch
import torch.nn.functional as F

from machete.utils.benchmark import Benchmark

try:
    import cutlass  # noqa: F401
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
    """Per-layer K and V caches as separate (B, H_kv, max_seq, D) tensors.

    Separate per-layer tensors (not one big block) so each has contiguous
    (H_kv, max_seq, D) — compatible with FlashDecoding's TMA requirements.
    """
    k_caches = [
        torch.zeros(batch, NUM_KV_HEADS, max_seq_len, HEAD_DIM, dtype=dtype, device=device)
        for _ in range(NUM_LAYERS)
    ]
    v_caches = [
        torch.zeros(batch, NUM_KV_HEADS, max_seq_len, HEAD_DIM, dtype=dtype, device=device)
        for _ in range(NUM_LAYERS)
    ]
    return k_caches, v_caches


# =============================================================================
# Sequential decode (PyTorch baseline)
# =============================================================================

def sequential_decode_step(x, residual, pos, k_caches, v_caches, weights):
    """Full-model decode step: 36 layers + final norm + LM head.

    Args:
        x: (B, S, HIDDEN) — current token embeddings (S=16 for MMA alignment)
        residual: (B, S, HIDDEN) — residual stream
        pos: int — starting sequence position
        k_caches, v_caches: lists of (B, H_kv, max_seq, D) per layer
        weights: dict

    Returns:
        logits: (B, S, VOCAB_SIZE), updated residual
    """
    B, S, _ = x.shape
    cos, sin = weights["cos"], weights["sin"]

    for i in range(NUM_LAYERS):
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
        k_caches[i][:, :, pos:pos + S, :] = k.view(B, S, NUM_KV_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
        v_caches[i][:, :, pos:pos + S, :] = v.view(B, S, NUM_KV_HEADS, HEAD_DIM).permute(0, 2, 1, 3)

        # Attention over full context
        q_4d = q.view(B, S, NUM_Q_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
        k_full = k_caches[i][:, :, :pos + S, :].repeat_interleave(KV_GROUP_SIZE, dim=1)
        v_full = v_caches[i][:, :, :pos + S, :].repeat_interleave(KV_GROUP_SIZE, dim=1)
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
    h_buf, q_buf, k_buf, v_buf, attn_out_buf, proj_buf,
    h2_buf, gate_up_buf, mlp_h_buf,
):
    """Schedule all ops for one decoder layer. Returns (ops, fa_config, extra_keep_alive)."""
    from machete.kernels.gemm import GemmOp
    from machete.kernels.rms_norm import RMSNormOp
    from machete.kernels.glu import GLUOp
    from machete.kernels.qknorm_rope import QKNormRopeOp
    from machete.kernels.attention import flash_attention_schedule

    cos, sin = weights["cos"], weights["sin"]
    pfx = f"layer.{i}"
    ops = []

    # 1. RMSNorm (fused-add residual)
    ops += RMSNormOp.schedule(
        x=x_in, weight=weights[f"{pfx}.attn_norm"], y=h_buf,
        residual_in=res_in, residual_out=res_out,
        tile_sizes={"S": 1}, page_size=page_size,
    )

    # 2-4. QKV projections (K/V to scratch buffers)
    ops += GemmOp.schedule(a=h_buf, b=weights[f"{pfx}.W_q"], c=q_buf, page_size=page_size)
    ops += GemmOp.schedule(a=h_buf, b=weights[f"{pfx}.W_k"], c=k_buf, page_size=page_size)
    ops += GemmOp.schedule(a=h_buf, b=weights[f"{pfx}.W_v"], c=v_buf, page_size=page_size)

    # 5. QKNormRope (per-head norm + partial RoPE, in-place)
    q_4d = q_buf.view(batch * S, NUM_Q_HEADS, HEAD_DIM)
    k_4d = k_buf.view(batch * S, NUM_KV_HEADS, HEAD_DIM)
    ops += QKNormRopeOp.schedule(
        q=q_4d, norm_weight=weights[f"{pfx}.w_q_norm"],
        cos=cos, sin=sin, page_size=page_size,
    )
    ops += QKNormRopeOp.schedule(
        q=k_4d, norm_weight=weights[f"{pfx}.w_k_norm"],
        cos=cos, sin=sin, page_size=page_size,
    )

    # 6. Attention over the pre-filled cache.
    # Use the shared scheduler so decode-like workloads can pick
    # FlashDecoding on Blackwell/Hopper instead of forcing the prefill FA path.
    # On SM120 the attention scheduler expects BSHD tensors, so keep the
    # logical decode views in (B, S, H, D) form and permute the cache from
    # its stored (B, H_kv, S, D) layout.
    q_fa = q_buf.view(batch, S, NUM_Q_HEADS, HEAD_DIM)
    k_full = k_caches[i][:, :, :pos + S, :].permute(0, 2, 1, 3)
    v_full = v_caches[i][:, :, :pos + S, :].permute(0, 2, 1, 3)
    o_fa = attn_out_buf.view(batch, S, NUM_Q_HEADS, HEAD_DIM)

    fa_ops, fa_config = flash_attention_schedule(
        q=q_fa, k=k_full, v=v_full, o=o_fa,
        kv_group_size=KV_GROUP_SIZE,
    )
    for op in fa_ops:
        op.dim_aliases["M"] = f"fa_M_{i}"
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
        tile_sizes={"S": 1}, page_size=page_size,
    )

    # 9-11. MLP
    ops += GemmOp.schedule(a=h2_buf, b=weights[f"{pfx}.W_gate_up"], c=gate_up_buf, page_size=page_size)
    ops += GLUOp.schedule(x=gate_up_buf, y=mlp_h_buf, activation="silu", tile_sizes={"S": 1}, page_size=page_size)
    ops += GemmOp.schedule(a=mlp_h_buf, b=weights[f"{pfx}.W_down"], c=x_out, page_size=page_size)

    extra_keep = [q_4d, k_4d, q_fa, k_full, v_full, o_fa]
    return ops, fa_config, extra_keep


def megakernel_decode_build(batch, pos, k_caches, v_caches, weights,
                            x_init, residual_init,
                            page_size=32768, num_layers=NUM_LAYERS):
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
    k_buf = torch.empty(batch, S, KV_DIM, dtype=dtype, device=device)
    v_buf = torch.empty(batch, S, KV_DIM, dtype=dtype, device=device)
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
        *x_buf, *res_buf, h_buf, q_buf, k_buf, v_buf,
        attn_out_buf, proj_buf, h2_buf, gate_up_buf, mlp_h_buf,
        h_final_buf, res_final, logits_buf,
    ]

    all_ops = []
    max_fa_tpb = 0

    for i in range(num_layers):
        cur_x = x_buf[i % 2]
        cur_res = res_buf[i % 2]
        next_x = x_buf[(i + 1) % 2]
        next_res = res_buf[(i + 1) % 2]

        layer_ops, fa_config, extra_keep = _schedule_layer_ops(
            i, batch, pos, S, k_caches, v_caches, weights, page_size,
            cur_x, cur_res, next_x, next_res,
            h_buf, q_buf, k_buf, v_buf, attn_out_buf, proj_buf,
            h2_buf, gate_up_buf, mlp_h_buf,
        )
        all_ops += layer_ops
        max_fa_tpb = max(max_fa_tpb, fa_config.threads_per_block)
        keep_alive += extra_keep

    # --- Final: RMSNorm + LM head ---
    final_x = x_buf[num_layers % 2]
    final_res = res_buf[num_layers % 2]
    all_ops += RMSNormOp.schedule(
        x=final_x, weight=weights["final_norm"], y=h_final_buf,
        residual_in=final_res, residual_out=res_final,
        tile_sizes={"S": 1}, page_size=page_size,
    )
    all_ops += GemmOp.schedule(
        a=h_final_buf, b=weights["lm_head"], c=logits_buf,
        page_size=page_size,
    )

    # --- Build megakernel ---
    gemm_like_ops = [op for op in all_ops if op.tile_sizes.get('S') is not None]
    gemm_config = GemmOp.kernel_config(gemm_like_ops or all_ops)
    effective_page_size = max(
        op.static_dims.get('page_size', page_size) for op in all_ops
    )
    config = MegakernelConfig(
        threads_per_block=max(gemm_config.threads_per_block, max_fa_tpb),
        page_size=effective_page_size,
        num_pages=1,
    )

    print(f"  Megakernel: {len(all_ops)} ops ({num_layers} layers), "
          f"{config.threads_per_block} threads, page={effective_page_size}")

    kernel = Megakernel(all_ops, config=config)

    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()
    torch.cuda.synchronize()

    spec = kernel.bench_spec(keep_alive=keep_alive)
    return spec, logits_buf, keep_alive


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
@Benchmark.parametrize("page_size", [32768, 49152])
def bench_qwen35_decode(context_len, batch, page_size):
    """Benchmark full-model Qwen 3.5 decode step (S=16 for MMA alignment)."""
    torch.manual_seed(42)
    dtype = torch.bfloat16
    device = "cuda"

    weights = allocate_model_weights(dtype=dtype, device=device)
    k_caches, v_caches = allocate_kv_cache(batch, context_len + DECODE_S, dtype=dtype, device=device)
    pos = context_len

    x = torch.randn(batch, DECODE_S, HIDDEN, dtype=dtype, device=device)
    residual = torch.zeros(batch, DECODE_S, HIDDEN, dtype=dtype, device=device)
    for i in range(NUM_LAYERS):
        k_caches[i][:, :, :pos, :].normal_()
        v_caches[i][:, :, :pos, :].normal_()

    # --- Sequential baseline ---
    kc_clone = [c.clone() for c in k_caches]
    vc_clone = [c.clone() for c in v_caches]
    sequential_decode_step(x, residual.clone(), pos, kc_clone, vc_clone, weights)
    torch.cuda.synchronize()

    funcs = {}
    funcs["sequential"] = lambda: sequential_decode_step(
        x, residual.clone(), pos,
        [c.clone() for c in k_caches],
        [c.clone() for c in v_caches],
        weights,
    )

    # --- Megakernel ---
    if is_sm90_or_newer() and CUTLASS_AVAILABLE:
        kc_pre = [c.clone() for c in kc_clone]
        vc_pre = [c.clone() for c in vc_clone]

        try:
            spec_1k, _, _ = megakernel_decode_build(
                batch, pos, kc_pre, vc_pre, weights,
                x_init=x, residual_init=residual.clone(),
                page_size=page_size,
            )
            funcs["megakernel_1k"] = spec_1k
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  megakernel build failed: {e}")

    return funcs


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 100)
    print("Qwen 3.5-0.8B Full-Model Decode Benchmark (Single Megakernel)")
    print(f"  {NUM_LAYERS} layers, hidden={HIDDEN}, intermediate={INTERMEDIATE}")
    print(f"  Q heads={NUM_Q_HEADS}, KV heads={NUM_KV_HEADS}, head_dim={HEAD_DIM}")
    print(f"  vocab={VOCAB_SIZE}")
    print("=" * 100)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.multi_processor_count} SMs)")
        print(f"SM90+: {is_sm90_or_newer()}")
        hbm_gb = props.total_mem / 1e9
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
        warmup=5,
        rep=20,
    )
