#!/usr/bin/env python
# Copyright (c) 2025, Machete Authors
"""Benchmark a full Qwen 3.5-0.8B attention decoder layer: sequential vs megakernel.

Qwen 3.5-0.8B attention layer dimensions:
    hidden_size = 1024, intermediate_size = 3584
    Q: 8 heads × head_dim 256 = 2048, KV: 2 heads × head_dim 256 = 512
    QKV total: 3072, O proj: 2048→1024
    MLP gate+up: 1024→7168 (2×3584), down: 3584→1024
    Partial RoPE: rotary_dim = 64 (cos/sin D2 = 32)

Forward pass (10 ops):
    RMSNorm → GEMM(Q) → GEMM(K) → GEMM(V) → QKNormRope(Q,K)
    → FlashAttention → GEMM(O) → RMSNorm(fused-add)
    → GEMM(gate_up) → GLU → GEMM(down)

Compares:
    sequential: PyTorch + cuBLAS (individual torch ops)
    megakernel: single fused megakernel (1 kernel launch)

Usage:
    python benchmarks/kernels/benchmark_qwen3_5_layer.py
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
# Qwen 3.5-0.8B Attention Layer Config
# =============================================================================

HIDDEN = 1024
INTERMEDIATE = 3584
NUM_Q_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = 256  # Q/K/V head dim (note: head_dim != hidden // num_heads for Qwen3.5)
ROTARY_DIM = 64  # partial RoPE: only first 64 dims rotated
D2 = ROTARY_DIM // 2  # cos/sin table width
Q_DIM = NUM_Q_HEADS * HEAD_DIM  # 2048
KV_DIM = NUM_KV_HEADS * HEAD_DIM  # 512
KV_GROUP_SIZE = NUM_Q_HEADS // NUM_KV_HEADS  # 4
EPS = 1e-6


def is_sm90_or_newer():
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


def _rmsnorm(x, weight, eps=EPS):
    """Manual RMSNorm (avoids torch.nn.functional version issues)."""
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)


def _per_head_rmsnorm(x, weight, num_heads, head_dim, eps=EPS):
    """Per-head RMSNorm: x is (B, S, num_heads * head_dim)."""
    B, S, _ = x.shape
    x = x.view(B, S, num_heads, head_dim)
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x = (x.float() * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)
    return x.view(B, S, num_heads * head_dim)


def _apply_rope(x, cos, sin, num_heads, head_dim, rotary_dim):
    """Apply partial RoPE to x of shape (B, S, num_heads * head_dim)."""
    B, S, _ = x.shape
    x = x.view(B, S, num_heads, head_dim)
    d2 = rotary_dim // 2
    x1 = x[..., :d2]
    x2 = x[..., d2:rotary_dim]
    cos_s = cos[:S].unsqueeze(0).unsqueeze(2)  # (1, S, 1, D2)
    sin_s = sin[:S].unsqueeze(0).unsqueeze(2)
    r1 = x1.float() * cos_s.float() - x2.float() * sin_s.float()
    r2 = x2.float() * cos_s.float() + x1.float() * sin_s.float()
    out = torch.cat([r1.to(x.dtype), r2.to(x.dtype), x[..., rotary_dim:]], dim=-1)
    return out.reshape(B, S, num_heads * head_dim)


# =============================================================================
# Sequential forward (PyTorch + cuBLAS)
# =============================================================================


def sequential_forward(x, residual,
                       w_attn_norm, W_q, W_k, W_v,
                       w_q_norm, w_k_norm, cos, sin,
                       W_o, w_mlp_norm, W_gate_up, W_down):
    """Full attention layer forward using PyTorch ops."""
    B, S, _ = x.shape

    # Pre-attention RMSNorm (fused add)
    residual = x + residual
    h = _rmsnorm(residual, w_attn_norm)

    # QKV projections
    q = torch.matmul(h, W_q.t())  # (B, S, Q_DIM)
    k = torch.matmul(h, W_k.t())  # (B, S, KV_DIM)
    v = torch.matmul(h, W_v.t())  # (B, S, KV_DIM)

    # Per-head QK norm
    q = _per_head_rmsnorm(q, w_q_norm, NUM_Q_HEADS, HEAD_DIM)
    k = _per_head_rmsnorm(k, w_k_norm, NUM_KV_HEADS, HEAD_DIM)

    # Partial RoPE
    q = _apply_rope(q, cos, sin, NUM_Q_HEADS, HEAD_DIM, ROTARY_DIM)
    k = _apply_rope(k, cos, sin, NUM_KV_HEADS, HEAD_DIM, ROTARY_DIM)

    # Reshape for attention: (B, S, H, D) → (B, H, S, D)
    q = q.view(B, S, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2)
    k = k.view(B, S, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    v = v.view(B, S, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

    # Expand KV for GQA
    k = k.repeat_interleave(KV_GROUP_SIZE, dim=1)
    v = v.repeat_interleave(KV_GROUP_SIZE, dim=1)

    # Scaled dot-product attention (causal)
    attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    # Reshape back: (B, H, S, D) → (B, S, H*D)
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, Q_DIM)

    # O projection
    proj = torch.matmul(attn_out, W_o.t())  # (B, S, HIDDEN)

    # Pre-MLP RMSNorm (fused add)
    residual = proj + residual
    h2 = _rmsnorm(residual, w_mlp_norm)

    # MLP: gate_up projection → GLU → down projection
    gate_up = torch.matmul(h2, W_gate_up.t())  # (B, S, 2*INTERMEDIATE)
    gate, up = gate_up.chunk(2, dim=-1)
    mlp_h = F.silu(gate) * up
    out = torch.matmul(mlp_h, W_down.t())  # (B, S, HIDDEN)

    return out, residual


# =============================================================================
# Megakernel fused forward
# =============================================================================


def megakernel_forward_build(B, S, x, residual,
                             w_attn_norm, W_q, W_k, W_v,
                             w_q_norm, w_k_norm, cos, sin,
                             W_o, w_mlp_norm, W_gate_up, W_down,
                             page_size=32768):
    """Build a single fused megakernel for full layer forward.

    Uses as_strided views so FlashAttention reads/writes directly from/to
    GEMM output buffers with correct strided layout — no inter-kernel copies.
    Requires B=1 (single batch).
    """
    from machete.megakernel import Megakernel, MegakernelConfig
    from machete.kernels.gemm import GemmOp
    from machete.kernels.rms_norm import RMSNormOp
    from machete.kernels.glu import GLUOp
    from machete.kernels.attention import FlashAttentionOp
    from machete.kernels.qknorm_rope import QKNormRopeOp

    assert B == 1, "Single fused megakernel requires B=1"
    dtype = x.dtype
    device = x.device

    # --- Allocate intermediates ---
    residual_out = torch.empty(B, S, HIDDEN, dtype=dtype, device=device)
    h = torch.empty(B, S, HIDDEN, dtype=dtype, device=device)
    q_3d = torch.empty(B, S, Q_DIM, dtype=dtype, device=device)
    k_3d = torch.empty(B, S, KV_DIM, dtype=dtype, device=device)
    v_3d = torch.empty(B, S, KV_DIM, dtype=dtype, device=device)

    # 4D views for QKNormRope: (B*S, H, D)
    q_4d = q_3d.view(B * S, NUM_Q_HEADS, HEAD_DIM)
    k_4d = k_3d.view(B * S, NUM_KV_HEADS, HEAD_DIM)

    # Strided FA views sharing memory with GEMM outputs.
    # q_fa[h,s,d] = q_3d[0, s, h*HEAD_DIM+d] via strides (HEAD_DIM, Q_DIM, 1).
    q_fa = q_3d.as_strided(
        (NUM_Q_HEADS, S, HEAD_DIM), (HEAD_DIM, Q_DIM, 1))
    k_fa = k_3d.as_strided(
        (NUM_KV_HEADS, S, HEAD_DIM), (HEAD_DIM, KV_DIM, 1))
    v_fa = v_3d.as_strided(
        (NUM_KV_HEADS, S, HEAD_DIM), (HEAD_DIM, KV_DIM, 1))

    attn_out_3d = torch.empty(B, S, Q_DIM, dtype=dtype, device=device)
    o_fa = attn_out_3d.as_strided(
        (NUM_Q_HEADS, S, HEAD_DIM), (HEAD_DIM, Q_DIM, 1))
    lse = torch.empty(NUM_Q_HEADS, S, dtype=torch.float32, device=device)

    proj = torch.empty(B, S, HIDDEN, dtype=dtype, device=device)
    residual_out2 = torch.empty(B, S, HIDDEN, dtype=dtype, device=device)
    h2 = torch.empty(B, S, HIDDEN, dtype=dtype, device=device)
    gate_up = torch.empty(B, S, 2 * INTERMEDIATE, dtype=dtype, device=device)
    mlp_h = torch.empty(B, S, INTERMEDIATE, dtype=dtype, device=device)
    out = torch.empty(B, S, HIDDEN, dtype=dtype, device=device)

    # --- Schedule all ops ---
    rmsnorm1_ops = RMSNormOp.schedule(
        x=x, weight=w_attn_norm, y=h,
        residual_in=residual, residual_out=residual_out,
        tile_sizes={"S": 16}, page_size=page_size,
    )
    gemm_q_ops = GemmOp.schedule(a=h, b=W_q, c=q_3d, page_size=page_size)
    gemm_k_ops = GemmOp.schedule(a=h, b=W_k, c=k_3d, page_size=page_size)
    gemm_v_ops = GemmOp.schedule(a=h, b=W_v, c=v_3d, page_size=page_size)
    qknorm_q_ops = QKNormRopeOp.schedule(
        q=q_4d, norm_weight=w_q_norm, cos=cos, sin=sin, page_size=page_size)
    qknorm_k_ops = QKNormRopeOp.schedule(
        q=k_4d, norm_weight=w_k_norm, cos=cos, sin=sin, page_size=page_size)

    fa_ops = FlashAttentionOp.schedule(
        q=q_fa, k=k_fa, v=v_fa, o=o_fa, lse=lse,
        causal=True, kv_group_size=KV_GROUP_SIZE, page_size=page_size,
    )

    gemm_o_ops = GemmOp.schedule(
        a=attn_out_3d, b=W_o, c=proj, page_size=page_size)
    rmsnorm2_ops = RMSNormOp.schedule(
        x=proj, weight=w_mlp_norm, y=h2,
        residual_in=residual_out, residual_out=residual_out2,
        tile_sizes={"S": 16}, page_size=page_size,
    )
    gemm_gu_ops = GemmOp.schedule(
        a=h2, b=W_gate_up, c=gate_up, page_size=page_size)
    glu_ops = GLUOp.schedule(
        x=gate_up, y=mlp_h, activation='silu',
        tile_sizes={"S": 2}, page_size=page_size)
    gemm_down_ops = GemmOp.schedule(
        a=mlp_h, b=W_down, c=out, page_size=page_size)

    all_ops = (rmsnorm1_ops + gemm_q_ops + gemm_k_ops + gemm_v_ops
               + qknorm_q_ops + qknorm_k_ops
               + fa_ops
               + gemm_o_ops + rmsnorm2_ops + gemm_gu_ops + glu_ops
               + gemm_down_ops)

    # Combined config: max threads from GEMM (8 MMA warps) and FA (4 MMA warps)
    fa_config = FlashAttentionOp.kernel_config(fa_ops)
    gemm_config = GemmOp.kernel_config(
        [op for op in all_ops if op not in fa_ops])
    config = MegakernelConfig(
        threads_per_block=max(gemm_config.threads_per_block,
                              fa_config.threads_per_block),
        page_size=page_size,
    )

    kernel = Megakernel(all_ops, config=config)

    # Warmup / compile
    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()
    torch.cuda.synchronize()

    spec = kernel.bench_spec(keep_alive=[
        x, residual, w_attn_norm, W_q, W_k, W_v, w_q_norm, w_k_norm,
        cos, sin, W_o, w_mlp_norm, W_gate_up, W_down,
        residual_out, h, q_3d, k_3d, v_3d, q_4d, k_4d,
        q_fa, k_fa, v_fa, o_fa, lse,
        attn_out_3d, proj, residual_out2, h2, gate_up, mlp_h, out,
    ])

    return spec, out, residual_out2


# =============================================================================
# Backward
# =============================================================================



# =============================================================================
# Benchmark Setup
# =============================================================================


def _total_bytes(seq_len, batch):
    """Approximate total bytes moved for one layer forward."""
    elem = 2  # bf16
    # Activations: x, residual, h, q, k, v, attn_out, proj, h2, gate_up, mlp_h, out
    act_elems = batch * seq_len * (
        HIDDEN * 6  # x, residual, h, proj, h2, out
        + Q_DIM * 2  # q, attn_out
        + KV_DIM * 2  # k, v
        + 2 * INTERMEDIATE  # gate_up
        + INTERMEDIATE  # mlp_h
    )
    # Weights: W_q, W_k, W_v, W_o, W_gate_up, W_down, norms
    weight_elems = (
        Q_DIM * HIDDEN  # W_q
        + KV_DIM * HIDDEN * 2  # W_k, W_v
        + HIDDEN * Q_DIM  # W_o
        + 2 * INTERMEDIATE * HIDDEN  # W_gate_up
        + HIDDEN * INTERMEDIATE  # W_down
        + HIDDEN * 2  # norm weights
    )
    return (act_elems + weight_elems) * elem


def _total_flops(seq_len, batch):
    """Approximate total FLOPs for one layer forward."""
    M = batch * seq_len
    # GEMMs: 2*M*N*K each
    gemm_flops = (
        2 * M * Q_DIM * HIDDEN  # Q proj
        + 2 * M * KV_DIM * HIDDEN  # K proj
        + 2 * M * KV_DIM * HIDDEN  # V proj
        + 2 * M * HIDDEN * Q_DIM  # O proj
        + 2 * M * 2 * INTERMEDIATE * HIDDEN  # gate_up
        + 2 * M * HIDDEN * INTERMEDIATE  # down
    )
    # Attention: 2*M*M*D per head (approx, causal halves it)
    attn_flops = batch * NUM_Q_HEADS * 2 * seq_len * seq_len * HEAD_DIM
    return gemm_flops + attn_flops


@Benchmark.parametrize("batch", [1])
@Benchmark.parametrize("seq_len", [128, 512, 2048, 8192])
def bench_qwen35_layer_fwd(seq_len, batch):
    """Benchmark Qwen 3.5 attention layer forward."""
    torch.manual_seed(42)
    dtype = torch.bfloat16
    device = "cuda"

    # --- Allocate tensors ---
    x = torch.randn(batch, seq_len, HIDDEN, dtype=dtype, device=device)
    residual = torch.randn(batch, seq_len, HIDDEN, dtype=dtype, device=device)

    # Weights
    w_attn_norm = torch.randn(HIDDEN, dtype=dtype, device=device)
    W_q = torch.randn(Q_DIM, HIDDEN, dtype=dtype, device=device) * 0.02
    W_k = torch.randn(KV_DIM, HIDDEN, dtype=dtype, device=device) * 0.02
    W_v = torch.randn(KV_DIM, HIDDEN, dtype=dtype, device=device) * 0.02
    w_q_norm = torch.ones(HEAD_DIM, dtype=dtype, device=device)
    w_k_norm = torch.ones(HEAD_DIM, dtype=dtype, device=device)
    cos = torch.randn(seq_len, D2, dtype=dtype, device=device)
    sin = torch.randn(seq_len, D2, dtype=dtype, device=device)
    W_o = torch.randn(HIDDEN, Q_DIM, dtype=dtype, device=device) * 0.02
    w_mlp_norm = torch.randn(HIDDEN, dtype=dtype, device=device)
    W_gate_up = torch.randn(2 * INTERMEDIATE, HIDDEN, dtype=dtype, device=device) * 0.02
    W_down = torch.randn(HIDDEN, INTERMEDIATE, dtype=dtype, device=device) * 0.02

    funcs = {}

    # --- Sequential baseline ---
    args = (x, residual, w_attn_norm, W_q, W_k, W_v, w_q_norm, w_k_norm,
            cos, sin, W_o, w_mlp_norm, W_gate_up, W_down)
    # Warmup
    sequential_forward(*args)
    torch.cuda.synchronize()
    funcs["sequential"] = lambda: sequential_forward(*args)

    # --- Megakernel fused ---
    if is_sm90_or_newer() and CUTLASS_AVAILABLE:
        try:
            spec, _, _ = megakernel_forward_build(
                batch, seq_len, x, residual,
                w_attn_norm, W_q, W_k, W_v, w_q_norm, w_k_norm, cos, sin,
                W_o, w_mlp_norm, W_gate_up, W_down,
            )
            funcs["megakernel"] = spec
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  megakernel build failed: {e}")

    return funcs


def sequential_no_attn(x, residual,
                       w_attn_norm, W_q, W_k, W_v,
                       w_q_norm, w_k_norm, cos, sin,
                       W_o, w_mlp_norm, W_gate_up, W_down):
    """Layer forward WITHOUT attention (GEMMs + norms + GLU only)."""
    B, S, _ = x.shape
    residual = x + residual
    h = _rmsnorm(residual, w_attn_norm)
    q = torch.matmul(h, W_q.t())
    k = torch.matmul(h, W_k.t())
    v = torch.matmul(h, W_v.t())
    q = _per_head_rmsnorm(q, w_q_norm, NUM_Q_HEADS, HEAD_DIM)
    k = _per_head_rmsnorm(k, w_k_norm, NUM_KV_HEADS, HEAD_DIM)
    q = _apply_rope(q, cos, sin, NUM_Q_HEADS, HEAD_DIM, ROTARY_DIM)
    k = _apply_rope(k, cos, sin, NUM_KV_HEADS, HEAD_DIM, ROTARY_DIM)
    # Skip attention — use q as dummy attn_out
    attn_out = q[:, :, :Q_DIM]
    proj = torch.matmul(attn_out, W_o.t())
    residual = proj + residual
    h2 = _rmsnorm(residual, w_mlp_norm)
    gate_up = torch.matmul(h2, W_gate_up.t())
    gate, up = gate_up.chunk(2, dim=-1)
    mlp_h = F.silu(gate) * up
    out = torch.matmul(mlp_h, W_down.t())
    return out


@Benchmark.parametrize("batch", [1])
@Benchmark.parametrize("seq_len", [128, 512, 2048, 8192])
def bench_qwen35_no_attn(seq_len, batch):
    """Benchmark non-attention ops only (K1 + K3): GEMMs + norms + GLU."""
    torch.manual_seed(42)
    dtype = torch.bfloat16
    device = "cuda"

    x = torch.randn(batch, seq_len, HIDDEN, dtype=dtype, device=device)
    residual = torch.randn(batch, seq_len, HIDDEN, dtype=dtype, device=device)
    w_attn_norm = torch.randn(HIDDEN, dtype=dtype, device=device)
    W_q = torch.randn(Q_DIM, HIDDEN, dtype=dtype, device=device) * 0.02
    W_k = torch.randn(KV_DIM, HIDDEN, dtype=dtype, device=device) * 0.02
    W_v = torch.randn(KV_DIM, HIDDEN, dtype=dtype, device=device) * 0.02
    w_q_norm = torch.ones(HEAD_DIM, dtype=dtype, device=device)
    w_k_norm = torch.ones(HEAD_DIM, dtype=dtype, device=device)
    cos = torch.randn(seq_len, D2, dtype=dtype, device=device)
    sin = torch.randn(seq_len, D2, dtype=dtype, device=device)
    W_o = torch.randn(HIDDEN, Q_DIM, dtype=dtype, device=device) * 0.02
    w_mlp_norm = torch.randn(HIDDEN, dtype=dtype, device=device)
    W_gate_up = torch.randn(2 * INTERMEDIATE, HIDDEN, dtype=dtype, device=device) * 0.02
    W_down = torch.randn(HIDDEN, INTERMEDIATE, dtype=dtype, device=device) * 0.02

    funcs = {}
    args = (x, residual, w_attn_norm, W_q, W_k, W_v, w_q_norm, w_k_norm,
            cos, sin, W_o, w_mlp_norm, W_gate_up, W_down)

    sequential_no_attn(*args)
    torch.cuda.synchronize()
    funcs["sequential"] = lambda: sequential_no_attn(*args)

    if is_sm90_or_newer() and CUTLASS_AVAILABLE:
        try:
            from machete.megakernel import Megakernel
            from machete.kernels.gemm import GemmOp
            from machete.kernels.rms_norm import RMSNormOp
            from machete.kernels.glu import GLUOp
            from machete.kernels.qknorm_rope import QKNormRopeOp

            page_size = 32768

            # K1: RMSNorm → GEMM(Q,K,V) → QKNormRope
            res_out = torch.empty_like(x)
            h = torch.empty_like(x)
            q_3d = torch.empty(batch, seq_len, Q_DIM, dtype=dtype, device=device)
            k_3d = torch.empty(batch, seq_len, KV_DIM, dtype=dtype, device=device)
            v_3d = torch.empty(batch, seq_len, KV_DIM, dtype=dtype, device=device)
            q_4d = q_3d.view(batch * seq_len, NUM_Q_HEADS, HEAD_DIM)
            k_4d = k_3d.view(batch * seq_len, NUM_KV_HEADS, HEAD_DIM)

            # All ops in a single megakernel (no attention = no layout incompatibility)
            res_out = torch.empty_like(x)
            h = torch.empty_like(x)
            q_3d = torch.empty(batch, seq_len, Q_DIM, dtype=dtype, device=device)
            k_3d = torch.empty(batch, seq_len, KV_DIM, dtype=dtype, device=device)
            v_3d = torch.empty(batch, seq_len, KV_DIM, dtype=dtype, device=device)
            q_4d = q_3d.view(batch * seq_len, NUM_Q_HEADS, HEAD_DIM)
            k_4d = k_3d.view(batch * seq_len, NUM_KV_HEADS, HEAD_DIM)
            attn_out = q_3d[:, :, :Q_DIM]  # dummy
            proj = torch.empty_like(x)
            res_out2 = torch.empty_like(x)
            h2 = torch.empty_like(x)
            gate_up = torch.empty(batch, seq_len, 2 * INTERMEDIATE, dtype=dtype, device=device)
            mlp_h = torch.empty(batch, seq_len, INTERMEDIATE, dtype=dtype, device=device)
            out = torch.empty_like(x)

            all_ops = (
                RMSNormOp.schedule(
                    x=x, weight=w_attn_norm, y=h,
                    residual_in=residual, residual_out=res_out,
                    tile_sizes={"S": 16}, page_size=page_size)
                + GemmOp.schedule(a=h, b=W_q, c=q_3d, page_size=page_size)
                + GemmOp.schedule(a=h, b=W_k, c=k_3d, page_size=page_size)
                + GemmOp.schedule(a=h, b=W_v, c=v_3d, page_size=page_size)
                + QKNormRopeOp.schedule(q=q_4d, norm_weight=w_q_norm, cos=cos, sin=sin, page_size=page_size)
                + QKNormRopeOp.schedule(q=k_4d, norm_weight=w_k_norm, cos=cos, sin=sin, page_size=page_size)
                + GemmOp.schedule(a=attn_out, b=W_o, c=proj, page_size=page_size)
                + RMSNormOp.schedule(
                    x=proj, weight=w_mlp_norm, y=h2,
                    residual_in=res_out, residual_out=res_out2,
                    tile_sizes={"S": 16}, page_size=page_size)
                + GemmOp.schedule(a=h2, b=W_gate_up, c=gate_up, page_size=page_size)
                + GLUOp.schedule(x=gate_up, y=mlp_h, activation='silu',
                                         tile_sizes={"S": 2}, page_size=page_size)
                + GemmOp.schedule(a=mlp_h, b=W_down, c=out, page_size=page_size)
            )
            config = GemmOp.kernel_config(all_ops)
            kernel = Megakernel(all_ops, config=config)

            with contextlib.redirect_stdout(io.StringIO()):
                kernel.run()
            torch.cuda.synchronize()

            funcs["megakernel"] = kernel.bench_spec(keep_alive=[
                x, residual, w_attn_norm, W_q, W_k, W_v, w_q_norm, w_k_norm,
                cos, sin, W_o, w_mlp_norm, W_gate_up, W_down,
                res_out, h, q_3d, k_3d, v_3d, q_4d, k_4d,
                attn_out, proj, res_out2, h2, gate_up, mlp_h, out,
            ])
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  megakernel build failed: {e}")

    return funcs


def verify_correctness(B, S):
    """Verify megakernel forward matches sequential forward."""
    torch.manual_seed(42)
    dtype = torch.bfloat16
    device = "cuda"

    x = torch.randn(B, S, HIDDEN, dtype=dtype, device=device)
    residual = torch.randn(B, S, HIDDEN, dtype=dtype, device=device)
    w_attn_norm = torch.randn(HIDDEN, dtype=dtype, device=device)
    W_q = torch.randn(Q_DIM, HIDDEN, dtype=dtype, device=device) * 0.02
    W_k = torch.randn(KV_DIM, HIDDEN, dtype=dtype, device=device) * 0.02
    W_v = torch.randn(KV_DIM, HIDDEN, dtype=dtype, device=device) * 0.02
    w_q_norm = torch.ones(HEAD_DIM, dtype=dtype, device=device)
    w_k_norm = torch.ones(HEAD_DIM, dtype=dtype, device=device)
    cos = torch.randn(S, D2, dtype=dtype, device=device)
    sin = torch.randn(S, D2, dtype=dtype, device=device)
    W_o = torch.randn(HIDDEN, Q_DIM, dtype=dtype, device=device) * 0.02
    w_mlp_norm = torch.randn(HIDDEN, dtype=dtype, device=device)
    W_gate_up = torch.randn(2 * INTERMEDIATE, HIDDEN, dtype=dtype, device=device) * 0.02
    W_down = torch.randn(HIDDEN, INTERMEDIATE, dtype=dtype, device=device) * 0.02

    args = (x, residual, w_attn_norm, W_q, W_k, W_v, w_q_norm, w_k_norm,
            cos, sin, W_o, w_mlp_norm, W_gate_up, W_down)

    # Sequential reference
    ref_out, ref_res = sequential_forward(*args)
    torch.cuda.synchronize()

    # Megakernel
    spec, mk_out, mk_res = megakernel_forward_build(B, S, *args)
    # Run once more to populate output tensors (warmup already ran in build)
    spec.setup_fn()  # Reset barriers before re-launch
    with contextlib.redirect_stdout(io.StringIO()):
        spec.launch_fn()
    torch.cuda.synchronize()

    out_err = (mk_out.float() - ref_out.float()).abs().max().item()
    res_err = (mk_res.float() - ref_res.float()).abs().max().item()
    return out_err, res_err


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    print("=" * 100)
    print("Qwen 3.5-0.8B Attention Layer Benchmark")
    print(f"  hidden={HIDDEN}, intermediate={INTERMEDIATE}")
    print(f"  Q heads={NUM_Q_HEADS}, KV heads={NUM_KV_HEADS}, head_dim={HEAD_DIM}")
    print(f"  rotary_dim={ROTARY_DIM}, kv_group_size={KV_GROUP_SIZE}")
    print("=" * 100)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.multi_processor_count} SMs)")
        print(f"SM90+: {is_sm90_or_newer()}")
    print(f"CUTLASS: {CUTLASS_AVAILABLE}")
    print()

    # Correctness verification
    if is_sm90_or_newer() and CUTLASS_AVAILABLE:
        print("-" * 80)
        print("Correctness Verification")
        print("-" * 80)
        for s in [128, 512]:
            try:
                out_err, res_err = verify_correctness(1, s)
                status = "PASS" if out_err < 1.0 and res_err < 3.0 else "FAIL"
                print(f"  S={s}: out_err={out_err:.4f}, res_err={res_err:.4f} [{status}]")
            except Exception as e:
                print(f"  S={s}: ERROR — {e}")
        print()

    print("-" * 80)
    print("Forward Pass (full layer)")
    print("-" * 80)
    bench_qwen35_layer_fwd._benchmark.run(
        mode="kernel",
        flops=_total_flops,
        warmup=10,
        rep=50,
    )

    print()
    print("-" * 80)
    print("Non-Attention Ops Only (K1 + K3: GEMMs + norms + GLU)")
    print("-" * 80)
    bench_qwen35_no_attn._benchmark.run(
        mode="kernel",
        warmup=10,
        rep=50,
    )
