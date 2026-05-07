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


def _is_sm120_single_batch_long_seq(batch, seq_len, threshold):
    """Return whether the current run is on the SM120 long-sequence regime."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 12 and batch == 1 and seq_len >= threshold


def _pick_single_layer_forward_tpb(batch, seq_len, gemm_tpb, fa_tpb):
    """Choose the fused forward thread geometry for the Qwen layer benchmark.

    Long prefill is GEMM-heavy even when attention is present. Keeping the GEMM
    warp geometry avoids starving the projection and MLP tiles once launch
    overhead is already amortized by sequence length.
    """
    if _is_sm120_single_batch_long_seq(batch, seq_len, threshold=2048):
        return fa_tpb
    return max(gemm_tpb, fa_tpb)


def _pick_single_layer_forward_mma_reg_count(batch, seq_len, default_mma_regs=232):
    """Tune the fused forward runtime MMA register budget on SM120."""
    if _is_sm120_single_batch_long_seq(batch, seq_len, threshold=2048):
        return 176
    if _is_sm120_single_batch_long_seq(batch, seq_len, threshold=1024):
        return 224
    return default_mma_regs


def _pick_single_layer_fused_gemm_tiles(batch, seq_len, page_size):
    """Tune fused-Qwen GEMM tiles for the single-kernel SM120 path.

    With a 32KB page, the single-kernel forward on Blackwell-class client GPUs
    is occupancy-limited. Keeping the dominant projection tiles at 128x64x32
    reduces the long-sequence penalty versus the default shape-aware chooser.
    """
    if _is_sm120_single_batch_long_seq(batch, seq_len, threshold=1024) and page_size <= 32 * 1024:
        tile = {"S": 128, "N": 64, "K": 32}
        return {
            "qkv": tile,
            "o": tile,
            "gate_up": tile,
            "down": tile,
        }
    return {}


def _pick_single_layer_backward_tpb(batch, seq_len, gemm_tpb, fa_bwd_tpb):
    """Choose the fused backward thread geometry for the Qwen layer benchmark."""
    # On SM120, the larger GEMM-driven geometry helps short sequences by
    # keeping more of the MLP-side work resident, but once sequence length
    # grows the FA backward slice becomes dominant and the extra shell/warp
    # overhead of the wider fused kernel loses.
    if _is_sm120_single_batch_long_seq(batch, seq_len, threshold=512):
        return fa_bwd_tpb
    return max(gemm_tpb, fa_bwd_tpb)


def _pick_single_layer_backward_mma_reg_count(batch, seq_len, default_mma_regs=232):
    """Tune the fused backward runtime MMA register budget."""
    return default_mma_regs

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

PAGE_SIZES = [32768, 49152, 65536, 98304]  # 32K, 48K, 64K, 96K


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


def sequential_forward(
    x, residual, w_attn_norm, W_q, W_k, W_v, w_q_norm, w_k_norm, cos, sin, W_o, w_mlp_norm, W_gate_up, W_down
):
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


# =============================================================================
# Megakernel fused forward
# =============================================================================


def megakernel_forward_build(
    B,
    S,
    x,
    residual,
    w_attn_norm,
    W_q,
    W_k,
    W_v,
    w_q_norm,
    w_k_norm,
    cos,
    sin,
    W_o,
    w_mlp_norm,
    W_gate_up,
    W_down,
    page_size=32768,
):
    """Build the single-megakernel full layer forward.

    Uses view+permute so FlashAttention reads/writes directly from/to
    GEMM output buffers with correct strided layout — no inter-kernel copies.

    Returns:
        (spec, out, residual_out2) — single-kernel spec and output tensors for
        correctness verification.
    """
    from machete.megakernel import Megakernel, MegakernelConfig
    from machete.kernels.gemm import GemmOp
    from machete.kernels.rms_norm import RMSNormOp
    from machete.kernels.glu import GLUOp
    from machete.kernels.qknorm_rope import PackedQKNormRopeOp

    dtype = x.dtype
    device = x.device
    W_qkv = torch.cat([W_q, W_k, W_v], dim=0).contiguous()

    # --- Allocate intermediates ---
    residual_out = torch.empty(B, S, HIDDEN, dtype=dtype, device=device)
    h = torch.empty(B, S, HIDDEN, dtype=dtype, device=device)
    qkv_3d = torch.empty(B, S, Q_DIM + 2 * KV_DIM, dtype=dtype, device=device)
    q_3d = qkv_3d[:, :, :Q_DIM]
    k_3d = qkv_3d[:, :, Q_DIM:Q_DIM + KV_DIM]
    v_3d = qkv_3d[:, :, Q_DIM + KV_DIM:]

    # 4D views for QKNormRope: (B*S, H, D)
    q_4d = q_3d.view(B * S, NUM_Q_HEADS, HEAD_DIM)
    k_4d = k_3d.view(B * S, NUM_KV_HEADS, HEAD_DIM)
    qk_4d = qkv_3d[:, :, :Q_DIM + KV_DIM].view(B * S, NUM_Q_HEADS + NUM_KV_HEADS, HEAD_DIM)

    # 4D FA views sharing memory with GEMM outputs: (B, S, H, D)
    q_fa = q_3d.view(B, S, NUM_Q_HEADS, HEAD_DIM)
    k_fa = k_3d.view(B, S, NUM_KV_HEADS, HEAD_DIM)
    v_fa = v_3d.view(B, S, NUM_KV_HEADS, HEAD_DIM)

    attn_out_3d = torch.empty(B, S, Q_DIM, dtype=dtype, device=device)
    o_fa = attn_out_3d.view(B, S, NUM_Q_HEADS, HEAD_DIM)
    lse = torch.empty(B, NUM_Q_HEADS, S, dtype=torch.float32, device=device)

    proj = torch.empty(B, S, HIDDEN, dtype=dtype, device=device)
    residual_out2 = torch.empty(B, S, HIDDEN, dtype=dtype, device=device)
    h2 = torch.empty(B, S, HIDDEN, dtype=dtype, device=device)
    gate_up = torch.empty(B, S, 2 * INTERMEDIATE, dtype=dtype, device=device)
    mlp_h = torch.empty(B, S, INTERMEDIATE, dtype=dtype, device=device)
    out = torch.empty(B, S, HIDDEN, dtype=dtype, device=device)

    # --- Schedule all ops (using FA's page_size for single-kernel) ---
    from machete.kernels.attention import flash_attention_schedule

    fa_ops, fa_config = flash_attention_schedule(
        q=q_fa, k=k_fa, v=v_fa, o=o_fa, lse=lse,
        causal=True, kv_group_size=KV_GROUP_SIZE, page_size=page_size,
        write_lse=False,
    )
    fa_page_size = fa_config.page_size
    gemm_tiles = _pick_single_layer_fused_gemm_tiles(B, S, fa_page_size)

    # Schedule non-FA ops with FA's page_size so they fit in the single kernel
    rmsnorm1_ops = RMSNormOp.schedule(
        x=x, weight=w_attn_norm, y=h,
        residual_in=residual, residual_out=residual_out,
        page_size=fa_page_size,
    )
    gemm_qkv_ops = GemmOp.schedule(
        a=h,
        b=W_qkv,
        c=qkv_3d,
        page_size=fa_page_size,
        tile_sizes=gemm_tiles.get("qkv"),
    )
    qknorm_qk_ops = PackedQKNormRopeOp.schedule(
        qk=qk_4d,
        q_norm_weight=w_q_norm,
        k_norm_weight=w_k_norm,
        cos=cos,
        sin=sin,
        page_size=fa_page_size,
        num_q_heads=NUM_Q_HEADS,
        num_k_heads=NUM_KV_HEADS,
    )

    for op in fa_ops:
        op.dim_aliases["M"] = "seq"

    gemm_o_ops = GemmOp.schedule(
        a=attn_out_3d,
        b=W_o,
        c=proj,
        page_size=fa_page_size,
        tile_sizes=gemm_tiles.get("o"),
    )
    rmsnorm2_ops = RMSNormOp.schedule(
        x=proj, weight=w_mlp_norm, y=h2,
        residual_in=residual_out, residual_out=residual_out2,
        page_size=fa_page_size,
    )
    gemm_gu_ops = GemmOp.schedule(
        a=h2,
        b=W_gate_up,
        c=gate_up,
        page_size=fa_page_size,
        tile_sizes=gemm_tiles.get("gate_up"),
    )
    glu_ops = GLUOp.schedule(x=gate_up, y=mlp_h, activation="silu", page_size=fa_page_size)
    gemm_down_ops = GemmOp.schedule(
        a=mlp_h,
        b=W_down,
        c=out,
        page_size=fa_page_size,
        tile_sizes=gemm_tiles.get("down"),
    )

    keep_alive = [
        x, residual, w_attn_norm, W_q, W_k, W_v, W_qkv, w_q_norm, w_k_norm,
        cos, sin, W_o, w_mlp_norm, W_gate_up, W_down,
        residual_out, h, qkv_3d, q_3d, k_3d, v_3d, q_4d, k_4d, qk_4d,
        q_fa, k_fa, v_fa, o_fa, lse, attn_out_3d, proj,
        residual_out2, h2, gate_up, mlp_h, out,
    ]

    pre_attn_ops = rmsnorm1_ops + gemm_qkv_ops + qknorm_qk_ops
    post_attn_ops = gemm_o_ops + rmsnorm2_ops + gemm_gu_ops + glu_ops + gemm_down_ops

    # --- Single kernel: all ops in one megakernel (two-page overlap) ---
    all_ops = pre_attn_ops + fa_ops + post_attn_ops
    gemm_config = GemmOp.kernel_config(pre_attn_ops + post_attn_ops)
    single_config = MegakernelConfig(
        threads_per_block=_pick_single_layer_forward_tpb(
            B,
            S,
            gemm_config.threads_per_block,
            fa_config.threads_per_block,
        ),
        page_size=fa_page_size,
        num_pages=2,
        mma_reg_count=_pick_single_layer_forward_mma_reg_count(B, S),
    )
    single_kernel = Megakernel(all_ops, config=single_config)

    with contextlib.redirect_stdout(io.StringIO()):
        single_kernel.run()
    torch.cuda.synchronize()
    out_1k = out.clone()
    residual_out2_1k = residual_out2.clone()

    spec_1k = single_kernel.bench_spec(keep_alive=keep_alive)

    return spec_1k, out_1k, residual_out2_1k


# =============================================================================
# Backward (full layer)
# =============================================================================


def sequential_layer_bwd(
    x, residual, w_attn_norm, W_q, W_k, W_v, w_q_norm, w_k_norm,
    cos, sin, W_o, w_mlp_norm, W_gate_up, W_down, d_out,
):
    """Full layer backward via PyTorch autograd on the sequential forward."""
    # Make all inputs require grad
    inputs = [x, residual, W_q, W_k, W_v, W_o, W_gate_up, W_down]
    detached = [t.detach().requires_grad_(True) for t in inputs]
    x_, res_, Wq_, Wk_, Wv_, Wo_, Wgu_, Wd_ = detached

    # Forward
    B, S, _ = x_.shape
    res_add = x_ + res_
    h = _rmsnorm(res_add, w_attn_norm)
    q = torch.matmul(h, Wq_.t())
    k = torch.matmul(h, Wk_.t())
    v = torch.matmul(h, Wv_.t())
    q = _per_head_rmsnorm(q, w_q_norm, NUM_Q_HEADS, HEAD_DIM)
    k = _per_head_rmsnorm(k, w_k_norm, NUM_KV_HEADS, HEAD_DIM)
    q = _apply_rope(q, cos, sin, NUM_Q_HEADS, HEAD_DIM, ROTARY_DIM)
    k = _apply_rope(k, cos, sin, NUM_KV_HEADS, HEAD_DIM, ROTARY_DIM)
    q4 = q.view(B, S, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2)
    k4 = k.view(B, S, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    v4 = v.view(B, S, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    k4 = k4.repeat_interleave(KV_GROUP_SIZE, dim=1)
    v4 = v4.repeat_interleave(KV_GROUP_SIZE, dim=1)
    attn_out = F.scaled_dot_product_attention(q4, k4, v4, is_causal=True)
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, Q_DIM)
    proj = torch.matmul(attn_out, Wo_.t())
    res2 = proj + res_add
    h2 = _rmsnorm(res2, w_mlp_norm)
    gate_up = torch.matmul(h2, Wgu_.t())
    gate, up = gate_up.chunk(2, dim=-1)
    mlp_h = F.silu(gate) * up
    out = torch.matmul(mlp_h, Wd_.t())

    # Backward
    out.backward(d_out)
    return tuple(t.grad for t in detached)


def sequential_attn_bwd(q, k, v, dout, causal=True):
    """Attention backward via PyTorch SDPA autograd."""
    q_ = q.detach().requires_grad_(True)
    k_ = k.detach().requires_grad_(True)
    v_ = v.detach().requires_grad_(True)
    k_exp = k_.repeat_interleave(KV_GROUP_SIZE, dim=1)
    v_exp = v_.repeat_interleave(KV_GROUP_SIZE, dim=1)
    o = F.scaled_dot_product_attention(q_, k_exp, v_exp, is_causal=causal)
    o.backward(dout)
    return q_.grad, k_.grad, v_.grad


def megakernel_layer_bwd_build(
    B, S, x, residual, w_attn_norm, W_q, W_k, W_v, w_q_norm, w_k_norm,
    cos, sin, W_o, w_mlp_norm, W_gate_up, W_down, page_size=32768,
    d_out=None, return_state=False,
):
    """Build megakernel for full layer backward.

    Backward pass (reverse order):
        GEMM(down) bwd → GLU bwd → GEMM(gate_up) bwd → RMSNorm2 bwd
        → GEMM(O) bwd → FA bwd → [QKNormRope bwd — TODO]
        → GEMM(Q/K/V) bwd → RMSNorm1 bwd
    """
    from machete.megakernel import Megakernel, MegakernelConfig
    from machete.kernels.gemm import GemmOp, GemmRowParallelOp
    from machete.kernels.rms_norm.rms_norm import RMSNormBwdOp
    from machete.kernels.glu.glu import GLUBwdOp
    from machete.kernels.activation import AddOp
    from machete.kernels.attention import (
        FlashAttentionSm120Op,
        FlashAttentionSm120BwdOp,
        AttentionDPSumOp,
    )
    from machete.kernels.qknorm_rope import QKNormRopeBwdOp

    dtype = x.dtype
    device = x.device

    # =========================================================================
    # Forward pass (to produce intermediates needed by backward)
    # =========================================================================
    residual_out = torch.empty(B, S, HIDDEN, dtype=dtype, device=device)
    h = torch.empty(B, S, HIDDEN, dtype=dtype, device=device)
    q_3d = torch.empty(B, S, Q_DIM, dtype=dtype, device=device)
    k_3d = torch.empty(B, S, KV_DIM, dtype=dtype, device=device)
    v_3d = torch.empty(B, S, KV_DIM, dtype=dtype, device=device)
    q_fa = q_3d.view(B, S, NUM_Q_HEADS, HEAD_DIM).contiguous()
    k_fa = k_3d.view(B, S, NUM_KV_HEADS, HEAD_DIM).contiguous()
    v_fa = v_3d.view(B, S, NUM_KV_HEADS, HEAD_DIM).contiguous()
    attn_out_3d = torch.empty(B, S, Q_DIM, dtype=dtype, device=device)
    o_fa = attn_out_3d.view(B, S, NUM_Q_HEADS, HEAD_DIM).contiguous()
    lse = torch.empty(B, NUM_Q_HEADS, S, dtype=torch.float32, device=device)
    proj = torch.empty(B, S, HIDDEN, dtype=dtype, device=device)
    residual_out2 = torch.empty(B, S, HIDDEN, dtype=dtype, device=device)
    h2 = torch.empty(B, S, HIDDEN, dtype=dtype, device=device)
    gate_up = torch.empty(B, S, 2 * INTERMEDIATE, dtype=dtype, device=device)
    mlp_h = torch.empty(B, S, INTERMEDIATE, dtype=dtype, device=device)
    out = torch.empty(B, S, HIDDEN, dtype=dtype, device=device)

    # Run forward to populate intermediates (use sequential for simplicity)
    fwd_out, fwd_res = sequential_forward(
        x, residual, w_attn_norm, W_q, W_k, W_v, w_q_norm, w_k_norm,
        cos, sin, W_o, w_mlp_norm, W_gate_up, W_down,
    )

    # Re-run forward to capture intermediates we need for backward
    res_add = (x + residual).to(dtype)
    h[:] = _rmsnorm(res_add, w_attn_norm).to(dtype)
    q_3d[:] = torch.matmul(h.float(), W_q.float().t()).to(dtype)
    k_3d[:] = torch.matmul(h.float(), W_k.float().t()).to(dtype)
    v_3d[:] = torch.matmul(h.float(), W_v.float().t()).to(dtype)
    # QKNormRope on Q/K
    q_prenorm = q_3d.clone()  # save pre-norm Q for backward
    k_prenorm = k_3d.clone()
    q_normed = _per_head_rmsnorm(q_3d, w_q_norm, NUM_Q_HEADS, HEAD_DIM)
    k_normed = _per_head_rmsnorm(k_3d, w_k_norm, NUM_KV_HEADS, HEAD_DIM)
    q_roped = _apply_rope(q_normed, cos, sin, NUM_Q_HEADS, HEAD_DIM, ROTARY_DIM)
    k_roped = _apply_rope(k_normed, cos, sin, NUM_KV_HEADS, HEAD_DIM, ROTARY_DIM)
    q_fa[:] = q_roped.view(B, S, NUM_Q_HEADS, HEAD_DIM).to(dtype)
    k_fa[:] = k_roped.view(B, S, NUM_KV_HEADS, HEAD_DIM).to(dtype)
    v_fa[:] = v_3d.view(B, S, NUM_KV_HEADS, HEAD_DIM).to(dtype)
    # Run FA forward to get o and lse
    fwd_ops = FlashAttentionSm120Op.schedule(
        q=q_fa, k=k_fa, v=v_fa, o=o_fa, lse=lse,
        causal=True, kv_group_size=KV_GROUP_SIZE, page_size=page_size,
    )
    fwd_config = FlashAttentionSm120Op.kernel_config(fwd_ops)
    fwd_kernel = Megakernel(fwd_ops, config=fwd_config)
    with contextlib.redirect_stdout(io.StringIO()):
        fwd_kernel.run()
    torch.cuda.synchronize()
    attn_out_3d[:] = o_fa.view(B, S, Q_DIM)
    proj[:] = torch.matmul(attn_out_3d.float(), W_o.float().t()).to(dtype)
    residual_out[:] = res_add
    residual_out2[:] = (proj + residual_out).to(dtype)
    h2[:] = _rmsnorm(residual_out2, w_mlp_norm).to(dtype)
    gate_up[:] = torch.matmul(h2.float(), W_gate_up.float().t()).to(dtype)
    gate, up = gate_up.chunk(2, dim=-1)
    mlp_h[:] = (F.silu(gate.float()) * up.float()).to(dtype)

    # =========================================================================
    # Backward pass — schedule all ops
    # =========================================================================
    if d_out is None:
        d_out = torch.randn(B, S, HIDDEN, dtype=dtype, device=device)

    # 11'. GEMM(down) bwd: d_mlp_h = d_out @ W_down, d_W_down = mlp_h.T @ d_out
    d_mlp_h = torch.empty(B, S, INTERMEDIATE, dtype=dtype, device=device)
    gemm_down_bwd_ops = GemmOp.schedule_backward(
        dout=d_out, a=mlp_h, b=W_down, da=d_mlp_h, page_size=page_size)

    # 10'. GLU bwd: d_gate_up from d_mlp_h
    d_gate_up = torch.empty(B, S, 2 * INTERMEDIATE, dtype=dtype, device=device)
    glu_bwd_ops = GLUBwdOp.schedule(
        dy=d_mlp_h, x=gate_up, dx=d_gate_up,
        activation='silu', page_size=page_size)

    # 9'. GEMM(gate_up) bwd: d_h2 = d_gate_up @ W_gate_up
    d_h2 = torch.empty(B, S, HIDDEN, dtype=dtype, device=device)
    gemm_gu_bwd_ops = GemmOp.schedule_backward(
        dout=d_gate_up, a=h2, b=W_gate_up, da=d_h2, page_size=page_size)

    # 8'. RMSNorm2 bwd: d_proj from d_h2 (with residual passthrough)
    d_proj = torch.empty(B, S, HIDDEN, dtype=dtype, device=device)
    d_res2 = torch.empty(B, S, HIDDEN, dtype=dtype, device=device)
    rmsnorm2_bwd_ops = RMSNormBwdOp.schedule(
        dout=d_h2, x=residual_out2, weight=w_mlp_norm, dx=d_proj,
        d_residual=d_res2,
        tile_sizes={"S": 16}, page_size=page_size)

    # 7'. GEMM(O) bwd: d_attn_out = d_proj @ W_o
    # d_proj is the gradient w.r.t. proj from RMSNorm2 bwd.
    # d_res2 flows separately back through the residual path.
    d_attn_out_3d = torch.empty(B, S, Q_DIM, dtype=dtype, device=device)
    gemm_o_bwd_ops = GemmOp.schedule_backward(
        dout=d_proj, a=attn_out_3d, b=W_o, da=d_attn_out_3d,
        page_size=page_size)

    # 6'. FA bwd: dQ, dK, dV from d_attn_out
    d_attn_fa = d_attn_out_3d.view(B, S, NUM_Q_HEADS, HEAD_DIM)
    dpsum = torch.empty(B, NUM_Q_HEADS, S, dtype=torch.float32, device=device)
    dq_fa = torch.zeros(B, S, NUM_Q_HEADS, HEAD_DIM, dtype=torch.float32, device=device)
    dk_fa = torch.zeros(B, S, NUM_KV_HEADS, HEAD_DIM, dtype=dtype, device=device)
    dv_fa = torch.zeros(B, S, NUM_KV_HEADS, HEAD_DIM, dtype=dtype, device=device)
    dpsum_ops = AttentionDPSumOp.schedule(
        dout=d_attn_fa,
        o=o_fa,
        dpsum=dpsum,
        tile_sizes={"S": 16, "H": 1},
        page_size=page_size,
    )

    fa_bwd_ops = FlashAttentionSm120BwdOp.schedule(
        k=k_fa, v=v_fa, q=q_fa, dout=d_attn_fa, lse=lse, dpsum=dpsum,
        dq=dq_fa, dk=dk_fa, dv=dv_fa,
        causal=True, kv_group_size=KV_GROUP_SIZE,
    )
    for op in fa_bwd_ops:
        op.dim_aliases["M"] = "seq"

    # 5'. QKNormRope bwd on Q/K activation path.
    q_prenorm_4d = q_prenorm.view(B * S, NUM_Q_HEADS, HEAD_DIM).contiguous()
    k_prenorm_4d = k_prenorm.view(B * S, NUM_KV_HEADS, HEAD_DIM).contiguous()
    dq_fa_flat = dq_fa.view(B * S, NUM_Q_HEADS, HEAD_DIM).contiguous()
    dk_fa_flat = dk_fa.view(B * S, NUM_KV_HEADS, HEAD_DIM).contiguous()
    dq_qnr_q_flat = torch.empty_like(q_prenorm_4d)
    dq_qnr_k_flat = torch.empty_like(k_prenorm_4d)
    qknorm_q_bwd_ops = QKNormRopeBwdOp.schedule(
        q=q_prenorm_4d,
        dout=dq_fa_flat,
        dq=dq_qnr_q_flat,
        norm_weight=w_q_norm,
        cos=cos,
        sin=sin,
        page_size=page_size,
    )
    qknorm_k_bwd_ops = QKNormRopeBwdOp.schedule(
        q=k_prenorm_4d,
        dout=dk_fa_flat,
        dq=dq_qnr_k_flat,
        norm_weight=w_k_norm,
        cos=cos,
        sin=sin,
        page_size=page_size,
    )
    for op in qknorm_q_bwd_ops + qknorm_k_bwd_ops:
        op.dim_aliases["M"] = "seq"

    # 4'. GEMM(Q/K/V) bwd: accumulate into d_h1, the gradient wrt the first
    # RMSNorm output h. The residual-path gradient d_res2 stays separate and
    # is added after RMSNorm1 backward.
    dq_qnr_q = dq_qnr_q_flat.view(B, S, Q_DIM)
    dq_qnr_k = dq_qnr_k_flat.view(B, S, KV_DIM)
    dv_3d = dv_fa.view(B, S, KV_DIM)
    d_h1 = torch.zeros(B, S, HIDDEN, dtype=dtype, device=device)
    W_q_t = W_q.t().contiguous()
    W_k_t = W_k.t().contiguous()
    W_v_t = W_v.t().contiguous()
    gemm_q_bwd_ops = GemmRowParallelOp.schedule(a=dq_qnr_q, b=W_q_t, c=d_h1, page_size=page_size)
    gemm_k_bwd_ops = GemmRowParallelOp.schedule(a=dq_qnr_k, b=W_k_t, c=d_h1, page_size=page_size)
    gemm_v_bwd_ops = GemmRowParallelOp.schedule(a=dv_3d, b=W_v_t, c=d_h1, page_size=page_size)

    # 3'. RMSNorm1 bwd: consume only d_h1, then merge in the later residual path.
    dx = torch.empty(B, S, HIDDEN, dtype=dtype, device=device)
    d_res1 = torch.empty(B, S, HIDDEN, dtype=dtype, device=device)
    rmsnorm1_bwd_ops = RMSNormBwdOp.schedule(
        dout=d_h1,
        x=residual_out,
        weight=w_attn_norm,
        dx=dx,
        d_residual=d_res1,
        add=d_res2,
        tile_sizes={"S": 16},
        page_size=page_size,
    )

    # =========================================================================
    # Build single backward megakernel
    # =========================================================================
    mlp_ops = (gemm_down_bwd_ops + glu_bwd_ops + gemm_gu_bwd_ops
               + rmsnorm2_bwd_ops + gemm_o_bwd_ops)
    tail_ops = (
        qknorm_q_bwd_ops
        + qknorm_k_bwd_ops
        + gemm_q_bwd_ops
        + gemm_k_bwd_ops
        + gemm_v_bwd_ops
        + rmsnorm1_bwd_ops
    )
    fa_bwd_config = FlashAttentionSm120BwdOp.kernel_config(fa_bwd_ops)

    keep_alive = [
        x, residual, w_attn_norm, W_q, W_k, W_v, w_q_norm, w_k_norm,
        cos, sin, W_o, w_mlp_norm, W_gate_up, W_down,
        h, q_3d, k_3d, v_3d, q_prenorm, k_prenorm, q_fa, k_fa, v_fa, o_fa, lse,
        attn_out_3d, proj, residual_out, residual_out2, h2, gate_up, mlp_h,
        d_out, d_mlp_h, d_gate_up, d_h2, d_proj, d_res2,
        d_attn_out_3d, d_attn_fa, dpsum, dq_fa, dk_fa, dv_fa,
        d_h1,
        W_q_t, W_k_t, W_v_t,
        q_prenorm_4d, k_prenorm_4d, dq_fa_flat, dk_fa_flat, dq_qnr_q_flat, dq_qnr_k_flat,
        dq_qnr_q, dq_qnr_k, dv_3d, dx, d_res1,
    ]
    setup_fn = lambda: (
        dq_fa.zero_(), dk_fa.zero_(), dv_fa.zero_(),
        d_res2.zero_(), d_h1.zero_(), dx.zero_(), d_res1.zero_(),
    )

    # --- Single kernel: all backward ops in one megakernel ---
    all_bwd_ops = mlp_ops + dpsum_ops + fa_bwd_ops + tail_ops
    gemm_config = GemmOp.kernel_config(mlp_ops)
    single_config = MegakernelConfig(
        threads_per_block=_pick_single_layer_backward_tpb(
            B,
            S,
            gemm_config.threads_per_block,
            fa_bwd_config.threads_per_block,
        ),
        page_size=fa_bwd_config.page_size,
        num_pages=1,
        mma_reg_count=_pick_single_layer_backward_mma_reg_count(B, S),
    )
    single_kernel = Megakernel(all_bwd_ops, config=single_config)

    dq_fa.zero_()
    dk_fa.zero_()
    dv_fa.zero_()
    d_res2.zero_()
    d_h1.zero_()
    dx.zero_()
    d_res1.zero_()
    with contextlib.redirect_stdout(io.StringIO()):
        single_kernel.run()
    torch.cuda.synchronize()

    single_state = None
    if return_state:
        single_state = {
            "d_out": d_out.clone(),
            "d_mlp_h": d_mlp_h.clone(),
            "d_gate_up": d_gate_up.clone(),
            "d_h2": d_h2.clone(),
            "d_proj": d_proj.clone(),
            "d_res2": d_res2.clone(),
            "d_attn_out_3d": d_attn_out_3d.clone(),
            "d_h1": d_h1.clone(),
            "dq_fa": dq_fa.clone(),
            "dk_fa": dk_fa.clone(),
            "dv_fa": dv_fa.clone(),
            "dq_qnr_q": dq_qnr_q.clone(),
            "dq_qnr_k": dq_qnr_k.clone(),
            "dx": dx.clone(),
            "d_res1": d_res1.clone(),
        }

    spec_1k = single_kernel.bench_spec(
        setup_fn=setup_fn, keep_alive=keep_alive,
    )

    if return_state:
        return spec_1k, d_out, single_state
    return spec_1k, d_out


def _layer_bwd_flops(seq_len, batch, page_size=32768):
    """Approximate FLOPs for full layer backward."""
    M = batch * seq_len
    # GEMM backward: each forward GEMM produces 2 backward GEMMs (dA and dB)
    # but dB is typically not needed for inference-focused benchmarks.
    # Here we compute dA only for each GEMM.
    gemm_flops = (
        2 * M * INTERMEDIATE * HIDDEN  # GEMM(down) bwd: dA
        + 2 * M * 2 * INTERMEDIATE * HIDDEN  # GEMM(gate_up) bwd: dA
        + 2 * M * Q_DIM * HIDDEN  # GEMM(O) bwd: dA
    )
    # FA backward: 5 GEMMs per (M-block, N-tile)
    attn_flops = batch * NUM_Q_HEADS * 5 * 2 * seq_len * seq_len * HEAD_DIM
    return gemm_flops + attn_flops


def verify_layer_backward(B, S):
    """Verify megakernel backward matches sequential (autograd) backward.

    Compares the partial backward chain (GEMM_down bwd → GLU bwd →
    GEMM_gate_up bwd → RMSNorm2 bwd → GEMM_O bwd → FA bwd) against
    the full sequential autograd backward. Only checks gradients that
    the megakernel backward computes (d_mlp_h through dQ/dK/dV).
    """
    from machete.megakernel import Megakernel
    from machete.kernels.gemm import GemmOp
    from machete.kernels.rms_norm.rms_norm import RMSNormBwdOp
    from machete.kernels.glu.glu import GLUBwdOp
    from machete.kernels.attention import FlashAttentionSm120Op, FlashAttentionSm120BwdOp

    torch.manual_seed(42)
    dtype = torch.bfloat16
    device = "cuda"
    page_size = 32768

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
    d_out = torch.randn(B, S, HIDDEN, dtype=dtype, device=device)

    # Compute forward intermediates for the ops we can check
    res_add = (x + residual).to(dtype)
    h = _rmsnorm(res_add, w_attn_norm).to(dtype)
    q_3d = torch.matmul(h.float(), W_q.float().t()).to(dtype)
    k_3d = torch.matmul(h.float(), W_k.float().t()).to(dtype)
    v_3d = torch.matmul(h.float(), W_v.float().t()).to(dtype)
    q_normed = _per_head_rmsnorm(q_3d, w_q_norm, NUM_Q_HEADS, HEAD_DIM)
    k_normed = _per_head_rmsnorm(k_3d, w_k_norm, NUM_KV_HEADS, HEAD_DIM)
    q_roped = _apply_rope(q_normed, cos, sin, NUM_Q_HEADS, HEAD_DIM, ROTARY_DIM)
    k_roped = _apply_rope(k_normed, cos, sin, NUM_KV_HEADS, HEAD_DIM, ROTARY_DIM)

    # FA forward via megakernel (to get o + lse)
    q_fa = q_roped.view(B, S, NUM_Q_HEADS, HEAD_DIM).contiguous()
    k_fa = k_roped.view(B, S, NUM_KV_HEADS, HEAD_DIM).contiguous()
    v_fa = v_3d.view(B, S, NUM_KV_HEADS, HEAD_DIM).contiguous()
    o_fa = torch.zeros_like(q_fa)
    lse = torch.empty(B, NUM_Q_HEADS, S, dtype=torch.float32, device=device)
    fwd_ops = FlashAttentionSm120Op.schedule(
        q=q_fa, k=k_fa, v=v_fa, o=o_fa, lse=lse,
        causal=True, kv_group_size=KV_GROUP_SIZE, page_size=page_size,
    )
    fwd_kernel = Megakernel(fwd_ops, config=FlashAttentionSm120Op.kernel_config(fwd_ops))
    with contextlib.redirect_stdout(io.StringIO()):
        fwd_kernel.run()
    torch.cuda.synchronize()

    attn_out_3d = o_fa.view(B, S, Q_DIM)
    proj = torch.matmul(attn_out_3d.float(), W_o.float().t()).to(dtype)
    residual_out2 = (proj + res_add).to(dtype)
    h2 = _rmsnorm(residual_out2, w_mlp_norm).to(dtype)
    gate_up = torch.matmul(h2.float(), W_gate_up.float().t()).to(dtype)
    gate, up = gate_up.chunk(2, dim=-1)
    mlp_h = (F.silu(gate.float()) * up.float()).to(dtype)

    # Reference backward via autograd on post-attention ops
    # (from mlp_h backward through FA backward)
    mlp_h_ = mlp_h.detach().float().requires_grad_(True)
    out_ref = torch.matmul(mlp_h_, W_down.float().t())
    out_ref.backward(d_out.float())
    ref_d_mlp_h = mlp_h_.grad.to(dtype)

    # --- Megakernel backward ---
    # GEMM(down) bwd
    d_mlp_h = torch.empty(B, S, INTERMEDIATE, dtype=dtype, device=device)
    gemm_down_bwd = GemmOp.schedule_backward(
        dout=d_out, a=mlp_h, b=W_down, da=d_mlp_h, page_size=page_size)

    # GLU bwd
    d_gate_up = torch.empty(B, S, 2 * INTERMEDIATE, dtype=dtype, device=device)
    glu_bwd = GLUBwdOp.schedule(
        dy=d_mlp_h, x=gate_up, dx=d_gate_up, activation='silu', page_size=page_size)

    # GEMM(gate_up) bwd
    d_h2 = torch.empty(B, S, HIDDEN, dtype=dtype, device=device)
    gemm_gu_bwd = GemmOp.schedule_backward(
        dout=d_gate_up, a=h2, b=W_gate_up, da=d_h2, page_size=page_size)

    # RMSNorm2 bwd
    d_proj = torch.empty(B, S, HIDDEN, dtype=dtype, device=device)
    d_res2 = torch.empty(B, S, HIDDEN, dtype=dtype, device=device)
    rmsnorm2_bwd = RMSNormBwdOp.schedule(
        dout=d_h2, x=residual_out2, weight=w_mlp_norm, dx=d_proj,
        d_residual=d_res2, tile_sizes={"S": 16}, page_size=page_size)

    # GEMM(O) bwd: d_attn_out = d_proj @ W_o
    d_attn_out_3d = torch.empty(B, S, Q_DIM, dtype=dtype, device=device)
    gemm_o_bwd = GemmOp.schedule_backward(
        dout=d_proj, a=attn_out_3d, b=W_o, da=d_attn_out_3d, page_size=page_size)

    # FA bwd
    d_attn_fa = d_attn_out_3d.view(B, S, NUM_Q_HEADS, HEAD_DIM)
    dpsum = (d_attn_fa.float() * o_fa.float()).sum(dim=-1).permute(0, 2, 1).contiguous()
    dq_fa = torch.zeros(B, S, NUM_Q_HEADS, HEAD_DIM, dtype=torch.float32, device=device)
    dk_fa = torch.zeros(B, S, NUM_KV_HEADS, HEAD_DIM, dtype=dtype, device=device)
    dv_fa = torch.zeros(B, S, NUM_KV_HEADS, HEAD_DIM, dtype=dtype, device=device)

    fa_bwd = FlashAttentionSm120BwdOp.schedule(
        k=k_fa, v=v_fa, q=q_fa, dout=d_attn_fa, lse=lse, dpsum=dpsum,
        dq=dq_fa, dk=dk_fa, dv=dv_fa,
        causal=True, kv_group_size=KV_GROUP_SIZE,
    )
    for op in fa_bwd:
        op.dim_aliases["M"] = "seq"

    # Two separate megakernels (FA bwd needs different config)
    mlp_ops = gemm_down_bwd + glu_bwd + gemm_gu_bwd + rmsnorm2_bwd + gemm_o_bwd
    mlp_kernel = Megakernel(mlp_ops, config=GemmOp.kernel_config(mlp_ops))
    fa_kernel = Megakernel(fa_bwd, config=FlashAttentionSm120BwdOp.kernel_config(fa_bwd))

    with contextlib.redirect_stdout(io.StringIO()):
        mlp_kernel.run()
    torch.cuda.synchronize()

    # Now d_attn_out_3d is populated → recompute dpsum and run FA bwd
    dpsum[:] = (d_attn_fa.float() * o_fa.float()).sum(dim=-1).permute(0, 2, 1)
    dq_fa.zero_()
    dk_fa.zero_()
    dv_fa.zero_()

    with contextlib.redirect_stdout(io.StringIO()):
        fa_kernel.run()
    torch.cuda.synchronize()

    # --- Compare d_mlp_h (first op in backward chain) ---
    d_mlp_h_err = (d_mlp_h.float() - ref_d_mlp_h.float()).abs().mean() / ref_d_mlp_h.float().abs().mean()

    # --- Compare FA backward (dQ, dK, dV) against SDPA autograd ---
    q_ref = q_fa.detach().requires_grad_(True)
    k_ref = k_fa.detach().requires_grad_(True)
    v_ref = v_fa.detach().requires_grad_(True)
    q_ref_bhsd = q_ref.permute(0, 2, 1, 3)
    k_ref_bhsd = k_ref.repeat_interleave(KV_GROUP_SIZE, dim=2).permute(0, 2, 1, 3)
    v_ref_bhsd = v_ref.repeat_interleave(KV_GROUP_SIZE, dim=2).permute(0, 2, 1, 3)
    o_ref = F.scaled_dot_product_attention(q_ref_bhsd, k_ref_bhsd, v_ref_bhsd, is_causal=True)
    o_ref.backward(d_attn_fa.float().permute(0, 2, 1, 3))
    dq_err = (dq_fa.float() - q_ref.grad.float()).abs().mean() / q_ref.grad.float().abs().mean()
    dk_err = (dk_fa.float() - k_ref.grad.float()).abs().mean() / k_ref.grad.float().abs().mean()
    dv_err = (dv_fa.float() - v_ref.grad.float()).abs().mean() / v_ref.grad.float().abs().mean()

    # --- Compare final activation-side grads now that the full tail is wired ---
    ref_dx, ref_dresidual, *_ = sequential_layer_bwd(
        x, residual, w_attn_norm, W_q, W_k, W_v, w_q_norm, w_k_norm,
        cos, sin, W_o, w_mlp_norm, W_gate_up, W_down, d_out,
    )
    _, _, single_state = megakernel_layer_bwd_build(
        B, S, x, residual, w_attn_norm, W_q, W_k, W_v, w_q_norm, w_k_norm,
        cos, sin, W_o, w_mlp_norm, W_gate_up, W_down, page_size=page_size,
        d_out=d_out, return_state=True,
    )
    dx_err = (
        (single_state["dx"].float() - ref_dx.float()).abs().mean()
        / ref_dx.float().abs().mean()
    )
    dres_err = (
        (single_state["d_res1"].float() - ref_dresidual.float()).abs().mean()
        / ref_dresidual.float().abs().mean()
    )

    return {
        "d_mlp_h": d_mlp_h_err.item(),
        "dQ": dq_err.item(),
        "dK": dk_err.item(),
        "dV": dv_err.item(),
        "dx": dx_err.item(),
        "d_res1": dres_err.item(),
    }


@Benchmark.parametrize("batch", [1])
@Benchmark.parametrize("seq_len", [128, 256, 512, 1024, 2048, 4096])
@Benchmark.parametrize("page_size", PAGE_SIZES)
def bench_qwen35_layer_bwd(seq_len, batch, page_size):
    """Benchmark Qwen 3.5 full layer backward."""
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
    d_out = torch.randn(batch, seq_len, HIDDEN, dtype=dtype, device=device)

    funcs = {}

    # --- Sequential (full autograd backward) ---
    args = (x, residual, w_attn_norm, W_q, W_k, W_v, w_q_norm, w_k_norm,
            cos, sin, W_o, w_mlp_norm, W_gate_up, W_down, d_out)
    sequential_layer_bwd(*args)
    torch.cuda.synchronize()
    funcs["sequential"] = lambda: sequential_layer_bwd(*args)

    # --- Megakernel backward: single kernel ---
    if is_sm90_or_newer() and CUTLASS_AVAILABLE:
        try:
            spec_1k, _ = megakernel_layer_bwd_build(
                batch, seq_len, x, residual, w_attn_norm, W_q, W_k, W_v,
                w_q_norm, w_k_norm, cos, sin, W_o, w_mlp_norm, W_gate_up, W_down,
                page_size=page_size,
            )
            funcs["megakernel"] = spec_1k
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  layer bwd megakernel build failed: {e}")

    return funcs


# =============================================================================
# Benchmark Setup
# =============================================================================


def _total_bytes(seq_len, batch, page_size=32768):
    """Approximate total bytes moved for one layer forward."""
    elem = 2  # bf16
    # Activations: x, residual, h, q, k, v, attn_out, proj, h2, gate_up, mlp_h, out
    act_elems = (
        batch
        * seq_len
        * (
            HIDDEN * 6  # x, residual, h, proj, h2, out
            + Q_DIM * 2  # q, attn_out
            + KV_DIM * 2  # k, v
            + 2 * INTERMEDIATE  # gate_up
            + INTERMEDIATE  # mlp_h
        )
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


def _total_flops(seq_len, batch, page_size=32768):
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


@Benchmark.parametrize("batch", [1, 8])
@Benchmark.parametrize("seq_len", [128, 256, 512, 1024, 2048, 4096, 8192, 16384])
@Benchmark.parametrize("page_size", PAGE_SIZES)
def bench_qwen35_layer_fwd(seq_len, batch, page_size):
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
    args = (x, residual, w_attn_norm, W_q, W_k, W_v, w_q_norm, w_k_norm, cos, sin, W_o, w_mlp_norm, W_gate_up, W_down)
    # Warmup
    sequential_forward(*args)
    torch.cuda.synchronize()
    funcs["sequential"] = lambda: sequential_forward(*args)
    compiled_forward = maybe_compiled_forward(sequential_forward, args)
    if compiled_forward is not None:
        funcs["torch_compile"] = lambda: compiled_forward(*args)

    # --- Megakernel: single kernel ---
    if is_sm90_or_newer() and CUTLASS_AVAILABLE:
        build_args = (batch, seq_len, x, residual, w_attn_norm, W_q, W_k, W_v,
                      w_q_norm, w_k_norm, cos, sin, W_o, w_mlp_norm, W_gate_up, W_down)
        try:
            spec_1k, _, _ = megakernel_forward_build(*build_args, page_size=page_size)
            funcs["megakernel"] = spec_1k
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  megakernel build failed: {e}")

    return funcs


def sequential_no_attn(
    x, residual, w_attn_norm, W_q, W_k, W_v, w_q_norm, w_k_norm, cos, sin, W_o, w_mlp_norm, W_gate_up, W_down
):
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


@Benchmark.parametrize("batch", [1, 8])
@Benchmark.parametrize("seq_len", [128, 256, 512, 1024, 2048, 4096, 8192, 16384])
@Benchmark.parametrize("page_size", PAGE_SIZES)
def bench_qwen35_no_attn(seq_len, batch, page_size):
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
    args = (x, residual, w_attn_norm, W_q, W_k, W_v, w_q_norm, w_k_norm, cos, sin, W_o, w_mlp_norm, W_gate_up, W_down)

    sequential_no_attn(*args)
    torch.cuda.synchronize()
    funcs["sequential"] = lambda: sequential_no_attn(*args)
    compiled_forward = maybe_compiled_forward(sequential_no_attn, args)
    if compiled_forward is not None:
        funcs["torch_compile"] = lambda: compiled_forward(*args)

    if is_sm90_or_newer() and CUTLASS_AVAILABLE:
        try:
            from machete.megakernel import Megakernel
            from machete.kernels.gemm import GemmOp
            from machete.kernels.rms_norm import RMSNormOp
            from machete.kernels.glu import GLUOp
            from machete.kernels.qknorm_rope import PackedQKNormRopeOp

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
            W_qkv = torch.cat([W_q, W_k, W_v], dim=0).contiguous()
            qkv_3d = torch.empty(batch, seq_len, Q_DIM + 2 * KV_DIM, dtype=dtype, device=device)
            q_3d = qkv_3d[:, :, :Q_DIM]
            k_3d = qkv_3d[:, :, Q_DIM:Q_DIM + KV_DIM]
            v_3d = qkv_3d[:, :, Q_DIM + KV_DIM:]
            q_4d = q_3d.view(batch * seq_len, NUM_Q_HEADS, HEAD_DIM)
            k_4d = k_3d.view(batch * seq_len, NUM_KV_HEADS, HEAD_DIM)
            qk_4d = qkv_3d[:, :, :Q_DIM + KV_DIM].view(batch * seq_len, NUM_Q_HEADS + NUM_KV_HEADS, HEAD_DIM)
            attn_out = q_3d[:, :, :Q_DIM]  # dummy
            proj = torch.empty_like(x)
            res_out2 = torch.empty_like(x)
            h2 = torch.empty_like(x)
            gate_up = torch.empty(batch, seq_len, 2 * INTERMEDIATE, dtype=dtype, device=device)
            mlp_h = torch.empty(batch, seq_len, INTERMEDIATE, dtype=dtype, device=device)
            out = torch.empty_like(x)
            gemm_tiles = _pick_single_layer_fused_gemm_tiles(batch, seq_len, page_size)

            all_ops = (
                RMSNormOp.schedule(
                    x=x,
                    weight=w_attn_norm,
                    y=h,
                    residual_in=residual,
                    residual_out=res_out,
                    page_size=page_size,
                )
                + GemmOp.schedule(a=h, b=W_qkv, c=qkv_3d, page_size=page_size, tile_sizes=gemm_tiles.get("qkv"))
                + PackedQKNormRopeOp.schedule(
                    qk=qk_4d,
                    q_norm_weight=w_q_norm,
                    k_norm_weight=w_k_norm,
                    cos=cos,
                    sin=sin,
                    page_size=page_size,
                    num_q_heads=NUM_Q_HEADS,
                    num_k_heads=NUM_KV_HEADS,
                )
                + GemmOp.schedule(a=attn_out, b=W_o, c=proj, page_size=page_size, tile_sizes=gemm_tiles.get("o"))
                + RMSNormOp.schedule(
                    x=proj,
                    weight=w_mlp_norm,
                    y=h2,
                    residual_in=res_out,
                    residual_out=res_out2,
                    page_size=page_size,
                )
                + GemmOp.schedule(a=h2, b=W_gate_up, c=gate_up, page_size=page_size, tile_sizes=gemm_tiles.get("gate_up"))
                + GLUOp.schedule(x=gate_up, y=mlp_h, activation="silu", page_size=page_size)
                + GemmOp.schedule(a=mlp_h, b=W_down, c=out, page_size=page_size, tile_sizes=gemm_tiles.get("down"))
            )
            config = GemmOp.kernel_config(all_ops)
            kernel = Megakernel(all_ops, config=config)

            with contextlib.redirect_stdout(io.StringIO()):
                kernel.run()
            torch.cuda.synchronize()

            funcs["megakernel"] = kernel.bench_spec(
                keep_alive=[
                    x,
                    residual,
                    w_attn_norm,
                    W_q,
                    W_k,
                    W_v,
                    W_qkv,
                    w_q_norm,
                    w_k_norm,
                    cos,
                    sin,
                    W_o,
                    w_mlp_norm,
                    W_gate_up,
                    W_down,
                    res_out,
                    h,
                    qkv_3d,
                    q_3d,
                    k_3d,
                    v_3d,
                    q_4d,
                    k_4d,
                    qk_4d,
                    attn_out,
                    proj,
                    res_out2,
                    h2,
                    gate_up,
                    mlp_h,
                    out,
                ]
            )
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

    args = (x, residual, w_attn_norm, W_q, W_k, W_v, w_q_norm, w_k_norm, cos, sin, W_o, w_mlp_norm, W_gate_up, W_down)

    # Sequential reference
    ref_out, ref_res = sequential_forward(*args)
    torch.cuda.synchronize()

    # Megakernel (outputs populated during build's warmup run)
    _, mk_out, mk_res = megakernel_forward_build(B, S, *args)

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
        print("Correctness Verification (Forward)")
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
        print("Correctness Verification (Layer Backward)")
        print("-" * 80)
        for s in [128, 512]:
            try:
                errs = verify_layer_backward(1, s)
                max_err = max(errs.values())
                status = "PASS" if max_err < 0.1 else "FAIL"
                parts = ", ".join(f"{k}={v:.4f}" for k, v in errs.items())
                print(f"  S={s}: {parts} [{status}]")
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
    print("Backward Pass (full layer)")
    print("-" * 80)
    bench_qwen35_layer_bwd._benchmark.run(
        mode="kernel",
        flops=_layer_bwd_flops,
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
