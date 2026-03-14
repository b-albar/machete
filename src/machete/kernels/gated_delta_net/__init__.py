# Copyright (c) 2025, Machete Authors
"""Gated Delta Net linear attention kernels (chunked algorithm).

Implements the Gated Delta Rule recurrence used in Qwen 3.5 and similar
architectures. The algorithm is decomposed into three pipelined megakernel
passes: prep (parallel), state recurrence (sequential, TMA-pipelined),
and output (parallel).

Forward pipeline: prep → state_recurrence → output
Backward pipeline: dv_local → bwd_state → dqkwg → wy_bwd → reverse_cumsum

Usage:
    from machete.kernels.gated_delta_net import chunk_gated_delta_rule

    o, final_state = chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta,
        scale=scale, initial_state=initial_state,
        output_final_state=True,
    )

Reference: fla-org/flash-linear-attention (Triton implementation).
"""

import torch

from machete.kernels.gated_delta_net.prep import run_prep
from machete.kernels.gated_delta_net.state import run_state_recurrence
from machete.kernels.gated_delta_net.output import run_output
from machete.kernels.gated_delta_net.state_bwd import run_bwd_state_recurrence
from machete.kernels.gated_delta_net.grad import (
    run_bwd_dv_local,
    run_bwd_dqkwg,
    run_bwd_wy,
    reverse_cumsum,
)

# Megakernel Ops (fusable with other Ops)
try:
    from machete.kernels.gated_delta_net.prep_op import GDNPrepOp
    from machete.kernels.gated_delta_net.fused_op import GDNFusedOp, GDNFusedBwdOp
    HAS_MEGAKERNEL_OPS = True
except ImportError:
    HAS_MEGAKERNEL_OPS = False

# 5-Op decomposition (Solve + WU + StateRecurrence + VNew + Output)
try:
    from machete.kernels.gated_delta_net.solve_op import GDNSolveOp
    from machete.kernels.gated_delta_net.wu_op import GDNWUOp
    from machete.kernels.gated_delta_net.state_recurrence_op import GDNStateRecurrenceOp
    from machete.kernels.gated_delta_net.vnew_op import GDNVNewOp
    from machete.kernels.gated_delta_net.output_op import GDNOutputOp
    HAS_5OP = True
except ImportError:
    HAS_5OP = False


def _run_fused_megakernel(q, k, v, g, beta, scale):
    """Run full forward as a single fused megakernel: PrepOp → FusedOp.

    All tensors use native [B, T, H, K/V] layout — no transposes.
    """
    from machete.megakernel import Megakernel

    B, T, H, K = q.shape
    V = v.shape[-1]
    dtype = q.dtype

    # Ensure contiguous inputs
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()

    # Allocate intermediate and output buffers (native layout)
    g_cumsum = torch.zeros(B, T, H, device=q.device, dtype=torch.float32)
    w = torch.zeros(B, T, H, K, device=q.device, dtype=dtype)
    u = torch.zeros(B, T, H, V, device=q.device, dtype=dtype)
    o = torch.zeros(B, T, H, V, device=q.device, dtype=dtype)

    # Schedule PrepOp + FusedOp — dependencies resolved via shared tensors
    prep_ops = GDNPrepOp.schedule_forward(
        k=k, v=v, g=g, beta=beta,
        g_cumsum=g_cumsum, w=w, u=u,
    )
    fused_ops = GDNFusedOp.schedule_forward(
        q=q, k=k, w=w, u=u,
        g_cumsum=g_cumsum, o=o, scale=scale,
    )

    all_ops = prep_ops + fused_ops
    config = GDNFusedOp.kernel_config(all_ops)
    Megakernel(all_ops, config=config).run()

    return o, g_cumsum, None, w, u


def _run_fused_bwd_megakernel(q, k, w, g_cumsum, do, scale):
    """Run backward stages 1+2 as a single fused megakernel: GDNFusedBwdOp.

    Fuses dv_local (Stage 1) + backward state recurrence (Stage 2).
    Returns dv2 (accumulated value gradient).
    """
    from machete.megakernel import Megakernel

    B, T, H, K = q.shape
    V = do.shape[-1]
    dtype = q.dtype

    q = q.contiguous()
    k = k.contiguous()
    w = w.contiguous()
    g_cumsum = g_cumsum.contiguous()
    do = do.contiguous()

    dv = torch.zeros(B, T, H, V, device=q.device, dtype=dtype)

    ops = GDNFusedBwdOp.schedule_forward(
        q=q, k=k, w=w, g_cumsum=g_cumsum, do=do, dv=dv, scale=scale,
    )
    config = GDNFusedBwdOp.kernel_config(ops)
    Megakernel(ops, config=config).run()

    return dv


def _run_5op_megakernel(q, k, v, g, beta, scale, page_size=None):
    """Run full forward as a 5-op fused megakernel: Solve → WU → StateRecurrence → VNew → Output.

    Splits StateOp into sequential StateRecurrence (h_states only) + parallel VNew (v_new = u - w@h).
    All tensors use native [B, T, H, K/V] layout — no transposes.
    """
    from machete.megakernel import Megakernel
    from machete.megakernel.ops import DEFAULT_PAGE_SIZE

    if page_size is None:
        page_size = DEFAULT_PAGE_SIZE

    B, T, H, K = q.shape
    V = v.shape[-1]
    NT = T // 64
    dtype = q.dtype

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()

    # Allocate intermediates and output
    g_cumsum = torch.zeros(B, T, H, device=q.device, dtype=torch.float32)
    a_solved = torch.zeros(B, T, H, 64, device=q.device, dtype=dtype)
    w = torch.zeros(B, T, H, K, device=q.device, dtype=dtype)
    u = torch.zeros(B, T, H, V, device=q.device, dtype=dtype)
    v_new = torch.zeros(B, T, H, V, device=q.device, dtype=dtype)
    h_states = torch.zeros(B, NT, H, K, V, device=q.device, dtype=dtype)
    o = torch.zeros(B, T, H, V, device=q.device, dtype=dtype)

    # Schedule all 5 ops
    solve_ops = GDNSolveOp.schedule_forward(
        k=k, g=g, beta=beta,
        g_cumsum=g_cumsum, a_solved=a_solved,
        page_size=page_size,
    )
    wu_ops = GDNWUOp.schedule_forward(
        a_solved=a_solved, k=k, v=v,
        g_cumsum=g_cumsum, beta=beta,
        w=w, u=u,
        page_size=page_size,
    )
    state_ops = GDNStateRecurrenceOp.schedule_forward(
        k=k, w=w, u=u, g_cumsum=g_cumsum,
        h_states=h_states,
        page_size=page_size,
    )
    vnew_ops = GDNVNewOp.schedule_forward(
        w=w, u=u, h_states=h_states,
        v_new=v_new,
        page_size=page_size,
    )
    output_ops = GDNOutputOp.schedule_forward(
        q=q, k=k, v_new=v_new, h_states=h_states,
        g_cumsum=g_cumsum, o=o, scale=scale,
        page_size=page_size,
    )

    all_ops = solve_ops + wu_ops + state_ops + vnew_ops + output_ops
    config = GDNOutputOp.kernel_config(all_ops)
    Megakernel(all_ops, config=config).run()

    return o


def _forward(q, k, v, g, beta, scale, initial_state, output_final_state):
    """Run full forward pipeline: prep → state → output."""
    g_cumsum, A, w, u = run_prep(k, v, g, beta)
    h, v_new = run_state_recurrence(k, w, u, g_cumsum, initial_state=initial_state)
    o = run_output(q, k, v_new, h, g_cumsum, scale=scale)
    return o, g_cumsum, A, w, u, h, v_new


def _backward(q, k, v, g_cumsum, beta, A, w, u, h, v_new, scale, initial_state, do, dht):
    """Run full backward pipeline."""
    # Stage 1: local dv from causal attention backward
    dv_local = run_bwd_dv_local(q, k, g_cumsum, do, scale)

    # Stage 2: backward state recurrence + weight gradient
    dh, dh0, dv2, dw = run_bwd_state_recurrence(
        q, k, w, g_cumsum, initial_state, dht, do, dv_local, h, scale,
    )

    # Stage 3: dq, dk, dg from attention + gate backward
    dq, dk, _, dg = run_bwd_dqkwg(
        q, k, v_new, w, g_cumsum, h, dv2, do, dh, scale,
    )

    # Stage 4: backward through WY representation
    dk2, dv, db, dg2 = run_bwd_wy(k, v, beta, g_cumsum, A, dw, dv2)

    # Accumulate
    dk = dk + dk2
    dg = dg + dg2

    # Stage 5: reverse cumsum of dg
    dg = reverse_cumsum(dg)

    return dq, dk, dv, db, dg, dh0


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    """Autograd wrapper for the chunked Gated Delta Rule."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
    ):
        o, g_cumsum, A, w, u, h, v_new = _forward(
            q, k, v, g, beta, scale, initial_state, output_final_state,
        )
        # Save tensors for backward
        ctx.save_for_backward(q, k, v, g_cumsum, beta, A, initial_state)
        ctx.scale = scale
        return o, None  # None for final_state (not yet implemented)

    @staticmethod
    def backward(ctx, do, _dht_unused):
        q, k, v, g_cumsum, beta, A, initial_state = ctx.saved_tensors
        scale = ctx.scale

        # Recompute w, u, h, v_new for backward (standard recomputation pattern)
        w, u = _recompute_w_u(k, v, beta, A, g_cumsum)
        h, v_new = run_state_recurrence(k, w, u, g_cumsum, initial_state=initial_state)

        dq, dk, dv, db, dg, dh0 = _backward(
            q, k, v, g_cumsum, beta, A, w, u, h, v_new, scale,
            initial_state, do, dht=None,
        )
        return dq, dk, dv, dg, db, None, dh0, None


def _recompute_w_u(k, v, beta, A, g_cumsum):
    """Recompute w, u from saved A and g_cumsum."""
    from machete.kernels.gated_delta_net.prep import _recompute_w_u
    return _recompute_w_u(k, v, beta, A, g_cumsum)


@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Chunked Gated Delta Rule forward with autograd backward.

    Args:
        q: Queries [B, T, H, K] (fp16/bf16)
        k: Keys [B, T, H, K] (fp16/bf16)
        v: Values [B, T, H, V] (fp16/bf16)
        g: Log-space gates [B, T, H] (fp32, always <= 0)
        beta: Beta values [B, T, H] (fp32, in [0,1])
        scale: Attention scale (default: K^-0.5)
        initial_state: Initial hidden state [B, H, K, V] (optional)
        output_final_state: Whether to output the final state

    Returns:
        o: Output [B, T, H, V] (same dtype as q)
        final_state: Final hidden state [B, H, K, V] or None
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q, k, v, g, beta, scale, initial_state, output_final_state,
    )
    return o, final_state


__all__ = [
    "chunk_gated_delta_rule",
    "HAS_MEGAKERNEL_OPS",
    "HAS_5OP",
    "_run_fused_megakernel",
    "_run_fused_bwd_megakernel",
    "_run_5op_megakernel",
]
