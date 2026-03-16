#!/usr/bin/env python
# Copyright (c) 2025, Machete Authors
"""Benchmark RoPE + Gated Delta Net: Fused Megakernel vs Sequential Launches.

Compares GPU kernel execution time of:
- fla: flash-linear-attention Triton kernels (RoPE via PyTorch + fla GDN)
- sequential: RoPE megakernel + GDN megakernel (2 kernel launches)
- fused: RoPE + GDN all in one megakernel (1 kernel launch)

GDN automatically uses 2-op (Prep+Fused) or 5-op decomposition depending
on page_size constraints.

Usage:
    python benchmarks/kernels/benchmark_rope_gdn.py
"""

import contextlib
import io

import torch

from machete.utils.benchmark import Benchmark
from machete.utils.benchmark_utils import KernelBenchSpec

try:
    import fla  # noqa: F401
    FLA_AVAILABLE = True
except ImportError:
    FLA_AVAILABLE = False
    print("WARNING: fla not available — fla benchmarks will be skipped.")

try:
    import cutlass  # noqa: F401
    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False

try:
    from machete.kernels.gated_delta_net import HAS_MEGAKERNEL_OPS, HAS_5OP
except ImportError:
    HAS_MEGAKERNEL_OPS = False
    HAS_5OP = False

PAGE_SIZES = [16384, 32768, 49152]
# GDN ops require >= 48KB for K=128 — only use 48KB for GDN benchmarks
GDN_PAGE_SIZES = [49152]


def is_hopper_or_newer():
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


def _make_inputs(B, T, H, K, V, dtype=torch.float16, device="cuda"):
    """Create random inputs for RoPE + GDN."""
    q = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    k = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    v = torch.randn(B, T, H, V, dtype=dtype, device=device) * 0.1
    g = -torch.rand(B, T, H, dtype=torch.float32, device=device) * 2.0
    beta = torch.sigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device))
    cos = torch.randn(T, K // 2, dtype=dtype, device=device)
    sin = torch.randn(T, K // 2, dtype=dtype, device=device)
    return q, k, v, g, beta, cos, sin


def _schedule_gdn_ops(q, k, v, g, beta, scale, page_size):
    """Schedule GDN ops: 2-op (Prep+Fused) if page_size allows, else 5-op.

    Returns (ops, config, bufs_to_zero) where bufs_to_zero includes o.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    dtype = q.dtype

    gc = torch.zeros(B, T, H, device=q.device, dtype=torch.float32)
    w = torch.zeros(B, T, H, K, device=q.device, dtype=dtype)
    u = torch.zeros(B, T, H, V, device=q.device, dtype=dtype)
    o = torch.zeros(B, T, H, V, device=q.device, dtype=dtype)

    # Try 2-op (Prep + Fused)
    try:
        from machete.kernels.gated_delta_net.prep_op import GDNPrepOp
        from machete.kernels.gated_delta_net.fused_op import GDNFusedOp

        prep_ops = GDNPrepOp.schedule_forward(
            k=k, v=v, g=g, beta=beta,
            g_cumsum=gc, w=w, u=u, page_size=page_size,
        )
        fused_ops = GDNFusedOp.schedule_forward(
            q=q, k=k, w=w, u=u,
            g_cumsum=gc, o=o, scale=scale, page_size=page_size,
        )
        ops = prep_ops + fused_ops
        config = GDNFusedOp.kernel_config(ops)
        return ops, config, [gc, w, u, o]
    except (ValueError, RuntimeError):
        if not HAS_5OP:
            raise

    # Fallback to 5-op decomposition
    from machete.kernels.gated_delta_net.solve_op import GDNSolveOp
    from machete.kernels.gated_delta_net.wu_op import GDNWUOp
    from machete.kernels.gated_delta_net.state_recurrence_op import GDNStateRecurrenceOp
    from machete.kernels.gated_delta_net.vnew_op import GDNVNewOp
    from machete.kernels.gated_delta_net.output_op import GDNOutputOp

    NT = T // 64
    a_solved = torch.zeros(B, T, H, 64, device=q.device, dtype=dtype)
    v_new = torch.zeros(B, T, H, V, device=q.device, dtype=dtype)
    h_states = torch.zeros(B, NT, H, K, V, device=q.device, dtype=dtype)

    solve_ops = GDNSolveOp.schedule_forward(
        k=k, g=g, beta=beta,
        g_cumsum=gc, a_solved=a_solved, page_size=page_size,
    )
    wu_ops = GDNWUOp.schedule_forward(
        a_solved=a_solved, k=k, v=v,
        g_cumsum=gc, beta=beta, w=w, u=u, page_size=page_size,
    )
    state_ops = GDNStateRecurrenceOp.schedule_forward(
        k=k, w=w, u=u, g_cumsum=gc,
        h_states=h_states, page_size=page_size,
    )
    vnew_ops = GDNVNewOp.schedule_forward(
        w=w, u=u, h_states=h_states,
        v_new=v_new, page_size=page_size,
    )
    output_ops = GDNOutputOp.schedule_forward(
        q=q, k=k, v_new=v_new, h_states=h_states,
        g_cumsum=gc, o=o, scale=scale, page_size=page_size,
    )

    ops = solve_ops + wu_ops + state_ops + vnew_ops + output_ops
    config = GDNOutputOp.kernel_config(ops)
    return ops, config, [gc, a_solved, w, u, v_new, h_states, o]


def _build_rope_megakernel(q, k, cos, sin, page_size):
    """Build megakernel with RoPE for q and k (2 RoPE ops)."""
    from machete.megakernel import Megakernel
    from machete.kernels.rope import RopeOp

    rope_q_ops = RopeOp.schedule_forward(q=q, cos=cos, sin=sin, page_size=page_size)
    rope_k_ops = RopeOp.schedule_forward(q=k, cos=cos, sin=sin, page_size=page_size)
    all_ops = rope_q_ops + rope_k_ops
    config = RopeOp.kernel_config(all_ops)
    kernel = Megakernel(all_ops, config=config)
    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()
    torch.cuda.synchronize()
    return kernel


def _build_gdn_megakernel(q, k, v, g, beta, scale, page_size):
    """Build megakernel with GDN (2-op or 5-op fallback)."""
    from machete.megakernel import Megakernel

    gdn_ops, gdn_config, bufs = _schedule_gdn_ops(
        q, k, v, g, beta, scale, page_size)
    kernel = Megakernel(gdn_ops, config=gdn_config)
    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()
    torch.cuda.synchronize()
    return kernel, bufs


def _build_fused_megakernel(q, k, v, g, beta, cos, sin, scale, page_size):
    """Build single megakernel: RoPE(q) + RoPE(k) + GDN (2-op or 5-op)."""
    from machete.megakernel import Megakernel, MegakernelConfig
    from machete.kernels.rope import RopeOp

    rope_q_ops = RopeOp.schedule_forward(q=q, cos=cos, sin=sin, page_size=page_size)
    rope_k_ops = RopeOp.schedule_forward(q=k, cos=cos, sin=sin, page_size=page_size)

    gdn_ops, gdn_config, bufs = _schedule_gdn_ops(
        q, k, v, g, beta, scale, page_size)

    all_ops = rope_q_ops + rope_k_ops + gdn_ops
    rope_config = RopeOp.kernel_config(rope_q_ops + rope_k_ops)
    config = MegakernelConfig(
        threads_per_block=max(rope_config.threads_per_block,
                              gdn_config.threads_per_block),
        page_size=max(rope_config.page_size, gdn_config.page_size),
    )
    kernel = Megakernel(all_ops, config=config)
    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()
    torch.cuda.synchronize()
    return kernel, bufs


# =============================================================================
# Pipeline benchmark: RoPE + GDN fused vs sequential
# =============================================================================

_BASE_CONFIGS = [
    # (B, T, H, K, V)
    (1, 2048, 32, 128, 128),
    (1, 4096, 32, 128, 128),
    (1, 8192, 32, 128, 128),
    (4, 2048, 32, 128, 128),
    (4, 4096, 32, 128, 128),
    (8, 2048, 32, 128, 128),
]

CONFIGS = [c + (ps,) for c in _BASE_CONFIGS for ps in GDN_PAGE_SIZES]


@Benchmark.configs(["B", "T", "H", "K", "V", "page_size"], CONFIGS)
def bench_rope_gdn(B, T, H, K, V, page_size):
    """Benchmark RoPE + GDN: fused megakernel vs sequential launches."""
    q, k, v, g, beta, cos, sin = _make_inputs(B, T, H, K, V)
    scale = K ** -0.5

    funcs = {}

    # --- fla baseline: PyTorch RoPE + fla GDN ---
    if FLA_AVAILABLE:
        from machete.kernels.rope.ref import rope_pytorch
        from machete.kernels.gated_delta_net.ref import fla_full_forward

        q_fla = q.clone()
        k_fla = k.clone()

        def fla_fn():
            q_fla.copy_(q)
            k_fla.copy_(k)
            rope_pytorch(q_fla, cos, sin)
            rope_pytorch(k_fla, cos, sin)
            return fla_full_forward(q_fla, k_fla, v, g, beta, scale=scale)

        funcs["fla"] = fla_fn

    if not (is_hopper_or_newer() and CUTLASS_AVAILABLE and HAS_MEGAKERNEL_OPS):
        return funcs

    import cuda.bindings.driver as cuda

    # --- Sequential ---
    try:
        print(f"  Building sequential (B={B}, T={T}, pg={page_size//1024}K)...", end="", flush=True)
        q_seq = q.clone().contiguous()
        k_seq = k.clone().contiguous()
        v_seq = v.contiguous()
        g_seq = g.contiguous()
        beta_seq = beta.contiguous()

        rope_kernel = _build_rope_megakernel(q_seq, k_seq, cos, sin, page_size=page_size)
        gdn_kernel, gdn_bufs_seq = _build_gdn_megakernel(
            q_seq, k_seq, v_seq, g_seq, beta_seq, scale, page_size=page_size)

        seq_stream = torch.cuda.Stream()
        seq_cu_stream = cuda.CUstream(seq_stream.cuda_stream)

        funcs["sequential"] = KernelBenchSpec(
            launch_fn=lambda rk=rope_kernel, gk=gdn_kernel, sc=seq_cu_stream: (
                rk.run(stream=sc, sync=False), gk.run(stream=sc, sync=False)),
            setup_fn=lambda qs=q_seq, ks=k_seq, bufs=gdn_bufs_seq: (
                qs.copy_(q), ks.copy_(k), [b.zero_() for b in bufs]),
            stream=(seq_stream, seq_cu_stream),
            _keep_alive=[rope_kernel, gdn_kernel, q_seq, k_seq, v_seq,
                         g_seq, beta_seq, cos, sin] + gdn_bufs_seq,
        )
        print(" ok", flush=True)
    except Exception as e:
        print(f" FAILED: {e}", flush=True)

    # --- Fused ---
    try:
        print(f"  Building fused (B={B}, T={T}, pg={page_size//1024}K)...", end="", flush=True)
        q_fused = q.clone().contiguous()
        k_fused = k.clone().contiguous()
        v_fused = v.contiguous()
        g_fused = g.contiguous()
        beta_fused = beta.contiguous()

        fused_kernel, fused_bufs = _build_fused_megakernel(
            q_fused, k_fused, v_fused, g_fused, beta_fused,
            cos, sin, scale, page_size=page_size)

        fused_stream = torch.cuda.Stream()
        fused_cu_stream = cuda.CUstream(fused_stream.cuda_stream)

        funcs["fused"] = KernelBenchSpec(
            launch_fn=lambda fk=fused_kernel, fc=fused_cu_stream: fk.run(
                stream=fc, sync=False),
            setup_fn=lambda qf=q_fused, kf=k_fused, bufs=fused_bufs: (
                qf.copy_(q), kf.copy_(k), [b.zero_() for b in bufs]),
            stream=(fused_stream, fused_cu_stream),
            _keep_alive=[fused_kernel, q_fused, k_fused, v_fused,
                         g_fused, beta_fused, cos, sin] + fused_bufs,
        )
        print(" ok", flush=True)
    except Exception as e:
        print(f" FAILED: {e}", flush=True)

    return funcs


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 100)
    print("RoPE + Gated Delta Net Benchmark: Fused Megakernel vs Sequential")
    print("=" * 100)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.multi_processor_count} SMs)")
        print(f"Hopper+: {is_hopper_or_newer()}")
    print(f"CUTLASS: {CUTLASS_AVAILABLE}")
    print(f"fla: {FLA_AVAILABLE}")
    print()

    print("-" * 100)
    print("RoPE(q) + RoPE(k) + GDN Prep + GDN Fused")
    print("-" * 100)
    bench_rope_gdn._benchmark.run(
        mode="kernel", warmup=10, rep=100,
        columns=["fla", "sequential", "fused"],
    )
