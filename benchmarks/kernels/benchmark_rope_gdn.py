#!/usr/bin/env python
# Copyright (c) 2025, Machete Authors
"""Benchmark RoPE + Gated Delta Net: Fused Megakernel vs Sequential Launches.

Compares GPU kernel execution time of:
- fla: flash-linear-attention Triton kernels (RoPE via PyTorch + fla GDN)
- sequential: RoPE megakernel + GDN megakernel (2 kernel launches)
- fused: RoPE + GDN all in one megakernel (1 kernel launch)

Also profiles individual ops to identify bottleneck contributions.

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
    from machete.kernels.gated_delta_net import HAS_MEGAKERNEL_OPS
except ImportError:
    HAS_MEGAKERNEL_OPS = False

PAGE_SIZES = [16384, 32768, 49152]


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
    """Build megakernel with GDN prep + fused (2 ops)."""
    from machete.megakernel import Megakernel
    from machete.kernels.gated_delta_net.prep_op import GDNPrepOp
    from machete.kernels.gated_delta_net.fused_op import GDNFusedOp

    B, T, H, K = q.shape
    V = v.shape[-1]
    dtype = q.dtype

    gc = torch.zeros(B, T, H, device=q.device, dtype=torch.float32)
    w = torch.zeros(B, T, H, K, device=q.device, dtype=dtype)
    u = torch.zeros(B, T, H, V, device=q.device, dtype=dtype)
    o = torch.zeros(B, T, H, V, device=q.device, dtype=dtype)

    prep_ops = GDNPrepOp.schedule_forward(
        k=k, v=v, g=g, beta=beta,
        g_cumsum=gc, w=w, u=u, page_size=page_size,
    )
    fused_ops = GDNFusedOp.schedule_forward(
        q=q, k=k, w=w, u=u,
        g_cumsum=gc, o=o, scale=scale, page_size=page_size,
    )
    all_ops = prep_ops + fused_ops
    config = GDNFusedOp.kernel_config(all_ops)
    kernel = Megakernel(all_ops, config=config)
    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()
    torch.cuda.synchronize()
    return kernel, gc, w, u, o


def _build_fused_megakernel(q, k, v, g, beta, cos, sin, scale, page_size):
    """Build single megakernel: RoPE(q) + RoPE(k) + GDN prep + GDN fused."""
    from machete.megakernel import Megakernel
    from machete.kernels.rope import RopeOp
    from machete.kernels.gated_delta_net.prep_op import GDNPrepOp
    from machete.kernels.gated_delta_net.fused_op import GDNFusedOp

    B, T, H, K = q.shape
    V = v.shape[-1]
    dtype = q.dtype

    gc = torch.zeros(B, T, H, device=q.device, dtype=torch.float32)
    w = torch.zeros(B, T, H, K, device=q.device, dtype=dtype)
    u = torch.zeros(B, T, H, V, device=q.device, dtype=dtype)
    o = torch.zeros(B, T, H, V, device=q.device, dtype=dtype)

    rope_q_ops = RopeOp.schedule_forward(q=q, cos=cos, sin=sin, page_size=page_size)
    rope_k_ops = RopeOp.schedule_forward(q=k, cos=cos, sin=sin, page_size=page_size)
    prep_ops = GDNPrepOp.schedule_forward(
        k=k, v=v, g=g, beta=beta,
        g_cumsum=gc, w=w, u=u, page_size=page_size,
    )
    fused_ops = GDNFusedOp.schedule_forward(
        q=q, k=k, w=w, u=u,
        g_cumsum=gc, o=o, scale=scale, page_size=page_size,
    )

    all_ops = rope_q_ops + rope_k_ops + prep_ops + fused_ops
    config = GDNFusedOp.kernel_config(all_ops)
    kernel = Megakernel(all_ops, config=config)
    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()
    torch.cuda.synchronize()
    return kernel, gc, w, u, o


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

CONFIGS = [c + (ps,) for c in _BASE_CONFIGS for ps in PAGE_SIZES]


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
        q_seq = q.clone().contiguous()
        k_seq = k.clone().contiguous()
        v_seq = v.contiguous()
        g_seq = g.contiguous()
        beta_seq = beta.contiguous()

        rope_kernel = _build_rope_megakernel(q_seq, k_seq, cos, sin, page_size=page_size)
        gdn_kernel, gc_seq, w_seq, u_seq, o_seq = _build_gdn_megakernel(
            q_seq, k_seq, v_seq, g_seq, beta_seq, scale, page_size=page_size)

        seq_stream = torch.cuda.Stream()
        seq_cu_stream = cuda.CUstream(seq_stream.cuda_stream)

        funcs["sequential"] = KernelBenchSpec(
            launch_fn=lambda rk=rope_kernel, gk=gdn_kernel, sc=seq_cu_stream: (
                rk.run(stream=sc, sync=False), gk.run(stream=sc, sync=False)),
            setup_fn=lambda qs=q_seq, ks=k_seq, gc=gc_seq, ws=w_seq, us=u_seq, os_=o_seq: (
                qs.copy_(q), ks.copy_(k), gc.zero_(), ws.zero_(), us.zero_(), os_.zero_()),
            stream=(seq_stream, seq_cu_stream),
            _keep_alive=[rope_kernel, gdn_kernel, q_seq, k_seq, v_seq,
                         g_seq, beta_seq, gc_seq, w_seq, u_seq, o_seq, cos, sin],
        )
    except Exception:
        pass

    # --- Fused ---
    try:
        q_fused = q.clone().contiguous()
        k_fused = k.clone().contiguous()
        v_fused = v.contiguous()
        g_fused = g.contiguous()
        beta_fused = beta.contiguous()

        fused_kernel, gc_f, w_f, u_f, o_f = _build_fused_megakernel(
            q_fused, k_fused, v_fused, g_fused, beta_fused,
            cos, sin, scale, page_size=page_size)

        fused_stream = torch.cuda.Stream()
        fused_cu_stream = cuda.CUstream(fused_stream.cuda_stream)

        funcs["fused"] = KernelBenchSpec(
            launch_fn=lambda fk=fused_kernel, fc=fused_cu_stream: fk.run(
                stream=fc, sync=False),
            setup_fn=lambda qf=q_fused, kf=k_fused, gc=gc_f, wf=w_f, uf=u_f, of=o_f: (
                qf.copy_(q), kf.copy_(k), gc.zero_(), wf.zero_(), uf.zero_(), of.zero_()),
            stream=(fused_stream, fused_cu_stream),
            _keep_alive=[fused_kernel, q_fused, k_fused, v_fused,
                         g_fused, beta_fused, gc_f, w_f, u_f, o_f, cos, sin],
        )
    except Exception:
        pass

    return funcs


# =============================================================================
# Individual op timing (to identify bottleneck contributions)
# =============================================================================

_BASE_INDIV_CONFIGS = [
    (4, 2048, 32, 128, 128),
    (8, 2048, 32, 128, 128),
]

INDIV_CONFIGS = [c + (ps,) for c in _BASE_INDIV_CONFIGS for ps in PAGE_SIZES]


@Benchmark.configs(["B", "T", "H", "K", "V", "page_size"], INDIV_CONFIGS)
def bench_individual_ops(B, T, H, K, V, page_size):
    """Time individual ops to identify bottleneck contributions."""
    q, k, v, g, beta, cos, sin = _make_inputs(B, T, H, K, V)
    scale = K ** -0.5
    dtype = q.dtype

    funcs = {}

    if not (is_hopper_or_newer() and CUTLASS_AVAILABLE and HAS_MEGAKERNEL_OPS):
        return funcs

    from machete.megakernel import Megakernel
    from machete.kernels.rope import RopeOp
    from machete.kernels.gated_delta_net.prep_op import GDNPrepOp
    from machete.kernels.gated_delta_net.fused_op import GDNFusedOp
    from machete.kernels.utils import SingleOpKernel

    # RoPE (q only, single op)
    try:
        q_rope = q.clone().contiguous()
        rope_ops = RopeOp.schedule_forward(q=q_rope, cos=cos, sin=sin, page_size=page_size)
        rope_kernel = Megakernel(rope_ops, config=RopeOp.kernel_config(rope_ops))
        with contextlib.redirect_stdout(io.StringIO()):
            rope_kernel.run()
        torch.cuda.synchronize()
        funcs["rope_q"] = rope_kernel.bench_spec(
            setup_fn=lambda q_rope=q_rope: q_rope.copy_(q),
            keep_alive=[q_rope, cos, sin],
        )
    except Exception:
        pass

    # GDN Prep only
    try:
        k_p = k.contiguous()
        v_p = v.contiguous()
        g_p = g.contiguous()
        beta_p = beta.contiguous()
        gc_p = torch.zeros(B, T, H, device=k.device, dtype=torch.float32)
        w_p = torch.zeros(B, T, H, K, device=k.device, dtype=dtype)
        u_p = torch.zeros(B, T, H, V, device=k.device, dtype=dtype)
        prep_ops = GDNPrepOp.schedule_forward(
            k=k_p, v=v_p, g=g_p, beta=beta_p,
            g_cumsum=gc_p, w=w_p, u=u_p, page_size=page_size,
        )
        prep_kernel = Megakernel(prep_ops, config=GDNPrepOp.kernel_config(prep_ops))
        with contextlib.redirect_stdout(io.StringIO()):
            prep_kernel.run()
        torch.cuda.synchronize()
        funcs["gdn_prep"] = prep_kernel.bench_spec(
            setup_fn=lambda gc=gc_p, w=w_p, u=u_p: (gc.zero_(), w.zero_(), u.zero_()),
            keep_alive=[k_p, v_p, g_p, beta_p, gc_p, w_p, u_p],
        )
    except Exception:
        pass

    # GDN Fused only
    try:
        gc_f = gc_p.clone()
        w_f = w_p.clone()
        u_f = u_p.clone()
        o_f = torch.zeros(B, T, H, V, device=q.device, dtype=dtype)
        fused_ops = GDNFusedOp.schedule_forward(
            q=q.contiguous(), k=k.contiguous(),
            w=w_f, u=u_f, g_cumsum=gc_f, o=o_f, scale=scale, page_size=page_size,
        )
        fused_kernel = Megakernel(fused_ops, config=GDNFusedOp.kernel_config(fused_ops))
        with contextlib.redirect_stdout(io.StringIO()):
            fused_kernel.run()
        torch.cuda.synchronize()
        funcs["gdn_fused"] = fused_kernel.bench_spec(
            setup_fn=lambda o_f=o_f: o_f.zero_(),
            keep_alive=[q, k, w_f, u_f, gc_f, o_f],
        )
    except Exception:
        pass

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
    print("Individual Op Timing (to identify bottleneck contributions)")
    print("-" * 100)
    bench_individual_ops._benchmark.run(mode="kernel", warmup=10, rep=100)

    print()
    print("-" * 100)
    print("Full Pipeline: RoPE(q) + RoPE(k) + GDN Prep + GDN Fused")
    print("-" * 100)
    bench_rope_gdn._benchmark.run(mode="kernel", warmup=10, rep=100)
