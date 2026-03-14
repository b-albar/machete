#!/usr/bin/env python
# Copyright (c) 2025, Machete Authors
"""Benchmark fused megakernels: sequential launches vs single fused launch.

Three fusion levels:
  1. GEMM+RoPE:     GEMM(Q,K,V) + RoPE(Q,K)           — 2 launches vs 1
  2. GEMM+GDN:      GEMM(Q,K,V) + GDN(prep+fused)     — 2 launches vs 1
  3. GEMM+RoPE+GDN: GEMM(Q,K,V) + RoPE(Q,K) + GDN     — 3 launches vs 1

Usage:
    python benchmarks/kernels/benchmark_gemm_rope_gdn.py
"""

import contextlib
import io

import torch

from machete.utils.benchmark import Benchmark
from machete.utils.benchmark_utils import KernelBenchSpec

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


def _make_inputs(B, T, H, K, V, dtype=torch.bfloat16, device="cuda"):
    """Create random inputs for GEMM + RoPE + GDN pipeline.

    Returns 3D tensors (B, T, D) for GEMM, 2D weights (N, K).
    """
    D_in = H * K
    D_v = H * V

    x = torch.randn(B, T, D_in, dtype=dtype, device=device) * 0.02
    wq = torch.randn(D_in, D_in, dtype=dtype, device=device) * 0.02
    wk = torch.randn(D_in, D_in, dtype=dtype, device=device) * 0.02
    wv = torch.randn(D_v, D_in, dtype=dtype, device=device) * 0.02

    q_3d = torch.zeros(B, T, D_in, dtype=dtype, device=device)
    k_3d = torch.zeros(B, T, D_in, dtype=dtype, device=device)
    v_3d = torch.zeros(B, T, D_v, dtype=dtype, device=device)

    g = -torch.rand(B, T, H, dtype=torch.float32, device=device) * 2.0
    beta = torch.sigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device))

    cos = torch.randn(T, K // 2, dtype=dtype, device=device)
    sin = torch.randn(T, K // 2, dtype=dtype, device=device)

    return x, wq, wk, wv, q_3d, k_3d, v_3d, g, beta, cos, sin


def _merged_config(*configs):
    """Merge multiple MegakernelConfigs by taking max threads and page_size."""
    from machete.megakernel import MegakernelConfig
    return MegakernelConfig(
        threads_per_block=max(c.threads_per_block for c in configs),
        page_size=max(c.page_size for c in configs),
    )


def _build_and_run(ops, config):
    """Build megakernel, warm up, return kernel."""
    from machete.megakernel import Megakernel
    kernel = Megakernel(ops, config=config)
    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()
    torch.cuda.synchronize()
    return kernel


# =============================================================================
# 1. GEMM + RoPE: sequential (2 launches) vs fused (1 launch)
# =============================================================================

_BASE_CONFIGS = [
    # (B, T, H, K, V)
    (1, 2048, 32, 128, 128),
    (1, 4096, 32, 128, 128),
    (4, 2048, 32, 128, 128),
    (4, 4096, 32, 128, 128),
    (8, 2048, 32, 128, 128),
]

GEMM_ROPE_CONFIGS = [c + (ps,) for c in _BASE_CONFIGS for ps in PAGE_SIZES]


@Benchmark.configs(["B", "T", "H", "K", "V", "page_size"], GEMM_ROPE_CONFIGS)
def bench_gemm_rope(B, T, H, K, V, page_size):
    """GEMM(Q,K,V) + RoPE(Q,K): 2 launches vs 1."""
    if not (is_hopper_or_newer() and CUTLASS_AVAILABLE):
        return {}

    from machete.kernels.gemm import GemmOp
    from machete.kernels.rope import RopeOp
    import cuda.bindings.driver as cuda

    dtype = torch.bfloat16
    x, wq, wk, wv, q_3d, k_3d, v_3d, _, _, cos, sin = _make_inputs(
        B, T, H, K, V, dtype=dtype)
    funcs = {}

    # --- Sequential: GEMM kernel + RoPE kernel ---
    try:
        x_s = x.clone()
        q_s = torch.zeros_like(q_3d)
        k_s = torch.zeros_like(k_3d)
        v_s = torch.zeros_like(v_3d)

        gemm_ops = (GemmOp.schedule_forward(a=x_s, b=wq, c=q_s, page_size=page_size)
                     + GemmOp.schedule_forward(a=x_s, b=wk, c=k_s, page_size=page_size)
                     + GemmOp.schedule_forward(a=x_s, b=wv, c=v_s, page_size=page_size))
        gemm_kern = _build_and_run(gemm_ops, GemmOp.kernel_config(gemm_ops))

        q_s4 = q_s.view(B, T, H, K)
        k_s4 = k_s.view(B, T, H, K)
        rope_ops = (RopeOp.schedule_forward(q=q_s4, cos=cos, sin=sin, page_size=page_size)
                     + RopeOp.schedule_forward(q=k_s4, cos=cos, sin=sin, page_size=page_size))
        rope_kern = _build_and_run(rope_ops, RopeOp.kernel_config(rope_ops))

        s_stream = torch.cuda.Stream()
        s_cu = cuda.CUstream(s_stream.cuda_stream)

        funcs["sequential"] = KernelBenchSpec(
            launch_fn=lambda gk=gemm_kern, rk=rope_kern, sc=s_cu: (
                gk.run(stream=sc, sync=False), rk.run(stream=sc, sync=False)),
            setup_fn=lambda qs=q_s, ks=k_s, vs=v_s: (qs.zero_(), ks.zero_(), vs.zero_()),
            stream=(s_stream, s_cu),
            _keep_alive=[gemm_kern, rope_kern, x_s, wq, wk, wv,
                         q_s, k_s, v_s, cos, sin],
        )
    except Exception:
        pass

    # --- Fused: single megakernel ---
    try:
        x_f = x.clone()
        q_f = torch.zeros_like(q_3d)
        k_f = torch.zeros_like(k_3d)
        v_f = torch.zeros_like(v_3d)

        q_f4 = q_f.view(B, T, H, K)
        k_f4 = k_f.view(B, T, H, K)

        gemm_ops = (GemmOp.schedule_forward(a=x_f, b=wq, c=q_f, page_size=page_size)
                     + GemmOp.schedule_forward(a=x_f, b=wk, c=k_f, page_size=page_size)
                     + GemmOp.schedule_forward(a=x_f, b=wv, c=v_f, page_size=page_size))
        rope_ops = (RopeOp.schedule_forward(q=q_f4, cos=cos, sin=sin, page_size=page_size)
                     + RopeOp.schedule_forward(q=k_f4, cos=cos, sin=sin, page_size=page_size))

        all_ops = gemm_ops + rope_ops
        config = GemmOp.kernel_config(all_ops)
        fused_kern = _build_and_run(all_ops, config)

        f_stream = torch.cuda.Stream()
        f_cu = cuda.CUstream(f_stream.cuda_stream)

        funcs["fused"] = KernelBenchSpec(
            launch_fn=lambda fk=fused_kern, fc=f_cu: fk.run(stream=fc, sync=False),
            setup_fn=lambda qf=q_f, kf=k_f, vf=v_f: (qf.zero_(), kf.zero_(), vf.zero_()),
            stream=(f_stream, f_cu),
            _keep_alive=[fused_kern, x_f, wq, wk, wv,
                         q_f, k_f, v_f, cos, sin],
        )
    except Exception:
        pass

    return funcs


# =============================================================================
# 2. GEMM + GDN: sequential (2 launches) vs fused (1 launch)
# =============================================================================

GEMM_GDN_CONFIGS = [c + (ps,) for c in _BASE_CONFIGS for ps in PAGE_SIZES]


@Benchmark.configs(["B", "T", "H", "K", "V", "page_size"], GEMM_GDN_CONFIGS)
def bench_gemm_gdn(B, T, H, K, V, page_size):
    """GEMM(Q,K,V) + GDN(prep+fused): 2 launches vs 1."""
    if not (is_hopper_or_newer() and CUTLASS_AVAILABLE and HAS_MEGAKERNEL_OPS):
        return {}

    from machete.kernels.gemm import GemmOp
    from machete.kernels.gated_delta_net.prep_op import GDNPrepOp
    from machete.kernels.gated_delta_net.fused_op import GDNFusedOp
    import cuda.bindings.driver as cuda

    dtype = torch.bfloat16
    x, wq, wk, wv, q_3d, k_3d, v_3d, g, beta, cos, sin = _make_inputs(
        B, T, H, K, V, dtype=dtype)
    scale = K ** -0.5
    funcs = {}

    def _gdn_intermediates():
        return (torch.zeros(B, T, H, device="cuda", dtype=torch.float32),
                torch.zeros(B, T, H, K, device="cuda", dtype=dtype),
                torch.zeros(B, T, H, V, device="cuda", dtype=dtype),
                torch.zeros(B, T, H, V, device="cuda", dtype=dtype))

    # --- Sequential: GEMM kernel + GDN kernel ---
    try:
        x_s = x.clone()
        q_s = torch.zeros_like(q_3d)
        k_s = torch.zeros_like(k_3d)
        v_s = torch.zeros_like(v_3d)

        gemm_ops = (GemmOp.schedule_forward(a=x_s, b=wq, c=q_s, page_size=page_size)
                     + GemmOp.schedule_forward(a=x_s, b=wk, c=k_s, page_size=page_size)
                     + GemmOp.schedule_forward(a=x_s, b=wv, c=v_s, page_size=page_size))
        gemm_kern = _build_and_run(gemm_ops, GemmOp.kernel_config(gemm_ops))

        q_s4 = q_s.view(B, T, H, K)
        k_s4 = k_s.view(B, T, H, K)
        v_s4 = v_s.view(B, T, H, V)
        gc_s, w_s, u_s, o_s = _gdn_intermediates()

        prep_ops = GDNPrepOp.schedule_forward(
            k=k_s4, v=v_s4, g=g, beta=beta, g_cumsum=gc_s, w=w_s, u=u_s, page_size=page_size)
        fused_ops = GDNFusedOp.schedule_forward(
            q=q_s4, k=k_s4, w=w_s, u=u_s, g_cumsum=gc_s, o=o_s, scale=scale, page_size=page_size)
        gdn_ops = prep_ops + fused_ops
        gdn_kern = _build_and_run(gdn_ops, GDNFusedOp.kernel_config(gdn_ops))

        s_stream = torch.cuda.Stream()
        s_cu = cuda.CUstream(s_stream.cuda_stream)

        funcs["sequential"] = KernelBenchSpec(
            launch_fn=lambda gk=gemm_kern, dk=gdn_kern, sc=s_cu: (
                gk.run(stream=sc, sync=False), dk.run(stream=sc, sync=False)),
            setup_fn=lambda qs=q_s, ks=k_s, vs=v_s, gc=gc_s, ws=w_s, us=u_s, os_=o_s: (
                qs.zero_(), ks.zero_(), vs.zero_(), gc.zero_(), ws.zero_(), us.zero_(), os_.zero_()),
            stream=(s_stream, s_cu),
            _keep_alive=[gemm_kern, gdn_kern, x_s, wq, wk, wv,
                         q_s, k_s, v_s, g, beta,
                         gc_s, w_s, u_s, o_s],
        )
    except Exception:
        pass

    # --- Fused: single megakernel ---
    try:
        x_f = x.clone()
        q_f = torch.zeros_like(q_3d)
        k_f = torch.zeros_like(k_3d)
        v_f = torch.zeros_like(v_3d)
        gc_f, w_f, u_f, o_f = _gdn_intermediates()

        q_f4 = q_f.view(B, T, H, K)
        k_f4 = k_f.view(B, T, H, K)
        v_f4 = v_f.view(B, T, H, V)

        gemm_ops = (GemmOp.schedule_forward(a=x_f, b=wq, c=q_f, page_size=page_size)
                     + GemmOp.schedule_forward(a=x_f, b=wk, c=k_f, page_size=page_size)
                     + GemmOp.schedule_forward(a=x_f, b=wv, c=v_f, page_size=page_size))
        prep_ops = GDNPrepOp.schedule_forward(
            k=k_f4, v=v_f4, g=g, beta=beta, g_cumsum=gc_f, w=w_f, u=u_f, page_size=page_size)
        fused_ops = GDNFusedOp.schedule_forward(
            q=q_f4, k=k_f4, w=w_f, u=u_f, g_cumsum=gc_f, o=o_f, scale=scale, page_size=page_size)

        all_ops = gemm_ops + prep_ops + fused_ops
        config = _merged_config(
            GemmOp.kernel_config(gemm_ops),
            GDNFusedOp.kernel_config(prep_ops + fused_ops),
        )
        fused_kern = _build_and_run(all_ops, config)

        f_stream = torch.cuda.Stream()
        f_cu = cuda.CUstream(f_stream.cuda_stream)

        funcs["fused"] = KernelBenchSpec(
            launch_fn=lambda fk=fused_kern, fc=f_cu: fk.run(stream=fc, sync=False),
            setup_fn=lambda qf=q_f, kf=k_f, vf=v_f, gc=gc_f, wf=w_f, uf=u_f, of=o_f: (
                qf.zero_(), kf.zero_(), vf.zero_(), gc.zero_(), wf.zero_(), uf.zero_(), of.zero_()),
            stream=(f_stream, f_cu),
            _keep_alive=[fused_kern, x_f, wq, wk, wv,
                         q_f, k_f, v_f, g, beta,
                         gc_f, w_f, u_f, o_f],
        )
    except Exception:
        pass

    return funcs


# =============================================================================
# 3. GEMM + RoPE + GDN: sequential (3 launches) vs fused (1 launch)
# =============================================================================

GEMM_ROPE_GDN_CONFIGS = [c + (ps,) for c in _BASE_CONFIGS for ps in PAGE_SIZES]


@Benchmark.configs(["B", "T", "H", "K", "V", "page_size"], GEMM_ROPE_GDN_CONFIGS)
def bench_gemm_rope_gdn(B, T, H, K, V, page_size):
    """GEMM(Q,K,V) + RoPE(Q,K) + GDN: 3 launches vs 1."""
    if not (is_hopper_or_newer() and CUTLASS_AVAILABLE and HAS_MEGAKERNEL_OPS):
        return {}

    from machete.kernels.gemm import GemmOp
    from machete.kernels.rope import RopeOp
    from machete.kernels.gated_delta_net.prep_op import GDNPrepOp
    from machete.kernels.gated_delta_net.fused_op import GDNFusedOp
    import cuda.bindings.driver as cuda

    dtype = torch.bfloat16
    x, wq, wk, wv, q_3d, k_3d, v_3d, g, beta, cos, sin = _make_inputs(
        B, T, H, K, V, dtype=dtype)
    scale = K ** -0.5
    funcs = {}

    def _gdn_intermediates():
        return (torch.zeros(B, T, H, device="cuda", dtype=torch.float32),
                torch.zeros(B, T, H, K, device="cuda", dtype=dtype),
                torch.zeros(B, T, H, V, device="cuda", dtype=dtype),
                torch.zeros(B, T, H, V, device="cuda", dtype=dtype))

    # --- Sequential: GEMM + RoPE + GDN (3 launches) ---
    try:
        x_s = x.clone()
        q_s = torch.zeros_like(q_3d)
        k_s = torch.zeros_like(k_3d)
        v_s = torch.zeros_like(v_3d)

        gemm_ops = (GemmOp.schedule_forward(a=x_s, b=wq, c=q_s, page_size=page_size)
                     + GemmOp.schedule_forward(a=x_s, b=wk, c=k_s, page_size=page_size)
                     + GemmOp.schedule_forward(a=x_s, b=wv, c=v_s, page_size=page_size))
        gemm_kern = _build_and_run(gemm_ops, GemmOp.kernel_config(gemm_ops))

        q_s4 = q_s.view(B, T, H, K)
        k_s4 = k_s.view(B, T, H, K)
        rope_ops = (RopeOp.schedule_forward(q=q_s4, cos=cos, sin=sin, page_size=page_size)
                     + RopeOp.schedule_forward(q=k_s4, cos=cos, sin=sin, page_size=page_size))
        rope_kern = _build_and_run(rope_ops, RopeOp.kernel_config(rope_ops))

        v_s4 = v_s.view(B, T, H, V)
        gc_s, w_s, u_s, o_s = _gdn_intermediates()

        prep_ops = GDNPrepOp.schedule_forward(
            k=k_s4, v=v_s4, g=g, beta=beta, g_cumsum=gc_s, w=w_s, u=u_s, page_size=page_size)
        fused_ops = GDNFusedOp.schedule_forward(
            q=q_s4, k=k_s4, w=w_s, u=u_s, g_cumsum=gc_s, o=o_s, scale=scale, page_size=page_size)
        gdn_ops = prep_ops + fused_ops
        gdn_kern = _build_and_run(gdn_ops, GDNFusedOp.kernel_config(gdn_ops))

        s_stream = torch.cuda.Stream()
        s_cu = cuda.CUstream(s_stream.cuda_stream)

        funcs["sequential"] = KernelBenchSpec(
            launch_fn=lambda gk=gemm_kern, rk=rope_kern, dk=gdn_kern, sc=s_cu: (
                gk.run(stream=sc, sync=False), rk.run(stream=sc, sync=False),
                dk.run(stream=sc, sync=False)),
            setup_fn=lambda qs=q_s, ks=k_s, vs=v_s, gc=gc_s, ws=w_s, us=u_s, os_=o_s: (
                qs.zero_(), ks.zero_(), vs.zero_(), gc.zero_(), ws.zero_(), us.zero_(), os_.zero_()),
            stream=(s_stream, s_cu),
            _keep_alive=[gemm_kern, rope_kern, gdn_kern,
                         x_s, wq, wk, wv, q_s, k_s, v_s,
                         g, beta, cos, sin, gc_s, w_s, u_s, o_s],
        )
    except Exception:
        pass

    # --- Fused: single megakernel ---
    try:
        x_f = x.clone()
        q_f = torch.zeros_like(q_3d)
        k_f = torch.zeros_like(k_3d)
        v_f = torch.zeros_like(v_3d)
        gc_f, w_f, u_f, o_f = _gdn_intermediates()

        q_f4 = q_f.view(B, T, H, K)
        k_f4 = k_f.view(B, T, H, K)
        v_f4 = v_f.view(B, T, H, V)

        gemm_ops = (GemmOp.schedule_forward(a=x_f, b=wq, c=q_f, page_size=page_size)
                     + GemmOp.schedule_forward(a=x_f, b=wk, c=k_f, page_size=page_size)
                     + GemmOp.schedule_forward(a=x_f, b=wv, c=v_f, page_size=page_size))
        rope_ops = (RopeOp.schedule_forward(q=q_f4, cos=cos, sin=sin, page_size=page_size)
                     + RopeOp.schedule_forward(q=k_f4, cos=cos, sin=sin, page_size=page_size))
        prep_ops = GDNPrepOp.schedule_forward(
            k=k_f4, v=v_f4, g=g, beta=beta, g_cumsum=gc_f, w=w_f, u=u_f, page_size=page_size)
        fused_ops = GDNFusedOp.schedule_forward(
            q=q_f4, k=k_f4, w=w_f, u=u_f, g_cumsum=gc_f, o=o_f, scale=scale, page_size=page_size)

        all_ops = gemm_ops + rope_ops + prep_ops + fused_ops
        config = _merged_config(
            GemmOp.kernel_config(gemm_ops),
            GDNFusedOp.kernel_config(prep_ops + fused_ops),
        )
        fused_kern = _build_and_run(all_ops, config)

        f_stream = torch.cuda.Stream()
        f_cu = cuda.CUstream(f_stream.cuda_stream)

        funcs["fused"] = KernelBenchSpec(
            launch_fn=lambda fk=fused_kern, fc=f_cu: fk.run(stream=fc, sync=False),
            setup_fn=lambda qf=q_f, kf=k_f, vf=v_f, gc=gc_f, wf=w_f, uf=u_f, of=o_f: (
                qf.zero_(), kf.zero_(), vf.zero_(), gc.zero_(), wf.zero_(), uf.zero_(), of.zero_()),
            stream=(f_stream, f_cu),
            _keep_alive=[fused_kern, x_f, wq, wk, wv,
                         q_f, k_f, v_f, g, beta, cos, sin,
                         gc_f, w_f, u_f, o_f],
        )
    except Exception:
        pass

    return funcs


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 100)
    print("Fused Megakernel Benchmark: Sequential vs Fused")
    print("=" * 100)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.multi_processor_count} SMs)")
        print(f"Hopper+: {is_hopper_or_newer()}")
    print(f"CUTLASS: {CUTLASS_AVAILABLE}")
    print()

    print("-" * 100)
    print("GEMM(Q,K,V) + RoPE(Q,K): 2 launches vs 1")
    print("-" * 100)
    bench_gemm_rope._benchmark.run(mode="kernel", warmup=10, rep=100)

    print()
    print("-" * 100)
    print("GEMM(Q,K,V) + GDN(prep+fused): 2 launches vs 1")
    print("-" * 100)
    bench_gemm_gdn._benchmark.run(mode="kernel", warmup=10, rep=100)

    print()
    print("-" * 100)
    print("GEMM(Q,K,V) + RoPE(Q,K) + GDN: 3 launches vs 1")
    print("-" * 100)
    bench_gemm_rope_gdn._benchmark.run(mode="kernel", warmup=10, rep=100)
