#!/usr/bin/env python
# Copyright (c) 2025, Machete Authors
"""Benchmark Gated Delta Net: Megakernel Ops vs PyTorch ref vs fla.

Compares GPU kernel execution time of prep, state recurrence, output, and
full forward pipeline across realistic Qwen 3.5 shapes.

Implementations:
- fla: flash-linear-attention Triton kernels (reference)
- PyTorch: Pure PyTorch reference implementation
- MegakernelOp: Machete megakernel Ops (fusable with other Ops)

Usage:
    python benchmarks/kernels/benchmark_gated_delta_net.py
"""

import contextlib
import io

import torch

from machete.kernels.utils import SingleOpKernel
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


def is_hopper_or_newer():
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


def _make_inputs(B, T, H, K, V, dtype=torch.float16, device="cuda"):
    """Create random inputs for gated delta net."""
    q = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    k = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    v = torch.randn(B, T, H, V, dtype=dtype, device=device) * 0.1
    g = -torch.rand(B, T, H, dtype=torch.float32, device=device) * 2.0
    beta = torch.sigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device))
    return q, k, v, g, beta


def _build_prep_megakernel(k, v, g, beta):
    """Build and compile a PrepOp megakernel with pre-allocated outputs.

    Returns (kernel, k, v, g, beta, gc, w, u) in native [B,T,H,K] layout.
    """
    from machete.megakernel import Megakernel
    from machete.kernels.gated_delta_net.prep_op import GDNPrepOp

    B, T, H, K_ = k.shape
    V = v.shape[-1]
    dtype = k.dtype

    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()

    gc = torch.zeros(B, T, H, device=k.device, dtype=torch.float32)
    w = torch.zeros(B, T, H, K_, device=k.device, dtype=dtype)
    u = torch.zeros(B, T, H, V, device=k.device, dtype=dtype)

    ops = GDNPrepOp.schedule_forward(
        k=k, v=v, g=g, beta=beta,
        g_cumsum=gc, w=w, u=u,
    )
    config = GDNPrepOp.kernel_config(ops)
    kernel = Megakernel(ops, config=config)
    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()
    torch.cuda.synchronize()

    return kernel, k, v, g, beta, gc, w, u


def _build_fused_megakernel(q, k, w, u, g_cumsum, scale):
    """Build and compile a FusedOp megakernel (state+output, no intermediates).

    Returns (kernel, q, k, w, u, gc, o) in native [B,T,H,K] layout.
    """
    from machete.megakernel import Megakernel
    from machete.kernels.gated_delta_net.fused_op import GDNFusedOp

    B, T, H, K_ = q.shape
    V = u.shape[-1]
    dtype = q.dtype

    q = q.contiguous()
    k = k.contiguous()
    w = w.contiguous()
    u = u.contiguous()
    g_cumsum = g_cumsum.contiguous()
    o = torch.zeros(B, T, H, V, device=q.device, dtype=dtype)

    ops = GDNFusedOp.schedule_forward(
        q=q, k=k, w=w, u=u, g_cumsum=g_cumsum, o=o,
        scale=scale,
    )
    config = GDNFusedOp.kernel_config(ops)
    kernel = Megakernel(ops, config=config)
    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()
    torch.cuda.synchronize()

    return kernel, q, k, w, u, g_cumsum, o


# =============================================================================
# Prep benchmark
# =============================================================================

PREP_CONFIGS = [
    (1, 1024, 32, 128, 128),
    (1, 2048, 32, 128, 128),
    (1, 4096, 32, 128, 128),
    (1, 8192, 32, 128, 128),
    (4, 2048, 32, 128, 128),
]


@Benchmark.configs(["B", "T", "H", "K", "V"], PREP_CONFIGS)
def bench_prep(B, T, H, K, V):
    """Setup prep kernel benchmarks: fla vs PyTorch vs MegakernelOp."""
    q, k, v, g, beta = _make_inputs(B, T, H, K, V)

    funcs = {}

    if FLA_AVAILABLE:
        from machete.kernels.gated_delta_net.ref import fla_prep_stage
        funcs["fla"] = lambda: fla_prep_stage(k, v, g, beta)

        from machete.kernels.gated_delta_net.prep import run_prep
        funcs["pytorch"] = lambda: run_prep(k, v, g, beta)

        if is_hopper_or_newer() and CUTLASS_AVAILABLE and HAS_MEGAKERNEL_OPS:
            try:
                result = _build_prep_megakernel(k, v, g, beta)
                prep_kernel = result[0]
                gc, w_out, u_out = result[5], result[6], result[7]

                funcs["megakernel_op"] = prep_kernel.bench_spec(
                    setup_fn=lambda: (gc.zero_(), w_out.zero_(), u_out.zero_()),
                    keep_alive=list(result[1:]),
                )
            except Exception as e:
                print(f"  MegakernelOp prep error: {e}")

        if is_hopper_or_newer() and CUTLASS_AVAILABLE and HAS_MEGAKERNEL_OPS:
            try:
                from machete.kernels.gated_delta_net.prep_op import GDNPrepOp

                gc_so = torch.zeros(B, T, H, device=k.device, dtype=torch.float32)
                w_so = torch.zeros(B, T, H, K, device=k.device, dtype=k.dtype)
                u_so = torch.zeros(B, T, H, V, device=k.device, dtype=k.dtype)

                so_ops = GDNPrepOp.schedule_forward(
                    k=k.contiguous(), v=v.contiguous(),
                    g=g.contiguous(), beta=beta.contiguous(),
                    g_cumsum=gc_so, w=w_so, u=u_so,
                )
                so_kernel = SingleOpKernel(so_ops)
                with contextlib.redirect_stdout(io.StringIO()):
                    so_kernel.run()
                torch.cuda.synchronize()

                funcs["single_op"] = so_kernel.bench_spec(
                    setup_fn=lambda: (gc_so.zero_(), w_so.zero_(), u_so.zero_()),
                    keep_alive=[k, v, g, beta, gc_so, w_so, u_so],
                )
            except Exception as e:
                print(f"  SingleOp prep error: {e}")

    return funcs


# =============================================================================
# Fused State+Output benchmark (GDNFusedOp vs fla)
# =============================================================================

FUSED_CONFIGS = [
    (1, 1024, 32, 128, 128),
    (1, 2048, 32, 128, 128),
    (1, 4096, 32, 128, 128),
    (1, 8192, 32, 128, 128),
    (4, 2048, 32, 128, 128),
]


@Benchmark.configs(["B", "T", "H", "K", "V"], FUSED_CONFIGS)
def bench_fused(B, T, H, K, V):
    """Benchmark fused state+output vs separate ops vs fla (state+output only)."""
    q, k, v, g, beta = _make_inputs(B, T, H, K, V)
    scale = K ** -0.5

    funcs = {}

    if FLA_AVAILABLE:
        from machete.kernels.gated_delta_net.ref import (
            fla_prep_stage, fla_state_recurrence, fla_output,
        )
        g_cumsum, A, w, u = fla_prep_stage(k, v, g, beta)

        # fla state+output combined
        def fla_state_output():
            h, v_new, _ = fla_state_recurrence(k, w, u, g_cumsum)
            return fla_output(q, k, v_new, h, g_cumsum, scale=scale)
        funcs["fla_state+output"] = fla_state_output

        if is_hopper_or_newer() and CUTLASS_AVAILABLE and HAS_MEGAKERNEL_OPS:
            try:
                fused_result = _build_fused_megakernel(
                    q, k, w, u, g_cumsum, scale)
                fused_kernel = fused_result[0]
                o_fused = fused_result[6]

                funcs["fused_op"] = fused_kernel.bench_spec(
                    setup_fn=lambda: o_fused.zero_(),
                    keep_alive=list(fused_result[1:]),
                )
            except Exception as e:
                print(f"  FusedOp error: {e}")

        if is_hopper_or_newer() and CUTLASS_AVAILABLE and HAS_MEGAKERNEL_OPS:
            try:
                from machete.kernels.gated_delta_net.fused_op import GDNFusedOp

                o_so = torch.zeros(B, T, H, V, device=q.device, dtype=q.dtype)
                so_ops = GDNFusedOp.schedule_forward(
                    q=q.contiguous(), k=k.contiguous(),
                    w=w.contiguous(), u=u.contiguous(),
                    g_cumsum=g_cumsum.contiguous(), o=o_so,
                    scale=scale,
                )
                so_kernel = SingleOpKernel(so_ops)
                with contextlib.redirect_stdout(io.StringIO()):
                    so_kernel.run()
                torch.cuda.synchronize()

                funcs["single_op"] = so_kernel.bench_spec(
                    setup_fn=lambda: o_so.zero_(),
                    keep_alive=[q, k, w, u, g_cumsum, o_so],
                )
            except Exception as e:
                print(f"  SingleOp fused error: {e}")

    return funcs


# =============================================================================
# Full forward pipeline benchmark
# =============================================================================

PIPELINE_CONFIGS = [
    (1, 1024, 16, 128, 128),
    (1, 2048, 32, 128, 128),
    (1, 4096, 32, 128, 128),
    (4, 2048, 16, 128, 128),
    (4, 4096, 32, 128, 128),
]


@Benchmark.configs(["B", "T", "H", "K", "V"], PIPELINE_CONFIGS)
def bench_pipeline(B, T, H, K, V):
    """Setup full pipeline benchmarks: fla vs Fused Megakernel.

    Pre-allocates all tensors and builds the fused megakernel once so only
    kernel execution time is measured.
    """
    q, k, v, g, beta = _make_inputs(B, T, H, K, V)
    scale = K ** -0.5
    dtype = k.dtype

    funcs = {}

    if FLA_AVAILABLE:
        from machete.kernels.gated_delta_net.ref import fla_full_forward
        funcs["fla"] = lambda: fla_full_forward(q, k, v, g, beta, scale=scale)

    if is_hopper_or_newer() and CUTLASS_AVAILABLE and HAS_MEGAKERNEL_OPS:
        try:
            import cuda.bindings.driver as cuda
            from machete.megakernel import Megakernel
            from machete.kernels.gated_delta_net.prep_op import GDNPrepOp
            from machete.kernels.gated_delta_net.fused_op import GDNFusedOp

            # Pre-allocate ALL tensors in native [B,T,H,K/V] layout
            q_c = q.contiguous()
            k_c = k.contiguous()
            v_c = v.contiguous()
            g_c = g.contiguous()
            beta_c = beta.contiguous()

            gc = torch.zeros(B, T, H, device=k.device, dtype=torch.float32)
            w = torch.zeros(B, T, H, K, device=k.device, dtype=dtype)
            u = torch.zeros(B, T, H, V, device=k.device, dtype=dtype)
            o = torch.zeros(B, T, H, V, device=k.device, dtype=dtype)

            # --- Fused megakernel: PrepOp + FusedOp in one kernel ---
            try:
                prep_ops = GDNPrepOp.schedule_forward(
                    k=k_c, v=v_c, g=g_c, beta=beta_c,
                    g_cumsum=gc, w=w, u=u,
                )
                fused_ops = GDNFusedOp.schedule_forward(
                    q=q_c, k=k_c, w=w, u=u,
                    g_cumsum=gc, o=o, scale=scale,
                )
                all_ops = prep_ops + fused_ops
                fused_config = GDNFusedOp.kernel_config(all_ops)
                fused_kernel = Megakernel(all_ops, config=fused_config)
                with contextlib.redirect_stdout(io.StringIO()):
                    fused_kernel.run()
                torch.cuda.synchronize()

                fused_stream = torch.cuda.Stream()
                fused_cu_stream = cuda.CUstream(fused_stream.cuda_stream)

                def fused_setup():
                    gc.zero_()
                    w.zero_()
                    u.zero_()
                    o.zero_()

                funcs["fused_megakernel"] = KernelBenchSpec(
                    launch_fn=lambda: fused_kernel.run(
                        stream=fused_cu_stream, sync=False),
                    setup_fn=fused_setup,
                    stream=(fused_stream, fused_cu_stream),
                    _keep_alive=[fused_kernel, k_c, v_c, g_c, beta_c,
                                 q_c, gc, w, u, o],
                )
            except Exception as e:
                print(f"  Fused megakernel error: {e}")

        except Exception as e:
            print(f"  Pipeline setup error: {e}")

    return funcs


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    print("=" * 100)
    print("Gated Delta Net Benchmark: Megakernel Ops vs PyTorch ref vs fla")
    print("=" * 100)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.multi_processor_count} SMs)")
        print(f"Hopper+: {is_hopper_or_newer()}")
    print(f"CUTLASS: {CUTLASS_AVAILABLE}")
    print(f"fla: {FLA_AVAILABLE}")
    print()

    print("-" * 100)
    print("Prep Kernel (fla vs PyTorch vs MegakernelOp)")
    print("-" * 100)
    bench_prep._benchmark.run(mode="kernel", warmup=10, rep=100)

    print()
    print("-" * 100)
    print("Fused State+Output (FusedOp vs fla)")
    print("-" * 100)
    bench_fused._benchmark.run(mode="kernel", warmup=10, rep=100)

    print()
    print("-" * 100)
    print("Full Pipeline (fla vs Fused Megakernel)")
    print("-" * 100)
    bench_pipeline._benchmark.run(mode="kernel", warmup=10, rep=100)
