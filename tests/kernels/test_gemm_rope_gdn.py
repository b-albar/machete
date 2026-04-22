# Copyright (c) 2025, Machete Authors
"""Tests for GEMM(Q,K,V) + RoPE(Q,K) + GDN megakernel pipeline.

Verifies correctness of the full LLM attention pre-projection pipeline:
  1. GEMM: X @ Wq -> Q, X @ Wk -> K, X @ Wv -> V
  2. RoPE: rotate Q and K
  3. GDN: prep + fused gated delta net recurrence

Tests include:
  - GEMM+RoPE fused (1 kernel launch)
  - Sequential pipeline (3 kernel launches)
  - Fully fused GEMM+RoPE+GDN (1 kernel launch)
"""

import contextlib
import io
import importlib.util

import pytest
import torch

if importlib.util.find_spec("cutlass") is None:
    pytest.skip("Requires CUTLASS", allow_module_level=True)


def is_hopper_or_newer():
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


try:
    import cutlass  # noqa: F401
    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False

try:
    import fla  # noqa: F401
    FLA_AVAILABLE = True
except ImportError:
    FLA_AVAILABLE = False

try:
    from machete.kernels.gated_delta_net import HAS_MEGAKERNEL_OPS
except ImportError:
    HAS_MEGAKERNEL_OPS = False


requires_gpu = pytest.mark.skipif(
    not (is_hopper_or_newer() and CUTLASS_AVAILABLE),
    reason="Requires Hopper+ GPU with CUTLASS",
)

requires_all = pytest.mark.skipif(
    not (is_hopper_or_newer() and CUTLASS_AVAILABLE and FLA_AVAILABLE
         and HAS_MEGAKERNEL_OPS),
    reason="Requires Hopper+ GPU with CUTLASS, fla, and megakernel ops",
)


# =============================================================================
# Helpers
# =============================================================================


def _gemm_reference(a, b_t):
    """Reference GEMM: C = A @ B_T^T, computed in fp32."""
    return (a.float() @ b_t.float().t()).to(a.dtype)


def _make_inputs(B, T, H, K, V, dtype=torch.bfloat16, device="cuda"):
    """Create random inputs for the full pipeline."""
    torch.manual_seed(42)
    M = B * T
    D_in = H * K
    D_v = H * V

    x = torch.randn(M, D_in, dtype=dtype, device=device) * 0.02
    wq = torch.randn(D_in, D_in, dtype=dtype, device=device) * 0.02
    wk = torch.randn(D_in, D_in, dtype=dtype, device=device) * 0.02
    wv = torch.randn(D_v, D_in, dtype=dtype, device=device) * 0.02

    g = -torch.rand(B, T, H, dtype=torch.float32, device=device) * 2.0
    beta = torch.sigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device))

    cos = torch.randn(T, K // 2, dtype=dtype, device=device)
    sin = torch.randn(T, K // 2, dtype=dtype, device=device)

    return x, wq, wk, wv, g, beta, cos, sin


def _reference_pipeline(x, wq, wk, wv, g, beta, cos, sin, B, T, H, K, V, scale):
    """Run the full pipeline in PyTorch as reference."""
    from machete.kernels.rope.ref import rope_pytorch
    from machete.kernels.gated_delta_net.ref import fla_full_forward

    q_flat = _gemm_reference(x, wq)
    k_flat = _gemm_reference(x, wk)
    v_flat = _gemm_reference(x, wv)

    q_4d = rope_pytorch(q_flat.view(B, T, H, K), cos, sin)
    k_4d = rope_pytorch(k_flat.view(B, T, H, K), cos, sin)
    v_4d = v_flat.view(B, T, H, V)

    o_ref, _ = fla_full_forward(
        q_4d.to(x.dtype), k_4d.to(x.dtype), v_4d, g, beta, scale=scale,
    )

    return q_4d.to(x.dtype), k_4d.to(x.dtype), v_4d, o_ref


def _run_gemm_rope_megakernel(x, wq, wk, wv, cos, sin, B, T, H, K, V):
    """Run fused GEMM(Q,K,V) + RoPE(Q,K) in a single megakernel."""
    from machete.megakernel import Megakernel
    from machete.kernels.gemm import GemmOp
    from machete.kernels.rope import RopeOp

    D_in = H * K
    D_v = H * V
    dtype = x.dtype

    # 3D tensors for GEMM (B, S=T, K/N)
    x_3d = x.view(B, T, D_in)
    q_3d = torch.zeros(B, T, D_in, dtype=dtype, device=x.device)
    k_3d = torch.zeros(B, T, D_in, dtype=dtype, device=x.device)
    v_3d = torch.zeros(B, T, D_v, dtype=dtype, device=x.device)

    # 4D views for RoPE (B, S=T, NH=H, HD=K)
    q_4d = q_3d.view(B, T, H, K)
    k_4d = k_3d.view(B, T, H, K)

    # GEMM ops
    gemm_q_ops = GemmOp.schedule(a=x_3d, b=wq, c=q_3d)
    gemm_k_ops = GemmOp.schedule(a=x_3d, b=wk, c=k_3d)
    gemm_v_ops = GemmOp.schedule(a=x_3d, b=wv, c=v_3d)

    # RoPE ops — q_4d/k_4d share data_ptr with q_3d/k_3d,
    # so framework auto-detects dependency (GEMM → RoPE)
    rope_q_ops = RopeOp.schedule(q=q_4d, cos=cos, sin=sin)
    rope_k_ops = RopeOp.schedule(q=k_4d, cos=cos, sin=sin)

    all_ops = gemm_q_ops + gemm_k_ops + gemm_v_ops + rope_q_ops + rope_k_ops
    config = GemmOp.kernel_config(all_ops)
    kernel = Megakernel(all_ops, config=config)
    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()
    torch.cuda.synchronize()

    return q_3d.view(B * T, D_in), k_3d.view(B * T, D_in), v_3d.view(B * T, D_v), kernel


def _run_gdn_megakernel(q_4d, k_4d, v_4d, g, beta, scale):
    """Run GDN prep + fused in a single megakernel."""
    from machete.megakernel import Megakernel
    from machete.kernels.gated_delta_net.prep_op import GDNPrepOp
    from machete.kernels.gated_delta_net.fused_op import GDNFusedOp

    B, T, H, K = q_4d.shape
    V = v_4d.shape[-1]
    dtype = q_4d.dtype

    gc = torch.zeros(B, T, H, device=q_4d.device, dtype=torch.float32)
    w = torch.zeros(B, T, H, K, device=q_4d.device, dtype=dtype)
    u = torch.zeros(B, T, H, V, device=q_4d.device, dtype=dtype)
    o = torch.zeros(B, T, H, V, device=q_4d.device, dtype=dtype)

    prep_ops = GDNPrepOp.schedule(
        k=k_4d.contiguous(), v=v_4d.contiguous(),
        g=g.contiguous(), beta=beta.contiguous(),
        g_cumsum=gc, w=w, u=u,
    )
    fused_ops = GDNFusedOp.schedule(
        q=q_4d.contiguous(), k=k_4d.contiguous(), w=w, u=u,
        g_cumsum=gc, o=o, scale=scale,
    )

    all_ops = prep_ops + fused_ops
    config = GDNFusedOp.kernel_config(all_ops)
    kernel = Megakernel(all_ops, config=config)
    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()
    torch.cuda.synchronize()

    return o, kernel


def _merged_config(*configs):
    """Merge multiple MegakernelConfigs by taking max threads and page_size."""
    from machete.megakernel import MegakernelConfig
    return MegakernelConfig(
        threads_per_block=max(c.threads_per_block for c in configs),
        page_size=max(c.page_size for c in configs),
    )


def _run_fully_fused_megakernel(x, wq, wk, wv, g, beta, cos, sin,
                                 B, T, H, K, V, scale):
    """Run GEMM+RoPE+GDN in a single megakernel (1 launch)."""
    from machete.megakernel import Megakernel
    from machete.kernels.gemm import GemmOp
    from machete.kernels.rope import RopeOp
    from machete.kernels.gated_delta_net.prep_op import GDNPrepOp
    from machete.kernels.gated_delta_net.fused_op import GDNFusedOp

    D_in = H * K
    D_v = H * V
    dtype = x.dtype

    # 3D GEMM outputs (B, T, D)
    x_3d = x.view(B, T, D_in)
    q_3d = torch.zeros(B, T, D_in, dtype=dtype, device=x.device)
    k_3d = torch.zeros(B, T, D_in, dtype=dtype, device=x.device)
    v_3d = torch.zeros(B, T, D_v, dtype=dtype, device=x.device)

    # 4D views sharing data_ptr with GEMM outputs
    q_4d = q_3d.view(B, T, H, K)
    k_4d = k_3d.view(B, T, H, K)
    v_4d = v_3d.view(B, T, H, V)

    # GDN intermediates and output
    gc = torch.zeros(B, T, H, device=x.device, dtype=torch.float32)
    w = torch.zeros(B, T, H, K, device=x.device, dtype=dtype)
    u = torch.zeros(B, T, H, V, device=x.device, dtype=dtype)
    o = torch.zeros(B, T, H, V, device=x.device, dtype=dtype)

    # Schedule all ops
    gemm_q_ops = GemmOp.schedule(a=x_3d, b=wq, c=q_3d)
    gemm_k_ops = GemmOp.schedule(a=x_3d, b=wk, c=k_3d)
    gemm_v_ops = GemmOp.schedule(a=x_3d, b=wv, c=v_3d)

    rope_q_ops = RopeOp.schedule(q=q_4d, cos=cos, sin=sin)
    rope_k_ops = RopeOp.schedule(q=k_4d, cos=cos, sin=sin)

    prep_ops = GDNPrepOp.schedule(
        k=k_4d, v=v_4d, g=g.contiguous(), beta=beta.contiguous(),
        g_cumsum=gc, w=w, u=u,
    )
    fused_ops = GDNFusedOp.schedule(
        q=q_4d, k=k_4d, w=w, u=u,
        g_cumsum=gc, o=o, scale=scale,
    )

    all_ops = (gemm_q_ops + gemm_k_ops + gemm_v_ops
               + rope_q_ops + rope_k_ops
               + prep_ops + fused_ops)

    config = _merged_config(
        GemmOp.kernel_config(gemm_q_ops + gemm_k_ops + gemm_v_ops),
        GDNFusedOp.kernel_config(prep_ops + fused_ops),
    )
    kernel = Megakernel(all_ops, config=config)
    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()
    torch.cuda.synchronize()

    return q_4d, k_4d, v_4d, o, kernel


def _run_sequential_megakernels(x, wq, wk, wv, g, beta, cos, sin,
                                B, T, H, K, V, scale):
    """Run the pipeline as 3 sequential megakernel launches."""
    from machete.megakernel import Megakernel
    from machete.kernels.gemm import GemmOp
    from machete.kernels.rope import RopeOp
    from machete.kernels.gated_delta_net.prep_op import GDNPrepOp
    from machete.kernels.gated_delta_net.fused_op import GDNFusedOp

    D_in = H * K
    D_v = H * V
    dtype = x.dtype

    x = x.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()

    # Step 1: GEMM (3D)
    x_3d = x.view(B, T, D_in)
    q_3d = torch.zeros(B, T, D_in, dtype=dtype, device=x.device)
    k_3d = torch.zeros(B, T, D_in, dtype=dtype, device=x.device)
    v_3d = torch.zeros(B, T, D_v, dtype=dtype, device=x.device)

    gemm_ops = (GemmOp.schedule(a=x_3d, b=wq, c=q_3d)
                + GemmOp.schedule(a=x_3d, b=wk, c=k_3d)
                + GemmOp.schedule(a=x_3d, b=wv, c=v_3d))
    gemm_kernel = Megakernel(gemm_ops, config=GemmOp.kernel_config(gemm_ops))
    with contextlib.redirect_stdout(io.StringIO()):
        gemm_kernel.run()
    torch.cuda.synchronize()

    # Step 2: RoPE (4D views sharing data_ptr with GEMM outputs)
    q_4d = q_3d.view(B, T, H, K)
    k_4d = k_3d.view(B, T, H, K)

    rope_ops = (RopeOp.schedule(q=q_4d, cos=cos, sin=sin)
                + RopeOp.schedule(q=k_4d, cos=cos, sin=sin))
    rope_kernel = Megakernel(rope_ops, config=RopeOp.kernel_config(rope_ops))
    with contextlib.redirect_stdout(io.StringIO()):
        rope_kernel.run()
    torch.cuda.synchronize()

    # Step 3: GDN (4D)
    q_4d = q_4d.contiguous()
    k_4d = k_4d.contiguous()
    v_4d = v_3d.view(B, T, H, V).contiguous()

    gc = torch.zeros(B, T, H, device=x.device, dtype=torch.float32)
    w = torch.zeros(B, T, H, K, device=x.device, dtype=dtype)
    u = torch.zeros(B, T, H, V, device=x.device, dtype=dtype)
    o = torch.zeros(B, T, H, V, device=x.device, dtype=dtype)

    gdn_ops = GDNPrepOp.schedule(
        k=k_4d, v=v_4d, g=g, beta=beta,
        g_cumsum=gc, w=w, u=u,
    )
    gdn_ops += GDNFusedOp.schedule(
        q=q_4d, k=k_4d, w=w, u=u,
        g_cumsum=gc, o=o, scale=scale,
    )
    gdn_kernel = Megakernel(gdn_ops, config=GDNFusedOp.kernel_config(gdn_ops))
    with contextlib.redirect_stdout(io.StringIO()):
        gdn_kernel.run()
    torch.cuda.synchronize()

    return q_4d, k_4d, v_4d, o


# =============================================================================
# Tests
# =============================================================================


@requires_gpu
class TestFusedGemmRope:
    """Test fused GEMM(Q,K,V) + RoPE(Q,K) in a single megakernel."""

    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 128, 2, 64, 64),
        (1, 128, 4, 128, 128),
        (2, 256, 4, 128, 128),
    ])
    def test_gemm_rope_vs_reference(self, B, T, H, K, V):
        """Compare fused GEMM+RoPE megakernel against PyTorch reference."""
        from machete.kernels.rope.ref import rope_pytorch

        x, wq, wk, wv, g, beta, cos, sin = _make_inputs(B, T, H, K, V)

        # Reference: fp32 GEMM + PyTorch RoPE
        q_ref = rope_pytorch(
            _gemm_reference(x, wq).view(B, T, H, K), cos, sin).to(x.dtype)
        k_ref = rope_pytorch(
            _gemm_reference(x, wk).view(B, T, H, K), cos, sin).to(x.dtype)
        v_ref = _gemm_reference(x, wv)

        # Megakernel: fused GEMM+RoPE
        q_flat, k_flat, v_flat, _ = _run_gemm_rope_megakernel(
            x, wq, wk, wv, cos, sin, B, T, H, K, V)

        torch.testing.assert_close(
            q_flat.view(B, T, H, K).float(), q_ref.float(),
            atol=1e-1, rtol=1e-2,
            msg="Q (GEMM+RoPE) mismatch",
        )
        torch.testing.assert_close(
            k_flat.view(B, T, H, K).float(), k_ref.float(),
            atol=1e-1, rtol=1e-2,
            msg="K (GEMM+RoPE) mismatch",
        )
        torch.testing.assert_close(
            v_flat.float(), v_ref.float(),
            atol=1e-1, rtol=1e-2,
            msg="V (GEMM) mismatch",
        )


@requires_all
class TestFullPipeline:
    """Test the full GEMM+RoPE+GDN pipeline (2 kernel launches) against reference."""

    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 128, 4, 128, 128),
    ])
    def test_pipeline_vs_reference(self, B, T, H, K, V):
        """Compare 2-launch pipeline against sequential PyTorch ops."""
        x, wq, wk, wv, g, beta, cos, sin = _make_inputs(B, T, H, K, V)
        scale = K ** -0.5

        _, _, _, o_ref = _reference_pipeline(
            x, wq, wk, wv, g, beta, cos, sin, B, T, H, K, V, scale)

        # Launch 1: fused GEMM+RoPE
        q_flat, k_flat, v_flat, _ = _run_gemm_rope_megakernel(
            x, wq, wk, wv, cos, sin, B, T, H, K, V)

        # Launch 2: GDN (separate kernel, contiguous 4D inputs)
        q_4d = q_flat.view(B, T, H, K).contiguous()
        k_4d = k_flat.view(B, T, H, K).contiguous()
        v_4d = v_flat.view(B, T, H, V).contiguous()
        o_mk, _ = _run_gdn_megakernel(q_4d, k_4d, v_4d, g, beta, scale)

        # GDN tolerances match existing GDN tests
        torch.testing.assert_close(
            o_mk.float(), o_ref.float(), atol=5e-1, rtol=2e-1,
            msg="GDN output mismatch",
        )


@requires_all
class TestFullyFused:
    """Test fully fused GEMM+RoPE+GDN in a single megakernel (1 launch)."""

    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 128, 2, 64, 64),
        (1, 128, 4, 128, 128),
        (2, 256, 4, 128, 128),
    ])
    def test_fully_fused_vs_reference(self, B, T, H, K, V):
        """Compare single-launch fused megakernel against PyTorch reference."""
        x, wq, wk, wv, g, beta, cos, sin = _make_inputs(B, T, H, K, V)
        scale = K ** -0.5

        _, _, _, o_ref = _reference_pipeline(
            x, wq, wk, wv, g, beta, cos, sin, B, T, H, K, V, scale)

        _, _, _, o_fused, _ = _run_fully_fused_megakernel(
            x, wq, wk, wv, g, beta, cos, sin, B, T, H, K, V, scale)

        torch.testing.assert_close(
            o_fused.float(), o_ref.float(), atol=5e-1, rtol=2e-1,
            msg="Fully fused GDN output mismatch vs reference",
        )
