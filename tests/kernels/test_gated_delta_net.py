# Copyright (c) 2025, Machete Authors
"""Tests for Gated Delta Net ops — correctness vs fla and naive PyTorch reference."""

import contextlib
import io

import pytest
import torch

from machete.kernels.gated_delta_net.ref import (
    gated_delta_rule_naive,
    fla_prep_stage,
    fla_state_recurrence,
    fla_output,
    fla_full_forward,
)


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


requires_gpu = pytest.mark.skipif(
    not (is_hopper_or_newer() and CUTLASS_AVAILABLE),
    reason="Requires Hopper+ GPU with CUTLASS",
)

requires_fla = pytest.mark.skipif(
    not FLA_AVAILABLE,
    reason="Requires fla (flash-linear-attention)",
)

requires_gpu_fla = pytest.mark.skipif(
    not (is_hopper_or_newer() and CUTLASS_AVAILABLE and FLA_AVAILABLE),
    reason="Requires Hopper+ GPU with CUTLASS and fla",
)


# =============================================================================
# Helpers
# =============================================================================


def _make_inputs(B, T, H, K, V, dtype=torch.float16, device="cuda"):
    """Create random inputs for gated delta net."""
    # Scale down inputs to prevent numerical instability
    q = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    k = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    v = torch.randn(B, T, H, V, dtype=dtype, device=device) * 0.1
    # g must be negative (log-space decay, always <= 0)
    g = -torch.rand(B, T, H, dtype=torch.float32, device=device) * 2.0
    # beta in [0, 1]
    beta = torch.sigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device))
    return q, k, v, g, beta


# =============================================================================
# Reference validation: naive vs fla
# =============================================================================


class TestNaiveVsFla:
    """Validate that our naive PyTorch impl matches fla's output."""

    @requires_gpu_fla
    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 64, 2, 64, 64),
        (1, 128, 2, 64, 64),
        (2, 128, 4, 128, 128),
    ])
    def test_naive_matches_fla(self, B, T, H, K, V):
        q, k, v, g, beta = _make_inputs(B, T, H, K, V)
        scale = K ** -0.5

        o_naive, _ = gated_delta_rule_naive(q, k, v, g, beta, scale=scale)
        o_fla, _ = fla_full_forward(q, k, v, g, beta, scale=scale)

        torch.testing.assert_close(o_naive.float(), o_fla.float(), atol=5e-1, rtol=2e-1)


# =============================================================================
# fla stage isolation: verify we can call stages independently
# =============================================================================


class TestFlaStages:
    """Verify that calling fla stages individually produces same result as end-to-end."""

    @requires_gpu_fla
    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 128, 4, 128, 128),
        (2, 256, 4, 128, 64),
    ])
    def test_stages_match_full(self, B, T, H, K, V):
        q, k, v, g, beta = _make_inputs(B, T, H, K, V)
        scale = K ** -0.5

        # Full forward
        o_full, _ = fla_full_forward(q, k, v, g, beta, scale=scale)

        # Stage-by-stage
        g_cumsum, A, w, u = fla_prep_stage(k, v, g, beta)
        h, v_new, _ = fla_state_recurrence(k, w, u, g_cumsum)
        o_staged = fla_output(q, k, v_new, h, g_cumsum, scale=scale)

        torch.testing.assert_close(o_staged.float(), o_full.float(), atol=1e-2, rtol=1e-2)


# =============================================================================
# Machete StateOp tests (to be enabled when implemented)
# =============================================================================


class TestGatedDeltaNetStateOp:
    """Test machete GatedDeltaNetStateOp against fla's state recurrence."""

    @requires_gpu_fla
    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 128, 2, 128, 64),
        (1, 256, 4, 128, 128),
        (2, 512, 4, 128, 64),
    ])
    def test_state_recurrence_forward(self, B, T, H, K, V):
        """Compare machete state recurrence against fla's chunk_gated_delta_rule_fwd_h."""
        q, k, v, g, beta = _make_inputs(B, T, H, K, V)

        # Use fla for prep stages
        g_cumsum, A, w, u = fla_prep_stage(k, v, g, beta)

        # fla reference
        h_ref, v_new_ref, _ = fla_state_recurrence(k, w, u, g_cumsum)

        # machete StateOp
        from machete.kernels.gated_delta_net.state import run_state_recurrence
        h_machete, v_new_machete = run_state_recurrence(k, w, u, g_cumsum)

        torch.testing.assert_close(
            h_machete.float(), h_ref.float(), atol=1e-2, rtol=1e-2,
        )
        torch.testing.assert_close(
            v_new_machete.float(), v_new_ref.float(), atol=1e-2, rtol=1e-2,
        )


# =============================================================================
# Machete OutputOp tests (to be enabled when implemented)
# =============================================================================


class TestGatedDeltaNetOutputOp:
    """Test machete GatedDeltaNetOutputOp against fla's output computation."""

    @requires_gpu_fla
    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 128, 2, 128, 64),
        (1, 256, 4, 128, 128),
    ])
    def test_output_forward(self, B, T, H, K, V):
        """Compare machete output op against fla's chunk_fwd_o."""
        q, k, v, g, beta = _make_inputs(B, T, H, K, V)
        scale = K ** -0.5

        # Use fla for all stages except output
        g_cumsum, A, w, u = fla_prep_stage(k, v, g, beta)
        h, v_new, _ = fla_state_recurrence(k, w, u, g_cumsum)

        # fla reference
        o_ref = fla_output(q, k, v_new, h, g_cumsum, scale=scale)

        # machete OutputOp
        from machete.kernels.gated_delta_net.output import run_output
        o_machete = run_output(q, k, v_new, h, g_cumsum, scale=scale)

        torch.testing.assert_close(
            o_machete.float(), o_ref.float(), atol=1e-2, rtol=1e-2,
        )


# =============================================================================
# Machete PrepOp tests
# =============================================================================


class TestGatedDeltaNetPrepOp:
    """Test machete prep stage against fla's preprocessing."""

    @requires_gpu_fla
    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 128, 2, 128, 64),
        (1, 256, 4, 128, 128),
        (2, 128, 4, 64, 64),
    ])
    def test_prep_forward(self, B, T, H, K, V):
        """Compare machete prep against fla's prep stages."""
        q, k, v, g, beta = _make_inputs(B, T, H, K, V)

        # fla reference
        g_ref, A_ref, w_ref, u_ref = fla_prep_stage(k, v, g, beta)

        # machete PrepOp
        from machete.kernels.gated_delta_net.prep import run_prep
        g_machete, A_machete, w_machete, u_machete = run_prep(k, v, g, beta)

        torch.testing.assert_close(
            g_machete.float(), g_ref.float(), atol=1e-5, rtol=1e-5,
        )
        torch.testing.assert_close(
            w_machete.float(), w_ref.float(), atol=5e-2, rtol=5e-2,
        )
        torch.testing.assert_close(
            u_machete.float(), u_ref.float(), atol=5e-2, rtol=5e-2,
        )


# =============================================================================
# Full forward pipeline tests
# =============================================================================


class TestGatedDeltaNetFullForward:
    """Test full machete forward pipeline (prep→state→output) vs fla."""

    @requires_gpu_fla
    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 128, 2, 128, 64),
        (1, 256, 4, 128, 128),
        (2, 512, 4, 128, 64),
    ])
    def test_full_forward(self, B, T, H, K, V):
        """Compare full machete pipeline against fla end-to-end."""
        q, k, v, g, beta = _make_inputs(B, T, H, K, V)
        scale = K ** -0.5

        # fla reference
        o_ref, _ = fla_full_forward(q, k, v, g, beta, scale=scale)

        # machete full pipeline
        from machete.kernels.gated_delta_net.prep import run_prep
        from machete.kernels.gated_delta_net.state import run_state_recurrence
        from machete.kernels.gated_delta_net.output import run_output

        g_cumsum, A, w, u = run_prep(k, v, g, beta)
        h, v_new = run_state_recurrence(k, w, u, g_cumsum)
        o_machete = run_output(q, k, v_new, h, g_cumsum, scale=scale)

        torch.testing.assert_close(
            o_machete.float(), o_ref.float(), atol=5e-1, rtol=2e-1,
        )


# =============================================================================
# Backward stage tests
# =============================================================================


class TestGatedDeltaNetBwdDvLocal:
    """Test machete bwd_dv_local against fla."""

    @requires_gpu_fla
    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 128, 2, 128, 64),
        (1, 256, 4, 128, 128),
    ])
    def test_bwd_dv_local(self, B, T, H, K, V):
        """Compare machete dv_local against fla's."""
        q, k, v, g, beta = _make_inputs(B, T, H, K, V)
        scale = K ** -0.5
        do = torch.randn(B, T, H, V, dtype=q.dtype, device="cuda") * 0.1

        g_cumsum, A, w, u = fla_prep_stage(k, v, g, beta)

        # fla reference
        from machete.kernels.gated_delta_net.ref import fla_bwd_dv_local
        dv_ref = fla_bwd_dv_local(q, k, g_cumsum, do, scale=scale)

        # machete
        from machete.kernels.gated_delta_net.grad import run_bwd_dv_local
        dv_machete = run_bwd_dv_local(q, k, g_cumsum, do, scale=scale)

        torch.testing.assert_close(
            dv_machete.float(), dv_ref.float(), atol=5e-2, rtol=5e-2,
        )


class TestGatedDeltaNetBwdState:
    """Test machete backward state recurrence against fla."""

    @requires_gpu_fla
    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 128, 2, 128, 64),
        (1, 256, 4, 128, 128),
    ])
    def test_bwd_state_recurrence(self, B, T, H, K, V):
        """Compare machete backward state recurrence against fla's."""
        q, k, v, g, beta = _make_inputs(B, T, H, K, V)
        scale = K ** -0.5
        do = torch.randn(B, T, H, V, dtype=q.dtype, device="cuda") * 0.1

        g_cumsum, A, w, u = fla_prep_stage(k, v, g, beta)

        # Get dv_local from fla
        from machete.kernels.gated_delta_net.ref import fla_bwd_dv_local, fla_bwd_state_recurrence
        dv_local = fla_bwd_dv_local(q, k, g_cumsum, do, scale=scale)

        # Forward state recurrence to get h (needed for dw computation)
        from machete.kernels.gated_delta_net.ref import fla_state_recurrence
        h, v_new, _ = fla_state_recurrence(k, w, u, g_cumsum)

        # fla reference
        dh_ref, dh0_ref, dv2_ref = fla_bwd_state_recurrence(
            q, k, w, g_cumsum, h0=None, dht=None, do=do, dv=dv_local, scale=scale,
        )

        # fla reference for dw (from chunk_bwd_dqkwg)
        from machete.kernels.gated_delta_net.grad import run_bwd_dqkwg
        _, _, dw_ref, _ = run_bwd_dqkwg(
            q, k, v_new, w, g_cumsum, h, dv2_ref, do,
            dh_ref, scale,
        )

        # machete
        from machete.kernels.gated_delta_net.state_bwd import run_bwd_state_recurrence
        dh_machete, dh0_machete, dv2_machete, dw_machete = run_bwd_state_recurrence(
            q, k, w, g_cumsum, h0=None, dht=None, do=do, dv_local=dv_local,
            h=h, scale=scale,
        )

        torch.testing.assert_close(
            dh_machete.float(), dh_ref.float(), atol=5e-2, rtol=5e-2,
        )
        torch.testing.assert_close(
            dv2_machete.float(), dv2_ref.float(), atol=5e-2, rtol=5e-2,
        )
        torch.testing.assert_close(
            dw_machete.float(), dw_ref.float(), atol=5e-2, rtol=5e-2,
        )


class TestGatedDeltaNetEndToEnd:
    """Test full machete end-to-end with autograd backward."""

    @requires_gpu_fla
    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 128, 2, 128, 64),
    ])
    def test_forward_matches_fla(self, B, T, H, K, V):
        """Test the public API forward matches fla."""
        q, k, v, g, beta = _make_inputs(B, T, H, K, V)
        scale = K ** -0.5

        from machete.kernels.gated_delta_net import chunk_gated_delta_rule

        o_ref, _ = fla_full_forward(q, k, v, g, beta, scale=scale)
        o_machete, _ = chunk_gated_delta_rule(q, k, v, g, beta, scale=scale)

        torch.testing.assert_close(
            o_machete.float(), o_ref.float(), atol=5e-1, rtol=2e-1,
        )


def _run_prep_op(k, v, g, beta):
    """Run GDNPrepOp megakernel and return outputs in [B, T, H, ...] layout.

    Native [B, T, H, K/V] layout — no transposes needed.
    """
    from machete.megakernel import Megakernel
    from machete.kernels.gated_delta_net.prep_op import GDNPrepOp

    B, T, H, K = k.shape
    V = v.shape[-1]
    dtype = k.dtype

    gc = torch.zeros(B, T, H, device=k.device, dtype=torch.float32)
    w = torch.zeros(B, T, H, K, device=k.device, dtype=dtype)
    u = torch.zeros(B, T, H, V, device=k.device, dtype=dtype)

    ops = GDNPrepOp.schedule(
        k=k.contiguous(), v=v.contiguous(),
        g=g.contiguous(), beta=beta.contiguous(),
        g_cumsum=gc, w=w, u=u,
    )
    config = GDNPrepOp.kernel_config(ops)
    kernel = Megakernel(ops, config=config)
    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()
    torch.cuda.synchronize()

    return gc, w, u


class TestGDNPrepMegakernel:
    """Test GDNPrepOp megakernel Op against fla's preprocessing."""

    @requires_gpu_fla
    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 128, 2, 128, 64),
        (1, 128, 4, 128, 128),
        (1, 256, 4, 128, 128),
        (2, 256, 4, 128, 64),
    ])
    def test_prep_op_vs_fla(self, B, T, H, K, V):
        """Compare GDNPrepOp megakernel against fla's prep stages."""
        q, k, v, g, beta = _make_inputs(B, T, H, K, V)

        # fla reference
        g_ref, A_ref, w_ref, u_ref = fla_prep_stage(k, v, g, beta)

        # GDNPrepOp megakernel
        gc_mk, w_mk, u_mk = _run_prep_op(k, v, g, beta)

        torch.testing.assert_close(
            gc_mk.float(), g_ref.float(), atol=1e-5, rtol=1e-5,
        )
        torch.testing.assert_close(
            w_mk.float(), w_ref.float(), atol=1e-1, rtol=5e-2,
        )
        torch.testing.assert_close(
            u_mk.float(), u_ref.float(), atol=1e-1, rtol=5e-2,
        )


# =============================================================================
# Fused megakernel tests (PrepOp + FusedOp in one kernel)
# =============================================================================


class TestGDNFusedMegakernel:
    """Test fused PrepOp+FusedOp single megakernel against fla."""

    @requires_gpu_fla
    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 128, 2, 128, 64),
        (1, 256, 4, 128, 128),
        (2, 256, 4, 128, 64),
    ])
    def test_fused_megakernel_vs_fla(self, B, T, H, K, V):
        """Compare fused PrepOp+FusedOp megakernel against fla end-to-end."""
        from machete.kernels.gated_delta_net import (
            HAS_MEGAKERNEL_OPS, _run_fused_megakernel,
        )
        if not HAS_MEGAKERNEL_OPS:
            pytest.skip("Megakernel Ops not available")

        q, k, v, g, beta = _make_inputs(B, T, H, K, V)
        scale = K ** -0.5

        # fla reference
        o_ref, _ = fla_full_forward(q, k, v, g, beta, scale=scale)

        # Fused megakernel
        o_mk, _, _, _, _ = _run_fused_megakernel(q, k, v, g, beta, scale)

        torch.testing.assert_close(
            o_mk.float(), o_ref.float(), atol=5e-1, rtol=2e-1,
        )


# =============================================================================
# GDNFusedOp tests (fused state+output, no h_states/v_new intermediates)
# =============================================================================


def _run_fused_op(q, k, w, u, g_cumsum, scale):
    """Run GDNFusedOp and return output."""
    from machete.megakernel import Megakernel
    from machete.kernels.gated_delta_net.fused_op import GDNFusedOp

    B, T, H, K = q.shape
    V = u.shape[-1]
    dtype = q.dtype

    o = torch.zeros(B, T, H, V, device=q.device, dtype=dtype)

    ops = GDNFusedOp.schedule(
        q=q.contiguous(), k=k.contiguous(),
        w=w.contiguous(), u=u.contiguous(),
        g_cumsum=g_cumsum.contiguous(), o=o,
        scale=scale,
    )
    config = GDNFusedOp.kernel_config(ops)
    kernel = Megakernel(ops, config=config)
    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()
    torch.cuda.synchronize()

    return o


class TestGDNFusedOp:
    """Test GDNFusedOp (fused state+output) against fla end-to-end."""

    @requires_gpu_fla
    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 64, 1, 128, 128),
        (1, 128, 2, 128, 64),
        (1, 256, 4, 128, 128),
        (2, 512, 4, 128, 64),
    ])
    def test_fused_op_vs_fla(self, B, T, H, K, V):
        """Compare GDNFusedOp against fla's full pipeline."""
        q, k, v, g, beta = _make_inputs(B, T, H, K, V)
        scale = K ** -0.5

        # fla prep + reference output
        g_cumsum, A, w, u = fla_prep_stage(k, v, g, beta)
        h_ref, v_new_ref, _ = fla_state_recurrence(k, w, u, g_cumsum)
        o_ref = fla_output(q, k, v_new_ref, h_ref, g_cumsum, scale=scale)

        # GDNFusedOp (no h_states or v_new intermediates)
        o_fused = _run_fused_op(q, k, w, u, g_cumsum, scale)

        torch.testing.assert_close(
            o_fused.float(), o_ref.float(), atol=8e-2, rtol=5e-2,
        )


# =============================================================================
# 4-Op decomposition tests (SolveOp, WUOp, StateOp, OutputOp)
# =============================================================================


def _run_solve_op(k, g, beta, page_size=49152):
    """Run GDNSolveOp megakernel."""
    from machete.megakernel import Megakernel
    from machete.kernels.gated_delta_net.solve_op import GDNSolveOp

    B, T, H, K = k.shape
    dtype = k.dtype

    gc = torch.zeros(B, T, H, device=k.device, dtype=torch.float32)
    a_solved = torch.zeros(B, T, H, 64, device=k.device, dtype=dtype)

    ops = GDNSolveOp.schedule(
        k=k.contiguous(), g=g.contiguous(), beta=beta.contiguous(),
        g_cumsum=gc, a_solved=a_solved,
        page_size=page_size,
    )
    config = GDNSolveOp.kernel_config(ops)
    kernel = Megakernel(ops, config=config)
    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()
    torch.cuda.synchronize()

    return gc, a_solved


def _run_wu_op(a_solved, k, v, g_cumsum, beta, page_size=49152):
    """Run GDNWUOp megakernel."""
    from machete.megakernel import Megakernel
    from machete.kernels.gated_delta_net.wu_op import GDNWUOp

    B, T, H, K = k.shape
    V = v.shape[-1]
    dtype = k.dtype

    w = torch.zeros(B, T, H, K, device=k.device, dtype=dtype)
    u = torch.zeros(B, T, H, V, device=k.device, dtype=dtype)

    ops = GDNWUOp.schedule(
        a_solved=a_solved.contiguous(), k=k.contiguous(),
        v=v.contiguous(), g_cumsum=g_cumsum.contiguous(),
        beta=beta.contiguous(), w=w, u=u,
        page_size=page_size,
    )
    config = GDNWUOp.kernel_config(ops)
    kernel = Megakernel(ops, config=config)
    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()
    torch.cuda.synchronize()

    return w, u


def _run_output_op(q, k, v_new, h_states, g_cumsum, scale, page_size=49152):
    """Run GDNOutputOp megakernel."""
    from machete.megakernel import Megakernel
    from machete.kernels.gated_delta_net.output_op import GDNOutputOp

    B, T, H, K = q.shape
    V = v_new.shape[-1]
    dtype = q.dtype

    o = torch.zeros(B, T, H, V, device=q.device, dtype=dtype)

    ops = GDNOutputOp.schedule(
        q=q.contiguous(), k=k.contiguous(),
        v_new=v_new.contiguous(), h_states=h_states.contiguous(),
        g_cumsum=g_cumsum.contiguous(), o=o, scale=scale,
        page_size=page_size,
    )
    config = GDNOutputOp.kernel_config(ops)
    kernel = Megakernel(ops, config=config)
    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()
    torch.cuda.synchronize()

    return o


class TestGDNSolveOp:
    """Test GDNSolveOp (phases 1-3) against fla prep."""

    @requires_gpu_fla
    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 128, 2, 128, 64),
        (1, 256, 4, 128, 128),
        (2, 256, 4, 128, 64),
    ])
    def test_solve_g_cumsum(self, B, T, H, K, V):
        """Verify g_cumsum matches fla."""
        q, k, v, g, beta = _make_inputs(B, T, H, K, V)
        g_ref, _, _, _ = fla_prep_stage(k, v, g, beta)
        gc_mk, _ = _run_solve_op(k, g, beta)

        torch.testing.assert_close(
            gc_mk.float(), g_ref.float(), atol=1e-5, rtol=1e-5,
        )


class TestGDNSolveWUOp:
    """Test SolveOp + WUOp combined matches PrepOp output."""

    @requires_gpu_fla
    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 128, 2, 128, 64),
        (1, 256, 4, 128, 128),
        (2, 256, 4, 128, 64),
    ])
    def test_solve_wu_vs_prep(self, B, T, H, K, V):
        """SolveOp + WUOp should produce same w, u as PrepOp."""
        q, k, v, g, beta = _make_inputs(B, T, H, K, V)

        # fla reference
        g_ref, _, w_ref, u_ref = fla_prep_stage(k, v, g, beta)

        # 2-op path: SolveOp → WUOp
        gc_mk, a_solved = _run_solve_op(k, g, beta)
        w_mk, u_mk = _run_wu_op(a_solved, k, v, gc_mk, beta)

        torch.testing.assert_close(
            gc_mk.float(), g_ref.float(), atol=1e-5, rtol=1e-5,
        )
        torch.testing.assert_close(
            w_mk.float(), w_ref.float(), atol=1e-1, rtol=5e-2,
        )
        torch.testing.assert_close(
            u_mk.float(), u_ref.float(), atol=1e-1, rtol=5e-2,
        )


class TestGDNOutputOp:
    """Test GDNOutputOp against fla output."""

    @requires_gpu_fla
    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 128, 2, 128, 64),
        (1, 256, 4, 128, 128),
    ])
    def test_output_op_vs_fla(self, B, T, H, K, V):
        """Compare GDNOutputOp against fla output."""
        q, k, v, g, beta = _make_inputs(B, T, H, K, V)
        scale = K ** -0.5

        g_cumsum, _, w, u = fla_prep_stage(k, v, g, beta)
        h_ref, v_new_ref, _ = fla_state_recurrence(k, w, u, g_cumsum)
        o_ref = fla_output(q, k, v_new_ref, h_ref, g_cumsum, scale=scale)

        # GDNOutputOp
        o_mk = _run_output_op(q, k, v_new_ref, h_ref, g_cumsum, scale)

        torch.testing.assert_close(
            o_mk.float(), o_ref.float(), atol=5e-2, rtol=2e-2,
        )


# =============================================================================
# 5-Op decomposition tests (StateRecurrenceOp + VNewOp)
# =============================================================================


def _run_state_recurrence_op(k, w, u, g_cumsum, page_size=49152):
    """Run GDNStateRecurrenceOp megakernel."""
    from machete.megakernel import Megakernel
    from machete.kernels.gated_delta_net.state_recurrence_op import GDNStateRecurrenceOp

    B, T, H, K = k.shape
    V = u.shape[-1]
    NT = T // 64
    dtype = k.dtype

    h_states = torch.zeros(B, NT, H, K, V, device=k.device, dtype=dtype)

    ops = GDNStateRecurrenceOp.schedule(
        k=k.contiguous(), w=w.contiguous(), u=u.contiguous(),
        g_cumsum=g_cumsum.contiguous(),
        h_states=h_states,
        page_size=page_size,
    )
    config = GDNStateRecurrenceOp.kernel_config(ops)
    kernel = Megakernel(ops, config=config)
    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()
    torch.cuda.synchronize()

    return h_states


def _run_vnew_op(w, u, h_states, page_size=49152):
    """Run GDNVNewOp megakernel."""
    from machete.megakernel import Megakernel
    from machete.kernels.gated_delta_net.vnew_op import GDNVNewOp

    B, T, H, K = w.shape
    V = u.shape[-1]
    dtype = w.dtype

    v_new = torch.zeros(B, T, H, V, device=w.device, dtype=dtype)

    ops = GDNVNewOp.schedule(
        w=w.contiguous(), u=u.contiguous(),
        h_states=h_states.contiguous(),
        v_new=v_new,
        page_size=page_size,
    )
    config = GDNVNewOp.kernel_config(ops)
    kernel = Megakernel(ops, config=config)
    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()
    torch.cuda.synchronize()

    return v_new


class TestGDNStateRecurrenceOp:
    """Test GDNStateRecurrenceOp h_states against fla."""

    @requires_gpu_fla
    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 128, 2, 128, 64),
        (1, 256, 4, 128, 128),
        (2, 256, 4, 128, 64),
    ])
    def test_h_states_vs_fla(self, B, T, H, K, V):
        """Compare GDNStateRecurrenceOp h_states against fla."""
        q, k, v, g, beta = _make_inputs(B, T, H, K, V)
        g_cumsum, _, w, u = fla_prep_stage(k, v, g, beta)

        # fla reference
        h_ref, _, _ = fla_state_recurrence(k, w, u, g_cumsum)

        # StateRecurrenceOp
        h_mk = _run_state_recurrence_op(k, w, u, g_cumsum)

        torch.testing.assert_close(
            h_mk.float(), h_ref.float(), atol=5e-2, rtol=2e-2,
        )


class TestGDNVNewOp:
    """Test GDNVNewOp v_new against fla."""

    @requires_gpu_fla
    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 128, 2, 128, 64),
        (1, 256, 4, 128, 128),
        (2, 256, 4, 128, 64),
    ])
    def test_vnew_vs_fla(self, B, T, H, K, V):
        """Compare GDNVNewOp v_new against fla state recurrence."""
        q, k, v, g, beta = _make_inputs(B, T, H, K, V)
        g_cumsum, _, w, u = fla_prep_stage(k, v, g, beta)

        # fla reference
        h_ref, v_new_ref, _ = fla_state_recurrence(k, w, u, g_cumsum)

        # VNewOp using fla h_states (isolate VNewOp correctness)
        v_new_mk = _run_vnew_op(w, u, h_ref)

        torch.testing.assert_close(
            v_new_mk.float(), v_new_ref.float(), atol=5e-2, rtol=2e-2,
        )


class TestGDNStateRecurrenceVNewOp:
    """Test combined StateRecurrenceOp + VNewOp pipeline."""

    @requires_gpu_fla
    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 128, 2, 128, 64),
        (1, 256, 4, 128, 128),
        (2, 256, 4, 128, 64),
    ])
    def test_combined_vs_fla(self, B, T, H, K, V):
        """StateRecurrenceOp + VNewOp should match fla state recurrence."""
        q, k, v, g, beta = _make_inputs(B, T, H, K, V)
        g_cumsum, _, w, u = fla_prep_stage(k, v, g, beta)

        # fla reference
        h_ref, v_new_ref, _ = fla_state_recurrence(k, w, u, g_cumsum)

        # 2-step: StateRecurrenceOp → VNewOp
        h_mk = _run_state_recurrence_op(k, w, u, g_cumsum)
        v_new_mk = _run_vnew_op(w, u, h_mk)

        torch.testing.assert_close(
            h_mk.float(), h_ref.float(), atol=5e-2, rtol=2e-2,
        )
        torch.testing.assert_close(
            v_new_mk.float(), v_new_ref.float(), atol=5e-2, rtol=2e-2,
        )


class TestGDN5OpMegakernel:
    """Test full 5-op fused megakernel against fla."""

    @requires_gpu_fla
    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 128, 2, 128, 64),
        (1, 256, 4, 128, 128),
        (2, 256, 4, 128, 64),
    ])
    def test_5op_vs_fla(self, B, T, H, K, V):
        """Compare 5-op fused megakernel against fla end-to-end."""
        from machete.kernels.gated_delta_net import HAS_5OP, _run_5op_megakernel
        if not HAS_5OP:
            pytest.skip("5-Op decomposition not available")

        q, k, v, g, beta = _make_inputs(B, T, H, K, V)
        scale = K ** -0.5

        o_ref, _ = fla_full_forward(q, k, v, g, beta, scale=scale)
        o_5op = _run_5op_megakernel(q, k, v, g, beta, scale)

        torch.testing.assert_close(
            o_5op.float(), o_ref.float(), atol=5e-1, rtol=2e-1,
        )
