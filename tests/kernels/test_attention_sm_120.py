# Copyright (c) 2025, Machete Authors
"""Tests for FlashAttentionSm120Op — cooperative forward correctness (fp16/bf16).

Tests run on GPU (Hopper+) and compare the megakernel FlashAttentionSm120Op
against a pure PyTorch reference implementation.
"""

import contextlib
import io
import importlib.util

import pytest
import torch

if importlib.util.find_spec("cutlass") is None:
    pytest.skip("Requires CUTLASS", allow_module_level=True)

from tests.kernels.support import requires_hopper_cutlass
from machete.kernels.attention.ref import flash_attention_pytorch, flash_attention_backward_pytorch

requires_gpu = requires_hopper_cutlass


# =============================================================================
# Helpers
# =============================================================================


def _run_attention_coop_forward(q, k, v, tile_m=None, causal=False,
                                kv_group_size=1):
    """Run FlashAttentionSm120Op forward and return output tensor."""
    from machete.megakernel import Megakernel
    from machete.kernels.attention import FlashAttentionSm120Op

    tile_sizes = {}
    if tile_m is not None:
        tile_sizes["M"] = tile_m
    o = torch.zeros_like(q)
    ops = FlashAttentionSm120Op.schedule(
        q=q, k=k, v=v, o=o,
        tile_sizes=tile_sizes,
        causal=causal,
        kv_group_size=kv_group_size,
    )
    config = FlashAttentionSm120Op.kernel_config(ops)
    kernel = Megakernel(ops, config=config)

    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()

    torch.cuda.synchronize()
    return o


def _run_attention_packgqa_forward(q, k, v, tile_m=64, causal=False,
                                   kv_group_size=1):
    from machete.megakernel import Megakernel
    from machete.kernels.attention import FlashAttentionSm120PackGQAOp

    o = torch.zeros_like(q)
    ops = FlashAttentionSm120PackGQAOp.schedule(
        q=q, k=k, v=v, o=o,
        tile_sizes={"M": tile_m},
        causal=causal,
        kv_group_size=kv_group_size,
        write_lse=False,
    )
    config = FlashAttentionSm120PackGQAOp.kernel_config(ops)
    kernel = Megakernel(ops, config=config)

    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()

    torch.cuda.synchronize()
    return o


def _run_attention_direct_forward(q, k, v, tile_m=64, causal=False,
                                  kv_group_size=1):
    from machete.megakernel import Megakernel
    from machete.kernels.attention import FlashAttentionSm120DirectOp

    o = torch.zeros_like(q)
    ops = FlashAttentionSm120DirectOp.schedule(
        q=q, k=k, v=v, o=o,
        tile_sizes={"M": tile_m},
        causal=causal,
        kv_group_size=kv_group_size,
        write_lse=False,
    )
    config = FlashAttentionSm120DirectOp.kernel_config(ops)
    kernel = Megakernel(ops, config=config)

    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()

    torch.cuda.synchronize()
    return o


def _build_attention_coop_kernel(q, k, v, tile_m=None, causal=False, kv_group_size=1):
    """Build and run FlashAttentionSm120Op, returning kernel and output."""
    from machete.megakernel import Megakernel
    from machete.kernels.attention import FlashAttentionSm120Op

    tile_sizes = {}
    if tile_m is not None:
        tile_sizes["M"] = tile_m
    o = torch.zeros_like(q)
    ops = FlashAttentionSm120Op.schedule(
        q=q, k=k, v=v, o=o,
        tile_sizes=tile_sizes,
        causal=causal,
        kv_group_size=kv_group_size,
    )
    config = FlashAttentionSm120Op.kernel_config(ops)
    kernel = Megakernel(ops, config=config)

    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()

    torch.cuda.synchronize()
    return kernel, o


def _run_attention_dpsum(dout, o):
    from machete.megakernel import Megakernel, MegakernelConfig
    from machete.kernels.attention import AttentionDPSumOp

    dpsum = torch.empty(
        dout.shape[0], dout.shape[2], dout.shape[1],
        dtype=torch.float32, device=dout.device,
    )
    ops = AttentionDPSumOp.schedule(dout=dout, o=o, dpsum=dpsum)
    kernel = Megakernel(ops, config=MegakernelConfig(num_sms=2))
    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()
    torch.cuda.synchronize()
    return dpsum


def _assert_attention_forward_close(q, k, v, *, tile_m=None, causal=False, kv_group_size=1):
    o_mk = _run_attention_coop_forward(
        q, k, v, tile_m=tile_m, causal=causal, kv_group_size=kv_group_size,
    )
    o_ref = flash_attention_pytorch(
        q.float(), k.float(), v.float(),
        causal=causal, kv_group_size=kv_group_size,
    ).half()
    torch.testing.assert_close(o_mk, o_ref, atol=5e-2, rtol=5e-2)


def _assert_attention_backward_close(q, k, v, dout, *, causal=False, kv_group_size=1):
    o, lse = _run_attention_coop_forward_with_lse(
        q, k, v, causal=causal, kv_group_size=kv_group_size,
    )
    dq, dk, dv = _run_attention_coop_backward(
        q, k, v, o, dout, lse, causal=causal, kv_group_size=kv_group_size,
    )
    dq_ref, dk_ref, dv_ref = flash_attention_backward_pytorch(
        q, k, v, o, dout, causal=causal, kv_group_size=kv_group_size,
    )
    torch.testing.assert_close(dv, dv_ref.half(), atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(dk, dk_ref.half(), atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(dq, dq_ref.half(), atol=5e-2, rtol=5e-2)


MMA_FORWARD_CASES = [
    pytest.param(1, 16, 16, 64, id="exact_tile_d64"),
    pytest.param(1, 16, 16, 128, id="exact_tile_d128"),
    pytest.param(1, 16, 20, 64, id="partial_n"),
    pytest.param(4, 32, 32, 128, id="multi_head_multi_m"),
    pytest.param(1, 16, 256, 128, id="large_n"),
]

MULTIWARP_FORWARD_CASES = [
    pytest.param(1, 64, 64, 64, id="base_d64"),
    pytest.param(1, 128, 128, 128, id="multi_m_tiles"),
    pytest.param(1, 80, 64, 128, id="partial_m"),
    pytest.param(1, 64, 50, 128, id="partial_n"),
    pytest.param(4, 16, 128, 128, id="decode_multi_head"),
]

CAUSAL_FORWARD_CASES = [
    pytest.param(1, 32, 32, 64, id="square"),
    pytest.param(1, 16, 64, 64, id="asymmetric"),
    pytest.param(4, 32, 32, 64, id="multi_head"),
    pytest.param(1, 64, 64, 128, id="d128"),
]

BACKWARD_CASES = [
    pytest.param(1, 32, 32, 64, False, id="base_d64"),
    pytest.param(1, 32, 32, 128, False, id="base_d128"),
    pytest.param(4, 32, 32, 128, False, id="multi_head"),
    pytest.param(1, 64, 32, 128, False, id="multi_m"),
    pytest.param(1, 32, 64, 128, False, id="multi_n"),
    pytest.param(1, 32, 32, 128, True, id="causal"),
]

GQA_FORWARD_CASES = [
    pytest.param(4, 2, 32, 32, 64, False, id="gqa2_d64"),
    pytest.param(8, 2, 32, 32, 128, False, id="gqa4_d128"),
    pytest.param(4, 2, 32, 32, 64, True, id="gqa2_d64_causal"),
]

GQA_BACKWARD_CASES = [
    pytest.param(4, 2, 32, 32, 64, False, id="gqa2_d64"),
    pytest.param(8, 2, 32, 32, 128, False, id="gqa4_d128"),
    pytest.param(4, 2, 32, 32, 64, True, id="gqa2_d64_causal"),
]


# =============================================================================
# fp16 MMA Tests (tensor core path)
# =============================================================================


class TestFlashAttentionCoopMMA:
    """fp16 tensor core MMA path correctness tests (cooperative)."""

    def test_3d_schedule_keeps_m_tile_and_tuning_static_dims(self):
        """3D BHMD compatibility should not run through BSHD view conversion."""
        from machete.kernels.attention import FlashAttentionSm120Op

        q = torch.empty(2, 32, 64, dtype=torch.float16, device="cuda")
        k = torch.empty(2, 32, 64, dtype=torch.float16, device="cuda")
        v = torch.empty(2, 32, 64, dtype=torch.float16, device="cuda")
        o = torch.empty_like(q)

        ops = FlashAttentionSm120Op.schedule(
            q=q,
            k=k,
            v=v,
            o=o,
            tile_sizes={"M": 16},
            num_mma_warps=1,
            write_lse=False,
        )

        assert ops[0].tile_sizes["M"] == 16
        assert ops[0].static_dims["H"] == 2
        assert ops[0].static_dims["num_mma_warps"] == 1
        assert ops[0].static_dims["write_lse"] == 0

    @requires_gpu
    @pytest.mark.parametrize("BH,M,N,D", MMA_FORWARD_CASES)
    def test_mma_forward_matrix(self, BH, M, N, D):
        """Representative exact-tile MMA forward coverage."""
        torch.manual_seed(42)
        q = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        _assert_attention_forward_close(q, k, v, tile_m=16)

    @requires_gpu
    def test_dpsum_matches_pytorch(self):
        torch.manual_seed(42)
        B, S, H, D = 2, 64, 8, 128
        dout = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
        o = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")

        mk = _run_attention_dpsum(dout, o)
        ref = (dout.float() * o.float()).sum(dim=-1).permute(0, 2, 1).contiguous()

        torch.testing.assert_close(mk, ref, atol=1e-4, rtol=1e-4)

    @requires_gpu
    def test_mma_uniform_attention(self):
        """Uniform keys -> each output row equals mean of V."""
        BH, M, N, D = 1, 16, 16, 64
        q = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        k_row = torch.randn(1, 1, D, dtype=torch.float16, device="cuda")
        k = k_row.expand(BH, N, D).contiguous()
        v = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")

        o_mk = _run_attention_coop_forward(q, k, v, tile_m=16)
        expected = v.float().mean(dim=1, keepdim=True).expand_as(q).half()

        torch.testing.assert_close(o_mk, expected, atol=5e-2, rtol=5e-2)

# =============================================================================
# Multi-warp MMA Tests (auto tile_M)
# =============================================================================


class TestFlashAttentionCoopMultiWarp:
    """Multi-warp MMA tests — let schedule pick optimal tile_M."""

    @requires_gpu
    @pytest.mark.parametrize("BH,M,N,D", MULTIWARP_FORWARD_CASES)
    def test_multi_warp_forward_matrix(self, BH, M, N, D):
        """Representative auto-tiled multi-warp forward coverage."""
        torch.manual_seed(42)
        q = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        _assert_attention_forward_close(q, k, v)

    @requires_gpu
    def test_sequence_dynamic_reuses_compiled_kernel(self):
        """Different N should reuse the same compiled FlashAttention kernel."""
        torch.manual_seed(0)
        BH, M, D = 1, 16, 256

        q1 = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        k1 = torch.randn(BH, 144, D, dtype=torch.float16, device="cuda")
        v1 = torch.randn(BH, 144, D, dtype=torch.float16, device="cuda")
        kfa1, o1 = _build_attention_coop_kernel(q1, k1, v1, tile_m=16)

        q2 = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        k2 = torch.randn(BH, 272, D, dtype=torch.float16, device="cuda")
        v2 = torch.randn(BH, 272, D, dtype=torch.float16, device="cuda")
        kfa2, o2 = _build_attention_coop_kernel(q2, k2, v2, tile_m=16)

        assert kfa1._make_cache_key() == kfa2._make_cache_key()
        assert kfa1._compiled_kernel is kfa2._compiled_kernel
        torch.testing.assert_close(o1, flash_attention_pytorch(q1.float(), k1.float(), v1.float()).half(), atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(o2, flash_attention_pytorch(q2.float(), k2.float(), v2.float()).half(), atol=5e-2, rtol=5e-2)


# =============================================================================
# Causal Tests (fp16 MMA)
# =============================================================================


class TestFlashAttentionCoopCausal:
    """Causal masking tests (fp16, cooperative)."""

    @requires_gpu
    @pytest.mark.parametrize("BH,M,N,D", CAUSAL_FORWARD_CASES)
    def test_causal_forward_matrix(self, BH, M, N, D):
        """Representative causal forward coverage."""
        torch.manual_seed(42)
        q = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        _assert_attention_forward_close(q, k, v, causal=True)


# =============================================================================
# Backward Helpers
# =============================================================================


def _run_attention_coop_forward_with_lse(q, k, v, causal=False,
                                         kv_group_size=1):
    """Run forward and return (o, lse) tensors."""
    from machete.megakernel import Megakernel
    from machete.kernels.attention import FlashAttentionSm120Op

    o = torch.zeros_like(q)
    lse = torch.empty(q.shape[0], q.shape[1], dtype=torch.float32, device=q.device)
    ops = FlashAttentionSm120Op.schedule(
        q=q, k=k, v=v, o=o, lse=lse, causal=causal,
        kv_group_size=kv_group_size,
    )
    config = FlashAttentionSm120Op.kernel_config(ops)
    kernel = Megakernel(ops, config=config)

    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()

    torch.cuda.synchronize()
    return o, lse


def _run_attention_coop_backward(q, k, v, o, dout, lse, causal=False,
                                 kv_group_size=1):
    """Run FlashAttentionSm120BwdOp and return (dq, dk, dv)."""
    from machete.megakernel import Megakernel
    from machete.kernels.attention import FlashAttentionSm120BwdOp

    dpsum = (dout.float() * o.float()).sum(dim=-1).contiguous()
    dq_accum = torch.zeros(q.shape[0], q.shape[1], q.shape[2],
                           dtype=torch.float32, device=q.device)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    ops = FlashAttentionSm120BwdOp.schedule(
        k=k, v=v, q=q, dout=dout, lse=lse, dpsum=dpsum,
        dq=dq_accum, dk=dk, dv=dv, causal=causal,
        kv_group_size=kv_group_size,
    )
    config = FlashAttentionSm120BwdOp.kernel_config(ops)
    kernel = Megakernel(ops, config=config)

    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()

    torch.cuda.synchronize()
    dq = dq_accum.to(q.dtype)
    return dq, dk, dv


def _run_attention_coop_forward_bshd(q, k, v, causal=False, kv_group_size=1):
    """Run BSHD wrapper forward and return output tensor."""
    from machete.megakernel import Megakernel
    from machete.kernels.attention import FlashAttentionSm120Op

    o = torch.zeros_like(q)
    lse = torch.empty(q.shape[0], q.shape[2], q.shape[1], dtype=torch.float32, device=q.device)
    ops = FlashAttentionSm120Op.schedule(
        q=q, k=k, v=v, o=o, lse=lse, causal=causal, kv_group_size=kv_group_size,
    )
    config = FlashAttentionSm120Op.kernel_config(ops)
    kernel = Megakernel(ops, config=config)

    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()

    torch.cuda.synchronize()
    return o, lse


def _run_attention_coop_backward_bshd(q, k, v, o, dout, lse, causal=False,
                                      kv_group_size=1):
    """Run BSHD wrapper backward and return (dq, dk, dv)."""
    from machete.megakernel import Megakernel
    from machete.kernels.attention import FlashAttentionSm120BwdOp

    dpsum = (dout.float() * o.float()).sum(dim=-1).permute(0, 2, 1).contiguous()
    dq_accum = torch.zeros(q.shape[0], q.shape[1], q.shape[2], q.shape[3],
                           dtype=torch.float32, device=q.device)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    ops = FlashAttentionSm120BwdOp.schedule(
        k=k, v=v, q=q, dout=dout, lse=lse, dpsum=dpsum,
        dq=dq_accum, dk=dk, dv=dv, causal=causal,
        kv_group_size=kv_group_size,
    )
    config = FlashAttentionSm120BwdOp.kernel_config(ops)
    kernel = Megakernel(ops, config=config)

    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()

    torch.cuda.synchronize()
    dq = dq_accum.to(q.dtype)
    return dq, dk, dv


# =============================================================================
# Backward Tests (fp16 MMA)
# =============================================================================


class TestFlashAttentionCoopBwd:
    """Backward pass correctness tests (cooperative, SM120)."""

    @requires_gpu
    @pytest.mark.parametrize("BH,M,N,D,causal", BACKWARD_CASES)
    def test_backward_matrix(self, BH, M, N, D, causal):
        """Representative backward coverage."""
        torch.manual_seed(42)
        q = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        dout = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        _assert_attention_backward_close(q, k, v, dout, causal=causal)


class TestFlashAttentionCoopBSHD:
    """BSHD wrapper correctness tests."""

    @requires_gpu
    def test_bshd_forward_matches_reference(self):
        torch.manual_seed(42)
        B, S, H, D = 1, 32, 4, 64
        q = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")
        k = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")
        v = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")

        o_mk, _ = _run_attention_coop_forward_bshd(q, k, v, causal=True)
        o_ref = flash_attention_pytorch(
            q.permute(0, 2, 1, 3).reshape(B * H, S, D).float(),
            k.permute(0, 2, 1, 3).reshape(B * H, S, D).float(),
            v.permute(0, 2, 1, 3).reshape(B * H, S, D).float(),
            causal=True,
        ).reshape(B, H, S, D).permute(0, 2, 1, 3).half()

        torch.testing.assert_close(o_mk, o_ref, atol=5e-2, rtol=5e-2)

    @requires_gpu
    def test_bshd_backward_matches_reference(self):
        torch.manual_seed(42)
        B, S, H, D = 1, 32, 4, 64
        q = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")
        k = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")
        v = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")
        dout = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")

        o, lse = _run_attention_coop_forward_bshd(q, k, v, causal=True)
        dq, dk, dv = _run_attention_coop_backward_bshd(q, k, v, o, dout, lse, causal=True)

        dq_ref, dk_ref, dv_ref = flash_attention_backward_pytorch(
            q.permute(0, 2, 1, 3).reshape(B * H, S, D),
            k.permute(0, 2, 1, 3).reshape(B * H, S, D),
            v.permute(0, 2, 1, 3).reshape(B * H, S, D),
            o.permute(0, 2, 1, 3).reshape(B * H, S, D),
            dout.permute(0, 2, 1, 3).reshape(B * H, S, D),
            causal=True,
        )
        dq_ref = dq_ref.reshape(B, H, S, D).permute(0, 2, 1, 3)
        dk_ref = dk_ref.reshape(B, H, S, D).permute(0, 2, 1, 3)
        dv_ref = dv_ref.reshape(B, H, S, D).permute(0, 2, 1, 3)

        torch.testing.assert_close(dq, dq_ref.half(), atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(dk, dk_ref.half(), atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(dv, dv_ref.half(), atol=5e-2, rtol=5e-2)

# =============================================================================
# GQA Tests (Grouped Query Attention)
# =============================================================================


class TestFlashAttentionCoopGQA:
    """GQA tests: multiple Q heads share K/V heads (cooperative forward)."""

    @requires_gpu
    @pytest.mark.parametrize("BH_q,BH_kv,M,N,D,causal", GQA_FORWARD_CASES)
    def test_gqa_forward(self, BH_q, BH_kv, M, N, D, causal):
        """GQA forward matches PyTorch reference with repeat_interleave."""
        kv_group_size = BH_q // BH_kv
        torch.manual_seed(42)
        q = torch.randn(BH_q, M, D, dtype=torch.float16, device="cuda")
        k = torch.randn(BH_kv, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(BH_kv, N, D, dtype=torch.float16, device="cuda")
        _assert_attention_forward_close(
            q, k, v, causal=causal, kv_group_size=kv_group_size,
        )

    @requires_gpu
    def test_packgqa_forward(self):
        """PackGQA path matches PyTorch reference on 4D Qwen-style layout."""
        torch.manual_seed(321)
        B, S, H, H_kv, D = 1, 32, 4, 2, 64
        kv_group_size = H // H_kv
        q = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")
        k = torch.randn(B, S, H_kv, D, dtype=torch.float16, device="cuda")
        v = torch.randn(B, S, H_kv, D, dtype=torch.float16, device="cuda")

        o_mk = _run_attention_packgqa_forward(
            q, k, v, tile_m=32, causal=True, kv_group_size=kv_group_size,
        )
        q_ref = q.permute(0, 2, 1, 3).reshape(B * H, S, D)
        k_ref = k.permute(0, 2, 1, 3).reshape(B * H_kv, S, D)
        v_ref = v.permute(0, 2, 1, 3).reshape(B * H_kv, S, D)
        o_ref = flash_attention_pytorch(
            q_ref.float(), k_ref.float(), v_ref.float(),
            causal=True, kv_group_size=kv_group_size,
        ).half().reshape(B, H, S, D).permute(0, 2, 1, 3)

        torch.testing.assert_close(o_mk, o_ref, atol=5e-2, rtol=5e-2)

    @requires_gpu
    def test_direct_forward(self):
        """Compute-owned direct path matches PyTorch reference."""
        torch.manual_seed(321)
        B, S, H, H_kv, D = 1, 32, 4, 2, 64
        kv_group_size = H // H_kv
        q = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")
        k = torch.randn(B, S, H_kv, D, dtype=torch.float16, device="cuda")
        v = torch.randn(B, S, H_kv, D, dtype=torch.float16, device="cuda")

        o_mk = _run_attention_direct_forward(
            q, k, v, tile_m=32, causal=True, kv_group_size=kv_group_size,
        )
        q_ref = q.permute(0, 2, 1, 3).reshape(B * H, S, D)
        k_ref = k.permute(0, 2, 1, 3).reshape(B * H_kv, S, D)
        v_ref = v.permute(0, 2, 1, 3).reshape(B * H_kv, S, D)
        o_ref = flash_attention_pytorch(
            q_ref.float(), k_ref.float(), v_ref.float(),
            causal=True, kv_group_size=kv_group_size,
        ).half().reshape(B, H, S, D).permute(0, 2, 1, 3)

        torch.testing.assert_close(o_mk, o_ref, atol=5e-2, rtol=5e-2)


class TestFlashAttentionCoopGQABwd:
    """GQA backward tests: dK/dV accumulate across Q-head groups."""

    @requires_gpu
    @pytest.mark.parametrize("BH_q,BH_kv,M,N,D,causal", GQA_BACKWARD_CASES)
    def test_gqa_backward(self, BH_q, BH_kv, M, N, D, causal):
        """GQA backward matches PyTorch reference."""
        kv_group_size = BH_q // BH_kv
        torch.manual_seed(42)
        q = torch.randn(BH_q, M, D, dtype=torch.float16, device="cuda")
        k = torch.randn(BH_kv, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(BH_kv, N, D, dtype=torch.float16, device="cuda")
        dout = torch.randn(BH_q, M, D, dtype=torch.float16, device="cuda")
        _assert_attention_backward_close(
            q, k, v, dout, causal=causal, kv_group_size=kv_group_size,
        )
