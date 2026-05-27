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


def _bmhd_to_bhmd(x):
    return x.transpose(1, 2).reshape(x.shape[0] * x.shape[2], x.shape[1], x.shape[3]).contiguous()


def _attention_ref_bmhd(q, k, v, *, causal=False, kv_group_size=1):
    B, M, H, D = q.shape
    H_kv = k.shape[2]
    q_ref = q.transpose(1, 2).reshape(B * H, M, D)
    k_ref = k.transpose(1, 2).reshape(B * H_kv, k.shape[1], D)
    v_ref = v.transpose(1, 2).reshape(B * H_kv, v.shape[1], D)
    return flash_attention_pytorch(
        q_ref.float(), k_ref.float(), v_ref.float(),
        causal=causal, kv_group_size=kv_group_size,
    ).half().reshape(B, H, M, D).transpose(1, 2).contiguous()


def _attention_lse_pytorch(q, k, *, causal=False, kv_group_size=1):
    if kv_group_size > 1:
        k = k.repeat_interleave(kv_group_size, dim=0)
    scale = q.shape[-1] ** -0.5
    scores = torch.bmm(q.float(), k.float().transpose(-2, -1)) * scale
    if causal:
        M, N = q.shape[1], k.shape[1]
        row_idx = torch.arange(M, device=q.device).unsqueeze(1)
        col_idx = torch.arange(N, device=q.device).unsqueeze(0)
        scores.masked_fill_(col_idx.unsqueeze(0) > row_idx.unsqueeze(0) + (N - M), float("-inf"))
    return torch.logsumexp(scores, dim=-1).contiguous()


def _attention_lse_bmhd(q, k, *, causal=False, kv_group_size=1):
    B, M, H, D = q.shape
    H_kv = k.shape[2]
    lse = _attention_lse_pytorch(
        q.transpose(1, 2).reshape(B * H, M, D),
        k.transpose(1, 2).reshape(B * H_kv, k.shape[1], D),
        causal=causal,
        kv_group_size=kv_group_size,
    )
    out = torch.empty(B, M, H, dtype=lse.dtype, device=lse.device)
    out.copy_(lse.reshape(B, H, M).transpose(1, 2))
    return out


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
        dout.shape[0], dout.shape[1], dout.shape[2],
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
    o_ref = _attention_ref_bmhd(q, k, v, causal=causal, kv_group_size=kv_group_size)
    torch.testing.assert_close(o_mk, o_ref, atol=5e-2, rtol=5e-2)


def _assert_attention_backward_close(q, k, v, dout, *, causal=False, kv_group_size=1):
    o = _attention_ref_bmhd(q, k, v, causal=causal, kv_group_size=kv_group_size)
    lse = _attention_lse_bmhd(q, k, causal=causal, kv_group_size=kv_group_size)
    dq, dk, dv = _run_attention_coop_backward(
        q, k, v, o, dout, lse, causal=causal, kv_group_size=kv_group_size,
    )
    q_ref = _bmhd_to_bhmd(q)
    k_ref = _bmhd_to_bhmd(k)
    v_ref = _bmhd_to_bhmd(v)
    o_ref = _bmhd_to_bhmd(o)
    dout_ref = _bmhd_to_bhmd(dout)
    dq_ref, dk_ref, dv_ref = flash_attention_backward_pytorch(
        q_ref, k_ref, v_ref, o_ref, dout_ref, causal=causal, kv_group_size=kv_group_size,
    )
    B, M, H, D = q.shape
    H_kv = k.shape[2]
    dq_ref = dq_ref.reshape(B, H, M, D).transpose(1, 2).contiguous()
    dk_ref = dk_ref.reshape(B, H_kv, k.shape[1], D).transpose(1, 2).contiguous()
    dv_ref = dv_ref.reshape(B, H_kv, v.shape[1], D).transpose(1, 2).contiguous()
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

    def test_bmhd_schedule_keeps_m_tile_and_tuning_static_dims(self):
        """Native BMHD scheduling keeps M-major tiling and tuning dims."""
        from machete.kernels.attention import FlashAttentionSm120Op

        q = torch.empty(1, 32, 2, 64, dtype=torch.float16, device="cuda")
        k = torch.empty(1, 32, 2, 64, dtype=torch.float16, device="cuda")
        v = torch.empty(1, 32, 2, 64, dtype=torch.float16, device="cuda")
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
        assert ops[0].dim_names == {"B": 0, "M": 1, "H": 2, "D": 3}
        assert ops[0].static_dims["H"] == 2
        assert ops[0].static_dims["num_mma_warps"] == 1
        assert ops[0].static_dims["write_lse"] == 0

    @requires_gpu
    @pytest.mark.parametrize("H,M,N,D", MMA_FORWARD_CASES)
    def test_mma_forward_matrix(self, H, M, N, D):
        """Representative exact-tile MMA forward coverage."""
        torch.manual_seed(42)
        q = torch.randn(1, M, H, D, dtype=torch.float16, device="cuda")
        k = torch.randn(1, N, H, D, dtype=torch.float16, device="cuda")
        v = torch.randn(1, N, H, D, dtype=torch.float16, device="cuda")
        _assert_attention_forward_close(q, k, v, tile_m=16)

    @requires_gpu
    def test_dpsum_matches_pytorch(self):
        torch.manual_seed(42)
        B, S, H, D = 2, 64, 8, 128
        dout = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
        o = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")

        mk = _run_attention_dpsum(dout, o)
        ref = (dout.float() * o.float()).sum(dim=-1).contiguous()

        torch.testing.assert_close(mk, ref, atol=1e-4, rtol=1e-4)

    @requires_gpu
    def test_mma_uniform_attention(self):
        """Uniform keys -> each output row equals mean of V."""
        B, M, N, H, D = 1, 16, 16, 1, 64
        q = torch.randn(B, M, H, D, dtype=torch.float16, device="cuda")
        k_row = torch.randn(B, 1, H, D, dtype=torch.float16, device="cuda")
        k = k_row.expand(B, N, H, D).contiguous()
        v = torch.randn(B, N, H, D, dtype=torch.float16, device="cuda")

        o_mk = _run_attention_coop_forward(q, k, v, tile_m=16)
        expected = v.float().mean(dim=1, keepdim=True).expand_as(q).half()

        torch.testing.assert_close(o_mk, expected, atol=5e-2, rtol=5e-2)

# =============================================================================
# Multi-warp MMA Tests (auto tile_M)
# =============================================================================


class TestFlashAttentionCoopMultiWarp:
    """Multi-warp MMA tests — let schedule pick optimal tile_M."""

    @requires_gpu
    @pytest.mark.parametrize("H,M,N,D", MULTIWARP_FORWARD_CASES)
    def test_multi_warp_forward_matrix(self, H, M, N, D):
        """Representative auto-tiled multi-warp forward coverage."""
        torch.manual_seed(42)
        q = torch.randn(1, M, H, D, dtype=torch.float16, device="cuda")
        k = torch.randn(1, N, H, D, dtype=torch.float16, device="cuda")
        v = torch.randn(1, N, H, D, dtype=torch.float16, device="cuda")
        _assert_attention_forward_close(q, k, v)

    @requires_gpu
    def test_sequence_dynamic_reuses_compiled_kernel(self):
        """Different N should reuse the same compiled FlashAttention kernel."""
        torch.manual_seed(0)
        B, H, M, D = 1, 1, 16, 256

        q1 = torch.randn(B, M, H, D, dtype=torch.float16, device="cuda")
        k1 = torch.randn(B, 144, H, D, dtype=torch.float16, device="cuda")
        v1 = torch.randn(B, 144, H, D, dtype=torch.float16, device="cuda")
        kfa1, o1 = _build_attention_coop_kernel(q1, k1, v1, tile_m=16)

        q2 = torch.randn(B, M, H, D, dtype=torch.float16, device="cuda")
        k2 = torch.randn(B, 272, H, D, dtype=torch.float16, device="cuda")
        v2 = torch.randn(B, 272, H, D, dtype=torch.float16, device="cuda")
        kfa2, o2 = _build_attention_coop_kernel(q2, k2, v2, tile_m=16)

        assert kfa1._make_cache_key() == kfa2._make_cache_key()
        assert kfa1._compiled_kernel is kfa2._compiled_kernel
        torch.testing.assert_close(o1, _attention_ref_bmhd(q1, k1, v1), atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(o2, _attention_ref_bmhd(q2, k2, v2), atol=5e-2, rtol=5e-2)


# =============================================================================
# Causal Tests (fp16 MMA)
# =============================================================================


class TestFlashAttentionCoopCausal:
    """Causal masking tests (fp16, cooperative)."""

    @requires_gpu
    @pytest.mark.parametrize("H,M,N,D", CAUSAL_FORWARD_CASES)
    def test_causal_forward_matrix(self, H, M, N, D):
        """Representative causal forward coverage."""
        torch.manual_seed(42)
        q = torch.randn(1, M, H, D, dtype=torch.float16, device="cuda")
        k = torch.randn(1, N, H, D, dtype=torch.float16, device="cuda")
        v = torch.randn(1, N, H, D, dtype=torch.float16, device="cuda")
        _assert_attention_forward_close(q, k, v, causal=True)


# =============================================================================
# Backward Helpers
# =============================================================================


def _run_attention_coop_backward(q, k, v, o, dout, lse, causal=False,
                                 kv_group_size=1):
    """Run FlashAttentionSm120BwdOp and return (dq, dk, dv)."""
    from machete.megakernel import Megakernel
    from machete.kernels.attention import FlashAttentionSm120BwdOp

    dpsum = torch.empty(q.shape[0], q.shape[1], q.shape[2], dtype=torch.float32, device=q.device)
    dpsum.copy_((dout.float() * o.float()).sum(dim=-1))
    dq_accum = torch.zeros_like(q, dtype=torch.float32)
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
        q = torch.randn(1, M, BH, D, dtype=torch.float16, device="cuda")
        k = torch.randn(1, N, BH, D, dtype=torch.float16, device="cuda")
        v = torch.randn(1, N, BH, D, dtype=torch.float16, device="cuda")
        dout = torch.randn(1, M, BH, D, dtype=torch.float16, device="cuda")
        _assert_attention_backward_close(q, k, v, dout, causal=causal)


class TestFlashAttentionCoopBMHD:
    """Native BMHD forward wrapper correctness tests."""

    @requires_gpu
    def test_bmhd_forward_matches_reference(self):
        torch.manual_seed(42)
        B, S, H, D = 1, 32, 4, 64
        q = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")
        k = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")
        v = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")

        o_mk = _run_attention_coop_forward(q, k, v, causal=True)
        o_ref = _attention_ref_bmhd(q, k, v, causal=True)

        torch.testing.assert_close(o_mk, o_ref, atol=5e-2, rtol=5e-2)

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
        q = torch.randn(1, M, BH_q, D, dtype=torch.float16, device="cuda")
        k = torch.randn(1, N, BH_kv, D, dtype=torch.float16, device="cuda")
        v = torch.randn(1, N, BH_kv, D, dtype=torch.float16, device="cuda")
        _assert_attention_forward_close(
            q, k, v, causal=causal, kv_group_size=kv_group_size,
        )

class TestFlashAttentionCoopGQABwd:
    """GQA backward tests: dK/dV accumulate across Q-head groups."""

    @requires_gpu
    @pytest.mark.parametrize("BH_q,BH_kv,M,N,D,causal", GQA_BACKWARD_CASES)
    def test_gqa_backward(self, BH_q, BH_kv, M, N, D, causal):
        """GQA backward matches PyTorch reference."""
        kv_group_size = BH_q // BH_kv
        torch.manual_seed(42)
        q = torch.randn(1, M, BH_q, D, dtype=torch.float16, device="cuda")
        k = torch.randn(1, N, BH_kv, D, dtype=torch.float16, device="cuda")
        v = torch.randn(1, N, BH_kv, D, dtype=torch.float16, device="cuda")
        dout = torch.randn(1, M, BH_q, D, dtype=torch.float16, device="cuda")
        _assert_attention_backward_close(
            q, k, v, dout, causal=causal, kv_group_size=kv_group_size,
        )
