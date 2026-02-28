# Copyright (c) 2025, Machete Authors
"""Tests for FlashAttentionSm120Op — cooperative forward correctness (fp16/bf16).

Tests run on GPU (Hopper+) and compare the megakernel FlashAttentionSm120Op
against a pure PyTorch reference implementation.
"""

import contextlib
import io

import pytest
import torch

from machete.kernels.attention.ref import flash_attention_pytorch, flash_attention_backward_pytorch


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


requires_gpu = pytest.mark.skipif(
    not (is_hopper_or_newer() and CUTLASS_AVAILABLE),
    reason="Requires Hopper+ GPU with CUTLASS",
)


# =============================================================================
# Helpers
# =============================================================================


def _run_attention_coop_forward(q, k, v, tile_m=None, causal=False):
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
    )
    config = FlashAttentionSm120Op.kernel_config(ops)
    kernel = Megakernel(ops, config=config)

    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()

    torch.cuda.synchronize()
    return o


# =============================================================================
# fp16 MMA Tests (tensor core path)
# =============================================================================


class TestFlashAttentionCoopMMA:
    """fp16 tensor core MMA path correctness tests (cooperative)."""

    @requires_gpu
    @pytest.mark.parametrize("BH,M,N,D", [
        (1, 16, 16, 64),     # Exact tile, D=64
        (1, 16, 16, 128),    # Exact tile, D=128
        (1, 16, 32, 64),     # Multi KV-block (if n_block < 32)
        (1, 16, 64, 128),    # N == n_block for D=128
    ])
    def test_mma_basic_shapes(self, BH, M, N, D):
        """MMA attention for various shapes (exact tile_M=16)."""
        torch.manual_seed(42)
        q = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")

        o_mk = _run_attention_coop_forward(q, k, v, tile_m=16)
        o_ref = flash_attention_pytorch(
            q.float(), k.float(), v.float()).half()

        torch.testing.assert_close(o_mk, o_ref, atol=5e-2, rtol=5e-2)

    @requires_gpu
    def test_mma_non_divisible_n(self):
        """N not a multiple of n_block."""
        torch.manual_seed(42)
        BH, M, N, D = 1, 16, 20, 64
        q = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")

        o_mk = _run_attention_coop_forward(q, k, v, tile_m=16)
        o_ref = flash_attention_pytorch(
            q.float(), k.float(), v.float()).half()

        torch.testing.assert_close(o_mk, o_ref, atol=5e-2, rtol=5e-2)

    @requires_gpu
    def test_mma_multi_head(self):
        """Multiple attention heads."""
        torch.manual_seed(42)
        BH, M, N, D = 4, 16, 32, 128
        q = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")

        o_mk = _run_attention_coop_forward(q, k, v, tile_m=16)
        o_ref = flash_attention_pytorch(
            q.float(), k.float(), v.float()).half()

        torch.testing.assert_close(o_mk, o_ref, atol=5e-2, rtol=5e-2)

    @requires_gpu
    def test_mma_multi_m_tile(self):
        """Multiple M tiles: M=32, tile_m=16."""
        torch.manual_seed(42)
        BH, M, N, D = 1, 32, 32, 64
        q = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")

        o_mk = _run_attention_coop_forward(q, k, v, tile_m=16)
        o_ref = flash_attention_pytorch(
            q.float(), k.float(), v.float()).half()

        torch.testing.assert_close(o_mk, o_ref, atol=5e-2, rtol=5e-2)

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

    @requires_gpu
    def test_mma_large_n(self):
        """Large N requiring multiple KV blocks (N > n_block)."""
        torch.manual_seed(42)
        BH, M, N, D = 1, 16, 256, 128
        q = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")

        o_mk = _run_attention_coop_forward(q, k, v, tile_m=16)
        o_ref = flash_attention_pytorch(
            q.float(), k.float(), v.float()).half()

        torch.testing.assert_close(o_mk, o_ref, atol=5e-2, rtol=5e-2)


# =============================================================================
# Multi-warp MMA Tests (auto tile_M)
# =============================================================================


class TestFlashAttentionCoopMultiWarp:
    """Multi-warp MMA tests — let schedule_forward pick optimal tile_M."""

    @requires_gpu
    @pytest.mark.parametrize("BH,M,N,D", [
        (1, 64, 64, 64),      # D=64: tile_M=64 (4 warps)
        (1, 64, 64, 128),     # D=128: tile_M=64 (4 warps)
        (1, 128, 128, 128),   # D=128: 2 M tiles
        (4, 64, 64, 128),     # Multi-head, D=128
    ])
    def test_multi_warp_shapes(self, BH, M, N, D):
        """Multi-warp MMA with auto tile_M for various shapes."""
        torch.manual_seed(42)
        q = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")

        o_mk = _run_attention_coop_forward(q, k, v)
        o_ref = flash_attention_pytorch(
            q.float(), k.float(), v.float()).half()

        torch.testing.assert_close(o_mk, o_ref, atol=5e-2, rtol=5e-2)

    @requires_gpu
    def test_multi_warp_non_divisible_n(self):
        """Multi-warp with N not divisible by n_block."""
        torch.manual_seed(42)
        BH, M, N, D = 1, 64, 50, 128
        q = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")

        o_mk = _run_attention_coop_forward(q, k, v)
        o_ref = flash_attention_pytorch(
            q.float(), k.float(), v.float()).half()

        torch.testing.assert_close(o_mk, o_ref, atol=5e-2, rtol=5e-2)

    @requires_gpu
    def test_multi_warp_non_divisible_m(self):
        """Multi-warp with M not divisible by tile_M."""
        torch.manual_seed(42)
        BH, M, N, D = 1, 80, 64, 128  # tile_M=64, so M=80 not divisible
        q = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")

        o_mk = _run_attention_coop_forward(q, k, v)
        o_ref = flash_attention_pytorch(
            q.float(), k.float(), v.float()).half()

        torch.testing.assert_close(o_mk, o_ref, atol=5e-2, rtol=5e-2)

    @requires_gpu
    def test_multi_warp_decode(self):
        """Decode with multi-warp: M=16 (single warp min), N large."""
        torch.manual_seed(42)
        BH, M, N, D = 4, 16, 128, 128
        q = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")

        o_mk = _run_attention_coop_forward(q, k, v)
        o_ref = flash_attention_pytorch(
            q.float(), k.float(), v.float()).half()

        torch.testing.assert_close(o_mk, o_ref, atol=5e-2, rtol=5e-2)

    @requires_gpu
    def test_multi_warp_large_n(self):
        """Large N exercising multiple KV blocks."""
        torch.manual_seed(42)
        BH, M, N, D = 1, 64, 512, 128  # 512/32 = 16 KV blocks
        q = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")

        o_mk = _run_attention_coop_forward(q, k, v)
        o_ref = flash_attention_pytorch(
            q.float(), k.float(), v.float()).half()

        torch.testing.assert_close(o_mk, o_ref, atol=5e-2, rtol=5e-2)


# =============================================================================
# Causal Tests (fp16 MMA)
# =============================================================================


class TestFlashAttentionCoopCausal:
    """Causal masking tests (fp16, cooperative)."""

    @requires_gpu
    def test_causal_square(self):
        """Square causal: M=N=32, lower-triangular attention."""
        torch.manual_seed(42)
        BH, M, N, D = 1, 32, 32, 64
        q = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")

        o_mk = _run_attention_coop_forward(q, k, v, causal=True)
        o_ref = flash_attention_pytorch(
            q.float(), k.float(), v.float(), causal=True).half()

        torch.testing.assert_close(o_mk, o_ref, atol=5e-2, rtol=5e-2)

    @requires_gpu
    def test_causal_asymmetric(self):
        """Asymmetric causal: M=16, N=64 (chunked prefill)."""
        torch.manual_seed(42)
        BH, M, N, D = 1, 16, 64, 64
        q = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")

        o_mk = _run_attention_coop_forward(q, k, v, causal=True)
        o_ref = flash_attention_pytorch(
            q.float(), k.float(), v.float(), causal=True).half()

        torch.testing.assert_close(o_mk, o_ref, atol=5e-2, rtol=5e-2)

    @requires_gpu
    def test_causal_multi_head(self):
        """Multi-head causal: BH=4."""
        torch.manual_seed(42)
        BH, M, N, D = 4, 32, 32, 64
        q = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")

        o_mk = _run_attention_coop_forward(q, k, v, causal=True)
        o_ref = flash_attention_pytorch(
            q.float(), k.float(), v.float(), causal=True).half()

        torch.testing.assert_close(o_mk, o_ref, atol=5e-2, rtol=5e-2)

    @requires_gpu
    def test_causal_d128(self):
        """Causal with D=128."""
        torch.manual_seed(42)
        BH, M, N, D = 1, 64, 64, 128
        q = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")

        o_mk = _run_attention_coop_forward(q, k, v, causal=True)
        o_ref = flash_attention_pytorch(
            q.float(), k.float(), v.float(), causal=True).half()

        torch.testing.assert_close(o_mk, o_ref, atol=5e-2, rtol=5e-2)


# =============================================================================
# Backward Helpers
# =============================================================================


def _run_attention_coop_forward_with_lse(q, k, v, causal=False):
    """Run forward and return (o, lse) tensors."""
    from machete.megakernel import Megakernel
    from machete.kernels.attention import FlashAttentionSm120Op

    o = torch.zeros_like(q)
    lse = torch.empty(q.shape[0], q.shape[1], dtype=torch.float32, device=q.device)
    ops = FlashAttentionSm120Op.schedule_forward(
        q=q, k=k, v=v, o=o, lse=lse, causal=causal,
    )
    config = FlashAttentionSm120Op.kernel_config(ops)
    kernel = Megakernel(ops, config=config)

    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()

    torch.cuda.synchronize()
    return o, lse


def _run_attention_coop_backward(q, k, v, o, dout, lse, causal=False):
    """Run FlashAttentionSm120BwdOp and return (dq, dk, dv)."""
    from machete.megakernel import Megakernel
    from machete.kernels.attention import FlashAttentionSm120BwdOp

    dpsum = (dout.float() * o.float()).sum(dim=-1).contiguous()
    dq_accum = torch.zeros(q.shape[0], q.shape[1], q.shape[2],
                           dtype=torch.float32, device=q.device)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    ops = FlashAttentionSm120BwdOp.schedule_backward(
        k=k, v=v, q=q, dout=dout, lse=lse, dpsum=dpsum,
        dq_accum=dq_accum, dk=dk, dv=dv, causal=causal,
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
    @pytest.mark.parametrize("BH,M,N,D", [
        (1, 32, 32, 64),
        (1, 32, 32, 128),
    ])
    def test_bwd_basic(self, BH, M, N, D):
        """Basic backward: square shapes."""
        torch.manual_seed(42)
        q = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        dout = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")

        o, lse = _run_attention_coop_forward_with_lse(q, k, v)
        dq, dk, dv = _run_attention_coop_backward(q, k, v, o, dout, lse)

        dq_ref, dk_ref, dv_ref = flash_attention_backward_pytorch(
            q, k, v, o, dout)

        torch.testing.assert_close(dv, dv_ref.half(), atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(dk, dk_ref.half(), atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(dq, dq_ref.half(), atol=5e-2, rtol=5e-2)

    @requires_gpu
    def test_bwd_multi_head(self):
        """Multi-head backward."""
        torch.manual_seed(42)
        BH, M, N, D = 4, 32, 32, 128
        q = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        dout = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")

        o, lse = _run_attention_coop_forward_with_lse(q, k, v)
        dq, dk, dv = _run_attention_coop_backward(q, k, v, o, dout, lse)

        dq_ref, dk_ref, dv_ref = flash_attention_backward_pytorch(
            q, k, v, o, dout)

        torch.testing.assert_close(dv, dv_ref.half(), atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(dk, dk_ref.half(), atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(dq, dq_ref.half(), atol=5e-2, rtol=5e-2)

    @requires_gpu
    def test_bwd_multi_m_blocks(self):
        """Multiple M-blocks in backward (M > tile_N)."""
        torch.manual_seed(42)
        BH, M, N, D = 1, 64, 32, 128
        q = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        dout = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")

        o, lse = _run_attention_coop_forward_with_lse(q, k, v)
        dq, dk, dv = _run_attention_coop_backward(q, k, v, o, dout, lse)

        dq_ref, dk_ref, dv_ref = flash_attention_backward_pytorch(
            q, k, v, o, dout)

        torch.testing.assert_close(dv, dv_ref.half(), atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(dk, dk_ref.half(), atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(dq, dq_ref.half(), atol=5e-2, rtol=5e-2)

    @requires_gpu
    def test_bwd_multi_n_tiles(self):
        """Multiple N-tiles in backward (N > tile_N)."""
        torch.manual_seed(42)
        BH, M, N, D = 1, 32, 64, 128
        q = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        dout = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")

        o, lse = _run_attention_coop_forward_with_lse(q, k, v)
        dq, dk, dv = _run_attention_coop_backward(q, k, v, o, dout, lse)

        dq_ref, dk_ref, dv_ref = flash_attention_backward_pytorch(
            q, k, v, o, dout)

        torch.testing.assert_close(dv, dv_ref.half(), atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(dk, dk_ref.half(), atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(dq, dq_ref.half(), atol=5e-2, rtol=5e-2)

    @requires_gpu
    def test_bwd_causal(self):
        """Causal backward."""
        torch.manual_seed(42)
        BH, M, N, D = 1, 32, 32, 128
        q = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        dout = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")

        o, lse = _run_attention_coop_forward_with_lse(q, k, v, causal=True)
        dq, dk, dv = _run_attention_coop_backward(
            q, k, v, o, dout, lse, causal=True)

        dq_ref, dk_ref, dv_ref = flash_attention_backward_pytorch(
            q, k, v, o, dout, causal=True)

        torch.testing.assert_close(dv, dv_ref.half(), atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(dk, dk_ref.half(), atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(dq, dq_ref.half(), atol=5e-2, rtol=5e-2)
