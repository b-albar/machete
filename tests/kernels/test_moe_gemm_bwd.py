# Copyright (c) 2025, Machete Authors
"""Tests for MoeGemmBwdOp — backward dx correctness for MoE grouped GEMM.

Tests run on GPU (SM_90+) and compare MoeGemmBwdOp against the PyTorch
reference implementation (moe_gemm_bwd_dx_ref).
"""

import contextlib
import io
import importlib.util

import pytest
import torch

if importlib.util.find_spec("cutlass") is None:
    pytest.skip("Requires CUTLASS", allow_module_level=True)
from tests.kernels.support import requires_sm90_cutlass


requires_gpu = requires_sm90_cutlass

BWD_CASES = [
    (32, 4, 2, 64, 32, 32, 32, 32),
    (128, 8, 2, 128, 64, 64, 32, 32),
    (64, 4, 2, 64, 128, 32, 32, 32),
    (256, 16, 2, 128, 256, 64, 32, 32),
]


# =============================================================================
# Helpers
# =============================================================================


def _run_moe_gemm_bwd(dc, w, expert_ids, dx,
                       tile_m=64, tile_k=32, tile_n=32):
    """Run MoeGemmBwdOp and return output tensor dx."""
    from machete.megakernel import Megakernel
    from machete.kernels.moe import MoeGemmBwdOp

    ops = MoeGemmBwdOp.schedule_backward(
        dc=dc, w=w, expert_ids=expert_ids, dx=dx,
        tile_sizes={"M": tile_m, "K": tile_k, "N": tile_n},
    )
    config = MoeGemmBwdOp.kernel_config(ops)
    kernel = Megakernel(ops, config=config)

    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()

    return dx


def _moe_bwd_dx_case(num_tokens, num_experts, topk, K, N, dtype,
                      tile_m=64, tile_k=32, tile_n=32,
                      atol=1e-1, rtol=1e-2):
    """Run a single MoE backward dx test case and assert correctness."""
    from machete.kernels.moe.align_sort import moe_align_sort
    from machete.kernels.moe.ref import moe_gemm_bwd_dx_ref

    torch.manual_seed(42)
    device = "cuda"

    # Create tensors
    x = torch.randn(num_tokens, K, dtype=dtype, device=device)
    w = torch.randn(num_experts, N, K, dtype=dtype, device=device)
    dc_full = torch.randn(num_tokens, N, dtype=dtype, device=device) * 0.1

    # Route tokens
    topk_ids = torch.randint(0, num_experts, (num_tokens, topk),
                             dtype=torch.int32, device=device)
    topk_weights = torch.randn(num_tokens, topk, dtype=torch.float32,
                               device=device).softmax(dim=-1)

    # Align and sort
    sorted_token_ids, expert_ids, sorted_weights, _ = (
        moe_align_sort(topk_ids, topk_weights, num_experts, block_size=tile_m)
    )
    total_padded = sorted_token_ids.shape[0]

    # Gather sorted dc (padding positions get clamped)
    clamped_ids = sorted_token_ids.clamp(max=num_tokens - 1).long()
    sorted_dc = dc_full[clamped_ids]

    # Reference
    ref = moe_gemm_bwd_dx_ref(sorted_dc, w, expert_ids, sorted_token_ids)

    # Run MoeGemmBwdOp
    dx = torch.zeros(total_padded, K, dtype=dtype, device=device)
    _run_moe_gemm_bwd(sorted_dc, w, expert_ids, dx,
                       tile_m=tile_m, tile_k=tile_k, tile_n=tile_n)

    # Only compare non-padding positions
    valid_mask = sorted_token_ids < num_tokens
    if valid_mask.any():
        torch.testing.assert_close(
            dx[valid_mask], ref[valid_mask], atol=atol, rtol=rtol)


# =============================================================================
# Tests: MoeGemmBwdOp (dx computation)
# =============================================================================


@requires_gpu
class TestMoeGemmBwd:
    """MoE grouped GEMM backward dx correctness tests."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "num_tokens,num_experts,topk,K,N,tile_m,tile_k,tile_n", BWD_CASES)
    def test_dx_matrix(
        self, dtype, num_tokens, num_experts, topk, K, N, tile_m, tile_k, tile_n
    ):
        """Representative MoE GEMM backward-dx matrix."""
        _moe_bwd_dx_case(
            num_tokens, num_experts, topk, K, N, dtype,
            tile_m=tile_m, tile_k=tile_k, tile_n=tile_n,
        )

    # ----- Topk variations -----

    @pytest.mark.parametrize("topk", [1, 2, 4])
    def test_topk_values(self, topk):
        """Different topk values."""
        _moe_bwd_dx_case(64, 8, topk, 64, 32, torch.float16,
                          tile_m=32, tile_k=32, tile_n=32)

    # ----- Edge cases -----

    def test_single_expert(self):
        """All tokens routed to one expert."""
        torch.manual_seed(42)
        from machete.kernels.moe.align_sort import moe_align_sort
        from machete.kernels.moe.ref import moe_gemm_bwd_dx_ref

        device = "cuda"
        dtype = torch.float16
        num_tokens, K, N, num_experts = 32, 64, 32, 4
        tile_m = 32

        dc = torch.randn(num_tokens, N, dtype=dtype, device=device) * 0.1
        w = torch.randn(num_experts, N, K, dtype=dtype, device=device)

        topk_ids = torch.zeros(num_tokens, 1, dtype=torch.int32, device=device)
        topk_weights = torch.ones(num_tokens, 1, dtype=torch.float32,
                                  device=device)

        sorted_token_ids, expert_ids, _, _ = (
            moe_align_sort(topk_ids, topk_weights, num_experts, tile_m)
        )
        total_padded = sorted_token_ids.shape[0]
        clamped = sorted_token_ids.clamp(max=num_tokens - 1).long()
        sorted_dc = dc[clamped]

        ref = moe_gemm_bwd_dx_ref(sorted_dc, w, expert_ids, sorted_token_ids)
        dx = torch.zeros(total_padded, K, dtype=dtype, device=device)
        _run_moe_gemm_bwd(sorted_dc, w, expert_ids, dx,
                           tile_m=tile_m, tile_k=32, tile_n=32)

        valid = sorted_token_ids < num_tokens
        torch.testing.assert_close(dx[valid], ref[valid], atol=1e-1, rtol=1e-2)

# =============================================================================
# Tests: Reference backward implementations
# =============================================================================


class TestMoeBwdRef:
    """Tests for backward reference implementations."""

    def test_dx_ref_identity(self):
        """dx = dc @ w should recover dc when w = I."""
        K, N = 32, 32
        dtype = torch.float32
        dc = torch.randn(4, N, dtype=dtype)
        w = torch.eye(N, K, dtype=dtype).unsqueeze(0)  # [1, N, K]
        expert_ids = torch.zeros(4, dtype=torch.int32)
        sorted_token_ids = torch.arange(4, dtype=torch.int32)

        from machete.kernels.moe.ref import moe_gemm_bwd_dx_ref
        dx = moe_gemm_bwd_dx_ref(dc, w, expert_ids, sorted_token_ids)
        torch.testing.assert_close(dx, dc, atol=1e-5, rtol=1e-5)

    def test_dw_ref_outer_product(self):
        """dw for 1 token, 1 expert = outer product dc^T @ x."""
        K, N = 4, 3
        dtype = torch.float32
        dc = torch.randn(1, N, dtype=dtype)
        x = torch.randn(1, K, dtype=dtype)
        expert_ids = torch.zeros(1, dtype=torch.int32)

        from machete.kernels.moe.ref import moe_gemm_bwd_dw_ref
        dw = moe_gemm_bwd_dw_ref(dc, x, expert_ids, num_experts=1)
        expected = dc.t() @ x  # [N, K]
        torch.testing.assert_close(dw[0], expected, atol=1e-5, rtol=1e-5)
