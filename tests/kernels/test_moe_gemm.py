# Copyright (c) 2025, Machete Authors
"""Tests for MoeGemmOp — grouped GEMM correctness for MoE.

Tests run on GPU (SM_90+) and compare MoeGemmOp against the PyTorch
reference implementation (moe_gemm_ref).
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

FORWARD_CASES = [
    (32, 4, 2, 64, 32, 32, 32, 32),
    (128, 8, 2, 128, 64, 64, 32, 32),
    (64, 4, 2, 128, 64, 64, 32, 32),
    (256, 16, 2, 256, 128, 64, 32, 32),
]


# =============================================================================
# Helpers
# =============================================================================


def _run_moe_align_sort(num_tokens, num_experts, topk, block_size, device="cuda"):
    """Create random routing and run moe_align_sort."""
    from machete.kernels.moe.align_sort import moe_align_sort

    topk_ids = torch.randint(0, num_experts, (num_tokens, topk),
                             dtype=torch.int32, device=device)
    topk_weights = torch.randn(num_tokens, topk, dtype=torch.float32,
                               device=device).softmax(dim=-1)

    sorted_token_ids, expert_ids, sorted_weights, num_tokens_per_expert = (
        moe_align_sort(topk_ids, topk_weights, num_experts, block_size)
    )
    return (topk_ids, topk_weights, sorted_token_ids, expert_ids,
            sorted_weights, num_tokens_per_expert)


def _run_moe_gemm(sorted_x, w, expert_ids, c,
                   tile_m=64, tile_n=32, tile_k=32):
    """Run MoeGemmOp and return output tensor c."""
    from machete.megakernel import Megakernel
    from machete.kernels.moe import MoeGemmOp

    ops = MoeGemmOp.schedule(
        sorted_x=sorted_x, w=w, expert_ids=expert_ids, c=c,
        tile_sizes={"M": tile_m, "N": tile_n, "K": tile_k},
    )
    config = MoeGemmOp.kernel_config(ops)
    kernel = Megakernel(ops, config=config)

    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()

    return c


# =============================================================================
# Tests: moe_align_sort
# =============================================================================


class TestMoeAlignSort:
    """Tests for moe_align_sort host-side preprocessing."""

    def test_basic_sorting(self):
        """Tokens are sorted by expert and padded correctly."""
        from machete.kernels.moe.align_sort import moe_align_sort

        num_tokens, topk, num_experts, block_size = 8, 2, 4, 4

        topk_ids = torch.tensor([
            [0, 1], [2, 3], [0, 2], [1, 3],
            [0, 0], [1, 1], [2, 2], [3, 3],
        ], dtype=torch.int32)
        topk_weights = torch.ones(num_tokens, topk, dtype=torch.float32) * 0.5

        sorted_token_ids, expert_ids, sorted_weights, num_per_expert = (
            moe_align_sort(topk_ids, topk_weights, num_experts, block_size)
        )

        total_padded = sorted_token_ids.shape[0]

        # All positions should have valid expert_ids
        assert (expert_ids >= 0).all()
        assert (expert_ids < num_experts).all()

        # Total padded should be multiple of block_size * num_experts_with_tokens
        assert total_padded % block_size == 0

        # Padding positions should have sentinel value
        sentinel = num_tokens
        for i in range(total_padded):
            if sorted_token_ids[i].item() == sentinel:
                assert sorted_weights[i].item() == 0.0

    def test_all_same_expert(self):
        """All tokens assigned to same expert."""
        from machete.kernels.moe.align_sort import moe_align_sort

        num_tokens, topk, num_experts, block_size = 16, 1, 4, 8

        topk_ids = torch.zeros(num_tokens, topk, dtype=torch.int32)
        topk_weights = torch.ones(num_tokens, topk, dtype=torch.float32)

        sorted_token_ids, expert_ids, _, num_per_expert = (
            moe_align_sort(topk_ids, topk_weights, num_experts, block_size)
        )

        # Expert 0 should have all 16 tokens, padded to 16
        assert num_per_expert[0].item() == 16
        assert num_per_expert[1:].sum().item() == 0

    def test_empty_experts(self):
        """Some experts get zero tokens."""
        from machete.kernels.moe.align_sort import moe_align_sort

        num_tokens, topk, num_experts, block_size = 4, 1, 8, 4

        topk_ids = torch.tensor([[0], [0], [1], [1]], dtype=torch.int32)
        topk_weights = torch.ones(num_tokens, topk, dtype=torch.float32)

        _, _, _, num_per_expert = (
            moe_align_sort(topk_ids, topk_weights, num_experts, block_size)
        )

        # Only experts 0 and 1 have tokens
        assert num_per_expert[0].item() == 2
        assert num_per_expert[1].item() == 2
        assert num_per_expert[2:].sum().item() == 0


# =============================================================================
# Tests: MoeGemmOp
# =============================================================================


def _moe_gemm_case(num_tokens, num_experts, topk, K, N, dtype,
                   tile_m=64, tile_n=32, tile_k=32,
                   atol=1e-1, rtol=1e-2):
    """Run a single MoE GEMM test case and assert correctness."""
    from machete.kernels.moe.align_sort import moe_align_sort
    from machete.kernels.moe.ref import moe_gemm_ref

    torch.manual_seed(42)
    device = "cuda"

    # Create input and weights
    x = torch.randn(num_tokens, K, dtype=dtype, device=device)
    w = torch.randn(num_experts, N, K, dtype=dtype, device=device)

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

    # Gather sorted input (padding positions get clamped to last token)
    clamped_ids = sorted_token_ids.clamp(max=num_tokens - 1).long()
    sorted_x = x[clamped_ids]

    # Run reference
    ref = moe_gemm_ref(sorted_x, w, expert_ids, sorted_token_ids)

    # Run MoeGemmOp
    c = torch.zeros(total_padded, N, dtype=dtype, device=device)
    _run_moe_gemm(sorted_x, w, expert_ids, c,
                  tile_m=tile_m, tile_n=tile_n, tile_k=tile_k)

    # Only compare non-padding positions
    valid_mask = sorted_token_ids < num_tokens
    if valid_mask.any():
        torch.testing.assert_close(
            c[valid_mask], ref[valid_mask], atol=atol, rtol=rtol)


@requires_gpu
class TestMoeGemmForward:
    """MoE grouped GEMM forward pass correctness tests."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "num_tokens,num_experts,topk,K,N,tile_m,tile_n,tile_k", FORWARD_CASES)
    def test_forward_matrix(
        self, dtype, num_tokens, num_experts, topk, K, N, tile_m, tile_n, tile_k
    ):
        """Representative MoE GEMM forward matrix."""
        _moe_gemm_case(
            num_tokens, num_experts, topk, K, N, dtype,
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        )

    # ----- Topk variations -----

    @pytest.mark.parametrize("topk", [1, 2, 4])
    def test_topk_values(self, topk):
        """Different topk values."""
        _moe_gemm_case(64, 8, topk, 64, 32, torch.float16,
                       tile_m=32, tile_n=32, tile_k=32)

    # ----- Edge cases -----

    def test_single_expert(self):
        """All tokens routed to one expert."""
        torch.manual_seed(42)
        from machete.kernels.moe.align_sort import moe_align_sort
        from machete.kernels.moe.ref import moe_gemm_ref

        device = "cuda"
        dtype = torch.float16
        num_tokens, K, N, num_experts = 32, 64, 32, 4
        tile_m = 32

        x = torch.randn(num_tokens, K, dtype=dtype, device=device)
        w = torch.randn(num_experts, N, K, dtype=dtype, device=device)

        topk_ids = torch.zeros(num_tokens, 1, dtype=torch.int32, device=device)
        topk_weights = torch.ones(num_tokens, 1, dtype=torch.float32,
                                  device=device)

        sorted_token_ids, expert_ids, _, _ = (
            moe_align_sort(topk_ids, topk_weights, num_experts, tile_m)
        )
        total_padded = sorted_token_ids.shape[0]
        clamped = sorted_token_ids.clamp(max=num_tokens - 1).long()
        sorted_x = x[clamped]

        ref = moe_gemm_ref(sorted_x, w, expert_ids, sorted_token_ids)
        c = torch.zeros(total_padded, N, dtype=dtype, device=device)
        _run_moe_gemm(sorted_x, w, expert_ids, c,
                      tile_m=tile_m, tile_n=32, tile_k=32)

        valid = sorted_token_ids < num_tokens
        torch.testing.assert_close(c[valid], ref[valid], atol=1e-1, rtol=1e-2)

# =============================================================================
# Tests: End-to-end MoE
# =============================================================================


@requires_gpu
class TestMoeEndToEnd:
    """End-to-end MoE: align_sort + grouped GEMM + unscatter."""

    def test_full_moe_pipeline(self):
        """Full pipeline: route → sort → grouped GEMM → unscatter + weight."""
        from machete.kernels.moe.align_sort import moe_align_sort
        from machete.kernels.moe.ref import moe_full_ref

        torch.manual_seed(42)
        device = "cuda"
        dtype = torch.float16
        num_tokens, K, N = 64, 64, 32
        num_experts, topk = 8, 2
        tile_m = 32

        x = torch.randn(num_tokens, K, dtype=dtype, device=device)
        w = torch.randn(num_experts, N, K, dtype=dtype, device=device)

        topk_ids = torch.randint(0, num_experts, (num_tokens, topk),
                                 dtype=torch.int32, device=device)
        topk_weights = torch.randn(num_tokens, topk, dtype=torch.float32,
                                   device=device).softmax(dim=-1)

        # Reference: full MoE
        ref = moe_full_ref(x, w, topk_ids, topk_weights, num_experts)

        # Kernel path: align_sort + grouped GEMM + unscatter
        sorted_token_ids, expert_ids, sorted_weights, _ = (
            moe_align_sort(topk_ids, topk_weights, num_experts, tile_m)
        )
        total_padded = sorted_token_ids.shape[0]
        clamped = sorted_token_ids.clamp(max=num_tokens - 1).long()
        sorted_x = x[clamped]

        c = torch.zeros(total_padded, N, dtype=dtype, device=device)
        _run_moe_gemm(sorted_x, w, expert_ids, c,
                      tile_m=tile_m, tile_n=32, tile_k=32)

        # Unscatter: accumulate weighted expert outputs back to token positions
        output = torch.zeros(num_tokens, N, dtype=dtype, device=device)
        valid = sorted_token_ids < num_tokens
        valid_indices = torch.where(valid)[0]
        for idx in valid_indices:
            token_id = sorted_token_ids[idx].long()
            weight = sorted_weights[idx]
            output[token_id] += weight * c[idx]

        torch.testing.assert_close(output, ref, atol=2e-1, rtol=5e-2)
