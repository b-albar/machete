# Copyright (c) 2025, Machete Authors
"""Tests for MoeGemmOp tensor parallelism (peer TMA stores).

Tests:
1. Single-GPU: MoeGemmOp with same-device peer buffer.
2. Multi-GPU integration: MoeGemmOp broadcasts C tile to peer GPU.
"""

import contextlib
import io
import importlib.util

import pytest
import torch

if importlib.util.find_spec("cutlass") is None:
    pytest.skip("Requires CUTLASS", allow_module_level=True)


def _is_sm90_or_newer():
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor >= 90


try:
    import cutlass  # noqa: F401
    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False


requires_gpu = pytest.mark.skipif(
    not (_is_sm90_or_newer() and CUTLASS_AVAILABLE),
    reason="Requires SM_90+ GPU with CUTLASS",
)


def _num_gpus():
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


requires_multi_gpu = pytest.mark.skipif(
    _num_gpus() < 2, reason="Requires 2+ GPUs",
)


def _enable_peer_access(src, dst):
    with torch.cuda.device(src):
        if torch.cuda.can_device_access_peer(src, dst):
            try:
                torch.cuda.enable_peer_access(dst)
            except RuntimeError:
                pass


def _alloc_peer_barriers(ops, device="cuda"):
    """Allocate peer barrier tensor sized for ops with peer stores."""
    n = sum(
        op.total_tiles for op in ops
        if getattr(op.op_cls, 'peer_stores', set())
        or getattr(op.op_cls, 'peer_reduce_stores', set())
    )
    return torch.zeros(max(n, 1), dtype=torch.int32, device=device)


def _moe_gemm_ref(sorted_x, w, expert_ids, tile_M):
    """Reference grouped GEMM matching MoeGemmOp tile-level expert selection.

    MoeGemmOp reads ONE expert_id per tile (expert_ids[tile_start]),
    so the reference must use the same expert for all tokens in a tile.
    """
    M = sorted_x.shape[0]
    N = w.shape[1]
    c = torch.zeros(M, N, dtype=sorted_x.dtype, device=sorted_x.device)
    for tile_start in range(0, M, tile_M):
        tile_end = min(tile_start + tile_M, M)
        eid = expert_ids[tile_start].item()
        for i in range(tile_start, tile_end):
            c[i] = (sorted_x[i].float() @ w[eid].float().t()).to(sorted_x.dtype)
    return c


# =============================================================================
# Single-GPU Test (same-device peer buffer)
# =============================================================================


class TestMoeTPSingleGPU:
    """MoeGemmOp with same-device peer buffer to exercise communicate()."""

    @requires_gpu
    def test_smoke_same_device_peer(self):
        """MoeGemmOp broadcasts C to same-device peer buffer."""
        from machete.megakernel import Megakernel, MegakernelConfig
        from machete.kernels.moe import MoeGemmOp

        tile_M, K, N, E = 64, 64, 32, 4
        M = tile_M * 2  # 2 tiles
        torch.manual_seed(42)
        sorted_x = torch.randn(M, K, dtype=torch.float16, device="cuda")
        w = torch.randn(E, N, K, dtype=torch.float16, device="cuda")
        # One expert per tile_M block (MoeGemmOp reads expert_ids[tile_start])
        expert_ids = torch.zeros(M, dtype=torch.int32, device="cuda")
        for t in range(0, M, tile_M):
            expert_ids[t:t + tile_M] = torch.randint(0, E, (1,)).item()
        c = torch.zeros(M, N, dtype=torch.float16, device="cuda")
        peer_c = torch.zeros(M, N, dtype=torch.float16, device="cuda")

        ops = MoeGemmOp.schedule(
            sorted_x=sorted_x, w=w, expert_ids=expert_ids, c=c,
            tile_sizes={"M": tile_M, "N": N, "K": 32},
        )
        base_config = MoeGemmOp.kernel_config(ops)
        config = MegakernelConfig(
            threads_per_block=base_config.threads_per_block,
            page_size=base_config.page_size,
            peer_buffers={"c": [peer_c]},
            peer_barriers=_alloc_peer_barriers(ops),
            device_idx=0,
            num_devices=2,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            kernel = Megakernel(ops, config=config)
            kernel.run()
        torch.cuda.synchronize()

        ref = _moe_gemm_ref(sorted_x, w, expert_ids, tile_M)
        torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)
        torch.testing.assert_close(peer_c, ref, atol=1e-1, rtol=1e-2)


# =============================================================================
# Multi-GPU Integration Tests
# =============================================================================


@requires_gpu
@requires_multi_gpu
class TestMoeTPMultiGPU:
    """Integration tests for MoeGemmOp peer TMA communication."""

    def test_broadcast_single_tile(self):
        """GPU 0 computes MoE GEMM, broadcasts C to GPU 1."""
        _enable_peer_access(0, 1)
        _enable_peer_access(1, 0)

        from machete.megakernel import Megakernel, MegakernelConfig
        from machete.kernels.moe import MoeGemmOp

        tile_M, K, N, E = 64, 64, 32, 4
        M = tile_M  # single tile
        torch.manual_seed(42)
        sorted_x = torch.randn(M, K, dtype=torch.float16, device="cuda:0")
        w = torch.randn(E, N, K, dtype=torch.float16, device="cuda:0")
        # Single tile → one expert for all tokens
        expert_ids = torch.full(
            (M,), 2, dtype=torch.int32, device="cuda:0")
        c = torch.zeros(M, N, dtype=torch.float16, device="cuda:0")
        peer_c = torch.zeros(M, N, dtype=torch.float16, device="cuda:1")

        ops = MoeGemmOp.schedule(
            sorted_x=sorted_x, w=w, expert_ids=expert_ids, c=c,
            tile_sizes={"M": tile_M, "N": N, "K": 32},
        )
        base_config = MoeGemmOp.kernel_config(ops)
        config = MegakernelConfig(
            threads_per_block=base_config.threads_per_block,
            page_size=base_config.page_size,
            peer_buffers={"c": [peer_c]},
            peer_barriers=_alloc_peer_barriers(ops, device="cuda:0"),
            device_idx=0,
            num_devices=2,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            kernel = Megakernel(ops, config=config)
            kernel.run()
        torch.cuda.synchronize()

        ref = _moe_gemm_ref(sorted_x, w, expert_ids, tile_M)
        torch.testing.assert_close(c, ref, atol=1e-1, rtol=1e-2)
        torch.testing.assert_close(
            peer_c.to("cuda:0"), ref, atol=1e-1, rtol=1e-2)
