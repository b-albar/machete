# Copyright (c) 2025, Machete Authors
"""Tests for GemmOp tensor parallelism.

Tests:
1. Single-GPU: GemmColumnParallelOp and GemmRowParallelOp with same-device peer.
2. Multi-GPU column-parallel: broadcast C shard to peer via TMA S2G.
3. Multi-GPU row-parallel: broadcast partial C to peer via TMA S2G.
"""

import contextlib
import io

import pytest
import torch


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


def _gemm_ref(a, b_t):
    """Reference GEMM: C = A @ B_T^T, computed in fp32."""
    return (a.float() @ b_t.float().t()).to(a.dtype)


# =============================================================================
# Single-GPU Tests (same-device peer buffer)
# =============================================================================


class TestGemmTPSingleGPU:
    """TP subclasses with same-device peer buffer to exercise communicate()."""

    @requires_gpu
    def test_column_parallel_smoke(self):
        """GemmColumnParallelOp broadcasts C to same-device peer buffer."""
        from machete.megakernel import Megakernel, MegakernelConfig
        from machete.kernels.gemm import GemmColumnParallelOp

        M, K, N = 64, 64, 32
        torch.manual_seed(42)
        a = torch.randn(1, M, K, dtype=torch.float16, device="cuda")
        b = torch.randn(N, K, dtype=torch.float16, device="cuda")
        c = torch.zeros(1, M, N, dtype=torch.float16, device="cuda")
        peer_c = torch.zeros(1, M, N, dtype=torch.float16, device="cuda")

        ops = GemmColumnParallelOp.schedule(a=a, b=b, c=c)
        base_config = GemmColumnParallelOp.kernel_config(ops)
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

        ref = _gemm_ref(a.squeeze(0), b)
        torch.testing.assert_close(c.squeeze(0), ref, atol=1e-1, rtol=1e-2)
        torch.testing.assert_close(peer_c.squeeze(0), ref, atol=1e-1, rtol=1e-2)

    @requires_gpu
    def test_row_parallel_smoke(self):
        """GemmRowParallelOp broadcasts C to same-device peer buffer."""
        from machete.megakernel import Megakernel, MegakernelConfig
        from machete.kernels.gemm import GemmRowParallelOp

        M, K, N = 64, 64, 32
        torch.manual_seed(42)
        a = torch.randn(1, M, K, dtype=torch.float16, device="cuda")
        b = torch.randn(N, K, dtype=torch.float16, device="cuda")
        c = torch.zeros(1, M, N, dtype=torch.float16, device="cuda")
        peer_c = torch.zeros(1, M, N, dtype=torch.float16, device="cuda")

        ops = GemmRowParallelOp.schedule(a=a, b=b, c=c)
        base_config = GemmRowParallelOp.kernel_config(ops)
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

        ref = _gemm_ref(a.squeeze(0), b)
        torch.testing.assert_close(c.squeeze(0), ref, atol=1e-1, rtol=1e-2)
        torch.testing.assert_close(peer_c.squeeze(0), ref, atol=1e-1, rtol=1e-2)

    @requires_gpu
    def test_schedule_forward_tp_column(self):
        """schedule_forward_tp with tp_mode='column'."""
        from machete.kernels.gemm import GemmOp, GemmColumnParallelOp

        M, K, N = 64, 64, 32
        a = torch.randn(1, M, K, dtype=torch.float16, device="cuda")
        b = torch.randn(N, K, dtype=torch.float16, device="cuda")
        c = torch.zeros(1, M, N, dtype=torch.float16, device="cuda")

        ops = GemmOp.schedule_forward_tp(tp_mode='column', a=a, b=b, c=c)
        assert ops[0].op_cls is GemmColumnParallelOp

    @requires_gpu
    def test_schedule_forward_tp_row(self):
        """schedule_forward_tp with tp_mode='row'."""
        from machete.kernels.gemm import GemmOp, GemmRowParallelOp

        M, K, N = 64, 64, 32
        a = torch.randn(1, M, K, dtype=torch.float16, device="cuda")
        b = torch.randn(N, K, dtype=torch.float16, device="cuda")
        c = torch.zeros(1, M, N, dtype=torch.float16, device="cuda")

        ops = GemmOp.schedule_forward_tp(tp_mode='row', a=a, b=b, c=c)
        assert ops[0].op_cls is GemmRowParallelOp


# =============================================================================
# Multi-GPU Column-Parallel Tests
# =============================================================================


@requires_gpu
@requires_multi_gpu
class TestGemmTPColumnParallel:
    """Column-parallel TP: broadcast C shard to peer."""

    def test_broadcast_single_tile(self):
        """GPU 0 computes C_shard = A @ W_shard^T, broadcasts to GPU 1."""
        _enable_peer_access(0, 1)
        _enable_peer_access(1, 0)

        from machete.megakernel import Megakernel, MegakernelConfig
        from machete.kernels.gemm import GemmColumnParallelOp

        M, K, N = 64, 64, 32
        torch.manual_seed(42)
        a = torch.randn(1, M, K, dtype=torch.float16, device="cuda:0")
        b = torch.randn(N, K, dtype=torch.float16, device="cuda:0")
        c = torch.zeros(1, M, N, dtype=torch.float16, device="cuda:0")
        peer_c = torch.zeros(1, M, N, dtype=torch.float16, device="cuda:1")

        ops = GemmColumnParallelOp.schedule(a=a, b=b, c=c)
        base_config = GemmColumnParallelOp.kernel_config(ops)
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

        ref = _gemm_ref(a.squeeze(0), b)
        torch.testing.assert_close(c.squeeze(0), ref, atol=1e-1, rtol=1e-2)
        torch.testing.assert_close(
            peer_c.squeeze(0).to("cuda:0"), ref, atol=1e-1, rtol=1e-2)

    def test_broadcast_multi_tile(self):
        """Multi-tile column-parallel broadcast."""
        _enable_peer_access(0, 1)
        _enable_peer_access(1, 0)

        from machete.megakernel import Megakernel, MegakernelConfig
        from machete.kernels.gemm import GemmColumnParallelOp

        M, K, N = 256, 64, 64
        torch.manual_seed(42)
        a = torch.randn(1, M, K, dtype=torch.float16, device="cuda:0")
        b = torch.randn(N, K, dtype=torch.float16, device="cuda:0")
        c = torch.zeros(1, M, N, dtype=torch.float16, device="cuda:0")
        peer_c = torch.zeros(1, M, N, dtype=torch.float16, device="cuda:1")

        ops = GemmColumnParallelOp.schedule(a=a, b=b, c=c)
        base_config = GemmColumnParallelOp.kernel_config(ops)
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

        ref = _gemm_ref(a.squeeze(0), b)
        torch.testing.assert_close(c.squeeze(0), ref, atol=1e-1, rtol=1e-2)
        torch.testing.assert_close(
            peer_c.squeeze(0).to("cuda:0"), ref, atol=1e-1, rtol=1e-2)


# =============================================================================
# Multi-GPU Row-Parallel Tests
# =============================================================================


@requires_gpu
@requires_multi_gpu
class TestGemmTPRowParallel:
    """Row-parallel TP: broadcast partial C to peer.

    Each GPU computes C_partial from its K-shard and broadcasts
    the partial result to peer buffers via TMA S2G copy.

    NOTE: True all-reduce (atomic add) requires CopyReduceBulkTensorTileS2GOp,
    which has a 2x output bug in the CUTLASS DSL. As a workaround,
    GemmRowParallelOp uses regular broadcast. The caller must handle
    reduction externally (e.g., NCCL all-reduce).
    """

    def test_broadcast_partial(self):
        """GPU 0 computes partial C and broadcasts to GPU 1."""
        _enable_peer_access(0, 1)
        _enable_peer_access(1, 0)

        from machete.megakernel import Megakernel, MegakernelConfig
        from machete.kernels.gemm import GemmRowParallelOp

        M, K, N = 64, 64, 32
        torch.manual_seed(42)
        a = torch.randn(1, M, K, dtype=torch.float16, device="cuda:0")
        b = torch.randn(N, K, dtype=torch.float16, device="cuda:0")
        c = torch.zeros(1, M, N, dtype=torch.float16, device="cuda:0")
        peer_c = torch.zeros(1, M, N, dtype=torch.float16, device="cuda:1")

        ops = GemmRowParallelOp.schedule(a=a, b=b, c=c)
        base_config = GemmRowParallelOp.kernel_config(ops)
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

        ref = _gemm_ref(a.squeeze(0), b)
        torch.testing.assert_close(c.squeeze(0), ref, atol=1e-1, rtol=1e-2)
        torch.testing.assert_close(
            peer_c.squeeze(0).to("cuda:0"), ref, atol=1e-1, rtol=1e-2)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))
