# Copyright (c) 2025, Machete Authors
"""Tests for Conv1dOp tensor parallelism (peer TMA stores).

Tests:
1. Single-GPU: Conv1dOp with same-device peer buffer.
2. Multi-GPU integration: Conv1dOp broadcasts y tile to peer GPU via TMA S2G.
"""

import contextlib
import io

import pytest
import torch

from machete.kernels.conv1d.ref import causal_conv1d_ref


def _is_hopper_or_newer():
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
    not (_is_hopper_or_newer() and CUTLASS_AVAILABLE),
    reason="Requires Hopper+ GPU with CUTLASS",
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


# =============================================================================
# Single-GPU Test (same-device peer buffer)
# =============================================================================


class TestConv1dTPSingleGPU:
    """Conv1dOp with same-device peer buffer to exercise communicate()."""

    @requires_gpu
    def test_smoke_same_device_peer(self):
        """Conv1dOp broadcasts y to same-device peer buffer."""
        from machete.megakernel import Megakernel, MegakernelConfig
        from machete.kernels.conv1d import Conv1dOp

        B, L, D, K = 1, 32, 64, 4
        torch.manual_seed(42)
        x = torch.randn(B, L, D, dtype=torch.float16, device="cuda")
        w = torch.randn(D, K, dtype=torch.float16, device="cuda")
        y = torch.empty_like(x)
        peer_y = torch.zeros(B, L, D, dtype=torch.float16, device="cuda")

        ops = Conv1dOp.schedule(x=x, w=w, y=y)
        base_config = Conv1dOp.kernel_config(ops)
        config = MegakernelConfig(
            threads_per_block=base_config.threads_per_block,
            page_size=base_config.page_size,
            peer_buffers={"y": [peer_y]},
            peer_barriers=_alloc_peer_barriers(ops),
            device_idx=0,
            num_devices=2,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            kernel = Megakernel(ops, config=config)
            kernel.run()
        torch.cuda.synchronize()

        ref = causal_conv1d_ref(x, w)
        torch.testing.assert_close(y, ref, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(peer_y, ref, atol=1e-2, rtol=1e-2)


# =============================================================================
# Multi-GPU Integration Tests
# =============================================================================


@requires_gpu
@requires_multi_gpu
class TestConv1dTPMultiGPU:
    """Integration tests for Conv1dOp peer TMA communication."""

    def test_broadcast_single_tile(self):
        """GPU 0 computes conv1d, broadcasts y to GPU 1 peer buffer."""
        _enable_peer_access(0, 1)
        _enable_peer_access(1, 0)

        from machete.megakernel import Megakernel, MegakernelConfig
        from machete.kernels.conv1d import Conv1dOp

        B, L, D, K = 1, 16, 64, 4
        torch.manual_seed(42)
        x = torch.randn(B, L, D, dtype=torch.float16, device="cuda:0")
        w = torch.randn(D, K, dtype=torch.float16, device="cuda:0")
        y = torch.empty_like(x)
        peer_y = torch.zeros(B, L, D, dtype=torch.float16, device="cuda:1")

        ops = Conv1dOp.schedule(x=x, w=w, y=y)
        base_config = Conv1dOp.kernel_config(ops)
        config = MegakernelConfig(
            threads_per_block=base_config.threads_per_block,
            page_size=base_config.page_size,
            peer_buffers={"y": [peer_y]},
            peer_barriers=_alloc_peer_barriers(ops, device="cuda:0"),
            device_idx=0,
            num_devices=2,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            kernel = Megakernel(ops, config=config)
            kernel.run()
        torch.cuda.synchronize()

        ref = causal_conv1d_ref(x, w)
        torch.testing.assert_close(y, ref, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(
            peer_y.to("cuda:0"), ref, atol=1e-2, rtol=1e-2)

    def test_broadcast_multi_tile(self):
        """Multi-tile conv1d peer broadcast."""
        _enable_peer_access(0, 1)
        _enable_peer_access(1, 0)

        from machete.megakernel import Megakernel, MegakernelConfig
        from machete.kernels.conv1d import Conv1dOp

        B, L, D, K = 2, 128, 64, 4
        torch.manual_seed(42)
        x = torch.randn(B, L, D, dtype=torch.float16, device="cuda:0")
        w = torch.randn(D, K, dtype=torch.float16, device="cuda:0")
        y = torch.empty_like(x)
        peer_y = torch.zeros(B, L, D, dtype=torch.float16, device="cuda:1")

        ops = Conv1dOp.schedule(x=x, w=w, y=y)
        base_config = Conv1dOp.kernel_config(ops)
        config = MegakernelConfig(
            threads_per_block=base_config.threads_per_block,
            page_size=base_config.page_size,
            peer_buffers={"y": [peer_y]},
            peer_barriers=_alloc_peer_barriers(ops, device="cuda:0"),
            device_idx=0,
            num_devices=2,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            kernel = Megakernel(ops, config=config)
            kernel.run()
        torch.cuda.synchronize()

        ref = causal_conv1d_ref(x, w)
        torch.testing.assert_close(y, ref, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(
            peer_y.to("cuda:0"), ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))
