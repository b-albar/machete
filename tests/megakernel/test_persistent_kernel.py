# Copyright (c) 2025, Machete Authors
"""
Tests for Megakernel with Instruction Stream and Fine-Grained Barriers.

Tests the instruction stream builder and persistent megakernel implementation.
"""

import pytest
import torch
from machete.megakernel.ops import Op


class _NOPOp(Op):
    """Test-only no-op for persistent kernel tests."""
    pass


# =============================================================================
# Host-Side Tests (No GPU Required)
# =============================================================================


class TestMegakernelHost:
    """Host-side tests for Megakernel (no GPU required)."""

    def test_init(self):
        """Test megakernel initialization."""
        from machete.megakernel import Megakernel, MegakernelConfig, ScheduledOp

        ops = [
            ScheduledOp(_NOPOp, tile_counts=(16,)),
            ScheduledOp(_NOPOp, tile_counts=(16,)),
        ]

        config = MegakernelConfig(num_sms=8)
        kernel = Megakernel(ops, config=config, device="cpu")

        assert kernel.num_sms == 8
        assert len(kernel.ops) == 2
        assert kernel.total_tiles == 32
        assert kernel.num_barriers == 32

    def test_repr(self):
        """Test string representation."""
        from machete.megakernel import Megakernel, MegakernelConfig, ScheduledOp

        ops = [ScheduledOp(_NOPOp, tile_counts=(4,))]
        config = MegakernelConfig(num_sms=8)
        kernel = Megakernel(ops, config=config, device="cpu")

        repr_str = repr(kernel)
        assert "Megakernel" in repr_str
        assert "_NOPOp" in repr_str

    def test_create_megakernel(self):
        """Test factory function."""
        from machete.megakernel import create_megakernel, ScheduledOp

        ops = [ScheduledOp(_NOPOp, tile_counts=(4,))]
        kernel = create_megakernel(ops, num_sms=4)

        assert kernel.num_sms == 4


# =============================================================================
# GPU Tests (Require Hopper)
# =============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestMegakernelGPU:
    """GPU tests for Megakernel."""

    @pytest.fixture(autouse=True)
    def check_gpu(self):
        """Skip if not Hopper (SM90+)."""
        major, _ = torch.cuda.get_device_capability()
        if major < 9:
            pytest.skip("Requires Hopper (SM90+) GPU")

    @pytest.mark.parametrize("num_ops", [1, 2])
    def test_nop_kernel_run(self, num_ops):
        """Test running _NOPOp kernels with barrier reset across multiple runs."""
        from machete.megakernel import Megakernel, ScheduledOp

        ops = [ScheduledOp(_NOPOp, tile_counts=(8,)) for _ in range(num_ops)]
        kernel = Megakernel(ops)

        # Run multiple times to verify barrier reset
        for _ in range(2):
            kernel.run()



class TestMegakernelPagedMemory:
    """Test megakernel with double-buffer memory integration."""

    def test_smem_size_from_layout(self):
        """Test that smem_size is computed from NPageLayout."""
        from machete.megakernel import Megakernel, MegakernelConfig, ScheduledOp

        ops = [ScheduledOp(_NOPOp, tile_counts=(4,))]
        config = MegakernelConfig(num_sms=8)
        kernel = Megakernel(ops, config=config, device="cpu")

        # smem_size should be > 0 and derived from the double-buffer layout
        # Double-buffer layout has scratch + 2 pages
        assert kernel.smem_size > 0
        assert kernel.smem_size >= 2 * config.page_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
