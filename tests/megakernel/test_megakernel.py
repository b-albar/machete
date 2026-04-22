# Copyright (c) 2025, Machete Authors
"""
Tests for Template-Based Megakernel.

Tests the megakernel that uses instruction stream and fine-grained barriers.
"""

import pytest
import torch

from tests.megakernel.support import get_nop_op


class TestMegakernel:
    """Test the template megakernel (now uses instruction stream)."""

    def test_megakernel_creation(self):
        """Test creating a megakernel instance."""
        from machete.megakernel import Megakernel, MegakernelConfig, ScheduledOp
        NopOp = get_nop_op()

        # Define some operations
        ops = [
            ScheduledOp(NopOp, tile_counts=(32,)),
            ScheduledOp(NopOp, tile_counts=(16,)),
        ]

        config = MegakernelConfig(num_sms=8)
        kernel = Megakernel(ops, config=config, device="cpu")

        # New API: total_tiles instead of total_blocks, grid is based on num_sms
        assert kernel.total_tiles == 48
        assert kernel.grid == (8, 1, 1)  # Now based on num_sms (persistent blocks)
        assert kernel.block == (256, 1, 1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_validation_check(self):
        """Test that validation checks for Hopper+ Architecture."""
        from machete.megakernel import Megakernel, ScheduledOp
        NopOp = get_nop_op()

        ops = [ScheduledOp(NopOp, tile_counts=(1,))]
        kernel = Megakernel(ops)

        major, _ = torch.cuda.get_device_capability()
        if major < 9:
            with pytest.raises(RuntimeError, match="requires Hopper"):
                kernel.run()
        else:
            # Should run (or fail with launch error if env is unstable, but NOT validation error)
            try:
                kernel.run()
            except Exception as e:
                # If it's a launch error, we passed the validation check (it tried to run)
                if "requires Hopper" in str(e):
                    raise e
                print(f"Kernel tried to run but failed (expected on some envs): {e}")
