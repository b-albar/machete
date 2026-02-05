# Copyright (c) 2025, Machete Authors
"""
Tests for Megakernel with Instruction Stream and Fine-Grained Barriers.

Tests the instruction stream builder and persistent megakernel implementation.
"""

import pytest
import torch


# =============================================================================
# Host-Side Tests (No GPU Required)
# =============================================================================


class TestInstructionStreamBuilder:
    """Tests for instruction stream generation."""

    def test_single_op(self):
        """Test instruction stream for a single operation."""
        from machete.megakernel import InstructionStreamBuilder, NOPOp

        builder = InstructionStreamBuilder()
        builder.add_op(NOPOp, tiles_m=4)

        assert builder.total_tiles == 4
        assert builder.num_barriers == 4

        instructions = builder.build()
        # 4 work instructions + 1 end marker
        assert len(instructions) == 5

        # Check first instruction (lightweight: op_idx, tile_m, tile_n, tile_l)
        instr0 = instructions[0]
        assert instr0.op_idx == 0
        assert instr0.tile_m == 0

        # Check last work instruction
        instr3 = instructions[3]
        assert instr3.op_idx == 0
        assert instr3.tile_m == 3

        # Check end marker
        end = instructions[4]
        assert end.op_idx == -1  # END_MARKER

        # Barrier formulas (separate from instructions)
        formulas = builder.get_op_barrier_formulas()
        assert len(formulas[0][0]) == 0  # No wait deps for first op
        assert len(formulas[0][1]) == 1  # One signal formula

    def test_two_ops_dependency(self):
        """Test that second op has barrier formulas for first op."""
        from machete.megakernel import InstructionStreamBuilder, NOPOp

        builder = InstructionStreamBuilder()
        builder.add_op(NOPOp, tiles_m=4)
        builder.add_op(NOPOp, tiles_m=4)

        assert builder.total_tiles == 8
        assert builder.num_barriers == 8  # 4 per op

        instructions = builder.build()
        # 8 work + 1 end marker
        assert len(instructions) == 9

        # Tiles are interleaved (not sequential) for pipeline overlap.
        # Verify all tiles from both ops are present.
        op0_count = sum(1 for i in instructions[:-1] if i.op_idx == 0)
        op1_count = sum(1 for i in instructions[:-1] if i.op_idx == 1)
        assert op0_count == 4
        assert op1_count == 4

        # Barrier formulas capture dependencies
        formulas = builder.get_op_barrier_formulas()
        # Op 0: no wait, signal base=0
        assert len(formulas[0][0]) == 0
        assert formulas[0][1][0].base == 0

        # Op 1: wait on op0's barriers (base=0), signal base=4
        wf = formulas[1][0][0]
        assert wf.base == 0
        assert wf.expected == 1
        assert formulas[1][1][0].base == 4

    def test_three_ops_chain(self):
        """Test dependency chain across three operations."""
        from machete.megakernel import InstructionStreamBuilder, NOPOp

        builder = InstructionStreamBuilder()
        builder.add_op(NOPOp, tiles_m=2)  # Barriers 0, 1
        builder.add_op(NOPOp, tiles_m=2)  # Barriers 2, 3
        builder.add_op(NOPOp, tiles_m=2)  # Barriers 4, 5

        assert builder.num_barriers == 6

        formulas = builder.get_op_barrier_formulas()

        # Op 0: no wait
        assert len(formulas[0][0]) == 0

        # Op 1: waits on op0 (barrier base 0)
        assert formulas[1][0][0].base == 0

        # Op 2: waits on op1 (barrier base 2)
        assert formulas[2][0][0].base == 2

    def test_build_tensor(self):
        """Test conversion to GPU tensor."""
        from machete.megakernel import InstructionStreamBuilder, NOPOp, INSTRUCTION_WORDS

        builder = InstructionStreamBuilder()
        builder.add_op(NOPOp, tiles_m=4)

        tensor = builder.build_tensor(device="cpu")  # Use CPU for test

        assert tensor.shape == (5, INSTRUCTION_WORDS)  # 4 tiles + end marker
        assert tensor.dtype == torch.int32

        # Verify op_idx column
        assert tensor[0, 0].item() == 0  # First instruction, op 0
        assert tensor[4, 0].item() == -1  # End marker

    def test_multidimensional_tiles(self):
        """Test 2D tile layout."""
        from machete.megakernel import InstructionStreamBuilder, NOPOp

        builder = InstructionStreamBuilder()
        builder.add_op(NOPOp, tiles_m=2, tiles_n=3)

        assert builder.total_tiles == 6
        instructions = builder.build()

        # Should have tiles (0,0), (1,0), (0,1), (1,1), (0,2), (1,2)
        # Ordering: tile_m varies fastest
        expected_tiles = [
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1),
            (0, 2),
            (1, 2),
        ]
        for i, (m, n) in enumerate(expected_tiles):
            assert instructions[i].tile_m == m, f"Wrong tile_m at index {i}"
            assert instructions[i].tile_n == n, f"Wrong tile_n at index {i}"


class TestMegakernelHost:
    """Host-side tests for Megakernel (no GPU required)."""

    def test_init(self):
        """Test megakernel initialization."""
        from machete.megakernel import Megakernel, MegakernelConfig, ScheduledOp, NOPOp

        ops = [
            ScheduledOp(NOPOp, tiles_m=16),
            ScheduledOp(NOPOp, tiles_m=16),
        ]

        config = MegakernelConfig(num_sms=8)
        kernel = Megakernel(ops, config=config, device="cpu")

        assert kernel.num_sms == 8
        assert len(kernel.ops) == 2
        assert kernel.total_tiles == 32
        assert kernel.num_barriers == 32

    def test_repr(self):
        """Test string representation."""
        from machete.megakernel import Megakernel, MegakernelConfig, ScheduledOp, NOPOp

        ops = [ScheduledOp(NOPOp, tiles_m=4)]
        config = MegakernelConfig(num_sms=8)
        kernel = Megakernel(ops, config=config, device="cpu")

        repr_str = repr(kernel)
        assert "Megakernel" in repr_str
        assert "NOPOp" in repr_str

    def test_create_megakernel(self):
        """Test factory function."""
        from machete.megakernel import create_megakernel, ScheduledOp, NOPOp

        ops = [ScheduledOp(NOPOp, tiles_m=4)]
        kernel = create_megakernel(ops, num_sms=4)

        assert kernel.num_sms == 4


# =============================================================================
# GPU Tests (Require Blackwell)
# =============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestMegakernelGPU:
    """GPU tests for Megakernel."""

    @pytest.fixture(autouse=True)
    def check_gpu(self):
        """Skip if not Blackwell (SM100+)."""
        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            pytest.skip("Requires Blackwell (SM100+) GPU")

    @pytest.mark.parametrize("num_ops", [1, 2])
    def test_nop_kernel_run(self, num_ops):
        """Test running NOPOp kernels with barrier reset across multiple runs."""
        from machete.megakernel import Megakernel, ScheduledOp, NOPOp

        ops = [ScheduledOp(NOPOp, tiles_m=8) for _ in range(num_ops)]
        kernel = Megakernel(ops)

        # Run multiple times to verify barrier reset
        for _ in range(2):
            kernel.run()



class TestSharedMemoryLayout:
    """Host-side tests for SharedMemoryLayout."""

    def test_layout_sizes(self):
        """Test that layout computes reasonable sizes."""
        from machete.megakernel.paged_memory import SharedMemoryLayout

        layout = SharedMemoryLayout(num_pages=4, page_size=16 * 1024)

        # 4 pages * 16KB = 64KB for page data alone
        assert layout.page_data_offset > 0
        assert layout.total_size >= 4 * 16 * 1024
        # Must fit in Blackwell's 228KB shared memory
        assert layout.total_size <= 228 * 1024

    def test_layout_offsets_are_aligned(self):
        """Test that all region offsets are 128-byte aligned."""
        from machete.megakernel.paged_memory import SharedMemoryLayout

        layout = SharedMemoryLayout(num_pages=4)

        assert layout.control_offset % 128 == 0
        assert layout.page_table_offset % 128 == 0
        # page_data_offset alignment depends on previous regions

    def test_layout_regions_dont_overlap(self):
        """Test that memory regions don't overlap."""
        from machete.megakernel.paged_memory import SharedMemoryLayout

        layout = SharedMemoryLayout(num_pages=4)

        assert layout.control_offset < layout.page_table_offset
        assert layout.page_table_offset < layout.page_data_offset
        assert layout.page_data_offset < layout.total_size

    def test_to_config(self):
        """Test PageTableConfig generation from layout."""
        from machete.megakernel.paged_memory import SharedMemoryLayout

        layout = SharedMemoryLayout(num_pages=4, page_size=16 * 1024)
        config = layout.to_config()

        assert config.num_pages == 4
        assert config.page_size == 16 * 1024
        assert config.base_offset == layout.page_data_offset


class TestMegakernelPagedMemory:
    """Test megakernel with paged memory integration."""

    def test_smem_size_from_layout(self):
        """Test that smem_size is computed from SharedMemoryLayout."""
        from machete.megakernel import Megakernel, MegakernelConfig, ScheduledOp, NOPOp

        ops = [ScheduledOp(NOPOp, tiles_m=4)]
        config = MegakernelConfig(num_sms=8, num_pages=4)
        kernel = Megakernel(ops, config=config, device="cpu")

        # smem_size should be > 0 and derived from the layout
        assert kernel.smem_size > 0
        assert kernel.smem_size >= 4 * config.page_size

    def test_auto_page_count(self):
        """Test that num_pages auto-sizes to fit the most demanding op."""
        from machete.megakernel import Megakernel, MegakernelConfig, ScheduledOp
        from machete.megakernel.ops import Op
        from typing import ClassVar

        class TwoPageOp(Op):
            NUM_INPUT_PAGES: ClassVar[int] = 2
            NUM_OUTPUT_PAGES: ClassVar[int] = 1

            @staticmethod
            def load_forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
                pass

            @staticmethod
            def compute_forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
                pass

            @staticmethod
            def store_forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
                pass

        ops = [ScheduledOp(TwoPageOp, tiles_m=4)]
        # Start with num_pages=1, should auto-size to 3
        config = MegakernelConfig(num_sms=8, num_pages=1)
        kernel = Megakernel(ops, config=config, device="cpu")

        assert kernel.config.num_pages >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
