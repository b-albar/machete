# Copyright (c) 2025, Machete Authors
"""
Test: One-to-One Dependency with Same Tile Size (GPU)

Dependency Pattern:
==================

    Producer (tile_size=1)          Consumer (tile_size=1)
    ┌───┬───┬───┬───┐              ┌───┬───┬───┬───┐
    │ 0 │ 1 │ 2 │ 3 │   ───────>   │ 0 │ 1 │ 2 │ 3 │
    └───┴───┴───┴───┘              └───┴───┴───┴───┘

Barrier Mapping:
    - 4 barriers (one per tile)
    - Producer tile k signals barrier k
    - Consumer tile k waits on barrier k
    - expected = 1 (single signal per barrier)

Test Strategy:
    Producer: writes value (tile_0 + 1) to output[tile_0]
    Consumer: reads input[tile_0], writes input[tile_0] * 2 to output[tile_0]
    Verify: output == (tile_0 + 1) * 2 for all tiles
"""

import pytest
import torch
from typing import ClassVar, List

import cutlass.cute as cute
from cutlass import Int32, Int64

from machete.megakernel import Megakernel, MegakernelConfig, ScheduledOp
from machete.megakernel.ops import Op
from machete.megakernel.interpreter import st_global_i32, ld_global_i32
from machete.utils.testing import is_hopper_available


# =============================================================================
# Global result tensor pointers
# =============================================================================

_producer_result_ptr = 0
_consumer_result_ptr = 0


# =============================================================================
# Test Ops with Real GPU Operations
# =============================================================================


class ProducerOp(Op):
    """
    Writes (tile_0 + 1) to producer_result[tile_0].

    This simple operation allows verifying that the producer ran
    and wrote the expected value before the consumer reads it.
    """
    INPUTS: ClassVar[List[str]] = []
    OUTPUTS: ClassVar[List[str]] = ["data"]

    @cute.jit
    def compute(self, page_ptr, tile_0):
        tidx = cute.arch.thread_idx()[0]
        if tidx == Int32(0):
            # Write (tile_0 + 1) to producer result
            value = tile_0 + Int32(1)
            st_global_i32(Int64(_producer_result_ptr), tile_0, value)


class ConsumerOp(Op):
    """
    Reads producer_result[tile_0], writes producer_result[tile_0] * 2 to consumer_result[tile_0].

    If barrier synchronization works correctly, the consumer will read
    the value written by the producer (tile_0 + 1) and write (tile_0 + 1) * 2.
    """
    INPUTS: ClassVar[List[str]] = ["data"]
    OUTPUTS: ClassVar[List[str]] = []

    @cute.jit
    def compute(self, page_ptr, tile_0):
        tidx = cute.arch.thread_idx()[0]
        if tidx == Int32(0):
            # Read from producer, multiply by 2, write to consumer result
            val = ld_global_i32(Int64(_producer_result_ptr), tile_0)
            st_global_i32(Int64(_consumer_result_ptr), tile_0, val * Int32(2))


# =============================================================================
# GPU Tests
# =============================================================================


@pytest.mark.skipif(not is_hopper_available(), reason="Hopper (SM90+) GPU required")
class TestOneToOneSameSizeGPU:
    """
    GPU test for 1:1 dependency with same tile size.

    Diagram:
        Producer tiles:  [0] [1] [2] [3]
                          │   │   │   │
                          ▼   ▼   ▼   ▼
        Barriers:        (0) (1) (2) (3)
                          │   │   │   │
                          ▼   ▼   ▼   ▼
        Consumer tiles:  [0] [1] [2] [3]

    Each producer writes (tile_0 + 1).
    Each consumer reads that value and multiplies by 2.
    Expected output: (tile_0 + 1) * 2 = [2, 4, 6, 8]
    """

    def test_simple_chain(self):
        """
        Simple 1:1 chain with 4 tiles.

        Producer writes: [1, 2, 3, 4]
        Consumer reads and doubles: [2, 4, 6, 8]
        """
        global _producer_result_ptr, _consumer_result_ptr
        num_tiles = 4

        producer_results = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        consumer_results = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")

        _producer_result_ptr = producer_results.data_ptr()
        _consumer_result_ptr = consumer_results.data_ptr()

        ops = [
            ScheduledOp(ProducerOp, tile_counts=(num_tiles,), dim_names={"batch": 0}),
            ScheduledOp(ConsumerOp, tile_counts=(num_tiles,), dim_names={"batch": 0}),
        ]

        config = MegakernelConfig(num_sms=2)
        kernel = Megakernel(ops, config=config)
        kernel.run()

        # Verify producer: tile_0 + 1
        expected_producer = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cuda")
        assert torch.equal(producer_results, expected_producer), (
            f"Producer mismatch: expected {expected_producer.tolist()}, got {producer_results.tolist()}"
        )

        # Verify consumer: (tile_0 + 1) * 2
        expected_consumer = torch.tensor([2, 4, 6, 8], dtype=torch.int32, device="cuda")
        assert torch.equal(consumer_results, expected_consumer), (
            f"Consumer mismatch: expected {expected_consumer.tolist()}, got {consumer_results.tolist()}"
        )

    def test_larger_chain(self):
        """
        1:1 chain with 16 tiles.

        Verifies barrier synchronization works correctly at larger scale.
        """
        global _producer_result_ptr, _consumer_result_ptr
        num_tiles = 16

        producer_results = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        consumer_results = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")

        _producer_result_ptr = producer_results.data_ptr()
        _consumer_result_ptr = consumer_results.data_ptr()

        ops = [
            ScheduledOp(ProducerOp, tile_counts=(num_tiles,), dim_names={"batch": 0}),
            ScheduledOp(ConsumerOp, tile_counts=(num_tiles,), dim_names={"batch": 0}),
        ]

        config = MegakernelConfig(num_sms=2)
        kernel = Megakernel(ops, config=config)
        kernel.run()

        expected_producer = torch.tensor(
            [i + 1 for i in range(num_tiles)],
            dtype=torch.int32, device="cuda"
        )
        expected_consumer = torch.tensor(
            [(i + 1) * 2 for i in range(num_tiles)],
            dtype=torch.int32, device="cuda"
        )

        assert torch.equal(producer_results, expected_producer), (
            f"Producer mismatch at some indices"
        )
        assert torch.equal(consumer_results, expected_consumer), (
            f"Consumer mismatch at some indices"
        )

    def test_three_op_chain(self):
        """
        Three-op chain: A -> B -> C

        Diagram:
            Op A writes: tile_0 + 1
            Op B reads, multiplies by 2, writes
            Op C reads, multiplies by 3, writes

            Expected: (tile_0 + 1) * 2 * 3 = (tile_0 + 1) * 6
        """
        global _opa_result_ptr, _opb_result_ptr, _opc_result_ptr
        num_tiles = 8

        # Define global pointers for this test
        _opa_result_ptr = 0
        _opb_result_ptr = 0
        _opc_result_ptr = 0

        class OpA(Op):
            INPUTS: ClassVar[List[str]] = []
            OUTPUTS: ClassVar[List[str]] = ["buf1"]

            @cute.jit
            def compute(self, page_ptr, tile_0):
                tidx = cute.arch.thread_idx()[0]
                if tidx == Int32(0):
                    st_global_i32(Int64(_opa_result_ptr), tile_0, tile_0 + Int32(1))

        class OpB(Op):
            INPUTS: ClassVar[List[str]] = ["buf1"]
            OUTPUTS: ClassVar[List[str]] = ["buf2"]

            @cute.jit
            def compute(self, page_ptr, tile_0):
                tidx = cute.arch.thread_idx()[0]
                if tidx == Int32(0):
                    val = ld_global_i32(Int64(_opa_result_ptr), tile_0)
                    st_global_i32(Int64(_opb_result_ptr), tile_0, val * Int32(2))

        class OpC(Op):
            INPUTS: ClassVar[List[str]] = ["buf2"]
            OUTPUTS: ClassVar[List[str]] = []

            @cute.jit
            def compute(self, page_ptr, tile_0):
                tidx = cute.arch.thread_idx()[0]
                if tidx == Int32(0):
                    val = ld_global_i32(Int64(_opb_result_ptr), tile_0)
                    st_global_i32(Int64(_opc_result_ptr), tile_0, val * Int32(3))

        buf1 = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        buf2 = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        buf3 = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")

        _opa_result_ptr = buf1.data_ptr()
        _opb_result_ptr = buf2.data_ptr()
        _opc_result_ptr = buf3.data_ptr()

        ops = [
            ScheduledOp(OpA, tile_counts=(num_tiles,), dim_names={"batch": 0}),
            ScheduledOp(OpB, tile_counts=(num_tiles,), dim_names={"batch": 0}),
            ScheduledOp(OpC, tile_counts=(num_tiles,), dim_names={"batch": 0}),
        ]

        config = MegakernelConfig(num_sms=2)
        kernel = Megakernel(ops, config=config)
        kernel.run()

        expected = torch.tensor(
            [(i + 1) * 6 for i in range(num_tiles)],
            dtype=torch.int32, device="cuda"
        )
        assert torch.equal(buf3, expected), (
            f"Expected {expected.tolist()}, got {buf3.tolist()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
