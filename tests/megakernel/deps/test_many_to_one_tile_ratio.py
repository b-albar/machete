# Copyright (c) 2025, Machete Authors
"""
Test: Many-to-One Dependency Pattern (GPU)

Dependency Pattern:
==================

Multiple producer tiles contribute to one consumer tile.
Producer has extra dimension (N) that consumer doesn't have.

    Producer (M x N)              Consumer (M only)
    ┌───────────────────┐        ┌───┬───┬───┬───┐
    │ (0,0)...(0,3)     │  ───>  │ 0 │ 1 │ 2 │ 3 │
    │ (1,0)...(1,3)     │        └───┴───┴───┴───┘
    │ (2,0)...(2,3)     │
    │ (3,0)...(3,3)     │
    └───────────────────┘

Barrier Mapping:
    - 4 barriers (indexed by M - the shared dimension)
    - All producer tiles (m, *) signal barrier m
    - Consumer tile m waits on barrier m
    - expected = N (number of producer signals per barrier)

Test Strategy:
    Producer: writes (tile_0 * 1000 + tile_1) to matrix[tile_0 * N + tile_1]
    Consumer: reads matrix[tile_0 * N + (N-1)], verifies last producer ran, writes to output
    Verify: consumer results indicate all producers for that dim 0 completed
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

_producer_matrix_ptr = 0
_consumer_result_ptr = 0
_num_cols = 4  # N dimension


# =============================================================================
# Test Ops with Real GPU Operations
# =============================================================================


class Producer2DOp(Op):
    """
    Writes (tile_0 * 1000 + tile_1) to matrix[tile_0 * N + tile_1].

    This producer iterates over dimensions 0 x 1.
    Each producer tile writes to a unique location in the matrix.
    """

    INPUTS: ClassVar[List[str]] = []
    OUTPUTS: ClassVar[List[str]] = ["data"]

    @staticmethod
    def compute(
        page_ptr: Int32,
        op_config_ptr: Int64,
    ) -> None:
        tidx = cute.arch.thread_idx()[0]
        if tidx == Int32(0):
            # Write tile_0 * 1000 + tile_1 to uniquely identify this producer
            idx = tile_0 * Int32(_num_cols) + tile_1
            val = tile_0 * Int32(1000) + tile_1
            st_global_i32(Int64(_producer_matrix_ptr), idx, val)


class Consumer1DOp(Op):
    """
    Reads the last producer value for this dim 0, writes m * 100 + 7 to output.

    If barrier synchronization works correctly, the last producer tile
    (m, N-1) will have already written its value before the consumer runs.
    """

    INPUTS: ClassVar[List[str]] = ["data"]
    OUTPUTS: ClassVar[List[str]] = []

    @staticmethod
    def compute(
        page_ptr: Int32,
        op_config_ptr: Int64,
    ) -> None:
        tidx = cute.arch.thread_idx()[0]
        if tidx == Int32(0):
            # Read the last producer's value to verify it completed
            last_n = Int32(_num_cols) - Int32(1)
            idx = tile_0 * Int32(_num_cols) + last_n
            _ = ld_global_i32(Int64(_producer_matrix_ptr), idx)
            # Write consumer result
            st_global_i32(Int64(_consumer_result_ptr), tile_0, tile_0 * Int32(100) + Int32(7))


# =============================================================================
# GPU Tests
# =============================================================================


@pytest.mark.skipif(not is_hopper_available(), reason="Hopper (SM90+) GPU required")
class TestManyToOnePatternGPU:
    """
    GPU test for many:1 dependency: 2D producer -> 1D consumer.

    Diagram (M=4, N=4):
        Producer tiles (M x N):
        ┌─────────────────────────────────┐
        │ (0,0) (0,1) (0,2) (0,3) → M=0   │
        │ (1,0) (1,1) (1,2) (1,3) → M=1   │
        │ (2,0) (2,1) (2,2) (2,3) → M=2   │
        │ (3,0) (3,1) (3,2) (3,3) → M=3   │
        └─────────────────────────────────┘
                    │
                    ▼ (aggregate by M)
        Barriers: (0) (1) (2) (3)
                    │
                    ▼
        Consumer tiles: [0] [1] [2] [3]

    Producer (m, n) writes m * 1000 + n to matrix[m * N + n].
    Consumer m waits for all producers with same M, writes m * 100 + 7.
    """

    def test_basic_2d_to_1d(self):
        """
        Basic 2D producer (4x4) to 1D consumer (4) test.

        All 16 producer tiles must complete before their corresponding consumer runs.
        """
        global _producer_matrix_ptr, _consumer_result_ptr, _num_cols
        tiles_m = 4
        tiles_n = 4
        _num_cols = tiles_n

        producer_matrix = torch.zeros(tiles_m * tiles_n, dtype=torch.int32, device="cuda")
        consumer_results = torch.zeros(tiles_m, dtype=torch.int32, device="cuda")

        _producer_matrix_ptr = producer_matrix.data_ptr()
        _consumer_result_ptr = consumer_results.data_ptr()

        ops = [
            ScheduledOp(
                Producer2DOp, tile_counts=(tiles_m, tiles_n),
                dim_names={"batch": 0, "seq": 1}
            ),
            ScheduledOp(
                Consumer1DOp, tile_counts=(tiles_m,),
                dim_names={"batch": 0}
            ),
        ]

        config = MegakernelConfig(num_sms=2)
        kernel = Megakernel(ops, config=config)
        kernel.run()

        # Verify all producers wrote their values: m * 1000 + n
        expected_matrix = torch.tensor([
            m * 1000 + n
            for m in range(tiles_m)
            for n in range(tiles_n)
        ], dtype=torch.int32, device="cuda")
        assert torch.equal(producer_matrix, expected_matrix), (
            f"Producer matrix mismatch: expected {expected_matrix.tolist()}, got {producer_matrix.tolist()}"
        )

        # Verify consumer: m * 100 + 7
        expected_consumer = torch.tensor([
            m * 100 + 7 for m in range(tiles_m)
        ], dtype=torch.int32, device="cuda")
        assert torch.equal(consumer_results, expected_consumer), (
            f"Consumer mismatch: expected {expected_consumer.tolist()}, got {consumer_results.tolist()}"
        )

    def test_larger_2d_to_1d(self):
        """
        Larger 2D producer (8x8) to 1D consumer (8) test.
        """
        global _producer_matrix_ptr, _consumer_result_ptr, _num_cols
        tiles_m = 8
        tiles_n = 8
        _num_cols = tiles_n

        producer_matrix = torch.zeros(tiles_m * tiles_n, dtype=torch.int32, device="cuda")
        consumer_results = torch.zeros(tiles_m, dtype=torch.int32, device="cuda")

        _producer_matrix_ptr = producer_matrix.data_ptr()
        _consumer_result_ptr = consumer_results.data_ptr()

        ops = [
            ScheduledOp(
                Producer2DOp, tile_counts=(tiles_m, tiles_n),
                dim_names={"batch": 0, "seq": 1}
            ),
            ScheduledOp(
                Consumer1DOp, tile_counts=(tiles_m,),
                dim_names={"batch": 0}
            ),
        ]

        config = MegakernelConfig(num_sms=2)
        kernel = Megakernel(ops, config=config)
        kernel.run()

        expected_consumer = torch.tensor([
            m * 100 + 7 for m in range(tiles_m)
        ], dtype=torch.int32, device="cuda")
        assert torch.equal(consumer_results, expected_consumer), (
            f"Consumer mismatch: expected {expected_consumer.tolist()}, got {consumer_results.tolist()}"
        )

    def test_three_op_chain(self):
        """
        Three-op chain: A(2D) -> B(1D) -> C(1D)

        Op A (2D): writes (tile_0 * 1000 + tile_1) to matrix[m * N + n]
        Op B (1D): reads matrix, writes m * 200 + 1 to buf[m]
        Op C (1D): reads buf[m], writes buf[m] * 3 to output[m]

        Expected: (m * 200 + 1) * 3
        """
        global _matrix_ptr, _buf_ptr, _output_ptr, _num_cols
        _matrix_ptr = 0
        _buf_ptr = 0
        _output_ptr = 0

        tiles_m = 4
        tiles_n = 4
        _num_cols = tiles_n

        class OpA(Op):

            INPUTS: ClassVar[List[str]] = []
            OUTPUTS: ClassVar[List[str]] = ["data"]

            @staticmethod
            def compute(page_ptr, op_config_ptr):
                tidx = cute.arch.thread_idx()[0]
                if tidx == Int32(0):
                    idx = tile_0 * Int32(_num_cols) + tile_1
                    st_global_i32(Int64(_matrix_ptr), idx, tile_0 * Int32(1000) + tile_1)

        class OpB(Op):

            INPUTS: ClassVar[List[str]] = ["data"]
            OUTPUTS: ClassVar[List[str]] = ["buf"]

            @staticmethod
            def compute(page_ptr, op_config_ptr):
                tidx = cute.arch.thread_idx()[0]
                if tidx == Int32(0):
                    # Read last value for this dim 0 to verify producers completed
                    _ = ld_global_i32(Int64(_matrix_ptr), tile_0 * Int32(_num_cols) + Int32(_num_cols - 1))
                    st_global_i32(Int64(_buf_ptr), tile_0, tile_0 * Int32(200) + Int32(1))

        class OpC(Op):

            INPUTS: ClassVar[List[str]] = ["buf"]
            OUTPUTS: ClassVar[List[str]] = []

            @staticmethod
            def compute(page_ptr, op_config_ptr):
                tidx = cute.arch.thread_idx()[0]
                if tidx == Int32(0):
                    val = ld_global_i32(Int64(_buf_ptr), tile_0)
                    st_global_i32(Int64(_output_ptr), tile_0, val * Int32(3))

        matrix = torch.zeros(tiles_m * tiles_n, dtype=torch.int32, device="cuda")
        buf = torch.zeros(tiles_m, dtype=torch.int32, device="cuda")
        output = torch.zeros(tiles_m, dtype=torch.int32, device="cuda")

        _matrix_ptr = matrix.data_ptr()
        _buf_ptr = buf.data_ptr()
        _output_ptr = output.data_ptr()

        ops = [
            ScheduledOp(OpA, tile_counts=(tiles_m, tiles_n), dim_names={"batch": 0, "seq": 1}),
            ScheduledOp(OpB, tile_counts=(tiles_m,), dim_names={"batch": 0}),
            ScheduledOp(OpC, tile_counts=(tiles_m,), dim_names={"batch": 0}),
        ]

        config = MegakernelConfig(num_sms=2)
        kernel = Megakernel(ops, config=config)
        kernel.run()

        # Expected: (m * 200 + 1) * 3
        expected = torch.tensor([
            (m * 200 + 1) * 3 for m in range(tiles_m)
        ], dtype=torch.int32, device="cuda")
        assert torch.equal(output, expected), (
            f"Expected {expected.tolist()}, got {output.tolist()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
