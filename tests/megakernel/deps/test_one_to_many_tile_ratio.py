# Copyright (c) 2025, Machete Authors
"""
Test: One-to-Many Dependency Pattern (GPU)

Dependency Pattern:
==================

This tests the case where a 1D producer feeds a 2D consumer.
The producer writes to shared data, and multiple consumer tiles
with the same M index but different N index read from the same data.

    Producer (M only)              Consumer (M x N)
    ┌───┬───┬───┬───┐            ┌───────────────────┐
    │ 0 │ 1 │ 2 │ 3 │  ───────>  │ (0,0)...(0,3)     │
    └───┴───┴───┴───┘            │ (1,0)...(1,3)     │
                                 │ (2,0)...(2,3)     │
                                 │ (3,0)...(3,3)     │
                                 └───────────────────┘

Barrier Mapping:
    - 4 barriers (indexed by M - the shared dimension)
    - Producer tile m signals barrier m
    - Consumer tiles (m, *) all wait on barrier m
    - expected = 1 per barrier

Test Strategy:
    Producer: writes value (tile_m + 1) to data[tile_m]
    Consumer: reads data[tile_m], writes data[tile_m] * (tile_n + 1) to output[tile_m * N + tile_n]
    Verify: output[m * N + n] = (m + 1) * (n + 1)
"""

import pytest
import torch
from typing import ClassVar, List

import cutlass.cute as cute
from cutlass import Int32, Int64

from machete.megakernel import Megakernel, MegakernelConfig, ScheduledOp
from machete.megakernel.ops import Op
from machete.megakernel.interpreter import st_global_i32, ld_global_i32


def is_hopper_available():
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


# =============================================================================
# Global result tensor pointers
# =============================================================================

_producer_data_ptr = 0
_consumer_result_ptr = 0
_num_cols = 4  # N dimension for output linearization


# =============================================================================
# Test Ops with Real GPU Operations
# =============================================================================


class Producer1DOp(Op):
    """
    Writes (tile_m + 1) to data[tile_m].

    This producer only iterates over M dimension.
    """
    NUM_INPUT_PAGES: ClassVar[int] = 0
    NUM_OUTPUT_PAGES: ClassVar[int] = 0
    INPUTS: ClassVar[List[str]] = []
    OUTPUTS: ClassVar[List[str]] = ["data"]

    @staticmethod
    def compute(
        page_ptr: Int32,
        tile_m: Int32, tile_n: Int32, tile_l: Int32,
        op_config_ptr: Int64,
    ) -> None:
        tidx = cute.arch.thread_idx()[0]
        if tidx == Int32(0):
            val = tile_m + Int32(1)
            st_global_i32(Int64(_producer_data_ptr), tile_m, val)


class Consumer2DOp(Op):
    """
    Reads data[tile_m], writes data[tile_m] * (tile_n + 1) to output[tile_m * N + tile_n].

    This consumer iterates over M x N dimensions.
    Multiple consumer tiles with the same M read from the same producer output.
    """
    NUM_INPUT_PAGES: ClassVar[int] = 0
    NUM_OUTPUT_PAGES: ClassVar[int] = 0
    INPUTS: ClassVar[List[str]] = ["data"]
    OUTPUTS: ClassVar[List[str]] = []

    @staticmethod
    def compute(
        page_ptr: Int32,
        tile_m: Int32, tile_n: Int32, tile_l: Int32,
        op_config_ptr: Int64,
    ) -> None:
        tidx = cute.arch.thread_idx()[0]
        if tidx == Int32(0):
            val = ld_global_i32(Int64(_producer_data_ptr), tile_m)
            out_idx = tile_m * Int32(_num_cols) + tile_n
            st_global_i32(Int64(_consumer_result_ptr), out_idx, val * (tile_n + Int32(1)))


# =============================================================================
# GPU Tests
# =============================================================================


@pytest.mark.skipif(not is_hopper_available(), reason="Hopper (SM90+) GPU required")
class TestOneToManyPatternGPU:
    """
    GPU test for one-to-many dependency: 1D producer -> 2D consumer.

    Diagram (M=4, N=4):
        Producer tiles: [0] [1] [2] [3]
                         │   │   │   │
                         ▼   ▼   ▼   ▼
        Barriers:       (0) (1) (2) (3)
                         │   │   │   │
                         ▼   ▼   ▼   ▼
        Consumer tiles (M x N):
        ┌─────────────────────────────────┐
        │ (0,0) (0,1) (0,2) (0,3) ← M=0   │
        │ (1,0) (1,1) (1,2) (1,3) ← M=1   │
        │ (2,0) (2,1) (2,2) (2,3) ← M=2   │
        │ (3,0) (3,1) (3,2) (3,3) ← M=3   │
        └─────────────────────────────────┘

    Producer m writes (m+1) to data[m].
    Consumer (m, n) reads data[m] and writes data[m] * (n+1) to output[m * N + n].
    Expected: output[m * N + n] = (m+1) * (n+1)
    """

    def test_basic_1d_to_2d(self):
        """
        Basic 1D producer (4) to 2D consumer (4x4) test.

        Producer writes: [1, 2, 3, 4]
        Consumer (m, n) computes: (m+1) * (n+1)
        """
        global _producer_data_ptr, _consumer_result_ptr, _num_cols
        tiles_m = 4
        tiles_n = 4
        _num_cols = tiles_n
        total_output = tiles_m * tiles_n

        producer_data = torch.zeros(tiles_m, dtype=torch.int32, device="cuda")
        consumer_results = torch.zeros(total_output, dtype=torch.int32, device="cuda")

        _producer_data_ptr = producer_data.data_ptr()
        _consumer_result_ptr = consumer_results.data_ptr()

        ops = [
            ScheduledOp(
                Producer1DOp, tiles_m=tiles_m,
                dim_names={"batch": "m"}
            ),
            ScheduledOp(
                Consumer2DOp, tiles_m=tiles_m, tiles_n=tiles_n,
                dim_names={"batch": "m", "seq": "n"}
            ),
        ]

        config = MegakernelConfig(num_sms=2)
        kernel = Megakernel(ops, config=config)
        kernel.run()

        # Verify producer data: [1, 2, 3, 4]
        expected_producer = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cuda")
        assert torch.equal(producer_data, expected_producer), (
            f"Producer mismatch: expected {expected_producer.tolist()}, got {producer_data.tolist()}"
        )

        # Verify consumer: output[m * N + n] = (m+1) * (n+1)
        expected_consumer = torch.tensor([
            (m + 1) * (n + 1)
            for m in range(tiles_m)
            for n in range(tiles_n)
        ], dtype=torch.int32, device="cuda")
        assert torch.equal(consumer_results, expected_consumer), (
            f"Consumer mismatch: expected {expected_consumer.tolist()}, got {consumer_results.tolist()}"
        )

    def test_larger_1d_to_2d(self):
        """
        Larger 1D producer (8) to 2D consumer (8x4) test.
        """
        global _producer_data_ptr, _consumer_result_ptr, _num_cols
        tiles_m = 8
        tiles_n = 4
        _num_cols = tiles_n
        total_output = tiles_m * tiles_n

        producer_data = torch.zeros(tiles_m, dtype=torch.int32, device="cuda")
        consumer_results = torch.zeros(total_output, dtype=torch.int32, device="cuda")

        _producer_data_ptr = producer_data.data_ptr()
        _consumer_result_ptr = consumer_results.data_ptr()

        ops = [
            ScheduledOp(
                Producer1DOp, tiles_m=tiles_m,
                dim_names={"batch": "m"}
            ),
            ScheduledOp(
                Consumer2DOp, tiles_m=tiles_m, tiles_n=tiles_n,
                dim_names={"batch": "m", "seq": "n"}
            ),
        ]

        config = MegakernelConfig(num_sms=2)
        kernel = Megakernel(ops, config=config)
        kernel.run()

        expected_consumer = torch.tensor([
            (m + 1) * (n + 1)
            for m in range(tiles_m)
            for n in range(tiles_n)
        ], dtype=torch.int32, device="cuda")
        assert torch.equal(consumer_results, expected_consumer), (
            f"Consumer mismatch at some indices"
        )

    def test_three_op_chain(self):
        """
        Three-op chain: A(1D) -> B(1D) -> C(2D)

        Op A (1D): writes (tile_m + 1) to buf1[tile_m]
        Op B (1D): reads buf1[m], writes buf1[m] * 2 to buf2[m]
        Op C (2D): reads buf2[m], writes buf2[m] * (n+1) to buf3[m * N + n]

        Expected: (m + 1) * 2 * (n + 1)
        """
        global _buf1_ptr, _buf2_ptr, _buf3_ptr, _num_cols
        _buf1_ptr = 0
        _buf2_ptr = 0
        _buf3_ptr = 0

        tiles_m = 4
        tiles_n = 4
        _num_cols = tiles_n
        total_output = tiles_m * tiles_n

        class OpA(Op):
            NUM_INPUT_PAGES: ClassVar[int] = 0
            NUM_OUTPUT_PAGES: ClassVar[int] = 0
            INPUTS: ClassVar[List[str]] = []
            OUTPUTS: ClassVar[List[str]] = ["buf1"]

            @staticmethod
            def compute(page_ptr, tile_m, tile_n, tile_l, op_config_ptr):
                tidx = cute.arch.thread_idx()[0]
                if tidx == Int32(0):
                    st_global_i32(Int64(_buf1_ptr), tile_m, tile_m + Int32(1))

        class OpB(Op):
            NUM_INPUT_PAGES: ClassVar[int] = 0
            NUM_OUTPUT_PAGES: ClassVar[int] = 0
            INPUTS: ClassVar[List[str]] = ["buf1"]
            OUTPUTS: ClassVar[List[str]] = ["buf2"]

            @staticmethod
            def compute(page_ptr, tile_m, tile_n, tile_l, op_config_ptr):
                tidx = cute.arch.thread_idx()[0]
                if tidx == Int32(0):
                    val = ld_global_i32(Int64(_buf1_ptr), tile_m)
                    st_global_i32(Int64(_buf2_ptr), tile_m, val * Int32(2))

        class OpC(Op):
            NUM_INPUT_PAGES: ClassVar[int] = 0
            NUM_OUTPUT_PAGES: ClassVar[int] = 0
            INPUTS: ClassVar[List[str]] = ["buf2"]
            OUTPUTS: ClassVar[List[str]] = []

            @staticmethod
            def compute(page_ptr, tile_m, tile_n, tile_l, op_config_ptr):
                tidx = cute.arch.thread_idx()[0]
                if tidx == Int32(0):
                    val = ld_global_i32(Int64(_buf2_ptr), tile_m)
                    out_idx = tile_m * Int32(_num_cols) + tile_n
                    st_global_i32(Int64(_buf3_ptr), out_idx, val * (tile_n + Int32(1)))

        buf1 = torch.zeros(tiles_m, dtype=torch.int32, device="cuda")
        buf2 = torch.zeros(tiles_m, dtype=torch.int32, device="cuda")
        buf3 = torch.zeros(total_output, dtype=torch.int32, device="cuda")

        _buf1_ptr = buf1.data_ptr()
        _buf2_ptr = buf2.data_ptr()
        _buf3_ptr = buf3.data_ptr()

        ops = [
            ScheduledOp(OpA, tiles_m=tiles_m, dim_names={"batch": "m"}),
            ScheduledOp(OpB, tiles_m=tiles_m, dim_names={"batch": "m"}),
            ScheduledOp(OpC, tiles_m=tiles_m, tiles_n=tiles_n, dim_names={"batch": "m", "seq": "n"}),
        ]

        config = MegakernelConfig(num_sms=2)
        kernel = Megakernel(ops, config=config)
        kernel.run()

        # Expected: (m + 1) * 2 * (n + 1)
        expected = torch.tensor([
            (m + 1) * 2 * (n + 1)
            for m in range(tiles_m)
            for n in range(tiles_n)
        ], dtype=torch.int32, device="cuda")
        assert torch.equal(buf3, expected), (
            f"Expected {expected.tolist()}, got {buf3.tolist()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
