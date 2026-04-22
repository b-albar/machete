# Copyright (c) 2025, Machete Authors
"""
Test: Tile Size Ratios on Shared Dimensions

Tests that the barrier formula correctly handles cases where two ops share
a dimension but use different tile sizes.  This exercises the divisor logic
(p_divs / c_divs) and the non-divisible detection in _resolve_named_formulas.

Cases:
  1. Divisible ratio (2:1) — producer fine-grained, consumer coarse.
  2. Divisible ratio (1:2) — producer coarse, consumer fine-grained.
  3. Non-divisible (4 vs 3) — M collapsed to single barrier group.
  4. Divisible ratio (3:1) — 3 producer tiles per consumer tile.
  5. 2D producer with M ratio → 1D consumer (ratio + many-to-one).
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


# Global pointers for inter-op communication
_data_ptr = 0
_result_ptr = 0

# Configurable constants (set before kernel launch)
_read_ratio = 2   # How many producer tiles per consumer tile
_read_count = 5   # Total producer tiles (for ReadAllOp)
_ncols = 4        # 2D column count


# =============================================================================
# Test Ops
# =============================================================================


class WriteOp(Op):
    """Writes (tile_0 + 1) to data[tile_0]."""
    INPUTS: ClassVar[List[str]] = []
    OUTPUTS: ClassVar[List[str]] = ["data"]

    @cute.jit
    def compute(self, page_ptr, tile_0):
        tidx = cute.arch.thread_idx()[0]
        if tidx == Int32(0):
            st_global_i32(Int64(_data_ptr), tile_0, tile_0 + Int32(1))


class ReadSumPair(Op):
    """Consumer: sums data[tile_0*2] + data[tile_0*2+1] → result[tile_0]."""
    INPUTS: ClassVar[List[str]] = ["data"]
    OUTPUTS: ClassVar[List[str]] = []

    @cute.jit
    def compute(self, page_ptr, tile_0):
        tidx = cute.arch.thread_idx()[0]
        if tidx == Int32(0):
            base = tile_0 * Int32(2)
            a = ld_global_i32(Int64(_data_ptr), base)
            b = ld_global_i32(Int64(_data_ptr), base + Int32(1))
            st_global_i32(Int64(_result_ptr), tile_0, a + b)


class ReadSumTriple(Op):
    """Consumer: sums data[tile_0*3..tile_0*3+2] → result[tile_0]."""
    INPUTS: ClassVar[List[str]] = ["data"]
    OUTPUTS: ClassVar[List[str]] = []

    @cute.jit
    def compute(self, page_ptr, tile_0):
        tidx = cute.arch.thread_idx()[0]
        if tidx == Int32(0):
            base = tile_0 * Int32(3)
            a = ld_global_i32(Int64(_data_ptr), base)
            b = ld_global_i32(Int64(_data_ptr), base + Int32(1))
            c = ld_global_i32(Int64(_data_ptr), base + Int32(2))
            st_global_i32(Int64(_result_ptr), tile_0, a + b + c)


class ReadHalf(Op):
    """Consumer: reads data[tile_0 // 2] → result[tile_0].

    Used when consumer is finer-grained than producer (1:2 ratio).
    Two consumer tiles per producer tile read the same value.
    """
    INPUTS: ClassVar[List[str]] = ["data"]
    OUTPUTS: ClassVar[List[str]] = []

    @cute.jit
    def compute(self, page_ptr, tile_0):
        tidx = cute.arch.thread_idx()[0]
        if tidx == Int32(0):
            src = tile_0 // Int32(2)
            val = ld_global_i32(Int64(_data_ptr), src)
            st_global_i32(Int64(_result_ptr), tile_0, val)


class ReadAll5(Op):
    """Consumer: sums all 5 data elements → result[tile_0].

    For non-divisible case: must wait for ALL producer tiles.
    """
    INPUTS: ClassVar[List[str]] = ["data"]
    OUTPUTS: ClassVar[List[str]] = []

    @cute.jit
    def compute(self, page_ptr, tile_0):
        tidx = cute.arch.thread_idx()[0]
        if tidx == Int32(0):
            total = Int32(0)
            total = total + ld_global_i32(Int64(_data_ptr), Int32(0))
            total = total + ld_global_i32(Int64(_data_ptr), Int32(1))
            total = total + ld_global_i32(Int64(_data_ptr), Int32(2))
            total = total + ld_global_i32(Int64(_data_ptr), Int32(3))
            total = total + ld_global_i32(Int64(_data_ptr), Int32(4))
            st_global_i32(Int64(_result_ptr), tile_0, total)


class Write2DOp(Op):
    """Writes (tile_0 * 100 + tile_1 + 1) to data[tile_0 * N + tile_1]."""
    INPUTS: ClassVar[List[str]] = []
    OUTPUTS: ClassVar[List[str]] = ["data"]

    @cute.jit
    def compute(self, page_ptr, tile_0, tile_1):
        tidx = cute.arch.thread_idx()[0]
        if tidx == Int32(0):
            idx = tile_0 * Int32(_ncols) + tile_1
            st_global_i32(Int64(_data_ptr), idx, tile_0 * Int32(100) + tile_1 + Int32(1))


class ReadTwoRows(Op):
    """Consumer: sums 2 rows of _ncols elements each.

    Reads data[tile_0*2*N .. tile_0*2*N + 2*N - 1] → result[tile_0].
    Used for testing many-to-one with M ratio=2:1.
    """
    INPUTS: ClassVar[List[str]] = ["data"]
    OUTPUTS: ClassVar[List[str]] = []

    @cute.jit
    def compute(self, page_ptr, tile_0):
        tidx = cute.arch.thread_idx()[0]
        if tidx == Int32(0):
            total = Int32(0)
            # Read 2 rows
            base = tile_0 * Int32(2) * Int32(_ncols)
            # Row 0
            total = total + ld_global_i32(Int64(_data_ptr), base + Int32(0))
            total = total + ld_global_i32(Int64(_data_ptr), base + Int32(1))
            total = total + ld_global_i32(Int64(_data_ptr), base + Int32(2))
            total = total + ld_global_i32(Int64(_data_ptr), base + Int32(3))
            # Row 1
            base2 = base + Int32(_ncols)
            total = total + ld_global_i32(Int64(_data_ptr), base2 + Int32(0))
            total = total + ld_global_i32(Int64(_data_ptr), base2 + Int32(1))
            total = total + ld_global_i32(Int64(_data_ptr), base2 + Int32(2))
            total = total + ld_global_i32(Int64(_data_ptr), base2 + Int32(3))
            st_global_i32(Int64(_result_ptr), tile_0, total)


# =============================================================================
# GPU Tests
# =============================================================================


@pytest.mark.skipif(not is_hopper_available(), reason="Hopper (SM90+) GPU required")
class TestTileSizeRatios:
    """Tests for barrier formulas with tile size ratios on shared dims."""

    def test_divisible_2_to_1(self):
        """Producer (8 tiles, tile_size=2) → consumer (4 tiles, tile_size=4).

        Two producer tiles signal each consumer barrier.
        Consumer sums pairs: data[2k] + data[2k+1].
        """
        global _data_ptr, _result_ptr

        data = torch.zeros(8, dtype=torch.int32, device="cuda")
        result = torch.zeros(4, dtype=torch.int32, device="cuda")
        _data_ptr = data.data_ptr()
        _result_ptr = result.data_ptr()

        ops = [
            ScheduledOp(
                WriteOp, tile_counts=(8,),
                dim_names={"M": 0}, tile_sizes={"M": 2},
            ),
            ScheduledOp(
                ReadSumPair, tile_counts=(4,),
                dim_names={"M": 0}, tile_sizes={"M": 4},
            ),
        ]

        kernel = Megakernel(ops, config=MegakernelConfig(num_sms=2))
        kernel.run()

        assert torch.equal(data, torch.arange(1, 9, dtype=torch.int32, device="cuda"))
        # tile k: (2k+1) + (2k+2) = 4k+3
        expected = torch.tensor([3, 7, 11, 15], dtype=torch.int32, device="cuda")
        assert torch.equal(result, expected), (
            f"expected {expected.tolist()}, got {result.tolist()}"
        )

    def test_divisible_1_to_2(self):
        """Producer (4 tiles, tile_size=4) → consumer (8 tiles, tile_size=2).

        One producer tile covers 2 consumer tiles.
        Consumer reads data[tile_0 // 2].
        """
        global _data_ptr, _result_ptr

        data = torch.zeros(4, dtype=torch.int32, device="cuda")
        result = torch.zeros(8, dtype=torch.int32, device="cuda")
        _data_ptr = data.data_ptr()
        _result_ptr = result.data_ptr()

        ops = [
            ScheduledOp(
                WriteOp, tile_counts=(4,),
                dim_names={"M": 0}, tile_sizes={"M": 4},
            ),
            ScheduledOp(
                ReadHalf, tile_counts=(8,),
                dim_names={"M": 0}, tile_sizes={"M": 2},
            ),
        ]

        kernel = Megakernel(ops, config=MegakernelConfig(num_sms=2))
        kernel.run()

        assert torch.equal(data, torch.arange(1, 5, dtype=torch.int32, device="cuda"))
        # tile k reads data[k // 2] = k//2 + 1
        expected = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4], dtype=torch.int32, device="cuda")
        assert torch.equal(result, expected), (
            f"expected {expected.tolist()}, got {result.tolist()}"
        )

    def test_divisible_3_to_1(self):
        """Producer (9 tiles, tile_size=1) → consumer (3 tiles, tile_size=3).

        Three producer tiles signal each consumer barrier.
        Consumer sums triples: data[3k] + data[3k+1] + data[3k+2].
        """
        global _data_ptr, _result_ptr

        data = torch.zeros(9, dtype=torch.int32, device="cuda")
        result = torch.zeros(3, dtype=torch.int32, device="cuda")
        _data_ptr = data.data_ptr()
        _result_ptr = result.data_ptr()

        ops = [
            ScheduledOp(
                WriteOp, tile_counts=(9,),
                dim_names={"M": 0}, tile_sizes={"M": 1},
            ),
            ScheduledOp(
                ReadSumTriple, tile_counts=(3,),
                dim_names={"M": 0}, tile_sizes={"M": 3},
            ),
        ]

        kernel = Megakernel(ops, config=MegakernelConfig(num_sms=2))
        kernel.run()

        assert torch.equal(data, torch.arange(1, 10, dtype=torch.int32, device="cuda"))
        # tile k: (3k+1)+(3k+2)+(3k+3) = 9k+6
        expected = torch.tensor([6, 15, 24], dtype=torch.int32, device="cuda")
        assert torch.equal(result, expected), (
            f"expected {expected.tolist()}, got {result.tolist()}"
        )

    def test_non_divisible_falls_back_to_many_to_one(self):
        """Producer (5 tiles, tile_size=4) → consumer (4 tiles, tile_size=3).

        4 and 3 don't divide evenly → falls back to many-to-one barrier
        (all producer tiles signal same barrier). Conservative but correct.
        """
        global _data_ptr, _result_ptr

        data = torch.zeros(5, dtype=torch.int32, device="cuda")
        result = torch.zeros(4, dtype=torch.int32, device="cuda")
        _data_ptr = data.data_ptr()
        _result_ptr = result.data_ptr()

        ops = [
            ScheduledOp(
                WriteOp, tile_counts=(5,),
                dim_names={"M": 0}, tile_sizes={"M": 4},
            ),
            ScheduledOp(
                ReadAll5, tile_counts=(4,),
                dim_names={"M": 0}, tile_sizes={"M": 3},
            ),
        ]

        # Should not raise — falls back to many-to-one barriers
        kernel = Megakernel(ops, config=MegakernelConfig(num_sms=2))
        kernel.run()
        torch.cuda.synchronize()

    def test_ratio_with_extra_dim(self):
        """2D producer (8M × 4N) → 1D consumer (4M), tile_size ratio 2:1 on M.

        Producer has 8 M tiles (tile_size=2), consumer has 4 M tiles (tile_size=4).
        N is producer-only (many-to-one). Each consumer barrier waits for
        2 (ratio) × 4 (N) = 8 producer signals.
        """
        global _data_ptr, _result_ptr, _ncols
        _ncols = 4

        data = torch.zeros(8 * 4, dtype=torch.int32, device="cuda")
        result = torch.zeros(4, dtype=torch.int32, device="cuda")
        _data_ptr = data.data_ptr()
        _result_ptr = result.data_ptr()

        ops = [
            ScheduledOp(
                Write2DOp, tile_counts=(8, 4),
                dim_names={"M": 0, "N": 1}, tile_sizes={"M": 2},
            ),
            ScheduledOp(
                ReadTwoRows, tile_counts=(4,),
                dim_names={"M": 0}, tile_sizes={"M": 4},
            ),
        ]

        kernel = Megakernel(ops, config=MegakernelConfig(num_sms=2))
        kernel.run()

        # Consumer tile k reads rows 2k and 2k+1:
        # Row r col c: r*100 + c + 1
        for k in range(4):
            expected_sum = 0
            for r in range(2 * k, 2 * k + 2):
                for c in range(4):
                    expected_sum += r * 100 + c + 1
            assert result[k].item() == expected_sum, (
                f"tile {k}: expected {expected_sum}, got {result[k].item()}"
            )
