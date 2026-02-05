#!/usr/bin/env python
# Copyright (c) 2025, Machete Authors
"""Extract traces for all megakernel dependency pattern tests.

Produces .nanotrace and .perfetto files that can be opened in trace viewers
to inspect per-SM tile timelines and barrier synchronization.

Usage:
    python scripts/trace_deps.py
    python scripts/trace_deps.py --output-dir traces/deps
    python scripts/trace_deps.py --test one_to_one  # run specific test

Output formats:
    - .nanotrace: Native cutedsl-trace format
    - .perfetto: Open at https://ui.perfetto.dev/
"""

import argparse
import os
import sys
from typing import ClassVar, List

import torch
import cutlass.cute as cute
from cutlass import Int32, Int64

from machete.megakernel import Megakernel, MegakernelConfig, ScheduledOp
from machete.megakernel.ops import Op
from machete.megakernel.interpreter import st_global_i32, ld_global_i32


def is_blackwell_available():
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 10


# =============================================================================
# Global pointers (set per-test)
# =============================================================================
_ptr_a = 0
_ptr_b = 0
_ptr_c = 0
_ptr_d = 0
_ptr_e = 0
_num_cols = 4


def _write_perfetto(kernel, output_path: str):
    """Write perfetto format trace from a traced kernel."""
    from cutedsl_trace import TraceWriter
    kernel._trace_builder.copy_to_host()
    writer = TraceWriter("megakernel")
    writer.set_block_type(kernel._trace_block_type)
    writer.add_tensor(kernel._trace_builder)
    for tt in kernel._trace_types.values():
        writer.register_trace_type(tt)
    writer.write_perfetto(output_path)
    print(f"  Written: {output_path}")


# =============================================================================
# Test: One-to-One Same Size
# =============================================================================

def trace_one_to_one(output_dir: str):
    """1:1 dependency pattern with same tile dimensions."""
    global _ptr_a, _ptr_b

    class ProducerOp(Op):
        NUM_INPUT_PAGES: ClassVar[int] = 0
        NUM_OUTPUT_PAGES: ClassVar[int] = 0
        INPUTS: ClassVar[List[str]] = []
        OUTPUTS: ClassVar[List[str]] = ["data"]

        @staticmethod
        def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
            tidx = cute.arch.thread_idx()[0]
            if tidx == Int32(0):
                st_global_i32(Int64(_ptr_a), tile_m, tile_m + Int32(1))

    class ConsumerOp(Op):
        NUM_INPUT_PAGES: ClassVar[int] = 0
        NUM_OUTPUT_PAGES: ClassVar[int] = 0
        INPUTS: ClassVar[List[str]] = ["data"]
        OUTPUTS: ClassVar[List[str]] = []

        @staticmethod
        def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
            tidx = cute.arch.thread_idx()[0]
            if tidx == Int32(0):
                val = ld_global_i32(Int64(_ptr_a), tile_m)
                st_global_i32(Int64(_ptr_b), tile_m, val * Int32(2))

    num_tiles = 8
    buf_a = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
    buf_b = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
    _ptr_a = buf_a.data_ptr()
    _ptr_b = buf_b.data_ptr()

    ops = [
        ScheduledOp(ProducerOp, tiles_m=num_tiles, dim_names={"batch": "m"}),
        ScheduledOp(ConsumerOp, tiles_m=num_tiles, dim_names={"batch": "m"}),
    ]

    config = MegakernelConfig(num_sms=2, tracing=True)
    kernel = Megakernel(ops, config=config)
    kernel.run()

    trace_path = os.path.join(output_dir, "one_to_one.nanotrace")
    kernel.write_trace(trace_path)
    print(f"  Written: {trace_path}")

    # Also write perfetto format
    perfetto_path = os.path.join(output_dir, "one_to_one.perfetto")
    _write_perfetto(kernel, perfetto_path)

    # Verify correctness
    expected = torch.tensor([(i + 1) * 2 for i in range(num_tiles)], dtype=torch.int32, device="cuda")
    assert torch.equal(buf_b, expected), f"Test failed: {buf_b.tolist()} != {expected.tolist()}"
    print("  Verified: PASS")


# =============================================================================
# Test: One-to-Many (1D producer -> 2D consumer)
# =============================================================================

def trace_one_to_many(output_dir: str):
    """1:many dependency - consumer has extra dimensions."""
    global _ptr_a, _ptr_b, _num_cols

    class ProducerOp(Op):
        NUM_INPUT_PAGES: ClassVar[int] = 0
        NUM_OUTPUT_PAGES: ClassVar[int] = 0
        INPUTS: ClassVar[List[str]] = []
        OUTPUTS: ClassVar[List[str]] = ["data"]

        @staticmethod
        def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
            tidx = cute.arch.thread_idx()[0]
            if tidx == Int32(0):
                st_global_i32(Int64(_ptr_a), tile_m, tile_m + Int32(1))

    class ConsumerOp(Op):
        NUM_INPUT_PAGES: ClassVar[int] = 0
        NUM_OUTPUT_PAGES: ClassVar[int] = 0
        INPUTS: ClassVar[List[str]] = ["data"]
        OUTPUTS: ClassVar[List[str]] = []

        @staticmethod
        def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
            tidx = cute.arch.thread_idx()[0]
            if tidx == Int32(0):
                val = ld_global_i32(Int64(_ptr_a), tile_m)
                out_idx = tile_m * Int32(_num_cols) + tile_n
                st_global_i32(Int64(_ptr_b), out_idx, val * (tile_n + Int32(1)))

    tiles_m, tiles_n = 4, 4
    _num_cols = tiles_n
    buf_a = torch.zeros(tiles_m, dtype=torch.int32, device="cuda")
    buf_b = torch.zeros(tiles_m * tiles_n, dtype=torch.int32, device="cuda")
    _ptr_a = buf_a.data_ptr()
    _ptr_b = buf_b.data_ptr()

    ops = [
        ScheduledOp(ProducerOp, tiles_m=tiles_m, dim_names={"batch": "m"}),
        ScheduledOp(ConsumerOp, tiles_m=tiles_m, tiles_n=tiles_n, dim_names={"batch": "m", "seq": "n"}),
    ]

    config = MegakernelConfig(num_sms=2, tracing=True)
    kernel = Megakernel(ops, config=config)
    kernel.run()

    trace_path = os.path.join(output_dir, "one_to_many.nanotrace")
    kernel.write_trace(trace_path)
    print(f"  Written: {trace_path}")
    _write_perfetto(kernel, os.path.join(output_dir, "one_to_many.perfetto"))

    # Verify
    expected = torch.tensor([(m + 1) * (n + 1) for m in range(tiles_m) for n in range(tiles_n)],
                           dtype=torch.int32, device="cuda")
    assert torch.equal(buf_b, expected), f"Test failed"
    print("  Verified: PASS")


# =============================================================================
# Test: Many-to-One (2D producer -> 1D consumer)
# =============================================================================

def trace_many_to_one(output_dir: str):
    """Many:1 dependency - producer has extra dimensions."""
    global _ptr_a, _ptr_b, _num_cols

    class ProducerOp(Op):
        NUM_INPUT_PAGES: ClassVar[int] = 0
        NUM_OUTPUT_PAGES: ClassVar[int] = 0
        INPUTS: ClassVar[List[str]] = []
        OUTPUTS: ClassVar[List[str]] = ["data"]

        @staticmethod
        def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
            tidx = cute.arch.thread_idx()[0]
            if tidx == Int32(0):
                idx = tile_m * Int32(_num_cols) + tile_n
                st_global_i32(Int64(_ptr_a), idx, tile_m * Int32(10) + tile_n)

    class ConsumerOp(Op):
        NUM_INPUT_PAGES: ClassVar[int] = 0
        NUM_OUTPUT_PAGES: ClassVar[int] = 0
        INPUTS: ClassVar[List[str]] = ["data"]
        OUTPUTS: ClassVar[List[str]] = []

        @staticmethod
        def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
            tidx = cute.arch.thread_idx()[0]
            if tidx == Int32(0):
                last_idx = tile_m * Int32(_num_cols) + Int32(_num_cols - 1)
                val = ld_global_i32(Int64(_ptr_a), last_idx)
                st_global_i32(Int64(_ptr_b), tile_m, val)

    tiles_m, tiles_n = 4, 4
    _num_cols = tiles_n
    buf_a = torch.zeros(tiles_m * tiles_n, dtype=torch.int32, device="cuda")
    buf_b = torch.zeros(tiles_m, dtype=torch.int32, device="cuda")
    _ptr_a = buf_a.data_ptr()
    _ptr_b = buf_b.data_ptr()

    ops = [
        ScheduledOp(ProducerOp, tiles_m=tiles_m, tiles_n=tiles_n, dim_names={"batch": "m", "seq": "n"}),
        ScheduledOp(ConsumerOp, tiles_m=tiles_m, dim_names={"batch": "m"}),
    ]

    config = MegakernelConfig(num_sms=2, tracing=True)
    kernel = Megakernel(ops, config=config)
    kernel.run()

    trace_path = os.path.join(output_dir, "many_to_one.nanotrace")
    kernel.write_trace(trace_path)
    print(f"  Written: {trace_path}")
    _write_perfetto(kernel, os.path.join(output_dir, "many_to_one.perfetto"))

    # Verify: consumer reads last column value for each row
    expected = torch.tensor([m * 10 + (tiles_n - 1) for m in range(tiles_m)], dtype=torch.int32, device="cuda")
    assert torch.equal(buf_b, expected), f"Test failed"
    print("  Verified: PASS")


# =============================================================================
# Test: Chain Mixed Sizes (2D -> 1D -> 2D)
# =============================================================================

def trace_chain_mixed(output_dir: str):
    """Chain with mixed dimensions: 2D -> 1D -> 2D."""
    global _ptr_a, _ptr_b, _ptr_c, _num_cols

    class OpA(Op):
        NUM_INPUT_PAGES: ClassVar[int] = 0
        NUM_OUTPUT_PAGES: ClassVar[int] = 0
        INPUTS: ClassVar[List[str]] = []
        OUTPUTS: ClassVar[List[str]] = ["matrix"]

        @staticmethod
        def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
            tidx = cute.arch.thread_idx()[0]
            if tidx == Int32(0):
                idx = tile_m * Int32(_num_cols) + tile_n
                st_global_i32(Int64(_ptr_a), idx, tile_m * Int32(1000) + tile_n)

    class OpB(Op):
        NUM_INPUT_PAGES: ClassVar[int] = 0
        NUM_OUTPUT_PAGES: ClassVar[int] = 0
        INPUTS: ClassVar[List[str]] = ["matrix"]
        OUTPUTS: ClassVar[List[str]] = ["buf"]

        @staticmethod
        def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
            tidx = cute.arch.thread_idx()[0]
            if tidx == Int32(0):
                _ = ld_global_i32(Int64(_ptr_a), tile_m * Int32(_num_cols) + Int32(_num_cols - 1))
                st_global_i32(Int64(_ptr_b), tile_m, tile_m * Int32(100) + Int32(1))

    class OpC(Op):
        NUM_INPUT_PAGES: ClassVar[int] = 0
        NUM_OUTPUT_PAGES: ClassVar[int] = 0
        INPUTS: ClassVar[List[str]] = ["buf"]
        OUTPUTS: ClassVar[List[str]] = []

        @staticmethod
        def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
            tidx = cute.arch.thread_idx()[0]
            if tidx == Int32(0):
                val = ld_global_i32(Int64(_ptr_b), tile_m)
                out_idx = tile_m * Int32(_num_cols) + tile_n
                st_global_i32(Int64(_ptr_c), out_idx, val * (tile_n + Int32(1)))

    tiles_m, tiles_n = 4, 4
    _num_cols = tiles_n
    buf_a = torch.zeros(tiles_m * tiles_n, dtype=torch.int32, device="cuda")
    buf_b = torch.zeros(tiles_m, dtype=torch.int32, device="cuda")
    buf_c = torch.zeros(tiles_m * tiles_n, dtype=torch.int32, device="cuda")
    _ptr_a = buf_a.data_ptr()
    _ptr_b = buf_b.data_ptr()
    _ptr_c = buf_c.data_ptr()

    ops = [
        ScheduledOp(OpA, tiles_m=tiles_m, tiles_n=tiles_n, dim_names={"batch": "m", "seq": "n"}),
        ScheduledOp(OpB, tiles_m=tiles_m, dim_names={"batch": "m"}),
        ScheduledOp(OpC, tiles_m=tiles_m, tiles_n=tiles_n, dim_names={"batch": "m", "seq": "n"}),
    ]

    config = MegakernelConfig(num_sms=2, tracing=True)
    kernel = Megakernel(ops, config=config)
    kernel.run()

    trace_path = os.path.join(output_dir, "chain_mixed.nanotrace")
    kernel.write_trace(trace_path)
    print(f"  Written: {trace_path}")
    _write_perfetto(kernel, os.path.join(output_dir, "chain_mixed.perfetto"))

    # Verify: (m * 100 + 1) * (n + 1)
    expected = torch.tensor([(m * 100 + 1) * (n + 1) for m in range(tiles_m) for n in range(tiles_n)],
                           dtype=torch.int32, device="cuda")
    assert torch.equal(buf_c, expected), f"Test failed"
    print("  Verified: PASS")


# =============================================================================
# Test: Diamond Dependency (A -> B,C -> D)
# =============================================================================

def trace_diamond(output_dir: str):
    """Diamond pattern: fork-join with parallel branches."""
    global _ptr_a, _ptr_b, _ptr_c, _ptr_d

    class OpA(Op):
        NUM_INPUT_PAGES: ClassVar[int] = 0
        NUM_OUTPUT_PAGES: ClassVar[int] = 0
        INPUTS: ClassVar[List[str]] = []
        OUTPUTS: ClassVar[List[str]] = ["data"]

        @staticmethod
        def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
            tidx = cute.arch.thread_idx()[0]
            if tidx == Int32(0):
                st_global_i32(Int64(_ptr_a), tile_m, tile_m + Int32(1))

    class OpB(Op):
        NUM_INPUT_PAGES: ClassVar[int] = 0
        NUM_OUTPUT_PAGES: ClassVar[int] = 0
        INPUTS: ClassVar[List[str]] = ["data"]
        OUTPUTS: ClassVar[List[str]] = ["buf_b"]

        @staticmethod
        def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
            tidx = cute.arch.thread_idx()[0]
            if tidx == Int32(0):
                val = ld_global_i32(Int64(_ptr_a), tile_m)
                st_global_i32(Int64(_ptr_b), tile_m, val * Int32(2))

    class OpC(Op):
        NUM_INPUT_PAGES: ClassVar[int] = 0
        NUM_OUTPUT_PAGES: ClassVar[int] = 0
        INPUTS: ClassVar[List[str]] = ["data"]
        OUTPUTS: ClassVar[List[str]] = ["buf_c"]

        @staticmethod
        def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
            tidx = cute.arch.thread_idx()[0]
            if tidx == Int32(0):
                val = ld_global_i32(Int64(_ptr_a), tile_m)
                st_global_i32(Int64(_ptr_c), tile_m, val * Int32(3))

    class OpD(Op):
        NUM_INPUT_PAGES: ClassVar[int] = 0
        NUM_OUTPUT_PAGES: ClassVar[int] = 0
        INPUTS: ClassVar[List[str]] = ["buf_b", "buf_c"]
        OUTPUTS: ClassVar[List[str]] = []

        @staticmethod
        def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
            tidx = cute.arch.thread_idx()[0]
            if tidx == Int32(0):
                b_val = ld_global_i32(Int64(_ptr_b), tile_m)
                c_val = ld_global_i32(Int64(_ptr_c), tile_m)
                st_global_i32(Int64(_ptr_d), tile_m, b_val + c_val)

    num_tiles = 4
    buf_a = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
    buf_b = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
    buf_c = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
    buf_d = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
    _ptr_a = buf_a.data_ptr()
    _ptr_b = buf_b.data_ptr()
    _ptr_c = buf_c.data_ptr()
    _ptr_d = buf_d.data_ptr()

    ops = [
        ScheduledOp(OpA, tiles_m=num_tiles, dim_names={"batch": "m"}),
        ScheduledOp(OpB, tiles_m=num_tiles, dim_names={"batch": "m"}),
        ScheduledOp(OpC, tiles_m=num_tiles, dim_names={"batch": "m"}),
        ScheduledOp(OpD, tiles_m=num_tiles, dim_names={"batch": "m"}),
    ]

    config = MegakernelConfig(num_sms=2, tracing=True)
    kernel = Megakernel(ops, config=config)
    kernel.run()

    trace_path = os.path.join(output_dir, "diamond.nanotrace")
    kernel.write_trace(trace_path)
    print(f"  Written: {trace_path}")
    _write_perfetto(kernel, os.path.join(output_dir, "diamond.perfetto"))

    # Verify: (m+1)*2 + (m+1)*3 = (m+1)*5
    expected = torch.tensor([(m + 1) * 5 for m in range(num_tiles)], dtype=torch.int32, device="cuda")
    assert torch.equal(buf_d, expected), f"Test failed"
    print("  Verified: PASS")


# =============================================================================
# Main
# =============================================================================

TESTS = {
    "one_to_one": trace_one_to_one,
    "one_to_many": trace_one_to_many,
    "many_to_one": trace_many_to_one,
    "chain_mixed": trace_chain_mixed,
    "diamond": trace_diamond,
}


def main():
    parser = argparse.ArgumentParser(description="Extract traces for megakernel dependency tests")
    parser.add_argument("--output-dir", "-o", type=str, default="traces/deps",
                       help="Output directory for trace files")
    parser.add_argument("--test", "-t", type=str, choices=list(TESTS.keys()),
                       help="Run specific test (default: all)")
    args = parser.parse_args()

    if not is_blackwell_available():
        print("ERROR: Blackwell GPU required (SM100+)")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    print()

    tests_to_run = [args.test] if args.test else list(TESTS.keys())

    for test_name in tests_to_run:
        print(f"Running: {test_name}")
        try:
            TESTS[test_name](args.output_dir)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        print()

    print("=" * 50)
    print("Trace extraction complete!")
    print()
    print("Generated formats:")
    print("  - .nanotrace: Can be viewed with nanotrace-compatible viewers")
    print("  - .perfetto: Can be opened at https://ui.perfetto.dev/")


if __name__ == "__main__":
    main()
