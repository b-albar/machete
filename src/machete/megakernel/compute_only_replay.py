# Copyright (c) 2026, Machete Authors
"""Compute-only replay loop for graphs without load/store/communicate phases."""

from __future__ import annotations

import cutlass.cute as cute
from cutlass import Int32, Int64, const_expr, range_constexpr

from .scheduling import TileInstruction


OP_META_HANDLER_IDX = 0
OP_META_COMPUTE_LOCAL_IDX = 1
OP_META_WAIT_COUNT = 2
OP_META_COMPUTE_WAIT_COUNT = 3
OP_META_SIGNAL_COUNT = 4
OP_META_WAIT_ACQUIRE = 5
OP_META_ORIGIN_0 = 6
OP_META_ORIGIN_1 = 7
OP_META_ORIGIN_2 = 8
OP_META_ORIGIN_3 = 9
OP_META_PAGE_COUNT = 10
OP_META_STRIDE = 11


def build_compute_only_op_metadata_entry(
    *,
    handler_idx: int,
    compute_local_idx: int,
    wait_count: int,
    compute_wait_count: int,
    signal_count: int,
    wait_acquire: int,
    origins,
    page_count: int,
) -> list[int]:
    """Build one compact compute-only op metadata record."""
    return [
        int(handler_idx),
        int(compute_local_idx),
        int(wait_count),
        int(compute_wait_count),
        int(signal_count),
        int(wait_acquire),
        *(int(origin) for origin in origins),
        int(page_count),
    ]


def compute_only_op_meta_exec_globals() -> dict[str, int]:
    """Return metadata indices consumed by the compute-only replay loop.

    The keys retain the generic generated-kernel names, but only fields read by
    the compute-only replay loop are present.
    """
    return {
        "_OP_META_ORIGIN_0": OP_META_ORIGIN_0,
        "_OP_META_ORIGIN_1": OP_META_ORIGIN_1,
        "_OP_META_ORIGIN_2": OP_META_ORIGIN_2,
        "_OP_META_ORIGIN_3": OP_META_ORIGIN_3,
        "_OP_META_HANDLER_IDX": OP_META_HANDLER_IDX,
        "_OP_META_COMPUTE_LOCAL_IDX": OP_META_COMPUTE_LOCAL_IDX,
        "_OP_META_WAIT_COUNT": OP_META_WAIT_COUNT,
        "_OP_META_COMPUTE_WAIT_COUNT": OP_META_COMPUTE_WAIT_COUNT,
        "_OP_META_SIGNAL_COUNT": OP_META_SIGNAL_COUNT,
        "_OP_META_WAIT_ACQUIRE": OP_META_WAIT_ACQUIRE,
        "_OP_META_PAGE_COUNT": OP_META_PAGE_COUNT,
        "_OP_META_STRIDE": OP_META_STRIDE,
    }


def build_compute_only_kernel_loop(kernel, kernel_cfg, runtime):
    """Build a lighter replay loop for graphs with compute phases only."""
    tracing = bool(kernel.config.tracing)
    dispatch_compute = runtime["dispatch_compute"]
    signal_barriers = runtime["signal_barriers"]
    max_waits = runtime["max_waits"]
    max_compute_waits = runtime["max_compute_waits"]
    max_signal_formulas = kernel._max_signal_formulas
    num_mma_warps = kernel_cfg["num_mma_warps"]
    num_compute_threads = kernel_cfg["num_compute_threads"]
    mma_reg_count = kernel_cfg["mma_reg_count"]
    relaxed_global_barriers = bool(kernel.config.relaxed_global_barriers)
    global_barrier_sleep_ns = int(kernel.config.global_barrier_sleep_ns)

    @cute.jit
    def _kernel_loop_compute_only(
        instructions_ptr: Int64,
        barriers_ptr: Int64,
        op_configs_ptr: Int64,
        op_meta_ptr: Int64,
        signal_meta_ptr: Int64,
        num_instructions: Int32,
        tidx: Int32,
        block_id: Int32,
        num_blocks: Int32,
        smem_base: Int32,
        trace_buffer_ptr: Int64,
        wait_info_ptr: Int64,
        compute_wait_info_ptr: Int64,
    ):
        warp_id = tidx // Int32(32)
        lane_id = tidx % Int32(32)
        if const_expr(tracing):
            _trace_buf = cute.make_tensor(
                cute.make_ptr(cute.Uint8, trace_buffer_ptr),
                cute.make_layout(1 << 24),
            )
            _mma_lane = begin_lane_dynamic_raw(
                Int32(4),
                Int32(trace_row_stride),
                block_id,
                Int32(1),
                (warp_id == Int32(0)) & (lane_id == Int32(0)),
            )

        if warp_id < Int32(num_mma_warps):
            setmaxregister_increase(mma_reg_count)

        iq_base = smem_base + Int32(iq_offset)
        _compute_page0 = _get_page_ptr(smem_base, Int32(0))
        if const_expr(MAX_REQUESTED_N > 1):
            _compute_page_table = smem_base + Int32(page_addr_table_offset)
            if warp_id == Int32(0) and lane_id == Int32(0):
                for _pi in range_constexpr(MAX_REQUESTED_N):
                    st_shared_i32(
                        _compute_page_table + Int32(_pi * 4),
                        _get_page_ptr(smem_base, Int32(_pi)),
                    )
            named_barrier_sync(Int32(1), Int32(num_compute_threads))
        _fetch_idx = block_id
        _fetch_limit = num_instructions
        _fetch_stride = num_blocks

        _cached_wait_count = Int32(0)
        _cached_wait_acquire = Int32(0)
        _cached_signal_count = Int32(0)
        _cached_compute_wait_count = Int32(0)
        _cached_wait_barrier = Int32(-2)
        _cached_wait_expected = Int32(-1)
        _cached_compute_wait_barrier = Int32(-2)
        _cached_compute_wait_expected = Int32(-1)
        _running = Int32(1)
        _cached_op_idx = Int32(TileInstruction.END_MARKER)
        _cached_handler = Int32(0)
        _cached_config_idx = Int32(-2)
        _cached_config = Int64(0)

        while _running == Int32(1):
            if warp_id == Int32(0) and lane_id == Int32(0):
                _instr_op = Int32(TileInstruction.END_MARKER)
                if _fetch_idx < _fetch_limit:
                    load_instruction_to_smem(instructions_ptr, _fetch_idx, iq_base)
                    _instr_word0 = ld_shared_i32(iq_base)
                    _instr_op = _instr_word0 & Int32(65535)
                    if _instr_op == Int32(65535):
                        _instr_op = Int32(TileInstruction.END_MARKER)
                    if _instr_op == Int32(TileInstruction.END_MARKER):
                        _fetch_idx = _fetch_limit
                    else:
                        _next_fetch_idx = _fetch_idx + _fetch_stride
                        if _next_fetch_idx < _fetch_limit:
                            prefetch_instruction(instructions_ptr, _next_fetch_idx)
                        _fetch_idx = _fetch_idx + _fetch_stride
                else:
                    st_shared_i32(iq_base, Int32(65535))
            named_barrier_sync(Int32(1), Int32(num_compute_threads))

            _op_word0 = ld_shared_i32(iq_base)
            op_idx = _op_word0 & Int32(65535)
            if op_idx == Int32(65535):
                op_idx = Int32(TileInstruction.END_MARKER)
            if op_idx == Int32(TileInstruction.END_MARKER):
                _running = Int32(0)

            if op_idx != Int32(TileInstruction.END_MARKER):
                _config = Int64(0)
                _compute_local = Int32(0)
                _meta_base = _op_meta_base(op_idx)
                _op_meta_changed = op_idx != _cached_op_idx
                if _op_meta_changed:
                    _cached_handler = _op_meta_i32_base(
                        op_meta_ptr, _meta_base, Int32(_OP_META_HANDLER_IDX)
                    )
                    _cached_signal_count = _op_meta_i32_base(
                        op_meta_ptr, _meta_base, Int32(_OP_META_SIGNAL_COUNT)
                    )
                    _cached_op_idx = op_idx
                _handler = _cached_handler
                _barrier_meta_idx = ld_shared_i32(iq_base + Int32(4 * _INSTR_BARRIER_META_IDX))
                if op_idx != _cached_config_idx:
                    _cached_config = ld_global_i64(op_configs_ptr, op_idx)
                    _cached_config_idx = op_idx
                _config = _cached_config
                if const_expr(dispatch_compute_uses_handler_local_idx):
                    _compute_local = _op_meta_i32_base(
                        op_meta_ptr, _meta_base, Int32(_OP_META_COMPUTE_LOCAL_IDX)
                    )

                if warp_id == Int32(0) and lane_id == Int32(0):
                    if _op_meta_changed:
                        _cached_wait_count = _op_meta_i32_base(
                            op_meta_ptr, _meta_base, Int32(_OP_META_WAIT_COUNT)
                        )
                        _cached_wait_acquire = _op_meta_i32_base(
                            op_meta_ptr, _meta_base, Int32(_OP_META_WAIT_ACQUIRE)
                        )
                        if const_expr(max_compute_waits > 0):
                            _cached_compute_wait_count = _op_meta_i32_base(
                                op_meta_ptr,
                                _meta_base,
                                Int32(_OP_META_COMPUTE_WAIT_COUNT),
                            )

                    if _cached_wait_count > Int32(0):
                        _done_waits = Int32(0)
                        for _w in range_constexpr(max_waits):
                            if _done_waits == Int32(0):
                                if _w < _cached_wait_count:
                                    _wi_off = (
                                        _barrier_meta_idx * Int32(max_waits * 2)
                                        + Int32(_w * 2)
                                    )
                                    _bar_idx = ld_global_i32(wait_info_ptr, _wi_off)
                                    if _bar_idx >= Int32(0):
                                        _bar_exp = ld_global_i32(wait_info_ptr, _wi_off + Int32(1))
                                        if (
                                            _bar_idx != _cached_wait_barrier
                                            or _bar_exp != _cached_wait_expected
                                        ):
                                            if const_expr(relaxed_global_barriers):
                                                if _cached_wait_acquire != Int32(0):
                                                    global_barrier_wait(barriers_ptr, _bar_idx, _bar_exp)
                                                else:
                                                    global_barrier_wait_relaxed(
                                                        barriers_ptr,
                                                        _bar_idx,
                                                        _bar_exp,
                                                        Int32(global_barrier_sleep_ns),
                                                    )
                                            else:
                                                global_barrier_wait(barriers_ptr, _bar_idx, _bar_exp)
                                            _cached_wait_barrier = _bar_idx
                                            _cached_wait_expected = _bar_exp
                                    else:
                                        _done_waits = Int32(1)
                                else:
                                    _done_waits = Int32(1)
                    if const_expr(max_compute_waits > 0):
                        if _cached_compute_wait_count > Int32(0):
                            _done_compute_waits = Int32(0)
                            for _cw in range_constexpr(max_compute_waits):
                                if _done_compute_waits == Int32(0):
                                    if _cw < _cached_compute_wait_count:
                                        _cwi_off = (
                                            _barrier_meta_idx * Int32(max_compute_waits * 2)
                                            + Int32(_cw * 2)
                                        )
                                        _cbar_idx = ld_global_i32(compute_wait_info_ptr, _cwi_off)
                                        if _cbar_idx >= Int32(0):
                                            _cbar_exp = ld_global_i32(compute_wait_info_ptr, _cwi_off + Int32(1))
                                            if (
                                                _cbar_idx != _cached_compute_wait_barrier
                                                or _cbar_exp != _cached_compute_wait_expected
                                            ):
                                                if const_expr(relaxed_global_barriers):
                                                    if _cached_wait_acquire != Int32(0):
                                                        global_barrier_wait(barriers_ptr, _cbar_idx, _cbar_exp)
                                                    else:
                                                        global_barrier_wait_relaxed(
                                                            barriers_ptr,
                                                            _cbar_idx,
                                                            _cbar_exp,
                                                            Int32(global_barrier_sleep_ns),
                                                        )
                                                else:
                                                    global_barrier_wait(barriers_ptr, _cbar_idx, _cbar_exp)
                                                _cached_compute_wait_barrier = _cbar_idx
                                                _cached_compute_wait_expected = _cbar_exp
                                        else:
                                            _done_compute_waits = Int32(1)
                                    else:
                                        _done_compute_waits = Int32(1)

                named_barrier_sync(Int32(1), Int32(num_compute_threads))

                _tile_01 = ld_shared_i32(iq_base + Int32(4 * _INSTR_TILE_01))
                _tile_23 = ld_shared_i32(iq_base + Int32(4 * _INSTR_TILE_23))
                tile_0 = _tile_01 & Int32(65535)
                tile_1 = (_tile_01 >> Int32(16)) & Int32(65535)
                tile_2 = _tile_23 & Int32(65535)
                tile_3 = (_tile_23 >> Int32(16)) & Int32(65535)
                origin_0 = _op_meta_i32_base(op_meta_ptr, _meta_base, Int32(_OP_META_ORIGIN_0))
                origin_1 = _op_meta_i32_base(op_meta_ptr, _meta_base, Int32(_OP_META_ORIGIN_1))
                origin_2 = _op_meta_i32_base(op_meta_ptr, _meta_base, Int32(_OP_META_ORIGIN_2))
                origin_3 = _op_meta_i32_base(op_meta_ptr, _meta_base, Int32(_OP_META_ORIGIN_3))
                page_ptr = _compute_page0
                if const_expr(MAX_REQUESTED_N > 1):
                    _page_count = _op_meta_i32_base(
                        op_meta_ptr, _meta_base, Int32(_OP_META_PAGE_COUNT)
                    )
                    if _page_count > Int32(1):
                        page_ptr = _compute_page_table
                _range_pos = Int32(0)
                _range_end = Int32(0)
                _range_stride = Int32(1)
                _range_offset = Int32(0)
                _range_axis = Int32(-1)
                _range_meta = (_op_word0 >> Int32(16)) & Int32(65535)
                _range_axis = (
                    _range_meta % Int32(16)
                ) - Int32(1)
                if _range_axis == Int32(0):
                    _range_pos = tile_0
                if _range_axis == Int32(1):
                    _range_pos = tile_1
                if _range_axis == Int32(2):
                    _range_pos = tile_2
                if _range_axis == Int32(3):
                    _range_pos = tile_3
                _range_end = ld_shared_i32(
                    iq_base + Int32(4 * _INSTR_RANGE_END)
                ) & Int32(65535)
                if _range_axis < Int32(0) or _range_end <= _range_pos:
                    _range_end = _range_pos + Int32(1)
                    _range_stride = Int32(1)

                while _range_pos < _range_end:
                    _current_meta_idx = _barrier_meta_idx + _range_offset
                    if (
                        _range_axis >= Int32(0)
                        and warp_id == Int32(0)
                        and lane_id == Int32(0)
                    ):
                        if _cached_wait_count > Int32(0):
                            _done_waits = Int32(0)
                            for _w in range_constexpr(max_waits):
                                if _done_waits == Int32(0):
                                    if _w < _cached_wait_count:
                                        _wi_off = (
                                            _current_meta_idx * Int32(max_waits * 2)
                                            + Int32(_w * 2)
                                        )
                                        _bar_idx = ld_global_i32(wait_info_ptr, _wi_off)
                                        if _bar_idx >= Int32(0):
                                            _bar_exp = ld_global_i32(wait_info_ptr, _wi_off + Int32(1))
                                            if (
                                                _bar_idx != _cached_wait_barrier
                                                or _bar_exp != _cached_wait_expected
                                            ):
                                                if const_expr(relaxed_global_barriers):
                                                    if _cached_wait_acquire != Int32(0):
                                                        global_barrier_wait(barriers_ptr, _bar_idx, _bar_exp)
                                                    else:
                                                        global_barrier_wait_relaxed(
                                                            barriers_ptr,
                                                            _bar_idx,
                                                            _bar_exp,
                                                            Int32(global_barrier_sleep_ns),
                                                        )
                                                else:
                                                    global_barrier_wait(barriers_ptr, _bar_idx, _bar_exp)
                                                _cached_wait_barrier = _bar_idx
                                                _cached_wait_expected = _bar_exp
                                        else:
                                            _done_waits = Int32(1)
                                    else:
                                        _done_waits = Int32(1)
                                else:
                                    _done_waits = Int32(1)
                    if (
                        const_expr(max_compute_waits > 0)
                        and _range_axis >= Int32(0)
                        and warp_id == Int32(0)
                        and lane_id == Int32(0)
                    ):
                        if _cached_compute_wait_count > Int32(0):
                            _done_compute_waits = Int32(0)
                            for _cw in range_constexpr(max_compute_waits):
                                if _done_compute_waits == Int32(0):
                                    if _cw < _cached_compute_wait_count:
                                        _cwi_off = (
                                            _current_meta_idx * Int32(max_compute_waits * 2)
                                            + Int32(_cw * 2)
                                        )
                                        _cbar_idx = ld_global_i32(compute_wait_info_ptr, _cwi_off)
                                        if _cbar_idx >= Int32(0):
                                            _cbar_exp = ld_global_i32(compute_wait_info_ptr, _cwi_off + Int32(1))
                                            if (
                                                _cbar_idx != _cached_compute_wait_barrier
                                                or _cbar_exp != _cached_compute_wait_expected
                                            ):
                                                if const_expr(relaxed_global_barriers):
                                                    if _cached_wait_acquire != Int32(0):
                                                        global_barrier_wait(barriers_ptr, _cbar_idx, _cbar_exp)
                                                    else:
                                                        global_barrier_wait_relaxed(
                                                            barriers_ptr,
                                                            _cbar_idx,
                                                            _cbar_exp,
                                                            Int32(global_barrier_sleep_ns),
                                                        )
                                                else:
                                                    global_barrier_wait(barriers_ptr, _cbar_idx, _cbar_exp)
                                                _cached_compute_wait_barrier = _cbar_idx
                                                _cached_compute_wait_expected = _cbar_exp
                                        else:
                                            _done_compute_waits = Int32(1)
                                    else:
                                        _done_compute_waits = Int32(1)
                    if _range_axis >= Int32(0):
                        named_barrier_sync(Int32(1), Int32(num_compute_threads))

                    if _range_axis == Int32(0):
                        tile_0 = _range_pos
                    if _range_axis == Int32(1):
                        tile_1 = _range_pos
                    if _range_axis == Int32(2):
                        tile_2 = _range_pos
                    if _range_axis == Int32(3):
                        tile_3 = _range_pos
                    dispatch_tile_0 = tile_0 + origin_0
                    dispatch_tile_1 = tile_1 + origin_1
                    dispatch_tile_2 = tile_2 + origin_2
                    dispatch_tile_3 = tile_3 + origin_3
                    if const_expr(dispatch_compute_uses_handler_local_idx):
                        if const_expr(tracing):
                            _tc = trace_start()
                        dispatch_compute(
                            _handler,
                            _compute_local,
                            page_ptr,
                            dispatch_tile_0,
                            dispatch_tile_1,
                            dispatch_tile_2,
                            dispatch_tile_3,
                            _config,
                        )
                        if const_expr(tracing):
                            _mma_lane = end_event_dynamic_raw_1(
                                _tc,
                                _trace_buf,
                                Int32(trace_row_stride),
                                _mma_lane,
                                ld_global_i32(trace_compute_fmt_ptr, op_idx),
                                op_idx,
                            )
                    else:
                        if const_expr(tracing):
                            _tc = trace_start()
                        dispatch_compute(
                            _handler,
                            page_ptr,
                            dispatch_tile_0,
                            dispatch_tile_1,
                            dispatch_tile_2,
                            dispatch_tile_3,
                            _config,
                        )
                        if const_expr(tracing):
                            _mma_lane = end_event_dynamic_raw_1(
                                _tc,
                                _trace_buf,
                                Int32(trace_row_stride),
                                _mma_lane,
                                ld_global_i32(trace_compute_fmt_ptr, op_idx),
                                op_idx,
                            )
                    named_barrier_sync(Int32(1), Int32(num_compute_threads))
                    if _range_axis >= Int32(0):
                        if _cached_signal_count > Int32(0):
                            global_memory_fence_gpu()
                            named_barrier_sync(Int32(1), Int32(num_compute_threads))
                            if warp_id == Int32(0) and lane_id == Int32(0):
                                if _cached_signal_count == Int32(1):
                                    _sig_barrier = ld_global_i32(
                                        signal_meta_ptr,
                                        _current_meta_idx * Int32(max_signal_formulas),
                                    )
                                    if _sig_barrier >= Int32(0):
                                        global_barrier_signal_gpu(barriers_ptr, _sig_barrier)
                                else:
                                    signal_barriers(
                                        signal_meta_ptr,
                                        _current_meta_idx,
                                        _cached_signal_count,
                                        barriers_ptr,
                                    )
                    _range_offset = _range_offset + Int32(1)
                    _range_pos = _range_pos + _range_stride

                if _range_axis == Int32(0):
                    tile_0 = _tile_01 & Int32(65535)
                if _range_axis == Int32(1):
                    tile_1 = (_tile_01 >> Int32(16)) & Int32(65535)
                if _range_axis == Int32(2):
                    tile_2 = _tile_23 & Int32(65535)
                if _range_axis == Int32(3):
                    tile_3 = (_tile_23 >> Int32(16)) & Int32(65535)

                if _range_axis < Int32(0):
                    if _cached_signal_count > Int32(0):
                        global_memory_fence_gpu()
                        named_barrier_sync(Int32(1), Int32(num_compute_threads))
                        if warp_id == Int32(0) and lane_id == Int32(0):
                            if _cached_signal_count == Int32(1):
                                _sig_barrier = ld_global_i32(
                                    signal_meta_ptr,
                                    _barrier_meta_idx * Int32(max_signal_formulas),
                                )
                                if _sig_barrier >= Int32(0):
                                    global_barrier_signal_gpu(barriers_ptr, _sig_barrier)
                            else:
                                signal_barriers(
                                    signal_meta_ptr,
                                    _barrier_meta_idx,
                                    _cached_signal_count,
                                    barriers_ptr,
                                )
            named_barrier_sync(Int32(1), Int32(num_compute_threads))
        if const_expr(tracing):
            finish_lane_dynamic_raw(_trace_buf, _mma_lane)

    return _kernel_loop_compute_only
