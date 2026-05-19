# Copyright (c) 2026, Machete Authors
"""Replay loop builders for persistent megakernels."""

from __future__ import annotations

from typing import Any, Dict

import cutlass.cute as cute
from cutlass import Int32, Int64, const_expr, range_constexpr

from .scheduling import TileInstruction



def build_ring_kernel_loop(kernel, kernel_cfg: Dict[str, Any], runtime: Dict[str, Any]):
    """Build the warp-specialized ring-buffer loop used by the persistent kernel.

    Returns a flat function whose body contains all dispatch_load/compute/store
    calls directly, so that _build_kernel_loop_source's regex can inject
    tensor and TMA parameters into every call site.
    """
    # Unpack into locals so they are captured by the closure below.
    # _build_kernel_loop_source extracts the function body as source text
    # and uses a regex to append tensor/TMA params to dispatch_* calls.
    # This only works when dispatch calls appear in the body itself (not
    # hidden inside nested @cute.jit sub-functions).
    def _kernel_loop_ring(
        instructions_ptr: Int64,
        barriers_ptr: Int64,
        op_configs_ptr: Int64,
        op_meta_ptr: Int64,
        signal_meta_ptr: Int64,
        peer_signal_ptr: Int64,
        num_instructions: Int32,
        tidx: Int32,
        block_id: Int32,
        num_blocks: Int32,
        smem_base: Int32,
        trace_buffer_ptr: Int64,
        wait_info_ptr: Int64,
        compute_wait_info_ptr: Int64,
    ) -> None:
        """Warp-specialized ring buffer loop.

        Warp roles (NUM_DMA_WARPS=3, W = num_mma_warps):
          Warps 0..W-1:  MMA warps — compute from ring buffer pages
          Warp W:        Controller — fetches instructions, pre-computed barrier wait
          Warp W+1:      Loader — dispatches TMA loads
          Warp W+2:      Store — waits for compute_done, dispatches TMA stores

        Mbarrier phases alternate 0/1 with each use (hardware auto-reset).
        """
        warp_id = tidx // Int32(32)
        lane_id = tidx % Int32(32)

        is_store_warp = warp_id == Int32(num_mma_warps + 2)

        if warp_id >= Int32(num_mma_warps):
            setmaxregister_decrease(dma_reg_count)
        if warp_id < Int32(num_mma_warps):
            setmaxregister_increase(mma_reg_count)

        iq_base = smem_base + Int32(iq_offset)
        flags_ptr = smem_base + Int32(flags_offset)

        if const_expr(tracing):
            _trace_buf = cute.make_tensor(
                cute.make_ptr(cute.Uint8, trace_buffer_ptr),
                cute.make_layout(1 << 24),
            )

        # ========== INIT (controller warp thread 0) ==========
        if warp_id == Int32(num_mma_warps):
            with cute.arch.elect_one():
                st_shared_i32(flags_ptr + FLAG_DISPATCH_LOAD, Int32(-1))
                st_shared_i32(flags_ptr + FLAG_PRODUCE_IDX, Int32(0))
                st_shared_i32(flags_ptr + FLAG_STORE_IDX, Int32(0))
                st_shared_i32(flags_ptr + FLAG_LOAD_DONE, Int32(0))
                st_shared_i32(flags_ptr + FLAG_DATA_RELEASE_IDX, Int32(0))
                st_shared_i32(flags_ptr + FLAG_DATA_PRODUCE_IDX, Int32(0))
                for _ip in range(num_slots):
                    _slot_ti = smem_base + Int32(ring_state_offset) + Int32(_ip) * Int32(tile_info_bytes)
                    st_shared_i32(_slot_ti, Int32(-1))
                    mbarrier_init(
                        _work_notify_mbar(smem_base, Int32(_ip)),
                        Int32(1),
                    )
                    mbarrier_init(
                        _compute_done_mbar(smem_base, Int32(_ip)),
                        Int32(num_mma_warps),
                    )
                mbarrier_init_fence()

        named_barrier_sync(Int32(0), Int32(threads_per_block))

        # ========== CONTROLLER WARP (fetch + pre-computed barrier wait) ==========
        if warp_id == Int32(num_mma_warps):
            if const_expr(tracing):
                _ctrl_lane = begin_lane_dynamic_raw(
                    Int32(4),
                    Int32(trace_row_stride),
                    block_id,
                    Int32(3),
                    lane_id == Int32(0),
                )
            produce_idx = Int32(0)
            _fetch_idx = block_id
            _fetch_limit = num_instructions
            _fetch_stride = num_blocks
            _ctrl_done = Int32(0)
            _ctrl_cached_config = Int64(0)
            _ctrl_cached_config_idx = Int32(-2)
            _ctrl_cached_handler = Int32(0)
            _ctrl_cached_wait_count = Int32(0)
            _ctrl_cached_wait_acquire = Int32(0)
            _ctrl_cached_barrier_meta_idx = Int32(0)
            _ctrl_cached_wait_barrier = Int32(-2)
            _ctrl_cached_wait_expected = Int32(-1)
            _ctrl_cached_op = Int32(TileInstruction.END_MARKER)
            _ctrl_cached_phase_mask = Int32(0)
            _ctrl_cached_t0 = Int32(0)
            _ctrl_cached_t1 = Int32(0)
            _ctrl_cached_t2 = Int32(0)
            _ctrl_cached_t3 = Int32(0)
            _ctrl_range_axis = Int32(-1)
            _ctrl_range_pos = Int32(0)
            _ctrl_range_end = Int32(0)
            _ctrl_range_offset = Int32(0)
            _ctrl_range_stride = Int32(1)
            _ctrl_range_active = Int32(0)

            produce_idx_ptr = flags_ptr + FLAG_PRODUCE_IDX
            store_idx_ptr = flags_ptr + FLAG_STORE_IDX
            load_done_ptr = flags_ptr + FLAG_LOAD_DONE
            _temp_instr = iq_base

            while _ctrl_done == Int32(0):
                if lane_id == Int32(0):
                    _instr_op = Int32(TileInstruction.END_MARKER)
                    _instr_word0 = Int32(0)
                    if _ctrl_range_active != Int32(0):
                        _instr_op = _ctrl_cached_op
                    if _ctrl_range_active == Int32(0) and _fetch_idx < _fetch_limit:
                        load_instruction_to_smem(instructions_ptr, _fetch_idx, _temp_instr)
                        _instr_word0 = ld_shared_i32(_temp_instr)
                        _instr_op = _instr_word0 & Int32(65535)
                        if _instr_op == Int32(65535):
                            _instr_op = Int32(TileInstruction.END_MARKER)
                        if _instr_op == Int32(TileInstruction.END_MARKER):
                            _fetch_idx = _fetch_limit
                        if _instr_op != Int32(TileInstruction.END_MARKER):
                            _next_fetch_idx = _fetch_idx + _fetch_stride
                            if _next_fetch_idx < _fetch_limit:
                                prefetch_instruction(instructions_ptr, _next_fetch_idx)
                            _fetch_idx = _fetch_idx + _fetch_stride

                    if _instr_op >= Int32(0) and _ctrl_range_active == Int32(0):
                        _ctrl_meta_base = _op_meta_base(_instr_op)
                        _phase_mask = _ctrl_cached_phase_mask
                        if _instr_op != _ctrl_cached_op:
                            _ctrl_cached_op = _instr_op
                            _ctrl_cached_handler = _op_meta_i32_base(
                                op_meta_ptr, _ctrl_meta_base, Int32(_OP_META_HANDLER_IDX)
                            )
                            _ctrl_cached_wait_count = _op_meta_i32_base(
                                op_meta_ptr, _ctrl_meta_base, Int32(_OP_META_WAIT_COUNT)
                            )
                            _ctrl_cached_wait_acquire = _op_meta_i32_base(
                                op_meta_ptr, _ctrl_meta_base, Int32(_OP_META_WAIT_ACQUIRE)
                            )
                            _phase_mask = _op_meta_i32_base(
                                op_meta_ptr, _ctrl_meta_base, Int32(_OP_META_PHASE_MASK)
                            )
                            _ctrl_cached_phase_mask = _phase_mask
                        _ctrl_cached_barrier_meta_idx = ld_shared_i32(
                            _temp_instr + Int32(4 * _INSTR_BARRIER_META_IDX)
                        )
                        if _instr_op != _ctrl_cached_config_idx:
                            _ctrl_cached_config = ld_global_i64(op_configs_ptr, _instr_op)
                            _ctrl_cached_config_idx = _instr_op

                        _ctrl_tile_01 = ld_shared_i32(_temp_instr + Int32(4 * _INSTR_TILE_01))
                        _ctrl_tile_23 = ld_shared_i32(_temp_instr + Int32(4 * _INSTR_TILE_23))
                        _ctrl_cached_t0 = _ctrl_tile_01 & Int32(65535)
                        _ctrl_cached_t1 = (_ctrl_tile_01 >> Int32(16)) & Int32(65535)
                        _ctrl_cached_t2 = _ctrl_tile_23 & Int32(65535)
                        _ctrl_cached_t3 = (_ctrl_tile_23 >> Int32(16)) & Int32(65535)
                        _ctrl_range_stride = Int32(1)
                        _ctrl_range_offset = Int32(0)
                        _ctrl_range_meta = (_instr_word0 >> Int32(16)) & Int32(65535)
                        _ctrl_range_axis = Int32(-1)
                        _ctrl_range_pos = Int32(0)
                        _ctrl_range_end = Int32(1)
                        if _ctrl_range_meta != Int32(0):
                            _ctrl_range_axis = (_ctrl_range_meta % Int32(16)) - Int32(1)
                            if _ctrl_range_axis == Int32(0):
                                _ctrl_range_pos = _ctrl_cached_t0
                            if _ctrl_range_axis == Int32(1):
                                _ctrl_range_pos = _ctrl_cached_t1
                            if _ctrl_range_axis == Int32(2):
                                _ctrl_range_pos = _ctrl_cached_t2
                            if _ctrl_range_axis == Int32(3):
                                _ctrl_range_pos = _ctrl_cached_t3
                            _ctrl_range_end = ld_shared_i32(
                                _temp_instr + Int32(4 * _INSTR_RANGE_END)
                            ) & Int32(65535)
                        if _ctrl_range_axis < Int32(0) or _ctrl_range_end <= _ctrl_range_pos:
                            _ctrl_range_end = _ctrl_range_pos + Int32(1)
                            _ctrl_range_stride = Int32(1)
                        _ctrl_range_active = Int32(1)

                    if _instr_op >= Int32(0):
                        _ctrl_current_meta_idx = (
                            _ctrl_cached_barrier_meta_idx
                            + _ctrl_range_offset
                        )
                        if _ctrl_cached_wait_count > Int32(0):
                            _done_waits = Int32(0)
                            for _w in range_constexpr(max_waits):
                                if _done_waits == Int32(0):
                                    if _w < _ctrl_cached_wait_count:
                                        _wi_off = (
                                            _ctrl_current_meta_idx * Int32(max_waits * 2)
                                            + Int32(_w * 2)
                                        )
                                        _bar_idx = ld_global_i32(wait_info_ptr, _wi_off)
                                        if _bar_idx >= Int32(0):
                                            _bar_exp = ld_global_i32(wait_info_ptr, _wi_off + Int32(1))
                                            if (
                                                _bar_idx != _ctrl_cached_wait_barrier
                                                or _bar_exp != _ctrl_cached_wait_expected
                                            ):
                                                if const_expr(relaxed_global_barriers):
                                                    if const_expr(tracing):
                                                        _tdw = trace_start()
                                                    if _ctrl_cached_wait_acquire != Int32(0):
                                                        global_barrier_wait(barriers_ptr, _bar_idx, _bar_exp)
                                                    else:
                                                        global_barrier_wait_relaxed(
                                                            barriers_ptr,
                                                            _bar_idx,
                                                            _bar_exp,
                                                            Int32(global_barrier_sleep_ns),
                                                        )
                                                    if const_expr(tracing):
                                                        _ctrl_lane = end_event_dynamic_raw_1(
                                                            _tdw,
                                                            _trace_buf,
                                                            Int32(trace_row_stride),
                                                            _ctrl_lane,
                                                            Int32(trace_dep_wait_fmt),
                                                            _instr_op,
                                                        )
                                                else:
                                                    if const_expr(tracing):
                                                        _tdw = trace_start()
                                                    global_barrier_wait(barriers_ptr, _bar_idx, _bar_exp)
                                                    if const_expr(tracing):
                                                        _ctrl_lane = end_event_dynamic_raw_1(
                                                            _tdw,
                                                            _trace_buf,
                                                            Int32(trace_row_stride),
                                                            _ctrl_lane,
                                                            Int32(trace_dep_wait_fmt),
                                                            _instr_op,
                                                        )
                                                _ctrl_cached_wait_barrier = _bar_idx
                                                _ctrl_cached_wait_expected = _bar_exp
                                        else:
                                            _done_waits = Int32(1)
                                    else:
                                        _done_waits = Int32(1)

                        _si = ld_shared_i32(store_idx_ptr)
                        while (produce_idx - _si) >= Int32(num_slots):
                            _wait_slot = produce_idx % Int32(num_slots)
                            _wait_phase = ((produce_idx // Int32(num_slots)) + Int32(1)) % Int32(2)
                            if const_expr(tracing):
                                _trfw = trace_start()
                            mbarrier_wait(
                                _compute_done_mbar(smem_base, _wait_slot), _wait_phase
                            )
                            if const_expr(tracing):
                                _ctrl_lane = end_event_dynamic_raw_1(
                                    _trfw,
                                    _trace_buf,
                                    Int32(trace_row_stride),
                                    _ctrl_lane,
                                    Int32(trace_ring_full_wait_fmt),
                                    _instr_op,
                                )
                            _si = ld_shared_i32(store_idx_ptr)

                        _p_slot = produce_idx % Int32(num_slots)
                        _p_ti = smem_base + Int32(ring_state_offset) + _p_slot * Int32(tile_info_bytes)
                        st_shared_i32(
                            _p_ti + Int32(4 * _TILE_INFO_PAGE_ID),
                            _p_slot % Int32(num_pages),
                        )
                        _p_t0 = _ctrl_cached_t0
                        _p_t1 = _ctrl_cached_t1
                        _p_t2 = _ctrl_cached_t2
                        _p_t3 = _ctrl_cached_t3
                        if _ctrl_range_axis == Int32(0):
                            _p_t0 = _ctrl_range_pos
                        if _ctrl_range_axis == Int32(1):
                            _p_t1 = _ctrl_range_pos
                        if _ctrl_range_axis == Int32(2):
                            _p_t2 = _ctrl_range_pos
                        if _ctrl_range_axis == Int32(3):
                            _p_t3 = _ctrl_range_pos
                        st_shared_i32(_p_ti, _instr_op)
                        st_shared_i32(
                            _p_ti + Int32(4 * _TILE_INFO_HANDLER_IDX),
                            _ctrl_cached_handler,
                        )
                        st_shared_i32(_p_ti + Int32(4 * _TILE_INFO_TILE_0), _p_t0)
                        st_shared_i32(_p_ti + Int32(4 * _TILE_INFO_TILE_1), _p_t1)
                        st_shared_i32(_p_ti + Int32(4 * _TILE_INFO_TILE_2), _p_t2)
                        st_shared_i32(_p_ti + Int32(4 * _TILE_INFO_TILE_3), _p_t3)
                        st_shared_i32(
                            _p_ti + Int32(4 * _TILE_INFO_INSTRUCTION_IDX),
                            _ctrl_current_meta_idx,
                        )
                        st_shared_i64(
                            _p_ti + Int32(4 * _TILE_INFO_OP_CONFIG),
                            _ctrl_cached_config,
                        )
                        produce_idx = produce_idx + Int32(1)
                        st_shared_release_cta_i32(produce_idx_ptr, produce_idx)
                        _ctrl_range_offset = _ctrl_range_offset + Int32(1)
                        _ctrl_range_pos = _ctrl_range_pos + _ctrl_range_stride
                        if _ctrl_range_pos >= _ctrl_range_end:
                            _ctrl_range_active = Int32(0)

                    if (
                        _instr_op == Int32(TileInstruction.END_MARKER)
                        and _ctrl_range_active == Int32(0)
                    ):
                        if _fetch_idx >= _fetch_limit:
                            _store_idx_done = ld_shared_i32(store_idx_ptr)
                            if (produce_idx - _store_idx_done) < Int32(num_slots):
                                _sent = produce_idx % Int32(num_slots)
                                st_shared_i32(
                                    smem_base + Int32(ring_state_offset) + _sent * Int32(tile_info_bytes),
                                    Int32(TileInstruction.END_MARKER),
                                )
                                mbarrier_arrive(_work_notify_mbar(smem_base, _sent))
                                st_shared_i32(load_done_ptr, Int32(1))

                _ctrl_done = ld_shared_i32(load_done_ptr)
            if const_expr(tracing):
                finish_lane_dynamic_raw(_trace_buf, _ctrl_lane)

        # ========== LOADER WARP (TMA dispatch) ==========
        if warp_id == Int32(num_mma_warps + 1):
            if const_expr(tracing):
                _dma_lane = begin_lane_dynamic_raw(
                    Int32(4),
                    Int32(trace_row_stride),
                    block_id,
                    Int32(0),
                    lane_id == Int32(0),
                )

            _ldr_done = Int32(0)
            _ldr_load_done_ptr = flags_ptr + FLAG_LOAD_DONE
            _ldr_produce_ptr = flags_ptr + FLAG_PRODUCE_IDX
            _ldr_idx = Int32(0)

            while _ldr_done == Int32(0):
                _p_idx = ld_shared_acquire_cta_i32(_ldr_produce_ptr)
                if _ldr_idx < _p_idx:
                    _dl_slot = _ldr_idx % Int32(num_slots)
                    _dl_ti = smem_base + Int32(ring_state_offset) + _dl_slot * Int32(tile_info_bytes)
                    _dl_op = ld_shared_i32(_dl_ti)
                    if _dl_op != Int32(TileInstruction.END_MARKER):
                        _dl_meta_base = _op_meta_base(_dl_op)
                        _dl_handler = ld_shared_i32(_dl_ti + Int32(4 * _TILE_INFO_HANDLER_IDX))
                        _dl_handler_local = Int32(0)
                        if const_expr(dispatch_load_uses_handler_local_idx):
                            _dl_handler_local = _op_meta_i32_base(
                                op_meta_ptr, _dl_meta_base, Int32(_OP_META_LOAD_LOCAL_IDX)
                            )
                        _dl_0 = ld_shared_i32(_dl_ti + Int32(4 * _TILE_INFO_TILE_0))
                        _dl_1 = ld_shared_i32(_dl_ti + Int32(4 * _TILE_INFO_TILE_1))
                        _dl_2 = ld_shared_i32(_dl_ti + Int32(4 * _TILE_INFO_TILE_2))
                        _dl_3 = ld_shared_i32(_dl_ti + Int32(4 * _TILE_INFO_TILE_3))
                        _dl_config = ld_shared_i64(_dl_ti + Int32(4 * _TILE_INFO_OP_CONFIG))
                        _dl_mbar = _work_notify_mbar(smem_base, _dl_slot)
                        if const_expr(tracing):
                            _tl = trace_start()
                        if const_expr(has_page_free_ops):
                            _dl_page = ld_shared_i32(_dl_ti + Int32(4 * _TILE_INFO_PAGE_ID))
                            _dl_pp = _get_page_ptr(smem_base, _dl_page)
                        else:
                            _dl_pp = _get_page_ptr(smem_base, _dl_slot)
                        if const_expr(dispatch_load_uses_handler_local_idx):
                            dispatch_load(
                                _dl_handler,
                                _dl_handler_local,
                                _dl_pp,
                                _dl_0,
                                _dl_1,
                                _dl_2,
                                _dl_3,
                                _dl_config,
                                _dl_mbar,
                            )
                        else:
                            dispatch_load(
                                _dl_handler,
                                _dl_pp,
                                _dl_0,
                                _dl_1,
                                _dl_2,
                                _dl_3,
                                _dl_config,
                                _dl_mbar,
                            )
                        if const_expr(tracing):
                            _dma_lane = end_event_dynamic_raw_1(
                                _tl,
                                _trace_buf,
                                Int32(trace_row_stride),
                                _dma_lane,
                                ld_global_i32(trace_load_fmt_ptr, _dl_op),
                                _dl_op,
                            )
                    _ldr_idx = _ldr_idx + Int32(1)
                else:
                    nanosleep(Int32(loader_idle_sleep_ns))

                _ldr_done = ld_shared_i32(_ldr_load_done_ptr)
                if _ldr_done == Int32(1) and _ldr_idx < ld_shared_acquire_cta_i32(_ldr_produce_ptr):
                    _ldr_done = Int32(0)

            if const_expr(tracing):
                finish_lane_dynamic_raw(_trace_buf, _dma_lane)

        # ========== STORE WARP LOOP ==========
        if is_store_warp:
            if const_expr(tracing):
                _store_lane = begin_lane_dynamic_raw(
                    Int32(4),
                    Int32(trace_row_stride),
                    block_id,
                    Int32(2),
                    lane_id == Int32(0),
                )
            _sw_done = Int32(0)
            store_idx_ptr = flags_ptr + FLAG_STORE_IDX
            produce_idx_ptr = flags_ptr + FLAG_PRODUCE_IDX
            load_done_ptr = flags_ptr + FLAG_LOAD_DONE
            while _sw_done == Int32(0):
                _s_idx = ld_shared_i32(store_idx_ptr)
                _p_idx = ld_shared_i32(produce_idx_ptr)

                if _s_idx < _p_idx:
                    _s_slot = _s_idx % Int32(num_slots)
                    _s_phase = (_s_idx // Int32(num_slots)) % Int32(2)

                    _ds_ti = smem_base + Int32(ring_state_offset) + _s_slot * Int32(tile_info_bytes)
                    _ds_op = ld_shared_i32(_ds_ti)
                    _ds_meta_base = _op_meta_base(_ds_op)
                    _ds_handler = ld_shared_i32(_ds_ti + Int32(4 * _TILE_INFO_HANDLER_IDX))
                    _ds_signal_count = _op_meta_i32_base(
                        op_meta_ptr, _ds_meta_base, Int32(_OP_META_SIGNAL_COUNT)
                    )
                    _ds_handler_local = Int32(0)
                    if const_expr(dispatch_store_uses_handler_local_idx):
                        _ds_handler_local = _op_meta_i32_base(
                            op_meta_ptr, _ds_meta_base, Int32(_OP_META_STORE_LOCAL_IDX)
                        )
                    _dc_handler_local = Int32(0)
                    if const_expr(dispatch_communicate_uses_handler_local_idx):
                        _dc_handler_local = _op_meta_i32_base(
                            op_meta_ptr, _ds_meta_base, Int32(_OP_META_COMM_LOCAL_IDX)
                        )
                    _ds_0 = ld_shared_i32(_ds_ti + Int32(4 * _TILE_INFO_TILE_0))
                    _ds_1 = ld_shared_i32(_ds_ti + Int32(4 * _TILE_INFO_TILE_1))
                    _ds_2 = ld_shared_i32(_ds_ti + Int32(4 * _TILE_INFO_TILE_2))
                    _ds_3 = ld_shared_i32(_ds_ti + Int32(4 * _TILE_INFO_TILE_3))
                    _ds_instruction_idx = ld_shared_i32(
                        _ds_ti + Int32(4 * _TILE_INFO_INSTRUCTION_IDX)
                    )
                    _ds_config = ld_shared_i64(_ds_ti + Int32(4 * _TILE_INFO_OP_CONFIG))
                    if const_expr(has_page_free_ops):
                        _ds_page = ld_shared_i32(_ds_ti + Int32(4 * _TILE_INFO_PAGE_ID))
                    else:
                        _ds_page = _s_slot

                    if const_expr(tracing):
                        _tsw = trace_start()
                    mbarrier_wait(_compute_done_mbar(smem_base, _s_slot), _s_phase)
                    if const_expr(tracing):
                        _store_lane = end_event_dynamic_raw_1(
                            _tsw,
                            _trace_buf,
                            Int32(trace_row_stride),
                            _store_lane,
                            Int32(trace_compute_wait_fmt),
                            _ds_op,
                        )

                    if const_expr(tracing):
                        _tss = trace_start()
                    if const_expr(has_page_free_ops):
                        _ds_pp = _get_page_ptr(smem_base, _ds_page)
                    else:
                        _ds_pp = _get_page_ptr(smem_base, _s_slot)
                    if const_expr(dispatch_store_uses_handler_local_idx):
                        dispatch_store(
                            _ds_handler,
                            _ds_handler_local,
                            _ds_pp,
                            _ds_0,
                            _ds_1,
                            _ds_2,
                            _ds_3,
                            _ds_config,
                        )
                    else:
                        dispatch_store(
                            _ds_handler,
                            _ds_pp,
                            _ds_0,
                            _ds_1,
                            _ds_2,
                            _ds_3,
                            _ds_config,
                        )
                    if const_expr(has_communicate):
                        if const_expr(dispatch_communicate_uses_handler_local_idx):
                            dispatch_communicate(
                                _ds_handler,
                                _dc_handler_local,
                                _ds_pp,
                                _ds_0,
                                _ds_1,
                                _ds_2,
                                _ds_3,
                                _ds_config,
                            )
                        else:
                            dispatch_communicate(
                                _ds_handler,
                                _ds_pp,
                                _ds_0,
                                _ds_1,
                                _ds_2,
                                _ds_3,
                                _ds_config,
                            )
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                    if const_expr(tracing):
                        _store_lane = end_event_dynamic_raw_1(
                            _tss,
                            _trace_buf,
                            Int32(trace_row_stride),
                            _store_lane,
                            ld_global_i32(trace_store_fmt_ptr, _ds_op),
                            _ds_op,
                        )

                    with cute.arch.elect_one():
                        if _ds_signal_count > Int32(0):
                            if _ds_signal_count == Int32(1):
                                _sig_barrier = ld_global_i32(
                                    signal_meta_ptr,
                                    _ds_instruction_idx * Int32(max_signal_formulas),
                                )
                                if _sig_barrier >= Int32(0):
                                    global_barrier_signal_gpu(barriers_ptr, _sig_barrier)
                            else:
                                signal_barriers(
                                    signal_meta_ptr,
                                    _ds_instruction_idx,
                                    _ds_signal_count,
                                    barriers_ptr,
                                )
                        if const_expr(has_communicate):
                            _ds_lin = Int32(0)
                            _ds_s0 = _op_meta_i32_base(
                                op_meta_ptr, _ds_meta_base, Int32(_OP_META_STRIDE_0)
                            )
                            _ds_s1 = _op_meta_i32_base(
                                op_meta_ptr, _ds_meta_base, Int32(_OP_META_STRIDE_1)
                            )
                            _ds_s2 = _op_meta_i32_base(
                                op_meta_ptr, _ds_meta_base, Int32(_OP_META_STRIDE_2)
                            )
                            _ds_s3 = _op_meta_i32_base(
                                op_meta_ptr, _ds_meta_base, Int32(_OP_META_STRIDE_3)
                            )
                            _ds_lin = (
                                _ds_0 * _ds_s0
                                + _ds_1 * _ds_s1
                                + _ds_2 * _ds_s2
                                + _ds_3 * _ds_s3
                            )
                            signal_peer_barriers(
                                peer_signal_ptr,
                                _ds_op,
                                _ds_lin,
                                Int64(_peer_barriers_data_ptr),
                            )
                        st_shared_i32(store_idx_ptr, _s_idx + Int32(1))

                if _s_idx >= _p_idx:
                    if ld_shared_i32(load_done_ptr) == Int32(1):
                        _sw_done = Int32(1)
                    if ld_shared_i32(load_done_ptr) != Int32(1):
                        _sw_next_slot = _s_idx % Int32(num_slots)
                        _sw_next_phase = (_s_idx // Int32(num_slots)) % Int32(2)
                        mbarrier_wait(
                            _work_notify_mbar(smem_base, _sw_next_slot),
                            _sw_next_phase,
                        )

            if const_expr(tracing):
                finish_lane_dynamic_raw(_trace_buf, _store_lane)

        # ========== MMA WARP LOOP ==========
        if warp_id < Int32(num_mma_warps):
            if const_expr(tracing):
                _mma_lane = begin_lane_dynamic_raw(
                    Int32(4),
                    Int32(trace_row_stride),
                    block_id,
                    Int32(1),
                    (warp_id == Int32(0)) & (lane_id == Int32(0)),
                )

            consume_ptr = Int32(0)
            mma_running = Int32(1)
            _cached_op_idx = Int32(-1)
            _active_op_warps = Int32(num_mma_warps)
            _cached_compute_wait_count = Int32(0)
            _cached_compute_wait_acquire = Int32(0)
            _cached_compute_wait_barrier = Int32(-2)
            _cached_compute_wait_expected = Int32(-1)

            while mma_running == Int32(1):
                slot = consume_ptr % Int32(num_slots)

                _wn_phase = (consume_ptr // Int32(num_slots)) % Int32(2)
                if const_expr(tracing):
                    _tw = trace_start()
                mbarrier_wait(_work_notify_mbar(smem_base, slot), _wn_phase)

                tile_info_ptr = smem_base + Int32(ring_state_offset) + slot * Int32(tile_info_bytes)
                op_idx = ld_shared_i32(tile_info_ptr)

                if op_idx == Int32(TileInstruction.END_MARKER):
                    mma_running = Int32(0)

                if op_idx != Int32(TileInstruction.END_MARKER):
                    if const_expr(tracing):
                        _mma_lane = end_event_dynamic_raw_1(
                            _tw,
                            _trace_buf,
                            Int32(trace_row_stride),
                            _mma_lane,
                            Int32(trace_data_wait_fmt),
                            op_idx,
                        )

                    tile_0 = ld_shared_i32(tile_info_ptr + Int32(4 * _TILE_INFO_TILE_0))
                    tile_1 = ld_shared_i32(tile_info_ptr + Int32(4 * _TILE_INFO_TILE_1))
                    tile_2 = ld_shared_i32(tile_info_ptr + Int32(4 * _TILE_INFO_TILE_2))
                    tile_3 = ld_shared_i32(tile_info_ptr + Int32(4 * _TILE_INFO_TILE_3))
                    _op_meta_base_cached = _op_meta_base(op_idx)
                    _handler_idx = ld_shared_i32(
                        tile_info_ptr + Int32(4 * _TILE_INFO_HANDLER_IDX)
                    )
                    _handler_local_idx = Int32(0)
                    if const_expr(dispatch_compute_uses_handler_local_idx):
                        _handler_local_idx = _op_meta_i32_base(
                            op_meta_ptr, _op_meta_base_cached, Int32(_OP_META_COMPUTE_LOCAL_IDX)
                        )
                    _op_config = ld_shared_i64(tile_info_ptr + Int32(4 * _TILE_INFO_OP_CONFIG))
                    if const_expr(has_page_free_ops):
                        _page_id = ld_shared_i32(tile_info_ptr + Int32(4 * _TILE_INFO_PAGE_ID))
                    else:
                        _page_id = slot
                    if op_idx != _cached_op_idx:
                        _cached_op_idx = op_idx

                    if const_expr(needs_warp_transition):
                        _active_op_warps = _op_meta_i32_base(
                            op_meta_ptr, _op_meta_base_cached, Int32(_OP_META_NUM_WARPS)
                        )
                        if warp_id >= _active_op_warps:
                            setmaxregister_decrease(MIN_IDLE_REGS)
                        named_barrier_sync(
                            Int32(1), Int32(num_compute_threads))
                        if warp_id < _active_op_warps:
                            setmaxregister_increase(mma_reg_count)
                    if const_expr(max_compute_waits > 0):
                        _cached_compute_wait_count = _op_meta_i32_base(
                            op_meta_ptr,
                            _op_meta_base_cached,
                            Int32(_OP_META_COMPUTE_WAIT_COUNT),
                        )
                        _cached_compute_wait_acquire = _op_meta_i32_base(
                            op_meta_ptr,
                            _op_meta_base_cached,
                            Int32(_OP_META_WAIT_ACQUIRE),
                        )

                    if const_expr(max_compute_waits > 0):
                        if _cached_compute_wait_count > Int32(0):
                            _barrier_meta_idx = ld_shared_i32(
                                tile_info_ptr + Int32(4 * _TILE_INFO_INSTRUCTION_IDX)
                            )
                            if warp_id == Int32(0) and lane_id == Int32(0):
                                _done_compute_waits = Int32(0)
                                for _cw in range_constexpr(max_compute_waits):
                                    if _done_compute_waits == Int32(0):
                                        if _cw < _cached_compute_wait_count:
                                            _cwi_off = (
                                                _barrier_meta_idx
                                                * Int32(max_compute_waits * 2)
                                                + Int32(_cw * 2)
                                            )
                                            _cbar_idx = ld_global_i32(
                                                compute_wait_info_ptr, _cwi_off
                                            )
                                            if _cbar_idx >= Int32(0):
                                                _cbar_exp = ld_global_i32(
                                                    compute_wait_info_ptr,
                                                    _cwi_off + Int32(1),
                                                )
                                                if (
                                                    _cbar_idx != _cached_compute_wait_barrier
                                                    or _cbar_exp != _cached_compute_wait_expected
                                                ):
                                                    if const_expr(tracing):
                                                        _tdw = trace_start()
                                                    if const_expr(relaxed_global_barriers):
                                                        if _cached_compute_wait_acquire != Int32(0):
                                                            global_barrier_wait(
                                                                barriers_ptr, _cbar_idx, _cbar_exp
                                                            )
                                                        else:
                                                            global_barrier_wait_relaxed(
                                                                barriers_ptr,
                                                                _cbar_idx,
                                                                _cbar_exp,
                                                                Int32(global_barrier_sleep_ns),
                                                            )
                                                    else:
                                                        global_barrier_wait(
                                                            barriers_ptr, _cbar_idx, _cbar_exp
                                                        )
                                                    if const_expr(tracing):
                                                        _mma_lane = end_event_dynamic_raw_1(
                                                            _tdw,
                                                            _trace_buf,
                                                            Int32(trace_row_stride),
                                                            _mma_lane,
                                                            Int32(trace_dep_wait_fmt),
                                                            op_idx,
                                                        )
                                                    _cached_compute_wait_barrier = _cbar_idx
                                                    _cached_compute_wait_expected = _cbar_exp
                                            else:
                                                _done_compute_waits = Int32(1)
                                        else:
                                            _done_compute_waits = Int32(1)
                            named_barrier_sync(Int32(1), Int32(num_compute_threads))

                    if const_expr(tracing):
                        _tc = trace_start()

                    if warp_id < _active_op_warps:
                        if const_expr(has_page_free_ops):
                            page_ptr = _get_page_ptr(smem_base, _page_id)
                        else:
                            page_ptr = _get_page_ptr(smem_base, slot)
                        if const_expr(dispatch_compute_uses_handler_local_idx):
                            dispatch_compute(
                                _handler_idx,
                                _handler_local_idx,
                                page_ptr,
                                tile_0,
                                tile_1,
                                tile_2,
                                tile_3,
                                _op_config,
                            )
                        else:
                            dispatch_compute(
                                _handler_idx,
                                page_ptr,
                                tile_0,
                                tile_1,
                                tile_2,
                                tile_3,
                                _op_config,
                            )

                    if const_expr(sync_compute_warps_after_tile):
                        named_barrier_sync(Int32(1), Int32(num_compute_threads))

                    if const_expr(tracing):
                        _mma_lane = end_event_dynamic_raw_1(
                            _tc,
                            _trace_buf,
                            Int32(trace_row_stride),
                            _mma_lane,
                            ld_global_i32(trace_compute_fmt_ptr, op_idx),
                            op_idx,
                        )

                    with cute.arch.elect_one():
                        mbarrier_arrive(_compute_done_mbar(smem_base, slot))

                    consume_ptr = consume_ptr + Int32(1)

            if const_expr(tracing):
                finish_lane_dynamic_raw(_trace_buf, _mma_lane)

    return _kernel_loop_ring


def build_compute_only_kernel_loop(kernel, kernel_cfg, runtime):
    """Build a lighter replay loop for graphs with compute phases only."""
    tracing = bool(kernel.config.tracing)
    dispatch_compute = runtime["dispatch_compute"]
    signal_barriers = runtime["signal_barriers"]
    max_waits = runtime["max_waits"]
    max_signal_formulas = kernel._max_signal_formulas
    num_mma_warps = kernel_cfg["num_mma_warps"]
    num_compute_threads = kernel_cfg["num_compute_threads"]
    mma_reg_count = kernel_cfg["mma_reg_count"]
    sync_compute_warps_after_tile = kernel._sync_compute_warps_after_tile()
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
        _fetch_idx = block_id
        _fetch_limit = num_instructions
        _fetch_stride = num_blocks

        _cached_wait_count = Int32(0)
        _cached_wait_acquire = Int32(0)
        _cached_signal_count = Int32(0)
        _cached_wait_barrier = Int32(-2)
        _cached_wait_expected = Int32(-1)
        _running = Int32(1)
        _cached_op_idx = Int32(TileInstruction.END_MARKER)
        _cached_handler = Int32(0)
        _cached_phase_mask = Int32(0)
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
                    _cached_phase_mask = _op_meta_i32_base(
                        op_meta_ptr, _meta_base, Int32(_OP_META_PHASE_MASK)
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
                        _cached_signal_count = _op_meta_i32_base(
                            op_meta_ptr, _meta_base, Int32(_OP_META_SIGNAL_COUNT)
                        )
                        _cached_wait_acquire = _op_meta_i32_base(
                            op_meta_ptr, _meta_base, Int32(_OP_META_WAIT_ACQUIRE)
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

                named_barrier_sync(Int32(1), Int32(num_compute_threads))

                _tile_01 = ld_shared_i32(iq_base + Int32(4 * _INSTR_TILE_01))
                _tile_23 = ld_shared_i32(iq_base + Int32(4 * _INSTR_TILE_23))
                tile_0 = _tile_01 & Int32(65535)
                tile_1 = (_tile_01 >> Int32(16)) & Int32(65535)
                tile_2 = _tile_23 & Int32(65535)
                tile_3 = (_tile_23 >> Int32(16)) & Int32(65535)
                page_ptr = _get_page_ptr(smem_base, Int32(0))
                _phase_mask = _cached_phase_mask
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
                    if const_expr(dispatch_compute_uses_handler_local_idx):
                        if const_expr(tracing):
                            _tc = trace_start()
                        dispatch_compute(
                            _handler,
                            _compute_local,
                            page_ptr,
                            tile_0,
                            tile_1,
                            tile_2,
                            tile_3,
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
                            tile_0,
                            tile_1,
                            tile_2,
                            tile_3,
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
                    if (
                        _range_axis >= Int32(0)
                        and warp_id == Int32(0)
                        and lane_id == Int32(0)
                    ):
                        if _cached_signal_count > Int32(0):
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

                if const_expr(sync_compute_warps_after_tile):
                    named_barrier_sync(
                        Int32(1),
                        Int32(num_compute_threads),
                    )

                if (
                    _range_axis < Int32(0)
                    and warp_id == Int32(0)
                    and lane_id == Int32(0)
                ):
                    if _cached_signal_count > Int32(0):
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
