# Copyright (c) 2026, Machete Authors
"""Runtime component assembly for generated persistent kernels."""

from __future__ import annotations

from typing import Any, Dict

import cutlass.cute as cute
from cutlass import Int32, Int64, range_constexpr

from .interpreter import (
    global_barrier_signal,
    global_barrier_signal_gpu,
    global_barrier_wait,
    global_barrier_wait_relaxed,
    global_memory_fence_gpu,
    ld_global_i32,
)
from .paged_memory import (
    FLAG_DATA_PRODUCE_IDX,
    FLAG_DATA_RELEASE_IDX,
    FLAG_DISPATCH_LOAD,
    FLAG_LOAD_DONE,
    FLAG_PRODUCE_IDX,
    FLAG_STORE_IDX,
    ld_shared_i64,
    ld_shared_v2_b32,
    st_shared_i64,
    st_shared_v2_b32,
)
from .scheduling import (
    INSTR_BARRIER_META_IDX,
    INSTR_RANGE_END,
    INSTR_RANGE_META,
    INSTR_TILE_01,
    INSTR_TILE_23,
)


def resolve_warp_register_api():
    """Resolve the CuTe warp register allocation API, with a no-op fallback."""
    try:
        from cutlass.cute.arch import (
            setmaxregister_decrease,
            setmaxregister_increase,
        )

        return setmaxregister_increase, setmaxregister_decrease
    except ImportError:
        try:
            from cutlass.cute.arch import (
                warpgroup_reg_alloc as setmaxregister_increase,
                warpgroup_reg_dealloc as setmaxregister_decrease,
            )

            return setmaxregister_increase, setmaxregister_decrease
        except ImportError:

            def _setmaxregister_noop(_n):
                """Fallback when the local CuTe build exposes no register API."""

            return _setmaxregister_noop, _setmaxregister_noop


def build_kernel_static_config(
    kernel,
    *,
    use_compute_only_replay: bool,
    num_dma_warps: int,
    mbarrier_stride: int,
    tile_info_bytes: int,
) -> Dict[str, Any]:
    """Collect the compile-time constants used to build the persistent kernel."""
    layout = kernel._layout
    threads_per_block = kernel.config.threads_per_block
    active_dma_warps = 0 if use_compute_only_replay else num_dma_warps
    num_mma_warps = (threads_per_block // 32) - active_dma_warps

    return {
        "num_sms": kernel.config.num_sms,
        "threads_per_block": threads_per_block,
        "smem_size": layout.total_size,
        "tracing": kernel.config.tracing,
        "num_pages": layout.num_pages,
        "num_slots": layout.num_slots,
        "has_page_free_ops": kernel._has_page_free_ops and not use_compute_only_replay,
        "iq_offset": layout.iq_offset,
        "flags_offset": layout.flags_offset,
        "ring_state_offset": layout.ring_state_offset,
        "pages_start": layout.pages_start,
        "aligned_page_size": layout.aligned_page_size,
        "work_notify_mbar_offset_0": layout.work_notify_mbar_offset(0),
        "compute_done_mbar_offset_0": layout.compute_done_mbar_offset(0),
        "num_mma_warps": num_mma_warps,
        "num_compute_threads": num_mma_warps * 32,
        "num_dma_warps": active_dma_warps,
        "dma_reg_count": kernel.config.dma_reg_count,
        "mma_reg_count": kernel.config.mma_reg_count,
        "actual_threads_per_block": threads_per_block,
        "mbarrier_stride": mbarrier_stride,
        "tile_info_bytes": tile_info_bytes,
        "peer_barriers_data_ptr": (
            kernel.config.peer_barriers.data_ptr()
            if kernel.config.peer_barriers is not None
            else 0
        ),
    }


def build_kernel_runtime_components(kernel, kernel_cfg: Dict[str, Any], op_meta: Dict[str, int]) -> Dict[str, Any]:
    """Build reusable runtime helpers needed by `_create_kernel`."""
    _OP_META_STRIDE = op_meta["_OP_META_STRIDE"]
    _OP_META_STRIDE_0 = op_meta["_OP_META_STRIDE_0"]
    _OP_META_STRIDE_1 = op_meta["_OP_META_STRIDE_1"]
    _OP_META_STRIDE_2 = op_meta["_OP_META_STRIDE_2"]
    _OP_META_STRIDE_3 = op_meta["_OP_META_STRIDE_3"]
    _OP_META_COUNT_0 = op_meta["_OP_META_COUNT_0"]
    _OP_META_COUNT_1 = op_meta["_OP_META_COUNT_1"]
    _OP_META_COUNT_2 = op_meta["_OP_META_COUNT_2"]
    _OP_META_COUNT_3 = op_meta["_OP_META_COUNT_3"]

    setmaxregister_increase, setmaxregister_decrease = resolve_warp_register_api()
    (
        dispatch_load,
        dispatch_compute,
        dispatch_store,
        dispatch_communicate,
        phase_uses_handler_local_idx,
        phase_uses_runtime_transport_selector,
        phase_uses_desc_slot_selector,
        has_communicate,
        per_op_warps,
        phase_tensor_names,
        phase_tma_names,
        all_tma_canonical,
    ) = kernel._build_pipelined_dispatch_fns()

    if kernel.config.tracing:
        from .tracing import get_trace_exec_globals

        trace_exec_globals = get_trace_exec_globals(kernel._tracing_state)
    else:
        trace_exec_globals = {}

    @cute.jit
    def _get_page_ptr(smem_base: Int32, page_idx: Int32) -> Int32:
        """Return the shared-memory base pointer for a page slot."""
        return (
            smem_base
            + Int32(kernel_cfg["pages_start"])
            + page_idx * Int32(kernel_cfg["aligned_page_size"])
        )

    @cute.jit
    def _work_notify_mbar(smem_base: Int32, slot: Int32) -> Int32:
        """Return the work-available mbarrier address for one slot."""
        return (
            smem_base
            + Int32(kernel_cfg["work_notify_mbar_offset_0"])
            + slot * Int32(kernel_cfg["mbarrier_stride"])
        )

    @cute.jit
    def _compute_done_mbar(smem_base: Int32, page_idx: Int32) -> Int32:
        """Return the compute-complete mbarrier address for one page."""
        return (
            smem_base
            + Int32(kernel_cfg["compute_done_mbar_offset_0"])
            + page_idx * Int32(kernel_cfg["mbarrier_stride"])
        )

    @cute.jit
    def _op_meta_i32(op_meta_ptr: Int64, op_idx: Int32, field: Int32) -> Int32:
        return ld_global_i32(op_meta_ptr, op_idx * Int32(_OP_META_STRIDE) + field)

    @cute.jit
    def _op_meta_base(op_idx: Int32) -> Int32:
        return op_idx * Int32(_OP_META_STRIDE)

    @cute.jit
    def _op_meta_i32_base(op_meta_ptr: Int64, op_meta_base: Int32, field: Int32) -> Int32:
        return ld_global_i32(op_meta_ptr, op_meta_base + field)

    @cute.jit
    def _decompose_tile(op_meta_ptr: Int64, op_meta_base: Int32, linear_idx: Int32):
        rem = linear_idx
        t0 = Int32(0)
        t1 = Int32(0)
        t2 = Int32(0)
        t3 = Int32(0)
        c0 = _op_meta_i32_base(op_meta_ptr, op_meta_base, Int32(_OP_META_COUNT_0))
        c1 = _op_meta_i32_base(op_meta_ptr, op_meta_base, Int32(_OP_META_COUNT_1))
        c2 = _op_meta_i32_base(op_meta_ptr, op_meta_base, Int32(_OP_META_COUNT_2))
        c3 = _op_meta_i32_base(op_meta_ptr, op_meta_base, Int32(_OP_META_COUNT_3))
        s0 = _op_meta_i32_base(op_meta_ptr, op_meta_base, Int32(_OP_META_STRIDE_0))
        s1 = _op_meta_i32_base(op_meta_ptr, op_meta_base, Int32(_OP_META_STRIDE_1))
        s2 = _op_meta_i32_base(op_meta_ptr, op_meta_base, Int32(_OP_META_STRIDE_2))
        s3 = _op_meta_i32_base(op_meta_ptr, op_meta_base, Int32(_OP_META_STRIDE_3))
        if c0 > Int32(1):
            if s0 > Int32(1):
                t0 = rem // s0
                rem = rem % s0
            else:
                t0 = rem
        if c1 > Int32(1):
            if s1 > Int32(1):
                t1 = rem // s1
                rem = rem % s1
            else:
                t1 = rem
        if c2 > Int32(1):
            if s2 > Int32(1):
                t2 = rem // s2
                rem = rem % s2
            else:
                t2 = rem
        if c3 > Int32(1):
            if s3 > Int32(1):
                t3 = rem // s3
                rem = rem % s3
            else:
                t3 = rem
        return t0, t1, t2, t3

    @cute.jit
    def _advance_tile(op_meta_ptr: Int64, op_meta_base: Int32, t0: Int32, t1: Int32, t2: Int32, t3: Int32):
        carry = Int32(1)
        c3 = _op_meta_i32_base(op_meta_ptr, op_meta_base, Int32(_OP_META_COUNT_3))
        if carry != Int32(0) and c3 > Int32(1):
            t3 = t3 + Int32(1)
            if t3 < c3:
                carry = Int32(0)
            else:
                t3 = Int32(0)
        c2 = _op_meta_i32_base(op_meta_ptr, op_meta_base, Int32(_OP_META_COUNT_2))
        if carry != Int32(0) and c2 > Int32(1):
            t2 = t2 + Int32(1)
            if t2 < c2:
                carry = Int32(0)
            else:
                t2 = Int32(0)
        c1 = _op_meta_i32_base(op_meta_ptr, op_meta_base, Int32(_OP_META_COUNT_1))
        if carry != Int32(0) and c1 > Int32(1):
            t1 = t1 + Int32(1)
            if t1 < c1:
                carry = Int32(0)
            else:
                t1 = Int32(0)
        c0 = _op_meta_i32_base(op_meta_ptr, op_meta_base, Int32(_OP_META_COUNT_0))
        if carry != Int32(0) and c0 > Int32(1):
            t0 = t0 + Int32(1)
            if t0 >= c0:
                t0 = Int32(0)
        return t0, t1, t2, t3

    @cute.jit
    def _signal_barriers_from_meta(
        signal_meta_ptr: Int64,
        instruction_idx: Int32,
        signal_count: Int32,
        barriers_ptr: Int64,
    ):
        done_signals = Int32(0)
        for sig_idx in range_constexpr(kernel._max_signal_formulas):
            if done_signals == Int32(0):
                if sig_idx < signal_count:
                    barrier_idx = ld_global_i32(
                        signal_meta_ptr,
                        instruction_idx * Int32(kernel._max_signal_formulas) + Int32(sig_idx),
                    )
                    if barrier_idx >= Int32(0):
                        global_barrier_signal_gpu(barriers_ptr, barrier_idx)
                    else:
                        done_signals = Int32(1)
                else:
                    done_signals = Int32(1)

    if has_communicate:
        @cute.jit
        def _signal_peer_barriers_from_meta(
            peer_signal_ptr: Int64,
            op_idx: Int32,
            linear_tile_idx: Int32,
            peer_barriers_ptr: Int64,
        ):
            barrier_offset = ld_global_i32(peer_signal_ptr, op_idx)
            if barrier_offset >= Int32(0):
                global_barrier_signal(
                    peer_barriers_ptr, barrier_offset + linear_tile_idx
                )
    else:
        _signal_peer_barriers_from_meta = None

    return {
        "setmaxregister_increase": setmaxregister_increase,
        "setmaxregister_decrease": setmaxregister_decrease,
        "dispatch_load": dispatch_load,
        "dispatch_compute": dispatch_compute,
        "dispatch_store": dispatch_store,
        "dispatch_communicate": dispatch_communicate,
        "phase_uses_handler_local_idx": phase_uses_handler_local_idx,
        "phase_uses_runtime_transport_selector": phase_uses_runtime_transport_selector,
        "phase_uses_desc_slot_selector": phase_uses_desc_slot_selector,
        "has_communicate": has_communicate,
        "needs_warp_transition": any(w < kernel_cfg["num_mma_warps"] for w in per_op_warps),
        "max_waits": kernel._max_wait_formulas,
        "max_compute_waits": kernel._max_compute_wait_formulas,
        "signal_barriers": _signal_barriers_from_meta,
        "signal_peer_barriers": _signal_peer_barriers_from_meta,
        "trace_exec_globals": trace_exec_globals,
        "decompose_tile": _decompose_tile,
        "_op_meta_i32": _op_meta_i32,
        "_op_meta_base": _op_meta_base,
        "_op_meta_i32_base": _op_meta_i32_base,
        "_get_page_ptr": _get_page_ptr,
        "_work_notify_mbar": _work_notify_mbar,
        "_compute_done_mbar": _compute_done_mbar,
        "phase_tensor_names": phase_tensor_names,
        "phase_tma_names": phase_tma_names,
        "all_tma_canonical": all_tma_canonical,
        "advance_tile": _advance_tile,
    }


def build_kernel_extra_exec_globals(
    kernel,
    kernel_cfg: Dict[str, Any],
    runtime: Dict[str, Any],
    *,
    op_meta: Dict[str, int],
    tile_info: Dict[str, int],
    sync_compute_warps_after_tile: bool,
    min_idle_regs: int,
    op_phase_load: int,
    op_phase_store: int,
    op_phase_communicate: int,
) -> Dict[str, Any]:
    """Build the extra globals consumed by the generated kernel loop."""
    exec_globals = {
        "_work_notify_mbar": runtime["_work_notify_mbar"],
        "_compute_done_mbar": runtime["_compute_done_mbar"],
        "decompose_tile": runtime["decompose_tile"],
        "advance_tile": runtime["advance_tile"],
        "ld_shared_v2_b32": ld_shared_v2_b32,
        "st_shared_v2_b32": st_shared_v2_b32,
        "ld_shared_i64": ld_shared_i64,
        "st_shared_i64": st_shared_i64,
        "num_mma_warps": kernel_cfg["num_mma_warps"],
        "num_compute_threads": kernel_cfg["num_compute_threads"],
        "threads_per_block": kernel_cfg["actual_threads_per_block"],
        "setmaxregister_increase": runtime["setmaxregister_increase"],
        "setmaxregister_decrease": runtime["setmaxregister_decrease"],
        "dma_reg_count": kernel_cfg["dma_reg_count"],
        "mma_reg_count": kernel_cfg["mma_reg_count"],
        "tile_info_bytes": kernel_cfg["tile_info_bytes"],
        "FLAG_DISPATCH_LOAD": FLAG_DISPATCH_LOAD,
        "FLAG_PRODUCE_IDX": FLAG_PRODUCE_IDX,
        "FLAG_STORE_IDX": FLAG_STORE_IDX,
        "FLAG_LOAD_DONE": FLAG_LOAD_DONE,
        "FLAG_DATA_RELEASE_IDX": FLAG_DATA_RELEASE_IDX,
        "FLAG_DATA_PRODUCE_IDX": FLAG_DATA_PRODUCE_IDX,
        "_op_meta_i32": runtime["_op_meta_i32"],
        "_op_meta_base": runtime["_op_meta_base"],
        "_op_meta_i32_base": runtime["_op_meta_i32_base"],
        "max_waits": runtime["max_waits"],
        "max_compute_waits": runtime["max_compute_waits"],
        "max_signal_formulas": kernel._max_signal_formulas,
        "global_barrier_wait": global_barrier_wait,
        "global_barrier_wait_relaxed": global_barrier_wait_relaxed,
        "global_memory_fence_gpu": global_memory_fence_gpu,
        "relaxed_global_barriers": kernel.config.relaxed_global_barriers,
        "global_barrier_sleep_ns": int(kernel.config.global_barrier_sleep_ns),
        "ld_global_i32": ld_global_i32,
        "has_communicate": runtime["has_communicate"],
        "needs_warp_transition": runtime["needs_warp_transition"],
        "sync_compute_warps_after_tile": sync_compute_warps_after_tile,
        "loader_idle_sleep_ns": int(kernel.config.loader_idle_sleep_ns),
        "has_page_free_ops": bool(kernel_cfg["has_page_free_ops"]),
        "dispatch_load_uses_handler_local_idx": runtime["phase_uses_handler_local_idx"]["load"],
        "dispatch_compute_uses_handler_local_idx": runtime["phase_uses_handler_local_idx"]["compute"],
        "dispatch_store_uses_handler_local_idx": runtime["phase_uses_handler_local_idx"]["store"],
        "dispatch_communicate_uses_handler_local_idx": runtime["phase_uses_handler_local_idx"]["communicate"],
        "MIN_IDLE_REGS": min_idle_regs,
        "_OP_PHASE_LOAD": op_phase_load,
        "_OP_PHASE_STORE": op_phase_store,
        "_OP_PHASE_COMMUNICATE": op_phase_communicate,
        **op_meta,
        **tile_info,
        **runtime["trace_exec_globals"],
    }
    if runtime["has_communicate"]:
        exec_globals.update(
            {
                "dispatch_communicate": runtime["dispatch_communicate"],
                "signal_peer_barriers": runtime["signal_peer_barriers"],
                "_peer_barriers_data_ptr": kernel_cfg["peer_barriers_data_ptr"],
            }
        )
    return exec_globals
