# Copyright (c) 2026, Machete Authors
"""Support helpers for generated megakernel source."""

from __future__ import annotations

from typing import Any, Dict

import cutlass.cute as cute
from cutlass import Int32, Int64

from .interpreter import (
    get_smem_base_ptr,
    global_barrier_signal,
    global_barrier_signal_gpu,
    global_memory_fence_gpu,
    ld_global_i64,
    load_instruction_to_smem,
    mbarrier_arrive,
    mbarrier_init,
    mbarrier_init_fence,
    mbarrier_wait,
    named_barrier_sync,
    nanosleep,
    prefetch_instruction,
)
from .paged_memory import ld_shared_acquire_cta_i32, ld_shared_i32, st_shared_i32, st_shared_release_cta_i32
from .scheduling import TileInstruction


def registry_uses_reduce_store(registry) -> bool:
    """Return whether any descriptor in the registry performs a reduce-store."""
    return any(getattr(desc, "direction", "s2g") == "s2g_reduce" for desc in registry.descriptors)


def build_kernel_exec_globals(
    *,
    tracing: bool,
    dispatch_load,
    dispatch_compute,
    dispatch_store,
    signal_barriers,
    get_page_ptr_fn,
    num_pages: int,
    num_slots: int,
    iq_offset: int,
    flags_offset: int,
    ring_state_offset: int,
    extra_exec_globals=None,
) -> Dict[str, Any]:
    """Build exec globals for the generated `_kernel_loop`."""
    cutlass = __import__("cutlass")
    exec_globals = {
        "cute": cute,
        "Int32": Int32,
        "Int64": Int64,
        "range_constexpr": cutlass.range_constexpr,
        "const_expr": cutlass.const_expr,
        "tracing": bool(tracing),
        "TileInstruction": TileInstruction,
        "dispatch_load": dispatch_load,
        "dispatch_compute": dispatch_compute,
        "dispatch_store": dispatch_store,
        "signal_barriers": signal_barriers,
        "_get_page_ptr": get_page_ptr_fn,
        "ld_shared_i32": ld_shared_i32,
        "st_shared_i32": st_shared_i32,
        "st_shared_release_cta_i32": st_shared_release_cta_i32,
        "ld_shared_acquire_cta_i32": ld_shared_acquire_cta_i32,
        "load_instruction_to_smem": load_instruction_to_smem,
        "prefetch_instruction": prefetch_instruction,
        "ld_global_i64": ld_global_i64,
        "mbarrier_init": mbarrier_init,
        "mbarrier_init_fence": mbarrier_init_fence,
        "mbarrier_arrive": mbarrier_arrive,
        "mbarrier_wait": mbarrier_wait,
        "nanosleep": nanosleep,
        "named_barrier_sync": named_barrier_sync,
        "global_barrier_signal": global_barrier_signal,
        "global_barrier_signal_gpu": global_barrier_signal_gpu,
        "global_memory_fence_gpu": global_memory_fence_gpu,
        "num_pages": num_pages,
        "num_slots": num_slots,
        "iq_offset": iq_offset,
        "flags_offset": flags_offset,
        "ring_state_offset": ring_state_offset,
    }
    if extra_exec_globals:
        exec_globals.update(extra_exec_globals)
    return exec_globals


def build_persistent_kernel_globals(tma_registry, peer_tma_registry, kernel_loop, sync_tma_desc_init_stream) -> Dict[str, Any]:
    """Build exec globals for the generated `PersistentKernel` class."""
    cutlass = __import__("cutlass")
    pk_globals = {
        "cute": cute,
        "cutlass": cutlass,
        "Int32": Int32,
        "Int64": Int64,
        "range_constexpr": cutlass.range_constexpr,
        "const_expr": cutlass.const_expr,
        "get_smem_base_ptr": get_smem_base_ptr,
        "_kernel_loop": kernel_loop,
        "_sync_tma_desc_init_stream": sync_tma_desc_init_stream,
    }
    from .transport import copy_runtime_desc_to_pool, fence_runtime_desc_pool, make_runtime_desc_tma_atom

    pk_globals["copy_runtime_desc_to_pool"] = copy_runtime_desc_to_pool
    pk_globals["fence_runtime_desc_pool"] = fence_runtime_desc_pool
    pk_globals["make_runtime_desc_tma_atom"] = make_runtime_desc_tma_atom

    if tma_registry.has_tma:
        from cutlass.cute.nvgpu.cpasync import (
            CopyBulkTensorTileG2SOp,
            CopyBulkTensorTileS2GOp,
        )

        pk_globals["CopyBulkTensorTileG2SOp"] = CopyBulkTensorTileG2SOp
        pk_globals["CopyBulkTensorTileS2GOp"] = CopyBulkTensorTileS2GOp

        if registry_uses_reduce_store(tma_registry):
            from cutlass.cute.nvgpu.cpasync import CopyReduceBulkTensorTileS2GOp
            from cutlass.cute.tensor import ReductionOp

            pk_globals["CopyReduceBulkTensorTileS2GOp"] = CopyReduceBulkTensorTileS2GOp
            pk_globals["ReductionOp"] = ReductionOp

    if peer_tma_registry.has_peer_tma and not tma_registry.has_tma:
        from cutlass.cute.nvgpu.cpasync import CopyBulkTensorTileS2GOp

        pk_globals["CopyBulkTensorTileS2GOp"] = CopyBulkTensorTileS2GOp

    if peer_tma_registry.has_peer_tma and registry_uses_reduce_store(peer_tma_registry):
        from cutlass.cute.nvgpu.cpasync import CopyReduceBulkTensorTileS2GOp
        from cutlass.cute.tensor import ReductionOp

        pk_globals["CopyReduceBulkTensorTileS2GOp"] = CopyReduceBulkTensorTileS2GOp
        pk_globals["ReductionOp"] = ReductionOp

    return pk_globals


def replace_required(source: str, old: str, new: str) -> str:
    """Replace a generated-source fragment, accepting one indentation variant."""
    if old not in source:
        def _shift_left(text: str) -> str:
            lines = text.splitlines(True)
            return "".join(
                line[8:] if line.startswith("        ") else line
                for line in lines
            )

        old_shifted = _shift_left(old)
        new_shifted = _shift_left(new)
        if old_shifted in source:
            return source.replace(old_shifted, new_shifted, 1)
        raise RuntimeError("Failed to patch generated page-free ring source")
    return source.replace(old, new, 1)


def enable_page_free_ring_source(source: str) -> str:
    """Specialize the generated ring loop for ops that do not own smem pages."""
    source = replace_required(
        source,
        "                _temp_instr = iq_base\n",
        "                data_release_idx_ptr = flags_ptr + FLAG_DATA_RELEASE_IDX\n"
        "                data_produce_idx_ptr = flags_ptr + FLAG_DATA_PRODUCE_IDX\n"
        "                _temp_instr = iq_base\n"
    )
    source = replace_required(
        source,
        "                            st_shared_i32(\n"
        "                                _p_ti + Int32(4 * _TILE_INFO_PAGE_ID),\n"
        "                                _p_slot % Int32(num_pages),\n"
        "                            )\n",
        "                            _ctrl_no_page = (\n"
        "                                (_ctrl_cached_phase_mask // Int32(1 << _INSTR_NO_SMEM_PAGE_BIT))\n"
        "                                % Int32(2)\n"
        "                            )\n"
        "                            st_shared_i32(\n"
        "                                _p_ti + Int32(4 * _TILE_INFO_PAGE_ID),\n"
        "                                Int32(-1),\n"
        "                            )\n"
        "                            if _ctrl_no_page == Int32(0):\n"
        "                                _dp = ld_shared_i32(data_produce_idx_ptr)\n"
        "                                _dr = ld_shared_acquire_cta_i32(data_release_idx_ptr)\n"
        "                                while (_dp - _dr) >= Int32(num_pages):\n"
        "                                    nanosleep(Int32(loader_idle_sleep_ns))\n"
        "                                    _dr = ld_shared_acquire_cta_i32(data_release_idx_ptr)\n"
        "                                st_shared_i32(\n"
        "                                    _p_ti + Int32(4 * _TILE_INFO_PAGE_ID),\n"
        "                                    _dp % Int32(num_pages),\n"
        "                                )\n"
        "                                st_shared_i32(data_produce_idx_ptr, _dp + Int32(1))\n",
    )
    source = replace_required(
        source,
        "                load_done_ptr = flags_ptr + FLAG_LOAD_DONE\n"
        "                while _sw_done == Int32(0):\n",
        "                load_done_ptr = flags_ptr + FLAG_LOAD_DONE\n"
        "                data_release_idx_ptr = flags_ptr + FLAG_DATA_RELEASE_IDX\n"
        "                while _sw_done == Int32(0):\n",
    )
    source = replace_required(
        source,
        "                            st_shared_i32(store_idx_ptr, _s_idx + Int32(1))\n",
        "                            if _ds_page >= Int32(0):\n"
        "                                _store_data_release_idx = ld_shared_i32(data_release_idx_ptr) + Int32(1)\n"
        "                                st_shared_release_cta_i32(\n"
        "                                    data_release_idx_ptr,\n"
        "                                    _store_data_release_idx,\n"
        "                                )\n"
        "                            st_shared_i32(store_idx_ptr, _s_idx + Int32(1))\n",
    )
    return source
