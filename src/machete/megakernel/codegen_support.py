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
    mbarrier_init_fence_async_proxy,
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
    dispatch_store_step=None,
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
        "dispatch_store_step": dispatch_store_step,
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
        "mbarrier_init_fence_async_proxy": mbarrier_init_fence_async_proxy,
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
