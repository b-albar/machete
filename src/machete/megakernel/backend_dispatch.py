# Copyright (c) 2025, Machete Authors
"""Dispatch generation helpers for the handler backend."""

from __future__ import annotations

from typing import Any, Dict, List

import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, scf

from .backend_ir import PHASE_NAMES
from .compile import compile_phase
from .ops import build_op_config


def make_switch_dispatch_callable(
    *,
    fn_name: str,
    phase_name: str,
    binder_fns: Dict[int, Any],
    handler_ids: List[int],
    use_handler_local_idx: bool,
):
    """Return a noinline JIT callable that emits a raw-MLIR switch."""
    if use_handler_local_idx:
        @cute.jit
        def _dispatch(*args):
            handler_idx = args[0]
            handler_local_idx = args[1]
            binder_rest = args[2:]

            switch_idx = arith.IndexCastOp(ir.IndexType.get(), handler_idx.value).result
            switch_op = scf.IndexSwitchOp([], switch_idx, handler_ids, len(handler_ids))

            default_block = switch_op.regions[0].blocks.append()
            with ir.InsertionPoint(default_block):
                scf.YieldOp([])

            for region_idx, handler_id in enumerate(handler_ids, start=1):
                case_block = switch_op.regions[region_idx].blocks.append()
                with ir.InsertionPoint(case_block):
                    binder_fns[handler_id](handler_local_idx, *binder_rest)
                    scf.YieldOp([])
            return None
    else:
        @cute.jit
        def _dispatch(*args):
            handler_idx = args[0]
            binder_rest = args[1:]

            switch_idx = arith.IndexCastOp(ir.IndexType.get(), handler_idx.value).result
            switch_op = scf.IndexSwitchOp([], switch_idx, handler_ids, len(handler_ids))

            default_block = switch_op.regions[0].blocks.append()
            with ir.InsertionPoint(default_block):
                scf.YieldOp([])

            for region_idx, handler_id in enumerate(handler_ids, start=1):
                case_block = switch_op.regions[region_idx].blocks.append()
                with ir.InsertionPoint(case_block):
                    binder_fns[handler_id](*binder_rest)
                    scf.YieldOp([])
            return None

    _dispatch.__name__ = fn_name
    _dispatch._noinline = True
    _dispatch._machete_switch_dispatch = True
    _dispatch._machete_phase_name = phase_name
    return _dispatch


def group_uses_handler_local_idx(
    *,
    handler_local_ids: List[int],
    op_indices: List[int],
    op_phase_tensor_args,
    op_phase_tma_args,
) -> bool:
    """Return whether one handler group still needs local binding dispatch."""
    representative_op_by_local_id = {}
    for local_id, op_idx in zip(handler_local_ids, op_indices):
        representative_op_by_local_id.setdefault(local_id, op_idx)
    unique_op_indices = list(representative_op_by_local_id.values())
    group_tensor_sigs = {tuple(op_phase_tensor_args[op_idx]) for op_idx in unique_op_indices}
    group_tma_sigs = {tuple(op_phase_tma_args[op_idx]) for op_idx in unique_op_indices}
    return not (len(group_tensor_sigs) == 1 and len(group_tma_sigs) == 1)


def make_local_switch_binder(
    *,
    binder_name: str,
    phase_name: str,
    handler_idx: int,
    handler_local_ids: List[int],
    op_indices: List[int],
    op_phase_tensor_args,
    op_phase_tma_args,
    phase_tensor_names: List[str],
    phase_tma_names: List[str],
    is_load: bool,
    phase_fn,
    accept_handler_local_idx: bool,
):
    """Return a callable that emits a local-index switch for one handler group."""
    all_arg_names = list(phase_tensor_names) + list(phase_tma_names)
    representative_op_by_local_id = {}
    for local_id, op_idx in zip(handler_local_ids, op_indices):
        representative_op_by_local_id.setdefault(local_id, op_idx)
    unique_local_ids = list(representative_op_by_local_id.keys())
    unique_op_indices = [representative_op_by_local_id[local_id] for local_id in unique_local_ids]

    group_tensor_sigs = {tuple(op_phase_tensor_args[op_idx]) for op_idx in unique_op_indices}
    group_tma_sigs = {tuple(op_phase_tma_args[op_idx]) for op_idx in unique_op_indices}
    direct_tensor_args = tuple(op_phase_tensor_args[op_indices[0]]) if len(group_tensor_sigs) == 1 else None
    direct_tma_args = tuple(op_phase_tma_args[op_indices[0]]) if len(group_tma_sigs) == 1 else None
    use_direct_call = direct_tensor_args is not None and direct_tma_args is not None

    if accept_handler_local_idx:
        def _binder(*args):
            handler_local_idx = args[0]
            page_ptr = args[1]
            tile_vals = args[2:7]
            op_config_ptr = args[7]

            cursor = 8
            work_mbar = None
            inner_iter_idx = None
            if is_load:
                work_mbar = args[cursor]
                inner_iter_idx = args[cursor + 1]
                cursor += 2

            named_args = {name: value for name, value in zip(all_arg_names, args[cursor:])}

            if use_direct_call:
                call_args = [page_ptr, *tile_vals, op_config_ptr]
                if is_load:
                    call_args.extend([work_mbar, inner_iter_idx])
                call_args.extend(named_args[name] for name in direct_tensor_args)
                call_args.extend(named_args[name] for name in direct_tma_args)
                phase_fn(*call_args)
                return None

            switch_idx = arith.IndexCastOp(ir.IndexType.get(), handler_local_idx.value).result
            switch_op = scf.IndexSwitchOp([], switch_idx, unique_local_ids, len(unique_local_ids))

            default_block = switch_op.regions[0].blocks.append()
            with ir.InsertionPoint(default_block):
                scf.YieldOp([])

            for region_idx, op_idx in enumerate(unique_op_indices, start=1):
                case_block = switch_op.regions[region_idx].blocks.append()
                with ir.InsertionPoint(case_block):
                    call_args = [page_ptr, *tile_vals, op_config_ptr]
                    if is_load:
                        call_args.extend([work_mbar, inner_iter_idx])
                    call_args.extend(named_args[name] for name in op_phase_tensor_args[op_idx])
                    call_args.extend(named_args[name] for name in op_phase_tma_args[op_idx])
                    phase_fn(*call_args)
                    scf.YieldOp([])
            return None
    else:
        def _binder(*args):
            page_ptr = args[0]
            tile_vals = args[1:6]
            op_config_ptr = args[6]

            cursor = 7
            work_mbar = None
            inner_iter_idx = None
            if is_load:
                work_mbar = args[cursor]
                inner_iter_idx = args[cursor + 1]
                cursor += 2

            named_args = {name: value for name, value in zip(all_arg_names, args[cursor:])}

            call_args = [page_ptr, *tile_vals, op_config_ptr]
            if is_load:
                call_args.extend([work_mbar, inner_iter_idx])
            call_args.extend(named_args[name] for name in direct_tensor_args)
            call_args.extend(named_args[name] for name in direct_tma_args)
            phase_fn(*call_args)
            return None

    _binder.__name__ = binder_name
    _binder._machete_phase_name = phase_name
    _binder._machete_handler_idx = handler_idx
    _binder._machete_uses_handler_local_idx = not use_direct_call
    return _binder


def build_exec_dispatch_fn(
    *,
    phase_fns,
    phase_name,
    op_handler_indices,
    op_phase_local_indices,
    op_phase_tensor_args,
    op_phase_tma_args,
    all_canonical,
    all_tma_canonical=None,
):
    """Build a two-level dispatch: handler tree plus per-handler local binders."""
    is_load = phase_name == "load"

    handler_to_ops: Dict[int, List[int]] = {}
    for op_idx_const, handler_idx in enumerate(op_handler_indices):
        handler_to_ops.setdefault(handler_idx, []).append(op_idx_const)

    handler_uses_local_idx = {}
    for handler_idx, op_indices in handler_to_ops.items():
        handler_local_ids = [op_phase_local_indices[i] for i in op_indices]
        handler_uses_local_idx[handler_idx] = group_uses_handler_local_idx(
            handler_local_ids=handler_local_ids,
            op_indices=op_indices,
            op_phase_tensor_args=op_phase_tensor_args,
            op_phase_tma_args=op_phase_tma_args,
        )
    phase_uses_handler_local_idx = any(handler_uses_local_idx.values())

    handler_ids = list(range(len(phase_fns)))
    binder_fns = {}
    for handler_idx, op_indices in handler_to_ops.items():
        handler_local_ids = [op_phase_local_indices[i] for i in op_indices]
        binder_fns[handler_idx] = make_local_switch_binder(
            binder_name=f"_bind_{handler_idx}",
            phase_name=phase_name,
            handler_idx=handler_idx,
            handler_local_ids=handler_local_ids,
            op_indices=op_indices,
            op_phase_tensor_args=op_phase_tensor_args,
            op_phase_tma_args=op_phase_tma_args,
            phase_tensor_names=all_canonical,
            phase_tma_names=all_tma_canonical or [],
            is_load=is_load,
            phase_fn=phase_fns[handler_idx],
            accept_handler_local_idx=phase_uses_handler_local_idx,
        )
    return make_switch_dispatch_callable(
        fn_name=f"dispatch_{phase_name}_switch",
        phase_name=phase_name,
        binder_fns=binder_fns,
        handler_ids=handler_ids,
        use_handler_local_idx=phase_uses_handler_local_idx,
    ), phase_uses_handler_local_idx


def compile_phase_dispatch_inputs(backend, kernel, *, num_dma_warps: int, phase_should_noinline) -> Dict[str, Any]:
    """Compile handler phase functions and build runtime dispatch inputs."""
    ops = kernel.ops
    threads_per_block = kernel.config.threads_per_block
    num_compute_threads = threads_per_block - num_dma_warps * 32
    num_mma_warps = num_compute_threads // 32

    all_canonical = backend.all_canonical(kernel)
    phase_tensor_names = {phase: [] for phase in PHASE_NAMES}
    phase_tma_names = {phase: [] for phase in PHASE_NAMES}
    phase_op_tensor_args = {phase: [[] for _ in ops] for phase in PHASE_NAMES}
    op_tma_args = {phase: [[] for _ in ops] for phase in PHASE_NAMES}
    has_communicate = kernel._peer_tma_registry.has_peer_tma
    op_weights = [spec.weight for spec in backend.ir.op_specs]
    compile_keys = backend.compile_keys()
    op_handler_indices = backend.handler_indices()
    op_phase_local_indices = {
        phase: backend.phase_local_indices(phase)
        for phase in PHASE_NAMES
    }
    kernel_tma_arg_names: List[str] = []
    seen_kernel_tma = set()
    seen_phase_tma = {phase: set() for phase in PHASE_NAMES}
    load_fns: List[Any] = [None] * len(backend.ir.handler_specs)
    compute_fns: List[Any] = [None] * len(backend.ir.handler_specs)
    store_fns: List[Any] = [None] * len(backend.ir.handler_specs)
    communicate_fns: List[Any] = [None] * len(backend.ir.handler_specs)
    inner_iters_by_handler: List[int] = [1] * len(backend.ir.handler_specs)
    handler_warps: List[int] = [num_mma_warps] * len(backend.ir.handler_specs)
    phase_fn_lists = {"load": load_fns, "compute": compute_fns, "store": store_fns}
    if has_communicate:
        phase_fn_lists["communicate"] = communicate_fns

    for i, op in enumerate(ops):
        handler_idx = op_handler_indices[i]
        if load_fns[handler_idx] is None:
            kernel_config = {"threads_per_row": num_compute_threads}
            instance = op.op_cls(**build_op_config(op, kernel_config=kernel_config))
            handler_spec = backend.ir.handler_specs[handler_idx]
            inner_iters_by_handler[handler_idx] = getattr(instance, "inner_iters", 1)
            handler_warps[handler_idx] = getattr(instance, "num_mma_warps", num_mma_warps)
            for phase_name, fn_list in phase_fn_lists.items():
                local_tma_args = list(handler_spec.local_tma_args[phase_name])
                phase_noinline = phase_should_noinline(instance, phase_name)
                compiled_fn = compile_phase(
                    instance,
                    phase_name,
                    tensor_param_names=list(handler_spec.local_tensor_names[phase_name]),
                    tma_param_names=local_tma_args,
                    tma_local_mapping={name: name for name in local_tma_args},
                    noinline=phase_noinline,
                    reconstruct_tensors=True,
                )
                fn_targets = [compiled_fn]
                wrapped = getattr(compiled_fn, "__wrapped__", None)
                if wrapped is not None and wrapped is not compiled_fn:
                    fn_targets.append(wrapped)
                debug_name = f"{type(instance).__name__}_{phase_name}_h{handler_idx}"
                for target in fn_targets:
                    target._machete_handler_idx = handler_idx
                    target._machete_compile_key = compile_keys[i]
                    target.__name__ = debug_name
                fn_list[handler_idx] = compiled_fn

    for i, _op in enumerate(ops):
        for phase_name in PHASE_NAMES:
            phase_args = tuple(backend.ir.op_specs[i].tma_args[phase_name])
            op_tma_args[phase_name][i] = list(phase_args)
            for name in phase_args:
                if name not in seen_phase_tma[phase_name]:
                    seen_phase_tma[phase_name].add(name)
                    phase_tma_names[phase_name].append(name)
                if name not in seen_kernel_tma:
                    seen_kernel_tma.add(name)
                    kernel_tma_arg_names.append(name)

    def _build_dispatch(phase_fns, phase_name):
        return build_exec_dispatch_fn(
            phase_fns=phase_fns,
            phase_name=phase_name,
            op_handler_indices=op_handler_indices,
            op_phase_local_indices=op_phase_local_indices[phase_name],
            op_phase_tensor_args=phase_op_tensor_args[phase_name],
            op_phase_tma_args=op_tma_args[phase_name],
            all_canonical=phase_tensor_names[phase_name],
            all_tma_canonical=phase_tma_names[phase_name],
        )

    dispatch_load, load_uses_local_idx = _build_dispatch(load_fns, "load")
    dispatch_compute, compute_uses_local_idx = _build_dispatch(compute_fns, "compute")
    dispatch_store, store_uses_local_idx = _build_dispatch(store_fns, "store")
    if has_communicate:
        dispatch_communicate, communicate_uses_local_idx = _build_dispatch(
            communicate_fns, "communicate"
        )
    else:
        dispatch_communicate = None
        communicate_uses_local_idx = False

    return {
        "dispatch_load": dispatch_load,
        "dispatch_compute": dispatch_compute,
        "dispatch_store": dispatch_store,
        "dispatch_communicate": dispatch_communicate,
        "phase_uses_handler_local_idx": {
            "load": load_uses_local_idx,
            "compute": compute_uses_local_idx,
            "store": store_uses_local_idx,
            "communicate": communicate_uses_local_idx,
        },
        "inner_iters_list": [inner_iters_by_handler[h] for h in op_handler_indices],
        "has_communicate": has_communicate,
        "per_op_warps": [handler_warps[h] for h in op_handler_indices],
        "op_tensor_args": phase_op_tensor_args,
        "op_tma_args": op_tma_args,
        "op_weights": op_weights,
        "compile_keys": compile_keys,
        "op_handler_indices": op_handler_indices,
        "op_phase_local_indices": op_phase_local_indices,
        "all_canonical": all_canonical,
        "all_tma_canonical": kernel_tma_arg_names,
        "phase_tensor_names": phase_tensor_names,
        "phase_tma_names": phase_tma_names,
    }


__all__ = [
    "build_exec_dispatch_fn",
    "compile_phase_dispatch_inputs",
    "group_uses_handler_local_idx",
    "make_local_switch_binder",
    "make_switch_dispatch_callable",
]
