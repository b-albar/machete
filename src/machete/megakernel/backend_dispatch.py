# Copyright (c) 2025, Machete Authors
"""Dispatch generation helpers for the handler backend."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cutlass.cute as cute
from cutlass import Int32
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, scf

from .backend_ir import PHASE_NAMES
from .compile import compile_phase
from .interpreter import ld_global_i32
from .noinline import _flatten_ir_values, _rebuild
from .ops import build_op_config


def _tma_field_namespace_and_mbar(direction: str) -> Tuple[str, bool]:
    if direction == "g2s":
        return "tmaload", True
    if direction == "s2g":
        return "tmastore", False
    if direction == "s2g_reduce":
        return "tmareduce", False
    raise ValueError(f"unknown TMA direction: {direction}")


def _build_tma_runtime_layout(backend, kernel):
    """Build compact per-phase TMA arg lists plus wrapper rebind specs."""
    local_desc_by_name: Dict[str, Any] = {
        desc.canonical_desc: desc for desc in kernel._tma_registry.descriptors
    }
    peer_desc_by_name: Dict[str, Any] = {
        desc.canonical_desc: desc for desc in kernel._peer_tma_registry.descriptors
    }

    op_phase_tma_args = {phase: [[] for _ in kernel.ops] for phase in PHASE_NAMES}
    phase_tma_names = {phase: [] for phase in PHASE_NAMES}
    seen_phase_tma = {phase: set() for phase in PHASE_NAMES}
    kernel_tma_arg_names: List[str] = []
    seen_kernel_tma = set()
    handler_rebind_specs: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}
    local_desc_slot_by_name = {
        desc.canonical_desc: idx for idx, desc in enumerate(kernel._tma_registry.descriptors)
    }
    peer_desc_slot_by_name = {
        desc.canonical_desc: idx for idx, desc in enumerate(kernel._peer_tma_registry.descriptors)
    }
    op_phase_desc_slots = {phase: [[] for _ in kernel.ops] for phase in PHASE_NAMES}
    local_desc_pool_name = "local_tma_desc_pool_ptr"
    peer_desc_pool_name = "peer_tma_desc_pool_ptr"

    for op_idx, spec in enumerate(backend.ir.op_specs):
        handler_idx = backend.ir.op_handler_indices[op_idx]
        phase_rebinds = handler_rebind_specs.setdefault(
            handler_idx,
            {phase: [] for phase in PHASE_NAMES},
        )
        for phase_name in PHASE_NAMES:
            local_names = spec.local_tma_args[phase_name]
            if not local_names:
                continue
            phase_mapping = (
                kernel._tma_registry.op_mappings.get((op_idx, phase_name), {})
                if phase_name != "communicate"
                else kernel._peer_tma_registry.op_mappings.get((op_idx, phase_name), {})
            )
            phase_args: List[str] = []
            seen_phase_args = set()
            phase_desc_slots: List[int] = []
            for local_name in local_names:
                if not local_name.endswith("_tma") or local_name.endswith("_tma_gmem"):
                    continue
                tensor_name = local_name[:-4]
                local_gmem_name = f"{local_name}_gmem"
                local_desc_name = f"{local_name}_desc"
                canonical_atom = phase_mapping[local_name]
                canonical_gmem = phase_mapping[local_gmem_name]
                canonical_desc = phase_mapping[local_desc_name]
                desc = (
                    local_desc_by_name.get(canonical_desc)
                    if phase_name != "communicate"
                    else peer_desc_by_name.get(canonical_desc)
                )
                if desc is None:
                    raise KeyError(f"missing descriptor metadata for {canonical_desc}")
                field_namespace, supports_mbar = _tma_field_namespace_and_mbar(
                    getattr(desc, "direction", "s2g")
                )
                shape_str = ", ".join(str(s) for s in desc.tile_shape)
                smem_layout_src = desc.smem_layout_src or f"cute.make_layout(({shape_str},))"
                cta_tiler_src = f"({shape_str},)"
                if phase_name != "communicate":
                    desc_pool_name = local_desc_pool_name
                    desc_slot = local_desc_slot_by_name[canonical_desc]
                    runtime_tensor_name = f"tma_{desc.tensor_canonical}_{len(desc.tile_shape)}d"
                    arg_triplet = (desc_pool_name, canonical_atom, runtime_tensor_name)
                else:
                    desc_pool_name = peer_desc_pool_name
                    desc_slot = peer_desc_slot_by_name[canonical_desc]
                    runtime_tensor_name = f"ptma_{desc.tensor_canonical}_p{desc.peer_idx}"
                    arg_triplet = (desc_pool_name, canonical_atom, runtime_tensor_name)
                for arg_name in arg_triplet:
                    if arg_name in seen_phase_args:
                        continue
                    seen_phase_args.add(arg_name)
                    phase_args.append(arg_name)
                    if arg_name not in seen_phase_tma[phase_name]:
                        seen_phase_tma[phase_name].add(arg_name)
                        phase_tma_names[phase_name].append(arg_name)
                    if arg_name not in seen_kernel_tma:
                        seen_kernel_tma.add(arg_name)
                        kernel_tma_arg_names.append(arg_name)
                if not phase_rebinds[phase_name]:
                    phase_rebinds[phase_name] = []
                if not any(r["local_atom_name"] == local_name for r in phase_rebinds[phase_name]):
                    phase_rebinds[phase_name].append(
                        {
                            "local_atom_name": local_name,
                            "local_gmem_name": local_gmem_name,
                            "wrapper_atom_name": canonical_atom,
                            "runtime_tensor_name": runtime_tensor_name,
                            "tensor_name": tensor_name,
                            "desc_pool_name": desc_pool_name,
                            "desc_slot_name": f"{local_name}_desc_slot",
                            "field_namespace": field_namespace,
                            "supports_mbar": supports_mbar,
                            "direction": getattr(desc, "direction", "s2g"),
                            "smem_layout_src": smem_layout_src,
                            "cta_tiler_src": cta_tiler_src,
                            "dim_perm": tuple(getattr(desc, "dim_perm", ()) or ()),
                        }
                    )
                phase_desc_slots.append(desc_slot)
            op_phase_tma_args[phase_name][op_idx] = phase_args
            op_phase_desc_slots[phase_name][op_idx] = phase_desc_slots

    phase_transport_records = {phase: [] for phase in PHASE_NAMES}
    op_phase_transport_indices = {phase: [] for phase in PHASE_NAMES}
    phase_transport_idx = {phase: {} for phase in PHASE_NAMES}
    for phase_name in PHASE_NAMES:
        for phase_args in op_phase_tma_args[phase_name]:
            key = tuple(phase_args)
            record_idx = phase_transport_idx[phase_name].get(key)
            if record_idx is None:
                record_idx = len(phase_transport_records[phase_name])
                phase_transport_idx[phase_name][key] = record_idx
                phase_transport_records[phase_name].append(key)
            op_phase_transport_indices[phase_name].append(record_idx)

    phase_local_transport_positions = {}
    phase_local_desc_slots = {}
    for phase_name in PHASE_NAMES:
        arg_index_by_name = {
            name: idx for idx, name in enumerate(phase_tma_names[phase_name])
        }
        handler_tables: List[Tuple[Tuple[int, ...], ...]] = []
        handler_desc_slot_tables: List[Tuple[Tuple[int, ...], ...]] = []
        for handler_idx in range(len(backend.ir.handler_specs)):
            local_positions: Dict[int, Tuple[int, ...]] = {}
            local_desc_slots: Dict[int, Tuple[int, ...]] = {}
            for op_idx, op_handler_idx in enumerate(backend.ir.op_handler_indices):
                if op_handler_idx != handler_idx:
                    continue
                local_idx = backend.ir.op_phase_local_indices[phase_name][op_idx]
                local_positions[local_idx] = tuple(
                    arg_index_by_name[name]
                    for name in op_phase_tma_args[phase_name][op_idx]
                )
                local_desc_slots[local_idx] = tuple(op_phase_desc_slots[phase_name][op_idx])
            if not local_positions:
                handler_tables.append(())
                handler_desc_slot_tables.append(())
                continue
            num_variants = max(local_positions) + 1
            handler_tables.append(
                tuple(local_positions.get(i, ()) for i in range(num_variants))
            )
            handler_desc_slot_tables.append(
                tuple(local_desc_slots.get(i, ()) for i in range(num_variants))
            )
        phase_local_transport_positions[phase_name] = tuple(handler_tables)
        phase_local_desc_slots[phase_name] = tuple(handler_desc_slot_tables)

    return (
        op_phase_tma_args,
        phase_tma_names,
        kernel_tma_arg_names,
        op_phase_transport_indices,
        phase_transport_records,
        phase_local_transport_positions,
        phase_local_desc_slots,
        handler_rebind_specs,
    )


def make_switch_dispatch_callable(
    *,
    fn_name: str,
    phase_name: str,
    binder_fns: Dict[int, Any],
    handler_ids: List[int],
    use_handler_local_idx: bool,
):
    """Return a noinline JIT callable that emits a raw-MLIR switch."""
    if len(handler_ids) == 1:
        only_handler_id = handler_ids[0]
        if use_handler_local_idx:
            @cute.jit
            def _dispatch(*args):
                handler_local_idx = args[1]
                binder_rest = args[2:]
                binder_fns[only_handler_id](handler_local_idx, *binder_rest)
                return None
        else:
            @cute.jit
            def _dispatch(*args):
                binder_rest = args[1:]
                binder_fns[only_handler_id](*binder_rest)
                return None

        _dispatch.__name__ = fn_name
        _dispatch._noinline = True
        _dispatch._machete_switch_dispatch = True
        _dispatch._machete_phase_name = phase_name
        return _dispatch

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


def group_uses_handler_local_idx_from_transport(
    *,
    handler_local_ids: List[int],
    handler_local_transport_positions: Tuple[Tuple[int, ...], ...],
) -> bool:
    """Return whether a tensorless handler group still needs local dispatch."""
    unique_positions = {
        tuple(handler_local_transport_positions[local_id])
        for local_id in set(handler_local_ids)
    }
    return len(unique_positions) != 1


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
    accept_transport_selector: bool,
):
    """Return a callable that emits a local-index switch for one handler group."""
    all_arg_names = list(phase_tensor_names) + list(phase_tma_names)
    arg_index_by_name = {name: idx for idx, name in enumerate(all_arg_names)}
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

    def _arg_positions(arg_names):
        return tuple(arg_index_by_name[name] for name in arg_names)

    direct_tensor_positions = _arg_positions(direct_tensor_args) if direct_tensor_args is not None else ()
    direct_tma_positions = _arg_positions(direct_tma_args) if direct_tma_args is not None else ()
    op_arg_positions = {
        op_idx: (
            _arg_positions(op_phase_tensor_args[op_idx]),
            _arg_positions(op_phase_tma_args[op_idx]),
        )
        for op_idx in unique_op_indices
    }

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
            if accept_transport_selector:
                cursor += 1

            phase_args = args[cursor:]

            if use_direct_call:
                call_args = [page_ptr, *tile_vals, op_config_ptr]
                if is_load:
                    call_args.extend([work_mbar, inner_iter_idx])
                call_args.extend(phase_args[pos] for pos in direct_tensor_positions)
                call_args.extend(phase_args[pos] for pos in direct_tma_positions)
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
                    tensor_positions, tma_positions = op_arg_positions[op_idx]
                    call_args = [page_ptr, *tile_vals, op_config_ptr]
                    if is_load:
                        call_args.extend([work_mbar, inner_iter_idx])
                    call_args.extend(phase_args[pos] for pos in tensor_positions)
                    call_args.extend(phase_args[pos] for pos in tma_positions)
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
            if accept_transport_selector:
                cursor += 1

            phase_args = args[cursor:]

            call_args = [page_ptr, *tile_vals, op_config_ptr]
            if is_load:
                call_args.extend([work_mbar, inner_iter_idx])
            call_args.extend(phase_args[pos] for pos in direct_tensor_positions)
            call_args.extend(phase_args[pos] for pos in direct_tma_positions)
            phase_fn(*call_args)
            return None

    _binder.__name__ = binder_name
    _binder._machete_phase_name = phase_name
    _binder._machete_handler_idx = handler_idx
    _binder._machete_uses_handler_local_idx = not use_direct_call
    return _binder


def make_transport_record_binder(
    *,
    binder_name: str,
    phase_name: str,
    handler_idx: int,
    handler_local_ids: List[int],
    handler_local_transport_positions: Tuple[Tuple[int, ...], ...],
    is_load: bool,
    phase_fn,
    accept_handler_local_idx: bool,
    accept_transport_selector: bool,
    accept_desc_slot_selector: bool,
    handler_local_desc_slots: Tuple[Tuple[int, ...], ...],
    handler_selector_base: int,
    selector_width: int,
    desc_slot_selector_base: int,
    desc_slot_width: int,
):
    """Return a binder for tensorless phases driven by transport records."""
    unique_local_ids = sorted(set(handler_local_ids))
    local_transport_positions = {
        local_id: handler_local_transport_positions[local_id]
        for local_id in unique_local_ids
    }
    local_desc_slots = {
        local_id: handler_local_desc_slots[local_id]
        for local_id in unique_local_ids
    }

    def _select_phase_arg(selected_pos, candidate_positions, phase_args):
        if len(candidate_positions) == 1:
            return phase_args[candidate_positions[0]]
        template = phase_args[candidate_positions[0]]
        template_vals = _flatten_ir_values(template)
        result_types = [v.type for v in template_vals]
        switch_idx = arith.IndexCastOp(ir.IndexType.get(), selected_pos.value).result
        switch_op = scf.IndexSwitchOp(result_types, switch_idx, list(candidate_positions), len(candidate_positions))

        default_block = switch_op.regions[0].blocks.append()
        with ir.InsertionPoint(default_block):
            scf.YieldOp(template_vals)

        for region_idx, pos in enumerate(candidate_positions, start=1):
            case_block = switch_op.regions[region_idx].blocks.append()
            with ir.InsertionPoint(case_block):
                vals = _flatten_ir_values(phase_args[pos])
                scf.YieldOp(vals)
        selected_results = list(switch_op.results)
        rebuilt, _ = _rebuild(template, selected_results, 0)
        return rebuilt

    local_arg_count = max((len(pos) for pos in local_transport_positions.values()), default=0)
    candidate_positions_by_slot = tuple(
        tuple(
            dict.fromkeys(
                positions[slot]
                for positions in local_transport_positions.values()
                if slot < len(positions)
            )
        )
        for slot in range(local_arg_count)
    )
    desc_slot_count = max((len(slots) for slots in local_desc_slots.values()), default=0)
    candidate_desc_slots_by_slot = tuple(
        tuple(
            dict.fromkeys(
                desc_slots[slot]
                for desc_slots in local_desc_slots.values()
                if slot < len(desc_slots)
            )
        )
        for slot in range(desc_slot_count)
    )
    direct_transport_positions = tuple(
        positions[0] if len(positions) == 1 else None
        for positions in candidate_positions_by_slot
    )
    direct_desc_slots = tuple(
        slots[0] if len(slots) == 1 else None
        for slots in candidate_desc_slots_by_slot
    )
    use_direct_call = (
        all(pos is not None for pos in direct_transport_positions)
        and all(slot is not None for slot in direct_desc_slots)
    )

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
            selector_ptr = None
            if accept_transport_selector:
                selector_ptr = args[cursor]
                cursor += 1
            desc_slot_selector_ptr = None
            if accept_desc_slot_selector:
                desc_slot_selector_ptr = args[cursor]
                cursor += 1

            phase_args = args[cursor:]

            call_args = [page_ptr, *tile_vals, op_config_ptr]
            if is_load:
                call_args.extend([work_mbar, inner_iter_idx])
            if use_direct_call:
                call_args.extend(
                    phase_args[pos] for pos in direct_transport_positions
                )
                call_args.extend(Int32(slot) for slot in direct_desc_slots)
                phase_fn(*call_args)
                return None
            selector_base = Int32(handler_selector_base) + handler_local_idx * Int32(selector_width)
            for slot, candidate_positions in enumerate(candidate_positions_by_slot):
                if len(candidate_positions) == 1:
                    selected_pos = Int32(candidate_positions[0])
                else:
                    selected_pos = ld_global_i32(selector_ptr, selector_base + Int32(slot))
                call_args.append(_select_phase_arg(selected_pos, candidate_positions, phase_args))
            if accept_desc_slot_selector:
                desc_slot_base = Int32(desc_slot_selector_base) + handler_local_idx * Int32(desc_slot_width)
                for slot in range(desc_slot_count):
                    if direct_desc_slots[slot] is not None:
                        call_args.append(Int32(direct_desc_slots[slot]))
                    else:
                        call_args.append(ld_global_i32(desc_slot_selector_ptr, desc_slot_base + Int32(slot)))
            phase_fn(*call_args)
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
            if accept_transport_selector:
                cursor += 1
            if accept_desc_slot_selector:
                cursor += 1

            phase_args = args[cursor:]

            call_args = [page_ptr, *tile_vals, op_config_ptr]
            if is_load:
                call_args.extend([work_mbar, inner_iter_idx])
            for slot, candidate_positions in enumerate(candidate_positions_by_slot):
                call_args.append(phase_args[direct_transport_positions[slot]])
            if desc_slot_count:
                call_args.extend(Int32(slot) for slot in direct_desc_slots)
            phase_fn(*call_args)
            return None

    _binder.__name__ = binder_name
    _binder._machete_phase_name = phase_name
    _binder._machete_handler_idx = handler_idx
    _binder._machete_uses_handler_local_idx = not use_direct_call
    _binder._machete_transport_record_binder = True
    return _binder


def build_exec_dispatch_fn(
    *,
    phase_fns,
    phase_name,
    op_handler_indices,
    op_phase_local_indices,
    op_phase_tensor_args,
    op_phase_tma_args,
    phase_handler_local_transport_positions,
    phase_handler_local_desc_slots,
    all_canonical,
    all_tma_canonical=None,
):
    """Build a two-level dispatch: handler tree plus per-handler local binders."""
    is_load = phase_name == "load"

    handler_to_ops: Dict[int, List[int]] = {}
    for op_idx_const, handler_idx in enumerate(op_handler_indices):
        handler_to_ops.setdefault(handler_idx, []).append(op_idx_const)

    handler_uses_local_idx = {}
    phase_uses_runtime_transport_selector = False
    phase_uses_desc_slot_selector = False
    for handler_idx, op_indices in handler_to_ops.items():
        handler_local_ids = [op_phase_local_indices[i] for i in op_indices]
        group_tensor_sigs = {tuple(op_phase_tensor_args[op_idx]) for op_idx in op_indices}
        if group_tensor_sigs == {()}:
            handler_uses_transport_selector = group_uses_handler_local_idx_from_transport(
                handler_local_ids=handler_local_ids,
                handler_local_transport_positions=phase_handler_local_transport_positions[handler_idx],
            )
            handler_uses_desc_slot_selector = len(
                {
                    tuple(phase_handler_local_desc_slots[handler_idx][local_id])
                    for local_id in set(handler_local_ids)
                }
            ) != 1
            handler_uses_local_idx[handler_idx] = (
                handler_uses_transport_selector or handler_uses_desc_slot_selector
            )
            if handler_uses_transport_selector:
                phase_uses_runtime_transport_selector = True
            if handler_uses_desc_slot_selector:
                phase_uses_desc_slot_selector = True
        else:
            handler_uses_local_idx[handler_idx] = group_uses_handler_local_idx(
                handler_local_ids=handler_local_ids,
                op_indices=op_indices,
                op_phase_tensor_args=op_phase_tensor_args,
                op_phase_tma_args=op_phase_tma_args,
            )
    phase_uses_handler_local_idx = any(handler_uses_local_idx.values())
    selector_width = max(
        (len(positions) for handler_positions in phase_handler_local_transport_positions for positions in handler_positions),
        default=0,
    )
    desc_slot_width = max(
        (len(slots) for handler_slots in phase_handler_local_desc_slots for slots in handler_slots),
        default=0,
    )
    handler_selector_bases: Dict[int, int] = {}
    running_selector_base = 0
    for handler_idx, handler_positions in enumerate(phase_handler_local_transport_positions):
        handler_selector_bases[handler_idx] = running_selector_base
        running_selector_base += len(handler_positions) * selector_width
    handler_desc_slot_selector_bases: Dict[int, int] = {}
    running_desc_slot_base = 0
    for handler_idx, handler_slots in enumerate(phase_handler_local_desc_slots):
        handler_desc_slot_selector_bases[handler_idx] = running_desc_slot_base
        running_desc_slot_base += len(handler_slots) * desc_slot_width

    handler_ids = list(range(len(phase_fns)))
    binder_fns = {}
    for handler_idx, op_indices in handler_to_ops.items():
        handler_local_ids = [op_phase_local_indices[i] for i in op_indices]
        group_tensor_sigs = {tuple(op_phase_tensor_args[op_idx]) for op_idx in op_indices}
        if group_tensor_sigs == {()}:
            binder_fns[handler_idx] = make_transport_record_binder(
                binder_name=f"_bind_{handler_idx}",
                phase_name=phase_name,
                handler_idx=handler_idx,
                handler_local_ids=handler_local_ids,
                handler_local_transport_positions=phase_handler_local_transport_positions[handler_idx],
                is_load=is_load,
                phase_fn=phase_fns[handler_idx],
                accept_handler_local_idx=phase_uses_handler_local_idx,
                accept_transport_selector=phase_uses_runtime_transport_selector,
                accept_desc_slot_selector=phase_uses_desc_slot_selector,
                handler_local_desc_slots=phase_handler_local_desc_slots[handler_idx],
                handler_selector_base=handler_selector_bases[handler_idx],
                selector_width=selector_width,
                desc_slot_selector_base=handler_desc_slot_selector_bases[handler_idx],
                desc_slot_width=desc_slot_width,
            )
        else:
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
                accept_transport_selector=phase_uses_runtime_transport_selector,
            )
    return make_switch_dispatch_callable(
        fn_name=f"dispatch_{phase_name}_switch",
        phase_name=phase_name,
        binder_fns=binder_fns,
        handler_ids=handler_ids,
        use_handler_local_idx=phase_uses_handler_local_idx,
    ), phase_uses_handler_local_idx, phase_uses_runtime_transport_selector, phase_uses_desc_slot_selector


def compile_phase_dispatch_inputs(backend, kernel, *, num_dma_warps: int, phase_should_noinline) -> Dict[str, Any]:
    """Compile handler phase functions and build runtime dispatch inputs."""
    ops = kernel.ops
    threads_per_block = kernel.config.threads_per_block
    num_compute_threads = threads_per_block - num_dma_warps * 32
    num_mma_warps = num_compute_threads // 32

    all_canonical = backend.all_canonical(kernel)
    phase_tensor_names = {phase: [] for phase in PHASE_NAMES}
    phase_op_tensor_args = {phase: [[] for _ in ops] for phase in PHASE_NAMES}
    has_communicate = kernel._peer_tma_registry.has_peer_tma
    op_weights = [spec.weight for spec in backend.ir.op_specs]
    compile_keys = backend.compile_keys()
    op_handler_indices = backend.handler_indices()
    op_phase_local_indices = {
        phase: backend.phase_local_indices(phase)
        for phase in PHASE_NAMES
    }
    (
        op_tma_args,
        phase_tma_names,
        kernel_tma_arg_names,
        op_phase_transport_indices,
        phase_transport_records,
        phase_local_transport_positions,
        phase_local_desc_slots,
        handler_rebind_specs,
    ) = _build_tma_runtime_layout(backend, kernel)
    load_fns: List[Any] = [None] * len(backend.ir.handler_specs)
    compute_fns: List[Any] = [None] * len(backend.ir.handler_specs)
    store_fns: List[Any] = [None] * len(backend.ir.handler_specs)
    communicate_fns: List[Any] = [None] * len(backend.ir.handler_specs)
    inner_iters_by_handler: List[int] = [1] * len(backend.ir.handler_specs)
    handler_warps: List[int] = [num_mma_warps] * len(backend.ir.handler_specs)
    phase_fn_lists = {"load": load_fns, "compute": compute_fns, "store": store_fns}
    if has_communicate:
        phase_fn_lists["communicate"] = communicate_fns
    ops_by_handler: Dict[int, List[int]] = {}
    for op_idx, handler_idx in enumerate(op_handler_indices):
        ops_by_handler.setdefault(handler_idx, []).append(op_idx)

    for i, op in enumerate(ops):
        handler_idx = op_handler_indices[i]
        if load_fns[handler_idx] is None:
            kernel_config = {"threads_per_row": num_compute_threads}
            instance = op.op_cls(**build_op_config(op, kernel_config=kernel_config))
            handler_spec = backend.ir.handler_specs[handler_idx]
            inner_iters_by_handler[handler_idx] = getattr(instance, "inner_iters", 1)
            handler_warps[handler_idx] = getattr(instance, "num_mma_warps", num_mma_warps)
            instance._machete_tma_rebind_specs = handler_rebind_specs.get(
                handler_idx,
                {phase: [] for phase in PHASE_NAMES},
            )
            handler_op_indices = ops_by_handler[handler_idx]
            for phase_name, fn_list in phase_fn_lists.items():
                handler_phase_rebinds = handler_rebind_specs.get(handler_idx, {}).get(phase_name, [])
                handler_local_ids = [
                    op_phase_local_indices[phase_name][op_idx]
                    for op_idx in handler_op_indices
                ]
                handler_uses_local_idx = group_uses_handler_local_idx(
                    handler_local_ids=handler_local_ids,
                    op_indices=handler_op_indices,
                    op_phase_tensor_args=phase_op_tensor_args[phase_name],
                    op_phase_tma_args=op_tma_args[phase_name],
                )
                effective_rebinds = handler_phase_rebinds
                handler_phase_tma_params = []
                seen_handler_tma_params = set()
                for spec in handler_phase_rebinds:
                    for name in (
                        spec["desc_pool_name"],
                        spec["wrapper_atom_name"],
                        spec["runtime_tensor_name"],
                    ):
                        if name in seen_handler_tma_params:
                            continue
                        seen_handler_tma_params.add(name)
                        handler_phase_tma_params.append(name)
                for spec in handler_phase_rebinds:
                    if spec["desc_slot_name"] in seen_handler_tma_params:
                        continue
                    seen_handler_tma_params.add(spec["desc_slot_name"])
                    handler_phase_tma_params.append(spec["desc_slot_name"])
                handler_phase_tma_mapping = {
                    name: name
                    for name in handler_spec.local_tma_args[phase_name]
                }
                instance._machete_tma_rebind_specs[phase_name] = effective_rebinds
                phase_noinline = phase_should_noinline(instance, phase_name)
                compiled_fn = compile_phase(
                    instance,
                    phase_name,
                    tensor_param_names=list(handler_spec.local_tensor_names[phase_name]),
                    tma_param_names=handler_phase_tma_params,
                    tma_local_mapping=handler_phase_tma_mapping,
                    noinline=phase_noinline,
                    reconstruct_tensors=True,
                    extra_reconstruct_tensor_names=[],
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

    def _build_dispatch(phase_fns, phase_name):
        return build_exec_dispatch_fn(
            phase_fns=phase_fns,
            phase_name=phase_name,
            op_handler_indices=op_handler_indices,
            op_phase_local_indices=op_phase_local_indices[phase_name],
            op_phase_tensor_args=phase_op_tensor_args[phase_name],
            op_phase_tma_args=op_tma_args[phase_name],
            phase_handler_local_transport_positions=phase_local_transport_positions[phase_name],
            phase_handler_local_desc_slots=phase_local_desc_slots[phase_name],
            all_canonical=phase_tensor_names[phase_name],
            all_tma_canonical=phase_tma_names[phase_name],
        )

    dispatch_load, load_uses_local_idx, load_uses_runtime_selector, load_uses_desc_slot_selector = _build_dispatch(load_fns, "load")
    dispatch_compute, compute_uses_local_idx, compute_uses_runtime_selector, compute_uses_desc_slot_selector = _build_dispatch(compute_fns, "compute")
    dispatch_store, store_uses_local_idx, store_uses_runtime_selector, store_uses_desc_slot_selector = _build_dispatch(store_fns, "store")
    if has_communicate:
        dispatch_communicate, communicate_uses_local_idx, communicate_uses_runtime_selector, communicate_uses_desc_slot_selector = _build_dispatch(
            communicate_fns, "communicate"
        )
    else:
        dispatch_communicate = None
        communicate_uses_local_idx = False
        communicate_uses_runtime_selector = False
        communicate_uses_desc_slot_selector = False

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
        "phase_uses_runtime_transport_selector": {
            "load": load_uses_runtime_selector,
            "compute": compute_uses_runtime_selector,
            "store": store_uses_runtime_selector,
            "communicate": communicate_uses_runtime_selector,
        },
        "phase_uses_desc_slot_selector": {
            "load": load_uses_desc_slot_selector,
            "compute": compute_uses_desc_slot_selector,
            "store": store_uses_desc_slot_selector,
            "communicate": communicate_uses_desc_slot_selector,
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
        "op_phase_transport_indices": op_phase_transport_indices,
        "all_canonical": all_canonical,
        "all_tma_canonical": kernel_tma_arg_names,
        "phase_tensor_names": phase_tensor_names,
        "phase_tma_names": phase_tma_names,
    }


__all__ = [
    "build_exec_dispatch_fn",
    "compile_phase_dispatch_inputs",
    "group_uses_handler_local_idx",
    "group_uses_handler_local_idx_from_transport",
    "make_local_switch_binder",
    "make_switch_dispatch_callable",
]
