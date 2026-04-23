# Copyright (c) 2025, Machete Authors
"""Backend implementations for the persistent megakernel."""

from __future__ import annotations

import inspect
from functools import lru_cache
from dataclasses import replace
from typing import Any, Dict, List, Tuple

from .backend_ir import BackendIR, HandlerSpec, OpCompileSpec
from .backend_dispatch import compile_phase_dispatch_inputs
from .ops import build_op_config


NUM_DMA_WARPS = 3
PHASE_NAMES = ("load", "compute", "store", "communicate")


def _phase_should_noinline(instance, phase_name: str) -> bool:
    """Return whether one phase should stay behind a noinline boundary.

    The handler backend defaults to noinline for every phase. A small set of
    thin wrapper phases are forced inline because they repeatedly became the
    dominant overhead in small GEMM-like kernels while adding little useful
    isolation. The heavy lifting either already lives in deeper helpers or is
    small enough that outlining only adds dispatch shell cost.
    """
    op_name = type(instance).__name__
    compute_name = getattr(getattr(instance, phase_name, None), "__name__", "")

    inline_phase_ops = {
        "load": {
            "GemmOp",
            "GemmSm100Op",
            "GLUOp",
            "FlashAttentionSm120Op",
            "QKNormRopeOp",
            "RMSNormOp",
        },
        "store": {
            "GemmOp",
            "GemmSm100Op",
            "GLUOp",
            "FlashAttentionSm120Op",
            "QKNormRopeOp",
            "RMSNormOp",
        },
        "compute": {
            "GemmOp",
            "GemmSm100Op",
            "GLUOp",
            "GLUBwdOp",
            "FlashAttentionSm120Op",
            "QKNormRopeOp",
            "RMSNormOp",
        },
    }

    if op_name in inline_phase_ops.get(phase_name, set()):
        return False
    if phase_name == "compute" and op_name == "GemmOp" and compute_name == "compute_unscaled":
        return False
    return True


def _get_local_tensor_names(op_cls, tensor_mapping: Dict[str, str]) -> Tuple[str, ...]:
    """Return op-local tensor names in declaration order."""
    if not hasattr(op_cls, "_UNIQUE_TENSORS"):
        return ()
    return tuple(name for name, _, _ in op_cls._UNIQUE_TENSORS if name in tensor_mapping)


@lru_cache(maxsize=None)
def _phase_param_names(op_cls, phase_name: str) -> Tuple[str, ...]:
    """Return cached method parameter names for one op class / phase."""
    method = getattr(op_cls, phase_name, None)
    if method is None:
        return ()
    return tuple(name for name in inspect.signature(method).parameters if name != "self")


@lru_cache(maxsize=None)
def _callable_param_names(method_target) -> Tuple[str, ...]:
    """Return cached parameter names for one concrete phase implementation."""
    return tuple(name for name in inspect.signature(method_target).parameters if name != "self")


def _phase_param_names_for_instance(op_obj, phase_name: str) -> Tuple[str, ...]:
    """Return parameter names for the concrete phase bound on one instance.

    Some ops, notably SM120 attention, rebind ``self.compute`` at construction
    time to a more specialized implementation. The backend must follow that
    concrete callable rather than the class-level placeholder method when it
    decides which tensors/TMA args a phase consumes.
    """
    method = getattr(op_obj, phase_name, None)
    if method is None:
        return ()
    method_target = getattr(method, "__func__", method)
    return _callable_param_names(method_target)


def _get_local_phase_tensor_names(
    op_obj,
    phase_name: str,
    tensor_mapping: Dict[str, str],
) -> Tuple[str, ...]:
    """Return the op-local tensor names actually consumed by one phase."""
    if not tensor_mapping:
        return ()
    method_param_names = frozenset(_phase_param_names_for_instance(op_obj, phase_name))
    return tuple(
        name
        for name, _, _ in getattr(type(op_obj), "_UNIQUE_TENSORS", ())
        if name in tensor_mapping and name in method_param_names
    )


def _get_local_tma_args(op_obj, phase_name: str, tma_mapping: Dict[str, str]) -> Tuple[str, ...]:
    """Return op-local TMA names in phase method declaration order."""
    if not tma_mapping:
        return ()
    return tuple(
        name for name in _phase_param_names_for_instance(op_obj, phase_name) if name in tma_mapping
    )


def _phase_reconstructs_all_tensors(instance, phase_tensor_names: Tuple[str, ...]) -> bool:
    """Return whether a phase can rebuild every local tensor from ``op_config_ptr``.

    When this is true, handler-local dispatch does not need to distinguish ops
    by concrete tensor bindings for that phase; the wrapper rebuilds them from
    packed config instead.
    """
    if not phase_tensor_names:
        return True

    op_cls = instance.__class__
    unique_tensors = {
        name: dims for name, _dtype, dims in getattr(op_cls, "_UNIQUE_TENSORS", ())
    }
    ptr_slots = getattr(op_cls, "_CONFIG_PTR_I64_INDEX", {})
    dynamic_offsets = getattr(op_cls, "_CONFIG_DYNAMIC_I32_OFFSET", {})

    for tensor_name in phase_tensor_names:
        dims = unique_tensors.get(tensor_name)
        ptr_slot = ptr_slots.get(tensor_name)
        dtype_attr = f"{tensor_name}_dtype"
        if dims is None or ptr_slot is None or not hasattr(instance, dtype_attr):
            return False
        for dim_name in dims:
            stride_attr = f"{tensor_name}_stride_{dim_name}"
            if not hasattr(instance, stride_attr):
                return False
            if dim_name not in dynamic_offsets and not hasattr(instance, dim_name):
                return False
    return True


def _build_compile_key(
    op,
    *,
    all_local_tensor_names: Tuple[str, ...],
    local_tensor_names: Dict[str, Tuple[str, ...]],
    local_tma_args: Dict[str, Tuple[str, ...]],
    tensor_args: Dict[str, Tuple[str, ...]],
    tma_args: Dict[str, Tuple[str, ...]],
) -> Tuple[Any, ...]:
    """Build the compile-time handler signature for one scheduled op."""
    static_dims_key = tuple(sorted(op.static_dims.items())) if op.static_dims else ()
    dtypes_key = (
        tuple(sorted((name, dtype.__name__) for name, dtype in op.tensor_dtypes.items()))
        if op.tensor_dtypes else ()
    )
    strides_key = (
        tuple(sorted(op.tensor_strides.items()))
        if op.tensor_strides else ()
    )
    return (
        op.op_cls,
        static_dims_key,
        dtypes_key,
        strides_key,
        all_local_tensor_names,
        tuple(local_tensor_names[phase] for phase in PHASE_NAMES),
        tuple(local_tma_args[phase] for phase in PHASE_NAMES),
    )


def build_handler_backend_ir(kernel) -> BackendIR:
    """Build backend IR from the current megakernel state."""
    registry = kernel._tensor_registry
    tma_registry = kernel._tma_registry
    peer_tma_registry = kernel._peer_tma_registry

    op_specs: List[OpCompileSpec] = []
    handler_specs: List[HandlerSpec] = []
    op_handler_indices: List[int] = []
    op_phase_local_indices: Dict[str, List[int]] = {phase: [] for phase in PHASE_NAMES}
    op_phase_transport_indices: Dict[str, List[int]] = {phase: [] for phase in PHASE_NAMES}
    handler_index_by_key: Dict[Tuple[Any, ...], int] = {}
    phase_variant_idx_by_handler: Dict[str, Dict[int, Dict[Tuple[Any, ...], int]]] = {
        phase: {}
        for phase in PHASE_NAMES
    }
    phase_transport_idx: Dict[str, Dict[Tuple[str, ...], int]] = {phase: {} for phase in PHASE_NAMES}
    phase_transport_records: Dict[str, List[Tuple[str, ...]]] = {phase: [] for phase in PHASE_NAMES}
    num_compute_threads = kernel.config.threads_per_block - NUM_DMA_WARPS * 32

    for i, op in enumerate(kernel.ops):
        instance = op.op_cls(**build_op_config(op, kernel_config={"threads_per_row": num_compute_threads}))
        tensor_mapping = registry.op_mappings[i]
        all_local_tensor_names = _get_local_tensor_names(op.op_cls, tensor_mapping)
        local_tensor_names = {
            phase: _get_local_phase_tensor_names(instance, phase, tensor_mapping)
            for phase in PHASE_NAMES
        }
        tensor_args = {
            phase: tuple(tensor_mapping[name] for name in local_tensor_names[phase])
            for phase in PHASE_NAMES
        }
        phase_mappings = {
            "load": tma_registry.op_mappings.get((i, "load"), {}),
            "compute": tma_registry.op_mappings.get((i, "compute"), {}),
            "store": tma_registry.op_mappings.get((i, "store"), {}),
            "communicate": peer_tma_registry.op_mappings.get((i, "communicate"), {}),
        }
        local_tma_args = {
            phase: _get_local_tma_args(instance, phase, phase_mappings[phase])
            for phase in PHASE_NAMES
        }
        phase_reconstructs_all_tensors = {
            phase: _phase_reconstructs_all_tensors(instance, local_tensor_names[phase])
            for phase in PHASE_NAMES
        }
        tma_args = {
            phase: tuple(phase_mappings[phase][name] for name in local_tma_args[phase])
            for phase in PHASE_NAMES
        }
        compile_key = _build_compile_key(
            op,
            all_local_tensor_names=all_local_tensor_names,
            local_tensor_names=local_tensor_names,
            local_tma_args=local_tma_args,
            tensor_args=tensor_args,
            tma_args=tma_args,
        )
        weight = max(1, op.total_tiles)

        op_specs.append(
            OpCompileSpec(
                op_index=i,
                compile_key=compile_key,
                tensor_args=tensor_args,
                local_tensor_names=local_tensor_names,
                tma_args=tma_args,
                local_tma_args=local_tma_args,
                weight=weight,
            )
        )

        handler_idx = handler_index_by_key.get(compile_key)
        if handler_idx is None:
            handler_idx = len(handler_specs)
            handler_index_by_key[compile_key] = handler_idx
            handler_specs.append(
                HandlerSpec(
                    handler_idx=handler_idx,
                    compile_key=compile_key,
                    local_tensor_names=local_tensor_names,
                    local_tma_args=local_tma_args,
                    weight=weight,
                )
            )
        else:
            prev = handler_specs[handler_idx]
            handler_specs[handler_idx] = HandlerSpec(
                handler_idx=prev.handler_idx,
                compile_key=prev.compile_key,
                local_tensor_names=prev.local_tensor_names,
                local_tma_args=prev.local_tma_args,
                weight=prev.weight + weight,
            )

        op_handler_indices.append(handler_idx)
        for phase in PHASE_NAMES:
            transport_key = op_specs[-1].tma_args[phase]
            transport_idx = phase_transport_idx[phase].get(transport_key)
            if transport_idx is None:
                transport_idx = len(phase_transport_records[phase])
                phase_transport_idx[phase][transport_key] = transport_idx
                phase_transport_records[phase].append(transport_key)
            op_phase_transport_indices[phase].append(transport_idx)

            phase_variant_map = phase_variant_idx_by_handler[phase].setdefault(handler_idx, {})
            phase_binding_key = (
                ()
                if phase_reconstructs_all_tensors[phase]
                else op_specs[-1].tensor_args[phase],
                op_specs[-1].tma_args[phase],
            )
            phase_local_idx = phase_variant_map.get(phase_binding_key)
            if phase_local_idx is None:
                # Keep hot-path dispatch keyed by compact handler-local ids.
                # Transport identity is tracked separately in
                # op_phase_transport_indices / phase_transport_records and can
                # later drive a transport-record-based dispatch path without
                # perturbing the current switch-based fast path.
                phase_local_idx = len(phase_variant_map)
                phase_variant_map[phase_binding_key] = phase_local_idx
            op_phase_local_indices[phase].append(phase_local_idx)

    phase_local_transport_positions: Dict[str, Tuple[Tuple[Tuple[int, ...], ...], ...]] = {}
    for phase in PHASE_NAMES:
        phase_names: List[str] = []
        seen_names = set()
        for spec in op_specs:
            for name in spec.tma_args[phase]:
                if name in seen_names:
                    continue
                seen_names.add(name)
                phase_names.append(name)
        arg_index_by_name = {name: idx for idx, name in enumerate(phase_names)}
        handler_variant_tables: List[Tuple[Tuple[int, ...], ...]] = []
        for handler_idx in range(len(handler_specs)):
            variant_map = phase_variant_idx_by_handler[phase].get(handler_idx, {})
            if not variant_map:
                handler_variant_tables.append(())
                continue
            num_variants = max(variant_map.values()) + 1
            variant_positions: List[Tuple[int, ...] | None] = [None] * num_variants
            for (_tensor_args, tma_args), local_idx in variant_map.items():
                variant_positions[local_idx] = tuple(arg_index_by_name[name] for name in tma_args)
            handler_variant_tables.append(
                tuple(() if pos is None else pos for pos in variant_positions)
            )
        phase_local_transport_positions[phase] = tuple(handler_variant_tables)

    return BackendIR(
        op_specs=tuple(op_specs),
        handler_specs=tuple(handler_specs),
        op_handler_indices=tuple(op_handler_indices),
        op_phase_local_indices={k: tuple(v) for k, v in op_phase_local_indices.items()},
        op_phase_transport_indices={k: tuple(v) for k, v in op_phase_transport_indices.items()},
        phase_transport_records={k: tuple(v) for k, v in phase_transport_records.items()},
        phase_local_transport_positions=phase_local_transport_positions,
    )


class HandlerBackend:
    """Backend that dispatches by unique handler index rather than raw op index."""

    def __init__(self, ir: BackendIR):
        self.ir = ir

    def compile_keys(self) -> List[Any]:
        return [spec.compile_key for spec in self.ir.op_specs]

    def handler_indices(self) -> List[int]:
        return list(self.ir.op_handler_indices)

    def phase_local_indices(self, phase_name: str) -> List[int]:
        return list(self.ir.op_phase_local_indices[phase_name])

    def all_canonical(self, kernel) -> List[str]:
        return []

    def compile_phase_dispatch_inputs(self, kernel) -> Dict[str, Any]:
        """Compile phase wrappers and synthesize runtime dispatch objects."""
        return compile_phase_dispatch_inputs(
            self,
            kernel,
            num_dma_warps=NUM_DMA_WARPS,
            phase_should_noinline=_phase_should_noinline,
        )


class RuntimeBackend(HandlerBackend):
    """Runtime-oriented backend facade.

    This backend currently reuses the handler IR and dispatch implementation
    while the framework is being moved away from hard-wired handler-specialized
    construction. Keeping it as a distinct backend makes backend selection
    explicit and lets tests lock the public contract before the runtime path
    diverges internally.
    """
    runtime_transport_records = True


def build_runtime_backend_ir(kernel) -> BackendIR:
    """Build backend IR for the runtime backend.

    The current runtime backend starts from the same IR as the handler path.
    Future work can relax handler specialization without changing the
    megakernel constructor or tests.
    """
    base_ir = build_handler_backend_ir(kernel)
    return replace(
        base_ir,
        op_phase_local_indices={
            phase: tuple(base_ir.op_phase_transport_indices[phase])
            for phase in PHASE_NAMES
        },
    )


def build_backend(kernel, backend_name: str):
    """Construct the configured backend and its IR."""
    if backend_name == "handler":
        ir = build_handler_backend_ir(kernel)
        return ir, HandlerBackend(ir)
    if backend_name == "runtime":
        ir = build_runtime_backend_ir(kernel)
        return ir, RuntimeBackend(ir)
    raise ValueError(f"Unknown megakernel backend: {backend_name}")


__all__ = [
    "build_handler_backend_ir",
    "build_runtime_backend_ir",
    "build_backend",
    "HandlerBackend",
    "RuntimeBackend",
]
