# Copyright (c) 2025, Machete Authors
"""Handler-based backend for the persistent megakernel."""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Tuple

import cutlass.cute as cute
from cutlass import Int32, Int64
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, scf

from .backend_ir import BackendIR, HandlerSpec, OpCompileSpec
from .compile import compile_phase, exec_generated_source
from .ops import build_op_config


NUM_DMA_WARPS = 3


def _get_local_tensor_names(op_cls, tensor_mapping: Dict[str, str]) -> Tuple[str, ...]:
    """Return op-local tensor names in declaration order."""
    if not hasattr(op_cls, "_UNIQUE_TENSORS"):
        return ()
    return tuple(name for name, _, _ in op_cls._UNIQUE_TENSORS if name in tensor_mapping)


def _get_local_phase_tensor_names(
    op_obj,
    phase_name: str,
    tensor_mapping: Dict[str, str],
) -> Tuple[str, ...]:
    """Return the op-local tensor names actually consumed by one phase."""
    if not tensor_mapping:
        return ()
    method = getattr(op_obj, phase_name, None)
    if method is None:
        return ()
    method_params = inspect.signature(method).parameters
    return tuple(
        name
        for name, _, _ in getattr(type(op_obj), "_UNIQUE_TENSORS", ())
        if name in tensor_mapping and name in method_params
    )


def _get_local_tma_args(op_obj, phase_name: str, tma_mapping: Dict[str, str]) -> Tuple[str, ...]:
    """Return op-local TMA names in phase method declaration order."""
    if not tma_mapping:
        return ()
    method = getattr(op_obj, phase_name, None)
    if method is None:
        return ()
    return tuple(
        name
        for name in inspect.signature(method).parameters
        if name != "self" and name in tma_mapping
    )


def _tensor_reconstruction_specs(op, local_tensor_names: Tuple[str, ...]) -> List[Dict[str, Any]]:
    """Describe how to rebuild local CuTe tensor views from op_config_ptr."""
    if not local_tensor_names:
        return []

    ptr_index_by_name = {
        name: idx
        for idx, (name, _dtype, _dims) in enumerate(getattr(op.op_cls, "_UNIQUE_TENSORS", ()))
    }
    specs: List[Dict[str, Any]] = []
    for local_name in local_tensor_names:
        meta = op.tensor_metas.get(local_name)
        if meta is None or local_name not in ptr_index_by_name:
            continue
        specs.append(
            {
                "canonical_name": local_name,
                "ptr_index": ptr_index_by_name[local_name],
                "dtype_expr": f"_dtype_{local_name}",
                "dtype_obj": meta.dtype,
                "shape": meta.shape,
                "strides": meta.strides,
            }
        )
    return specs


def _make_switch_dispatch_callable(
    *,
    fn_name: str,
    phase_name: str,
    binder_fns: Dict[int, Any],
    handler_ids: List[int],
):
    """Return a noinline JIT callable that emits a raw-MLIR switch."""

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

    _dispatch.__name__ = fn_name
    _dispatch._noinline = True
    _dispatch._machete_switch_dispatch = True
    _dispatch._machete_phase_name = phase_name
    return _dispatch


def _make_local_switch_binder(
    *,
    binder_name: str,
    phase_name: str,
    handler_idx: int,
    handler_local_ids: List[int],
    op_indices: List[int],
    op_specs,
    phase_tensor_names: List[str],
    phase_tma_names: List[str],
    is_load: bool,
    phase_fn,
):
    """Return a callable that emits a local-index switch for one handler group."""

    all_arg_names = list(phase_tensor_names) + list(phase_tma_names)

    def _binder(*args):
        if ir.InsertionPoint.current is None:
            return None

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

        named_args = {
            name: value for name, value in zip(all_arg_names, args[cursor:])
        }

        switch_idx = arith.IndexCastOp(ir.IndexType.get(), handler_local_idx.value).result
        switch_op = scf.IndexSwitchOp([], switch_idx, handler_local_ids, len(handler_local_ids))

        default_block = switch_op.regions[0].blocks.append()
        with ir.InsertionPoint(default_block):
            scf.YieldOp([])

        for region_idx, op_idx in enumerate(op_indices, start=1):
            case_block = switch_op.regions[region_idx].blocks.append()
            with ir.InsertionPoint(case_block):
                spec = op_specs[op_idx]
                call_args = [page_ptr, *tile_vals, op_config_ptr]
                if is_load:
                    call_args.extend([work_mbar, inner_iter_idx])
                call_args.extend(named_args[name] for name in spec.tensor_args[phase_name])
                call_args.extend(named_args[name] for name in spec.tma_args[phase_name])
                phase_fn(*call_args)
                scf.YieldOp([])
        return None

    _binder.__name__ = binder_name
    _binder._machete_phase_name = phase_name
    _binder._machete_handler_idx = handler_idx
    return _binder


def build_handler_backend_ir(kernel) -> BackendIR:
    """Build backend IR from the current megakernel state."""
    registry = kernel._tensor_registry
    tma_registry = kernel._tma_registry
    peer_tma_registry = kernel._peer_tma_registry

    op_specs: List[OpCompileSpec] = []
    handler_specs: List[HandlerSpec] = []
    op_handler_indices: List[int] = []
    op_handler_local_indices: List[int] = []
    handler_index_by_key: Dict[Tuple[Any, ...], int] = {}
    next_local_idx_by_handler: Dict[int, int] = {}
    num_compute_threads = kernel.config.threads_per_block - NUM_DMA_WARPS * 32

    for i, op in enumerate(kernel.ops):
        instance = op.op_cls(**build_op_config(op, kernel_config={"threads_per_row": num_compute_threads}))
        tensor_mapping = registry.op_mappings[i]
        all_local_tensor_names = _get_local_tensor_names(op.op_cls, tensor_mapping)
        local_tensor_names = {
            phase: _get_local_phase_tensor_names(instance, phase, tensor_mapping)
            for phase in ("load", "compute", "store", "communicate")
        }
        tensor_args = {
            phase: tuple(
                tensor_mapping[name]
                for name in local_tensor_names[phase]
            )
            for phase in ("load", "compute", "store", "communicate")
        }
        phase_mappings = {
            "load": tma_registry.op_mappings.get((i, "load"), {}),
            "compute": tma_registry.op_mappings.get((i, "compute"), {}),
            "store": tma_registry.op_mappings.get((i, "store"), {}),
            "communicate": peer_tma_registry.op_mappings.get((i, "communicate"), {}),
        }
        local_tma_args = {
            phase: _get_local_tma_args(instance, phase, phase_mappings[phase])
            for phase in ("load", "compute", "store", "communicate")
        }
        tma_args = {
            phase: tuple(phase_mappings[phase][name] for name in local_tma_args[phase])
            for phase in ("load", "compute", "store", "communicate")
        }

        static_dims_key = tuple(sorted(op.static_dims.items())) if op.static_dims else ()
        dtypes_key = (
            tuple(sorted((k, v.__name__) for k, v in op.tensor_dtypes.items()))
            if op.tensor_dtypes else ()
        )
        strides_key = (
            tuple(sorted((k, v) for k, v in op.tensor_strides.items()))
            if op.tensor_strides else ()
        )
        compile_key = (
            op.op_cls,
            static_dims_key,
            dtypes_key,
            strides_key,
            all_local_tensor_names,
            tuple(local_tensor_names[ph] for ph in ("load", "compute", "store", "communicate")),
            tuple(local_tma_args[ph] for ph in ("load", "compute", "store", "communicate")),
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
        op_handler_local_indices.append(next_local_idx_by_handler.get(handler_idx, 0))
        next_local_idx_by_handler[handler_idx] = op_handler_local_indices[-1] + 1

    return BackendIR(
        op_specs=tuple(op_specs),
        handler_specs=tuple(handler_specs),
        op_handler_indices=tuple(op_handler_indices),
        op_handler_local_indices=tuple(op_handler_local_indices),
    )


class HandlerBackend:
    """Backend that dispatches by unique handler index rather than raw op index."""

    def __init__(self, ir: BackendIR):
        self.ir = ir

    def compile_keys(self) -> List[Any]:
        return [spec.compile_key for spec in self.ir.op_specs]

    def handler_indices(self) -> List[int]:
        return list(self.ir.op_handler_indices)

    def handler_local_indices(self) -> List[int]:
        return list(self.ir.op_handler_local_indices)

    def all_canonical(self, kernel) -> List[str]:
        seen = set()
        names: List[str] = []
        for spec in self.ir.op_specs:
            for phase in ("load", "compute", "store", "communicate"):
                for name in spec.tensor_args[phase]:
                    if name in seen:
                        continue
                    seen.add(name)
                    names.append(name)
        return names

    def all_tma_canonical(self, kernel) -> List[str]:
        return (
            kernel._tma_registry.all_canonical_names
            + kernel._peer_tma_registry.all_canonical_names
        )

    def op_tensor_args(self) -> List[List[str]]:
        tensor_args: List[List[str]] = []
        for spec in self.ir.op_specs:
            seen = set()
            names: List[str] = []
            for phase in ("load", "compute", "store", "communicate"):
                for name in spec.tensor_args[phase]:
                    if name in seen:
                        continue
                    seen.add(name)
                    names.append(name)
            tensor_args.append(names)
        return tensor_args

    def phase_op_tensor_args(self) -> Dict[str, List[List[str]]]:
        return {
            phase: [list(spec.tensor_args[phase]) for spec in self.ir.op_specs]
            for phase in ("load", "compute", "store", "communicate")
        }

    def op_tma_args(self) -> Dict[str, List[List[str]]]:
        return {
            phase: [list(spec.tma_args[phase]) for spec in self.ir.op_specs]
            for phase in ("load", "compute", "store", "communicate")
        }

    def phase_tensor_names(self) -> Dict[str, List[str]]:
        """Return the canonical tensor names actually used by each phase."""
        phase_names: Dict[str, List[str]] = {}
        for phase in ("load", "compute", "store", "communicate"):
            seen = set()
            names: List[str] = []
            for spec in self.ir.op_specs:
                for name in spec.tensor_args[phase]:
                    if name in seen:
                        continue
                    seen.add(name)
                    names.append(name)
            phase_names[phase] = names
        return phase_names

    def phase_tma_names(self) -> Dict[str, List[str]]:
        """Return the canonical TMA names actually used by each phase."""
        phase_names: Dict[str, List[str]] = {}
        for phase in ("load", "compute", "store", "communicate"):
            seen = set()
            names: List[str] = []
            for spec in self.ir.op_specs:
                for name in spec.tma_args[phase]:
                    if name in seen:
                        continue
                    seen.add(name)
                    names.append(name)
            phase_names[phase] = names
        return phase_names

    def compile_phase_dispatch_inputs(self, kernel) -> Dict[str, Any]:
        """Compile phase wrappers once per unique handler and build handler dispatch."""
        ops = kernel.ops
        threads_per_block = kernel.config.threads_per_block
        num_compute_threads = threads_per_block - NUM_DMA_WARPS * 32
        num_mma_warps = num_compute_threads // 32

        all_canonical = self.all_canonical(kernel)
        all_tma_canonical = self.all_tma_canonical(kernel)
        phase_tensor_names = self.phase_tensor_names()
        phase_tma_names = self.phase_tma_names()
        phase_op_tensor_args = self.phase_op_tensor_args()
        op_tma_args = self.op_tma_args()
        op_weights = [spec.weight for spec in self.ir.op_specs]
        compile_keys = self.compile_keys()
        op_handler_indices = self.handler_indices()
        op_handler_local_indices = self.handler_local_indices()

        load_fns: List[Any] = [None] * len(self.ir.handler_specs)
        compute_fns: List[Any] = [None] * len(self.ir.handler_specs)
        store_fns: List[Any] = [None] * len(self.ir.handler_specs)
        communicate_fns: List[Any] = [None] * len(self.ir.handler_specs)
        inner_iters_by_handler: List[int] = [1] * len(self.ir.handler_specs)
        handler_warps: List[int] = [num_mma_warps] * len(self.ir.handler_specs)
        per_op_load_uses_config: List[bool] = []
        per_op_compute_uses_config: List[bool] = []
        per_op_store_uses_config: List[bool] = []
        per_op_communicate_uses_config: List[bool] = []

        phase_fn_lists = {
            "load": load_fns,
            "compute": compute_fns,
            "store": store_fns,
            "communicate": communicate_fns,
        }

        for i, op in enumerate(ops):
            handler_idx = op_handler_indices[i]
            if load_fns[handler_idx] is None:
                kernel_config = {"threads_per_row": num_compute_threads}
                instance = op.op_cls(**build_op_config(op, kernel_config=kernel_config))
                handler_spec = self.ir.handler_specs[handler_idx]
                inner_iters_by_handler[handler_idx] = getattr(instance, "inner_iters", 1)
                handler_warps[handler_idx] = getattr(instance, "num_mma_warps", num_mma_warps)

                for phase_name, fn_list in phase_fn_lists.items():
                    local_tma_args = list(handler_spec.local_tma_args[phase_name])
                    phase_noinline = True
                    if (
                        phase_name == "load"
                        and type(instance).__name__ == "GemmOp"
                    ):
                        # After inlining the other thin store/compute wrappers,
                        # the next single-kernel decode blocker is the GEMM
                        # load wrapper. Keep the phase inline and let the real
                        # transport work stay inside the op body.
                        phase_noinline = False
                    if (
                        phase_name == "load"
                        and type(instance).__name__ == "GLUOp"
                    ):
                        # After inlining GEMM load, the next single-kernel
                        # decode blocker is the GLU load wrapper.
                        phase_noinline = False
                    if (
                        phase_name == "load"
                        and type(instance).__name__ == "FlashAttentionSm120Op"
                    ):
                        # After inlining GLU load, the next single-kernel
                        # decode blocker is the FlashAttention load wrapper.
                        phase_noinline = False
                    if (
                        phase_name == "load"
                        and type(instance).__name__ == "QKNormRopeOp"
                    ):
                        # After inlining FlashAttention load, the next
                        # single-kernel decode blocker is the QKNormRope load
                        # wrapper.
                        phase_noinline = False
                    if (
                        phase_name == "load"
                        and type(instance).__name__ == "RMSNormOp"
                    ):
                        # After inlining QKNormRope load, the next single-kernel
                        # decode blocker is the RMSNorm load wrapper.
                        phase_noinline = False
                    if (
                        phase_name == "compute"
                        and type(instance).__name__ == "GemmOp"
                        and getattr(getattr(instance, phase_name), "__name__", "") == "compute_unscaled"
                    ):
                        # The real work is already outlined into
                        # _gemm_compute_unscaled_core(). Keeping the thin phase
                        # wrapper itself noinline only adds another large
                        # device-function boundary with no register-pressure
                        # benefit.
                        phase_noinline = False
                    if (
                        phase_name == "compute"
                        and type(instance).__name__ == "GLUOp"
                    ):
                        # GLU forward compute is outlined into _glu_forward_core().
                        # Keep the phase wrapper inline and noinline only the
                        # actual helper body.
                        phase_noinline = False
                    if (
                        phase_name == "compute"
                        and type(instance).__name__ == "FlashAttentionSm120Op"
                    ):
                        # The cooperative FA compute body is currently the
                        # blocking noinline symbol for larger decode megakernels.
                        # Keep the phase inline so it no longer has to fit into
                        # a separate noinline LLVM/NVVM function boundary.
                        phase_noinline = False
                    if (
                        phase_name == "compute"
                        and type(instance).__name__ == "QKNormRopeOp"
                    ):
                        # QKNormRope is another small compute phase whose
                        # noinline function boundary becomes the next NVVM
                        # blocker in larger decode megakernels.
                        phase_noinline = False
                    if (
                        phase_name == "compute"
                        and type(instance).__name__ == "RMSNormOp"
                    ):
                        # RMSNorm compute is the current noinline compile wall
                        # on larger single-kernel decode graphs. Keep the phase
                        # inline for now; unlike the failed tensor-ABI
                        # compaction attempt, this does not change semantics.
                        phase_noinline = False
                    if (
                        phase_name == "store"
                        and type(instance).__name__ == "GemmOp"
                    ):
                        # After inlining RMSNorm compute, the next single-kernel
                        # decode blocker is the noinline GEMM store wrapper.
                        # Keep store inline for now; the method body is already
                        # a thin TMA store shell.
                        phase_noinline = False
                    if (
                        phase_name == "store"
                        and type(instance).__name__ == "GLUOp"
                    ):
                        # After inlining GEMM store, the next large decode
                        # blocker is the GLU store wrapper. Keep this thin TMA
                        # store inline as well.
                        phase_noinline = False
                    if (
                        phase_name == "store"
                        and type(instance).__name__ == "FlashAttentionSm120Op"
                    ):
                        # The next single-kernel decode blocker after GLU store
                        # is the FlashAttention store wrapper.
                        phase_noinline = False
                    if (
                        phase_name == "store"
                        and type(instance).__name__ == "QKNormRopeOp"
                    ):
                        # The next single-kernel decode blocker after
                        # FlashAttention store is the QKNormRope store wrapper.
                        # Keep this thin store shell inline as well.
                        phase_noinline = False
                    if (
                        phase_name == "store"
                        and type(instance).__name__ == "RMSNormOp"
                    ):
                        # After inlining QKNormRope store, the next single-kernel
                        # decode blocker is the RMSNorm store wrapper.
                        phase_noinline = False
                    compiled_fn = compile_phase(
                        instance,
                        phase_name,
                        tensor_param_names=list(handler_spec.local_tensor_names[phase_name]),
                        tma_param_names=local_tma_args,
                        tma_local_mapping={name: name for name in local_tma_args},
                        noinline=phase_noinline,
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

            per_op_load_uses_config.append(bool(getattr(load_fns[handler_idx], "_uses_op_config_ptr", True)))
            per_op_compute_uses_config.append(bool(getattr(compute_fns[handler_idx], "_uses_op_config_ptr", True)))
            per_op_store_uses_config.append(bool(getattr(store_fns[handler_idx], "_uses_op_config_ptr", True)))
            per_op_communicate_uses_config.append(
                bool(getattr(communicate_fns[handler_idx], "_uses_op_config_ptr", True))
            )

        def _build_dispatch(phase_fns, phase_name):
            return self._build_exec_dispatch_fn(
                phase_fns,
                phase_name,
                self.ir.op_specs,
                op_handler_indices,
                op_handler_local_indices,
                phase_op_tensor_args[phase_name],
                phase_tensor_names[phase_name],
                phase_tma_names[phase_name],
                op_weights,
            )

        return {
            "dispatch_load": _build_dispatch(load_fns, "load"),
            "dispatch_compute": _build_dispatch(compute_fns, "compute"),
            "dispatch_store": _build_dispatch(store_fns, "store"),
            "dispatch_communicate": _build_dispatch(communicate_fns, "communicate"),
            "inner_iters_list": [inner_iters_by_handler[h] for h in op_handler_indices],
            "has_communicate": kernel._peer_tma_registry.has_peer_tma,
            "per_op_warps": [handler_warps[h] for h in op_handler_indices],
            "per_op_load_uses_config": per_op_load_uses_config,
            "per_op_compute_uses_config": per_op_compute_uses_config,
            "per_op_store_uses_config": per_op_store_uses_config,
            "per_op_communicate_uses_config": per_op_communicate_uses_config,
            "op_tensor_args": phase_op_tensor_args,
            "op_tma_args": op_tma_args,
            "op_weights": op_weights,
            "compile_keys": compile_keys,
            "op_handler_indices": op_handler_indices,
            "op_handler_local_indices": op_handler_local_indices,
            "all_canonical": all_canonical,
            "all_tma_canonical": all_tma_canonical,
            "phase_tensor_names": phase_tensor_names,
            "phase_tma_names": phase_tma_names,
        }

    def _build_exec_dispatch_fn(
        self,
        phase_fns,
        phase_name,
        op_specs,
        op_handler_indices,
        op_handler_local_indices,
        op_phase_tensor_args,
        all_canonical,
        all_tma_canonical=None,
        op_weights=None,
    ):
        """Build a two-level dispatch: handler tree + per-handler op binder."""
        is_load = phase_name == "load"
        tensor_params = ", ".join(all_canonical)
        tile_params = ", ".join(f"tile_{i}" for i in range(5))
        tma_params = ", ".join(all_tma_canonical) if all_tma_canonical else ""

        handler_to_ops: Dict[int, List[int]] = {}
        for op_idx_const, handler_idx in enumerate(op_handler_indices):
            handler_to_ops.setdefault(handler_idx, []).append(op_idx_const)

        tensor_sig = f", {tensor_params}" if tensor_params else ""
        tma_sig = f", {tma_params}" if tma_params else ""

        fn_name = f"dispatch_{phase_name}_switch"
        handler_ids = list(range(len(phase_fns)))
        binder_fns = {}
        for handler_idx, op_indices in handler_to_ops.items():
            handler_local_ids = [op_handler_local_indices[i] for i in op_indices]
            binder_fns[handler_idx] = _make_local_switch_binder(
                binder_name=f"_bind_{handler_idx}",
                phase_name=phase_name,
                handler_idx=handler_idx,
                handler_local_ids=handler_local_ids,
                op_indices=op_indices,
                op_specs=op_specs,
                phase_tensor_names=all_canonical,
                phase_tma_names=all_tma_canonical or [],
                is_load=is_load,
                phase_fn=phase_fns[handler_idx],
            )
        return _make_switch_dispatch_callable(
            fn_name=fn_name,
            phase_name=phase_name,
            binder_fns=binder_fns,
            handler_ids=handler_ids,
        )


__all__ = [
    "build_handler_backend_ir",
    "HandlerBackend",
]
