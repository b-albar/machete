# Copyright (c) 2025, Machete Authors
"""Noinline MLIR function support for megakernel phases.

This module monkey-patches a small part of the CuTe DSL stack so megakernel
phases can cross ``func.call`` boundaries without losing TMA support. The key
piece is runtime-descriptor TMA handling: instead of threading large exec-TMA
objects through every noinline call, we keep the non-exec atom plus a typed
descriptor pointer and rebuild the exec form locally in the callee.

The patch is intentionally local to Machete. It avoids modifying site-packages
while still giving the handler backend a practical noinline path for large
kernels.
"""

import copy as copy_mod

from cutlass._mlir import ir
from cutlass._mlir.dialects import cute_nvgpu as cn
from cutlass._mlir.dialects import func as func_dialect
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.base_dsl.dsl import (
    BaseDSL,
    DSLSingletonMeta,
    extract_mlir_values,
    new_from_mlir_values,
)
from cutlass.cute import atom
from cutlass.cute.nvgpu.cpasync.copy import (
    CopyBulkTensorTileG2SNonExecTrait,
    CopyBulkTensorTileG2SMulticastNonExecTrait,
    CopyBulkTensorTileS2GNonExecTrait,
    CopyReduceBulkTensorTileS2GNonExecTrait,
)
from cutlass._mlir.dialects._cute_nvgpu_ops_gen import (
    atom_make_exec_tma,
    atom_set_value,
)
from .transport import (
    RuntimeDescTMATrait,
    make_runtime_desc_tma_atom,
    runtime_desc_ptr_type,
)

_noinline_counter = 0
_orig_func = None
_orig_make_tiled_tma_atom = None
_noinline_func_cache = {}

# TMA atom field names for atom_set_value (from CuTe DSL copy.py).
_TMA_MBAR_PTR_FIELD = "tma_bar"
_TMA_DESC_PTR_FIELD = "tma_descriptor_ptr"
_TMA_CACHE_POLICY_FIELD = "cache_policy"


def _non_mlir_cache_key(value):
    """Return a stable cache key for non-MLIR noinline arguments."""
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return ("lit", value)
    if isinstance(value, tuple):
        return ("tuple", tuple(_non_mlir_cache_key(v) for v in value))
    if isinstance(value, list):
        return ("list", tuple(_non_mlir_cache_key(v) for v in value))
    if isinstance(value, type):
        return ("type", value.__module__, value.__qualname__)
    return ("obj", type(value).__name__, id(value))


class _PreExecTMATrait(atom.Trait):
    """TMA trait wrapping a pre-exec'd value.

    ``make_exec_tma`` was already called in the caller scope (inside
    ``gpu.func`` where it legalizes). This trait's ``unpack()`` only
    sets runtime fields (mbarrier pointer, etc.) via ``atom_set_value``.
    """

    def __init__(self, value, *, field_namespace: str, supports_mbar: bool):
        self.value = value
        self.field_namespace = field_namespace
        self.supports_mbar = supports_mbar

    def __new_from_mlir_values__(self, values):
        return self.__class__(
            values[0],
            field_namespace=self.field_namespace,
            supports_mbar=self.supports_mbar,
        )

    def unpack(self, *, tma_bar_ptr=None, tma_desc_ptr=None,
               cache_policy=None, loc=None, ip=None, **kwargs):
        exec_value = self.value
        if self.supports_mbar and tma_bar_ptr is not None:
            attr = ir.Attribute.parse(
                f"#cute_nvgpu.atom_copy_field_{self.field_namespace}<{_TMA_MBAR_PTR_FIELD}>")
            exec_value = atom_set_value(
                exec_value, attr, tma_bar_ptr.value, loc=loc, ip=ip)
        if tma_desc_ptr is not None:
            attr = ir.Attribute.parse(
                f"#cute_nvgpu.atom_copy_field_{self.field_namespace}<{_TMA_DESC_PTR_FIELD}>")
            exec_value = atom_set_value(
                exec_value, attr, tma_desc_ptr.value, loc=loc, ip=ip)
        if cache_policy is not None:
            attr = ir.Attribute.parse(
                f"#cute_nvgpu.atom_copy_field_{self.field_namespace}<{_TMA_CACHE_POLICY_FIELD}>")
            exec_value = atom_set_value(
                exec_value, attr, cache_policy.value, loc=loc, ip=ip)
        return exec_value

@dsl_user_op
def _patched_make_tiled_tma_atom(
    op,
    gmem_tensor,
    smem_layout,
    cta_tiler,
    num_multicast=1,
    *,
    internal_type=None,
    loc=None,
    ip=None,
):
    """Create TMA atoms that already carry a runtime descriptor pointer.

    CuTe's default `make_tiled_tma_atom()` returns non-exec TMA traits whose
    `unpack()` path still calls `atom_make_exec_tma`. For large megakernels and
    wrapper-local TMA recreation that legalization point is too restrictive.

    This monkey patch keeps the standard non-exec atom for `tma_partition`, but
    also materializes a typed descriptor pointer directly with
    `cute_nvgpu.make_tma_desc_tiled`. The returned trait can then build
    `atom_make_tma_*` directly without ever invoking `atom_make_exec_tma`.
    """
    try:
        return make_runtime_desc_tma_atom(
            op,
            gmem_tensor,
            smem_layout,
            cta_tiler,
            num_multicast=num_multicast,
            internal_type=internal_type,
            loc=loc,
            ip=ip,
        )
    except TypeError:
        pass

    return _orig_make_tiled_tma_atom(
        op,
        gmem_tensor,
        smem_layout,
        cta_tiler,
        num_multicast=num_multicast,
        internal_type=internal_type,
        loc=loc,
        ip=ip,
    )


def _install_runtime_desc_tma_patch():
    global _orig_make_tiled_tma_atom
    if _orig_make_tiled_tma_atom is not None:
        return
    import cutlass.cute as cute
    import cutlass.cute.nvgpu.cpasync.helpers as cpasync_helpers

    _orig_make_tiled_tma_atom = cpasync_helpers.make_tiled_tma_atom
    cpasync_helpers.make_tiled_tma_atom = _patched_make_tiled_tma_atom
    cute.nvgpu.cpasync.make_tiled_tma_atom = _patched_make_tiled_tma_atom


def _uninstall_runtime_desc_tma_patch():
    global _orig_make_tiled_tma_atom
    if _orig_make_tiled_tma_atom is None:
        return
    import cutlass.cute as cute
    import cutlass.cute.nvgpu.cpasync.helpers as cpasync_helpers

    cpasync_helpers.make_tiled_tma_atom = _orig_make_tiled_tma_atom
    cute.nvgpu.cpasync.make_tiled_tma_atom = _orig_make_tiled_tma_atom
    _orig_make_tiled_tma_atom = None


def _flatten_ir_values(obj):
    """Recursively extract a flat list of ir.Value from a CuTe DSL object.

    Unlike ``extract_mlir_values`` which may return CuTe wrappers (e.g.
    _ComposedLayout) mixed with ir.Values, this function guarantees every
    element in the returned list is a raw ``ir.Value``.
    """
    if isinstance(obj, type):
        return []
    if isinstance(obj, ir.Value):
        return [obj]
    values = extract_mlir_values(obj)
    result = []
    for v in values:
        result.extend(_flatten_ir_values(v))
    return result


def _rebuild(template, flat_values, offset):
    """Reconstruct a CuTe DSL value from flat ir.Values using the original as template.

    Returns (reconstructed_object, next_offset).

    Mirrors the structure of ``extract_mlir_values``: for each item in
    the mixed list, if it was an ir.Value we consume one flat value; if it
    was a wrapper we recurse into it and then call ``new_from_mlir_values``
    to reconstruct the wrapper.
    """
    if isinstance(template, ir.Value):
        return flat_values[offset], offset + 1
    mixed = extract_mlir_values(template)
    if not mixed:
        return template, offset
    reconstructed_parts = []
    for part in mixed:
        rebuilt, offset = _rebuild(part, flat_values, offset)
        reconstructed_parts.append(rebuilt)
    return new_from_mlir_values(template, reconstructed_parts), offset


def _pre_exec_tma_args(args):
    """Prepare TMA args in caller scope (inside gpu.func).

    For supported non-exec TMA CopyAtom args, we still materialize an exec atom
    in caller scope, but only to derive a typed runtime descriptor pointer.
    The callee receives the original non-exec atom plus this descriptor pointer
    and rebuilds the exec TMA atom locally with raw NVGPU ops.

    Returns a new args list; original args are not modified.
    """
    new_args = list(args)
    for i, a in enumerate(new_args):
        trait = getattr(a, '_trait', None)
        field_namespace = None
        supports_mbar = False
        if isinstance(trait, (CopyBulkTensorTileG2SNonExecTrait, CopyBulkTensorTileG2SMulticastNonExecTrait)):
            field_namespace = "tmaload"
            supports_mbar = True
        elif isinstance(trait, CopyBulkTensorTileS2GNonExecTrait):
            field_namespace = "tmastore"
        elif isinstance(trait, CopyReduceBulkTensorTileS2GNonExecTrait):
            field_namespace = "tmareduce"

        if field_namespace is not None:
            exec_value = atom_make_exec_tma(trait.value)
            # Use a generic typed pointer here. Lowering from a byval kernel
            # argument-backed descriptor field to addrspace(1) leaves an
            # unrealized conversion cast in LLVM translation, while generic
            # typed descriptor pointers lower cleanly and are accepted by the
            # raw NVGPU ``atom_make_tma_*`` ops.
            ptr_type = runtime_desc_ptr_type()
            desc_ptr = cn.get_tma_desc_addr(ptr_type, exec_value)
            new_atom = copy_mod.copy(a)
            new_atom._trait = RuntimeDescTMATrait(
                trait.value,
                desc_ptr,
                field_namespace=field_namespace,
                supports_mbar=supports_mbar,
            )
            new_args[i] = new_atom
    return new_args


def _emit_noinline_call(funcBody, args, kwargs):
    """Emit a ``func.func`` with ``noinline`` and a ``func.call`` to it.

    MLIR-trackable arguments are deeply flattened to pure ``ir.Value``
    objects for the ``func.call`` operands, then reconstructed inside
    the callee using the modified argument as a structural template.

    CuTe types pass through directly in the func.func signature — the
    CuTe legalization handles type conversion. TMA atoms are pre-exec'd
    in the caller scope so the callee only needs ``atom_set_value``
    (not ``make_exec_tma``).
    """
    # Pre-exec TMA atoms in caller scope (inside gpu.func where
    # make_exec_tma can be legalized by the CuTe pass pipeline).
    args = _pre_exec_tma_args(args)

    # Build flat list of ir.Values + per-arg mapping
    mlir_args = []       # flat list of ir.Value for func.call
    mlir_types = []      # flat list of ir.Type for func signature
    # Per-arg: ('mlir', start_idx, count) or ('non_mlir', value)
    arg_mapping = []

    for a in args:
        values = _flatten_ir_values(a)
        if values:
            start = len(mlir_args)
            for v in values:
                mlir_args.append(v)
                mlir_types.append(v.type)
            arg_mapping.append(("mlir", start, len(values)))
        else:
            arg_mapping.append(("non_mlir", a))

    instances = DSLSingletonMeta._instances
    if not instances:
        return funcBody(*args, **kwargs)
    gpu_mod = next(iter(instances.values())).gpu_module
    if gpu_mod is None:
        return funcBody(*args, **kwargs)

    gpu_body_block = gpu_mod.bodyRegion.blocks[0]
    cache_key = (
        id(gpu_mod),
        id(funcBody),
        tuple(str(t) for t in mlir_types),
        tuple(
            ("non_mlir", _non_mlir_cache_key(mapping[1]))
            if mapping[0] == "non_mlir"
            else ("mlir", mapping[1], mapping[2])
            for mapping in arg_mapping
        ),
        tuple(sorted(kwargs.items())),
    )

    fn_name = _noinline_func_cache.get(cache_key)
    if fn_name is None:
        global _noinline_counter
        _noinline_counter += 1
        fn_name = f"_noinline_{funcBody.__name__}_{_noinline_counter}"
        func_type = ir.FunctionType.get(mlir_types, [])

        with ir.InsertionPoint.at_block_begin(gpu_body_block):
            fn_op = func_dialect.FuncOp(fn_name, func_type)
            fn_op.attributes["sym_visibility"] = ir.StringAttr.get("private")
            fn_op.attributes["passthrough"] = ir.ArrayAttr.get(
                [ir.StringAttr.get("noinline")]
            )
            entry_block = fn_op.add_entry_block()
            with ir.InsertionPoint(entry_block):
                # Block args have original types — no bridging needed
                block_args = list(entry_block.arguments)

                # Reconstruct Python-level args from block args
                reconstructed_args = []
                for i, mapping in enumerate(arg_mapping):
                    if mapping[0] == "non_mlir":
                        reconstructed_args.append(mapping[1])
                    else:
                        _, start, count = mapping
                        block_vals = block_args[start:start + count]
                        rebuilt, _ = _rebuild(args[i], block_vals, 0)
                        reconstructed_args.append(rebuilt)
                funcBody(*reconstructed_args, **kwargs)
                func_dialect.ReturnOp([])
        _noinline_func_cache[cache_key] = fn_name

    func_dialect.CallOp([], ir.FlatSymbolRefAttr.get(fn_name), mlir_args)
    return None


def _patched_func(self, funcBody, *args, **kwargs):
    """Replacement for ``BaseDSL._func`` that intercepts noinline-marked functions."""
    if ir.Context.current is None:
        pass
    elif ir.InsertionPoint.current is not None:
        if getattr(funcBody, "_noinline", False):
            return _emit_noinline_call(funcBody, args, kwargs)
    return _orig_func(self, funcBody, *args, **kwargs)


def install():
    """Install the noinline patch on ``BaseDSL._func``."""
    global _orig_func, _noinline_counter, _noinline_func_cache
    if _orig_func is not None:
        return  # already installed
    _orig_func = BaseDSL._func
    _noinline_counter = 0
    _noinline_func_cache = {}
    BaseDSL._func = _patched_func
    _install_runtime_desc_tma_patch()


def uninstall():
    """Restore the original ``BaseDSL._func``."""
    global _orig_func, _noinline_func_cache
    if _orig_func is None:
        return
    BaseDSL._func = _orig_func
    _orig_func = None
    _noinline_func_cache = {}
    _uninstall_runtime_desc_tma_patch()
