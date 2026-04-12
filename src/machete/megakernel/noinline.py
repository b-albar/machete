# Copyright (c) 2025, Machete Authors
"""Noinline device function support for megakernel compute phases.

When enabled, each op's compute function is emitted as a separate MLIR
device function with the ``noinline`` attribute. This reduces register
pressure by preventing the LLVM optimizer from merging all op compute
bodies into a single monolithic function.

CuTe-specific MLIR types appear directly in the ``func.func`` signature.
The CuTe legalization passes handle type conversion for both ``gpu.func``
and ``func.func``.

TMA atoms require special handling: the ``make_exec_tma`` MLIR op can
only be legalized inside ``gpu.func``, so we pre-execute it in the
caller (which IS inside ``gpu.func``) and pass the exec'd value through
the ``func.call`` boundary. A custom trait (``_PreExecTMATrait``) sets
runtime fields (mbarrier pointer) via ``atom_set_value`` inside the
callee without needing ``make_exec_tma``.

Requires ``opt_level <= 2`` — at opt-level 3 the LLVM inliner ignores
the ``noinline`` attribute.

Usage::

    config = MegakernelConfig(noinline=True)
    kernel = Megakernel(ops, config=config)
    kernel.run()
"""

import copy as copy_mod

from cutlass._mlir import ir
from cutlass._mlir.dialects import func as func_dialect
from cutlass.base_dsl.dsl import (
    BaseDSL,
    DSLSingletonMeta,
    extract_mlir_values,
    new_from_mlir_values,
)
from cutlass.cute.atom import Trait
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

_noinline_counter = 0
_orig_func = None
_noinline_func_cache = {}
_last_noinline_stats = {}

# TMA atom field names for atom_set_value (from CuTe DSL copy.py).
_TMA_MBAR_PTR_FIELD = "tma_bar"
_TMA_DESC_PTR_FIELD = "tma_descriptor_ptr"
_TMA_CACHE_POLICY_FIELD = "cache_policy"


class _PreExecTMATrait(Trait):
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


def _flatten_ir_values(obj):
    """Recursively extract a flat list of ir.Value from a CuTe DSL object.

    Unlike ``extract_mlir_values`` which may return CuTe wrappers (e.g.
    _ComposedLayout) mixed with ir.Values, this function guarantees every
    element in the returned list is a raw ``ir.Value``.
    """
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
    """Pre-execute non-exec TMA atoms in caller scope (inside gpu.func).

    For supported non-exec TMA CopyAtom args, calls ``make_exec_tma`` at the
    current insertion point (gpu.func body) and replaces the trait with
    ``_PreExecTMATrait``.

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
            new_atom = copy_mod.copy(a)
            new_atom._trait = _PreExecTMATrait(
                exec_value,
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
            ("non_mlir", type(mapping[1]).__name__, id(mapping[1]))
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
    global _orig_func, _noinline_counter, _noinline_func_cache, _last_noinline_stats
    if _orig_func is not None:
        return  # already installed
    _orig_func = BaseDSL._func
    _noinline_counter = 0
    _noinline_func_cache = {}
    _last_noinline_stats = {}
    BaseDSL._func = _patched_func


def uninstall():
    """Restore the original ``BaseDSL._func``."""
    global _orig_func, _noinline_func_cache, _last_noinline_stats
    if _orig_func is None:
        return
    per_func_body = {}
    for key in _noinline_func_cache:
        func_body_id = key[1]
        per_func_body[func_body_id] = per_func_body.get(func_body_id, 0) + 1
    _last_noinline_stats = {
        "cache_entries": len(_noinline_func_cache),
        "unique_func_bodies": len(per_func_body),
        "per_func_body": per_func_body,
    }
    BaseDSL._func = _orig_func
    _orig_func = None
    _noinline_func_cache = {}
