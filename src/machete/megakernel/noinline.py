# Copyright (c) 2025, Machete Authors
"""Noinline device function support for megakernel compute phases.

When enabled, each op's compute function is emitted as a separate MLIR
device function with the ``noinline`` attribute. This reduces register
pressure by preventing the LLVM optimizer from merging all op compute
bodies into a single monolithic function.

Requires ``opt_level <= 2`` — at opt-level 3 the LLVM inliner ignores
the ``noinline`` attribute on ``func.func`` ops inside ``gpu.module``.

Usage::

    config = MegakernelConfig(noinline=True)
    kernel = Megakernel(ops, config=config)
    kernel.run()
"""

from cutlass import Int32, Int64
from cutlass._mlir import ir
from cutlass._mlir.dialects import func as func_dialect
from cutlass.base_dsl.dsl import BaseDSL, DSLSingletonMeta
from cutlass.cute.tensor import _Tensor as CoreTensor

_noinline_counter = 0
_orig_func = None


def _get_ir_value(arg):
    """Extract the underlying ``ir.Value`` from a CuTe DSL value."""
    if isinstance(arg, (Int32, Int64)):
        return arg.ir_value()
    elif type(arg).__name__ == "_Tensor":
        return arg.value
    elif isinstance(arg, ir.Value):
        return arg
    else:
        raise TypeError(f"Cannot extract ir.Value from {type(arg).__name__}")


def _wrap_block_arg(block_arg, original_arg):
    """Wrap a block argument back into the original CuTe DSL type."""
    if isinstance(original_arg, Int32):
        return Int32(block_arg)
    elif isinstance(original_arg, Int64):
        return Int64(block_arg)
    elif type(original_arg).__name__ == "_Tensor":
        return CoreTensor(block_arg, original_arg._dtype)
    else:
        return block_arg


def _emit_noinline_call(funcBody, args, kwargs):
    """Emit a ``func.func`` with ``noinline`` and a ``func.call`` to it.

    MLIR-trackable arguments are passed through ``func.call``.
    Non-MLIR arguments (e.g. CopyAtom) are captured via closure.
    """
    global _noinline_counter
    _noinline_counter += 1
    fn_name = f"_noinline_{funcBody.__name__}_{_noinline_counter}"

    mlir_args = []
    mlir_types = []
    non_mlir_map = {}

    for i, a in enumerate(args):
        try:
            ir_val = _get_ir_value(a)
            mlir_args.append(ir_val)
            mlir_types.append(ir_val.type)
        except TypeError:
            non_mlir_map[i] = a

    instances = DSLSingletonMeta._instances
    if not instances:
        return funcBody(*args, **kwargs)
    gpu_mod = next(iter(instances.values())).gpu_module
    if gpu_mod is None:
        return funcBody(*args, **kwargs)

    func_type = ir.FunctionType.get(mlir_types, [])
    gpu_body_block = gpu_mod.bodyRegion.blocks[0]

    with ir.InsertionPoint.at_block_begin(gpu_body_block):
        fn_op = func_dialect.FuncOp(fn_name, func_type)
        fn_op.attributes["sym_visibility"] = ir.StringAttr.get("private")
        fn_op.attributes["passthrough"] = ir.ArrayAttr.get(
            [ir.StringAttr.get("noinline")]
        )
        entry_block = fn_op.add_entry_block()
        with ir.InsertionPoint(entry_block):
            reconstructed_args = []
            mlir_idx = 0
            for i in range(len(args)):
                if i in non_mlir_map:
                    reconstructed_args.append(non_mlir_map[i])
                else:
                    reconstructed_args.append(
                        _wrap_block_arg(entry_block.arguments[mlir_idx], args[i])
                    )
                    mlir_idx += 1
            funcBody(*reconstructed_args, **kwargs)
            func_dialect.ReturnOp([])

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
    global _orig_func, _noinline_counter
    if _orig_func is not None:
        return  # already installed
    _orig_func = BaseDSL._func
    _noinline_counter = 0
    BaseDSL._func = _patched_func


def uninstall():
    """Restore the original ``BaseDSL._func``."""
    global _orig_func
    if _orig_func is None:
        return
    BaseDSL._func = _orig_func
    _orig_func = None
