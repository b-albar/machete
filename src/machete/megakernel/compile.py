# Copyright (c) 2025, Machete Authors
"""
Phase Function Compilation for Megakernel.

Compiles an Op's pipelined phase methods (load, compute, store) into
@cute.jit wrapper functions. The wrapper delegates to the Op instance's
method, mapping positional tile indices to named parameters.

Usage:
    from machete.megakernel.compile import compile_compute

    config = build_op_config(scheduled_op, kernel_config=kernel_config)
    instance = MyOp(**config)
    tile_fn = compile_compute(instance, tensor_param_names=['t0', 't1'])

    # Returns @cute.jit function with signature:
    #   fn(page_ptr: Int32, tile_0..tile_4: Int32,
    #      op_config_ptr: Int64, t0, t1) -> None
"""

import ast
import inspect
import linecache
import textwrap

import cutlass.cute as cute
from cutlass import Int32, Int64

# Counter for unique filenames in linecache
_compile_counter = 0

# Track linecache entries for cleanup
_linecache_entries: list = []


# =============================================================================
# Source Extraction
# =============================================================================


def _extract_body(fn):
    """Extract a function's body as dedented source code.

    Uses inspect.getsource + AST parsing to reliably strip decorators
    (@staticmethod, @abstractmethod), the def line, and class-level
    indentation — returning just the executable body.
    """
    source = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(source)

    func_node = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_node = node
            break

    if func_node is None:
        raise ValueError(f"Could not find function def in source of {fn}")

    lines = source.splitlines()
    body_start = func_node.body[0].lineno - 1  # 0-indexed
    body_end = func_node.body[-1].end_lineno
    return textwrap.dedent("\n".join(lines[body_start:body_end]))


# =============================================================================
# Phase Compilation (Class-Based)
# =============================================================================


def _build_phase_wrapper(
    instance,
    phase_name,
    tensor_param_names=None,
    extra_params=None,
    append_mbar=False,
    filename="<compile_phase>",
    tma_param_names=None,
    tma_local_mapping=None,
):
    """Build a @cute.jit wrapper that delegates to an Op instance method.

    The wrapper has the standard dispatch signature (page_ptr, tile_0..4,
    op_config_ptr, [work_mbar,] t0, t1, ..., tma0_atom, tma0_gmem, ...)
    and maps positional tile indices, canonical tensor names, and canonical
    TMA names to the instance method's named parameters.

    The instance method signature convention:
        def phase(self, page_ptr, tile_M, [tile_D, ...], x, [weight, ...], y,
                  [x_tma, x_tma_gmem, ...], [work_mbar]):

    CuTe DSL traces through the wrapper into the real instance method,
    which is in a real .py file and has full source access.

    Args:
        instance: Op instance with config stored as self attributes.
        phase_name: Method name ('load', 'compute', 'store', etc.).
        tensor_param_names: Canonical tensor parameter names for dispatch.
        extra_params: Extra params in wrapper signature (e.g., ['work_mbar']).
        append_mbar: If True, append mbarrier_arrive(work_mbar) after call.
        filename: Base filename for linecache registration.
        tma_param_names: Canonical TMA parameter names for dispatch signature.
        tma_local_mapping: Dict mapping local TMA names (e.g., 'x_tma') to
            canonical names (e.g., 'tma0_atom'). Used to map method params.
    """
    global _compile_counter
    _compile_counter += 1
    unique_filename = f"{filename}_{_compile_counter}"

    op_cls = instance.__class__
    method = getattr(instance, phase_name)
    method_params = set(inspect.signature(method).parameters.keys())
    method_params_ordered = list(inspect.signature(method).parameters.keys())

    # Build instance method call arguments:
    # page_ptr, then tile indices (mapped to positional tile_i), then tensors,
    # then TMA params, then special params (op_config_ptr, work_mbar).
    # Only pass args that the method signature actually accepts (base Op
    # default methods only take page_ptr).
    call_args = ["page_ptr"]

    # Map tile dims to positional tile_i params
    if hasattr(op_cls, "_TILE_DIM_NAMES_ORDERED"):
        for dim_name in op_cls._TILE_DIM_NAMES_ORDERED:
            if f"tile_{dim_name}" in method_params:
                axis = op_cls.DIM_NAMES[dim_name]
                call_args.append(f"tile_{axis}")
    else:
        # Raw op (no reads/writes/tile declarations): detect tile_N params
        # from method signature and pass them through positionally.
        for i in range(5):
            if f"tile_{i}" in method_params:
                call_args.append(f"tile_{i}")

    # Build reverse TMA mapping: local_name -> canonical_name
    tma_reverse = tma_local_mapping or {}

    # Add tensor params only if method expects them (has params beyond
    # page_ptr, tile_*, op_config_ptr, work_mbar, and TMA params)
    if tensor_param_names:
        known_special = {"page_ptr", "op_config_ptr", "work_mbar"}
        tma_local_names = set(tma_reverse.keys())
        expects_tensors = any(
            p not in known_special
            and not p.startswith("tile_")
            and p not in tma_local_names
            for p in method_params
        )
        if expects_tensors:
            call_args.extend(tensor_param_names)

    # Add TMA params: detect local TMA names in method signature and map
    # to canonical names. Preserve method signature order.
    if tma_reverse:
        for p in method_params_ordered:
            if p in tma_reverse:
                call_args.append(tma_reverse[p])

    # Check if method expects op_config_ptr
    if "op_config_ptr" in method_params:
        call_args.append("op_config_ptr")

    # Check if method expects work_mbar
    if "work_mbar" in method_params and extra_params and "work_mbar" in extra_params:
        call_args.append("work_mbar")

    call_str = ", ".join(call_args)

    # Build wrapper function signature (standard dispatch format)
    tile_params = ", ".join(f"tile_{i}" for i in range(5))
    extra_str = ""
    if extra_params:
        extra_str = ", " + ", ".join(extra_params)
    tensor_str = ""
    if tensor_param_names:
        tensor_str = ", " + ", ".join(tensor_param_names)
    tma_str = ""
    if tma_param_names:
        tma_str = ", " + ", ".join(tma_param_names)

    # For load/store phases: dispatch is called by all DMA warp threads
    # (needed for TMA warp convergence on both G2S loads and S2G stores).
    # Non-TMA ops must be wrapped in elect_one() so only one thread executes.
    # TMA ops handle thread selection internally (elect_one for mbarrier,
    # cute.copy outside for warp-convergent TMA copy).
    is_load_phase = phase_name in ("load", "backward_load")
    is_store_phase = phase_name in ("store", "backward_store")
    has_tma = bool(tma_local_mapping)
    if is_load_phase and not has_tma:
        # Non-TMA loads: elect_one so only one thread issues the G2S copy.
        body = f"    with cute.arch.elect_one():\n"
        body += f"        _instance.{phase_name}({call_str})\n"
        if append_mbar:
            body += f"        mbarrier_arrive(work_mbar)\n"
    else:
        body = f"    _instance.{phase_name}({call_str})\n"
        if append_mbar:
            body += "    mbarrier_arrive(work_mbar)\n"

    fn_source = (
        "@cute.jit\n"
        f"def phase_fn(page_ptr, {tile_params}, op_config_ptr"
        f"{extra_str}{tensor_str}{tma_str}):\n"
        f"{body}"
    )

    exec_globals = {
        "cute": cute, "Int32": Int32, "Int64": Int64,
        "_instance": instance,
    }
    if append_mbar:
        from .interpreter import mbarrier_arrive
        exec_globals["mbarrier_arrive"] = mbarrier_arrive

    linecache.cache[unique_filename] = (
        len(fn_source), None, fn_source.splitlines(True), unique_filename,
    )
    _linecache_entries.append(unique_filename)

    code = compile(fn_source, unique_filename, "exec")
    exec(code, exec_globals)
    return exec_globals["phase_fn"]


def compile_load(instance, tensor_param_names=None,
                 tma_param_names=None, tma_local_mapping=None):
    """Compile Op's load method.

    Detects async vs sync load from method signature.
    Always includes work_mbar in wrapper signature.
    """
    method = getattr(instance, "load")
    is_async = "work_mbar" in inspect.signature(method).parameters

    if is_async:
        return _build_phase_wrapper(
            instance, "load", tensor_param_names,
            extra_params=["work_mbar"],
            filename="<compile_load>",
            tma_param_names=tma_param_names,
            tma_local_mapping=tma_local_mapping,
        )
    else:
        return _build_phase_wrapper(
            instance, "load", tensor_param_names,
            extra_params=["work_mbar"],
            append_mbar=True,
            filename="<compile_load>",
            tma_param_names=tma_param_names,
            tma_local_mapping=tma_local_mapping,
        )


def compile_compute(instance, tensor_param_names=None,
                    tma_param_names=None, tma_local_mapping=None):
    """Compile Op's compute method."""
    return _build_phase_wrapper(
        instance, "compute", tensor_param_names,
        filename="<compile_compute>",
        tma_param_names=tma_param_names,
        tma_local_mapping=tma_local_mapping,
    )


def compile_store(instance, tensor_param_names=None,
                  tma_param_names=None, tma_local_mapping=None):
    """Compile Op's store method."""
    return _build_phase_wrapper(
        instance, "store", tensor_param_names,
        filename="<compile_store>",
        tma_param_names=tma_param_names,
        tma_local_mapping=tma_local_mapping,
    )


def compile_backward_load(instance, tensor_param_names=None,
                          tma_param_names=None, tma_local_mapping=None):
    """Compile Op's backward_load method."""
    method = getattr(instance, "backward_load")
    is_async = "work_mbar" in inspect.signature(method).parameters

    if is_async:
        return _build_phase_wrapper(
            instance, "backward_load", tensor_param_names,
            extra_params=["work_mbar"],
            filename="<compile_backward_load>",
            tma_param_names=tma_param_names,
            tma_local_mapping=tma_local_mapping,
        )
    else:
        return _build_phase_wrapper(
            instance, "backward_load", tensor_param_names,
            extra_params=["work_mbar"],
            append_mbar=True,
            filename="<compile_backward_load>",
            tma_param_names=tma_param_names,
            tma_local_mapping=tma_local_mapping,
        )


def compile_backward_compute(instance, tensor_param_names=None,
                             tma_param_names=None, tma_local_mapping=None):
    """Compile Op's backward_compute method."""
    return _build_phase_wrapper(
        instance, "backward_compute", tensor_param_names,
        filename="<compile_backward_compute>",
        tma_param_names=tma_param_names,
        tma_local_mapping=tma_local_mapping,
    )


def compile_backward_store(instance, tensor_param_names=None,
                           tma_param_names=None, tma_local_mapping=None):
    """Compile Op's backward_store method."""
    return _build_phase_wrapper(
        instance, "backward_store", tensor_param_names,
        filename="<compile_backward_store>",
        tma_param_names=tma_param_names,
        tma_local_mapping=tma_local_mapping,
    )


# =============================================================================
# Cleanup
# =============================================================================


def cleanup_linecache() -> int:
    """Remove compile-time linecache entries.

    Returns the number of entries removed. Call this to reclaim memory
    after kernels are no longer needed.
    """
    count = 0
    for filename in _linecache_entries:
        if linecache.cache.pop(filename, None) is not None:
            count += 1
    _linecache_entries.clear()
    return count


__all__ = [
    "compile_load",
    "compile_compute",
    "compile_store",
    "compile_backward_load",
    "compile_backward_compute",
    "compile_backward_store",
    "cleanup_linecache",
    # Internals
    "_extract_body",
    "_build_phase_wrapper",
]
