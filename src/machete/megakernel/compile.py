# Copyright (c) 2025, Machete Authors
"""
Phase Function Compilation for Megakernel.

Compiles an Op's pipelined phase methods (load, compute, store) into
@cute.jit wrapper functions. The wrapper delegates to the Op instance's
method, mapping positional tile indices to named parameters.

Usage:
    from machete.megakernel.compile import compile_phase

    config = build_op_config(scheduled_op, kernel_config=kernel_config)
    instance = MyOp(**config)
    tile_fn = compile_phase(instance, "compute", tensor_param_names=['t0', 't1'])

    # Returns @cute.jit function with signature:
    #   fn(page_ptr: Int32, tile_0..tile_4: Int32,
    #      op_config_ptr: Int64, t0, t1) -> None
"""

import ast
import inspect
import linecache
import textwrap
from typing import Dict, List

import cutlass.cute as cute
from cutlass import Int32, Int64

from .ops import MAX_TILE_DIMS

# Counter for unique filenames in linecache
_compile_counter = 0

# Track linecache entries for cleanup
_linecache_entries: list = []


def exec_generated_source(source: str, label: str, exec_globals: dict) -> dict:
    """Compile and exec generated source code with linecache support.

    Registers the source in linecache so that tracebacks show meaningful
    file names and line numbers for generated code.

    Args:
        source: Python source code string.
        label: Descriptive label for linecache (e.g., "kernel_loop", "dispatch_load").
        exec_globals: Globals dict for exec(). Updated in-place with defined names.

    Returns:
        The exec_globals dict (same object, for convenience).
    """
    global _compile_counter
    _compile_counter += 1
    filename = f"<{label}>_{_compile_counter}"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(True),
        filename,
    )
    _linecache_entries.append(filename)
    code = compile(source, filename, "exec")
    exec(code, exec_globals)
    return exec_globals


def _register_generated_source(source: str, filename: str) -> None:
    """Register generated source in linecache for readable tracebacks."""
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(True),
        filename,
    )
    _linecache_entries.append(filename)


# =============================================================================
# Source Extraction
# =============================================================================


def _extract_body(fn):
    """Extract a function's body as dedented source code.

    Prefers parsing the full source file and locating the exact AST node by
    function name and line number. That is more robust for large nested
    functions than reparsing ``inspect.getsource(fn)`` directly, which can
    occasionally return an invalid snippet when the surrounding definition is
    complex. Falls back to snippet-based parsing when no source file is
    available.
    """
    source_file = inspect.getsourcefile(fn)
    if source_file:
        file_source = inspect.getsource(inspect.getmodule(fn))
        tree = ast.parse(file_source)
        source_lines, start_lineno = inspect.getsourcelines(fn)
        def_lineno = start_lineno
        for offset, line in enumerate(source_lines):
            stripped = line.lstrip()
            if stripped.startswith("def ") or stripped.startswith("async def "):
                def_lineno = start_lineno + offset
                break

        func_node = None
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == fn.__name__
                and node.lineno == def_lineno
            ):
                func_node = node
                break

        if func_node is None:
            raise ValueError(f"Could not find function def for {fn.__name__} at line {def_lineno}")

        lines = file_source.splitlines()
        body_start = func_node.body[0].lineno - 1
        body_end = func_node.body[-1].end_lineno
        return textwrap.dedent("\n".join(lines[body_start:body_end]))

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
    body_start = func_node.body[0].lineno - 1
    body_end = func_node.body[-1].end_lineno
    return textwrap.dedent("\n".join(lines[body_start:body_end]))


def _ordered_method_params(method) -> List[str]:
    """Return method parameters in declared order."""
    return list(inspect.signature(method).parameters.keys())


def _tile_call_args(op_cls, method_params: set) -> List[str]:
    """Build tile-index arguments passed from the wrapper into the phase method."""
    call_args = []
    if hasattr(op_cls, "_TILE_DIM_NAMES_ORDERED"):
        for dim_name in op_cls._TILE_DIM_NAMES_ORDERED:
            tile_param = f"tile_{dim_name}"
            if tile_param in method_params:
                call_args.append(f"tile_{op_cls.DIM_NAMES[dim_name]}")
    else:
        for axis in range(5):
            tile_param = f"tile_{axis}"
            if tile_param in method_params:
                call_args.append(tile_param)
    return call_args


def _tensor_call_args(
    method_params: set,
    tensor_param_names,
    tma_local_mapping: Dict[str, str],
) -> List[str]:
    """Build wrapper tensor arguments if the phase method actually expects tensors."""
    if not tensor_param_names:
        return []

    known_special = {
        "page_ptr",
        "op_config_ptr",
        "work_mbar",
        "inner_iter_idx",
    }
    tma_local_names = set(tma_local_mapping)
    expects_tensors = any(
        param not in known_special
        and not param.startswith("tile_")
        and param not in tma_local_names
        for param in method_params
    )
    return list(tensor_param_names) if expects_tensors else []


def _tma_call_args(method_params_ordered: List[str], tma_local_mapping: Dict[str, str]) -> List[str]:
    """Map local TMA method parameters back to canonical wrapper parameter names."""
    return [tma_local_mapping[param] for param in method_params_ordered if param in tma_local_mapping]


def _wrapper_signature_suffix(is_load_phase, extra_params, tensor_param_names, tma_param_names) -> str:
    """Build the wrapper function signature suffix after op_config_ptr."""
    parts = []
    if extra_params:
        parts.extend(extra_params)
    if is_load_phase:
        parts.append("inner_iter_idx")
    if tensor_param_names:
        parts.extend(dict.fromkeys(tensor_param_names))
    if tma_param_names:
        parts.extend(tma_param_names)
    return f", {', '.join(parts)}" if parts else ""


def _wrapper_body(phase_name, call_str, append_mbar: bool, has_tma: bool) -> str:
    """Build the wrapper body for one phase method."""
    is_load_phase = phase_name == "load"
    is_compute = phase_name == "compute"
    is_store = phase_name in ("store", "communicate")

    if is_load_phase and not has_tma:
        body = "    with cute.arch.elect_one():\n"
        body += f"        _instance.{phase_name}({call_str})\n"
        if append_mbar:
            body += "        mbarrier_arrive(work_mbar)\n"
    else:
        body = f"    _instance.{phase_name}({call_str})\n"
        if append_mbar:
            body += "    mbarrier_arrive(work_mbar)\n"

    if is_compute or is_store:
        body += "    _compiler_fence()\n"
    return body


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
    method_params_ordered = _ordered_method_params(method)
    method_params = set(method_params_ordered)

    # Build instance method call arguments:
    # page_ptr, then tile indices (mapped to positional tile_i), then tensors,
    # then TMA params, then special params (op_config_ptr, work_mbar).
    # Only pass args that the method signature actually accepts (base Op
    # default methods only take page_ptr).
    call_args = ["page_ptr"]
    tma_reverse = tma_local_mapping or {}
    call_args.extend(_tile_call_args(op_cls, method_params))
    call_args.extend(_tensor_call_args(method_params, tensor_param_names, tma_reverse))
    call_args.extend(_tma_call_args(method_params_ordered, tma_reverse))

    uses_op_config_ptr = "op_config_ptr" in method_params
    if uses_op_config_ptr:
        call_args.append("op_config_ptr")

    # Check if method expects special framework params
    if "work_mbar" in method_params and extra_params and "work_mbar" in extra_params:
        call_args.append("work_mbar")

    is_load_phase = phase_name == "load"

    if is_load_phase:
        # inner_iter_idx: store warp passes iteration index for K-block loading
        if "inner_iter_idx" in method_params:
            call_args.append("inner_iter_idx")

    call_str = ", ".join(call_args)

    # Build wrapper function signature (standard dispatch format)
    tile_params = ", ".join(f"tile_{i}" for i in range(MAX_TILE_DIMS))
    signature_suffix = _wrapper_signature_suffix(
        is_load_phase,
        extra_params,
        tensor_param_names,
        tma_param_names,
    )

    # For load/store phases: dispatch is called by all DMA warp threads
    # (needed for TMA warp convergence on both G2S loads and S2G stores).
    # Non-TMA ops must be wrapped in elect_one() so only one thread executes.
    # TMA ops handle thread selection internally (elect_one for mbarrier,
    # cute.copy outside for warp-convergent TMA copy).
    has_tma = bool(tma_local_mapping)
    body = _wrapper_body(phase_name, call_str, append_mbar, has_tma)
    needs_fence = phase_name == "compute" or phase_name in ("store", "communicate")

    fn_source = (
        "@cute.jit\n"
        f"def phase_fn(page_ptr, {tile_params}, op_config_ptr"
        f"{signature_suffix}):\n"
        f"{body}"
    )

    exec_globals = {
        "cute": cute,
        "Int32": Int32,
        "Int64": Int64,
        "_instance": instance,
    }
    if append_mbar:
        from .interpreter import mbarrier_arrive

        exec_globals["mbarrier_arrive"] = mbarrier_arrive
    if needs_fence:
        from .interpreter import nanosleep

        @cute.jit
        def _compiler_fence():
            """Emit a tiny side-effecting instruction to block over-aggressive reordering."""
            nanosleep(Int32(0))

        exec_globals["_compiler_fence"] = _compiler_fence

    _register_generated_source(fn_source, unique_filename)

    code = compile(fn_source, unique_filename, "exec")
    exec(code, exec_globals)
    phase_fn = exec_globals["phase_fn"]
    phase_fn._uses_op_config_ptr = uses_op_config_ptr
    return phase_fn


def compile_phase(instance, phase_name, tensor_param_names=None,
                  tma_param_names=None, tma_local_mapping=None,
                  noinline=False):
    """Compile any Op phase method into a @cute.jit dispatch wrapper.

    For load phases, detects async vs sync from method
    signature. Async loads manage their own mbarrier; sync loads get an
    automatic mbarrier_arrive appended. All load wrappers include work_mbar
    in their signature.

    All other phases (compute, store, communicate, backward_*) are compiled
    directly with no extra parameters.
    """
    is_load = phase_name == "load"

    if is_load:
        method = getattr(instance, phase_name)
        is_async = "work_mbar" in inspect.signature(method).parameters
        fn = _build_phase_wrapper(
            instance,
            phase_name,
            tensor_param_names,
            extra_params=["work_mbar"],
            append_mbar=not is_async,
            filename=f"<compile_{phase_name}>",
            tma_param_names=tma_param_names,
            tma_local_mapping=tma_local_mapping,
        )
    else:
        fn = _build_phase_wrapper(
            instance,
            phase_name,
            tensor_param_names,
            filename=f"<compile_{phase_name}>",
            tma_param_names=tma_param_names,
            tma_local_mapping=tma_local_mapping,
        )

    if noinline and hasattr(fn, "__wrapped__"):
        fn.__wrapped__._noinline = True

    return fn


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
    "compile_phase",
    "cleanup_linecache",
    # Internals
    "_extract_body",
    "_build_phase_wrapper",
]
