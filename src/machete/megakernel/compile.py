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
import hashlib
import inspect
import linecache
import textwrap
from functools import lru_cache
from typing import Dict, List, Tuple

import cutlass.cute as cute
from cutlass import Int32, Int64

from .ops import MAX_TILE_DIMS

# Track linecache entries for cleanup
_linecache_entries: list = []

# Counter for phase-wrapper filenames in linecache.
_compile_counter = 0


@lru_cache(maxsize=None)
def _compile_generated_source(source: str, label: str):
    """Compile generated source once per `(source, label)` pair."""
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]
    filename = f"<{label}>_{digest}"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(True),
        filename,
    )
    if filename not in _linecache_entries:
        _linecache_entries.append(filename)
    return compile(source, filename, "exec")


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
    code = _compile_generated_source(source, label)
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


@lru_cache(maxsize=None)
def _extract_body_from_file(source_file: str, fn_name: str, def_lineno: int) -> str:
    """Extract a function body from a source file, cached by stable location."""
    with open(source_file, "r", encoding="utf-8") as f:
        file_source = f.read()
    tree = ast.parse(file_source)

    func_node = None
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == fn_name
            and node.lineno == def_lineno
        ):
            func_node = node
            break

    if func_node is None:
        raise ValueError(f"Could not find function def for {fn_name} at line {def_lineno}")

    lines = file_source.splitlines()
    body_start = func_node.body[0].lineno - 1
    body_end = func_node.body[-1].end_lineno
    return textwrap.dedent("\n".join(lines[body_start:body_end]))


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
        source_lines, start_lineno = inspect.getsourcelines(fn)
        def_lineno = start_lineno
        for offset, line in enumerate(source_lines):
            stripped = line.lstrip()
            if stripped.startswith("def ") or stripped.startswith("async def "):
                def_lineno = start_lineno + offset
                break
        try:
            return _extract_body_from_file(source_file, fn.__name__, def_lineno)
        except ValueError:
            pass

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
    for axis in range(5):
        tile_param = f"tile_{axis}"
        if tile_param in method_params and tile_param not in call_args:
            call_args.append(tile_param)
    return call_args


def _tma_call_args(method_params_ordered: List[str], tma_local_mapping: Dict[str, str]) -> List[str]:
    """Map local TMA method parameters back to canonical wrapper parameter names."""
    return [param for param in method_params_ordered if param in tma_local_mapping]


def _wrapper_signature_suffix(is_load_phase, extra_params, tensor_param_names, tma_param_names) -> str:
    """Build the wrapper function signature suffix after op_config_ptr."""
    parts = []
    if extra_params:
        parts.extend(extra_params)
    if tensor_param_names:
        parts.extend(dict.fromkeys(tensor_param_names))
    if tma_param_names:
        parts.extend(dict.fromkeys(tma_param_names))
    return f", {', '.join(parts)}" if parts else ""


def _tma_rebind_preamble(instance, tma_rebind_specs) -> str:
    """Build wrapper source that rebinds shared TMA atoms to runtime desc ptrs."""
    if not tma_rebind_specs:
        return ""

    lines: List[str] = []
    for spec in tma_rebind_specs:
        gmem_line = (
            f"\n    {spec['local_gmem_name']} = make_runtime_tma_gmem(\n"
            f"        {spec['direction']!r},\n"
            f"        {spec['runtime_tensor_name']},\n"
            f"        {spec['smem_layout_src']},\n"
            f"        {spec['cta_tiler_src']},\n"
            f"    )"
        )
        lines.append(
            f"    {spec['local_atom_name']} = copy.copy({spec['wrapper_atom_name']})\n"
            f"    {spec['local_atom_name']}._trait = RuntimeDescTMATrait(\n"
            f"        {spec['wrapper_atom_name']}._trait.value,\n"
            f"        runtime_desc_ptr_from_pool({spec['desc_pool_name']}, {spec['desc_slot_name']}),\n"
            f"        field_namespace={spec['field_namespace']!r},\n"
            f"        supports_mbar={spec['supports_mbar']},\n"
            f"    ){gmem_line}"
        )
    return "\n".join(lines) + "\n"


def _tensor_reconstruction_preamble(instance, tensor_names: List[str]) -> Tuple[str, List[str]]:
    """Build wrapper source that reconstructs tensors from ``op_config_ptr``.

    Returns:
        A tuple of:
        - source preamble lines
        - tensor names that must remain in the wrapper signature because they
          could not be reconstructed generically
    """
    if not tensor_names:
        return "", []

    op_cls = instance.__class__
    unique_tensors = {
        name: dims for name, _dtype, dims in getattr(op_cls, "_UNIQUE_TENSORS", ())
    }
    ptr_slots = getattr(op_cls, "_CONFIG_PTR_I64_INDEX", {})
    dynamic_offsets = getattr(op_cls, "_CONFIG_DYNAMIC_I32_OFFSET", {})

    preamble_lines: List[str] = []
    fallback_names: List[str] = []
    loaded_dims = set()

    def _shape_expr(dim_name: str) -> str | None:
        if dim_name in dynamic_offsets:
            local_name = f"_dim_{dim_name}"
            if dim_name not in loaded_dims:
                preamble_lines.append(
                    f"    {local_name} = ld_global_i32(op_config_ptr, Int32({dynamic_offsets[dim_name]}))"
                )
                loaded_dims.add(dim_name)
            return local_name
        if hasattr(instance, dim_name):
            return f"_instance.{dim_name}"
        return None

    for tensor_name in tensor_names:
        dims = unique_tensors.get(tensor_name)
        ptr_slot = ptr_slots.get(tensor_name)
        dtype_attr = f"{tensor_name}_dtype"
        if dims is None or ptr_slot is None or not hasattr(instance, dtype_attr):
            fallback_names.append(tensor_name)
            continue

        shape_exprs = []
        stride_exprs = []
        missing_dim = False
        for dim_name in dims:
            shape_expr = _shape_expr(dim_name)
            stride_attr = f"{tensor_name}_stride_{dim_name}"
            if shape_expr is None or not hasattr(instance, stride_attr):
                missing_dim = True
                break
            shape_exprs.append(shape_expr)
            stride_exprs.append(f"_instance.{stride_attr}")

        if missing_dim:
            fallback_names.append(tensor_name)
            continue

        shape_src = "(" + ", ".join(shape_exprs) + ("," if len(shape_exprs) == 1 else "") + ")"
        stride_src = "(" + ", ".join(stride_exprs) + ("," if len(stride_exprs) == 1 else "") + ")"
        stride_vals = [getattr(instance, f"{tensor_name}_stride_{dim_name}") for dim_name in dims]
        one_stride_dims = [i for i, s in enumerate(stride_vals) if s == 1]
        if len(one_stride_dims) == 1:
            leading_dim = one_stride_dims[0]
        elif stride_vals:
            leading_dim = min(range(len(stride_vals)), key=lambda i: stride_vals[i])
        else:
            leading_dim = 0
        tensor_src = (
            f"cute.make_tensor("
            f"cute.make_ptr(_instance.{dtype_attr}, ld_global_i64(op_config_ptr, Int32({ptr_slot})), "
            f"cute.AddressSpace.gmem, assumed_align=16), "
            f"cute.make_layout({shape_src}, stride={stride_src})"
            f")"
        )
        preamble_lines.append(f"    {tensor_name} = {tensor_src}")
        preamble_lines.append(
            f"    {tensor_name}.mark_layout_dynamic(leading_dim={leading_dim})"
        )

    if not preamble_lines:
        return "", fallback_names
    return "\n".join(preamble_lines) + "\n", fallback_names


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
    reconstruct_tensors=False,
    extra_reconstruct_tensor_names=None,
    tma_rebind_specs=None,
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
    reconstruct_preamble = ""
    wrapper_tensor_param_names = list(tensor_param_names or [])
    reconstructed_tensor_names: List[str] = []
    reconstruct_names = list(tensor_param_names or [])
    for name in extra_reconstruct_tensor_names or []:
        if name not in reconstruct_names:
            reconstruct_names.append(name)
    if reconstruct_names and reconstruct_tensors:
        reconstruct_preamble, remaining_tensor_names = _tensor_reconstruction_preamble(
            instance,
            reconstruct_names,
        )
        wrapper_tensor_param_names = [
            name for name in wrapper_tensor_param_names if name in set(remaining_tensor_names)
        ]
        reconstructed_tensor_names = [
            name for name in reconstruct_names if name not in set(remaining_tensor_names)
        ]

    if tensor_param_names:
        known_special = {
            "page_ptr",
            "op_config_ptr",
            "work_mbar",
        }
        tma_local_names = set(tma_reverse)
        expects_tensors = any(
            param not in known_special
            and not param.startswith("tile_")
            and param not in tma_local_names
            for param in method_params
        )
        if expects_tensors:
            wrapper_tensor_param_name_set = set(wrapper_tensor_param_names)
            reconstructed_tensor_name_set = set(reconstructed_tensor_names)
            call_args.extend(
                name
                for name in tensor_param_names
                if name in wrapper_tensor_param_name_set or name in reconstructed_tensor_name_set
            )

    call_args.extend(_tma_call_args(method_params_ordered, tma_reverse))

    method_uses_op_config_ptr = "op_config_ptr" in method_params
    uses_op_config_ptr = method_uses_op_config_ptr
    if method_uses_op_config_ptr:
        call_args.append("op_config_ptr")

    # Check if method expects special framework params
    if "work_mbar" in method_params and extra_params and "work_mbar" in extra_params:
        call_args.append("work_mbar")

    call_str = ", ".join(call_args)

    # Build wrapper function signature (standard dispatch format)
    tile_params = ", ".join(f"tile_{i}" for i in range(MAX_TILE_DIMS))
    signature_suffix = _wrapper_signature_suffix(
        phase_name == "load",
        extra_params,
        wrapper_tensor_param_names,
        tma_param_names,
    )

    # For load/store phases: dispatch is called by all DMA warp threads
    # (needed for TMA warp convergence on both G2S loads and S2G stores).
    # Non-TMA ops must be wrapped in elect_one() so only one thread executes.
    # TMA ops handle thread selection internally (elect_one for mbarrier,
    # cute.copy outside for warp-convergent TMA copy).
    has_tma = bool(tma_local_mapping)

    if phase_name == "communicate" and not has_tma and not tma_rebind_specs:
        body = "    pass\n"
    else:
        body = _wrapper_body(phase_name, call_str, append_mbar, has_tma)
    if tma_local_mapping and not tma_rebind_specs:
        alias_lines = [
            f"    {local_name} = {canonical_name}"
            for local_name, canonical_name in tma_local_mapping.items()
            if local_name != canonical_name
        ]
        if alias_lines:
            body = "\n".join(alias_lines) + "\n" + body
    if tma_rebind_specs:
        body = _tma_rebind_preamble(instance, tma_rebind_specs) + body
    if reconstruct_preamble:
        body = reconstruct_preamble + body
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
    if tma_rebind_specs:
        import copy

        from .transport import RuntimeDescTMATrait, make_runtime_tma_gmem, runtime_desc_ptr_from_pool

        exec_globals["copy"] = copy
        exec_globals["RuntimeDescTMATrait"] = RuntimeDescTMATrait
        exec_globals["make_runtime_tma_gmem"] = make_runtime_tma_gmem
        exec_globals["runtime_desc_ptr_from_pool"] = runtime_desc_ptr_from_pool
    if reconstruct_preamble:
        from .interpreter import ld_global_i32, ld_global_i64

        exec_globals["ld_global_i32"] = ld_global_i32
        exec_globals["ld_global_i64"] = ld_global_i64
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
    phase_targets = [phase_fn]
    wrapped = getattr(phase_fn, "__wrapped__", None)
    if wrapped is not None and wrapped is not phase_fn:
        phase_targets.append(wrapped)
    for target in phase_targets:
        target._uses_op_config_ptr = uses_op_config_ptr
        target._machete_phase_name = phase_name
        target._machete_phase_owner = instance.__class__.__name__
        target._machete_wrapper_tensor_params = tuple(wrapper_tensor_param_names)
        target._machete_reconstructed_tensor_params = tuple(reconstructed_tensor_names)
    return phase_fn


def compile_phase(instance, phase_name, tensor_param_names=None,
                  tma_param_names=None, tma_local_mapping=None,
                  noinline=False,
                  reconstruct_tensors=False,
                  extra_reconstruct_tensor_names=None):
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
            reconstruct_tensors=reconstruct_tensors,
            extra_reconstruct_tensor_names=extra_reconstruct_tensor_names,
            tma_rebind_specs=getattr(instance, "_machete_tma_rebind_specs", {}).get(phase_name),
        )
    else:
        fn = _build_phase_wrapper(
            instance,
            phase_name,
            tensor_param_names,
            filename=f"<compile_{phase_name}>",
            tma_param_names=tma_param_names,
            tma_local_mapping=tma_local_mapping,
            reconstruct_tensors=reconstruct_tensors,
            extra_reconstruct_tensor_names=extra_reconstruct_tensor_names,
            tma_rebind_specs=getattr(instance, "_machete_tma_rebind_specs", {}).get(phase_name),
        )

    if noinline:
        noinline_targets = [fn]
        wrapped = getattr(fn, "__wrapped__", None)
        if wrapped is not None and wrapped is not fn:
            noinline_targets.append(wrapped)
        for target in noinline_targets:
            target._noinline = True

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
