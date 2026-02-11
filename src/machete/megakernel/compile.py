# Copyright (c) 2025, Machete Authors
"""
Phase Function Compilation for Megakernel.

Compiles an Op's pipelined phase methods (load, compute, store) into
@cute.jit functions. The op method body is extracted via inspect and inlined
with init_source prepended. This allows CuTe DSL's AST preprocessor to
transform all control flow (if/while over dynamic values) in a single pass.

Usage:
    from machete.megakernel.compile import compile_compute

    tile_fn = compile_compute(MyOp, init_source, tensor_param_names=['t0', 't1'])

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
# Source Extraction & Function Building
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


def _merge_globals(*fns):
    """Merge __globals__ from multiple functions.

    Always includes cutlass dtypes so that generated init source strings
    (dtype references like BFloat16) work without needing a function
    object to extract globals from.
    """
    merged = {}
    for fn in fns:
        if hasattr(fn, "__globals__"):
            merged.update(fn.__globals__)
    merged["cute"] = cute
    merged["Int32"] = Int32
    merged["Int64"] = Int64

    # Cutlass dtype names used in tensor declarations
    import cutlass
    for name in dir(cutlass):
        obj = getattr(cutlass, name)
        if isinstance(obj, type) or (hasattr(obj, '__name__') and
                                      name[0].isupper()):
            merged[name] = obj

    return merged


def _is_pass_only(fn):
    """Check if a function body is just 'pass' (no-op)."""
    body = _extract_body(fn).strip()
    return body == "pass"


# =============================================================================
# Phase Compilation
# =============================================================================


def _build_phase_fn(body_source, exec_globals, tensor_param_names,
                    extra_params=None, filename="<compile_phase>"):
    """Build a @cute.jit phase function with optional tensor parameters.

    Adds tensor parameter names to the function signature when provided.
    The tensors are cute.Tensor objects passed from the dispatch
    function using the op's local tensor names.

    Signature:
        fn(page_ptr, tile_0, tile_1, tile_2, tile_3, tile_4, op_config_ptr,
           [work_mbar,] x, weight, y) -> None

    Args:
        body_source: Combined init + op body source code.
        exec_globals: Globals dict for exec(). May include 'compute_barrier'
            for warp-specialized compute phases.
        tensor_param_names: List of tensor parameter names (e.g., ['x', 'weight', 'y']).
        extra_params: Extra parameter names inserted before tensor params
            (e.g., ['work_mbar'] for async load phases).
        filename: Base filename for linecache registration.
    """
    global _compile_counter
    _compile_counter += 1
    unique_filename = f"{filename}_{_compile_counter}"

    # Build extra params (e.g., work_mbar for async load)
    extra_params_str = ", ".join(extra_params or [])
    if extra_params_str:
        extra_params_str = ", " + extra_params_str

    # Build tensor params string for the function signature
    tensor_params_str = ", ".join(tensor_param_names)
    if tensor_params_str:
        tensor_params_str = ", " + tensor_params_str

    tile_params = ", ".join(f"tile_{i}" for i in range(5))
    fn_source = (
        "@cute.jit\n"
        f"def phase_fn(page_ptr, {tile_params}, op_config_ptr"
        f"{extra_params_str}{tensor_params_str}):\n"
        + textwrap.indent(body_source, "    ")
        + "\n"
    )

    linecache.cache[unique_filename] = (
        len(fn_source),
        None,
        fn_source.splitlines(True),
        unique_filename,
    )
    _linecache_entries.append(unique_filename)

    code = compile(fn_source, unique_filename, "exec")
    exec(code, exec_globals)
    return exec_globals["phase_fn"]


def compile_phase(phase_fn, init_source=None, tensor_param_names=None,
                  replace_barrier=False, num_compute_threads=None,
                  extra_params=None):
    """Compile a PipelinedOp's phase method into a @cute.jit function.

    Generates a function that accepts cute.Tensor objects as additional
    parameters (passed through the dispatch chain from the kernel).

    Args:
        phase_fn: PipelinedOp.load, .compute, or .store static method.
        init_source: Pre-generated init source string (tensor aliases, dims, dtypes).
        tensor_param_names: List of tensor parameter names for the signature.
        replace_barrier: If True, replace cute.arch.barrier() and
            cute.arch.sync_threads() calls with compute_barrier() in the
            extracted body. Used for warp-specialized compute phases where
            __syncthreads would deadlock (DMA warp doesn't participate).
        num_compute_threads: Number of MMA warp threads for the named barrier.
            Required when replace_barrier=True.
        extra_params: Extra parameter names inserted before tensor params
            (e.g., ['work_mbar'] for async load phases).

    Returns:
        @cute.jit function with signature:
            (page_ptr, tile_0..4, op_config_ptr, [work_mbar,] x, weight, ...) -> None
    """
    if tensor_param_names is None:
        tensor_param_names = []

    parts = []
    if init_source is not None:
        parts.append(init_source)
    if not _is_pass_only(phase_fn):
        body = _extract_body(phase_fn)
        if replace_barrier:
            body = body.replace("cute.arch.barrier()", "compute_barrier()")
            body = body.replace("cute.arch.sync_threads()", "compute_barrier()")
        parts.append(body)

    combined = "\n".join(parts) if parts else "pass"

    exec_globals = _merge_globals(phase_fn)

    # Inject compute_barrier function for warp-specialized compute phases
    if replace_barrier and num_compute_threads is not None:
        from .interpreter import named_barrier_sync as _nbs

        _count = num_compute_threads

        @cute.jit
        def compute_barrier():
            _nbs(Int32(1), Int32(_count))

        exec_globals["compute_barrier"] = compute_barrier

    return _build_phase_fn(
        combined,
        exec_globals,
        tensor_param_names,
        extra_params=extra_params,
        filename="<compile_phase>",
    )


def compile_load(op_cls, init_source=None, tensor_param_names=None):
    """Compile load phase. Always includes work_mbar in the signature.

    Detection is automatic via the load method's signature:
    - If load declares ``work_mbar`` as a parameter, the op signals it
      itself (async TMA load).
    - Otherwise, ``mbarrier_arrive(work_mbar)`` is appended automatically.
    """
    is_async = "work_mbar" in inspect.signature(op_cls.load).parameters

    if is_async:
        return compile_phase(op_cls.load, init_source, tensor_param_names,
                             extra_params=["work_mbar"])

    # Sync load: build manually so we can append mbarrier_arrive(work_mbar)
    if tensor_param_names is None:
        tensor_param_names = []

    parts = []
    if init_source is not None:
        parts.append(init_source)
    if not _is_pass_only(op_cls.load):
        parts.append(_extract_body(op_cls.load))
    parts.append("mbarrier_arrive(work_mbar)")
    combined = "\n".join(parts)

    exec_globals = _merge_globals(op_cls.load)
    from .interpreter import mbarrier_arrive
    exec_globals["mbarrier_arrive"] = mbarrier_arrive

    return _build_phase_fn(combined, exec_globals, tensor_param_names,
                           extra_params=["work_mbar"],
                           filename="<compile_load>")


def compile_compute(op_cls, init_source=None, tensor_param_names=None,
                    replace_barrier=False, num_compute_threads=None):
    """Compile compute phase."""
    return compile_phase(op_cls.compute, init_source, tensor_param_names,
                         replace_barrier=replace_barrier,
                         num_compute_threads=num_compute_threads)


def compile_store(op_cls, init_source=None, tensor_param_names=None):
    """Compile store phase."""
    return compile_phase(op_cls.store, init_source, tensor_param_names)


def compile_backward_load(op_cls, init_source=None, tensor_param_names=None):
    """Compile backward_load phase. Always includes work_mbar in the signature.

    Mirrors compile_load: detects async (work_mbar in signature) vs sync,
    and for sync loads appends mbarrier_arrive(work_mbar) automatically.
    """
    is_async = "work_mbar" in inspect.signature(op_cls.backward_load).parameters

    if is_async:
        return compile_phase(op_cls.backward_load, init_source, tensor_param_names,
                             extra_params=["work_mbar"])

    # Sync backward load: append mbarrier_arrive(work_mbar)
    if tensor_param_names is None:
        tensor_param_names = []

    parts = []
    if init_source is not None:
        parts.append(init_source)
    if not _is_pass_only(op_cls.backward_load):
        parts.append(_extract_body(op_cls.backward_load))
    parts.append("mbarrier_arrive(work_mbar)")
    combined = "\n".join(parts)

    exec_globals = _merge_globals(op_cls.backward_load)
    from .interpreter import mbarrier_arrive
    exec_globals["mbarrier_arrive"] = mbarrier_arrive

    return _build_phase_fn(combined, exec_globals, tensor_param_names,
                           extra_params=["work_mbar"],
                           filename="<compile_backward_load>")


def compile_backward_compute(op_cls, init_source=None, tensor_param_names=None,
                             replace_barrier=False, num_compute_threads=None):
    """Compile backward_compute phase."""
    return compile_phase(op_cls.backward_compute, init_source, tensor_param_names,
                         replace_barrier=replace_barrier,
                         num_compute_threads=num_compute_threads)


def compile_backward_store(op_cls, init_source=None, tensor_param_names=None):
    """Compile backward_store phase."""
    return compile_phase(op_cls.backward_store, init_source, tensor_param_names)


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
    "compile_load",
    "compile_compute",
    "compile_store",
    "compile_backward_load",
    "compile_backward_compute",
    "compile_backward_store",
    "cleanup_linecache",
    # Internals
    "_extract_body",
    "_build_phase_fn",
    "_merge_globals",
    "_is_pass_only",
]
