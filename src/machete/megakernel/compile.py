# Copyright (c) 2025, Machete Authors
"""
Op Execution Compilation for Megakernel.

Compiles an Op's forward/backward method into a @cute.jit tile function.
The op method body is extracted via inspect and inlined with init_source
prepended. This allows CuTe DSL's AST preprocessor to transform all control
flow (if/while over dynamic values) in a single pass.

Usage:
    from machete.megakernel.compile import compile_op

    # Forward pass
    tile_fn = compile_op(MyOp.forward, init_source)

    # Backward pass
    tile_fn = compile_op(MyOp.backward, init_source)

    # Returns @cute.jit function with signature:
    #   fn(smem_base: Int32, config_ptr: Int32, page_ids: tuple,
    #      tile_m: Int32, tile_n: Int32, tile_l: Int32,
    #      op_config_ptr: Int64) -> None
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


def _build_tile_fn(body_source, exec_globals, filename="<compile>"):
    """Build a @cute.jit tile function from combined body source.

    Creates a new function via exec() and registers the source in
    linecache so CuTe DSL's AST preprocessor (which uses
    inspect.getsource) can find and transform the control flow.
    """
    global _compile_counter
    _compile_counter += 1
    unique_filename = f"{filename}_{_compile_counter}"

    # The @cute.jit decorator MUST be present in the source for the DSL
    # preprocessor to recognize and transform control flow (if/while over
    # dynamic values). Without it, check_decorator() returns False and
    # the AST transformation is skipped entirely.
    fn_source = (
        "@cute.jit\n"
        "def tile_fn(smem_base, config_ptr, page_ids,"
        " tile_m, tile_n, tile_l, op_config_ptr):\n"
        + textwrap.indent(body_source, "    ")
        + "\n"
    )

    # Register in linecache so inspect.getsource works on exec'd functions
    linecache.cache[unique_filename] = (
        len(fn_source),
        None,
        fn_source.splitlines(True),
        unique_filename,
    )
    _linecache_entries.append(unique_filename)

    code = compile(fn_source, unique_filename, "exec")
    exec(code, exec_globals)
    # tile_fn is already @cute.jit-decorated in the source, so exec
    # produces the jit-wrapped function directly.
    return exec_globals["tile_fn"]


def _merge_globals(*fns):
    """Merge __globals__ from multiple functions.

    Always includes init-required globals (interpreter primitives, _FLAT,
    cutlass dtypes) so that generated init source strings work without
    needing a function object to extract globals from.
    """
    merged = {}
    for fn in fns:
        if hasattr(fn, "__globals__"):
            merged.update(fn.__globals__)
    merged["cute"] = cute
    merged["Int32"] = Int32
    merged["Int64"] = Int64

    # Init-required globals (needed for generated init source strings)
    from machete.megakernel.interpreter import ld_global_i64, ld_global_i32
    merged["ld_global_i64"] = ld_global_i64
    merged["ld_global_i32"] = ld_global_i32
    merged["_FLAT"] = 1 << 24

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
# Op Compilation
# =============================================================================


def compile_op(op_fn, init_source=None):
    """Compile an Op's forward/backward method into a @cute.jit tile function.

    Extracts the op method's body via inspect and inlines it with init_source
    prepended (if provided). The init_source typically contains tensor pointer
    loading and dimension setup generated by Op.gen_init_source().

    Args:
        op_fn: Op.forward or Op.backward static method
        init_source: Pre-generated init source string (optional).
            Contains tensor pointer loads and dimension setup.

    Returns:
        @cute.jit function with signature:
            (smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr) -> None
    """
    parts = []

    # Inline init source (pointers, dims, tensors) before op body
    if init_source is not None:
        parts.append(init_source)

    # Inline op body (skip if just 'pass')
    if not _is_pass_only(op_fn):
        parts.append(_extract_body(op_fn))

    combined = "\n".join(parts) if parts else "pass"

    return _build_tile_fn(
        combined,
        _merge_globals(op_fn),
        filename="<compile_op>",
    )


# =============================================================================
# Pipelined Phase Compilation
# =============================================================================


def _build_phase_fn(body_source, exec_globals, filename="<compile_phase>"):
    """Build a @cute.jit phase function for pipelined ops.

    Phase functions have a different signature than forward/backward:
        fn(page_ptr, tile_m, tile_n, tile_l, op_config_ptr) -> None

    Where page_ptr is the shared memory page base address (Int32).
    """
    global _compile_counter
    _compile_counter += 1
    unique_filename = f"{filename}_{_compile_counter}"

    fn_source = (
        "@cute.jit\n"
        "def phase_fn(page_ptr, tile_m, tile_n, tile_l, op_config_ptr):\n"
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


def compile_phase(phase_fn, init_source=None):
    """Compile a PipelinedOp's phase method (load_async/compute/store).

    Extracts the phase method's body and inlines it with init_source prepended.
    The init_source typically contains tensor pointer loading and dimension
    setup generated by Op.gen_init_source().

    Args:
        phase_fn: PipelinedOp.load_async, .compute, or .store static method
        init_source: Pre-generated init source string (optional).
            Contains tensor pointer loads and dimension setup.

    Returns:
        @cute.jit function with signature:
            (page_ptr, tile_m, tile_n, tile_l, op_config_ptr) -> None
    """
    parts = []

    if init_source is not None:
        parts.append(init_source)

    if not _is_pass_only(phase_fn):
        parts.append(_extract_body(phase_fn))

    combined = "\n".join(parts) if parts else "pass"

    return _build_phase_fn(
        combined,
        _merge_globals(phase_fn),
        filename="<compile_phase>",
    )


def compile_load_async(op_cls, init_source=None):
    """Compile a PipelinedOp's load_async method.

    Args:
        op_cls: PipelinedOp subclass
        init_source: Pre-generated init source string (optional)

    Returns:
        @cute.jit function for async loading.
    """
    return compile_phase(op_cls.load_async, init_source)


def compile_compute(op_cls, init_source=None):
    """Compile a PipelinedOp's compute method.

    Args:
        op_cls: PipelinedOp subclass
        init_source: Pre-generated init source string (optional)

    Returns:
        @cute.jit function for compute phase.
    """
    return compile_phase(op_cls.compute, init_source)


def compile_store(op_cls, init_source=None):
    """Compile a PipelinedOp's store method.

    Args:
        op_cls: PipelinedOp subclass
        init_source: Pre-generated init source string (optional)

    Returns:
        @cute.jit function for store phase.
    """
    return compile_phase(op_cls.store, init_source)


def compile_backward_load_async(op_cls, init_source=None):
    """Compile a PipelinedOp's backward_load_async method.

    Args:
        op_cls: PipelinedOp subclass
        init_source: Pre-generated init source string (optional)

    Returns:
        @cute.jit function for backward async loading.
    """
    return compile_phase(op_cls.backward_load_async, init_source)


def compile_backward_compute(op_cls, init_source=None):
    """Compile a PipelinedOp's backward_compute method.

    Args:
        op_cls: PipelinedOp subclass
        init_source: Pre-generated init source string (optional)

    Returns:
        @cute.jit function for backward compute phase.
    """
    return compile_phase(op_cls.backward_compute, init_source)


def compile_backward_store(op_cls, init_source=None):
    """Compile a PipelinedOp's backward_store method.

    Args:
        op_cls: PipelinedOp subclass
        init_source: Pre-generated init source string (optional)

    Returns:
        @cute.jit function for backward store phase.
    """
    return compile_phase(op_cls.backward_store, init_source)


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
    "compile_op",
    "compile_phase",
    "compile_load_async",
    "compile_compute",
    "compile_store",
    "compile_backward_load_async",
    "compile_backward_compute",
    "compile_backward_store",
    "cleanup_linecache",
    # Keep internals available for advanced use
    "_extract_body",
    "_build_tile_fn",
    "_build_phase_fn",
    "_merge_globals",
    "_is_pass_only",
]
