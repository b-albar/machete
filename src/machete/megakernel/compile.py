# Copyright (c) 2025, Machete Authors
"""
Op Execution Compilation for Megakernel.

Provides two compilation strategies that fuse an Op's load/compute/store
methods into a single @cute.jit tile function:

1. compile_sequential: All threads execute load → sync → compute → sync → store
2. compile_warp_specialized: Producer warps do load/store, consumer warps do
   compute, synchronized via hardware mbarrier

Op methods are plain @staticmethod functions whose bodies are extracted via
inspect and inlined into a new function. This allows CuTe DSL's AST
preprocessor to transform all control flow (if/while over dynamic values)
in a single pass.

These functions are agnostic to forward vs backward — the caller selects
which methods to pass (e.g., load_forward or load_backward).

Usage:
    from machete.megakernel.compile import compile_sequential, compile_warp_specialized

    # Forward pass
    tile_fn = compile_sequential(MyOp.load_forward, MyOp.compute_forward, MyOp.store_forward)

    # Backward pass
    tile_fn = compile_sequential(MyOp.load_backward, MyOp.compute_backward, MyOp.store_backward)

    # Both return @cute.jit functions with signature:
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


# =============================================================================
# Sequential Compilation
# =============================================================================


def _is_pass_only(fn):
    """Check if a function body is just 'pass' (no-op).

    Used to skip inlining init_forward when it has no useful body.
    """
    body = _extract_body(fn).strip()
    return body == "pass"


def compile_sequential(load_fn, compute_fn, store_fn, init_fn=None,
                       init_source=None):
    """Fuse init/load/compute/store into a sequential @cute.jit function.

    Extracts each op method's body via inspect and inlines them into a
    single function: [init →] load → sync → compute → sync → store.

    Init can be provided as either a function (init_fn, extracted via inspect)
    or a pre-generated source string (init_source, from Op.gen_init_source).
    init_source takes precedence over init_fn when both are provided.

    Args:
        load_fn: Op.load_forward static method
        compute_fn: Op.compute_forward static method
        store_fn: Op.store_forward static method
        init_fn: Op.init_forward static method (optional, legacy)
        init_source: Pre-generated init source string (optional, preferred)

    Returns:
        @cute.jit function with signature:
            (smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr) -> None
    """
    load_body = _extract_body(load_fn)
    compute_body = _extract_body(compute_fn)
    store_body = _extract_body(store_fn)

    parts = []
    all_fns = [load_fn, compute_fn, store_fn]

    # Inline init body before everything else
    if init_source is not None:
        # Generated init source with static dims baked in
        parts.append(init_source)
    elif init_fn is not None and not _is_pass_only(init_fn):
        parts.append(_extract_body(init_fn))
        all_fns.append(init_fn)

    parts.extend([
        load_body,
        "cute.arch.sync_threads()",
        compute_body,
        "cute.arch.sync_threads()",
        store_body,
    ])

    combined = "\n".join(parts)

    return _build_tile_fn(
        combined,
        _merge_globals(*all_fns),
        filename="<compile_sequential>",
    )


# =============================================================================
# Warp-Specialized Compilation
# =============================================================================


def compile_warp_specialized(
    load_fn,
    compute_fn,
    store_fn,
    init_fn=None,
    init_source=None,
    num_producer_warps=1,
    warps_per_block=8,
):
    """Fuse init/load/compute/store into a warp-specialized @cute.jit function.

    Extracts each op method's body via inspect and inlines them into a
    producer/consumer pattern synchronized via hardware mbarriers:

        All warps (before split)
        ────────────────────────
        init_forward()              ← shared setup (if provided)

        Producer warps              Consumer warps
        ──────────────              ──────────────
        load_forward()              ─ (idle) ─
        mbarrier_arrive(full)       mbarrier_wait(full, phase)
        ─ (idle) ─                  compute_forward()
        mbarrier_wait(empty, phase) mbarrier_arrive(empty)
        store_forward()             ─ (idle) ─
        sync_threads()              sync_threads()

    The init body is inlined before the warp_id branch, so any
    variables it defines (pipeline objects, TMA partitions, smem tensor
    views) are naturally in scope for both producer and consumer paths.

    Args:
        load_fn: Op.load_forward static method
        compute_fn: Op.compute_forward static method
        store_fn: Op.store_forward static method
        init_fn: Op.init_forward static method (optional, legacy)
        init_source: Pre-generated init source string (optional, preferred)
        num_producer_warps: Number of warps dedicated to load/store (default: 1)
        warps_per_block: Total warps per block (default: 8)

    Returns:
        @cute.jit function with signature:
            (smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr) -> None
    """
    load_body = _extract_body(load_fn)
    compute_body = _extract_body(compute_fn)
    store_body = _extract_body(store_fn)

    all_fns = [load_fn, compute_fn, store_fn]

    num_producer_threads = num_producer_warps * 32
    num_consumer_threads = (warps_per_block - num_producer_warps) * 32

    parts = []

    # Inline init source (pointers, dims, tensors) before the warp split
    if init_source is not None:
        parts.append(init_source)

    # Also inline init_fn body if it has custom setup (e.g., smem allocations)
    if init_fn is not None and not _is_pass_only(init_fn):
        parts.append(_extract_body(init_fn))
        all_fns.append(init_fn)

    parts.extend([
        "warp_id = cute.arch.warp_idx()",
        "",
        "mbar_storage = cute.arch.alloc_smem(cute.Uint64, 2, alignment=8)",
        "mbar_full = mbar_storage",
        "mbar_empty = mbar_storage + 1",
        "",
        "tidx = cute.arch.thread_idx()[0]",
        "if tidx == 0:",
        f"    cute.arch.mbarrier_init(mbar_full, {num_producer_threads})",
        f"    cute.arch.mbarrier_init(mbar_empty, {num_consumer_threads})",
        "cute.arch.sync_threads()",
        "",
        f"if warp_id < Int32({num_producer_warps}):",
        textwrap.indent(load_body, "    "),
        "    cute.arch.mbarrier_arrive(mbar_full)",
        "    cute.arch.mbarrier_wait(mbar_empty, Int32(0))",
        textwrap.indent(store_body, "    "),
        "else:",
        "    cute.arch.mbarrier_wait(mbar_full, Int32(0))",
        textwrap.indent(compute_body, "    "),
        "    cute.arch.mbarrier_arrive(mbar_empty)",
        "",
        "cute.arch.sync_threads()",
    ])

    combined = "\n".join(parts)

    return _build_tile_fn(
        combined,
        _merge_globals(*all_fns),
        filename="<compile_warp_specialized>",
    )


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
    "compile_sequential",
    "compile_warp_specialized",
    "cleanup_linecache",
]
