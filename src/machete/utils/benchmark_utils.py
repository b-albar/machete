# Copyright (c) 2025, Machete Authors.
"""
Benchmark utilities for measuring kernel performance.

Supports both:
- PyTorch-based benchmarking (for autograd functions)
- CUTLASS cute.testing-based benchmarking (for JIT kernels, more accurate)
"""

import math
import time
from dataclasses import dataclass
from typing import Callable, Any, Optional

import torch
import torch.utils.benchmark as benchmark

# CUTLASS testing utilities for precise kernel benchmarking
try:
    import cuda.bindings.driver as cuda
    import cutlass.cute.testing as cute_testing
    from cutlass.cute.testing import JitArguments

    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False
    cuda = None
    cute_testing = None
    JitArguments = None


@dataclass
class KernelBenchSpec:
    """Wraps a compiled CuTe JIT kernel for benchmarking.

    Used by Benchmark.run(mode="kernel") to distinguish megakernel launches
    from regular callables. The persistent megakernel requires barrier resets
    between invocations, so CUDA graph replay is NOT used — each iteration
    calls launch_fn() which launches the kernel.

    Attributes:
        launch_fn: Callable that launches the kernel (timed).
            Must be called on the stream specified in ``stream``. If it accepts
            one positional argument, the benchmark framework passes the input
            group index for L2-evicting input rotation.
        setup_fn: Optional callable for per-iteration setup (barrier resets,
            output zeroing, etc.). If it accepts one positional argument, the
            benchmark framework passes the input group index.
        stream: A (torch.cuda.Stream, CUstream) pair.
        use_host_timer: If True, benchmark with host wall-clock time and
            explicit CUDA synchronization instead of CUDA events. This is less
            pure, but some persistent megakernel paths are not stable under
            event-based timing.
        input_size_bytes: Total bytes read/written by one input group. When
            provided, benchmark timing rotates through enough independent input
            groups to naturally evict the previous group from L2.
        num_input_groups: Explicit input group count. Overrides the L2-derived
            count when set.
        cooldown_ms: Optional post-benchmark idle time. Defaults to the shared
            benchmark convention when unset.
    """

    launch_fn: Callable
    setup_fn: Optional[Callable] = None
    stream: Any = None
    use_host_timer: bool = False
    input_size_bytes: Optional[int] = None
    num_input_groups: Optional[int] = None
    cooldown_ms: Optional[int] = None
    metadata: Optional[str] = None
    _keep_alive: Any = None  # Prevent GC of objects whose GPU memory is referenced by the kernel


def cuda_l2_cache_size(device: Optional[Any] = None) -> int:
    """Return the selected CUDA device L2 cache size in bytes."""
    props = torch.cuda.get_device_properties(device if device is not None else torch.cuda.current_device())
    return int(getattr(props, "l2_cache_size", 0) or 0)


def recommended_input_group_count(
    input_size_bytes: Optional[int],
    *,
    device: Optional[Any] = None,
    l2_multiplier: int = 3,
) -> int:
    """Choose an input rotation count that naturally evicts L2 residency.

    The convention is one group when a single input group is already at least
    ``l2_multiplier`` times larger than L2; otherwise allocate enough groups so
    the rotation footprint exceeds that threshold.
    """
    if input_size_bytes is None:
        return 1
    input_size_bytes = int(input_size_bytes)
    if input_size_bytes <= 0:
        return 1
    l2_size = cuda_l2_cache_size(device)
    if l2_size <= 0:
        return 1
    target_bytes = int(l2_multiplier) * l2_size
    if input_size_bytes >= target_bytes:
        return 1
    return target_bytes // input_size_bytes + 1


def tensor_tree_nbytes(obj: Any, *, unique_storage: bool = True) -> int:
    """Estimate tensor bytes in a nested object.

    This is useful for benchmark setup code that needs to populate
    ``KernelBenchSpec.input_size_bytes``. By default each storage is counted
    once so views do not inflate the input footprint.
    """
    seen: set[tuple[int, int]] = set()

    def _walk(value: Any) -> int:
        if isinstance(value, torch.Tensor):
            if unique_storage:
                try:
                    storage = value.untyped_storage()
                    key = (int(storage.data_ptr()), int(storage.nbytes()))
                except RuntimeError:
                    key = (int(value.data_ptr()), int(value.numel() * value.element_size()))
                if key in seen:
                    return 0
                seen.add(key)
            return int(value.numel() * value.element_size())
        if isinstance(value, dict):
            return sum(_walk(v) for v in value.values())
        if isinstance(value, (list, tuple, set, frozenset)):
            return sum(_walk(v) for v in value)
        return 0

    return _walk(obj)


def combine_megakernel_bench_spec(
    kernels: list[Any],
    setup_fn: Optional[Callable] = None,
    keep_alive: Any = None,
) -> KernelBenchSpec:
    """Create one benchmark spec that launches several megakernels in sequence.

    This is used for split-kernel baselines such as pre-attention / attention /
    post-attention pipelines. The helper deliberately reuses each megakernel's
    real launch path instead of rebuilding benchmark-only argument lists in the
    benchmark scripts.
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS is not available")

    for kernel in kernels:
        kernel.compile()
        kernel._cache_launch_state()

    def _setup():
        if setup_fn is not None:
            setup_fn()
        # Each kernel resets its own barriers in run(). Keep the benchmark
        # setup free of kernel-specific launch plumbing.

    def _launch():
        for kernel in kernels:
            kernel.run(sync=False, validate=False)

    torch_stream = torch.cuda.current_stream()

    return KernelBenchSpec(
        launch_fn=_launch,
        setup_fn=_setup,
        stream=(torch_stream, None),
        use_host_timer=True,
        _keep_alive=(kernels, keep_alive),
    )


def chain_kernel_bench_specs(
    specs: list[KernelBenchSpec],
    setup_fn: Optional[Callable] = None,
    keep_alive: Any = None,
) -> KernelBenchSpec:
    """Create one benchmark spec that launches several existing specs in sequence.

    Unlike ``combine_megakernel_bench_spec()``, this operates on already-built
    ``KernelBenchSpec`` objects. It is useful for end-to-end model benchmarks
    that are composed from several fused megakernels built through existing
    helpers.
    """
    if not specs:
        raise ValueError("expected at least one KernelBenchSpec")

    torch_stream, cu_stream = specs[0].stream

    def _setup():
        if setup_fn is not None:
            setup_fn()
        for spec in specs:
            if spec.setup_fn is not None:
                spec.setup_fn()

    def _launch():
        for spec in specs:
            spec.launch_fn()

    return KernelBenchSpec(
        launch_fn=_launch,
        setup_fn=_setup,
        stream=(torch_stream, cu_stream),
        use_host_timer=any(spec.use_host_timer for spec in specs),
        _keep_alive=(specs, keep_alive),
    )


# =============================================================================
# CUDA graph benchmarking (for regular callables)
# =============================================================================


def benchmark_cuda_graph(
    fn: Callable,
    warmup: int = 500,
    rep: int = 100,
) -> float:
    """Benchmark a callable using CUDA graph capture + CUDA event timing.

    Captures the callable into a CUDA graph on a non-default stream, then
    replays it with CUDA event timing. This removes CPU launch overhead and
    provides measurements comparable to cute.testing.benchmark.

    Args:
        fn: Callable to benchmark (must be CUDA-only, no CPU side effects).
        warmup: Number of warmup iterations (both pre-capture and post-capture).
        rep: Number of timed iterations.

    Returns:
        Average execution time in milliseconds.
    """
    # Warmup on default stream
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Capture on non-default stream (required for CUDA graphs)
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            fn()
    torch.cuda.synchronize()

    # Warmup graph replay
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            graph.replay()
    torch.cuda.synchronize()

    # Timed runs with two CUDA events around the full profiling loop.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record(stream)
        for _ in range(rep):
            graph.replay()
        end.record(stream)

    end.synchronize()
    time.sleep(0.5)
    return start.elapsed_time(end) / rep


# =============================================================================
# CUTLASS-based benchmarking (for JIT kernels)
# =============================================================================


def benchmark_jit_kernel(
    compiled_kernel: Callable,
    *,
    workspace_generator: Optional[Callable[[], Any]] = None,
    kernel_arguments: Optional[Any] = None,
    warmup_iterations: int = 500,
    iterations: int = 100,
    workspace_count: int = 10,
    stream: Optional[Any] = None,
    use_cuda_graphs: bool = False,
) -> float:
    """
    Benchmark a CUTLASS JIT-compiled kernel using cute.testing.benchmark.

    This provides more accurate timing than PyTorch benchmarking because it:
    - Uses CUDA events for precise GPU timing
    - Supports workspace rotation to avoid L2 cache effects
    - Uses a 500 warmup / 100 profiling iteration convention by default
    - Can use CUDA graphs for reduced launch overhead

    Args:
        compiled_kernel: A compiled @cute.jit annotated function
        workspace_generator: Function returning JitArguments for each iteration
        kernel_arguments: Static arguments if not using workspace_generator
        warmup_iterations: Number of warmup iterations
        iterations: Number of benchmark iterations
        workspace_count: Number of workspaces to rotate through
        stream: CUDA stream (required for CUDA graphs)
        use_cuda_graphs: Enable CUDA graph capture

    Returns:
        Execution time in microseconds

    Example:
        >>> def gen_workspace():
        ...     a = torch.randn(1024, 1024, device='cuda')
        ...     return JitArguments(a_ptr, stream)
        >>> time_us = benchmark_jit_kernel(
        ...     compiled_kernel,
        ...     workspace_generator=gen_workspace,
        ...     workspace_count=10,
        ...     iterations=100
        ... )
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS cute.testing is not available. Install nvidia-cutlass-dsl.")

    if stream is None:
        torch_stream = torch.cuda.current_stream()
        stream = cuda.CUstream(torch_stream.cuda_stream)

    return cute_testing.benchmark(
        compiled_kernel,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        stream=stream,
        kernel_arguments=kernel_arguments,
        workspace_generator=workspace_generator,
        workspace_count=workspace_count,
        use_cuda_graphs=use_cuda_graphs,
    )


def get_cuda_stream():
    """Get the current CUDA stream as a CUstream object for CUTLASS APIs."""
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS is not available")
    torch_stream = torch.cuda.current_stream()
    return cuda.CUstream(torch_stream.cuda_stream)


# =============================================================================
# PyTorch-based benchmarking (for autograd functions)
# =============================================================================


def benchmark_forward(
    fn: Callable,
    *inputs: Any,
    repeats: int = 10,
    desc: str = "",
    verbose: bool = True,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    **kwinputs: Any,
) -> tuple[benchmark.Timer, benchmark.Measurement]:
    """Use Pytorch Benchmark on the forward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Forward pass")

    def amp_wrapper(*inputs: Any, **kwinputs: Any) -> None:
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)

    t = benchmark.Timer(
        stmt="fn_amp(*inputs, **kwinputs)",
        globals={"fn_amp": amp_wrapper, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(m)

    return t, m


def benchmark_backward(
    fn: Callable,
    *inputs: Any,
    grad: Optional[torch.Tensor] = None,
    repeats: int = 10,
    desc: str = "",
    verbose: bool = True,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    **kwinputs: Any,
) -> tuple[benchmark.Timer, benchmark.Measurement]:
    """Use Pytorch Benchmark on the backward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Backward pass")
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
        y = fn(*inputs, **kwinputs)
        if isinstance(y, tuple):
            y = y[0]
    if grad is None:
        grad = torch.randn_like(y)
    else:
        if grad.shape != y.shape:
            raise RuntimeError("Grad shape does not match output shape")

    def f(*inputs: Any, y: torch.Tensor, grad: torch.Tensor) -> None:
        # Set .grad to None to avoid extra operation of gradient accumulation
        for x in inputs:
            if isinstance(x, torch.Tensor):
                x.grad = None
        y.backward(grad, retain_graph=True)

    t = benchmark.Timer(
        stmt="f(*inputs, y=y, grad=grad)",
        globals={"f": f, "inputs": inputs, "y": y, "grad": grad},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


def benchmark_combined(
    fn: Callable,
    *inputs: Any,
    grad: Optional[torch.Tensor] = None,
    repeats: int = 10,
    desc: str = "",
    verbose: bool = True,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    **kwinputs: Any,
) -> tuple[benchmark.Timer, benchmark.Measurement]:
    """Use Pytorch Benchmark on the forward+backward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Forward + Backward pass")
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
        y = fn(*inputs, **kwinputs)
        if isinstance(y, tuple):
            y = y[0]
    if grad is None:
        grad = torch.randn_like(y)
    else:
        if grad.shape != y.shape:
            raise RuntimeError("Grad shape does not match output shape")

    def f(grad: torch.Tensor, *inputs: Any, **kwinputs: Any) -> None:
        for x in inputs:
            if isinstance(x, torch.Tensor):
                x.grad = None
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            y = fn(*inputs, **kwinputs)
            if isinstance(y, tuple):
                y = y[0]
        y.backward(grad, retain_graph=True)

    t = benchmark.Timer(
        stmt="f(grad, *inputs, **kwinputs)",
        globals={"f": f, "fn": fn, "inputs": inputs, "grad": grad, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


def benchmark_fwd_bwd(
    fn: Callable,
    *inputs: Any,
    grad: Optional[torch.Tensor] = None,
    repeats: int = 10,
    desc: str = "",
    verbose: bool = True,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    **kwinputs: Any,
) -> tuple[
    tuple[benchmark.Timer, benchmark.Measurement],
    tuple[benchmark.Timer, benchmark.Measurement],
]:
    """Use Pytorch Benchmark on the forward+backward pass of an arbitrary function."""
    return (
        benchmark_forward(
            fn,
            *inputs,
            repeats=repeats,
            desc=desc,
            verbose=verbose,
            amp=amp,
            amp_dtype=amp_dtype,
            **kwinputs,
        ),
        benchmark_backward(
            fn,
            *inputs,
            grad=grad,
            repeats=repeats,
            desc=desc,
            verbose=verbose,
            amp=amp,
            amp_dtype=amp_dtype,
            **kwinputs,
        ),
    )


def benchmark_all(
    fn: Callable,
    *inputs: Any,
    grad: Optional[torch.Tensor] = None,
    repeats: int = 10,
    desc: str = "",
    verbose: bool = True,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    **kwinputs: Any,
) -> tuple[
    tuple[benchmark.Timer, benchmark.Measurement],
    tuple[benchmark.Timer, benchmark.Measurement],
    tuple[benchmark.Timer, benchmark.Measurement],
]:
    """Use Pytorch Benchmark on fwd, bwd, and combined passes of an arbitrary function."""
    return (
        benchmark_forward(
            fn,
            *inputs,
            repeats=repeats,
            desc=desc,
            verbose=verbose,
            amp=amp,
            amp_dtype=amp_dtype,
            **kwinputs,
        ),
        benchmark_backward(
            fn,
            *inputs,
            grad=grad,
            repeats=repeats,
            desc=desc,
            verbose=verbose,
            amp=amp,
            amp_dtype=amp_dtype,
            **kwinputs,
        ),
        benchmark_combined(
            fn,
            *inputs,
            grad=grad,
            repeats=repeats,
            desc=desc,
            verbose=verbose,
            amp=amp,
            amp_dtype=amp_dtype,
            **kwinputs,
        ),
    )


# =============================================================================
# Profiling utilities
# =============================================================================


def pytorch_profiler(
    fn: Callable,
    *inputs: Any,
    trace_filename: Optional[str] = None,
    backward: bool = False,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    cpu: bool = False,
    verbose: bool = True,
    **kwinputs: Any,
) -> None:
    """Wrap benchmark functions in Pytorch profiler to see CUDA information."""
    if backward:
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            out = fn(*inputs, **kwinputs)
            if isinstance(out, tuple):
                out = out[0]
            g = torch.randn_like(out)

    for _ in range(30):  # Warm up
        if backward:
            for x in inputs:
                if isinstance(x, torch.Tensor):
                    x.grad = None
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            out = fn(*inputs, **kwinputs)
            if isinstance(out, tuple):
                out = out[0]
        # Backward should be done outside autocast
        if backward:
            out.backward(g, retain_graph=True)

    activities = ([torch.profiler.ProfilerActivity.CPU] if cpu else []) + [torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=True,
    ) as prof:
        if backward:
            for x in inputs:
                if isinstance(x, torch.Tensor):
                    x.grad = None
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            out = fn(*inputs, **kwinputs)
            if isinstance(out, tuple):
                out = out[0]
        if backward:
            out.backward(g, retain_graph=True)

    if verbose:
        print(prof.key_averages().table(row_limit=50))
    if trace_filename is not None:
        prof.export_chrome_trace(trace_filename)


# =============================================================================
# Memory and efficiency utilities
# =============================================================================


def benchmark_memory(
    fn: Callable,
    *inputs: Any,
    desc: str = "",
    verbose: bool = True,
    **kwinputs: Any,
) -> float:
    """Measure peak GPU memory usage of a function call."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn(*inputs, **kwinputs)
    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated() / ((2**20) * 1000)
    if verbose:
        print(f"{desc} max memory: {mem}GB")
    torch.cuda.empty_cache()
    return mem


def efficiency(flop: float, time: float) -> float:
    """Calculate TFLOPS efficiency from FLOP count and time in seconds."""
    return (flop / time / 10**12) if not math.isnan(time) else 0.0


def memory_throughput(bytes_transferred: int, time_us: float) -> float:
    """Calculate memory throughput in GB/s from bytes and time in microseconds."""
    return (bytes_transferred / (time_us / 1e6)) / 1e9


# =============================================================================
# Comparison utilities
# =============================================================================


def compare_kernels(
    kernels: dict[str, Callable],
    workspace_generator: Optional[Callable] = None,
    warmup_iterations: int = 500,
    iterations: int = 100,
    workspace_count: int = 10,
    bytes_transferred: Optional[int] = None,
    verbose: bool = True,
) -> dict[str, dict[str, float]]:
    """
    Compare multiple kernel implementations.

    Args:
        kernels: Dictionary of {name: callable} for each kernel to benchmark
        workspace_generator: Function returning inputs for each iteration
        warmup_iterations: Number of warmup iterations
        iterations: Number of benchmark iterations
        workspace_count: Number of workspaces to rotate through
        bytes_transferred: Total bytes read/written for throughput calculation
        verbose: Print results

    Returns:
        Dictionary with timing and throughput for each kernel

    Example:
        >>> results = compare_kernels({
        ...     'cutlass': cutlass_kernel,
        ...     'triton': triton_kernel,
        ... }, workspace_generator=gen_inputs, bytes_transferred=total_bytes)
    """
    results = {}
    baseline_time = None

    for name, kernel in kernels.items():
        try:
            if CUTLASS_AVAILABLE and hasattr(kernel, "_jit_kernel"):
                # Use CUTLASS benchmark for JIT kernels
                time_us = benchmark_jit_kernel(
                    kernel,
                    workspace_generator=workspace_generator,
                    warmup_iterations=warmup_iterations,
                    iterations=iterations,
                    workspace_count=workspace_count,
                )
            else:
                # Use triton.testing.do_bench for other kernels
                try:
                    from triton.testing import do_bench

                    time_ms = do_bench(kernel, warmup=warmup_iterations, rep=iterations)
                    time_us = time_ms * 1000
                except ImportError:
                    # Fallback to PyTorch benchmark
                    _, m = benchmark_forward(kernel, repeats=iterations, verbose=False)
                    time_us = m.mean * 1e6

            results[name] = {"time_us": time_us}

            if bytes_transferred is not None:
                results[name]["throughput_gbps"] = memory_throughput(bytes_transferred, time_us)

            if baseline_time is None:
                baseline_time = time_us
            results[name]["speedup"] = baseline_time / time_us

        except Exception as e:
            results[name] = {"time_us": float("inf"), "error": str(e)}

    if verbose:
        print("\n" + "=" * 60)
        print("Kernel Comparison Results:")
        print("=" * 60)
        for name, data in results.items():
            if "error" in data:
                print(f"  {name}: ERROR - {data['error']}")
            else:
                line = f"  {name}: {data['time_us']:.2f} us"
                if "throughput_gbps" in data:
                    line += f", {data['throughput_gbps']:.2f} GB/s"
                line += f", {data['speedup']:.2f}x vs baseline"
                print(line)
        print("=" * 60)

    return results
