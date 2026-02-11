# Copyright (c) 2025, Machete Authors.
"""
Benchmark utilities for measuring kernel performance.

Supports both:
- PyTorch-based benchmarking (for autograd functions)
- CUTLASS cute.testing-based benchmarking (for JIT kernels, more accurate)
"""

import math
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
    calls launch_fn() which resets barriers and launches the kernel.

    Attributes:
        launch_fn: Callable that resets barriers and launches the kernel.
            Must be called on the stream specified in ``stream``.
        stream: A (torch.cuda.Stream, CUstream) pair.
    """

    launch_fn: Callable
    stream: Any = None
    _keep_alive: Any = None  # Prevent GC of objects whose GPU memory is referenced by the kernel


# =============================================================================
# CUDA graph benchmarking (for regular callables)
# =============================================================================


def benchmark_cuda_graph(
    fn: Callable,
    warmup: int = 25,
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

    # Timed runs with CUDA events
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    with torch.cuda.stream(stream):
        for _ in range(rep):
            start.record(stream)
            graph.replay()
            end.record(stream)
            stream.synchronize()
            times.append(start.elapsed_time(end))

    return sum(times) / len(times)


# =============================================================================
# CUTLASS-based benchmarking (for JIT kernels)
# =============================================================================


def benchmark_jit_kernel(
    compiled_kernel: Callable,
    *,
    workspace_generator: Optional[Callable[[], Any]] = None,
    kernel_arguments: Optional[Any] = None,
    warmup_iterations: int = 10,
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
    warmup_iterations: int = 10,
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
