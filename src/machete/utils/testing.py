# Copyright (c) 2025, Machete Authors
import torch
import triton
from typing import Callable, Dict, Tuple


def benchmark_op(
    name: str, configs: Dict[str, Tuple], op_map: Dict[str, Callable], numel_provider: Callable[[Tuple], int]
):
    """
    Generic benchmarking utility for kernels.

    Args:
        name: Name of the operation being benchmarked.
        configs: Dictionary mapping configuration names to tuples of arguments.
        op_map: Dictionary mapping provider names to callable functions.
        numel_provider: Function that takes config arguments and returns total elements transferred.
    """
    print(f"\n{'=' * 20} {name} {'=' * 20}")
    print(f"{'Config':<20} | {'Provider':<15} | {'Speed (GB/s)':<15} | {'Time (ms)':<10} | {'Peak Mem (MB)':<12}")
    print("-" * 85)

    iterator = configs.items() if isinstance(configs, dict) else configs
    failed_providers = set()
    for config_name, args in iterator:
        if args is None:
            for provider in op_map:
                print(f"{config_name:<20} | {provider:<15} | {'OOM':<15} | {'-':<10} | {'-':<12}")
            continue

        for provider, func in op_map.items():
            if provider in failed_providers:
                print(f"{config_name:<20} | {provider:<15} | {'OOM':<15} | {'-':<10} | {'-':<12}")
                continue
            # Estimate throughput (GB/s)
            numel = numel_provider(args)
            bytes_transferred = numel * args[0].element_size()

            try:
                # Warm up
                for _ in range(5):
                    func(*args)

                # Time benchmark
                ms = triton.testing.do_bench(lambda: func(*args))
                if ms > 0:
                    gbps = bytes_transferred / (ms * 1e-3) / 1e9
                else:
                    gbps = 0

                # Memory benchmark
                torch.cuda.reset_peak_memory_stats()
                base_mem = torch.cuda.memory_allocated()
                _ = func(*args)
                peak_mem = torch.cuda.max_memory_allocated()
                peak_delta_mb = (peak_mem - base_mem) / (1024 * 1024)

                print(f"{config_name:<20} | {provider:<15} | {gbps:<15.2f} | {ms:<10.4f} | {peak_delta_mb:<12.2f}")
            except torch.cuda.OutOfMemoryError:
                print(f"{config_name:<20} | {provider:<15} | {'OOM':<15} | {'-':<10} | {'-':<12}")
                failed_providers.add(provider)
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"{config_name:<20} | {provider:<15} | {'Error':<15} | {'-':<10} | {'-':<12}")
                print(f"Error details: {e}")


def verify_kernel(
    name: str,
    func: Callable,
    ref_func: Callable,
    inputs: Tuple,
    dtype: torch.dtype,
    atol: float = None,
    check_grad: bool = True,
):
    """
    Generic verification utility to check forward and backward output against a reference.

    Args:
        name: Name of the kernel.
        func: The kernel function to test.
        ref_func: The reference implementation.
        inputs: Tuple of input tensors (requires_grad should be set if check_grad is True).
        dtype: Data type being tested.
        atol: Absolute tolerance for comparison.
        check_grad: Whether to check gradients.
    """
    if atol is None:
        atol = 1e-1 if dtype == torch.bfloat16 else 1e-2

    # Forward Pass
    print(f"Testing {name} for dtype: {dtype} (Forward)")
    out_ref = ref_func(*inputs)
    out_mac = func(*inputs)

    diff_fwd = (out_mac - out_ref).abs().max().item()
    print(f"  Forward Max Diff: {diff_fwd}")
    assert diff_fwd < atol, f"{name} forward mismatch for {dtype}: {diff_fwd}"

    if not check_grad:
        return

    # Backward Pass
    print(f"Testing {name} for dtype: {dtype} (Backward)")
    dy = torch.randn_like(out_ref)

    # Reference backward
    for x in inputs:
        if isinstance(x, torch.Tensor) and x.requires_grad:
            x.grad = None
    out_ref.backward(dy, retain_graph=True)
    grads_ref = [x.grad.clone() for x in inputs if isinstance(x, torch.Tensor) and x.requires_grad]

    # Kernel backward
    for x in inputs:
        if isinstance(x, torch.Tensor) and x.requires_grad:
            x.grad = None
    if hasattr(out_mac, "backward"):
        out_mac.backward(dy, retain_graph=True)
    else:
        # If func is not an autograd object, we can't call backward on it directly
        pass

    grads_mac = [x.grad.clone() for x in inputs if isinstance(x, torch.Tensor) and x.requires_grad]

    for i, (g_mac, g_ref) in enumerate(zip(grads_mac, grads_ref)):
        diff_grad = (g_mac - g_ref).abs().max().item()
        print(f"  Grad[{i}] Max Diff: {diff_grad}")
        assert diff_grad < atol, f"{name} grad[{i}] mismatch for {dtype}: {diff_grad}"

    print(f"  {name} passed for {dtype}!")
