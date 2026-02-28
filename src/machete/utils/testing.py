# Copyright (c) 2025, Machete Authors
import torch
from typing import Callable, Tuple


# =============================================================================
# GPU Availability Checks
# =============================================================================


def is_hopper_available() -> bool:
    """Check if Hopper (SM90+) GPU is available.

    Hopper GPUs support advanced features like:
    - Hardware mbarrier synchronization
    - Distributed shared memory
    - TMA (Tensor Memory Accelerator)

    Returns:
        True if a Hopper or newer GPU is available.
    """
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


def is_blackwell_available() -> bool:
    """Check if Blackwell (SM120+) GPU is available.

    Blackwell GPUs provide additional optimizations for:
    - Enhanced tensor cores
    - Improved memory bandwidth
    - Advanced warp scheduling

    Returns:
        True if a Blackwell or newer GPU is available.
    """
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 12


def get_device_capability() -> Tuple[int, int]:
    """Get the CUDA compute capability of the current device.

    Returns:
        Tuple of (major, minor) version numbers, or (0, 0) if CUDA unavailable.
    """
    if not torch.cuda.is_available():
        return (0, 0)
    return torch.cuda.get_device_capability()


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
    assert diff_fwd <= atol, f"{name} forward mismatch for {dtype}: {diff_fwd}"

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
        assert diff_grad <= atol, f"{name} grad[{i}] mismatch for {dtype}: {diff_grad}"

    print(f"  {name} passed for {dtype}!")
