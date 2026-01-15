# Copyright (c) 2025, Machete Authors
from abc import abstractmethod
from typing import Callable
import cutlass.cute as cute

# Re-export dependency decorators for convenience
from machete.megakernel.scheduler import reads, writes, independent  # noqa: F401

__all__ = ["reads", "writes", "independent", "machete_op", "MegakernelOp", "FusableOp", "FusableKernel"]


def machete_op(num_tensors: int, needs_global_sync: bool = False, smem_per_stage: int = 0, num_stages: int = 1):
    """Decorator to mark a function or method as a Machete Megakernel operation."""

    def decorator(func):
        func._machete_is_op = True
        func._machete_num_tensors = num_tensors
        func._machete_needs_sync = needs_global_sync
        func._machete_smem_per_stage = smem_per_stage
        func._machete_num_stages = num_stages
        return func

    return decorator


class MegakernelOp:
    """
    Base class for operations that can be run either as a simple kernel
    or fused into a megakernel.

    Operations implement Load/Compute/Store phases:
    - load(): Load data from global memory to shared memory
    - compute(): Perform computation using shared memory
    - store(): Store results from shared memory back to global memory
    """

    @property
    @abstractmethod
    def num_tensors(self) -> int:
        """Number of tensors this operation expects in its compute method."""
        pass

    @property
    def needs_global_sync(self) -> bool:
        """Whether this operation needs a global barrier after its execution."""
        return False

    @property
    def smem_per_stage(self) -> int:
        """Shared memory size needed per page in bytes."""
        return 0

    @property
    def num_stages(self) -> int:
        """Number of pages to allocate for this operation (e.g. 2 for double buffering)."""
        return 1

    @cute.jit
    def load(self, paged_pool, page_idx, *args):
        """Load phase: move data from global memory to shared memory."""
        pass  # Default no-op

    @abstractmethod
    @cute.jit
    def compute(self, *args, **kwargs):
        """Compute phase: perform the operation logic."""
        pass

    @cute.jit
    def store(self, paged_pool, page_idx, *args):
        """Store phase: move results from shared memory to global memory."""
        pass  # Default no-op

    @abstractmethod
    def launch(self, *args, **kwargs):
        """Launch the operation as a standalone kernel."""
        pass


class FusableOp(MegakernelOp):
    """
    A generic wrapper that turns any @cute.jit function or method into a MegakernelOp.
    """

    def __init__(
        self,
        compute_func: Callable,
        num_tensors: int,
        needs_sync: bool = False,
        smem_per_stage: int = 0,
        num_stages: int = 1,
        load_func: Callable = None,
        store_func: Callable = None,
        launch_func: Callable = None,
    ):
        self._compute_func = compute_func
        self._load_func = load_func
        self._store_func = store_func
        self._num_tensors = num_tensors
        self._needs_sync = needs_sync
        self._smem_per_stage = smem_per_stage
        self._num_stages = num_stages
        self._launch_func = launch_func

    @property
    def num_tensors(self) -> int:
        return self._num_tensors

    @property
    def needs_global_sync(self) -> bool:
        return self._needs_sync

    @property
    def smem_per_stage(self) -> int:
        return self._smem_per_stage

    @property
    def num_stages(self) -> int:
        return self._num_stages

    @cute.jit
    def load(self, paged_pool, page_idx, *args):
        # Default no-op, subclasses override if needed
        pass

    @cute.jit
    def compute(self, *args, **kwargs):
        self._compute_func(*args, **kwargs)

    @cute.jit
    def store(self, paged_pool, page_idx, *args):
        # Default no-op, subclasses override if needed
        pass

    def launch(self, *args, **kwargs):
        if self._launch_func:
            self._launch_func(*args, **kwargs)
        else:
            raise NotImplementedError("Standalone launch not configured for this FusableOp")


class FusableKernel:
    """
    Base class for kernels that support both forward and backward
    passes in a megakernel with Load/Compute/Store phases.
    """

    @property
    def smem_per_stage(self) -> int:
        return 0

    @property
    def num_stages(self) -> int:
        return 1

    # ========== Forward Pass L/C/S ==========

    @cute.jit
    def load_forward(self, paged_pool, page_idx, *args):
        """Forward load phase (optional)."""
        pass

    @cute.jit
    def compute_forward(self, *args, **kwargs):
        """Forward pass compute logic."""
        pass

    @cute.jit
    def store_forward(self, paged_pool, page_idx, *args):
        """Forward store phase (optional)."""
        pass

    # ========== Backward Pass L/C/S ==========

    @cute.jit
    def load_backward(self, paged_pool, page_idx, *args):
        """Backward load phase (optional)."""
        pass

    @cute.jit
    def compute_backward(self, *args, **kwargs):
        """Backward pass compute logic."""
        pass

    @cute.jit
    def store_backward(self, paged_pool, page_idx, *args):
        """Backward store phase (optional)."""
        pass
