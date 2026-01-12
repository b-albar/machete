# Copyright (c) 2025, Machete Authors
from abc import ABC, abstractmethod
from typing import Callable
import cutlass.cute as cute


def machete_op(num_tensors: int, needs_global_sync: bool = False, smem_per_page: int = 0, num_pages: int = 1):
    """Decorator to mark a function or method as a Machete Megakernel operation."""

    def decorator(func):
        func._machete_is_op = True
        func._machete_num_tensors = num_tensors
        func._machete_needs_sync = needs_global_sync
        func._machete_smem_per_page = smem_per_page
        func._machete_num_pages = num_pages
        return func

    return decorator


class MegakernelOp(ABC):
    """
    Base class for operations that can be run either as a simple kernel
    or fused into a megakernel.
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
    def smem_per_page(self) -> int:
        """Shared memory size needed per page in bytes."""
        return 0

    @property
    def num_pages(self) -> int:
        """Number of pages to allocate for this operation (e.g. 2 for double buffering)."""
        return 1

    @abstractmethod
    @cute.jit
    def compute(self, *args, **kwargs):
        """
        The core logic of the operation.
        If smem_per_page > 0, the first argument (after self) will be a cute.Tensor
        representing the shared memory page.
        """
        pass

    @abstractmethod
    def launch(self, *args, **kwargs):
        """
        Launch the operation as a standalone kernel.
        """
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
        smem_per_page: int = 0,
        num_pages: int = 1,
        launch_func: Callable = None,
    ):
        self._compute_func = compute_func
        self._num_tensors = num_tensors
        self._needs_sync = needs_sync
        self._smem_per_page = smem_per_page
        self._num_pages = num_pages
        self._launch_func = launch_func

    @property
    def num_tensors(self) -> int:
        return self._num_tensors

    @property
    def needs_global_sync(self) -> bool:
        return self._needs_sync

    @property
    def smem_per_page(self) -> int:
        return self._smem_per_page

    @property
    def num_pages(self) -> int:
        return self._num_pages

    @cute.jit
    def compute(self, *args, **kwargs):
        self._compute_func(*args, **kwargs)

    def launch(self, *args, **kwargs):
        if self._launch_func:
            self._launch_func(*args, **kwargs)
        else:
            raise NotImplementedError("Standalone launch not configured for this FusableOp")


class FusableKernel(ABC):
    """
    Abstract base class for kernels that support both forward and backward
    passes in a megakernel.
    """

    @property
    def smem_per_page(self) -> int:
        return 0

    @property
    def num_pages(self) -> int:
        return 1

    @abstractmethod
    @cute.jit
    def compute_forward(self, *args, **kwargs):
        """Forward pass compute logic."""
        pass

    @abstractmethod
    @cute.jit
    def compute_backward(self, *args, **kwargs):
        """Backward pass compute logic."""
        pass
