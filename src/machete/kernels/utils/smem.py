# Copyright (c) 2025, Machete Authors
"""
Shared Memory Allocation Utilities for Machete Kernels.

This module provides:
- MacheteSmemAllocator: A bumping allocator for shared memory tensors
- Helper methods for TMA-compatible tensor allocation

Usage:
    @cute.jit
    def _setup_storage(self, smem):
        alloc = MacheteSmemAllocator(smem)
        # Allocate with TMA-friendly alignment (1024 bytes)
        sA = alloc.allocate_tensor(cute.Float16, cute.make_layout((128, 64, 4)), byte_alignment=1024)
        sB = alloc.allocate_tensor(cute.Float16, cute.make_layout((64, 128, 4)), byte_alignment=1024)
        # Allocate barriers
        barriers = alloc.allocate_tensor(cute.Int64, cute.make_layout((16,)))
        return sA, sB, barriers.iterator
"""

import cutlass.cute as cute


class MacheteSmemAllocator:
    """
    A simple bumping allocator for shared memory.

    If initialized with a base_tensor, it allocates within that tensor (e.g. a page).
    Otherwise, it allocates from the start of dynamic shared memory.

    TMA Compatibility Notes:
    ------------------------
    For TMA operations, shared memory tensors need:
    1. Proper address space metadata (handled automatically)
    2. 1024-byte alignment for optimal performance (use byte_alignment=1024)
    3. Contiguous layouts matching the TMA tile shape

    The allocator preserves address space information through slicing operations
    when using cute.slice_() instead of Python slicing syntax.

    Example for TMA-compatible allocation:
        alloc = MacheteSmemAllocator(smem)
        # Multi-stage buffer for pipelined loads
        sA = alloc.allocate_tensor(
            cute.Float16,
            cute.make_layout((tile_m, tile_k, num_stages), stride=(tile_k, 1, tile_m * tile_k)),
            byte_alignment=1024
        )
        # Access stage i with: cute.slice_(sA, (None, None, stage_idx))
    """

    # Default alignment for TMA operations
    TMA_ALIGNMENT = 1024

    def __init__(self, base_tensor=None):
        if base_tensor is not None:
            # Start from the provided base tensor
            # We must use Uint8 for byte-level math and alignment
            self._base = cute.recast_ptr(base_tensor.iterator, dtype=cute.Uint8)
            # Store capacity for bounds checking if possible
            self._capacity = cute.size(base_tensor.layout) * (base_tensor.element_type.width // 8)
            self._allocated_bytes = 0
        else:
            # Allocate from dynamic shared memory starting at 0
            # Alignment 1024 is the default for SmemAllocator
            self._base = cute.runtime.make_ptr(cute.Uint8, 0, cute.AddressSpace.shared, assumed_align=1024)
            self._capacity = None

    def allocate_tensor(self, element_type, layout, byte_alignment=16):
        """Allocate a tensor in shared memory.

        Args:
            element_type: CuTe dtype (e.g., cute.Float16, cute.BFloat16)
            layout: CuTe layout for the tensor
            byte_alignment: Alignment in bytes (use 1024 for TMA)

        Returns:
            CuTe tensor backed by shared memory
        """
        # Align current base pointer
        self._base = self._base.align(byte_alignment)

        # Record current position as our allocation start
        allocated_ptr = self._base

        # Increment base by size in bytes
        # Since self._base is a Uint8 pointer, addition is at byte granularity
        element_size = element_type.width // 8
        size_bytes = cute.size(layout) * element_size
        self._base += size_bytes

        if self._capacity is not None:
            self._allocated_bytes += size_bytes
            # Note: doesn't account for alignment padding yet

        # Recast the pointer to target type and return the tensor
        return cute.make_tensor(cute.recast_ptr(allocated_ptr, dtype=element_type), layout)

    def allocate_array(self, element_type, num_elems, byte_alignment=16):
        """Allocate a 1D array in shared memory."""
        layout = cute.make_layout(num_elems)
        return self.allocate_tensor(element_type, layout, byte_alignment)

    def allocate_tma_buffer(self, element_type, tile_shape, num_stages, byte_alignment=1024):
        """Allocate a multi-stage buffer suitable for TMA pipelining.

        This is a convenience method for the common pattern of allocating
        a multi-stage shared memory buffer for pipelined TMA loads.

        Args:
            element_type: CuTe dtype (e.g., cute.Float16)
            tile_shape: Tuple of (dim0, dim1) for the tile size
            num_stages: Number of pipeline stages
            byte_alignment: Alignment in bytes (default 1024 for TMA)

        Returns:
            CuTe tensor with shape (dim0, dim1, num_stages) and row-major layout per stage

        Example:
            # Allocate a 4-stage buffer for 128x64 tiles
            sA = alloc.allocate_tma_buffer(cute.Float16, (128, 64), num_stages=4)
            # Access stage i: cute.slice_(sA, (None, None, stage_idx))
        """
        dim0, dim1 = tile_shape
        # Row-major per stage: stride = (dim1, 1, dim0 * dim1)
        layout = cute.make_layout((dim0, dim1, num_stages), stride=(dim1, 1, dim0 * dim1))
        return self.allocate_tensor(element_type, layout, byte_alignment)

    def allocate_barriers(self, num_barriers):
        """Allocate barrier storage for pipeline synchronization.

        Args:
            num_barriers: Number of 64-bit barriers to allocate

        Returns:
            Iterator to the barrier storage (pass to PipelineTmaAsync.create)
        """
        barrier_tensor = self.allocate_tensor(cute.Int64, cute.make_layout((num_barriers,)))
        return barrier_tensor.iterator

    def allocate_swizzled_tensor(self, element_type, composed_layout, byte_alignment=1024):
        """Allocate a tensor with a swizzled (ComposedLayout) layout.

        This method handles ComposedLayout objects returned by utilities like
        `make_smem_layout_a/b` which include swizzle patterns for LdMatrix compatibility.

        Args:
            element_type: CuTe dtype (e.g., cute.Float16)
            composed_layout: A ComposedLayout with .outer (logical shape) and .inner (swizzle)
            byte_alignment: Alignment in bytes (default 1024 for TMA)

        Returns:
            CuTe tensor backed by shared memory with the swizzled layout

        Example:
            from cutlass.utils.hopper_helpers import make_smem_layout_a
            layout_a = make_smem_layout_a(cute.Float16, (128, 64), LayoutEnum.RowMajor)
            sA = alloc.allocate_swizzled_tensor(cute.Float16, layout_a, 1024)
        """
        # Align current base pointer
        self._base = self._base.align(byte_alignment)

        # Record current position as our allocation start
        allocated_ptr = self._base

        # For ComposedLayout, the size is computed from the outer layout
        # The swizzle (.inner) doesn't change the total element count
        element_size = element_type.width // 8
        size_bytes = cute.size(composed_layout) * element_size
        self._base += size_bytes

        if self._capacity is not None:
            self._allocated_bytes += size_bytes

        # Create tensor with the composed layout (includes swizzle)
        return cute.make_tensor(cute.recast_ptr(allocated_ptr, dtype=element_type), composed_layout)
