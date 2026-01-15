# Copyright (c) 2025, Machete Authors
import cutlass.cute as cute


class MacheteSmemAllocator:
    """
    A simple bumping allocator for shared memory.
    If initialized with a base_tensor, it allocates within that tensor (e.g. a page).
    Otherwise, it allocates from the start of dynamic shared memory.
    """

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
        # Align current base pointer
        self._base = self._base.align(byte_alignment)

        # Increment allocated bytes due to alignment
        # (This is a bit tricky to track exactly without integer ptrs, but we can approximate or use static checks)

        # Record current position as our allocation start
        allocated_ptr = self._base

        # Increment base by size in bytes
        # Since self._base is a Uint8 pointer, addition is at byte granularity
        element_size = element_type.width // 8
        size_bytes = cute.size(layout) * element_size
        self._base += size_bytes

        if self._capacity is not None:
            self._allocated_bytes += size_bytes  # Note: doesn't account for alignment padding yet
            # In a real JIT we'd use a static assert or a runtime check
            # For now, this is mostly for documentation and future-proofing
            pass

        # Recast the pointer to target type and return the tensor
        return cute.make_tensor(cute.recast_ptr(allocated_ptr, dtype=element_type), layout)

    def allocate_array(self, element_type, num_elems, byte_alignment=16):
        layout = cute.make_layout(num_elems)
        return self.allocate_tensor(element_type, layout, byte_alignment)
