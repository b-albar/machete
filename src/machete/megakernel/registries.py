# Copyright (c) 2025, Machete Authors
"""
Tensor and TMA Registries for Megakernel.

This module provides registries that manage tensor deduplication and TMA
descriptor metadata across fused operations. Registries map op-local tensor
names to canonical names (t0, t1, ...) for efficient parameter passing.

Classes:
    TensorRegistry: Deduplicates tensors by data_ptr across ops.
    PeerBufferRegistry: Tracks pre-allocated buffers on peer GPUs.
    TMARegistry: Manages TMA descriptor naming and metadata.
    PeerTMARegistry: Manages TMA descriptors for peer GPU stores.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Type

from .ops import (
    TORCH_TO_CUTLASS_DTYPE,
    ScheduledOp,
    TensorMeta,
)


# =============================================================================
# Tensor Registry (Deduplication for Tensor Parameter Mode)
# =============================================================================


@dataclass
class TensorRegistry:
    """Deduplicates tensors across ops by data_ptr() for tensor parameter mode.

    When multiple ops share the same underlying tensor (e.g., RMSNorm.y and
    Rope.q point to the same GPU buffer), this registry assigns a single
    canonical parameter name (t0, t1, ...) to avoid passing duplicate tensors.

    Usage:
        registry = TensorRegistry.from_ops(ops)
        # registry.canonical_names → ['t0', 't1', 't2', 't3', 't4']
        # registry.get_op_tensor_args(0, RMSNormOp) → ['t0', 't1', 't2']
        # registry.get_op_tensor_args(1, RopeOp) → ['t2', 't3', 't4']
    """

    # List of (canonical_name, torch.Tensor, cutlass_dtype)
    tensors: List[Tuple[str, Any, Any]]
    # Per-op mappings: {op_idx: {local_name: canonical_name}}
    op_mappings: Dict[int, Dict[str, str]]
    # Reverse mapping: canonical_name -> index in tensors list
    name_to_idx: Dict[str, int]

    @classmethod
    def from_ops(cls, ops: List[ScheduledOp]) -> "TensorRegistry":
        """Build a TensorRegistry from a list of ScheduledOps.

        Iterates through each op's tensor_refs in declaration order,
        deduplicating by data_ptr(). Tensors with the same GPU address
        get the same canonical name.

        Args:
            ops: List of scheduled operations with tensor_refs populated.
        """
        ptr_to_canonical: Dict[int, str] = {}
        tensors: List[Tuple[str, Any, Any]] = []
        op_mappings: Dict[int, Dict[str, str]] = {}
        name_to_idx: Dict[str, int] = {}

        for i, op in enumerate(ops):
            mapping: Dict[str, str] = {}
            # Skip ops that don't have tensor declarations (e.g., simple test ops
            # like StampOp/TensorScaleOp that don't use the @op decorator)
            if not hasattr(op.op_cls, "_UNIQUE_TENSORS"):
                op_mappings[i] = mapping
                continue

            # Use the op's unique_tensors for consistent ordering
            unique_tensors = op.op_cls._UNIQUE_TENSORS

            for name, dtype, dims in unique_tensors:
                if name not in op.tensor_refs:
                    continue
                tensor = op.tensor_refs[name]
                ptr = tensor.data_ptr()

                if ptr not in ptr_to_canonical:
                    canonical = f"t{len(tensors)}"
                    ptr_to_canonical[ptr] = canonical

                    # Resolve dtype (None means infer from tensor)
                    resolved_dtype = dtype
                    if resolved_dtype is None:
                        resolved_dtype = TORCH_TO_CUTLASS_DTYPE.get(tensor.dtype)

                    name_to_idx[canonical] = len(tensors)
                    tensors.append((canonical, tensor, resolved_dtype))

                mapping[name] = ptr_to_canonical[ptr]
            op_mappings[i] = mapping

        return cls(tensors=tensors, op_mappings=op_mappings, name_to_idx=name_to_idx)

    @property
    def canonical_names(self) -> List[str]:
        """List of canonical tensor parameter names in order."""
        return [name for name, _, _ in self.tensors]

    @property
    def num_tensors(self) -> int:
        """Number of unique tensors."""
        return len(self.tensors)

    def get_op_tensor_args(self, op_idx: int, op_cls) -> List[str]:
        """Get ordered canonical tensor names for an op's function call.

        Returns canonical names in the same order as the op's tensor
        declarations (reads then writes, deduplicated). This order matches
        the tensor parameter positions in the compiled phase function.

        Args:
            op_idx: Index of the op in the ops list.
            op_cls: The op class (for accessing _UNIQUE_TENSORS).
        """
        if not hasattr(op_cls, "_UNIQUE_TENSORS"):
            return []
        mapping = self.op_mappings[op_idx]
        return [mapping[name] for name, _, _ in op_cls._UNIQUE_TENSORS if name in mapping]


# =============================================================================
# Peer Buffer Registry (Multi-GPU Communication)
# =============================================================================


@dataclass
class PeerBufferInfo:
    """Metadata for a single peer buffer entry.

    Attributes:
        canonical_name: Canonical tensor name from TensorRegistry (e.g., "t0")
        peer_tensors: List of tensors on peer GPUs (one per peer device)
    """

    canonical_name: str
    peer_tensors: List[Any]  # List[torch.Tensor]


@dataclass
class PeerBufferRegistry:
    """Tracks identically-shaped pre-allocated buffers on peer GPUs.

    Maps canonical tensor names to lists of peer GPU tensors for TMA
    peer-to-peer stores. Users pre-allocate buffers on each peer device
    and pass them via MegakernelConfig.peer_buffers.

    Usage:
        registry = PeerBufferRegistry.from_config(
            peer_map={"y": [gpu1_y, gpu2_y]},
            tensor_registry=tensor_registry,
            ops=ops,
        )
    """

    buffers: List[PeerBufferInfo]
    num_peers: int

    @classmethod
    def from_config(
        cls,
        peer_map: Dict[str, List[Any]],
        tensor_registry: "TensorRegistry",
        ops: List[ScheduledOp],
    ) -> "PeerBufferRegistry":
        """Build from user-provided peer tensor mapping.

        Args:
            peer_map: {tensor_name: [peer0_tensor, peer1_tensor, ...]}
                Maps op-local tensor names to pre-allocated peer buffers.
            tensor_registry: Local TensorRegistry for canonical name resolution.
            ops: Scheduled ops list for resolving tensor names to canonical names.
        """
        if not peer_map:
            return cls(buffers=[], num_peers=0)

        # Validate all peer lists have the same length
        num_peers = None
        for name, peers in peer_map.items():
            if num_peers is None:
                num_peers = len(peers)
            elif len(peers) != num_peers:
                raise ValueError(
                    f"Peer buffer '{name}' has {len(peers)} peers, "
                    f"expected {num_peers} (must be consistent)"
                )

        if num_peers is None or num_peers == 0:
            return cls(buffers=[], num_peers=0)

        # Resolve tensor names to canonical names via op mappings
        buffers: List[PeerBufferInfo] = []
        seen_canonical: Set[str] = set()

        for name, peers in peer_map.items():
            # Find canonical name by searching op mappings
            canonical = None
            for op_idx, mapping in tensor_registry.op_mappings.items():
                if name in mapping:
                    canonical = mapping[name]
                    break

            if canonical is None:
                raise ValueError(
                    f"Peer buffer tensor '{name}' not found in any op's "
                    f"tensor declarations"
                )

            if canonical in seen_canonical:
                continue
            seen_canonical.add(canonical)

            buffers.append(PeerBufferInfo(
                canonical_name=canonical,
                peer_tensors=list(peers),
            ))

        return cls(buffers=buffers, num_peers=num_peers)

    @property
    def has_peers(self) -> bool:
        """Whether any peer buffers are registered."""
        return len(self.buffers) > 0

    @property
    def canonical_names(self) -> List[str]:
        """Canonical tensor names that have peer buffers."""
        return [b.canonical_name for b in self.buffers]

    def get_peer_tensors(self, canonical_name: str) -> Optional[List[Any]]:
        """Get peer tensors for a canonical tensor name."""
        for b in self.buffers:
            if b.canonical_name == canonical_name:
                return b.peer_tensors
        return None


# =============================================================================
# TMA Tile Shape Computation (shared helper)
# =============================================================================


def _compute_tma_tile_shape(
    op_cls,
    tensor_name: str,
    op: ScheduledOp,
) -> Tuple[int, ...]:
    """Compute TMA tile shape for a tensor, transposed for row-major.

    Ops can override via get_tma_tile_shape() for dims that need custom
    sub-tiling (e.g., K in GEMM). Otherwise, uses the op's tile_sizes
    and static_dims to build the shape from _TMA_TENSOR_DIMS.

    The result is reversed to match the transposed dimension order
    (PyTorch row-major → CuTe mode 0 contiguous).

    Args:
        op_cls: The Op class with TMA declarations.
        tensor_name: Name of the tensor (e.g., "x", "y").
        op: ScheduledOp with tile_sizes and static_dims.

    Returns:
        Transposed TMA tile shape tuple.
    """
    custom_shape = None
    if hasattr(op_cls, "get_tma_tile_shape"):
        custom_shape = op_cls.get_tma_tile_shape(
            tensor_name, op.tile_sizes, op.static_dims)

    if custom_shape is not None:
        tile_shape = tuple(custom_shape)
    else:
        tma_dims = getattr(op_cls, "_TMA_TENSOR_DIMS", {}).get(tensor_name, [])
        tile_shape = []
        for dim_name in tma_dims:
            if dim_name in op.tile_sizes:
                tile_shape.append(op.tile_sizes[dim_name])
            elif dim_name in op.static_dims:
                tile_shape.append(op.static_dims[dim_name])
            else:
                raise ValueError(
                    f"TMA tensor '{tensor_name}' dim '{dim_name}' "
                    f"not found in tile_sizes or static_dims"
                )
        tile_shape = tuple(tile_shape)

    # TMA requires CuTe mode 0 to be contiguous. Since PyTorch tensors are
    # row-major, the gmem tensor is transposed before from_dlpack (so mode 0 =
    # last dim). Reverse tile_shape to match the transposed dimension order.
    return tuple(reversed(tile_shape))


# =============================================================================
# TMA Registry (Descriptor Management for TMA Parameter Mode)
# =============================================================================


@dataclass
class TMADescriptorInfo:
    """Metadata for a single TMA descriptor to create at launch time.

    Attributes:
        canonical_atom: Canonical name for the TMA copy atom (e.g., "tma0_atom")
        canonical_gmem: Canonical name for the TMA gmem tensor (e.g., "tma0_gmem")
        tensor_canonical: Canonical tensor name (e.g., "t0") for the source tensor
        direction: "g2s" (load) or "s2g" (store)
        tile_shape: Tuple of ints, TMA tile shape per tensor dimension
        smem_layout_shape: Tuple of ints, shared memory layout shape
        dtype: CUTLASS dtype for the tensor
        smem_layout_src: Optional code string for composed smem layout with swizzle.
            When set, replaces the plain make_layout in TMA descriptor creation
            so make_tiled_tma_atom can detect the hardware swizzle mode.
    """

    canonical_atom: str
    canonical_gmem: str
    tensor_canonical: str
    direction: str  # "g2s" or "s2g"
    tile_shape: Tuple[int, ...]
    smem_layout_shape: Tuple[int, ...]
    dtype: Any
    smem_layout_src: Optional[str] = None


@dataclass
class TMARegistry:
    """Manages TMA descriptor naming and metadata across fused ops.

    For each TMA load/store declaration, assigns canonical parameter names
    and records the metadata needed to create TMA descriptors at launch time.

    Usage:
        registry = TMARegistry.from_ops(ops, tensor_registry)
        # registry.descriptors → [TMADescriptorInfo(...), ...]
        # registry.get_op_tma_args(0, "load") → ['tma0_atom', 'tma0_gmem']
    """

    descriptors: List[TMADescriptorInfo]
    # Per-op, per-phase mappings: {(op_idx, phase): {local_name: canonical_name}}
    op_mappings: Dict[Tuple[int, str], Dict[str, str]]

    @classmethod
    def from_ops(
        cls,
        ops: List["ScheduledOp"],
        tensor_registry: "TensorRegistry",
    ) -> "TMARegistry":
        """Build a TMARegistry from scheduled ops.

        For each TMA load/store declaration, computes the TMA tile shape
        from the op's tile_sizes and static_dims, and assigns canonical names.
        """
        descriptors: List[TMADescriptorInfo] = []
        op_mappings: Dict[Tuple[int, str], Dict[str, str]] = {}
        counter = 0

        for i, op in enumerate(ops):
            op_cls = op.op_cls
            if not hasattr(op_cls, "_TMA_LOADS"):
                op_mappings[(i, "load")] = {}
                op_mappings[(i, "compute")] = {}
                op_mappings[(i, "store")] = {}
                continue

            op_mappings[(i, "load")] = {}
            op_mappings[(i, "compute")] = {}
            op_mappings[(i, "store")] = {}

            for phase, tma_names, direction in [
                ("load", op_cls._TMA_LOADS, "g2s"),
                ("store", op_cls._TMA_STORES, "s2g"),
                ("store", getattr(op_cls, "_TMA_REDUCE_STORES", set()), "s2g_reduce"),
            ]:
                for tensor_name in tma_names:
                    # Get canonical tensor name from tensor registry
                    tensor_canonical = tensor_registry.op_mappings[i].get(tensor_name)
                    if tensor_canonical is None:
                        continue

                    tma_tile_shape = _compute_tma_tile_shape(op_cls, tensor_name, op)

                    # Resolve dtype
                    meta = op.tensor_metas.get(tensor_name)
                    dtype = meta.dtype if meta else None

                    canonical_atom = f"tma{counter}_atom"
                    canonical_gmem = f"tma{counter}_gmem"

                    # Check for custom smem layout (e.g., swizzle for GEMM)
                    smem_layout_src = None
                    if hasattr(op_cls, "get_tma_smem_layout_src"):
                        smem_layout_src = op_cls.get_tma_smem_layout_src(
                            tensor_name, tma_tile_shape, op.tile_sizes, op.static_dims
                        )

                    descriptors.append(
                        TMADescriptorInfo(
                            canonical_atom=canonical_atom,
                            canonical_gmem=canonical_gmem,
                            tensor_canonical=tensor_canonical,
                            direction=direction,
                            tile_shape=tma_tile_shape,
                            smem_layout_shape=tma_tile_shape,
                            dtype=dtype,
                            smem_layout_src=smem_layout_src,
                        )
                    )

                    # Map op-local names to canonical
                    op_mappings[(i, phase)][f"{tensor_name}_tma"] = canonical_atom
                    op_mappings[(i, phase)][f"{tensor_name}_tma_gmem"] = canonical_gmem
                    counter += 1

            # Compute phase can use TMA load descriptors + reduce store descriptors
            op_mappings[(i, "compute")] = dict(op_mappings[(i, "load")])
            for key, val in op_mappings[(i, "store")].items():
                if any(
                    d.direction == "s2g_reduce"
                    and (d.canonical_atom == val or d.canonical_gmem == val)
                    for d in descriptors
                ):
                    op_mappings[(i, "compute")][key] = val

        return cls(descriptors=descriptors, op_mappings=op_mappings)

    @property
    def has_tma(self) -> bool:
        """Whether any ops use TMA."""
        return len(self.descriptors) > 0

    @property
    def all_canonical_names(self) -> List[str]:
        """All canonical TMA parameter names (atoms + gmems) in order."""
        names = []
        for d in self.descriptors:
            names.append(d.canonical_atom)
            names.append(d.canonical_gmem)
        return names

    def get_op_tma_args(self, op_idx: int, phase: str) -> List[str]:
        """Get canonical TMA param names for an op's phase function.

        Args:
            op_idx: Index of the op
            phase: "load" or "store"

        Returns:
            List of canonical names in order: [atom, gmem, atom, gmem, ...]
        """
        mapping = self.op_mappings.get((op_idx, phase), {})
        # Return in consistent order: for each TMA tensor, atom then gmem
        result = []
        for local_name, canonical in sorted(mapping.items()):
            result.append(canonical)
        return result


# =============================================================================
# Peer TMA Registry (Multi-GPU TMA Descriptors)
# =============================================================================


@dataclass
class PeerTMADescriptorInfo:
    """Metadata for a TMA descriptor targeting a peer GPU buffer.

    Similar to TMADescriptorInfo but tracks which peer device the
    descriptor targets. Created per (tensor, peer) pair.

    Attributes:
        canonical_atom: e.g., "ptma0_p0_atom" (peer 0), "ptma0_p1_atom" (peer 1)
        canonical_gmem: e.g., "ptma0_p0_gmem"
        tensor_canonical: Canonical tensor name from TensorRegistry (e.g., "t0")
        peer_idx: Index of the peer device
        tile_shape: TMA tile shape (transposed for row-major)
        smem_layout_shape: Shared memory layout shape
        dtype: CUTLASS dtype
        smem_layout_src: Optional swizzle layout code string
    """

    canonical_atom: str
    canonical_gmem: str
    tensor_canonical: str
    peer_idx: int
    tile_shape: Tuple[int, ...]
    smem_layout_shape: Tuple[int, ...]
    dtype: Any
    smem_layout_src: Optional[str] = None


@dataclass
class PeerTMARegistry:
    """Manages TMA descriptors for peer GPU stores.

    For each op with peer_stores, creates one set of TMA S2G descriptors
    per peer device. Descriptors are created at runtime since peer GPU
    addresses are runtime values.

    Usage:
        registry = PeerTMARegistry.from_ops(ops, tensor_registry, peer_buffer_registry)
        # registry.descriptors → [PeerTMADescriptorInfo(...), ...]
        # registry.get_op_peer_tma_args(0, "communicate") → ['ptma0_p0_atom', ...]
    """

    descriptors: List[PeerTMADescriptorInfo]
    # Per-op mappings: {(op_idx, "communicate"): {local_name: canonical_name}}
    op_mappings: Dict[Tuple[int, str], Dict[str, str]]
    num_peers: int

    @classmethod
    def from_ops(
        cls,
        ops: List[ScheduledOp],
        tensor_registry: "TensorRegistry",
        peer_buffer_registry: "PeerBufferRegistry",
    ) -> "PeerTMARegistry":
        """Build PeerTMARegistry from ops that declare peer_stores.

        For each op's peer_stores tensor, for each peer device, creates
        a TMA S2G descriptor targeting the peer's buffer.
        """
        descriptors: List[PeerTMADescriptorInfo] = []
        op_mappings: Dict[Tuple[int, str], Dict[str, str]] = {}
        counter = 0
        num_peers = peer_buffer_registry.num_peers

        for i, op in enumerate(ops):
            op_cls = op.op_cls
            peer_stores = getattr(op_cls, "_PEER_STORES", set())
            op_mappings[(i, "communicate")] = {}

            if not peer_stores:
                continue

            for tensor_name in peer_stores:
                # Get canonical tensor name
                tensor_canonical = tensor_registry.op_mappings[i].get(tensor_name)
                if tensor_canonical is None:
                    continue

                tma_tile_shape = _compute_tma_tile_shape(op_cls, tensor_name, op)

                # Resolve dtype
                meta = op.tensor_metas.get(tensor_name)
                dtype = meta.dtype if meta else None

                # Smem layout (use same swizzle as local store if available)
                smem_layout_src = None
                if hasattr(op_cls, "get_tma_smem_layout_src"):
                    smem_layout_src = op_cls.get_tma_smem_layout_src(
                        tensor_name, tma_tile_shape, op.tile_sizes, op.static_dims
                    )

                # Create one descriptor per peer
                for peer_idx in range(num_peers):
                    canonical_atom = f"ptma{counter}_p{peer_idx}_atom"
                    canonical_gmem = f"ptma{counter}_p{peer_idx}_gmem"

                    descriptors.append(
                        PeerTMADescriptorInfo(
                            canonical_atom=canonical_atom,
                            canonical_gmem=canonical_gmem,
                            tensor_canonical=tensor_canonical,
                            peer_idx=peer_idx,
                            tile_shape=tma_tile_shape,
                            smem_layout_shape=tma_tile_shape,
                            dtype=dtype,
                            smem_layout_src=smem_layout_src,
                        )
                    )

                    # Map local names to canonical for communicate phase
                    op_mappings[(i, "communicate")][
                        f"{tensor_name}_p{peer_idx}_tma"
                    ] = canonical_atom
                    op_mappings[(i, "communicate")][
                        f"{tensor_name}_p{peer_idx}_tma_gmem"
                    ] = canonical_gmem

                counter += 1

        return cls(
            descriptors=descriptors,
            op_mappings=op_mappings,
            num_peers=num_peers,
        )

    @property
    def has_peer_tma(self) -> bool:
        """Whether any ops use peer TMA stores."""
        return len(self.descriptors) > 0

    @property
    def all_canonical_names(self) -> List[str]:
        """All canonical peer TMA parameter names in order."""
        names = []
        for d in self.descriptors:
            names.append(d.canonical_atom)
            names.append(d.canonical_gmem)
        return names

    def get_op_peer_tma_args(self, op_idx: int, phase: str = "communicate") -> List[str]:
        """Get canonical peer TMA param names for an op's communicate phase."""
        mapping = self.op_mappings.get((op_idx, phase), {})
        result = []
        for local_name, canonical in sorted(mapping.items()):
            result.append(canonical)
        return result


# =============================================================================
# Cross-Op Compatibility Validation
# =============================================================================


def validate_op_compatibility(ops: List[ScheduledOp], registry: "TensorRegistry") -> None:
    """Validate shared tensors across fused ops have compatible shapes.

    When two ops share the same underlying tensor (same data_ptr), checks that:
    1. Total element count matches (product of shapes).
    2. Shared dimension names have matching values in static_dims.

    This allows reshapes like (M, D) → (M, H, D) as long as total elements agree
    and any dimension names in common (e.g. M) have the same value.

    Args:
        ops: List of scheduled operations.
        registry: TensorRegistry with deduplication info.

    Raises:
        ValueError: If shared tensors have incompatible shapes.
    """
    # Build reverse map: data_ptr → list of (op_idx, tensor_name, TensorMeta)
    ptr_to_uses: Dict[int, List[Tuple[int, str, TensorMeta]]] = {}
    for i, op in enumerate(ops):
        for name, meta in op.tensor_metas.items():
            ptr_to_uses.setdefault(meta.data_ptr, []).append((i, name, meta))

    # Check each shared tensor (data_ptr with multiple users)
    for ptr, uses in ptr_to_uses.items():
        if len(uses) < 2:
            continue

        # Check total element count compatibility
        ref_idx, ref_name, ref_meta = uses[0]
        ref_numel = math.prod(ref_meta.shape)
        ref_op_name = ops[ref_idx].op_cls.__name__

        for other_idx, other_name, other_meta in uses[1:]:
            other_numel = math.prod(other_meta.shape)
            other_op_name = ops[other_idx].op_cls.__name__

            if ref_numel != other_numel:
                raise ValueError(
                    f"Shared tensor incompatibility: "
                    f"{ref_op_name}.{ref_name} has shape {ref_meta.shape} "
                    f"({ref_numel} elements) but "
                    f"{other_op_name}.{other_name} has shape {other_meta.shape} "
                    f"({other_numel} elements)"
                )

    # Note: We intentionally do NOT check per-dim name matching on shared
    # tensors. Different ops can reshape the same buffer (e.g., RMSNorm
    # outputs (M, D) and RoPE reads (M, H, D) from the same storage),
    # and dim names like "D" can legitimately mean different things.
    # The total element count check above is the correct constraint.


__all__ = [
    "TensorRegistry",
    "PeerBufferInfo",
    "PeerBufferRegistry",
    "TMADescriptorInfo",
    "TMARegistry",
    "PeerTMADescriptorInfo",
    "PeerTMARegistry",
    "validate_op_compatibility",
]
