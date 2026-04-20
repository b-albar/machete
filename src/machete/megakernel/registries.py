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
from typing import Any, Dict, List, Optional, Set, Tuple

from .ops import (
    TORCH_TO_CUTLASS_DTYPE,
    ScheduledOp,
    TensorMeta,
)

TMA_PHASE_SPECS = (
    ("load", "_TMA_LOADS", "g2s"),
    ("store", "_TMA_STORES", "s2g"),
    ("store", "_TMA_REDUCE_STORES", "s2g_reduce"),
)

PEER_TMA_DIRECTION_SPECS = (
    ("_PEER_STORES", "s2g"),
    ("_PEER_REDUCE_STORES", "s2g_reduce"),
)


def _canonical_name_list(descriptors) -> List[str]:
    """Flatten descriptor atom/gmem names in declaration order."""
    names = []
    seen = set()
    for descriptor in descriptors:
        if descriptor.canonical_atom not in seen:
            seen.add(descriptor.canonical_atom)
            names.append(descriptor.canonical_atom)
        if descriptor.canonical_gmem not in seen:
            seen.add(descriptor.canonical_gmem)
            names.append(descriptor.canonical_gmem)
    return names


def _sorted_mapping_values(mapping: Dict[str, str]) -> List[str]:
    """Return canonical names ordered by sorted local parameter name."""
    return [canonical for _local_name, canonical in sorted(mapping.items())]


def _iter_tma_phase_specs(op_cls):
    """Yield `(phase, tensor_names, direction)` triples for TMA declarations."""
    for phase, attr_name, direction in TMA_PHASE_SPECS:
        yield phase, getattr(op_cls, attr_name, set()), direction


def _iter_peer_tma_specs(op_cls):
    """Yield `(tensor_name, direction)` pairs for peer TMA declarations."""
    for attr_name, direction in PEER_TMA_DIRECTION_SPECS:
        for tensor_name in getattr(op_cls, attr_name, set()):
            yield tensor_name, direction


def _tensor_dim_permutation(tensor) -> Tuple[int, ...]:
    """Return a stride-sorted permutation suitable for TMA descriptor creation."""
    return tuple(sorted(range(tensor.ndim), key=lambda axis: (tensor.stride(axis), -axis)))


def _resolve_tensor_canonical_name(
    tensor_name: str,
    tensor_registry: "TensorRegistry",
) -> Optional[str]:
    """Resolve an op-local tensor name to its canonical registry name."""
    for mapping in tensor_registry.op_mappings.values():
        if tensor_name in mapping:
            return mapping[tensor_name]
    return None


def _mapping_values(mapping_store: Dict[Tuple[int, str], Dict[str, str]], op_idx: int, phase: str) -> List[str]:
    """Return canonical names for one `(op_idx, phase)` mapping in stable order."""
    return _sorted_mapping_values(mapping_store.get((op_idx, phase), {}))


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

        buffers: List[PeerBufferInfo] = []
        seen_canonical: Set[str] = set()

        for name, peers in peer_map.items():
            canonical = _resolve_tensor_canonical_name(name, tensor_registry)
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
    dim_perm: Tuple[int, ...] = (),
) -> Tuple[int, ...]:
    """Compute TMA tile shape for a tensor, permuted for TMA compatibility.

    Ops can override via get_tma_tile_shape() for dims that need custom
    sub-tiling (e.g., K in GEMM). Otherwise, uses the op's tile_sizes
    and static_dims to build the shape from _TMA_TENSOR_DIMS.

    The result is permuted to match the gmem tensor's dimension order after
    stride-sorting (see _prepare_tma_tensors). For contiguous tensors this
    is equivalent to simple reversal.

    Args:
        op_cls: The Op class with TMA declarations.
        tensor_name: Name of the tensor (e.g., "x", "y").
        op: ScheduledOp with tile_sizes and static_dims.
        dim_perm: Stride-sorted dimension permutation from the tensor.

    Returns:
        Permuted TMA tile shape tuple.
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

    # Permute tile shape to match the gmem tensor's stride-sorted dim order.
    # TMA requires CuTe mode 0 to be contiguous (stride 1). Sorting dims by
    # stride (ascending) ensures this and guarantees monotonically increasing
    # strides for the TMA descriptor. For contiguous row-major tensors, this
    # is equivalent to simple reversal.
    if dim_perm and len(dim_perm) == len(tile_shape):
        return tuple(tile_shape[p] for p in dim_perm)
    return tuple(reversed(tile_shape))


def _resolve_tma_dtype(op: ScheduledOp, tensor_name: str) -> Any:
    """Resolve the CUTLASS dtype for a TMA-managed tensor."""
    meta = op.tensor_metas.get(tensor_name)
    return meta.dtype if meta else None


def _resolve_tma_smem_layout_src(
    op_cls,
    tensor_name: str,
    tma_tile_shape: Tuple[int, ...],
    op: ScheduledOp,
) -> Optional[str]:
    """Resolve an optional custom smem layout source for a TMA descriptor."""
    if not hasattr(op_cls, "get_tma_smem_layout_src"):
        return None
    return op_cls.get_tma_smem_layout_src(
        tensor_name,
        tma_tile_shape,
        op.tile_sizes,
        op.static_dims,
    )


def _set_tma_mapping(mapping: Dict[str, str], tensor_name: str, canonical_atom: str, canonical_gmem: str) -> None:
    """Store op-local TMA parameter names for a tensor."""
    mapping[f"{tensor_name}_tma"] = canonical_atom
    mapping[f"{tensor_name}_tma_gmem"] = canonical_gmem


def _set_peer_tma_mapping(
    mapping: Dict[str, str],
    tensor_name: str,
    peer_idx: int,
    canonical_atom: str,
    canonical_gmem: str,
) -> None:
    """Store op-local peer TMA parameter names for a tensor/peer pair."""
    mapping[f"{tensor_name}_p{peer_idx}_tma"] = canonical_atom
    mapping[f"{tensor_name}_p{peer_idx}_tma_gmem"] = canonical_gmem


def _append_tma_descriptor(
    descriptors: List["TMADescriptorInfo"],
    op_mappings: Dict[Tuple[int, str], Dict[str, str]],
    *,
    op_idx: int,
    phase: str,
    tensor_name: str,
    tensor_canonical: str,
    direction: str,
    tile_shape: Tuple[int, ...],
    dtype: Any,
    canonical_atom: str,
    canonical_gmem: str,
    tensor_shape: Tuple[int, ...],
    original_tensor: Any,
    smem_layout_src: Optional[str],
    dim_perm: Tuple[int, ...],
) -> None:
    """Append one local TMA descriptor and update the phase mapping."""
    descriptors.append(
        TMADescriptorInfo(
            canonical_atom=canonical_atom,
            canonical_gmem=canonical_gmem,
            tensor_canonical=tensor_canonical,
            direction=direction,
            tile_shape=tile_shape,
            smem_layout_shape=tile_shape,
            dtype=dtype,
            tensor_shape=tensor_shape,
            original_tensor=original_tensor,
            smem_layout_src=smem_layout_src,
            dim_perm=dim_perm,
        )
    )
    _set_tma_mapping(
        op_mappings[(op_idx, phase)],
        tensor_name,
        canonical_atom,
        canonical_gmem,
    )


def _local_tma_desc_key(
    *,
    tensor_canonical: str,
    direction: str,
    tile_shape: Tuple[int, ...],
    dtype: Any,
    tensor_shape: Tuple[int, ...],
    original_tensor: Any,
    smem_layout_src: Optional[str],
    dim_perm: Tuple[int, ...],
) -> Tuple[Any, ...]:
    """Return a stable dedup key for one local TMA descriptor."""
    return (
        tensor_canonical,
        direction,
        tuple(tile_shape),
        getattr(dtype, "__name__", str(dtype)),
        tuple(tensor_shape),
        original_tensor.data_ptr() if original_tensor is not None else None,
        smem_layout_src,
        tuple(dim_perm),
    )


def _local_tma_atom_key(
    *,
    tensor_canonical: str,
    direction: str,
    tile_shape: Tuple[int, ...],
    dtype: Any,
    tensor_shape: Tuple[int, ...],
    smem_layout_src: Optional[str],
    dim_perm: Tuple[int, ...],
) -> Tuple[Any, ...]:
    """Return a structural dedup key for a reusable local TMA atom."""
    return (
        tensor_canonical,
        direction,
        tuple(tile_shape),
        getattr(dtype, "__name__", str(dtype)),
        tuple(tensor_shape),
        smem_layout_src,
        tuple(dim_perm),
    )


def _append_peer_tma_descriptor(
    descriptors: List["PeerTMADescriptorInfo"],
    op_mappings: Dict[Tuple[int, str], Dict[str, str]],
    *,
    op_idx: int,
    tensor_name: str,
    tensor_canonical: str,
    peer_idx: int,
    direction: str,
    tile_shape: Tuple[int, ...],
    dtype: Any,
    canonical_atom: str,
    canonical_gmem: str,
    smem_layout_src: Optional[str],
) -> None:
    """Append one peer TMA descriptor and update the communicate mapping."""
    descriptors.append(
        PeerTMADescriptorInfo(
            canonical_atom=canonical_atom,
            canonical_gmem=canonical_gmem,
            tensor_canonical=tensor_canonical,
            peer_idx=peer_idx,
            tile_shape=tile_shape,
            smem_layout_shape=tile_shape,
            dtype=dtype,
            smem_layout_src=smem_layout_src,
            direction=direction,
        )
    )
    _set_peer_tma_mapping(
        op_mappings[(op_idx, "communicate")],
        tensor_name,
        peer_idx,
        canonical_atom,
        canonical_gmem,
    )


def _peer_tma_desc_key(
    *,
    tensor_canonical: str,
    peer_idx: int,
    direction: str,
    tile_shape: Tuple[int, ...],
    dtype: Any,
    smem_layout_src: Optional[str],
) -> Tuple[Any, ...]:
    """Return a stable dedup key for one peer TMA descriptor."""
    return (
        tensor_canonical,
        peer_idx,
        direction,
        tuple(tile_shape),
        getattr(dtype, "__name__", str(dtype)),
        smem_layout_src,
    )


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
    tensor_shape: Tuple[int, ...] = ()  # original tensor shape from the op
    original_tensor: Any = None  # actual tensor from op.tensor_refs (for strided TMA)
    smem_layout_src: Optional[str] = None
    dim_perm: Tuple[int, ...] = ()  # stride-sorted dimension permutation for TMA


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
        atom_counter = 0
        gmem_counter = 0
        desc_name_cache: Dict[Tuple[Any, ...], Tuple[str, str]] = {}
        atom_name_cache: Dict[Tuple[Any, ...], str] = {}
        gmem_name_cache: Dict[Tuple[Any, ...], str] = {}

        for i, op in enumerate(ops):
            op_cls = op.op_cls
            for phase in ("load", "compute", "store"):
                op_mappings[(i, phase)] = {}

            for phase, tma_names, direction in _iter_tma_phase_specs(op_cls):
                for tensor_name in tma_names:
                    tensor_canonical = tensor_registry.op_mappings[i].get(tensor_name)
                    if tensor_canonical is None:
                        continue

                    original_ref = op.tensor_refs[tensor_name]
                    dim_perm = _tensor_dim_permutation(original_ref)
                    tma_tile_shape = _compute_tma_tile_shape(
                        op_cls, tensor_name, op, dim_perm=dim_perm)
                    dtype = _resolve_tma_dtype(op, tensor_name)
                    smem_layout_src = _resolve_tma_smem_layout_src(
                        op_cls, tensor_name, tma_tile_shape, op
                    )
                    desc_key = _local_tma_desc_key(
                        tensor_canonical=tensor_canonical,
                        direction=direction,
                        tile_shape=tma_tile_shape,
                        dtype=dtype,
                        tensor_shape=tuple(original_ref.shape),
                        original_tensor=original_ref,
                        smem_layout_src=smem_layout_src,
                        dim_perm=dim_perm,
                    )
                    if desc_key in desc_name_cache:
                        canonical_atom, canonical_gmem = desc_name_cache[desc_key]
                        _set_tma_mapping(
                            op_mappings[(i, phase)],
                            tensor_name,
                            canonical_atom,
                            canonical_gmem,
                        )
                    else:
                        share_atom = (
                            op_cls.__name__ in {"GemmOp", "GemmSm100Op"}
                            and op.static_dims.get("activation", 0) == 0
                            and op.static_dims.get("has_a_scale", 0) == 0
                        )
                        canonical_atom = None
                        if share_atom:
                            atom_key = _local_tma_atom_key(
                                tensor_canonical=tensor_canonical,
                                direction=direction,
                                tile_shape=tma_tile_shape,
                                dtype=dtype,
                                tensor_shape=tuple(original_ref.shape),
                                smem_layout_src=smem_layout_src,
                                dim_perm=dim_perm,
                            )
                            canonical_atom = atom_name_cache.get(atom_key)
                        if canonical_atom is None:
                            canonical_atom = f"tma{atom_counter}_atom"
                            if share_atom:
                                atom_name_cache[atom_key] = canonical_atom
                            atom_counter += 1
                        gmem_key = None
                        canonical_gmem = None
                        if share_atom:
                            # Reusing the atom is safe because it only depends on
                            # structural transport state. Reusing the gmem-side
                            # transport object across different tensors is not:
                            # it carries the concrete source/destination tensor
                            # binding. Keep one gmem handle per canonical tensor.
                            gmem_key = (
                                tensor_canonical,
                                direction,
                                tuple(tma_tile_shape),
                                getattr(dtype, "__name__", str(dtype)),
                                tuple(original_ref.shape),
                                smem_layout_src,
                                tuple(dim_perm),
                            )
                            canonical_gmem = gmem_name_cache.get(gmem_key)
                        if canonical_gmem is None:
                            canonical_gmem = f"tma{gmem_counter}_gmem"
                            if share_atom:
                                gmem_name_cache[gmem_key] = canonical_gmem
                            gmem_counter += 1
                        desc_name_cache[desc_key] = (canonical_atom, canonical_gmem)
                        _append_tma_descriptor(
                            descriptors,
                            op_mappings,
                            op_idx=i,
                            phase=phase,
                            tensor_name=tensor_name,
                            tensor_canonical=tensor_canonical,
                            direction=direction,
                            tile_shape=tma_tile_shape,
                            dtype=dtype,
                            canonical_atom=canonical_atom,
                            canonical_gmem=canonical_gmem,
                            tensor_shape=tuple(original_ref.shape),
                            original_tensor=original_ref,
                            smem_layout_src=smem_layout_src,
                            dim_perm=dim_perm,
                        )

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
        return _canonical_name_list(self.descriptors)

    def get_op_tma_args(self, op_idx: int, phase: str) -> List[str]:
        """Get canonical TMA param names for an op's phase function.

        Args:
            op_idx: Index of the op
            phase: "load" or "store"

        Returns:
            List of canonical names in order: [atom, gmem, atom, gmem, ...]
        """
        return _mapping_values(self.op_mappings, op_idx, phase)


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
    direction: str = "s2g"  # "s2g" for regular copy, "s2g_reduce" for atomic add


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
        """Build PeerTMARegistry from ops that declare peer_stores or peer_reduce_stores.

        For each op's peer_stores tensor, creates TMA S2G descriptors (regular copy).
        For each op's peer_reduce_stores tensor, creates TMA S2G reduce descriptors
        (atomic add) targeting the peer's buffer.
        """
        descriptors: List[PeerTMADescriptorInfo] = []
        op_mappings: Dict[Tuple[int, str], Dict[str, str]] = {}
        counter = 0
        desc_name_cache: Dict[Tuple[Any, ...], Tuple[str, str]] = {}
        num_peers = peer_buffer_registry.num_peers

        for i, op in enumerate(ops):
            op_cls = op.op_cls
            peer_stores = getattr(op_cls, "_PEER_STORES", set())
            peer_reduce_stores = getattr(op_cls, "_PEER_REDUCE_STORES", set())
            op_mappings[(i, "communicate")] = {}

            if not peer_stores and not peer_reduce_stores:
                continue

            for tensor_name, direction in _iter_peer_tma_specs(op_cls):
                tensor_canonical = tensor_registry.op_mappings[i].get(tensor_name)
                if tensor_canonical is None:
                    continue

                tma_tile_shape = _compute_tma_tile_shape(op_cls, tensor_name, op)
                dtype = _resolve_tma_dtype(op, tensor_name)
                smem_layout_src = _resolve_tma_smem_layout_src(
                    op_cls, tensor_name, tma_tile_shape, op
                )

                for peer_idx in range(num_peers):
                    desc_key = _peer_tma_desc_key(
                        tensor_canonical=tensor_canonical,
                        peer_idx=peer_idx,
                        direction=direction,
                        tile_shape=tma_tile_shape,
                        dtype=dtype,
                        smem_layout_src=smem_layout_src,
                    )
                    if desc_key in desc_name_cache:
                        canonical_atom, canonical_gmem = desc_name_cache[desc_key]
                        _set_peer_tma_mapping(
                            op_mappings[(i, "communicate")],
                            tensor_name,
                            peer_idx,
                            canonical_atom,
                            canonical_gmem,
                        )
                    else:
                        canonical_atom = f"ptma{counter}_p{peer_idx}_atom"
                        canonical_gmem = f"ptma{counter}_p{peer_idx}_gmem"
                        desc_name_cache[desc_key] = (canonical_atom, canonical_gmem)
                        _append_peer_tma_descriptor(
                            descriptors,
                            op_mappings,
                            op_idx=i,
                            tensor_name=tensor_name,
                            tensor_canonical=tensor_canonical,
                            peer_idx=peer_idx,
                            direction=direction,
                            tile_shape=tma_tile_shape,
                            dtype=dtype,
                            canonical_atom=canonical_atom,
                            canonical_gmem=canonical_gmem,
                            smem_layout_src=smem_layout_src,
                        )

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
        return _canonical_name_list(self.descriptors)

    def get_op_peer_tma_args(self, op_idx: int, phase: str = "communicate") -> List[str]:
        """Get canonical peer TMA param names for an op's communicate phase."""
        return _mapping_values(self.op_mappings, op_idx, phase)


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
