# Copyright (c) 2025, Machete Authors
"""Backend IR for megakernel handler-based dispatch."""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

PHASE_NAMES = ("load", "compute", "store", "communicate")


@dataclass(frozen=True)
class OpCompileSpec:
    """Compile-relevant signature for one scheduled op."""

    op_index: int
    compile_key: Tuple[Any, ...]
    tensor_args: Dict[str, Tuple[str, ...]]
    local_tensor_names: Dict[str, Tuple[str, ...]]
    tma_args: Dict[str, Tuple[str, ...]]
    local_tma_args: Dict[str, Tuple[str, ...]]
    weight: int


@dataclass(frozen=True)
class HandlerSpec:
    """One unique handler shared by all ops with the same compile key."""

    handler_idx: int
    compile_key: Tuple[Any, ...]
    local_tensor_names: Dict[str, Tuple[str, ...]]
    local_tma_args: Dict[str, Tuple[str, ...]]
    weight: int


@dataclass(frozen=True)
class ProtocolRoleSpec:
    """One page or semaphore role in an instruction-owned protocol."""

    name: str
    offset: int
    count: int
    participants: int = 1


@dataclass(frozen=True)
class RegionCompileSpec:
    """Compile-relevant signature for one persistent region."""

    region_idx: int
    name: str
    start_op: int
    end_op: int
    lowering: str
    page_roles: Tuple[ProtocolRoleSpec, ...]
    semaphore_roles: Tuple[ProtocolRoleSpec, ...]
    page_count: int
    page_bytes: int
    semaphore_count: int
    scratch_bytes: int
    resource_bytes: int
    input_stages: int
    output_stages: int
    stage_pages: int
    range_axis: int
    range_end_axis: int
    range_block_size: int
    coalesce_ranges: bool

    @property
    def op_count(self) -> int:
        return self.end_op - self.start_op


@dataclass(frozen=True)
class BackendIR:
    """Static backend IR used by handler-based dispatch."""

    op_specs: Tuple[OpCompileSpec, ...]
    region_specs: Tuple[RegionCompileSpec, ...]
    handler_specs: Tuple[HandlerSpec, ...]
    op_handler_indices: Tuple[int, ...]
    op_phase_local_indices: Dict[str, Tuple[int, ...]]
    op_phase_transport_indices: Dict[str, Tuple[int, ...]]
    phase_transport_records: Dict[str, Tuple[Tuple[str, ...], ...]]
    phase_local_transport_positions: Dict[str, Tuple[Tuple[Tuple[int, ...], ...], ...]]


__all__ = [
    "PHASE_NAMES",
    "OpCompileSpec",
    "HandlerSpec",
    "ProtocolRoleSpec",
    "RegionCompileSpec",
    "BackendIR",
]
