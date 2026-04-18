# Copyright (c) 2025, Machete Authors
"""Backend IR for megakernel handler-based dispatch."""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


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
class BackendIR:
    """Static backend IR used by handler-based dispatch."""

    op_specs: Tuple[OpCompileSpec, ...]
    handler_specs: Tuple[HandlerSpec, ...]
    op_handler_indices: Tuple[int, ...]
    op_handler_local_indices: Tuple[int, ...]


__all__ = [
    "OpCompileSpec",
    "HandlerSpec",
    "BackendIR",
]
