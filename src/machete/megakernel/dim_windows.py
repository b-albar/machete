# Copyright (c) 2026, Machete Authors
"""Dimension-window helpers for scheduled ops.

Windows are expressed in element coordinates by semantic dimension name, while
the runtime dispatches tile coordinates.  This module keeps the normalization
and tile-domain math independent of concrete op classes.
"""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Tuple


DimWindows = Dict[str, Tuple[int, int]]


def parse_dim_windows(dim_windows) -> DimWindows:
    """Normalize dimension windows to ``dim -> (origin, extent)`` in elements."""
    if not dim_windows:
        return {}

    parsed: DimWindows = {}
    for dim_name, window in dim_windows.items():
        if isinstance(window, Mapping):
            origin = window.get("origin", window.get("start", 0))
            extent = window.get("extent", window.get("size"))
        else:
            if len(window) != 2:
                raise ValueError(
                    f"dim_windows[{dim_name!r}] must be (origin, extent), got {window!r}"
                )
            origin, extent = window
        if extent is None:
            raise ValueError(f"dim_windows[{dim_name!r}] is missing an extent")
        origin = int(origin)
        extent = int(extent)
        if origin < 0:
            raise ValueError(f"dim_windows[{dim_name!r}] origin must be >= 0, got {origin}")
        if extent < 1:
            raise ValueError(f"dim_windows[{dim_name!r}] extent must be >= 1, got {extent}")
        parsed[dim_name] = (origin, extent)
    return parsed


def resolve_schedule_domain(tile_dim_names, tile_sizes, dim_values, dim_windows, resolve_tile_sizes):
    """Resolve tile sizes/counts plus tile origins for a scheduled window."""
    resolved_tile_sizes, tile_counts = resolve_tile_sizes(
        tile_dim_names,
        tile_sizes,
        {dim: dim_windows.get(dim, (0, dim_values[dim]))[1] for dim in dim_values},
    )

    tile_origins = {}
    tile_dim_set = set(tile_dim_names)
    for dim_name, (origin, extent) in dim_windows.items():
        if dim_name not in dim_values:
            raise ValueError(f"dim_windows references unknown dim {dim_name!r}")
        full_extent = dim_values[dim_name]
        if origin + extent > full_extent:
            raise ValueError(
                f"dim_windows[{dim_name!r}]={origin, extent} exceeds full extent {full_extent}"
            )
        if dim_name not in tile_dim_set:
            raise ValueError(
                f"dim_windows[{dim_name!r}] targets a non-tiled dimension. "
                "Only tiled dimensions can be windowed generically."
            )
        tile_size = resolved_tile_sizes[dim_name]
        if origin % tile_size != 0:
            raise ValueError(
                f"dim_windows[{dim_name!r}] origin {origin} must be aligned to "
                f"tile_size_{dim_name}={tile_size}"
            )
        tile_origins[dim_name] = origin // tile_size

    return resolved_tile_sizes, tile_counts, tile_origins


def tensor_window_for_dims(meta, dim_windows: DimWindows) -> Tuple[int, Tuple[int, ...]]:
    """Return storage-offset delta and logical shape for a dim-windowed tensor."""
    storage_delta = 0
    shape = list(meta.shape)
    for axis, dim_name in enumerate(meta.declared_dims):
        window = dim_windows.get(dim_name)
        if window is None:
            continue
        origin, extent = window
        storage_delta += origin * meta.strides[axis]
        shape[axis] = extent
    return storage_delta, tuple(shape)


def window_tensor_metas(
    tensor_metas,
    dim_windows: DimWindows,
    *,
    element_size_for_name: Callable[[str], int],
    meta_factory: Callable,
):
    """Return tensor metadata restricted to the logical dimension window.

    Kernels still receive full tensor refs. These adjusted metas are only for
    host-side dependency matching, so independent windows do not serialize on
    each other.
    """
    if not dim_windows:
        return tensor_metas

    windowed_metas = {}
    for name, meta in tensor_metas.items():
        storage_delta, shape = tensor_window_for_dims(meta, dim_windows)
        if storage_delta == 0 and shape == meta.shape:
            windowed_metas[name] = meta
            continue
        windowed_metas[name] = meta_factory(
            meta=meta,
            shape=shape,
            data_ptr=meta.data_ptr + storage_delta * element_size_for_name(name),
            storage_offset=meta.storage_offset + storage_delta,
        )
    return windowed_metas


def iter_dim_windows(dim_name: str, extent: int, window_extent: int):
    """Yield contiguous element windows for one dimension."""
    if extent < 0:
        raise ValueError(f"extent must be >= 0, got {extent}")
    if window_extent < 1:
        raise ValueError(f"window_extent must be >= 1, got {window_extent}")
    for origin in range(0, extent, window_extent):
        yield {dim_name: (origin, min(window_extent, extent - origin))}


__all__ = [
    "DimWindows",
    "parse_dim_windows",
    "resolve_schedule_domain",
    "tensor_window_for_dims",
    "window_tensor_metas",
    "iter_dim_windows",
]
