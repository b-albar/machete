# Copyright (c) 2025, Machete Authors
"""Tracing support for the megakernel.

Provides cutedsl-trace integration for visualizing per-SM tile timelines
(load, compute, store, and wait events).

Four lanes per CTA:
    lane 0 = Load warp (load)
    lane 1 = MMA warp 0 (compute, data_wait)
    lane 2 = Store warp (store, compute_wait)
    lane 3 = Controller warp (dep_wait, ring_full_wait)
"""

import math
from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class TracingState:
    """Holds all tracing state for a megakernel compilation."""

    builder: object = None       # DynamicTraceBuilder
    block_type: object = None    # BlockType
    trace_types: dict = field(default_factory=dict)
    load_fmts: List[int] = field(default_factory=list)
    compute_fmts: List[int] = field(default_factory=list)
    store_fmts: List[int] = field(default_factory=list)
    dep_wait_fmt: int = 0
    data_wait_fmt: int = 0
    compute_wait_fmt: int = 0
    ring_full_wait_fmt: int = 0
    load_fmts_tensor: object = None
    compute_fmts_tensor: object = None
    store_fmts_tensor: object = None


def setup_tracing(ops, num_sms, total_tiles, device="cuda") -> TracingState:
    """Set up cutedsl-trace builder and trace types.

    Creates per-(op_class, phase) TraceTypes so load/compute/store events
    appear as distinct colored spans in the trace viewer.
    """
    from cutedsl_trace import (
        TraceType,
        BlockType,
        TrackType,
        DynamicTraceBuilder,
    )
    from cutedsl_trace.types import LaneType

    state = TracingState()

    # Per-(op_class, phase) TraceTypes
    seen_classes: dict = {}
    for op in ops:
        cls_name = op.op_cls.__name__
        if cls_name not in seen_classes:
            seen_classes[cls_name] = {}
            for phase in ("load", "compute", "store"):
                key = f"{cls_name}_{phase}"
                tt = TraceType(
                    name=key,
                    label_string=f"{cls_name} {phase}",
                    tooltip_string=f"{cls_name} {phase} op={{0}}",
                    param_count=1,
                    lane_type=LaneType.DYNAMIC,
                )
                seen_classes[cls_name][phase] = tt
                state.trace_types[key] = tt
        state.load_fmts.append(seen_classes[cls_name]["load"].id)
        state.compute_fmts.append(seen_classes[cls_name]["compute"].id)
        state.store_fmts.append(seen_classes[cls_name]["store"].id)

    # Wait event types
    for wait_name in ("dep_wait", "data_wait", "compute_wait", "ring_full_wait"):
        tt = TraceType(
            name=wait_name,
            label_string=wait_name.replace("_", " "),
            tooltip_string=f"{wait_name} op={{0}}",
            param_count=1,
            lane_type=LaneType.DYNAMIC,
        )
        state.trace_types[wait_name] = tt
    state.dep_wait_fmt = state.trace_types["dep_wait"].id
    state.data_wait_fmt = state.trace_types["data_wait"].id
    state.compute_wait_fmt = state.trace_types["compute_wait"].id
    state.ring_full_wait_fmt = state.trace_types["ring_full_wait"].id

    state.block_type = BlockType(
        name="CTA",
        label_string="CTA {blockLinear}",
        tooltip_string="CTA {blockLinear} on SM {smId}",
    )

    load_track = TrackType(
        name="Load",
        label_string="SM {lane} Load",
        tooltip_string="Load warp on SM {lane}",
    )
    mma_track = TrackType(
        name="MMA",
        label_string="SM {lane} MMA",
        tooltip_string="MMA warp 0 on SM {lane}",
    )
    store_track = TrackType(
        name="Store",
        label_string="SM {lane} Store",
        tooltip_string="Store warp on SM {lane}",
    )
    controller_track = TrackType(
        name="Controller",
        label_string="SM {lane} Controller",
        tooltip_string="Controller warp on SM {lane}",
    )

    # 4 lanes: load, MMA, store, controller; enough events per tile.
    tiles_per_sm = math.ceil(total_tiles / num_sms)
    max_events = tiles_per_sm * 8 + 256
    state.builder = DynamicTraceBuilder(
        num_lanes=4,
        max_events_per_lane=max_events,
        grid_dims=(num_sms, 1, 1),
    )
    state.builder.set_track_type(load_track, lane=0)
    state.builder.set_track_type(mma_track, lane=1)
    state.builder.set_track_type(store_track, lane=2)
    state.builder.set_track_type(controller_track, lane=3)

    state.load_fmts_tensor = torch.tensor(
        state.load_fmts, dtype=torch.int32, device=device
    )
    state.compute_fmts_tensor = torch.tensor(
        state.compute_fmts, dtype=torch.int32, device=device
    )
    state.store_fmts_tensor = torch.tensor(
        state.store_fmts, dtype=torch.int32, device=device
    )

    return state


def _make_writer(state: TracingState):
    """Create a TraceWriter from tracing state."""
    if state is None or state.builder is None:
        raise RuntimeError("Tracing not enabled. Set MegakernelConfig(tracing=True).")
    from cutedsl_trace import TraceWriter

    state.builder.copy_to_host()
    writer = TraceWriter("megakernel")
    writer.set_block_type(state.block_type)
    writer.add_tensor(state.builder)
    for tt in state.trace_types.values():
        writer.register_trace_type(tt)
    return writer


def write_trace(state: TracingState, filename: str) -> None:
    """Write trace to .nanotrace file. Only valid after run() with tracing=True."""
    _make_writer(state).write(filename)


def write_trace_perfetto(state: TracingState, filename: str) -> None:
    """Write trace as Perfetto JSON. Only valid after run() with tracing=True."""
    _make_writer(state).write_perfetto(filename)


def ensure_device_trace_buffer(state: TracingState) -> None:
    """Ensure the trace builder owns a CUDA buffer before kernel launch."""
    if state is None or state.builder is None:
        return
    buffer = getattr(state.builder, "_buffer", None)
    if hasattr(buffer, "data_ptr"):
        return
    grid_x, grid_y, grid_z = state.builder.grid_dims
    total_bytes = (
        state.builder.row_stride_bytes
        * state.builder.num_lanes
        * grid_x
        * grid_y
        * grid_z
    )
    state.builder._allocate_buffer(total_bytes)


def resolve_tracing_blocks(source: str, tracing: bool) -> str:
    """Resolve ``if tracing:`` blocks at the source level.

    CuTe DSL has no constexpr-if: its AST rewriter transforms every ``if``
    into a dynamic SCF op that traces both branches.  We therefore resolve
    the tracing guard *before* the source reaches CuTe DSL:

    * ``tracing=False`` → the ``if tracing:`` line and its entire indented
      body are stripped (zero trace code in the compiled kernel).
    * ``tracing=True``  → the ``if tracing:`` guard is removed and the body
      is dedented by one level (trace code always runs).
    """
    lines = source.split("\n")
    result: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        if stripped == "if tracing:":
            if_indent = len(line) - len(stripped)
            # Collect the indented block body
            block: list[str] = []
            i += 1
            while i < len(lines):
                nxt = lines[i]
                nxt_stripped = nxt.lstrip()
                # Blank lines belong to the block; non-blank lines at
                # the same or lesser indent terminate it.
                if nxt_stripped and (len(nxt) - len(nxt_stripped)) <= if_indent:
                    break
                block.append(nxt)
                i += 1
            if tracing:
                # Inline: dedent body by one level (first non-blank
                # line determines the block indent).
                body_indent = None
                for bl in block:
                    bs = bl.lstrip()
                    if bs:
                        body_indent = len(bl) - len(bs)
                        break
                if body_indent is not None:
                    dedent = body_indent - if_indent
                    for bl in block:
                        if bl.strip():
                            result.append(bl[dedent:])
                        else:
                            result.append(bl)
            # tracing=False → drop the entire block (nothing added)
        else:
            result.append(line)
            i += 1
    return "\n".join(result)


def get_trace_exec_globals(state: TracingState) -> dict:
    """Build the exec_globals dict entries for trace instrumentation.

    When tracing is enabled, returns the device-side functions and format_ids.
    When state is None (tracing disabled), returns an empty dict (the trace
    code has been stripped from the source by resolve_tracing_blocks).
    """
    if state is not None:
        from cutlass import Int64
        from cutedsl_trace.device import (
            start as trace_start,
            begin_lane_dynamic_raw,
            end_event_dynamic_raw_1,
            finish_lane_dynamic_raw,
        )
        return {
            "trace_start": trace_start,
            "begin_lane_dynamic_raw": begin_lane_dynamic_raw,
            "end_event_dynamic_raw_1": end_event_dynamic_raw_1,
            "finish_lane_dynamic_raw": finish_lane_dynamic_raw,
            "trace_row_stride": state.builder.row_stride_bytes,
            "trace_load_fmt_ptr": Int64(state.load_fmts_tensor.data_ptr()),
            "trace_compute_fmt_ptr": Int64(state.compute_fmts_tensor.data_ptr()),
            "trace_store_fmt_ptr": Int64(state.store_fmts_tensor.data_ptr()),
            "trace_dep_wait_fmt": state.dep_wait_fmt,
            "trace_data_wait_fmt": state.data_wait_fmt,
            "trace_compute_wait_fmt": state.compute_wait_fmt,
            "trace_ring_full_wait_fmt": state.ring_full_wait_fmt,
        }
    else:
        # Trace code is stripped from the source by resolve_tracing_blocks,
        # so no globals are needed.
        return {}
