# Copyright (c) 2025, Machete Authors
"""CSV export helpers for megakernel dependency graphs."""

import csv
from typing import Dict, List, Tuple

from .ops import MAX_TILE_DIMS, ScheduledOp


def _tile_str(tile: Tuple[int, ...]) -> str:
    return ":".join(str(v) for v in tile)


def _linear_strides(tile_counts: Tuple[int, ...]) -> Tuple[int, ...]:
    ndims = len(tile_counts)
    strides = [0] * MAX_TILE_DIMS
    stride = 1
    for i in range(ndims - 1, -1, -1):
        strides[i] = stride
        stride *= tile_counts[i]
    return tuple(strides)


def _linear_tile(tile: Tuple[int, ...], tile_counts: Tuple[int, ...]) -> int:
    linear = 0
    strides = _linear_strides(tile_counts)
    padded = tuple(tile) + (0,) * (MAX_TILE_DIMS - len(tile))
    for axis, stride in enumerate(strides):
        linear += padded[axis] * stride
    return linear


def export_dependency_graph_csv(builder, op_csv: str, tile_csv: str) -> None:
    """Export op-level and tile-level dependency graphs as CSV files.

    The tile graph is reconstructed from the same host-side barrier formulas
    used to build runtime wait/signal metadata. Each tile-dependency row joins
    one consumer wait barrier to every producer tile that signals that barrier.
    """
    plan = builder.dependency_plan()
    formulas = plan.formulas
    edges = plan.edges
    controller_waits = plan.controller_wait_formulas
    compute_waits = plan.compute_wait_formulas

    edge_rows = []
    edge_by_pair: Dict[Tuple[int, int], List[object]] = {}
    for edge_id, edge in enumerate(edges):
        producer = builder._op_records[edge.producer_idx].op
        consumer = builder._op_records[edge.consumer_idx].op
        edge_by_pair.setdefault((edge.producer_idx, edge.consumer_idx), []).append(edge)
        edge_rows.append(
            {
                "edge_id": edge_id,
                "producer_op_idx": edge.producer_idx,
                "producer_op": producer.op_cls.__name__,
                "producer_tiles": producer.total_tiles,
                "producer_tile_counts": _tile_str(producer.tile_counts),
                "producer_buffer": edge.producer_buffer,
                "consumer_op_idx": edge.consumer_idx,
                "consumer_op": consumer.op_cls.__name__,
                "consumer_tiles": consumer.total_tiles,
                "consumer_tile_counts": _tile_str(consumer.tile_counts),
                "consumer_buffer": edge.consumer_buffer,
                "kind": edge.kind,
            }
        )

    with open(op_csv, "w", newline="") as f:
        fieldnames = [
            "edge_id",
            "producer_op_idx",
            "producer_op",
            "producer_tiles",
            "producer_tile_counts",
            "producer_buffer",
            "consumer_op_idx",
            "consumer_op",
            "consumer_tiles",
            "consumer_tile_counts",
            "consumer_buffer",
            "kind",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(edge_rows)

    barrier_signals: Dict[int, List[Tuple[int, ScheduledOp, Tuple[int, ...], int]]] = {}
    for rec in builder._op_records:
        signal_formulas = formulas.get(rec.op_idx, ([], []))[1]
        for tile in rec.tiles:
            for formula in signal_formulas:
                if formula.has_guard and not formula.is_guarded(tile):
                    continue
                barrier_idx = formula.compute_index(tile)
                barrier_signals.setdefault(barrier_idx, []).append(
                    (rec.op_idx, rec.op, tile, _linear_tile(tile, rec.op.tile_counts))
                )

    tile_fieldnames = [
        "dep_id",
        "wait_phase",
        "barrier_idx",
        "expected",
        "producer_op_idx",
        "producer_op",
        "producer_tile_linear",
        "producer_tile",
        "producer_buffer",
        "consumer_op_idx",
        "consumer_op",
        "consumer_tile_linear",
        "consumer_tile",
        "consumer_buffer",
        "kind",
    ]
    dep_id = 0
    with open(tile_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=tile_fieldnames)
        writer.writeheader()

        for wait_phase, wait_by_op in (
            ("controller", controller_waits),
            ("compute", compute_waits),
        ):
            for rec in builder._op_records:
                wait_formulas = wait_by_op.get(rec.op_idx, [])
                if not wait_formulas:
                    continue
                for consumer_tile in rec.tiles:
                    consumer_linear = _linear_tile(consumer_tile, rec.op.tile_counts)
                    for formula in wait_formulas:
                        if formula.has_guard and not formula.is_guarded(consumer_tile):
                            continue
                        barrier_idx = formula.compute_index(consumer_tile)
                        for prod_idx, prod_op, prod_tile, prod_linear in barrier_signals.get(barrier_idx, []):
                            pair_edges = edge_by_pair.get((prod_idx, rec.op_idx), [])
                            producer_buffer = "|".join(sorted({edge.producer_buffer for edge in pair_edges}))
                            consumer_buffer = "|".join(sorted({edge.consumer_buffer for edge in pair_edges}))
                            kind = "|".join(sorted({edge.kind for edge in pair_edges}))
                            writer.writerow(
                                {
                                    "dep_id": dep_id,
                                    "wait_phase": wait_phase,
                                    "barrier_idx": barrier_idx,
                                    "expected": formula.expected,
                                    "producer_op_idx": prod_idx,
                                    "producer_op": prod_op.op_cls.__name__,
                                    "producer_tile_linear": prod_linear,
                                    "producer_tile": _tile_str(prod_tile),
                                    "producer_buffer": producer_buffer,
                                    "consumer_op_idx": rec.op_idx,
                                    "consumer_op": rec.op.op_cls.__name__,
                                    "consumer_tile_linear": consumer_linear,
                                    "consumer_tile": _tile_str(consumer_tile),
                                    "consumer_buffer": consumer_buffer,
                                    "kind": kind,
                                }
                            )
                            dep_id += 1
