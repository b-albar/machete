#!/usr/bin/env python
"""Measure framework overhead and selector-table scaling across repeated layers.

This benchmark intentionally focuses on host/runtime framework work:
- Megakernel construction
- backend IR / dispatch metadata preparation
- selector-table growth

It does not compile or launch the GPU kernel. The goal is to catch framework
scaling regressions early while runtime-backend transport selection is being
refactored toward layer-count-independent behavior.
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import time

import torch

if importlib.util.find_spec("cutlass") is None:
    raise SystemExit("Requires CUTLASS")

from machete.megakernel.megakernel import Megakernel, MegakernelConfig
from tests.megakernel.support_tma import (
    SYNTHETIC_TMA_N,
    SYNTHETIC_TMA_TILE_M,
    SyntheticTMAAddOneOp,
)


def make_ops(num_layers: int, *, device: str):
    ops = []
    for _ in range(num_layers):
        x = torch.zeros(SYNTHETIC_TMA_TILE_M, SYNTHETIC_TMA_N, dtype=torch.float16, device=device)
        y = torch.zeros(SYNTHETIC_TMA_TILE_M, SYNTHETIC_TMA_N, dtype=torch.float16, device=device)
        ops.extend(
            SyntheticTMAAddOneOp.schedule(
                x=x,
                y=y,
                tile_sizes={"M": SYNTHETIC_TMA_TILE_M},
            )
        )
    return ops


def selector_size_bytes(kernel: Megakernel, phase: str, kind: str) -> int:
    if kind == "transport":
        tensor = kernel._phase_local_transport_position_tensors[phase]
    else:
        tensor = kernel._phase_local_desc_slot_tensors[phase]
    return 0 if tensor is None else tensor.numel() * tensor.element_size()


def benchmark_case(num_layers: int, backend: str, repeats: int, device: str):
    make_ops_samples_ms = []
    init_samples_ms = []
    prepare_samples_ms = []
    dispatch_samples_ms = []
    create_samples_ms = []
    total_samples_ms = []
    summary = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        ops = make_ops(num_layers, device=device)
        t1 = time.perf_counter()
        kernel = Megakernel(
            ops,
            config=MegakernelConfig(num_sms=1, backend=backend),
            device=device,
        )
        t2 = time.perf_counter()
        kernel._prepare_tensors()
        t3 = time.perf_counter()
        kernel._prepare_cute_tensors()
        kernel._prepare_tma_tensors()
        kernel._prepare_peer_tma_tensors()
        t4 = time.perf_counter()
        kernel._backend.compile_phase_dispatch_inputs(kernel)
        t5 = time.perf_counter()
        kernel._create_kernel()
        t6 = time.perf_counter()
        make_ops_samples_ms.append((t1 - t0) * 1000.0)
        init_samples_ms.append((t2 - t1) * 1000.0)
        prepare_samples_ms.append((t3 - t2) * 1000.0)
        dispatch_samples_ms.append((t5 - t4) * 1000.0)
        create_samples_ms.append((t6 - t5) * 1000.0)
        total_samples_ms.append((t6 - t0) * 1000.0)
        summary = {
            "handlers": len(kernel._backend_ir.handler_specs),
            "transport_records": {
                phase: len(kernel._backend_ir.phase_transport_records[phase])
                for phase in ("load", "compute", "store", "communicate")
            },
            "transport_table_bytes": {
                phase: selector_size_bytes(kernel, phase, "transport")
                for phase in ("load", "compute", "store", "communicate")
            },
            "desc_slot_table_bytes": {
                phase: selector_size_bytes(kernel, phase, "desc")
                for phase in ("load", "compute", "store", "communicate")
            },
        }
    return {
        "make_ops_mean_ms": statistics.mean(make_ops_samples_ms),
        "init_mean_ms": statistics.mean(init_samples_ms),
        "prepare_mean_ms": statistics.mean(prepare_samples_ms),
        "dispatch_mean_ms": statistics.mean(dispatch_samples_ms),
        "create_mean_ms": statistics.mean(create_samples_ms),
        "total_mean_ms": statistics.mean(total_samples_ms),
        "total_min_ms": min(total_samples_ms),
        "total_max_ms": max(total_samples_ms),
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print(
        "backend,layers,make_ops_mean_ms,init_mean_ms,prepare_mean_ms,"
        "dispatch_mean_ms,create_mean_ms,"
        "total_mean_ms,total_min_ms,total_max_ms,handlers,load_records,"
        "store_records,load_transport_B,store_transport_B,load_desc_B,store_desc_B"
    )
    for backend in ("handler", "runtime"):
        for num_layers in args.layers:
            result = benchmark_case(num_layers, backend, args.repeats, args.device)
            summary = result["summary"]
            print(
                f"{backend},{num_layers},"
                f"{result['make_ops_mean_ms']:.3f},"
                f"{result['init_mean_ms']:.3f},"
                f"{result['prepare_mean_ms']:.3f},"
                f"{result['dispatch_mean_ms']:.3f},"
                f"{result['create_mean_ms']:.3f},"
                f"{result['total_mean_ms']:.3f},"
                f"{result['total_min_ms']:.3f},"
                f"{result['total_max_ms']:.3f},"
                f"{summary['handlers']},"
                f"{summary['transport_records']['load']},"
                f"{summary['transport_records']['store']},"
                f"{summary['transport_table_bytes']['load']},"
                f"{summary['transport_table_bytes']['store']},"
                f"{summary['desc_slot_table_bytes']['load']},"
                f"{summary['desc_slot_table_bytes']['store']}"
            )


if __name__ == "__main__":
    main()
