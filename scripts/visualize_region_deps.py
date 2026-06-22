#!/usr/bin/env python3
"""Visualize megakernel region dependencies from dependency CSVs.

The generated HTML is intentionally dependency-free: open it directly in a
browser to inspect op tile shapes, producer/consumer buffers, dependency edge
kinds, per-edge tile counts, and optional Perfetto-derived timing.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _read_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _parse_counts(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(part) for part in value.split(":") if part != ""]


def _region_axis_to_dict(axis) -> dict[str, Any]:
    return {
        "name": axis.name,
        "extent": int(axis.extent),
        "tile_dim": axis.tile_dim,
        "tile_size": int(axis.tile_size),
        "tile_origin": int(axis.tile_origin),
        "start": int(axis.start),
        "stop": None if axis.stop is None else int(axis.stop),
    }


def _region_to_dict(region) -> dict[str, Any]:
    return {
        "tensor": region.tensor,
        "axes": [_region_axis_to_dict(axis) for axis in region.axes],
        "group_dim": region.group_dim,
        "group_tiles": int(region.group_tiles),
        "group_count": region.group_count,
        "group_index": region.group_index,
        "group_index_offset": int(region.group_index_offset),
        "group_index_dim": region.group_index_dim,
        "group_index_group_tiles": int(region.group_index_group_tiles),
        "group_index_mode": region.group_index_mode,
        "group_index_all": bool(region.group_index_all),
        "prefix_dim": region.prefix_dim,
        "prefix_index_dim": region.prefix_index_dim,
        "prefix_group_tiles": int(region.prefix_group_tiles),
        "prefix_group_count": region.prefix_group_count,
    }


def _regions_to_dict(access) -> list[dict[str, Any]]:
    from machete.megakernel.ops import iter_tensor_access_regions

    return [_region_to_dict(region) for region in iter_tensor_access_regions(access)]


def _format_region(region: dict[str, Any]) -> str:
    axes = []
    for axis in region["axes"]:
        span = f"{axis['start']}:{axis['stop']}"
        tile = ""
        if axis["tile_dim"] is not None:
            tile = f" tile={axis['tile_dim']}[{axis['tile_origin']}]+{axis['tile_size']}"
        axes.append(f"{axis['name']}={axis['extent']}({span}{tile})")
    group_parts = []
    for key in (
        "group_dim",
        "group_tiles",
        "group_count",
        "group_index",
        "group_index_offset",
        "group_index_dim",
        "group_index_group_tiles",
        "group_index_mode",
        "group_index_all",
        "prefix_dim",
        "prefix_index_dim",
        "prefix_group_tiles",
        "prefix_group_count",
    ):
        value = region.get(key)
        if value not in (None, False):
            group_parts.append(f"{key}={value}")
    suffix = f" groups[{', '.join(group_parts)}]" if group_parts else ""
    return f"{region['tensor']}: " + ", ".join(axes) + suffix


def _collect_qwen_forward_regions(args: argparse.Namespace) -> list[dict[str, Any]]:
    import torch

    from benchmarks.kernels.benchmark_qwen3_5_layer import (
        BENCH_ATTENTION_TILE_M,
        BENCH_SPLIT_ATTENTION_SPLITS,
        BENCH_USE_FUSED_RMS_PROJ,
        BENCH_USE_PACKED_QKV_PROJECTION,
        BENCH_USE_QKNORM_4D,
        BENCH_USE_SPLIT_ATTENTION,
        BENCH_USE_TMA_ATTENTION,
        BENCH_USE_TWO_PAGE_ATTENTION,
        _alloc_layer,
        _scheduler,
    )
    from machete.kernels.qwen_3_5.qwen_3_5_forward import schedule_qwen3_5_forward_ops

    if not torch.cuda.is_available():
        raise RuntimeError("--qwen-forward requires CUDA to allocate schedule tensors")

    alloc = _alloc_layer(args.batch, args.seq_len)
    forward = schedule_qwen3_5_forward_ops(
        *alloc,
        page_size=args.page_size,
        scheduler=_scheduler("qwen-forward", args.fetch_stride or None),
        use_packed_qkv_projection=BENCH_USE_PACKED_QKV_PROJECTION,
        use_fused_rms_proj=BENCH_USE_FUSED_RMS_PROJ,
        use_qknorm_4d=BENCH_USE_QKNORM_4D,
        use_tma_attention=BENCH_USE_TMA_ATTENTION,
        use_two_page_attention=BENCH_USE_TWO_PAGE_ATTENTION,
        use_split_attention=BENCH_USE_SPLIT_ATTENTION,
        split_attention_splits=BENCH_SPLIT_ATTENTION_SPLITS,
        attention_tile_m=BENCH_ATTENTION_TILE_M,
    )

    out = []
    for op_idx, op in enumerate(forward.ops):
        regions = op.access_regions()
        out.append(
            {
                "op_idx": op_idx,
                "op": op.op_cls.__name__,
                "tile_counts": list(map(int, op.tile_counts)),
                "tile_sizes": {k: int(v) for k, v in op.tile_sizes.items()},
                "dim_names": dict(op.dim_names),
                "reads": {
                    name: _regions_to_dict(access)
                    for name, access in regions.reads.items()
                },
                "writes": {
                    name: _regions_to_dict(access)
                    for name, access in regions.writes.items()
                },
            }
        )
    return out


def _load_timing(perfetto: Path | None) -> dict[int, dict[str, float]]:
    if perfetto is None:
        return {}
    from machete.megakernel import TimingProfile

    profile = TimingProfile.from_perfetto(perfetto)
    return {
        int(op_idx): {
            "total_us": timing.total_us,
            "compute_us": timing.compute_us,
            "data_wait_us": timing.data_wait_us,
            "dep_wait_us": timing.dep_wait_us,
            "compute_wait_us": timing.compute_wait_us,
            "ring_full_wait_us": timing.ring_full_wait_us,
            "event_count": timing.event_count,
        }
        for op_idx, timing in profile.ops.items()
    }


def _split_pipe(value: str) -> list[str]:
    return value.split("|") if "|" in value else [value]


def _tile_edge_keys(row: dict[str, str]) -> list[tuple[str, str, str, str, str]]:
    producer_buffers = _split_pipe(row["producer_buffer"])
    consumer_buffers = _split_pipe(row["consumer_buffer"])
    kinds = _split_pipe(row["kind"])
    width = max(len(producer_buffers), len(consumer_buffers), len(kinds))

    def _expand(values: list[str]) -> list[str]:
        if len(values) == width:
            return values
        if len(values) == 1:
            return values * width
        return values

    producer_buffers = _expand(producer_buffers)
    consumer_buffers = _expand(consumer_buffers)
    kinds = _expand(kinds)
    if not (len(producer_buffers) == len(consumer_buffers) == len(kinds)):
        return [
            (
                row["producer_op_idx"],
                row["producer_buffer"],
                row["consumer_op_idx"],
                row["consumer_buffer"],
                row["kind"],
            )
        ]
    return [
        (
            row["producer_op_idx"],
            producer_buffer,
            row["consumer_op_idx"],
            consumer_buffer,
            kind,
        )
        for producer_buffer, consumer_buffer, kind in zip(
            producer_buffers,
            consumer_buffers,
            kinds,
        )
    ]


def _build_model(
    op_deps: list[dict[str, str]],
    tile_deps: list[dict[str, str]],
    timing: dict[int, dict[str, float]],
    regions: list[dict[str, Any]],
) -> dict[str, Any]:
    ops: dict[int, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    tile_edge_counts: Counter[tuple[str, str, str, str, str]] = Counter()
    expected_by_edge: dict[tuple[str, str, str, str, str], Counter[int]] = defaultdict(Counter)

    for row in tile_deps:
        for edge_key in _tile_edge_keys(row):
            tile_edge_counts[edge_key] += 1
            if row.get("expected"):
                expected_by_edge[edge_key][int(row["expected"])] += 1

    region_by_op = {int(item["op_idx"]): item for item in regions}

    for row in op_deps:
        edge_id = int(row["edge_id"])
        producer = int(row["producer_op_idx"])
        consumer = int(row["consumer_op_idx"])
        edge_key = (
            row["producer_op_idx"],
            row["producer_buffer"],
            row["consumer_op_idx"],
            row["consumer_buffer"],
            row["kind"],
        )
        producer_counts = _parse_counts(row.get("producer_tile_counts"))
        consumer_counts = _parse_counts(row.get("consumer_tile_counts"))
        ops.setdefault(
            producer,
            {
                "op_idx": producer,
                "op": row["producer_op"],
                "tile_counts": producer_counts,
            },
        )
        ops.setdefault(
            consumer,
            {
                "op_idx": consumer,
                "op": row["consumer_op"],
                "tile_counts": consumer_counts,
            },
        )
        edges.append(
            {
                "edge_id": edge_id,
                "producer_op_idx": producer,
                "producer_op": row["producer_op"],
                "producer_buffer": row["producer_buffer"],
                "producer_tiles": int(row["producer_tiles"]),
                "producer_tile_counts": producer_counts,
                "consumer_op_idx": consumer,
                "consumer_op": row["consumer_op"],
                "consumer_buffer": row["consumer_buffer"],
                "consumer_tiles": int(row["consumer_tiles"]),
                "consumer_tile_counts": consumer_counts,
                "kind": row["kind"],
                "tile_deps": tile_edge_counts.get(edge_key, 0),
                "expected_histogram": dict(sorted(expected_by_edge.get(edge_key, {}).items())),
            }
        )

    for op_idx, op in ops.items():
        if op_idx in timing:
            op["timing"] = timing[op_idx]
        if op_idx in region_by_op:
            declared = region_by_op[op_idx]
            op["tile_sizes"] = declared["tile_sizes"]
            op["dim_names"] = declared["dim_names"]
            op["reads"] = declared["reads"]
            op["writes"] = declared["writes"]

    return {
        "ops": [ops[idx] for idx in sorted(ops)],
        "edges": edges,
        "summary": {
            "num_ops": len(ops),
            "num_edges": len(edges),
            "num_tile_deps": len(tile_deps),
        },
    }


def _json_for_script(model: dict[str, Any]) -> str:
    return json.dumps(model, separators=(",", ":")).replace("</", "<\\/")


def _html_doc(model: dict[str, Any], title: str) -> str:
    data = _json_for_script(model)
    escaped_title = html.escape(title)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{escaped_title}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: system-ui, sans-serif; margin: 18px; color: #172026; background: #ffffff; }}
    h1 {{ font-size: 20px; margin: 0 0 12px; }}
    h2 {{ font-size: 15px; margin: 12px 0 8px; }}
    button, input, select {{ font: inherit; }}
    button {{ border: 1px solid #c8d0d8; background: #f7f9fb; border-radius: 5px; padding: 5px 8px; cursor: pointer; }}
    button:hover {{ background: #edf2f7; }}
    .summary {{ display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }}
    .pill {{ border: 1px solid #ccd3da; border-radius: 6px; padding: 6px 8px; background: #f7f9fb; }}
    .toolbar {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin: 8px 0 14px; padding: 10px; border: 1px solid #d7dee5; border-radius: 6px; background: #fbfcfd; }}
    .toolbar label {{ display: flex; gap: 5px; align-items: center; white-space: nowrap; font-size: 12px; }}
    .toolbar input[type="search"] {{ min-width: 240px; padding: 5px 7px; border: 1px solid #c8d0d8; border-radius: 5px; }}
    .toolbar input[type="number"] {{ width: 92px; padding: 4px 6px; border: 1px solid #c8d0d8; border-radius: 5px; }}
    .layout {{ display: grid; grid-template-columns: minmax(720px, 1.35fr) minmax(420px, .65fr); gap: 16px; align-items: start; }}
    .graph-wrap {{ position: sticky; top: 12px; }}
    svg {{ width: 100%; height: min(72vh, 760px); min-height: 520px; border: 1px solid #d7dee5; border-radius: 6px; background: #fbfcfd; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border-bottom: 1px solid #e3e8ed; padding: 5px 6px; text-align: left; vertical-align: top; }}
    th {{ background: #f2f5f8; position: sticky; top: 0; }}
    .small {{ color: #596671; font-size: 11px; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .node {{ cursor: pointer; }}
    .opbox {{ fill: white; stroke: #7e8b97; stroke-width: 1.2; rx: 5; transition: fill .12s, stroke .12s; }}
    .node:hover .opbox {{ fill: #eef6ff; stroke: #2d7dd2; }}
    .node.selected .opbox {{ fill: #dff0ff; stroke: #0b6fc6; stroke-width: 2.2; }}
    .node.dimmed {{ opacity: .18; }}
    .edge-hit {{ fill: none; stroke: transparent; stroke-width: 11; cursor: pointer; }}
    .edge {{ fill: none; stroke: #4b7bec; stroke-width: 1.6; opacity: .72; marker-end: url(#arrow); transition: opacity .12s, stroke-width .12s; pointer-events: none; }}
    .edge.many_to_one {{ stroke: #e17055; }}
    .edge.one_to_one {{ stroke: #0984e3; }}
    .edge.one_to_many {{ stroke: #00b894; }}
    .edge.selected {{ stroke-width: 3.3; opacity: 1; }}
    .edge.dimmed {{ opacity: .08; }}
    .hidden {{ display: none; }}
    .label {{ font-size: 11px; fill: #172026; }}
    .muted {{ fill: #65717c; color: #65717c; font-size: 10px; }}
    .edge-label {{ cursor: pointer; }}
    .edge-label.dimmed {{ opacity: .12; }}
    .row-click {{ cursor: pointer; }}
    .row-click:hover {{ background: #eef6ff; }}
    .row-selected {{ background: #dff0ff; }}
    .panel {{ border: 1px solid #d7dee5; border-radius: 6px; padding: 10px; background: #fff; margin-bottom: 12px; }}
    .panel h2 {{ margin-top: 0; }}
    .bars {{ display: grid; gap: 4px; margin-top: 6px; }}
    .bar-row {{ display: grid; grid-template-columns: 88px 1fr 70px; gap: 6px; align-items: center; font-size: 11px; }}
    .bar-bg {{ height: 8px; background: #eef2f5; border-radius: 4px; overflow: hidden; }}
    .bar-fill {{ height: 100%; background: #4b7bec; }}
    .bar-fill.compute {{ background: #00a8ff; }}
    .bar-fill.dep {{ background: #e17055; }}
    .bar-fill.data {{ background: #6c5ce7; }}
    .bar-fill.ring {{ background: #00b894; }}
    .scroll {{ max-height: 34vh; overflow: auto; border: 1px solid #e3e8ed; border-radius: 6px; }}
    details {{ margin: 8px 0; }}
    summary {{ cursor: pointer; font-weight: 600; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #f7f9fb; border: 1px solid #e3e8ed; border-radius: 6px; padding: 8px; font-size: 11px; }}
  </style>
</head>
<body>
  <h1>{escaped_title}</h1>
  <div class="summary" id="summary"></div>
  <div class="toolbar">
    <label>Search <input id="search" type="search" placeholder="op, buffer, edge id"></label>
    <label><input id="kind-one" type="checkbox" checked> one-to-one</label>
    <label><input id="kind-many" type="checkbox" checked> many-to-one</label>
    <label><input id="kind-one-many" type="checkbox" checked> one-to-many</label>
    <label>min tile deps <input id="min-deps" type="number" min="0" value="0"></label>
    <label>color <select id="color-mode"><option value="op">op type</option><option value="time">total time</option><option value="dep">dep wait</option></select></label>
    <button id="reset">Reset</button>
  </div>
  <div class="layout">
    <div class="graph-wrap">
      <svg id="graph" viewBox="0 0 980 620" role="img" aria-label="region dependency graph"></svg>
      <p class="small">Click nodes or edges to inspect details. Blue: one-to-one, orange: many-to-one, green: one-to-many. Edge labels show buffer mapping and tile-dependency count.</p>
    </div>
    <div>
      <div class="panel" id="details"><h2>Selection</h2><div class="small">Click an op or edge.</div></div>
      <h2>Edges</h2>
      <div id="edges" class="scroll"></div>
      <h2>Ops / Regions</h2>
      <div id="ops"></div>
    </div>
  </div>
  <script id="model-data" type="application/json">{data}</script>
  <script>
  const model = JSON.parse(document.getElementById("model-data").textContent);
  let selected = null;
  let graphEls = {{ nodes: new Map(), edgePaths: new Map(), edgeHits: new Map(), edgeLabels: new Map() }};

  function fmtCounts(counts) {{ return counts && counts.length ? counts.join("x") : "1"; }}
  function fmtTiming(t) {{
    if (!t) return "";
    return `total ${{t.total_us.toFixed(1)}}us, compute ${{t.compute_us.toFixed(1)}}us, dep ${{t.dep_wait_us.toFixed(1)}}us, data ${{t.data_wait_us.toFixed(1)}}us`;
  }}
  function regionText(r) {{
    const axes = r.axes.map(a => {{
      const tile = a.tile_dim == null ? "" : ` tile=${{a.tile_dim}}[${{a.tile_origin}}]+${{a.tile_size}}`;
      return `${{a.name}}=${{a.extent}}(${{a.start}}:${{a.stop}}${{tile}})`;
    }}).join(", ");
    const groups = ["group_dim","group_tiles","group_count","group_index","group_index_offset","group_index_dim","group_index_group_tiles","group_index_mode","group_index_all","prefix_dim","prefix_index_dim","prefix_group_tiles","prefix_group_count"]
      .filter(k => r[k] !== null && r[k] !== false && r[k] !== undefined)
      .map(k => `${{k}}=${{r[k]}}`).join(", ");
    return `${{r.tensor}}: ${{axes}}${{groups ? " groups[" + groups + "]" : ""}}`;
  }}
  function shortOp(name) {{
    return name.replace("Qwen3_5Forward", "").replace("Qwen3_5", "").replace("Op", "");
  }}
  function edgeText(e) {{
    return `${{e.producer_op_idx}}:${{e.producer_buffer}} → ${{e.consumer_op_idx}}:${{e.consumer_buffer}}`;
  }}
  function edgeSearchText(e) {{
    return `${{e.edge_id}} ${{edgeText(e)}} ${{e.kind}} ${{e.producer_op}} ${{e.consumer_op}}`.toLowerCase();
  }}
  function opSearchText(op) {{
    const regions = [];
    for (const side of ["reads", "writes"]) {{
      for (const regs of Object.values(op[side] || {{}})) {{
        for (const r of regs) regions.push(regionText(r));
      }}
    }}
    return `${{op.op_idx}} ${{op.op}} ${{fmtCounts(op.tile_counts)}} ${{regions.join(" ")}}`.toLowerCase();
  }}
  function timingRows(t) {{
    if (!t) return "<div class='small'>No timing profile loaded.</div>";
    const rows = [
      ["total", "total", t.total_us],
      ["compute", "compute", t.compute_us],
      ["dep wait", "dep", t.dep_wait_us],
      ["data wait", "data", t.data_wait_us],
      ["ring full", "ring", t.ring_full_wait_us],
      ["compute wait", "ring", t.compute_wait_us],
    ];
    const max = Math.max(1, t.total_us);
    return `<div class="bars">` + rows.map(([name, cls, value]) =>
      `<div class="bar-row"><div>${{name}}</div><div class="bar-bg"><div class="bar-fill ${{cls}}" style="width:${{Math.min(100, value / max * 100).toFixed(1)}}%"></div></div><div class="mono">${{value.toFixed(1)}}us</div></div>`
    ).join("") + `</div>`;
  }}
  function regionBlock(title, regionsByName) {{
    const lines = [];
    for (const [name, regs] of Object.entries(regionsByName || {{}})) {{
      for (const r of regs) lines.push(regionText(r));
    }}
    return `<h3 style="font-size:13px">${{title}}</h3><pre>${{lines.join("\\n") || "(none)"}}</pre>`;
  }}

  document.getElementById("summary").innerHTML = [
    ["ops", model.summary.num_ops],
    ["edges", model.summary.num_edges],
    ["tile deps", model.summary.num_tile_deps],
  ].map(([k,v]) => `<div class="pill"><b>${{k}}</b> ${{v}}</div>`).join("");

  const svg = document.getElementById("graph");
  const ops = model.ops.slice().sort((a,b) => a.op_idx - b.op_idx);
  const edges = model.edges.slice().sort((a,b) => a.edge_id - b.edge_id);
  const opByIdx = new Map(ops.map(op => [op.op_idx, op]));
  const maxTime = Math.max(1, ...ops.map(op => op.timing ? op.timing.total_us : 0));
  const maxDep = Math.max(1, ...ops.map(op => op.timing ? op.timing.dep_wait_us : 0));
  const xByIdx = new Map();
  const yByIdx = new Map();
  const dx = Math.max(105, Math.floor(900 / Math.max(1, ops.length)));
  ops.forEach((op, i) => {{
    const x = 24 + i * dx;
    const y = 90 + (i % 2) * 130;
    xByIdx.set(op.op_idx, x);
    yByIdx.set(op.op_idx, y);
  }});

  function nodeFill(op) {{
    const mode = document.getElementById("color-mode").value;
    if (mode === "time" && op.timing) {{
      const v = Math.min(1, op.timing.total_us / maxTime);
      return `rgb(${{Math.round(255 - 40*v)}}, ${{Math.round(255 - 105*v)}}, ${{Math.round(255 - 150*v)}})`;
    }}
    if (mode === "dep" && op.timing) {{
      const v = Math.min(1, op.timing.dep_wait_us / maxDep);
      return `rgb(${{Math.round(255)}}, ${{Math.round(255 - 115*v)}}, ${{Math.round(255 - 155*v)}})`;
    }}
    if (op.op.includes("Projection") || op.op.includes("Down")) return "#fff8ec";
    if (op.op.includes("Attention") || op.op.includes("Prefill")) return "#eef7ff";
    if (op.op.includes("RMS")) return "#f3fff2";
    if (op.op.includes("GLU")) return "#fff0fb";
    return "white";
  }}

  function renderGraph() {{
    graphEls = {{ nodes: new Map(), edgePaths: new Map(), edgeHits: new Map(), edgeLabels: new Map() }};
    svg.innerHTML = `<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#4b7bec"/></marker></defs>`;
    const maxX = 70 + (ops.length - 1) * dx + 120;
    svg.setAttribute("viewBox", `0 0 ${{Math.max(980, maxX)}} 620`);
    for (const edge of edges) {{
    const x1 = xByIdx.get(edge.producer_op_idx) + 82;
    const y1 = yByIdx.get(edge.producer_op_idx) + 26;
    const x2 = xByIdx.get(edge.consumer_op_idx);
    const y2 = yByIdx.get(edge.consumer_op_idx) + 26;
    const mid = (x1 + x2) / 2;
    const d = `M ${{x1}} ${{y1}} C ${{mid}} ${{y1}}, ${{mid}} ${{y2}}, ${{x2}} ${{y2}}`;
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", d);
    path.setAttribute("class", `edge ${{edge.kind}}`);
    svg.appendChild(path);
    const hit = document.createElementNS("http://www.w3.org/2000/svg", "path");
    hit.setAttribute("d", d);
    hit.setAttribute("class", "edge-hit");
    hit.addEventListener("click", () => selectEdge(edge.edge_id));
    svg.appendChild(hit);
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", mid - 28);
    label.setAttribute("y", Math.min(y1, y2) - 8 - (edge.edge_id % 3) * 13);
    label.setAttribute("class", "muted edge-label");
    label.textContent = `${{edge.producer_buffer}}→${{edge.consumer_buffer}} ${{edge.tile_deps}}`;
    label.addEventListener("click", () => selectEdge(edge.edge_id));
    svg.appendChild(label);
    graphEls.edgePaths.set(edge.edge_id, path);
    graphEls.edgeHits.set(edge.edge_id, hit);
    graphEls.edgeLabels.set(edge.edge_id, label);
  }}
  for (const op of ops) {{
    const x = xByIdx.get(op.op_idx), y = yByIdx.get(op.op_idx);
    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    group.setAttribute("class", "node");
    group.addEventListener("click", () => selectOp(op.op_idx));
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", x); rect.setAttribute("y", y); rect.setAttribute("width", 86); rect.setAttribute("height", 54); rect.setAttribute("class", "opbox");
    rect.style.fill = nodeFill(op);
    group.appendChild(rect);
    const text1 = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text1.setAttribute("x", x + 5); text1.setAttribute("y", y + 16); text1.setAttribute("class", "label");
    text1.textContent = `${{op.op_idx}} ${{shortOp(op.op)}}`;
    group.appendChild(text1);
    const text2 = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text2.setAttribute("x", x + 5); text2.setAttribute("y", y + 34); text2.setAttribute("class", "muted");
    text2.textContent = `tiles ${{fmtCounts(op.tile_counts)}}`;
    group.appendChild(text2);
    const text3 = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text3.setAttribute("x", x + 5); text3.setAttribute("y", y + 48); text3.setAttribute("class", "muted");
    text3.textContent = op.timing ? `${{op.timing.total_us.toFixed(0)}}us agg` : "";
    group.appendChild(text3);
    svg.appendChild(group);
    graphEls.nodes.set(op.op_idx, group);
  }}
    applyFilters();
    applySelection();
  }}

  function renderTables() {{
    document.getElementById("edges").innerHTML = `<table><thead><tr><th>edge</th><th>mapping</th><th>kind</th><th>tile deps</th><th>expected</th></tr></thead><tbody>` +
    edges.map(e => `<tr class="row-click edge-row" data-edge="${{e.edge_id}}"><td>${{e.edge_id}}</td><td>${{edgeText(e)}}</td><td>${{e.kind}}</td><td>${{e.tile_deps}}</td><td>${{JSON.stringify(e.expected_histogram)}}</td></tr>`).join("") +
    `</tbody></table>`;
    for (const row of document.querySelectorAll(".edge-row")) {{
      row.addEventListener("click", () => selectEdge(Number(row.dataset.edge)));
    }}

    document.getElementById("ops").innerHTML = ops.map(op => {{
    const readLines = op.reads ? Object.entries(op.reads).flatMap(([name, regs]) => regs.map(regionText)) : [];
    const writeLines = op.writes ? Object.entries(op.writes).flatMap(([name, regs]) => regs.map(regionText)) : [];
    return `<details class="op-details" data-op="${{op.op_idx}}"><summary>${{op.op_idx}} ${{op.op}} <span class="small">tiles=${{fmtCounts(op.tile_counts)}} ${{fmtTiming(op.timing)}}</span></summary>` +
      `<pre>tile_sizes: ${{JSON.stringify(op.tile_sizes || {{}})}}\\ndim_names: ${{JSON.stringify(op.dim_names || {{}})}}\\n\\nreads:\\n${{readLines.join("\\n") || "(not loaded)"}}\\n\\nwrites:\\n${{writeLines.join("\\n") || "(not loaded)"}}</pre></details>`;
  }}).join("");
    for (const detail of document.querySelectorAll(".op-details")) {{
      detail.querySelector("summary").addEventListener("click", () => selectOp(Number(detail.dataset.op)));
    }}
  }}

  function enabledKinds() {{
    const kinds = new Set();
    if (document.getElementById("kind-one").checked) kinds.add("one_to_one");
    if (document.getElementById("kind-many").checked) kinds.add("many_to_one");
    if (document.getElementById("kind-one-many").checked) kinds.add("one_to_many");
    return kinds;
  }}
  function edgeVisible(e) {{
    const q = document.getElementById("search").value.trim().toLowerCase();
    const minDeps = Number(document.getElementById("min-deps").value || 0);
    return enabledKinds().has(e.kind) && e.tile_deps >= minDeps && (!q || edgeSearchText(e).includes(q));
  }}
  function opVisible(op) {{
    const q = document.getElementById("search").value.trim().toLowerCase();
    if (!q) return true;
    return opSearchText(op).includes(q) || edges.some(e => edgeVisible(e) && (e.producer_op_idx === op.op_idx || e.consumer_op_idx === op.op_idx));
  }}
  function applyFilters() {{
    const visibleEdges = new Set();
    const visibleOps = new Set();
    for (const e of edges) {{
      const visible = edgeVisible(e);
      if (visible) {{
        visibleEdges.add(e.edge_id);
        visibleOps.add(e.producer_op_idx);
        visibleOps.add(e.consumer_op_idx);
      }}
      graphEls.edgePaths.get(e.edge_id)?.classList.toggle("hidden", !visible);
      graphEls.edgeHits.get(e.edge_id)?.classList.toggle("hidden", !visible);
      graphEls.edgeLabels.get(e.edge_id)?.classList.toggle("hidden", !visible);
    }}
    for (const op of ops) {{
      const visible = opVisible(op) || visibleOps.has(op.op_idx);
      graphEls.nodes.get(op.op_idx)?.classList.toggle("hidden", !visible);
    }}
    for (const row of document.querySelectorAll(".edge-row")) {{
      const e = edges.find(item => item.edge_id === Number(row.dataset.edge));
      row.classList.toggle("hidden", !edgeVisible(e));
    }}
  }}

  function selectOp(opIdx) {{
    selected = {{ type: "op", id: opIdx }};
    const op = opByIdx.get(opIdx);
    const connected = edges.filter(e => e.producer_op_idx === opIdx || e.consumer_op_idx === opIdx);
    document.getElementById("details").innerHTML =
      `<h2>Op ${{op.op_idx}}: ${{op.op}}</h2>` +
      `<div class="small">tiles=${{fmtCounts(op.tile_counts)}} tile_sizes=${{JSON.stringify(op.tile_sizes || {{}})}}</div>` +
      timingRows(op.timing) +
      `<h3 style="font-size:13px">Connected edges</h3><pre>${{connected.map(e => `e${{e.edge_id}} ${{edgeText(e)}} ${{e.kind}} deps=${{e.tile_deps}} expected=${{JSON.stringify(e.expected_histogram)}}`).join("\\n") || "(none)"}}</pre>` +
      regionBlock("Reads", op.reads) + regionBlock("Writes", op.writes);
    applySelection();
  }}
  function selectEdge(edgeId) {{
    selected = {{ type: "edge", id: edgeId }};
    const e = edges.find(item => item.edge_id === edgeId);
    const producer = opByIdx.get(e.producer_op_idx);
    const consumer = opByIdx.get(e.consumer_op_idx);
    document.getElementById("details").innerHTML =
      `<h2>Edge ${{e.edge_id}}: ${{edgeText(e)}}</h2>` +
      `<div class="small">${{e.kind}}, tile deps=${{e.tile_deps}}, expected=${{JSON.stringify(e.expected_histogram)}}</div>` +
      `<pre>producer: ${{producer.op}} tiles=${{fmtCounts(producer.tile_counts)}}\\nconsumer: ${{consumer.op}} tiles=${{fmtCounts(consumer.tile_counts)}}\\nproducer buffer: ${{e.producer_buffer}}\\nconsumer buffer: ${{e.consumer_buffer}}</pre>` +
      `<h3 style="font-size:13px">Producer timing</h3>${{timingRows(producer.timing)}}` +
      `<h3 style="font-size:13px">Consumer timing</h3>${{timingRows(consumer.timing)}}`;
    applySelection();
  }}
  function applySelection() {{
    for (const [idx, el] of graphEls.nodes) {{
      el.classList.remove("selected", "dimmed");
      if (!selected) continue;
      if (selected.type === "op") {{
        if (idx === selected.id) el.classList.add("selected");
        else if (!edges.some(e => e.edge_id === selected.id || ((e.producer_op_idx === selected.id || e.consumer_op_idx === selected.id) && (e.producer_op_idx === idx || e.consumer_op_idx === idx)))) el.classList.add("dimmed");
      }} else {{
        const e = edges.find(item => item.edge_id === selected.id);
        if (idx === e.producer_op_idx || idx === e.consumer_op_idx) el.classList.add("selected");
        else el.classList.add("dimmed");
      }}
    }}
    for (const e of edges) {{
      const active = selected && (
        (selected.type === "edge" && e.edge_id === selected.id) ||
        (selected.type === "op" && (e.producer_op_idx === selected.id || e.consumer_op_idx === selected.id))
      );
      graphEls.edgePaths.get(e.edge_id)?.classList.toggle("selected", !!active);
      graphEls.edgePaths.get(e.edge_id)?.classList.toggle("dimmed", !!selected && !active);
      graphEls.edgeLabels.get(e.edge_id)?.classList.toggle("dimmed", !!selected && !active);
    }}
    for (const row of document.querySelectorAll(".edge-row")) {{
      row.classList.toggle("row-selected", selected?.type === "edge" && Number(row.dataset.edge) === selected.id);
    }}
  }}
  function resetSelection() {{
    selected = null;
    document.getElementById("search").value = "";
    document.getElementById("min-deps").value = "0";
    document.getElementById("kind-one").checked = true;
    document.getElementById("kind-many").checked = true;
    document.getElementById("kind-one-many").checked = true;
    document.getElementById("color-mode").value = "op";
    document.getElementById("details").innerHTML = "<h2>Selection</h2><div class='small'>Click an op or edge.</div>";
    renderGraph();
    renderTables();
  }}

  for (const id of ["search", "kind-one", "kind-many", "kind-one-many", "min-deps"]) {{
    document.getElementById(id).addEventListener("input", () => {{ applyFilters(); applySelection(); }});
  }}
  document.getElementById("color-mode").addEventListener("input", renderGraph);
  document.getElementById("reset").addEventListener("click", resetSelection);
  renderGraph();
  renderTables();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op-deps", type=Path, required=True)
    parser.add_argument("--tile-deps", type=Path, default=None)
    parser.add_argument("--perfetto", type=Path, default=None)
    parser.add_argument("--output-html", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--title", default="Machete Region Dependencies")
    parser.add_argument("--qwen-forward", action="store_true", help="Add live Qwen 3.5 forward region declarations.")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=32768)
    parser.add_argument("--fetch-stride", type=int, default=0)
    args = parser.parse_args()

    op_deps = _read_csv(args.op_deps)
    tile_deps = _read_csv(args.tile_deps)
    timing = _load_timing(args.perfetto)
    regions = _collect_qwen_forward_regions(args) if args.qwen_forward else []
    model = _build_model(op_deps, tile_deps, timing, regions)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(model, indent=2), encoding="utf-8")
    if args.output_html:
        args.output_html.parent.mkdir(parents=True, exist_ok=True)
        args.output_html.write_text(_html_doc(model, args.title), encoding="utf-8")

    print(json.dumps(model["summary"], indent=2))
    if args.output_html:
        print(f"html={args.output_html}")
    if args.output_json:
        print(f"json={args.output_json}")


if __name__ == "__main__":
    main()
