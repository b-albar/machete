#!/usr/bin/env python
"""Compatibility wrapper for the Qwen 3.5 MXFP4 decode benchmark."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if __name__ != "__main__":
    from benchmarks.kernels.qwen_3_5.benchmark_qwen3_5_mxfp4_simt_decode import *  # noqa: F401,F403


if __name__ == "__main__":
    runpy.run_module(
        "benchmarks.kernels.qwen_3_5.benchmark_qwen3_5_mxfp4_simt_decode",
        run_name="__main__",
    )
