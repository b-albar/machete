#!/usr/bin/env python
"""Compatibility wrapper for the Qwen 3.5 layer benchmark."""

from __future__ import annotations

import runpy

from benchmarks.kernels.qwen_3_5.benchmark_qwen3_5_layer import *  # noqa: F401,F403


if __name__ == "__main__":
    runpy.run_module("benchmarks.kernels.qwen_3_5.benchmark_qwen3_5_layer", run_name="__main__")
