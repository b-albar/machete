import itertools
import math
from typing import Any, List, Callable, Optional
import copy
import time
import matplotlib.pyplot as plt
import os
import numpy as np
import torch

from .benchmark_utils import (
    benchmark_forward,
    benchmark_backward,
    benchmark_memory,
    benchmark_fwd_bwd,
    benchmark_jit_kernel,
    KernelBenchSpec,
    efficiency,
    memory_throughput,
    CUTLASS_AVAILABLE,
)


class Benchmark:
    def __init__(self):
        self.parameters = {}
        self.func = {}
        self._configs = None
        self._config_names = None

    @classmethod
    def parametrize(cls, param_name: str, values: List[Any]):
        def decorator(func):
            if not hasattr(func, "_benchmark"):
                func._benchmark = cls()
            func._benchmark.parameters[param_name] = values
            func._benchmark.func = func
            return func

        return decorator

    @classmethod
    def configs(cls, param_names: List[str], config_list: List[tuple]):
        """Parametrize with explicit configs (no cartesian product).

        Unlike ``parametrize`` which generates all combinations, this accepts
        a list of specific parameter tuples to benchmark.

        Args:
            param_names: Parameter names, e.g. ["BH", "M", "N", "D"].
            config_list: List of tuples, each matching param_names order.

        Example::

            @Benchmark.configs(["BH", "M", "N", "D"], [
                (32, 16, 128, 128),
                (32, 64, 256, 128),
            ])
            def bench_fn(BH, M, N, D):
                ...
        """
        def decorator(func):
            if not hasattr(func, "_benchmark"):
                func._benchmark = cls()
            func._benchmark._config_names = param_names
            func._benchmark._configs = config_list
            func._benchmark.func = func
            return func
        return decorator

    def _plot_graphics(self, results, path_graphics, key_split, memory=False, is_flops=False):
        dirname = path_graphics if path_graphics is not None else "benchmark_results"
        os.makedirs(dirname, exist_ok=True)

        param_names = list(self.parameters.keys())
        param_values = list(self.parameters.values())

        index_split = param_names.index(key_split)
        split_values = param_values[index_split]

        # extract function names
        names = list(list(results.values())[0].keys())

        # remove split key
        param_values.pop(index_split)
        param_names_without_split = [name for name in param_names if name != key_split]

        modes = ["fwd", "bwd"]
        if memory:
            modes.append("memory")

        for mode in modes:
            # compute the max value to set the ylim with a bit of margin
            max_value = max([x[mode] for xs in results.values() for x in xs.values()])
            max_value += 20 / 100 * max_value

            for combination in itertools.product(*param_values):
                combination = list(combination)

                values_per_func = {}
                for name in names:
                    values_per_func[name] = []

                for split in split_values:
                    comb = copy.copy(combination)
                    comb.insert(index_split, split)

                    param_key = str(dict(zip(param_names, comb)))
                    result = results[param_key]

                    for name in names:
                        values_per_func[name].append(round(result[name][mode], 1))

                x = np.arange(len(split_values))  # the label locations
                width = 1.0 / (len(names) + len(split_values) - 1)  # the width of the bars
                multiplier = 0

                fig, ax = plt.subplots(layout="constrained")

                for attribute, measurement in values_per_func.items():
                    offset = width * multiplier
                    rects = ax.bar(x + offset, measurement, width, label=attribute)
                    ax.bar_label(rects, padding=3)
                    multiplier += 1

                # Add some text for labels, title and custom x-axis tick labels, etc.
                string_params = " ".join(
                    [str(x) + " " + str(combination[i]) for i, x in enumerate(param_names_without_split)]
                )
                filename_params_str = ",".join(
                    [str(x) + "-" + str(combination[i]) for i, x in enumerate(param_names_without_split)]
                )
                plt.xlabel("Sequence Length")
                if mode == "memory":
                    plt.ylabel("Memory (MB)")
                elif is_flops:
                    plt.ylabel("TFLOPS")
                else:
                    plt.ylabel("Time (ms)")
                plt.title(f"{mode.upper()}\n" + string_params, wrap=True)
                ax.set_xticks(x + width, split_values)
                ax.legend(loc="upper left", ncols=2)
                ax.set_ylim(0, max_value)

                filename = dirname + "/" + f"{mode.upper()}" + "-" + filename_params_str + ".png"
                plt.savefig(filename)
                plt.close()

    def _bench_kernel_func(self, func_or_spec, warmup, rep, force_host_timer: bool = False):
        """Benchmark a kernel callable or `KernelBenchSpec`.

        Kernel-mode benchmarks are timed directly with CUDA events on a
        dedicated stream. `KernelBenchSpec` instances may also provide a
        `setup_fn` that runs before each launch.

        This avoids CUDA graph capture entirely for kernel benchmarks, which
        is important for megakernels, dynamic launch sequences, and plain
        callables that allocate outputs during execution.

        Args:
            func_or_spec: A callable or a KernelBenchSpec.
            warmup: Number of warmup iterations.
            rep: Number of timed iterations.

        Returns:
            Execution time in milliseconds.
        """
        if isinstance(func_or_spec, KernelBenchSpec):
            torch_stream, _ = func_or_spec.stream
            launch = func_or_spec.launch_fn
            setup = func_or_spec.setup_fn
            use_host_timer = func_or_spec.use_host_timer
        else:
            torch_stream = torch.cuda.Stream()
            launch = func_or_spec
            setup = None
            use_host_timer = False

        use_host_timer = use_host_timer or force_host_timer

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = []
        if use_host_timer:
            # Host timing is too noisy for very small kernels if we only time
            # one launch per sample. Probe once, then batch multiple launches
            # into each timed sample and divide back down.
            if setup is not None:
                setup()
            t0 = time.perf_counter()
            launch()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            probe_ms = max((t1 - t0) * 1000.0, 1e-3)
            inner_repeats = max(1, min(128, int(math.ceil(5.0 / probe_ms))))

            for _ in range(warmup):
                for _ in range(inner_repeats):
                    if setup is not None:
                        setup()
                    launch()
                torch.cuda.synchronize()

            for _ in range(rep):
                t0 = time.perf_counter()
                for _ in range(inner_repeats):
                    if setup is not None:
                        setup()
                    launch()
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append(((t1 - t0) * 1000.0) / inner_repeats)

            return sum(times) / len(times)

        with torch.cuda.stream(torch_stream):
            for _ in range(warmup):
                if setup is not None:
                    setup()
                launch()
            torch_stream.synchronize()

            for _ in range(rep):
                if setup is not None:
                    setup()
                start.record(torch_stream)
                launch()
                end.record(torch_stream)
                torch_stream.synchronize()
                times.append(start.elapsed_time(end))

        return sum(times) / len(times)

    def _print_kernel_summary(self, results, names, bytes_fn):
        """Print a formatted summary table for kernel benchmark results."""
        print()
        print("=" * 120)
        print("SUMMARY")
        print("=" * 120)

        has_gbps = bytes_fn is not None
        baseline_name = names[0] if names else None

        # Header
        header = f"{'Shape':<35}"
        for name in names:
            if has_gbps:
                header += f" {name:>16}"
            else:
                header += f" {name + ' (ms)':>14}"
            if name != baseline_name:
                header += f" {'speedup':>8}"
        print(header)

        if has_gbps:
            sub = f"{'':<35}"
            for name in names:
                sub += f" {'ms':>8} {'GB/s':>7}"
                if name != baseline_name:
                    sub += f" {'':<8}"
            print(sub)

        print("-" * len(header))

        # Rows
        for param_key, func_results in results.items():
            # Build a short label from param dict
            label = param_key.replace("'", "").replace("{", "").replace("}", "")
            if len(label) > 34:
                label = label[:31] + "..."
            line = f"{label:<35}"

            baseline_ms = None
            for name in names:
                data = func_results.get(name, {})
                time_ms = data.get("time_ms", float("nan"))

                if baseline_ms is None:
                    baseline_ms = time_ms

                if has_gbps:
                    gbps = data.get("gbps", float("nan"))
                    line += f" {time_ms:>8.3f} {gbps:>7.1f}"
                else:
                    line += f" {time_ms:>14.3f}"

                if name != baseline_name and baseline_ms and baseline_ms > 0 and time_ms > 0:
                    speedup = baseline_ms / time_ms
                    line += f" {speedup:>7.2f}x"

            print(line)

        print("=" * 120)

    def _print_kernel_header(self, names, bytes_fn):
        """Print table header for line-by-line kernel benchmark output."""
        has_gbps = bytes_fn is not None
        baseline_name = names[0] if names else None

        show_check = any(
            isinstance(candidate, KernelBenchSpec) and candidate.metadata
            for candidate in self._current_kernel_funcs.values()
        ) if hasattr(self, "_current_kernel_funcs") else False

        header = f"{'Config':<36}"
        for name in names:
            if has_gbps:
                header += f" {name:>16}"
            else:
                header += f" {name + ' (ms)':>14}"
            if name != baseline_name:
                header += f" {'speedup':>8}"
        if show_check:
            header += f" {'check':<24}"
        print(header.rstrip())

        if has_gbps:
            sub = f"{'':<36}"
            for name in names:
                sub += f" {'ms':>8} {'GB/s':>7}"
                if name != baseline_name:
                    sub += f" {'':<8}"
            if show_check:
                sub += f" {'':<24}"
            print(sub.rstrip())

        print("-" * len(header.rstrip()))

    def _print_kernel_row(self, params, func_results, names, bytes_fn):
        """Print a single result row for kernel benchmark."""
        has_gbps = bytes_fn is not None
        baseline_name = names[0] if names else None

        # Abbreviate param names to fit more info: hidden_dim→D, seq_len→S, etc.
        _ABBREV = {
            "hidden_dim": "D", "seq_len": "S", "batch": "B", "page_size": "pg",
            "context_len": "ctx", "num_pages": "np",
            "BH": "BH", "H": "H", "M": "M", "N": "N", "K": "K", "D": "D",
        }
        def _fmt_val(k, v):
            if k == "page_size" and isinstance(v, int):
                return f"{v // 1024}K"
            return str(v)
        label = ", ".join(
            f"{_ABBREV.get(k, k)}={_fmt_val(k, v)}" for k, v in params.items()
        )
        if len(label) > 35:
            label = label[:32] + "..."
        line = f"{label:<36}"

        baseline_ms = None
        checks = []
        for name in names:
            data = func_results.get(name, {})
            time_ms = data.get("time_ms", float("nan"))
            meta = data.get("metadata")
            if meta:
                checks.append(str(meta))

            if baseline_ms is None:
                baseline_ms = time_ms

            if has_gbps:
                gbps = data.get("gbps", float("nan"))
                line += f" {time_ms:>8.3f} {gbps:>7.1f}"
            else:
                line += f" {time_ms:>14.3f}"

            if name != baseline_name and baseline_ms and baseline_ms > 0 and time_ms > 0:
                speedup = baseline_ms / time_ms
                line += f" {speedup:>7.2f}x"

        if checks:
            check = "; ".join(checks)
            if len(check) > 24:
                check = check[:21] + "..."
            line += f" {check:<24}"

        print(line, flush=True)

    def run(
        self,
        flops: Optional[Callable] = None,
        mode: str = "fwd",
        memory: bool = False,
        export_csv: bool = True,
        export_graphics: bool = False,
        key_split: Optional[str] = None,
        path_graphics: Optional[str] = None,
        bytes_fn: Optional[Callable] = None,
        warmup: int = 25,
        rep: int = 100,
        print_summary: bool = True,
        columns: Optional[list] = None,
    ):
        if self._configs is not None:
            param_names = self._config_names
            all_combinations = self._configs
        else:
            param_names = list(self.parameters.keys())
            param_values = list(self.parameters.values())
            all_combinations = list(itertools.product(*param_values))

        results = {}

        # Use explicit columns if provided, otherwise discover dynamically
        if columns is not None:
            names_seen = set(columns)
            names_ordered = list(columns)
        else:
            names_seen = set()
            names_ordered = []
        _header_printed_for = None

        for combination in all_combinations:
            params = dict(zip(param_names, combination))

            try:
                funcs = self.func(**params)
            except torch.cuda.OutOfMemoryError:
                print(f"OOM during setup for params: {params}")
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"Error during setup for params: {params}, error: {e}")
                continue

            for func_name, func in funcs.items():
                if func_name not in names_seen:
                    names_seen.add(func_name)
                    names_ordered.append(func_name)

                if str(params) not in results:
                    results[str(params)] = {}
                results[str(params)][func_name] = {}

                try:
                    if mode == "kernel":
                        row_use_host_timer = any(
                            isinstance(candidate, KernelBenchSpec) and candidate.use_host_timer
                            for candidate in funcs.values()
                        )
                        time_ms = self._bench_kernel_func(
                            func, warmup, rep, force_host_timer=row_use_host_timer
                        )
                        results[str(params)][func_name]["time_ms"] = time_ms
                        if isinstance(func, KernelBenchSpec) and func.metadata:
                            results[str(params)][func_name]["metadata"] = func.metadata
                        if bytes_fn is not None:
                            total_bytes = bytes_fn(**params)
                            gbps = memory_throughput(total_bytes, time_ms * 1000)
                            results[str(params)][func_name]["gbps"] = gbps
                    elif mode == "fwd":
                        result = benchmark_forward(func, verbose=False)
                        if flops is not None:
                            flops_op = flops(**params, mode=mode)
                            results[str(params)][func_name]["fwd"] = efficiency(flops_op, result[1].mean)
                        else:
                            results[str(params)][func_name]["fwd"] = result[1].mean * 1000
                        result = benchmark_backward(func, backward=True, verbose=False)
                        if flops is not None:
                            flops_op = flops(**params, mode=mode)
                            results[str(params)][func_name]["bwd"] = efficiency(flops_op, result[1].mean)
                        else:
                            results[str(params)][func_name]["bwd"] = result[1].mean * 1000
                    elif mode == "fwd_bwd":
                        result = benchmark_fwd_bwd(func, verbose=False)
                        if flops is not None:
                            flops_fwd = flops(**params, mode="fwd")
                            flops_bwd = flops(**params, mode="bwd")
                            results[str(params)][func_name]["fwd"] = efficiency(flops_fwd, result[0][1].mean)
                            results[str(params)][func_name]["bwd"] = efficiency(flops_bwd, result[1][1].mean)
                        else:
                            results[str(params)][func_name]["fwd"] = result[0][1].mean * 1000
                            results[str(params)][func_name]["bwd"] = result[1][1].mean * 1000
                    else:
                        raise ValueError(f"Invalid mode: {mode}")

                    if memory and mode != "kernel":
                        result = benchmark_memory(func, verbose=False)
                        results[str(params)][func_name]["memory"] = result * 1024
                except torch.cuda.OutOfMemoryError:
                    print(f"OOM for {func_name} with params: {params}")
                    torch.cuda.empty_cache()
                    if mode == "kernel":
                        results[str(params)][func_name]["time_ms"] = 0.0
                        results[str(params)][func_name]["gbps"] = 0.0
                    else:
                        results[str(params)][func_name]["fwd"] = 0.0
                        results[str(params)][func_name]["bwd"] = 0.0
                        results[str(params)][func_name]["memory"] = 0.0
                except Exception as e:
                    print(f"Error for {func_name} with params: {params}, error: {e}")
                    if mode == "kernel":
                        results[str(params)][func_name]["time_ms"] = 0.0
                        results[str(params)][func_name]["gbps"] = 0.0
                    else:
                        results[str(params)][func_name]["fwd"] = 0.0
                        results[str(params)][func_name]["bwd"] = 0.0
                        results[str(params)][func_name]["memory"] = 0.0

            # Line-by-line printing for kernel mode
            if mode == "kernel":
                header_key = tuple(names_ordered)
                if _header_printed_for != header_key and names_ordered:
                    self._current_kernel_funcs = funcs
                    self._print_kernel_header(names_ordered, bytes_fn)
                    del self._current_kernel_funcs
                    _header_printed_for = header_key
                if str(params) in results:
                    self._print_kernel_row(
                        params, results[str(params)], names_ordered, bytes_fn,
                    )
                torch.cuda.empty_cache()

        if export_graphics and mode != "kernel":
            self._plot_graphics(results, path_graphics, key_split=key_split, memory=memory, is_flops=flops is not None)

        return results
