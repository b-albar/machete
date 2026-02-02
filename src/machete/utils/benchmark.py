import itertools
from typing import Any, List, Callable, Optional
import copy
import matplotlib.pyplot as plt
import os
import numpy as np
import torch

from .benchmark_utils import (
    benchmark_forward,
    benchmark_backward,
    benchmark_memory,
    benchmark_fwd_bwd,
    benchmark_cuda_graph,
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

    @classmethod
    def parametrize(cls, param_name: str, values: List[Any]):
        def decorator(func):
            if not hasattr(func, "_benchmark"):
                func._benchmark = cls()
            func._benchmark.parameters[param_name] = values
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

    def _bench_kernel_func(self, func_or_spec, warmup, rep):
        """Benchmark a single function or KernelBenchSpec using CUDA graphs.

        Args:
            func_or_spec: A callable (benchmarked with torch CUDA graphs)
                or a KernelBenchSpec (benchmarked with cute.testing.benchmark).
            warmup: Number of warmup iterations.
            rep: Number of timed iterations.

        Returns:
            Execution time in milliseconds.
        """
        if isinstance(func_or_spec, KernelBenchSpec):
            torch_stream, cu_stream = func_or_spec.stream
            with torch.cuda.stream(torch_stream):
                time_us = benchmark_jit_kernel(
                    func_or_spec.compiled_kernel,
                    workspace_generator=func_or_spec.workspace_generator,
                    warmup_iterations=warmup,
                    iterations=rep,
                    workspace_count=func_or_spec.workspace_count,
                    stream=cu_stream,
                    use_cuda_graphs=True,
                )
            return time_us / 1000.0
        else:
            return benchmark_cuda_graph(func_or_spec, warmup=warmup, rep=rep)

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

                if name != baseline_name and baseline_ms and baseline_ms > 0:
                    speedup = baseline_ms / time_ms if time_ms > 0 else float("inf")
                    line += f" {speedup:>7.2f}x"

            print(line)

        print("=" * 120)

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
    ):
        param_names = list(self.parameters.keys())
        param_values = list(self.parameters.values())

        results = {}

        names_seen = set()
        names_ordered = []

        for combination in itertools.product(*param_values):
            # get functions list
            params = dict(zip(param_names, combination))

            try:
                funcs = self.func(**params)
                # Update known names from successful retrieval
                if funcs and not names_seen:
                    names_seen = set(funcs.keys())
                    names_ordered = list(funcs.keys())
            except torch.cuda.OutOfMemoryError:
                print(f"OOM during setup for params: {params}")
                torch.cuda.empty_cache()
                if names_seen:
                    if mode == "kernel":
                        results[str(params)] = {
                            name: {"time_ms": 0.0, "gbps": 0.0} for name in names_seen
                        }
                    else:
                        results[str(params)] = {
                            name: {"fwd": 0.0, "bwd": 0.0, "memory": 0.0} for name in names_seen
                        }
                continue
            except Exception as e:
                print(f"Error during setup for params: {params}, error: {e}")
                if names_seen:
                    if mode == "kernel":
                        results[str(params)] = {
                            name: {"time_ms": 0.0, "gbps": 0.0} for name in names_seen
                        }
                    else:
                        results[str(params)] = {
                            name: {"fwd": 0.0, "bwd": 0.0, "memory": 0.0} for name in names_seen
                        }
                continue

            for func_name, func in funcs.items():
                if str(params) not in results:
                    results[str(params)] = {}
                results[str(params)][func_name] = {}

                try:
                    if mode == "kernel":
                        time_ms = self._bench_kernel_func(func, warmup, rep)
                        results[str(params)][func_name]["time_ms"] = time_ms
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

            # Cleanup between parameter combinations
            if mode == "kernel":
                torch.cuda.empty_cache()

        if mode == "kernel" and print_summary:
            self._print_kernel_summary(results, names_ordered, bytes_fn)

        if export_graphics and mode != "kernel":
            self._plot_graphics(results, path_graphics, key_split=key_split, memory=memory, is_flops=flops is not None)

        return results
