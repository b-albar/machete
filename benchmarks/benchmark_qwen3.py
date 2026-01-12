#!/usr/bin/env python
# Copyright (c) 2025, Machete Authors
"""Benchmark script for comparing Machete-patched vs vanilla HuggingFace models.

Usage:
    python benchmarks/benchmark_qwen3.py
"""

import argparse
import time
from contextlib import contextmanager
import logging
import machete

import torch

logging.basicConfig(level=logging.INFO)


@contextmanager
def cuda_timer():
    """Context manager for accurate CUDA timing."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed


def benchmark_generation(model, tokenizer, prompt: str, max_new_tokens: int = 128, warmup: int = 3, repeats: int = 10):
    """Benchmark text generation speed."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None
            )

    # Benchmark
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    times = []
    total_tokens = 0

    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        total_tokens += outputs.shape[1] - inputs.input_ids.shape[1]

    max_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB

    avg_time = sum(times) / len(times)
    avg_tokens = total_tokens / repeats
    tokens_per_sec = avg_tokens / avg_time

    return {
        "avg_time_ms": avg_time * 1000,
        "tokens_per_sec": tokens_per_sec,
        "avg_new_tokens": avg_tokens,
        "max_memory_mb": max_memory,
    }


def benchmark_forward(model, input_ids, warmup: int = 5, repeats: int = 20):
    """Benchmark forward pass speed."""
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(input_ids)

    # Benchmark
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    times = []

    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            _ = model(input_ids)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    max_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB

    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

    return {
        "avg_time_ms": avg_time * 1000,
        "std_time_ms": std_time * 1000,
        "min_time_ms": min(times) * 1000,
        "max_memory_mb": max_memory,
    }


def benchmark_backward(model, input_ids, warmup: int = 5, repeats: int = 20):
    """Benchmark backward pass speed (forward + backward)."""
    # Warmup
    for _ in range(warmup):
        outputs = model(input_ids)
        loss = outputs.logits.sum()
        loss.backward()
        model.zero_grad()

    # Benchmark
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    times = []

    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()

        outputs = model(input_ids)
        loss = outputs.logits.sum()
        loss.backward()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        model.zero_grad()

    max_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB

    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

    return {
        "avg_time_ms": avg_time * 1000,
        "std_time_ms": std_time * 1000,
        "min_time_ms": min(times) * 1000,
        "max_memory_mb": max_memory,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Machete vs HuggingFace")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B", help="Model name or path")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length for forward benchmark")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens for generation")
    parser.add_argument("--repeats", type=int, default=10, help="Number of benchmark repeats")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"], help="Model dtype")
    parser.add_argument("--skip-backward", action="store_true", help="Skip backward pass benchmark")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print("=" * 60)
    print(f"Machete Benchmark: {args.model}")
    print("=" * 60)
    print(f"Sequence length: {args.seq_len}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Dtype: {args.dtype}")
    print(f"Repeats: {args.repeats}")
    print()

    # Load model and tokenizer
    print("Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map="cuda",
        attn_implementation="eager",  # Use regular attention as baseline
    )

    # Create test inputs
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, args.seq_len), device="cuda")
    prompt = "The quick brown fox jumps over the lazy dog. " * 10

    print(f"Model loaded on {model.device}")
    print()

    # Benchmark vanilla HuggingFace
    print("-" * 60)
    print("Benchmarking: Vanilla HuggingFace")
    print("-" * 60)

    model.eval()
    hf_forward = benchmark_forward(model, input_ids, repeats=args.repeats)
    print(f"Forward pass: {hf_forward['avg_time_ms']:.2f} ± {hf_forward['std_time_ms']:.2f} ms")

    if not args.skip_backward:
        model.train()
        hf_backward = benchmark_backward(model, input_ids, repeats=args.repeats)
        print(f"Fwd+Bwd pass: {hf_backward['avg_time_ms']:.2f} ± {hf_backward['std_time_ms']:.2f} ms")

    model.eval()
    hf_gen = benchmark_generation(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens, repeats=args.repeats)
    print(f"Generation: {hf_gen['avg_time_ms']:.2f} ms ({hf_gen['tokens_per_sec']:.1f} tokens/sec)")
    print()

    # Apply Machete patches
    print("-" * 60)
    print("Applying Machete patches...")
    print("-" * 60)

    machete.patch(model)
    print()

    # Inspect module classes
    print("Inspecting model module classes:")
    class_names = set()
    for module in model.modules():
        class_names.add(module.__class__.__name__)
    print(f"  Classes found: {sorted(list(class_names))}")

    # Verify patches
    print("Verifying patches...")
    patch_count = 0
    for name, module in model.named_modules():
        if hasattr(module, "_machete_original_forward"):
            patch_count += 1
            if patch_count <= 5:
                print(f"  [PATCHED] {name} ({module.__class__.__name__})")

    print(f"  Total patched modules: {patch_count}")
    if patch_count == 0:
        print("  WARNING: No modules were patched!")
    print()

    # Benchmark Machete-patched model
    print("-" * 60)
    print("Benchmarking: Machete (flash-attn-cute + quack)")
    print("-" * 60)

    model.eval()
    machete_forward = benchmark_forward(model, input_ids, repeats=args.repeats)
    print(f"Forward pass: {machete_forward['avg_time_ms']:.2f} ± {machete_forward['std_time_ms']:.2f} ms")

    if not args.skip_backward:
        model.train()
        machete_backward = benchmark_backward(model, input_ids, repeats=args.repeats)
        print(f"Fwd+Bwd pass: {machete_backward['avg_time_ms']:.2f} ± {machete_backward['std_time_ms']:.2f} ms")

    model.eval()
    machete_gen = benchmark_generation(
        model, tokenizer, prompt, max_new_tokens=args.max_new_tokens, repeats=args.repeats
    )
    print(f"Generation: {machete_gen['avg_time_ms']:.2f} ms ({machete_gen['tokens_per_sec']:.1f} tokens/sec)")
    print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    forward_speedup = hf_forward["avg_time_ms"] / machete_forward["avg_time_ms"]
    gen_speedup = hf_gen["avg_time_ms"] / machete_gen["avg_time_ms"]

    print(f"Forward pass speedup: {forward_speedup:.2f}x")
    if not args.skip_backward:
        backward_speedup = hf_backward["avg_time_ms"] / machete_backward["avg_time_ms"]
        print(f"Fwd+Bwd speedup:      {backward_speedup:.2f}x")
    print(f"Generation speedup:   {gen_speedup:.2f}x")
    print()

    print(f"{'Metric':<25} {'HuggingFace':<20} {'Machete':<20} {'Speedup':<10}")
    print("-" * 75)
    print(
        f"{'Forward (ms)':<25} {hf_forward['avg_time_ms']:<20.2f} "
        f"{machete_forward['avg_time_ms']:<20.2f} {forward_speedup:<10.2f}x"
    )
    print(
        f"{'Forward Mem (MB)':<25} {hf_forward['max_memory_mb']:<20.2f} "
        f"{machete_forward['max_memory_mb']:<20.2f} {'-':<10}"
    )

    if not args.skip_backward:
        print(
            f"{'Fwd+Bwd (ms)':<25} {hf_backward['avg_time_ms']:<20.2f} "
            f"{machete_backward['avg_time_ms']:<20.2f} {backward_speedup:<10.2f}x"
        )
        print(
            f"{'Fwd+Bwd Mem (MB)':<25} {hf_backward['max_memory_mb']:<20.2f} "
            f"{machete_backward['max_memory_mb']:<20.2f} {'-':<10}"
        )

    print(
        f"{'Generation (ms)':<25} {hf_gen['avg_time_ms']:<20.2f} "
        f"{machete_gen['avg_time_ms']:<20.2f} {gen_speedup:<10.2f}x"
    )
    print(f"{'Gen Mem (MB)':<25} {hf_gen['max_memory_mb']:<20.2f} {machete_gen['max_memory_mb']:<20.2f} {'-':<10}")
    print(f"{'Tokens/sec':<25} {hf_gen['tokens_per_sec']:<20.1f} {machete_gen['tokens_per_sec']:<20.1f}")


if __name__ == "__main__":
    main()
