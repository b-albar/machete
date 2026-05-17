# Test Suite Layout

The suite is organized by responsibility first, then by subsystem.

## Coverage Tiers

- `unit`
  Fast, narrowly scoped logic checks.
  Examples:
  - scheduler/dependency math in `tests/megakernel/deps/`
  - instruction-stream and barrier tests in `tests/megakernel/test_ops.py`
  - lightweight patching behavior tests

- `smoke`
  Compile-and-run coverage for the default maintained paths.
  These are the first GPU checks to run when validating framework changes.
  This bucket should stay intentionally small.

- `integration`
  Cross-subsystem or model-shaped coverage.
  Examples:
  - persistent megakernel integration chains
  - Qwen prefill coverage
  - heavier attention / MoE / patching flows

- `slow`
  Expensive end-to-end tests that should be run deliberately, not as the first gate.

## Suite Markers

- `kernels`
  Kernel-level op tests under `tests/kernels`
- `megakernel`
  Framework and persistent-runtime tests under `tests/megakernel`
- `patching`
  Model patching tests under `tests/patching`
- `deps`
  Scheduler dependency-mapping coverage under `tests/megakernel/deps`
- `arch_sm120`
  Architecture-specific SM120 coverage

Markers are attached automatically from `tests/conftest.py`, so file layout and coverage intent stay aligned.

## Recommended Validation Order

1. Fast host-side coverage:
   `pytest -m unit`
2. Maintained GPU smoke gate:
   `pytest -m "smoke and gpu and not slow"`
3. Focused subsystem validation:
   `pytest -m "kernels or megakernel or patching"`
4. Heavier end-to-end checks:
   `pytest -m integration`
5. Expensive full checks:
   `pytest -m slow`

## Canonical Coverage Shape

- Fast framework logic:
  - `tests/megakernel/test_ops.py`
  - `tests/megakernel/deps/`
  - `tests/megakernel/test_autograd.py`
- Fast framework smoke:
  - `tests/megakernel/test_megakernel.py`
  - `tests/megakernel/test_persistent_kernel.py`
  - `tests/megakernel/test_tma_megakernel.py`
- Heavy framework integration:
  - `tests/megakernel/test_communicate.py`
  - `tests/megakernel/test_integration_gpu.py`
  - `tests/megakernel/test_qwen_nvfp4_ops.py`
- Fast kernel smoke:
  - `tests/kernels/test_activation.py`
  - `tests/kernels/test_glu.py`
  - `tests/kernels/test_rope.py`
- Heavy kernel regressions:
  - `tests/kernels/test_gemm.py`
  - `tests/kernels/test_rmsnorm.py`
  - `tests/kernels/test_attention_sm_120.py`
  - `tests/kernels/test_gated_delta_net.py`

The goal is to keep `smoke` small and representative, and push broad shape matrices into `integration` / `slow`.
