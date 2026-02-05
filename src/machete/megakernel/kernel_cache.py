# Copyright (c) 2025, Machete Authors
"""
Thread-safe cache for compiled megakernels.

Caches compiled kernel objects keyed on structural shape to avoid
~77ms CuTe DSL re-tracing per call. Instruction streams and config
tensors are rebuilt per call (cheap CPU work).

The cache key captures everything that affects the compiled MLIR/PTX:
- Op class types and their execution modes
- Tile dimensions (barrier formulas bake tiles_m/n into coefficients)
- num_sms, threads_per_block, num_pages, backward flag

Usage:
    cache = KernelCache.get()
    compiled = cache.lookup(scheduled_ops, mk_config, backward=False)
    if compiled is None:
        # compile, then store
        cache.store(scheduled_ops, mk_config, backward=False, compiled_kernel=kernel)
"""

import threading
from typing import Dict, List, Optional, Tuple, Type

from .ops import ScheduledOp, Op
from .megakernel import MegakernelConfig


# Structural cache key type
CacheKey = Tuple


class KernelCache:
    """Thread-safe singleton cache for compiled megakernels.

    Caches ``Megakernel._compiled_kernel`` objects. Each autograd call
    creates a fresh ``Megakernel``, injects the cached compiled kernel,
    and calls ``run()``. Only ``_prepare_tensors()`` runs per call
    (instruction stream + config tensor building — cheap CPU work).
    """

    _instance: Optional["KernelCache"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._cache: Dict[CacheKey, object] = {}

    @classmethod
    def get(cls) -> "KernelCache":
        """Get the singleton cache instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @staticmethod
    def make_key(
        ops: List[ScheduledOp],
        config: MegakernelConfig,
        backward: bool,
    ) -> CacheKey:
        """Build a cache key from op structure and config.

        Includes tile dimensions because barrier formula coefficients
        (computed in ``_compute_formula_coeffs``) use ``tiles_m``/``tiles_n``
        at JIT compile time.
        """
        op_structure = tuple(
            (
                op.op_cls,
                op.tiles_m,
                op.tiles_n,
                op.tiles_l,
                tuple(sorted(op.static_dims.items())) if op.static_dims else (),
            )
            for op in ops
        )
        return (
            op_structure,
            config.num_sms,
            backward,
            config.threads_per_block,
            config.num_pages,
        )

    def lookup(
        self,
        ops: List[ScheduledOp],
        config: MegakernelConfig,
        backward: bool,
    ) -> object:
        """Look up a cached compiled kernel. Returns None on miss."""
        key = self.make_key(ops, config, backward)
        return self._cache.get(key)

    def store(
        self,
        ops: List[ScheduledOp],
        config: MegakernelConfig,
        backward: bool,
        compiled_kernel: object,
    ) -> None:
        """Store a compiled kernel in the cache."""
        key = self.make_key(ops, config, backward)
        with self._lock:
            self._cache[key] = compiled_kernel

    def clear(self):
        """Clear all cached kernels."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


__all__ = ["KernelCache"]
