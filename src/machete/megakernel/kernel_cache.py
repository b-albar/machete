# Copyright (c) 2025, Machete Authors
"""
Thread-safe cache for compiled megakernels.

The preferred path is to key this cache from a prepared ``Megakernel`` via its
runtime-dedicated structural cache key. That keeps the autograd wrapper aligned
with ``Megakernel.compile()`` as more scheduling state moves out of the emitted
kernel and into runtime metadata.

Legacy ``lookup(ops, config)`` / ``store(ops, config, ...)`` entry points are
still supported, but callers that already constructed a ``Megakernel`` should
use ``lookup_key()`` / ``store_key()`` to avoid duplicating key derivation.
"""

import threading
from typing import Dict, List, Optional, Tuple

from .ops import ScheduledOp
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
    ) -> CacheKey:
        """Build the runtime-dedicated structural cache key for one workload."""
        from .megakernel import Megakernel

        mk = Megakernel(ops, config=config, device="cpu")
        mk._prepare_tensors()
        mk._prepare_tma_tensors()
        mk._prepare_peer_tma_tensors()
        return mk._make_cache_key()

    def lookup_key(self, key: CacheKey) -> object:
        """Look up a precomputed structural cache key. Returns None on miss."""
        return self._cache.get(key)

    def lookup(
        self,
        ops: List[ScheduledOp],
        config: MegakernelConfig,
    ) -> object:
        """Look up a cached compiled kernel. Returns None on miss."""
        return self.lookup_key(self.make_key(ops, config))

    def store_key(
        self,
        key: CacheKey,
        compiled_kernel: object,
    ) -> None:
        """Store a compiled kernel under a precomputed structural key."""
        with self._lock:
            self._cache[key] = compiled_kernel

    def store(
        self,
        ops: List[ScheduledOp],
        config: MegakernelConfig,
        compiled_kernel: object,
    ) -> None:
        """Store a compiled kernel in the cache."""
        self.store_key(self.make_key(ops, config), compiled_kernel)

    def clear(self):
        """Clear all cached kernels."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


__all__ = ["KernelCache"]
