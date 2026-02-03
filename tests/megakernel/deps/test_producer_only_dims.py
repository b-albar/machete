# Copyright (c) 2025, Machete Authors
"""
Test: Producer-Only Dimensions (GPU)

This is essentially the same as many-to-one: producer has extra dimensions
that the consumer doesn't use. Re-exports test_many_to_one_tile_ratio tests.

See test_many_to_one_tile_ratio.py for the full implementation.
"""

from tests.megakernel.deps.test_many_to_one_tile_ratio import (
    TestManyToOnePatternGPU as TestProducerOnlyDimsGPU,
)

__all__ = ["TestProducerOnlyDimsGPU"]
