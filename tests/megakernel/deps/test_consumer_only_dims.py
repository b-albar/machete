# Copyright (c) 2025, Machete Authors
"""
Test: Consumer-Only Dimensions (GPU)

This is essentially the same as one-to-many: consumer has extra dimensions
that the producer doesn't use. Re-exports test_one_to_many_tile_ratio tests.

See test_one_to_many_tile_ratio.py for the full implementation.
"""

from tests.megakernel.deps.test_one_to_many_tile_ratio import (
    TestOneToManyPatternGPU as TestConsumerOnlyDimsGPU,
)

__all__ = ["TestConsumerOnlyDimsGPU"]
