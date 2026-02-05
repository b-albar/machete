# Copyright (c) 2025, Machete Authors
"""
Dependency pattern tests for megakernel barrier synchronization.

Each test file covers a specific dependency pattern with ASCII diagrams
explaining the tile mapping and barrier synchronization. All tests execute
on GPU with real kernels (requires Hopper+ GPU).

Test Files:
-----------

test_one_to_one_same_size.py
    1:1 dependency with same tile dimensions.
    Producer tile k signals barrier k, consumer tile k waits on barrier k.
    Pattern: A(M) -> B(M)

test_one_to_many_tile_ratio.py
    1:many dependency where consumer has extra dimensions.
    Producer (1D) feeds consumer (2D), consumer expands over N dimension.
    Pattern: A(M) -> B(M x N)

test_many_to_one_tile_ratio.py
    Many:1 dependency where producer has extra dimensions.
    Producer (2D) feeds consumer (1D), producer collapses over N dimension.
    Pattern: A(M x N) -> B(M), expected = N signals per barrier

test_producer_only_dims.py
    Alias for many:1 pattern - producer has dims consumer doesn't use.

test_consumer_only_dims.py
    Alias for 1:many pattern - consumer has dims producer doesn't use.

test_chain_mixed_sizes.py
    Chains with alternating 1D and 2D ops.
    Tests dimension expansion and collapse in multi-op chains.
    Pattern: A(2D) -> B(1D) -> C(2D) and variations

test_diamond_dependency.py
    Diamond pattern where multiple ops depend on the same producer,
    and another op depends on all of them.
    Pattern: A -> B,C -> D (fork-join with parallel branches)
"""
