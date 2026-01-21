# Copyright (c) 2025, Machete Authors
"""Machete: Optimized kernels for HuggingFace Transformer models.

Machete patches existing HuggingFace models with optimized implementations
using flash-attn-cute and quack, while maintaining full compatibility with
Transformers, TRL, and LoRA/PEFT.

Example:
    >>> from transformers import AutoModelForCausalLM
    >>> import machete
    >>>
    >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
    >>> machete.patch(model)
    >>>
    >>> # Works with LoRA
    >>> from peft import get_peft_model, LoraConfig
    >>> model = get_peft_model(model, LoraConfig(...))
    >>>
    >>> # Works with TRL
    >>> from trl import SFTTrainer
    >>> trainer = SFTTrainer(model=model, ...)
"""

__version__ = "0.1.0"

from machete.patch import patch, unpatch, is_patched

# Copyright (c) 2025, Machete Authors

from .cluster_launch_control import (
    # Data structures
    CLCResponse,
    WorkTileInfo,
    CLCPipelineState,
    # Constants
    CLC_RESPONSE_BYTES,
    # Core CLC operations
    clc_try_cancel,
    clc_query_is_canceled,
    clc_query_get_first_ctaid,
    clc_load_response,
    clc_store_response,
    work_tile_info_from_clc_response,
    work_tile_info_from_smem,
    cast_smem_ptr_to_uint,
    # Mbarrier operations for CLC
    mbarrier_init_for_clc,
    mbarrier_arrive_expect_tx_for_clc,
    mbarrier_try_wait,
    mbarrier_try_wait_parity_ticks,
    mbarrier_arrive,
    mbarrier_invalidate,
    fence_mbarrier_init,
    # Pipeline helpers
    clc_pipeline_issue_query,
    clc_pipeline_wait_and_get_work,
    # Host-side configuration
    TileSchedulerParams,
    # CLC Pipeline classes
    CLCFetchPipeline,
    CLCPipelineProducerState,
    CLCPipelineConsumerState,
    # Persistent tile scheduler
    PersistentTileScheduler,
    # Device-side helpers
    get_block_idx,
    get_cluster_idx,
    get_cluster_dim,
    get_block_idx_in_cluster,
    compute_initial_tile_coords,
    swizzle_tile_coords,
    # Scheduler initialization
    scheduler_init_mbarriers,
    scheduler_prefill_pipeline,
    # Cluster synchronization
    cluster_barrier_sync,
    cluster_barrier_arrive,
    cluster_barrier_wait,
    # Leader election
    elect_one_in_cluster,
    is_cluster_leader,
)

from .utils import (
    nanosleep,
    atomic_add_i32,
    atomic_load_acquire_i32,
    atomic_store_release_i32,
    semaphore_init,
    semaphore_signal,
    semaphore_wait,
    semaphore_try_wait,
)

__all__ = [
    # Data structures
    "CLCResponse",
    "WorkTileInfo",
    "CLCPipelineState",
    # Constants
    "CLC_RESPONSE_BYTES",
    # Core CLC operations
    "clc_try_cancel",
    "clc_query_is_canceled",
    "clc_query_get_first_ctaid",
    "clc_load_response",
    "clc_store_response",
    "work_tile_info_from_clc_response",
    "work_tile_info_from_smem",
    "cast_smem_ptr_to_uint",
    # Mbarrier operations for CLC
    "mbarrier_init_for_clc",
    "mbarrier_arrive_expect_tx_for_clc",
    "mbarrier_try_wait",
    "mbarrier_try_wait_parity_ticks",
    "mbarrier_arrive",
    "mbarrier_invalidate",
    "fence_mbarrier_init",
    # Pipeline helpers
    "clc_pipeline_issue_query",
    "clc_pipeline_wait_and_get_work",
    # Host-side configuration
    "TileSchedulerParams",
    # CLC Pipeline classes
    "CLCFetchPipeline",
    "CLCPipelineProducerState",
    "CLCPipelineConsumerState",
    # Persistent tile scheduler
    "PersistentTileScheduler",
    # Device-side helpers
    "get_block_idx",
    "get_cluster_idx",
    "get_cluster_dim",
    "get_block_idx_in_cluster",
    "compute_initial_tile_coords",
    "swizzle_tile_coords",
    # Scheduler initialization
    "scheduler_init_mbarriers",
    "scheduler_prefill_pipeline",
    # Cluster synchronization
    "cluster_barrier_sync",
    "cluster_barrier_arrive",
    "cluster_barrier_wait",
    # Leader election
    "elect_one_in_cluster",
    "is_cluster_leader",
    # Utilities - atomics and sleep
    "nanosleep",
    "atomic_add_i32",
    "atomic_load_acquire_i32",
    "atomic_store_release_i32",
    # Utilities - semaphores
    "semaphore_init",
    "semaphore_signal",
    "semaphore_wait",
    "semaphore_try_wait",
    # Others
    "patch",
    "unpatch",
    "is_patched",
    "__version__",
]
