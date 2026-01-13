# Copyright (c) 2025, Machete Authors
"""
No Bubbles Scheduler for Megakernels.

This module implements the scheduling infrastructure for the No Bubbles pattern:
1. Paged shared memory management
2. Instruction sequencing per SM
3. Global synchronization via atomic counters
4. Page request/release coordination

Based on: "Look Ma, No Bubbles!" - HazyResearch
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
import torch
from enum import Enum, auto


class MicroOpType(Enum):
    LOAD = auto()
    COMPUTE = auto()
    STORE = auto()
    SYNC_BLOCK = auto()
    SYNC_GLOBAL = auto()


@dataclass
class MicroOp:
    type: MicroOpType
    op_idx: int
    desc: str = ""


@dataclass
class Instruction:
    """An instruction for a single SM to execute."""

    opcode: int
    op_idx: int
    args: List
    page_ids: List[int]
    depends_on: List[int] = field(default_factory=list)
    signals: List[int] = field(default_factory=list)


@dataclass
class NoBubblesConfig:
    num_pages: int = 13
    page_size_bytes: int = 16384
    num_sync_counters: int = 64
    max_instructions_per_sm: int = 256


class NoBubblesScheduler:
    def __init__(self, config: Optional[NoBubblesConfig] = None):
        self.config = config or NoBubblesConfig()
        self.instructions: List[Instruction] = []
        self.micro_ops: List[MicroOp] = []

    def add_micro_op(self, type: MicroOpType, op_idx: int, desc: str = ""):
        self.micro_ops.append(MicroOp(type, op_idx, desc))

    def generate_pipeline_schedule(self, ops: List[dict], use_pipeline: bool = True):
        """Generates the main execution loop schedule."""
        self.micro_ops.clear()
        n_ops = len(ops)

        if not use_pipeline:
            # Serial Schedule: L -> Sync -> C -> S -> Sync
            for i in range(n_ops):
                self.add_micro_op(MicroOpType.LOAD, i)
                if ops[i].get("needs_block_sync", True):
                    self.add_micro_op(MicroOpType.SYNC_BLOCK, i, "Wait for Load")

                self.add_micro_op(MicroOpType.COMPUTE, i)
                self.add_micro_op(MicroOpType.STORE, i)

                if ops[i].get("needs_block_sync", True):
                    self.add_micro_op(MicroOpType.SYNC_BLOCK, i, "Wait for Store")

                if ops[i].get("needs_sync", False):
                    self.add_micro_op(MicroOpType.SYNC_GLOBAL, i)
            return

        # No Bubbles Pipeline Schedule
        # Prologue: Load[0] -> Sync
        if n_ops > 0:
            self.add_micro_op(MicroOpType.LOAD, 0, "Prologue")
            self.add_micro_op(MicroOpType.SYNC_BLOCK, 0)

        for i in range(n_ops):
            # Compute[i]
            self.add_micro_op(MicroOpType.COMPUTE, i)

            # Load[i+1] (Overlapped)
            if i + 1 < n_ops:
                self.add_micro_op(MicroOpType.LOAD, i + 1, "Prefetch")

            # Store[i]
            self.add_micro_op(MicroOpType.STORE, i)

            # Synchronization Logic
            # If current op needs global sync OR next op depends on current

            # If i+1 exists, we usually need sync unless independent or thread-local
            if i + 1 < n_ops and ops[i].get("needs_block_sync", True):
                self.add_micro_op(MicroOpType.SYNC_BLOCK, i)

            if ops[i].get("needs_sync", False):
                self.add_micro_op(MicroOpType.SYNC_GLOBAL, i)

    # ... (rest of methods like allocate_pages kept for future use)
