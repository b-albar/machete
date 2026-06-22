# Copyright (c) 2026, Machete Authors
"""Small source-fragment builders shared by megakernel codegen."""

from __future__ import annotations

from dataclasses import dataclass
import re
import textwrap
from typing import Dict, Optional

import torch

from .backend_ir import PHASE_NAMES
from .compile import _extract_body


@dataclass(frozen=True)
class PhasePointerFragments:
    local_sig: str
    local_arg: str
    selector_sig: str
    selector_arg: str
    desc_slot_sig: str
    desc_slot_arg: str
    local_init: str
    selector_init: str
    desc_slot_init: str


def build_phase_pointer_fragments(
    *,
    local_idx_tensors: Dict[str, Optional[torch.Tensor]],
    transport_position_tensors: Dict[str, Optional[torch.Tensor]],
    desc_slot_tensors: Dict[str, Optional[torch.Tensor]],
    init_indent: str,
    include_args: bool,
) -> PhasePointerFragments:
    """Build generated-source signature, argument, and default-init fragments."""
    local_sig_parts = []
    local_arg_parts = []
    selector_sig_parts = []
    selector_arg_parts = []
    desc_slot_sig_parts = []
    desc_slot_arg_parts = []
    local_init = ""
    selector_init = ""
    desc_slot_init = ""

    for phase in PHASE_NAMES:
        ptr_name = f"{phase}_local_idx_ptr"
        if local_idx_tensors[phase] is not None:
            local_sig_parts.append(ptr_name)
            local_arg_parts.append(ptr_name)
        else:
            local_init += f"{init_indent}{ptr_name} = Int64(0)\n"

        selector_ptr_name = f"{phase}_local_transport_positions_ptr"
        if transport_position_tensors[phase] is not None:
            selector_sig_parts.append(selector_ptr_name)
            selector_arg_parts.append(selector_ptr_name)
        else:
            selector_init += f"{init_indent}{selector_ptr_name} = Int64(0)\n"

        desc_slot_ptr_name = f"{phase}_local_desc_slots_ptr"
        if desc_slot_tensors[phase] is not None:
            desc_slot_sig_parts.append(desc_slot_ptr_name)
            desc_slot_arg_parts.append(desc_slot_ptr_name)
        else:
            desc_slot_init += f"{init_indent}{desc_slot_ptr_name} = Int64(0)\n"

    def _sig(parts) -> str:
        return f", {', '.join(parts)}" if parts else ""

    def _arg(parts) -> str:
        return f", {', '.join(parts)}" if include_args and parts else ""

    return PhasePointerFragments(
        local_sig=_sig(local_sig_parts),
        local_arg=_arg(local_arg_parts),
        selector_sig=_sig(selector_sig_parts),
        selector_arg=_arg(selector_arg_parts),
        desc_slot_sig=_sig(desc_slot_sig_parts),
        desc_slot_arg=_arg(desc_slot_arg_parts),
        local_init=local_init,
        selector_init=selector_init,
        desc_slot_init=desc_slot_init,
    )


def build_kernel_loop_source(
    kernel_loop_fn,
    *,
    tensor_sig: str,
    tma_sig: str,
    peer_tma_sig: str,
    has_communicate: bool,
    tracing: bool,
    local_idx_tensors: Dict[str, Optional[torch.Tensor]],
    transport_position_tensors: Dict[str, Optional[torch.Tensor]],
    desc_slot_tensors: Dict[str, Optional[torch.Tensor]],
    dispatch_extra_params: Dict[str, str],
) -> str:
    """Render the generated `_kernel_loop` source."""
    body = _extract_body(kernel_loop_fn)
    if dispatch_extra_params:
        def _rewrite_dispatch(match):
            fn_name = match.group(1)
            call_args = match.group(2).rstrip().rstrip(",")
            phase_name = fn_name.removeprefix("dispatch_")
            extra_params = dispatch_extra_params.get(phase_name, "")
            if not extra_params:
                return f"{fn_name}({call_args})"
            return f"{fn_name}({call_args}, {extra_params})"

        body = re.sub(
            r"(dispatch_(?:load|compute|store|communicate))\(([^)]*)\)",
            _rewrite_dispatch,
            body,
        )

    peer_signal_sig = ", peer_signal_ptr" if has_communicate else ""
    peer_signal_init = "" if has_communicate else "    peer_signal_ptr = Int64(0)\n"
    static_sig = ""
    trace_sig = ", trace_buffer_ptr" if tracing else ""
    trace_init = "" if tracing else "    trace_buffer_ptr = Int64(0)\n"
    phase_ptrs = build_phase_pointer_fragments(
        local_idx_tensors=local_idx_tensors,
        transport_position_tensors=transport_position_tensors,
        desc_slot_tensors=desc_slot_tensors,
        init_indent="    ",
        include_args=False,
    )
    return (
        "@cute.jit\n"
        "def _kernel_loop(instructions_ptr, barriers_ptr, op_configs_ptr,\n"
        f"                  op_meta_ptr{phase_ptrs.local_sig}{phase_ptrs.selector_sig}{phase_ptrs.desc_slot_sig}, signal_meta_ptr{peer_signal_sig}{static_sig},\n"
        "                  num_instructions, tidx, block_id, num_blocks,\n"
        f"                  smem_base{trace_sig}, wait_info_ptr, compute_wait_info_ptr{tensor_sig}{tma_sig}{peer_tma_sig}):\n"
        f"{phase_ptrs.local_init}"
        f"{phase_ptrs.selector_init}"
        f"{phase_ptrs.desc_slot_init}"
        f"{peer_signal_init}"
        f"{trace_init}"
        + textwrap.indent(body, "    ")
        + "\n"
    )


def build_persistent_kernel_source(
    *,
    num_sms: int,
    threads_per_block: int,
    smem_size: int,
    tensor_sig: str,
    kernel_tma_sig: str,
    tma_components: Dict[str, object],
    has_communicate: bool,
    tracing: bool,
    local_idx_tensors: Dict[str, Optional[torch.Tensor]],
    transport_position_tensors: Dict[str, Optional[torch.Tensor]],
    desc_slot_tensors: Dict[str, Optional[torch.Tensor]],
) -> str:
    """Render the generated `PersistentKernel` class source."""
    peer_signal_sig = ", peer_signal_ptr" if has_communicate else ""
    peer_signal_arg = ", peer_signal_ptr" if has_communicate else ""
    peer_signal_init = "" if has_communicate else "        peer_signal_ptr = Int64(0)\n"
    static_sig = ""
    static_arg = ""
    trace_sig = ", trace_buffer_ptr" if tracing else ""
    trace_arg = ", trace_buffer_ptr" if tracing else ""
    trace_init = "" if tracing else "        trace_buffer_ptr = Int64(0)\n"
    phase_ptrs = build_phase_pointer_fragments(
        local_idx_tensors=local_idx_tensors,
        transport_position_tensors=transport_position_tensors,
        desc_slot_tensors=desc_slot_tensors,
        init_indent="        ",
        include_args=True,
    )
    return (
        f"{tma_components['helper_definitions_code']}"
        "class PersistentKernel:\n"
        "    def __init__(self):\n"
        f"        self.num_sms = {num_sms}\n"
        f"        self.threads_per_block = {threads_per_block}\n"
        f"        self.smem_size = {smem_size}\n"
        "\n"
        "    @cute.kernel\n"
        f"    def init_tma_desc_pool(self, local_tma_desc_pool_ptr, peer_tma_desc_pool_ptr"
        f"{tma_components['desc_pool_init_sig']}):\n"
        f"{tma_components['desc_pool_init_body'] or '        return\n'}"
        "\n"
        "    @cute.jit\n"
        "    def __call__(self, instructions_ptr, barriers_ptr, op_configs_ptr,\n"
        f"                 op_meta_ptr{phase_ptrs.local_sig}{phase_ptrs.selector_sig}{phase_ptrs.desc_slot_sig}, signal_meta_ptr{peer_signal_sig}{static_sig},\n"
        f"                 wait_info_ptr, compute_wait_info_ptr, num_instructions{trace_sig}"
        f"{tensor_sig}{tma_components['desc_pool_sig']}{tma_components['tma_tensor_sig']}{tma_components['peer_tma_tensor_input_sig']}, desc_pool_init_needed, stream):\n"
        f"{phase_ptrs.local_init}"
        f"{phase_ptrs.selector_init}"
        f"{phase_ptrs.desc_slot_init}"
        f"{peer_signal_init}"
        f"{trace_init}"
        f"{tma_components['tma_creation_code']}"
        "        if desc_pool_init_needed:\n"
        "            self.init_tma_desc_pool(\n"
        "                local_tma_desc_pool_ptr,\n"
        "                peer_tma_desc_pool_ptr"
        f"{', ' if tma_components['desc_pool_init_params'] else ''}{tma_components['desc_pool_init_params']}\n"
        "            ).launch(grid=[1, 1, 1], block=[32, 1, 1], stream=stream)\n"
        "        if not desc_pool_init_needed:\n"
        "            self.kernel(instructions_ptr, barriers_ptr, op_configs_ptr,\n"
        f"                        op_meta_ptr{phase_ptrs.local_arg}{phase_ptrs.selector_arg}{phase_ptrs.desc_slot_arg}, signal_meta_ptr{peer_signal_arg}{static_arg},\n"
        f"                        wait_info_ptr, compute_wait_info_ptr,\n"
        f"                        num_instructions{trace_arg}{tensor_sig}{kernel_tma_sig}).launch(\n"
        "                grid=[self.num_sms, 1, 1],\n"
        "                block=[self.threads_per_block, 1, 1],\n"
        "                smem=self.smem_size,\n"
        "                stream=stream,\n"
        "                min_blocks_per_mp=1,\n"
        "            )\n"
        "\n"
        "    @cute.kernel\n"
        "    def kernel(self, instructions_ptr, barriers_ptr, op_configs_ptr,\n"
        f"               op_meta_ptr{phase_ptrs.local_sig}{phase_ptrs.selector_sig}{phase_ptrs.desc_slot_sig}, signal_meta_ptr{peer_signal_sig}{static_sig},\n"
        f"               wait_info_ptr, compute_wait_info_ptr, num_instructions{trace_sig}{tensor_sig}{kernel_tma_sig}):\n"
        f"{phase_ptrs.local_init}"
        f"{phase_ptrs.selector_init}"
        f"{phase_ptrs.desc_slot_init}"
        "        tidx = cute.arch.thread_idx()[0]\n"
        "        block_id = cute.arch.block_idx()[0]\n"
        "        num_blocks = cute.arch.grid_dim()[0]\n"
        "        smem_base = get_smem_base_ptr()\n"
        "        _kernel_loop(instructions_ptr, barriers_ptr, op_configs_ptr,\n"
        f"                     op_meta_ptr{phase_ptrs.local_arg}{phase_ptrs.selector_arg}{phase_ptrs.desc_slot_arg}, signal_meta_ptr{peer_signal_arg}{static_arg},\n"
        "                     num_instructions, tidx, block_id, num_blocks,\n"
        f"                     smem_base{trace_arg}, wait_info_ptr, compute_wait_info_ptr{tensor_sig}{kernel_tma_sig})\n"
    )
