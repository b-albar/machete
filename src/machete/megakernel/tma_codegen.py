# Copyright (c) 2026, Machete Authors
"""TMA source-generation helpers for the persistent megakernel."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple


def collect_tma_tensor_names(tma_registry) -> List[str]:
    """Return unique parameter names for local static-layout TMA tensors."""
    tensor_names: List[str] = []
    seen = set()
    for desc in tma_registry.descriptors:
        ndim = len(desc.tile_shape)
        tensor_name = f"tma_{desc.tensor_canonical}_{ndim}d"
        if tensor_name in seen:
            continue
        seen.add(tensor_name)
        tensor_names.append(tensor_name)
    return tensor_names


def collect_peer_tma_tensor_names(peer_tma_registry) -> List[str]:
    """Return unique parameter names for peer static-layout TMA tensors."""
    tensor_names: List[str] = []
    seen = set()
    for desc in peer_tma_registry.descriptors:
        tensor_name = f"ptma_{desc.tensor_canonical}_p{desc.peer_idx}"
        if tensor_name in seen:
            continue
        seen.add(tensor_name)
        tensor_names.append(tensor_name)
    return tensor_names


def render_tma_creation_expr(desc, tensor_source: str) -> str:
    """Render the `make_tiled_tma_atom` expression for one descriptor."""
    direction = getattr(desc, "direction", "s2g")
    if direction == "g2s":
        copy_op = "CopyBulkTensorTileG2SOp()"
    elif direction == "s2g":
        copy_op = "CopyBulkTensorTileS2GOp()"
    elif direction == "s2g_reduce":
        copy_op = "CopyReduceBulkTensorTileS2GOp(reduction_kind=ReductionOp.ADD)"
    else:
        raise ValueError(f"Unknown TMA direction: {direction}")

    shape_str = ", ".join(str(s) for s in desc.tile_shape)
    smem_layout_code = desc.smem_layout_src or f"cute.make_layout(({shape_str},))"
    return (
        "cute.nvgpu.cpasync.make_tiled_tma_atom(\n"
        f"        {copy_op},\n"
        f"        {tensor_source},\n"
        f"        {smem_layout_code},\n"
        f"        ({shape_str},),\n"
        "        num_multicast=1,\n"
        "    )"
    )


def append_tma_descriptor_code(
    tma_creation_lines,
    desc_pool_init_specs,
    desc,
    tensor_source: str,
    helper_name: str | None = None,
    *,
    create_atom_binding: bool,
    create_gmem_binding: bool,
    pool_name: str,
    pool_slot: int,
) -> None:
    """Append source that constructs one TMA descriptor pair."""
    concrete_atom_name = desc.canonical_atom if create_atom_binding else f"_{desc.canonical_desc}_atom"
    concrete_gmem_name = desc.canonical_gmem if create_gmem_binding else f"_{desc.canonical_desc}_gmem"
    concrete_desc_name = desc.canonical_desc
    if helper_name is not None:
        tma_creation_lines.append(
            f"        {concrete_atom_name}, {concrete_gmem_name}, {concrete_desc_name} = "
            f"{helper_name}({tensor_source})"
        )
    else:
        tma_creation_lines.append(
            f"        {concrete_atom_name}, {concrete_gmem_name} = "
            f"{render_tma_creation_expr(desc, tensor_source)}"
        )
        tma_creation_lines.append(f"        {concrete_desc_name} = {concrete_atom_name}._trait.desc_ptr")
    desc_pool_init_specs.append((concrete_desc_name, pool_name, pool_slot))
    if create_atom_binding and concrete_atom_name != desc.canonical_atom:
        tma_creation_lines.append(f"        {desc.canonical_atom} = {concrete_atom_name}")
    if create_gmem_binding and concrete_gmem_name != desc.canonical_gmem:
        tma_creation_lines.append(f"        {desc.canonical_gmem} = {concrete_gmem_name}")


def build_tma_kernel_components(
    tma_registry,
    peer_tma_registry,
    *,
    signature_suffix: Callable[[List[str]], str],
) -> Dict[str, Any]:
    """Assemble the TMA-specific signature fragments and descriptor setup code."""
    tma_tensor_names = collect_tma_tensor_names(tma_registry)
    peer_tma_tensor_names = collect_peer_tma_tensor_names(peer_tma_registry)
    tma_creation_lines: List[str] = []
    desc_pool_init_specs: List[Tuple[str, str, int]] = []
    helper_sources: List[str] = []
    helper_name_by_key: Dict[Tuple[Any, ...], str] = {}
    seen_atoms = set()
    seen_gmems = set()

    def _helper_name_for_desc(desc) -> str:
        key = (
            getattr(desc, "direction", "s2g"),
            tuple(desc.tile_shape),
            desc.smem_layout_src or "",
            tuple(getattr(desc, "dim_perm", ()) or ()),
        )
        helper_name = helper_name_by_key.get(key)
        if helper_name is not None:
            return helper_name
        helper_name = f"_make_tma_helper_{len(helper_name_by_key)}"
        helper_name_by_key[key] = helper_name
        direction = getattr(desc, "direction", "s2g")
        if direction == "g2s":
            copy_op = "CopyBulkTensorTileG2SOp()"
        elif direction == "s2g":
            copy_op = "CopyBulkTensorTileS2GOp()"
        elif direction == "s2g_reduce":
            copy_op = "CopyReduceBulkTensorTileS2GOp(reduction_kind=ReductionOp.ADD)"
        else:
            raise ValueError(f"Unknown TMA direction: {direction}")
        shape_str = ", ".join(str(s) for s in desc.tile_shape)
        smem_layout_code = desc.smem_layout_src or f"cute.make_layout(({shape_str},))"
        helper_sources.append(
            "@cute.jit\n"
            f"def {helper_name}(tensor):\n"
            "    atom, gmem = make_runtime_desc_tma_atom(\n"
            f"        {copy_op},\n"
            "        tensor,\n"
            f"        {smem_layout_code},\n"
            f"        ({shape_str},),\n"
            "        num_multicast=1,\n"
            "    )\n"
            "    return atom, gmem, atom._trait.desc_ptr\n"
        )
        return helper_name

    for slot, desc in enumerate(tma_registry.descriptors):
        ndim = len(desc.tile_shape)
        append_tma_descriptor_code(
            tma_creation_lines,
            desc_pool_init_specs,
            desc,
            f"tma_{desc.tensor_canonical}_{ndim}d",
            helper_name=_helper_name_for_desc(desc),
            create_atom_binding=desc.canonical_atom not in seen_atoms,
            create_gmem_binding=desc.canonical_gmem not in seen_gmems,
            pool_name="local_tma_desc_pool_ptr",
            pool_slot=slot,
        )
        seen_atoms.add(desc.canonical_atom)
        seen_gmems.add(desc.canonical_gmem)

    for slot, desc in enumerate(peer_tma_registry.descriptors):
        append_tma_descriptor_code(
            tma_creation_lines,
            desc_pool_init_specs,
            desc,
            f"ptma_{desc.tensor_canonical}_p{desc.peer_idx}",
            helper_name=_helper_name_for_desc(desc),
            create_atom_binding=desc.canonical_atom not in seen_atoms,
            create_gmem_binding=desc.canonical_gmem not in seen_gmems,
            pool_name="peer_tma_desc_pool_ptr",
            pool_slot=slot,
        )
        seen_atoms.add(desc.canonical_atom)
        seen_gmems.add(desc.canonical_gmem)

    tma_creation_code = "\n".join(tma_creation_lines)
    if tma_creation_code:
        tma_creation_code = "\n" + tma_creation_code + "\n"

    init_param_names: List[str] = []
    for desc_name, _pool_name, _slot in desc_pool_init_specs:
        if desc_name not in init_param_names:
            init_param_names.append(desc_name)
    desc_pool_init_params = ", ".join(init_param_names)
    desc_pool_init_sig = signature_suffix(init_param_names)
    desc_pool_init_body_lines = []
    for desc_name, pool_name, slot in desc_pool_init_specs:
        desc_pool_init_body_lines.append(
            f"        copy_runtime_desc_to_pool({desc_name}, {pool_name}, Int32({slot}))"
        )
    if desc_pool_init_body_lines:
        desc_pool_init_body_lines.append("        fence_runtime_desc_pool()")
    desc_pool_init_body = "\n".join(desc_pool_init_body_lines)

    return {
        "desc_pool_sig": ", local_tma_desc_pool_ptr, peer_tma_desc_pool_ptr",
        "tma_tensor_sig": signature_suffix(tma_tensor_names),
        "peer_tma_tensor_input_sig": signature_suffix(peer_tma_tensor_names),
        "helper_definitions_code": "\n".join(helper_sources) + ("\n" if helper_sources else ""),
        "tma_creation_code": tma_creation_code,
        "desc_pool_init_sig": desc_pool_init_sig,
        "desc_pool_init_params": desc_pool_init_params,
        "desc_pool_init_body": desc_pool_init_body,
    }
