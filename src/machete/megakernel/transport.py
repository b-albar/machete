# Copyright (c) 2025, Machete Authors
"""Runtime transport backend for megakernel TMA paths.

This module keeps the CuTe DSL authoring surface intact while moving the
runtime-descriptor transport logic out of ``noinline.py``. Ops still write the
same CuTe code:

- ``cute.nvgpu.cpasync.make_tiled_tma_atom(...)``
- ``cute.nvgpu.cpasync.tma_partition(...)``
- ``cute.copy(...)``

The difference is under the hood. The monkey patch installed by
``machete.megakernel.noinline`` redirects TMA atom construction to this module,
which returns standard CuTe ``CopyAtom`` objects carrying runtime descriptor
traits. Those traits reconstruct load/store/reduce exec atoms on demand using
backend-owned primitives.

The goal is not to replace all of CuTe transport yet. It is to define the
minimal backend layer that:

- owns runtime descriptor reconstruction
- provides explicit load/store/reduce primitives
- preserves the existing CuTe DSL code in kernels
"""

import re

from cutlass._mlir import ir
from cutlass._mlir.dialects import cute as cute_ir
from cutlass._mlir.dialects import cute_nvgpu as cn
from cutlass.base_dsl.dsl import extract_mlir_values, new_from_mlir_values
import cutlass.cute as cute
from cutlass.cute import atom, core
from cutlass.cute.atom import Trait
from cutlass.cute.nvgpu.cpasync.copy import (
    CopyBulkTensorTileG2SOp,
    CopyBulkTensorTileG2SMulticastOp,
    CopyBulkTensorTileS2GOp,
    CopyReduceBulkTensorTileS2GOp,
)
from cutlass.cute.typing import NumericMeta
from cutlass import Int64, Uint8
from cutlass.cute.typing import AddressSpace
from cutlass.utils.tensormap_manager import TensorMapManager, TensorMapUpdateMode


_TMA_DESC_BYTES = 128
_TMA_DESC_MANAGER = TensorMapManager(
    TensorMapUpdateMode.GMEM,
    bytes_per_tensormap=_TMA_DESC_BYTES,
)


def runtime_desc_ptr_type():
    """Return the generic typed pointer used for runtime TMA descriptors."""
    return cute_ir.PtrType.get(cn.TmaDescriptorTiledType.get(), 0, 64)


def runtime_desc_gmem_ptr_from_pool(pool_ptr, slot, *, loc=None, ip=None):
    """Return a gmem typed tensormap pointer for one descriptor pool slot."""
    byte_ptr = cute.make_ptr(
        Uint8,
        pool_ptr,
        AddressSpace.gmem,
        assumed_align=_TMA_DESC_BYTES,
        loc=loc,
        ip=ip,
    )
    return _TMA_DESC_MANAGER.get_tensormap_ptr(
        byte_ptr + Int64(slot) * Int64(_TMA_DESC_BYTES),
        address_space=AddressSpace.gmem,
        loc=loc,
        ip=ip,
    )


def runtime_desc_ptr_from_pool(pool_ptr, slot, *, loc=None, ip=None):
    """Return a generic typed tensormap pointer for one descriptor pool slot."""
    byte_ptr = cute.make_ptr(
        Uint8,
        pool_ptr,
        AddressSpace.gmem,
        assumed_align=_TMA_DESC_BYTES,
        loc=loc,
        ip=ip,
    )
    desc_ptr = _TMA_DESC_MANAGER.get_tensormap_ptr(
        byte_ptr + Int64(slot) * Int64(_TMA_DESC_BYTES),
        address_space=AddressSpace.generic,
        loc=loc,
        ip=ip,
    )
    return desc_ptr


def init_runtime_desc_pool_slot(copy_atom, pool_ptr, slot, *, warp_id=0, loc=None, ip=None):
    """Initialize one descriptor pool slot from a copy atom."""
    _TMA_DESC_MANAGER.init_tensormap_from_atom(
        copy_atom,
        runtime_desc_gmem_ptr_from_pool(pool_ptr, slot, loc=loc, ip=ip),
        warp_id=warp_id,
        loc=loc,
        ip=ip,
    )


def fence_runtime_desc_pool(*, loc=None, ip=None):
    """Fence descriptor-pool initialization before runtime use."""
    _TMA_DESC_MANAGER.fence_tensormap_initialization(loc=loc, ip=ip)


def copy_runtime_desc_to_pool(desc_ptr, pool_ptr, slot, *, loc=None, ip=None):
    """Copy one runtime descriptor into the descriptor pool as raw words."""
    src_ptr_val = extract_mlir_values(desc_ptr)[0]
    src_i64 = Int64(cute_ir.ptrtoint(Int64.mlir_type, src_ptr_val, loc=loc, ip=ip))
    src_words = cute.make_ptr(
        Int64,
        src_i64,
        AddressSpace.generic,
        assumed_align=8,
        loc=loc,
        ip=ip,
    )
    dst_words = cute.make_ptr(
        Int64,
        pool_ptr + Int64(slot) * Int64(_TMA_DESC_BYTES),
        AddressSpace.gmem,
        assumed_align=8,
        loc=loc,
        ip=ip,
    )
    for i in range(_TMA_DESC_BYTES // 8):
        src_word_ptr = extract_mlir_values(src_words + Int64(i))[0]
        dst_word_ptr = extract_mlir_values(dst_words + Int64(i))[0]
        cute_ir.ptr_store(
            cute_ir.ptr_load(src_word_ptr, loc=loc, ip=ip),
            dst_word_ptr,
            loc=loc,
            ip=ip,
        )


def make_runtime_tma_gmem(
    direction,
    gmem_tensor,
    smem_layout,
    cta_tiler,
    num_multicast=1,
    *,
    internal_type=None,
    loc=None,
    ip=None,
):
    """Rebuild only the non-exec GMEM-side TMA partition tensor.

    This intentionally does not create or touch a TMA descriptor. It is used
    by noinline wrappers to recover the GMEM partition object from the runtime
    tensor binding while the executable descriptor still comes from the
    runtime descriptor pool.
    """
    cta_v_map = core.composition(
        core.make_identity_layout(gmem_tensor.shape, loc=loc, ip=ip),
        cta_tiler,
        loc=loc,
        ip=ip,
    )

    if isinstance(smem_layout, core._ComposedLayout):
        smem_layout = smem_layout.value

    op = (
        CopyBulkTensorTileG2SOp()
        if direction == "g2s"
        else CopyBulkTensorTileS2GOp()
        if direction == "s2g"
        else CopyReduceBulkTensorTileS2GOp()
    )
    op.smem_layout = (
        smem_layout.value
        if isinstance(smem_layout, core._ComposedLayout)
        else smem_layout
    )

    tma_format = None
    if internal_type is not None:
        if not isinstance(internal_type, NumericMeta):
            raise TypeError(f"internal_type must be a Numeric, but got {internal_type}")
        use_unpack = (
            internal_type.width == 8
            and isinstance(gmem_tensor.element_type, NumericMeta)
            and gmem_tensor.element_type.width < 8
        )
        internal_mlir_type = (
            gmem_tensor.element_type.mlir_type
            if use_unpack
            else internal_type.mlir_type
        )
        tma_format = cn.TmaDataFormat(
            cn.get_default_tma_format(internal_mlir_type, use_unpack)
        )

    if direction == "g2s":
        _atom_res, gmem_res = cn.atom_make_non_exec_tiled_tma_load(
            gmem_tensor.value,
            smem_layout,
            cta_v_map,
            op._to_ir(),
            num_multicast=num_multicast,
            tma_format=tma_format,
            loc=loc,
            ip=ip,
        )
        return gmem_res

    if direction == "s2g":
        _atom_res, gmem_res = cn.atom_make_non_exec_tiled_tma_store(
            gmem_tensor.value,
            smem_layout,
            cta_v_map,
            tma_format=tma_format,
            loc=loc,
            ip=ip,
        )
        return gmem_res

    if direction == "s2g_reduce":
        _atom_res, gmem_res = cn.atom_make_non_exec_tiled_tma_reduce(
            gmem_tensor.value,
            smem_layout,
            cta_v_map,
            op._to_ir(),
            tma_format=tma_format,
            loc=loc,
            ip=ip,
        )
        return gmem_res

    raise ValueError(f"unsupported TMA direction: {direction}")


def _parse_exec_tma_type(nonexec_type, field_namespace: str):
    """Build the exec-TMA type corresponding to a non-exec TMA atom type."""
    type_str = str(nonexec_type)

    if field_namespace == "tmaload":
        match = re.match(
            r'!cute_nvgpu\.atom\.non_exec_tiled_tma_load<([^,]+),\s*([^,]+),\s*copy_bits = ([0-9]+),\s*tma_gbasis = <"([^"]+)">,\s*tma_format = ([^>]+)>',
            type_str,
        )
        if not match:
            raise ValueError(f"unsupported non-exec TMA load type: {type_str}")
        arch, dtype, copy_bits, gbasis, _fmt = match.groups()
        num_cta = 2 if "2sm" in arch else 1
        return ir.Type.parse(
            f'!cute_nvgpu.atom.tma_load<{dtype}, copy_bits = {copy_bits}, mode = tiled, '
            f'num_cta = {num_cta}, g_stride = <"()"> tma_gbasis = <"{gbasis}">>'
        )

    if field_namespace == "tmastore":
        match = re.match(
            r'!cute_nvgpu\.atom\.non_exec_tiled_tma_store<([^,]+),\s*copy_bits = ([0-9]+),\s*tma_gbasis = <"([^"]+)">,\s*tma_format = ([^>]+)>',
            type_str,
        )
        if not match:
            raise ValueError(f"unsupported non-exec TMA store type: {type_str}")
        dtype, copy_bits, gbasis, _fmt = match.groups()
        return ir.Type.parse(
            f'!cute_nvgpu.atom.tma_store<{dtype}, copy_bits = {copy_bits}, mode = tiled, '
            f'g_stride = <"()"> tma_gbasis = <"{gbasis}">>'
        )

    if field_namespace == "tmareduce":
        match = re.match(
            r'!cute_nvgpu\.atom\.non_exec_tiled_tma_reduce<([^,]+),\s*([^,]+),\s*copy_bits = ([0-9]+),\s*tma_gbasis = <"([^"]+)">,\s*tma_format = ([^>,]+)(?:,\s*op = ([^>]+))?>',
            type_str,
        )
        if not match:
            raise ValueError(f"unsupported non-exec TMA reduce type: {type_str}")
        kind, dtype, copy_bits, gbasis, _fmt, red = match.groups()
        if red is not None:
            kind = red
        return ir.Type.parse(
            f'!cute_nvgpu.atom.tma_reduce<{dtype}, copy_bits = {copy_bits}, mode = tiled, '
            f'kind = {kind}, g_stride = <"()"> tma_gbasis = <"{gbasis}">>'
        )

    raise ValueError(f"unsupported TMA field namespace: {field_namespace}")


def _make_tma_g_stride(*, loc=None, ip=None):
    return cute_ir.make_stride(
        ir.Type.parse('!cute.stride<"()">'),
        [],
        loc=loc,
        ip=ip,
    )



def make_runtime_tma_load(exec_type, desc_ptr, *, tma_bar_ptr, cache_policy=None, loc=None, ip=None):
    """Backend-owned runtime TMA load primitive."""
    return cn.atom_make_tma_load(
        exec_type,
        desc_ptr,
        tma_bar_ptr.value,
        _make_tma_g_stride(loc=loc, ip=ip),
        cache_policy=cache_policy.value if cache_policy is not None else None,
        loc=loc,
        ip=ip,
    )


def make_runtime_tma_store(exec_type, desc_ptr, *, cache_policy=None, loc=None, ip=None):
    """Backend-owned runtime TMA store primitive."""
    return cn.atom_make_tma_store(
        exec_type,
        desc_ptr,
        _make_tma_g_stride(loc=loc, ip=ip),
        cache_policy=cache_policy.value if cache_policy is not None else None,
        loc=loc,
        ip=ip,
    )


def make_runtime_tma_reduce(exec_type, desc_ptr, *, cache_policy=None, loc=None, ip=None):
    """Backend-owned runtime TMA reduce-store primitive."""
    return cn.atom_make_tma_reduce(
        exec_type,
        desc_ptr,
        _make_tma_g_stride(loc=loc, ip=ip),
        cache_policy=cache_policy.value if cache_policy is not None else None,
        loc=loc,
        ip=ip,
    )


class RuntimeDescTMATrait(Trait):
    """TMA trait carrying a non-exec atom plus runtime descriptor pointer."""

    def __init__(self, value, desc_ptr, *, field_namespace: str, supports_mbar: bool):
        self.value = value
        self.desc_ptr = desc_ptr
        self.field_namespace = field_namespace
        self.supports_mbar = supports_mbar

    def __extract_mlir_values__(self):
        return [self.value] + extract_mlir_values(self.desc_ptr)

    def __new_from_mlir_values__(self, values):
        desc_template_vals = extract_mlir_values(self.desc_ptr)
        desc_vals = values[1:1 + len(desc_template_vals)]
        desc_ptr = new_from_mlir_values(self.desc_ptr, desc_vals)
        return self.__class__(
            values[0],
            desc_ptr,
            field_namespace=self.field_namespace,
            supports_mbar=self.supports_mbar,
        )

    def unpack(self, *, tma_bar_ptr=None, tma_desc_ptr=None, cache_policy=None, loc=None, ip=None, **kwargs):
        # Allow callers to override the bound descriptor pointer. This keeps a
        # structurally shared non-exec atom reusable across bindings while the
        # actual runtime transport selects the concrete descriptor separately.
        desc_source = tma_desc_ptr if tma_desc_ptr is not None else self.desc_ptr
        desc_value = extract_mlir_values(desc_source)[0]
        exec_type = _parse_exec_tma_type(self.value.type, self.field_namespace)

        if self.field_namespace == "tmaload":
            return make_runtime_tma_load(
                exec_type,
                desc_value,
                tma_bar_ptr=tma_bar_ptr,
                cache_policy=cache_policy,
                loc=loc,
                ip=ip,
            )
        if self.field_namespace == "tmastore":
            return make_runtime_tma_store(
                exec_type,
                desc_value,
                cache_policy=cache_policy,
                loc=loc,
                ip=ip,
            )
        if self.field_namespace == "tmareduce":
            return make_runtime_tma_reduce(
                exec_type,
                desc_value,
                cache_policy=cache_policy,
                loc=loc,
                ip=ip,
            )
        raise ValueError(f"unsupported TMA field namespace: {self.field_namespace}")


def make_runtime_desc_tma_atom(
    op,
    gmem_tensor,
    smem_layout,
    cta_tiler,
    num_multicast=1,
    *,
    internal_type=None,
    loc=None,
    ip=None,
):
    """Create a CuTe CopyAtom backed by runtime descriptor transport.

    This preserves the normal CuTe return shape:
    ``(CopyAtom, gmem_partition_tensor)``
    so existing ops keep using ``tma_partition`` and ``cute.copy`` unchanged.
    """
    cta_v_map = core.composition(
        core.make_identity_layout(gmem_tensor.shape, loc=loc, ip=ip),
        cta_tiler,
        loc=loc,
        ip=ip,
    )

    if isinstance(smem_layout, core._ComposedLayout):
        smem_layout = smem_layout.value

    op.smem_layout = (
        smem_layout.value
        if isinstance(smem_layout, core._ComposedLayout)
        else smem_layout
    )

    tma_format = None
    if internal_type is not None:
        if not isinstance(internal_type, NumericMeta):
            raise TypeError(f"internal_type must be a Numeric, but got {internal_type}")
        use_unpack = (
            internal_type.width == 8
            and isinstance(gmem_tensor.element_type, NumericMeta)
            and gmem_tensor.element_type.width < 8
        )
        internal_mlir_type = (
            gmem_tensor.element_type.mlir_type
            if use_unpack
            else internal_type.mlir_type
        )
        tma_format = cn.TmaDataFormat(
            cn.get_default_tma_format(internal_mlir_type, use_unpack)
        )

    desc_ptr_type = runtime_desc_ptr_type()

    if isinstance(op, CopyBulkTensorTileG2SOp):
        if num_multicast != 1:
            raise ValueError(
                f"expects num_multicast to be 1 for non multicast G2S copies, got {num_multicast}"
            )
        atom_res, gmem_res = cn.atom_make_non_exec_tiled_tma_load(
            gmem_tensor.value,
            smem_layout,
            cta_v_map,
            op._to_ir(),
            num_multicast=num_multicast,
            tma_format=tma_format,
            loc=loc,
            ip=ip,
        )
        desc_ptr = cn.make_tma_desc_tiled(
            desc_ptr_type,
            gmem_tensor.value,
            smem_layout,
            cta_v_map,
            cn.TmaLoadMode.tiled,
            num_multicast=num_multicast,
            tma_format=tma_format,
            loc=loc,
            ip=ip,
        )
        return (
            atom.CopyAtom(
                op,
                RuntimeDescTMATrait(
                    atom_res,
                    desc_ptr,
                    field_namespace="tmaload",
                    supports_mbar=True,
                ),
            ),
            gmem_res,
        )

    if isinstance(op, CopyBulkTensorTileG2SMulticastOp):
        if num_multicast < 1:
            raise ValueError(
                f"expects num_multicast to be >= 1 for multicast G2S copies, got {num_multicast}"
            )
        atom_res, gmem_res = cn.atom_make_non_exec_tiled_tma_load(
            gmem_tensor.value,
            smem_layout,
            cta_v_map,
            op._to_ir(),
            num_multicast=num_multicast,
            tma_format=tma_format,
            loc=loc,
            ip=ip,
        )
        desc_ptr = cn.make_tma_desc_tiled(
            desc_ptr_type,
            gmem_tensor.value,
            smem_layout,
            cta_v_map,
            cn.TmaLoadMode.tiled,
            num_multicast=num_multicast,
            tma_format=tma_format,
            loc=loc,
            ip=ip,
        )
        return (
            atom.CopyAtom(
                op,
                RuntimeDescTMATrait(
                    atom_res,
                    desc_ptr,
                    field_namespace="tmaload",
                    supports_mbar=True,
                ),
            ),
            gmem_res,
        )

    if isinstance(op, CopyBulkTensorTileS2GOp):
        atom_res, gmem_res = cn.atom_make_non_exec_tiled_tma_store(
            gmem_tensor.value,
            smem_layout,
            cta_v_map,
            tma_format=tma_format,
            loc=loc,
            ip=ip,
        )
        desc_ptr = cn.make_tma_desc_tiled(
            desc_ptr_type,
            gmem_tensor.value,
            smem_layout,
            cta_v_map,
            cn.TmaStoreMode.tiled,
            tma_format=tma_format,
            loc=loc,
            ip=ip,
        )
        return (
            atom.CopyAtom(
                op,
                RuntimeDescTMATrait(
                    atom_res,
                    desc_ptr,
                    field_namespace="tmastore",
                    supports_mbar=False,
                ),
            ),
            gmem_res,
        )

    if isinstance(op, CopyReduceBulkTensorTileS2GOp):
        atom_res, gmem_res = cn.atom_make_non_exec_tiled_tma_reduce(
            gmem_tensor.value,
            smem_layout,
            cta_v_map,
            op._to_ir(),
            tma_format=tma_format,
            loc=loc,
            ip=ip,
        )
        desc_ptr = cn.make_tma_desc_tiled(
            desc_ptr_type,
            gmem_tensor.value,
            smem_layout,
            cta_v_map,
            cn.TmaStoreMode.tiled,
            tma_format=tma_format,
            loc=loc,
            ip=ip,
        )
        return (
            atom.CopyAtom(
                op,
                RuntimeDescTMATrait(
                    atom_res,
                    desc_ptr,
                    field_namespace="tmareduce",
                    supports_mbar=False,
                ),
            ),
            gmem_res,
        )

    raise TypeError(f"unsupported runtime transport op: {type(op).__name__}")
