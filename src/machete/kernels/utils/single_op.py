# Copyright (c) 2025, Machete Authors
"""Lightweight single-op kernel for benchmarking without megakernel overhead.

Launches one block per tile with load/store/compute warp specialization.
No instruction stream, no dependency checking, no ring buffer.

Usage:
    ops = MyOp.schedule(x=x, y=y, tile_sizes={"M": 4})
    kernel = SingleOpKernel(ops)
    kernel.run()

    # For benchmarking:
    spec = kernel.bench_spec()
"""

import linecache
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple

import torch

import cutlass.cute as cute
from cutlass import Int32, Int64

from machete.megakernel.ops import MAX_TILE_DIMS, ScheduledOp, build_op_config
from machete.megakernel.registries import (
    TensorRegistry,
    TMARegistry,
    validate_op_compatibility,
)
from machete.megakernel.compile import compile_phase, _extract_body
from machete.megakernel.interpreter import (
    mbarrier_init,
    mbarrier_init_fence,
    mbarrier_arrive,
    mbarrier_wait,
    named_barrier_sync,
    get_smem_base_ptr,
    ld_global_i64,
)
from machete.megakernel.megakernel import MegakernelConfig, NUM_DMA_WARPS


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class SingleOpKernel:
    """Lightweight kernel: one block per tile, no megakernel overhead.

    Keeps warp specialization (load/store/compute) but eliminates:
    - Instruction stream and IQ scanning
    - Dependency checking and barrier DAG
    - Ring buffer and paged memory
    - Persistent kernel (nanosleep, sentinel)
    """

    _compiled_kernel_cache: Dict[Tuple, Any] = {}

    def __init__(
        self,
        ops: List[ScheduledOp],
        config: Optional[MegakernelConfig] = None,
        device: str = "cuda",
    ):
        self.ops = ops
        self.device = device

        # Auto-detect config from op if not provided
        if config is None:
            op_cls = ops[0].op_cls
            if hasattr(op_cls, "kernel_config"):
                config = op_cls.kernel_config(ops)
            else:
                config = MegakernelConfig()
        self.config = config

        self._total_tiles = sum(op.total_tiles for op in ops)

        # Smem: [work_mbar(8B) compute_done_mbar(8B) pad→128B] [page]
        self._work_mbar_offset = 0
        self._compute_done_offset = 8
        self._page_offset = 128
        self._aligned_page_size = _align_up(config.page_size, 128)
        self._smem_size = self._page_offset + self._aligned_page_size

        # Registries (same as Megakernel)
        self._tensor_registry = TensorRegistry.from_ops(ops)
        validate_op_compatibility(ops, self._tensor_registry)
        self._tma_registry = TMARegistry.from_ops(ops, self._tensor_registry)

        self._op_configs_tensor = None
        self._cute_tensors = None
        self._tma_cute_tensors = None
        self._compiled_kernel = None

    @property
    def total_tiles(self) -> int:
        return self._total_tiles

    @property
    def smem_size(self) -> int:
        return self._smem_size

    # =========================================================================
    # Tensor / TMA Preparation (reuses Megakernel patterns)
    # =========================================================================

    def _prepare_op_configs(self):
        if self._op_configs_tensor is not None:
            return
        ptrs = [
            op.config_data.data_ptr() if op.config_data is not None else 0
            for op in self.ops
        ]
        self._op_configs_tensor = torch.tensor(
            ptrs, dtype=torch.int64, device=self.device
        )

    def _prepare_cute_tensors(self):
        if self._cute_tensors is not None:
            return
        self._cute_tensors = []
        for _, tensor, _ in self._tensor_registry.tensors:
            t = tensor.detach().contiguous()
            strides = t.stride()
            unit_dims = [i for i in range(t.ndim) if strides[i] == 1]
            if len(unit_dims) > 1:
                from cutlass.cute.runtime import from_dlpack

                self._cute_tensors.append(
                    from_dlpack(t, assumed_align=16).mark_layout_dynamic(
                        leading_dim=t.ndim - 1
                    )
                )
            else:
                self._cute_tensors.append(t)

    def _prepare_tma_tensors(self):
        if self._tma_cute_tensors is not None:
            return
        if not self._tma_registry.has_tma:
            self._tma_cute_tensors = []
            return
        from cutlass.cute.runtime import from_dlpack

        self._tma_cute_tensors = []
        seen = set()
        for desc in self._tma_registry.descriptors:
            if desc.tensor_canonical in seen:
                continue
            seen.add(desc.tensor_canonical)
            for cn, tensor, _ in self._tensor_registry.tensors:
                if cn == desc.tensor_canonical:
                    t = tensor.detach()
                    if t.ndim >= 2:
                        t = t.permute(*reversed(range(t.ndim)))
                    self._tma_cute_tensors.append(
                        (cn, from_dlpack(t, assumed_align=16))
                    )
                    break

    # =========================================================================
    # JIT Function Builders
    # =========================================================================

    @staticmethod
    def _exec_jit(src, fn_name, cm, extra_globals=None):
        """Compile and exec a JIT function source string."""
        globs = {"cute": cute, "Int32": Int32, "Int64": Int64}
        if extra_globals:
            globs.update(extra_globals)
        cm._compile_counter += 1
        fname = f"<{fn_name}>_{cm._compile_counter}"
        linecache.cache[fname] = (len(src), None, src.splitlines(True), fname)
        cm._linecache_entries.append(fname)
        exec(compile(src, fname, "exec"), globs)
        return globs[fn_name]

    def _build_decompose_fn(self):
        """Build JIT fn: block_id → (op_idx, tile_0..4)."""
        import machete.megakernel.compile as cm

        lines = []
        cumulative = 0
        for i, op in enumerate(self.ops):
            total = op.total_tiles
            kw = "if" if i == 0 else "elif"
            lines.append(f"    {kw} block_id < Int32({cumulative + total}):")
            lines.append(f"        op_idx = Int32({i})")
            lines.append(f"        _local = block_id - Int32({cumulative})")
            tc = op.tile_counts
            for d in range(len(tc)):
                if tc[d] > 1:
                    divisor = 1
                    for k in range(d):
                        divisor *= tc[k]
                    if divisor == 1:
                        lines.append(
                            f"        tile_{d} = _local % Int32({tc[d]})"
                        )
                    else:
                        lines.append(
                            f"        tile_{d} = (_local // Int32({divisor})) % Int32({tc[d]})"
                        )
            cumulative += total

        body = "\n".join(lines) if lines else "    pass"
        tile_inits = "".join(
            f"    tile_{i} = Int32(0)\n" for i in range(MAX_TILE_DIMS)
        )
        src = (
            "@cute.jit\n"
            "def decompose_block_id(block_id):\n"
            "    op_idx = Int32(0)\n"
            f"{tile_inits}"
            f"{body}\n"
            "    return op_idx, tile_0, tile_1, tile_2, tile_3, tile_4\n"
        )
        return self._exec_jit(src, "decompose_block_id", cm)

    def _build_inner_iters_fn(self, inner_iters_list):
        """Build JIT fn: op_idx → inner_iters count."""
        import machete.megakernel.compile as cm

        lines = []
        for idx, n in enumerate(inner_iters_list):
            kw = "if" if idx == 0 else "elif"
            lines.append(
                f"    {kw} op_idx == Int32({idx}):\n        _r = Int32({n})"
            )
        body = "\n".join(lines) if lines else "    _r = Int32(1)"
        src = (
            "@cute.jit\n"
            "def _get_inner_iters(op_idx) -> Int32:\n"
            f"    _r = Int32(1)\n{body}\n    return _r\n"
        )
        return self._exec_jit(src, "_get_inner_iters", cm)

    def _build_dispatch_fns(self):
        """Build dispatch_load/compute/store and return inner_iters_list."""
        ops = self.ops
        registry = self._tensor_registry
        tma_registry = self._tma_registry
        all_canonical = registry.canonical_names
        all_tma_canonical = tma_registry.all_canonical_names

        threads_per_block = self.config.threads_per_block
        num_compute_threads = threads_per_block - NUM_DMA_WARPS * 32

        load_fns, compute_fns, store_fns = [], [], []
        inner_iters_list = []
        op_tensor_args = []
        op_tma_args = {"load": [], "compute": [], "store": []}

        for i, op in enumerate(ops):
            ta = registry.get_op_tensor_args(i, op.op_cls)
            op_tensor_args.append(ta)
            for phase in ("load", "compute", "store"):
                op_tma_args[phase].append(
                    tma_registry.get_op_tma_args(i, phase)
                )

            config = build_op_config(
                op, kernel_config={"threads_per_row": num_compute_threads}
            )
            instance = op.op_cls(**config)
            inner_iters_list.append(getattr(instance, "inner_iters", 1))

            for phase, fns in [
                ("load", load_fns),
                ("compute", compute_fns),
                ("store", store_fns),
            ]:
                mapping = tma_registry.op_mappings.get((i, phase), {})
                fns.append(
                    compile_phase(
                        instance,
                        phase,
                        tensor_param_names=ta,
                        tma_param_names=op_tma_args[phase][-1],
                        tma_local_mapping=mapping,
                    )
                )

        dispatch_load = self._build_exec_dispatch(
            "load", load_fns, op_tensor_args,
            all_canonical, op_tma_args["load"], all_tma_canonical,
        )
        dispatch_compute = self._build_exec_dispatch(
            "compute", compute_fns, op_tensor_args,
            all_canonical, op_tma_args["compute"], all_tma_canonical,
        )
        dispatch_store = self._build_exec_dispatch(
            "store", store_fns, op_tensor_args,
            all_canonical, op_tma_args["store"], all_tma_canonical,
        )
        return dispatch_load, dispatch_compute, dispatch_store, inner_iters_list

    def _build_exec_dispatch(
        self, phase_name, phase_fns, op_tensor_args,
        all_canonical, op_tma_args, all_tma_canonical,
    ):
        """Build dispatch fn (replicates Megakernel._build_exec_dispatch_fn)."""
        import machete.megakernel.compile as cm

        is_load = phase_name == "load"
        tensor_params = ", ".join(all_canonical)
        tile_params = ", ".join(f"tile_{i}" for i in range(MAX_TILE_DIMS))
        tma_params = (
            ", ".join(all_tma_canonical) if all_tma_canonical else ""
        )

        lines = []
        for i, args in enumerate(op_tensor_args):
            all_args = list(dict.fromkeys(args))
            if op_tma_args and i < len(op_tma_args):
                all_args.extend(op_tma_args[i])
            args_str = (", " + ", ".join(all_args)) if all_args else ""
            kw = "if" if i == 0 else "elif"
            if is_load:
                lines.append(
                    f"    {kw} op_idx == Int32({i}):\n"
                    f"        _fn_{i}(page_ptr, {tile_params}, op_config_ptr, "
                    f"work_mbar, inner_iter_idx{args_str})"
                )
            else:
                lines.append(
                    f"    {kw} op_idx == Int32({i}):\n"
                    f"        _fn_{i}(page_ptr, {tile_params}, "
                    f"op_config_ptr{args_str})"
                )

        body = "\n".join(lines) if lines else "    pass"
        fn_name = f"dispatch_{phase_name}"
        tensor_sig = f", {tensor_params}" if tensor_params else ""
        tma_sig = f", {tma_params}" if tma_params else ""
        extra = (
            f", work_mbar, inner_iter_idx"
            if is_load
            else ""
        )
        src = (
            f"@cute.jit\ndef {fn_name}(op_idx, page_ptr, {tile_params}, "
            f"op_config_ptr{extra}{tensor_sig}{tma_sig}):\n{body}\n"
        )
        globs = {"cute": cute, "Int32": Int32, "Int64": Int64}
        for i, fn in enumerate(phase_fns):
            globs[f"_fn_{i}"] = fn
        return self._exec_jit(src, fn_name, cm, extra_globals=globs)

    # =========================================================================
    # Kernel Creation
    # =========================================================================

    def _create_kernel(self):
        """Build the grid kernel with warp specialization."""
        import machete.megakernel.compile as cm

        cfg = self.config
        threads_per_block = cfg.threads_per_block
        num_mma_warps = (threads_per_block // 32) - NUM_DMA_WARPS
        num_compute_threads = num_mma_warps * 32
        dma_reg_count = cfg.dma_reg_count
        mma_reg_count = cfg.mma_reg_count
        total_tiles = self._total_tiles
        smem_size = self._smem_size
        work_mbar_offset = self._work_mbar_offset
        compute_done_offset = self._compute_done_offset
        page_offset = self._page_offset

        try:
            from cutlass.cute.arch import (
                setmaxregister_increase,
                setmaxregister_decrease,
            )
        except ImportError:
            try:
                from cutlass.cute.arch import (
                    warpgroup_reg_alloc as setmaxregister_increase,
                    warpgroup_reg_dealloc as setmaxregister_decrease,
                )
            except ImportError:
                setmaxregister_increase = setmaxregister_decrease = (
                    lambda n: None
                )

        # Build dispatch functions
        (
            dispatch_load,
            dispatch_compute,
            dispatch_store,
            inner_iters_list,
        ) = self._build_dispatch_fns()
        decompose_block_id = self._build_decompose_fn()
        _get_inner_iters = self._build_inner_iters_fn(inner_iters_list)

        # Tensor/TMA param strings for signature threading
        all_canonical = self._tensor_registry.canonical_names
        tensor_params = ", ".join(all_canonical)
        tensor_sig = f", {tensor_params}" if tensor_params else ""

        tma_registry = self._tma_registry
        all_tma_canonical = tma_registry.all_canonical_names
        tma_params = ", ".join(all_tma_canonical)
        tma_sig = f", {tma_params}" if tma_params else ""

        # Define kernel body (warp-specialized, one tile per block)
        def _kernel_body(
            op_configs_ptr: Int64,
            tidx: Int32,
            block_id: Int32,
            smem_base: Int32,
        ) -> None:
            warp_id = tidx // Int32(32)
            lane_id = tidx % Int32(32)
            is_load_warp = warp_id == Int32(num_mma_warps)
            is_store_warp = warp_id == Int32(num_mma_warps + 1)

            if warp_id >= Int32(num_mma_warps):
                setmaxregister_decrease(dma_reg_count)
            if warp_id < Int32(num_mma_warps):
                setmaxregister_increase(mma_reg_count)

            op_idx, tile_0, tile_1, tile_2, tile_3, tile_4 = (
                decompose_block_id(block_id)
            )
            _wm = smem_base + Int32(work_mbar_offset)
            _cd = smem_base + Int32(compute_done_offset)
            page_ptr = smem_base + Int32(page_offset)
            op_config = ld_global_i64(op_configs_ptr, op_idx)

            # Init mbarriers (load warp thread 0)
            if is_load_warp:
                if lane_id == Int32(0):
                    mbarrier_init(_wm, Int32(1))
                    mbarrier_init(_cd, Int32(num_mma_warps))
                    mbarrier_init_fence()

            named_barrier_sync(Int32(0), Int32(threads_per_block))

            # LOAD WARP: dispatch iter 0
            if is_load_warp:
                _zero = Int32(0)
                dispatch_load(
                    op_idx, page_ptr,
                    tile_0, tile_1, tile_2, tile_3, tile_4,
                    op_config, _wm,
                    _zero,
                )

            # STORE WARP: inner iters 1+, wait compute, store
            if is_store_warp:
                _n_iters = _get_inner_iters(op_idx)
                if _n_iters > Int32(1):
                    mbarrier_wait(_wm, Int32(0))
                _iter = Int32(1)
                while _iter < _n_iters:
                    dispatch_load(
                        op_idx, page_ptr,
                        tile_0, tile_1, tile_2, tile_3, tile_4,
                        op_config, _wm,
                        _iter,
                    )
                    _iter = _iter + Int32(1)
                mbarrier_wait(_cd, Int32(0))
                dispatch_store(
                    op_idx, page_ptr,
                    tile_0, tile_1, tile_2, tile_3, tile_4,
                    op_config,
                )
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)

            # MMA WARPS: wait data, compute, signal done
            if warp_id < Int32(num_mma_warps):
                mbarrier_wait(_wm, Int32(0))
                dispatch_compute(
                    op_idx, page_ptr,
                    tile_0, tile_1, tile_2, tile_3, tile_4,
                    op_config,
                )
                named_barrier_sync(Int32(1), Int32(num_compute_threads))
                if lane_id == Int32(0):
                    mbarrier_arrive(_cd)

        # Source-transform: add tensor+TMA params to dispatch calls
        body = _extract_body(_kernel_body)
        extra_dispatch = ", ".join(filter(None, [tensor_params, tma_params]))
        if extra_dispatch:
            body = re.sub(
                r"(dispatch_(?:load|compute|store))\(([^)]*)\)",
                lambda m: (
                    f"{m.group(1)}("
                    f"{m.group(2).rstrip().rstrip(',')}, {extra_dispatch})"
                ),
                body,
            )

        kb_src = (
            "@cute.jit\n"
            "def _kernel_body("
            f"op_configs_ptr, tidx, block_id, smem_base{tensor_sig}{tma_sig}"
            "):\n"
            + textwrap.indent(body, "    ")
            + "\n"
        )
        kb_globals = {
            "cute": cute,
            "Int32": Int32,
            "Int64": Int64,
            "dispatch_load": dispatch_load,
            "dispatch_compute": dispatch_compute,
            "dispatch_store": dispatch_store,
            "decompose_block_id": decompose_block_id,
            "_get_inner_iters": _get_inner_iters,
            "ld_global_i64": ld_global_i64,
            "mbarrier_init": mbarrier_init,
            "mbarrier_init_fence": mbarrier_init_fence,
            "mbarrier_arrive": mbarrier_arrive,
            "mbarrier_wait": mbarrier_wait,
            "named_barrier_sync": named_barrier_sync,
            "setmaxregister_increase": setmaxregister_increase,
            "setmaxregister_decrease": setmaxregister_decrease,
            "num_mma_warps": num_mma_warps,
            "num_compute_threads": num_compute_threads,
            "threads_per_block": threads_per_block,
            "dma_reg_count": dma_reg_count,
            "mma_reg_count": mma_reg_count,
            "work_mbar_offset": work_mbar_offset,
            "compute_done_offset": compute_done_offset,
            "page_offset": page_offset,
        }
        kernel_body = self._exec_jit(
            kb_src, "_kernel_body", cm, extra_globals=kb_globals
        )

        # Build GridKernel class (TMA descriptor creation + launch)
        tma_tensor_names = []
        seen = set()
        for desc in tma_registry.descriptors:
            tn = f"tma_{desc.tensor_canonical}"
            if tn not in seen:
                seen.add(tn)
                tma_tensor_names.append(tn)
        tma_tensor_sig = (
            ", " + ", ".join(tma_tensor_names) if tma_tensor_names else ""
        )

        tma_creation_lines = []
        tma_kernel_args = []
        if tma_registry.has_tma:
            for desc in tma_registry.descriptors:
                tma_src = f"tma_{desc.tensor_canonical}"
                if desc.direction == "g2s":
                    copy_op = "CopyBulkTensorTileG2SOp()"
                elif desc.direction == "s2g":
                    copy_op = "CopyBulkTensorTileS2GOp()"
                elif desc.direction == "s2g_reduce":
                    copy_op = (
                        "CopyReduceBulkTensorTileS2GOp("
                        "reduction_kind=ReductionOp.ADD)"
                    )
                else:
                    raise ValueError(
                        f"Unknown TMA direction: {desc.direction}"
                    )
                ss = ", ".join(str(s) for s in desc.tile_shape)
                sl = desc.smem_layout_src or f"cute.make_layout(({ss},))"
                tma_creation_lines.append(
                    f"        _sl_{desc.canonical_atom} = {sl}\n"
                    f"        {desc.canonical_atom}, {desc.canonical_gmem} = "
                    f"cute.nvgpu.cpasync.make_tiled_tma_atom(\n"
                    f"            {copy_op}, {tma_src}, "
                    f"_sl_{desc.canonical_atom}, ({ss},), num_multicast=1)"
                )
                tma_kernel_args.append(desc.canonical_atom)
                tma_kernel_args.append(desc.canonical_gmem)

        tma_code = "\n".join(tma_creation_lines)
        if tma_code:
            tma_code = "\n" + tma_code + "\n"
        tma_ka_sig = (
            ", " + ", ".join(tma_kernel_args) if tma_kernel_args else ""
        )

        gk_src = (
            "class GridKernel:\n"
            "    def __init__(self):\n"
            f"        self.total_tiles = {total_tiles}\n"
            f"        self.threads_per_block = {threads_per_block}\n"
            f"        self.smem_size = {smem_size}\n"
            "\n"
            "    @cute.jit\n"
            "    def __call__(self, op_configs_ptr"
            f"{tensor_sig}{tma_tensor_sig}, stream):\n"
            f"{tma_code}"
            f"        self.kernel(op_configs_ptr{tensor_sig}{tma_ka_sig}"
            ").launch(\n"
            f"            grid=[self.total_tiles, 1, 1],\n"
            f"            block=[self.threads_per_block, 1, 1],\n"
            f"            smem=self.smem_size, stream=stream)\n"
            "\n"
            "    @cute.kernel\n"
            f"    def kernel(self, op_configs_ptr{tensor_sig}{tma_sig}):\n"
            "        tidx = cute.arch.thread_idx()[0]\n"
            "        block_id = cute.arch.block_idx()[0]\n"
            "        smem_base = get_smem_base_ptr()\n"
            f"        _kernel_body(op_configs_ptr, tidx, block_id, smem_base"
            f"{tensor_sig}{tma_sig})\n"
        )

        gk_globals = {
            "cute": cute,
            "get_smem_base_ptr": get_smem_base_ptr,
            "_kernel_body": kernel_body,
        }
        if tma_registry.has_tma:
            from cutlass.cute.nvgpu.cpasync import (
                CopyBulkTensorTileG2SOp,
                CopyBulkTensorTileS2GOp,
            )

            gk_globals["CopyBulkTensorTileG2SOp"] = CopyBulkTensorTileG2SOp
            gk_globals["CopyBulkTensorTileS2GOp"] = CopyBulkTensorTileS2GOp

            if any(
                d.direction == "s2g_reduce" for d in tma_registry.descriptors
            ):
                from cutlass.cute.nvgpu.cpasync import (
                    CopyReduceBulkTensorTileS2GOp,
                )
                from cutlass.cute.tensor import ReductionOp

                gk_globals["CopyReduceBulkTensorTileS2GOp"] = (
                    CopyReduceBulkTensorTileS2GOp
                )
                gk_globals["ReductionOp"] = ReductionOp

        GridKernel = self._exec_jit(
            gk_src, "GridKernel", cm, extra_globals=gk_globals
        )
        return GridKernel()

    # =========================================================================
    # Cache Key
    # =========================================================================

    def _make_cache_key(self) -> Tuple:
        op_keys = []
        for op in self.ops:
            static_dims = (
                tuple(sorted(op.static_dims.items()))
                if op.static_dims
                else ()
            )
            dtypes = (
                tuple(
                    sorted((k, v.__name__) for k, v in op.tensor_dtypes.items())
                )
                if op.tensor_dtypes
                else ()
            )
            strides = (
                tuple(sorted((k, v) for k, v in op.tensor_strides.items()))
                if op.tensor_strides
                else ()
            )
            tma_loads = frozenset(getattr(op.op_cls, "_TMA_LOADS", set()))
            tma_stores = frozenset(getattr(op.op_cls, "_TMA_STORES", set()))
            op_keys.append(
                (
                    op.op_cls,
                    static_dims,
                    dtypes,
                    strides,
                    op.tile_counts,
                    tma_loads,
                    tma_stores,
                )
            )
        config_key = (
            self.config.threads_per_block,
            self.config.page_size,
            self.config.dma_reg_count,
            self.config.mma_reg_count,
        )
        return (tuple(op_keys), config_key)

    # =========================================================================
    # Compile / Run / Bench
    # =========================================================================

    def compile(self):
        """JIT compile the kernel (cached across instances with same config)."""
        self._prepare_op_configs()
        self._prepare_cute_tensors()
        self._prepare_tma_tensors()

        if self._compiled_kernel is not None:
            return

        cache_key = self._make_cache_key()
        if cache_key in SingleOpKernel._compiled_kernel_cache:
            self._compiled_kernel = SingleOpKernel._compiled_kernel_cache[
                cache_key
            ]
            return

        from cutedsl_trace.config import set_tracing_enabled

        set_tracing_enabled(False)

        tma_str = " [TMA]" if self._tma_registry.has_tma else ""
        print(
            f"Compiling grid kernel{tma_str} for {len(self.ops)} ops, "
            f"{self._total_tiles} tiles, "
            f"{self._smem_size // 1024}KB smem..."
        )
        self._compiled_kernel = self._create_kernel()

        import cuda.bindings.driver as cuda

        torch_stream = torch.cuda.current_stream()
        cu_stream = cuda.CUstream(torch_stream.cuda_stream)

        tma_tensor_args = (
            [ct for _, ct in self._tma_cute_tensors]
            if self._tma_cute_tensors
            else []
        )
        self._compiled_kernel = cute.compile(
            self._compiled_kernel,
            Int64(self._op_configs_tensor.data_ptr()),
            *self._cute_tensors,
            *tma_tensor_args,
            cu_stream,
        )

        SingleOpKernel._compiled_kernel_cache[cache_key] = (
            self._compiled_kernel
        )
        print("Compilation complete.")

    def run(self, stream=None, sync: bool = True):
        """Launch the kernel."""
        self.compile()

        if stream is None:
            import cuda.bindings.driver as cuda

            torch_stream = torch.cuda.current_stream()
            stream = cuda.CUstream(torch_stream.cuda_stream)

        tma_tensor_args = (
            [ct for _, ct in self._tma_cute_tensors]
            if self._tma_cute_tensors
            else []
        )
        self._compiled_kernel(
            Int64(self._op_configs_tensor.data_ptr()),
            *self._cute_tensors,
            *tma_tensor_args,
            stream,
        )
        if sync:
            torch.cuda.synchronize()

    def bench_spec(self, setup_fn=None, keep_alive=None):
        """Create a KernelBenchSpec for raw GPU kernel timing."""
        import cuda.bindings.driver as cuda
        from machete.utils.benchmark_utils import KernelBenchSpec

        self.compile()

        bench_stream = torch.cuda.Stream()
        cu_stream = cuda.CUstream(bench_stream.cuda_stream)

        compiled_kernel = self._compiled_kernel
        op_configs_tensor = self._op_configs_tensor
        cute_tensors = (
            list(self._cute_tensors) if self._cute_tensors else []
        )
        tma_tensor_args = (
            [ct for _, ct in self._tma_cute_tensors]
            if self._tma_cute_tensors
            else []
        )

        def _setup():
            if setup_fn is not None:
                setup_fn()

        def _launch():
            compiled_kernel(
                Int64(op_configs_tensor.data_ptr()),
                *cute_tensors,
                *tma_tensor_args,
                cu_stream,
            )

        return KernelBenchSpec(
            launch_fn=_launch,
            setup_fn=_setup,
            stream=(bench_stream, cu_stream),
            _keep_alive=(self, keep_alive),
        )

    def __repr__(self) -> str:
        op_names = ", ".join(
            f"{op.op_cls.__name__}({op.total_tiles})"
            for op in self.ops
        )
        return (
            f"SingleOpKernel(\n"
            f"  ops=[{op_names}],\n"
            f"  total_tiles={self._total_tiles},\n"
            f"  smem={self._smem_size // 1024}KB\n"
            f")"
        )


__all__ = ["SingleOpKernel"]
