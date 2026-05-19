# Copyright (c) 2025, Machete Authors
"""
Element-wise Activation Op for the Megakernel.

Applies an activation function in-place:
    y = act(x)

Designed to fuse with GEMM: GEMM writes C via TMA store_add, then
ActivationOp reads C, applies activation, writes back. The framework
auto-detects the dependency via tensor pointer matching.

Supported activations:
    1 = ReLU:  max(0, x)
    2 = SiLU:  x * sigmoid(x)

Pipelined load/compute/store:
    load:    TMA G->S (x tile)
    compute: read x from smem, apply activation, write back to smem
    store:   TMA S->G (y = same tensor as x)

Usage:
    from machete.kernels.activation import ActivationOp
    from machete.kernels.gemm import GemmOp
    from machete.megakernel import Megakernel

    c = torch.zeros(1, M, N, dtype=dtype, device="cuda")
    ops = GemmOp.schedule(a=a, b=b, c=c, tile_sizes={"S": 64, "N": 32, "K": 32})
    ops += ActivationOp.schedule(x=c, activation='relu', tile_sizes={"S": 4})
    kernel = Megakernel(ops)
    kernel.run()
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32

from machete.megakernel.ops import Op, DEFAULT_PAGE_SIZE
from machete.megakernel.interpreter import mbarrier_arrive_expect_tx


# Activation type constants
ACT_RELU = 1
ACT_SILU = 2

ACT_MAP = {'relu': ACT_RELU, 'silu': ACT_SILU}


def _align_up(x: int, align: int) -> int:
    return ((x + align - 1) // align) * align


def _pick_tma_tile_n(N: int, tile_s: int, elem_bytes: int) -> int:
    upper = min(N, 256)
    for tile_n in range(upper, 15, -1):
        if N % tile_n == 0 and (tile_n * tile_s * elem_bytes) % 128 == 0:
            return tile_n
    for tile_n in range(upper, 15, -1):
        if N % tile_n == 0:
            return tile_n
    return 16


class ActivationOp(Op):
    """Element-wise activation for the megakernel framework.

    Reads x from global, applies activation in registers, writes back.
    In-place: schedule with y=x (default when y is omitted).

    Tensor declarations:
        x: (B, S, N) -- input tensor
        y: (B, S, N) -- output tensor (typically same as x)

    Tiling:
        tile_B=1, tile_S indexes row groups. N is full-extent (1 tile).
        tile_size_S * N * elem_bytes <= page_size.
    """

    reads = {"x": (None, ("B", "S", "N"))}
    writes = {"y": (None, ("B", "S", "N"))}
    tile = ("B", "S", "N")
    dynamic_dims = ("B",)

    tma_loads = {"x"}
    tma_stores = {"y"}

    @classmethod
    def get_tma_tile_shape(cls, tensor_name, tile_sizes, static_dims):
        if tensor_name in ("x", "y"):
            return (1, tile_sizes["S"], static_dims["tma_tile_N"])
        return None

    @classmethod
    def get_tma_smem_layout_src(cls, tensor_name, tma_tile_shape, tile_sizes, static_dims):
        if tensor_name in ("x", "y"):
            tile_n, tile_s, tile_b = tma_tile_shape
            return f"cute.make_layout(({tile_n}, {tile_s}, {tile_b}))"
        return None

    def __init__(self, **config):
        super().__init__(**config)
        self.activation = getattr(self, 'activation', ACT_RELU)
        self.page_size = getattr(self, 'page_size', DEFAULT_PAGE_SIZE)

        if self.x_dtype == cutlass.Float32:
            self.elem_bytes = 4
        elif self.x_dtype in (cutlass.Float16, cutlass.BFloat16):
            self.elem_bytes = 2
        else:
            self.elem_bytes = 4
        self.tma_tile_N = getattr(
            self,
            'tma_tile_N',
            _pick_tma_tile_n(self.N, self.tile_size_S, self.elem_bytes),
        )
        self.num_tma_tiles = self.N // self.tma_tile_N
        self.chunk_tile_elems = self.tma_tile_N * self.tile_size_S
        self.chunk_tile_bytes = self.chunk_tile_elems * self.elem_bytes
        self.chunk_stride_bytes = _align_up(self.chunk_tile_bytes, 128)
        self.chunk_stride_elems = self.chunk_stride_bytes // self.elem_bytes
        self.x_tile_bytes = self.num_tma_tiles * self.chunk_stride_bytes

        assert self.x_tile_bytes <= self.page_size, (
            f"ActivationOp: tile smem ({self.x_tile_bytes}B) exceeds page_size ({self.page_size}B). "
            f"Reduce tile_size_S={self.tile_size_S}."
        )

    # =========================================================================
    # Scheduling
    # =========================================================================

    @classmethod
    def schedule(cls, tile_sizes=None, activation='relu', page_size=DEFAULT_PAGE_SIZE, **tensors):
        """Schedule activation forward.

        Args:
            activation: 'relu' or 'silu'
            page_size: Shared memory page size in bytes.
            **tensors: x (B,S,N) required, y optional (defaults to x for in-place)
        """
        tile_sizes = dict(tile_sizes or {})
        if "S" not in tile_sizes:
            x = tensors.get('x')
            if x is not None:
                N = x.shape[2]
                elem_bytes = x.element_size()
                tile_sizes["S"] = max(1, page_size // (N * elem_bytes))
        tile_sizes.setdefault("B", 1)
        if 'y' not in tensors:
            tensors['y'] = tensors['x']
        act_id = ACT_MAP.get(activation, ACT_RELU)
        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        ops[0].static_dims['activation'] = act_id
        ops[0].static_dims['page_size'] = page_size
        ops[0].static_dims['tma_tile_N'] = _pick_tma_tile_n(
            tensors['x'].shape[2],
            tile_sizes["S"],
            tensors['x'].element_size(),
        )
        return ops

    @classmethod
    def kernel_config(cls, ops):
        """Return recommended MegakernelConfig for scheduled ActivationOps."""
        from machete.megakernel import MegakernelConfig
        page_size = ops[0].static_dims.get('page_size', DEFAULT_PAGE_SIZE)
        return MegakernelConfig(page_size=page_size)

    # =========================================================================
    # Forward Load (TMA G->S)
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_B, tile_S, tile_N, x_tma, x_tma_gmem, work_mbar):
        """TMA load of x tile from global to shared memory."""
        nbytes = Int32(self.x_tile_bytes)
        mbar_ptr = cute.make_ptr(
            cutlass.Int64, work_mbar, cute.AddressSpace.smem
        )
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, nbytes)
        for wi in range(self.num_tma_tiles):
            gXi = cute.local_tile(
                x_tma_gmem,
                (self.tma_tile_N, self.tile_size_S, 1),
                (Int32(wi), tile_S, tile_B),
            )
            sXi = cute.make_tensor(
                cute.make_ptr(
                    self.x_dtype,
                    page_ptr + Int32(wi * self.chunk_stride_bytes),
                    cute.AddressSpace.smem,
                ),
                cute.make_layout((self.tma_tile_N, self.tile_size_S, 1)),
            )
            tXsXi, tXgXi = cute.nvgpu.cpasync.tma_partition(
                x_tma, Int32(0), cute.make_layout(1),
                cute.group_modes(sXi, 0, 3),
                cute.group_modes(gXi, 0, 3),
            )
            cute.copy(x_tma, tXgXi, tXsXi, tma_bar_ptr=mbar_ptr)

    # =========================================================================
    # Forward Compute
    # =========================================================================

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_N):
        """Apply activation element-wise: read from smem, activate, write back."""
        x_smem = cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem)

        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.threads_per_row // 32
        thr_layout = cute.make_layout(32)

        row_start = tile_S * self.tile_size_S

        for local_row in range(warp_idx, self.tile_size_S, num_warps):
            row_idx = row_start + local_row

            if row_idx < self.S:
                for wi in range(self.num_tma_tiles):
                    chunk_ptr = (
                        x_smem
                        + wi * self.chunk_stride_elems
                        + local_row * self.tma_tile_N
                    )
                    x_row = cute.make_tensor(
                        chunk_ptr,
                        cute.make_layout(self.tma_tile_N),
                    )
                    x_part = cute.local_partition(x_row, thr_layout, lane_idx)
                    x_reg = cute.make_fragment_like(x_part)
                    cute.autovec_copy(x_part, x_reg)

                    y_reg = cute.make_fragment_like(x_reg)
                    for i in range(cute.size(x_reg)):
                        val = x_reg[i].to(Float32)
                        if self.activation == ACT_RELU:
                            act_val = val
                            if val < Float32(0.0):
                                act_val = Float32(0.0)
                            y_reg[i] = act_val.to(self.x_dtype)
                        elif self.activation == ACT_SILU:
                            neg_val = Float32(0.0) - val
                            exp_neg = cute.math.exp(neg_val, fastmath=True)
                            act_val = val / (Float32(1.0) + exp_neg)
                            y_reg[i] = act_val.to(self.x_dtype)

                    y_row = cute.make_tensor(
                        chunk_ptr,
                        cute.make_layout(self.tma_tile_N),
                    )
                    y_part = cute.local_partition(y_row, thr_layout, lane_idx)
                    cute.autovec_copy(y_reg, y_part)

    # =========================================================================
    # Forward Store (TMA S->G)
    # =========================================================================

    @cute.jit
    def store(self, page_ptr, tile_B, tile_S, tile_N, y_tma, y_tma_gmem):
        """TMA store of activated result from shared to global memory."""
        with cute.arch.elect_one():
            for wi in range(self.num_tma_tiles):
                gYi = cute.local_tile(
                    y_tma_gmem,
                    (self.tma_tile_N, self.tile_size_S, 1),
                    (Int32(wi), tile_S, tile_B),
                )
                sYi = cute.make_tensor(
                    cute.make_ptr(
                        self.x_dtype,
                        page_ptr + Int32(wi * self.chunk_stride_bytes),
                        cute.AddressSpace.smem,
                    ),
                    cute.make_layout((self.tma_tile_N, self.tile_size_S, 1)),
                )
                tYsYi, tYgYi = cute.nvgpu.cpasync.tma_partition(
                    y_tma, Int32(0), cute.make_layout(1),
                    cute.group_modes(sYi, 0, 3),
                    cute.group_modes(gYi, 0, 3),
                )
                cute.copy(y_tma, tYsYi, tYgYi)


__all__ = ["ActivationOp"]
