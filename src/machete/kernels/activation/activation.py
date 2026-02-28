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

    c = torch.zeros(M, N, dtype=dtype, device="cuda")
    ops = GemmOp.schedule(a=a, b=b, c=c, tile_sizes={"M": 64, "N": 32, "K": 32})
    ops += ActivationOp.schedule(x=c, activation='relu', tile_sizes={"M": 4})
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


class ActivationOp(Op):
    """Element-wise activation for the megakernel framework.

    Reads x from global, applies activation in registers, writes back.
    In-place: schedule with y=x (default when y is omitted).

    Tensor declarations:
        x: (M, D) -- input tensor
        y: (M, D) -- output tensor (typically same as x)

    Tiling:
        tile_M indexes row groups. D is full-extent (1 tile).
        tile_size_M * D * elem_bytes <= page_size.
    """

    reads = {"x": (None, ("M", "D"))}
    writes = {"y": (None, ("M", "D"))}
    tile = ("M", "D")

    tma_loads = {"x"}
    tma_stores = {"y"}

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

        self.x_tile_bytes = self.tile_size_M * self.D * self.elem_bytes

        assert self.x_tile_bytes <= self.page_size, (
            f"ActivationOp: tile smem ({self.x_tile_bytes}B) exceeds page_size ({self.page_size}B). "
            f"Reduce tile_size_M={self.tile_size_M}."
        )

        self.num_warps = self.threads_per_row // 32

    # =========================================================================
    # Scheduling
    # =========================================================================

    @classmethod
    def schedule_forward(cls, tile_sizes=None, activation='relu', page_size=DEFAULT_PAGE_SIZE, **tensors):
        """Schedule activation forward.

        Args:
            activation: 'relu' or 'silu'
            page_size: Shared memory page size in bytes.
            **tensors: x required, y optional (defaults to x for in-place)
        """
        tile_sizes = dict(tile_sizes or {})
        if "M" not in tile_sizes:
            x = tensors.get('x')
            if x is not None:
                D = x.shape[1]
                elem_bytes = x.element_size()
                tile_sizes["M"] = max(1, page_size // (D * elem_bytes))
        if 'y' not in tensors:
            tensors['y'] = tensors['x']
        act_id = ACT_MAP.get(activation, ACT_RELU)
        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        ops[0].static_dims['activation'] = act_id
        ops[0].static_dims['page_size'] = page_size
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
    def load(self, page_ptr, tile_M, tile_D, x_tma, x_tma_gmem, work_mbar):
        """TMA load of x tile from global to shared memory."""
        sX = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.D, self.tile_size_M)),
        )
        gX = cute.local_tile(
            x_tma_gmem, (self.D, self.tile_size_M), (None, None),
        )
        tXsX, tXgX = cute.nvgpu.cpasync.tma_partition(
            x_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sX, 0, 2),
            cute.group_modes(gX, 0, 2),
        )

        nbytes = Int32(self.x_tile_bytes)
        mbar_ptr = cute.make_ptr(
            cutlass.Int64, work_mbar, cute.AddressSpace.smem
        )
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, nbytes)
        cute.copy(x_tma, tXgX[(None, tile_D, tile_M)], tXsX,
                  tma_bar_ptr=mbar_ptr)

    # =========================================================================
    # Forward Compute
    # =========================================================================

    @cute.jit
    def compute(self, page_ptr, tile_M, tile_D):
        """Apply activation element-wise: read from smem, activate, write back."""
        x_smem = cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem)

        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.threads_per_row // 32
        thr_layout = cute.make_layout(32)

        row_start = tile_M * self.tile_size_M

        for local_row in range(warp_idx, self.tile_size_M, num_warps):
            row_idx = row_start + local_row

            if row_idx < self.M:
                x_row = cute.make_tensor(
                    x_smem + local_row * self.D,
                    cute.make_layout(self.D),
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
                    x_smem + local_row * self.D,
                    cute.make_layout(self.D),
                )
                y_part = cute.local_partition(y_row, thr_layout, lane_idx)
                cute.autovec_copy(y_reg, y_part)

    # =========================================================================
    # Forward Store (TMA S->G)
    # =========================================================================

    @cute.jit
    def store(self, page_ptr, tile_M, tile_D, y_tma, y_tma_gmem):
        """TMA store of activated result from shared to global memory."""
        sY = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.D, self.tile_size_M)),
        )
        gY = cute.local_tile(
            y_tma_gmem, (self.D, self.tile_size_M), (None, None),
        )
        tYsY, tYgY = cute.nvgpu.cpasync.tma_partition(
            y_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sY, 0, 2),
            cute.group_modes(gY, 0, 2),
        )
        with cute.arch.elect_one():
            cute.copy(y_tma, tYsY, tYgY[(None, tile_D, tile_M)])


__all__ = ["ActivationOp"]
