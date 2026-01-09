# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from typing import Tuple, Type

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack

from quack.compile_utils import make_fake_tensor as fake_tensor

"""
Simple GEMM Forward Kernel for SM80 (Ampere).
Minimal implementation to establish a working baseline.
"""


class GatedMLPSM80:
    def __init__(
        self,
        ab_dtype: Type[cutlass.Numeric] = cutlass.Float16,
        c_dtype: Type[cutlass.Numeric] = cutlass.Float16,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        tile_shape: Tuple[int, int, int] = (128, 128, 32),
    ):
        self.ab_dtype = ab_dtype
        self.c_dtype = c_dtype
        self.acc_dtype = acc_dtype
        self.tile_m, self.tile_n, self.tile_k = tile_shape

        # MMA shape (16, 8, 16) for Ampere
        self.mma_inst_shape = (16, 8, 16)
        # 4 warps = 128 threads
        self.atom_layout_mnk = (2, 2, 1)
        self.num_threads = 128

    @cute.jit
    def __call__(
        self,
        ma: cute.Tensor,
        mb: cute.Tensor,
        mc: cute.Tensor,
    ):
        # Layouts
        # ma: (M, K)
        # mb: (N, K)
        # mc: (M, N)

        # Shared memory layouts - simple, no swizzling
        sa_layout = cute.make_layout((self.tile_m, self.tile_k), stride=(self.tile_k, 1))
        sb_layout = cute.make_layout((self.tile_n, self.tile_k), stride=(self.tile_k, 1))

        # Tiled Copy for Global to Shared
        copy_bits = 128
        atom_copy = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), ma.element_type, num_bits_per_copy=copy_bits)

        # Define thread layouts
        thr_layout = cute.make_layout((self.num_threads // 8, 8), stride=(8, 1))
        val_layout = cute.make_layout((1, 8))

        tiled_copy_a = cute.make_tiled_copy_tv(atom_copy, thr_layout, val_layout)
        tiled_copy_b = cute.make_tiled_copy_tv(atom_copy, thr_layout, val_layout)

        # MMA
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(self.ab_dtype, self.acc_dtype, self.mma_inst_shape)
        permutation_mnk = (
            self.atom_layout_mnk[0] * self.mma_inst_shape[0],
            self.atom_layout_mnk[1] * self.mma_inst_shape[1],
            self.atom_layout_mnk[2] * self.mma_inst_shape[2],
        )
        tiled_mma = cute.make_tiled_mma(mma_op, cute.make_layout(self.atom_layout_mnk), permutation_mnk=permutation_mnk)

        # Grid
        grid_dim = (cute.ceil_div(ma.shape[0], self.tile_m), cute.ceil_div(mb.shape[0], self.tile_n), 1)

        smem_size = cute.size_in_bytes(ma.element_type, sa_layout) + cute.size_in_bytes(mb.element_type, sb_layout)

        self.kernel(ma, mb, mc, sa_layout, sb_layout, tiled_copy_a, tiled_copy_b, tiled_mma).launch(
            grid=grid_dim, block=[self.num_threads, 1, 1], smem=smem_size
        )

    @cute.kernel
    def kernel(
        self,
        ma: cute.Tensor,
        mb: cute.Tensor,
        mc: cute.Tensor,
        sa_layout: cute.Layout,
        sb_layout: cute.Layout,
        tiled_copy_a: cute.TiledCopy,
        tiled_copy_b: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        # Local tiles
        ga = cute.local_tile(ma, (self.tile_m, self.tile_k), (bidx, None))
        gb = cute.local_tile(mb, (self.tile_n, self.tile_k), (bidy, None))
        gc = cute.local_tile(mc, (self.tile_m, self.tile_n), (bidx, bidy))

        # Shared memory
        smem = cutlass.utils.SmemAllocator()
        sa = smem.allocate_tensor(ma.element_type, sa_layout)
        sb = smem.allocate_tensor(mb.element_type, sb_layout)

        # Partitioning
        thr_copy_a = tiled_copy_a.get_slice(tidx)
        thr_copy_b = tiled_copy_b.get_slice(tidx)

        taga = thr_copy_a.partition_S(ga)
        tasa = thr_copy_a.partition_D(sa)
        tbgb = thr_copy_b.partition_S(gb)
        tbsb = thr_copy_b.partition_D(sb)

        thr_mma = tiled_mma.get_slice(tidx)
        tcsa = thr_mma.partition_A(sa)
        tcsb = thr_mma.partition_B(sb)

        tcra = tiled_mma.make_fragment_A(tcsa)
        tcrb = tiled_mma.make_fragment_B(tcsb)
        tcrc = tiled_mma.make_fragment_C(thr_mma.partition_C(gc))
        tcrc.fill(0.0)

        # Main loop
        k_tiles = cute.size(taga, mode=[2])
        for k in range(k_tiles):
            # Load to shared
            cute.copy(tiled_copy_a, taga[None, None, None, k], tasa[None, None, None])
            cute.copy(tiled_copy_b, tbgb[None, None, None, k], tbsb[None, None, None])
            cute.arch.sync_threads()

            # GEMM on current tile
            # Simple shared to register and MMA
            num_k_inner = cute.size(tcra, mode=[2])
            for k_in in range(num_k_inner):
                # Using basic copy atom for s2r
                atom_s2r = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), ma.element_type)
                cute.copy(atom_s2r, tcsa[None, None, k_in], tcra[None, None, k_in])
                cute.copy(atom_s2r, tcsb[None, None, k_in], tcrb[None, None, k_in])

                cute.gemm(tiled_mma, tcrc, tcra[None, None, k_in], tcrb[None, None, k_in], tcrc)

            cute.arch.sync_threads()

        # Recast accumulator to output dtype
        tcrd = cute.make_fragment_like(tcrc, mc.element_type)
        tcrd[None] = tcrc.load().to(mc.element_type)

        # Store back to global
        tcgc = thr_mma.partition_C(gc)
        atom_r2g = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mc.element_type)
        cute.copy(atom_r2g, tcrd, tcgc)


class GatedMLPSM80Func(torch.autograd.Function):
    _compile_cache = {}

    @staticmethod
    def forward(ctx, x, weight):
        # weight is (K, N)
        m, k = x.shape
        kw, n = weight.shape
        assert k == kw

        device = x.device
        dtype = x.dtype
        cute_dtype = {torch.float16: cutlass.Float16, torch.bfloat16: cutlass.BFloat16}[dtype]

        impl = GatedMLPSM80(ab_dtype=cute_dtype)
        y = torch.empty((m, n), dtype=dtype, device=device)

        ma = from_dlpack(x, assumed_align=16)
        mb = from_dlpack(weight.T.contiguous(), assumed_align=16)  # (N, K)
        mc = from_dlpack(y, assumed_align=16)

        key = (dtype,)
        if key not in GatedMLPSM80Func._compile_cache:
            ms = cute.sym_int()
            ma_f = fake_tensor(cute_dtype, (ms, k), divisibility=16)
            mb_f = fake_tensor(cute_dtype, (n, k), divisibility=16)
            mc_f = fake_tensor(cute_dtype, (ms, n), divisibility=16)
            GatedMLPSM80Func._compile_cache[key] = cute.compile(impl, ma_f, mb_f, mc_f)

        GatedMLPSM80Func._compile_cache[key](ma, mb, mc)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return None, None


def gated_mlp_sm80(x, weight, act_type="silu"):
    # Ignoring act_type for simple GEMM
    return GatedMLPSM80Func.apply(x, weight)
