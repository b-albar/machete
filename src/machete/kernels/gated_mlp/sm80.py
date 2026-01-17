# Copyright (c) 2025, Machete Authors
"""
Gated MLP kernel with L/C/S decomposition for SM80+.

This kernel computes: output = activation(x @ W_gate) * (x @ W_up)
Where W is (K, 2*N) with gate/up columns interleaved: [g0, u0, g1, u1, ...]

Uses L/C/S pattern for No Bubbles pipelining and tile-based GEMM.
"""

import torch
from torch import Tensor
import cutlass.cute as cute
from cutlass import Float32, const_expr

import quack.activation as qact
from quack.cute_dsl_utils import torch2cute_dtype_map
from machete.megakernel.interface import FusableKernel
from machete.megakernel.single import SingleKernel


class GatedMLPSM80(SingleKernel, FusableKernel):
    """
    Gated MLP kernel with activation (SiLU/SwiGLU, GELU/GeGLU).

    Computes: output = activation(x @ W_gate) * (x @ W_up)
    Where W is (K, 2*N) with interleaved columns [g0, u0, g1, u1, ...].

    This implementation uses tiled GEMM with shared memory for efficient computation.
    Each block computes a tile of the output matrix.
    """

    # Tile sizes - optimized for SM80+
    TILE_M = 64  # Rows per block
    TILE_N = 64  # Output columns per block (N, not 2N)
    TILE_K = 32  # K dimension tile
    NUM_THREADS = 128

    def __init__(self, dtype: torch.dtype, act_type: str, m_dim: int, k_dim: int, n_dim: int):
        """
        Initialize the GatedMLP kernel.

        Args:
            dtype: Input/output data type
            act_type: Activation type ("silu" or "gelu")
            m_dim: Number of rows (batch * seq_len)
            k_dim: Input dimension (d_model)
            n_dim: Output dimension (half of weight's N dimension)
        """
        self.torch_dtype = dtype
        self.cute_dtype = torch2cute_dtype_map[dtype]
        self.act_type = act_type
        self.m_dim = m_dim
        self.k_dim = k_dim
        self.n_dim = n_dim  # Output dimension (N, not 2N)
        SingleKernel.__init__(self, self, self.grid_fn, self.block_fn)

    @property
    def smem_size_fwd(self) -> int:
        """
        Shared memory for forward pass:
        - x_tile: TILE_M x TILE_K
        - w_gate_tile: TILE_K x TILE_N
        - w_up_tile: TILE_K x TILE_N
        """
        element_size = 2 if self.torch_dtype in [torch.float16, torch.bfloat16] else 4
        x_size = self.TILE_M * self.TILE_K * element_size
        w_size = 2 * self.TILE_K * self.TILE_N * element_size  # gate + up
        return x_size + w_size

    @property
    def smem_size_bwd(self) -> int:
        """Shared memory for backward pass (not fully implemented yet)."""
        return self.smem_size_fwd

    # ========== Forward Pass L/C/S ==========

    @cute.jit
    def load_forward(self, logical_idx, smem, x_ptr, w_ptr, out_ptr):
        """
        Load x tile and weight tiles (gate & up) into shared memory.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx_m, bidx_n, _ = cute.arch.block_idx()
        num_threads = const_expr(self.NUM_THREADS)

        m_dim = const_expr(self.m_dim)
        k_dim = const_expr(self.k_dim)
        n_dim = const_expr(self.n_dim)
        TILE_M = const_expr(self.TILE_M)
        TILE_N = const_expr(self.TILE_N)
        TILE_K = const_expr(self.TILE_K)

        # Create tensors from pointers
        m_x = cute.make_tensor(
            x_ptr,
            cute.make_layout((m_dim, k_dim), stride=(k_dim, 1)),
        )
        # Weight is (K, 2*N) interleaved: [g0, u0, g1, u1, ...]
        m_w = cute.make_tensor(
            w_ptr,
            cute.make_layout((k_dim, 2 * n_dim), stride=(2 * n_dim, 1)),
        )

        # Compute tile offsets
        m_start = bidx_m * TILE_M
        n_start = bidx_n * TILE_N

        # Allocate shared memory regions
        # smem layout: [x_tile | w_gate_tile | w_up_tile]
        x_offset = 0
        w_gate_offset = TILE_M * TILE_K
        w_up_offset = w_gate_offset + TILE_K * TILE_N

        # Load x tile (TILE_M x TILE_K) - will accumulate across K
        # For first K tile only
        total_x = TILE_M * TILE_K
        for idx in range(tidx, total_x, num_threads):
            m_idx = idx // TILE_K
            k_idx = idx % TILE_K
            global_m = m_start + m_idx
            global_k = k_idx
            if global_m < m_dim and global_k < k_dim:
                smem[x_offset + idx] = m_x[global_m, global_k]
            else:
                smem[x_offset + idx] = Float32(0.0).to(self.cute_dtype)

        # Load weight tiles for gate and up
        # Gate columns are at indices 2*n, up columns are at 2*n+1
        total_w = TILE_K * TILE_N
        for idx in range(tidx, total_w, num_threads):
            k_idx = idx // TILE_N
            n_idx = idx % TILE_N
            global_n = n_start + n_idx
            if k_idx < k_dim and global_n < n_dim:
                # Gate: column 2*global_n
                smem[w_gate_offset + idx] = m_w[k_idx, 2 * global_n]
                # Up: column 2*global_n + 1
                smem[w_up_offset + idx] = m_w[k_idx, 2 * global_n + 1]
            else:
                smem[w_gate_offset + idx] = Float32(0.0).to(self.cute_dtype)
                smem[w_up_offset + idx] = Float32(0.0).to(self.cute_dtype)

    @cute.jit
    def compute_forward(self, logical_idx, smem, x_ptr, w_ptr, out_ptr):
        """
        Compute the tiled GEMM with gated activation:
        - acc_gate = x_tile @ w_gate_tile
        - acc_up = x_tile @ w_up_tile
        - out = activation(acc_gate) * acc_up
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx_m, bidx_n, _ = cute.arch.block_idx()
        num_threads = const_expr(self.NUM_THREADS)

        m_dim = const_expr(self.m_dim)
        k_dim = const_expr(self.k_dim)
        n_dim = const_expr(self.n_dim)
        TILE_M = const_expr(self.TILE_M)
        TILE_N = const_expr(self.TILE_N)
        TILE_K = const_expr(self.TILE_K)

        # Create output tensor from pointer
        m_out = cute.make_tensor(
            out_ptr,
            cute.make_layout((m_dim, n_dim), stride=(n_dim, 1)),
        )

        # Create input tensor for k-loop continuation
        m_x = cute.make_tensor(
            x_ptr,
            cute.make_layout((m_dim, k_dim), stride=(k_dim, 1)),
        )
        m_w = cute.make_tensor(
            w_ptr,
            cute.make_layout((k_dim, 2 * n_dim), stride=(2 * n_dim, 1)),
        )

        m_start = bidx_m * TILE_M
        n_start = bidx_n * TILE_N

        # Shared memory offsets
        x_offset = 0
        w_gate_offset = TILE_M * TILE_K
        w_up_offset = w_gate_offset + TILE_K * TILE_N

        # Each thread computes multiple output elements
        # Simple row-major distribution
        total_out = TILE_M * TILE_N
        for out_idx in range(tidx, total_out, num_threads):
            m_local = out_idx // TILE_N
            n_local = out_idx % TILE_N

            global_m = m_start + m_local
            global_n = n_start + n_local

            if global_m < m_dim and global_n < n_dim:
                # Accumulate dot products across all K tiles
                acc_gate = Float32(0.0)
                acc_up = Float32(0.0)

                # First K tile from shared memory
                for k in range(TILE_K):
                    if k < k_dim:
                        x_val = smem[x_offset + m_local * TILE_K + k].to(Float32)
                        g_val = smem[w_gate_offset + k * TILE_N + n_local].to(Float32)
                        u_val = smem[w_up_offset + k * TILE_N + n_local].to(Float32)
                        acc_gate = acc_gate + x_val * g_val
                        acc_up = acc_up + x_val * u_val

                # Remaining K tiles from global memory
                for k_tile in range(TILE_K, k_dim, TILE_K):
                    for k in range(TILE_K):
                        global_k = k_tile + k
                        if global_k < k_dim:
                            x_val = m_x[global_m, global_k].to(Float32)
                            g_val = m_w[global_k, 2 * global_n].to(Float32)
                            u_val = m_w[global_k, 2 * global_n + 1].to(Float32)
                            acc_gate = acc_gate + x_val * g_val
                            acc_up = acc_up + x_val * u_val

                # Apply gated activation
                if const_expr(self.act_type == "silu"):
                    gate = qact.silu(acc_gate)
                elif const_expr(self.act_type == "gelu"):
                    gate = qact.geglu(acc_gate, Float32(1.0))
                else:
                    gate = acc_gate

                result = gate * acc_up
                m_out[global_m, global_n] = result.to(self.cute_dtype)

    @cute.jit
    def store_forward(self, logical_idx, smem, x_ptr, w_ptr, out_ptr):
        """No-op: results written directly in compute phase."""
        pass

    # ========== Backward Pass L/C/S ==========
    # Backward requires computing:
    # 1. dh_gate, dh_up = d_activation(h_gate, h_up, dout)
    # 2. dx = dh_gate @ W_gate.T + dh_up @ W_up.T
    # 3. dW_gate = x.T @ dh_gate, dW_up = x.T @ dh_up

    @cute.jit
    def load_backward(self, logical_idx, smem, dout_ptr, x_ptr, w_ptr, dx_ptr, dw_ptr):
        """Load data for backward pass."""
        pass  # Not yet implemented

    @cute.jit
    def compute_backward(self, logical_idx, smem, dout_ptr, x_ptr, w_ptr, dx_ptr, dw_ptr):
        """Compute backward pass."""
        pass  # Not yet implemented

    @cute.jit
    def store_backward(self, logical_idx, smem, dout_ptr, x_ptr, w_ptr, dx_ptr, dw_ptr):
        """Store backward results."""
        pass  # Not yet implemented

    # ========== Launch Helpers ==========

    def grid_fn(self, *args):
        """Grid: one block per (TILE_M, TILE_N) output tile."""
        n_tiles_m = (self.m_dim + self.TILE_M - 1) // self.TILE_M
        n_tiles_n = (self.n_dim + self.TILE_N - 1) // self.TILE_N
        return [n_tiles_m, n_tiles_n, 1]

    def block_fn(self, *args):
        return [self.NUM_THREADS, 1, 1]

    def run_forward(self, ctx, x_2d, weight, out):
        ctx.save_for_backward(x_2d, weight)
        args = (x_2d, weight, out)
        grid = self.grid_fn(*args)
        block = self.block_fn(*args)
        n_blocks = grid[0] * grid[1] * grid[2]

        self._update_or_add(self.mk_fwd, args)
        self.mk_fwd.launch(n_blocks, grid, block)
        return out

    def run_backward(self, ctx, dout):
        x_2d, weight = ctx.saved_tensors
        # Backward not yet implemented - return None gradients
        dx = torch.zeros_like(x_2d)
        dw = torch.zeros_like(weight)
        return dx, dw, None

    def __call__(self, x: Tensor, weight: Tensor) -> Tensor:
        """
        Forward pass for Gated MLP.

        Args:
            x: Input tensor of shape (..., K)
            weight: Weight tensor of shape (K, 2*N) with interleaved gate/up columns

        Returns:
            Output tensor of shape (..., N)
        """
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        m_dim, k_dim = x_2d.shape
        n2_dim = weight.shape[1]
        n_dim = n2_dim // 2

        out = torch.empty((m_dim, n_dim), dtype=x.dtype, device=x.device)

        return self.apply_autograd(x_2d, weight, out).view(*orig_shape[:-1], n_dim)
