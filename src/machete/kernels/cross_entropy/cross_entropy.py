# Copyright (c) 2025, Machete Authors
"""Cross-Entropy Op — Megakernel Op with TMA V-block chunking.

Fuses forward loss computation AND backward gradient computation in a single
kernel using online softmax (2-pass over vocabulary dimension V):

    Pass 1: Streaming max + sum for log-sum-exp (V-blocks loaded via TMA)
    Pass 2: Compute softmax gradients, write to grad_logits (V-blocks via TMA)

Architecture:
    DMA warps:  Load logits V-blocks via TMA (double-buffered).
                Store warp dispatches remaining TMA loads (iter 1+).
    MMA warps:  All warps cooperatively reduce over V via cross-warp reduction.
                Read V-blocks from smem, write gradients directly to global.

Mbarrier protocol (double-buffered, same as FlashAttentionSm100Op):
    4 op-managed mbarriers: smem_consumed[0,1], kblock_ready[0,1]
    Phase formula: phase(g) = ((g - 1) // 2) % 2 for global_iter g >= 1.
    Iter 0 uses framework work_mbar (not kblock_ready).
"""

import operator

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Float32

from machete.megakernel.ops import Op, DEFAULT_PAGE_SIZE
from machete.megakernel.interpreter import (
    mbarrier_init,
    mbarrier_init_fence,
    mbarrier_arrive,
    mbarrier_arrive_expect_tx,
    mbarrier_wait,
    named_barrier_sync,
)


def _compute_block_v(page_size, elem_bytes, V=None, num_warps=16):
    """Compute BLOCK_V from page_size and element size.

    BLOCK_V is capped at 4096 elements to keep compilation fast,
    and buf_bytes is aligned to 128 for TMA alignment.
    """
    mbar_bytes = 32  # 4 mbarriers × 8 bytes
    scratch_bytes = num_warps * 4
    meta_bytes = ((mbar_bytes + scratch_bytes + 127) // 128) * 128
    usable = page_size - meta_bytes
    buf_bytes = usable // 2
    # Align buffer to 128 bytes for TMA
    buf_bytes = (buf_bytes // 128) * 128
    block_v = buf_bytes // elem_bytes
    # Cap BLOCK_V to keep compilation manageable
    max_block_v = 4096
    if block_v > max_block_v:
        block_v = max_block_v
        buf_bytes = block_v * elem_bytes
    if V is not None and block_v > V:
        block_v = V
        buf_bytes = ((block_v * elem_bytes + 127) // 128) * 128
    return block_v, buf_bytes, meta_bytes


class CrossEntropyOp(Op):
    """Cross-entropy forward+backward Op with TMA V-block chunking.

    Tensors:
        logits:      (BT, V)  -- input logits (bf16 or fp32)
        targets:     (BT,)    -- target class indices (int32)
        loss:        (BT,)    -- output per-row CE loss (fp32)
        grad_logits: (BT, V)  -- output dL/dlogits (same dtype as logits)

    Tiling:
        tile_BT=1 (one row per tile, all warps cooperate on V reduction).
        V is chunked into BLOCK_V-sized blocks loaded via TMA.
    """

    reads = {
        "logits":  (None, ("BT", "V")),
        "targets": (cutlass.Int32, ("BT",)),
    }
    writes = {
        "loss":        (cutlass.Float32, ("BT",)),
        "grad_logits": (None, ("BT", "V")),
    }
    tile = ("BT",)
    tma_loads = {"logits"}

    @classmethod
    def get_tma_tile_shape(cls, tensor_name, tile_sizes, static_dims):
        """Custom TMA tile shape: (1, BLOCK_V) instead of (1, V)."""
        if tensor_name == "logits":
            return (1, static_dims["block_v"])
        return None

    def __init__(self, **config):
        super().__init__(**config)
        self.page_size = getattr(self, "page_size", DEFAULT_PAGE_SIZE)
        self.ignore_index = getattr(self, "ignore_index", -100)

        if self.logits_dtype in (cutlass.Float16, cutlass.BFloat16):
            self.elem_bytes = 2
        else:
            self.elem_bytes = 4

        # Thread config: 16 compute warps = 512 threads
        self.num_mma_warps = 16
        self.num_mma_threads = self.num_mma_warps * 32
        self.effective_threads = self.num_mma_threads
        self.effective_warps = self.num_mma_warps

        # V-block layout
        self.BLOCK_V, self.buf_bytes, meta_bytes = _compute_block_v(
            self.page_size, self.elem_bytes, V=self.V,
            num_warps=self.num_mma_warps)
        self.n_v_blocks = (self.V + self.BLOCK_V - 1) // self.BLOCK_V

        # Elements per thread per V-block
        self.elems_per_thread = (
            (self.BLOCK_V + self.effective_threads - 1)
            // self.effective_threads
        )

        # Smem offsets: buf0 | buf1 | mbarriers | scratch
        self.mbar_offset = self.buf_bytes * 2
        self.scratch_offset = self.mbar_offset + 32  # after 4×8 mbarriers

        # Inner iters for DMA: pass 1 + pass 2
        self.inner_iters = 2 * self.n_v_blocks
        self.inner_depth = 2

        # Override compute and load methods
        self.compute = self.compute_ce
        self.load = self.load_logits

    # =========================================================================
    # Scheduling
    # =========================================================================

    @classmethod
    def schedule_forward(cls, page_size=None, ignore_index=-100,
                         tile_sizes=None, **tensors):
        """Schedule cross-entropy Op."""
        if page_size is None:
            page_size = DEFAULT_PAGE_SIZE
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("BT", 1)

        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        ops[0].static_dims["page_size"] = page_size
        ops[0].static_dims["ignore_index"] = ignore_index

        # Compute BLOCK_V for TMA tile shape
        import torch
        logits_t = tensors["logits"]
        elem_bytes = 2 if logits_t.dtype in (torch.bfloat16, torch.float16) else 4
        V = ops[0].static_dims["V"]
        block_v, _, _ = _compute_block_v(page_size, elem_bytes, V=V)
        ops[0].static_dims["block_v"] = block_v
        ops[0].static_dims["inner_depth"] = 2

        return ops

    @classmethod
    def kernel_config(cls, ops):
        """Return recommended MegakernelConfig."""
        from machete.megakernel import MegakernelConfig
        from machete.megakernel.megakernel import NUM_DMA_WARPS

        compute_threads = 512  # 16 warps
        threads_per_block = compute_threads + NUM_DMA_WARPS * 32
        page_size = ops[0].static_dims.get("page_size", DEFAULT_PAGE_SIZE)
        return MegakernelConfig(
            threads_per_block=threads_per_block,
            page_size=page_size,
        )

    # =========================================================================
    # TMA Load — V-block chunking
    # =========================================================================

    @cute.jit
    def load_logits(self, page_ptr, tile_BT,
                    logits_tma, logits_tma_gmem,
                    work_mbar, inner_iter_idx):
        """TMA load dispatched by inner_iter_idx.

        iter 0:   Init mbarriers. TMA logits V-block 0 → buf 0 (work_mbar).
        iter k>0: Wait smem_consumed[buf], TMA V-block → buf (kblock_ready[buf]).
        """
        buf_idx = inner_iter_idx % Int32(2)
        buf_base = page_ptr + buf_idx * Int32(self.buf_bytes)
        v_block = inner_iter_idx % Int32(self.n_v_blocks)

        # Op-managed mbarrier addresses
        _sc_0 = page_ptr + Int32(self.mbar_offset)
        _sc_1 = page_ptr + Int32(self.mbar_offset + 8)
        _kr_0 = page_ptr + Int32(self.mbar_offset + 16)
        _kr_1 = page_ptr + Int32(self.mbar_offset + 24)

        if inner_iter_idx == Int32(0):
            # --- Init mbarriers ---
            with cute.arch.elect_one():
                mbarrier_init(_sc_0, Int32(1))
                mbarrier_init(_sc_1, Int32(1))
                mbarrier_init(_kr_0, Int32(1))
                mbarrier_init(_kr_1, Int32(1))
            mbarrier_init_fence()
            # Pre-arrive smem_consumed[1]: buf 1 starts empty
            with cute.arch.elect_one():
                mbarrier_arrive(_sc_1)

            # TMA load V-block 0 → buf 0 using framework work_mbar
            sL = cute.make_tensor(
                cute.make_ptr(self.logits_dtype, buf_base,
                              cute.AddressSpace.smem),
                cute.make_layout((self.BLOCK_V, 1)),
            )
            gL = cute.local_tile(
                logits_tma_gmem, (self.BLOCK_V, 1), (None, None),
            )
            tLsL, tLgL = cute.nvgpu.cpasync.tma_partition(
                logits_tma, Int32(0), cute.make_layout(1),
                cute.group_modes(sL, 0, 2),
                cute.group_modes(gL, 0, 2),
            )
            mbar_ptr = cute.make_ptr(
                cutlass.Int64, work_mbar, cute.AddressSpace.smem)
            nbytes = Int32(self.buf_bytes)
            with cute.arch.elect_one():
                mbarrier_arrive_expect_tx(work_mbar, nbytes)
            cute.copy(logits_tma,
                      tLgL[(None, v_block, tile_BT)],
                      tLsL, tma_bar_ptr=mbar_ptr)

        if inner_iter_idx > Int32(0):
            # Wait smem_consumed[buf] — compute freed this buffer
            sc_phase = ((inner_iter_idx - Int32(1)) // Int32(2)) % Int32(2)
            if buf_idx == Int32(0):
                mbarrier_wait(_sc_0, sc_phase)
            if buf_idx == Int32(1):
                mbarrier_wait(_sc_1, sc_phase)

            # TMA load V-block → buf, signal kblock_ready[buf]
            sL = cute.make_tensor(
                cute.make_ptr(self.logits_dtype, buf_base,
                              cute.AddressSpace.smem),
                cute.make_layout((self.BLOCK_V, 1)),
            )
            gL = cute.local_tile(
                logits_tma_gmem, (self.BLOCK_V, 1), (None, None),
            )
            tLsL, tLgL = cute.nvgpu.cpasync.tma_partition(
                logits_tma, Int32(0), cute.make_layout(1),
                cute.group_modes(sL, 0, 2),
                cute.group_modes(gL, 0, 2),
            )
            nbytes = Int32(self.buf_bytes)
            if buf_idx == Int32(0):
                _kr_ptr = cute.make_ptr(
                    cutlass.Int64, _kr_0, cute.AddressSpace.smem)
                with cute.arch.elect_one():
                    mbarrier_arrive_expect_tx(_kr_0, nbytes)
                cute.copy(logits_tma,
                          tLgL[(None, v_block, tile_BT)],
                          tLsL, tma_bar_ptr=_kr_ptr)
            if buf_idx == Int32(1):
                _kr_ptr = cute.make_ptr(
                    cutlass.Int64, _kr_1, cute.AddressSpace.smem)
                with cute.arch.elect_one():
                    mbarrier_arrive_expect_tx(_kr_1, nbytes)
                cute.copy(logits_tma,
                          tLgL[(None, v_block, tile_BT)],
                          tLsL, tma_bar_ptr=_kr_ptr)

    # =========================================================================
    # Compute — Fused Forward + Backward with TMA V-blocks
    # =========================================================================

    @cute.jit
    def compute_ce(self, page_ptr, tile_BT,
                   logits, targets, loss, grad_logits):
        """Cross-entropy forward+backward for one row.

        Pass 1: Online softmax — streaming max + sum over V-blocks from smem.
        Pass 2: Compute softmax gradients from V-blocks, write to global.
        """
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()

        if warp_idx < Int32(self.num_mma_warps):
            _kr_0 = page_ptr + Int32(self.mbar_offset + 16)
            _kr_1 = page_ptr + Int32(self.mbar_offset + 24)
            _sc_0 = page_ptr + Int32(self.mbar_offset)
            _sc_1 = page_ptr + Int32(self.mbar_offset + 8)

            scratch = cute.make_tensor(
                cute.make_ptr(Float32,
                              page_ptr + Int32(self.scratch_offset),
                              cute.AddressSpace.smem, assumed_align=16),
                cute.make_layout(self.effective_warps),
            )
            row = tile_BT

            # Load target from global
            target_g = cute.make_tensor(
                targets.iterator, cute.make_layout(self.BT),
            )
            target = target_g[row]

            # Global grad row and loss tensors
            grad_row = cute.make_tensor(
                (grad_logits.iterator + row * Int32(self.V)).align(16),
                cute.make_layout(self.V),
            )
            loss_g = cute.make_tensor(
                loss.iterator, cute.make_layout(self.BT),
            )

            # ====== Pass 1: Online softmax over V-blocks ======
            local_max = Float32(-1e30)
            local_sum = Float32(0.0)

            vb = Int32(0)
            while vb < Int32(self.n_v_blocks):
                buf = vb % Int32(2)
                buf_base = page_ptr + buf * Int32(self.buf_bytes)

                # Wait for V-block: iter 0 used work_mbar (framework),
                # iter 1+ use kblock_ready
                if vb > Int32(0):
                    kr_phase = ((vb - Int32(1)) // Int32(2)) % Int32(2)
                    if buf == Int32(0):
                        mbarrier_wait(_kr_0, kr_phase)
                    if buf == Int32(1):
                        mbarrier_wait(_kr_1, kr_phase)

                # Read V-block from smem
                sL = cute.make_tensor(
                    cute.make_ptr(self.logits_dtype, buf_base,
                                  cute.AddressSpace.smem),
                    cute.make_layout(self.BLOCK_V),
                )
                v_base = vb * Int32(self.BLOCK_V)

                lidx = tidx
                while lidx < Int32(self.BLOCK_V):
                    v_idx = v_base + lidx
                    if v_idx < Int32(self.V):
                        val = sL[lidx].to(Float32)
                        new_max = cute.arch.fmax(local_max, val)
                        local_sum = (
                            local_sum
                            * cute.math.exp(
                                local_max - new_max, fastmath=True)
                            + cute.math.exp(
                                val - new_max, fastmath=True)
                        )
                        local_max = new_max
                    lidx = lidx + Int32(self.effective_threads)

                # Signal buffer consumed
                named_barrier_sync(
                    Int32(2), Int32(self.effective_threads))
                if tidx == Int32(0):
                    if buf == Int32(0):
                        mbarrier_arrive(_sc_0)
                    if buf == Int32(1):
                        mbarrier_arrive(_sc_1)

                vb = vb + Int32(1)

            # --- Cross-warp max reduction ---
            thread_max = local_max
            warp_max = cute.arch.warp_reduction_max(local_max)
            if lane_idx == 0:
                scratch[warp_idx] = warp_max
            named_barrier_sync(Int32(2), Int32(self.effective_threads))

            global_max = scratch[Int32(0)]
            for wi in range(1, self.effective_warps):
                global_max = cute.arch.fmax(global_max, scratch[wi])
            named_barrier_sync(Int32(2), Int32(self.effective_threads))

            # --- Correct sum with global max ---
            corrected_sum = local_sum * cute.math.exp(
                thread_max - global_max, fastmath=True)
            warp_sum = cute.arch.warp_reduction(
                corrected_sum, operator.add)
            if lane_idx == 0:
                scratch[warp_idx] = warp_sum
            named_barrier_sync(Int32(2), Int32(self.effective_threads))

            global_sum = Float32(0.0)
            for wi in range(self.effective_warps):
                global_sum = global_sum + scratch[wi]
            named_barrier_sync(Int32(2), Int32(self.effective_threads))

            # --- LSE and loss ---
            lse = global_max + cute.math.log(global_sum, fastmath=True)

            # Read target logit from global
            logits_row_g = cute.make_tensor(
                (logits.iterator + row * Int32(self.V)).align(16),
                cute.make_layout(self.V),
            )
            target_logit = Float32(0.0)
            if target >= Int32(0):
                target_logit = logits_row_g[target].to(Float32)

            if tidx == Int32(0):
                if target != Int32(self.ignore_index):
                    loss_g[row] = lse - target_logit
                else:
                    loss_g[row] = Float32(0.0)

            # ====== Pass 2: Gradients from V-blocks ======
            vb2 = Int32(0)
            while vb2 < Int32(self.n_v_blocks):
                # Global iter for pass 2
                g_iter = Int32(self.n_v_blocks) + vb2
                buf2 = g_iter % Int32(2)
                buf_base2 = page_ptr + buf2 * Int32(self.buf_bytes)

                # Wait kblock_ready
                kr_phase2 = (
                    (g_iter - Int32(1)) // Int32(2)
                ) % Int32(2)
                if buf2 == Int32(0):
                    mbarrier_wait(_kr_0, kr_phase2)
                if buf2 == Int32(1):
                    mbarrier_wait(_kr_1, kr_phase2)

                sL2 = cute.make_tensor(
                    cute.make_ptr(self.logits_dtype, buf_base2,
                                  cute.AddressSpace.smem),
                    cute.make_layout(self.BLOCK_V),
                )
                v_base2 = vb2 * Int32(self.BLOCK_V)

                lidx2 = tidx
                while lidx2 < Int32(self.BLOCK_V):
                    v_idx2 = v_base2 + lidx2
                    if v_idx2 < Int32(self.V):
                        if target != Int32(self.ignore_index):
                            val = sL2[lidx2].to(Float32)
                            softmax_val = cute.math.exp(
                                val - lse, fastmath=True)
                            grad_val = softmax_val
                            if v_idx2 == target:
                                grad_val = grad_val - Float32(1.0)
                            grad_row[v_idx2] = grad_val.to(
                                self.logits_dtype)
                        else:
                            grad_row[v_idx2] = Float32(0.0).to(
                                self.logits_dtype)
                    lidx2 = lidx2 + Int32(self.effective_threads)

                # Signal buffer consumed
                named_barrier_sync(
                    Int32(2), Int32(self.effective_threads))
                if tidx == Int32(0):
                    if buf2 == Int32(0):
                        mbarrier_arrive(_sc_0)
                    if buf2 == Int32(1):
                        mbarrier_arrive(_sc_1)

                vb2 = vb2 + Int32(1)


__all__ = ["CrossEntropyOp"]
