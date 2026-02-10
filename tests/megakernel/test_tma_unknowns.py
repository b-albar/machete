# Copyright (c) 2025, Machete Authors
"""
Investigate two unknowns for megakernel TMA integration:

Unknown 1: Does cute.copy(..., mbar_ptr=Int32) accept a raw Int32 smem address?
           (instead of SmemAllocator handle)

Unknown 2: Does cute.make_copy_atom() work inside a @cute.jit sub-function
           that gets inlined into @cute.kernel? (mimics compile_phase pattern)
"""

import pytest
import torch
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu.cpasync import (
    CopyBulkG2SOp,
    CopyBulkS2GOp,
    group_bulk_copy_modes,
)
import cuda.bindings.driver as cuda

from machete.utils.testing import is_hopper_available
from machete.megakernel.interpreter import (
    get_smem_base_ptr,
    mbarrier_init,
    mbarrier_init_fence,
    mbarrier_arrive_expect_tx,
    mbarrier_wait,
)

requires_hopper = pytest.mark.skipif(
    not is_hopper_available(), reason="Requires Hopper (SM90+) GPU",
)

THREADS = 128
TILE = 256


def _make_stream():
    s = torch.cuda.Stream()
    return s, cuda.CUstream(s.cuda_stream)


# =============================================================================
# Unknown 1: cute.copy with Int32 mbar_ptr
# =============================================================================

class BulkCopyInt32Mbar:
    """Same as test_tma_ops.BulkCopyG2S but using raw Int32 smem address for mbar."""

    @cute.jit
    def __call__(self, inp: cute.Tensor, out: cute.Tensor, stream: cuda.CUstream):
        atom = cute.make_copy_atom(
            CopyBulkG2SOp(), cutlass.Float16, num_bits_per_copy=TILE * 16,
        )
        self.kernel(inp, out, atom).launch(
            grid=[1, 1, 1], block=[THREADS, 1, 1],
            smem=TILE * 2 + 256, stream=stream,
        )

    @cute.kernel
    def kernel(self, inp: cute.Tensor, out: cute.Tensor, atom: cute.CopyAtom):
        tidx = cute.arch.thread_idx()[0]
        smem_base = get_smem_base_ptr()

        # Mbar at offset 0, data at offset 128 (128B-aligned)
        mbar_addr = smem_base           # Int32!
        data_addr = smem_base + Int32(128)

        sdata = cute.make_tensor(
            cute.make_ptr(cutlass.Float16, data_addr, cute.AddressSpace.smem),
            cute.make_layout((TILE,)),
        )
        gdata = cute.make_tensor(inp.iterator, cute.make_layout((TILE,)))

        num_bytes = TILE * 2
        if tidx == 0:
            mbarrier_init(mbar_addr, Int32(1))
            mbarrier_arrive_expect_tx(mbar_addr, Int32(num_bytes))
        mbarrier_init_fence()
        cute.arch.barrier()

        # KEY TEST: create Pointer from Int32, pass to cute.copy
        mbar_ptr = cute.make_ptr(cutlass.Int64, mbar_addr, cute.AddressSpace.smem)
        if tidx == 0:
            gsrc, sdst = group_bulk_copy_modes(gdata, sdata)
            cute.copy(atom, gsrc, sdst, mbar_ptr=mbar_ptr)

        mbarrier_wait(mbar_addr, Int32(0))

        # Read back to verify
        per_thread = TILE // THREADS
        for i in range(per_thread):
            idx = tidx + Int32(i * THREADS)
            out[idx] = sdata[idx]


# =============================================================================
# Unknown 2: cute.make_copy_atom inside @cute.jit sub-function
# (mimics compile_phase: body wrapped in @cute.jit, called from @cute.kernel)
# =============================================================================

class BulkCopyAtomInSubJit:
    """Create the CopyAtom inside a @cute.jit function called from the kernel."""

    @cute.jit
    def __call__(self, inp: cute.Tensor, out: cute.Tensor, stream: cuda.CUstream):
        self.kernel(inp, out).launch(
            grid=[1, 1, 1], block=[THREADS, 1, 1],
            smem=TILE * 2 + 256, stream=stream,
        )

    @cute.kernel
    def kernel(self, inp: cute.Tensor, out: cute.Tensor):
        tidx = cute.arch.thread_idx()[0]

        # KEY TEST: create atom inside kernel context (not passed as parameter)
        atom = cute.make_copy_atom(
            CopyBulkG2SOp(), cutlass.Float16, num_bits_per_copy=TILE * 16,
        )

        smem = cutlass.utils.SmemAllocator()
        mbar = smem.allocate(cutlass.Int64)
        sdata = smem.allocate_tensor(
            cutlass.Float16,
            cute.make_layout((TILE,)),
            byte_alignment=128,
        )
        gdata = cute.make_tensor(inp.iterator, cute.make_layout((TILE,)))

        num_bytes = TILE * 2
        if tidx == 0:
            cute.arch.mbarrier_init(mbar, 1)
            cute.arch.mbarrier_arrive_and_expect_tx(mbar, num_bytes)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        if tidx == 0:
            gsrc, sdst = group_bulk_copy_modes(gdata, sdata)
            cute.copy(atom, gsrc, sdst, mbar_ptr=mbar)

        cute.arch.mbarrier_wait(mbar, 0)

        per_thread = TILE // THREADS
        for i in range(per_thread):
            idx = tidx + Int32(i * THREADS)
            out[idx] = sdata[idx]


# =============================================================================
# Unknown 2b: cute.make_copy_atom inside an exec'd @cute.jit (mimics compile_phase exactly)
# =============================================================================

def make_sub_jit_via_exec():
    """Create a @cute.jit function via exec(), like compile_phase does.

    Key: register source in linecache so inspect.getsourcelines works
    (the DSL preprocessor needs it for AST transformation).
    """
    import linecache

    filename = "<test_exec_jit_phase>"
    src = """\
@cute.jit
def phase_fn(inp, out, mbar_addr):
    tidx = cute.arch.thread_idx()[0]
    atom = cute.make_copy_atom(
        CopyBulkG2SOp(), cutlass.Float16, num_bits_per_copy={num_bits},
    )
    sdata = cute.make_tensor(
        cute.make_ptr(cutlass.Float16, mbar_addr + Int32(128), cute.AddressSpace.smem),
        cute.make_layout(({tile},)),
    )
    gdata = cute.make_tensor(inp.iterator, cute.make_layout(({tile},)))

    mbar_ptr = cute.make_ptr(cutlass.Int64, mbar_addr, cute.AddressSpace.smem)
    if tidx == 0:
        mbarrier_init(mbar_addr, Int32(1))
        mbarrier_arrive_expect_tx(mbar_addr, Int32({nbytes}))
    mbarrier_init_fence()
    cute.arch.barrier()

    if tidx == 0:
        gsrc, sdst = group_bulk_copy_modes(gdata, sdata)
        cute.copy(atom, gsrc, sdst, mbar_ptr=mbar_ptr)

    mbarrier_wait(mbar_addr, Int32(0))

    per_thread = {tile} // {threads}
    for i in range(per_thread):
        idx = tidx + Int32(i * {threads})
        out[idx] = sdata[idx]
""".format(num_bits=TILE * 16, tile=TILE, nbytes=TILE * 2, threads=THREADS)

    # Register in linecache so DSL preprocessor can find the source
    linecache.cache[filename] = (
        len(src), None, src.splitlines(True), filename,
    )

    exec_globals = {
        "cute": cute, "cutlass": cutlass, "Int32": Int32, "Int64": Int64,
        "CopyBulkG2SOp": CopyBulkG2SOp, "group_bulk_copy_modes": group_bulk_copy_modes,
        "mbarrier_init": mbarrier_init, "mbarrier_init_fence": mbarrier_init_fence,
        "mbarrier_wait": mbarrier_wait, "mbarrier_arrive_expect_tx": mbarrier_arrive_expect_tx,
    }
    exec(compile(src, filename, "exec"), exec_globals)
    return exec_globals["phase_fn"]


class BulkCopyExecJit:
    """Test 2b: CopyAtom + Int32 mbar inside exec'd @cute.jit (closest to compile_phase)."""

    def __init__(self):
        self.phase_fn = make_sub_jit_via_exec()

    @cute.jit
    def __call__(self, inp: cute.Tensor, out: cute.Tensor, stream: cuda.CUstream):
        self.kernel(inp, out).launch(
            grid=[1, 1, 1], block=[THREADS, 1, 1],
            smem=TILE * 2 + 256, stream=stream,
        )

    @cute.kernel
    def kernel(self, inp: cute.Tensor, out: cute.Tensor):
        mbar_addr = get_smem_base_ptr()
        self.phase_fn(inp, out, mbar_addr)


# =============================================================================
# Test Functions
# =============================================================================

@requires_hopper
def test_unknown1_int32_mbar():
    """Unknown 1: cute.copy with raw Int32 mbar_ptr."""
    inp = torch.randn(TILE, dtype=torch.float16, device="cuda")
    out = torch.zeros(TILE, dtype=torch.float16, device="cuda")
    inp_ct = from_dlpack(inp, assumed_align=16)
    out_ct = from_dlpack(out, assumed_align=16)
    _, cu = _make_stream()

    kernel = BulkCopyInt32Mbar()
    kernel(inp_ct, out_ct, cu)
    torch.cuda.synchronize()

    diff = (out - inp).abs().max().item()
    print(f"Unknown 1 (Int32 mbar): max diff = {diff}")
    assert diff < 1e-5, f"Failed: max diff {diff}"


@requires_hopper
def test_unknown2_atom_in_kernel():
    """Unknown 2: cute.make_copy_atom inside @cute.kernel (not passed as param)."""
    inp = torch.randn(TILE, dtype=torch.float16, device="cuda")
    out = torch.zeros(TILE, dtype=torch.float16, device="cuda")
    inp_ct = from_dlpack(inp, assumed_align=16)
    out_ct = from_dlpack(out, assumed_align=16)
    _, cu = _make_stream()

    kernel = BulkCopyAtomInSubJit()
    kernel(inp_ct, out_ct, cu)
    torch.cuda.synchronize()

    diff = (out - inp).abs().max().item()
    print(f"Unknown 2 (atom in kernel): max diff = {diff}")
    assert diff < 1e-5, f"Failed: max diff {diff}"


@requires_hopper
def test_unknown2b_atom_in_exec_jit():
    """Unknown 2b: CopyAtom + Int32 mbar in exec'd @cute.jit (mimics compile_phase)."""
    inp = torch.randn(TILE, dtype=torch.float16, device="cuda")
    out = torch.zeros(TILE, dtype=torch.float16, device="cuda")
    inp_ct = from_dlpack(inp, assumed_align=16)
    out_ct = from_dlpack(out, assumed_align=16)
    _, cu = _make_stream()

    kernel = BulkCopyExecJit()
    kernel(inp_ct, out_ct, cu)
    torch.cuda.synchronize()

    diff = (out - inp).abs().max().item()
    print(f"Unknown 2b (exec'd jit + Int32 mbar): max diff = {diff}")
    assert diff < 1e-5, f"Failed: max diff {diff}"
