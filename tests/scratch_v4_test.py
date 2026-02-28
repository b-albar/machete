#!/usr/bin/env python
"""Test: can @cute.jit return a tuple? And can a dsl_user_op do 2 v4 loads?"""

import torch
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass._mlir import ir as mlir_ir
import cuda.bindings.driver as cuda_driver
from typing import Tuple


@dsl_user_op
def get_smem_base_ptr(*, loc=None, ip=None) -> Int32:
    result = llvm.inline_asm(
        mlir_ir.IntegerType.get_signless(32), [],
        "{ .reg .u64 p; cvta.shared.u64 p, 0; cvt.u32.u64 $0, p; }", "=r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return Int32(result)

@dsl_user_op
def st_shared_i32(addr: Int32, val: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None, [addr.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
        "st.shared.b32 [$0], $1;", "r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT)

@dsl_user_op
def ld_shared_i32(addr: Int32, *, loc=None, ip=None) -> Int32:
    i32 = mlir_ir.IntegerType.get_signless(32)
    r = llvm.inline_asm(
        i32, [addr.ir_value(loc=loc, ip=ip)],
        "ld.shared.b32 $0, [$1];", "=r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return Int32(r)

@dsl_user_op
def st_shared_v4_b32(addr: Int32, v0: Int32, v1: Int32, v2: Int32, v3: Int32,
                      *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [addr.ir_value(loc=loc, ip=ip),
         v0.ir_value(loc=loc, ip=ip), v1.ir_value(loc=loc, ip=ip),
         v2.ir_value(loc=loc, ip=ip), v3.ir_value(loc=loc, ip=ip)],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};", "r,r,r,r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT)

@dsl_user_op
def ld_shared_v4_b32(addr: Int32, *, loc=None, ip=None
                      ) -> Tuple[Int32, Int32, Int32, Int32]:
    i32 = mlir_ir.IntegerType.get_signless(32)
    struct_ty = llvm.StructType.get_literal([i32, i32, i32, i32])
    r = llvm.inline_asm(
        struct_ty, [addr.ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (Int32(llvm.extractvalue(i32, r, [0], loc=loc, ip=ip)),
            Int32(llvm.extractvalue(i32, r, [1], loc=loc, ip=ip)),
            Int32(llvm.extractvalue(i32, r, [2], loc=loc, ip=ip)),
            Int32(llvm.extractvalue(i32, r, [3], loc=loc, ip=ip)))


# =========================================================================
# Test 1: @cute.jit helper returning a tuple (calls v4 load internally)
# =========================================================================

@cute.jit
def read_tile_info(addr: Int32) -> Tuple[Int32, Int32, Int32, Int32, Int32, Int32]:
    """Read 6-field tile info from smem using two v4 loads (slot = 32 bytes)."""
    a, b, c, d = ld_shared_v4_b32(addr)
    e, f, _pad0, _pad1 = ld_shared_v4_b32(addr + 16)
    return a, b, c, d, e, f


@cute.jit
def write_tile_info(addr: Int32, v0: Int32, v1: Int32, v2: Int32,
                    v3: Int32, v4: Int32, v5: Int32):
    """Write 6-field tile info to smem using two v4 stores (slot = 32 bytes)."""
    st_shared_v4_b32(addr, v0, v1, v2, v3)
    st_shared_v4_b32(addr + 16, v4, v5, Int32(0), Int32(0))


# =========================================================================
# Kernel
# =========================================================================

class TestRunner:
    def __init__(self):
        self.smem_size = 256

    @cute.jit
    def __call__(self, out_ptr, stream):
        self.kernel(out_ptr).launch(
            grid=[1, 1, 1], block=[32, 1, 1],
            smem=self.smem_size, stream=stream)

    @cute.kernel
    def kernel(self, out_ptr):
        tid = cute.arch.thread_idx()[0]
        smem = get_smem_base_ptr()

        if tid == Int32(0):
            # Write with helper
            write_tile_info(smem, Int32(10), Int32(20), Int32(30),
                           Int32(40), Int32(50), Int32(60))
            cute.arch.fence_view_async_shared()

            # Read back with helper
            op, t0, t1, t2, t3, t4 = read_tile_info(smem)
            out_ptr[0] = op
            out_ptr[1] = t0
            out_ptr[2] = t1
            out_ptr[3] = t2
            out_ptr[4] = t3
            out_ptr[5] = t4


if __name__ == "__main__":
    out = torch.zeros(8, dtype=torch.int32, device="cuda")
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)

    print("Compiling...")
    try:
        runner = TestRunner()
        compiled = cute.compile(runner, out, stream)
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback; traceback.print_exc()
        exit(1)

    print("Running...")
    compiled(out, stream)
    torch.cuda.synchronize()

    r = out.tolist()[:6]
    print(f"Results: {r}")
    expected = [10, 20, 30, 40, 50, 60]
    if r == expected:
        print("  @cute.jit tuple return + v4 helpers: PASS")
        print("\nAll passed!")
    else:
        print(f"  FAIL: expected {expected}")
        exit(1)
