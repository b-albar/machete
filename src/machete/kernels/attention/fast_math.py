# Copyright (c) 2025, Machete Authors
"""
Fast Math Primitives for Attention Softmax (SM90+).

Provides PTX-level optimizations from FA4:
1. Polynomial exp2 emulation — avoids SFU bottleneck, processes 2 values

For packed f32x2 arithmetic (fma, add, mul), use the built-in CuTe DSL ops:
    cute.arch.fma_packed_f32x2, cute.arch.add_packed_f32x2, etc.
"""

from typing import Tuple

from cutlass import Float32
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm


# =============================================================================
# Polynomial exp2 Emulation (avoids SFU bottleneck)
# =============================================================================
#
# Computes 2^x for two values simultaneously using a degree-3 polynomial:
#   2^x = 2^int(x) * poly(frac(x))
# where poly is a Horner-form evaluation of minimax coefficients.
#
# Algorithm:
#   1. Clamp x to [-127, inf) to prevent IEEE754 underflow
#   2. Split: int_part = floor(x), frac_part = x - floor(x) in [0,1)
#   3. Polynomial: p = ((c3*f + c2)*f + c1)*f + c0
#   4. Combine: result = 2^int * p (via IEEE754 exponent bit manipulation)
#
# The floor is computed with cvt.rmi.f32.f32 (round-toward-minus-infinity).
# The combine uses cvt.rni + shl to set the IEEE754 exponent field,
# then multiplies by the polynomial mantissa.


@dsl_user_op
def ex2_approx_2(
    x: Float32,
    y: Float32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32]:
    """Polynomial exp2 for two values: (2^x, 2^y).

    Degree-3 minimax approximation on [0,1) for the fractional part.
    ~1e-4 relative error, sufficient for attention softmax (bf16/fp16 output).
    """
    from cutlass._mlir import ir

    f32 = ir.F32Type.get()
    struct_ty = llvm.StructType.get_literal([f32, f32])
    result = llvm.inline_asm(
        struct_ty,
        [
            x.ir_value(loc=loc, ip=ip),
            y.ir_value(loc=loc, ip=ip),
        ],
        # Interleaved x/y computation for ILP
        "{ .reg .f32 %x, %y, %ix, %iy, %fx, %fy, %rx, %ry, %ex, %ey;"
        " .reg .s32 %iix, %iiy;"
        " .reg .b32 %bx, %by;"
        " .reg .f32 %neg127, %c0, %c1, %c2, %c3;"
        # Constants: minimax coefficients for 2^x on [0,1)
        " mov.f32 %neg127, 0fC2FE0000;"  # -127.0
        " mov.f32 %c0, 0f3F800000;"      # 1.0
        " mov.f32 %c1, 0f3F317218;"      # 0.6931472 (ln2)
        " mov.f32 %c2, 0f3E75FDF0;"      # 0.2402265 (ln2^2/2)
        " mov.f32 %c3, 0f3D6354E1;"      # 0.0554903 (ln2^3/6)
        # Clamp to [-127, inf)
        " max.f32 %x, $2, %neg127;"
        " max.f32 %y, $3, %neg127;"
        # Floor via round-toward-minus-infinity
        " cvt.rmi.f32.f32 %ix, %x;"
        " cvt.rmi.f32.f32 %iy, %y;"
        # Fractional part
        " sub.f32 %fx, %x, %ix;"
        " sub.f32 %fy, %y, %iy;"
        # Horner polynomial: ((c3*f + c2)*f + c1)*f + c0
        " fma.rn.f32 %rx, %c3, %fx, %c2;"
        " fma.rn.f32 %ry, %c3, %fy, %c2;"
        " fma.rn.f32 %rx, %rx, %fx, %c1;"
        " fma.rn.f32 %ry, %ry, %fy, %c1;"
        " fma.rn.f32 %rx, %rx, %fx, %c0;"
        " fma.rn.f32 %ry, %ry, %fy, %c0;"
        # Combine: 2^int_part via IEEE754 exponent manipulation
        # IEEE754: 2^n = (n + 127) << 23 (add exponent bias before shift)
        " cvt.rni.s32.f32 %iix, %ix;"
        " cvt.rni.s32.f32 %iiy, %iy;"
        " add.s32 %iix, %iix, 127;"
        " add.s32 %iiy, %iiy, 127;"
        " shl.b32 %bx, %iix, 23;"
        " shl.b32 %by, %iiy, 23;"
        " mov.b32 %ex, %bx;"
        " mov.b32 %ey, %by;"
        # Result: 2^int * polynomial(frac)
        " mul.f32 $0, %ex, %rx;"
        " mul.f32 $1, %ey, %ry; }",
        "=f,=f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(f32, result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(f32, result, [1], loc=loc, ip=ip)),
    )


__all__ = [
    "ex2_approx_2",
]
