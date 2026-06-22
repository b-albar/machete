"""llama.cpp-style MXFP4 decode matvec: int8-quantized activation (q8_1) x fp4
weight via `dp4a` integer dot, vs machete's current hardware-e2m1 f16x2 dot.

llama's `vec_dot_mxfp4_q8_1`: quantize activation -> int8 (per-32 scale), fp4
weight -> int8 via the `kvalues_mxfp4` table (= 2x the fp4 value), `dp4a` dot,
then result = e8m0_scale * 0.5 * act_scale * sumi. This builds that path in
machete and measures it head-to-head against the existing f16x2 MXFP4 matvec.
"""
import os
os.environ.setdefault("CUTE_DSL_ARCH", "sm_120a")
import argparse, io, contextlib, operator
import numpy as np
import torch

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from machete.megakernel import Megakernel, MegakernelConfig
from machete.megakernel.megakernel import NUM_DMA_WARPS
from machete.kernels.decode_matvec.sm120 import _DecodeMatvecSm120Base, _MatvecNvfp4Sm120Base
from machete.kernels.qwen_3_5.mxfp4_ops import _e8m0_byte_to_f32, MatvecMxfp4SimtSm120Op
from machete.quantization.mxfp4 import quantize_mxfp4_weight


# ---------------------------------------------------------------- dp4a dot-8
@dsl_user_op
def _mxfp4_dp4a_dot8(w: Int32, a_lo: Int32, a_hi: Int32, scale: Float32, *, loc=None, ip=None) -> Float32:
    """sumi = sum_{k=0..7} kvalues_mxfp4[code_k] * a_int8[k] via dp4a, then *scale.
    w = u32 of 8 packed fp4 codes; a_lo/a_hi = u32 of int8 activations [0..3]/[4..7];
    scale = e8m0_weight_scale * 0.5 * act_scale (kvalues = 2x fp4 value -> the *0.5)."""
    from cutlass._mlir import ir
    result = llvm.inline_asm(
        ir.F32Type.get(),
        [Int32(w).ir_value(loc=loc, ip=ip), Int32(a_lo).ir_value(loc=loc, ip=ip),
         Int32(a_hi).ir_value(loc=loc, ip=ip), Float32(scale).ir_value(loc=loc, ip=ip)],
        "{\n"
        ".reg .b32 q4, q4m, m, lhs, t0, t1, t2, t3, low, high, vx, vy;\n"
        ".reg .s32 sumi;\n"
        ".reg .f32 sf;\n"
        "mov.b32 q4, $1;\n"
        "mov.b32 t0, 0x03020100;\n"   # kvalues[0..3]  = {0,1,2,3}
        "mov.b32 t1, 0x0C080604;\n"   # kvalues[4..7]  = {4,6,8,12}
        "mov.b32 t2, 0xFDFEFF00;\n"   # kvalues[8..11] = {0,-1,-2,-3}
        "mov.b32 t3, 0xF4F8FAFC;\n"   # kvalues[12..15]= {-4,-6,-8,-12}
        "and.b32 q4m, q4, 0x77777777;\n"          # clear bit3 of each NIBBLE (PTX prmt selectors are per-nibble)
        "and.b32 m, q4, 0x88888888;\n"
        "shr.u32 m, m, 1;\n"
        "or.b32 lhs, m, 0x32103210;\n"            # pick low(0-3)/high(8-15) by bit3
        "prmt.b32 low, t0, t1, q4m;\n"
        "prmt.b32 high, t2, t3, q4m;\n"
        "prmt.b32 vx, low, high, lhs;\n"          # int8x4 for k=0..3
        "shr.u32 q4m, q4m, 16;\n"
        "shr.u32 lhs, lhs, 16;\n"
        "prmt.b32 low, t0, t1, q4m;\n"
        "prmt.b32 high, t2, t3, q4m;\n"
        "prmt.b32 vy, low, high, lhs;\n"          # int8x4 for k=4..7
        "dp4a.s32.s32 sumi, vx, $2, 0;\n"
        "dp4a.s32.s32 sumi, vy, $3, sumi;\n"
        "cvt.rn.f32.s32 sf, sumi;\n"
        "mul.rn.f32 $0, sf, $4;\n"
        "}\n",
        "=f,r,r,r,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return Float32(result)


class MatvecMXFP4Dp4aSm120Op(_DecodeMatvecSm120Base):
    """MXFP4 decode matvec, llama-style: int8 q8_1 activation x fp4 weight via dp4a."""
    pipeline = None  # set below
    reads = {
        "a_q8": (cutlass.Int8, ("B", "S", "K")),
        "a_scale": (cutlass.Float32, ("B", "S", "G")),
        "weight_packed": (cutlass.Uint8, ("O", "K2")),
        "weight_scales": (cutlass.Uint8, ("O", "G")),
    }
    writes = {"y": (None, ("B", "S", "O"))}
    tile = ("B", "S", "O")
    dynamic_dims = ("B",)

    def __init__(self, **config):
        super().__init__(**config)
        self.group_size = getattr(self, "group_size", 32)

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=49152, group_size=32, **tensors):
        ts = dict(tile_sizes or {})
        ts.setdefault("B", 1); ts.setdefault("S", 16); ts.setdefault("O", 16)
        op = cls._schedule_single(tile_sizes=ts, **tensors)
        op.static_dims["page_size"] = page_size
        op.static_dims["K"] = tensors["a_q8"].shape[-1]
        op.static_dims["group_size"] = group_size
        return [op]

    @cute.jit
    def _dot_dp4a(self, a_u32, a_scale_row, w_u32, scale_row):
        lane_idx = cute.arch.lane_idx()
        acc = Float32(0.0)
        full_k = Int32((self.K // 8) * 8)
        k = lane_idx * Int32(8)
        while k < full_k:
            wu = w_u32[k >> Int32(3)].to(Int32)
            a_lo = a_u32[k >> Int32(2)].to(Int32)
            a_hi = a_u32[(k >> Int32(2)) + Int32(1)].to(Int32)
            grp = k >> Int32(5)
            e8 = _e8m0_byte_to_f32(scale_row[grp].to(Int32))
            scale = e8 * a_scale_row[grp] * Float32(0.5)
            acc = acc + _mxfp4_dp4a_dot8(wu, a_lo, a_hi, scale)
            k = k + Int32(256)
        return cute.arch.warp_reduction(acc, operator.add)

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_O, a_q8, a_scale, weight_packed, weight_scales, y):
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.threads_per_row // 32
        row_start = tile_S * Int32(self.tile_size_S)
        out_start = tile_O * Int32(self.tile_size_O)
        for local_work in range(warp_idx, self.tile_size_S * self.tile_size_O, num_warps):
            local_row = local_work // self.tile_size_O
            local_o = local_work - local_row * self.tile_size_O
            row_idx = row_start + Int32(local_row)
            if row_idx < Int32(self.S):
                out_idx = out_start + Int32(local_o)
                if out_idx < Int32(self.O):
                    a_base = tile_B * Int32(self.a_q8_stride_B) + row_idx * Int32(self.a_q8_stride_S)
                    a_row = cute.make_tensor(a_q8.iterator + a_base, cute.make_layout(self.K))
                    a_u32 = cute.make_tensor(cute.recast_ptr(a_row.iterator, dtype=Int32), cute.make_layout(self.K // 4))
                    asc_base = tile_B * Int32(self.a_scale_stride_B) + row_idx * Int32(self.a_scale_stride_S)
                    a_scale_row = cute.make_tensor(a_scale.iterator + asc_base, cute.make_layout(self.G))
                    w_base = out_idx * Int32(self.weight_packed_stride_O)
                    w_row = cute.make_tensor(weight_packed.iterator + w_base, cute.make_layout(self.K // 2))
                    w_u32 = cute.make_tensor(cute.recast_ptr(w_row.iterator, dtype=Int32), cute.make_layout(self.K // 8))
                    s_base = out_idx * Int32(self.weight_scales_stride_O)
                    scale_row = cute.make_tensor(weight_scales.iterator + s_base, cute.make_layout(self.G))
                    total = self._dot_dp4a(a_u32, a_scale_row, w_u32, scale_row)
                    if lane_idx == Int32(0):
                        y_base = tile_B * Int32(self.y_stride_B) + row_idx * Int32(self.y_stride_S)
                        y_tile = cute.make_tensor(y.iterator + y_base + out_start, cute.make_layout(self.tile_size_O))
                        y_tile[local_o] = total.to(self.y_dtype)


def quantize_q8_1(x, group=32):
    """bf16/f32 activation -> int8 (per-32-group, scale=max|x|/127) + f32 scales."""
    K = x.shape[-1]; G = K // group
    xg = x.float().reshape(-1, G, group)
    amax = xg.abs().amax(dim=-1, keepdim=True)
    scale = (amax / 127.0).clamp(min=1e-12)
    q = torch.round(xg / scale).clamp(-127, 127).to(torch.int8)
    return q.reshape(x.shape).contiguous(), scale.squeeze(-1).reshape(x.shape[0], x.shape[1], G).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--O", type=int, default=2048)
    ap.add_argument("--K", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--only", choices=["dp4a", "f16x2"], default=None,
                    help="build/run only one kernel (for clean ncu profiling)")
    args = ap.parse_args()
    dev = "cuda"; torch.manual_seed(0)
    O, K, G = args.O, args.K, args.K // 32

    Wf = (torch.randn(O, K, device=dev) * 0.05)
    Wq = quantize_mxfp4_weight(Wf)          # .packed (O,K/2) uint8, .scales (O,G) uint8 E8M0
    x = torch.randn(1, 1, K, device=dev, dtype=torch.bfloat16)

    # reference: dequant(W) @ dequant(q8_1(x))
    a_q8, a_scale = quantize_q8_1(x)
    cfg = MegakernelConfig(threads_per_block=512, page_size=49152, mma_reg_count=96)

    def time_kernel(k):
        k.compile()
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(30): k.run(validate=False, sync=False)
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
            for _ in range(args.iters): k.run(validate=False, sync=False)
            e1.record(); torch.cuda.synchronize()
        return e0.elapsed_time(e1) / args.iters * 1000

    t_dp = t_f16 = float("nan"); y_dp = y_f16 = None
    # ---- dp4a matvec ----
    if args.only != "f16x2":
        y_dp = torch.zeros(1, 1, O, device=dev, dtype=torch.float32)
        ops = MatvecMXFP4Dp4aSm120Op.schedule(a_q8=a_q8, a_scale=a_scale, weight_packed=Wq.packed,
                                              weight_scales=Wq.scales_e8m0, y=y_dp, page_size=49152)
        kd = Megakernel(ops, config=cfg); kd._keep_alive = [a_q8, a_scale, Wq.packed, Wq.scales_e8m0, y_dp]
        t_dp = time_kernel(kd); torch.cuda.synchronize()

    # ---- current f16x2 MXFP4 SIMT matvec ----
    if args.only != "dp4a":
        y_f16 = torch.zeros(1, 1, O, device=dev, dtype=torch.float32)
        ops2 = MatvecMxfp4SimtSm120Op.schedule(a=x, weight_packed=Wq.packed, weight_scales=Wq.scales_e8m0, y=y_f16, page_size=49152)
        kf = Megakernel(ops2, config=cfg); kf._keep_alive = [x, Wq.packed, Wq.scales_e8m0, y_f16]
        t_f16 = time_kernel(kf); torch.cuda.synchronize()

    if args.only is not None:
        print(f"ran only {args.only}: t={t_dp if args.only=='dp4a' else t_f16:.2f}us (wall, may be contended)")
        return

    # reference (dequant W in fp32 @ x)
    kv = torch.tensor([0,0.5,1,1.5,2,3,4,6,0,-0.5,-1,-1.5,-2,-3,-4,-6], device=dev)
    codes = torch.stack([(Wq.packed & 0xF), (Wq.packed >> 4)], dim=-1).reshape(O, K).long()
    e8 = (Wq.scales_e8m0.to(torch.int32).reshape(O, G, 1) << 23).view(torch.float32) if False else None
    sc = torch.pow(2.0, (Wq.scales_e8m0.float() - 127.0)).reshape(O, G, 1)
    Wdq = (kv[codes].reshape(O, G, 32) * sc).reshape(O, K)
    y_ref = (Wdq @ x.float().reshape(K, 1)).reshape(1, 1, O)

    # dp4a-math reference (numpy): exactly what the dp4a kernel SHOULD compute
    tbl = torch.tensor([0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12], device=dev, dtype=torch.float32)
    wint = tbl[codes].reshape(O, G, 32)                          # 2x fp4 value, int8
    aq = a_q8.float().reshape(1, G, 32)
    sumi = (wint * aq).sum(-1)                                   # (O,G)
    e8v = torch.pow(2.0, (Wq.scales_e8m0.float() - 127.0))       # (O,G)
    y_dpref = (e8v * 0.5 * a_scale.reshape(1, G) * sumi).sum(-1).reshape(1, 1, O)

    def stats(y, name, ref=y_ref):
        err = (y - ref).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(y.flatten(), ref.flatten(), dim=0).item()
        print(f"  {name:18s}: max|err|={err:.3e}  cos={cos:.6f}")
    print(f"MXFP4 matvec O={O} K={K}:")
    stats(y_dp, "dp4a vs full-x ref")
    stats(y_dp, "dp4a vs dp4a-math", y_dpref)
    stats(y_f16, "f16x2 vs full-x ref")
    print(f"SPEED (incl launch): dp4a={t_dp:.2f}us   f16x2={t_f16:.2f}us   "
          f"{'dp4a FASTER' if t_dp < t_f16 else 'f16x2 faster'} by {abs(t_dp-t_f16):.2f}us")


if __name__ == "__main__":
    main()
