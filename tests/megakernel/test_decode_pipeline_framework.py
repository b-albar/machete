# Copyright (c) 2025, Machete Authors
"""Framework contracts for staged pipeline ops."""

import pytest
import subprocess
import sys
import textwrap
import torch
import cutlass.cute as cute
from cutlass import Float32, Int32

from machete.megakernel import Megakernel, MegakernelConfig, ScheduledOp
from machete.megakernel import megakernel as mk
from machete.megakernel.ops import (
    DEFAULT_PAGE_SIZE,
    InstructionPageProtocol,
    PageRole,
    PipelineABI,
    PipelineSpec,
    Op,
    SemaphoreRole,
    build_op_config,
)
from machete.megakernel.paged_memory import PipelinePageLayout
from machete.megakernel.scheduling import (
    BarrierFormula,
    InstructionStreamBuilder,
    OverlapTileScheduler,
    TileInstruction,
)


class _PipelineNoop(Op):
    pipeline = PipelineSpec.streaming(
        page_bytes=256,
        scratch_bytes=64,
    )


class _OpOwnedPipelineNoop(Op):
    pipeline = PipelineSpec.streaming(
        page_bytes=256,
        scratch_bytes=64,
    )
    pipeline_abi = PipelineABI.op_owned()


class _PlainNoop(Op):
    pass


class _TooLargePipelineNoop(Op):
    pipeline = PipelineSpec(
        page_count=4,
        page_bytes=1024,
        semaphore_count=2,
        scratch_bytes=512,
    )


class _CustomProtocolNoop(Op):
    pipeline = PipelineSpec(
        page_count=2,
        page_bytes=128,
        semaphore_count=2,
        scratch_bytes=32,
    )
    pipeline_page_protocol = InstructionPageProtocol(
        page_roles=(
            PageRole("activation", 0, 1),
            PageRole("weights", 1, 1),
        ),
        semaphore_roles=(
            SemaphoreRole("loaded", 0, 1, participants=1),
            SemaphoreRole("computed", 1, 1, participants=4),
        ),
        page_bytes=128,
        scratch_bytes=32,
    )


class _CoalescedPipelineNoop(Op):
    pipeline = PipelineSpec.streaming(
        range_axis=0,
        range_end_axis=1,
        range_block_size=16,
        coalesce_ranges=True,
    )


class _CoalescedRangeFillOp(Op):
    pipeline = PipelineSpec.streaming(
        range_axis=0,
        range_end_axis=1,
        range_block_size=1,
        coalesce_ranges=True,
    )
    reads = {}
    writes = {"y": (None, ("N",))}
    tile = ("N",)

    @classmethod
    def schedule(cls, *, y):
        return [cls._schedule_single(tile_sizes={"N": 1}, y=y)]

    @cute.jit
    def compute(self, page_ptr, tile_N, tile_1, y):
        tidx = cute.arch.thread_idx()[0]
        out = cute.make_tensor(y.iterator, cute.make_layout(self.N))
        idx = tile_N + tidx
        while idx < tile_1:
            out[idx] = Float32(1.0).to(self.y_dtype)
            idx = idx + Int32(self.threads_per_row)


class _AddOneOp(Op):
    reads = {"x": (None, ("N",))}
    writes = {"y": (None, ("N",))}
    tile = ("N",)

    @classmethod
    def schedule(cls, *, x, y, tile_n=16):
        return [cls._schedule_single(tile_sizes={"N": tile_n}, x=x, y=y)]

    @cute.jit
    def compute(self, page_ptr, tile_N, x, y):
        tidx = cute.arch.thread_idx()[0]
        x_flat = cute.make_tensor(x.iterator, cute.make_layout(Int32(self.N)))
        y_flat = cute.make_tensor(y.iterator, cute.make_layout(Int32(self.N)))
        base = tile_N * Int32(self.tile_size_N)
        for offset in range(tidx, self.tile_size_N, self.threads_per_row):
            idx = base + offset
            if idx < Int32(self.N):
                y_flat[idx] = (x_flat[idx].to(Float32) + Float32(1.0)).to(self.y_dtype)


class _RangeLoopAddOneOp(_AddOneOp):
    pipeline = PipelineSpec.streaming(
        range_axis=0,
        range_end_axis=1,
        range_block_size=16,
        coalesce_ranges=True,
    )


class _StaticSingleWriterProducerOp(Op):
    reads = {"x": (None, ("N",))}
    writes = {"y": (None, ("N",))}
    tile = ("N",)


class _StaticSingleWriterConsumerOp(Op):
    reads = {"y": (None, ("N",))}
    writes = {"z": (None, ("N",))}
    tile = ("N",)


class _StaticReadScratchOp(Op):
    reads = {"x": (None, ("N",))}
    writes = {"y": (None, ("N",))}
    tile = ("N",)


class _StaticOverwriteScratchOp(Op):
    reads = {}
    writes = {"x": (None, ("N",))}
    tile = ("N",)


class _StaticQkvChunkProducerOp(Op):
    reads = {}
    writes = {
        "q": (None, ("O",)),
        "k_cache": (None, ("O",)),
    }
    tile = ("O",)


class _StaticQHeadConsumerOp(Op):
    reads = {"q": (None, ("H",))}
    writes = {"o": (None, ("H",))}
    tile = ("H",)


class _FullMlpProducerOp(Op):
    reads = {}
    writes = {"y": (None, ("B", "S", "O"))}
    tile = ("B", "S", "O")


class _SlicedMlpConsumerOp(Op):
    reads = {"a": (None, ("B", "S", "K"))}
    writes = {"z": (None, ("B", "S", "O"))}
    tile = ("B", "S", "O")


class _InvalidStagedLoadLoopOp(Op):
    pipeline = PipelineSpec.streaming(range_axis=0)
    reads = {}
    writes = {"y": (None, ("N",))}
    tile = ("N",)

    @cute.jit
    def load(self, page_ptr, y, inner_iter_idx):
        pass


class _InvalidPlainLoadLoopOp(Op):
    reads = {}
    writes = {"y": (None, ("N",))}
    tile = ("N",)

    @cute.jit
    def load(self, page_ptr, y, inner_iter_idx):
        pass


class _PartialRangeOwnershipStagedOp(Op):
    pipeline = PipelineSpec.streaming(
        range_axis=0,
        range_end_axis=1,
        range_block_size=4,
        coalesce_ranges=True,
    )
    reads = {}
    writes = {"y": (None, ("N",))}
    tile = ("N",)

    @cute.jit
    def load(self, page_ptr, tile_N, y, work_mbar):
        pass

    @cute.jit
    def compute(self, page_ptr, tile_N, tile_1, y):
        pass


def test_streaming_spec_matches_reference_shape():
    spec = PipelineSpec.streaming(
        page_bytes=2048,
        scratch_bytes=384,
        range_axis=2,
        range_end_axis=3,
        range_block_size=16,
        coalesce_ranges=True,
    )

    assert spec.page_count == 13  # activation page + 3 input stages * 4 pages
    assert spec.input_stages == 3
    assert spec.output_stages == 3
    assert spec.stage_pages == 4
    assert spec.semaphore_count == 13  # activation + arrived/finished pairs
    assert spec.resource_bytes == 13 * 2048 + 13 * 8 + 384
    assert spec.range_axis == 2
    assert spec.range_end_axis == 3
    assert spec.range_block_size == 16
    assert spec.coalesce_ranges is True


def test_streaming_protocol_names_pages_and_semaphores():
    spec = PipelineSpec.streaming(
        input_stages=3,
        output_stages=3,
        stage_pages=4,
        page_bytes=1024,
        scratch_bytes=128,
    )
    protocol = spec.page_protocol()

    assert isinstance(protocol, InstructionPageProtocol)
    assert protocol.page_count == spec.page_count
    assert protocol.semaphore_count == spec.semaphore_count
    assert protocol.resource_bytes == spec.resource_bytes
    assert protocol.page("activation") == 0
    assert protocol.page("input_stage_0", 0) == 1
    assert protocol.page("input_stage_0", 3) == 4
    assert protocol.page("input_stage_1", 0) == 5
    assert protocol.page("input_stage_2", 3) == 12
    assert protocol.semaphore("activations_arrived") == 0
    assert protocol.semaphore("weights_arrived_0") == 1
    assert protocol.semaphore("weights_arrived_2") == 3
    assert protocol.semaphore("weights_finished_0") == 4
    assert protocol.semaphore("outputs_arrived_0") == 7
    assert protocol.semaphore("outputs_finished_2") == 12

    with pytest.raises(IndexError):
        protocol.page("input_stage_0", 4)
    with pytest.raises(KeyError):
        protocol.semaphore("missing")


def test_non_staged_pipeline_protocol_falls_back_to_generic_names():
    spec = PipelineSpec(
        page_count=4,
        page_bytes=512,
        semaphore_count=2,
        scratch_bytes=64,
    )
    protocol = spec.page_protocol()

    assert protocol.page_count == 4
    assert protocol.semaphore_count == 2
    assert protocol.page("page_3") == 3
    assert protocol.semaphore("semaphore_1") == 1
    assert protocol.resource_bytes == spec.resource_bytes


def test_pipeline_abi_is_op_owned():
    op_owned = PipelineABI.op_owned()

    assert op_owned.kind == "staged"
    assert op_owned.execution == "op_owned"



def test_op_owned_pipeline_exposes_named_resource_offsets():
    op = ScheduledOp(_CustomProtocolNoop, tile_counts=(1,))
    Megakernel(
        [op],
        config=MegakernelConfig(num_sms=1, num_pages=1, page_size=1024),
    )
    instance = _CustomProtocolNoop(**build_op_config(op))

    assert _CustomProtocolNoop.pipeline_protocol().page("weights") == 1
    assert instance.pipeline_page_offset("activation") == 0
    assert instance.pipeline_page_offset("weights") == 128
    assert instance.pipeline_semaphore_offset("loaded") == 256
    assert instance.pipeline_semaphore_offset("computed") == 264
    assert instance.pipeline_semaphore_participants("computed") == 4
    assert instance.pipeline_scratch_offset() == 272


def test_op_pipeline_drives_flat_range_coalescing():
    op = ScheduledOp(_CoalescedPipelineNoop, tile_counts=(16,))
    builder = InstructionStreamBuilder()
    builder.add_op(op)
    instructions = builder.coalesce_pipeline_instructions(builder.build())

    work = [
        instr
        for instr in instructions
        if instr.op_idx != TileInstruction.END_MARKER
    ]
    assert len(work) == 1
    assert work[0].tiles[:2] == (0, 16)


def test_pipeline_is_resource_metadata_not_a_second_phase_api():
    assert not hasattr(Op, "pipeline_load")
    assert not hasattr(Op, "pipeline_consume")
    assert not hasattr(Op, "pipeline_store")


def test_pipeline_page_layout_offsets_are_aligned_and_bounds_checked():
    layout = PipelinePageLayout(
        page_count=3,
        page_bytes=1024,
        semaphore_count=5,
        scratch_bytes=96,
    )

    assert layout.page_offset(0) == 0
    assert layout.page_offset(2) == 2048
    assert layout.activation_page_offset() == 0
    assert layout.weight_page_offset(
        input_stage=0,
        page_in_stage=1,
        input_stages=1,
        stage_pages=2,
    ) == 2048
    assert layout.semaphores_offset == 3072
    assert layout.semaphore_offset(4) == 3072 + 4 * 8
    assert layout.weights_arrived_sem(2) == 3
    assert layout.weights_finished_sem(3, 2) == 6
    assert layout.outputs_arrived_sem(3, 1) == 8
    assert layout.outputs_finished_sem(3, 3, 2) == 12
    assert layout.scratch_offset % 16 == 0
    assert layout.output_scratch_offset(output_stage=1, output_stages=3) == (
        layout.scratch_offset + 32
    )
    assert layout.total_size % 128 == 0

    with pytest.raises(IndexError):
        layout.page_offset(3)
    with pytest.raises(IndexError):
        layout.semaphore_offset(5)


def test_pipeline_device_metadata_does_not_carry_range_loop_state():
    op = ScheduledOp(_PipelineNoop, tile_counts=(2,))
    kernel = Megakernel(
        [op],
        config=MegakernelConfig(num_sms=1, page_size=4096),
        device="cpu",
    )
    kernel._prepare_tensors()

    meta = kernel._op_metadata_tensor.cpu().tolist()
    assert len(meta) == mk._OP_META_STRIDE


def test_scheduled_op_can_override_pipeline_resources_without_specializing_config():
    op = ScheduledOp(
        _PipelineNoop,
        tile_counts=(1,),
        static_dims={
            "pipeline_page_count": 2,
            "pipeline_page_bytes": 1024,
            "pipeline_semaphore_count": 4,
            "pipeline_scratch_bytes": 32,
            "pipeline_range_axis": 2,
            "pipeline_range_end_axis": 3,
            "pipeline_range_block_size": 16,
            "pipeline_coalesce_ranges": True,
        },
    )
    kernel = Megakernel(
        [op],
        config=MegakernelConfig(num_sms=1, page_size=4096),
        device="cpu",
    )
    kernel._prepare_tensors()

    meta = kernel._op_metadata_tensor.cpu().tolist()
    assert len(meta) == mk._OP_META_STRIDE
    assert "pipeline_page_bytes" not in build_op_config(op)
    assert "pipeline_range_axis" not in build_op_config(op)


def test_non_pipeline_ops_get_zero_pipeline_metadata():
    class _PlainNoop(Op):
        pass

    kernel = Megakernel(
        [ScheduledOp(_PlainNoop, tile_counts=(1,))],
        config=MegakernelConfig(num_sms=1, page_size=4096),
        device="cpu",
    )
    kernel._prepare_tensors()

    meta = kernel._op_metadata_tensor.cpu().tolist()
    assert len(meta) == mk._OP_META_STRIDE


def test_pipeline_resource_validation_uses_effective_page_size():
    op = ScheduledOp(_TooLargePipelineNoop, tile_counts=(1,))

    with pytest.raises(ValueError, match="pipeline resources"):
        Megakernel(
            [op],
            config=MegakernelConfig(num_sms=1, page_size=2048),
            device="cpu",
        )

    kernel = Megakernel(
        [op],
        config=MegakernelConfig(num_sms=1, page_size=8192),
        device="cpu",
    )
    assert kernel.smem_size >= 8192


def test_staged_pipeline_rejects_runtime_load_loop_signature():
    with pytest.raises(ValueError, match="own staged loops"):
        Megakernel(
            [ScheduledOp(_InvalidStagedLoadLoopOp, tile_counts=(1,))],
            config=MegakernelConfig(num_sms=1, page_size=4096),
            device="cpu",
        )


def test_plain_op_rejects_runtime_load_loop_signature():
    with pytest.raises(ValueError, match="runtime load loops"):
        kernel = Megakernel(
            [ScheduledOp(_InvalidPlainLoadLoopOp, tile_counts=(1,))],
            config=MegakernelConfig(num_sms=1, page_size=4096),
            device="cpu",
        )
        kernel._prepare_tensors()


def test_staged_range_rejects_partial_phase_ownership():
    with pytest.raises(ValueError, match="partial coalesced-range ownership"):
        kernel = Megakernel(
            [ScheduledOp(_PartialRangeOwnershipStagedOp, tile_counts=(8,))],
            config=MegakernelConfig(num_sms=1, page_size=4096),
            device="cpu",
        )
        kernel._prepare_tensors()


def test_pipeline_ops_can_coalesce_flat_instruction_stream():
    op = ScheduledOp(_CoalescedPipelineNoop, tile_counts=(10,))
    builder = InstructionStreamBuilder()
    builder.add_op(op)
    instructions = builder.coalesce_pipeline_instructions(builder.build())

    ranges = [
        (instr.tiles[0], instr.tiles[1])
        for instr in instructions
        if instr.op_idx != TileInstruction.END_MARKER
    ]

    assert ranges == [(0, 10)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_coalesced_range_instruction_reaches_device_code():
    y = torch.zeros(10, device="cuda", dtype=torch.float32)
    kernel = Megakernel(
        _CoalescedRangeFillOp.schedule(y=y),
        config=MegakernelConfig(
            num_sms=3,
            num_pages=1,
            page_size=4096,
        ),
    )

    kernel.run()
    torch.cuda.synchronize()

    assert torch.all(y == 1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_coalesced_range_instruction_can_use_framework_fast_loop():
    x = torch.arange(32, device="cuda", dtype=torch.float32)
    y = torch.zeros_like(x)
    kernel = Megakernel(
        _RangeLoopAddOneOp.schedule(x=x, y=y, tile_n=1),
        config=MegakernelConfig(
            num_sms=2,
            num_pages=1,
            page_size=4096,
        ),
    )

    kernel.run()
    torch.cuda.synchronize()

    torch.testing.assert_close(y, x + 1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")








def test_qwen_final_head_public_api_does_not_expose_scalar_lm_head_variant():
    import machete.kernels.qwen_3_5 as qwen

    assert "Qwen3_5FinalRmsLmHeadSm120Op" not in qwen.__all__
    assert "Qwen3_5FinalRmsLmHeadStagedSm120Op" not in qwen.__all__


def test_qwen_final_schedule_keeps_lm_head_inside_megakernel_without_scalar_variant():
    import machete.kernels.qwen_3_5 as qwen
    from machete.kernels.rms_norm import RMSNormOp

    dtype = torch.bfloat16
    x = torch.empty(1, 16, qwen.QWEN3_5_HIDDEN, dtype=dtype)
    residual = torch.empty_like(x)
    h_final = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    final_norm = torch.empty(qwen.QWEN3_5_HIDDEN, dtype=dtype)
    lm_head = torch.empty(128, qwen.QWEN3_5_HIDDEN, dtype=dtype)
    logits = torch.empty(1, 16, 128, dtype=dtype)

    ops = qwen.schedule_final_qwen3_5_sm120(
        x=x,
        residual_in=residual,
        residual_out=residual_out,
        h_final=h_final,
        final_norm=final_norm,
        lm_head=lm_head,
        logits=logits,
        seq_len=16,
    )

    assert [op.op_cls for op in ops] == [RMSNormOp, qwen.Qwen3_5LmHeadGemmSm120Op]
    assert qwen.Qwen3_5LmHeadGemmSm120Op.pipeline is not None
    assert qwen.Qwen3_5DecodeMatvecGemmSm120Op.pipeline is not None
    assert qwen.Qwen3_5LmHeadGemmSm120Op.pipeline_abi == PipelineABI.op_owned()
    assert qwen.Qwen3_5DecodeMatvecGemmSm120Op.pipeline_abi == PipelineABI.op_owned()
    assert qwen.Qwen3_5RangedLmHeadSm120Op.pipeline_abi == PipelineABI.op_owned()
    assert qwen.Qwen3_5LmHeadGemmSm120Op.pipeline.range_axis == 2
    assert qwen.Qwen3_5LmHeadGemmSm120Op.pipeline.range_block_size == 16
    assert qwen.Qwen3_5RangedLmHeadSm120Op.tma_loads == {"weight"}
    assert ops[1].tile_sizes["S"] == qwen.QWEN3_5_DECODE_S
    assert ops[1].tile_sizes["N"] == 128
    assert ops[1].static_dims["tile_K"] == 32


def test_qwen_lm_head_rejects_non_divisible_vocab_tile():
    import machete.kernels.qwen_3_5 as qwen

    dtype = torch.bfloat16
    h = torch.empty(1, 16, qwen.QWEN3_5_HIDDEN, dtype=dtype)
    lm_head = torch.empty(130, qwen.QWEN3_5_HIDDEN, dtype=dtype)
    logits = torch.empty(1, 16, 130, dtype=dtype)

    with pytest.raises(ValueError, match="divisible"):
        qwen.Qwen3_5LmHeadGemmSm120Op.schedule(
            a=h,
            b=lm_head,
            c=logits,
            tile_sizes={"S": 16, "N": 64, "K": 64},
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qwen_top1_lm_head_matches_torch_small():
    script = r"""
import torch
from machete.megakernel import Megakernel, MegakernelConfig
from machete.kernels.qwen_3_5.sm120 import Qwen3_5Top1LmHeadSm120Op

torch.manual_seed(17)
dtype = torch.bfloat16
h = torch.randn(1, 4, 64, device="cuda", dtype=dtype)
weight = torch.randn(256, 64, device="cuda", dtype=dtype)
top_values = torch.empty(1, 4, device="cuda", dtype=torch.float32)
top_indices = torch.empty(1, 4, device="cuda", dtype=torch.int32)

ops = Qwen3_5Top1LmHeadSm120Op.schedule(
    h=h,
    weight=weight,
    top_values=top_values,
    top_indices=top_indices,
    page_size=8192,
)
kernel = Megakernel(
    ops,
    config=MegakernelConfig(
        num_sms=4,
        num_pages=1,
        page_size=ops[0].static_dims["page_size"],
        threads_per_block=224,
    ),
)
kernel.run()
torch.cuda.synchronize()

ref = torch.matmul(h.float(), weight.float().t())
ref_values, ref_indices = ref.max(dim=-1)
torch.testing.assert_close(top_values, ref_values, rtol=2e-3, atol=2e-2)
torch.testing.assert_close(top_indices, ref_indices.to(torch.int32), rtol=0, atol=0)
"""
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=True,
        timeout=180,
    )


def test_qwen_layer_qkv_projection_uses_staged_decode_gemm():
    import machete.kernels.qwen_3_5 as qwen

    dtype = torch.bfloat16
    batch = 1
    seq_len = 16
    hidden = qwen.QWEN3_5_HIDDEN
    pfx = "layer.0"
    weights = {
        "cos": torch.empty(128, qwen.QWEN3_5_ROTARY_D2, dtype=dtype),
        "sin": torch.empty(128, qwen.QWEN3_5_ROTARY_D2, dtype=dtype),
        f"{pfx}.attn_norm": torch.empty(hidden, dtype=dtype),
        f"{pfx}.W_qkv": torch.empty(
            (qwen.QWEN3_5_NUM_Q_HEADS + 2 * qwen.QWEN3_5_NUM_KV_HEADS)
            * qwen.QWEN3_5_HEAD_DIM,
            hidden,
            dtype=dtype,
        ),
        f"{pfx}.w_q_norm": torch.empty(qwen.QWEN3_5_HEAD_DIM, dtype=dtype),
        f"{pfx}.w_k_norm": torch.empty(qwen.QWEN3_5_HEAD_DIM, dtype=dtype),
        f"{pfx}.W_o": torch.empty(hidden, qwen.QWEN3_5_Q_DIM, dtype=dtype),
        f"{pfx}.mlp_norm": torch.empty(hidden, dtype=dtype),
        f"{pfx}.W_gate_up": torch.empty(2 * qwen.QWEN3_5_INTERMEDIATE, hidden, dtype=dtype),
        f"{pfx}.W_down": torch.empty(hidden, qwen.QWEN3_5_INTERMEDIATE, dtype=dtype),
    }
    x = torch.empty(batch, seq_len, hidden, dtype=dtype)
    qkv_buf = torch.empty(
        batch,
        seq_len,
        qwen.QWEN3_5_NUM_Q_HEADS + 2 * qwen.QWEN3_5_NUM_KV_HEADS,
        qwen.QWEN3_5_HEAD_DIM,
        dtype=dtype,
    )
    qk_sumsq_buf = torch.empty(
        batch,
        seq_len,
        qwen.QWEN3_5_NUM_Q_HEADS + qwen.QWEN3_5_NUM_KV_HEADS,
        qwen.QWEN3_5_HEAD_DIM // 64,
        dtype=torch.float32,
    )

    layer = qwen.schedule_decode_layer_qwen3_5_sm120(
        layer_idx=0,
        batch=batch,
        seq_len=seq_len,
        cache_pos=0,
        weights=weights,
        k_cache=torch.empty(batch, 128, qwen.QWEN3_5_NUM_KV_HEADS, qwen.QWEN3_5_HEAD_DIM, dtype=dtype),
        v_cache=torch.empty(batch, 128, qwen.QWEN3_5_NUM_KV_HEADS, qwen.QWEN3_5_HEAD_DIM, dtype=dtype),
        x_in=x,
        residual_in=torch.empty_like(x),
        x_out=torch.empty_like(x),
        residual_out=torch.empty_like(x),
        h_buf=torch.empty_like(x),
        q_buf=torch.empty(batch, seq_len, qwen.QWEN3_5_Q_DIM, dtype=dtype),
        kv_buf=torch.empty(batch, seq_len, 2 * qwen.QWEN3_5_KV_DIM, dtype=dtype),
        attn_out_buf=torch.empty(batch, seq_len, qwen.QWEN3_5_Q_DIM, dtype=dtype),
        proj_buf=torch.empty_like(x),
        h2_buf=torch.empty_like(x),
        gate_up_buf=torch.empty(batch, seq_len, 2 * qwen.QWEN3_5_INTERMEDIATE, dtype=dtype),
        mlp_h_buf=torch.empty(batch, seq_len, qwen.QWEN3_5_INTERMEDIATE, dtype=dtype),
        qkv_buf=qkv_buf,
        qk_sumsq_buf=qk_sumsq_buf,
        packed_qkv=True,
    )

    assert layer.ops[0].op_cls is qwen.Qwen3_5ComputeTmaRMSAddPackedQkvChunkProjectSm120Op
    assert layer.ops[0].op_cls.pipeline_abi == PipelineABI.op_owned()
    assert layer.ops[0].dim_aliases["N"] == "qkv_chunk_0"
    assert layer.ops[0].static_dims["barrier_group_count_N"] == (
        qwen.QWEN3_5_NUM_Q_HEADS + 2 * qwen.QWEN3_5_NUM_KV_HEADS
    )
    assert layer.ops[1].op_cls is qwen.Qwen3_5PackedQkvFinalizeSm120Op
    assert layer.ops[1].dim_aliases["H"] == "q_head_0"
    assert layer.ops[1].static_dims["barrier_wait_alias_H"] == "qkv_chunk_0"
    assert layer.ops[1].static_dims["barrier_signal_alias_H"] == "q_head_0"

    layer.ops[0].static_dims["pipeline_range_block_size"] = 4
    builder = InstructionStreamBuilder()
    for op in layer.ops[:2]:
        builder.add_op(op)
    instructions = builder.coalesce_pipeline_instructions(builder.build())
    formulas = builder.get_op_barrier_formulas()
    first_qkv = next(instr for instr in instructions if instr.op_idx == 0)
    first_finalize = next(instr for instr in instructions if instr.op_idx == 1)

    assert first_qkv.tiles == (0, 0, 0, 4, 0)
    assert builder._build_signal_info_entry(first_qkv, formulas) == [0, 0, 0, 0]
    assert builder._build_wait_info_entry(first_finalize, formulas) == [0, 4]


def test_barrier_formula_supports_half_open_tile_interval_guard():
    formula = BarrierFormula(
        base=17,
        coeffs=(1, 0, 0, 0, 0),
        divs=(2, 1, 1, 1, 1),
        guard_min=4,
        guard_max=10,
    )

    assert formula.has_guard
    assert not formula.is_guarded((3,))
    assert formula.is_guarded((4,))
    assert formula.is_guarded((9,))
    assert not formula.is_guarded((10,))
    assert formula.compute_index((8,)) == 21

    axis_guard = BarrierFormula(
        base=0,
        coeffs=(8, 0, 1, 0, 0),
        guard_min=4,
        guard_max=8,
        guard_coeffs=(0, 0, 1, 0, 0),
    )
    assert axis_guard.is_guarded((0, 0, 4))
    assert axis_guard.is_guarded((1, 0, 4))
    assert not axis_guard.is_guarded((0, 0, 8))


def test_buffer_specific_barrier_aliases_drive_dependency_formulas():
    producer = ScheduledOp(
        _StaticQkvChunkProducerOp,
        tile_counts=(8,),
        dim_names={"O": 0},
        tile_sizes={"O": 64},
        static_dims={
            "barrier_signal_q_alias_O": "q_group",
            "barrier_signal_q_tile_size_O": 64,
        },
    )
    consumer = ScheduledOp(
        _StaticQHeadConsumerOp,
        tile_counts=(2,),
        dim_names={"H": 0},
        tile_sizes={"H": 1},
        static_dims={
            "barrier_wait_q_alias_H": "q_group",
            "barrier_wait_q_tile_size_H": 256,
        },
    )

    builder = InstructionStreamBuilder()
    builder.add_op(producer)
    builder.add_op(consumer)
    formulas = builder.get_op_barrier_formulas()
    signal_formula = formulas[0][1][0]
    wait_formula = formulas[1][0][0]

    assert signal_formula.compute_index((0,)) == signal_formula.compute_index((3,))
    assert signal_formula.compute_index((4,)) == signal_formula.compute_index((7,))
    assert signal_formula.compute_index((0,)) != signal_formula.compute_index((4,))
    assert wait_formula.compute_index((0,)) == signal_formula.compute_index((0,))
    assert wait_formula.compute_index((1,)) == signal_formula.compute_index((4,))
    assert wait_formula.expected == 4


def test_last_dim_slice_consumer_waits_on_matching_producer_region():
    y = torch.empty((1, 1, 8192), dtype=torch.float32)
    a = y[:, :, 2048:4096]
    z = torch.empty((1, 1, 2048), dtype=torch.float32)

    producer = _FullMlpProducerOp.schedule(
        y=y,
        tile_sizes={"B": 1, "S": 1, "O": 8},
    )[0]
    consumer = _SlicedMlpConsumerOp.schedule(
        a=a,
        z=z,
        tile_sizes={"B": 1, "S": 1, "O": 8},
    )[0]

    builder = InstructionStreamBuilder()
    builder.add_op(producer)
    builder.add_op(consumer)
    formulas = builder.get_op_barrier_formulas()
    signal_formula = formulas[0][1][0]
    wait_formula = formulas[1][0][0]

    assert builder.num_barriers == 4
    assert signal_formula.divs[2] == 256
    assert wait_formula.expected == 256
    assert wait_formula.compute_index((0, 0, 0)) == signal_formula.compute_index((0, 0, 256))
    assert signal_formula.compute_index((0, 0, 255)) != wait_formula.compute_index((0, 0, 0))
    assert signal_formula.compute_index((0, 0, 511)) == wait_formula.compute_index((0, 0, 0))
    assert signal_formula.compute_index((0, 0, 512)) != wait_formula.compute_index((0, 0, 0))


def test_overlap_scheduler_can_prioritize_newly_ready_consumers():
    producer = ScheduledOp(
        _StaticQkvChunkProducerOp,
        tile_counts=(8,),
        dim_names={"O": 0},
        tile_sizes={"O": 64},
        static_dims={
            "barrier_signal_q_alias_O": "q_group",
            "barrier_signal_q_tile_size_O": 64,
        },
    )
    consumer = ScheduledOp(
        _StaticQHeadConsumerOp,
        tile_counts=(2,),
        dim_names={"H": 0},
        tile_sizes={"H": 1},
        static_dims={
            "barrier_wait_q_alias_H": "q_group",
            "barrier_wait_q_tile_size_H": 256,
        },
    )

    builder = InstructionStreamBuilder()
    builder.add_op(producer)
    builder.add_op(consumer)
    instructions = builder.build(
        scheduler=OverlapTileScheduler(
            fetch_stride=1,
            prefer_data_movement=False,
            prefer_ready_consumers=True,
        )
    )

    assert [(instr.op_idx, instr.tiles) for instr in instructions[:6]] == [
        (0, (0,)),
        (0, (1,)),
        (0, (2,)),
        (0, (3,)),
        (1, (0,)),
        (0, (4,)),
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qwen_fused_rms_add_staged_decode_gemm_matches_torch_small():
    from machete.kernels.qwen_3_5.sm120 import (
        Qwen3_5RMSAddStagedDecodeGemmSm120Op,
    )

    torch.manual_seed(4)
    dtype = torch.bfloat16
    a = torch.randn(1, 16, 128, device="cuda", dtype=dtype)
    residual = torch.randn_like(a)
    norm_weight = torch.randn(128, device="cuda", dtype=dtype)
    b = torch.randn(256, 128, device="cuda", dtype=dtype)
    residual_out = torch.empty_like(a)
    c = torch.empty(1, 16, 256, device="cuda", dtype=dtype)

    kernel = Megakernel(
        Qwen3_5RMSAddStagedDecodeGemmSm120Op.schedule(
            a=a,
            residual_in=residual,
            norm_weight=norm_weight,
            b=b,
            residual_out=residual_out,
            c=c,
            page_size=32768,
            tile_sizes={"S": 16, "N": 64, "K": 32},
        ),
        config=MegakernelConfig(
            num_sms=4,
            num_pages=1,
            page_size=32768,
            threads_per_block=224,
        ),
    )
    kernel.run()
    torch.cuda.synchronize()

    ref_residual = (a.float() + residual.float()).to(dtype)
    ref_norm = (
        ref_residual.float()
        * torch.rsqrt(ref_residual.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        * norm_weight.float()
    )
    ref = torch.matmul(ref_norm, b.float().t())
    torch.testing.assert_close(residual_out.float(), ref_residual.float(), rtol=0, atol=0)
    torch.testing.assert_close(c.float(), ref, rtol=2e-2, atol=2e-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qwen_ranged_rms_add_decode_matvec_reuses_norm_for_long_ranges():
    script = r"""
import torch
from machete.megakernel import Megakernel, MegakernelConfig
from machete.kernels.qwen_3_5.sm120 import (
    Qwen3_5RMSAddRangedDecodeMatvecSm120Op,
)

torch.manual_seed(9)
dtype = torch.bfloat16
a = torch.randn(1, 16, 64, device="cuda", dtype=dtype)
residual = torch.randn_like(a)
norm_weight = torch.randn(64, device="cuda", dtype=dtype)
b = torch.randn(4096, 64, device="cuda", dtype=dtype)
residual_out = torch.empty_like(a)
c = torch.empty(1, 16, 4096, device="cuda", dtype=dtype)

kernel = Megakernel(
    Qwen3_5RMSAddRangedDecodeMatvecSm120Op.schedule(
        a=a,
        residual_in=residual,
        norm_weight=norm_weight,
        b=b,
        residual_out=residual_out,
        c=c,
        page_size=32768,
        tile_sizes={"S": 16, "N": 16},
    ),
    config=MegakernelConfig(
        num_sms=4,
        num_pages=1,
        page_size=32768,
        threads_per_block=224,
    ),
)
kernel.run()
torch.cuda.synchronize()

ref_residual = a + residual
ref_norm = (
    ref_residual.float()
    * torch.rsqrt(ref_residual.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    * norm_weight.float()
)
ref = torch.matmul(ref_norm, b.float().t())
torch.testing.assert_close(residual_out.float(), ref_residual.float(), rtol=0, atol=0)
torch.testing.assert_close(c.float(), ref, rtol=2e-2, atol=2e-1)
"""
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=True,
        timeout=180,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qwen_ranged_rms_add_decode_gemm_matches_torch_long_ranges():
    script = r"""
import torch
from machete.megakernel import Megakernel, MegakernelConfig
from machete.kernels.qwen_3_5.sm120 import (
    Qwen3_5RMSAddRangedDecodeGemmSm120Op,
)

torch.manual_seed(13)
dtype = torch.bfloat16
a = torch.randn(1, 16, 128, device="cuda", dtype=dtype)
residual = torch.randn_like(a)
norm_weight = torch.randn(128, device="cuda", dtype=dtype)
b = torch.randn(4096, 128, device="cuda", dtype=dtype)
residual_out = torch.empty_like(a)
c = torch.empty(1, 16, 4096, device="cuda", dtype=dtype)

ops = Qwen3_5RMSAddRangedDecodeGemmSm120Op.schedule(
    a=a,
    residual_in=residual,
    norm_weight=norm_weight,
    b=b,
    residual_out=residual_out,
    c=c,
    page_size=49152,
    tile_sizes={"S": 16, "N": 128, "K": 32},
)
kernel = Megakernel(
    ops,
    config=MegakernelConfig(
        num_sms=8,
        num_pages=1,
        page_size=ops[0].static_dims["page_size"],
        threads_per_block=224,
    ),
)
kernel.run()
torch.cuda.synchronize()

ref_residual = a + residual
ref_norm = (
    ref_residual.float()
    * torch.rsqrt(ref_residual.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    * norm_weight.float()
)
ref = torch.matmul(ref_norm, b.float().t())
torch.testing.assert_close(residual_out.float(), ref_residual.float(), rtol=0, atol=0)
torch.testing.assert_close(c.float(), ref, rtol=2e-2, atol=3e-1)
"""
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=True,
        timeout=180,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

def test_qwen_ranged_rms_add_decode_gemm_rejects_unstable_tile_k():
    from machete.kernels.qwen_3_5.sm120 import (
        Qwen3_5RMSAddRangedDecodeGemmSm120Op,
    )

    dtype = torch.bfloat16
    a = torch.empty(1, 16, 128, dtype=dtype)
    residual = torch.empty_like(a)
    norm_weight = torch.empty(128, dtype=dtype)
    b = torch.empty(256, 128, dtype=dtype)
    residual_out = torch.empty_like(a)
    c = torch.empty(1, 16, 256, dtype=dtype)

    with pytest.raises(ValueError, match="K=32"):
        Qwen3_5RMSAddRangedDecodeGemmSm120Op.schedule(
            a=a,
            residual_in=residual,
            norm_weight=norm_weight,
            b=b,
            residual_out=residual_out,
            c=c,
            tile_sizes={"S": 16, "N": 128, "K": 64},
        )


def test_qwen_decode_gemm_uses_staged_abi():
    import machete.kernels.qwen_3_5 as qwen

    assert qwen.Qwen3_5DecodeMatvecGemmSm120Op.pipeline_abi == PipelineABI.op_owned()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qwen_staged_decode_gemm_matches_torch_small():
    from machete.kernels.qwen_3_5.sm120 import Qwen3_5DecodeMatvecGemmSm120Op

    torch.manual_seed(3)
    dtype = torch.bfloat16
    a = torch.randn(1, 16, 128, device="cuda", dtype=dtype)
    b = torch.randn(256, 128, device="cuda", dtype=dtype)
    c = torch.empty(1, 16, 256, device="cuda", dtype=dtype)

    kernel = Megakernel(
        Qwen3_5DecodeMatvecGemmSm120Op.schedule(
            a=a,
            b=b,
            c=c,
            page_size=32768,
            tile_sizes={"S": 16, "N": 64, "K": 32},
        ),
        config=MegakernelConfig(
            num_sms=4,
            num_pages=1,
            page_size=32768,
            threads_per_block=224,
        ),
    )
    kernel.run()
    torch.cuda.synchronize()

    ref = torch.matmul(a.float(), b.float().t())
    torch.testing.assert_close(c.float(), ref, rtol=2e-2, atol=2e-1)


def test_qwen_ranged_lm_head_matches_torch_small():
    from machete.kernels.qwen_3_5.sm120 import Qwen3_5RangedLmHeadSm120Op

    dtype = torch.bfloat16
    h = torch.empty(1, 16, 64, dtype=dtype)
    weight = torch.empty(64, 64, dtype=dtype)
    logits = torch.empty(1, 16, 64, dtype=dtype)

    ops = Qwen3_5RangedLmHeadSm120Op.schedule(
        h=h,
        weight=weight,
        logits=logits,
        page_size=8192,
    )

    assert ops[0].op_cls.pipeline_abi == PipelineABI.op_owned()
    assert ops[0].op_cls.pipeline.coalesce_ranges is True
    assert ops[0].tile_sizes["N"] == 16









def test_qwen_ranged_lm_head_handles_long_coalesced_ranges():
    script = r"""
import torch
from machete.megakernel import Megakernel, MegakernelConfig
from machete.kernels.qwen_3_5.sm120 import Qwen3_5RangedLmHeadSm120Op

torch.manual_seed(1)
dtype = torch.bfloat16
h = torch.randn(1, 16, 64, device="cuda", dtype=dtype)
weight = torch.randn(4096, 64, device="cuda", dtype=dtype)
logits = torch.empty(1, 16, 4096, device="cuda", dtype=dtype)

kernel = Megakernel(
    Qwen3_5RangedLmHeadSm120Op.schedule(
        h=h,
        weight=weight,
        logits=logits,
        page_size=8192,
    ),
    config=MegakernelConfig(
        num_sms=8,
        num_pages=1,
        page_size=8192,
    ),
)
kernel.run()
torch.cuda.synchronize()

ref = torch.matmul(h, weight.t())
torch.testing.assert_close(logits.float(), ref.float(), rtol=2e-2, atol=2e-1)
"""
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=True,
        timeout=180,
    )
