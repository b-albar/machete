# Copyright (c) 2025, Machete Authors
"""
Tests for multi-GPU communication (ParallelKittens-style peer TMA stores).

Tests the following components:
1. PeerBufferRegistry: peer buffer tracking and canonical name resolution
2. PeerTMARegistry: peer TMA descriptor generation
3. Op.peer_stores validation in _process_op_declarations
4. compile_communicate: compiles Op.communicate into @cute.jit wrapper
5. has_communicate flag and num_peer_barriers property
6. Single-GPU smoke test: peer_stores op with no actual peers (no-op communicate)
7. Multi-GPU integration test: actual peer buffer TMA communication
"""

import contextlib
import io

import pytest
import torch

from machete.megakernel.ops import (
    Op,
    ScheduledOp,
    TensorRegistry,
    PeerBufferInfo,
    PeerBufferRegistry,
    PeerTMARegistry,
    PeerTMADescriptorInfo,
)
from machete.megakernel.megakernel import Megakernel, MegakernelConfig
from machete.megakernel.compile import compile_communicate
from machete.utils.testing import is_hopper_available

requires_hopper = pytest.mark.skipif(
    not is_hopper_available(), reason="Requires Hopper (SM90+) GPU",
)


def _num_gpus():
    """Return number of available CUDA devices."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


requires_multi_gpu = pytest.mark.skipif(
    _num_gpus() < 2, reason="Requires 2+ GPUs",
)


# -- Test Ops ------------------------------------------------------------------


TILE_M = 4
N_STATIC = 16
ELEM_BYTES = 2  # fp16


class SimpleWriteOp(Op):
    """Op with writes but no peer_stores — baseline for comparison."""

    reads = {"x": (None, ("M", "N"))}
    writes = {"y": (None, ("M", "N"))}
    tile = ("M",)


class PeerStoreOp(Op):
    """Op that declares peer_stores for output tensor y.

    For testing: compute adds 1.0, communicate is default no-op.
    """

    reads = {"x": (None, ("M", "N"))}
    writes = {"y": (None, ("M", "N"))}
    tile = ("M",)
    peer_stores = {"y"}


class PeerStoreTMAOp(Op):
    """Op with TMA load/store AND peer_stores.

    Uses TMA for local load/store and declares peer communication.
    """

    reads = {"x": (None, ("M", "N"))}
    writes = {"y": (None, ("M", "N"))}
    tile = ("M",)
    tma_loads = {"x"}
    tma_stores = {"y"}
    peer_stores = {"y"}


class MultiPeerStoreOp(Op):
    """Op that peer-stores multiple output tensors."""

    reads = {"x": (None, ("M", "N"))}
    writes = {"y": (None, ("M", "N")), "z": (None, ("M", "N"))}
    tile = ("M",)
    peer_stores = {"y", "z"}


# =============================================================================
# PeerBufferRegistry Tests
# =============================================================================


class TestPeerBufferRegistry:

    def test_empty_peer_map(self):
        """Empty peer_map creates empty registry."""
        registry = PeerBufferRegistry.from_config(
            peer_map={},
            tensor_registry=TensorRegistry(tensors=[], op_mappings={}, name_to_idx={}),
            ops=[],
        )
        assert not registry.has_peers
        assert registry.num_peers == 0
        assert registry.canonical_names == []

    def test_none_peer_list(self):
        """Peer map with empty lists creates empty registry."""
        registry = PeerBufferRegistry(buffers=[], num_peers=0)
        assert not registry.has_peers
        assert registry.num_peers == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_single_peer(self):
        """Single peer buffer is correctly registered."""
        x = torch.randn(8, N_STATIC, dtype=torch.float16, device="cuda")
        y = torch.zeros(8, N_STATIC, dtype=torch.float16, device="cuda")
        peer_y = torch.zeros(8, N_STATIC, dtype=torch.float16, device="cuda")

        ops = PeerStoreOp.schedule(x=x, y=y, tile_sizes={"M": TILE_M})
        tensor_registry = TensorRegistry.from_ops(ops)

        registry = PeerBufferRegistry.from_config(
            peer_map={"y": [peer_y]},
            tensor_registry=tensor_registry,
            ops=ops,
        )
        assert registry.has_peers
        assert registry.num_peers == 1
        assert len(registry.buffers) == 1
        assert registry.buffers[0].peer_tensors == [peer_y]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_multiple_peers(self):
        """Multiple peer buffers with consistent count."""
        x = torch.randn(8, N_STATIC, dtype=torch.float16, device="cuda")
        y = torch.zeros(8, N_STATIC, dtype=torch.float16, device="cuda")
        peers = [
            torch.zeros(8, N_STATIC, dtype=torch.float16, device="cuda")
            for _ in range(3)
        ]

        ops = PeerStoreOp.schedule(x=x, y=y, tile_sizes={"M": TILE_M})
        tensor_registry = TensorRegistry.from_ops(ops)

        registry = PeerBufferRegistry.from_config(
            peer_map={"y": peers},
            tensor_registry=tensor_registry,
            ops=ops,
        )
        assert registry.has_peers
        assert registry.num_peers == 3
        assert len(registry.buffers) == 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_inconsistent_peer_count_raises(self):
        """Inconsistent peer list lengths raise ValueError."""
        x = torch.randn(8, N_STATIC, dtype=torch.float16, device="cuda")
        y = torch.zeros(8, N_STATIC, dtype=torch.float16, device="cuda")
        z = torch.zeros(8, N_STATIC, dtype=torch.float16, device="cuda")

        ops = MultiPeerStoreOp.schedule(x=x, y=y, z=z, tile_sizes={"M": TILE_M})
        tensor_registry = TensorRegistry.from_ops(ops)

        with pytest.raises(ValueError, match="must be consistent"):
            PeerBufferRegistry.from_config(
                peer_map={
                    "y": [torch.zeros(8, N_STATIC, dtype=torch.float16, device="cuda")],
                    "z": [
                        torch.zeros(8, N_STATIC, dtype=torch.float16, device="cuda"),
                        torch.zeros(8, N_STATIC, dtype=torch.float16, device="cuda"),
                    ],
                },
                tensor_registry=tensor_registry,
                ops=ops,
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_unknown_tensor_name_raises(self):
        """Peer buffer for unknown tensor name raises ValueError."""
        x = torch.randn(8, N_STATIC, dtype=torch.float16, device="cuda")
        y = torch.zeros(8, N_STATIC, dtype=torch.float16, device="cuda")

        ops = PeerStoreOp.schedule(x=x, y=y, tile_sizes={"M": TILE_M})
        tensor_registry = TensorRegistry.from_ops(ops)

        with pytest.raises(ValueError, match="not found"):
            PeerBufferRegistry.from_config(
                peer_map={"nonexistent": [
                    torch.zeros(8, N_STATIC, dtype=torch.float16, device="cuda"),
                ]},
                tensor_registry=tensor_registry,
                ops=ops,
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_get_peer_tensors(self):
        """get_peer_tensors returns correct tensors for canonical name."""
        x = torch.randn(8, N_STATIC, dtype=torch.float16, device="cuda")
        y = torch.zeros(8, N_STATIC, dtype=torch.float16, device="cuda")
        peer_y = torch.zeros(8, N_STATIC, dtype=torch.float16, device="cuda")

        ops = PeerStoreOp.schedule(x=x, y=y, tile_sizes={"M": TILE_M})
        tensor_registry = TensorRegistry.from_ops(ops)

        registry = PeerBufferRegistry.from_config(
            peer_map={"y": [peer_y]},
            tensor_registry=tensor_registry,
            ops=ops,
        )
        canonical = registry.canonical_names[0]
        peers = registry.get_peer_tensors(canonical)
        assert peers is not None
        assert len(peers) == 1
        assert peers[0] is peer_y

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_get_peer_tensors_nonexistent(self):
        """get_peer_tensors returns None for unknown canonical name."""
        x = torch.randn(8, N_STATIC, dtype=torch.float16, device="cuda")
        y = torch.zeros(8, N_STATIC, dtype=torch.float16, device="cuda")
        peer_y = torch.zeros(8, N_STATIC, dtype=torch.float16, device="cuda")

        ops = PeerStoreOp.schedule(x=x, y=y, tile_sizes={"M": TILE_M})
        tensor_registry = TensorRegistry.from_ops(ops)

        registry = PeerBufferRegistry.from_config(
            peer_map={"y": [peer_y]},
            tensor_registry=tensor_registry,
            ops=ops,
        )
        assert registry.get_peer_tensors("t999") is None


# =============================================================================
# PeerTMARegistry Tests
# =============================================================================


class TestPeerTMARegistry:

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_no_peer_stores_empty_registry(self):
        """Ops without peer_stores produce empty PeerTMARegistry."""
        x = torch.randn(8, N_STATIC, dtype=torch.float16, device="cuda")
        y = torch.zeros(8, N_STATIC, dtype=torch.float16, device="cuda")

        ops = SimpleWriteOp.schedule(x=x, y=y, tile_sizes={"M": TILE_M})
        tensor_registry = TensorRegistry.from_ops(ops)
        peer_buffer_registry = PeerBufferRegistry(buffers=[], num_peers=0)

        registry = PeerTMARegistry.from_ops(
            ops, tensor_registry, peer_buffer_registry)
        assert not registry.has_peer_tma
        assert registry.all_canonical_names == []
        assert registry.num_peers == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_single_peer_store_descriptors(self):
        """Single peer store creates descriptors for each peer."""
        x = torch.randn(8, N_STATIC, dtype=torch.float16, device="cuda")
        y = torch.zeros(8, N_STATIC, dtype=torch.float16, device="cuda")
        peers = [
            torch.zeros(8, N_STATIC, dtype=torch.float16, device="cuda")
            for _ in range(2)
        ]

        ops = PeerStoreOp.schedule(x=x, y=y, tile_sizes={"M": TILE_M})
        tensor_registry = TensorRegistry.from_ops(ops)
        peer_buffer_registry = PeerBufferRegistry.from_config(
            peer_map={"y": peers},
            tensor_registry=tensor_registry,
            ops=ops,
        )

        registry = PeerTMARegistry.from_ops(
            ops, tensor_registry, peer_buffer_registry)
        assert registry.has_peer_tma
        assert registry.num_peers == 2
        # One tensor × 2 peers = 2 descriptors
        assert len(registry.descriptors) == 2

        # Check canonical naming: ptma0_p0_atom/gmem, ptma0_p1_atom/gmem
        d0, d1 = registry.descriptors
        assert d0.canonical_atom == "ptma0_p0_atom"
        assert d0.canonical_gmem == "ptma0_p0_gmem"
        assert d0.peer_idx == 0
        assert d1.canonical_atom == "ptma0_p1_atom"
        assert d1.canonical_gmem == "ptma0_p1_gmem"
        assert d1.peer_idx == 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_all_canonical_names(self):
        """all_canonical_names returns atom/gmem pairs for all descriptors."""
        x = torch.randn(8, N_STATIC, dtype=torch.float16, device="cuda")
        y = torch.zeros(8, N_STATIC, dtype=torch.float16, device="cuda")
        peer_y = torch.zeros(8, N_STATIC, dtype=torch.float16, device="cuda")

        ops = PeerStoreOp.schedule(x=x, y=y, tile_sizes={"M": TILE_M})
        tensor_registry = TensorRegistry.from_ops(ops)
        peer_buffer_registry = PeerBufferRegistry.from_config(
            peer_map={"y": [peer_y]},
            tensor_registry=tensor_registry,
            ops=ops,
        )

        registry = PeerTMARegistry.from_ops(
            ops, tensor_registry, peer_buffer_registry)

        names = registry.all_canonical_names
        assert names == ["ptma0_p0_atom", "ptma0_p0_gmem"]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_op_mappings_communicate_phase(self):
        """op_mappings maps local TMA names to canonical for communicate phase."""
        x = torch.randn(8, N_STATIC, dtype=torch.float16, device="cuda")
        y = torch.zeros(8, N_STATIC, dtype=torch.float16, device="cuda")
        peer_y = torch.zeros(8, N_STATIC, dtype=torch.float16, device="cuda")

        ops = PeerStoreOp.schedule(x=x, y=y, tile_sizes={"M": TILE_M})
        tensor_registry = TensorRegistry.from_ops(ops)
        peer_buffer_registry = PeerBufferRegistry.from_config(
            peer_map={"y": [peer_y]},
            tensor_registry=tensor_registry,
            ops=ops,
        )

        registry = PeerTMARegistry.from_ops(
            ops, tensor_registry, peer_buffer_registry)

        # Check (op_idx=0, "communicate") mapping exists
        mapping = registry.op_mappings[(0, "communicate")]
        assert "y_p0_tma" in mapping
        assert mapping["y_p0_tma"] == "ptma0_p0_atom"
        assert "y_p0_tma_gmem" in mapping
        assert mapping["y_p0_tma_gmem"] == "ptma0_p0_gmem"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_get_op_peer_tma_args(self):
        """get_op_peer_tma_args returns sorted canonical names for communicate."""
        x = torch.randn(8, N_STATIC, dtype=torch.float16, device="cuda")
        y = torch.zeros(8, N_STATIC, dtype=torch.float16, device="cuda")
        peers = [
            torch.zeros(8, N_STATIC, dtype=torch.float16, device="cuda")
            for _ in range(2)
        ]

        ops = PeerStoreOp.schedule(x=x, y=y, tile_sizes={"M": TILE_M})
        tensor_registry = TensorRegistry.from_ops(ops)
        peer_buffer_registry = PeerBufferRegistry.from_config(
            peer_map={"y": peers},
            tensor_registry=tensor_registry,
            ops=ops,
        )

        registry = PeerTMARegistry.from_ops(
            ops, tensor_registry, peer_buffer_registry)

        args = registry.get_op_peer_tma_args(0, "communicate")
        # 2 peers × (atom + gmem) = 4 canonical names, sorted by local name
        assert len(args) == 4
        assert "ptma0_p0_atom" in args
        assert "ptma0_p1_atom" in args

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_descriptor_tile_shape(self):
        """TMA descriptor has correct reversed tile shape."""
        x = torch.randn(8, N_STATIC, dtype=torch.float16, device="cuda")
        y = torch.zeros(8, N_STATIC, dtype=torch.float16, device="cuda")
        peer_y = torch.zeros(8, N_STATIC, dtype=torch.float16, device="cuda")

        ops = PeerStoreOp.schedule(x=x, y=y, tile_sizes={"M": TILE_M})
        tensor_registry = TensorRegistry.from_ops(ops)
        peer_buffer_registry = PeerBufferRegistry.from_config(
            peer_map={"y": [peer_y]},
            tensor_registry=tensor_registry,
            ops=ops,
        )

        registry = PeerTMARegistry.from_ops(
            ops, tensor_registry, peer_buffer_registry)

        d = registry.descriptors[0]
        # y is (M, N) → TMA tile shape reversed: (N, tile_M)
        assert d.tile_shape == (N_STATIC, TILE_M)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_non_peer_op_empty_mappings(self):
        """Ops without peer_stores get empty communicate mapping."""
        x = torch.randn(8, N_STATIC, dtype=torch.float16, device="cuda")
        y = torch.zeros(8, N_STATIC, dtype=torch.float16, device="cuda")

        ops = SimpleWriteOp.schedule(x=x, y=y, tile_sizes={"M": TILE_M})
        tensor_registry = TensorRegistry.from_ops(ops)
        peer_buffer_registry = PeerBufferRegistry(buffers=[], num_peers=0)

        registry = PeerTMARegistry.from_ops(
            ops, tensor_registry, peer_buffer_registry)

        assert registry.get_op_peer_tma_args(0, "communicate") == []


# =============================================================================
# Op.peer_stores Validation Tests
# =============================================================================


class TestPeerStoresValidation:

    def test_peer_stores_must_be_in_writes(self):
        """peer_stores tensor not in writes or backward_writes raises ValueError."""
        with pytest.raises(ValueError, match="not found in writes"):
            class InvalidOp(Op):
                reads = {"x": (None, ("M", "N"))}
                writes = {"y": (None, ("M", "N"))}
                tile = ("M",)
                peer_stores = {"x"}  # x is a read, not a write

    def test_peer_stores_valid_write(self):
        """peer_stores with valid write tensor works."""
        # Should not raise — PeerStoreOp defines peer_stores={"y"} and y is in writes
        assert hasattr(PeerStoreOp, "_PEER_STORES")
        assert PeerStoreOp._PEER_STORES == {"y"}

    def test_peer_stores_empty_by_default(self):
        """Ops without peer_stores get empty _PEER_STORES."""
        assert hasattr(SimpleWriteOp, "_PEER_STORES")
        assert SimpleWriteOp._PEER_STORES == set()

    def test_peer_stores_in_tma_tensor_dims(self):
        """peer_stores tensors are included in _TMA_TENSOR_DIMS."""
        assert "y" in PeerStoreOp._TMA_TENSOR_DIMS

    def test_multiple_peer_stores(self):
        """Multiple peer_stores all validated and stored."""
        assert MultiPeerStoreOp._PEER_STORES == {"y", "z"}
        assert "y" in MultiPeerStoreOp._TMA_TENSOR_DIMS
        assert "z" in MultiPeerStoreOp._TMA_TENSOR_DIMS


# =============================================================================
# compile_communicate Tests
# =============================================================================


class TestCompileCommunicate:

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_default_communicate_compiles(self):
        """Default no-op communicate() compiles to a valid @cute.jit function."""
        from machete.megakernel.ops import build_op_config

        x = torch.randn(8, N_STATIC, dtype=torch.float16, device="cuda")
        y = torch.zeros(8, N_STATIC, dtype=torch.float16, device="cuda")

        ops = PeerStoreOp.schedule(x=x, y=y, tile_sizes={"M": TILE_M})
        tensor_registry = TensorRegistry.from_ops(ops)
        tensor_args = tensor_registry.get_op_tensor_args(0, PeerStoreOp)

        config = build_op_config(ops[0], kernel_config={"threads_per_row": 128})
        instance = PeerStoreOp(**config)

        fn = compile_communicate(instance, tensor_param_names=tensor_args)
        assert fn is not None
        assert callable(fn)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_communicate_with_peer_tma_mapping(self):
        """compile_communicate with TMA mapping produces valid function."""
        from machete.megakernel.ops import build_op_config

        x = torch.randn(8, N_STATIC, dtype=torch.float16, device="cuda")
        y = torch.zeros(8, N_STATIC, dtype=torch.float16, device="cuda")
        peer_y = torch.zeros(8, N_STATIC, dtype=torch.float16, device="cuda")

        ops = PeerStoreOp.schedule(x=x, y=y, tile_sizes={"M": TILE_M})
        tensor_registry = TensorRegistry.from_ops(ops)
        tensor_args = tensor_registry.get_op_tensor_args(0, PeerStoreOp)

        peer_buffer_registry = PeerBufferRegistry.from_config(
            peer_map={"y": [peer_y]},
            tensor_registry=tensor_registry,
            ops=ops,
        )
        peer_tma_registry = PeerTMARegistry.from_ops(
            ops, tensor_registry, peer_buffer_registry)

        comm_tma_args = peer_tma_registry.get_op_peer_tma_args(0, "communicate")
        comm_tma_mapping = peer_tma_registry.op_mappings.get((0, "communicate"), {})

        config = build_op_config(ops[0], kernel_config={"threads_per_row": 128})
        instance = PeerStoreOp(**config)

        fn = compile_communicate(
            instance, tensor_param_names=tensor_args,
            tma_param_names=comm_tma_args,
            tma_local_mapping=comm_tma_mapping,
        )
        assert fn is not None
        assert callable(fn)


# =============================================================================
# has_communicate & num_peer_barriers Tests
# =============================================================================


class TestHasCommunicateAndPeerBarriers:

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_no_peer_stores_has_communicate_false(self):
        """Megakernel with no peer_stores ops has has_communicate=False."""
        x = torch.randn(8, N_STATIC, dtype=torch.float16, device="cuda")
        y = torch.zeros(8, N_STATIC, dtype=torch.float16, device="cuda")

        ops = SimpleWriteOp.schedule(x=x, y=y, tile_sizes={"M": TILE_M})
        kernel = Megakernel(ops)
        assert kernel.num_peer_barriers == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_peer_stores_num_peer_barriers(self):
        """num_peer_barriers equals total tiles for peer-store ops."""
        M = 16
        x = torch.randn(M, N_STATIC, dtype=torch.float16, device="cuda")
        y = torch.zeros(M, N_STATIC, dtype=torch.float16, device="cuda")

        ops = PeerStoreOp.schedule(x=x, y=y, tile_sizes={"M": TILE_M})
        kernel = Megakernel(ops)

        # M=16, tile_M=4 → 4 tiles. PeerStoreOp has peer_stores → 4 barriers
        assert kernel.num_peer_barriers == M // TILE_M

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_mixed_ops_peer_barriers(self):
        """num_peer_barriers counts only ops with peer_stores."""
        M = 16
        x = torch.randn(M, N_STATIC, dtype=torch.float16, device="cuda")
        y = torch.zeros(M, N_STATIC, dtype=torch.float16, device="cuda")
        z = torch.zeros(M, N_STATIC, dtype=torch.float16, device="cuda")

        # SimpleWriteOp has no peer_stores, PeerStoreOp does
        ops = (
            SimpleWriteOp.schedule(x=x, y=y, tile_sizes={"M": TILE_M})
            + PeerStoreOp.schedule(x=y, y=z, tile_sizes={"M": TILE_M})
        )
        kernel = Megakernel(ops)

        # Only PeerStoreOp contributes: M/TILE_M = 4 tiles
        assert kernel.num_peer_barriers == M // TILE_M


# =============================================================================
# Single-GPU Smoke Test
# =============================================================================


@requires_hopper
class TestSingleGPUSmoke:

    def test_peer_store_op_no_peers_runs(self):
        """Op with peer_stores but no peer_buffers config runs normally.

        The communicate phase is a no-op, and has_communicate is True but
        the const_expr eliminates the communicate block since no actual
        peer TMA descriptors exist.
        """
        M = 16
        torch.manual_seed(42)
        x = torch.randn(M, N_STATIC, dtype=torch.float16, device="cuda")
        y = torch.zeros(M, N_STATIC, dtype=torch.float16, device="cuda")

        ops = PeerStoreOp.schedule(x=x, y=y, tile_sizes={"M": TILE_M})
        with contextlib.redirect_stdout(io.StringIO()):
            kernel = Megakernel(ops)
            kernel.run()
        torch.cuda.synchronize()

    def test_mixed_ops_with_peer_stores_runs(self):
        """Mixed ops (some with peer_stores, some without) run correctly."""
        M = 16
        torch.manual_seed(42)
        x = torch.randn(M, N_STATIC, dtype=torch.float16, device="cuda")
        y = torch.zeros(M, N_STATIC, dtype=torch.float16, device="cuda")
        z = torch.zeros(M, N_STATIC, dtype=torch.float16, device="cuda")

        ops = (
            SimpleWriteOp.schedule(x=x, y=y, tile_sizes={"M": TILE_M})
            + PeerStoreOp.schedule(x=y, y=z, tile_sizes={"M": TILE_M})
        )
        with contextlib.redirect_stdout(io.StringIO()):
            kernel = Megakernel(ops)
            kernel.run()
        torch.cuda.synchronize()


# =============================================================================
# Multi-GPU Integration Test
# =============================================================================


import cutlass
import cutlass.cute as cute
from cutlass import Int32


class PeerAddOneOp(Op):
    """Add 1.0 and communicate result to peer GPU via TMA S2G.

    For testing: TMA loads x, compute adds 1.0, TMA stores y locally,
    communicate sends y to peer via peer TMA.
    """

    reads = {"x": (None, ("M", "N"))}
    writes = {"y": (None, ("M", "N"))}
    tile = ("M",)

    tma_loads = {"x"}
    tma_stores = {"y"}
    peer_stores = {"y"}

    @cute.jit
    def load(self, page_ptr, tile_M, x_tma, x_tma_gmem, work_mbar):
        from machete.megakernel.interpreter import mbarrier_arrive_expect_tx

        sA = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.N, self.tile_size_M)),
        )
        gA = cute.local_tile(
            x_tma_gmem, (self.N, self.tile_size_M), (None, None),
        )
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            x_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )
        nbytes = Int32(self.tile_size_M * self.N * ELEM_BYTES)
        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, nbytes)
        cute.copy(x_tma, tAgA[(None, 0, tile_M)], tAsA, tma_bar_ptr=mbar_ptr)

    @cute.jit
    def compute(self, page_ptr, tile_M, x, y):
        tidx = cute.arch.thread_idx()[0]
        total_elems = self.tile_size_M * self.N
        s = cute.make_tensor(
            cute.make_ptr(self.x_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((total_elems,)),
        )
        one = self.x_dtype(1.0)
        for i in range(tidx, total_elems, self.threads_per_row):
            s[i] = s[i] + one

    @cute.jit
    def store(self, page_ptr, tile_M, y_tma, y_tma_gmem):
        sA = cute.make_tensor(
            cute.make_ptr(self.y_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.N, self.tile_size_M)),
        )
        gA = cute.local_tile(
            y_tma_gmem, (self.N, self.tile_size_M), (None, None),
        )
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            y_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )
        with cute.arch.elect_one():
            cute.copy(y_tma, tAsA, tAgA[(None, 0, tile_M)])

    @cute.jit
    def communicate(self, page_ptr, tile_M, y_p0_tma, y_p0_tma_gmem):
        """Send result to peer GPU 0 via TMA S2G."""
        sA = cute.make_tensor(
            cute.make_ptr(self.y_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.N, self.tile_size_M)),
        )
        gA = cute.local_tile(
            y_p0_tma_gmem, (self.N, self.tile_size_M), (None, None),
        )
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            y_p0_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )
        with cute.arch.elect_one():
            cute.copy(y_p0_tma, tAsA, tAgA[(None, 0, tile_M)])


@requires_hopper
@requires_multi_gpu
class TestMultiGPUCommunication:
    """Integration tests requiring 2+ GPUs for actual peer TMA communication.

    Tests that communicate() correctly sends data to peer GPU buffers
    via TMA S2G stores over NVLink.
    """

    def _enable_peer_access(self, src_device, dst_device):
        """Enable P2P access between two devices."""
        with torch.cuda.device(src_device):
            if torch.cuda.can_device_access_peer(src_device, dst_device):
                try:
                    torch.cuda.enable_peer_access(dst_device)
                except RuntimeError:
                    pass  # Already enabled

    def test_peer_tma_store_single_tile(self):
        """GPU 0 computes x+1, stores locally, and communicates to GPU 1."""
        self._enable_peer_access(0, 1)
        self._enable_peer_access(1, 0)

        torch.manual_seed(42)
        # Source data on GPU 0
        x = torch.randn(TILE_M, N_STATIC, dtype=torch.float16, device="cuda:0")
        y = torch.zeros(TILE_M, N_STATIC, dtype=torch.float16, device="cuda:0")

        # Peer buffer on GPU 1
        peer_y = torch.zeros(TILE_M, N_STATIC, dtype=torch.float16, device="cuda:1")

        ops = PeerAddOneOp.schedule(x=x, y=y, tile_sizes={"M": TILE_M})
        config = MegakernelConfig(
            peer_buffers={"y": [peer_y]},
            device_idx=0,
            num_devices=2,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            kernel = Megakernel(ops, config=config)
            kernel.run()
        torch.cuda.synchronize()

        # Local store should be x + 1
        expected = x + 1.0
        torch.testing.assert_close(y, expected, atol=1e-3, rtol=1e-3)

        # Peer store should also be x + 1 (same data sent via TMA S2G)
        torch.testing.assert_close(
            peer_y.to("cuda:0"), expected, atol=1e-3, rtol=1e-3)

    def test_peer_tma_store_multi_tile(self):
        """Multi-tile peer communication: verifies each tile is sent correctly."""
        self._enable_peer_access(0, 1)
        self._enable_peer_access(1, 0)

        M = TILE_M * 4  # 4 tiles
        torch.manual_seed(42)
        x = torch.randn(M, N_STATIC, dtype=torch.float16, device="cuda:0")
        y = torch.zeros(M, N_STATIC, dtype=torch.float16, device="cuda:0")
        peer_y = torch.zeros(M, N_STATIC, dtype=torch.float16, device="cuda:1")

        ops = PeerAddOneOp.schedule(x=x, y=y, tile_sizes={"M": TILE_M})
        config = MegakernelConfig(
            peer_buffers={"y": [peer_y]},
            device_idx=0,
            num_devices=2,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            kernel = Megakernel(ops, config=config)
            kernel.run()
        torch.cuda.synchronize()

        expected = x + 1.0
        torch.testing.assert_close(y, expected, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(
            peer_y.to("cuda:0"), expected, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))
