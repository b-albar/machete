import pytest
from typing import Any
from machete.jit.jit import load_cuda_ops
from machete.jit.jit_env import ROOT_DIR

@pytest.fixture
def flash_attention_op() -> Any:

    # Load CUDA operations
    op = load_cuda_ops(
        "flash_attention_3",
        arch_target="hopper",
        sources=[
            ROOT_DIR / "csrc/kernels/flash-attention/h100/h100_fwd.cu",
            ROOT_DIR / "csrc/kernels/flash-attention/h100/h100_bwd.cu",
            ROOT_DIR / "csrc/kernels/flash-attention/h100/h100_interface.cu",
        ],
    )
    return op

def test_flash_attention_compilation(flash_attention_op: Any) -> None:
    # Test that the forward operation exists
    assert hasattr(flash_attention_op, "fwd")

    # Test that the backward operation exists
    assert hasattr(flash_attention_op, "bwd")
