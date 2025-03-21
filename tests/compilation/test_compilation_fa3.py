import pytest
from machete.jit.jit import load_cuda_ops
import os

@pytest.fixture
def flash_attention_op():
    # Change to the root directory
    os.chdir("../../")

    # Load CUDA operations
    op = load_cuda_ops(
        "flash_attention_3",
        arch_target="hopper",
        sources=[
            "csrc/kernels/flash-attention/h100/h100_fwd.cu",
            "csrc/kernels/flash-attention/h100/h100_bwd.cu",
            "csrc/kernels/flash-attention/h100/h100_interface.cu",
        ],
    )
    return op

def test_flash_attention_compilation(flash_attention_op):
    # Test that the forward operation exists
    assert hasattr(flash_attention_op, "fwd")

    # Test that the backward operation exists
    assert hasattr(flash_attention_op, "bwd")
