import torch

from machete.utils.benchmark import _accepts_group_index
from machete.utils import benchmark_utils


def test_recommended_input_group_count_rotates_small_inputs(monkeypatch):
    monkeypatch.setattr(benchmark_utils, "cuda_l2_cache_size", lambda device=None: 128)

    assert benchmark_utils.recommended_input_group_count(384) == 1
    assert benchmark_utils.recommended_input_group_count(192) == 3
    assert benchmark_utils.recommended_input_group_count(64) == 7


def test_recommended_input_group_count_handles_unknown_or_invalid_size(monkeypatch):
    monkeypatch.setattr(benchmark_utils, "cuda_l2_cache_size", lambda device=None: 0)

    assert benchmark_utils.recommended_input_group_count(None) == 1
    assert benchmark_utils.recommended_input_group_count(0) == 1
    assert benchmark_utils.recommended_input_group_count(128) == 1


def test_tensor_tree_nbytes_counts_unique_tensor_storage_once():
    tensor = torch.empty(8, dtype=torch.float16)
    view = tensor.view(2, 4)
    other = torch.empty(4, dtype=torch.int32)

    assert benchmark_utils.tensor_tree_nbytes({"a": tensor, "b": [view, other]}) == 32
    assert benchmark_utils.tensor_tree_nbytes([tensor, view], unique_storage=False) == 32


def test_benchmark_group_index_detection_accepts_optional_positional():
    def no_group():
        pass

    def required_group(group_idx):
        pass

    def optional_group(group_idx=0):
        pass

    assert not _accepts_group_index(no_group)
    assert _accepts_group_index(required_group)
    assert _accepts_group_index(optional_group)
