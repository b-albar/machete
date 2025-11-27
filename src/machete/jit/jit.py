from typing import List, Optional, Union, Callable, Literal
from pathlib import Path
import re
import torch
import torch.utils.cpp_extension as torch_cpp_ext
import os
from filelock import FileLock
from .jit_env import CFLAGS, CUDA_CFLAGS, LD_FLAGS, JIT_DIR, TK_INCLUDE_DIR, MACHETE_INCLUDE_DIR

def get_cuda_arch() -> Optional[str]:
    for cuda_arch_flags in torch_cpp_ext._get_cuda_arch_flags():
        arch = re.search(r"compute_(\d+)", cuda_arch_flags)
        if arch is not None:
            return arch.group(1)
    return None

def get_gpu_device(device_id: int) -> str:
    return torch.cuda.get_device_name(device_id)

def load_cuda_ops(
    name: str,
    sources: List[Union[str, Path]],
    extra_cflags: Optional[List[str]] = None,
    extra_cuda_cflags: Optional[List[str]] = None,
    extra_ldflags: Optional[List[str]] = None,
    extra_include_paths: Optional[List[str]] = None,
    gpu_target: Literal["4090", "a100", "h100", "5070"] = "h100",
    debug: bool = False,
    verbose: bool = False,
) -> Callable:
    if extra_cflags is None:
        extra_cflags = []
    if extra_cuda_cflags is None:
        extra_cuda_cflags = []

    cflags = CFLAGS
    cuda_cflags = CUDA_CFLAGS
    ldflags = LD_FLAGS

    cflags += extra_cflags
    cuda_cflags += extra_cuda_cflags
    if debug:
        cuda_cflags.append("-g -G -DDEBUG")

    if extra_ldflags is not None:
        ldflags += extra_ldflags
    print(f"Loading JIT ops: {name}")

    if gpu_target == "4090":
        if os.environ.get("TORCH_CUDA_ARCH_LIST") is None:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
            cuda_cflags.append("-DKITTENS_ARCH=890")
    elif gpu_target == "a100":
        if os.environ.get("TORCH_CUDA_ARCH_LIST") is None:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
        cuda_cflags.append("-DKITTENS_ARCH=800")
    elif gpu_target == "h100":
        if os.environ.get("TORCH_CUDA_ARCH_LIST") is None:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
        cuda_cflags.append("-DKITTENS_ARCH=900")
    elif gpu_target == "5070":
        if os.environ.get("TORCH_CUDA_ARCH_LIST") is None:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
        cuda_cflags.append("-DKITTENS_ARCH=1000")

    build_directory = JIT_DIR + "/" + name
    os.makedirs(build_directory, exist_ok=True)

    if extra_include_paths is None:
        extra_include_paths = []
    extra_include_paths += [
        str(TK_INCLUDE_DIR),
        str(MACHETE_INCLUDE_DIR)
    ]

    lock = FileLock(JIT_DIR + "/" + f"{name}.lock")
    with lock:

        mod = torch_cpp_ext.load(
            name,
            [str(_) for _ in sources],
            extra_cflags=cflags,
            extra_cuda_cflags=cuda_cflags,
            extra_ldflags=ldflags,
            extra_include_paths=[str(_) for _ in extra_include_paths],
            build_directory=build_directory,
            verbose=verbose,
            with_cuda=True
        )

    print(f"Finished loading JIT ops: {name}")
    return mod  # type: ignore
