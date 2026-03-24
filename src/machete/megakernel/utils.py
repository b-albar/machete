# Copyright (c) 2025, Machete Authors
"""Debug utilities for inspecting compiled megakernels (PTX, SASS, CUBIN)."""

import os
import struct
import subprocess
import tempfile


def extract_cubin(kernel) -> bytes:
    """Extract the CUBIN binary from a compiled Megakernel.

    Args:
        kernel: A compiled Megakernel instance (must have called compile() or run()).

    Returns:
        Raw CUBIN bytes.
    """
    if kernel._compiled_kernel is None:
        raise RuntimeError("Kernel not compiled yet. Call compile() or run() first.")

    engine = kernel._compiled_kernel.engine
    with tempfile.NamedTemporaryFile(suffix=".o", delete=False) as f:
        obj_path = f.name
    engine.dump_to_object_file(obj_path)

    with open(obj_path, "rb") as f:
        data = f.read()
    os.unlink(obj_path)

    # Find CUBIN ELF (machine type 190 = EM_CUDA)
    elf_magic = b"\x7fELF"
    pos = 0
    while True:
        idx = data.find(elf_magic, pos)
        if idx == -1:
            raise RuntimeError("No CUBIN found in compiled kernel object")
        e_machine = struct.unpack("<H", data[idx + 18 : idx + 20])[0]
        if e_machine == 190:
            next_elf = data.find(elf_magic, idx + 1)
            end = next_elf if next_elf != -1 else len(data)
            return data[idx:end]
        pos = idx + 1


def dump_sass(kernel, path: str = None) -> str:
    """Dump SASS assembly of a compiled Megakernel.

    Args:
        kernel: A compiled Megakernel instance.
        path: Optional file path to write SASS to.

    Returns:
        The SASS assembly as a string.
    """
    cubin = extract_cubin(kernel)
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
        f.write(cubin)
        cubin_path = f.name
    try:
        result = subprocess.run(
            ["cuobjdump", "-sass", cubin_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        sass = result.stdout
    finally:
        os.unlink(cubin_path)

    if path:
        with open(path, "w") as f:
            f.write(sass)
    return sass


def dump_ptx(kernel, path: str = None) -> str:
    """Dump PTX of a compiled Megakernel.

    Recompiles the kernel with KeepPTX enabled to capture PTX output,
    since PTX is not embedded in the CUBIN by default.

    Args:
        kernel: A compiled Megakernel instance.
        path: Optional file path to write PTX to.

    Returns:
        The PTX source as a string.
    """
    import torch
    import cuda.bindings.driver as cuda_drv
    import cutlass.cute as cute
    from cutlass import Int64

    if kernel._compiled_kernel is None:
        raise RuntimeError("Kernel not compiled yet. Call compile() or run() first.")

    with tempfile.TemporaryDirectory() as dump_dir:
        torch_stream = torch.cuda.current_stream()
        cu_stream = cuda_drv.CUstream(torch_stream.cuda_stream)

        tma_tensor_args = (
            [ct for _, ct in kernel._tma_cute_tensors] if kernel._tma_cute_tensors else []
        )
        peer_tma_tensor_args = (
            [ct for _, _, ct in kernel._peer_tma_cute_tensors]
            if kernel._peer_tma_cute_tensors
            else []
        )

        compiled = cute.compile(
            kernel._create_kernel(),
            Int64(kernel._instructions_tensor.data_ptr()),
            Int64(kernel._barriers_tensor.data_ptr()),
            Int64(kernel._op_configs_tensor.data_ptr()),
            Int64(0),
            kernel._num_instructions_i32,
            *kernel._cute_tensors,
            *tma_tensor_args,
            *peer_tma_tensor_args,
            cu_stream,
            options=f"--dump-dir={dump_dir} --keep-ptx",
        )

        ptx = compiled.artifacts.PTX
        if ptx is None:
            ptx_files = [f for f in os.listdir(dump_dir) if f.endswith(".ptx")]
            if ptx_files:
                with open(os.path.join(dump_dir, ptx_files[0]), "r") as f:
                    ptx = f.read()
            else:
                raise RuntimeError("PTX capture failed — no PTX file produced")

    if path:
        with open(path, "w") as f:
            f.write(ptx)
    return ptx


def dump_cubin(kernel, path: str) -> None:
    """Dump the raw CUBIN binary to a file.

    Args:
        kernel: A compiled Megakernel instance.
        path: File path to write the CUBIN to.
    """
    cubin = extract_cubin(kernel)
    with open(path, "wb") as f:
        f.write(cubin)
