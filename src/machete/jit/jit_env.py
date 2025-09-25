import os
import pathlib

CFLAGS = [
    "-O3",
    "-Wno-switch-bool"
]

CUDA_CFLAGS = [
    "-O3",
    "-std=c++20",
    "-use_fast_math",
    "-Xnvlink=--verbose",
    "-Xptxas=--warn-on-spills --verbose",
    "-MD", "-MT", "-MF",
    "-x", "cu",
    "-w", "--extended-lambda"
]

LD_FLAGS = [
    "-lrt", "-lpthread", "-ldl",
    "-lcuda", "-lcudadevrt", "-lcudart_static"
]

JIT_DIR = os.environ.get("MACHETE_JIT_DIR", "/tmp/machete_jit")

ROOT_DIR = pathlib.Path(__file__).parent.parent.parent.parent
TK_INCLUDE_DIR = ROOT_DIR / "csrc/ThunderKittens/include"
MACHETE_INCLUDE_DIR = ROOT_DIR / "csrc/utils"
