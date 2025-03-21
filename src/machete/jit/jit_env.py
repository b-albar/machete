import os

CFLAGS = [
    "-O3",
    "-Wno-switch-bool"
]

CUDA_CFLAGS = [
    "-O3",
    "-std=c++20",
    "-use_fast_math",
    "-Xnvlink=--verbose",
    "-Xptxas=--verbose",
    "-Xptxas=--warn-on-spills",
    "-MD", "-MT", "-MF",
    "-x", "cu",
    "-w"
]

LD_FLAGS = [
    "-lrt", "-lpthread", "-ldl",
    "-lcuda", "-lcudadevrt", "-lcudart_static"
]

JIT_DIR = os.environ.get("MACHETE_JIT_DIR", "/tmp/machete_jit")

TK_INCLUDE_DIR = "csrc/ThunderKittens/include"
