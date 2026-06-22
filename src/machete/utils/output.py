"""Output-control helpers for benchmark setup paths."""

from __future__ import annotations

from contextlib import contextmanager
import os
import sys


@contextmanager
def suppress_stdout_stderr(enabled: bool = True):
    """Temporarily redirect process stdout/stderr to /dev/null.

    This is intentionally file-descriptor based, not just `redirect_stdout`,
    because TorchInductor/Triton autotune logging may be emitted below Python's
    `sys.stdout`/`sys.stderr` objects.
    """
    if not enabled:
        yield
        return

    sys.stdout.flush()
    sys.stderr.flush()
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)


def suppress_torch_compile_logs() -> bool:
    """Return whether benchmark torch.compile setup logs should be suppressed."""
    return os.environ.get("MACHETE_SHOW_TORCH_COMPILE_LOGS", "").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }

