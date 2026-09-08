"""Linux execution supervisor; adopted descendants belong to this one figure run."""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys
import time

import psutil


def _adopt_orphans() -> None:
    if sys.platform != "linux":
        raise RuntimeError("The source-preserving figure runner requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _reap() -> None:
    while True:
        try:
            if os.waitpid(-1, os.WNOHANG)[0] == 0:
                return
        except ChildProcessError:
            return


def _cleanup(parent: psutil.Process) -> None:
    deadline = time.monotonic() + 1.5
    graceful = True
    while True:
        children = parent.children(recursive=True)
        if not children:
            _reap()
            return
        for child in children:
            try:
                child.terminate() if graceful else child.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(children, timeout=0.1)
        _reap()
        graceful = False
        if time.monotonic() >= deadline:
            if any(child.status() != psutil.STATUS_ZOMBIE for child in parent.children(recursive=True)):
                raise RuntimeError("Owned descendant cleanup did not complete")
            return


def main(command: list[str]) -> int:
    _adopt_orphans()
    stopping = False

    def stop(signum: int, frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    parent = psutil.Process()
    process = subprocess.Popen(command)
    try:
        while not stopping:
            try:
                return process.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                pass
        return 124
    finally:
        _cleanup(parent)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
