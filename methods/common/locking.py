"""Small POSIX primitives for shared benchmark artifacts.

The multi-node scheduler runs independent method/task processes against one
shared filesystem.  A lock is therefore part of the artifact protocol, not a
scheduler-specific workaround: checkpoint groups, vendor caches, and masks
all have a single writer while any number of later processes may reuse them.
"""
from __future__ import annotations

import fcntl
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def exclusive_file_lock(path: str | Path) -> Iterator[None]:
    """Hold an advisory exclusive lock for the duration of the context."""
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def exclusive_directory_lock(
    path: str | Path,
    *,
    timeout_seconds: float = 120.0,
    stale_after_seconds: float = 600.0,
    poll_seconds: float = 0.1,
) -> Iterator[None]:
    """Hold a mkdir-based lock that also works on shared/NFS filesystems.

    POSIX ``mkdir`` is atomic across hosts, whereas advisory ``flock`` support
    varies across distributed filesystems.  The lock directory is deliberately
    empty so a writer can recover a stale lock after the bounded grace period.
    This primitive is intended for short metadata publications (for example,
    deterministic mask bundles), not for long model-training sections.
    """
    lock_dir = Path(path)
    lock_dir.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + float(timeout_seconds)
    owned_inode: int | None = None
    while True:
        try:
            lock_dir.mkdir()
            owned_inode = lock_dir.stat().st_ino
            break
        except FileExistsError:
            try:
                age = time.time() - lock_dir.stat().st_mtime
                if age > float(stale_after_seconds):
                    lock_dir.rmdir()
                    continue
            except FileNotFoundError:
                continue
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for shared lock: {lock_dir}")
            time.sleep(float(poll_seconds))
    try:
        yield
    finally:
        try:
            # A stale-lock recovery may have replaced this directory while a
            # badly delayed owner was alive. Never remove the new owner's lock.
            if lock_dir.stat().st_ino == owned_inode:
                lock_dir.rmdir()
        except FileNotFoundError:
            pass


@contextmanager
def atomic_output_path(path: str | Path) -> Iterator[Path]:
    """Yield a same-directory temporary path, then atomically replace target."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(fd)
    tmp = Path(raw_tmp)
    try:
        yield tmp
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
