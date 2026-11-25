"""One `run` or `calibrate` at a time (and one `subs resolve`, on its own lock).

The lock is an exclusive OS lock on a file beside the state database.

A second run would reset the first one's live records, sweep its scratch mid-encode and race on
``manifest.csv``. The lock is an OS lock (``msvcrt.locking`` on Windows, ``fcntl.flock``
elsewhere), so it disappears with the process: a lock file left by a crash never blocks. The
file holds the holder's PID for the refusal message; the lock itself is taken on a byte far
past the PID so the PID stays readable on Windows (where locked bytes cannot be read).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

LOCK_NAME = "video-cleaner.lock"
SUBS_LOCK_NAME = "video-cleaner.subs.lock"
# Windows locks a byte range; this byte lies beyond any PID text (locking past EOF is allowed).
_LOCK_OFFSET = 1 << 20
_PID_BYTES = 32


class AlreadyRunning(Exception):
    """Another process holds the lock."""

    def __init__(self, lock: Path, pid: int | None) -> None:
        holder = f"process {pid}" if pid is not None else "another process"
        what = (
            "another `subs resolve` is running"
            if lock.name == SUBS_LOCK_NAME
            else "another video-cleaner run or calibration is active"
        )
        super().__init__(f"{what} ({holder} holds {lock}); only one may run at a time")
        self.pid = pid


def lock_path_for(state_db: Path) -> Path:
    return state_db.parent / LOCK_NAME


def subs_lock_path_for(state_db: Path) -> Path:
    """The lock of `subs resolve`: its own file, so a subtitle lookup may run beside a `run`
    (which keeps `lock_path_for`) while two lookups never overlap."""
    return state_db.parent / SUBS_LOCK_NAME


def _try_lock(fd: int) -> bool:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _unlock(fd: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


def _read_pid(fd: int) -> int | None:
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        text = os.read(fd, _PID_BYTES).decode("ascii", "replace").strip()
        return int(text)
    except (OSError, ValueError):
        return None


@contextmanager
def instance_lock(path: Path) -> Iterator[None]:
    """Hold the exclusive lock on *path* for the duration of the block.

    Raises `AlreadyRunning` (with the holder's PID when readable) if another handle holds it,
    without touching the holder's PID.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if not _try_lock(fd):
            raise AlreadyRunning(path, _read_pid(fd))
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, f"{os.getpid()}\n".encode("ascii").ljust(_PID_BYTES))
            yield
        finally:
            _unlock(fd)
    finally:
        os.close(fd)
