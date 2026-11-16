"""Scratch space: verified copy from the library, a size cap, and background prefetch.

The library is read-only: this module only ever reads the source and writes under the
scratch root. A copy is streamed to ``<dest>.part`` and renamed into place only once it is
complete and the source is seen unchanged, so a file at its final name is always a whole,
verified copy. Any failure (NAS drop, disk full, source changed, Ctrl+C) removes the partial
file and is re-raised for the caller to record.
"""

from __future__ import annotations

import errno
import hashlib
import logging
import os
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from video_cleaner.paths import path_key as _key
from video_cleaner.progress import Progress, Reporter

_log = logging.getLogger(__name__)

_MIB = 2**20
_PART_SUFFIX = ".part"
_REPORT_INTERVAL_S = 0.1  # at most ~10 progress updates per second
_PREFETCH_CHUNK = 8 * _MIB
_JOIN_POLL_S = 0.2
# ERROR_HANDLE_DISK_FULL, ERROR_DISK_FULL: belt and braces, Python maps these to ENOSPC.
_WIN_DISK_FULL = frozenset({39, 112})

_monotonic = time.monotonic


class SourceChanged(Exception):
    """The source's size or mtime differed after the copy: the copy is stale and discarded."""


class ScratchFull(Exception):
    """The scratch volume ran out of space, or the scratch cap would be exceeded."""


class _Cancelled(Exception):
    """Internal: a prefetch was cancelled mid-copy."""


@dataclass(frozen=True)
class CopyResult:
    dest: Path
    size: int
    sha256: str


def _is_disk_full(exc: OSError) -> bool:
    return exc.errno == errno.ENOSPC or getattr(exc, "winerror", None) in _WIN_DISK_FULL


def _remove_quietly(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        _log.warning("could not remove partial file %s: %s", path, exc)


def _report(reporter: Reporter, copied: int, total: int, elapsed_s: float) -> None:
    percent = 100.0 if total <= 0 else min(100.0, copied / total * 100.0)
    rate = copied / elapsed_s if elapsed_s > 0 and copied > 0 else None
    eta = max(0, total - copied) / rate if rate else None
    note = f"{rate / _MIB:.0f} MB/s" if rate else ""
    reporter.update(Progress(percent=percent, fps=None, speed=None, eta_s=eta, note=note))


def _copy_verified(
    src: Path,
    dest: Path,
    part: Path,
    reporter: Reporter,
    chunk: int,
    cancelled: Callable[[], bool] | None,
) -> CopyResult:
    before = src.stat()
    total = before.st_size
    digest = hashlib.sha256()
    copied = 0
    reported = -1
    started = last_report = _monotonic()
    with open(src, "rb") as fin, open(part, "wb") as fout:
        while True:
            if cancelled is not None and cancelled():
                raise _Cancelled
            data = fin.read(chunk)
            if not data:
                break
            digest.update(data)
            fout.write(data)
            copied += len(data)
            now = _monotonic()
            if now - last_report >= _REPORT_INTERVAL_S:
                last_report = now
                reported = copied
                _report(reporter, copied, total, now - started)
    if reported != copied:
        _report(reporter, copied, total, _monotonic() - started)

    after = src.stat()
    if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
        raise SourceChanged(
            f"{src} changed during the copy "
            f"(size {before.st_size} -> {after.st_size}, "
            f"mtime_ns {before.st_mtime_ns} -> {after.st_mtime_ns})"
        )
    written = part.stat().st_size
    if copied != total or written != total:
        raise OSError(
            errno.EIO,
            f"copy size mismatch for {src}: source {total}, read {copied}, written {written}",
        )
    os.replace(part, dest)
    return CopyResult(dest=dest, size=total, sha256=digest.hexdigest())


def _copy(
    src: Path,
    dest: Path,
    reporter: Reporter,
    chunk: int,
    cancelled: Callable[[], bool] | None,
) -> CopyResult:
    part = dest.with_name(dest.name + _PART_SUFFIX)
    try:
        try:
            return _copy_verified(src, dest, part, reporter, chunk, cancelled)
        except OSError as exc:
            if _is_disk_full(exc):
                raise ScratchFull(f"scratch is full while copying {src}: {exc}") from exc
            raise
    except BaseException:
        # Both file handles are closed by now, so the partial can be removed on Windows too.
        _remove_quietly(part)
        raise


def copy_with_progress(
    src: Path, dest: Path, reporter: Reporter, *, chunk: int = 8 * _MIB
) -> CopyResult:
    """Copy *src* to *dest* with progress, a SHA-256 of the bytes read and a size check.

    Raises `SourceChanged` if the source's size or mtime differs after the copy, `ScratchFull`
    if the destination volume is full, and re-raises anything else. On any error (including
    KeyboardInterrupt) the partial file is removed and *dest* is never created or replaced.
    """
    return _copy(src, dest, reporter, chunk, None)


class ScratchManager:
    """Keeps the scratch folder under a byte cap. Thread-safe.

    ``reserve`` counts what is already promised (reservations, in memory) plus what is on disk
    under *root* (so leftovers from an earlier process count). Pass the file's final *path* to
    ``reserve`` so that the file and its ``.part`` are not counted a second time once they are
    written; a reservation without a path is counted on top of whatever is on disk.
    """

    def __init__(self, root: Path, cap_bytes: int) -> None:
        self.root = root
        self.cap_bytes = cap_bytes
        root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._bound: dict[str, int] = {}
        self._unbound = 0

    def _files(self) -> Iterator[tuple[str, int]]:
        for folder, _dirs, names in os.walk(self.root):
            for name in names:
                full = os.path.join(folder, name)
                try:
                    yield _key(full), os.stat(full).st_size
                except OSError:  # removed while scanning
                    continue

    def used_bytes(self) -> int:
        """Bytes of all files currently under the scratch root."""
        return sum(size for _, size in self._files())

    def _committed(self) -> int:
        covered = set(self._bound) | {key + _PART_SUFFIX for key in self._bound}
        untracked = sum(size for key, size in self._files() if key not in covered)
        return sum(self._bound.values()) + self._unbound + untracked

    def reserve(self, size: int, path: Path | None = None) -> bool:
        """Atomically claim *size* bytes; False if that would exceed the cap."""
        if size < 0:
            raise ValueError(f"cannot reserve a negative size: {size}")
        with self._lock:
            key = _key(path) if path is not None else None
            if key is not None and key in self._bound:
                raise ValueError(f"{path} is already reserved")
            if self._committed() + size > self.cap_bytes:
                return False
            if key is None:
                self._unbound += size
            else:
                self._bound[key] = size
            return True

    def release(self, path: Path) -> None:
        """Delete *path* (and its ``.part``; missing is fine) and free its reservation.

        Refuses anything outside the scratch root: this is the only place that deletes a
        caller-supplied path, and it must never reach the library.
        """
        key = _key(path)
        root = _key(self.root)
        if not key.startswith(root.rstrip("\\/") + os.sep):
            raise ValueError(f"refusing to delete {path}: outside the scratch root {self.root}")
        with self._lock:
            try:
                size = os.stat(key).st_size
            except OSError:
                size = 0
            # A failed delete propagates and keeps the reservation: the bytes are still there.
            Path(key).unlink(missing_ok=True)
            Path(key + _PART_SUFFIX).unlink(missing_ok=True)
            if key in self._bound:
                del self._bound[key]
            else:
                self._unbound -= min(self._unbound, size)


class _Job:
    def __init__(self) -> None:
        self.cancel = threading.Event()
        self.thread: threading.Thread | None = None
        self.dest: Path | None = None
        self.result: CopyResult | None = None
        self.error: BaseException | None = None


def _join(thread: threading.Thread, timeout: float | None = None) -> None:
    """Join in short slices so Ctrl+C is still delivered on Windows."""
    deadline = None if timeout is None else _monotonic() + timeout
    while thread.is_alive():
        if deadline is None:
            thread.join(_JOIN_POLL_S)
            continue
        remaining = deadline - _monotonic()
        if remaining <= 0:
            return
        thread.join(min(_JOIN_POLL_S, remaining))


class Prefetcher:
    """Copies the next file into scratch in a background thread while the current one encodes.

    One prefetch at a time: after ``start`` the outcome must be collected with ``result`` (which
    re-raises whatever went wrong in the thread) or dropped with ``cancel`` before the next
    ``start``. A failed or cancelled prefetch leaves nothing in scratch and holds no reservation;
    a successful one keeps its reservation until ``scratch.release(result.dest)``.
    """

    def __init__(self, scratch: ScratchManager, reporter: Reporter) -> None:
        self._scratch = scratch
        self._reporter = reporter
        self._lock = threading.Lock()
        self._job: _Job | None = None

    def dest_for(self, src: Path) -> Path:
        digest = hashlib.sha1(
            str(src).encode("utf-8", errors="surrogatepass"), usedforsecurity=False
        ).hexdigest()[:8]
        return self._scratch.root / f"{digest}-{src.name}"

    def start(self, src: Path) -> None:
        with self._lock:
            if self._job is not None:
                raise RuntimeError("a prefetch is already in flight: call result() or cancel()")
            job = _Job()
            job.thread = threading.Thread(
                target=self._run, args=(job, src), name=f"prefetch-{src.name}", daemon=True
            )
            self._job = job
            job.thread.start()

    def _run(self, job: _Job, src: Path) -> None:
        try:
            size = src.stat().st_size
            dest = self.dest_for(src)
            if not self._scratch.reserve(size, dest):
                raise ScratchFull(f"no room in scratch for {src} ({size} bytes)")
            job.dest = dest
            result = _copy(src, dest, self._reporter, _PREFETCH_CHUNK, job.cancel.is_set)
            if job.cancel.is_set():
                raise _Cancelled
            job.result = result
        except BaseException as exc:
            job.error = exc  # handed to result(); never swallowed
            if job.dest is not None:
                try:
                    self._scratch.release(job.dest)
                except OSError as cleanup_exc:
                    _log.warning("could not clean up %s: %s", job.dest, cleanup_exc)

    def result(self) -> CopyResult:
        """Wait for the prefetch and return its result, re-raising any error from the thread."""
        with self._lock:
            job = self._job
        if job is None or job.thread is None:
            raise RuntimeError("no prefetch has been started")
        _join(job.thread)
        with self._lock:
            if self._job is job:
                self._job = None
        if isinstance(job.error, _Cancelled):
            raise RuntimeError("the prefetch was cancelled") from job.error
        if job.error is not None:
            raise job.error
        if job.result is None:  # pragma: no cover - the thread always sets one of the two
            raise RuntimeError("the prefetch finished without a result")
        return job.result

    def cancel(self, timeout: float | None = None) -> None:
        """Best effort: stop the copy, remove what it wrote and free its reservation.

        The copy notices between chunks, so a read stalled on a dead NAS only stops when the
        read returns. With a *timeout* that expires first, the thread cleans up after itself
        when it does stop and the prefetch stays "in flight" until ``cancel`` is called again.
        """
        with self._lock:
            job = self._job
        if job is None or job.thread is None:
            return
        job.cancel.set()
        _join(job.thread, timeout)
        if job.thread.is_alive():
            return
        with self._lock:
            if self._job is job:
                self._job = None
        if job.result is not None:
            self._scratch.release(job.result.dest)
