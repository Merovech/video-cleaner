"""Scratch space, verified copy and background prefetch."""

from __future__ import annotations

import errno
import hashlib
import os
import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

import pytest

from video_cleaner import scratch
from video_cleaner.progress import Progress
from video_cleaner.scratch import (
    CopyResult,
    Prefetcher,
    ScratchFull,
    ScratchManager,
    SourceChanged,
    copy_with_progress,
)

CHUNK = 64 * 1024


class FakeReporter:
    def __init__(self) -> None:
        self.updates: list[Progress] = []
        self.stages: list[str] = []
        self._lock = threading.Lock()

    def stage(self, label: str) -> None:
        with self._lock:
            self.stages.append(label)

    def update(self, p: Progress) -> None:
        with self._lock:
            self.updates.append(p)

    def heartbeat(self, label: str, elapsed_s: float) -> None:
        pass

    def file_done(self, summary: str) -> None:
        pass


class FileProxy:
    """Wraps a real binary file; hooks run before each read / write."""

    def __init__(
        self,
        inner: IO[bytes],
        *,
        on_read: Callable[[int], None] | None = None,
        on_write: Callable[[int], None] | None = None,
        drop_writes: bool = False,
    ) -> None:
        self._inner = inner
        self._on_read = on_read
        self._on_write = on_write
        self._drop_writes = drop_writes
        self._reads = 0
        self._writes = 0

    def read(self, n: int = -1) -> bytes:
        if self._on_read is not None:
            self._on_read(self._reads)
        self._reads += 1
        return self._inner.read(n)

    def write(self, data: bytes) -> int:
        if self._on_write is not None:
            self._on_write(self._writes)
        self._writes += 1
        if self._drop_writes:
            return len(data)
        return self._inner.write(data)

    def __enter__(self) -> FileProxy:
        return self

    def __exit__(self, *exc: object) -> None:
        self._inner.close()


def patch_open(monkeypatch: pytest.MonkeyPatch, target: Path, **proxy_kwargs: Any) -> None:
    """Make `scratch` open *target* through a `FileProxy`; other files open normally."""

    def fake_open(path: Path, mode: str, *args: Any, **kwargs: Any) -> Any:
        real = open(path, mode, *args, **kwargs)  # noqa: SIM115
        if Path(path) == target:
            return FileProxy(real, **proxy_kwargs)
        return real

    monkeypatch.setattr(scratch, "open", fake_open, raising=False)


def make_src(tmp_path: Path, size: int = 5 * CHUNK + 123, name: str = "movie.mkv") -> Path:
    folder = tmp_path / "nas"
    folder.mkdir(exist_ok=True)
    src = folder / name
    src.write_bytes(os.urandom(size))
    return src


def raise_after(n: int, exc: BaseException) -> Callable[[int], None]:
    def hook(count: int) -> None:
        if count >= n:
            raise exc

    return hook


def assert_nothing_left(dest: Path) -> None:
    assert not dest.exists()
    assert list(dest.parent.iterdir()) == []


# --- copy_with_progress -------------------------------------------------------------------


def test_copy_verifies_size_and_hash(tmp_path: Path) -> None:
    src = make_src(tmp_path)
    out = tmp_path / "scratch"
    out.mkdir()
    dest = out / "copy.mkv"

    result = copy_with_progress(src, dest, FakeReporter(), chunk=CHUNK)

    data = src.read_bytes()
    assert result == CopyResult(dest=dest, size=len(data), sha256=hashlib.sha256(data).hexdigest())
    assert dest.read_bytes() == data
    assert [p.name for p in out.iterdir()] == ["copy.mkv"]


def test_copy_of_empty_file(tmp_path: Path) -> None:
    src = make_src(tmp_path, size=0)
    dest = tmp_path / "copy.mkv"
    reporter = FakeReporter()

    result = copy_with_progress(src, dest, reporter)

    assert result.size == 0
    assert result.sha256 == hashlib.sha256(b"").hexdigest()
    assert dest.read_bytes() == b""
    assert reporter.updates[-1].percent == 100.0


def test_copy_handles_odd_file_names(tmp_path: Path) -> None:
    src = make_src(tmp_path, size=1000, name="The First Omen (2024).en.MX] é中.mkv")
    dest = tmp_path / ("abcd1234-" + src.name)

    result = copy_with_progress(src, dest, FakeReporter())

    assert result.dest == dest
    assert dest.read_bytes() == src.read_bytes()


def test_copy_raises_when_dest_size_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = make_src(tmp_path)
    out = tmp_path / "scratch"
    out.mkdir()
    dest = out / "copy.mkv"
    patch_open(monkeypatch, out / "copy.mkv.part", drop_writes=True)

    with pytest.raises(OSError, match="size"):
        copy_with_progress(src, dest, FakeReporter(), chunk=CHUNK)

    assert_nothing_left(dest)


def test_copy_removes_partial_file_on_read_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = make_src(tmp_path)
    out = tmp_path / "scratch"
    out.mkdir()
    dest = out / "copy.mkv"
    seen_partial: list[bool] = []

    def hook(count: int) -> None:
        if count >= 2:
            seen_partial.append((out / "copy.mkv.part").exists())
            raise OSError(errno.EIO, "the NAS went away")

    patch_open(monkeypatch, src, on_read=hook)

    with pytest.raises(OSError, match="NAS went away") as info:
        copy_with_progress(src, dest, FakeReporter(), chunk=CHUNK)

    assert not isinstance(info.value, ScratchFull)
    assert seen_partial == [True]
    assert_nothing_left(dest)


def test_copy_removes_partial_file_on_keyboard_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = make_src(tmp_path)
    out = tmp_path / "scratch"
    out.mkdir()
    dest = out / "copy.mkv"
    patch_open(monkeypatch, src, on_read=raise_after(2, KeyboardInterrupt()))

    with pytest.raises(KeyboardInterrupt):
        copy_with_progress(src, dest, FakeReporter(), chunk=CHUNK)

    assert_nothing_left(dest)


def test_copy_raises_when_source_is_missing(tmp_path: Path) -> None:
    out = tmp_path / "scratch"
    out.mkdir()
    dest = out / "copy.mkv"

    with pytest.raises(FileNotFoundError):
        copy_with_progress(tmp_path / "gone.mkv", dest, FakeReporter())

    assert_nothing_left(dest)


def test_copy_raises_source_changed_when_mtime_changes_during_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = make_src(tmp_path)
    out = tmp_path / "scratch"
    out.mkdir()
    dest = out / "copy.mkv"
    before = src.stat()

    def touch(count: int) -> None:
        if count == 1:
            os.utime(src, ns=(before.st_atime_ns, before.st_mtime_ns + 5_000_000_000))

    patch_open(monkeypatch, src, on_read=touch)

    with pytest.raises(SourceChanged):
        copy_with_progress(src, dest, FakeReporter(), chunk=CHUNK)

    assert src.stat().st_size == before.st_size
    assert_nothing_left(dest)


def test_copy_raises_source_changed_when_size_changes_during_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = make_src(tmp_path)
    out = tmp_path / "scratch"
    out.mkdir()
    dest = out / "copy.mkv"
    before = src.stat()

    def grow(count: int) -> None:
        if count == 1:
            with open(src, "ab") as fh:
                fh.write(b"more")
            # Same mtime as before: only the size gives the change away.
            os.utime(src, ns=(before.st_atime_ns, before.st_mtime_ns))

    patch_open(monkeypatch, src, on_read=grow)

    with pytest.raises(SourceChanged):
        copy_with_progress(src, dest, FakeReporter(), chunk=CHUNK)

    assert_nothing_left(dest)


def test_failed_copy_keeps_the_source_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = make_src(tmp_path)
    data = src.read_bytes()
    before = src.stat()
    dest = tmp_path / "copy.mkv"
    patch_open(monkeypatch, src, on_read=raise_after(1, OSError(errno.EIO, "boom")))

    with pytest.raises(OSError, match="boom"):
        copy_with_progress(src, dest, FakeReporter(), chunk=CHUNK)

    assert src.read_bytes() == data
    assert src.stat().st_mtime_ns == before.st_mtime_ns


def test_disk_full_raises_scratch_full_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = make_src(tmp_path)
    out = tmp_path / "scratch"
    out.mkdir()
    dest = out / "copy.mkv"
    patch_open(
        monkeypatch,
        out / "copy.mkv.part",
        on_write=raise_after(2, OSError(errno.ENOSPC, "No space left on device")),
    )

    with pytest.raises(ScratchFull) as info:
        copy_with_progress(src, dest, FakeReporter(), chunk=CHUNK)

    assert isinstance(info.value.__cause__, OSError)
    assert_nothing_left(dest)


def test_disk_full_when_creating_the_file_raises_scratch_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = make_src(tmp_path)
    out = tmp_path / "scratch"
    out.mkdir()
    dest = out / "copy.mkv"

    def fake_open(path: Path, mode: str, *args: Any, **kwargs: Any) -> Any:
        if "w" in mode:
            raise OSError(errno.ENOSPC, "No space left on device")
        return open(path, mode, *args, **kwargs)

    monkeypatch.setattr(scratch, "open", fake_open, raising=False)

    with pytest.raises(ScratchFull):
        copy_with_progress(src, dest, FakeReporter(), chunk=CHUNK)

    assert_nothing_left(dest)


def test_copy_progress_reports_percent_and_throughput(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mib = 2**20
    src = make_src(tmp_path, size=4 * mib)
    reporter = FakeReporter()
    ticks = iter(range(1000))
    monkeypatch.setattr(scratch, "_monotonic", lambda: float(next(ticks)))  # 1 s per call

    copy_with_progress(src, tmp_path / "copy.mkv", reporter, chunk=mib)

    percents = [p.percent for p in reporter.updates]
    assert percents[:4] == [25.0, 50.0, 75.0, 100.0]
    assert percents[-1] == 100.0
    assert all(p.fps is None and p.speed is None for p in reporter.updates)
    assert all(re.fullmatch(r"\d+ MB/s", p.note) for p in reporter.updates)
    first = reporter.updates[0]  # 1 MiB after 1 s, 3 MiB to go
    assert first.note == "1 MB/s"
    assert first.eta_s == pytest.approx(3.0)
    assert reporter.updates[-1].eta_s == pytest.approx(0.0)


def test_copy_progress_is_rate_limited(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = make_src(tmp_path, size=200 * 1024)
    reporter = FakeReporter()
    monkeypatch.setattr(scratch, "_monotonic", lambda: 50.0)  # time stands still

    copy_with_progress(src, tmp_path / "copy.mkv", reporter, chunk=1024)

    # 200 chunks within the same instant: only the final 100% gets through.
    assert [p.percent for p in reporter.updates] == [100.0]
    assert reporter.updates[0].eta_s is None
    assert reporter.updates[0].note == ""


# --- ScratchManager -----------------------------------------------------------------------


def test_scratch_manager_creates_root(tmp_path: Path) -> None:
    root = tmp_path / "a" / "b" / "scratch"

    manager = ScratchManager(root, 100)

    assert root.is_dir()
    assert manager.root == root
    assert manager.used_bytes() == 0


def test_reserve_refuses_over_cap(tmp_path: Path) -> None:
    manager = ScratchManager(tmp_path / "scratch", 100)

    assert manager.reserve(60) is True
    assert manager.reserve(41) is False
    assert manager.reserve(40) is True
    assert manager.reserve(1) is False


def test_reserve_rejects_negative_size(tmp_path: Path) -> None:
    manager = ScratchManager(tmp_path / "scratch", 100)

    with pytest.raises(ValueError, match="negative"):
        manager.reserve(-1)


def test_reserve_counts_leftovers_from_an_earlier_process(tmp_path: Path) -> None:
    root = tmp_path / "scratch"
    (root / "sub").mkdir(parents=True)
    (root / "left.mkv").write_bytes(b"x" * 30)
    (root / "sub" / "left.part").write_bytes(b"x" * 20)

    manager = ScratchManager(root, 100)

    assert manager.used_bytes() == 50
    assert manager.reserve(51) is False
    assert manager.reserve(50) is True


def test_reserved_file_is_not_counted_twice_once_written(tmp_path: Path) -> None:
    manager = ScratchManager(tmp_path / "scratch", 100)
    path = manager.root / "a.mkv"

    assert manager.reserve(60, path) is True
    (manager.root / "a.mkv.part").write_bytes(b"x" * 30)  # copy in flight
    assert manager.reserve(40) is True
    manager.release(manager.root / "other")  # frees nothing that exists
    path.write_bytes(b"x" * 60)
    assert manager.used_bytes() == 90
    assert manager.reserve(1) is False


def test_reserving_the_same_path_twice_is_an_error(tmp_path: Path) -> None:
    manager = ScratchManager(tmp_path / "scratch", 100)
    path = manager.root / "a.mkv"
    assert manager.reserve(10, path) is True

    with pytest.raises(ValueError, match="already reserved"):
        manager.reserve(10, path)


def test_release_deletes_file_and_frees_cap(tmp_path: Path) -> None:
    manager = ScratchManager(tmp_path / "scratch", 100)
    path = manager.root / "a.mkv"
    assert manager.reserve(100, path) is True
    path.write_bytes(b"x" * 100)
    (manager.root / "a.mkv.part").write_bytes(b"x" * 5)
    assert manager.reserve(1) is False

    manager.release(path)

    assert list(manager.root.iterdir()) == []
    assert manager.used_bytes() == 0
    assert manager.reserve(100) is True


def test_release_of_missing_file_is_fine_and_frees_reservation(tmp_path: Path) -> None:
    manager = ScratchManager(tmp_path / "scratch", 100)
    path = manager.root / "never-written.mkv"
    assert manager.reserve(100, path) is True

    manager.release(path)
    manager.release(path)

    assert manager.reserve(100) is True


def test_release_frees_an_unbound_reservation_by_file_size(tmp_path: Path) -> None:
    manager = ScratchManager(tmp_path / "scratch", 100)
    assert manager.reserve(50) is True
    path = manager.root / "a.mkv"
    path.write_bytes(b"x" * 50)
    assert manager.reserve(1) is False  # conservative: reservation and file both count

    manager.release(path)

    assert manager.reserve(100) is True


def test_release_refuses_paths_outside_the_scratch_root(tmp_path: Path) -> None:
    manager = ScratchManager(tmp_path / "scratch", 100)
    outside = tmp_path / "library" / "movie.mkv"
    outside.parent.mkdir()
    outside.write_bytes(b"precious")

    with pytest.raises(ValueError, match="outside"):
        manager.release(outside)
    with pytest.raises(ValueError, match="outside"):
        manager.release(manager.root / ".." / "library" / "movie.mkv")

    assert outside.read_bytes() == b"precious"


def test_reserve_is_atomic_across_threads(tmp_path: Path) -> None:
    manager = ScratchManager(tmp_path / "scratch", 100)
    barrier = threading.Barrier(32)
    granted: list[bool] = []

    def worker() -> None:
        barrier.wait()
        granted.append(manager.reserve(10))

    threads = [threading.Thread(target=worker) for _ in range(32)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert granted.count(True) == 10
    assert granted.count(False) == 22


# --- Prefetcher ---------------------------------------------------------------------------


def expected_dest(manager: ScratchManager, src: Path) -> Path:
    digest = hashlib.sha1(str(src).encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
    return manager.root / f"{digest}-{src.name}"


def gate_reads(
    monkeypatch: pytest.MonkeyPatch, src: Path
) -> tuple[threading.Event, threading.Event]:
    """Block the copy inside its second read until the returned `go` event is set."""
    reading = threading.Event()
    go = threading.Event()

    def hook(count: int) -> None:
        if count == 1:
            reading.set()
            assert go.wait(10)

    patch_open(monkeypatch, src, on_read=hook)
    return reading, go


def test_prefetch_runs_in_background_and_returns_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = make_src(tmp_path)
    manager = ScratchManager(tmp_path / "scratch", 10 * 2**20)
    reporter = FakeReporter()
    reading, go = gate_reads(monkeypatch, src)
    monkeypatch.setattr(scratch, "_PREFETCH_CHUNK", CHUNK)
    prefetcher = Prefetcher(manager, reporter)

    prefetcher.start(src)  # returns while the copy is still blocked mid-file

    assert reading.wait(10)
    dest = expected_dest(manager, src)
    assert not dest.exists()
    assert dest.with_name(dest.name + ".part").exists()
    with pytest.raises(RuntimeError, match="in flight"):
        prefetcher.start(src)
    go.set()
    result = prefetcher.result()

    data = src.read_bytes()
    assert result == CopyResult(dest=dest, size=len(data), sha256=hashlib.sha256(data).hexdigest())
    assert dest.read_bytes() == data
    assert reporter.updates[-1].percent == 100.0
    # The finished copy still holds its reservation until released.
    assert manager.reserve(10 * 2**20 - len(data) + 1) is False
    manager.release(result.dest)
    assert manager.reserve(10 * 2**20) is True


def test_prefetch_thread_is_a_daemon(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = make_src(tmp_path)
    manager = ScratchManager(tmp_path / "scratch", 10 * 2**20)
    reading, go = gate_reads(monkeypatch, src)
    monkeypatch.setattr(scratch, "_PREFETCH_CHUNK", CHUNK)
    prefetcher = Prefetcher(manager, FakeReporter())

    prefetcher.start(src)
    assert reading.wait(10)
    workers = [t for t in threading.enumerate() if t.name.startswith("prefetch")]
    go.set()
    prefetcher.result()

    assert len(workers) == 1
    assert workers[0].daemon is True


def test_prefetch_result_reraises_copy_errors_and_frees_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = make_src(tmp_path)
    manager = ScratchManager(tmp_path / "scratch", 10 * 2**20)
    patch_open(monkeypatch, src, on_read=raise_after(2, OSError(errno.EIO, "the NAS went away")))
    monkeypatch.setattr(scratch, "_PREFETCH_CHUNK", CHUNK)
    prefetcher = Prefetcher(manager, FakeReporter())

    prefetcher.start(src)
    with pytest.raises(OSError, match="NAS went away"):
        prefetcher.result()

    assert list(manager.root.iterdir()) == []
    assert manager.reserve(10 * 2**20) is True


def test_prefetch_result_raises_for_a_missing_source(tmp_path: Path) -> None:
    manager = ScratchManager(tmp_path / "scratch", 100)
    prefetcher = Prefetcher(manager, FakeReporter())

    prefetcher.start(tmp_path / "gone.mkv")
    with pytest.raises(FileNotFoundError):
        prefetcher.result()

    assert manager.reserve(100) is True


def test_prefetch_result_raises_scratch_full_when_reserve_fails(tmp_path: Path) -> None:
    src = make_src(tmp_path, size=1000)
    manager = ScratchManager(tmp_path / "scratch", 999)
    prefetcher = Prefetcher(manager, FakeReporter())

    prefetcher.start(src)
    with pytest.raises(ScratchFull):
        prefetcher.result()

    assert list(manager.root.iterdir()) == []
    assert manager.reserve(999) is True


def test_prefetch_result_without_start_is_an_error(tmp_path: Path) -> None:
    prefetcher = Prefetcher(ScratchManager(tmp_path / "scratch", 100), FakeReporter())

    with pytest.raises(RuntimeError, match="no prefetch"):
        prefetcher.result()


def test_prefetch_start_refuses_to_drop_an_uncollected_result(tmp_path: Path) -> None:
    manager = ScratchManager(tmp_path / "scratch", 10_000)
    first = make_src(tmp_path, size=1000, name="one.mkv")
    second = make_src(tmp_path, size=1000, name="two.mkv")
    prefetcher = Prefetcher(manager, FakeReporter())
    prefetcher.start(first)
    for t in threading.enumerate():
        if t.name.startswith("prefetch"):
            t.join(10)

    with pytest.raises(RuntimeError, match="in flight"):
        prefetcher.start(second)

    assert prefetcher.result().dest == expected_dest(manager, first)
    prefetcher.start(second)
    assert prefetcher.result().dest == expected_dest(manager, second)


def test_prefetch_cancel_removes_partial_and_frees_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = make_src(tmp_path)
    manager = ScratchManager(tmp_path / "scratch", 10 * 2**20)
    reading, go = gate_reads(monkeypatch, src)
    monkeypatch.setattr(scratch, "_PREFETCH_CHUNK", CHUNK)
    prefetcher = Prefetcher(manager, FakeReporter())
    prefetcher.start(src)
    assert reading.wait(10)

    canceller = threading.Thread(target=prefetcher.cancel)
    canceller.start()
    go.set()
    canceller.join(10)

    assert not canceller.is_alive()
    assert list(manager.root.iterdir()) == []
    assert manager.reserve(10 * 2**20) is True
    with pytest.raises(RuntimeError, match="no prefetch"):
        prefetcher.result()


def test_prefetch_cancel_after_completion_removes_the_copy(tmp_path: Path) -> None:
    src = make_src(tmp_path, size=1000)
    manager = ScratchManager(tmp_path / "scratch", 1000)
    prefetcher = Prefetcher(manager, FakeReporter())
    prefetcher.start(src)
    for t in threading.enumerate():
        if t.name.startswith("prefetch"):
            t.join(10)
    assert expected_dest(manager, src).exists()

    prefetcher.cancel()
    prefetcher.cancel()  # nothing in flight: a no-op

    assert list(manager.root.iterdir()) == []
    assert manager.reserve(1000) is True
    assert src.exists()


# --- the tool-owned scratch folder ---------------------------------------------------------


def test_prepare_tool_scratch_creates_marked_subfolder(tmp_path: Path) -> None:
    root = scratch.prepare_tool_scratch(tmp_path / "s")
    assert root == tmp_path / "s" / scratch.TOOL_FOLDER
    assert (root / scratch.MARKER_NAME).is_file()
    assert scratch.is_tool_scratch(root)
    assert scratch.prepare_tool_scratch(tmp_path / "s") == root  # idempotent


def test_prepare_tool_scratch_marks_an_empty_existing_folder(tmp_path: Path) -> None:
    (tmp_path / scratch.TOOL_FOLDER).mkdir()
    root = scratch.prepare_tool_scratch(tmp_path)
    assert scratch.is_tool_scratch(root)


def test_prepare_tool_scratch_refuses_a_foreign_non_empty_folder(tmp_path: Path) -> None:
    folder = tmp_path / scratch.TOOL_FOLDER
    folder.mkdir()
    (folder / "x.txt").write_bytes(b"x")
    with pytest.raises(scratch.ScratchNotOurs):
        scratch.prepare_tool_scratch(tmp_path)
    assert not scratch.is_tool_scratch(folder)
    assert (folder / "x.txt").exists()
