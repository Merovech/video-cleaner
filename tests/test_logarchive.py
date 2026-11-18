from __future__ import annotations

import os
import time
import zipfile
from pathlib import Path

import pytest

from video_cleaner import logarchive
from video_cleaner.logarchive import ARCHIVE_DIR_NAME, archive_old_logs

DAY = 86_400.0


def _log(directory: Path, name: str, size: int, age_days: float, now: float) -> Path:
    path = directory / name
    path.write_bytes(b"x" * size)
    stamp = now - age_days * DAY
    os.utime(path, (stamp, stamp))
    return path


def _archived(log_dir: Path) -> dict[str, bytes]:
    """Every entry of every zip in the archive folder, by entry name."""
    entries: dict[str, bytes] = {}
    for archive in sorted((log_dir / ARCHIVE_DIR_NAME).glob("*.zip")):
        with zipfile.ZipFile(archive) as zf:
            entries.update({name: zf.read(name) for name in zf.namelist()})
    return entries


def test_archive_old_logs_by_age(tmp_path: Path) -> None:
    now = time.time()
    old = _log(tmp_path, "run-20260101-000000.log", 10, 40, now)
    new = _log(tmp_path, "run-20261110-000000.log", 10, 5, now)
    removed = archive_old_logs(tmp_path, keep_days=30, now=now)
    assert removed == [old]
    assert not old.exists() and new.exists()
    assert _archived(tmp_path) == {old.name: b"x" * 10}


def test_archive_old_logs_by_size_oldest_first(tmp_path: Path) -> None:
    now = time.time()
    a = _log(tmp_path, "run-a.log", 100, 3, now)
    b = _log(tmp_path, "run-b.log", 100, 2, now)
    c = _log(tmp_path, "run-c.log", 100, 1, now)
    removed = archive_old_logs(tmp_path, max_bytes=150, now=now)
    assert removed == [a, b]
    assert c.exists()
    assert set(_archived(tmp_path)) == {a.name, b.name}


def test_archive_keeps_the_log_text_intact(tmp_path: Path) -> None:
    now = time.time()
    text = (
        b"2026-01-01 DEBUG started"
        + bytes([10])
        + b"2026-01-01 ERROR something broke"
        + bytes([10])
    )
    old = tmp_path / "run-20260101-000000.log"
    old.write_bytes(text)
    os.utime(old, (now - 40 * DAY, now - 40 * DAY))
    archive_old_logs(tmp_path, keep_days=30, now=now)
    assert _archived(tmp_path) == {old.name: text}


def test_archives_are_grouped_by_month_of_the_log(tmp_path: Path) -> None:
    jan = tmp_path / "run-jan.log"
    feb = tmp_path / "run-feb.log"
    for path, stamp in ((jan, "2025-01-15"), (feb, "2025-02-15")):
        path.write_bytes(b"x")
        mtime = time.mktime(time.strptime(stamp, "%Y-%m-%d"))
        os.utime(path, (mtime, mtime))
    archive_old_logs(tmp_path, keep_days=30, now=time.time())
    zips = sorted(p.name for p in (tmp_path / ARCHIVE_DIR_NAME).glob("*.zip"))
    assert zips == ["run-logs-2025-01.zip", "run-logs-2025-02.zip"]


def test_archive_appends_to_an_existing_zip(tmp_path: Path) -> None:
    now = time.time()
    _log(tmp_path, "run-a.log", 10, 40, now)
    archive_old_logs(tmp_path, keep_days=30, now=now)
    _log(tmp_path, "run-b.log", 20, 40, now)
    archive_old_logs(tmp_path, keep_days=30, now=now)
    assert _archived(tmp_path) == {"run-a.log": b"x" * 10, "run-b.log": b"x" * 20}


def test_archive_does_not_duplicate_a_log_already_in_the_zip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = time.time()
    locked = _log(tmp_path, "run-locked.log", 10, 40, now)
    original = Path.unlink

    def unlink(self: Path, missing_ok: bool = False) -> None:
        if self == locked:
            raise PermissionError("in use")
        original(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    assert archive_old_logs(tmp_path, now=now) == []  # archived, but could not be removed
    assert archive_old_logs(tmp_path, now=now) == []  # the second try finds it already stored
    assert list(_archived(tmp_path)) == [locked.name]


def test_archive_keeps_a_different_log_with_the_same_name(tmp_path: Path) -> None:
    now = time.time()
    _log(tmp_path, "run-a.log", 10, 40, now)
    archive_old_logs(tmp_path, keep_days=30, now=now)
    _log(tmp_path, "run-a.log", 30, 40, now)
    archive_old_logs(tmp_path, keep_days=30, now=now)
    assert sorted(len(data) for data in _archived(tmp_path).values()) == [10, 30]


def test_log_is_kept_when_the_archive_cannot_be_written(tmp_path: Path) -> None:
    now = time.time()
    old = _log(tmp_path, "run-a.log", 10, 40, now)
    (tmp_path / ARCHIVE_DIR_NAME).write_bytes(b"a file where the folder should be")
    assert archive_old_logs(tmp_path, keep_days=30, now=now) == []
    assert old.exists()


def test_log_is_kept_when_the_archive_entry_does_not_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = time.time()
    old = _log(tmp_path, "run-a.log", 10, 40, now)
    monkeypatch.setattr(logarchive, "_stored_intact", lambda archive, name, crc, size: False)
    assert archive_old_logs(tmp_path, keep_days=30, now=now) == []
    assert old.exists()


def test_archive_old_logs_ignores_other_files(tmp_path: Path) -> None:
    now = time.time()
    other = _log(tmp_path, "stderr-movie.log", 1000, 90, now)
    notes = _log(tmp_path, "run-notes.txt", 1000, 90, now)
    (tmp_path / "run-dir.log").mkdir()
    assert archive_old_logs(tmp_path, keep_days=1, max_bytes=1, now=now) == []
    assert other.exists() and notes.exists()
    assert not (tmp_path / ARCHIVE_DIR_NAME).exists()


def test_archive_old_logs_never_touches_the_excluded_log(tmp_path: Path) -> None:
    now = time.time()
    current = _log(tmp_path, "run-current.log", 100, 40, now)
    old = _log(tmp_path, "run-old.log", 100, 40, now)
    removed = archive_old_logs(tmp_path, keep_days=30, max_bytes=1, now=now, exclude=current)
    assert removed == [old]
    assert current.exists()


def test_archive_old_logs_missing_dir(tmp_path: Path) -> None:
    assert archive_old_logs(tmp_path / "nope") == []


def test_archive_old_logs_skips_undeletable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = time.time()
    locked = _log(tmp_path, "run-locked.log", 10, 40, now)
    original = Path.unlink

    def unlink(self: Path, missing_ok: bool = False) -> None:
        if self == locked:
            raise PermissionError("in use")
        original(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    assert archive_old_logs(tmp_path, now=now) == []
