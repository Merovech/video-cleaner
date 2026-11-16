"""Atomic file writes shared by the plan, report, survey CSV and manifest."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import pytest

from video_cleaner.atomicio import write_csv_atomic, write_text_atomic


def _leftovers(folder: Path, keep: str) -> list[str]:
    return sorted(p.name for p in folder.iterdir() if p.name != keep)


def test_writes_utf8_without_bom_and_leaves_no_temp(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "out.txt"  # missing folder is created
    write_text_atomic(target, "Amélie ✓ [1080p]\na\r\nb")
    assert target.read_bytes() == "Amélie ✓ [1080p]\na\r\nb".encode()  # no newline translation
    assert _leftovers(target.parent, "out.txt") == []


def test_replaces_an_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    target.write_text("old", encoding="utf-8")
    write_text_atomic(target, "new")
    assert target.read_text(encoding="utf-8") == "new"
    assert _leftovers(tmp_path, "out.txt") == []


def test_failed_replace_keeps_the_old_file_and_no_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "out.txt"
    target.write_text("old", encoding="utf-8")

    def locked(src: Any, dst: Any) -> None:
        raise PermissionError(13, "in use", str(dst))

    monkeypatch.setattr(os, "replace", locked)
    with pytest.raises(PermissionError):
        write_text_atomic(target, "new")
    assert target.read_text(encoding="utf-8") == "old"
    assert _leftovers(tmp_path, "out.txt") == []


def test_encoding_error_leaves_no_temp(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    target.write_text("old", encoding="utf-8")
    with pytest.raises(UnicodeEncodeError):
        write_text_atomic(target, "bad \udc80 name")  # strict by default
    assert target.read_text(encoding="utf-8") == "old"
    assert _leftovers(tmp_path, "out.txt") == []


def test_errors_backslashreplace_writes_lone_surrogates(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    write_text_atomic(target, "bad \udc80 name", errors="backslashreplace")
    assert target.read_text(encoding="utf-8") == "bad \\udc80 name"


@pytest.mark.parametrize("flag", [True, False])
def test_fsync_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: bool) -> None:
    synced: list[int] = []
    real = os.fsync

    def spy(fd: int) -> None:
        synced.append(fd)
        real(fd)

    monkeypatch.setattr(os, "fsync", spy)
    write_text_atomic(tmp_path / "out.txt", "x", fsync=flag)
    assert bool(synced) is flag
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "x"


def test_write_csv_atomic(tmp_path: Path) -> None:
    target = tmp_path / "out.csv"
    write_csv_atomic(target, ("a", "b"), [["x, y", 1], ['q"uote', ""]])
    with target.open(encoding="utf-8", newline="") as fh:
        assert list(csv.reader(fh)) == [["a", "b"], ["x, y", "1"], ['q"uote', ""]]
    assert _leftovers(tmp_path, "out.csv") == []
