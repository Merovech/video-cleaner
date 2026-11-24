"""Atomic file writes shared by the plan, report, survey CSV and manifest."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import pytest

from video_cleaner.atomicio import (
    csv_cell,
    csv_cell_value,
    read_bytes_capped,
    write_csv_atomic,
    write_text_atomic,
)


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


@pytest.mark.parametrize(
    "cell",
    ['=HYPERLINK("http://x")', "+1+cmd", "-2+3", "@SUM(A1)", "\tTab", "\rCR", "'quoted"],
)
def test_csv_cell_neutralises_formula_triggers(cell: str) -> None:
    assert csv_cell(cell) == "'" + cell
    assert csv_cell_value(str(csv_cell(cell))) == cell


@pytest.mark.parametrize(
    "cell", ["plain", "", "C:\\lib\\a=b.mkv", "-1.5", "-3", "1e-7", 7, -2, -1.25, None]
)
def test_csv_cell_leaves_safe_values_alone(cell: object) -> None:
    assert csv_cell(cell) == cell
    if isinstance(cell, str):
        assert csv_cell_value(cell) == cell


def test_write_csv_atomic_neutralises_formulas(tmp_path: Path) -> None:
    target = tmp_path / "out.csv"
    write_csv_atomic(target, ("name", "n"), [["=1+1", -4], ["@cmd", "-0.5"]])
    with target.open(encoding="utf-8", newline="") as fh:
        assert list(csv.reader(fh)) == [["name", "n"], ["'=1+1", "-4"], ["'@cmd", "-0.5"]]


# --- read_bytes_capped ---------------------------------------------------------------------


def test_read_bytes_capped_returns_a_file_of_exactly_the_cap(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"abcde")

    assert read_bytes_capped(path, 5) == b"abcde"


def test_read_bytes_capped_returns_none_for_one_byte_over(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"abcdef")

    assert read_bytes_capped(path, 5) is None


def test_read_bytes_capped_returns_empty_bytes_for_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"")

    assert read_bytes_capped(path, 5) == b""


@pytest.mark.parametrize("max_bytes", [0, -1])
def test_read_bytes_capped_refuses_a_cap_below_one(tmp_path: Path, max_bytes: int) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")

    with pytest.raises(ValueError, match="max_bytes"):
        read_bytes_capped(path, max_bytes)


def test_read_bytes_capped_propagates_os_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_bytes_capped(tmp_path / "missing.bin", 5)


def test_read_bytes_capped_never_asks_for_more_than_one_byte_past_the_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "big.bin"
    path.write_bytes(b"x" * 100_000)  # the file grew after the caller looked at its size
    requested: list[int] = []
    real_open = Path.open

    class Spy:
        def __init__(self, handle: Any) -> None:
            self._handle = handle

        def __enter__(self) -> Spy:
            return self

        def __exit__(self, *exc: object) -> None:
            self._handle.close()

        def read(self, size: int = -1) -> bytes:
            requested.append(size)
            data: bytes = self._handle.read(size)
            return data

    def spy_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        handle = real_open(self, *args, **kwargs)
        return Spy(handle) if self == path else handle

    monkeypatch.setattr(Path, "open", spy_open)

    assert read_bytes_capped(path, 10) is None
    assert requested == [11]
