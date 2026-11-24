"""Shared path helpers: one definition of "same file" and "inside a folder"."""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from video_cleaner import paths
from video_cleaner.config import Settings
from video_cleaner.paths import (
    OutputInLibrary,
    is_under,
    library_overlaps,
    name_key,
    outputs_outside_library,
    path_key,
)

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows path semantics")


def test_is_under_plain_and_equal(tmp_path: Path) -> None:
    root = tmp_path / "library"
    assert is_under(root / "a" / "b.mkv", root)
    assert is_under(root, root)
    assert not is_under(tmp_path / "scratch", root)


def test_sibling_prefix_is_not_under(tmp_path: Path) -> None:
    assert not is_under(tmp_path / "library2" / "x.mkv", tmp_path / "library")
    assert not is_under(tmp_path / "library.old", tmp_path / "library")


def test_dot_dot_and_relative_spellings_are_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "library"
    root.mkdir()
    assert is_under(tmp_path / "other" / ".." / "library" / "x", root)
    assert not is_under(root / ".." / "scratch", root)
    monkeypatch.chdir(root)
    assert is_under(Path("sub") / "x.mkv", root)
    assert path_key(Path("x.mkv")) == path_key(root / "x.mkv")


@windows_only
def test_case_and_separator_differences(tmp_path: Path) -> None:
    root = tmp_path / "Library"
    assert is_under(Path(str(root).upper() + "/Films/A.mkv"), root)
    assert path_key(root / "A.MKV") == path_key(Path(str(root).lower()) / "a.mkv")


@windows_only
def test_unc_paths_compare_case_insensitively(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "_resolve", lambda p: os.path.abspath(p))  # no network lookups
    assert is_under(Path(r"\\NAS\Media\Films\x.mkv"), Path(r"\\nas\media"))
    assert not is_under(Path(r"\\nas\media2\x.mkv"), Path(r"\\nas\media"))


@windows_only
def test_mapped_drive_alias_of_the_share_is_under(monkeypatch: pytest.MonkeyPatch) -> None:
    # What realpath does for a mapped network drive: W:\ resolves to the share it maps.
    def resolve(p: str | os.PathLike[str]) -> str:
        text = os.path.abspath(p)
        if text.upper().startswith("W:\\"):
            return r"\\nas\media" + text[2:]
        return text

    monkeypatch.setattr(paths, "_resolve", resolve)
    assert is_under(Path(r"W:\scratch"), Path(r"\\nas\media"))
    assert is_under(Path(r"\\nas\media\scratch"), Path("W:/"))
    assert not is_under(Path(r"D:\scratch"), Path(r"\\nas\media"))


def test_unresolvable_path_falls_back_to_absolute(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(p: str | os.PathLike[str]) -> str:
        raise OSError("unreachable")

    monkeypatch.setattr(paths, "_resolve", broken)
    assert path_key(Path("a")) == os.path.normcase(os.path.abspath("a"))


# --- the one library-overlap guard ---------------------------------------------------------


def _guard_settings(tmp_path: Path, **changes: Any) -> Settings:
    base = Settings(
        library_root=tmp_path / "library",
        scratch_dir=tmp_path / "scratch",
        completed_dir=tmp_path / "completed",
        state_db=tmp_path / "appdata" / "state.db",
        log_dir=tmp_path / "appdata" / "logs",
        subtitle_cache_dir=tmp_path / "appdata" / "subtitles",
    )
    return dataclasses.replace(base, **changes)


def test_outputs_outside_library_accepts_separate_folders(tmp_path: Path) -> None:
    settings = _guard_settings(tmp_path)
    assert library_overlaps(settings) == []
    outputs_outside_library(settings, extra=(("report", tmp_path / "r.csv"),))


@pytest.mark.parametrize(
    ("field", "label"),
    [
        ("scratch_dir", "scratch folder"),
        ("completed_dir", "completed folder"),
        ("state_db", "state database"),
        ("log_dir", "log folder"),
        ("subtitle_cache_dir", "subtitle cache"),
    ],
)
def test_every_output_under_the_library_is_refused(tmp_path: Path, field: str, label: str) -> None:
    settings = _guard_settings(tmp_path, **{field: tmp_path / "library" / "x"})
    with pytest.raises(OutputInLibrary, match=label):
        outputs_outside_library(settings)
    assert [lab for lab, _ in library_overlaps(settings)] == [label]


@pytest.mark.parametrize("field", ["scratch_dir", "completed_dir", "log_dir", "subtitle_cache_dir"])
def test_library_inside_an_output_folder_is_refused(tmp_path: Path, field: str) -> None:
    settings = _guard_settings(
        tmp_path, library_root=getattr(_guard_settings(tmp_path), field) / "library"
    )
    with pytest.raises(OutputInLibrary, match="is inside the"):
        outputs_outside_library(settings)


def test_extra_outputs_and_extra_roots_are_checked(tmp_path: Path) -> None:
    settings = _guard_settings(tmp_path)
    inside = tmp_path / "library" / "report.csv"
    with pytest.raises(OutputInLibrary, match="report"):
        outputs_outside_library(settings, extra=(("report", inside),))
    other_root = tmp_path / "elsewhere"
    with pytest.raises(OutputInLibrary, match="log folder"):
        outputs_outside_library(
            _guard_settings(tmp_path, log_dir=other_root / "logs"), roots=(other_root,)
        )


def test_name_key_is_string_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def boom(p: object) -> str:
        raise AssertionError("name_key must not resolve")

    monkeypatch.setattr(paths, "_resolve", boom)
    assert name_key(tmp_path / "a" / ".." / "b.mkv") == name_key(tmp_path / "b.mkv")
    assert name_key(str(tmp_path / "b.mkv")) == name_key(tmp_path / "b.mkv")


@windows_only
def test_name_key_folds_case_on_windows(tmp_path: Path) -> None:
    assert name_key(tmp_path / "A.MKV") == name_key(Path(str(tmp_path).lower()) / "a.mkv")


def test_file_key_is_the_name_key_with_size_and_mtime(tmp_path: Path) -> None:
    path = tmp_path / "lib" / ".." / "a.mkv"
    assert paths.file_key(path, 10, 2.5) == (name_key(tmp_path / "a.mkv"), 10, 2.5)


def test_record_key_serves_file_and_subtitle_records(tmp_path: Path) -> None:
    from video_cleaner.state import FileRecord, Status, SubtitleRecord, SubtitleStatus

    path = tmp_path / "a.mkv"
    rec = FileRecord(path, 7, 1.5, Status.PENDING, "encode", (), 0, (), "")
    sub = SubtitleRecord(path, 7, 1.5, SubtitleStatus.WANTED)
    assert paths.record_key(rec) == paths.record_key(sub) == paths.file_key(path, 7, 1.5)
    other = dataclasses.replace(sub, mtime=2.0)
    assert paths.record_key(other) != paths.record_key(sub)
