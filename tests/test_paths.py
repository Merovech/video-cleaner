"""Shared path helpers: one definition of "same file" and "inside a folder"."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from video_cleaner import paths
from video_cleaner.paths import is_under, path_key

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
