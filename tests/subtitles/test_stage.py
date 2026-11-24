"""Checking a FOUND subtitle's cache file and staging a private copy of it for mkvmerge."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from video_cleaner.atomicio import read_bytes_capped
from video_cleaner.config import Settings
from video_cleaner.mux import SubtitleInput
from video_cleaner.state import SubtitleRecord, SubtitleStatus
from video_cleaner.subtitles import stage as stage_module
from video_cleaner.subtitles.stage import (
    STAGED_NAME,
    SubtitleCacheStale,
    check_cached_subtitle,
    stage_subtitle,
)

DATA = b"1\n00:00:01,000 --> 00:00:02,000\nHello there.\n"
TITLE = "English (OpenSubtitles)"


def _settings(tmp_path: Path, **changes: object) -> Settings:
    values: dict[str, object] = {
        "library_root": tmp_path / "library",
        "scratch_dir": tmp_path / "scratch",
        "completed_dir": tmp_path / "completed",
        "state_db": tmp_path / "appdata" / "state.db",
        "log_dir": tmp_path / "appdata" / "logs",
        "subtitle_cache_dir": tmp_path / "cache",
    }
    values.update(changes)
    return Settings(**values)  # type: ignore[arg-type]


def _record(cache: Path | None, data: bytes = DATA, **changes: object) -> SubtitleRecord:
    values: dict[str, object] = {
        "path": Path("D:/library/a.mkv"),
        "size": 10,
        "mtime": 1.0,
        "status": SubtitleStatus.FOUND,
        "provider": "opensubtitles",
        "cache_path": cache,
        "sha256": hashlib.sha256(data).hexdigest(),
        "provenance": TITLE,
    }
    values.update(changes)
    return SubtitleRecord(**values)  # type: ignore[arg-type]


@pytest.fixture
def cached(tmp_path: Path) -> tuple[Settings, Path]:
    settings = _settings(tmp_path)
    settings.subtitle_cache_dir.mkdir(parents=True)
    path = settings.subtitle_cache_dir / "0123456789abcdef-opensubtitles-1.srt"
    path.write_bytes(DATA)
    return settings, path


def test_a_valid_cache_file_passes(cached: tuple[Settings, Path]) -> None:
    settings, path = cached
    check_cached_subtitle(_record(path), settings)


def test_stage_copies_the_bytes_under_the_work_folder(
    cached: tuple[Settings, Path], tmp_path: Path
) -> None:
    settings, path = cached
    work = tmp_path / "work"
    work.mkdir()
    staged = stage_subtitle(_record(path), settings, work)
    assert staged == SubtitleInput(path=work / STAGED_NAME, title=TITLE)
    assert staged.path.read_bytes() == DATA
    assert path.read_bytes() == DATA  # the cache file itself is untouched


def test_stale_reason_names_the_problem_not_the_content(cached: tuple[Settings, Path]) -> None:
    settings, path = cached
    path.unlink()
    with pytest.raises(SubtitleCacheStale, match="missing"):
        check_cached_subtitle(_record(path), settings)


@pytest.mark.parametrize("hole", ["no_path", "no_hash"])
def test_a_record_without_a_cache_path_or_hash_is_stale(
    cached: tuple[Settings, Path], hole: str
) -> None:
    settings, path = cached
    record = _record(None) if hole == "no_path" else _record(path, sha256="")
    with pytest.raises(SubtitleCacheStale):
        check_cached_subtitle(record, settings)


def test_changed_bytes_are_stale(cached: tuple[Settings, Path]) -> None:
    settings, path = cached
    path.write_bytes(DATA + b"extra")
    with pytest.raises(SubtitleCacheStale, match="changed"):
        check_cached_subtitle(_record(path), settings)


def test_a_relative_cache_path_is_stale(cached: tuple[Settings, Path]) -> None:
    settings, _ = cached
    with pytest.raises(SubtitleCacheStale, match="absolute"):
        check_cached_subtitle(_record(Path("cache") / "x.srt"), settings)


def test_a_path_outside_the_cache_folder_is_stale(
    cached: tuple[Settings, Path], tmp_path: Path
) -> None:
    settings, _ = cached
    outside = tmp_path / "elsewhere.srt"
    outside.write_bytes(DATA)
    with pytest.raises(SubtitleCacheStale, match="outside"):
        check_cached_subtitle(_record(outside), settings)


def test_a_dotdot_path_that_leaves_the_cache_folder_is_stale(
    cached: tuple[Settings, Path], tmp_path: Path
) -> None:
    settings, _ = cached
    (tmp_path / "elsewhere.srt").write_bytes(DATA)
    sneaky = settings.subtitle_cache_dir / ".." / "elsewhere.srt"
    with pytest.raises(SubtitleCacheStale, match="outside"):
        check_cached_subtitle(_record(sneaky), settings)


def test_a_sibling_folder_with_the_same_prefix_is_outside(
    cached: tuple[Settings, Path], tmp_path: Path
) -> None:
    settings, _ = cached
    sibling = tmp_path / "cache2"
    sibling.mkdir()
    (sibling / "x.srt").write_bytes(DATA)
    with pytest.raises(SubtitleCacheStale, match="outside"):
        check_cached_subtitle(_record(sibling / "x.srt"), settings)


def test_a_folder_in_place_of_the_file_is_stale(cached: tuple[Settings, Path]) -> None:
    settings, path = cached
    path.unlink()
    path.mkdir()
    with pytest.raises(SubtitleCacheStale, match="regular file"):
        check_cached_subtitle(_record(path), settings)


def test_an_oversize_file_is_stale_without_being_read(tmp_path: Path) -> None:
    settings = _settings(tmp_path, subtitle_max_bytes=10)
    settings.subtitle_cache_dir.mkdir()
    path = settings.subtitle_cache_dir / "big.srt"
    path.write_bytes(DATA)  # longer than 10 bytes
    with pytest.raises(SubtitleCacheStale, match="larger than"):
        check_cached_subtitle(_record(path), settings)


def test_a_symlinked_cache_file_is_stale(cached: tuple[Settings, Path]) -> None:
    settings, path = cached
    link = settings.subtitle_cache_dir / "link.srt"
    try:
        os.symlink(path, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks cannot be created here")
    with pytest.raises(SubtitleCacheStale, match="link"):
        check_cached_subtitle(_record(link), settings)


@pytest.mark.skipif(sys.platform != "win32", reason="junctions exist on Windows only")
def test_a_cache_file_reached_through_a_junction_is_stale(
    cached: tuple[Settings, Path], tmp_path: Path
) -> None:
    settings, _ = cached
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.srt").write_bytes(DATA)
    junction = settings.subtitle_cache_dir / "junction"
    made = subprocess.run(  # noqa: S603 - fixed arguments, paths made by this test
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],  # noqa: S607
        capture_output=True,
        check=False,
    )
    if made.returncode != 0:
        pytest.skip("junctions cannot be created here")
    try:
        with pytest.raises(SubtitleCacheStale):
            check_cached_subtitle(_record(junction / "x.srt"), settings)
    finally:
        os.rmdir(junction)


def test_stage_rejects_a_file_changed_after_the_check(
    cached: tuple[Settings, Path], tmp_path: Path
) -> None:
    settings, path = cached
    record = _record(path)
    check_cached_subtitle(record, settings)
    path.write_bytes(b"something else entirely")  # the check passed; now it changes
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(SubtitleCacheStale, match="changed"):
        stage_subtitle(record, settings, work)
    assert not (work / STAGED_NAME).exists()  # no half-trusted copy is left behind


def test_stage_removes_its_copy_when_the_copy_does_not_read_back_the_same(
    cached: tuple[Settings, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, path = cached
    work = tmp_path / "work"
    work.mkdir()
    real = read_bytes_capped

    def corrupting(target: Path, max_bytes: int) -> bytes | None:
        data = real(target, max_bytes)
        return data if target == path else b"corrupted" + (data or b"")

    monkeypatch.setattr(stage_module, "read_bytes_capped", corrupting, raising=True)
    with pytest.raises(SubtitleCacheStale, match="staged copy"):
        stage_subtitle(_record(path), settings, work)
    assert not (work / STAGED_NAME).exists()


def test_stage_never_overwrites_an_existing_staged_file(
    cached: tuple[Settings, Path], tmp_path: Path
) -> None:
    settings, path = cached
    work = tmp_path / "work"
    work.mkdir()
    (work / STAGED_NAME).write_bytes(b"left over")
    with pytest.raises(FileExistsError):
        stage_subtitle(_record(path), settings, work)
    assert (work / STAGED_NAME).read_bytes() == b"left over"


def test_stage_refuses_a_hostile_title(cached: tuple[Settings, Path], tmp_path: Path) -> None:
    settings, path = cached
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(ValueError, match="title"):
        stage_subtitle(_record(path, provenance="bad\ntitle"), settings, work)
    assert not (work / STAGED_NAME).exists()
