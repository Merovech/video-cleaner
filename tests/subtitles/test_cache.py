"""The cache folder helpers that list a video's own candidate files."""

from __future__ import annotations

import os
from pathlib import Path

from video_cleaner.subtitles.cache import prune_kept_downloads, reusable_ids

KEY = "0123456789abcdef"
PROVIDER = "opensubtitles"


def _touch(folder: Path, name: str, age_s: int = 0) -> Path:
    path = folder / name
    path.write_text("x", encoding="utf-8")
    stamp = 1_900_000_000 - age_s
    os.utime(path, (stamp, stamp))
    return path


def test_reusable_ids_are_the_finished_and_kept_files_of_this_video_only(tmp_path: Path) -> None:
    _touch(tmp_path, f"{KEY}-{PROVIDER}-7.srt")
    _touch(tmp_path, f"{KEY}-{PROVIDER}-8.dl.srt")
    _touch(tmp_path, f"{KEY}-{PROVIDER}-9.aligned.srt")  # an aligner output is not reusable
    _touch(tmp_path, f"{KEY}-{PROVIDER}-x.srt")  # not one of our numbers
    _touch(tmp_path, f"{KEY}-sidecar-5.srt")  # another provider
    _touch(tmp_path, f"ffffffffffffffff-{PROVIDER}-6.srt")  # another video
    assert reusable_ids(tmp_path, KEY, PROVIDER) == {7, 8}


def test_reusable_ids_of_a_missing_folder_is_empty(tmp_path: Path) -> None:
    assert reusable_ids(tmp_path / "nowhere", KEY, PROVIDER) == set()


def test_prune_removes_the_oldest_kept_downloads_beyond_the_limit(tmp_path: Path) -> None:
    oldest = _touch(tmp_path, f"{KEY}-{PROVIDER}-1.dl.srt", age_s=300)
    middle = _touch(tmp_path, f"{KEY}-{PROVIDER}-2.dl.srt", age_s=200)
    newest = _touch(tmp_path, f"{KEY}-{PROVIDER}-3.dl.srt", age_s=100)
    final = _touch(tmp_path, f"{KEY}-{PROVIDER}-4.srt", age_s=900)
    other = _touch(tmp_path, f"ffffffffffffffff-{PROVIDER}-1.dl.srt", age_s=900)
    assert prune_kept_downloads(tmp_path, KEY, PROVIDER, keep=2) == 1
    assert not oldest.exists()
    assert middle.exists() and newest.exists() and final.exists() and other.exists()


def test_prune_never_removes_the_protected_download(tmp_path: Path) -> None:
    protected = _touch(tmp_path, f"{KEY}-{PROVIDER}-1.dl.srt", age_s=300)
    _touch(tmp_path, f"{KEY}-{PROVIDER}-2.dl.srt", age_s=200)
    _touch(tmp_path, f"{KEY}-{PROVIDER}-3.dl.srt", age_s=100)
    assert prune_kept_downloads(tmp_path, KEY, PROVIDER, keep=1, protect=protected) == 2
    assert protected.exists()
    assert [p.name for p in tmp_path.iterdir()] == [protected.name]
