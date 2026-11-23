"""Finding a movie's subtitle among the user's own local subtitle folders."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from video_cleaner.subtitles import local
from video_cleaner.subtitles.local import (
    AMBIGUITY_MARGIN,
    SIMILARITY_SAME_YEAR,
    SIMILARITY_YEAR_OFF_BY_ONE,
    LocalCandidate,
    LocalSubtitle,
    index_local_folders,
    match_local,
)
from video_cleaner.subtitles.names import TitleYear, title_similarity

MB = 1024 * 1024


def write(path: Path, size: int = 10) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def entry(title: str, year: int | None, name: str | None = None) -> LocalSubtitle:
    file_name = name if name is not None else f"{title}.{year}.srt"
    return LocalSubtitle(Path("subs") / file_name, TitleYear(title, year))


def names_of(index: list[LocalSubtitle]) -> list[str]:
    return [item.path.name for item in index]


# ---------------------------------------------------------------- thresholds


def test_thresholds_are_the_agreed_values() -> None:
    assert SIMILARITY_SAME_YEAR == 0.90
    assert SIMILARITY_YEAR_OFF_BY_ONE == 0.95
    assert AMBIGUITY_MARGIN == 0.02


# ---------------------------------------------------------------- index


def test_index_lists_subtitle_files_with_their_parsed_names(tmp_path: Path) -> None:
    write(tmp_path / "Alien.1979.1080p.BluRay.en.srt")

    index = index_local_folders([tmp_path], max_bytes=MB)

    assert len(index) == 1
    assert index[0].path == tmp_path / "Alien.1979.1080p.BluRay.en.srt"
    assert (index[0].parsed.title, index[0].parsed.year) == ("Alien", 1979)


def test_index_takes_only_text_subtitle_extensions_case_insensitively(tmp_path: Path) -> None:
    for name in ("a.2001.srt", "b.2002.ASS", "c.2003.ssa", "d.2004.vtt", "e.2005.SRT"):
        write(tmp_path / name)
    for name in ("f.2006.sub", "g.2007.idx", "h.2008.txt", "i.2009", "j.2010.mkv"):
        write(tmp_path / name)

    index = index_local_folders([tmp_path], max_bytes=MB)

    assert names_of(index) == ["a.2001.srt", "b.2002.ASS", "c.2003.ssa", "d.2004.vtt", "e.2005.SRT"]


def test_index_skips_a_file_over_the_size_limit_and_keeps_one_at_the_limit(
    tmp_path: Path,
) -> None:
    write(tmp_path / "Big.2000.srt", size=3 * MB)
    write(tmp_path / "Exact.2001.srt", size=MB)

    index = index_local_folders([tmp_path], max_bytes=MB)

    assert names_of(index) == ["Exact.2001.srt"]


def test_index_skips_empty_files(tmp_path: Path) -> None:
    write(tmp_path / "Empty.2000.srt", size=0)

    assert index_local_folders([tmp_path], max_bytes=MB) == []


def test_index_skips_a_directory_with_a_subtitle_name(tmp_path: Path) -> None:
    (tmp_path / "Folder.2000.srt").mkdir()

    assert index_local_folders([tmp_path], max_bytes=MB) == []


def test_index_skips_episodes(tmp_path: Path) -> None:
    write(tmp_path / "Show.S01E08.720p.en.srt")
    write(tmp_path / "Movie.2010.en.srt")

    index = index_local_folders([tmp_path], max_bytes=MB)

    assert names_of(index) == ["Movie.2010.en.srt"]


def test_index_skips_a_file_with_no_readable_title(tmp_path: Path) -> None:
    write(tmp_path / "....srt")
    write(tmp_path / "2010.en.srt")

    assert index_local_folders([tmp_path], max_bytes=MB) == []


def test_index_skips_a_missing_folder_and_still_lists_the_others(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    write(tmp_path / "there" / "Movie.2010.srt")
    caplog.set_level(logging.WARNING, logger="video_cleaner.subtitles.local")

    index = index_local_folders([tmp_path / "missing", tmp_path / "there"], max_bytes=MB)

    assert names_of(index) == ["Movie.2010.srt"]
    assert "missing" in caplog.text


def test_index_given_a_file_instead_of_a_folder_returns_nothing(tmp_path: Path) -> None:
    file = write(tmp_path / "Movie.2010.srt")

    assert index_local_folders([file], max_bytes=MB) == []


def test_index_lists_a_folder_given_twice_once(tmp_path: Path) -> None:
    write(tmp_path / "Movie.2010.srt")

    index = index_local_folders([tmp_path, tmp_path, tmp_path / "."], max_bytes=MB)

    assert len(index) == 1


def test_index_descends_into_subfolders(tmp_path: Path) -> None:
    write(tmp_path / "one" / "two" / "Deep.2010.srt")

    assert names_of(index_local_folders([tmp_path], max_bytes=MB)) == ["Deep.2010.srt"]


def test_index_order_is_deterministic(tmp_path: Path) -> None:
    for name in ("C.2003.srt", "A.2001.srt", "B.2002.srt"):
        write(tmp_path / name)

    assert names_of(index_local_folders([tmp_path], max_bytes=MB)) == [
        "A.2001.srt",
        "B.2002.srt",
        "C.2003.srt",
    ]


def test_index_with_a_zero_size_limit_is_empty(tmp_path: Path) -> None:
    write(tmp_path / "Movie.2010.srt")

    assert index_local_folders([tmp_path], max_bytes=0) == []


def test_index_survives_very_long_and_odd_unicode_names(tmp_path: Path) -> None:
    write(tmp_path / ("A" * 100 + ".2010.srt"))
    write(tmp_path / "\u00dcn\u00efc\u00f6d\u00e9 \u202e \uff2dovie.2011.srt")
    write(tmp_path / "千と千尋の神隠し.2001.srt")

    index = index_local_folders([tmp_path], max_bytes=MB)

    assert len(index) == 3


def test_index_stops_at_the_entry_cap_with_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    for number in range(10):
        write(tmp_path / f"Movie{number}.2010.srt")
    monkeypatch.setattr(local, "MAX_ENTRIES_EXAMINED", 4)
    caplog.set_level(logging.WARNING, logger="video_cleaner.subtitles.local")

    index = index_local_folders([tmp_path], max_bytes=MB)

    assert 0 < len(index) <= 4
    assert "stopping" in caplog.text


def test_index_stops_at_the_depth_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write(tmp_path / "Top.2010.srt")
    write(tmp_path / "a" / "One.2011.srt")
    write(tmp_path / "a" / "b" / "Two.2012.srt")
    monkeypatch.setattr(local, "MAX_DEPTH", 1)

    assert names_of(index_local_folders([tmp_path], max_bytes=MB)) == [
        "Top.2010.srt",
        "One.2011.srt",
    ]


class _FailingStatEntry:
    """A scandir entry whose stat fails, standing in for an unreadable file."""

    def __init__(self, real: os.DirEntry[str]) -> None:
        self._real = real
        self.name = real.name
        self.path = real.path
        self._stats = 0

    def is_symlink(self) -> bool:
        return self._real.is_symlink()

    def is_dir(self, *, follow_symlinks: bool = True) -> bool:
        return self._real.is_dir(follow_symlinks=follow_symlinks)

    def is_file(self, *, follow_symlinks: bool = True) -> bool:
        return self._real.is_file(follow_symlinks=follow_symlinks)

    def stat(self, *, follow_symlinks: bool = True) -> os.stat_result:
        self._stats += 1
        if self.name.startswith("Locked"):
            raise PermissionError("denied")
        if self.name.startswith("Late") and self._stats > 1:
            raise PermissionError("denied after the link check")
        return self._real.stat(follow_symlinks=follow_symlinks)


def test_index_skips_an_entry_it_cannot_stat_without_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write(tmp_path / "Locked.2010.srt")
    write(tmp_path / "Late.2012.srt")
    write(tmp_path / "Fine.2011.srt")
    real_scandir = os.scandir

    class _Wrapped:
        def __init__(self, path: str) -> None:
            self._inner = real_scandir(path)

        def __enter__(self) -> Iterator[Any]:
            return (_FailingStatEntry(item) for item in self._inner.__enter__())

        def __exit__(self, *exc: object) -> None:
            self._inner.__exit__(*exc)

    monkeypatch.setattr(local, "_scandir", _Wrapped)

    assert names_of(index_local_folders([tmp_path], max_bytes=MB)) == ["Fine.2011.srt"]


def test_index_skips_a_folder_that_cannot_be_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def deny(path: str) -> Any:
        raise PermissionError("denied")

    monkeypatch.setattr(local, "_scandir", deny)

    assert index_local_folders([tmp_path], max_bytes=MB) == []


def test_index_skips_a_symlinked_file_and_a_symlinked_folder(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    write(outside / "Secret.2010.srt")
    listed = tmp_path / "listed"
    listed.mkdir()
    write(listed / "Real.2011.srt")
    try:
        os.symlink(outside / "Secret.2010.srt", listed / "Linked.2010.srt")
        os.symlink(outside, listed / "linkdir", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks cannot be created here")

    assert names_of(index_local_folders([listed], max_bytes=MB)) == ["Real.2011.srt"]


@pytest.mark.skipif(sys.platform != "win32", reason="junctions exist on Windows only")
def test_index_skips_a_junction_to_a_folder_outside(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    write(outside / "Secret.2010.srt")
    listed = tmp_path / "listed"
    write(listed / "Real.2011.srt")
    made = subprocess.run(  # noqa: S603 - fixed arguments, paths made by this test
        ["cmd", "/c", "mklink", "/J", str(listed / "junction"), str(outside)],  # noqa: S607
        capture_output=True,
        check=False,
    )
    if made.returncode != 0:
        pytest.skip("junctions cannot be created here")

    assert names_of(index_local_folders([listed], max_bytes=MB)) == ["Real.2011.srt"]
    # Removing the junction itself leaves its target alone.
    os.rmdir(listed / "junction")


def test_index_skips_a_broken_symlink(tmp_path: Path) -> None:
    try:
        os.symlink(tmp_path / "nowhere.srt", tmp_path / "Broken.2010.srt")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks cannot be created here")
    write(tmp_path / "Real.2011.srt")

    assert names_of(index_local_folders([tmp_path], max_bytes=MB)) == ["Real.2011.srt"]


def test_a_windows_reparse_point_counts_as_a_link() -> None:
    reparse = types.SimpleNamespace(
        is_symlink=lambda: False,
        stat=lambda follow_symlinks=False: types.SimpleNamespace(st_file_attributes=0x400),
    )
    plain = types.SimpleNamespace(
        is_symlink=lambda: False,
        stat=lambda follow_symlinks=False: types.SimpleNamespace(st_file_attributes=0x20),
    )
    posix = types.SimpleNamespace(
        is_symlink=lambda: False,
        stat=lambda follow_symlinks=False: types.SimpleNamespace(),
    )

    symlink = types.SimpleNamespace(is_symlink=lambda: True)

    assert local._is_link_like(cast("os.DirEntry[str]", symlink)) is True
    assert local._is_link_like(cast("os.DirEntry[str]", reparse)) is True
    assert local._is_link_like(cast("os.DirEntry[str]", plain)) is False
    assert local._is_link_like(cast("os.DirEntry[str]", posix)) is False


# ---------------------------------------------------------------- matching


def test_exact_title_and_same_year_is_full_confidence() -> None:
    index = [entry("Alien", 1979)]

    result = match_local(index, TitleYear("Alien", 1979))

    assert result == LocalCandidate(index[0].path, 1.0)


def test_year_off_by_one_still_matches_a_near_identical_title() -> None:
    index = [entry("Promising Young Woman", 2020, "Promising.Young.Woman.2020.1080p.en.srt")]

    result = match_local(index, TitleYear("Promising Young Woman", 2021))

    assert result is not None
    assert result.path == index[0].path
    assert 0.9 < result.confidence < 1.0


def test_year_off_by_one_needs_a_higher_similarity() -> None:
    # A title between 0.90 and 0.95 similar is accepted in the same year only.
    close_title = "The Shawshank Redemptio"  # one letter off: >= 0.95
    loose_title = "The Shawshank Redemp"  # 0.9 <= sim < 0.95
    assert 0.90 <= title_similarity(loose_title, "The Shawshank Redemption") < 0.95
    assert title_similarity(close_title, "The Shawshank Redemption") >= 0.95

    target = TitleYear("The Shawshank Redemption", 1994)

    assert match_local([entry(close_title, 1995)], target) is not None
    assert match_local([entry(loose_title, 1995)], target) is None
    assert match_local([entry(loose_title, 1994)], target) is not None


def test_a_year_two_apart_never_matches() -> None:
    assert match_local([entry("Alien", 1981)], TitleYear("Alien", 1979)) is None


def test_alien_never_matches_aliens() -> None:
    target = TitleYear("Alien", 1979)

    assert match_local([entry("Aliens", 1986)], target) is None
    assert match_local([entry("Aliens", None, "Aliens.en.srt")], target) is None


def test_sequels_are_not_the_original() -> None:
    target = TitleYear("Toy Story", 1995)

    assert match_local([entry("Toy Story 2", 1999)], target) is None
    assert match_local([entry("Toy Story 2", None, "Toy.Story.2.srt")], target) is None
    assert match_local([entry("Toy Story 2", 1995)], target) is None


def test_the_sequel_is_picked_over_the_original_for_the_sequel() -> None:
    index = [entry("Toy Story", 1995), entry("Toy Story 2", 1999)]

    result = match_local(index, TitleYear("Toy Story 2", 1999))

    assert result is not None
    assert result.path == index[1].path


def test_accents_and_ampersand_match() -> None:
    assert match_local([entry("Amelie", 2001)], TitleYear("Amélie", 2001)) is not None
    assert match_local([entry("Tom and Jerry", 2021)], TitleYear("Tom & Jerry", 2021)) is not None


def test_two_equally_good_files_are_ambiguous() -> None:
    index = [
        entry("Alien", 1979, "Alien.1979.en.srt"),
        LocalSubtitle(Path("other") / "Alien.1979.en.srt", TitleYear("Alien", 1979)),
    ]

    assert match_local(index, TitleYear("Alien", 1979)) is None


def test_two_files_within_the_margin_are_ambiguous() -> None:
    target = TitleYear("The Lord of the Rings the Fellowship of the Ring", 2001)
    index = [
        entry("he Lord of the Rings the Fellowship of the Ring", 2001, "a.srt"),
        entry("The Lord of the Rings the Fellowship of the Rin", 2001, "b.srt"),
    ]
    scores = sorted(title_similarity(item.parsed.title, target.title) for item in index)
    assert scores[1] - scores[0] <= AMBIGUITY_MARGIN
    assert min(scores) >= SIMILARITY_YEAR_OFF_BY_ONE

    assert match_local(index, target) is None


def test_a_clearly_better_file_wins_over_a_weaker_one() -> None:
    target = TitleYear("The Lord of the Rings the Fellowship of the Ring", 2001)
    index = [
        entry("The Lord of the Rings the Fellowship of the R", 2001, "weak.srt"),
        entry("The Lord of the Rings the Fellowship of the Ring", 2001, "exact.srt"),
    ]
    assert title_similarity(index[0].parsed.title, target.title) >= SIMILARITY_YEAR_OFF_BY_ONE

    result = match_local(index, target)

    assert result == LocalCandidate(index[1].path, 1.0)


def test_a_file_without_a_year_matches_a_unique_title_with_capped_confidence() -> None:
    index = [entry("Alien", None, "Alien.1080p.en.srt")]

    result = match_local(index, TitleYear("Alien", 1979))

    assert result is not None
    assert result.path == index[0].path
    assert 0.0 < result.confidence <= 0.9


def test_a_file_without_a_year_needs_a_near_exact_title() -> None:
    target = TitleYear("The Shawshank Redemption", 1994)

    assert match_local([entry("The Shawshank Redem", None, "x.srt")], target) is None


def test_a_dated_file_beats_a_yearless_one_when_clearly_better() -> None:
    index = [entry("Alien", None, "Alien.en.srt"), entry("Alien", 1979, "Alien.1979.en.srt")]

    # 1.0 for the dated file against at most 0.9 for the undated one: the margin is cleared.
    result = match_local(index, TitleYear("Alien", 1979))

    assert result == LocalCandidate(index[1].path, 1.0)


def test_a_target_without_a_year_cannot_be_verified_and_is_capped() -> None:
    index = [entry("Alien", 1979)]

    result = match_local(index, TitleYear("Alien", None))

    assert result is not None
    assert result.confidence <= 0.9


def test_an_episode_target_never_matches() -> None:
    index = [entry("Show", None, "Show.srt")]

    assert match_local(index, TitleYear("Show", None, is_episode=True)) is None


def test_episode_entries_in_the_index_are_ignored() -> None:
    episode = LocalSubtitle(Path("Alien.srt"), TitleYear("Alien", 1979, is_episode=True))

    assert match_local([episode], TitleYear("Alien", 1979)) is None


def test_an_empty_index_or_an_empty_title_never_matches() -> None:
    assert match_local([], TitleYear("Alien", 1979)) is None
    assert match_local([entry("Alien", 1979)], TitleYear("", 1979)) is None
    assert match_local([entry("", 1979, "x.srt")], TitleYear("", 1979)) is None
    assert match_local([entry("!!!", 1979, "x.srt")], TitleYear("???", 1979)) is None


def test_the_same_file_listed_twice_is_not_ambiguous() -> None:
    item = entry("Alien", 1979)

    result = match_local([item, item], TitleYear("Alien", 1979))

    assert result == LocalCandidate(item.path, 1.0)


def test_matching_does_not_depend_on_the_order_of_the_index() -> None:
    target = TitleYear("The Lord of the Rings the Fellowship of the Ring", 2001)
    a = entry("The Lord of the Rings the Fellowship of the Ring", 2001, "a.srt")
    b = entry("The Lord of the Rings the Fellowship of the Rin", 2001, "b.srt")

    assert match_local([a, b], target) == match_local([b, a], target)


def test_a_hostile_long_title_is_matched_quickly() -> None:
    hostile = entry("ab" * 500_000, 2000, "hostile.srt")

    started = time.monotonic()
    result = match_local([hostile] * 50, TitleYear("cd" * 500_000, 2000))

    assert time.monotonic() - started < 5.0
    assert result is None
