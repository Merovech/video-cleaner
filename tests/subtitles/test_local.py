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
    title_year_confidence,
)
from video_cleaner.subtitles.names import (
    TitleYear,
    parse_library_name,
    parse_release_name,
    title_similarity,
)

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


SRT = b"1\r\n00:00:01,000 --> 00:00:02,500\r\nHello there.\r\n"
ASS = (
    b"[Script Info]\n[V4+ Styles]\n[Events]\n"
    b"Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    b"Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,Hello there.\n"
)
VTT = b"WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHello there.\n"


def put(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_index_takes_text_subtitle_extensions_case_insensitively_without_reading_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("a.2001.srt", "b.2002.ASS", "c.2003.ssa", "d.2004.vtt", "e.2005.SRT"):
        write(tmp_path / name)  # the content is judged later, by the resolver

    def refuse(*args: object, **kwargs: object) -> bytes:
        raise AssertionError("a file with a subtitle extension was read")

    monkeypatch.setattr(local, "read_bytes_capped", refuse)

    index = index_local_folders([tmp_path], max_bytes=MB)

    assert names_of(index) == ["a.2001.srt", "b.2002.ASS", "c.2003.ssa", "d.2004.vtt", "e.2005.SRT"]


@pytest.mark.parametrize(
    "name",
    [
        "Movie.2010.en.264-EVO",
        "Movie (2010).en.AC3-EVO",
        "Movie (2010).en.MX]",
        "Movie (2010).en.The Movie - 4K Disc Eng",
        "Movie (2010)",
        "Movie.2010.txt",
    ],
)
def test_index_takes_a_file_with_any_other_name_when_it_holds_a_subtitle(
    tmp_path: Path, name: str
) -> None:
    put(tmp_path / name, SRT)

    index = index_local_folders([tmp_path], max_bytes=MB)

    assert match_local(index, TitleYear("Movie", 2010)) == LocalCandidate(tmp_path / name, 1.0)


@pytest.mark.parametrize("content", [ASS, VTT, b"\xef\xbb\xbf" + SRT])
def test_index_recognises_each_format_by_content(tmp_path: Path, content: bytes) -> None:
    put(tmp_path / "Movie (2010).en.DTS-GRP", content)

    assert names_of(index_local_folders([tmp_path], max_bytes=MB)) == ["Movie (2010).en.DTS-GRP"]


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("Movie.2010.sub", b"{1}{25}Hello there."),
        ("Movie.2010.nfo", b"Movie (2010), a film. Enjoy."),
        ("Movie.2010.jpg", bytes(range(256)) * 4),
        ("Movie.2010.html", b"<html><body>00:00:01,000 --> 00:00:02,000</body></html>"),
        ("Movie.2010", b"x" * 10),
    ],
)
def test_index_skips_a_file_with_another_name_that_holds_no_subtitle(
    tmp_path: Path, name: str, content: bytes
) -> None:
    put(tmp_path / name, content)

    assert index_local_folders([tmp_path], max_bytes=MB) == []


def test_other_files_beside_a_subtitle_do_not_make_it_ambiguous(tmp_path: Path) -> None:
    put(tmp_path / "Movie (2010).nfo", b"Movie (2010), a film. Enjoy.")
    put(tmp_path / "Movie (2010).en.264-EVO", SRT)

    result = match_local(index_local_folders([tmp_path], max_bytes=MB), TitleYear("Movie", 2010))

    assert result == LocalCandidate(tmp_path / "Movie (2010).en.264-EVO", 1.0)


@pytest.mark.parametrize("name", ["Show.S01E08.en.264-EVO", "Movie.2010.fr.264-EVO"])
def test_index_filters_a_file_with_another_name_by_its_name_too(tmp_path: Path, name: str) -> None:
    put(tmp_path / name, SRT)

    assert index_local_folders([tmp_path], max_bytes=MB) == []


def test_index_never_reads_a_file_over_the_size_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write(tmp_path / "Movie (2010).mkv", size=2 * MB)

    def refuse(*args: object, **kwargs: object) -> bytes:
        raise AssertionError("an oversized file was read")

    monkeypatch.setattr(local, "read_bytes_capped", refuse)

    assert index_local_folders([tmp_path], max_bytes=MB) == []


def test_index_reads_another_name_only_as_the_regular_file_it_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    put(tmp_path / "Movie (2010).en.264-EVO", SRT)
    calls: list[tuple[int, bool]] = []

    def swapped(path: Path, max_bytes: int, *, regular_only: bool = False) -> bytes:
        calls.append((max_bytes, regular_only))
        raise OSError("not the regular file the path names")

    monkeypatch.setattr(local, "read_bytes_capped", swapped)
    write(tmp_path / "Other (2011).srt")

    assert names_of(index_local_folders([tmp_path], max_bytes=MB)) == ["Other (2011).srt"]
    assert calls == [(MB, True)]


def test_index_uses_no_other_names_when_there_are_too_many_to_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # With part of the list missing, the film of one year could take the subtitle of the next.
    put(tmp_path / "Harbour (2010).en.x264-GRP", SRT)
    put(tmp_path / "Harbour (2011).en.x264-GRP", SRT)
    put(tmp_path / "Plain (2012).srt", SRT)
    monkeypatch.setattr(local, "MAX_FILES_READ", 1)
    caplog.set_level(logging.WARNING, logger="video_cleaner.subtitles.local")

    index = index_local_folders([tmp_path], max_bytes=MB)

    assert names_of(index) == ["Plain (2012).srt"]
    assert match_local(index, TitleYear("Harbour", 2011)) is None
    assert sum("more than 1 files" in record.getMessage() for record in caplog.records) == 1


def test_index_uses_every_other_name_when_they_just_fit_under_the_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    put(tmp_path / "Harbour (2010).en.x264-GRP", SRT)
    put(tmp_path / "Harbour (2011).en.x264-GRP", SRT)
    monkeypatch.setattr(local, "MAX_FILES_READ", 2)
    caplog.set_level(logging.WARNING, logger="video_cleaner.subtitles.local")

    assert len(index_local_folders([tmp_path], max_bytes=MB)) == 2
    assert caplog.records == []


def test_files_that_cannot_be_used_do_not_count_toward_the_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "Show.S01E08.en.264-GRP",
        "Other.2010.fr.264-GRP",
        "Movie (2010).nfo",
        "Movie (2010).jpg",
        "Movie (2010).en.srt.part",
    ):
        put(tmp_path / name, SRT)
    put(tmp_path / "Movie (2010).en.264-GRP", SRT)
    monkeypatch.setattr(local, "MAX_FILES_READ", 1)

    assert names_of(index_local_folders([tmp_path], max_bytes=MB)) == ["Movie (2010).en.264-GRP"]


@pytest.mark.parametrize(
    "name",
    [
        "Movie (2010).en.srt.part",
        "Movie (2010).en.srt.crdownload",
        "Movie (2010).en.srt.bak",
        "Movie (2010).en.srt.orig",
        "Movie (2010).en.ASS.old",
        "Movie (2010).en.srt.disabled",
        "Movie (2010).en.srt.1",
        "Movie (2010).en.srt.txt",
        "Movie (2010).en.srt (1)",
        "Movie (2010).en.srt - Copy",
        "Movie (2010).en.srt~",
        "Movie (2010).en.264-GRP.part",
        "Movie (2010).en.264-GRP.!ut",
        "Movie (2010).partial",
        "Movie (2010).tmp",
        "Movie (2010).download",
    ],
)
def test_index_skips_unfinished_downloads_and_backup_copies(tmp_path: Path, name: str) -> None:
    put(tmp_path / name, SRT)

    assert index_local_folders([tmp_path], max_bytes=MB) == []


@pytest.mark.parametrize("name", ["Kick.Ass.2010", "The.Smart.Ass.2010.en.x264-GRP"])
def test_a_title_that_spells_an_extension_is_not_taken_for_a_copy(
    tmp_path: Path, name: str
) -> None:
    put(tmp_path / name, SRT)

    assert names_of(index_local_folders([tmp_path], max_bytes=MB)) == [name]


def test_a_leftover_copy_does_not_make_the_real_subtitle_ambiguous(tmp_path: Path) -> None:
    put(tmp_path / "Movie (2010).en.srt", SRT)
    put(tmp_path / "Movie (2010).en.srt.bak", SRT)

    result = match_local(index_local_folders([tmp_path], max_bytes=MB), TitleYear("Movie", 2010))

    assert result == LocalCandidate(tmp_path / "Movie (2010).en.srt", 1.0)


@pytest.mark.parametrize("suffix", [".nfo", ".jpg", ".png", ".idx", ".zip"])
def test_index_never_opens_a_file_that_cannot_be_a_subtitle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    write(tmp_path / f"Movie (2010){suffix}")

    def refuse(*args: object, **kwargs: object) -> bytes:
        raise AssertionError("a file that cannot be a subtitle was read")

    monkeypatch.setattr(local, "read_bytes_capped", refuse)

    assert index_local_folders([tmp_path], max_bytes=MB) == []


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


@pytest.mark.parametrize(
    "name",
    [
        "Alien.1979.fr.srt",
        "Alien.1979.spa.srt",
        "Alien.1979.pt-BR.srt",
        "Alien.1979.fre.srt",
        "Alien.1979.French.srt",
        "Alien.1979.en.fr.srt",
        "Alien.1979.VOSTFR.srt",
    ],
)
def test_index_skips_a_file_named_for_a_language_other_than_english(
    tmp_path: Path, name: str
) -> None:
    write(tmp_path / name)

    assert index_local_folders([tmp_path], max_bytes=MB) == []


@pytest.mark.parametrize(
    "name",
    ["Alien.1979.en.srt", "Alien.1979.eng.srt", "Alien.1979.English.srt", "Alien.1979.srt"],
)
def test_index_keeps_english_and_untagged_files(tmp_path: Path, name: str) -> None:
    # An untagged name says nothing about the language: the resolver checks the content.
    write(tmp_path / name)

    assert names_of(index_local_folders([tmp_path], max_bytes=MB)) == [name]


HEARING_IMPAIRED_NAMES = [
    "Movie.2010.en.hi.srt",
    "Movie.2010.en.HI.srt",
    "Movie.2010.eng.hi.srt",
    "Movie.2010.English.HI.srt",
    "Movie.2010.hi.en.srt",
    "Movie.2010.HI.srt",
    "Movie.2010.1080p.WEB-DL.H.264-GRP-HI.srt",
]


@pytest.mark.parametrize("name", HEARING_IMPAIRED_NAMES)
def test_index_keeps_a_hearing_impaired_english_file_at_the_plain_names_confidence(
    tmp_path: Path, name: str
) -> None:
    write(tmp_path / "hi" / name)
    write(tmp_path / "plain" / "Movie.2010.en.srt")
    target = TitleYear("Movie", 2010)

    hearing_impaired = match_local(index_local_folders([tmp_path / "hi"], max_bytes=MB), target)
    baseline = match_local(index_local_folders([tmp_path / "plain"], max_bytes=MB), target)

    assert hearing_impaired == LocalCandidate(tmp_path / "hi" / name, 1.0)
    assert baseline is not None
    assert hearing_impaired.confidence == baseline.confidence


@pytest.mark.parametrize("name", ["Movie.2010.Hindi.srt", "Movie.2010.en.fr.srt"])
def test_index_skips_a_hindi_file_and_an_english_plus_other_language_file(
    tmp_path: Path, name: str
) -> None:
    write(tmp_path / name)

    assert index_local_folders([tmp_path], max_bytes=MB) == []


def test_a_title_with_the_word_hi_is_indexed_and_matched(tmp_path: Path) -> None:
    write(tmp_path / "Hi.Mom.1970.en.srt")

    index = index_local_folders([tmp_path], max_bytes=MB)

    assert index[0].parsed.title == "Hi Mom"
    assert match_local(index, TitleYear("Hi Mom", 1970)) == LocalCandidate(
        tmp_path / "Hi.Mom.1970.en.srt", 1.0
    )


def test_a_french_sibling_does_not_make_the_english_file_ambiguous(tmp_path: Path) -> None:
    write(tmp_path / "Alien.1979.en.srt")
    write(tmp_path / "Alien.1979.fr.srt")

    result = match_local(index_local_folders([tmp_path], max_bytes=MB), TitleYear("Alien", 1979))

    assert result == LocalCandidate(tmp_path / "Alien.1979.en.srt", 1.0)


def test_a_folder_with_only_a_french_file_matches_nothing(tmp_path: Path) -> None:
    write(tmp_path / "Alien.1979.fr.srt")

    index = index_local_folders([tmp_path], max_bytes=MB)

    assert match_local(index, TitleYear("Alien", 1979)) is None


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

    assert local.is_link_like(cast("os.DirEntry[str]", symlink)) is True
    assert local.is_link_like(cast("os.DirEntry[str]", reparse)) is True
    assert local.is_link_like(cast("os.DirEntry[str]", plain)) is False
    assert local.is_link_like(cast("os.DirEntry[str]", posix)) is False


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


def test_a_library_file_named_aliens_in_the_alien_folder_never_gets_the_alien_subtitle(
    tmp_path: Path,
) -> None:
    write(tmp_path / "Alien.1979.en.srt")
    index = index_local_folders([tmp_path], max_bytes=MB)

    aliens = parse_library_name(Path("Alien (1979)") / "Aliens.mkv")
    alien = parse_library_name(Path("Alien (1979)") / "Alien.mkv")

    assert match_local(index, aliens) is None
    assert match_local(index, alien) == LocalCandidate(tmp_path / "Alien.1979.en.srt", 1.0)


def test_title_year_confidence_is_public_and_applies_the_match_rule() -> None:
    assert title_year_confidence(TitleYear("Alien", 1979), TitleYear("Alien", 1979)) == 1.0
    assert title_year_confidence(TitleYear("Alien", 1980), TitleYear("Alien", 1979)) == (1.0 * 0.95)
    assert title_year_confidence(TitleYear("Alien", 1981), TitleYear("Alien", 1979)) is None
    assert title_year_confidence(TitleYear("Alien", None), TitleYear("Alien", 1979)) == 0.9
    assert title_year_confidence(TitleYear("Alien", None), TitleYear("Aliens", 1986)) is None


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


def subtitle(name: str) -> LocalSubtitle:
    """An indexed subtitle named like a downloaded release."""
    return LocalSubtitle(Path("subs") / f"{name}.srt", parse_release_name(name))


@pytest.mark.parametrize(
    ("release", "library"),
    [
        ("The.Godfather.Part.II.en", "The Godfather (1972).mkv"),
        ("The Godfather Part II", "The Godfather (1972).mkv"),
        ("Dune.Part.Two.en", "Dune (2021).mkv"),
        ("Dune.Part.Two.en", "Dune (1984).mkv"),
        ("Back.to.the.Future.Part.III.en", "Back to the Future (1985).mkv"),
        ("Kill.Bill.Vol.2.en", "Kill Bill (2003).mkv"),
        ("Breaking.Dawn.Part.2.2012", "Breaking Dawn (2011).mkv"),
        ("Mockingjay.Part.2.2015", "Mockingjay (2014).mkv"),
        ("Dune.Part.Two.2024", "Dune (2023).mkv"),
        ("Dune.Part.Two.2024", "Dune (2024).mkv"),
        # And the other way round: the library file is the sequel.
        ("Dune.en", "Dune Part Two (2024).mkv"),
        ("Dune.2021", "Dune Part Two (2021).mkv"),
        ("The.Godfather.1972.en", "The Godfather Part II (1974).mkv"),
        ("The Godfather", "The Godfather Part II (1972).mkv"),
        ("Kill.Bill.2003", "Kill Bill Vol. 2 (2004).mkv"),
        ("Dune.Part.One.2021", "Dune Part Two (2021).mkv"),
    ],
)
def test_a_sequel_and_its_original_never_match(release: str, library: str) -> None:
    result = match_local([subtitle(release)], parse_library_name(Path(library)))

    assert result is None


@pytest.mark.parametrize(
    ("release", "library"),
    [
        ("Dune.Part.Two.2024", "Dune Part Two (2024).mkv"),
        ("Dune.Part.2.2024", "Dune Part Two (2024).mkv"),
        ("Dune.Part.Two.en", "Dune Part Two (2024).mkv"),
        ("The Godfather Part II", "The Godfather Part 2 (1974).mkv"),
        ("The.Godfather.Part.II.1974.en", "The Godfather Part II (1974).mkv"),
        ("Kill.Bill.Vol.2.2004.en", "Kill Bill Vol. 2 (2004).mkv"),
        ("Kill.Bill.Volume.2.2004", "Kill Bill Vol 2 (2004).mkv"),
        ("Rocky.II.1979", "Rocky 2 (1979).mkv"),
        ("Back.to.the.Future.Part.III.1990", "Back to the Future Part III (1990).mkv"),
    ],
)
def test_the_same_sequel_matches_however_its_number_is_written(release: str, library: str) -> None:
    result = match_local([subtitle(release)], parse_library_name(Path(library)))

    assert result is not None
    assert result.path == Path("subs") / f"{release}.srt"


def test_a_sequel_subtitle_is_picked_over_the_original_for_the_sequel() -> None:
    index = [subtitle("Dune.2021.en"), subtitle("Dune.Part.Two.2024.en")]

    result = match_local(index, parse_library_name(Path("Dune Part Two (2024).mkv")))

    assert result == LocalCandidate(index[1].path, 1.0)


@pytest.mark.parametrize(
    ("release", "library"),
    [
        ("Jackass.2.5.2007", "Jackass (2002).mkv"),
        ("Jackass.2.5.2007", "Jackass (2007).mkv"),
        ("Jackass.2.5.2007.1080p.BluRay.x264.en", "Jackass (2002).mkv"),
        ("Naked.Gun.2.5.1991", "Naked Gun (1991).mkv"),
        ("Naked.Gun.2.5.1991", "The Naked Gun (1988).mkv"),
        ("Naked.Gun.33.1-3.1994", "Naked Gun (1994).mkv"),
        ("Naked.Gun.33.1-3.1994", "The Naked Gun (1988).mkv"),
        # The other way round: the library file carries the number, the subtitle does not.
        ("Jackass.2002.en", "Jackass 2.5 (2007).mkv"),
        ("Jackass.en", "Jackass 2.5 (2007).mkv"),
        ("Naked.Gun.1991", "The Naked Gun 2 1-2 (1991).mkv"),
        ("Naked.Gun.1991", "The Naked Gun 2½ (1991).mkv"),
        ("Dune.2021", "Dune Pt 2.mkv"),
        ("Dune.2021", "Dune Pt. 2.mkv"),
        ("Dune.2021", "Dune (2021)/Dune Pt 2.mkv"),
        ("Dune.2021", "Dune (2021)/Dune Pt. 2.mkv"),
    ],
)
def test_a_dropped_or_odd_number_never_leads_to_the_wrong_movie(release: str, library: str) -> None:
    result = match_local([subtitle(release)], parse_library_name(Path(library)))

    assert result is None


@pytest.mark.parametrize(
    ("release", "library"),
    [
        ("Jackass.2.5.1080p.en", "Jackass 2.5.mkv"),
        ("Ocean's.11.2001", "Ocean's 11 (2001).mkv"),
        ("Se7en.1995", "Se7en (1995).mkv"),
        ("Apollo.13.1995", "Apollo 13 (1995).mkv"),
        ("1917.2019", "1917 (2019).mkv"),
        ("2012.2009", "2012 (2009).mkv"),
        ("10.Things.I.Hate.About.You.1999", "10 Things I Hate About You (1999).mkv"),
        ("300.2006", "300 (2006).mkv"),
        ("21.Jump.Street.2012", "21 Jump Street (2012).mkv"),
        ("District.9.2009", "District 9 (2009).mkv"),
        ("Blade.Runner.2049.2017", "Blade Runner 2049 (2017).mkv"),
        ("Movie.2010.1080p.BluRay.DD5.1.x264-GRP", "Movie (2010).mkv"),
        ("Movie.2010.AAC2.0.h264", "Movie (2010).mkv"),
        ("Movie.2010.DTS-HD.7.1", "Movie (2010).mkv"),
        ("Movie.2010.2160p.HDR.x265.10bit", "Movie (2010).mkv"),
        ("Movie.2010.720p.WEB-DL.DDP5.1.H.264-GRP", "Movie (2010).mkv"),
        ("Movie.2010.BluRay.1080p.x264.AC3.5.1-GRP", "Movie (2010).mkv"),
        ("Movie.2010.WEBRip.x264.AAC.2.0", "Movie (2010).mkv"),
        ("Promising.Young.Woman.2020.1080p.BluRay.x264.en.MX]", "Promising Young Woman (2020).mkv"),
    ],
)
def test_legitimate_releases_with_numbers_still_match_their_own_library_name(
    release: str, library: str
) -> None:
    result = match_local([subtitle(release)], parse_library_name(Path(library)))

    assert result is not None
    assert result.path == Path("subs") / f"{release}.srt"


@pytest.mark.parametrize(
    "stem",
    [
        "Harbour - Tidal Wake (2013)",
        "Harbour - Part III - The Long Night - Ends (2014)",
        "Harbour - Tidal Wake (Extended Edition) (2002)",
        "Harbour - Director's Cut (2001)",
        "Harbour Special Edition (1989)",
        "Silver Ultra (2015)",
        "VHS 94 (2021)",
        "12-12-12 (2011)",
        "[REC] (2007)",
        "[REC] 2 (2009)",
    ],
)
def test_a_subtitle_named_like_its_film_matches_it(stem: str) -> None:
    result = match_local([subtitle(f"{stem}.en")], parse_library_name(Path(f"{stem}.mkv")))

    assert result is not None
    assert result.path == Path("subs") / f"{stem}.en.srt"


@pytest.mark.parametrize(
    ("release", "library"),
    [
        ("Harbour - Tidal Wake.en", "Harbour - Tidal Wake (2018).avi"),
        ("Harbour - Tidal Wake 2013", "Harbour - Tidal Wake (2013).mkv"),
        ("Harbour.-.Tidal.Wake.2013.1080p.BluRay.x264-GRP", "Harbour - Tidal Wake (2013).mkv"),
        ("Harbour - Tidal Wake.en", "Harbour - Tidal Wake.mkv"),
        ("[YTS.MX] Harbour (2010).en", "Harbour (2010).mkv"),
        ("[1080p] Harbour (2010)", "Harbour (2010).mkv"),
        ("[1080P] Harbour (2010)", "Harbour (2010).mkv"),
        ("[4K] Harbour (2010)", "Harbour (2010).mkv"),
        ("[1080p BluRay] Harbour (2010)", "Harbour (2010).mkv"),
        ("www.site.com - Harbour (2010).en", "Harbour (2010).mkv"),
        ("Harbour (2010) - English", "Harbour (2010).mkv"),
    ],
)
def test_text_after_a_dash_and_site_tags_do_not_hide_the_right_film(
    release: str, library: str
) -> None:
    result = match_local([subtitle(release)], parse_library_name(Path(library)))

    assert result is not None


@pytest.mark.parametrize(
    ("release", "library"),
    [
        ("Harbour - Tidal Wake (2013).en", "Harbour (2012).mkv"),
        ("Harbour - For Good (2025).en", "Harbour (2024).mkv"),
        ("Harbour - Tidal Wake 2013", "Harbour (2012).mkv"),
        ("Harbour.-.Tidal.Wake.2013.1080p.BluRay.x264-GRP", "Harbour (2012).mkv"),
        ("Harbour - Tidal Wake.en", "Harbour.mkv"),
        ("Harbour.en", "Harbour - Tidal Wake.mkv"),
        # Another cut of a film has other timings and lines: its subtitle is not the film's.
        ("Harbour - Director's Cut (1979).en", "Harbour (1979).mkv"),
        ("Harbour (Extended Edition) (2002).en", "Harbour (2002).mkv"),
    ],
)
def test_a_sequels_subtitle_never_matches_the_shorter_titled_film(
    release: str, library: str
) -> None:
    result = match_local([subtitle(release)], parse_library_name(Path(library)))

    assert result is None
