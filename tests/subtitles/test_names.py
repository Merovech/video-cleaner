"""Reading a title and year out of library file names and subtitle release names."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from video_cleaner.subtitles import names
from video_cleaner.subtitles.names import (
    TitleYear,
    normalise_title,
    parse_library_name,
    parse_release_name,
    title_similarity,
)


@pytest.mark.parametrize(
    ("name", "title", "year"),
    [
        ("Promising.Young.Woman.2020.1080p.BluRay.x264.en.MX]", "Promising Young Woman", 2020),
        ("The.First.Omen.2024.WEB-DL.en.264-EVO", "The First Omen", 2024),
        # No source tag: guessit alone reads "2020.en.264" as season 2 episode 64.
        ("Some.Film.2020.en.264-EVO", "Some Film", 2020),
        ("Alien (1979)", "Alien", 1979),
        ("Alien.1979.en", "Alien", 1979),
        ("1917.2019.1080p.BluRay", "1917", 2019),
        ("Blade Runner 2049 (2017)", "Blade Runner 2049", 2017),
        ("Toy Story 2 (1999)", "Toy Story 2", 1999),
    ],
)
def test_release_name_yields_title_and_year(name: str, title: str, year: int) -> None:
    parsed = parse_release_name(name)

    assert parsed == TitleYear(title=title, year=year, is_episode=False)


def test_release_name_without_a_year_has_year_none() -> None:
    parsed = parse_release_name("Some.Film.1080p.en")

    assert parsed.title == "Some Film"
    assert parsed.year is None
    assert parsed.is_episode is False


@pytest.mark.parametrize(
    "name",
    ["Show.S01E08.720p.en", "Show 1x08", "Show.2020.S02.COMPLETE", "show.s10e101.en"],
)
def test_episode_names_are_flagged(name: str) -> None:
    assert parse_release_name(name).is_episode is True


def test_a_name_with_nothing_readable_gives_an_empty_title() -> None:
    assert parse_release_name("") == TitleYear(title="", year=None)
    assert parse_release_name("...") == TitleYear(title="", year=None)


def test_a_name_with_a_lone_surrogate_does_not_raise() -> None:
    parsed = parse_release_name("Odd\udc80Name.2020.en")

    assert parsed.year in (2020, None)


def test_a_huge_name_is_bounded_and_does_not_hang() -> None:
    started = time.monotonic()
    for hostile in ("a" * 1_000_000, "Movie.2020." * 100_000, "S01E02." * 100_000):
        parse_release_name(hostile)

    assert time.monotonic() - started < 10.0


def test_only_the_start_of_a_long_name_is_read() -> None:
    parsed = parse_release_name("Short.Title.2001." + "x" * 5000)

    assert parsed.title == "Short Title"
    assert parsed.year == 2001


@pytest.mark.parametrize(
    ("path", "title", "year"),
    [
        (Path("Alien (1979).mkv"), "Alien", 1979),
        (Path("Dr. Mabuse the Gambler (1922) - pt1.mkv"), "Dr. Mabuse the Gambler", 1922),
        (Path("Dr. Mabuse the Gambler (1922) - pt2.mkv"), "Dr. Mabuse the Gambler", 1922),
        (Path("Amélie (2001).mkv"), "Amélie", 2001),
        (Path("Tom & Jerry (2021).mp4"), "Tom & Jerry", 2021),
        (Path("2012 (2009).mkv"), "2012", 2009),
        (Path("Movie.Name.2020.1080p.mkv"), "Movie Name", 2020),
    ],
)
def test_library_name_yields_title_and_year(path: Path, title: str, year: int) -> None:
    parsed = parse_library_name(path)

    assert parsed == TitleYear(title=title, year=year, is_episode=False)


def test_library_name_keeps_a_dot_inside_the_title_of_a_file_without_an_extension() -> None:
    parsed = parse_library_name(Path("Dr. Mabuse (1922)"))

    assert (parsed.title, parsed.year) == ("Dr. Mabuse", 1922)


def test_library_name_flags_episodes() -> None:
    assert parse_library_name(Path("Show S01E08.mkv")).is_episode is True


def test_library_name_takes_the_year_from_the_folder_when_the_file_has_none() -> None:
    parsed = parse_library_name(Path("Alien (1979)") / "Alien.mkv")

    assert (parsed.title, parsed.year) == ("Alien", 1979)


def test_library_name_uses_the_folder_when_the_file_name_is_generic() -> None:
    parsed = parse_library_name(Path("Alien (1979)") / "movie.mkv")

    assert (parsed.title, parsed.year) == ("Alien", 1979)


def test_library_name_does_not_borrow_the_folder_for_an_unrelated_file() -> None:
    parsed = parse_library_name(Path("Alien (1979)") / "Behind The Scenes.mkv")

    assert parsed.title == "Behind The Scenes"
    assert parsed.year is None


def test_library_name_prefers_the_file_year_over_the_folder_year() -> None:
    parsed = parse_library_name(Path("Alien (1979)") / "Alien (1980).mkv")

    assert parsed.year == 1980


def test_library_name_of_a_bare_file_name_without_a_year_has_year_none() -> None:
    parsed = parse_library_name(Path("Alien.mkv"))

    assert (parsed.title, parsed.year) == ("Alien", None)


def test_library_name_ignores_a_folder_that_is_an_episode_or_has_no_year() -> None:
    assert parse_library_name(Path("Show S01E01") / "Alien.mkv").year is None
    assert parse_library_name(Path("Some Folder") / "Alien.mkv").year is None


def test_a_failing_guessit_gives_an_empty_title(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*args: object, **kwargs: object) -> object:
        raise ValueError("odd name")

    monkeypatch.setattr(names, "guessit", explode)

    assert parse_release_name("Alien.1979") == TitleYear(title="", year=None)


def test_library_name_with_no_year_anywhere_has_year_none() -> None:
    parsed = parse_library_name(Path("Movies") / "Alien.mkv")

    assert (parsed.title, parsed.year) == ("Alien", None)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Amélie", "amelie"),
        ("Amélie", "amelie"),  # decomposed accent
        ("Tom & Jerry", "tom and jerry"),
        ("Tom&Jerry", "tom and jerry"),
        ("  The   Matrix:  Reloaded! ", "the matrix reloaded"),
        ("Dr. Mabuse", "dr mabuse"),
        ("Schindler's List", "schindlers list"),
        ("Schindler\u2019s List", "schindlers list"),
        ("STRASSE", "strasse"),
        ("Straße", "strasse"),
        ("Se7en", "se7en"),
        ("...", ""),
        ("", ""),
    ],
)
def test_normalise_title(raw: str, expected: str) -> None:
    assert normalise_title(raw) == expected


def test_normalise_title_keeps_non_latin_letters() -> None:
    assert normalise_title("千と千尋の神隠し") == "千と千尋の神隠し"


def test_normalise_title_bounds_a_huge_input() -> None:
    started = time.monotonic()
    result = normalise_title("é" * 1_000_000)

    assert time.monotonic() - started < 5.0
    assert 0 < len(result) <= 1000


def test_identical_titles_score_one() -> None:
    assert title_similarity("Alien", "alien!") == 1.0


def test_accents_and_ampersand_do_not_lower_the_score() -> None:
    assert title_similarity("Amélie", "Amelie") == 1.0
    assert title_similarity("Tom & Jerry", "Tom and Jerry") == 1.0


def test_unrelated_titles_score_low() -> None:
    assert title_similarity("Alien", "Casablanca") <= 0.5


def test_empty_titles_never_match() -> None:
    assert title_similarity("", "") == 0.0
    assert title_similarity("Alien", "") == 0.0
    assert title_similarity("", "Alien") == 0.0
    assert title_similarity("...", "!!!") == 0.0


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Alien", "Aliens"),
        ("Toy Story", "Toy Story 2"),
        ("The Lord of the Rings", "The Lord of the Ring"),
        ("abcab", "abab"),
        ("aaaa bbbb aaaa", "bbbb aaaa bbbb"),
        ("Dune", "Dune Part Two"),
    ],
)
def test_similarity_is_symmetric(a: str, b: str) -> None:
    assert title_similarity(a, b) == title_similarity(b, a)


def test_similarity_is_deterministic() -> None:
    scores = {title_similarity("The Matrix Reloaded", "The Matrix Revolutions") for _ in range(5)}

    assert len(scores) == 1


def test_a_sequel_number_is_never_a_near_match() -> None:
    assert title_similarity("Toy Story", "Toy Story 2") <= 0.5
    assert title_similarity("Toy Story 2", "Toy Story 3") <= 0.5
    assert title_similarity("Rocky", "Rocky V") <= 0.5
    assert title_similarity("Rocky IV", "Rocky V") <= 0.5
    assert title_similarity("Toy Story 2", "Toy Story 2") == 1.0


def test_similarity_stays_fast_for_hostile_long_titles() -> None:
    a = "ab" * 500_000
    b = "ba" * 500_000

    started = time.monotonic()
    score = title_similarity(a, b)

    assert time.monotonic() - started < 5.0
    assert 0.0 <= score <= 1.0
    assert score == title_similarity(b, a)


def test_the_name_length_cap_is_a_module_constant() -> None:
    assert names.MAX_NAME_CHARS == 256
