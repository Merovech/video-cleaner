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


@pytest.mark.parametrize(
    ("name", "languages"),
    [
        ("Alien.1979", ()),
        ("Alien.1979.fr", ("fr",)),
        ("Alien.1979.fre", ("fr",)),
        ("Alien.1979.spa", ("es",)),
        ("Alien.1979.pt-BR", ("pt-br",)),
        ("Alien.1979.en", ("en",)),
        ("Alien.1979.eng", ("en",)),
        ("Alien.1979.English", ("en",)),
        ("Alien.1979.German.en", ("de", "en")),
        ("Alien.1979.VOSTFR", ("fr",)),
        ("Alien.1979.Multi", ("mul",)),
    ],
)
def test_release_name_reports_the_language_tags_of_the_name(
    name: str, languages: tuple[str, ...]
) -> None:
    parsed = parse_release_name(name)

    assert parsed.languages == languages
    assert (parsed.title, parsed.year) == ("Alien", 1979)


@pytest.mark.parametrize(
    ("name", "non_english"),
    [
        ("Alien.1979", False),
        ("Alien.1979.en", False),
        ("Alien.1979.eng", False),
        ("Alien.1979.English", False),
        ("Alien.1979.fr", True),
        ("Alien.1979.pt-BR", True),
        ("Alien.1979.en.fr", True),
        ("Alien.1979.German.en", True),
        ("Alien.1979.VOSTFR", True),
        ("Alien.1979.Multi", True),
    ],
)
def test_a_name_is_non_english_unless_english_is_its_only_language(
    name: str, non_english: bool
) -> None:
    assert parse_release_name(name).non_english is non_english


@pytest.mark.parametrize(
    "name",
    [
        "Movie.2010.en.hi",
        "Movie.2010.EN.HI",
        "Movie.2010.eng.hi",
        "Movie.2010.English.HI",
        "Movie.2010.hi.en",
    ],
)
def test_hi_beside_an_english_tag_is_the_hearing_impaired_flag_not_hindi(name: str) -> None:
    parsed = parse_release_name(name)

    assert parsed.languages == ("en",)
    assert parsed.non_english is False
    assert (parsed.title, parsed.year) == ("Movie", 2010)


@pytest.mark.parametrize(
    "name",
    [
        "Movie.2010.HI",
        "Movie.2010.hi",
        "Movie (2010).hi",
        "Movie.2010.1080p.WEB-DL.H.264-GRP-HI",
    ],
)
def test_a_bare_hi_tag_is_the_hearing_impaired_flag_too(name: str) -> None:
    # The text of every candidate is checked for English, so a Hindi file named this way is
    # still refused, by its content.
    parsed = parse_release_name(name)

    assert parsed.languages == ()
    assert parsed.non_english is False
    assert (parsed.title, parsed.year) == ("Movie", 2010)


@pytest.mark.parametrize(
    "name",
    [
        "Movie.2010.Hindi",
        "Movie.2010.hin",
        "Movie.2010.hi.fr",
        "Movie.2010.en.fr",
        "Movie.2010.en.hi.fr",  # English and a real language
        "Movie.2010.en.Hindi",
    ],
)
def test_hindi_written_out_and_other_languages_stay_non_english(name: str) -> None:
    assert parse_release_name(name).non_english is True


def test_a_title_containing_hi_is_unaffected_by_the_hearing_impaired_rule() -> None:
    parsed = parse_release_name("Hi.Mom.1970.en")

    assert (parsed.title, parsed.year, parsed.languages) == ("Hi Mom", 1970, ("en",))
    assert parse_release_name("Hi.Mom.1970.en.hi").title == "Hi Mom"


def test_the_languages_of_a_name_are_not_part_of_its_identity() -> None:
    assert parse_release_name("Alien.1979.fr") == TitleYear("Alien", 1979)


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


def test_library_name_does_not_give_the_folders_year_to_a_different_film() -> None:
    # "Aliens" is 0.91 similar to "Alien" but is another film: borrowing the year 1979 would make
    # it match the subtitle of the original at 0.909.
    parsed = parse_library_name(Path("Alien (1979)") / "Aliens.mkv")

    assert (parsed.title, parsed.year) == ("Aliens", None)


def test_library_name_still_borrows_the_folders_year_for_the_same_title() -> None:
    parsed = parse_library_name(Path("Dune (2021)") / "Dune.mkv")

    assert (parsed.title, parsed.year) == ("Dune", 2021)


def test_library_name_borrows_the_folders_year_when_only_spelling_differs() -> None:
    parsed = parse_library_name(Path("Amelie (2001)") / "Amélie.mkv")

    assert (parsed.title, parsed.year) == ("Amélie", 2001)


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


@pytest.mark.parametrize(
    ("name", "title"),
    [
        ("The.Godfather.Part.II.en", "The Godfather Part 2"),
        ("The Godfather Part II", "The Godfather Part 2"),
        ("Dune.Part.Two.2024", "Dune Part 2"),
        ("Dune.Part.2.2024", "Dune Part 2"),
        ("Back.to.the.Future.Part.III.en", "Back to the Future Part 3"),
        ("Kill.Bill.Vol.2.en", "Kill Bill Volume 2"),
        ("Kill.Bill.Volume.2.2004", "Kill Bill Volume 2"),
        ("Breaking.Dawn.Part.2.2012", "Breaking Dawn Part 2"),
    ],
)
def test_release_name_keeps_the_part_or_volume_number_in_the_title(name: str, title: str) -> None:
    assert parse_release_name(name).title == title


def test_release_name_keeps_the_year_next_to_a_part_number() -> None:
    parsed = parse_release_name("Dune.Part.Two.2024")

    assert parsed.year == 2024
    assert parsed.is_episode is False


def test_a_disc_marker_or_an_edition_is_not_part_of_the_title() -> None:
    assert parse_release_name("Alien.1979.CD1").title == "Alien"
    assert parse_release_name("Alien.1979.Directors.Cut").title == "Alien"
    assert parse_release_name("Movie.2010.Extended.Edition").title == "Movie"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("The Godfather Part II", "the godfather part 2"),
        ("The Godfather Part 2", "the godfather part 2"),
        ("Dune Part Two", "dune part 2"),
        ("Dune Pt. 2", "dune part 2"),
        ("Back to the Future Part III", "back to the future part 3"),
        ("Godfather Part I", "godfather part 1"),
        ("Kill Bill Vol. 2", "kill bill volume 2"),
        ("Kill Bill Volume Two", "kill bill volume 2"),
        ("Rocky IV", "rocky 4"),
        ("Rocky IIII", "rocky 4"),  # the clock-face form of four
        ("Rocky 4", "rocky 4"),
        ("Rocky", "rocky"),
        ("Malcolm X", "malcolm 10"),
    ],
)
def test_normalise_title_writes_word_and_roman_numbers_as_digits(raw: str, expected: str) -> None:
    assert normalise_title(raw) == expected


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("The Godfather Part II", "The Godfather"),
        ("The Godfather Part 2", "The Godfather"),
        ("Dune Part Two", "Dune"),
        ("Back to the Future Part III", "Back to the Future"),
        ("Kill Bill Vol. 2", "Kill Bill"),
        ("Breaking Dawn Part 2", "Breaking Dawn"),
        ("Mockingjay Part 2", "Mockingjay"),
        ("Dune Part One", "Dune Part Two"),
        ("Malcolm X", "Malcolm"),
    ],
)
def test_titles_that_differ_in_a_part_number_are_never_similar(a: str, b: str) -> None:
    assert title_similarity(a, b) <= 0.5
    assert title_similarity(b, a) <= 0.5


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("The Godfather Part II", "The Godfather Part 2"),
        ("Dune Part Two", "Dune Part 2"),
        ("Dune Pt 2", "Dune Part Two"),
        ("Kill Bill Vol. 2", "Kill Bill Volume Two"),
        ("Rocky II", "Rocky 2"),
        ("Rocky IIII", "Rocky IV"),
        ("Rocky IIII", "Rocky 4"),
        ("Godfather Part IIII", "Godfather Part Four"),
        ("Godfather Part I", "Godfather Part 1"),
    ],
)
def test_the_same_part_number_written_differently_is_the_same_title(a: str, b: str) -> None:
    assert title_similarity(a, b) == 1.0


def test_library_name_with_a_part_number_in_the_title_keeps_it() -> None:
    assert parse_library_name(Path("Dune Part Two (2024).mkv")) == TitleYear("Dune Part Two", 2024)
    assert parse_library_name(Path("Dune Part 2.mkv")).title == "Dune Part 2"


def test_a_part_marker_is_stripped_only_where_it_marks_a_split_file() -> None:
    assert parse_library_name(Path("Movie (2010) - pt1.mkv")).title == "Movie"
    assert parse_library_name(Path("Movie (2010) - part 2.mkv")).title == "Movie"
    assert parse_library_name(Path("Movie - cd1.mkv")).title == "Movie"
    # No closing year: "pt" is a sequel marker exactly like "part", so the number stays.
    assert parse_library_name(Path("Movie.pt2.mkv")).title == "Movie Part 2"
    assert parse_library_name(Path("Movie - pt1.mkv")).title == "Movie Part 1"
    assert parse_library_name(Path("Dune Pt 2.mkv")).title == "Dune Part 2"
    assert parse_library_name(Path("Dune Pt. 2.mkv")).title in {"Dune Part 2", "Dune Pt 2"}
    # A bare "Part 2" on a name with no year is a sequel's title, not a split file.
    assert parse_library_name(Path("Dune - Part 2.mkv")).title == "Dune Part 2"


def test_a_name_made_of_separators_is_bounded_before_any_pattern_runs() -> None:
    started = time.monotonic()
    for hostile in (
        "a" + " " * 100_000 + "x.mkv",
        "a" + "-" * 100_000 + "x.mkv",
        "." * 100_000 + "x",
    ):
        parse_library_name(Path(hostile))
        parse_library_name(Path("Folder " + hostile[:200]) / hostile)

    assert time.monotonic() - started < 10.0


@pytest.mark.parametrize("name", ["Dune Pt 2.mkv", "Dune Pt. 2.mkv", "Dune - Pt 2.mkv"])
def test_a_year_less_pt_sequel_keeps_its_number_and_the_folders_year_is_not_borrowed(
    name: str,
) -> None:
    parsed = parse_library_name(Path("Dune (2021)") / name)

    assert normalise_title(parsed.title) == "dune part 2"
    assert parsed.year is None


def test_a_split_file_marker_after_the_year_is_still_stripped() -> None:
    assert parse_library_name(Path("Dr. Mabuse the Gambler (1922) - pt1.mkv")) == TitleYear(
        "Dr. Mabuse the Gambler", 1922
    )
    assert parse_library_name(Path("Movie (2010) - cd2.mkv")).title == "Movie"


@pytest.mark.parametrize(
    "name",
    [
        "Jackass.2.5.2007",
        "Jackass.2.5.2007.1080p.BluRay.x264.en",
        "Naked.Gun.2.5.1991",
        "Naked.Gun.33.1-3.1994",
        "Movie.2010.5.1",
        "The Naked Gun 2 1/2.en",
    ],
)
def test_a_release_whose_number_guessit_dropped_is_unmatchable(name: str) -> None:
    parsed = parse_release_name(name)

    assert parsed.title == ""
    assert parsed.is_episode is False


@pytest.mark.parametrize(
    ("name", "title", "year"),
    [
        ("Ocean's.11.2001", "Ocean's 11", 2001),
        ("Se7en.1995", "Se7en", 1995),
        ("Apollo.13.1995", "Apollo 13", 1995),
        ("1917.2019", "1917", 2019),
        ("2012.2009", "2012", 2009),
        ("10.Things.I.Hate.About.You.1999", "10 Things I Hate About You", 1999),
        ("300.2006", "300", 2006),
        ("21.Jump.Street.2012", "21 Jump Street", 2012),
        ("District.9.2009", "District 9", 2009),
        ("Blade.Runner.2049.2017", "Blade Runner 2049", 2017),
        ("Toy.Story.3.2010.en", "Toy Story 3", 2010),
        ("Mad.Max.2.1981", "Mad Max 2", 1981),
        ("Alien.3.1992", "Alien 3", 1992),
        ("Jackass.2.5.1080p", "Jackass 2.5.", None),
        ("Movie.2010.1080p.BluRay.DD5.1.x264-GRP", "Movie", 2010),
        ("Movie.2010.AAC2.0.h264", "Movie", 2010),
        ("Movie.2010.DTS-HD.7.1", "Movie", 2010),
        ("Movie.2010.2160p.HDR.x265.10bit", "Movie", 2010),
        ("Movie.2010.720p.WEB-DL.DDP5.1.H.264-GRP", "Movie", 2010),
        ("Movie.2010.BluRay.1080p.x264.AC3.5.1-GRP", "Movie", 2010),
        ("Movie.Name.1080p.BluRay.x264.5.1", "Movie Name", None),
        ("Movie.2010.NF.WEB-DL.DDP2.0.H.264", "Movie", 2010),
        ("Movie.2010.1080p.BluRay.FLAC.2.0.x264", "Movie", 2010),
        ("Movie.2010.WEBRip.x264.AAC.2.0", "Movie", 2010),
        ("Movie.2010.Dual.Audio.DD+5.1", "Movie", 2010),
        ("Movie.2010.2.0", "Movie", 2010),
        ("Movie.2010.v2", "Movie", 2010),
        ("Movie.2010.REPACK2", "Movie", 2010),
        ("Movie.2010.24fps", "Movie", 2010),
        ("[1080p] Movie.2010", "Movie", 2010),
        ("[YTS.MX] Movie.2010.1080p", "Movie", 2010),
        ("Promising.Young.Woman.2020.1080p.BluRay.x264.en.MX]", "Promising Young Woman", 2020),
    ],
)
def test_legitimate_numbers_in_release_names_still_parse(
    name: str, title: str, year: int | None
) -> None:
    parsed = parse_release_name(name)

    assert (parsed.title, parsed.year) == (title, year)


def test_library_names_keep_a_decimal_or_fractional_number() -> None:
    # A file name cannot hold "1/2", so a fraction is written "1-2" or as the one-half sign.
    jackass = parse_library_name(Path("Jackass 2.5 (2007).mkv"))
    dashed = parse_library_name(Path("The Naked Gun 2 1-2 (1991).mkv"))
    half = parse_library_name(Path("The Naked Gun 2½ (1991).mkv"))

    assert jackass == TitleYear("Jackass 2.5", 2007)
    assert dashed == TitleYear("The Naked Gun 2 1-2", 1991)
    assert half == TitleYear("The Naked Gun 2½", 1991)
    for parsed, original in (
        (jackass, "Jackass"),
        (dashed, "The Naked Gun"),
        (half, "The Naked Gun"),
    ):
        assert title_similarity(parsed.title, original) <= 0.5


def test_a_library_name_without_a_year_keeps_the_text_after_a_dash() -> None:
    # This is also the title the OpenSubtitles title search sends.
    parsed = parse_library_name(Path("Harbour - Tidal Wake.mkv"))

    assert (parsed.title, parsed.year) == ("Harbour Tidal Wake", None)
