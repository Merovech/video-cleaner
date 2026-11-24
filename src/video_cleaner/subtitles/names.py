"""Reading a title and year out of a movie's file name, and comparing two titles.

Names come from two places the user does not fully control: the library's file names and the
names of subtitle files downloaded by hand ("Movie.2020.1080p.BluRay.x264.en.MX]"). Every name
is bounded before it reaches `guessit`, which slows down sharply on long repetitive input, and
nothing here raises on an odd name: an unreadable one parses to an empty title, which never
matches anything.
"""

from __future__ import annotations

import functools
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from guessit import guessit

from video_cleaner.multipart import split_part_marker

_log = logging.getLogger("video_cleaner.subtitles.names")

# Longest name (and title) read: file names top out at 255 characters, and guessit takes
# seconds, not milliseconds, on a few thousand characters of repetitive text.
MAX_NAME_CHARS = 256

_MIN_YEAR = 1888  # the first film
_MAX_YEAR = 2100
# A library name: "Title (2010)", anything after the closing bracket is ignored.
_LIBRARY_NAME = re.compile(r"(?P<title>.*?)\s*\((?P<year>(?:18|19|20)[0-9]{2})\)", re.DOTALL)
# What a short extension looks like; "Dr. Mabuse the Gambler" has a dot but no extension.
_EXTENSION = re.compile(r"\.[A-Za-z0-9]{1,5}")
# File names that say nothing about the film, so the folder's name is used instead.
_GENERIC_STEMS = frozenset({"movie", "video", "film", "main", "feature"})
# Number words and roman numerals count as the digit they stand for, so "Part II", "Part Two"
# and "Part 2" are one title. A lone "i" is an ordinary word unless a part word comes before it.
_NUMBER_WORDS = {
    **{
        word: str(number)
        for number, word in enumerate(
            ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"), start=1
        )
    },
    **{
        numeral: str(number)
        for number, numeral in enumerate(
            ("ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"), start=2
        )
    },
    "iiii": "4",  # the clock-face form of four, still used for historical titles
}
# guessit results that say nothing about where the title area ends.
_STRUCTURAL_KEYS = frozenset({"title", "type"})
_PART_WORDS = frozenset({"part", "volume", "chapter", "episode"})
_WORD_ALIASES = {"pt": "part", "vol": "volume"}
# A trailing "pt1"/"cd2"/"part 3" marker, and a "(1922)" year closing the name before it.
_PART_WORD = re.compile(r"part|pt", re.IGNORECASE)
_YEAR_AT_END = re.compile(r"\((?:18|19|20)[0-9]{2}\)\s*\Z")
# Titles that differ in a sequel number are different films however alike the rest is.
_NUMBER_MISMATCH_CAP = 0.5
_LATIN_LIMIT = 0x250  # end of Latin Extended-B: accents are stripped from these letters only
_ENGLISH_CODES = frozenset({"en", "eng"})
# guessit reads the bare token "hi", the usual tag for hearing-impaired subtitles (Bazarr,
# Subliminal, Jellyfin), as Hindi.
_HEARING_IMPAIRED_CODE = "hi"
# guessit reports the language of a name as "language" and, for "VOSTFR"-style tags, as
# "subtitle_language".
_LANGUAGE_KEYS = ("language", "subtitle_language")


@dataclass(frozen=True)
class TitleYear:
    """What a name says about a film: its title and year, whether it is an episode, and the
    languages tagged in it (lower-case codes such as "en" or "pt-br", only set by
    `parse_release_name`). The languages describe a file, not the film, so they do not take part
    in comparing two of these.
    """

    title: str
    year: int | None
    is_episode: bool = False
    languages: tuple[str, ...] = field(default=(), compare=False)

    @property
    def non_english(self) -> bool:
        """Whether the name is tagged with a language and English is not the only one."""
        return any(_primary_subtag(code) not in _ENGLISH_CODES for code in self.languages)


def _empty() -> TitleYear:
    return TitleYear(title="", year=None)


def parse_release_name(name: str) -> TitleYear:
    """Title, year and episode flag of a release-style name such as "Film.2020.1080p.en".

    The whole name is read, including a mangled "extension" like ".en.264-EVO", so callers pass
    the name without a real extension. The type is forced to "movie" because guessit otherwise
    reads "2020.en.264" as season 2, episode 64; an episode is still told by its season and
    episode numbers. Only the first `MAX_NAME_CHARS` characters are read.

    A number guessit drops from the title would make a different film look the same ("Jackass
    2.5" is read as the title "Jackass" and a date), so such a name is unmatchable: its title is
    empty. That is a safe miss rather than a repair, because the lost number cannot be told from
    a release tag in general; see `_dropped_a_number`.
    """
    text = name[:MAX_NAME_CHARS]
    try:
        found = guessit(text, {"type": "movie", "advanced": True})
    except Exception:  # guessit's rule engine can fail on odd input; one name must not stop a run
        _log.debug("guessit failed on a name of %d characters", len(text), exc_info=True)
        return _empty()
    guess = {key: _matches(match) for key, match in found.items()}
    raw_title = _value(guess, "title")
    title = raw_title.strip() if isinstance(raw_title, str) else ""
    raw_year = _value(guess, "year")
    year = raw_year if isinstance(raw_year, int) and _MIN_YEAR <= raw_year <= _MAX_YEAR else None
    is_episode = _value(guess, "season") is not None or _value(guess, "episode") is not None
    languages = _language_codes(guess)
    if title and _dropped_a_number(text, title, guess):
        return TitleYear(title="", year=year, is_episode=is_episode, languages=languages)
    if title:
        # guessit moves "Part II" and "Vol 2" out of the title; they say which film it is.
        title = " ".join([title, *_numbered(_value(guess, "part"), "Part")])
        title = " ".join([title, *_numbered(_value(guess, "volume"), "Volume")])
    return TitleYear(title=title, year=year, is_episode=is_episode, languages=languages)


def _language_codes(guess: dict[str, list[Any]]) -> tuple[str, ...]:
    """The languages guessit found in the name, as lower-case codes, in name order.

    A "hi" code beside an English one is the hearing-impaired flag, not Hindi, and is dropped.
    Without an English tag it stays: a bare ".hi" is ambiguous, so the file counts as non-English.
    """
    found = [match for key in _LANGUAGE_KEYS for match in guess.get(key, [])]
    found.sort(key=lambda match: match.span)
    codes: list[str] = []
    for match in found:
        code = str(match.value).strip().lower()
        if code and code not in codes:
            codes.append(code)
    if _HEARING_IMPAIRED_CODE in codes and any(_primary_subtag(c) in _ENGLISH_CODES for c in codes):
        codes.remove(_HEARING_IMPAIRED_CODE)
    return tuple(codes)


def _primary_subtag(code: str) -> str:
    """ "pt" for "pt-br"."""
    return code.replace("_", "-").split("-", 1)[0]


def _matches(found: object) -> list[Any]:
    return list(found) if isinstance(found, list) else [found]


def _value(guess: dict[str, list[Any]], key: str) -> Any:
    """The value guessit found for *key* (a list when it found several), or None."""
    matches = guess.get(key)
    if not matches:
        return None
    values = [match.value for match in matches]
    return values[0] if len(values) == 1 else values


def _dropped_a_number(text: str, title: str, guess: dict[str, list[Any]]) -> bool:
    """Whether a number in the name's title area did not reach the parsed title.

    The title area is the text before the first tag guessit recognised after the title (the
    year, a resolution, a part marker...). Digit runs inside tags that came before the title
    ("[1080p] Movie") are ignored. A date match means guessit read "2.5.2007" or "33.1-3" as a
    date, which is always a number lost from the title.
    """
    if "date" in guess:
        return True
    title_start, title_end = guess["title"][0].span  # the caller has a title, so it exists
    tags = [
        match.span
        for key, matches in guess.items()
        if key not in _STRUCTURAL_KEYS
        for match in matches
        if match.span[1] > match.span[0]
    ]
    cutoff = min((start for start, _ in tags if start >= title_end), default=len(text))
    before_title = [(start, end) for start, end in tags if end <= title_start]
    kept = set(re.findall(r"[0-9]+", title))
    for run in re.finditer(r"[0-9]+", text[:cutoff]):
        if any(start <= run.start() < end for start, end in before_title):
            continue
        if run.group() not in kept:
            return True
    return False


def _numbered(value: object, label: str) -> list[str]:
    """ "Part 2" for a guessit part/volume value (a number, or a list of them)."""
    numbers = value if isinstance(value, list) else [value]
    return [f"{label} {number}" for number in numbers if isinstance(number, int)]


def _parse_stem(stem: str) -> TitleYear:
    """A library name without its extension: "Title (2010)" first, else a release name."""
    release = parse_release_name(stem)
    match = _LIBRARY_NAME.match(stem[:MAX_NAME_CHARS])
    if match is None or not match.group("title").strip():
        return release
    return TitleYear(
        title=match.group("title").strip(),
        year=int(match.group("year")),
        is_episode=release.is_episode,
    )


def _without_extension(name: str) -> str:
    suffix = Path(name).suffix
    return name[: -len(suffix)] if _EXTENSION.fullmatch(suffix) else name


def _without_split_marker(stem: str) -> str:
    """Drop a trailing "pt1"/"cd1" marker that says the file is one piece of the movie.

    A "part" or "pt" marker only counts after the year ("Title (1922) - pt1"): on a bare name,
    "Dune Part 2" and "Dune Pt 2" are a sequel's title. A "cd" marker is always a split file.
    """
    split = split_part_marker(stem)
    if split is None:
        return stem
    base = split[0]
    if _PART_WORD.search(stem, len(base)) and not _YEAR_AT_END.search(base):
        return stem
    return base


def parse_library_name(path: Path) -> TitleYear:
    """Title and year of a library file: "Title (2010).mkv", with a " - pt1" marker stripped.

    The folder's name fills in what the file name lacks: its year when the titles are the same
    once normalised, or the whole identity when the file is named something generic like
    "movie.mkv". A merely similar title does not borrow the year: "Aliens" in the folder
    "Alien (1979)" is another film, and giving it 1979 would match it to the first film's subtitle.
    """
    stem = _without_split_marker(_without_extension(path.name[:MAX_NAME_CHARS]))
    parsed = _parse_stem(stem)
    if parsed.is_episode or parsed.year is not None:
        return parsed
    folder_name = path.parent.name
    if not folder_name:
        return parsed
    folder = _parse_stem(folder_name)
    if folder.is_episode or folder.year is None or not folder.title:
        return parsed
    own = normalise_title(parsed.title)
    if not own or own in _GENERIC_STEMS:
        return folder
    if own == normalise_title(folder.title):
        return TitleYear(title=parsed.title, year=folder.year)
    return parsed


def normalise_title(title: str) -> str:
    """Casefold, strip Latin accents, "&" to "and", drop punctuation, collapse spaces.

    Number words and roman numerals become digits ("Part Two", "Part II" -> "part 2"; "Pt" and
    "Vol" are "part" and "volume").

    Only the first `MAX_NAME_CHARS` characters count. Accents come off Latin letters only, so
    kana and Cyrillic keep their marks (a voiced kana is not its unvoiced twin).
    """
    return _normalise(title[:MAX_NAME_CHARS])


@functools.lru_cache(maxsize=8192)
def _normalise(title: str) -> str:
    text = unicodedata.normalize("NFKD", title).casefold()
    text = text.replace("&", " and ")
    kept: list[str] = []
    for char in text:
        if unicodedata.combining(char) and kept and ord(kept[-1]) < _LATIN_LIMIT:
            continue
        kept.append(char)
    text = unicodedata.normalize("NFC", "".join(kept))
    text = text.replace("'", "").replace("\u2019", "")
    spaced = "".join(char if char.isalnum() else " " for char in text)
    words: list[str] = []
    for word in spaced.split():
        word = _WORD_ALIASES.get(word, word)
        if word in _NUMBER_WORDS:
            word = _NUMBER_WORDS[word]
        elif word == "i" and words and words[-1] in _PART_WORDS:
            word = "1"
        words.append(word)
    return " ".join(words)


def _number_tokens(normalised: str) -> frozenset[str]:
    return frozenset(token for token in normalised.split() if token.isdigit())


def title_similarity(a: str, b: str) -> float:
    """How alike two titles are, 0.0 to 1.0, after normalising both.

    Symmetric and deterministic. An empty title matches nothing. Titles whose numbers differ
    ("Toy Story" and "Toy Story 2") score at most 0.5: the characters are nearly the same but
    the films are not.
    """
    first = normalise_title(a)
    second = normalise_title(b)
    if not first or not second:
        return 0.0
    if first == second:
        return 1.0
    low, high = sorted((first, second))
    ratio = SequenceMatcher(None, low, high, autojunk=False).ratio()
    if _number_tokens(first) != _number_tokens(second):
        return min(ratio, _NUMBER_MISMATCH_CAP)
    return ratio
