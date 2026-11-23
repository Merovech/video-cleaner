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
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

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
_SAME_TITLE_AS_FOLDER = 0.90
# Roman numerals read as part of a title ("Rocky IV"); "i" is left out as an ordinary word.
_ROMAN_NUMERALS = frozenset({"ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"})
# Titles that differ in a sequel number are different films however alike the rest is.
_NUMBER_MISMATCH_CAP = 0.5
_LATIN_LIMIT = 0x250  # end of Latin Extended-B: accents are stripped from these letters only


@dataclass(frozen=True)
class TitleYear:
    title: str
    year: int | None
    is_episode: bool = False


def _empty() -> TitleYear:
    return TitleYear(title="", year=None)


def parse_release_name(name: str) -> TitleYear:
    """Title, year and episode flag of a release-style name such as "Film.2020.1080p.en".

    The whole name is read, including a mangled "extension" like ".en.264-EVO", so callers pass
    the name without a real extension. The type is forced to "movie" because guessit otherwise
    reads "2020.en.264" as season 2, episode 64; an episode is still told by its season and
    episode numbers. Only the first `MAX_NAME_CHARS` characters are read.
    """
    text = name[:MAX_NAME_CHARS]
    try:
        guess = guessit(text, {"type": "movie"})
    except Exception:  # guessit's rule engine can fail on odd input; one name must not stop a run
        _log.debug("guessit failed on a name of %d characters", len(text), exc_info=True)
        return _empty()
    raw_title = guess.get("title")
    title = raw_title if isinstance(raw_title, str) else ""
    raw_year = guess.get("year")
    year = raw_year if isinstance(raw_year, int) and _MIN_YEAR <= raw_year <= _MAX_YEAR else None
    is_episode = guess.get("season") is not None or guess.get("episode") is not None
    return TitleYear(title=title.strip(), year=year, is_episode=is_episode)


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


def parse_library_name(path: Path) -> TitleYear:
    """Title and year of a library file: "Title (2010).mkv", with a " - pt1" marker stripped.

    The folder's name fills in what the file name lacks: its year when the titles agree, or the
    whole identity when the file is named something generic like "movie.mkv".
    """
    stem = _without_extension(path.name)
    split = split_part_marker(stem)
    if split is not None:
        stem = split[0]
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
    if title_similarity(parsed.title, folder.title) >= _SAME_TITLE_AS_FOLDER:
        return TitleYear(title=parsed.title, year=folder.year)
    return parsed


def normalise_title(title: str) -> str:
    """Casefold, strip Latin accents, "&" to "and", drop punctuation, collapse spaces.

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
    return " ".join(spaced.split())


def _number_tokens(normalised: str) -> frozenset[str]:
    return frozenset(
        token for token in normalised.split() if token.isdigit() or token in _ROMAN_NUMERALS
    )


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
