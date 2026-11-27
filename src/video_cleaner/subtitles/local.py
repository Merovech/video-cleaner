"""Finding a movie's subtitle among the user's own local subtitle folders.

`index_local_folders` lists the folders: names come from the directory listing and sizes from
`stat`. A file named like a subtitle is not read there. Downloaded subtitles often lose their
extension to a release tag ("Film (2010).en.264-GRP"), so a file with any other name is read
(within the size limit and a cap on how many) and kept only if its content is a subtitle.
File names there are untrusted, so a name that cannot be read sensibly is skipped, and symlinks
and Windows junctions are never followed out of a listed folder. `match_local` is conservative
on purpose: a wrong subtitle muxed into a movie is worse than none, so a doubtful or ambiguous
match is `None`.
"""

from __future__ import annotations

import logging
import os
import re
import stat
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from video_cleaner.atomicio import read_bytes_capped
from video_cleaner.subtitles.names import (
    TitleYear,
    normalise_title,
    parse_release_name,
    title_similarity,
)
from video_cleaner.subtitles.textfmt import SubtitleRejected, normalise_subtitle

_log = logging.getLogger("video_cleaner.subtitles.local")

# The extensions of the text subtitle formats: such a file is indexed on its name alone.
SUBTITLE_EXTENSIONS = (".srt", ".ass", ".ssa", ".vtt")
# Most directory entries looked at in one index, and the deepest folder level listed.
MAX_ENTRIES_EXAMINED = 100_000
MAX_DEPTH = 8
# Most files with another name that are opened to see whether they hold a subtitle.
MAX_FILES_READ = 2_000

# Endings of unfinished downloads and backups: such a file may hold only part of a subtitle,
# or repeat one that is there.
_LEFTOVER_ENDINGS = (
    ".part",
    ".partial",
    ".crdownload",
    ".download",
    ".!ut",
    ".tmp",
    ".bak",
    ".old",
    ".orig",
    ".disabled",
    "~",
)
# Endings of files that are never text subtitles, so not worth opening.
_NEVER_SUBTITLE_ENDINGS = (".nfo", ".jpg", ".jpeg", ".png", ".gif", ".idx", ".zip", ".rar", ".7z")
# A subtitle's name with a copy mark after it: "Film.srt.1", "Film.srt.txt", "Film.srt (1)".
# Only at the very end, so a title that spells an extension ("The.Smart.Ass.2010") is not one.
_COPY_NAME = re.compile(
    r"\.(?:srt|ass|ssa|vtt)"
    r"(?:\.(?:[0-9]{1,3}|txt|bk|save|backup)| ?\([0-9]{1,3}\)| - copy(?: \([0-9]{1,3}\))?)\Z"
)

# A title must be at least this similar to match when both years agree...
SIMILARITY_SAME_YEAR = 0.90
# ...and when the years differ by one (a year the library and the release disagree on).
SIMILARITY_YEAR_OFF_BY_ONE = 0.95
# The best match must lead every other candidate by more than this, or there is no match.
AMBIGUITY_MARGIN = 0.02
# Without a year on one side the title alone has to carry the match: it must be near-exact and
# the confidence stays below certainty.
SIMILARITY_NO_YEAR = 0.95
CONFIDENCE_NO_YEAR_MAX = 0.9
# A year-off-by-one match is less sure than a same-year one.
_CONFIDENCE_YEAR_OFF_BY_ONE = 0.95

_scandir = os.scandir  # a seam for tests that need a listing to fail


@dataclass(frozen=True)
class LocalSubtitle:
    path: Path
    parsed: TitleYear


@dataclass(frozen=True)
class LocalCandidate:
    path: Path
    confidence: float


def is_link_like(entry: os.DirEntry[str]) -> bool:
    """A symlink, or on Windows any reparse point (a junction is not reported as a symlink)."""
    if entry.is_symlink():
        return True
    attributes = getattr(entry.stat(follow_symlinks=False), "st_file_attributes", 0)
    return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _subtitle_extension(name: str) -> str | None:
    lowered = name.lower()
    for extension in SUBTITLE_EXTENSIONS:
        if lowered.endswith(extension):
            return extension
    return None


def _walk(roots: Sequence[str]) -> Iterator[os.DirEntry[str]]:
    """Yield the regular files under *roots*, never following a link, within the caps."""
    examined = 0
    stack = [(root, 0) for root in reversed(roots)]
    while stack:
        directory, depth = stack.pop()
        try:
            with _scandir(directory) as entries:
                for entry in entries:
                    examined += 1
                    if examined > MAX_ENTRIES_EXAMINED:
                        _log.warning("stopping the listing after %d entries", examined - 1)
                        return
                    try:
                        if is_link_like(entry):
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            if depth < MAX_DEPTH:
                                stack.append((entry.path, depth + 1))
                        elif entry.is_file(follow_symlinks=False):
                            yield entry
                    except OSError:
                        _log.debug("skipping an entry that cannot be read: %r", entry.name)
        except OSError as err:
            _log.warning("cannot list %r: %s", directory, err)


def _worth_opening(name: str) -> bool:
    """Whether a file without a subtitle extension could be a whole subtitle, by its name."""
    lowered = name.lower()
    return not (
        lowered.endswith(_LEFTOVER_ENDINGS + _NEVER_SUBTITLE_ENDINGS) or _COPY_NAME.search(lowered)
    )


def _holds_subtitle(path: Path, max_bytes: int) -> bool:
    """Whether the content of *path* reads as a text subtitle. A file that cannot be read, or
    that is no longer the regular file that was listed, does not."""
    try:
        raw = read_bytes_capped(path, max_bytes, regular_only=True)
        if raw is None:
            return False
        # Only "is this a subtitle": the resolver applies the user's own minimum of cues.
        normalise_subtitle(raw, max_bytes=max_bytes, min_cues=1)
    except (OSError, SubtitleRejected):
        return False
    return True


def index_local_folders(dirs: Sequence[Path], *, max_bytes: int) -> list[LocalSubtitle]:
    """Index the text subtitle files under *dirs*, parsed by their release names.

    A file with a subtitle extension is taken on its name; any other file only if its content
    is a subtitle. Unfinished downloads, backups and copies (`_worth_opening`) are never taken
    that way. When more than `MAX_FILES_READ` such files would have to be opened, none of them
    is used (and a warning says so): judged on part of the list, one film's subtitle could go
    to its neighbour.

    Skipped without error: anything that is not a regular file, empty files, files over
    *max_bytes*, entries that cannot be read or listed, symlinks and junctions, names that parse
    to no title, episodes, and names tagged with a language other than English alone ("fr",
    "spa", "pt-BR", "en.fr"). The result is sorted by path.
    """
    roots: dict[str, str] = {}
    for folder in dirs:
        root = os.path.abspath(folder)
        roots.setdefault(os.path.normcase(root), root)
    found: dict[str, LocalSubtitle] = {}
    by_content: set[str] = set()  # the keys of `found` that were taken for what they hold
    files_read = 0
    too_many = False
    for entry in _walk(list(roots.values())):
        extension = _subtitle_extension(entry.name)
        if extension is None and (too_many or not _worth_opening(entry.name)):
            continue
        try:
            size = entry.stat(follow_symlinks=False).st_size
        except OSError:
            _log.debug("skipping a file that cannot be read: %r", entry.name)
            continue
        if not 0 < size <= max_bytes:
            continue
        # A mangled "extension" stays: `parse_release_name` reads the whole name.
        stem = entry.name if extension is None else entry.name[: -len(extension)]
        parsed = parse_release_name(stem)
        if parsed.is_episode or not normalise_title(parsed.title):
            continue
        # A name tagged with a language other than English (or with several, English among
        # them) is not trusted to hold English text. A name with no language tag stays: its
        # content language is not known here, the resolver checks the content.
        if parsed.non_english:
            continue
        key = os.path.normcase(entry.path)
        # Opened last, so that only a file the name leaves usable counts toward the cap.
        if extension is None:
            if files_read >= MAX_FILES_READ:
                too_many = True
                continue
            files_read += 1
            if not _holds_subtitle(Path(entry.path), max_bytes):
                continue
            by_content.add(key)
        found.setdefault(key, LocalSubtitle(Path(entry.path), parsed))
    if too_many:
        _log.warning(
            "more than %d files without a subtitle extension: none of them is used "
            "(point local_subtitle_dirs at folders that hold only subtitles)",
            MAX_FILES_READ,
        )
        for key in by_content:
            del found[key]
    return sorted(found.values(), key=lambda item: str(item.path))


def title_year_confidence(file: TitleYear, target: TitleYear) -> float | None:
    """How sure a file is the target's subtitle, or None when it does not qualify.

    Title and year are compared by the thresholds above; the same rule is meant for any
    candidate that has a title and a year, wherever it comes from.
    """
    similarity = title_similarity(file.title, target.title)
    if file.year is None or target.year is None:
        if similarity < SIMILARITY_NO_YEAR:
            return None
        return min(similarity, CONFIDENCE_NO_YEAR_MAX)
    gap = abs(file.year - target.year)
    if gap == 0 and similarity >= SIMILARITY_SAME_YEAR:
        return similarity
    if gap == 1 and similarity >= SIMILARITY_YEAR_OFF_BY_ONE:
        return similarity * _CONFIDENCE_YEAR_OFF_BY_ONE
    return None


def match_local(index: Sequence[LocalSubtitle], target: TitleYear) -> LocalCandidate | None:
    """The one indexed file that is *target*'s subtitle, or None.

    A file qualifies when its title is similar enough for the year gap (see the thresholds
    above). The best qualifying file wins only if no other file scores within
    `AMBIGUITY_MARGIN` of it. Episodes, and titles with nothing to compare, never match.
    """
    if target.is_episode or not normalise_title(target.title):
        return None
    scored: dict[Path, float] = {}
    for item in index:
        if item.parsed.is_episode:
            continue
        confidence = title_year_confidence(item.parsed, target)
        if confidence is not None and confidence > scored.get(item.path, 0.0):
            scored[item.path] = confidence
    if not scored:
        return None
    ranked = sorted(scored.items(), key=lambda pair: (-pair[1], str(pair[0])))
    best_path, best = ranked[0]
    if len(ranked) > 1 and best - ranked[1][1] <= AMBIGUITY_MARGIN:
        return None
    return LocalCandidate(best_path, best)
