"""Finding a movie's subtitle among the user's own local subtitle folders.

`index_local_folders` lists the folders without reading any file: names come from the directory
listing and sizes from `stat`. File names there are untrusted, so a name that cannot be read
sensibly is skipped, and symlinks and Windows junctions are never followed out of a listed
folder. `match_local` is conservative on purpose: a wrong subtitle muxed into a movie is worse
than none, so a doubtful or ambiguous match is `None`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from video_cleaner.subtitles.names import (
    TitleYear,
    normalise_title,
    parse_release_name,
    title_similarity,
)

_log = logging.getLogger("video_cleaner.subtitles.local")

# Text subtitle formats only: ".sub" is a different format (image-based or MicroDVD).
SUBTITLE_EXTENSIONS = (".srt", ".ass", ".ssa", ".vtt")
# Most directory entries looked at in one index, and the deepest folder level listed.
MAX_ENTRIES_EXAMINED = 100_000
MAX_DEPTH = 8

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
_REPARSE_POINT = 0x400  # FILE_ATTRIBUTE_REPARSE_POINT: symlinks, junctions, mount points


@dataclass(frozen=True)
class LocalSubtitle:
    path: Path
    parsed: TitleYear


@dataclass(frozen=True)
class LocalCandidate:
    path: Path
    confidence: float


def _is_link_like(entry: os.DirEntry[str]) -> bool:
    """A symlink, or on Windows any reparse point (a junction is not reported as a symlink)."""
    if entry.is_symlink():
        return True
    attributes = getattr(entry.stat(follow_symlinks=False), "st_file_attributes", 0)
    return bool(attributes & _REPARSE_POINT)


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
                        if _is_link_like(entry):
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


def index_local_folders(dirs: Sequence[Path], *, max_bytes: int) -> list[LocalSubtitle]:
    """Index the text subtitle files under *dirs*, parsed by their release names.

    Skipped without error: anything that is not a regular file, empty files, files over
    *max_bytes*, entries that cannot be read or listed, symlinks and junctions, names that parse
    to no title, and episodes. The result is sorted by path.
    """
    roots: dict[str, str] = {}
    for folder in dirs:
        root = os.path.abspath(folder)
        roots.setdefault(os.path.normcase(root), root)
    found: dict[str, LocalSubtitle] = {}
    for entry in _walk(list(roots.values())):
        extension = _subtitle_extension(entry.name)
        if extension is None:
            continue
        try:
            size = entry.stat(follow_symlinks=False).st_size
        except OSError:
            _log.debug("skipping a file that cannot be read: %r", entry.name)
            continue
        if not 0 < size <= max_bytes:
            continue
        parsed = parse_release_name(entry.name[: -len(extension)])
        if parsed.is_episode or not normalise_title(parsed.title):
            continue
        found.setdefault(os.path.normcase(entry.path), LocalSubtitle(Path(entry.path), parsed))
    return sorted(found.values(), key=lambda item: str(item.path))


def _confidence(file: TitleYear, target: TitleYear) -> float | None:
    """How sure a file is the target's subtitle, or None when it does not qualify."""
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
        confidence = _confidence(item.parsed, target)
        if confidence is not None and confidence > scored.get(item.path, 0.0):
            scored[item.path] = confidence
    if not scored:
        return None
    ranked = sorted(scored.items(), key=lambda pair: (-pair[1], str(pair[0])))
    best_path, best = ranked[0]
    if len(ranked) > 1 and best - ranked[1][1] <= AMBIGUITY_MARGIN:
        return None
    return LocalCandidate(best_path, best)
