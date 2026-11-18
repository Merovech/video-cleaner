"""Multi-part movie detection (``Movie pt1.mkv`` + ``Movie pt2.mkv`` -> ``Movie.mkv``).

Pure functions, no I/O: `find_part_groups` works on file names only and `parts_compatible`
compares two already probed `MediaInfo` objects. Anything that does not form a clean group
(a lone part, a gap, a duplicate number) is simply not returned; nothing here raises on odd
names.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from video_cleaner.probe import MediaInfo

OUTPUT_SUFFIX = ".mkv"
MIN_PARTS = 2
MAX_PART_NUMBER = 99
FPS_TOLERANCE = 0.001

# A trailing part marker on the file stem: optional separators, pt/part/cd, optional
# separators, one or two ASCII digits. The marker must not continue a word ("Concept1" and
# "Apart 1" are not parts), hence the look-behind for a letter or digit (Unicode-aware, so an
# accented letter also counts). Matched with fullmatch: "$" would also match before a
# trailing newline. [0-9], not \d, which accepts other scripts' digits.
_PART_MARKER = re.compile(
    r"(?P<base>.*?)[ ._-]*(?<![^\W_])(?:pt|part|cd)[ ._-]*(?P<number>[0-9]{1,2})",
    re.IGNORECASE | re.DOTALL,
)


class _Part(NamedTuple):
    """One file of a candidate group: its part number, the name before the marker, its path."""

    number: int
    base: str
    path: Path


@dataclass(frozen=True)
class PartGroup:
    """The parts of one movie, in part order, and the name of the combined output file."""

    output_name: str
    parts: tuple[Path, ...]


def _split_marker(path: Path) -> tuple[str, int] | None:
    """Return (base name, part number) if *path* ends in a part marker, else None."""
    match = _PART_MARKER.fullmatch(path.stem)
    if match is None:
        return None
    base = match.group("base")
    number = int(match.group("number"))
    if not base or not 1 <= number <= MAX_PART_NUMBER:
        return None
    return base, number


def find_part_groups(files: Sequence[Path]) -> list[PartGroup]:
    """Group files of one folder whose names differ only in a trailing part marker.

    Markers are ``pt``, ``part`` and ``cd`` (case-insensitive) with a number 1-99 and optional
    space, dot, dash or underscore separators around them. A group needs at least two parts,
    numbered from 1 without gaps, each number exactly once; everything else is left out.
    Groups come back ordered by folder, then output name.
    """
    # Files of one folder whose names agree up to the part marker (compared ignoring case).
    candidates: dict[tuple[Path, str], list[_Part]] = defaultdict(list)
    for path in files:
        split = _split_marker(path)
        if split is not None:
            base, number = split
            candidates[(path.parent, base.casefold())].append(_Part(number, base, path))

    groups: list[PartGroup] = []
    for members in candidates.values():
        members.sort(key=lambda part: (part.number, str(part.path)))
        if len(members) < MIN_PARTS:
            continue
        if [part.number for part in members] != list(range(1, len(members) + 1)):
            continue  # a gap or a duplicate number: not a clean group
        groups.append(
            PartGroup(
                output_name=members[0].base + OUTPUT_SUFFIX,
                parts=tuple(part.path for part in members),
            )
        )
    groups.sort(key=lambda group: (str(group.parts[0].parent), group.output_name))
    return groups


def parts_compatible(a: MediaInfo, b: MediaInfo) -> bool:
    """True if two parts can be joined without re-encoding around a layout change.

    Both need a video stream with the same codec, frame size, pixel format and frame rate,
    and the same audio and subtitle layout (codec and language, stream by stream, in order).
    """
    if a.video is None or b.video is None:
        return False
    va, vb = a.video, b.video
    if (va.codec, va.width, va.height, va.pix_fmt) != (vb.codec, vb.width, vb.height, vb.pix_fmt):
        return False
    if abs(va.fps - vb.fps) > FPS_TOLERANCE:
        return False
    if [(s.codec, s.language) for s in a.audio] != [(s.codec, s.language) for s in b.audio]:
        return False
    return [(s.codec, s.language) for s in a.subtitles] == [
        (s.codec, s.language) for s in b.subtitles
    ]
