"""Building external-command argument lists out of named flag/value pairs.

A long command written as one flat list of strings hides which value belongs to which flag.
The command builders write each option as a tuple, ``("-crf", "20")`` or a bare ``("-an",)``,
one per line with a comment, and `flatten` turns them into the list ``subprocess`` needs.
"""

from __future__ import annotations

# A flag followed by its values, or a flag on its own.
Option = tuple[str, ...]

# ffmpeg: write machine-readable progress to stdout and suppress the human progress line.
PROGRESS_OPTIONS: tuple[Option, ...] = (("-progress", "pipe:1"), ("-nostats",))


def flatten(*options: Option) -> list[str]:
    """The argument list of *options*, in order."""
    return [part for option in options for part in option]
