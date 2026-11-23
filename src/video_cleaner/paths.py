"""One definition of "the same path" and "inside a folder", shared by every module.

Two keys, for two jobs:

* `name_key` is string-only (absolute spelling, ``..`` collapsed, the platform's case and
  separator folding; no I/O). It identifies a path in sets, dicts and state comparisons, where
  a run may evaluate it for every record and a NAS round trip per call would be O(n^2).
* `path_key` resolves the real path (symbolic links and, on Windows, a mapped network drive to
  the share it maps). It is for the safety guards only (`is_under`, `outputs_outside_library`),
  where an alias must not slip past. A path that cannot be resolved (an unreachable share, an
  embedded NUL) falls back to its absolute spelling rather than raising.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from video_cleaner.config import Settings

_resolve = os.path.realpath


class OutputInLibrary(Exception):
    """Something the tool writes, or cleans, would overlap the library root."""


def name_key(path: Path | str) -> str:
    """String-only identity of *path*: absolute, ``..`` collapsed, case- and separator-folded."""
    return os.path.normcase(os.path.abspath(path))


def name_under(path: Path | str, root: Path | str) -> bool:
    """String-only `is_under`: for selection by scope, never for a safety guard."""
    inner = name_key(path)
    outer = name_key(root).rstrip("\\/")
    return inner == outer or inner.startswith(outer + os.sep)


def path_key(path: Path | str) -> str:
    """Comparable form of *path*: resolved, absolute, case- and separator-folded (does I/O)."""
    try:
        resolved = _resolve(path)
    except (OSError, ValueError):
        resolved = os.path.abspath(path)
    return os.path.normcase(resolved)


def is_under(path: Path | str, root: Path | str) -> bool:
    """True if *path* is *root* or lies inside it (``library2`` is not inside ``library``)."""
    inner = path_key(path)
    outer = path_key(root).rstrip("\\/")
    return inner == outer or inner.startswith(outer + os.sep)


def _outputs(settings: Settings, extra: Sequence[tuple[str, Path]]) -> list[tuple[str, Path]]:
    return [
        ("scratch folder", settings.scratch_dir),
        ("completed folder", settings.completed_dir),
        ("state database", settings.state_db),
        ("log folder", settings.log_dir),
        ("subtitle cache", settings.subtitle_cache_dir),
        *extra,
    ]


def library_overlaps(
    settings: Settings,
    *,
    extra: Sequence[tuple[str, Path]] = (),
    roots: Sequence[Path] = (),
) -> list[tuple[str, str]]:
    """Every ``(label, problem)`` where an output overlaps a library root, in either direction.

    The outputs are the scratch and completed folders, the state database, the log folder, the
    subtitle cache and the labelled *extra* paths; the roots are ``settings.library_root`` and
    *roots* (a survey ``--root``). An output under a root would write to the library; a root
    inside an output could be written to (the completed folder mirrors library paths) or
    swept. Empty when safe.
    """
    found: list[tuple[str, str]] = []
    bases = list(dict.fromkeys((settings.library_root, *roots)))
    for label, target in _outputs(settings, extra):
        for base in bases:
            if is_under(target, base):
                problem = f"{label} {target} is under the library root {base}"
            elif is_under(base, target):
                problem = f"the library root {base} is inside the {label} {target}"
            else:
                continue
            found.append((label, f"{problem}; nothing is ever written to the library"))
            break
    return found


def outputs_outside_library(
    settings: Settings,
    *,
    extra: Sequence[tuple[str, Path]] = (),
    roots: Sequence[Path] = (),
) -> None:
    """Raise `OutputInLibrary` for the first problem `library_overlaps` finds."""
    found = library_overlaps(settings, extra=extra, roots=roots)
    if found:
        raise OutputInLibrary(found[0][1])
