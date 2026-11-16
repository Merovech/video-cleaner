"""One definition of "the same path" and "inside a folder", shared by every module.

Paths are compared by `path_key`: the real path (symbolic links, ``..`` and, on Windows, a
mapped network drive resolved to the share it maps) with the platform's case and separator
folding. A path that cannot be resolved (an unreachable share, an embedded NUL) falls back to
its absolute spelling rather than raising.
"""

from __future__ import annotations

import os
from pathlib import Path

_resolve = os.path.realpath


def path_key(path: Path | str) -> str:
    """Comparable form of *path*: resolved, absolute, case- and separator-folded."""
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
