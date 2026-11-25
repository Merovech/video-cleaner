"""OpenSubtitles movie hash, captured during the survey while the originals still exist.

The hash is the file size plus the sum of the 64-bit little-endian words of the first and the
last 64 KiB, modulo 2**64, written as 16 lower-case hex digits. It identifies a release for
subtitle lookups; after the user replaces an original with its re-encode it can never be
computed again, so `survey` stores it in the state database.

Read-only: the file is opened for reading only, and only 128 KiB of it are read.
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path

_log = logging.getLogger("video_cleaner.moviehash")

CHUNK = 64 * 1024
# The algorithm needs a full head and a full tail chunk; OpenSubtitles refuses smaller files,
# so a hash for one would never match anything. Such a file gets no hash.
MIN_SIZE = 2 * CHUNK
_MASK = 2**64 - 1
_WORDS = struct.Struct(f"<{CHUNK // 8}Q")


def movie_hash(path: Path) -> str | None:
    """The OpenSubtitles hash of *path*, or None.

    None for a file smaller than `MIN_SIZE` (debug log) and when the file cannot be read in
    full (missing, locked, shrunk while reading; logged as a warning). Never raises OSError.
    """
    try:
        size = path.stat().st_size
        if size < MIN_SIZE:
            _log.debug("no movie hash for %s: %d bytes is below %d", path, size, MIN_SIZE)
            return None
        with path.open("rb") as fh:
            head = fh.read(CHUNK)
            fh.seek(size - CHUNK)
            tail = fh.read(CHUNK)
    except (OSError, ValueError) as exc:
        _log.warning("no movie hash for %s: %s", path, exc)
        return None
    if len(head) != CHUNK or len(tail) != CHUNK:
        _log.warning("no movie hash for %s: the file changed size while it was read", path)
        return None
    total = size + sum(_WORDS.unpack(head)) + sum(_WORDS.unpack(tail))
    return f"{total & _MASK:016x}"
