"""The subtitle cache folder: its guards, its file names, and the folder listings around it.

Nothing here reads a subtitle. The names are built from a hash and a number only, so no text
from a server or a subtitle file ever reaches a path. The guards raise `ValueError` for a
configuration that could write into the library or into a video's own folder tree.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from video_cleaner.atomicio import read_bytes_capped
from video_cleaner.config import Settings
from video_cleaner.paths import is_under
from video_cleaner.state import SubtitleRecord
from video_cleaner.subtitles.align import WORK_PREFIX
from video_cleaner.subtitles.local import is_link_like

_log = logging.getLogger("video_cleaner.subtitles.cache")

SIDECAR_EXTENSIONS = (".srt", ".ass", ".vtt")
_SIDECAR_LANGUAGES = ("", ".en", ".eng")
MAX_SIDECARS = 3
_MAX_ENTRIES_EXAMINED = 100_000
_STAGED_SUFFIX = ".dl.srt"
_ALIGNED_SUFFIX = ".aligned.srt"
_PROBE_PREFIX = ".probe-"


@dataclass(frozen=True)
class CacheNames:
    """The cache files of one candidate: the finished subtitle, the download kept while it is
    aligned (`.dl.srt`) and the aligner's output (`.aligned.srt`)."""

    final: Path
    staged: Path
    aligned: Path


def cache_key(sub: SubtitleRecord) -> str:
    """16 hex digits identifying one file by its path, size and mtime (the record's key)."""
    text = f"{sub.path}|{sub.size}|{sub.mtime}"
    return hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def cache_names(cache_dir: Path, key: str, provider: str, ident: str) -> CacheNames:
    """The names for one candidate; *provider* is a constant and *ident* a hex hash or a number."""
    stem = f"{key}-{provider}-{ident}"
    return CacheNames(
        cache_dir / f"{stem}.srt",
        cache_dir / f"{stem}{_STAGED_SUFFIX}",
        cache_dir / f"{stem}{_ALIGNED_SUFFIX}",
    )


def discard(*paths: Path) -> None:
    """Remove the resolver's own intermediate files; a failure is not worth an error."""
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            _log.debug("could not remove %r", path.name)


def check_cache_dir(settings: Settings, video: Path) -> Path:
    """The cache folder, after checking it is absolute and apart from the library and the video.

    A violation is a configuration bug, so it raises `ValueError` before anything is written.
    """
    cache = settings.subtitle_cache_dir
    if not cache.is_absolute():
        raise ValueError(f"subtitle_cache_dir must be an absolute path, not {str(cache)!r}")
    root = settings.library_root
    if is_under(cache, root) or is_under(root, cache):
        raise ValueError(
            f"subtitle_cache_dir {str(cache)!r} overlaps the library {str(root)!r}; "
            "nothing is ever written to the library"
        )
    if is_under(cache, video) or is_under(video, cache):
        raise ValueError(
            f"subtitle_cache_dir {str(cache)!r} overlaps the video {str(video)!r}; "
            "nothing is ever written beside a video"
        )
    return cache


def probe_writable(cache_dir: Path) -> None:
    """Create *cache_dir* and write and remove a tiny uniquely named file in it.

    Raises `OSError` when that cannot be done (disk full, no permission): better to learn it
    before a download is spent than after.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=_PROBE_PREFIX, suffix=".tmp", dir=cache_dir)
    try:
        os.write(fd, b"x")
    finally:
        os.close(fd)
        discard(Path(name))


def sidecar_candidates(path: Path) -> list[Path]:
    """Subtitle files that sit beside *path*, named like it: same folder, same stem (any case),
    an optional ".en" or ".eng", and an extension from `SIDECAR_EXTENSIONS`.

    Sorted by name, at most `MAX_SIDECARS`, regular files only (never a link). The folder is
    only listed; one that cannot be listed gives an empty list.
    """
    stem = path.stem.lower()
    wanted = {f"{stem}{lang}{ext}" for lang in _SIDECAR_LANGUAGES for ext in SIDECAR_EXTENSIONS}
    found: list[str] = []
    try:
        with os.scandir(path.parent) as entries:
            for entry in itertools.islice(entries, _MAX_ENTRIES_EXAMINED):
                if entry.name.lower() not in wanted:
                    continue
                try:
                    if is_link_like(entry) or not entry.is_file(follow_symlinks=False):
                        continue
                except OSError:
                    continue
                found.append(entry.name)
    except OSError:
        return []
    return [path.parent / name for name in sorted(found)[:MAX_SIDECARS]]


def sweep_align_workdirs(cache_dir: Path) -> int:
    """Remove the work folders ffsubsync runs left in *cache_dir* (killed runs) and count them.

    Only direct children that are real folders named with the align work prefix are removed;
    files, links, junctions and everything else stay. A folder that cannot be removed is logged
    and skipped.
    """
    folders: list[str] = []
    try:
        with os.scandir(cache_dir) as entries:
            for entry in itertools.islice(entries, _MAX_ENTRIES_EXAMINED):
                try:
                    if (
                        entry.name.startswith(WORK_PREFIX)
                        and entry.is_dir(follow_symlinks=False)
                        and not is_link_like(entry)
                    ):
                        folders.append(entry.path)
                except OSError:
                    continue
    except OSError:
        return 0
    removed = 0
    for folder in folders:
        try:
            shutil.rmtree(folder)
        except OSError as exc:
            _log.warning("could not remove an old align work folder: %s", type(exc).__name__)
            continue
        removed += 1
    return removed


def cached_result_is_valid(sub: SubtitleRecord, cache_dir: Path, max_bytes: int) -> bool:
    """Whether a FOUND record's cache file is still there, inside the cache, and unchanged."""
    path = sub.cache_path
    if path is None or not sub.sha256 or not path.is_absolute() or not is_under(path, cache_dir):
        return False
    try:
        data = read_bytes_capped(path, max_bytes)
    except OSError:
        return False
    return data is not None and hashlib.sha256(data).hexdigest() == sub.sha256
