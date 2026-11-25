"""Check a FOUND subtitle's cache file and stage a private copy of it for mkvmerge.

The state database says where the cached subtitle is and what its SHA-256 was when the resolver
stored it. Before a run copies any video that subtitle must still be there and unchanged
(`check_cached_subtitle`). The file mkvmerge reads is not the cache file but a copy made in the
file's scratch work folder, hashed again after the copy (`stage_subtitle`), so nothing can change
it between the last check and the mux. Both raise `SubtitleCacheStale` for any problem with the
cache file, and the caller then sends the record back to the resolver.

Nothing in the cache folder is written, moved or removed here.
"""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

from video_cleaner.atomicio import read_bytes_capped
from video_cleaner.config import Settings
from video_cleaner.mux import SubtitleInput
from video_cleaner.paths import is_under
from video_cleaner.state import SubtitleRecord

STAGED_NAME = "subtitle.srt"


class SubtitleCacheStale(Exception):
    """The cache file of a FOUND record is missing, outside the cache, or no longer what was stored.

    The message says which, without quoting any file content.
    """


def _cache_file(sub: SubtitleRecord, settings: Settings) -> Path:
    """The cache file of *sub* after the location checks, before its bytes are read."""
    path = sub.cache_path
    if path is None or not sub.sha256:
        raise SubtitleCacheStale("the record has no cache file or hash")
    if not path.is_absolute():
        raise SubtitleCacheStale("the cache path is not absolute")
    if not is_under(path, settings.subtitle_cache_dir):
        raise SubtitleCacheStale("the cache file is outside the subtitle cache folder")
    try:
        st = os.lstat(path)
    except OSError:
        raise SubtitleCacheStale("the cache file is missing or unreadable") from None
    attributes: int = getattr(st, "st_file_attributes", 0)
    if stat.S_ISLNK(st.st_mode) or attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        raise SubtitleCacheStale("the cache file is a link")
    if not stat.S_ISREG(st.st_mode):
        raise SubtitleCacheStale("the cache path is not a regular file")
    if st.st_size > settings.subtitle_max_bytes:
        raise SubtitleCacheStale(
            f"the cache file is larger than the limit of {settings.subtitle_max_bytes} bytes"
        )
    return path


def _read_cache(sub: SubtitleRecord, settings: Settings) -> bytes:
    """The bytes of the cache file when they hash to the stored value."""
    path = _cache_file(sub, settings)
    try:
        data = read_bytes_capped(path, settings.subtitle_max_bytes)
    except OSError:
        raise SubtitleCacheStale("the cache file is missing or unreadable") from None
    if data is None or hashlib.sha256(data).hexdigest() != sub.sha256:
        raise SubtitleCacheStale("the cache file changed since it was stored")
    return data


def check_cached_subtitle(sub: SubtitleRecord, settings: Settings) -> None:
    """Raise `SubtitleCacheStale` unless the cache file of the FOUND record *sub* is intact.

    Intact means: an absolute path inside ``settings.subtitle_cache_dir`` (links resolved), a
    regular file that is not a link, at most ``settings.subtitle_max_bytes`` long, with the
    stored SHA-256.
    """
    _read_cache(sub, settings)


def stage_subtitle(sub: SubtitleRecord, settings: Settings, workdir: Path) -> SubtitleInput:
    """Copy the cache file of *sub* to ``workdir/subtitle.srt`` and return it as a mux input.

    The cache bytes are checked again (location, size and hash, as `check_cached_subtitle`),
    written to a new file that must not exist yet, and read back from that file and hashed once
    more. A mismatch removes the copy and raises `SubtitleCacheStale`. The track name is the
    record's provenance; `ValueError` for one that `SubtitleInput` refuses, before anything is
    written. A failure to write the copy (full disk) is an `OSError` for the caller.
    """
    target = workdir / STAGED_NAME
    result = SubtitleInput(path=target, title=sub.provenance)
    data = _read_cache(sub, settings)
    with target.open("xb") as handle:
        handle.write(data)
    try:
        copy = read_bytes_capped(target, settings.subtitle_max_bytes)
        if copy is None or hashlib.sha256(copy).hexdigest() != sub.sha256:
            raise SubtitleCacheStale("the staged copy does not match the stored hash")
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return result
