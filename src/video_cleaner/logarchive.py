"""Run-log retention: old logs are archived into zips, never deleted.

`archive_old_logs` moves ``run-*.log`` files out of the log folder once they are old or the
folder is large. Each goes into a zip named for the month of its last write, under an
``archive`` folder in the log folder, and is removed only after its copy has been read back
and verified.
"""

from __future__ import annotations

import time
import zipfile
import zlib
from pathlib import Path

from video_cleaner.paths import name_key

LOG_GLOB = "run-*.log"
ARCHIVE_DIR_NAME = "archive"  # under the log folder; holds the zips of archived run logs


def _crc32(path: Path) -> int:
    """CRC-32 of a file, read in chunks (a log can be large)."""
    crc = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            crc = zlib.crc32(chunk, crc)
    return crc


def _stored_intact(archive: Path, name: str, crc: int, size: int) -> bool:
    """True if *archive* holds an entry *name* of *size* bytes with checksum *crc*.

    Reads the entry back to its end, which makes zipfile verify the stored checksum too.
    """
    try:
        with zipfile.ZipFile(archive) as zf:
            info = zf.getinfo(name)
            if (info.CRC, info.file_size) != (crc, size):
                return False
            with zf.open(info) as stored:
                while stored.read(1 << 20):
                    pass
    except (KeyError, OSError, zipfile.BadZipFile):
        return False
    return True


def _archive_log(log: Path, archive_dir: Path) -> bool:
    """Store *log* in its month's zip in *archive_dir*; True once it is there and verified.

    The zip is named for the month of the log's modification time (UTC), so no single archive
    grows without end and a damaged one costs a month, not everything. A log already stored
    (a failed earlier removal) is not stored twice; a different log of the same name gets a
    numbered entry name.
    """
    stat = log.stat()
    crc = _crc32(log)
    archive = archive_dir / time.strftime("run-logs-%Y-%m.zip", time.gmtime(stat.st_mtime))
    try:
        archive_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "a", zipfile.ZIP_DEFLATED) as zf:
            taken = {info.filename: info for info in zf.infolist()}
            name, number = log.name, 1
            while name in taken and (taken[name].CRC, taken[name].file_size) != (crc, stat.st_size):
                number += 1
                name = f"{log.stem}-{number}{log.suffix}"
            if name not in taken:
                zf.write(log, name)
    except (OSError, ValueError, zipfile.BadZipFile):  # ValueError: a date before 1980
        return False
    return _stored_intact(archive, name, crc, stat.st_size)


def archive_old_logs(
    log_dir: Path,
    *,
    keep_days: int = 30,
    max_bytes: int = 200 * 10**6,
    now: float | None = None,
    exclude: Path | None = None,
) -> list[Path]:
    """Move ``run-*.log`` files older than *keep_days*, then the oldest until the total
    size is at most *max_bytes*, into zip archives under ``<log_dir>/archive``.

    Nothing is deleted outright: a log is removed from the folder only after its copy in the
    archive has been read back and matches. Other files are never touched, nor is *exclude*
    (the current run's log). A log that cannot be archived or removed (for example one still
    open) stays where it is. Returns the paths moved to the archive; never raises for a
    missing directory.
    """
    current = time.time() if now is None else now
    cutoff = current - keep_days * 86_400
    try:
        skip = None if exclude is None else name_key(exclude)
        candidates = [p for p in log_dir.glob(LOG_GLOB) if p.is_file() and name_key(p) != skip]
        logs = sorted((p.stat().st_mtime, p.stat().st_size, p) for p in candidates)
    except OSError:
        return []

    archived: list[Path] = []
    kept: list[tuple[float, int, Path]] = []

    def move(path: Path) -> bool:
        if not _archive_log(path, log_dir / ARCHIVE_DIR_NAME):
            return False
        try:
            path.unlink()
        except OSError:
            return False
        archived.append(path)
        return True

    for mtime, size, path in logs:
        if not (mtime < cutoff and move(path)):
            kept.append((mtime, size, path))
    total = sum(size for _, size, _ in kept)
    for _, size, path in kept:  # oldest first
        if total <= max_bytes:
            break
        if move(path):
            total -= size
    return archived
