"""Completed-folder layout and the manifest of what was produced from which originals.

The completed folder mirrors the library: a file at ``<library_root>/a/b/c.avi`` is delivered as
``<completed_dir>/a/b/c.mkv``. ``manifest.csv`` records, per original, the size and mtime it had
when it was processed, so a later check can tell that an original changed afterwards and the
completed file is stale (it must then never replace the original).

Everything here is path arithmetic and CSV; nothing touches the library except `check_unchanged`,
which only stats the originals.
"""

from __future__ import annotations

import contextlib
import csv
import math
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from video_cleaner.config import Settings

COMPLETED_SUFFIX = ".mkv"
MANIFEST_COLUMNS: tuple[str, ...] = (
    "original",
    "original_size",
    "original_mtime",
    "completed",
    "completed_size",
    "vmaf_mean",
    "vmaf_worst",
    "crf",
    "verified",
    "action",
)
# Columns that describe the completed file; every row of a multi-part entry repeats them.
_SHARED_COLUMNS = MANIFEST_COLUMNS[3:]
_TRUE = "true"
_FALSE = "false"


class ManifestError(ValueError):
    """The manifest cannot be written as asked, or an existing manifest is malformed."""


@dataclass(frozen=True)
class ManifestEntry:
    """One completed file and the original(s) it was made from.

    ``original``, ``original_sizes`` and ``original_mtimes`` are parallel: index *i* of each
    describes the same source file. A multi-part movie has several originals and one completed
    file.
    """

    original: tuple[Path, ...]
    original_sizes: tuple[int, ...]
    original_mtimes: tuple[float, ...]
    completed: Path
    completed_size: int
    vmaf_mean: float | None
    vmaf_worst: float | None
    crf: int | None
    verified: bool
    action: str

    def __post_init__(self) -> None:
        count = len(self.original)
        if count == 0:
            raise ValueError("a manifest entry needs at least one original")
        if len(self.original_sizes) != count:
            raise ValueError(
                f"original_sizes has {len(self.original_sizes)} value(s) for {count} original(s)"
            )
        if len(self.original_mtimes) != count:
            raise ValueError(
                f"original_mtimes has {len(self.original_mtimes)} value(s) for {count} original(s)"
            )


def _same(a: str, b: str) -> bool:
    return os.path.normcase(a) == os.path.normcase(b)


def _check_new_name(new_name: str) -> None:
    plain = (
        new_name not in ("", ".", "..")
        and "/" not in new_name
        and "\\" not in new_name
        # "C:name" would silently re-anchor the joined path on another drive.
        and not PureWindowsPath(new_name).drive
    )
    if not plain:
        raise ValueError(f"new_name must be a plain file name, got {new_name!r}")


def completed_path_for(settings: Settings, original: Path, *, new_name: str | None = None) -> Path:
    """Where the completed file for *original* goes: its relative folder under ``completed_dir``.

    The file name becomes *new_name* when given (the combined name of a multi-part movie),
    otherwise the original's stem plus ``.mkv``. Folder and file names keep the casing of
    *original*; only the comparison against ``library_root`` ignores case (on Windows).

    Raises `ValueError` if *original* is not a file path strictly inside ``library_root``, if it
    contains ``..`` (which could climb out of the completed folder), or if *new_name* is not a
    plain file name.
    """
    root_parts = settings.library_root.parts
    parts = original.parts
    inside = len(parts) > len(root_parts) and all(
        _same(mine, theirs) for mine, theirs in zip(parts, root_parts, strict=False)
    )
    if not inside:
        raise ValueError(f"{original} is not under the library root {settings.library_root}")
    relative = parts[len(root_parts) :]
    if ".." in relative:
        raise ValueError(f"{original} contains '..'; refusing to mirror it")
    if new_name is None:
        name = Path(relative[-1]).stem + COMPLETED_SUFFIX
    else:
        _check_new_name(new_name)
        name = new_name
    return settings.completed_dir.joinpath(*relative[:-1], name)


def check_unchanged(entry: ManifestEntry) -> bool:
    """True only if every original still exists as a file with the recorded size and mtime.

    A vanished, unreadable or replaced original is "changed"; this never raises for one.
    """
    for path, size, mtime in zip(
        entry.original, entry.original_sizes, entry.original_mtimes, strict=True
    ):
        try:
            st = os.stat(path)
        except (OSError, ValueError):
            return False
        if not stat.S_ISREG(st.st_mode) or st.st_size != size or st.st_mtime != mtime:
            return False
    return True


def _optional(value: float | int | None) -> str:
    return "" if value is None else repr(value)


def _rows(entries: Sequence[ManifestEntry]) -> list[list[str]]:
    rows: list[list[str]] = []
    seen: set[str] = set()
    for entry in entries:
        completed = str(entry.completed)
        key = os.path.normcase(completed)
        if key in seen:
            raise ManifestError(f"more than one entry for the completed file {completed}")
        seen.add(key)
        shared = [
            completed,
            str(entry.completed_size),
            _optional(entry.vmaf_mean),
            _optional(entry.vmaf_worst),
            _optional(entry.crf),
            _TRUE if entry.verified else _FALSE,
            entry.action,
        ]
        for path, size, mtime in zip(
            entry.original, entry.original_sizes, entry.original_mtimes, strict=True
        ):
            # repr() of a float is the shortest text that parses back to the same float.
            rows.append([str(path), str(size), repr(float(mtime)), *shared])
    for row in rows:
        for cell in row:
            try:
                cell.encode("utf-8")
            except UnicodeEncodeError:
                raise ManifestError(
                    f"cannot record {cell!r} in the manifest: not representable as UTF-8"
                ) from None
    return rows


def write_manifest(path: Path, entries: Sequence[ManifestEntry]) -> None:
    """Write the manifest CSV (UTF-8, no BOM), one row per original, replacing *path* atomically.

    A multi-part entry yields one row per original with the completed columns repeated. The file
    is written beside *path* and moved over it with `os.replace`, so a crash or a full disk
    leaves the previous manifest intact.
    """
    rows = _rows(entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(MANIFEST_COLUMNS)
            writer.writerows(rows)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def _parse_int(column: str, text: str) -> int:
    try:
        return int(text)
    except ValueError:
        raise ManifestError(f"{column}: expected an integer, got {text!r}") from None


def _parse_float(column: str, text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise ManifestError(f"{column}: expected a number, got {text!r}") from None
    if not math.isfinite(value):
        raise ManifestError(f"{column}: expected a finite number, got {text!r}")
    return value


def _parse_bool(column: str, text: str) -> bool:
    if text == _TRUE:
        return True
    if text == _FALSE:
        return False
    raise ManifestError(f"{column}: expected {_TRUE!r} or {_FALSE!r}, got {text!r}")


def _parse_path(column: str, text: str) -> Path:
    if not text:
        raise ManifestError(f"{column}: path must not be empty")
    return Path(text)


@dataclass
class _Group:
    """Rows collected so far for one completed file."""

    shared: tuple[str, ...]
    completed: Path
    completed_size: int
    vmaf_mean: float | None
    vmaf_worst: float | None
    crf: int | None
    verified: bool
    action: str
    originals: list[Path]
    sizes: list[int]
    mtimes: list[float]

    def entry(self) -> ManifestEntry:
        return ManifestEntry(
            original=tuple(self.originals),
            original_sizes=tuple(self.sizes),
            original_mtimes=tuple(self.mtimes),
            completed=self.completed,
            completed_size=self.completed_size,
            vmaf_mean=self.vmaf_mean,
            vmaf_worst=self.vmaf_worst,
            crf=self.crf,
            verified=self.verified,
            action=self.action,
        )


def _new_group(row: Mapping[str, str]) -> _Group:
    return _Group(
        shared=tuple(row[c] for c in _SHARED_COLUMNS),
        completed=_parse_path("completed", row["completed"]),
        completed_size=_parse_int("completed_size", row["completed_size"]),
        vmaf_mean=_parse_float("vmaf_mean", row["vmaf_mean"]) if row["vmaf_mean"] else None,
        vmaf_worst=_parse_float("vmaf_worst", row["vmaf_worst"]) if row["vmaf_worst"] else None,
        crf=_parse_int("crf", row["crf"]) if row["crf"] else None,
        verified=_parse_bool("verified", row["verified"]),
        action=row["action"],
        originals=[],
        sizes=[],
        mtimes=[],
    )


def read_manifest(path: Path) -> list[ManifestEntry]:
    """Read a manifest written by `write_manifest`, regrouping the rows by completed path.

    Entries come back in the order their first row appears; the originals of an entry keep row
    order. Raises `FileNotFoundError` if there is no manifest and `ManifestError` (naming the
    line) if it is malformed.
    """
    groups: dict[str, _Group] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, strict=True)
        try:
            header = reader.fieldnames or []
            missing = [c for c in MANIFEST_COLUMNS if c not in header]
            if missing:
                raise ManifestError(f"{path}: missing column(s): {', '.join(missing)}")
            for row in reader:
                line = reader.line_num
                try:
                    if None in row or any(row[c] is None for c in MANIFEST_COLUMNS):
                        raise ManifestError(f"expected {len(header)} fields")
                    original = _parse_path("original", row["original"])
                    size = _parse_int("original_size", row["original_size"])
                    mtime = _parse_float("original_mtime", row["original_mtime"])
                    key = os.path.normcase(row["completed"])
                    group = groups.get(key)
                    if group is None:
                        group = groups[key] = _new_group(row)
                    elif group.shared != tuple(row[c] for c in _SHARED_COLUMNS):
                        raise ManifestError(
                            f"rows for {row['completed']} disagree about the completed file"
                        )
                    group.originals.append(original)
                    group.sizes.append(size)
                    group.mtimes.append(mtime)
                except ManifestError as exc:
                    raise ManifestError(f"{path}, line {line}: {exc}") from None
        except csv.Error as exc:
            raise ManifestError(f"{path}, line {reader.line_num}: {exc}") from exc
        except UnicodeDecodeError as exc:
            raise ManifestError(f"{path}: not valid UTF-8: {exc}") from exc
    return [group.entry() for group in groups.values()]
