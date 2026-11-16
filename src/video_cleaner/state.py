"""Resumable run state in SQLite.

One row per (first-part path, size, mtime). A file whose size or mtime changed gets a new row and
the old one becomes ``stale``, so a result computed from earlier bytes is never applied to newer
ones. All access goes through one connection guarded by a lock (the runner calls from several
threads); every statement is parameterised. Paths are stored as POSIX-style strings and returned
as `Path`.
"""

from __future__ import annotations

import contextlib
import json
import math
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Any

from video_cleaner.probe import MediaInfo

SCHEMA_VERSION = 2
_BUSY_TIMEOUT_S = 30.0


class StateError(Exception):
    """The state database cannot be used (closed, or written by a newer version)."""


class StaleRecord(StateError):
    """The (path, size, mtime) record named by the caller is not the live one any more."""


class UnrepresentablePath(StateError):
    """A path cannot be stored (for example a file name with undecodable bytes)."""


class Status(StrEnum):
    PENDING = "pending"
    QUEUED = "queued"
    COPYING = "copying"
    ENCODING = "encoding"
    MUXING = "muxing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    LEFT_ALONE = "left_alone"
    FLAGGED = "flagged"
    RETRY = "retry"
    STALE = "stale"


# decision -> status a freshly planned file starts in
_INITIAL_STATUS: dict[str, Status] = {
    "encode": Status.PENDING,
    "remux": Status.PENDING,
    "leave_alone": Status.LEFT_ALONE,
    "flag": Status.FLAGGED,
}


def _unstarted(status: Status, decision: str, note: str) -> bool:
    """Whether a re-plan of an unchanged file may still replace this record's plan.

    True for a STALE record (its source came back) and for a record that is still exactly as
    survey left it: initial status for its decision and no note. A LEFT_ALONE or FLAGGED status
    set by a run (low projected saving, guard or verify failure, collision) does not count, so a
    re-survey never sends the file through the same hours-long encode again.
    """
    if status is Status.STALE:
        return True
    return status is _INITIAL_STATUS.get(decision) and not note


# Statuses a run passes through; a record left in one of them means the run was interrupted.
IN_PROGRESS = (
    Status.QUEUED,
    Status.COPYING,
    Status.ENCODING,
    Status.MUXING,
    Status.VERIFYING,
)
_INTERRUPTED_NOTE = "reset after interrupted run"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime REAL NOT NULL,
    status TEXT NOT NULL,
    decision TEXT NOT NULL,
    reasons TEXT NOT NULL,
    est_saving_bytes INTEGER NOT NULL,
    parts TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL,
    movie_hash TEXT,
    UNIQUE (path, size, mtime)
);
CREATE INDEX IF NOT EXISTS files_status_saving ON files (status, est_saving_bytes DESC);
CREATE INDEX IF NOT EXISTS files_path ON files (path);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    line TEXT NOT NULL
);
"""

_PLAN_HASH_KEY = "plan_ack_hash"
_CALIBRATION_KEY = "calibration"
_CALIBRATION_SWEPT_KEY = "calibration_swept_at"
# The only text interpolated into SQL (the `nosec B608` lines) is this constant and runs of
# "?" placeholders; every value is bound as a parameter.
_COLUMNS = "path, size, mtime, status, decision, reasons, est_saving_bytes, parts, note, movie_hash"
# Schema upgrades, applied in order to a database at the version before each key.
_MIGRATIONS: dict[int, tuple[str, ...]] = {
    2: ("ALTER TABLE files ADD COLUMN movie_hash TEXT",),
}


@dataclass(frozen=True)
class FileRecord:
    path: Path
    size: int
    mtime: float
    status: Status
    decision: str
    reasons: tuple[str, ...]
    est_saving_bytes: int
    parts: tuple[Path, ...]
    note: str
    # OpenSubtitles hash of this record's file (for a multi-part group: of its first part; each
    # member part has its own record and hash). None when not computed or not computable.
    movie_hash: str | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _key(path: Path) -> str:
    """The stored form of *path*; refuses paths SQLite cannot encode, before any write."""
    text = path.as_posix()
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        raise UnrepresentablePath(
            f"path cannot be stored in the state database: {path!r}"
        ) from None
    return text


def _record(row: Sequence[Any]) -> FileRecord:
    return FileRecord(
        path=Path(row[0]),
        size=int(row[1]),
        mtime=float(row[2]),
        status=Status(row[3]),
        decision=str(row[4]),
        reasons=tuple(json.loads(row[5])),
        est_saving_bytes=int(row[6]),
        parts=tuple(Path(p) for p in json.loads(row[7])),
        note=str(row[8]),
        movie_hash=None if row[9] is None else str(row[9]),
    )


class State:
    """Thread-safe handle on the state database. Use `State.open`."""

    def __init__(self, conn: sqlite3.Connection, db_path: Path) -> None:
        self._conn: sqlite3.Connection | None = conn
        self._db_path = db_path
        self._lock = threading.RLock()

    @classmethod
    def open(cls, db_path: Path) -> State:
        """Open (creating if needed) the database at *db_path*, in WAL mode."""
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            db_path, timeout=_BUSY_TIMEOUT_S, isolation_level=None, check_same_thread=False
        )
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            cls._init_schema(conn)
        except BaseException:
            conn.close()
            raise
        return cls(conn, db_path)

    @staticmethod
    def _init_schema(conn: sqlite3.Connection) -> None:
        conn.execute("BEGIN IMMEDIATE")
        try:
            for statement in _SCHEMA.split(";"):
                if statement.strip():
                    conn.execute(statement)
            row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
            current = row[0]
            if current is None:
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
            elif current > SCHEMA_VERSION:
                raise StateError(
                    f"state database schema version {current} is newer than "
                    f"supported version {SCHEMA_VERSION}"
                )
            else:
                for version in range(current + 1, SCHEMA_VERSION + 1):
                    for statement in _MIGRATIONS[version]:
                        conn.execute(statement)
                    conn.execute("UPDATE schema_version SET version = ?", (version,))
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")

    # -- plumbing ---------------------------------------------------------------------------

    @contextmanager
    def _tx(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = self._conn
            if conn is None:
                raise StateError("state database is closed")
            if not write:
                yield conn
                return
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                # Inside the try: a failed COMMIT (disk full, I/O error) is rolled back too,
                # instead of leaving the handle stuck in an open transaction.
                conn.execute("COMMIT")
            except BaseException:
                with contextlib.suppress(sqlite3.Error):  # never mask the original error
                    conn.execute("ROLLBACK")
                raise

    @staticmethod
    def _event(conn: sqlite3.Connection, text: str) -> None:
        conn.execute("INSERT INTO events (line) VALUES (?)", (f"{_now()} {text}",))

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> State:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- files ------------------------------------------------------------------------------

    def upsert_file(
        self,
        info: MediaInfo,
        *,
        decision: str,
        reasons: Sequence[str],
        est_saving_bytes: int,
        parts: Sequence[Path] = (),
        movie_hash: str | None = None,
    ) -> None:
        """Record the plan for *info*, keyed on path + size + mtime.

        A new key creates a record (initial status from *decision*) and marks any other live
        record for the path ``stale``. Re-planning an unchanged key refreshes the plan only while
        the record is untouched since survey (see `_unstarted`); a record a run has advanced or
        decided on (including a run-set LEFT_ALONE/FLAGGED) is left alone. A *movie_hash* is
        stored on the record for that key whenever given, even on an advanced record that
        lacks one (a database surveyed before hashes existed); None never clears a stored hash.
        """
        initial = _INITIAL_STATUS.get(decision)
        if initial is None:
            raise ValueError(
                f"unknown decision {decision!r}; expected one of {sorted(_INITIAL_STATUS)}"
            )
        path = _key(info.path)
        reasons_json = json.dumps(list(reasons))
        parts_json = json.dumps([_key(p) for p in parts])
        with self._tx(write=True) as conn:
            existing = conn.execute(
                "SELECT id, status, decision, note FROM files "
                "WHERE path = ? AND size = ? AND mtime = ?",
                (path, info.size, info.mtime),
            ).fetchone()
            if existing is not None and movie_hash is not None:
                conn.execute(
                    "UPDATE files SET movie_hash = ? WHERE id = ?", (movie_hash, existing[0])
                )
            if existing is not None and not _unstarted(
                Status(existing[1]), str(existing[2]), str(existing[3])
            ):
                return
            superseded = conn.execute(
                "UPDATE files SET status = ?, updated_at = ? "
                "WHERE path = ? AND status != ? AND NOT (size = ? AND mtime = ?)",
                (Status.STALE.value, _now(), path, Status.STALE.value, info.size, info.mtime),
            ).rowcount
            if existing is None:
                conn.execute(
                    "INSERT INTO files (path, size, mtime, status, decision, reasons, "
                    "est_saving_bytes, parts, note, updated_at, movie_hash) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, '', ?, ?)",
                    (
                        path,
                        info.size,
                        info.mtime,
                        initial.value,
                        decision,
                        reasons_json,
                        est_saving_bytes,
                        parts_json,
                        _now(),
                        movie_hash,
                    ),
                )
                suffix = f"; superseded {superseded} stale record(s)" if superseded else ""
                self._event(conn, f"{path}: new ({decision}) -> {initial.value}{suffix}")
            else:
                conn.execute(
                    "UPDATE files SET status = ?, decision = ?, reasons = ?, est_saving_bytes = ?, "
                    "parts = ?, note = '', updated_at = ? WHERE id = ?",
                    (
                        initial.value,
                        decision,
                        reasons_json,
                        est_saving_bytes,
                        parts_json,
                        _now(),
                        existing[0],
                    ),
                )
                if Status(existing[1]) is Status.STALE:
                    self._event(conn, f"{path}: stale -> {initial.value} (source reverted)")

    def set_status(
        self,
        path: Path,
        status: Status,
        note: str = "",
        *,
        size: int | None = None,
        mtime: float | None = None,
    ) -> None:
        """Set the status and note of the live (non-stale) record for *path*.

        With *size* and *mtime* the update targets exactly that (path, size, mtime) record and
        raises `StaleRecord` if it is not the live one (superseded or unknown), changing nothing;
        use this for results computed from specific source bytes. Without them, the live record
        is updated. Raises KeyError when there is no live record at all.
        """
        key = _key(path)
        if (size is None) != (mtime is None):
            raise ValueError("size and mtime must be given together")
        with self._tx(write=True) as conn:
            if size is not None:
                row = conn.execute(
                    "SELECT id, status FROM files WHERE path = ? AND size = ? AND mtime = ?",
                    (key, size, mtime),
                ).fetchone()
                if row is None or Status(row[1]) is Status.STALE:
                    raise StaleRecord(
                        f"no live state record for {path} with size={size} mtime={mtime}"
                    )
            else:
                row = conn.execute(
                    "SELECT id, status FROM files WHERE path = ? AND status != ? "
                    "ORDER BY id DESC LIMIT 1",
                    (key, Status.STALE.value),
                ).fetchone()
                if row is None:
                    raise KeyError(f"no live state record for {path}")
            conn.execute(
                "UPDATE files SET status = ?, note = ?, updated_at = ? WHERE id = ?",
                (status.value, note, _now(), row[0]),
            )
            detail = f" ({note})" if note else ""
            self._event(conn, f"{key}: {row[1]} -> {status.value}{detail}")

    def requeue(self, path: Path) -> FileRecord:
        """Put the live record for *path* back to the status survey gave it, with no note.

        The deliberate way to retry a file a run left alone or flagged (a re-survey keeps such
        outcomes). Raises KeyError when there is no live record, and StateError for a completed
        record (its output is already delivered). Returns the updated record.
        """
        key = _key(path)
        with self._tx(write=True) as conn:
            row = conn.execute(
                "SELECT id, status, decision FROM files WHERE path = ? AND status != ? "
                "ORDER BY id DESC LIMIT 1",
                (key, Status.STALE.value),
            ).fetchone()
            if row is None:
                raise KeyError(f"no live state record for {path}")
            if Status(row[1]) is Status.COMPLETED:
                raise StateError(f"{path} is completed; nothing to retry")
            initial = _INITIAL_STATUS[str(row[2])]
            conn.execute(
                "UPDATE files SET status = ?, note = '', updated_at = ? WHERE id = ?",
                (initial.value, _now(), row[0]),
            )
            self._event(conn, f"{key}: {row[1]} -> {initial.value} (requeued)")
            updated = conn.execute(
                f"SELECT {_COLUMNS} FROM files WHERE id = ?",  # noqa: S608  # nosec B608
                (row[0],),
            ).fetchone()
        return _record(updated)

    def records(self, *statuses: Status) -> list[FileRecord]:
        """Records with any of *statuses*, ordered by path (then age).

        With no statuses: every live record. STALE records appear only when requested.
        """
        with self._tx() as conn:
            if statuses:
                marks = ", ".join("?" for _ in statuses)
                rows = conn.execute(
                    f"SELECT {_COLUMNS} FROM files WHERE status IN ({marks}) "  # noqa: S608  # nosec B608
                    "ORDER BY path, id",
                    tuple(s.value for s in statuses),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT {_COLUMNS} FROM files WHERE status != ? "  # noqa: S608  # nosec B608
                    "ORDER BY path, id",
                    (Status.STALE.value,),
                ).fetchall()
        return [_record(r) for r in rows]

    def reset_in_progress(self, to: Status = Status.RETRY) -> int:
        """Move every record a killed run left in an in-progress status to *to*.

        One transaction, one event per record; returns how many were moved.
        """
        marks = ", ".join("?" for _ in IN_PROGRESS)
        with self._tx(write=True) as conn:
            rows = conn.execute(
                f"SELECT id, path, status FROM files WHERE status IN ({marks}) "  # noqa: S608  # nosec B608
                "ORDER BY path, id",
                tuple(s.value for s in IN_PROGRESS),
            ).fetchall()
            for row_id, path, old in rows:
                conn.execute(
                    "UPDATE files SET status = ?, note = ?, updated_at = ? WHERE id = ?",
                    (to.value, _INTERRUPTED_NOTE, _now(), row_id),
                )
                self._event(conn, f"{path}: {old} -> {to.value} ({_INTERRUPTED_NOTE})")
        return len(rows)

    def get(self, path: Path) -> FileRecord | None:
        """The live record for *path*, or the most recent stale one if none is live."""
        with self._tx() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM files WHERE path = ? "  # noqa: S608  # nosec B608
                "ORDER BY (status = ?), id DESC LIMIT 1",
                (_key(path), Status.STALE.value),
            ).fetchone()
        return None if row is None else _record(row)

    def _next_with_status(self, status: Status) -> FileRecord | None:
        with self._tx() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM files WHERE status = ? "  # noqa: S608  # nosec B608
                "ORDER BY est_saving_bytes DESC, id ASC LIMIT 1",
                (status.value,),
            ).fetchone()
        return None if row is None else _record(row)

    def next_pending(self) -> FileRecord | None:
        """The PENDING record with the largest estimated saving (RETRY is `next_retry`)."""
        return self._next_with_status(Status.PENDING)

    def next_retry(self) -> FileRecord | None:
        """The RETRY record with the largest estimated saving."""
        return self._next_with_status(Status.RETRY)

    def counts(self) -> dict[Status, int]:
        """Number of records per status, with every status present (zero when none)."""
        with self._tx() as conn:
            rows = conn.execute("SELECT status, COUNT(*) FROM files GROUP BY status").fetchall()
        result = dict.fromkeys(Status, 0)
        for status, n in rows:
            result[Status(status)] = int(n)
        return result

    # -- plan acknowledgement and events ----------------------------------------------------

    def acknowledge_plan(self, plan_hash: str) -> None:
        with self._tx(write=True) as conn:
            self._set_meta(conn, _PLAN_HASH_KEY, plan_hash)
            self._event(conn, f"plan acknowledged: {plan_hash}")

    def plan_acknowledged(self) -> str | None:
        return self._meta(_PLAN_HASH_KEY)

    def _meta(self, key: str) -> str | None:
        with self._tx() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def _set_meta(self, conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def approve_calibration(self, vmaf_target: float, crf_max: int) -> None:
        """Record the calibration Ari approved: the VMAF target and CRF ceiling a run uses.

        Stored like the plan acknowledgement (one meta value, one event line). Range checks
        against the configuration are the caller's job; this only refuses values that are not
        a finite number and an integer.
        """
        if isinstance(vmaf_target, bool) or not isinstance(vmaf_target, int | float):
            raise ValueError(f"vmaf_target must be a number, got {vmaf_target!r}")
        if not math.isfinite(vmaf_target):
            raise ValueError(f"vmaf_target must be a finite number, got {vmaf_target!r}")
        if type(crf_max) is not int:
            raise ValueError(f"crf_max must be an int, got {crf_max!r}")
        value = json.dumps({"vmaf_target": float(vmaf_target), "crf_max": crf_max})
        with self._tx(write=True) as conn:
            self._set_meta(conn, _CALIBRATION_KEY, value)
            self._event(
                conn, f"calibration approved: vmaf_target={float(vmaf_target)} crf_max={crf_max}"
            )

    def record_calibration_sweep(self) -> None:
        """Note (no event line) that a calibration sweep finished; approval requires one."""
        with self._tx(write=True) as conn:
            self._set_meta(conn, _CALIBRATION_SWEPT_KEY, _now())

    def calibration_swept(self) -> str | None:
        """When the last calibration sweep finished (ISO timestamp), or None if never."""
        return self._meta(_CALIBRATION_SWEPT_KEY)

    def calibration(self) -> tuple[float, int] | None:
        """The approved ``(vmaf_target, crf_max)``, or None before `approve_calibration`."""
        raw = self._meta(_CALIBRATION_KEY)
        if raw is None:
            return None
        data = json.loads(raw)
        return float(data["vmaf_target"]), int(data["crf_max"])

    def events(self, limit: int = 100) -> list[str]:
        """The most recent event lines, newest first, each prefixed with an ISO timestamp."""
        if limit <= 0:
            return []
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT line FROM events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [str(r[0]) for r in rows]
