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
from video_cleaner.subtitles.models import QuotaInfo

SCHEMA_VERSION = 4
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


class SubtitleStatus(StrEnum):
    WANTED = "wanted"
    FOUND = "found"
    NOT_NEEDED = "not_needed"
    UNAVAILABLE = "unavailable"
    UNCLEAR = "unclear"


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

# One row per (first-part path, size, mtime), like `files`, so an original that was overwritten
# starts with a fresh subtitle outcome. Created for a new database by `_SCHEMA` and added to an
# older one by migration 4; hence "IF NOT EXISTS".
_SUBTITLES_TABLE = """
CREATE TABLE IF NOT EXISTS subtitles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime REAL NOT NULL,
    status TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT '',
    cache_path TEXT,
    sha256 TEXT NOT NULL DEFAULT '',
    provenance TEXT NOT NULL DEFAULT '',
    hash_exact INTEGER NOT NULL DEFAULT 0,
    english_audio INTEGER NOT NULL DEFAULT 0,
    reason TEXT NOT NULL DEFAULT '',
    tried_ids TEXT NOT NULL DEFAULT '[]',
    attempts INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    UNIQUE (path, size, mtime)
)
"""
_SUBTITLES_INDEX = "CREATE INDEX IF NOT EXISTS subtitles_status ON subtitles (status)"

_SCHEMA = (
    """
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
    attempts INTEGER NOT NULL DEFAULT 0,
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
    + _SUBTITLES_TABLE
    + ";\n"
    + _SUBTITLES_INDEX
    + ";\n"
)

_PLAN_HASH_KEY = "plan_ack_hash"
_QUOTA_REMAINING_KEY = "subtitle_quota_remaining"
_QUOTA_RESET_KEY = "subtitle_quota_reset_at"
_CALIBRATION_KEY = "calibration"
_CALIBRATION_SWEPT_KEY = "calibration_swept_at"
# The only text interpolated into SQL (the `nosec B608` lines) is this constant and runs of
# "?" placeholders; every value is bound as a parameter.
_COLUMNS = (
    "path, size, mtime, status, decision, reasons, est_saving_bytes, parts, note, movie_hash, "
    "attempts"
)
# Schema upgrades, applied in order to a database at the version before each key.
_MIGRATIONS: dict[int, tuple[str, ...]] = {
    2: ("ALTER TABLE files ADD COLUMN movie_hash TEXT",),
    3: ("ALTER TABLE files ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0",),
    4: (_SUBTITLES_TABLE, _SUBTITLES_INDEX),
}
_SUB_COLUMNS = (
    "path, size, mtime, status, provider, cache_path, sha256, provenance, hash_exact, "
    "english_audio, reason, tried_ids, attempts"
)
_SUB_COLUMNS_S = ", ".join(f"s.{c.strip()}" for c in _SUB_COLUMNS.split(","))


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
    # Runs that ended for this file in a RETRY it caused (tool failure, I/O error).
    attempts: int = 0


@dataclass(frozen=True)
class SubtitleRecord:
    """The subtitle outcome for one file, keyed like its `FileRecord` (path, size, mtime)."""

    path: Path
    size: int
    mtime: float
    status: SubtitleStatus
    provider: str = ""  # "embedded" | "sidecar" | "local" | "opensubtitles"
    cache_path: Path | None = None
    sha256: str = ""
    provenance: str = ""  # the track title to mux
    hash_exact: bool = False
    english_audio: bool = False  # the video has an English audio track
    reason: str = ""
    # Provider file ids already downloaded and rejected for this file, never tried again.
    tried_ids: tuple[int, ...] = ()
    attempts: int = 0


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
        attempts=int(row[10]),
    )


def _tried_ids(raw: object) -> tuple[int, ...]:
    """The ids in a stored ``tried_ids`` value; anything but a JSON array of non-negative ints
    reads as empty, so a damaged row costs a repeat download at worst and never an error."""
    try:
        data = json.loads(str(raw))
    except ValueError:
        return ()
    if not isinstance(data, list):
        return ()
    ids: list[int] = []
    for item in data:
        if type(item) is not int or item < 0:
            return ()
        ids.append(item)
    return tuple(ids)


def _subtitle_record(row: Sequence[Any]) -> SubtitleRecord:
    return SubtitleRecord(
        path=Path(row[0]),
        size=int(row[1]),
        mtime=float(row[2]),
        status=SubtitleStatus(row[3]),
        provider=str(row[4]),
        cache_path=None if row[5] is None else Path(row[5]),
        sha256=str(row[6]),
        provenance=str(row[7]),
        hash_exact=bool(row[8]),
        english_audio=bool(row[9]),
        reason=str(row[10]),
        tried_ids=_tried_ids(row[11]),
        attempts=int(row[12]),
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
        count_attempt: bool = False,
    ) -> None:
        """Set the status and note of the live (non-stale) record for *path*.

        *count_attempt* adds one to the record's ``attempts`` (a run that failed on this file).

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
                "UPDATE files SET status = ?, note = ?, updated_at = ?, "
                "attempts = attempts + ? WHERE id = ?",
                (status.value, note, _now(), int(count_attempt), row[0]),
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
                "UPDATE files SET status = ?, note = '', attempts = 0, updated_at = ? WHERE id = ?",
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

    def counts(self) -> dict[Status, int]:
        """Number of records per status, with every status present (zero when none)."""
        with self._tx() as conn:
            rows = conn.execute("SELECT status, COUNT(*) FROM files GROUP BY status").fetchall()
        result = dict.fromkeys(Status, 0)
        for status, n in rows:
            result[Status(status)] = int(n)
        return result

    # -- subtitles --------------------------------------------------------------------------

    def subtitle(self, path: Path, size: int, mtime: float) -> SubtitleRecord | None:
        """The subtitle record for exactly this (path, size, mtime), or None."""
        key = _key(path)
        with self._tx() as conn:
            row = self._subtitle_row(conn, key, size, mtime)
        return None if row is None else _subtitle_record(row)

    @staticmethod
    def _subtitle_row(
        conn: sqlite3.Connection, key: str, size: int, mtime: float
    ) -> Sequence[Any] | None:
        row: Sequence[Any] | None = conn.execute(
            f"SELECT {_SUB_COLUMNS} FROM subtitles "  # noqa: S608  # nosec B608
            "WHERE path = ? AND size = ? AND mtime = ?",
            (key, size, mtime),
        ).fetchone()
        return row

    def set_subtitle(self, rec: SubtitleRecord) -> None:
        """Store *rec*, replacing any record for its (path, size, mtime), in one transaction.

        A FOUND record is never downgraded to WANTED (a stale writer cannot undo a result);
        that write changes nothing. Every other change goes through, including FOUND to
        another outcome. Raises ValueError for a *tried_ids* entry that is not a
        non-negative int.
        """
        key = _key(rec.path)
        cache = None if rec.cache_path is None else _key(rec.cache_path)
        if any(type(i) is not int or i < 0 for i in rec.tried_ids):
            raise ValueError(f"tried_ids must be non-negative ints, got {rec.tried_ids!r}")
        tried = json.dumps(list(rec.tried_ids))
        with self._tx(write=True) as conn:
            existing = self._subtitle_row(conn, key, rec.size, rec.mtime)
            old = None if existing is None else SubtitleStatus(existing[3])
            if old is SubtitleStatus.FOUND and rec.status is SubtitleStatus.WANTED:
                return
            conn.execute(
                "INSERT INTO subtitles (path, size, mtime, status, provider, cache_path, sha256, "
                "provenance, hash_exact, english_audio, reason, tried_ids, attempts, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(path, size, mtime) DO UPDATE SET status = excluded.status, "
                "provider = excluded.provider, cache_path = excluded.cache_path, "
                "sha256 = excluded.sha256, provenance = excluded.provenance, "
                "hash_exact = excluded.hash_exact, english_audio = excluded.english_audio, "
                "reason = excluded.reason, tried_ids = excluded.tried_ids, "
                "attempts = excluded.attempts, updated_at = excluded.updated_at",
                (
                    key,
                    rec.size,
                    rec.mtime,
                    rec.status.value,
                    rec.provider,
                    cache,
                    rec.sha256,
                    rec.provenance,
                    int(rec.hash_exact),
                    int(rec.english_audio),
                    rec.reason,
                    tried,
                    rec.attempts,
                    _now(),
                ),
            )
            if old is not rec.status:
                self._event(
                    conn,
                    f"{key}: subtitle {'new' if old is None else old.value} -> {rec.status.value}",
                )

    def ensure_wanted(
        self, path: Path, size: int, mtime: float, *, english_audio: bool
    ) -> SubtitleRecord:
        """The subtitle record for this (path, size, mtime), creating a WANTED one if absent.

        An existing record is never overwritten; only a still-WANTED one takes the new
        *english_audio* flag. Returns the record as stored.
        """
        key = _key(path)
        with self._tx(write=True) as conn:
            row = self._subtitle_row(conn, key, size, mtime)
            if row is None:
                conn.execute(
                    "INSERT INTO subtitles (path, size, mtime, status, english_audio, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (key, size, mtime, SubtitleStatus.WANTED.value, int(english_audio), _now()),
                )
                self._event(conn, f"{key}: subtitle new -> {SubtitleStatus.WANTED.value}")
            elif (
                SubtitleStatus(row[3]) is SubtitleStatus.WANTED
                and bool(row[9]) is not english_audio
            ):
                conn.execute(
                    "UPDATE subtitles SET english_audio = ?, updated_at = ? "
                    "WHERE path = ? AND size = ? AND mtime = ?",
                    (int(english_audio), _now(), key, size, mtime),
                )
            stored = self._subtitle_row(conn, key, size, mtime)
        if stored is None:  # pragma: no cover - inserted or found above, in the same transaction
            raise StateError(f"subtitle record for {path} vanished during its own transaction")
        return _subtitle_record(stored)

    def wanted_subtitles(self, limit: int | None = None) -> list[SubtitleRecord]:
        """WANTED subtitle records, ordered like the encode queue.

        Largest estimated saving first, taken from the live `files` record for the same
        (path, size, mtime); ties by path; a record with no live match comes last. *limit*
        keeps the first that many (none for zero or less).
        """
        if limit is not None and limit <= 0:
            return []
        with self._tx() as conn:
            rows = conn.execute(
                f"SELECT {_SUB_COLUMNS_S} FROM subtitles AS s "  # noqa: S608  # nosec B608
                "LEFT JOIN files AS f ON f.path = s.path AND f.size = s.size "
                "AND f.mtime = s.mtime AND f.status != ? "
                "WHERE s.status = ? "
                "ORDER BY f.id IS NULL, f.est_saving_bytes DESC, s.path, s.size, s.mtime "
                "LIMIT ?",
                (Status.STALE.value, SubtitleStatus.WANTED.value, -1 if limit is None else limit),
            ).fetchall()
        return [_subtitle_record(r) for r in rows]

    def subtitle_counts(self) -> dict[SubtitleStatus, int]:
        """Number of subtitle records per status, with every status present (zero when none)."""
        with self._tx() as conn:
            rows = conn.execute("SELECT status, COUNT(*) FROM subtitles GROUP BY status").fetchall()
        result = dict.fromkeys(SubtitleStatus, 0)
        for status, n in rows:
            result[SubtitleStatus(status)] = int(n)
        return result

    def reset_subtitle(self, path: Path, size: int, mtime: float) -> None:
        """Make an UNAVAILABLE or UNCLEAR record WANTED again, forgetting what was tried.

        Clears the tried ids, reason and attempt count so the next search starts afresh. A
        record in any other status, or none, is left as it is.
        """
        key = _key(path)
        with self._tx(write=True) as conn:
            changed = conn.execute(
                "UPDATE subtitles SET status = ?, tried_ids = '[]', reason = '', attempts = 0, "
                "updated_at = ? WHERE path = ? AND size = ? AND mtime = ? AND status IN (?, ?)",
                (
                    SubtitleStatus.WANTED.value,
                    _now(),
                    key,
                    size,
                    mtime,
                    SubtitleStatus.UNAVAILABLE.value,
                    SubtitleStatus.UNCLEAR.value,
                ),
            ).rowcount
            if changed:
                self._event(conn, f"{key}: subtitle reset -> {SubtitleStatus.WANTED.value}")

    # -- subtitle download allowance --------------------------------------------------------

    def quota(self) -> QuotaInfo | None:
        """The last stored download allowance, or None when none is stored or it is unreadable."""
        with self._tx() as conn:
            rows = dict(
                conn.execute(
                    "SELECT key, value FROM meta WHERE key IN (?, ?)",
                    (_QUOTA_REMAINING_KEY, _QUOTA_RESET_KEY),
                ).fetchall()
            )
        try:
            remaining = int(rows[_QUOTA_REMAINING_KEY])
            if remaining < 0:
                return None
            return QuotaInfo(
                remaining=remaining, reset_at=datetime.fromisoformat(rows[_QUOTA_RESET_KEY])
            )
        except (KeyError, ValueError):
            return None

    def set_quota(self, q: QuotaInfo) -> None:
        """Store the download allowance (both values in one transaction)."""
        with self._tx(write=True) as conn:
            self._set_meta(conn, _QUOTA_REMAINING_KEY, str(q.remaining))
            self._set_meta(conn, _QUOTA_RESET_KEY, q.reset_at.astimezone(UTC).isoformat())

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
        """Record the calibration the user approved: the VMAF target and CRF ceiling a run uses.

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
        """The approved ``(vmaf_target, crf_max)``, or None before `approve_calibration`.

        A stored value that is not what `approve_calibration` writes (not JSON, a missing key,
        a non-finite target, a CRF that is not an integer) raises `StateError`.
        """
        raw = self._meta(_CALIBRATION_KEY)
        if raw is None:
            return None
        try:
            data = json.loads(raw)
            vmaf_target, crf_max = data["vmaf_target"], data["crf_max"]
        except (ValueError, TypeError, KeyError) as exc:
            raise StateError(f"the stored calibration is unreadable: {exc}") from None
        target_ok = (
            isinstance(vmaf_target, int | float)
            and not isinstance(vmaf_target, bool)
            and math.isfinite(vmaf_target)
        )
        crf_ok = isinstance(crf_max, int) and not isinstance(crf_max, bool)
        if not (target_ok and crf_ok):
            raise StateError(
                f"the stored calibration is unreadable: vmaf_target {vmaf_target!r}, "
                f"crf_max {crf_max!r}; approve it again with `calibrate --approve`"
            )
        return float(vmaf_target), int(crf_max)

    def events(self, limit: int = 100) -> list[str]:
        """The most recent event lines, newest first, each prefixed with an ISO timestamp."""
        if limit <= 0:
            return []
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT line FROM events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [str(r[0]) for r in rows]
