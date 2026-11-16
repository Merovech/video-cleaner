"""Resumable run state in SQLite.

One row per (first-part path, size, mtime). A file whose size or mtime changed gets a new row and
the old one becomes ``stale``, so a result computed from earlier bytes is never applied to newer
ones. All access goes through one connection guarded by a lock (the runner calls from several
threads); every statement is parameterised. Paths are stored as POSIX-style strings and returned
as `Path`.
"""

from __future__ import annotations

import json
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

SCHEMA_VERSION = 1
_BUSY_TIMEOUT_S = 30.0


class StateError(Exception):
    """The state database cannot be used (closed, or written by a newer version)."""


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
# Statuses in which a re-plan of an unchanged file may still replace the plan.
_UNSTARTED = (Status.PENDING, Status.LEFT_ALONE, Status.FLAGGED, Status.STALE)

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
_COLUMNS = "path, size, mtime, status, decision, reasons, est_saving_bytes, parts, note"


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


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _key(path: Path) -> str:
    return path.as_posix()


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
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

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
    ) -> None:
        """Record the plan for *info*, keyed on path + size + mtime.

        A new key creates a record (initial status from *decision*) and marks any other live
        record for the path ``stale``. Re-planning an unchanged key refreshes the plan only while
        the record has not started; a record that has advanced is left alone.
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
                "SELECT id, status FROM files WHERE path = ? AND size = ? AND mtime = ?",
                (path, info.size, info.mtime),
            ).fetchone()
            if existing is not None and Status(existing[1]) not in _UNSTARTED:
                return
            superseded = conn.execute(
                "UPDATE files SET status = ?, updated_at = ? "
                "WHERE path = ? AND status != ? AND NOT (size = ? AND mtime = ?)",
                (Status.STALE.value, _now(), path, Status.STALE.value, info.size, info.mtime),
            ).rowcount
            if existing is None:
                conn.execute(
                    "INSERT INTO files (path, size, mtime, status, decision, reasons, "
                    "est_saving_bytes, parts, note, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, '', ?)",
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
                    ),
                )
                suffix = f"; superseded {superseded} stale record(s)" if superseded else ""
                self._event(conn, f"{path}: new ({decision}) -> {initial.value}{suffix}")
            else:
                conn.execute(
                    "UPDATE files SET status = ?, decision = ?, reasons = ?, est_saving_bytes = ?, "
                    "parts = ?, updated_at = ? WHERE id = ?",
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

    def set_status(self, path: Path, status: Status, note: str = "") -> None:
        """Set the status and note of the live (non-stale) record for *path*.

        Raises KeyError when there is none.
        """
        key = _key(path)
        with self._tx(write=True) as conn:
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

    def get(self, path: Path) -> FileRecord | None:
        """The live record for *path*, or the most recent stale one if none is live."""
        with self._tx() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM files WHERE path = ? "  # noqa: S608 - constant columns
                "ORDER BY (status = ?), id DESC LIMIT 1",
                (_key(path), Status.STALE.value),
            ).fetchone()
        return None if row is None else _record(row)

    def _next_with_status(self, status: Status) -> FileRecord | None:
        with self._tx() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM files WHERE status = ? "  # noqa: S608 - constant columns
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
            conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (_PLAN_HASH_KEY, plan_hash),
            )
            self._event(conn, f"plan acknowledged: {plan_hash}")

    def plan_acknowledged(self) -> str | None:
        with self._tx() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = ?", (_PLAN_HASH_KEY,)).fetchone()
        return None if row is None else str(row[0])

    def events(self, limit: int = 100) -> list[str]:
        """The most recent event lines, newest first, each prefixed with an ISO timestamp."""
        if limit <= 0:
            return []
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT line FROM events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [str(r[0]) for r in rows]
