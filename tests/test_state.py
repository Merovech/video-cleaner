from __future__ import annotations

import os
import re
import sqlite3
import threading
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

import pytest

from video_cleaner.probe import MediaInfo
from video_cleaner.state import (
    SCHEMA_VERSION,
    FileRecord,
    StaleRecord,
    State,
    StateError,
    Status,
    SubtitleRecord,
    SubtitleStatus,
    UnrepresentablePath,
)
from video_cleaner.subtitles.models import QuotaInfo

SIZE = 8_000_000_000
MTIME = 1_700_000_000.5
ISO_PREFIX = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d")


def make_info(
    path: str | Path = "/media/movies/a.mkv", size: int = SIZE, mtime: float = MTIME
) -> MediaInfo:
    return MediaInfo(
        path=Path(path),
        size=size,
        mtime=mtime,
        duration_s=7200.0,
        container="matroska,webm",
        video=None,
        audio=(),
        subtitles=(),
        chapters=0,
        cover_art=0,
    )


@pytest.fixture
def state(tmp_path: Path) -> Iterator[State]:
    s = State.open(tmp_path / "state.db")
    yield s
    s.close()


def upsert(
    s: State,
    info: MediaInfo,
    decision: str = "encode",
    saving: int = 1000,
    reasons: tuple[str, ...] = ("x264 high bitrate",),
    parts: tuple[Path, ...] = (),
) -> None:
    s.upsert_file(info, decision=decision, reasons=reasons, est_saving_bytes=saving, parts=parts)


def test_upsert_is_idempotent(state: State) -> None:
    info = make_info()
    upsert(state, info)
    upsert(state, info)
    rec = state.get(info.path)
    assert sum(state.counts().values()) == 1
    assert state.counts()[Status.PENDING] == 1
    assert rec == FileRecord(
        path=info.path,
        size=SIZE,
        mtime=MTIME,
        status=Status.PENDING,
        decision="encode",
        reasons=("x264 high bitrate",),
        est_saving_bytes=1000,
        parts=(),
        note="",
    )
    assert len(state.events()) == 1


def test_changed_size_or_mtime_creates_new_record_and_marks_old_stale(state: State) -> None:
    info = make_info()
    upsert(state, info)
    upsert(state, make_info(size=SIZE + 1))
    upsert(state, make_info(size=SIZE + 1, mtime=MTIME + 10))
    counts = state.counts()
    assert counts[Status.STALE] == 2
    assert counts[Status.PENDING] == 1
    current = state.get(info.path)
    assert current is not None
    assert (current.size, current.mtime, current.status) == (SIZE + 1, MTIME + 10, Status.PENDING)
    assert state.records(Status.PENDING) == [current]


def test_resume_after_reopen_keeps_status(tmp_path: Path) -> None:
    db = tmp_path / "sub" / "state.db"
    parts = (Path("/m/x - cd1.mkv"), Path("/m/x - cd2.mkv"))
    with State.open(db) as s:
        upsert(s, make_info(parts[0]), saving=77, parts=parts)
        s.set_status(parts[0], Status.ENCODING, "crf 20")
        s.acknowledge_plan("abc")
    with State.open(db) as s2:
        rec = s2.get(parts[0])
        assert rec is not None
        assert rec.status is Status.ENCODING
        assert rec.note == "crf 20"
        assert rec.parts == parts
        assert s2.plan_acknowledged() == "abc"
        assert s2.records(Status.PENDING) == []
        # Re-planning the unchanged file must not reset an advanced status.
        upsert(s2, make_info(parts[0]), saving=77, parts=parts)
        again = s2.get(parts[0])
        assert again is not None
        assert again.status is Status.ENCODING


def test_plan_ack_roundtrip(state: State) -> None:
    assert state.plan_acknowledged() is None
    state.acknowledge_plan("hash-1")
    assert state.plan_acknowledged() == "hash-1"
    state.acknowledge_plan("hash-2")
    assert state.plan_acknowledged() == "hash-2"


def test_counts(state: State) -> None:
    upsert(state, make_info("/m/a.mkv"))
    upsert(state, make_info("/m/b.mkv"))
    upsert(state, make_info("/m/c.mkv"), decision="leave_alone")
    upsert(state, make_info("/m/d.mkv"), decision="flag", reasons=("4K",))
    state.set_status(Path("/m/b.mkv"), Status.COMPLETED)
    counts = state.counts()
    assert set(counts) == set(Status)
    assert counts[Status.PENDING] == 1
    assert counts[Status.COMPLETED] == 1
    assert counts[Status.LEFT_ALONE] == 1
    assert counts[Status.FLAGGED] == 1
    assert counts[Status.ENCODING] == 0


@pytest.mark.parametrize(
    ("decision", "status"),
    [
        ("encode", Status.PENDING),
        ("remux", Status.PENDING),
        ("leave_alone", Status.LEFT_ALONE),
        ("flag", Status.FLAGGED),
    ],
)
def test_decision_maps_to_initial_status(state: State, decision: str, status: Status) -> None:
    upsert(state, make_info(), decision=decision)
    rec = state.get(make_info().path)
    assert rec is not None
    assert rec.status is status
    assert rec.decision == decision


def test_unknown_decision_rejected(state: State) -> None:
    with pytest.raises(ValueError, match="decision"):
        upsert(state, make_info(), decision="explode")
    assert state.get(make_info().path) is None


def test_reupsert_refreshes_plan_while_still_unstarted(state: State) -> None:
    info = make_info()
    upsert(state, info, decision="encode", saving=100)
    upsert(state, info, decision="flag", saving=0, reasons=("HDR",))
    rec = state.get(info.path)
    assert rec is not None
    assert (rec.status, rec.decision, rec.reasons, rec.est_saving_bytes) == (
        Status.FLAGGED,
        "flag",
        ("HDR",),
        0,
    )


def test_reupsert_does_not_touch_advanced_record(state: State) -> None:
    info = make_info()
    upsert(state, info, saving=100)
    state.set_status(info.path, Status.COMPLETED, "done")
    upsert(state, info, decision="flag", saving=0, reasons=("other",))
    rec = state.get(info.path)
    assert rec is not None
    assert (rec.status, rec.decision, rec.est_saving_bytes) == (Status.COMPLETED, "encode", 100)


@pytest.mark.parametrize(
    ("status", "note"),
    [
        (Status.LEFT_ALONE, "projected saving 4% is below 10%"),
        (Status.FLAGGED, "guard failed: worst window 71.0"),
        (Status.FLAGGED, ""),
        (Status.LEFT_ALONE, ""),
    ],
)
def test_reupsert_keeps_run_outcome_on_encode_record(
    state: State, status: Status, note: str
) -> None:
    """A run's LEFT_ALONE/FLAGGED verdict survives a re-survey of the unchanged file."""
    info = make_info()
    upsert(state, info, decision="encode", saving=100)
    state.set_status(info.path, status, note)
    upsert(state, info, decision="encode", saving=100)
    rec = state.get(info.path)
    assert rec is not None
    assert (rec.status, rec.note) == (status, note)


def test_reupsert_keeps_noted_record_in_initial_status(state: State) -> None:
    """A note means the run touched the record, even when the status equals the initial one."""
    info = make_info()
    upsert(state, info, decision="flag", saving=0)
    state.set_status(info.path, Status.FLAGGED, "output name collides with Film.mkv")
    upsert(state, info, decision="encode", saving=100)
    rec = state.get(info.path)
    assert rec is not None
    assert (rec.status, rec.decision, rec.note) == (
        Status.FLAGGED,
        "flag",
        "output name collides with Film.mkv",
    )


def test_reupsert_refreshes_unstarted_left_alone_record(state: State) -> None:
    info = make_info()
    upsert(state, info, decision="leave_alone", saving=0)
    upsert(state, info, decision="encode", saving=100)
    rec = state.get(info.path)
    assert rec is not None
    assert (rec.status, rec.decision) == (Status.PENDING, "encode")


def test_set_status_unknown_path_raises(state: State) -> None:
    with pytest.raises(KeyError):
        state.set_status(Path("/nope.mkv"), Status.QUEUED)


def test_set_status_on_current_record_only(state: State) -> None:
    upsert(state, make_info())
    upsert(state, make_info(size=SIZE + 5))
    state.set_status(make_info().path, Status.QUEUED)
    counts = state.counts()
    assert counts[Status.QUEUED] == 1
    assert counts[Status.STALE] == 1


def test_get_unknown_returns_none(state: State) -> None:
    assert state.get(Path("/nope.mkv")) is None


def test_events_are_timestamped_newest_first_and_limited(state: State) -> None:
    upsert(state, make_info("/m/a.mkv"))
    state.set_status(Path("/m/a.mkv"), Status.QUEUED, "picked")
    state.set_status(Path("/m/a.mkv"), Status.COPYING)
    lines = state.events()
    assert len(lines) == 3
    assert all(ISO_PREFIX.match(line) for line in lines)
    assert "copying" in lines[0]
    assert "queued" in lines[1]
    assert "picked" in lines[1]
    assert "a.mkv" in lines[2]
    assert len(state.events(limit=2)) == 2
    assert state.events(limit=0) == []


def test_odd_paths_round_trip(state: State) -> None:
    odd = Path(
        "/m/The First Omen (2024).en.MX]/Cafe\u0301 \u00fcber [x] q'; DROP TABLE files;--.mkv"
    )
    upsert(state, make_info(odd))
    rec = state.get(odd)
    assert rec is not None
    assert rec.path == odd
    assert state.counts()[Status.PENDING] == 1


@pytest.mark.skipif(os.name != "nt", reason="UNC paths are Windows-specific")
def test_unc_path_round_trip(tmp_path: Path) -> None:
    unc = Path(PureWindowsPath(r"\\nas\media\Movies\A (2020)\A.mkv"))
    db = tmp_path / "unc.db"
    with State.open(db) as s:
        upsert(s, make_info(unc))
        rec = s.get(unc)
        assert rec is not None
        assert rec.path == unc
    raw = sqlite3.connect(db)
    try:
        stored = raw.execute("SELECT path FROM files").fetchone()[0]
    finally:
        raw.close()
    assert "\\" not in stored


def test_wal_mode_and_schema_version(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    with State.open(db):
        pass
    raw = sqlite3.connect(db)
    try:
        assert raw.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert raw.execute("SELECT version FROM schema_version").fetchall() == [(SCHEMA_VERSION,)]
    finally:
        raw.close()


_V1_FILES = """
CREATE TABLE schema_version (version INTEGER NOT NULL);
INSERT INTO schema_version (version) VALUES (1);
CREATE TABLE files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL, size INTEGER NOT NULL, mtime REAL NOT NULL, status TEXT NOT NULL,
    decision TEXT NOT NULL, reasons TEXT NOT NULL, est_saving_bytes INTEGER NOT NULL,
    parts TEXT NOT NULL, note TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL,
    UNIQUE (path, size, mtime)
);
INSERT INTO files (path, size, mtime, status, decision, reasons, est_saving_bytes, parts,
    note, updated_at)
VALUES ('/media/movies/a.mkv', 8000000000, 1700000000.5, 'completed', 'encode', '[]', 5, '[]',
    'done', '2026-11-01T00:00:00+00:00');
"""


def test_version_1_database_is_migrated(tmp_path: Path) -> None:
    db = tmp_path / "v1.db"
    raw = sqlite3.connect(db)
    raw.executescript(_V1_FILES)
    raw.close()
    with State.open(db) as s:
        rec = s.get(make_info().path)
        assert rec is not None
        assert (rec.status, rec.movie_hash) == (Status.COMPLETED, None)
        upsert_hash(s, make_info(), "8e245d9679d31e12")
        rec = s.get(make_info().path)
        assert rec is not None and rec.movie_hash == "8e245d9679d31e12"
    raw = sqlite3.connect(db)
    try:
        assert raw.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == (
            SCHEMA_VERSION
        )
    finally:
        raw.close()
    with State.open(db) as s:  # reopening a migrated database is a no-op
        assert s.get(make_info().path) is not None


def upsert_hash(s: State, info: MediaInfo, movie_hash: str | None) -> None:
    s.upsert_file(info, decision="encode", reasons=(), est_saving_bytes=1, movie_hash=movie_hash)


def test_movie_hash_is_stored_and_returned(state: State) -> None:
    upsert_hash(state, make_info(), "00000000000abcde")
    rec = state.get(make_info().path)
    assert rec is not None and rec.movie_hash == "00000000000abcde"


def test_movie_hash_defaults_to_none(state: State) -> None:
    upsert(state, make_info())
    rec = state.get(make_info().path)
    assert rec is not None and rec.movie_hash is None


def test_movie_hash_filled_in_on_advanced_record_but_never_cleared(state: State) -> None:
    info = make_info()
    upsert(state, info)
    state.set_status(info.path, Status.COMPLETED, "done")
    upsert_hash(state, info, "1111111111111111")
    upsert_hash(state, info, None)
    rec = state.get(info.path)
    assert rec is not None
    assert (rec.status, rec.movie_hash) == (Status.COMPLETED, "1111111111111111")


def test_newer_schema_version_refused(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    with State.open(db):
        pass
    raw = sqlite3.connect(db)
    raw.execute("UPDATE schema_version SET version = 99")
    raw.commit()
    raw.close()
    with pytest.raises(StateError, match="schema"):
        State.open(db)


def test_close_is_idempotent_and_use_after_close_fails(tmp_path: Path) -> None:
    s = State.open(tmp_path / "s.db")
    s.close()
    s.close()
    with pytest.raises(StateError, match="closed"):
        s.counts()


def test_concurrent_use_from_threads(state: State) -> None:
    errors: list[BaseException] = []

    def worker(n: int) -> None:
        try:
            for i in range(25):
                upsert(state, make_info(f"/m/t{n}-{i}.mkv"), saving=i)
                state.counts()
                state.records(Status.PENDING)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert state.counts()[Status.PENDING] == 150


def test_status_values_are_lowercase_names() -> None:
    assert [s.value for s in Status] == [s.name.lower() for s in Status]
    assert [s.name for s in Status] == [
        "PENDING",
        "QUEUED",
        "COPYING",
        "ENCODING",
        "MUXING",
        "VERIFYING",
        "COMPLETED",
        "LEFT_ALONE",
        "FLAGGED",
        "RETRY",
        "STALE",
    ]


def test_source_reverting_to_old_key_revives_that_record(state: State) -> None:
    upsert(state, make_info(size=SIZE))
    upsert(state, make_info(size=SIZE + 1))
    upsert(state, make_info(size=SIZE))
    counts = state.counts()
    assert counts[Status.PENDING] == 1
    assert counts[Status.STALE] == 1
    rec = state.get(make_info().path)
    assert rec is not None
    assert rec.size == SIZE
    assert rec.status is Status.PENDING


IN_PROGRESS = (
    Status.QUEUED,
    Status.COPYING,
    Status.ENCODING,
    Status.MUXING,
    Status.VERIFYING,
)


def test_reset_in_progress_after_simulated_crash(tmp_path: Path) -> None:
    db = tmp_path / "crash.db"
    with State.open(db) as s:
        for i, status in enumerate(IN_PROGRESS):
            upsert(s, make_info(f"/m/f{i}.mkv"), saving=100 - i)
            s.set_status(Path(f"/m/f{i}.mkv"), status)
        upsert(s, make_info("/m/pending.mkv"))
        upsert(s, make_info("/m/done.mkv"))
        s.set_status(Path("/m/done.mkv"), Status.COMPLETED)
        upsert(s, make_info("/m/ret.mkv"))
        s.set_status(Path("/m/ret.mkv"), Status.RETRY, "earlier")
    # "kill" happened: reopen. The interrupted records are not RETRY until reset.
    with State.open(db) as s2:
        assert [r.path for r in s2.records(Status.RETRY)] == [Path("/m/ret.mkv")]
        assert s2.reset_in_progress() == len(IN_PROGRESS)
        counts = s2.counts()
        assert counts[Status.RETRY] == len(IN_PROGRESS) + 1
        for status in IN_PROGRESS:
            assert counts[status] == 0
        assert counts[Status.PENDING] == 1
        assert counts[Status.COMPLETED] == 1
        rec = s2.get(Path("/m/f2.mkv"))
        assert rec is not None
        assert rec.status is Status.RETRY
        assert rec.note == "reset after interrupted run"
        untouched = s2.get(Path("/m/ret.mkv"))
        assert untouched is not None
        assert untouched.note == "earlier"


def test_reset_in_progress_is_idempotent_and_writes_events(state: State) -> None:
    for i, status in enumerate((Status.COPYING, Status.ENCODING)):
        upsert(state, make_info(f"/m/f{i}.mkv"))
        state.set_status(Path(f"/m/f{i}.mkv"), status)
    before = len(state.events(limit=1000))
    assert state.reset_in_progress() == 2
    lines = state.events(limit=1000)
    assert len(lines) == before + 2
    assert all("reset after interrupted run" in line for line in lines[:2])
    assert any("copying -> retry" in line for line in lines[:2])
    assert any("encoding -> retry" in line for line in lines[:2])
    assert state.reset_in_progress() == 0
    assert len(state.events(limit=1000)) == before + 2


def test_reset_in_progress_custom_target_and_ignores_stale(state: State) -> None:
    upsert(state, make_info("/m/a.mkv"))
    state.set_status(Path("/m/a.mkv"), Status.ENCODING)
    upsert(state, make_info("/m/a.mkv", size=SIZE + 1))  # old encoding record becomes stale
    upsert(state, make_info("/m/b.mkv"))
    state.set_status(Path("/m/b.mkv"), Status.MUXING)
    assert state.reset_in_progress(Status.PENDING) == 1
    rec = state.get(Path("/m/b.mkv"))
    assert rec is not None
    assert rec.status is Status.PENDING
    assert state.counts()[Status.STALE] == 1


def test_records_filtering_and_order(state: State) -> None:
    upsert(state, make_info("/m/b.mkv"))
    upsert(state, make_info("/m/a.mkv"))
    upsert(state, make_info("/m/c.mkv"), decision="flag")
    upsert(state, make_info("/m/a.mkv", size=SIZE + 1))  # stales the first a.mkv
    state.set_status(Path("/m/b.mkv"), Status.ENCODING)
    live = state.records()
    assert [r.path.name for r in live] == ["a.mkv", "b.mkv", "c.mkv"]
    assert all(r.status is not Status.STALE for r in live)
    assert [r.path.name for r in state.records(Status.PENDING)] == ["a.mkv"]
    assert [r.path.name for r in state.records(Status.PENDING, Status.ENCODING)] == [
        "a.mkv",
        "b.mkv",
    ]
    stale = state.records(Status.STALE)
    assert [(r.path.name, r.size) for r in stale] == [("a.mkv", SIZE)]
    assert state.records(Status.COMPLETED) == []


def test_set_status_with_old_key_cannot_touch_new_record(state: State) -> None:
    old = make_info()
    upsert(state, old)
    state.set_status(old.path, Status.COPYING)
    new = make_info(size=SIZE + 7, mtime=MTIME + 3)
    upsert(state, new)
    events_before = state.events(limit=1000)
    with pytest.raises(StaleRecord):
        state.set_status(old.path, Status.COMPLETED, size=old.size, mtime=old.mtime)
    current = state.get(old.path)
    assert current is not None
    assert (current.size, current.mtime, current.status) == (new.size, new.mtime, Status.PENDING)
    assert state.counts()[Status.COMPLETED] == 0
    assert state.events(limit=1000) == events_before
    assert issubclass(StaleRecord, StateError)


def test_set_status_with_unknown_key_raises_stale_record(state: State) -> None:
    upsert(state, make_info())
    with pytest.raises(StaleRecord):
        state.set_status(make_info().path, Status.COMPLETED, size=1, mtime=2.0)
    assert state.counts()[Status.PENDING] == 1


def test_set_status_with_current_key_updates(state: State) -> None:
    info = make_info()
    upsert(state, info)
    state.set_status(info.path, Status.COPYING, "go", size=info.size, mtime=info.mtime)
    rec = state.get(info.path)
    assert rec is not None
    assert (rec.status, rec.note) == (Status.COPYING, "go")


def test_unrepresentable_path_raises_before_any_write(state: State) -> None:
    bad = Path("/m/bad\udc80name.mkv")
    with pytest.raises(UnrepresentablePath) as excinfo:
        upsert(state, make_info(bad))
    assert repr(bad) in str(excinfo.value)
    assert isinstance(excinfo.value, StateError)
    assert sum(state.counts().values()) == 0
    assert state.events() == []
    with pytest.raises(UnrepresentablePath):
        state.get(bad)
    with pytest.raises(UnrepresentablePath):
        state.set_status(bad, Status.QUEUED)
    good = make_info("/m/ok.mkv")
    with pytest.raises(UnrepresentablePath):
        upsert(state, good, parts=(good.path, bad))
    assert sum(state.counts().values()) == 0


# --- calibration approval ---------------------------------------------


def test_calibration_is_none_until_approved(state: State) -> None:
    assert state.calibration() is None


def test_approve_calibration_stores_values_and_one_event(state: State) -> None:
    state.approve_calibration(95.5, 23)
    assert state.calibration() == (95.5, 23)
    events = state.events()
    assert len(events) == 1
    assert "calibration approved" in events[0]
    assert "95.5" in events[0] and "23" in events[0]
    state.approve_calibration(94.0, 22)
    assert state.calibration() == (94.0, 22)


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "[]",
        "{}",
        '{"vmaf_target": "x", "crf_max": 23}',
        '{"vmaf_target": 95.0, "crf_max": "high"}',
        '{"vmaf_target": NaN, "crf_max": 23}',
        '{"vmaf_target": 95.0, "crf_max": 23.5}',
        '{"vmaf_target": 95.0, "crf_max": true}',
    ],
)
def test_corrupt_stored_calibration_is_a_state_error(tmp_path: Path, raw: str) -> None:
    db = tmp_path / "s.db"
    State.open(db).close()
    conn = sqlite3.connect(db)
    with conn:
        conn.execute("INSERT INTO meta (key, value) VALUES ('calibration', ?)", (raw,))
    conn.close()
    with State.open(db) as state, pytest.raises(StateError, match="calibration"):
        state.calibration()


def test_calibration_survives_reopen(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    with State.open(db) as s:
        s.approve_calibration(96.0, 24)
    with State.open(db) as s2:
        assert s2.calibration() == (96.0, 24)
        assert s2.plan_acknowledged() is None  # separate from the plan acknowledgement


@pytest.mark.parametrize(
    ("vmaf", "crf"),
    [(float("nan"), 23), (float("inf"), 23), (95.0, 23.5), (95.0, True)],
)
def test_approve_calibration_rejects_non_values(state: State, vmaf: float, crf: int) -> None:
    with pytest.raises(ValueError):
        state.approve_calibration(vmaf, crf)
    assert state.calibration() is None
    assert state.events() == []


# --- deliberate retry of a run outcome -----------------------------------------------------


def test_requeue_puts_run_outcome_back_to_initial_status(state: State) -> None:
    info = make_info()
    upsert(state, info, decision="encode", saving=100)
    state.set_status(info.path, Status.FLAGGED, "guard failed")
    rec = state.requeue(info.path)
    assert (rec.status, rec.note) == (Status.PENDING, "")
    stored = state.get(info.path)
    assert stored is not None and (stored.status, stored.note) == (Status.PENDING, "")
    assert "requeued" in state.events(1)[0]


def test_requeue_refuses_completed_and_unknown(state: State) -> None:
    info = make_info()
    with pytest.raises(KeyError):
        state.requeue(info.path)
    upsert(state, info)
    state.set_status(info.path, Status.COMPLETED, "done")
    with pytest.raises(StateError, match="completed"):
        state.requeue(info.path)


class _FailingCommit:
    """A connection whose next COMMIT fails (disk full, I/O error)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self.fail_next_commit = True

    def execute(self, sql: str, *args: Any) -> sqlite3.Cursor:
        if sql == "COMMIT" and self.fail_next_commit:
            self.fail_next_commit = False
            raise sqlite3.OperationalError("disk I/O error")
        return self._conn.execute(sql, *args)

    def close(self) -> None:
        self._conn.close()


def test_failed_commit_rolls_back_and_leaves_the_handle_usable(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    State.open(db).close()
    raw = sqlite3.connect(db, isolation_level=None, check_same_thread=False)
    state = State(_FailingCommit(raw), db)  # type: ignore[arg-type]
    try:
        with pytest.raises(sqlite3.OperationalError):
            upsert(state, make_info())
        assert state.get(make_info().path) is None  # rolled back, nothing half-written
        upsert(state, make_info())  # the next transaction works: no "within a transaction"
        assert state.get(make_info().path) is not None
    finally:
        state.close()


# -- subtitle outcomes and the quota ledger ----------------------------------------------------

SUB_PATH = Path("/media/movies/a.mkv")


def sub_rec(
    status: SubtitleStatus = SubtitleStatus.WANTED,
    path: Path = SUB_PATH,
    size: int = SIZE,
    mtime: float = MTIME,
    **fields: Any,
) -> SubtitleRecord:
    return SubtitleRecord(path=path, size=size, mtime=mtime, status=status, **fields)


def test_fresh_database_has_the_subtitle_table_at_the_current_version(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    State.open(db).close()
    raw = sqlite3.connect(db)
    try:
        assert SCHEMA_VERSION == 4
        assert raw.execute("SELECT version FROM schema_version").fetchall() == [(4,)]
        assert raw.execute("SELECT COUNT(*) FROM subtitles").fetchone() == (0,)
    finally:
        raw.close()


def test_version_3_database_migrates_without_losing_rows(tmp_path: Path) -> None:
    db = tmp_path / "v3.db"
    with State.open(db) as s:
        upsert(s, make_info(), saving=42)
        s.set_status(make_info().path, Status.COMPLETED, "done")
        s.acknowledge_plan("abc")
    raw = sqlite3.connect(db)
    raw.executescript("DROP TABLE subtitles; UPDATE schema_version SET version = 3;")
    raw.close()
    with State.open(db) as s:
        rec = s.get(make_info().path)
        assert rec is not None
        assert (rec.status, rec.est_saving_bytes, rec.note) == (Status.COMPLETED, 42, "done")
        assert s.plan_acknowledged() == "abc"
        assert s.subtitle(SUB_PATH, SIZE, MTIME) is None
        s.set_subtitle(sub_rec(SubtitleStatus.FOUND, provider="local"))
        assert s.subtitle_counts()[SubtitleStatus.FOUND] == 1
    raw = sqlite3.connect(db)
    try:
        assert raw.execute("SELECT version FROM schema_version").fetchall() == [(SCHEMA_VERSION,)]
    finally:
        raw.close()


def test_schema_newer_than_this_version_is_still_refused(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    State.open(db).close()
    raw = sqlite3.connect(db)
    raw.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION + 1,))
    raw.commit()
    raw.close()
    with pytest.raises(StateError, match="newer"):
        State.open(db)


def test_subtitle_statuses_are_the_stored_words() -> None:
    assert [s.value for s in SubtitleStatus] == [
        "wanted",
        "found",
        "not_needed",
        "unavailable",
        "unclear",
    ]


def test_unknown_subtitle_is_none(state: State) -> None:
    assert state.subtitle(SUB_PATH, SIZE, MTIME) is None


def test_set_subtitle_round_trips_every_field(state: State) -> None:
    rec = SubtitleRecord(
        path=SUB_PATH,
        size=SIZE,
        mtime=MTIME,
        status=SubtitleStatus.FOUND,
        provider="opensubtitles",
        cache_path=Path("/cache/subs/ab/abcdef.srt"),
        sha256="ab" * 32,
        provenance="English (OpenSubtitles)",
        hash_exact=True,
        english_audio=True,
        reason="hash match",
        tried_ids=(11, 4_000_000_000, 7),
        attempts=3,
    )
    state.set_subtitle(rec)
    assert state.subtitle(SUB_PATH, SIZE, MTIME) == rec


def test_subtitle_defaults_round_trip(state: State) -> None:
    state.set_subtitle(sub_rec())
    got = state.subtitle(SUB_PATH, SIZE, MTIME)
    assert got == sub_rec()
    assert got is not None and got.cache_path is None and got.tried_ids == ()


def test_subtitle_windows_style_paths_round_trip(state: State) -> None:
    path = Path(r"D:\Movies\Some Film (2020)\Some Film.mkv")
    cache = Path(r"C:\Users\x\AppData\Local\video-cleaner\subs\a.srt")
    state.set_subtitle(sub_rec(SubtitleStatus.FOUND, path=path, cache_path=cache))
    got = state.subtitle(path, SIZE, MTIME)
    assert got is not None
    assert (got.path, got.cache_path) == (path, cache)


def test_subtitle_paths_are_stored_posix_style(state: State) -> None:
    path = Path(r"D:\Movies\a.mkv")
    state.set_subtitle(sub_rec(path=path))
    raw = sqlite3.connect(state._db_path)
    try:
        stored = raw.execute("SELECT path FROM subtitles").fetchone()[0]
    finally:
        raw.close()
    assert stored == path.as_posix()
    if os.name == "nt":
        assert stored == "D:/Movies/a.mkv"


def test_set_subtitle_updates_in_place(state: State) -> None:
    state.set_subtitle(sub_rec(attempts=1))
    state.set_subtitle(sub_rec(SubtitleStatus.UNAVAILABLE, reason="no results", attempts=2))
    got = state.subtitle(SUB_PATH, SIZE, MTIME)
    assert got is not None
    assert (got.status, got.reason, got.attempts) == (SubtitleStatus.UNAVAILABLE, "no results", 2)
    assert sum(state.subtitle_counts().values()) == 1


def test_found_subtitle_survives_a_wanted_write(state: State) -> None:
    found = sub_rec(SubtitleStatus.FOUND, provider="sidecar", sha256="cd" * 32)
    state.set_subtitle(found)
    state.set_subtitle(sub_rec(SubtitleStatus.WANTED, english_audio=True))
    assert state.subtitle(SUB_PATH, SIZE, MTIME) == found


def test_found_subtitle_may_still_be_replaced_by_another_outcome(state: State) -> None:
    state.set_subtitle(sub_rec(SubtitleStatus.FOUND, provider="sidecar"))
    state.set_subtitle(sub_rec(SubtitleStatus.NOT_NEEDED, reason="embedded english"))
    got = state.subtitle(SUB_PATH, SIZE, MTIME)
    assert got is not None and got.status is SubtitleStatus.NOT_NEEDED


def test_subtitles_are_keyed_by_path_size_and_mtime(state: State) -> None:
    state.set_subtitle(sub_rec(SubtitleStatus.FOUND, provider="local"))
    assert state.subtitle(SUB_PATH, SIZE + 1, MTIME) is None  # overwritten original: fresh
    assert state.subtitle(SUB_PATH, SIZE, MTIME + 1) is None
    assert state.subtitle(Path("/media/movies/b.mkv"), SIZE, MTIME) is None
    fresh = state.ensure_wanted(SUB_PATH, SIZE + 1, MTIME, english_audio=False)
    assert fresh.status is SubtitleStatus.WANTED
    old = state.subtitle(SUB_PATH, SIZE, MTIME)
    assert old is not None and old.status is SubtitleStatus.FOUND
    assert sum(state.subtitle_counts().values()) == 2


def test_ensure_wanted_inserts_a_wanted_row(state: State) -> None:
    rec = state.ensure_wanted(SUB_PATH, SIZE, MTIME, english_audio=True)
    assert rec == sub_rec(english_audio=True)
    assert state.subtitle(SUB_PATH, SIZE, MTIME) == rec


@pytest.mark.parametrize("status", [SubtitleStatus.FOUND, SubtitleStatus.UNAVAILABLE])
def test_ensure_wanted_never_overwrites_a_decided_row(state: State, status: SubtitleStatus) -> None:
    decided = sub_rec(status, provider="local", reason="r", tried_ids=(5,), english_audio=False)
    state.set_subtitle(decided)
    got = state.ensure_wanted(SUB_PATH, SIZE, MTIME, english_audio=True)
    assert got == decided
    assert state.subtitle(SUB_PATH, SIZE, MTIME) == decided


@pytest.mark.parametrize("status", [SubtitleStatus.NOT_NEEDED, SubtitleStatus.UNCLEAR])
def test_ensure_wanted_keeps_other_decided_rows_untouched(
    state: State, status: SubtitleStatus
) -> None:
    decided = sub_rec(status, english_audio=False)
    state.set_subtitle(decided)
    assert state.ensure_wanted(SUB_PATH, SIZE, MTIME, english_audio=True) == decided
    assert state.subtitle(SUB_PATH, SIZE, MTIME) == decided


def test_ensure_wanted_refreshes_the_audio_flag_of_a_wanted_row(state: State) -> None:
    state.set_subtitle(sub_rec(tried_ids=(9,), attempts=2))
    got = state.ensure_wanted(SUB_PATH, SIZE, MTIME, english_audio=True)
    assert got == sub_rec(english_audio=True, tried_ids=(9,), attempts=2)
    assert state.subtitle(SUB_PATH, SIZE, MTIME) == got
    assert state.ensure_wanted(SUB_PATH, SIZE, MTIME, english_audio=False).english_audio is False


def test_wanted_subtitles_are_ordered_like_the_encode_queue(state: State) -> None:
    big, mid, small, tie_a, tie_b, orphan = (Path(f"/m/{n}.mkv") for n in "bmstuo")
    for path, saving in ((big, 900), (mid, 500), (small, 5), (tie_a, 500), (tie_b, 500)):
        upsert(state, make_info(path), saving=saving)
        state.ensure_wanted(path, SIZE, MTIME, english_audio=False)
    state.ensure_wanted(orphan, SIZE, MTIME, english_audio=False)  # no files row: last
    got = [r.path for r in state.wanted_subtitles()]
    assert got == [big, mid, tie_a, tie_b, small, orphan]
    assert [r.path for r in state.wanted_subtitles(limit=2)] == [big, mid]
    assert state.wanted_subtitles(limit=0) == []
    assert len(state.wanted_subtitles(limit=None)) == 6


def test_wanted_subtitles_exclude_other_statuses_and_rank_by_the_live_source(
    state: State,
) -> None:
    a, b, c, d = (Path(f"/m/{n}.mkv") for n in "abcd")
    for path in (a, b, c):
        upsert(state, make_info(path), saving=10)
    state.set_subtitle(sub_rec(SubtitleStatus.FOUND, path=a))
    state.set_subtitle(sub_rec(SubtitleStatus.UNAVAILABLE, path=b))
    state.ensure_wanted(c, SIZE, MTIME, english_audio=False)
    assert [r.path for r in state.wanted_subtitles()] == [c]
    upsert(state, make_info(d), saving=50)
    state.ensure_wanted(d, SIZE, MTIME, english_audio=False)
    assert [r.path for r in state.wanted_subtitles()] == [d, c]
    # c's original is replaced: its old subtitle row matches no live file, so it ranks last.
    upsert(state, make_info(c, size=SIZE + 1), saving=999_999)
    assert [r.path for r in state.wanted_subtitles()] == [d, c]
    state.ensure_wanted(c, SIZE + 1, MTIME, english_audio=False)
    assert [(r.path, r.size) for r in state.wanted_subtitles()] == [
        (c, SIZE + 1),
        (d, SIZE),
        (c, SIZE),
    ]


def test_subtitle_counts_has_every_status(state: State) -> None:
    assert state.subtitle_counts() == dict.fromkeys(SubtitleStatus, 0)
    state.set_subtitle(sub_rec(SubtitleStatus.FOUND, path=Path("/m/1.mkv")))
    state.set_subtitle(sub_rec(SubtitleStatus.FOUND, path=Path("/m/2.mkv")))
    state.set_subtitle(sub_rec(SubtitleStatus.UNCLEAR, path=Path("/m/3.mkv")))
    state.ensure_wanted(Path("/m/4.mkv"), SIZE, MTIME, english_audio=False)
    counts = state.subtitle_counts()
    assert counts[SubtitleStatus.FOUND] == 2
    assert counts[SubtitleStatus.UNCLEAR] == 1
    assert counts[SubtitleStatus.WANTED] == 1
    assert counts[SubtitleStatus.UNAVAILABLE] == counts[SubtitleStatus.NOT_NEEDED] == 0


@pytest.mark.parametrize("status", [SubtitleStatus.UNAVAILABLE, SubtitleStatus.UNCLEAR])
def test_reset_subtitle_makes_it_wanted_and_forgets_tried_ids(
    state: State, status: SubtitleStatus
) -> None:
    state.set_subtitle(
        sub_rec(status, tried_ids=(1, 2, 3), reason="no luck", attempts=4, english_audio=True)
    )
    state.reset_subtitle(SUB_PATH, SIZE, MTIME)
    assert state.subtitle(SUB_PATH, SIZE, MTIME) == sub_rec(english_audio=True)


@pytest.mark.parametrize(
    "status", [SubtitleStatus.FOUND, SubtitleStatus.NOT_NEEDED, SubtitleStatus.WANTED]
)
def test_reset_subtitle_leaves_other_statuses_alone(state: State, status: SubtitleStatus) -> None:
    rec = sub_rec(status, provider="local", tried_ids=(1, 2), reason="kept")
    state.set_subtitle(rec)
    state.reset_subtitle(SUB_PATH, SIZE, MTIME)
    assert state.subtitle(SUB_PATH, SIZE, MTIME) == rec


def test_reset_subtitle_of_an_unknown_row_is_a_no_op(state: State) -> None:
    state.reset_subtitle(SUB_PATH, SIZE, MTIME)
    assert state.subtitle(SUB_PATH, SIZE, MTIME) is None


@pytest.mark.parametrize(
    "stored",
    ["not json", '{"a": 1}', '"7"', '[1, "2"]', "[true]", "[1.5]", "[-3]", "[[1]]", "", "null"],
)
def test_corrupt_tried_ids_read_as_empty_and_never_crash_a_read(state: State, stored: str) -> None:
    state.set_subtitle(sub_rec(tried_ids=(1, 2)))
    raw = sqlite3.connect(state._db_path)
    raw.execute("UPDATE subtitles SET tried_ids = ?", (stored,))
    raw.commit()
    raw.close()
    got = state.subtitle(SUB_PATH, SIZE, MTIME)
    assert got is not None and got.tried_ids == ()
    assert [r.tried_ids for r in state.wanted_subtitles()] == [()]


def test_tried_ids_are_stored_as_a_json_array_of_ints(state: State) -> None:
    state.set_subtitle(sub_rec(tried_ids=(3, 1, 2)))
    raw = sqlite3.connect(state._db_path)
    try:
        assert raw.execute("SELECT tried_ids FROM subtitles").fetchone() == ("[3, 1, 2]",)
    finally:
        raw.close()


@pytest.mark.parametrize("bad", [(-1,), (True,), (1.5,), ("7",)])
def test_tried_ids_that_are_not_non_negative_ints_are_refused(state: State, bad: Any) -> None:
    with pytest.raises(ValueError, match="tried_ids"):
        state.set_subtitle(sub_rec(tried_ids=bad))
    assert sum(state.subtitle_counts().values()) == 0


def test_hostile_text_is_stored_as_data(state: State) -> None:
    nasty = "x'); DROP TABLE subtitles; --"
    state.set_subtitle(sub_rec(provenance=nasty, reason=nasty, provider=nasty))
    got = state.subtitle(SUB_PATH, SIZE, MTIME)
    assert got is not None
    assert (got.provenance, got.reason, got.provider) == (nasty, nasty, nasty)


def test_subtitle_path_that_cannot_be_stored_is_refused(state: State) -> None:
    with pytest.raises(UnrepresentablePath):
        state.set_subtitle(sub_rec(path=Path("bad\udc80.mkv")))
    assert sum(state.subtitle_counts().values()) == 0


def test_set_subtitle_is_one_transaction(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    State.open(db).close()
    raw = sqlite3.connect(db, isolation_level=None, check_same_thread=False)
    state = State(_FailingCommit(raw), db)  # type: ignore[arg-type]
    try:
        with pytest.raises(sqlite3.OperationalError):
            state.set_subtitle(sub_rec())
        assert state.subtitle(SUB_PATH, SIZE, MTIME) is None  # rolled back
        state.set_subtitle(sub_rec())  # handle still usable
        assert state.subtitle(SUB_PATH, SIZE, MTIME) is not None
    finally:
        state.close()


def test_subtitle_rows_survive_reopen(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    rec = sub_rec(SubtitleStatus.FOUND, tried_ids=(8, 9), provider="local")
    with State.open(db) as s:
        s.set_subtitle(rec)
    with State.open(db) as s2:
        assert s2.subtitle(SUB_PATH, SIZE, MTIME) == rec


def test_concurrent_subtitle_writes_from_threads(state: State) -> None:
    errors: list[BaseException] = []

    def worker(n: int) -> None:
        try:
            for i in range(20):
                state.ensure_wanted(Path(f"/m/t{n}-{i}.mkv"), SIZE, MTIME, english_audio=False)
                state.wanted_subtitles(limit=5)
                state.subtitle_counts()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert state.subtitle_counts()[SubtitleStatus.WANTED] == 100


RESET = datetime(2026, 11, 24, 0, 0, 0, tzinfo=UTC)


def test_quota_is_none_until_stored(state: State) -> None:
    assert state.quota() is None


def test_quota_round_trips_through_meta(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    q = QuotaInfo(remaining=17, reset_at=RESET)
    with State.open(db) as s:
        s.set_quota(q)
        assert s.quota() == q
        s.set_quota(QuotaInfo(remaining=0, reset_at=RESET))
        assert s.quota() == QuotaInfo(remaining=0, reset_at=RESET)
    with State.open(db) as s2:
        assert s2.quota() == QuotaInfo(remaining=0, reset_at=RESET)
    raw = sqlite3.connect(db)
    try:
        stored = dict(raw.execute("SELECT key, value FROM meta WHERE key LIKE '%quota%'"))
    finally:
        raw.close()
    assert sorted(stored.values()) == ["0", "2026-11-24T00:00:00+00:00"]


def test_quota_reset_time_comes_back_in_utc(state: State) -> None:
    local = datetime(2026, 11, 24, 2, 0, tzinfo=timezone(timedelta(hours=2)))
    state.set_quota(QuotaInfo(remaining=5, reset_at=local))
    got = state.quota()
    assert got is not None
    assert got.reset_at == local
    assert got.reset_at.utcoffset() == timedelta(0)


@pytest.mark.parametrize(
    ("remaining", "reset_at"),
    [
        ("abc", "2026-11-24T00:00:00+00:00"),
        ("5", "not a time"),
        ("5", "2026-11-24T00:00:00"),  # naive
        ("-1", "2026-11-24T00:00:00+00:00"),
        ("5", ""),
    ],
)
def test_unreadable_stored_quota_reads_as_none(state: State, remaining: str, reset_at: str) -> None:
    state.set_quota(QuotaInfo(remaining=3, reset_at=RESET))
    raw = sqlite3.connect(state._db_path)
    raw.execute("UPDATE meta SET value = ? WHERE key LIKE '%remaining'", (remaining,))
    raw.execute("UPDATE meta SET value = ? WHERE key LIKE '%reset_at'", (reset_at,))
    raw.commit()
    raw.close()
    assert state.quota() is None


def test_half_stored_quota_reads_as_none(state: State) -> None:
    state.set_quota(QuotaInfo(remaining=3, reset_at=RESET))
    raw = sqlite3.connect(state._db_path)
    raw.execute("DELETE FROM meta WHERE key LIKE '%reset_at'")
    raw.commit()
    raw.close()
    assert state.quota() is None


def test_set_quota_is_one_transaction(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    State.open(db).close()
    raw = sqlite3.connect(db, isolation_level=None, check_same_thread=False)
    state = State(_FailingCommit(raw), db)  # type: ignore[arg-type]
    try:
        with pytest.raises(sqlite3.OperationalError):
            state.set_quota(QuotaInfo(remaining=9, reset_at=RESET))
        assert state.quota() is None  # neither half was written
        state.set_quota(QuotaInfo(remaining=9, reset_at=RESET))
        assert state.quota() == QuotaInfo(remaining=9, reset_at=RESET)
    finally:
        state.close()
