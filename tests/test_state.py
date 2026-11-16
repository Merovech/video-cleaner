from __future__ import annotations

import os
import re
import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path, PureWindowsPath

import pytest

from video_cleaner.probe import MediaInfo
from video_cleaner.state import (
    FileRecord,
    StaleRecord,
    State,
    StateError,
    Status,
    UnrepresentablePath,
)

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
    assert state.next_pending() == current


def test_next_pending_orders_by_estimated_saving(state: State) -> None:
    upsert(state, make_info("/m/small.mkv"), saving=10)
    upsert(state, make_info("/m/big.mkv"), saving=900)
    upsert(state, make_info("/m/mid.mkv"), saving=500)
    upsert(state, make_info("/m/left.mkv"), decision="leave_alone", saving=99999)
    order: list[str] = []
    while (rec := state.next_pending()) is not None:
        order.append(rec.path.name)
        state.set_status(rec.path, Status.QUEUED)
    assert order == ["big.mkv", "mid.mkv", "small.mkv"]


def test_next_pending_empty_is_none(state: State) -> None:
    assert state.next_pending() is None
    assert state.next_retry() is None


def test_next_retry_is_separate_from_pending(state: State) -> None:
    upsert(state, make_info("/m/a.mkv"), saving=10)
    upsert(state, make_info("/m/b.mkv"), saving=50)
    upsert(state, make_info("/m/c.mkv"), saving=20)
    state.set_status(Path("/m/b.mkv"), Status.RETRY, "nas dropped")
    state.set_status(Path("/m/c.mkv"), Status.RETRY)
    pending = state.next_pending()
    assert pending is not None
    assert pending.path == Path("/m/a.mkv")
    retry = state.next_retry()
    assert retry is not None
    assert retry.path == Path("/m/b.mkv")
    assert retry.note == "nas dropped"


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
        assert s2.next_pending() is None
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
        assert raw.execute("SELECT version FROM schema_version").fetchall() == [(1,)]
    finally:
        raw.close()


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
                state.next_pending()
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
    # "kill" happened: reopen. The pickers never return the interrupted records.
    with State.open(db) as s2:
        retry = s2.next_retry()
        assert retry is not None
        assert retry.path == Path("/m/ret.mkv")
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


# --- calibration approval (Task 14, ruling R3) ---------------------------------------------


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
