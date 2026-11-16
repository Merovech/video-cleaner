from __future__ import annotations

import os
import re
import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path, PureWindowsPath

import pytest

from video_cleaner.probe import MediaInfo
from video_cleaner.state import FileRecord, State, StateError, Status

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
