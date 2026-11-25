"""The daily subtitle job: selection, quota safety, persistence order, failures, re-deciding.

No network, no real ffmpeg/ffsubsync/mkvmerge: the resolver is a scripted fake (the real one is
used where the persisted tried ids matter), the probe is a fake and the client answers from lists.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from helpers import MTIME, make_info
from media import RecordingReporter
from resolve_fakes import NOW, FakeAligner, FakeClient, result, srt_bytes
from video_cleaner.config import Settings
from video_cleaner.consolefmt import Level
from video_cleaner.probe import MediaInfo, ProbeError
from video_cleaner.progress import Reporter
from video_cleaner.state import FileRecord, State, Status, SubtitleRecord, SubtitleStatus
from video_cleaner.subtitles import job as job_module
from video_cleaner.subtitles.job import (
    DEADLINE_REACHED,
    INTERRUPTED,
    LIMIT_REACHED,
    NOT_ANSWERING,
    JobSummary,
    persist_subtitle,
    run_resolve_job,
)
from video_cleaner.subtitles.models import QuotaInfo
from video_cleaner.subtitles.opensubtitles import ProviderError
from video_cleaner.subtitles.resolve import (
    CacheUnwritable,
    OnlineFailure,
    QuotaStop,
    ResolveContext,
    resolve_subtitle,
)
from video_cleaner.tools import ToolNotFound, Tools

GIB = 1024**3
TOOLS = Tools(Path("ffmpeg"), Path("ffprobe"), Path("mkvmerge"))
MIDNIGHT = datetime(2030, 1, 2, 0, 0, tzinfo=UTC)
HASH = "0123456789abcdef"


# --- fakes ---------------------------------------------------------------------------------


class RecordingState(State):
    """A real State that remembers the order of the writes the job makes."""

    def __init__(self, conn: Any, db_path: Path) -> None:
        super().__init__(conn, db_path)
        self.log: list[str] = []

    def set_subtitle(self, rec: SubtitleRecord) -> None:
        self.log.append(f"subtitle {rec.path.name} {rec.status.value}")
        super().set_subtitle(rec)

    def downgrade_stale_subtitle(self, path: Path, size: int, mtime: float, reason: str) -> bool:
        self.log.append(f"downgrade {path.name}")
        return super().downgrade_stale_subtitle(path, size, mtime, reason)

    def set_quota(self, q: QuotaInfo) -> None:
        self.log.append(f"quota {q.remaining}")
        super().set_quota(q)


Outcome = SubtitleRecord | BaseException | Callable[[FileRecord, SubtitleRecord], SubtitleRecord]


@dataclass
class Call:
    name: str
    status: SubtitleStatus
    has_client: bool
    offline_reason: str = ""


class FakeResolver:
    """Stands in for `resolve_subtitle`: a scripted outcome per file name (found by default)."""

    def __init__(self, cache: Path) -> None:
        self.cache = cache
        self.script: dict[str, Outcome] = {}
        self.calls: list[Call] = []

    def found(self, sub: SubtitleRecord, provider: str = "opensubtitles") -> SubtitleRecord:
        return replace(
            sub,
            status=SubtitleStatus.FOUND,
            provider=provider,
            cache_path=self.cache / f"{sub.path.stem}.srt",
            sha256="ab" * 32,
            provenance="English (OpenSubtitles)",
            reason="aligned",
        )

    def __call__(
        self,
        rec: FileRecord,
        info: MediaInfo,
        movie_hash: str | None,
        sub: SubtitleRecord,
        ctx: ResolveContext,
    ) -> SubtitleRecord:
        self.calls.append(
            Call(rec.path.name, sub.status, ctx.client is not None, ctx.offline_reason)
        )
        item = self.script.get(rec.path.name)
        if item is None:
            return self.found(sub)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, SubtitleRecord):
            return item
        return item(rec, sub)

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.calls]


def unavailable(sub: SubtitleRecord, tried: tuple[int, ...] = (5,)) -> SubtitleRecord:
    return replace(
        sub,
        status=SubtitleStatus.UNAVAILABLE,
        reason="no English subtitle found: sidecar none, local none, OpenSubtitles tried 1",
        tried_ids=tried,
        attempts=len(tried),
    )


def still_wanted(sub: SubtitleRecord, reason: str, tried: tuple[int, ...] = ()) -> SubtitleRecord:
    return replace(sub, reason=reason, tried_ids=tried, attempts=len(tried))


@dataclass
class Rig:
    tmp_path: Path
    settings: Settings
    state: RecordingState
    reporter: RecordingReporter
    client: FakeClient | None
    resolver: FakeResolver
    kinds: dict[str, str] = field(default_factory=dict)
    sizes: dict[str, int] = field(default_factory=dict)
    probed: list[str] = field(default_factory=list)
    probe_errors: dict[str, Exception] = field(default_factory=dict)
    clock_now: datetime = field(default_factory=lambda: datetime(2030, 1, 1, 6, 0))

    @property
    def library(self) -> Path:
        return self.settings.library_root

    def info(self, path: Path) -> MediaInfo:
        info = make_info(path, self.sizes.get(path.name, 6 * GIB))
        kind = self.kinds.get(path.name, "h264")
        video = info.video
        assert video is not None
        if kind == "hevc":
            video = replace(video, codec="hevc")
        elif kind == "hdr":
            video = replace(video, transfer="smpte2084")
        return replace(info, video=video)

    def probe(self, path: Path, tools: Tools, reporter: Reporter) -> MediaInfo:
        self.probed.append(path.name)
        error = self.probe_errors.get(path.name)
        if error is not None:
            raise error
        return self.info(path)

    def add(
        self,
        name: str,
        *,
        kind: str = "h264",
        decision: str = "encode",
        est: int = 0,
        size: int = 6 * GIB,
        parts: Sequence[Path] = (),
        subtitle: str | None = "wanted",
        hash_value: str | None = HASH,
        reasons: Sequence[str] = ("h264 1080p",),
    ) -> Path:
        path = self.library / name
        self.kinds[path.name] = kind
        self.sizes[path.name] = size
        self.state.upsert_file(
            self.info(path),
            decision=decision,
            reasons=reasons,
            est_saving_bytes=est,
            parts=parts,
            movie_hash=hash_value,
        )
        if subtitle == "wanted":
            self.state.ensure_wanted(path, size, MTIME, english_audio=False)
        return path

    def ctx(self, *client: FakeClient | None) -> ResolveContext:
        """The context for the job; with an argument, that client (None: no credentials)."""
        use = client[0] if client else self.client
        return ResolveContext(
            settings=self.settings,
            tools=TOOLS,
            reporter=self.reporter,
            local_index=(),
            client=use,
            align=FakeAligner(),
            now=lambda: NOW,
        )

    def run(self, **kwargs: Any) -> JobSummary:
        options: dict[str, Any] = {"limit": None, "deadline": None, "only": None}
        options.update(kwargs)
        resolve = options.pop("resolve", self.resolver)
        ctx = options.pop("ctx", None) or self.ctx()
        return run_resolve_job(
            self.settings,
            self.state,
            ctx,
            resolve=resolve,
            probe_fn=options.pop("probe_fn", self.probe),
            clock=options.pop("clock", lambda: self.clock_now),
            **options,
        )

    def sub(self, path: Path) -> SubtitleRecord:
        rec = self.state.get(path)
        assert rec is not None
        row = self.state.subtitle(path, rec.size, rec.mtime)
        assert row is not None
        return row

    def file(self, path: Path) -> FileRecord:
        rec = self.state.get(path)
        assert rec is not None
        return rec


@pytest.fixture
def rig(tmp_path: Path) -> Iterator[Rig]:
    settings = Settings(
        library_root=tmp_path / "library",
        scratch_dir=tmp_path / "scratch",
        completed_dir=tmp_path / "completed",
        state_db=tmp_path / "appdata" / "state.db",
        log_dir=tmp_path / "appdata" / "logs",
        subtitle_cache_dir=tmp_path / "cache",
        subtitle_candidates_per_file=3,
    )
    state = RecordingState.open(settings.state_db)
    assert isinstance(state, RecordingState)
    try:
        yield Rig(
            tmp_path,
            settings,
            state,
            RecordingReporter(),
            FakeClient(),
            FakeResolver(settings.subtitle_cache_dir),
        )
    finally:
        state.close()


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(f"{record.levelname} {record.getMessage()}")


@contextmanager
def watch(level: int = logging.INFO) -> Iterator[list[str]]:
    """The job's log lines (the tool's logger does not propagate to pytest's capture)."""
    logger = logging.getLogger("video_cleaner.subtitles.job")
    handler = _Records()
    before = logger.level
    logger.setLevel(level)
    logger.addHandler(handler)
    try:
        yield handler.messages
    finally:
        logger.removeHandler(handler)
        logger.setLevel(before)


# --- selection and order -------------------------------------------------------------------


def test_wanted_files_are_resolved_in_queue_order(rig: Rig) -> None:
    rig.add("small.mkv", est=10)
    rig.add("big.mkv", est=300)
    rig.add("middle.mkv", est=100)
    summary = rig.run()
    assert rig.resolver.names == ["big.mkv", "middle.mkv", "small.mkv"]
    assert summary == JobSummary(3, 0, 0, 0, "")
    assert all(rig.sub(rig.library / n).status is SubtitleStatus.FOUND for n in rig.resolver.names)


def test_the_limit_counts_files_attempted(rig: Rig) -> None:
    for n, name in enumerate(("a.mkv", "b.mkv", "c.mkv")):
        rig.add(name, est=100 - n)
    summary = rig.run(limit=2)
    assert rig.resolver.names == ["a.mkv", "b.mkv"]
    assert summary.stopped_reason == LIMIT_REACHED
    assert summary.resolved_found == 2
    assert rig.sub(rig.library / "c.mkv").status is SubtitleStatus.WANTED


def test_a_file_that_cannot_be_probed_does_not_use_the_limit(rig: Rig) -> None:
    rig.add("a.mkv", est=3)
    rig.add("b.mkv", est=2)
    rig.add("c.mkv", est=1)
    rig.probe_errors["a.mkv"] = ProbeError(rig.library / "a.mkv", "ffprobe exited with code 1")
    summary = rig.run(limit=1)
    assert rig.resolver.names == ["b.mkv"]
    assert summary.skipped == 1 and summary.resolved_found == 1


def test_only_limits_the_job_to_a_file_a_group_or_a_folder(rig: Rig) -> None:
    one = rig.add("one.mkv", est=5)
    first = rig.library / "Two cd1.mkv"
    second = rig.library / "Two cd2.mkv"
    rig.add("Two cd1.mkv", decision="remux", parts=(first, second), est=4)
    rig.add("Two cd2.mkv", decision="leave_alone", subtitle=None, reasons=("part of group x",))
    rig.add("films/deep.mkv", est=3)
    assert rig.run(only=one).resolved_found == 1
    assert rig.resolver.names == ["one.mkv"]
    rig.resolver.calls.clear()
    rig.run(only=second)  # a member part selects its group
    assert rig.resolver.names == ["Two cd1.mkv"]
    rig.resolver.calls.clear()
    rig.run(only=rig.library / "films")  # a folder selects everything under it
    assert rig.resolver.names == ["deep.mkv"]


def test_orphan_completed_and_run_decided_rows_are_skipped_and_counted(rig: Rig) -> None:
    done = rig.add("done.mkv", est=9)
    rig.state.set_status(done, Status.COMPLETED, "delivered")
    decided = rig.add("decided.mkv", est=8)
    rig.state.set_status(decided, Status.LEFT_ALONE, "output not smaller")
    busy = rig.add("busy.mkv", est=7)
    rig.state.set_status(busy, Status.ENCODING)
    rig.add("live.mkv", est=1)
    # A WANTED row with no file record at all, and one for a size that is not current.
    ghost = rig.library / "ghost.mkv"
    rig.state.ensure_wanted(ghost, 1, MTIME, english_audio=False)
    rig.state.ensure_wanted(rig.library / "live.mkv", 12345, MTIME, english_audio=False)
    summary = rig.run()
    assert rig.resolver.names == ["live.mkv"]
    assert summary.skipped == 5
    assert summary.resolved_found == 1
    assert rig.probed == ["live.mkv"]


def test_a_changed_or_unreadable_file_is_skipped_not_resolved(rig: Rig) -> None:
    rig.add("changed.mkv", est=3)
    rig.add("broken.mkv", est=2)
    rig.add("fine.mkv", est=1)
    rig.probe_errors["broken.mkv"] = ProbeError(rig.library / "broken.mkv", "not a media file")

    def probe(path: Path, tools: Tools, reporter: Reporter) -> MediaInfo:
        info = rig.probe(path, tools, reporter)
        # The file on disk no longer matches what the survey recorded.
        return replace(info, size=7) if path.name == "changed.mkv" else info

    summary = rig.run(probe_fn=probe)
    assert rig.resolver.names == ["fine.mkv"]
    assert summary.skipped == 2


def test_a_missing_ffprobe_stops_the_job(rig: Rig) -> None:
    rig.add("a.mkv")
    rig.probe_errors["a.mkv"] = ToolNotFound("ffprobe", "not on PATH")
    with pytest.raises(ToolNotFound):
        rig.run()
    assert rig.sub(rig.library / "a.mkv").status is SubtitleStatus.WANTED


def test_the_align_work_folders_are_swept_once_at_the_start(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    swept: list[Path] = []

    def fake_sweep(folder: Path) -> int:
        swept.append(folder)
        return 2

    monkeypatch.setattr(job_module, "sweep_align_workdirs", fake_sweep)
    rig.add("a.mkv")
    rig.add("b.mkv")
    with watch() as lines:
        rig.run()
    assert swept == [rig.settings.subtitle_cache_dir]
    assert any("2" in line and "align" in line for line in lines)


def test_a_cache_folder_inside_the_library_never_starts(rig: Rig) -> None:
    rig.add("a.mkv")
    bad = replace(rig.settings, subtitle_cache_dir=rig.library / "subs")
    ctx = replace(rig.ctx(), settings=bad)
    summary = run_resolve_job(
        bad,
        rig.state,
        ctx,
        limit=None,
        deadline=None,
        only=None,
        resolve=rig.resolver,
        probe_fn=rig.probe,
    )
    assert summary.stopped_reason.startswith("cannot start")
    assert rig.resolver.calls == []


# --- deadline ------------------------------------------------------------------------------


def test_the_deadline_stops_before_the_next_file(rig: Rig) -> None:
    rig.add("a.mkv", est=2)
    rig.add("b.mkv", est=1)
    deadline = datetime(2030, 1, 1, 7, 0)

    def late(rec: FileRecord, sub: SubtitleRecord) -> SubtitleRecord:
        rig.clock_now = deadline + timedelta(minutes=1)  # the first file ran past the deadline
        return rig.resolver.found(sub)

    rig.resolver.script["a.mkv"] = late
    summary = rig.run(deadline=deadline)
    assert rig.resolver.names == ["a.mkv"]  # the file in progress finished, none started after
    assert summary.stopped_reason == DEADLINE_REACHED
    assert rig.sub(rig.library / "b.mkv").status is SubtitleStatus.WANTED


def test_an_aware_deadline_is_compared_with_the_context_clock(rig: Rig) -> None:
    rig.add("a.mkv")
    summary = rig.run(deadline=NOW - timedelta(seconds=1))
    assert summary.stopped_reason == DEADLINE_REACHED
    assert rig.resolver.calls == []


# --- quota ---------------------------------------------------------------------------------


def test_a_spent_allowance_stops_the_job_before_any_file(rig: Rig) -> None:
    rig.add("a.mkv", est=2)
    rig.add("b.mkv", est=1)
    rig.state.set_quota(QuotaInfo(0, MIDNIGHT))
    summary = rig.run()
    assert rig.resolver.calls == [] and rig.probed == []
    assert summary == JobSummary(0, 0, 2, 0, "quota exhausted until 2030-01-02 00:00 UTC")


def test_a_renewed_allowance_does_not_stop_the_job(rig: Rig) -> None:
    rig.add("a.mkv")
    rig.state.set_quota(QuotaInfo(0, NOW - timedelta(hours=1)))
    assert rig.run().resolved_found == 1


def test_without_a_client_a_spent_allowance_does_not_matter(rig: Rig) -> None:
    rig.add("a.mkv")
    rig.state.set_quota(QuotaInfo(0, MIDNIGHT))
    summary = rig.run(ctx=rig.ctx(None))
    assert summary.resolved_found == 1 and summary.stopped_reason == ""
    assert rig.resolver.calls[0].has_client is False


def test_quota_stop_keeps_the_record_stores_the_reset_and_defers_the_rest(rig: Rig) -> None:
    rig.add("a.mkv", est=3)
    rig.add("b.mkv", est=2)
    rig.add("c.mkv", est=1)
    b = rig.library / "b.mkv"
    partial = still_wanted(rig.sub(b), "OpenSubtitles quota used up", (11, 12))
    rig.resolver.script["b.mkv"] = QuotaStop(partial, MIDNIGHT)
    summary = rig.run()
    assert rig.resolver.names == ["a.mkv", "b.mkv"]
    assert summary == JobSummary(1, 0, 2, 0, "quota exhausted until 2030-01-02 00:00 UTC")
    stored = rig.sub(b)
    assert (stored.status, stored.tried_ids, stored.attempts) == (
        SubtitleStatus.WANTED,
        (11, 12),
        2,
    )
    assert rig.state.quota() == QuotaInfo(0, MIDNIGHT)
    assert rig.sub(rig.library / "c.mkv").status is SubtitleStatus.WANTED


def test_quota_stop_without_a_reset_time_assumes_the_next_midnight_utc(rig: Rig) -> None:
    rig.add("a.mkv")
    a = rig.library / "a.mkv"
    rig.resolver.script["a.mkv"] = QuotaStop(still_wanted(rig.sub(a), "quota"), None)
    summary = rig.run()
    assert rig.state.quota() == QuotaInfo(0, MIDNIGHT)
    assert summary.stopped_reason == "quota exhausted until 2030-01-02 00:00 UTC"


@pytest.mark.parametrize(
    ("reset", "expected"),
    [
        (NOW - timedelta(days=1), MIDNIGHT),  # not in the future: the next midnight
        (NOW, MIDNIGHT),
        (MIDNIGHT + timedelta(minutes=30), MIDNIGHT + timedelta(minutes=30)),  # believable
        (MIDNIGHT + timedelta(hours=1), MIDNIGHT + timedelta(hours=1)),  # the limit itself
        (MIDNIGHT + timedelta(hours=1, minutes=1), MIDNIGHT + timedelta(hours=1)),
        (datetime(2999, 1, 1, tzinfo=UTC), MIDNIGHT + timedelta(hours=1)),
    ],
    ids=["past", "now", "soon", "limit", "just-over", "far"],
)
def test_a_reset_time_is_clamped_to_an_hour_past_the_next_midnight(
    rig: Rig, reset: datetime, expected: datetime
) -> None:
    rig.add("a.mkv")
    a = rig.library / "a.mkv"
    rig.resolver.script["a.mkv"] = QuotaStop(still_wanted(rig.sub(a), "q"), reset)
    summary = rig.run()
    assert rig.state.quota() == QuotaInfo(0, expected)
    assert summary.stopped_reason == f"quota exhausted until {expected:%Y-%m-%d %H:%M} UTC"


def test_the_clients_allowance_is_stored_after_each_file_and_clamped(rig: Rig) -> None:
    assert rig.client is not None
    rig.add("a.mkv", est=2)
    rig.add("b.mkv", est=1)
    client = rig.client

    def downloaded(
        left: int, reset: datetime
    ) -> Callable[[FileRecord, SubtitleRecord], SubtitleRecord]:
        def outcome(rec: FileRecord, sub: SubtitleRecord) -> SubtitleRecord:
            client.quota = QuotaInfo(left, reset)
            return rig.resolver.found(sub)

        return outcome

    rig.resolver.script["a.mkv"] = downloaded(7, MIDNIGHT)
    rig.resolver.script["b.mkv"] = downloaded(6, datetime(2999, 1, 1, tzinfo=UTC))
    rig.run()
    # A far-away reset is not kept: at most an hour past the next midnight.
    assert rig.state.quota() == QuotaInfo(6, MIDNIGHT + timedelta(hours=1))


def test_a_stops_allowance_wins_over_the_clients_older_report(rig: Rig) -> None:
    assert rig.client is not None
    rig.add("a.mkv")
    a = rig.library / "a.mkv"
    client = rig.client

    def downloaded_then_refused(rec: FileRecord, sub: SubtitleRecord) -> SubtitleRecord:
        client.quota = QuotaInfo(5, MIDNIGHT)  # a download reported 5 left ...
        raise QuotaStop(still_wanted(rig.sub(a), "q", (3,)), MIDNIGHT)  # ... then the refusal

    rig.resolver.script["a.mkv"] = downloaded_then_refused
    rig.run()
    assert rig.state.quota() == QuotaInfo(0, MIDNIGHT)


def test_the_subtitle_row_is_written_before_the_allowance(rig: Rig) -> None:
    assert rig.client is not None
    rig.add("a.mkv")
    client = rig.client

    def downloaded(rec: FileRecord, sub: SubtitleRecord) -> SubtitleRecord:
        client.quota = QuotaInfo(4, MIDNIGHT)
        return rig.resolver.found(sub)

    rig.resolver.script["a.mkv"] = downloaded
    rig.state.log.clear()
    rig.run()
    assert rig.state.log == ["subtitle a.mkv found", "quota 4"]


def test_a_stop_persists_its_record_before_the_allowance(rig: Rig) -> None:
    rig.add("a.mkv")
    a = rig.library / "a.mkv"
    rig.resolver.script["a.mkv"] = QuotaStop(still_wanted(rig.sub(a), "q", (3,)), MIDNIGHT)
    rig.state.log.clear()
    rig.run()
    assert rig.state.log == ["subtitle a.mkv wanted", "quota 0"]


def test_the_allowance_is_stored_even_when_the_resolver_fails_after_a_download(rig: Rig) -> None:
    assert rig.client is not None
    rig.add("a.mkv")
    client = rig.client

    def boom(rec: FileRecord, sub: SubtitleRecord) -> SubtitleRecord:
        client.quota = QuotaInfo(3, MIDNIGHT)
        raise ToolNotFound("ffsubsync", "not installed")

    rig.resolver.script["a.mkv"] = boom
    with pytest.raises(ToolNotFound):
        rig.run()
    assert rig.state.quota() == QuotaInfo(3, MIDNIGHT)
    assert rig.sub(rig.library / "a.mkv").status is SubtitleStatus.WANTED


# --- failures ------------------------------------------------------------------------------


def test_an_unwritable_cache_stops_the_job_after_the_record_is_kept(rig: Rig) -> None:
    rig.add("a.mkv", est=2)
    rig.add("b.mkv", est=1)
    a = rig.library / "a.mkv"
    why = "the subtitle cache folder X cannot be written (PermissionError)"
    rig.resolver.script["a.mkv"] = CacheUnwritable(still_wanted(rig.sub(a), why, (9,)), why)
    summary = rig.run()
    assert rig.resolver.names == ["a.mkv"]
    assert summary.stopped_reason == f"cache folder is not writable: {why}"
    assert rig.sub(a).tried_ids == (9,)


def test_a_refused_login_turns_the_online_step_off_for_the_rest_of_the_job(rig: Rig) -> None:
    rig.add("a.mkv", est=3)
    rig.add("b.mkv", est=2)
    rig.add("c.mkv", est=1)
    a = rig.library / "a.mkv"
    reason = "OpenSubtitles refused the login: bad key"
    rig.resolver.script["a.mkv"] = OnlineFailure(still_wanted(rig.sub(a), reason), reason, True)
    with watch(logging.WARNING) as lines:
        summary = rig.run()
    assert [c.has_client for c in rig.resolver.calls] == [True, False, False]
    assert summary.resolved_found == 2 and summary.stopped_reason == ""
    assert rig.sub(a).reason == reason
    warned = [line for line in lines if "online lookups are off" in line]
    assert len(warned) == 1 and warned[0].startswith("WARNING")


def test_after_a_refused_login_later_files_are_not_told_there_are_no_credentials(
    rig: Rig,
) -> None:
    from video_cleaner.subtitles.resolve import LOGIN_REFUSED, NO_CREDENTIALS

    rig.add("a.mkv", est=3)
    rig.add("b.mkv", est=2)
    a = rig.library / "a.mkv"
    reason = "OpenSubtitles refused the login: bad key"
    rig.resolver.script["a.mkv"] = OnlineFailure(still_wanted(rig.sub(a), reason), reason, True)
    rig.run()
    assert [c.has_client for c in rig.resolver.calls] == [True, False]
    assert rig.resolver.calls[0].offline_reason == NO_CREDENTIALS  # the default, never used
    assert rig.resolver.calls[1].offline_reason == LOGIN_REFUSED
    assert "refused" in LOGIN_REFUSED and "credentials" not in LOGIN_REFUSED


def test_the_real_resolver_stores_the_refused_login_reason_and_keeps_the_row_wanted(
    rig: Rig,
) -> None:
    from video_cleaner.subtitles.resolve import LOGIN_REFUSED

    rig.add("a.mkv", est=3)
    rig.add("b.mkv", est=2)
    a = rig.library / "a.mkv"
    reason = "OpenSubtitles refused the login: bad key"
    first = OnlineFailure(still_wanted(rig.sub(a), reason), reason, True)

    def resolve(
        rec: FileRecord, info: MediaInfo, h: str | None, sub: SubtitleRecord, ctx: Any
    ) -> Any:
        if rec.path.name == "a.mkv":
            raise first
        return resolve_subtitle(rec, info, h, sub, ctx)

    rig.run(resolve=resolve)
    b = rig.sub(rig.library / "b.mkv")
    assert (b.status, b.reason) == (SubtitleStatus.WANTED, LOGIN_REFUSED)


def failure(rig: Rig, name: str, fatal: bool = False) -> OnlineFailure:
    sub = rig.sub(rig.library / name)
    return OnlineFailure(still_wanted(sub, "OpenSubtitles failed: HTTP 503"), "503", fatal)


def test_three_failures_in_a_row_stop_the_job(rig: Rig) -> None:
    for n, name in enumerate(("a.mkv", "b.mkv", "c.mkv", "d.mkv")):
        rig.add(name, est=10 - n)
    for name in ("a.mkv", "b.mkv", "c.mkv"):
        rig.resolver.script[name] = failure(rig, name)
    summary = rig.run()
    assert rig.resolver.names == ["a.mkv", "b.mkv", "c.mkv"]
    assert summary.stopped_reason == NOT_ANSWERING
    assert rig.sub(rig.library / "d.mkv").status is SubtitleStatus.WANTED
    assert rig.sub(rig.library / "a.mkv").reason == "OpenSubtitles failed: HTTP 503"


def test_an_online_success_between_failures_resets_the_count(rig: Rig) -> None:
    names = ("a.mkv", "b.mkv", "c.mkv", "d.mkv", "e.mkv")
    for n, name in enumerate(names):
        rig.add(name, est=10 - n)
    for name in ("a.mkv", "b.mkv", "d.mkv", "e.mkv"):
        rig.resolver.script[name] = failure(rig, name)
    summary = rig.run()
    assert rig.resolver.names == list(names)
    assert summary.stopped_reason == ""


def test_a_local_find_does_not_reset_the_count_of_failures(rig: Rig) -> None:
    names = ("a.mkv", "b.mkv", "c.mkv", "d.mkv")
    for n, name in enumerate(names):
        rig.add(name, est=10 - n)
    for name in ("a.mkv", "b.mkv", "d.mkv"):
        rig.resolver.script[name] = failure(rig, name)
    c = rig.library / "c.mkv"
    rig.resolver.script["c.mkv"] = rig.resolver.found(rig.sub(c), "sidecar")
    assert rig.run().stopped_reason == NOT_ANSWERING


def test_an_unexpected_error_in_one_file_does_not_stop_the_job(rig: Rig) -> None:
    rig.add("a.mkv", est=2)
    rig.add("b.mkv", est=1)
    rig.resolver.script["a.mkv"] = RuntimeError("bug in the parser")
    with watch(logging.ERROR) as lines:
        summary = rig.run()
    assert rig.resolver.names == ["a.mkv", "b.mkv"]
    assert summary.skipped == 1 and summary.resolved_found == 1
    assert rig.sub(rig.library / "a.mkv").status is SubtitleStatus.WANTED
    assert any("RuntimeError" in line for line in lines)


def test_ctrl_c_stops_cleanly_and_keeps_what_was_done(rig: Rig) -> None:
    assert rig.client is not None
    rig.add("a.mkv", est=3)
    rig.add("b.mkv", est=2)
    rig.add("c.mkv", est=1)
    client = rig.client

    def first(rec: FileRecord, sub: SubtitleRecord) -> SubtitleRecord:
        client.quota = QuotaInfo(8, MIDNIGHT)
        return rig.resolver.found(sub)

    def interrupted(rec: FileRecord, sub: SubtitleRecord) -> SubtitleRecord:
        client.quota = QuotaInfo(7, MIDNIGHT)  # the download went through, then Ctrl+C
        raise KeyboardInterrupt

    rig.resolver.script["a.mkv"] = first
    rig.resolver.script["b.mkv"] = interrupted
    summary = rig.run()
    assert summary.stopped_reason == INTERRUPTED
    assert summary.resolved_found == 1
    assert rig.sub(rig.library / "a.mkv").status is SubtitleStatus.FOUND
    assert rig.sub(rig.library / "b.mkv").status is SubtitleStatus.WANTED
    assert rig.sub(rig.library / "c.mkv").status is SubtitleStatus.WANTED
    assert rig.state.quota() == QuotaInfo(7, MIDNIGHT)


def test_a_malformed_stored_hash_is_not_sent_to_the_resolver(rig: Rig) -> None:
    rig.add("a.mkv", hash_value="not-a-hash")
    seen: list[str | None] = []

    def spy(
        rec: FileRecord, info: MediaInfo, h: str | None, sub: SubtitleRecord, ctx: ResolveContext
    ) -> SubtitleRecord:
        seen.append(h)
        return rig.resolver.found(sub)

    rig.run(resolve=spy)
    assert seen == [None]


# --- stale FOUND rows ----------------------------------------------------------------------


def test_persist_downgrades_a_found_row_before_a_weaker_outcome_is_stored(rig: Rig) -> None:
    path = rig.add("a.mkv", subtitle=None)
    found = SubtitleRecord(
        path,
        6 * GIB,
        MTIME,
        SubtitleStatus.FOUND,
        provider="sidecar",
        cache_path=rig.tmp_path / "cache" / "x.srt",
        sha256="cd" * 32,
        tried_ids=(7,),
    )
    rig.state.set_subtitle(found)
    after = replace(
        unavailable(found, (7, 9)), provider="", cache_path=None, sha256="", provenance=""
    )
    rig.state.log.clear()
    persist_subtitle(rig.state, found, after, "cache gone")
    assert rig.state.log == ["downgrade a.mkv", "subtitle a.mkv unavailable"]
    stored = rig.sub(path)
    assert stored.status is SubtitleStatus.UNAVAILABLE
    assert stored.tried_ids == (7, 9) and stored.cache_path is None and stored.sha256 == ""


def test_persist_does_not_downgrade_a_row_that_is_not_found(rig: Rig) -> None:
    path = rig.add("a.mkv")
    before = rig.sub(path)
    rig.state.log.clear()
    persist_subtitle(rig.state, before, unavailable(before), "")
    assert rig.state.log == ["subtitle a.mkv unavailable"]


def test_a_stale_found_row_that_fails_to_resolve_keeps_its_tried_ids_and_is_not_asked_again(
    rig: Rig,
) -> None:
    video = rig.library / "The Great Movie (2010).mkv"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"not a real video")
    size = video.stat().st_size
    rig.add(video.name, size=size, subtitle=None)
    wanted = SubtitleRecord(video, size, MTIME, SubtitleStatus.WANTED, tried_ids=(7,))
    rig.state.set_subtitle(wanted)
    found = SubtitleRecord(
        video,
        size,
        MTIME,
        SubtitleStatus.FOUND,
        provider="opensubtitles",
        cache_path=rig.tmp_path / "cache" / "gone.srt",
        sha256="ab" * 32,
        tried_ids=(7,),
    )

    def flip(path: Path, tools: Tools, reporter: Reporter) -> MediaInfo:
        rig.state.set_subtitle(found)  # another process stored a FOUND row after the listing
        return rig.info(path)

    first = FakeClient(hash_results=[result(1001)], downloads={1001: ProviderError("HTTP 500")})
    summary = rig.run(resolve=resolve_subtitle, probe_fn=flip, ctx=rig.ctx(first))
    assert first.download_calls == [1001]
    assert summary.stopped_reason == ""
    stored = rig.state.subtitle(video, size, MTIME)
    assert stored is not None
    assert stored.status is SubtitleStatus.WANTED  # downgraded first, so the weaker row was stored
    assert stored.tried_ids == (7, 1001)
    assert stored.cache_path is None

    second = FakeClient(
        hash_results=[result(1001), result(1002, hash_match=True)],
        downloads={1002: srt_bytes(30)},
    )
    again = rig.run(resolve=resolve_subtitle, ctx=rig.ctx(second))
    assert second.download_calls == [1002]  # the id that failed is never asked for again
    assert again.resolved_found == 1


# --- re-deciding what turned FOUND ---------------------------------------------------------


def test_a_left_alone_file_that_found_a_subtitle_becomes_a_pending_remux(rig: Rig) -> None:
    path = rig.add("hevc.mkv", kind="hevc", decision="leave_alone", reasons=("already hevc",))
    assert rig.file(path).status is Status.LEFT_ALONE
    rig.run()
    rec = rig.file(path)
    assert (rec.decision, rec.status) == ("remux", Status.PENDING)
    assert "new subtitles to mux in" in rec.reasons
    assert rec.movie_hash == HASH


def test_a_flagged_hdr_file_that_found_a_subtitle_becomes_a_remux(rig: Rig) -> None:
    path = rig.add("hdr.mkv", kind="hdr", decision="flag", reasons=("HDR (transfer smpte2084)",))
    rig.run()
    rec = rig.file(path)
    assert (rec.decision, rec.status) == ("remux", Status.PENDING)
    assert "video left as is" in rec.reasons


def test_an_encode_stays_an_encode(rig: Rig) -> None:
    path = rig.add("enc.mkv", est=100)
    rig.run()
    rec = rig.file(path)
    assert (rec.decision, rec.status) == ("encode", Status.PENDING)
    assert rec.reasons[-1] == "new subtitles to mux in"


def test_a_multi_part_group_keeps_its_decision(rig: Rig) -> None:
    first = rig.library / "Film cd1.mkv"
    second = rig.library / "Film cd2.mkv"
    rig.add(
        "Film cd1.mkv",
        kind="hevc",
        decision="remux",
        parts=(first, second),
        reasons=("multi-part file to join",),
    )
    rig.add(
        "Film cd2.mkv", decision="leave_alone", subtitle=None, reasons=("part of group Film.mkv",)
    )
    rig.run()
    rec = rig.file(first)
    assert (rec.decision, rec.reasons, rec.parts) == (
        "remux",
        ("multi-part file to join",),
        (first, second),
    )
    assert rig.sub(first).status is SubtitleStatus.FOUND


def test_a_file_that_did_not_find_one_is_not_touched(rig: Rig) -> None:
    path = rig.add("hevc.mkv", kind="hevc", decision="leave_alone", reasons=("already hevc",))
    rig.resolver.script["hevc.mkv"] = unavailable(rig.sub(path))
    summary = rig.run()
    rec = rig.file(path)
    assert (rec.decision, rec.status, rec.reasons) == (
        "leave_alone",
        Status.LEFT_ALONE,
        ("already hevc",),
    )
    assert summary == JobSummary(0, 1, 0, 0, "")
    assert rig.sub(path).tried_ids == (5,)


def test_a_multi_part_film_that_cannot_be_looked_up_counts_as_unavailable(rig: Rig) -> None:
    first = rig.library / "Film cd1.mkv"
    rig.add("Film cd1.mkv", decision="remux", parts=(first, rig.library / "Film cd2.mkv"))
    sub = rig.sub(first)
    rig.resolver.script["Film cd1.mkv"] = replace(
        sub,
        status=SubtitleStatus.UNCLEAR,
        reason="multi-part film: subtitle lookup needs the parts combined",
    )
    assert rig.run() == JobSummary(0, 1, 0, 0, "")


# --- progress lines ------------------------------------------------------------------------


def test_progress_lines_name_the_file_the_source_the_outcome_and_the_allowance(rig: Rig) -> None:
    assert rig.client is not None
    rig.add("a.mkv", est=3, kind="hevc", decision="leave_alone", reasons=("already hevc",))
    rig.add("b.mkv", est=2)
    rig.add("c.mkv", est=1)
    b, c = rig.library / "b.mkv", rig.library / "c.mkv"
    client = rig.client

    def downloaded(rec: FileRecord, sub: SubtitleRecord) -> SubtitleRecord:
        client.quota = QuotaInfo(17, MIDNIGHT)
        return rig.resolver.found(sub)

    rig.resolver.script["a.mkv"] = rig.resolver.found(rig.sub(rig.library / "a.mkv"), "sidecar")
    rig.resolver.script["b.mkv"] = downloaded
    rig.resolver.script["c.mkv"] = unavailable(rig.sub(c))
    rig.run()
    done = rig.reporter.done
    assert len(done) == 3
    assert done[0].startswith("[1/3] a.mkv: sidecar")
    assert "found" in done[0] and "remux" in done[0]
    assert done[1].startswith("[2/3] b.mkv: OpenSubtitles")
    assert "OpenSubtitles: 17 downloads left" in done[1]
    assert "OpenSubtitles: 17 downloads left" not in done[0]
    assert done[2].startswith("[3/3] c.mkv: no subtitle found")
    assert "unavailable" in done[2]
    assert [t for t in rig.reporter.stages if t.startswith("[")] == [
        "[1/3] a.mkv",
        "[2/3] b.mkv",
        "[3/3] c.mkv",
    ]
    assert b.name in done[1]


def test_a_hostile_file_name_cannot_inject_terminal_codes_into_a_progress_line(rig: Rig) -> None:
    rig.add("evil\x1b[31m‮.mkv")
    rig.run()
    (line,) = rig.reporter.done
    assert "\x1b" not in line and "‮" not in line
    assert "evil" in line


def test_stops_are_reported_as_warnings(rig: Rig) -> None:
    rig.add("a.mkv")
    a = rig.library / "a.mkv"
    rig.resolver.script["a.mkv"] = QuotaStop(still_wanted(rig.sub(a), "q"), MIDNIGHT)
    rig.run()
    assert rig.reporter.done_levels[-1] is Level.WARN
    assert "allowance" in rig.reporter.done[-1]


# --- the real resolver, no credentials -----------------------------------------------------


def test_without_credentials_a_sidecar_is_still_found_and_the_file_is_re_decided(rig: Rig) -> None:
    video = rig.library / "The Great Movie (2010).mkv"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"not a real video")
    (rig.library / "The Great Movie (2010).srt").write_bytes(srt_bytes(30))
    size = video.stat().st_size
    rig.add(video.name, size=size, kind="hevc", decision="leave_alone", reasons=("already hevc",))
    infos = rig.info(video)
    assert infos.duration_s == 100.0
    summary = rig.run(resolve=resolve_subtitle, ctx=rig.ctx(None))
    assert summary == JobSummary(1, 0, 0, 0, "")
    sub = rig.state.subtitle(video, size, MTIME)
    assert sub is not None and sub.status is SubtitleStatus.FOUND and sub.provider == "sidecar"
    assert sub.cache_path is not None and sub.cache_path.is_file()
    assert rig.file(video).decision == "remux"
    assert any("sidecar" in line for line in rig.reporter.done)


# --- a cached subtitle that went missing or changed is looked up again -----------------------


def _found_row(rig: Rig, name: str, data: bytes, *, on_disk: bytes | None) -> Path:
    """A FOUND row for *name* whose cache file holds *on_disk* (None: no file) and whose
    stored hash is that of *data*."""
    import hashlib

    path = rig.add(name, subtitle=None, est=10)
    rec = rig.file(path)
    cache = rig.settings.subtitle_cache_dir / f"{path.stem}.srt"
    if on_disk is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(on_disk)
    rig.state.set_subtitle(
        SubtitleRecord(
            path=path,
            size=rec.size,
            mtime=rec.mtime,
            status=SubtitleStatus.FOUND,
            provider="opensubtitles",
            cache_path=cache,
            sha256=hashlib.sha256(data).hexdigest(),
            provenance="English (OpenSubtitles)",
            tried_ids=(7,),
            attempts=1,
        )
    )
    return path


def test_found_rows_whose_cache_file_is_missing_or_changed_are_looked_up_again(rig: Rig) -> None:
    good = _found_row(rig, "good.mkv", b"one", on_disk=b"one")
    gone = _found_row(rig, "gone.mkv", b"two", on_disk=None)
    changed = _found_row(rig, "changed.mkv", b"three", on_disk=b"tampered")
    kept = rig.sub(good)
    summary = rig.run()
    assert sorted(rig.resolver.names) == ["changed.mkv", "gone.mkv"]
    assert [c.status for c in rig.resolver.calls] == [SubtitleStatus.WANTED] * 2
    assert summary.resolved_found == 2
    assert rig.sub(good) == kept  # untouched
    lines = [line for line in rig.reporter.done if "cached subtitle" in line]
    assert lines == ["2 cached subtitle(s) missing or changed: looking them up again"]
    assert rig.state.log.count(f"downgrade {gone.name}") == 1
    assert rig.state.log.count(f"downgrade {changed.name}") == 1


def test_no_line_is_printed_when_every_cached_subtitle_is_intact(rig: Rig) -> None:
    _found_row(rig, "good.mkv", b"one", on_disk=b"one")
    rig.run()
    assert rig.resolver.names == []
    assert not [line for line in rig.reporter.done if "cached subtitle" in line]


def test_the_cache_check_stays_within_the_path_scope_and_skips_delivered_files(rig: Rig) -> None:
    inside = _found_row(rig, "in.mkv", b"x", on_disk=None)
    outside = _found_row(rig, "out.mkv", b"y", on_disk=None)
    delivered = _found_row(rig, "done.mkv", b"z", on_disk=None)
    rig.state.set_status(delivered, Status.COMPLETED, "delivered")
    rig.run(only=inside)
    assert rig.resolver.names == ["in.mkv"]
    assert rig.sub(outside).status is SubtitleStatus.FOUND
    assert rig.sub(delivered).status is SubtitleStatus.FOUND


def test_a_reset_time_is_shown_in_utc() -> None:
    plus_two = timezone(timedelta(hours=2))
    assert job_module._format_reset(datetime(2030, 1, 2, 1, 30, tzinfo=plus_two)) == (
        "2030-01-01 23:30 UTC"
    )
