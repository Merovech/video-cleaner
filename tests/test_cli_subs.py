"""The `subs resolve|status|retry` commands (the job itself is faked: see subtitles/test_job.py)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from helpers import MTIME, add_record, write_config
from video_cleaner import credentials
from video_cleaner.cli import app
from video_cleaner.credentials import CredentialError
from video_cleaner.instancelock import instance_lock, lock_path_for, subs_lock_path_for
from video_cleaner.state import State, SubtitleRecord, SubtitleStatus
from video_cleaner.subtitles import align
from video_cleaner.subtitles import job as job_module
from video_cleaner.subtitles.job import (
    DEADLINE_REACHED,
    INTERRUPTED,
    LIMIT_REACHED,
    NOT_ANSWERING,
    JobSummary,
)
from video_cleaner.subtitles.models import QuotaInfo
from video_cleaner.subtitles.opensubtitles import OpenSubtitlesClient
from video_cleaner.subtitles.resolve import ResolveContext
from video_cleaner.tools import ToolNotFound, Tools

TOOLS = Tools(Path("ffmpeg"), Path("ffprobe"), Path("mkvmerge"))
DONE = JobSummary(2, 1, 0, 0, "")
LOGIN = {
    "VC_OPENSUBTITLES_API_KEY": "key-123",
    "VC_OPENSUBTITLES_USERNAME": "someone",
    "VC_OPENSUBTITLES_PASSWORD": "hunter2-secret",
}


class EmptyStore:
    def get(self, name: str) -> str | None:
        return None

    def set(self, name: str, value: str) -> None:
        raise AssertionError("nothing is stored by subs resolve")

    def delete(self, name: str) -> None:
        raise AssertionError("nothing is deleted by subs resolve")


class FakeJob:
    """Replaces `run_resolve_job`; remembers how it was called."""

    def __init__(
        self,
        outcome: JobSummary | BaseException = DONE,
        effect: Callable[[State], None] | None = None,
    ) -> None:
        self.outcome = outcome
        self.effect = effect  # what the job did to the state (it changes the plan hash)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, settings: Any, state: State, ctx: ResolveContext, **kwargs: Any
    ) -> JobSummary:
        self.calls.append({"settings": settings, "state": state, "ctx": ctx, **kwargs})
        if self.effect is not None:
            self.effect(state)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


@pytest.fixture(autouse=True)
def _no_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in LOGIN:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(credentials, "open_store", lambda env=None: EmptyStore())
    monkeypatch.setattr("video_cleaner.cli.locate_tools", lambda: TOOLS)
    monkeypatch.setattr(align, "probe_ffsubsync", lambda *a, **k: align.ProbeResult(True, "runs"))


def _invoke(*args: str, env: dict[str, str] | None = None) -> Any:
    return CliRunner().invoke(app, list(args), env=env)


def _setup(tmp_path: Path) -> Path:
    cfg = write_config(tmp_path)
    with State.open(tmp_path / "appdata" / "state.db"):
        pass
    return cfg


def _resolve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: JobSummary | BaseException = DONE,
    *args: str,
    env: dict[str, str] | None = None,
) -> tuple[Any, FakeJob]:
    cfg = _setup(tmp_path)
    fake = FakeJob(outcome)
    monkeypatch.setattr(job_module, "run_resolve_job", fake)
    result = _invoke("subs", "resolve", "--config", str(cfg), "-q", *args, env=env)
    return result, fake


# --- the command group -----------------------------------------------------------------------


def test_subs_is_listed_with_its_three_commands() -> None:
    top = CliRunner().invoke(app, ["--help"]).stdout
    assert "subs" in top
    sub = CliRunner().invoke(app, ["subs", "--help"]).stdout
    for name in ("resolve", "status", "retry"):
        assert name in sub


@pytest.mark.parametrize("args", [["resolve"], ["status"], ["retry", "--path", "x.mkv"]])
def test_a_missing_config_exits_2(tmp_path: Path, args: list[str]) -> None:
    result = _invoke("subs", *args, "--config", str(tmp_path / "missing.toml"))
    assert result.exit_code == 2
    assert "config" in result.output


def test_the_two_locks_are_different_files(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    assert subs_lock_path_for(db) != lock_path_for(db)
    assert subs_lock_path_for(db).parent == lock_path_for(db).parent


# --- resolve ---------------------------------------------------------------------------------


def test_resolve_runs_the_job_with_the_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "library" / "Films"
    result, fake = _resolve(
        tmp_path, monkeypatch, DONE, "--limit", "5", "--until", "07:00", "--path", str(target)
    )
    assert result.exit_code == 0, result.output
    (call,) = fake.calls
    assert call["limit"] == 5
    assert call["only"] == target
    assert call["deadline"] is not None and call["deadline"] > datetime.now()
    assert call["deadline"] - datetime.now() <= timedelta(hours=24)
    assert call["ctx"].tools == TOOLS
    assert "found 2" in result.output and "unavailable 1" in result.output
    assert "Finished" in result.output


def test_the_real_job_with_nothing_wanted_finishes(tmp_path: Path) -> None:
    cfg = _setup(tmp_path)
    result = _invoke("subs", "resolve", "--config", str(cfg), "-q")
    assert result.exit_code == 0, result.output
    assert "found 0" in result.output and "Finished" in result.output
    assert "plan --ack" not in result.output


def test_resolve_defaults_have_no_limit_deadline_or_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _result, fake = _resolve(tmp_path, monkeypatch)
    (call,) = fake.calls
    assert (call["limit"], call["deadline"], call["only"]) == (None, None, None)


def test_without_credentials_it_says_so_and_runs_without_a_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, fake = _resolve(tmp_path, monkeypatch)
    assert result.exit_code == 0, result.output
    assert fake.calls[0]["ctx"].client is None
    assert (
        "no OpenSubtitles credentials: sidecar and local folders only; run `auth` to add them"
        in result.output
    )


def test_with_credentials_the_client_is_built_and_no_secret_is_printed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, fake = _resolve(tmp_path, monkeypatch, DONE, env=LOGIN)
    assert result.exit_code == 0, result.output
    assert isinstance(fake.calls[0]["ctx"].client, OpenSubtitlesClient)
    assert "no OpenSubtitles credentials" not in result.output
    for value in LOGIN.values():
        assert value not in result.output


def test_a_key_that_cannot_be_sent_exits_2_without_printing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = {**LOGIN, "VC_OPENSUBTITLES_API_KEY": "bad key\twith space"}
    result, fake = _resolve(tmp_path, monkeypatch, DONE, env=bad)
    assert result.exit_code == 2
    assert fake.calls == []
    assert "credentials" in result.output
    assert "bad key" not in result.output


def test_the_local_folders_are_indexed_for_the_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "subs"
    folder.mkdir()
    (folder / "The Great Movie (2010).srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nhi\n")
    cfg = _setup(tmp_path)
    with cfg.open("a", encoding="utf-8") as fh:
        fh.write(f"local_subtitle_dirs = ['{folder.as_posix()}']\n")
    fake = FakeJob()
    monkeypatch.setattr(job_module, "run_resolve_job", fake)
    result = _invoke("subs", "resolve", "--config", str(cfg), "-q")
    assert result.exit_code == 0, result.output
    assert len(fake.calls[0]["ctx"].local_index) == 1


@pytest.mark.parametrize("error", [CredentialError("no usable keyring"), OSError("denied")])
def test_an_unreadable_credential_store_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def broken(env: Any = None) -> Any:
        raise error

    monkeypatch.setattr(credentials, "open_store", broken)
    result, fake = _resolve(tmp_path, monkeypatch)
    assert result.exit_code == 2
    assert fake.calls == []
    assert "credentials:" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("args", [["--limit", "0"], ["--limit", "-3"], ["--until", "25:00"]])
def test_bad_options_exit_2_before_anything_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    result, fake = _resolve(tmp_path, monkeypatch, DONE, *args)
    assert result.exit_code == 2
    assert fake.calls == []


def test_without_a_state_database_resolve_exits_2(tmp_path: Path) -> None:
    cfg = write_config(tmp_path)
    result = _invoke("subs", "resolve", "--config", str(cfg), "-q")
    assert result.exit_code == 2
    assert "run `survey` first" in result.output
    assert not (tmp_path / "appdata" / "state.db").exists()


def test_a_second_resolve_exits_2_with_the_already_running_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _setup(tmp_path)
    fake = FakeJob()
    monkeypatch.setattr(job_module, "run_resolve_job", fake)
    with instance_lock(subs_lock_path_for(tmp_path / "appdata" / "state.db")):
        result = _invoke("subs", "resolve", "--config", str(cfg), "-q")
    assert result.exit_code == 2
    assert "another `subs resolve` is running" in " ".join(result.output.split())
    assert fake.calls == []
    # Released again once the holder is gone.
    assert _invoke("subs", "resolve", "--config", str(cfg), "-q").exit_code == 0


def test_resolve_may_run_beside_a_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _setup(tmp_path)
    fake = FakeJob()
    monkeypatch.setattr(job_module, "run_resolve_job", fake)
    with instance_lock(lock_path_for(tmp_path / "appdata" / "state.db")):
        result = _invoke("subs", "resolve", "--config", str(cfg), "-q")
    assert result.exit_code == 0, result.output
    assert len(fake.calls) == 1


@pytest.mark.parametrize(
    ("reason", "code"),
    [
        ("", 0),
        (LIMIT_REACHED, 0),
        (DEADLINE_REACHED, 0),
        ("quota exhausted until 2030-01-02 00:00 UTC", 0),
        (NOT_ANSWERING, 1),
        ("cache folder is not writable: the subtitle cache folder X cannot be written", 1),
        ("cannot start: subtitle_cache_dir overlaps the library", 1),
        (INTERRUPTED, 130),
    ],
)
def test_exit_codes_follow_why_the_job_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str, code: int
) -> None:
    result, _ = _resolve(tmp_path, monkeypatch, JobSummary(1, 2, 3, 4, reason))
    assert result.exit_code == code, result.output
    text = " ".join(result.output.split())
    assert "found 1" in text and "unavailable 2" in text
    assert "deferred 3" in text and "skipped 4" in text
    if reason:
        assert reason in text


def _changes_the_plan(tmp_path: Path, status: SubtitleStatus) -> Callable[[State], None]:
    """What a job did: it set the subtitle row of a.mkv to *status* (a plan-visible change)."""
    path = _wanted(tmp_path, "a.mkv")

    def effect(state: State) -> None:
        state.set_subtitle(SubtitleRecord(path, 1000, MTIME, status, reason="x", sha256="ab" * 32))

    return effect


HINT_PLAN = "plan --ack"


@pytest.mark.parametrize(
    ("status", "summary"),
    [
        (SubtitleStatus.FOUND, JobSummary(1, 0, 0, 0, "")),
        (SubtitleStatus.UNAVAILABLE, JobSummary(0, 1, 0, 0, "")),
        (SubtitleStatus.UNCLEAR, JobSummary(0, 1, 0, 0, "")),
    ],
)
def test_the_plan_hint_follows_the_plan_not_the_found_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: SubtitleStatus, summary: JobSummary
) -> None:
    cfg = _setup(tmp_path)
    fake = FakeJob(summary, _changes_the_plan(tmp_path, status))
    monkeypatch.setattr(job_module, "run_resolve_job", fake)
    result = _invoke("subs", "resolve", "--config", str(cfg), "-q")
    assert result.exit_code == 0, result.output
    assert HINT_PLAN in result.output and "The plan changed" in result.output


def test_no_plan_hint_when_the_job_left_the_plan_as_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _setup(tmp_path)
    _wanted(tmp_path, "a.mkv")
    monkeypatch.setattr(job_module, "run_resolve_job", FakeJob(JobSummary(2, 1, 0, 0, "")))
    result = _invoke("subs", "resolve", "--config", str(cfg), "-q")
    assert HINT_PLAN not in result.output  # the fake job changed nothing, whatever it reports


def test_a_reset_time_is_shown_in_utc_by_status() -> None:
    from video_cleaner.cli import _quota_line

    moment = datetime(2030, 1, 2, 1, 30, tzinfo=timezone(timedelta(hours=2)))
    line = _quota_line(QuotaInfo(4, moment), datetime(2030, 1, 1, 0, 0, tzinfo=UTC))
    assert "2030-01-01 23:30 UTC" in line
    renewed = _quota_line(QuotaInfo(0, moment), datetime(2030, 1, 5, 0, 0, tzinfo=UTC))
    assert "2030-01-01 23:30 UTC" in renewed


def test_a_missing_tool_exits_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result, _ = _resolve(tmp_path, monkeypatch, ToolNotFound("ffsubsync", "not installed"))
    assert result.exit_code == 2
    assert "ffsubsync" in result.output
    assert "Traceback" not in result.output


def _broken_ffsubsync(monkeypatch: pytest.MonkeyPatch, result: align.ProbeResult) -> list[object]:
    probes: list[object] = []

    def probe(*args: object, **kwargs: object) -> align.ProbeResult:
        probes.append(args)
        return result

    monkeypatch.setattr(align, "probe_ffsubsync", probe)
    return probes


def test_resolve_refuses_when_ffsubsync_is_not_usable_before_any_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def download(*args: object, **kwargs: object) -> bytes:
        raise AssertionError("a download was made")

    monkeypatch.setattr(OpenSubtitlesClient, "download", download)
    probes = _broken_ffsubsync(
        monkeypatch, align.ProbeResult(False, "ModuleNotFoundError: No module named 'ffsubsync'")
    )
    result, fake = _resolve(tmp_path, monkeypatch, DONE, env=LOGIN)
    assert result.exit_code == 2
    assert len(probes) == 1
    assert fake.calls == []  # the job, which would download, never started
    assert "ffsubsync is not usable" in result.output
    assert "ModuleNotFoundError" in result.output
    assert "uv sync" in result.output
    assert len(result.output.strip().splitlines()) == 1


def test_a_probe_timeout_is_refused_without_the_install_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _broken_ffsubsync(
        monkeypatch, align.ProbeResult(False, "did not answer within 20 s", timed_out=True)
    )
    result, fake = _resolve(tmp_path, monkeypatch)
    assert result.exit_code == 2
    assert fake.calls == []
    assert "did not answer within 20 s" in result.output
    assert "uv sync" not in result.output


def test_resolve_without_credentials_checks_ffsubsync_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Sidecar and local candidates are aligned with ffsubsync as well, so a broken one would
    # reject them all and mark the files unavailable.
    _broken_ffsubsync(monkeypatch, align.ProbeResult(False, "exit 1"))
    result, fake = _resolve(tmp_path, monkeypatch)
    assert result.exit_code == 2
    assert fake.calls == []


def test_resolve_goes_on_when_ffsubsync_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probes = _broken_ffsubsync(monkeypatch, align.ProbeResult(True, "runs"))
    result, fake = _resolve(tmp_path, monkeypatch)
    assert result.exit_code == 0, result.output
    assert len(probes) == 1
    assert len(fake.calls) == 1


def test_ctrl_c_during_the_job_exits_130(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result, _ = _resolve(tmp_path, monkeypatch, KeyboardInterrupt())
    assert result.exit_code == 130
    assert "interrupted" in result.output
    assert "Traceback" not in result.output


# --- status ----------------------------------------------------------------------------------


def _wanted(tmp_path: Path, name: str, **kw: Any) -> Path:
    path = tmp_path / "library" / name
    with State.open(tmp_path / "appdata" / "state.db") as state:
        add_record(state, path, "encode", ("h264",), **kw)
        state.ensure_wanted(path, kw.get("size", 1000), MTIME, english_audio=False)
    return path


def test_status_lists_every_status_with_zeros_the_allowance_and_the_files_waiting(
    tmp_path: Path,
) -> None:
    cfg = write_config(tmp_path)
    _wanted(tmp_path, "a.mkv")
    _wanted(tmp_path, "b.mkv")
    reset = datetime.now(UTC) + timedelta(hours=5)
    with State.open(tmp_path / "appdata" / "state.db") as state:
        state.set_quota(QuotaInfo(13, reset))
    result = _invoke("subs", "status", "--config", str(cfg))
    assert result.exit_code == 0, result.output
    lines = [" ".join(line.split()) for line in result.output.splitlines()]
    for name, count in (
        ("wanted", 2),
        ("found", 0),
        ("not_needed", 0),
        ("unavailable", 0),
        ("unclear", 0),
    ):
        assert f"{name} {count}" in lines
    assert any(
        line.startswith("OpenSubtitles downloads left: 13")
        and f"{reset:%Y-%m-%d %H:%M} UTC" in line
        for line in lines
    )
    assert "Files waiting on subtitles: 2" in result.output


def test_status_with_no_stored_allowance_says_unknown(tmp_path: Path) -> None:
    cfg = write_config(tmp_path)
    _wanted(tmp_path, "a.mkv")
    result = _invoke("subs", "status", "--config", str(cfg))
    assert "downloads left: unknown" in result.output


def test_status_with_a_renewed_allowance_does_not_claim_it_is_spent(tmp_path: Path) -> None:
    cfg = write_config(tmp_path)
    _wanted(tmp_path, "a.mkv")
    with State.open(tmp_path / "appdata" / "state.db") as state:
        state.set_quota(QuotaInfo(0, datetime.now(UTC) - timedelta(hours=1)))
    result = _invoke("subs", "status", "--config", str(cfg))
    assert "downloads left: unknown" in result.output
    assert "renewed" in result.output


def test_status_without_a_database_says_so_and_creates_nothing(tmp_path: Path) -> None:
    cfg = write_config(tmp_path)
    result = _invoke("subs", "status", "--config", str(cfg))
    assert result.exit_code == 0
    assert "No state database" in result.output
    assert not (tmp_path / "appdata" / "state.db").exists()


# --- retry -----------------------------------------------------------------------------------


def _stored(tmp_path: Path, path: Path, status: SubtitleStatus, size: int = 1000) -> None:
    with State.open(tmp_path / "appdata" / "state.db") as state:
        state.set_subtitle(
            SubtitleRecord(
                path, size, MTIME, status, reason="old reason", tried_ids=(4, 5), attempts=2
            )
        )


def _row(tmp_path: Path, path: Path, size: int = 1000) -> SubtitleRecord:
    with State.open(tmp_path / "appdata" / "state.db") as state:
        row = state.subtitle(path, size, MTIME)
    assert row is not None
    return row


@pytest.mark.parametrize("status", [SubtitleStatus.UNAVAILABLE, SubtitleStatus.UNCLEAR])
def test_retry_makes_an_unavailable_or_unclear_subtitle_wanted_again(
    tmp_path: Path, status: SubtitleStatus
) -> None:
    cfg = write_config(tmp_path)
    path = _wanted(tmp_path, "a.mkv")
    _stored(tmp_path, path, status)
    result = _invoke("subs", "retry", "--path", str(path), "--config", str(cfg))
    assert result.exit_code == 0, result.output
    row = _row(tmp_path, path)
    assert (row.status, row.tried_ids, row.reason, row.attempts) == (
        SubtitleStatus.WANTED,
        (),
        "",
        0,
    )
    assert status.value in result.output and "wanted" in result.output
    assert "subs resolve" in result.output


def test_retry_through_a_member_part_resets_the_groups_first_part(tmp_path: Path) -> None:
    cfg = write_config(tmp_path)
    lib = tmp_path / "library"
    first, second = lib / "Film cd1.mkv", lib / "Film cd2.mkv"
    with State.open(tmp_path / "appdata" / "state.db") as state:
        add_record(state, first, "remux", ("multi-part file to join",), parts=(first, second))
        add_record(state, second, "leave_alone", ("part of group Film.mkv",))
    _stored(tmp_path, first, SubtitleStatus.UNCLEAR)
    result = _invoke("subs", "retry", "--path", str(second), "--config", str(cfg))
    assert result.exit_code == 0, result.output
    assert _row(tmp_path, first).status is SubtitleStatus.WANTED


def test_retry_for_a_file_with_no_record_exits_1(tmp_path: Path) -> None:
    cfg = write_config(tmp_path)
    _setup(tmp_path)
    result = _invoke(
        "subs", "retry", "--path", str(tmp_path / "library" / "nope.mkv"), "--config", str(cfg)
    )
    assert result.exit_code == 1
    assert "nope.mkv" in result.output and "survey" in result.output


def test_retry_for_a_file_with_no_subtitle_row_exits_1(tmp_path: Path) -> None:
    cfg = write_config(tmp_path)
    with State.open(tmp_path / "appdata" / "state.db") as state:
        add_record(state, tmp_path / "library" / "a.mkv", "encode", ("h264",))
    result = _invoke(
        "subs", "retry", "--path", str(tmp_path / "library" / "a.mkv"), "--config", str(cfg)
    )
    assert result.exit_code == 1
    assert "no subtitle record" in result.output


@pytest.mark.parametrize(
    "status", [SubtitleStatus.WANTED, SubtitleStatus.FOUND, SubtitleStatus.NOT_NEEDED]
)
def test_retry_leaves_other_subtitle_rows_alone(tmp_path: Path, status: SubtitleStatus) -> None:
    cfg = write_config(tmp_path)
    path = _wanted(tmp_path, "a.mkv")
    _stored(tmp_path, path, status)
    before = _row(tmp_path, path)
    result = _invoke("subs", "retry", "--path", str(path), "--config", str(cfg))
    assert result.exit_code == 1
    assert status.value in result.output
    assert _row(tmp_path, path) == before


def test_retry_needs_a_path() -> None:
    assert _invoke("subs", "retry").exit_code == 2


def test_retry_without_a_database_exits_2(tmp_path: Path) -> None:
    cfg = write_config(tmp_path)
    result = _invoke("subs", "retry", "--path", "x.mkv", "--config", str(cfg))
    assert result.exit_code == 2
    assert "survey" in result.output


# --- wanted subtitles a run has already dealt with -------------------------------------------

HINT = "put a run-decided file back with `plan --retry` to look up its subtitle"


def _held_back(tmp_path: Path) -> None:
    """Three wanted rows that `subs resolve` will not touch, and one that it will."""
    from video_cleaner.state import Status

    with State.open(tmp_path / "appdata" / "state.db") as state:
        for name, status, note in (
            ("decided.mkv", Status.LEFT_ALONE, "output not smaller"),
            ("also-decided.mkv", Status.FLAGGED, "verify failed"),
            ("done.mkv", Status.COMPLETED, "delivered"),
            ("busy.mkv", Status.ENCODING, ""),
            ("open.mkv", None, ""),
        ):
            path = tmp_path / "library" / name
            add_record(state, path, "encode", ("h264",))
            state.ensure_wanted(path, 1000, MTIME, english_audio=False)
            if status is not None:
                state.set_status(path, status, note)


def test_resolve_says_how_many_wanted_files_a_run_already_dealt_with(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _setup(tmp_path)
    _held_back(tmp_path)
    monkeypatch.setattr(job_module, "run_resolve_job", FakeJob())
    result = _invoke("subs", "resolve", "--config", str(cfg), "-q")
    text = " ".join(result.output.split())
    assert "2 already decided by a run, 1 already delivered, 1 being processed" in text
    assert HINT in text


def test_status_says_it_too(tmp_path: Path) -> None:
    cfg = _setup(tmp_path)
    _held_back(tmp_path)
    text = " ".join(_invoke("subs", "status", "--config", str(cfg)).output.split())
    assert "2 already decided by a run, 1 already delivered, 1 being processed" in text
    assert HINT in text


def test_nothing_is_said_when_no_wanted_file_is_held_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _setup(tmp_path)
    _wanted(tmp_path, "a.mkv")
    monkeypatch.setattr(job_module, "run_resolve_job", FakeJob())
    for args in (["resolve", "-q"], ["status"]):
        out = _invoke("subs", *args, "--config", str(cfg)).output
        assert "already decided" not in out and "plan --retry" not in out


def _kept_download(cache: Path, path: Path, file_id: int, size: int = 1000) -> None:
    from video_cleaner.subtitles.cache import cache_key, cache_names

    key = cache_key(SubtitleRecord(path, size, MTIME, SubtitleStatus.WANTED))
    cache.mkdir(parents=True, exist_ok=True)
    cache_names(cache, key, "opensubtitles", str(file_id)).staged.write_text("kept")


def test_retry_says_how_many_remembered_candidates_are_kept_and_free_to_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "appdata" / "subs"
    monkeypatch.setenv("VC_SUBTITLE_CACHE_DIR", str(cache))
    cfg = write_config(tmp_path)
    path = _wanted(tmp_path, "a.mkv")
    _stored(tmp_path, path, SubtitleStatus.UNAVAILABLE)  # tried ids 4 and 5
    _kept_download(cache, path, 4)
    result = _invoke("subs", "retry", "--path", str(path), "--config", str(cfg))
    assert result.exit_code == 0, result.output
    text = " ".join(result.output.split())
    assert "1 of 2 remembered candidate(s) are kept" in text
    assert "re-checked without a download" in text
    assert "anything else may cost downloads" in text


def test_retry_with_nothing_remembered_says_a_lookup_may_cost_downloads(tmp_path: Path) -> None:
    cfg = write_config(tmp_path)
    path = _wanted(tmp_path, "a.mkv")
    with State.open(tmp_path / "appdata" / "state.db") as state:
        state.set_subtitle(SubtitleRecord(path, 1000, MTIME, SubtitleStatus.UNCLEAR, reason="x"))
    result = _invoke("subs", "retry", "--path", str(path), "--config", str(cfg))
    text = " ".join(result.output.split())
    assert "no candidate is remembered" in text and "may cost downloads" in text
