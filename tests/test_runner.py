"""run_queue orchestration, against a fake pipeline, plus one integration run of the real one.

The unit tests inject a `Pipeline` of fakes that write small dummy files; the real copy code
copies small dummy sources and no ffmpeg, ffprobe or mkvmerge process is started. The
``integration`` tests at the end run the real pipeline on a tiny generated clip.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import re
import sqlite3
import stat
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from media import RecordingReporter as MediaReporter
from media import make_clip, vmaf_tools
from video_cleaner import cli as cli_module
from video_cleaner import encode as encode_module
from video_cleaner import logarchive
from video_cleaner import paths as paths_module
from video_cleaner import probe as probe_module
from video_cleaner import quality as quality_module
from video_cleaner import runner as runner_module
from video_cleaner import scratch as scratch_module
from video_cleaner import survey as survey_module
from video_cleaner import verify as verify_module
from video_cleaner.config import Settings
from video_cleaner.consolefmt import Level
from video_cleaner.encode import CrfSearchResult, EncodeError
from video_cleaner.instancelock import AlreadyRunning, instance_lock, lock_path_for
from video_cleaner.manifest import ManifestEntry, ManifestError, read_manifest, write_manifest
from video_cleaner.mux import SubtitleInput
from video_cleaner.planning import build_plan, plan_hash
from video_cleaner.probe import AudioStream, MediaInfo, ProbeError, SubtitleStream, VideoStream
from video_cleaner.progress import Progress, Reporter
from video_cleaner.quality import VmafError
from video_cleaner.runner import (
    CalibrationNotApproved,
    Pipeline,
    PlanNotAcknowledged,
    RunRefused,
    RunSummary,
    deadline_from,
    run_queue,
)
from video_cleaner.scratch import CopyResult, ScratchFull, SourceChanged, copy_with_progress
from video_cleaner.state import FileRecord, State, Status, SubtitleRecord, SubtitleStatus
from video_cleaner.tools import (
    ToolFailed,
    ToolNotFound,
    ToolOutputError,
    ToolResult,
    Tools,
)
from video_cleaner.verify import VerifyResult, verify_output

TOOLS = Tools(ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe"), mkvmerge=Path("mkvmerge"))
START = datetime(2026, 11, 15, 22, 0, 0)
# Calibration every run fixture approves; distinct from the Settings defaults (95.0, 24).
APPROVED_VMAF = 94.5
APPROVED_CRF_MAX = 23


# --- helpers -------------------------------------------------------------------------------


class RecordingReporter:
    def __init__(self) -> None:
        self.stages: list[str] = []
        self.updates: list[Progress] = []
        self.done: list[str] = []
        self.done_levels: list[Level] = []

    def stage(self, label: str) -> None:
        self.stages.append(label)

    def update(self, p: Progress) -> None:
        self.updates.append(p)

    def heartbeat(self, label: str, elapsed_s: float) -> None:
        pass

    def file_done(self, summary: str, level: Level = Level.INFO) -> None:
        self.done.append(summary)
        self.done_levels.append(level)


class Clock:
    def __init__(self, start: datetime = START) -> None:
        self.t = start

    def __call__(self) -> datetime:
        return self.t


def _settings(tmp_path: Path, **changes: Any) -> Settings:
    values: dict[str, Any] = {
        "library_root": tmp_path / "library",
        "scratch_dir": tmp_path / "scratch",
        "completed_dir": tmp_path / "completed",
        "state_db": tmp_path / "appdata" / "state.db",
        "log_dir": tmp_path / "appdata" / "logs",
        "subtitle_cache_dir": tmp_path / "appdata" / "subtitles",
    }
    values.update(changes)
    settings = Settings(**values)
    settings.library_root.mkdir(parents=True, exist_ok=True)
    return settings


def _info(
    path: Path, size: int, mtime: float, duration_s: float = 100.0, codec: str = "h264"
) -> MediaInfo:
    return MediaInfo(
        path=path,
        size=size,
        mtime=mtime,
        duration_s=duration_s,
        container="matroska,webm",
        video=VideoStream(
            codec=codec,
            width=1920,
            height=1080,
            pix_fmt="yuv420p",
            fps=23.976,
            bit_rate=None,
            transfer="bt709",
            dv_profile=None,
            field_order="progressive",
        ),
        audio=(AudioStream(index=1, codec="ac3", language="eng", title=""),),
        subtitles=(),
        chapters=0,
        cover_art=0,
        format_bit_rate=8_000_000,  # well above the bit rate floor: an encode candidate
    )


def _source(settings: Settings, rel: str, size: int = 1000) -> Path:
    path = settings.library_root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(range(256)) * (size // 256) + b"x" * (size % 256))
    return path


def _plan(
    state: State,
    path: Path,
    *,
    decision: str = "encode",
    est: int = 500,
    parts: Sequence[Path] = (),
) -> FileRecord:
    st = path.stat()
    state.upsert_file(
        _info(path, st.st_size, st.st_mtime),
        decision=decision,
        reasons=("test",),
        est_saving_bytes=est,
        parts=parts,
    )
    record = state.get(path)
    assert record is not None
    return record


def _open_state(settings: Settings) -> State:
    """The state with an approved calibration; `_run` acknowledges the plan as it stands."""
    state = State.open(settings.state_db)
    state.approve_calibration(APPROVED_VMAF, APPROVED_CRF_MAX)
    return state


def _ack(state: State) -> str:
    """Acknowledge the plan exactly as it stands now (what `plan` + `plan --ack` record)."""
    digest = plan_hash(build_plan(state))
    state.acknowledge_plan(digest)
    return digest


def _status(state: State, path: Path) -> FileRecord:
    record = state.get(path)
    assert record is not None
    return record


def _transitions(state: State, path: Path) -> list[str]:
    """Status names the record passed through, oldest first, from the event log."""
    key = path.as_posix()
    found: list[str] = []
    for line in reversed(state.events(1000)):
        match = re.search(re.escape(key) + r": (\w+) -> (\w+)", line)
        if match:
            found.append(match.group(2))
    return found


def _tree(root: Path) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for folder, dirs, files in os.walk(root):
        for name in dirs + files:
            full = Path(folder) / name
            st = full.stat()
            out[str(full.relative_to(root))] = (st.st_size, st.st_mtime_ns)
    return out


def _scratch_files(settings: Settings) -> list[str]:
    """Files under the scratch folder, apart from the tool folder's marker."""
    return sorted(
        str(Path(folder, name).relative_to(settings.scratch_dir))
        for folder, _dirs, names in os.walk(settings.scratch_dir)
        for name in names
        if name != scratch_module.MARKER_NAME
    )


def _tool_scratch(settings: Settings) -> Path:
    """The tool-owned scratch folder, created and marked the way a run would."""
    return scratch_module.prepare_tool_scratch(settings.scratch_dir)


class Fakes:
    """A recording fake pipeline. Every stage can be told to raise via `fail`."""

    def __init__(self, clock: Clock | None = None) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.durations: dict[str, float] = {}
        self.crf = CrfSearchResult(crf=20, projected_saving=0.4, points=((20, 95.5), (24, 93.0)))
        self.scores: list[float] = [96.0, 95.0, 97.0]
        self.ratio = 0.5
        self.verify_result: VerifyResult | None = None
        self.fail: dict[str, Callable[[Path], BaseException | None]] = {}
        self.clock = clock
        self.encode_advances = timedelta(0)
        self.after_copy: Callable[[Path], None] | None = None
        self.codec = "h264"
        self.transfer = "bt709"  # "smpte2084" makes every probed file HDR
        # output path -> title of the subtitle muxed into it (what its probe then shows)
        self.subtitled: dict[str, str] = {}
        # the subtitle path and bytes each mux call was given, read at the time of the call
        self.staged: list[tuple[Path, bytes]] = []
        self.drop_subtitle = False  # a broken mux: accepts the subtitle, writes none

    def _check(self, stage: str, path: Path) -> None:
        rule = self.fail.get(stage)
        if rule is not None:
            exc = rule(path)
            if exc is not None:
                raise exc

    def copy(self, src: Path, dest: Path, reporter: Reporter) -> CopyResult:
        self.calls.append(("copy", src, dest))
        self._check("copy", src)
        result = copy_with_progress(src, dest, reporter)
        if self.after_copy is not None:
            self.after_copy(src)
        return result

    def probe(self, path: Path, tools: Tools, reporter: Reporter) -> MediaInfo:
        self.calls.append(("probe", path))
        self._check("probe", path)
        st = path.stat()
        duration = self.durations.get(str(path), 100.0)
        info = _info(path, st.st_size, st.st_mtime, duration, self.codec)
        if self.transfer != "bt709" and info.video is not None:
            info = dataclasses.replace(
                info, video=dataclasses.replace(info.video, transfer=self.transfer)
            )
        title = self.subtitled.get(str(path))
        if title is None:
            return info
        added = SubtitleStream(
            index=9,
            codec="subrip",
            language="eng",
            forced=False,
            sdh=False,
            is_image=False,
            default=False,
            title=title,
        )
        return dataclasses.replace(info, subtitles=(*info.subtitles, added))

    def search_crf(
        self, info: MediaInfo, tools: Tools, settings: Settings, workdir: Path, rep: Reporter
    ) -> CrfSearchResult:
        self.calls.append(("search_crf", info.path))
        self._check("search_crf", info.path)
        return self.crf

    def encode_full(
        self,
        info: MediaInfo,
        tools: Tools,
        settings: Settings,
        crf: int,
        dst: Path,
        reporter: Reporter,
    ) -> None:
        self.calls.append(("encode_full", info.path, crf, dst))
        self._check("encode_full", info.path)
        dst.write_bytes(b"v" * 10)
        if self.clock is not None:
            self.clock.t += self.encode_advances

    def append_parts(
        self, tools: Tools, parts: Sequence[Path], output: Path, reporter: Reporter
    ) -> None:
        self.calls.append(("append_parts", list(parts), output))
        self._check("append_parts", output)
        output.write_bytes(b"".join(p.read_bytes() for p in parts))
        self.durations[str(output)] = sum(self.durations.get(str(p), 100.0) for p in parts)

    def mux(
        self,
        tools: Tools,
        sources: Sequence[Path],
        new_video: Path | None,
        output: Path,
        reporter: Reporter,
        subtitle: SubtitleInput | None = None,
    ) -> None:
        self.calls.append(("mux", list(sources), new_video, output, subtitle))
        self._check("mux", output)
        size = sum(p.stat().st_size for p in sources)
        if new_video is not None:
            size = int(size * self.ratio)
        if subtitle is not None:
            data = subtitle.path.read_bytes()
            self.staged.append((subtitle.path, data))
            size += len(data)
            if not self.drop_subtitle:
                self.subtitled[str(output)] = subtitle.title
        output.write_bytes(b"m" * size)
        self.durations[str(output)] = self.durations.get(str(sources[0]), 100.0)

    def window_scores(
        self,
        tools: Tools,
        source: MediaInfo,
        output: Path,
        settings: Settings,
        workdir: Path,
        reporter: Reporter,
    ) -> Sequence[float]:
        self.calls.append(("window_scores", source.path, output))
        return self.scores

    def _probe_encoded(self, path: Path, tools: Tools, reporter: Reporter) -> MediaInfo:
        """What the probe of an encoded output shows: 10-bit HEVC (like build_x265_cmd)."""
        info = self.probe(path, tools, reporter)
        assert info.video is not None
        video = dataclasses.replace(info.video, codec="hevc", pix_fmt="yuv420p10le")
        return dataclasses.replace(info, video=video)

    def verify(
        self, tools: Tools, original: MediaInfo, output: Path, reporter: Reporter, **kw: Any
    ) -> VerifyResult:
        self.calls.append(("verify", original.path, output, kw))
        if self.verify_result is not None:
            return self.verify_result
        probe_fn = self._probe_encoded if kw.get("encoded") else self.probe
        return verify_output(
            tools, original, output, reporter, probe_fn=probe_fn, decode_fn=_no_errors, **kw
        )

    def pipeline(self) -> Pipeline:
        return Pipeline(
            copy=self.copy,
            search_crf=self.search_crf,
            encode_full=self.encode_full,
            append_parts=self.append_parts,
            mux=self.mux,
            window_scores=self.window_scores,
            verify=self.verify,
            probe=self.probe,
        )

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]


def _named(path: Path, name: str) -> bool:
    """True if *path* is the source *name* or its scratch copy ``<8 hex>-<name>``.

    Never a substring match: the hex prefix of a scratch name can spell "bad", "add" and so on.
    """
    return path.name == name or (path.name.endswith("-" + name) and len(path.name) == len(name) + 9)


def _no_errors(tools: Tools, output: Path, reporter: Reporter) -> str:
    return ""


def _run(
    settings: Settings,
    state: State,
    fakes: Fakes,
    *,
    reporter: Reporter | None = None,
    deadline: datetime | None = None,
    only: Path | None = None,
    clock: Clock | None = None,
    current_log: Path | None = None,
    ack: bool = True,
    include_unresolved: bool = False,
) -> RunSummary:
    """run_queue with the fakes; acknowledges the current plan first unless *ack* is False."""
    if ack:
        _ack(state)
    return run_queue(
        settings,
        TOOLS,
        state,
        reporter if reporter is not None else RecordingReporter(),
        deadline=deadline,
        only=only,
        now=clock if clock is not None else Clock(),
        pipeline=fakes.pipeline(),
        current_log=current_log,
        include_unresolved=include_unresolved,
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return _settings(tmp_path)


@pytest.fixture
def state(settings: Settings) -> Any:
    st = _open_state(settings)
    yield st
    st.close()


# --- deadline_from -------------------------------------------------------------------------


def test_deadline_rolls_to_next_day() -> None:
    now = datetime(2026, 11, 15, 8, 30)
    assert deadline_from("07:00", None, now) == datetime(2026, 11, 16, 7, 0)
    assert deadline_from("23:15", None, now) == datetime(2026, 11, 15, 23, 15)
    assert deadline_from("08:30", None, now) == datetime(2026, 11, 16, 8, 30)


def test_deadline_max_hours_and_earlier_of_both() -> None:
    now = datetime(2026, 11, 15, 22, 0)
    assert deadline_from(None, 2.5, now) == datetime(2026, 11, 16, 0, 30)
    assert deadline_from("07:00", 2.0, now) == datetime(2026, 11, 16, 0, 0)
    assert deadline_from("23:00", 2.0, now) == datetime(2026, 11, 15, 23, 0)
    assert deadline_from(None, None, now) is None


@pytest.mark.parametrize("until", ["7", "24:00", "07:60", "ab:cd", "7:00:00", ""])
def test_deadline_rejects_bad_until(until: str) -> None:
    with pytest.raises(ValueError, match="HH:MM"):
        deadline_from(until, None, START)


@pytest.mark.parametrize("hours", [0.0, -1.0, float("nan"), float("inf")])
def test_deadline_rejects_bad_max_hours(hours: float) -> None:
    with pytest.raises(ValueError, match="max-hours"):
        deadline_from(None, hours, START)


# --- refusals and startup ------------------------------------------------------------------


def test_run_refuses_without_plan_ack(settings: Settings) -> None:
    with State.open(settings.state_db) as state:
        _plan(state, _source(settings, "a.avi"))
        with pytest.raises(PlanNotAcknowledged, match="not been acknowledged"):
            _run(settings, state, Fakes(), ack=False)
        assert _status(state, settings.library_root / "a.avi").status is Status.PENDING


@pytest.mark.parametrize("field", ["scratch_dir", "completed_dir", "state_db", "log_dir"])
def test_run_refuses_outputs_under_library_root(tmp_path: Path, field: str) -> None:
    base = _settings(tmp_path)
    settings = _settings(tmp_path, **{field: base.library_root / "inside"})
    with _open_state(base) as state, pytest.raises(RunRefused, match="library root"):
        _run(settings, state, Fakes())


@pytest.mark.parametrize("field", ["scratch_dir", "completed_dir", "log_dir"])
def test_run_refuses_library_root_inside_an_output_folder(tmp_path: Path, field: str) -> None:
    # completed_dir = <x> with library_root = <x>/library would mirror <library>/library/a.avi
    # to <x>/library/a.mkv, inside the library; scratch sweeps would reach it too.
    outer = tmp_path / "outer"
    base = _settings(tmp_path)
    settings = _settings(tmp_path, library_root=outer / "library", **{field: outer})
    with _open_state(base) as state, pytest.raises(RunRefused, match="library root"):
        _run(settings, state, Fakes())


def test_library_unreachable_stops_before_anything(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with _open_state(settings) as state:
        src = _source(settings, "a.avi")
        _plan(state, src)
        src.unlink()
        settings.library_root.rmdir()
        fakes = Fakes()
        summary = _run(settings, state, fakes)
        assert summary.stop_reason == "library unreachable"
        assert fakes.calls == []
        assert _status(state, src).status is Status.PENDING


def test_startup_resets_sweeps_and_prunes(
    settings: Settings, state: State, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = _source(settings, "a.avi")
    _plan(state, src)
    state.set_status(src, Status.ENCODING)  # a killed run left it in progress
    ours = _tool_scratch(settings)
    (ours / "deadbeef-old.avi.part").write_bytes(b"p")
    (ours / "deadbeef-orphan.mkv").write_bytes(b"o")
    (ours / "work-deadbeef0123").mkdir()
    (ours / "work-deadbeef0123" / "video.mkv").write_bytes(b"w")
    (ours / "calibration" / "01").mkdir(parents=True)
    (ours / "calibration" / "01" / "crf17.mkv").write_bytes(b"c")
    # A shared scratch folder: foreign files, even with names like ours, must survive.
    (settings.scratch_dir / "notes.txt").write_bytes(b"not ours")
    (settings.scratch_dir / "cafebabe-holiday.mkv").write_bytes(b"not ours")
    (settings.scratch_dir / "work-0123456789ab").mkdir()
    (settings.scratch_dir / "work-0123456789ab" / "keep.txt").write_bytes(b"not ours")
    pruned: list[tuple[Path, Path | None]] = []

    def fake_prune(log_dir: Path, **kw: Any) -> list[Path]:
        pruned.append((log_dir, kw.get("exclude")))
        return []

    monkeypatch.setattr(logarchive, "archive_old_logs", fake_prune)
    current = settings.log_dir / "run-now.log"
    summary = _run(settings, state, Fakes(), current_log=current)
    assert pruned == [(settings.log_dir, current)]
    assert summary.completed == 1  # the reset record was retried
    assert "retry" in _transitions(state, src)
    # only our own leftovers are swept, and only inside the tool folder
    assert _scratch_files(settings) == [
        "cafebabe-holiday.mkv",
        "notes.txt",
        os.path.join("work-0123456789ab", "keep.txt"),
    ]


def test_second_run_is_refused_while_one_holds_the_lock(settings: Settings, state: State) -> None:
    src = _source(settings, "a.avi")
    _plan(state, src)
    state.set_status(src, Status.ENCODING)  # the live run's file: must not be reset
    live = _tool_scratch(settings) / "deadbeef-a.avi.part"
    live.write_bytes(b"p")  # the live run's copy: must not be swept
    fakes = Fakes()
    with instance_lock(lock_path_for(settings.state_db)), pytest.raises(AlreadyRunning) as info:
        _run(settings, state, fakes)
    assert info.value.pid == os.getpid()
    assert fakes.calls == []
    assert _status(state, src).status is Status.ENCODING
    assert live.exists()
    summary = _run(settings, state, Fakes())  # the lock is free again
    assert summary.completed == 1


# --- the happy paths -----------------------------------------------------------------------


def test_run_encodes_delivers_and_records(settings: Settings, state: State) -> None:
    src = _source(settings, "Films/Heat (1995)/Heat (1995).avi", size=4000)
    _plan(state, src)
    fakes = Fakes()
    summary = _run(settings, state, fakes)

    out = settings.completed_dir / "Films" / "Heat (1995)" / "Heat (1995).mkv"
    assert out.is_file() and out.stat().st_size == 2000
    assert summary == RunSummary(
        completed=1,
        left_alone=0,
        flagged=0,
        retry=0,
        bytes_saved=2000,
        stopped_by_deadline=False,
        stop_reason="",
    )
    assert _status(state, src).status is Status.COMPLETED
    assert _transitions(state, src) == [
        "queued",
        "copying",
        "encoding",
        "muxing",
        "verifying",
        "completed",
    ]
    (entry,) = read_manifest(settings.completed_dir / "manifest.csv")
    st = src.stat()
    assert entry.original == (src,)
    assert entry.original_sizes == (st.st_size,)
    assert entry.original_mtimes == (st.st_mtime,)
    assert entry.completed == out
    assert entry.completed_size == 2000
    assert entry.crf == 20
    assert entry.vmaf_mean == pytest.approx(96.0)
    assert entry.vmaf_worst == pytest.approx(95.0)
    assert entry.verified is True
    assert entry.action == "encode"
    assert _scratch_files(settings) == []
    # the encode and mux work on the scratch copy, never on the library file
    copy_dest = fakes.calls[0][2]
    assert ("search_crf", copy_dest) in fakes.calls
    mux = next(c for c in fakes.calls if c[0] == "mux")
    assert mux[1] == [copy_dest] and mux[2] is not None


def test_manifest_appends_to_existing_rows(settings: Settings, state: State) -> None:
    a = _source(settings, "a.avi")
    b = _source(settings, "b.avi")
    _plan(state, a, est=900)
    _plan(state, b, est=100)
    _run(settings, state, Fakes())
    entries = read_manifest(settings.completed_dir / "manifest.csv")
    assert [e.original for e in entries] == [(a,), (b,)]


def test_remux_only_never_drops_video(settings: Settings, state: State) -> None:
    src = _source(settings, "Show/clip.mp4", size=3000)
    _plan(state, src, decision="remux", est=0)
    fakes = Fakes()
    summary = _run(settings, state, fakes)
    assert summary.completed == 1
    assert "search_crf" not in fakes.names() and "encode_full" not in fakes.names()
    (mux,) = [c for c in fakes.calls if c[0] == "mux"]
    assert mux[2] is None  # no new video: the source's video streams all survive
    (verify,) = [c for c in fakes.calls if c[0] == "verify"]
    assert verify[3]["encoded"] is False
    (entry,) = read_manifest(settings.completed_dir / "manifest.csv")
    assert entry.action == "remux" and entry.crf is None and entry.vmaf_mean is None
    assert summary.bytes_saved == 0
    assert "encoding" not in _transitions(state, src)


def test_multi_part_encode_order_of_operations(settings: Settings, state: State) -> None:
    pt1 = _source(settings, "Films/Epic/Epic pt1.avi", size=2000)
    pt2 = _source(settings, "Films/Epic/Epic pt2.avi", size=3000)
    _plan(state, pt1, parts=(pt1, pt2))
    _plan(state, pt2, decision="leave_alone", est=0)
    fakes = Fakes()
    summary = _run(settings, state, fakes)

    assert summary.completed == 1
    names = fakes.names()
    assert names[:2] == ["copy", "copy"]
    assert [fakes.calls[0][1], fakes.calls[1][1]] == [pt1, pt2]
    copies = [fakes.calls[0][2], fakes.calls[1][2]]
    append = fakes.calls[names.index("append_parts")]
    combined = append[2]
    assert append[1] == copies
    order = [n for n in names if n not in ("probe", "copy")]
    assert order == ["append_parts", "search_crf", "encode_full", "mux", "window_scores", "verify"]
    assert fakes.calls[names.index("search_crf")][1] == combined
    mux = fakes.calls[names.index("mux")]
    assert mux[1] == [combined]
    assert mux[2] == fakes.calls[names.index("encode_full")][3]
    verify = fakes.calls[names.index("verify")][3]
    assert verify["original_duration_s"] == pytest.approx(200.0)
    assert verify["original_size"] == 5000

    out = settings.completed_dir / "Films" / "Epic" / "Epic.mkv"
    assert out.is_file()
    (entry,) = read_manifest(settings.completed_dir / "manifest.csv")
    assert entry.original == (pt1, pt2)
    assert entry.original_sizes == (2000, 3000)
    assert entry.original_mtimes == (pt1.stat().st_mtime, pt2.stat().st_mtime)
    assert summary.bytes_saved == 5000 - out.stat().st_size
    assert _status(state, pt2).status is Status.LEFT_ALONE  # member record untouched
    assert _scratch_files(settings) == []


def test_multi_part_remux_delivers_the_appended_file(settings: Settings, state: State) -> None:
    pt1 = _source(settings, "Epic cd1.mkv", size=1000)
    pt2 = _source(settings, "Epic cd2.mkv", size=1000)
    _plan(state, pt1, decision="remux", parts=(pt1, pt2))
    _plan(state, pt2, decision="leave_alone", est=0)
    fakes = Fakes()
    fakes.codec = "hevc"  # already efficient: the combined file is only joined
    summary = _run(settings, state, fakes)
    assert summary.completed == 1
    assert "mux" not in fakes.names() and "encode_full" not in fakes.names()
    assert (settings.completed_dir / "Epic.mkv").stat().st_size == 2000


# --- outcomes that keep the original -------------------------------------------------------


def test_low_projected_saving_leaves_alone(settings: Settings, state: State) -> None:
    src = _source(settings, "a.avi")
    _plan(state, src)
    fakes = Fakes()
    fakes.crf = CrfSearchResult(crf=24, projected_saving=0.05, points=())
    summary = _run(settings, state, fakes)
    assert summary.left_alone == 1
    record = _status(state, src)
    assert record.status is Status.LEFT_ALONE
    assert "projected saving 5.0%" in record.note
    assert "encode_full" not in fakes.names()
    assert not settings.completed_dir.exists() or not any(settings.completed_dir.iterdir())
    assert _scratch_files(settings) == []


def test_guard_failure_discards_output(settings: Settings, state: State) -> None:
    src = _source(settings, "a.avi")
    _plan(state, src)
    fakes = Fakes()
    fakes.scores = [94.0, 86.0, 95.0]
    summary = _run(settings, state, fakes)
    assert summary.left_alone == 1 and summary.completed == 0
    record = _status(state, src)
    assert record.status is Status.LEFT_ALONE
    assert "worst window VMAF 86.00 is below the floor" in record.note
    assert "verify" not in fakes.names()
    assert not (settings.completed_dir / "a.mkv").exists()
    assert not (settings.completed_dir / "manifest.csv").exists()
    assert _scratch_files(settings) == []


def test_output_not_small_enough_leaves_alone(settings: Settings, state: State) -> None:
    src = _source(settings, "a.avi")
    _plan(state, src)
    fakes = Fakes()
    fakes.ratio = 0.95
    summary = _run(settings, state, fakes)
    assert summary.left_alone == 1
    record = _status(state, src)
    assert record.status is Status.LEFT_ALONE
    assert "smaller" in record.note
    assert not (settings.completed_dir / "a.mkv").exists()


def test_verify_failure_flags(settings: Settings, state: State) -> None:
    src = _source(settings, "a.avi")
    _plan(state, src)
    fakes = Fakes()
    fakes.verify_result = VerifyResult(ok=False, reason="audio streams: 0 in the output")
    summary = _run(settings, state, fakes)
    assert summary.flagged == 1
    record = _status(state, src)
    assert record.status is Status.FLAGGED
    assert "verify failed: audio streams: 0 in the output" in record.note
    assert not (settings.completed_dir / "a.mkv").exists()


def test_a_files_closing_console_line_is_tagged_by_outcome(
    settings: Settings, state: State
) -> None:
    """completed is OK (green), left alone is INFO, flagged is WARN (yellow)."""
    done, kept, bad = (_source(settings, n) for n in ("done.avi", "kept.avi", "bad.avi"))
    for src, est in ((done, 900), (kept, 800), (bad, 700)):
        _plan(state, src, est=est)
    fakes = Fakes()
    fakes.fail["probe"] = lambda p: ProbeError(p, "zero duration") if _named(p, "bad.avi") else None

    reporter = RecordingReporter()
    summary = _run(settings, state, fakes, reporter=reporter)
    assert (summary.completed, summary.flagged) == (2, 1)
    levels = dict(zip((d.split(":")[0] for d in reporter.done), reporter.done_levels, strict=True))
    assert levels["done.avi"] is Level.OK
    assert levels["bad.avi"] is Level.WARN


def test_left_alone_closing_line_is_info(settings: Settings, state: State) -> None:
    src = _source(settings, "a.avi")
    _plan(state, src)
    fakes = Fakes()
    fakes.ratio = 0.95  # saves under the minimum
    reporter = RecordingReporter()
    assert _run(settings, state, fakes, reporter=reporter).left_alone == 1
    assert reporter.done_levels == [Level.INFO]


# --- errors --------------------------------------------------------------------------------


def test_run_continues_after_corrupt_file(settings: Settings, state: State) -> None:
    bad = _source(settings, "bad.avi")
    good = _source(settings, "good.avi")
    _plan(state, bad, est=900)
    _plan(state, good, est=100)
    fakes = Fakes()
    fakes.fail["probe"] = lambda p: ProbeError(p, "zero duration") if _named(p, "bad.avi") else None
    summary = _run(settings, state, fakes)
    assert summary.flagged == 1 and summary.completed == 1
    record = _status(state, bad)
    assert record.status is Status.FLAGGED
    assert "zero duration" in record.note
    assert _status(state, good).status is Status.COMPLETED
    assert _scratch_files(settings) == []


def test_unexpected_exception_flags_and_continues(settings: Settings, state: State) -> None:
    a = _source(settings, "a.avi")
    b = _source(settings, "b.avi")
    _plan(state, a, est=900)
    _plan(state, b, est=100)
    fakes = Fakes()
    fakes.fail["search_crf"] = lambda p: RuntimeError("boom") if _named(p, "a.avi") else None
    summary = _run(settings, state, fakes)
    assert summary.flagged == 1 and summary.completed == 1
    assert "RuntimeError: boom" in _status(state, a).note


def test_tool_failed_and_tool_output_error_go_to_retry(settings: Settings, state: State) -> None:
    a = _source(settings, "a.avi")
    b = _source(settings, "b.avi")
    c = _source(settings, "c.avi")
    _plan(state, a, est=900)
    _plan(state, b, est=500)
    _plan(state, c, est=100)
    log = settings.scratch_dir / "x.stderr.log"

    def fail(p: Path) -> BaseException | None:
        if _named(p, "a.avi"):
            return ToolFailed(3, "ffmpeg -i a", log)
        if _named(p, "b.avi"):
            return ToolOutputError("encode", "stdout", "parser broke")
        return None

    fakes = Fakes()
    fakes.fail["encode_full"] = fail
    summary = _run(settings, state, fakes)
    assert summary.retry == 2 and summary.completed == 1
    note_a = _status(state, a).note
    assert _status(state, a).status is Status.RETRY
    assert "exit code 3" in note_a
    note_b = _status(state, b).note
    assert _status(state, b).status is Status.RETRY
    assert "tool output handling failed" in note_b
    # each failure is attempted once per run, not retried in a loop
    assert fakes.names().count("encode_full") == 3


def test_tool_failed_log_is_kept_out_of_scratch(settings: Settings, state: State) -> None:
    src = _source(settings, "a.avi")
    _plan(state, src)
    fakes = Fakes()

    def fail(p: Path) -> BaseException:
        log = p.parent / "mux.stderr.log"
        log.write_text("mkvmerge: error\n", encoding="utf-8")
        return ToolFailed(2, "mkvmerge -o x", log)

    fakes.fail["mux"] = fail
    _run(settings, state, fakes)
    note = _status(state, src).note
    kept = list(settings.log_dir.glob("run-*.log"))
    assert len(kept) == 1
    assert str(kept[0]) in note
    assert kept[0].read_text(encoding="utf-8") == "mkvmerge: error\n"
    assert _scratch_files(settings) == []


def test_copy_oserror_retries_and_continues(settings: Settings, state: State) -> None:
    a = _source(settings, "a.avi")
    b = _source(settings, "b.avi")
    _plan(state, a, est=900)
    _plan(state, b, est=100)
    fakes = Fakes()
    fakes.fail["copy"] = lambda p: (
        OSError(64, "network name no longer available") if p == a else None
    )
    summary = _run(settings, state, fakes)
    assert _status(state, a).status is Status.RETRY
    assert summary.retry == 1 and summary.completed == 1


def test_library_drop_mid_run_stops(settings: Settings, state: State) -> None:
    a = _source(settings, "a.avi")
    b = _source(settings, "b.avi")
    _plan(state, a, est=900)
    _plan(state, b, est=100)
    fakes = Fakes()
    renamed = settings.library_root.with_name("library-offline")

    def drop(p: Path) -> BaseException:
        settings.library_root.rename(renamed)
        return FileNotFoundError(2, "gone", str(p))

    fakes.fail["copy"] = drop
    summary = _run(settings, state, fakes)
    renamed.rename(settings.library_root)
    assert summary.stop_reason == "library unreachable"
    assert _status(state, a).status is Status.RETRY  # never stale for an unreachable share
    assert _status(state, b).status is Status.PENDING


def test_scratch_full_stops_the_run(settings: Settings, state: State) -> None:
    a = _source(settings, "a.avi")
    b = _source(settings, "b.avi")
    _plan(state, a, est=900)
    _plan(state, b, est=100)
    fakes = Fakes()
    fakes.fail["copy"] = lambda p: ScratchFull("disk full")
    summary = _run(settings, state, fakes)
    assert summary.stop_reason == "scratch full"
    assert _status(state, a).status is Status.RETRY
    assert _status(state, b).status is Status.PENDING
    assert summary.retry == 1


def test_scratch_cap_counts_as_scratch_full(tmp_path: Path) -> None:
    settings = _settings(tmp_path, scratch_cap_bytes=500)
    with _open_state(settings) as state:
        src = _source(settings, "a.avi", size=1000)
        _plan(state, src)
        fakes = Fakes()
        summary = _run(settings, state, fakes)
        assert summary.stop_reason == "scratch full"
        assert "copy" not in fakes.names()
        assert _status(state, src).status is Status.RETRY


def test_tool_not_found_stops_the_run(settings: Settings, state: State) -> None:
    a = _source(settings, "a.avi")
    b = _source(settings, "b.avi")
    _plan(state, a, est=900)
    _plan(state, b, est=100)
    fakes = Fakes()
    fakes.fail["mux"] = lambda p: ToolNotFound("mkvmerge")
    summary = _run(settings, state, fakes)
    assert summary.stop_reason.startswith("tool not found")
    assert _status(state, a).status is Status.RETRY
    assert _status(state, b).status is Status.PENDING


@pytest.mark.parametrize(
    ("stage", "error"),
    [
        ("search_crf", EncodeError("a.avi: the CRF 24 window 1/3 sample is empty")),
        ("search_crf", VmafError("the VMAF log vmaf-1.json is missing")),
        ("encode_full", EncodeError("a.avi: the encode at CRF 20 is empty")),
        ("window_scores", VmafError("the VMAF log vmaf-2.json has an unusable mean nan")),
    ],
)
def test_unusable_encode_or_score_flags_the_file_and_the_run_continues(
    settings: Settings, state: State, stage: str, error: Exception
) -> None:
    a = _source(settings, "a.avi")
    b = _source(settings, "b.avi")
    _plan(state, a, est=900)
    _plan(state, b, est=100)
    fakes = Fakes()
    if stage == "window_scores":
        original = fakes.window_scores

        def failing(*args: Any) -> Sequence[float]:
            if args[1].path.name.endswith("a.avi"):
                raise error
            return original(*args)

        pipeline = fakes.pipeline()
        pipeline.window_scores = failing
        _ack(state)
        summary = run_queue(
            settings,
            TOOLS,
            state,
            RecordingReporter(),
            deadline=None,
            now=Clock(),
            pipeline=pipeline,
        )
    else:
        fakes.fail[stage] = lambda p: error if p.name.endswith("a.avi") else None
        summary = _run(settings, state, fakes)
    assert summary.stop_reason == ""
    rec = _status(state, a)
    assert rec.status is Status.FLAGGED
    assert rec.note == str(error)
    assert rec.attempts == 0
    assert _status(state, b).status is Status.COMPLETED
    assert _scratch_files(settings) == []


def test_default_pipeline_uses_the_real_stages() -> None:
    pipeline = Pipeline()
    assert pipeline.search_crf is encode_module.search_crf
    assert pipeline.encode_full is encode_module.encode_full
    assert pipeline.window_scores is runner_module.guard_window_scores


def test_guard_window_scores_scores_four_20s_windows_of_both_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    scores = iter([96.0, 95.0, 97.0, 94.0])

    def fake_vmaf(
        tools: Tools, distorted: Path, reference: Path, workdir: Path, rep: Reporter, **kw: Any
    ) -> float:
        calls.append({"distorted": distorted, "reference": reference, "workdir": workdir, **kw})
        return next(scores)

    monkeypatch.setattr(quality_module, "vmaf_mean", fake_vmaf)
    source = _info(tmp_path / "copy.mkv", 1, 1.0, duration_s=1000.0)
    out = tmp_path / "work" / "Film.mkv"
    reporter = RecordingReporter()
    result = runner_module.guard_window_scores(
        TOOLS, source, out, _settings_stub(), tmp_path / "work", reporter
    )
    assert list(result) == [96.0, 95.0, 97.0, 94.0]
    assert len(calls) == 4
    for call in calls:
        assert call["distorted"] == out
        assert call["reference"] == source.path
        assert call["reference_length_s"] == call["distorted_length_s"] == 20.0
        assert call["reference_start_s"] == call["distorted_start_s"]
        assert 100.0 <= call["reference_start_s"] <= 880.0  # inside the middle 80%
    assert sum("quality guard" in s for s in reporter.stages) == 4


def test_guard_window_scores_short_film_uses_one_shorter_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_vmaf(*args: Any, **kw: Any) -> float:
        calls.append(kw)
        return 99.0

    monkeypatch.setattr(quality_module, "vmaf_mean", fake_vmaf)
    source = _info(tmp_path / "copy.mkv", 1, 1.0, duration_s=10.0)
    result = runner_module.guard_window_scores(
        TOOLS, source, tmp_path / "o.mkv", _settings_stub(), tmp_path, RecordingReporter()
    )
    assert list(result) == [99.0]
    assert calls[0]["reference_length_s"] == pytest.approx(9.0)


def test_scratch_is_reserved_for_the_encode_before_it_runs(tmp_path: Path) -> None:
    """The encoded video is reserved at the original's size before it is written."""
    settings = _settings(tmp_path, scratch_cap_bytes=1999)
    with _open_state(settings) as state:
        a = _source(settings, "a.avi", size=1000)
        b = _source(settings, "b.avi", size=1000)
        _plan(state, a, est=900)
        _plan(state, b, est=100)
        fakes = Fakes()
        summary = _run(settings, state, fakes)
        assert summary.stop_reason == "scratch full"
        assert "search_crf" in fakes.names()
        assert "encode_full" not in fakes.names()
        rec = _status(state, a)
        assert rec.status is Status.RETRY
        assert "video.mkv" in rec.note
        assert rec.attempts == 0  # like the copy path: a stop condition, not the file's fault
        assert _status(state, b).status is Status.PENDING
        assert _scratch_files(settings) == []


def test_scratch_is_reserved_for_the_muxed_output_before_it_runs(tmp_path: Path) -> None:
    settings = _settings(tmp_path, scratch_cap_bytes=2999)
    with _open_state(settings) as state:
        a = _source(settings, "a.avi", size=1000)
        _plan(state, a)
        fakes = Fakes()
        summary = _run(settings, state, fakes)
        assert summary.stop_reason == "scratch full"
        assert "encode_full" in fakes.names()
        assert "mux" not in fakes.names()
        assert _status(state, a).status is Status.RETRY
        assert _scratch_files(settings) == []


def test_encode_fits_when_scratch_has_room_for_copy_video_and_output(tmp_path: Path) -> None:
    # 1000 bytes each for the copy, the encoded video and the output, plus the tool's marker
    settings = _settings(tmp_path, scratch_cap_bytes=3200)
    with _open_state(settings) as state:
        a = _source(settings, "a.avi", size=1000)
        _plan(state, a)
        summary = _run(settings, state, Fakes())
        assert summary.completed == 1, _status(state, a).note


def test_a_run_never_changes_the_plan_hash(settings: Settings, state: State) -> None:
    """The run writes neither est_saving_bytes nor decision: the acknowledged plan stays valid."""
    enc = _source(settings, "a.avi")
    small = _source(settings, "b.avi")
    remux = _source(settings, "c.mp4")
    _plan(state, enc, est=900)
    _plan(state, small, est=800)
    _plan(state, remux, decision="remux", est=0)
    before = plan_hash(build_plan(state))
    fakes = Fakes()
    original = fakes.search_crf

    def low_saving_for_b(*args: Any) -> CrfSearchResult:
        result = original(*args)
        if args[0].path.name.endswith("b.avi"):
            return dataclasses.replace(result, projected_saving=0.02)
        return result

    pipeline = fakes.pipeline()
    pipeline.search_crf = low_saving_for_b
    _ack(state)
    summary = run_queue(
        settings, TOOLS, state, RecordingReporter(), deadline=None, now=Clock(), pipeline=pipeline
    )
    assert (summary.completed, summary.left_alone) == (2, 1)
    assert plan_hash(build_plan(state)) == before
    # a second run is accepted with the same acknowledgement
    again = run_queue(
        settings, TOOLS, state, RecordingReporter(), deadline=None, now=Clock(), pipeline=pipeline
    )
    assert again.stop_reason == ""


def _settings_stub() -> Settings:
    return Settings(
        library_root=Path("L"),
        scratch_dir=Path("S"),
        completed_dir=Path("C"),
        state_db=Path("D"),
        log_dir=Path("G"),
    )


def test_keyboard_interrupt_leaves_in_progress_and_cleans(settings: Settings, state: State) -> None:
    src = _source(settings, "a.avi")
    _plan(state, src)
    fakes = Fakes()
    fakes.fail["encode_full"] = lambda p: KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        _run(settings, state, fakes)
    assert _status(state, src).status is Status.ENCODING
    assert _scratch_files(settings) == []
    # the next run resets it and finishes it
    summary = _run(settings, state, Fakes())
    assert summary.completed == 1


# --- stale sources -------------------------------------------------------------------------


def test_stale_source_discarded(settings: Settings, state: State) -> None:
    a = _source(settings, "a.avi")
    b = _source(settings, "b.avi")
    _plan(state, a, est=900)
    _plan(state, b, est=100)
    fakes = Fakes()

    def touch(src: Path) -> None:
        if src == a:
            st = src.stat()
            os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))

    fakes.after_copy = touch
    summary = _run(settings, state, fakes)
    record = state.records(Status.STALE)
    assert [r.path for r in record] == [a]
    assert record[0].note == "source changed"
    assert not (settings.completed_dir / "a.mkv").exists()
    entries = read_manifest(settings.completed_dir / "manifest.csv")
    assert [e.original for e in entries] == [(b,)]
    assert summary.completed == 1
    assert _scratch_files(settings) == []


def test_source_changed_during_copy_is_stale(settings: Settings, state: State) -> None:
    src = _source(settings, "a.avi")
    _plan(state, src)
    fakes = Fakes()
    fakes.fail["copy"] = lambda p: SourceChanged("grew")
    _run(settings, state, fakes)
    (record,) = state.records(Status.STALE)
    assert record.path == src and "source changed" in record.note


def test_superseded_record_never_gets_results(settings: Settings, state: State) -> None:
    src = _source(settings, "a.avi")
    _plan(state, src)
    fakes = Fakes()

    def replan(p: Path) -> None:
        # a re-survey while the encode runs: the file has a new size, a new record
        state.upsert_file(_info(src, 1, 1.0), decision="encode", reasons=(), est_saving_bytes=0)

    fakes.fail["encode_full"] = lambda p: replan(p)
    summary = _run(settings, state, fakes)
    assert summary.completed == 0
    assert not (settings.completed_dir / "manifest.csv").exists()
    assert not (settings.completed_dir / "a.mkv").exists()
    assert _scratch_files(settings) == []


# --- collisions ----------------------------------------------------------------------------


# The library is a case-insensitive Windows/SMB share; name keys fold case only on Windows
# (os.path.normcase), so collisions that differ only in case are detected there only.
windows_only_case = pytest.mark.skipif(
    sys.platform != "win32", reason="case-only name collisions are folded on Windows only"
)


@windows_only_case
def test_output_name_collision_flags_both(settings: Settings, state: State) -> None:
    avi = _source(settings, "Clips/Clip.avi")
    mp4 = _source(settings, "Clips/clip.MP4")
    other = _source(settings, "Clips/Other.avi")
    _plan(state, avi, est=900)
    _plan(state, mp4, est=800)
    _plan(state, other, est=100)
    fakes = Fakes()
    summary = _run(settings, state, fakes)
    assert summary.flagged == 2 and summary.completed == 1
    assert _status(state, avi).status is Status.FLAGGED
    assert f"output name collides with {mp4}" in _status(state, avi).note
    assert f"output name collides with {avi}" in _status(state, mp4).note
    assert [c[1] for c in fakes.calls if c[0] == "copy"] == [other]


def test_existing_destination_is_never_overwritten(settings: Settings, state: State) -> None:
    src = _source(settings, "a.avi")
    _plan(state, src)
    settings.completed_dir.mkdir(parents=True)
    existing = settings.completed_dir / "a.mkv"
    existing.write_bytes(b"keep me")
    fakes = Fakes()
    summary = _run(settings, state, fakes)
    assert summary.flagged == 1
    assert "already exists" in _status(state, src).note
    assert existing.read_bytes() == b"keep me"
    assert "copy" not in fakes.names()


# --- deadline, resume, scope ---------------------------------------------------------------


def test_run_stops_after_current_file_at_deadline(settings: Settings, state: State) -> None:
    paths = [_source(settings, f"{n}.avi") for n in ("a", "b", "c")]
    for est, path in zip((900, 500, 100), paths, strict=True):
        _plan(state, path, est=est)
    clock = Clock()
    fakes = Fakes(clock)
    fakes.encode_advances = timedelta(hours=2)
    summary = _run(settings, state, fakes, deadline=START + timedelta(hours=1), clock=clock)
    assert summary.stopped_by_deadline
    assert summary.completed == 1
    assert _status(state, paths[0]).status is Status.COMPLETED
    assert _status(state, paths[1]).status is Status.PENDING
    assert _status(state, paths[2]).status is Status.PENDING
    assert _scratch_files(settings) == []  # a prefetch of the next file is dropped


def test_deadline_already_passed_starts_nothing(settings: Settings, state: State) -> None:
    _plan(state, _source(settings, "a.avi"))
    fakes = Fakes()
    summary = _run(settings, state, fakes, deadline=START - timedelta(minutes=1))
    assert summary.stopped_by_deadline and summary.completed == 0
    assert fakes.calls == []


def test_run_resumes_without_redoing_completed(settings: Settings, state: State) -> None:
    a = _source(settings, "a.avi")
    b = _source(settings, "b.avi")
    _plan(state, a, est=900)
    _plan(state, b, est=100)
    clock = Clock()
    first = Fakes(clock)
    first.encode_advances = timedelta(hours=2)
    _run(settings, state, first, deadline=START + timedelta(hours=1), clock=clock)
    assert first.names().count("encode_full") == 1
    assert _status(state, a).status is Status.COMPLETED

    second = Fakes()
    summary = _run(settings, state, second)
    assert summary.completed == 1
    encoded = [c[1].name for c in second.calls if c[0] == "encode_full"]
    assert len(encoded) == 1 and _named(Path(encoded[0]), "b.avi")
    assert _status(state, a).status is Status.COMPLETED
    assert len(read_manifest(settings.completed_dir / "manifest.csv")) == 2


def test_retry_records_run_after_pending(settings: Settings, state: State) -> None:
    retry = _source(settings, "r.avi")
    pending = _source(settings, "p.avi")
    _plan(state, retry, est=10_000)
    _plan(state, pending, est=1)
    state.set_status(retry, Status.RETRY, "earlier failure")
    fakes = Fakes()
    _run(settings, state, fakes)
    encoded = [c[1].name for c in fakes.calls if c[0] == "encode_full"]
    assert [e.split("-", 1)[1] for e in encoded] == ["p.avi", "r.avi"]


def test_only_limits_to_a_folder_or_file(settings: Settings, state: State) -> None:
    a = _source(settings, "Films/a.avi")
    b = _source(settings, "Shows/b.avi")
    c = _source(settings, "Shows/c.avi")
    for p in (a, b, c):
        _plan(state, p)
    summary = _run(settings, state, Fakes(), only=settings.library_root / "Shows")
    assert summary.completed == 2
    assert _status(state, a).status is Status.PENDING
    summary = _run(settings, state, Fakes(), only=a)
    assert summary.completed == 1


def test_run_never_writes_under_library_root(settings: Settings, state: State) -> None:
    ok = _source(settings, "Films/ok.avi", size=3000)
    bad = _source(settings, "Films/bad.avi")
    remux = _source(settings, "Films/remux.mp4")
    low = _source(settings, "Films/low.avi")
    pt1 = _source(settings, "Films/Two pt1.avi")
    pt2 = _source(settings, "Films/Two pt2.avi")
    _plan(state, ok, est=900)
    _plan(state, bad, est=800)
    _plan(state, remux, decision="remux", est=0)
    _plan(state, low, est=700)
    _plan(state, pt1, est=600, parts=(pt1, pt2))
    _plan(state, pt2, decision="leave_alone", est=0)  # survey records every member part
    before = _tree(settings.library_root)
    fakes = Fakes()
    fakes.fail["probe"] = lambda p: ProbeError(p, "bad") if _named(p, "bad.avi") else None
    low_crf = CrfSearchResult(crf=24, projected_saving=0.01, points=())
    good_crf = fakes.crf
    original_search = fakes.search_crf

    def search(info: MediaInfo, *args: Any) -> CrfSearchResult:
        original_search(info, *args)
        return low_crf if _named(info.path, "low.avi") else good_crf

    pipeline = fakes.pipeline()
    pipeline.search_crf = search
    _ack(state)
    summary = run_queue(
        settings, TOOLS, state, RecordingReporter(), deadline=None, now=Clock(), pipeline=pipeline
    )
    assert summary.completed == 3 and summary.flagged == 1 and summary.left_alone == 1
    assert _tree(settings.library_root) == before


# --- prefetch ------------------------------------------------------------------------------


def test_prefetch_uses_a_separate_reporter(
    settings: Settings, state: State, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = _source(settings, "a.avi")
    b = _source(settings, "b.avi")
    _plan(state, a, est=900)
    _plan(state, b, est=100)
    made: list[Any] = []
    started: list[Path] = []
    real = scratch_module.Prefetcher

    class Spy(real):  # type: ignore[misc, valid-type]
        def __init__(self, scratch: Any, reporter: Reporter) -> None:
            made.append(reporter)
            super().__init__(scratch, reporter)

        def start(self, src: Path) -> None:
            started.append(src)
            super().start(src)

    foreground = RecordingReporter()
    fakes = Fakes()
    pipeline = fakes.pipeline()
    pipeline.prefetcher = Spy
    _ack(state)
    summary = run_queue(
        settings, TOOLS, state, foreground, deadline=None, now=Clock(), pipeline=pipeline
    )
    assert summary.completed == 2
    assert len(made) == 1 and made[0] is not foreground
    assert started == [b]
    # b came from the prefetch, so the foreground copy ran for a only
    assert [c[1] for c in fakes.calls if c[0] == "copy"] == [a]


# --- default mkvmerge stages (run_tool faked: no process starts) ---------------------------


def _fake_run_tool(
    monkeypatch: pytest.MonkeyPatch, returncode: int
) -> list[tuple[list[str], dict[str, Any]]]:
    seen: list[tuple[list[str], dict[str, Any]]] = []

    def fake(cmd: Sequence[str], **kwargs: Any) -> ToolResult:
        seen.append((list(cmd), kwargs))
        return ToolResult(returncode=returncode, stdout="", stderr_tail="")

    monkeypatch.setattr(runner_module, "run_tool", fake)
    return seen


def test_default_mux_swaps_video_and_accepts_warnings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _fake_run_tool(monkeypatch, returncode=1)  # mkvmerge: 1 = warnings only
    out = tmp_path / "out.mkv"
    runner_module.mux(
        TOOLS, [tmp_path / "src.mkv"], tmp_path / "video.mkv", out, RecordingReporter()
    )
    ((cmd, kwargs),) = seen
    assert cmd[:4] == ["mkvmerge", "--gui-mode", "-o", str(out)]
    assert "--no-video" in cmd
    assert kwargs["check"] is False
    assert kwargs["parser"] is not None
    assert kwargs["stderr_log"] == tmp_path / "out.mkv.stderr.log"


def test_default_append_never_drops_video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _fake_run_tool(monkeypatch, returncode=0)
    parts = [tmp_path / "a pt1.mkv", tmp_path / "a pt2.mkv"]
    runner_module.append_parts(TOOLS, parts, tmp_path / "c.mkv", RecordingReporter())
    ((cmd, _),) = seen
    assert "--no-video" not in cmd
    assert cmd[-2:] == [str(parts[0]), f"+{parts[1]}"]


def test_default_mux_error_exit_raises_tool_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_run_tool(monkeypatch, returncode=2)
    out = tmp_path / "out.mkv"
    with pytest.raises(ToolFailed) as info:
        runner_module.mux(TOOLS, [tmp_path / "src.mkv"], None, out, RecordingReporter())
    assert info.value.returncode == 2
    assert info.value.log_path == tmp_path / "out.mkv.stderr.log"
    assert "mkvmerge" in info.value.cmd_text


# --- more edges ----------------------------------------------------------------------------


def test_source_changed_during_encode_is_stale(settings: Settings, state: State) -> None:
    src = _source(settings, "a.avi")
    _plan(state, src)
    fakes = Fakes()

    def touch(p: Path) -> None:
        st = src.stat()
        os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 7_000_000_000))

    fakes.fail["encode_full"] = lambda p: touch(p)
    summary = _run(settings, state, fakes)
    assert summary.completed == 0
    (record,) = state.records(Status.STALE)
    assert record.note == "source changed"
    assert not (settings.completed_dir / "a.mkv").exists()
    assert not (settings.completed_dir / "manifest.csv").exists()


def test_source_deleted_after_copy_is_stale(settings: Settings, state: State) -> None:
    src = _source(settings, "a.avi")
    _plan(state, src)
    fakes = Fakes()
    fakes.after_copy = lambda p: p.unlink()
    _run(settings, state, fakes)
    (record,) = state.records(Status.STALE)
    assert record.path == src


def test_manifest_write_failure_removes_output(
    settings: Settings, state: State, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = _source(settings, "a.avi")
    other = _source(settings, "b.avi")
    _plan(state, src, est=900)
    _plan(state, other, est=100)

    def broken(path: Path, entries: Any) -> None:
        raise OSError(28, "no space left on device")

    monkeypatch.setattr(runner_module, "write_manifest", broken)
    summary = _run(settings, state, Fakes())
    assert summary.retry == 1
    assert summary.stop_reason.startswith("manifest not writable: ")
    assert _status(state, src).status is Status.RETRY
    assert _status(state, other).status is Status.PENDING
    assert _scratch_files(settings) == []
    assert not (settings.completed_dir / "a.mkv").exists()
    assert not list(settings.completed_dir.rglob("*.part"))


def test_parts_that_no_longer_group_are_flagged(settings: Settings, state: State) -> None:
    a = _source(settings, "a.avi")
    b = _source(settings, "b.avi")
    _plan(state, a, parts=(a, b))
    summary = _run(settings, state, Fakes())
    assert summary.flagged == 1
    assert "multi-part group" in _status(state, a).note


def test_record_outside_library_root_is_flagged(tmp_path: Path, state: State) -> None:
    settings = _settings(tmp_path)
    outside = tmp_path / "elsewhere" / "a.avi"
    outside.parent.mkdir()
    outside.write_bytes(b"x" * 100)
    _plan(state, outside)
    summary = _run(settings, state, Fakes())
    assert summary.flagged == 1
    assert "not under the library root" in _status(state, outside).note


def test_other_not_implemented_error_flags(settings: Settings, state: State) -> None:
    src = _source(settings, "a.avi")
    _plan(state, src)
    fakes = Fakes()
    fakes.fail["search_crf"] = lambda p: NotImplementedError("something else")
    summary = _run(settings, state, fakes)
    assert summary.flagged == 1 and summary.stop_reason == ""


def test_remux_verify_failure_flags(settings: Settings, state: State) -> None:
    src = _source(settings, "a.mp4")
    _plan(state, src, decision="remux")
    fakes = Fakes()
    fakes.verify_result = VerifyResult(ok=False, reason="extra video streams: 0 in the output")
    summary = _run(settings, state, fakes)
    assert summary.flagged == 1
    assert "extra video streams" in _status(state, src).note


# --- CLI `run` (run_queue faked) -----------------------------------------------------------


def _cli_config(tmp_path: Path, **changes: Any) -> Path:
    settings = _settings(tmp_path, **changes)
    config = tmp_path / "config.toml"
    lines = [
        f"{name} = '{getattr(settings, name)}'"
        for name in ("library_root", "scratch_dir", "completed_dir", "state_db", "log_dir")
    ]
    config.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config


def _summary(**changes: Any) -> RunSummary:
    values: dict[str, Any] = {
        "completed": 2,
        "left_alone": 1,
        "flagged": 0,
        "retry": 0,
        "bytes_saved": 3 * 1024**3,
        "stopped_by_deadline": False,
        "stop_reason": "",
    }
    values.update(changes)
    return RunSummary(**values)


def _invoke_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: RunSummary | BaseException,
    *args: str,
) -> tuple[Any, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    def fake_run_queue(*a: Any, **kw: Any) -> RunSummary:
        calls.append(kw)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(runner_module, "run_queue", fake_run_queue)
    monkeypatch.setattr(cli_module, "locate_tools", lambda: TOOLS)
    config = _cli_config(tmp_path)
    State.open(tmp_path / "appdata" / "state.db").close()  # `run` needs a surveyed state
    result = CliRunner().invoke(cli_module.app, ["run", "--config", str(config), *args])
    return result, calls


def test_cli_run_completed_exits_0_and_summarises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, calls = _invoke_run(tmp_path, monkeypatch, _summary())
    assert result.exit_code == 0, result.output
    assert "completed 2" in result.output
    assert "left alone 1" in result.output
    assert "3.0 GiB" in result.output
    assert "queue finished" in result.output
    (kw,) = calls
    assert kw["deadline"] is None and kw["only"] is None
    assert kw["current_log"].parent == tmp_path / "appdata" / "logs"


def test_cli_run_passes_deadline_and_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "library" / "Films"
    result, calls = _invoke_run(
        tmp_path,
        monkeypatch,
        _summary(stopped_by_deadline=True),
        "--until",
        "07:00",
        "--max-hours",
        "3",
        "--path",
        str(target),
    )
    assert result.exit_code == 0, result.output
    assert "deadline reached" in result.output
    (kw,) = calls
    assert kw["deadline"] is not None
    assert kw["deadline"] - datetime.now() <= timedelta(hours=3)
    assert kw["only"] == target


@pytest.mark.parametrize(
    "reason",
    [
        "scratch full",
        "library unreachable",
        "tool not found: ffmpeg",
        "manifest unreadable: bad header",
        "completed folder not writable: denied",
    ],
)
def test_cli_run_stop_reason_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    result, _ = _invoke_run(tmp_path, monkeypatch, _summary(stop_reason=reason))
    assert result.exit_code == 1
    assert f"Run stopped: {reason}" in result.output


@pytest.mark.parametrize(
    "error",
    [
        PlanNotAcknowledged("the plan has not been acknowledged"),
        RunRefused("under the library"),
        CalibrationNotApproved("calibration has not been approved: run `calibrate`"),
        AlreadyRunning(Path("video-cleaner.lock"), 4242),
    ],
)
def test_cli_run_refusals_exit_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    result, _ = _invoke_run(tmp_path, monkeypatch, error)
    assert result.exit_code == 2
    assert " ".join(str(error).split()) in " ".join(result.output.split())


def test_cli_run_bad_until_exits_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result, calls = _invoke_run(tmp_path, monkeypatch, _summary(), "--until", "25:00")
    assert result.exit_code == 2
    assert "HH:MM" in result.output
    assert calls == []


def test_cli_run_config_error_exits_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner_module, "run_queue", lambda *a, **k: _summary())
    result = CliRunner().invoke(cli_module.app, ["run", "--config", str(tmp_path / "missing.toml")])
    assert result.exit_code == 2
    assert "config error" in result.output


def test_cli_run_interrupt_exits_130(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result, _ = _invoke_run(tmp_path, monkeypatch, KeyboardInterrupt())
    assert result.exit_code == 130
    assert "interrupted" in result.output


# --- fix round 1: readiness, early refusal, manifest preflight, library collisions ---------


def _forbid_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_tool(*args: Any, **kwargs: Any) -> ToolResult:
        raise AssertionError("no tool may run")

    for module in (runner_module, probe_module, verify_module):
        monkeypatch.setattr(module, "run_tool", no_tool)

    def no_prefetch(self: Any, src: Path) -> None:
        raise AssertionError("no prefetch may start")

    monkeypatch.setattr(scratch_module.Prefetcher, "start", no_prefetch)


def test_not_implemented_in_a_stage_is_flagged_not_a_stop(settings: Settings, state: State) -> None:
    """With the hold gone, NotImplementedError is just an unexpected error: flag, carry on."""
    a = _source(settings, "a.avi")
    b = _source(settings, "b.avi")
    _plan(state, a, est=900)
    _plan(state, b, est=100)
    fakes = Fakes()
    fakes.fail["search_crf"] = lambda p: (
        NotImplementedError("unfinished stage") if p.name.endswith("a.avi") else None
    )
    summary = _run(settings, state, fakes)
    assert summary.stop_reason == ""
    assert _status(state, a).status is Status.FLAGGED
    assert _status(state, b).status is Status.COMPLETED


def test_refusal_uses_resolved_paths(
    tmp_path: Path, state: State, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _settings(tmp_path)
    alias = tmp_path / "W"
    real_resolve = paths_module._resolve

    def resolve(p: Any) -> str:
        text = str(real_resolve(p))
        prefix = str(alias)
        if os.path.normcase(text).startswith(os.path.normcase(prefix)):
            return str(base.library_root) + text[len(prefix) :]
        return text

    monkeypatch.setattr(paths_module, "_resolve", resolve)
    settings = _settings(tmp_path, scratch_dir=alias / "scratch")
    with pytest.raises(RunRefused, match="scratch folder"):
        _run(settings, state, Fakes())


def test_cli_refuses_before_creating_log_or_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = tmp_path / "library"
    called: list[Any] = []
    monkeypatch.setattr(runner_module, "run_queue", lambda *a, **k: called.append(k))
    monkeypatch.setattr(cli_module, "locate_tools", lambda: TOOLS)
    config = _cli_config(
        tmp_path, log_dir=library / "logs", state_db=library / "state" / "state.db"
    )
    result = CliRunner().invoke(cli_module.app, ["run", "--config", str(config)])
    assert result.exit_code == 2
    assert "library root" in result.output
    assert not (library / "logs").exists()
    assert not (library / "state").exists()
    assert called == []


def test_corrupt_manifest_stops_before_any_copy(settings: Settings, state: State) -> None:
    src = _source(settings, "a.avi")
    _plan(state, src)
    settings.completed_dir.mkdir(parents=True)
    (settings.completed_dir / "manifest.csv").write_text("not,a,manifest\n", encoding="utf-8")
    fakes = Fakes()
    summary = _run(settings, state, fakes)
    assert summary.stop_reason.startswith("manifest unreadable: ")
    assert fakes.calls == []
    assert _status(state, src).status is Status.PENDING


def test_read_only_manifest_stops_before_any_copy(settings: Settings, state: State) -> None:
    src = _source(settings, "a.avi")
    _plan(state, src)
    settings.completed_dir.mkdir(parents=True)
    manifest = settings.completed_dir / "manifest.csv"
    write_manifest(manifest, [])
    before = manifest.read_bytes()
    manifest.chmod(stat.S_IREAD)
    try:
        fakes = Fakes()
        summary = _run(settings, state, fakes)
    finally:
        manifest.chmod(stat.S_IREAD | stat.S_IWRITE)
    assert summary.stop_reason.startswith("manifest not writable: ")
    assert fakes.calls == []
    assert manifest.read_bytes() == before


def test_unwritable_completed_folder_stops_before_any_copy(
    settings: Settings, state: State
) -> None:
    _plan(state, _source(settings, "a.avi"))
    settings.completed_dir.write_bytes(b"a file where the folder should be")
    fakes = Fakes()
    summary = _run(settings, state, fakes)
    assert summary.stop_reason.startswith("completed folder not writable: ")
    assert fakes.calls == []


def test_manifest_turning_corrupt_mid_run_stops(
    settings: Settings, state: State, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = _source(settings, "a.avi")
    b = _source(settings, "b.avi")
    _plan(state, a, est=900)
    _plan(state, b, est=100)
    real_read = read_manifest
    reads: list[Path] = []

    def read(path: Path) -> Any:
        reads.append(path)
        if len(reads) > 2:  # fine for the startup check and reconcile, corrupt at delivery
            raise ManifestError("line 3: expected 10 fields")
        return real_read(path)

    monkeypatch.setattr(runner_module, "read_manifest", read)
    summary = _run(settings, state, Fakes())
    assert summary.stop_reason.startswith("manifest unreadable: ")
    assert _status(state, a).status is Status.RETRY
    assert _status(state, b).status is Status.PENDING
    assert not (settings.completed_dir / "a.mkv").exists()
    assert _scratch_files(settings) == []


def test_combined_name_colliding_with_library_file_is_flagged(
    settings: Settings, state: State
) -> None:
    pt1 = _source(settings, "Films/Film pt1.avi")
    pt2 = _source(settings, "Films/Film pt2.avi")
    sibling = _source(settings, "Films/Film.mkv")  # an unrelated, unplanned file
    _plan(state, pt1, parts=(pt1, pt2))
    fakes = Fakes()
    summary = _run(settings, state, fakes)
    assert summary.flagged == 1
    assert _status(state, pt1).note == f"output name collides with {sibling}"
    assert fakes.calls == []


@windows_only_case
def test_mkv_name_colliding_with_left_alone_library_file_is_flagged(
    settings: Settings, state: State
) -> None:
    avi = _source(settings, "Clips/Clip.avi")
    mkv = _source(settings, "Clips/clip.MKV")
    _plan(state, avi)
    _plan(state, mkv, decision="leave_alone", est=0)
    summary = _run(settings, state, Fakes())
    assert summary.flagged == 1
    assert "output name collides with" in _status(state, avi).note
    assert str(mkv) in _status(state, avi).note


def test_delivering_beside_its_own_source_is_fine(settings: Settings, state: State) -> None:
    avi = _source(settings, "Films/Film.avi")
    mkv = _source(settings, "Shows/Show.mkv")
    _plan(state, avi)
    _plan(state, mkv, decision="remux", est=0)  # Show.mkv -> completed Show.mkv: itself
    summary = _run(settings, state, Fakes())
    assert summary.completed == 2


# --- calibration approval ---------------------------------------------


def test_run_refused_without_calibration_approval(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_tools(monkeypatch)
    with State.open(settings.state_db) as state:
        src = _source(settings, "a.avi")
        _plan(state, src)
        _ack(state)
        events = len(state.events(10_000))
        fakes = Fakes()
        with pytest.raises(CalibrationNotApproved, match="calibrate"):
            _run(settings, state, fakes, ack=False)
        # The default (real) pipeline is refused the same way, before any tool runs.
        with pytest.raises(CalibrationNotApproved, match="calibrate"):
            run_queue(settings, TOOLS, state, RecordingReporter(), deadline=None, now=Clock())
        assert fakes.calls == []
        assert _status(state, src).status is Status.PENDING
        assert len(state.events(10_000)) == events
    assert not settings.scratch_dir.exists()
    assert not settings.completed_dir.exists()


def test_plan_ack_is_checked_before_calibration(settings: Settings) -> None:
    with State.open(settings.state_db) as state:
        state.approve_calibration(APPROVED_VMAF, APPROVED_CRF_MAX)
        with pytest.raises(PlanNotAcknowledged):
            _run(settings, state, Fakes(), ack=False)
        summary = _run(settings, state, Fakes())  # acknowledged and approved: no refusal
        assert summary.stop_reason == ""


def test_run_uses_the_approved_calibration(settings: Settings, state: State) -> None:
    assert (settings.vmaf_target, settings.crf_max) != (APPROVED_VMAF, APPROVED_CRF_MAX)
    src = _source(settings, "a.avi", size=4000)
    _plan(state, src)
    seen: list[Settings] = []
    fakes = Fakes()
    original_search = fakes.search_crf

    def search(
        info: MediaInfo, tools: Tools, s: Settings, workdir: Path, rep: Reporter
    ) -> CrfSearchResult:
        seen.append(s)
        return original_search(info, tools, s, workdir, rep)

    pipeline = fakes.pipeline()
    pipeline.search_crf = search
    _ack(state)
    summary = run_queue(
        settings, TOOLS, state, RecordingReporter(), deadline=None, now=Clock(), pipeline=pipeline
    )
    assert summary.completed == 1
    (used,) = seen
    assert (used.vmaf_target, used.crf_max) == (APPROVED_VMAF, APPROVED_CRF_MAX)
    # everything else is the configuration as given
    assert used.crf_min == settings.crf_min and used.completed_dir == settings.completed_dir
    encode_calls = [c for c in fakes.calls if c[0] == "encode_full"]
    assert len(encode_calls) == 1


def test_approved_crf_max_below_configured_crf_min_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_tools(monkeypatch)
    settings = _settings(tmp_path, crf_min=20)
    with State.open(settings.state_db) as state:
        state.approve_calibration(95.0, 18)  # approved before crf_min was raised in the config
        _plan(state, _source(settings, "a.avi"))
        fakes = Fakes()
        with pytest.raises(CalibrationNotApproved, match="crf_min"):
            _run(settings, state, fakes)
        assert fakes.calls == []


# --- plan acknowledgement must match the current plan -------------


def test_run_refused_when_plan_changed_since_ack(
    settings: Settings, state: State, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_tools(monkeypatch)
    src = _source(settings, "a.avi")
    _plan(state, src)
    _ack(state)
    src.write_bytes(b"y" * 3000)  # the file changed and a survey recorded it
    _plan(state, src)
    events = len(state.events(10_000))
    fakes = Fakes()
    with pytest.raises(
        PlanNotAcknowledged, match="plan changed since it was acknowledged"
    ) as raised:
        run_queue(
            settings,
            TOOLS,
            state,
            RecordingReporter(),
            deadline=None,
            now=Clock(),
            pipeline=fakes.pipeline(),
        )
    message = str(raised.value)
    assert "a subtitle was found or went stale" in message
    assert "run `plan`, review it, then `plan --ack`" in message
    assert fakes.calls == []
    assert len(state.events(10_000)) == events
    assert not settings.scratch_dir.exists()
    assert not settings.completed_dir.exists()


def test_a_change_between_the_plan_check_and_the_snapshot_is_refused(
    settings: Settings, state: State, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `subs resolve` that finishes a file after the early plan check, before the run takes
    its snapshot, must not smuggle an unacknowledged plan into the run."""
    a = _source(settings, "a.avi")
    b = _source(settings, "b.avi")
    _plan(state, a, est=900)
    rec_b = _plan(state, b, est=100)
    _want_subtitle(state, rec_b)
    _ack(state)
    real = runner_module.refuse_outputs_in_library
    fired: list[int] = []

    def with_a_resolve_in_between(checked: Settings) -> None:
        real(checked)  # the early plan check has passed by now
        fired.append(1)
        state.set_subtitle(
            SubtitleRecord(
                path=rec_b.path,
                size=rec_b.size,
                mtime=rec_b.mtime,
                status=SubtitleStatus.UNAVAILABLE,
                reason="none found",
            )
        )

    monkeypatch.setattr(runner_module, "refuse_outputs_in_library", with_a_resolve_in_between)
    fakes = Fakes()
    with pytest.raises(PlanNotAcknowledged, match="plan changed since it was acknowledged"):
        run_queue(
            settings,
            TOOLS,
            state,
            RecordingReporter(),
            deadline=None,
            now=Clock(),
            pipeline=fakes.pipeline(),
        )
    assert fired and fakes.calls == []
    assert not settings.completed_dir.exists()
    assert _status(state, a).status is Status.PENDING


def test_a_run_is_not_blocked_by_a_subs_resolve(settings: Settings, state: State) -> None:
    from video_cleaner.instancelock import instance_lock, subs_lock_path_for

    a = _source(settings, "a.avi")
    _plan(state, a)
    with instance_lock(subs_lock_path_for(settings.state_db)):
        assert _run(settings, state, Fakes()).completed == 1


def test_an_unchanged_plan_passes_the_second_check(settings: Settings, state: State) -> None:
    a = _source(settings, "a.avi")
    _plan(state, a)
    assert _run(settings, state, Fakes()).completed == 1


def test_stale_found_by_a_run_needs_a_new_ack(settings: Settings, state: State) -> None:
    """Deliberate: a run that marks a record STALE changes the plan (STALE drops out of it),
    so the next run is refused until `plan` is reviewed and acknowledged again."""
    a = _source(settings, "a.avi")
    b = _source(settings, "b.avi")
    _plan(state, a, est=900)
    _plan(state, b, est=100)
    _ack(state)
    fakes = Fakes()

    def touch(src: Path) -> None:
        if src == a:
            st = src.stat()
            os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))

    fakes.after_copy = touch
    pipeline = fakes.pipeline()
    first = run_queue(
        settings, TOOLS, state, RecordingReporter(), deadline=None, now=Clock(), pipeline=pipeline
    )
    assert first.completed == 1
    assert [r.path for r in state.records(Status.STALE)] == [a]
    with pytest.raises(PlanNotAcknowledged, match="plan changed"):
        run_queue(
            settings,
            TOOLS,
            state,
            RecordingReporter(),
            deadline=None,
            now=Clock(),
            pipeline=Fakes().pipeline(),
        )
    _ack(state)  # `plan` reviewed and acknowledged again
    again = run_queue(
        settings,
        TOOLS,
        state,
        RecordingReporter(),
        deadline=None,
        now=Clock(),
        pipeline=Fakes().pipeline(),
    )
    assert again.stop_reason == ""


def test_run_progress_keeps_the_ack_valid(settings: Settings, state: State) -> None:
    a = _source(settings, "a.avi")
    b = _source(settings, "b.avi")
    _plan(state, a, est=900)
    _plan(state, b, est=100)
    _ack(state)
    first = run_queue(
        settings,
        TOOLS,
        state,
        RecordingReporter(),
        deadline=None,
        now=Clock(),
        pipeline=Fakes().pipeline(),
    )
    assert first.completed == 2
    second = run_queue(  # statuses and notes changed, the decisions did not: no new ack needed
        settings,
        TOOLS,
        state,
        RecordingReporter(),
        deadline=None,
        now=Clock(),
        pipeline=Fakes().pipeline(),
    )
    assert second.stop_reason == ""


def test_approved_vmaf_target_below_guard_mean_floor_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_tools(monkeypatch)
    settings = _settings(tmp_path, guard_mean_floor=95.0, guard_worst_floor=90.0)
    with State.open(settings.state_db) as state:
        state.approve_calibration(94.0, 23)  # approved before the guard floor was raised
        _plan(state, _source(settings, "a.avi"))
        fakes = Fakes()
        with pytest.raises(CalibrationNotApproved, match="guard_mean_floor"):
            _run(settings, state, fakes)
        assert fakes.calls == []
    assert not settings.scratch_dir.exists()


# --- delivery order and crash recovery ------------------------------------


class _Kill(BaseException):
    """Stands for the process being killed: no `except Exception` cleanup runs."""


def _kill_after(monkeypatch: pytest.MonkeyPatch, point: str) -> None:
    """Make delivery die right after *point*: 'move', 'manifest' or 'rename'."""
    if point == "move":

        def no_manifest(path: Path, entries: Any) -> None:
            raise _Kill

        monkeypatch.setattr(runner_module, "write_manifest", no_manifest)
        return
    if point == "manifest":
        real_write = write_manifest

        def write_then_die(path: Path, entries: Any) -> None:
            real_write(path, entries)
            raise _Kill

        monkeypatch.setattr(runner_module, "write_manifest", write_then_die)
        return
    real_rename = os.rename

    def rename_then_die(src: Any, dst: Any) -> None:
        real_rename(src, dst)
        if str(dst).endswith(".mkv"):
            raise _Kill

    monkeypatch.setattr(os, "rename", rename_then_die)


def test_delivery_writes_manifest_row_before_the_final_rename(
    settings: Settings, state: State, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = _source(settings, "a.avi")
    _plan(state, src)
    target = settings.completed_dir / "a.mkv"
    part = settings.completed_dir / "a.mkv.part"
    seen: list[tuple[bool, bool]] = []
    real_write = write_manifest

    def spy(path: Path, entries: Any) -> None:
        seen.append((part.exists(), target.exists()))
        real_write(path, entries)

    monkeypatch.setattr(runner_module, "write_manifest", spy)
    assert _run(settings, state, Fakes()).completed == 1
    assert seen == [(True, False)]  # row written while the output is still `.part`
    assert target.is_file() and not part.exists()


def test_delivery_never_overwrites_a_target_that_appears_late(
    settings: Settings, state: State, monkeypatch: pytest.MonkeyPatch
) -> None:
    """os.rename refuses an existing target, so a file placed meanwhile survives."""
    src = _source(settings, "a.avi")
    _plan(state, src)
    target = settings.completed_dir / "a.mkv"
    real_write = write_manifest

    def write_then_someone_places_target(path: Path, entries: Any) -> None:
        real_write(path, entries)
        target.write_bytes(b"not ours")

    monkeypatch.setattr(runner_module, "write_manifest", write_then_someone_places_target)
    if os.name != "nt":  # POSIX rename overwrites; refuse the same way Windows does

        def refusing_rename(src_: Any, dst: Any) -> None:
            if Path(dst).exists():
                raise FileExistsError(17, "exists", str(dst))
            os.replace(src_, dst)

        monkeypatch.setattr(os, "rename", refusing_rename)
    summary = _run(settings, state, Fakes())
    assert summary.completed == 0
    assert target.read_bytes() == b"not ours"


@pytest.mark.parametrize("point", ["move", "manifest", "rename"])
def test_resume_after_a_crash_in_delivery(
    settings: Settings, state: State, monkeypatch: pytest.MonkeyPatch, point: str
) -> None:
    src = _source(settings, "Films/a.avi")
    _plan(state, src)
    target = settings.completed_dir / "Films" / "a.mkv"
    with monkeypatch.context() as patch:
        _kill_after(patch, point)
        with pytest.raises(_Kill):
            _run(settings, state, Fakes())
    assert _status(state, src).status is Status.VERIFYING  # what a killed run leaves

    fakes = Fakes()
    summary = _run(settings, state, fakes)

    assert _status(state, src).status is Status.COMPLETED
    assert target.is_file()
    assert not list(settings.completed_dir.rglob("*.part"))  # no partial file left
    entries = read_manifest(settings.completed_dir / "manifest.csv")
    assert [e.completed for e in entries] == [target]
    if point == "move":
        assert "encode_full" in fakes.names() and summary.completed == 1  # redone
    else:
        assert fakes.calls == []  # the finished encode was recovered, not redone


def test_startup_reconciles_manifest_and_completed_folder(settings: Settings, state: State) -> None:
    gone = _source(settings, "gone.avi")
    done = _source(settings, "done.avi")
    _plan(state, gone)
    _plan(state, done)
    state.set_status(gone, Status.RETRY, "interrupted")
    state.set_status(done, Status.COMPLETED, "encode: saved 1 bytes")
    folder = settings.completed_dir
    folder.mkdir(parents=True)

    def entry(src: Path) -> Any:
        st = src.stat()
        return ManifestEntry(
            original=(src,),
            original_sizes=(st.st_size,),
            original_mtimes=(st.st_mtime,),
            completed=folder / (src.stem + ".mkv"),
            completed_size=5,
            vmaf_mean=None,
            vmaf_worst=None,
            crf=None,
            verified=True,
            action="remux",
        )

    # gone: a row with neither the file nor its .part, record not completed -> row dropped.
    # done: delivered earlier and moved away by the owner -> row kept.
    write_manifest(folder / "manifest.csv", [entry(gone), entry(done)])
    orphan = folder / "sub" / "orphan.mkv.part"
    orphan.parent.mkdir()
    orphan.write_bytes(b"o")
    foreign = folder / "notes.part"
    foreign.write_bytes(b"not ours")

    fakes = Fakes()
    fakes.fail["copy"] = lambda p: ToolNotFound("stop")  # stop before any real work
    _run(settings, state, fakes)

    kept = [e.completed.name for e in read_manifest(folder / "manifest.csv")]
    assert kept == ["done.mkv"]
    assert not orphan.exists()
    assert foreign.exists()


def test_reconcile_keeps_history_after_overwrite_and_resurvey(
    settings: Settings, state: State
) -> None:
    """A delivered row stays after the owner overwrote the original and re-surveyed it."""
    src = _source(settings, "a.avi")
    _plan(state, src)
    assert _run(settings, state, Fakes()).completed == 1
    folder = settings.completed_dir
    before = read_manifest(folder / "manifest.csv")
    assert len(before) == 1
    # The owner overwrites the original with the delivered file, moves it away, and re-surveys:
    # the path now has a new size and mtime, so the old record is replaced by a LEFT_ALONE one.
    before[0].completed.unlink()
    src.write_bytes(b"h" * 321)
    _plan(state, src, decision="leave_alone", est=0)
    assert _status(state, src).status is Status.LEFT_ALONE

    fakes = Fakes()
    fakes.fail["copy"] = lambda p: ToolNotFound("stop")  # stop before any real work
    _run(settings, state, fakes)

    assert read_manifest(folder / "manifest.csv") == before


def test_failed_final_rename_leaves_no_row_and_no_part(
    settings: Settings, state: State, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = _source(settings, "a.avi")
    _plan(state, src)

    def denied(src_: Any, dst: Any) -> None:
        raise PermissionError(13, "denied", str(dst))

    monkeypatch.setattr(os, "rename", denied)
    summary = _run(settings, state, Fakes())
    assert summary.flagged == 1
    assert "could not be put in place" in _status(state, src).note
    assert read_manifest(settings.completed_dir / "manifest.csv") == []
    assert not list(settings.completed_dir.rglob("*.part"))
    assert not (settings.completed_dir / "a.mkv").exists()


def test_scratch_work_stays_inside_the_tool_folder(settings: Settings, state: State) -> None:
    src = _source(settings, "a.avi")
    _plan(state, src)
    fakes = Fakes()
    assert _run(settings, state, fakes).completed == 1
    ours = settings.scratch_dir / scratch_module.TOOL_FOLDER
    copy_dest = fakes.calls[0][2]
    assert copy_dest.parent == ours
    mux = next(c for c in fakes.calls if c[0] == "mux")
    assert mux[3].parent.parent == ours and mux[3].parent.name.startswith("work-")


def test_foreign_files_in_scratch_do_not_count_toward_the_cap(tmp_path: Path) -> None:
    # room for the copy, the encoded video and the output (1000 each) plus the marker
    settings = _settings(tmp_path, scratch_cap_bytes=3200)
    settings.scratch_dir.mkdir(parents=True)
    (settings.scratch_dir / "big-foreign.bin").write_bytes(b"f" * 5000)
    src = _source(settings, "a.avi")
    with _open_state(settings) as state:
        _plan(state, src)
        summary = _run(settings, state, Fakes())
    assert summary.completed == 1
    assert (settings.scratch_dir / "big-foreign.bin").exists()


def test_unmarked_tool_folder_is_never_swept(settings: Settings, state: State) -> None:
    ours = settings.scratch_dir / scratch_module.TOOL_FOLDER
    ours.mkdir(parents=True)
    keep = ours / "deadbeef-someone.mkv"
    keep.write_bytes(b"k")  # a folder of that name that we did not create
    src = _source(settings, "a.avi")
    _plan(state, src)
    fakes = Fakes()
    with pytest.raises(RunRefused, match="not created by video-cleaner"):
        _run(settings, state, fakes)
    assert keep.exists()
    assert fakes.calls == []


def test_scope_selection_does_no_path_resolution_per_record(
    settings: Settings, state: State, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Selecting records is string-only: no NAS round trip per record and per pick."""
    wanted = _source(settings, "Films/a.avi")
    _plan(state, wanted)
    for n in range(60):
        _plan(state, _source(settings, f"Other/{n:02d}.avi"))
    calls: list[str] = []
    real = paths_module._resolve

    def counting(p: str) -> str:
        calls.append(str(p))
        return real(p)

    monkeypatch.setattr(paths_module, "_resolve", counting)
    summary = _run(settings, state, Fakes(), only=settings.library_root / "Films")
    assert summary.completed == 1
    assert not any("Other" in c for c in calls)


def test_only_with_nothing_surveyed_in_scope_says_survey_first(
    settings: Settings, state: State
) -> None:
    _plan(state, _source(settings, "Films/a.avi"))
    unsurveyed = settings.library_root / "New"
    unsurveyed.mkdir()
    fakes = Fakes()
    summary = _run(settings, state, fakes, only=unsurveyed)
    assert summary.stop_reason.startswith(f"no surveyed files under {unsurveyed}")
    assert "survey --root" in summary.stop_reason and "plan --ack" in summary.stop_reason
    assert fakes.calls == []


def test_only_with_finished_files_in_scope_is_not_an_error(
    settings: Settings, state: State
) -> None:
    a = _source(settings, "Films/a.avi")
    _plan(state, a)
    state.set_status(a, Status.COMPLETED, "done")
    summary = _run(settings, state, Fakes(), only=settings.library_root / "Films")
    assert summary.stop_reason == ""


def test_cli_run_keeps_the_machine_awake_only_while_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from video_cleaner import power

    calls: list[int] = []
    monkeypatch.setattr(power, "_set_execution_state", calls.append)
    monkeypatch.setattr(power, "_WINDOWS", True)
    result, _ = _invoke_run(tmp_path, monkeypatch, _summary())
    assert result.exit_code == 0, result.output
    assert calls == [power.ES_CONTINUOUS | power.ES_SYSTEM_REQUIRED, power.ES_CONTINUOUS]


@pytest.mark.parametrize("member_state", ["missing", "stale"])
def test_multi_part_member_without_a_live_record_is_stale(
    settings: Settings, state: State, member_state: str
) -> None:
    pt1 = _source(settings, "Epic pt1.avi", size=2000)
    pt2 = _source(settings, "Epic pt2.avi", size=3000)
    _plan(state, pt1, parts=(pt1, pt2))
    if member_state == "stale":
        _plan(state, pt2, decision="leave_alone", est=0)
        state.upsert_file(  # pt2 re-surveyed with other bytes, then that record superseded too
            _info(pt2, 1, 1.0), decision="leave_alone", reasons=(), est_saving_bytes=0
        )
        state.set_status(pt2, Status.STALE)
    fakes = Fakes()
    summary = _run(settings, state, fakes)
    assert summary.completed == 0
    assert _status(state, pt1).status is Status.STALE
    assert "copy" not in fakes.names()  # never falls back to a fresh stat of the part


def test_file_failing_three_runs_is_flagged(settings: Settings, state: State) -> None:
    a = _source(settings, "a.avi")
    _plan(state, a)
    fakes = Fakes()
    fakes.fail["encode_full"] = lambda p: ToolFailed(3, "ffmpeg -i a", None)
    for run in (1, 2):
        assert _run(settings, state, fakes).retry == 1, run
        assert _status(state, a).status is Status.RETRY
        assert _status(state, a).attempts == run
    summary = _run(settings, state, fakes)
    assert summary.flagged == 1 and summary.retry == 0
    rec = _status(state, a)
    assert rec.status is Status.FLAGGED
    assert "failed 3 times" in rec.note and "exit code 3" in rec.note


def test_stop_conditions_do_not_count_as_attempts(settings: Settings, state: State) -> None:
    a = _source(settings, "a.avi")
    _plan(state, a)
    fakes = Fakes()
    fakes.fail["copy"] = lambda p: ToolNotFound("ffmpeg")  # a setup problem, not the file's
    for _ in range(4):
        _run(settings, state, fakes)
    rec = _status(state, a)
    assert rec.status is Status.RETRY and rec.attempts == 0


def test_requeue_resets_attempts(settings: Settings, state: State) -> None:
    a = _source(settings, "a.avi")
    _plan(state, a)
    fakes = Fakes()
    fakes.fail["encode_full"] = lambda p: ToolFailed(3, "ffmpeg -i a", None)
    _run(settings, state, fakes)
    assert _status(state, a).attempts == 1
    state.requeue(a)
    assert _status(state, a).attempts == 0


def test_default_pipeline_still_requires_the_plan_ack(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with (
        State.open(settings.state_db) as state,  # no ack, no calibration
        pytest.raises(PlanNotAcknowledged),
    ):
        run_queue(settings, TOOLS, state, RecordingReporter(), deadline=None, now=Clock())
    assert not settings.scratch_dir.exists()


def test_cli_run_with_corrupt_stored_calibration_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli_module, "locate_tools", lambda: TOOLS)
    config = _cli_config(tmp_path)
    db = tmp_path / "appdata" / "state.db"
    with State.open(db) as state:
        _ack(state)
    conn = sqlite3.connect(db)
    with conn:
        conn.execute("INSERT INTO meta (key, value) VALUES ('calibration', 'garbage')")
    conn.close()
    for command in ("run", "status"):
        result = CliRunner().invoke(cli_module.app, [command, "--config", str(config)])
        assert result.exit_code == 2, (command, result.output)
        assert "calibration" in result.output
    assert not (tmp_path / "scratch").exists()


def test_cli_run_without_state_database_exits_2_and_creates_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner_module, "run_queue", lambda *a, **k: _summary())
    config = _cli_config(tmp_path)
    result = CliRunner().invoke(cli_module.app, ["run", "--config", str(config)])
    assert result.exit_code == 2
    assert "survey" in result.output
    assert not (tmp_path / "appdata").exists()


# --- subtitles: deferral, verification, staging, mux, verify and delivery --------------------

SRT_BYTES = b"1\n00:00:01,000 --> 00:00:02,000\nHello there.\n"
SUB_TITLE = "English (OpenSubtitles)"
STALE_NOTE = "subtitle cache missing or changed"


def _want_subtitle(state: State, rec: FileRecord) -> None:
    state.ensure_wanted(rec.path, rec.size, rec.mtime, english_audio=True)


def _found_subtitle(
    settings: Settings,
    state: State,
    rec: FileRecord,
    *,
    data: bytes = SRT_BYTES,
    title: str = SUB_TITLE,
    name: str = "0123456789abcdef-opensubtitles-1.srt",
) -> Path:
    """Cache *data* as the resolver does, record the file as FOUND; returns the cache file."""
    settings.subtitle_cache_dir.mkdir(parents=True, exist_ok=True)
    cache = settings.subtitle_cache_dir / name
    cache.write_bytes(data)
    state.set_subtitle(
        SubtitleRecord(
            path=rec.path,
            size=rec.size,
            mtime=rec.mtime,
            status=SubtitleStatus.FOUND,
            provider="opensubtitles",
            cache_path=cache,
            sha256=hashlib.sha256(data).hexdigest(),
            provenance=title,
            hash_exact=True,
            english_audio=True,
            tried_ids=(7,),
            attempts=1,
        )
    )
    return cache


def _subtitle_of(state: State, rec: FileRecord) -> SubtitleRecord:
    sub = state.subtitle(rec.path, rec.size, rec.mtime)
    assert sub is not None
    return sub


def _is_under_scratch(settings: Settings, path: Path) -> bool:
    return paths_module.is_under(path, settings.scratch_dir)


def test_a_file_still_waiting_for_its_subtitle_is_deferred_and_counted(
    settings: Settings, state: State
) -> None:
    waiting = _source(settings, "wait.avi")
    other = _source(settings, "other.avi")
    _want_subtitle(state, _plan(state, waiting, est=900))
    _plan(state, other, est=100)
    fakes = Fakes()
    summary = _run(settings, state, fakes)
    assert summary.completed == 1
    assert summary.waiting_on_subtitles == 1
    assert [c[1] for c in fakes.calls if c[0] == "copy"] == [other]
    record = _status(state, waiting)
    assert record.status is Status.PENDING and record.note == ""
    assert not (settings.completed_dir / "wait.mkv").exists()


def test_a_deferred_retry_file_is_counted_too(settings: Settings, state: State) -> None:
    src = _source(settings, "wait.avi")
    _want_subtitle(state, _plan(state, src))
    state.set_status(src, Status.RETRY, "scratch full")
    fakes = Fakes()
    summary = _run(settings, state, fakes)
    assert fakes.calls == [] and summary.waiting_on_subtitles == 1 and summary.completed == 0


def test_deferred_files_outside_the_path_scope_are_not_counted(
    settings: Settings, state: State
) -> None:
    inside = _source(settings, "Films/in.avi")
    outside = _source(settings, "Shows/out.avi")
    _want_subtitle(state, _plan(state, inside))
    _want_subtitle(state, _plan(state, outside))
    summary = _run(settings, state, Fakes(), only=settings.library_root / "Films")
    assert summary.waiting_on_subtitles == 1


def test_include_unresolved_runs_a_waiting_file_without_a_subtitle(
    settings: Settings, state: State
) -> None:
    src = _source(settings, "wait.avi")
    _want_subtitle(state, _plan(state, src))
    fakes = Fakes()
    summary = _run(settings, state, fakes, include_unresolved=True)
    assert summary.completed == 1 and summary.waiting_on_subtitles == 0
    (mux,) = [c for c in fakes.calls if c[0] == "mux"]
    assert mux[4] is None
    (entry,) = read_manifest(settings.completed_dir / "manifest.csv")
    assert entry.subtitle == ""
    (verify,) = [c for c in fakes.calls if c[0] == "verify"]
    assert "expected_new_subtitles" not in verify[3]
    assert _subtitle_of(state, _status(state, src)).status is SubtitleStatus.WANTED


@pytest.mark.parametrize(
    "status", [SubtitleStatus.UNAVAILABLE, SubtitleStatus.NOT_NEEDED, SubtitleStatus.UNCLEAR]
)
def test_other_subtitle_outcomes_do_not_defer_a_file(
    settings: Settings, state: State, status: SubtitleStatus
) -> None:
    src = _source(settings, "a.avi")
    rec = _plan(state, src)
    state.set_subtitle(SubtitleRecord(path=rec.path, size=rec.size, mtime=rec.mtime, status=status))
    summary = _run(settings, state, Fakes())
    assert summary.completed == 1 and summary.waiting_on_subtitles == 0


def test_a_found_subtitle_is_muxed_into_a_remux_and_the_file_is_delivered(
    settings: Settings, state: State
) -> None:
    src = _source(settings, "Films/clip.mp4", size=3000)
    rec = _plan(state, src, decision="remux", est=0)
    cache = _found_subtitle(settings, state, rec)
    cache_before = (cache.read_bytes(), cache.stat().st_mtime_ns)
    fakes = Fakes()
    fakes.codec = "hevc"  # already efficient: it is only in the queue for its subtitle
    summary = _run(settings, state, fakes)

    assert summary.completed == 1 and summary.waiting_on_subtitles == 0
    (mux,) = [c for c in fakes.calls if c[0] == "mux"]
    assert mux[2] is None  # remux: no new video
    subtitle = mux[4]
    assert isinstance(subtitle, SubtitleInput)
    assert (subtitle.title, subtitle.language) == (SUB_TITLE, "eng")
    # the muxer got a private copy under the file's scratch work folder, with the cached bytes
    assert subtitle.path != cache
    assert _is_under_scratch(settings, subtitle.path)
    assert subtitle.path.parent.name.startswith("work-")
    assert fakes.staged == [(subtitle.path, SRT_BYTES)]
    assert not subtitle.path.exists()  # released with the work folder
    assert (cache.read_bytes(), cache.stat().st_mtime_ns) == cache_before

    (verify,) = [c for c in fakes.calls if c[0] == "verify"]
    assert verify[3]["encoded"] is False
    assert verify[3]["expected_new_subtitles"] == 1
    assert verify[3]["expected_subtitle_title"] == SUB_TITLE
    # delivered although the output is not smaller: a subtitle-only remux
    out = settings.completed_dir / "Films" / "clip.mkv"
    assert out.stat().st_size == 3000 + len(SRT_BYTES)
    (entry,) = read_manifest(settings.completed_dir / "manifest.csv")
    assert (entry.action, entry.subtitle, entry.verified) == ("remux", SUB_TITLE, True)
    record = _status(state, src)
    assert record.status is Status.COMPLETED
    assert "subtitle" in record.note and SUB_TITLE in record.note
    assert _scratch_files(settings) == []


def test_a_found_subtitle_is_muxed_into_an_encode(settings: Settings, state: State) -> None:
    src = _source(settings, "a.avi")
    rec = _plan(state, src)
    _found_subtitle(settings, state, rec)
    fakes = Fakes()
    summary = _run(settings, state, fakes)
    assert summary.completed == 1
    (mux,) = [c for c in fakes.calls if c[0] == "mux"]
    assert mux[2] is not None and isinstance(mux[4], SubtitleInput)
    assert fakes.staged[0][1] == SRT_BYTES
    (verify,) = [c for c in fakes.calls if c[0] == "verify"]
    assert verify[3]["encoded"] is True and verify[3]["expected_new_subtitles"] == 1
    (entry,) = read_manifest(settings.completed_dir / "manifest.csv")
    assert (entry.action, entry.subtitle, entry.crf) == ("encode", SUB_TITLE, 20)


def test_a_multi_part_group_is_appended_then_muxed_again_with_the_subtitle(
    settings: Settings, state: State
) -> None:
    pt1 = _source(settings, "Epic cd1.mkv", size=1000)
    pt2 = _source(settings, "Epic cd2.mkv", size=1000)
    rec = _plan(state, pt1, decision="remux", parts=(pt1, pt2))
    _plan(state, pt2, decision="leave_alone", est=0)
    _found_subtitle(settings, state, rec)
    fakes = Fakes()
    fakes.codec = "hevc"
    summary = _run(settings, state, fakes)
    assert summary.completed == 1, _status(state, pt1).note
    names = [n for n in fakes.names() if n not in ("probe", "copy")]
    assert names == ["append_parts", "mux", "verify"]
    combined = fakes.calls[fakes.names().index("append_parts")][2]
    (mux,) = [c for c in fakes.calls if c[0] == "mux"]
    assert mux[1] == [combined] and mux[2] is None and isinstance(mux[4], SubtitleInput)
    (verify,) = [c for c in fakes.calls if c[0] == "verify"]
    assert verify[3]["expected_new_subtitles"] == 1 and verify[3]["original_size"] == 2000
    (entry,) = read_manifest(settings.completed_dir / "manifest.csv")
    assert entry.original == (pt1, pt2) and entry.subtitle == SUB_TITLE
    assert (settings.completed_dir / "Epic.mkv").stat().st_size == 2000 + len(SRT_BYTES)
    assert _scratch_files(settings) == []


def _break_cache(kind: str, cache: Path, tmp_path: Path) -> Path | None:
    """Damage the cache file as *kind* says; returns a new cache path for the record, if any."""
    if kind == "missing":
        cache.unlink()
    elif kind == "changed":
        cache.write_bytes(SRT_BYTES + b"tampered")
    elif kind == "outside":
        # the record points at an identical file that lives outside the cache folder
        elsewhere = tmp_path / "elsewhere.srt"
        elsewhere.write_bytes(SRT_BYTES)
        return elsewhere
    elif kind == "directory":
        cache.unlink()
        cache.mkdir()
    elif kind == "symlink":
        real = cache.with_name("real.srt")
        real.write_bytes(SRT_BYTES)
        cache.unlink()
        try:
            os.symlink(real, cache)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks cannot be created here")
    return None


@pytest.mark.parametrize(
    "kind", ["missing", "changed", "outside", "directory", "symlink", "oversize"]
)
def test_a_stale_subtitle_cache_sends_the_file_to_retry_and_the_subtitle_back_to_wanted(
    tmp_path: Path, kind: str
) -> None:
    settings = _settings(tmp_path)
    with _open_state(settings) as state:
        src = _source(settings, "a.avi")
        rec = _plan(state, src)
        cache = _found_subtitle(settings, state, rec)
        moved = _break_cache(kind, cache, tmp_path)
        if moved is not None:
            state.set_subtitle(dataclasses.replace(_subtitle_of(state, rec), cache_path=moved))
        if kind == "oversize":
            settings = dataclasses.replace(settings, subtitle_max_bytes=len(SRT_BYTES) - 1)
        fakes = Fakes()
        summary = _run(settings, state, fakes)

        assert fakes.calls == []  # nothing was copied, probed or muxed
        assert summary.retry == 1 and summary.completed == 0
        assert summary.waiting_on_subtitles == 0  # it was tried, not skipped
        record = _status(state, src)
        assert record.status is Status.RETRY and record.note == STALE_NOTE
        assert record.attempts == 0  # not the file's fault
        sub = _subtitle_of(state, rec)
        assert sub.status is SubtitleStatus.WANTED
        assert (sub.cache_path, sub.sha256, sub.provenance) == (None, "", "")
        assert (sub.tried_ids, sub.attempts) == ((7,), 1)
        assert STALE_NOTE in sub.reason
        assert not (settings.completed_dir / "a.mkv").exists()
        assert _scratch_files(settings) == []

        # and the next run leaves it alone until the resolver finds a subtitle again
        again = Fakes()
        summary = _run(settings, state, again)
        assert again.calls == [] and summary.waiting_on_subtitles == 1


def test_a_stale_cache_of_the_next_file_is_not_prefetched(
    settings: Settings, state: State, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _source(settings, "first.avi")
    second = _source(settings, "second.avi")
    _plan(state, first, est=900)
    rec = _plan(state, second, est=100)
    cache = _found_subtitle(settings, state, rec)
    cache.unlink()
    started: list[Path] = []
    real_start = scratch_module.Prefetcher.start

    def recording_start(self: scratch_module.Prefetcher, src: Path) -> None:
        started.append(src)
        real_start(self, src)

    monkeypatch.setattr(scratch_module.Prefetcher, "start", recording_start)
    fakes = Fakes()
    summary = _run(settings, state, fakes)
    assert summary.completed == 1 and summary.retry == 1
    assert started == []  # `second` was never copied from the library, not even in advance
    assert [c[1] for c in fakes.calls if c[0] == "copy"] == [first]


def test_a_cache_file_changed_between_the_check_and_the_staging_is_caught(
    settings: Settings, state: State
) -> None:
    src = _source(settings, "a.avi")
    rec = _plan(state, src)
    cache = _found_subtitle(settings, state, rec)
    fakes = Fakes()

    def swap(_src: Path) -> None:
        cache.write_bytes(b"swapped after the first check")

    fakes.after_copy = swap
    summary = _run(settings, state, fakes)
    assert "copy" in fakes.names()  # the first check had passed
    assert "mux" not in fakes.names() and "verify" not in fakes.names()
    assert summary.retry == 1 and summary.completed == 0
    record = _status(state, src)
    assert record.status is Status.RETRY and record.note == STALE_NOTE
    assert _subtitle_of(state, rec).status is SubtitleStatus.WANTED
    assert not (settings.completed_dir / "a.mkv").exists()
    assert _scratch_files(settings) == []


def test_an_unusable_subtitle_title_flags_the_file(settings: Settings, state: State) -> None:
    src = _source(settings, "a.avi")
    rec = _plan(state, src)
    _found_subtitle(settings, state, rec, title="bad\ntitle")
    fakes = Fakes()
    summary = _run(settings, state, fakes)
    assert summary.flagged == 1 and "mux" not in fakes.names()
    record = _status(state, src)
    assert record.status is Status.FLAGGED and "subtitle title" in record.note
    assert _subtitle_of(state, rec).status is SubtitleStatus.FOUND
    assert _scratch_files(settings) == []


def test_an_output_that_lost_the_subtitle_is_flagged_and_never_delivered(
    settings: Settings, state: State
) -> None:
    src = _source(settings, "a.avi")
    rec = _plan(state, src)
    _found_subtitle(settings, state, rec)
    fakes = Fakes()
    fakes.drop_subtitle = True
    summary = _run(settings, state, fakes)
    assert summary.flagged == 1 and summary.completed == 0
    record = _status(state, src)
    assert record.status is Status.FLAGGED and "subtitle streams" in record.note
    assert not (settings.completed_dir / "a.mkv").exists()
    assert not (settings.completed_dir / "manifest.csv").exists()
    assert _scratch_files(settings) == []


def test_a_mux_failure_keeps_the_subtitle_found_and_retries_the_file(
    settings: Settings, state: State
) -> None:
    src = _source(settings, "a.avi")
    rec = _plan(state, src)
    _found_subtitle(settings, state, rec)
    fakes = Fakes()
    fakes.fail["mux"] = lambda p: ToolFailed(2, "mkvmerge ...", None)
    summary = _run(settings, state, fakes)
    assert summary.retry == 1
    assert _status(state, src).status is Status.RETRY
    assert _status(state, src).attempts == 1
    assert _subtitle_of(state, rec).status is SubtitleStatus.FOUND
    assert _scratch_files(settings) == []


def test_ctrl_c_during_the_mux_leaves_no_staged_subtitle_behind(
    settings: Settings, state: State
) -> None:
    src = _source(settings, "a.avi")
    rec = _plan(state, src)
    cache = _found_subtitle(settings, state, rec)
    fakes = Fakes()
    fakes.fail["mux"] = lambda p: KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        _run(settings, state, fakes)
    assert _scratch_files(settings) == []
    assert cache.read_bytes() == SRT_BYTES
    assert _subtitle_of(state, rec).status is SubtitleStatus.FOUND
    assert not (settings.completed_dir / "a.mkv").exists()


# --- a remux that existed only for its subtitle ends when the subtitle is gone -----------------

GONE_NOTE = "subtitle no longer available; nothing left to do for this file"


def _delivered_nothing(settings: Settings, fakes: Fakes) -> None:
    assert "mux" not in fakes.names() and "verify" not in fakes.names()
    assert not (settings.completed_dir / "manifest.csv").exists()
    assert (
        not list(settings.completed_dir.rglob("*.mkv")) if settings.completed_dir.exists() else True
    )
    assert _scratch_files(settings) == []


def test_a_remux_whose_subtitle_went_stale_and_ended_unavailable_delivers_nothing(
    settings: Settings, state: State
) -> None:
    src = _source(settings, "Films/clip.mp4", size=3000)
    rec = _plan(state, src, decision="remux", est=0)
    cache = _found_subtitle(settings, state, rec)
    cache.unlink()  # the cache was cleared
    first = Fakes()
    first.codec = "hevc"
    summary = _run(settings, state, first)
    assert summary.retry == 1 and first.calls == []
    assert _subtitle_of(state, rec).status is SubtitleStatus.WANTED

    # `subs resolve` finds nothing, the user acknowledges the plan, and runs again.
    state.set_subtitle(
        dataclasses.replace(
            _subtitle_of(state, rec), status=SubtitleStatus.UNAVAILABLE, reason="none found"
        )
    )
    library_before = _tree(settings.library_root)
    again = Fakes()
    again.codec = "hevc"
    summary = _run(settings, state, again)
    assert (summary.completed, summary.left_alone, summary.flagged) == (0, 1, 0)
    record = _status(state, src)
    assert record.status is Status.LEFT_ALONE and record.note == GONE_NOTE
    _delivered_nothing(settings, again)
    assert _tree(settings.library_root) == library_before


def test_include_unresolved_on_a_wanted_remux_delivers_nothing(
    settings: Settings, state: State
) -> None:
    src = _source(settings, "Films/clip.mp4", size=3000)
    rec = _plan(state, src, decision="remux", est=0)
    _want_subtitle(state, rec)
    fakes = Fakes()
    fakes.codec = "hevc"
    summary = _run(settings, state, fakes, include_unresolved=True)
    assert (summary.completed, summary.left_alone) == (0, 1)
    record = _status(state, src)
    assert record.status is Status.LEFT_ALONE and record.note == GONE_NOTE
    _delivered_nothing(settings, fakes)


def test_an_hdr_remux_that_lost_its_subtitle_is_flagged_with_its_reasons(
    settings: Settings, state: State
) -> None:
    src = _source(settings, "Films/hdr.avi", size=3000)
    rec = _plan(state, src, decision="remux", est=0)
    _want_subtitle(state, rec)
    fakes = Fakes()
    fakes.transfer = "smpte2084"
    summary = _run(settings, state, fakes, include_unresolved=True)
    assert (summary.completed, summary.flagged) == (0, 1)
    record = _status(state, src)
    assert record.status is Status.FLAGGED
    assert record.note == "HDR (transfer smpte2084)"
    _delivered_nothing(settings, fakes)


def test_a_single_remux_that_the_probe_still_wants_encoded_or_remuxed_goes_on(
    settings: Settings, state: State
) -> None:
    src = _source(settings, "Show/clip.mp4", size=3000)
    _plan(state, src, decision="remux", est=0)  # no subtitle record; h264 would be encoded
    fakes = Fakes()
    assert _run(settings, state, fakes).completed == 1
    assert fakes.names().count("mux") == 1


def test_a_multi_part_join_without_a_subtitle_still_delivers(
    settings: Settings, state: State
) -> None:
    pt1 = _source(settings, "Epic cd1.mkv", size=1000)
    pt2 = _source(settings, "Epic cd2.mkv", size=1000)
    rec = _plan(state, pt1, decision="remux", parts=(pt1, pt2))
    _plan(state, pt2, decision="leave_alone", est=0)
    _want_subtitle(state, rec)
    fakes = Fakes()
    fakes.codec = "hevc"
    summary = _run(settings, state, fakes, include_unresolved=True)
    assert summary.completed == 1 and summary.left_alone == 0
    assert (settings.completed_dir / "Epic.mkv").stat().st_size == 2000


def _abandon_by_projection(fakes: Fakes) -> None:
    fakes.crf = CrfSearchResult(crf=24, projected_saving=0.05, points=())


def _abandon_by_guard(fakes: Fakes) -> None:
    fakes.scores = [94.0, 86.0, 95.0]


def _abandon_by_size(fakes: Fakes) -> None:
    fakes.ratio = 0.95  # the encode is only 5% smaller: verify says too large


ABANDONED = {
    "projection": (_abandon_by_projection, "projected saving 5.0%", 0, 1),
    "guard": (_abandon_by_guard, "quality guard failed", 1, 1),
    "too_large": (_abandon_by_size, "output not kept", 1, 2),
}


@pytest.mark.parametrize("path", list(ABANDONED))
def test_an_abandoned_encode_delivers_a_subtitle_only_remux_instead(
    settings: Settings, state: State, path: str
) -> None:
    abandon, why, encodes, verifies = ABANDONED[path]
    src = _source(settings, "Films/a.avi", size=3000)
    rec = _plan(state, src)
    cache = _found_subtitle(settings, state, rec)
    library_before = _tree(settings.library_root)
    fakes = Fakes()
    abandon(fakes)
    summary = _run(settings, state, fakes)

    assert (summary.completed, summary.left_alone, summary.flagged) == (1, 0, 0)
    assert fakes.names().count("encode_full") == encodes
    muxes = [c for c in fakes.calls if c[0] == "mux"]
    assert muxes[-1][2] is None  # the delivered file is a remux: the video is the original's
    assert isinstance(muxes[-1][4], SubtitleInput)
    assert fakes.staged[-1][1] == SRT_BYTES
    verifies_made = [c for c in fakes.calls if c[0] == "verify"]
    assert len(verifies_made) == verifies
    assert verifies_made[-1][3]["encoded"] is False  # no size gate for a remux
    assert verifies_made[-1][3]["expected_new_subtitles"] == 1
    out = settings.completed_dir / "Films" / "a.mkv"
    assert out.stat().st_size == 3000 + len(SRT_BYTES)
    (entry,) = read_manifest(settings.completed_dir / "manifest.csv")
    assert (entry.action, entry.subtitle, entry.crf) == ("remux", SUB_TITLE, None)
    assert (entry.vmaf_mean, entry.vmaf_worst, entry.verified) == (None, None, True)
    record = _status(state, src)
    assert record.status is Status.COMPLETED
    assert why in record.note and "subtitle-only remux" in record.note
    assert _subtitle_of(state, rec).status is SubtitleStatus.FOUND
    assert _scratch_files(settings) == []
    assert _tree(settings.library_root) == library_before
    assert cache.read_bytes() == SRT_BYTES


@pytest.mark.parametrize("path", ["projection", "guard"])
def test_an_abandoned_encode_whose_remux_fails_verify_is_flagged_and_delivers_nothing(
    settings: Settings, state: State, path: str
) -> None:
    abandon, _, _, _ = ABANDONED[path]
    src = _source(settings, "a.avi")
    rec = _plan(state, src)
    _found_subtitle(settings, state, rec)
    fakes = Fakes()
    abandon(fakes)
    fakes.drop_subtitle = True  # the fallback mux loses the subtitle
    summary = _run(settings, state, fakes)
    assert (summary.flagged, summary.completed, summary.left_alone) == (1, 0, 0)
    record = _status(state, src)
    assert record.status is Status.FLAGGED and "subtitle streams" in record.note
    assert not (settings.completed_dir / "a.mkv").exists()
    assert not (settings.completed_dir / "manifest.csv").exists()
    assert _scratch_files(settings) == []


def test_a_too_large_encode_whose_fallback_remux_fails_verify_is_flagged_and_delivers_nothing(
    settings: Settings, state: State
) -> None:
    """The size gate rejects the encode (its subtitle is fine), then the subtitle-only remux
    that replaces it loses the subtitle: nothing may be delivered."""
    src = _source(settings, "a.avi")
    rec = _plan(state, src)
    _found_subtitle(settings, state, rec)
    fakes = Fakes()
    _abandon_by_size(fakes)
    real_mux = fakes.mux
    muxes: list[int] = []

    def mux_then_lose_the_subtitle(*args: Any, **kwargs: Any) -> None:
        muxes.append(1)
        fakes.drop_subtitle = len(muxes) > 1  # only the fallback remux is broken
        fakes.subtitled.clear()  # the fallback writes the same output path as the encode did
        real_mux(*args, **kwargs)

    fakes.mux = mux_then_lose_the_subtitle  # type: ignore[method-assign]
    summary = _run(settings, state, fakes)

    assert len(muxes) == 2 and fakes.names().count("encode_full") == 1
    verifies = [c for c in fakes.calls if c[0] == "verify"]
    assert [v[3]["encoded"] for v in verifies] == [True, False]
    assert (summary.flagged, summary.completed, summary.left_alone) == (1, 0, 0)
    record = _status(state, src)
    assert record.status is Status.FLAGGED and "subtitle streams" in record.note
    assert not (settings.completed_dir / "a.mkv").exists()
    assert not (settings.completed_dir / "manifest.csv").exists()
    assert _scratch_files(settings) == []


@pytest.mark.parametrize("path", list(ABANDONED))
def test_an_abandoned_multi_part_encode_muxes_the_combined_file_with_the_subtitle(
    settings: Settings, state: State, path: str
) -> None:
    abandon, why, _, _ = ABANDONED[path]
    pt1 = _source(settings, "Epic cd1.avi", size=2000)
    pt2 = _source(settings, "Epic cd2.avi", size=3000)
    rec = _plan(state, pt1, parts=(pt1, pt2))
    _plan(state, pt2, decision="leave_alone", est=0)
    _found_subtitle(settings, state, rec)
    fakes = Fakes()
    abandon(fakes)
    summary = _run(settings, state, fakes)
    assert summary.completed == 1, _status(state, pt1).note
    combined = fakes.calls[fakes.names().index("append_parts")][2]
    last = [c for c in fakes.calls if c[0] == "mux"][-1]
    assert last[1] == [combined] and last[2] is None and isinstance(last[4], SubtitleInput)
    assert (settings.completed_dir / "Epic.mkv").stat().st_size == 5000 + len(SRT_BYTES)
    (entry,) = read_manifest(settings.completed_dir / "manifest.csv")
    assert (entry.action, entry.original, entry.subtitle) == ("remux", (pt1, pt2), SUB_TITLE)
    assert why in _status(state, pt1).note
    assert _scratch_files(settings) == []


@pytest.mark.parametrize("path", list(ABANDONED))
def test_an_abandoned_encode_without_a_found_subtitle_is_left_alone_as_before(
    settings: Settings, state: State, path: str
) -> None:
    abandon, why, encodes, _ = ABANDONED[path]
    src = _source(settings, "a.avi")
    rec = _plan(state, src)
    _want_subtitle(state, rec)
    fakes = Fakes()
    abandon(fakes)
    summary = _run(settings, state, fakes, include_unresolved=True)
    assert (summary.left_alone, summary.completed) == (1, 0)
    record = _status(state, src)
    assert record.status is Status.LEFT_ALONE and why in record.note
    assert "subtitle" not in record.note
    assert fakes.names().count("encode_full") == encodes
    assert fakes.names().count("mux") == encodes  # no extra remux
    assert not (settings.completed_dir / "manifest.csv").exists()


def test_the_remux_reservation_leaves_room_for_the_subtitle(tmp_path: Path) -> None:
    # room for the copy and an output of the video's size (1000 each) but not for the subtitle
    settings = _settings(tmp_path, scratch_cap_bytes=3200, subtitle_max_bytes=5000)
    src = _source(settings, "a.mp4", size=1000)
    with _open_state(settings) as state:
        rec = _plan(state, src, decision="remux", est=0)
        _found_subtitle(settings, state, rec)
        fakes = Fakes()
        summary = _run(settings, state, fakes)
        assert summary.stop_reason == "scratch full" and summary.completed == 0
        assert "mux" not in fakes.names()
        assert _status(state, src).status is Status.RETRY
        assert _scratch_files(settings) == []


def test_the_library_and_the_cache_are_never_written_when_a_subtitle_is_muxed(
    settings: Settings, state: State
) -> None:
    src = _source(settings, "Films/a.mp4", size=3000)
    rec = _plan(state, src, decision="remux", est=0)
    _found_subtitle(settings, state, rec)
    library_before = _tree(settings.library_root)
    cache_before = _tree(settings.subtitle_cache_dir)
    fakes = Fakes()
    fakes.codec = "hevc"
    assert _run(settings, state, fakes).completed == 1
    assert _tree(settings.library_root) == library_before
    assert _tree(settings.subtitle_cache_dir) == cache_before
    for call in fakes.calls:
        if call[0] == "mux":
            for written in (call[3], call[4].path):
                assert not paths_module.is_under(written, settings.library_root)
                assert not paths_module.is_under(written, settings.subtitle_cache_dir)


def test_the_nas_write_guard_still_refuses_a_cache_folder_inside_the_library(
    tmp_path: Path,
) -> None:
    base = _settings(tmp_path)
    settings = _settings(tmp_path, subtitle_cache_dir=base.library_root / "subs")
    with _open_state(base) as state, pytest.raises(RunRefused, match="library root"):
        _run(settings, state, Fakes())


def test_cli_run_passes_include_unresolved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, calls = _invoke_run(tmp_path, monkeypatch, _summary())
    assert calls[0]["include_unresolved"] is False
    _, calls = _invoke_run(tmp_path, monkeypatch, _summary(), "--include-unresolved")
    assert calls[0]["include_unresolved"] is True


def test_cli_run_summary_says_how_many_files_wait_on_subtitles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _ = _invoke_run(tmp_path, monkeypatch, _summary(waiting_on_subtitles=3))
    assert "3 files waiting on subtitles" in result.output
    assert "--include-unresolved" in result.output
    result, _ = _invoke_run(tmp_path, monkeypatch, _summary())
    assert "waiting on subtitles" not in result.output


# --- a run ignores what changed after it started (a `subs resolve` beside it) --------------


def _mid_run(fakes: Fakes, trigger: Path, change: Callable[[], object]) -> list[int]:
    """Run *change* once, right after the copy of *trigger* (between two files of a run)."""
    done: list[int] = []

    def after_copy(src: Path) -> None:
        if src == trigger and not done:
            done.append(1)
            change()

    fakes.after_copy = after_copy
    return done


def test_a_file_that_turns_found_after_the_run_started_is_not_processed(
    settings: Settings, state: State
) -> None:
    first = _source(settings, "first.avi")
    later = _source(settings, "later.avi")
    _plan(state, first, est=900)
    later_rec = _plan(state, later, decision="remux", est=100)
    fakes = Fakes()
    done = _mid_run(fakes, first, lambda: _found_subtitle(settings, state, later_rec))
    summary = _run(settings, state, fakes)
    assert done and summary.completed == 1
    assert summary.changed_since_start == 1
    assert [c[1] for c in fakes.calls if c[0] == "copy"] == [first]
    assert _status(state, later).status is Status.PENDING
    assert not (settings.completed_dir / "later.mkv").exists()


def test_a_left_alone_file_the_job_decides_again_mid_run_is_not_processed(
    settings: Settings, state: State
) -> None:
    first = _source(settings, "first.avi")
    later = _source(settings, "later.avi")
    _plan(state, first, est=900)
    later_rec = _plan(state, later, decision="leave_alone", est=0)
    assert later_rec.status is Status.LEFT_ALONE

    def job_turns_it_into_a_remux() -> None:
        _found_subtitle(settings, state, later_rec)
        survey_module.redecide_found(
            _info(later, later_rec.size, later_rec.mtime, codec="hevc"),
            settings,
            state,
            movie_hash=None,
        )

    fakes = Fakes()
    _mid_run(fakes, first, job_turns_it_into_a_remux)
    summary = _run(settings, state, fakes)
    assert _status(state, later).decision == "remux"  # the change really happened
    assert summary.completed == 1 and summary.changed_since_start == 1
    assert [c[1] for c in fakes.calls if c[0] == "copy"] == [first]
    assert _status(state, later).status is Status.PENDING


def test_a_file_that_was_found_at_the_start_is_processed_as_before(
    settings: Settings, state: State
) -> None:
    first = _source(settings, "first.avi")
    found = _source(settings, "found.avi", size=3000)
    other = _source(settings, "other.avi")
    _plan(state, first, est=900)
    found_rec = _plan(state, found, decision="remux", est=0)
    other_rec = _plan(state, other, est=100)
    _found_subtitle(settings, state, found_rec)
    fakes = Fakes()
    # An unrelated change mid-run (another file's subtitle row) does not touch the found one.
    _mid_run(fakes, first, lambda: _want_subtitle(state, other_rec))
    summary = _run(settings, state, fakes)
    assert summary.changed_since_start == 0
    assert summary.completed == 2  # `other` waits for its (new) subtitle lookup
    assert (settings.completed_dir / "found.mkv").is_file()
    assert _subtitle_of(state, _status(state, found)).status is SubtitleStatus.FOUND
    (entry,) = [e for e in read_manifest(settings.completed_dir / "manifest.csv") if e.subtitle]
    assert entry.original == (found,)


def test_a_file_that_was_waiting_at_the_start_stays_deferred_when_it_turns_found(
    settings: Settings, state: State
) -> None:
    first = _source(settings, "first.avi")
    waiting = _source(settings, "wait.avi")
    _plan(state, first, est=900)
    waiting_rec = _plan(state, waiting, est=100)
    _want_subtitle(state, waiting_rec)
    fakes = Fakes()
    _mid_run(fakes, first, lambda: _found_subtitle(settings, state, waiting_rec))
    summary = _run(settings, state, fakes)
    assert summary.completed == 1 and summary.changed_since_start == 1
    assert summary.waiting_on_subtitles == 0  # it is no longer WANTED, only unreviewed
    assert [c[1] for c in fakes.calls if c[0] == "copy"] == [first]
    assert _status(state, waiting).status is Status.PENDING


def test_a_new_record_for_a_changed_file_is_not_picked_up_mid_run(
    settings: Settings, state: State
) -> None:
    first = _source(settings, "first.avi")
    other = _source(settings, "other.avi")
    _plan(state, first, est=900)
    _plan(state, other, est=100)

    def resurvey() -> None:
        other.write_bytes(b"y" * 2000)  # the file changed and a survey recorded the new one
        _plan(state, other, est=100)

    fakes = Fakes()
    _mid_run(fakes, first, resurvey)
    summary = _run(settings, state, fakes)
    assert summary.completed == 1 and summary.changed_since_start == 1
    assert [c[1] for c in fakes.calls if c[0] == "copy"] == [first]


def test_a_retry_inside_the_same_run_is_still_not_repeated(
    settings: Settings, state: State
) -> None:
    src = _source(settings, "a.avi")
    _plan(state, src)
    fakes = Fakes()
    fakes.fail["search_crf"] = lambda p: OSError("disk")
    summary = _run(settings, state, fakes)
    assert summary.retry == 1 and summary.changed_since_start == 0
    assert len([c for c in fakes.calls if c[0] == "copy"]) == 1


def test_cli_run_summary_says_how_many_files_changed_after_the_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _ = _invoke_run(tmp_path, monkeypatch, _summary(changed_since_start=2))
    text = " ".join(result.output.split())
    assert "2 files changed after the run started" in text
    assert "a subtitle was found or the plan was updated" in text
    assert "run `plan`, review it, then `plan --ack`" in text
    result, _ = _invoke_run(tmp_path, monkeypatch, _summary(changed_since_start=1))
    assert "1 file changed after the run started" in " ".join(result.output.split())
    result, _ = _invoke_run(tmp_path, monkeypatch, _summary())
    assert "changed after the run started" not in result.output


# --- integration: the real pipeline on a tiny clip -----------------------------------------


def _content_tree(root: Path) -> dict[str, tuple[bytes, int]]:
    """Every file under *root* with its bytes and mtime: a byte-identical comparison."""
    return {
        str(Path(folder, name).relative_to(root)): (
            Path(folder, name).read_bytes(),
            Path(folder, name).stat().st_mtime_ns,
        )
        for folder, _dirs, names in os.walk(root)
        for name in names
    }


@pytest.mark.integration
def test_real_guard_window_scores_of_a_real_mux(tmp_path: Path) -> None:
    tools = vmaf_tools()
    reporter = MediaReporter()
    clip = make_clip(tools, tmp_path / "src.mkv", duration_s=6.0, rich=True)
    info = probe_module.probe(clip, tools, reporter)
    settings = _settings(tmp_path)
    video = tmp_path / "video.mkv"
    encode_module.encode_full(info, tools, settings, 24, video, reporter)
    output = tmp_path / "out.mkv"
    runner_module.mux(tools, [clip], video, output, reporter)
    scores = runner_module.guard_window_scores(
        tools, info, output, settings, tmp_path / "work", reporter
    )
    assert len(scores) == 1  # a 6 s film has room for one (shortened) window
    assert 80.0 < scores[0] <= 100.0
    # the original against itself scores about 100
    (same,) = runner_module.guard_window_scores(
        tools, info, clip, settings, tmp_path / "work", reporter
    )
    assert same > 99.0


@pytest.mark.integration
def test_real_run_queue_encodes_one_tiny_clip_end_to_end(tmp_path: Path) -> None:
    tools = vmaf_tools()
    settings = _settings(tmp_path)
    clip = make_clip(
        tools, settings.library_root / "Films" / "Tiny [x] (2024).mkv", duration_s=8.0, rich=True
    )
    library_before = _content_tree(settings.library_root)
    reporter = MediaReporter()
    with State.open(settings.state_db) as state:
        info = probe_module.probe(clip, tools, reporter)
        state.upsert_file(info, decision="encode", reasons=("test",), est_saving_bytes=info.size)
        state.approve_calibration(95.0, 24)
        _ack(state)
        hash_before = plan_hash(build_plan(state))
        summary = run_queue(settings, tools, state, reporter, deadline=None)
        record = _status(state, clip)
        assert plan_hash(build_plan(state)) == hash_before
    assert summary.stop_reason == ""
    assert (summary.completed, summary.flagged, summary.retry) == (1, 0, 0), record.note
    assert record.status is Status.COMPLETED

    target = settings.completed_dir / "Films" / "Tiny [x] (2024).mkv"
    out = probe_module.probe(target, tools, reporter)
    assert out.video is not None
    assert (out.video.codec, out.video.pix_fmt) == ("hevc", "yuv420p10le")
    assert [a.codec for a in out.audio] == [a.codec for a in info.audio]
    assert len(out.subtitles) == len(info.subtitles) == 1
    assert out.chapters == info.chapters == 2
    assert out.cover_art == info.cover_art >= 1
    assert out.size < info.size * 0.9

    (row,) = read_manifest(settings.completed_dir / "manifest.csv")
    assert row.completed == target
    assert row.original == (clip,)
    assert row.action == "encode"
    assert row.crf is not None and 16 <= row.crf <= 24
    assert row.vmaf_mean is not None and row.vmaf_mean >= settings.guard_mean_floor
    assert row.verified

    assert _content_tree(settings.library_root) == library_before  # byte-identical
    assert _scratch_files(settings) == []
    assert any(s.startswith("encode CRF") for s in reporter.stages)


def test_the_stale_subtitle_warning_shows_a_hostile_file_name_only_escaped(
    state: State, caplog: pytest.LogCaptureFixture
) -> None:
    from types import SimpleNamespace

    from video_cleaner.subtitles.stage import SubtitleCacheStale

    sub = SubtitleRecord(Path("D:/lib/evil\u202e\u200b.mkv"), 1, 1.0, SubtitleStatus.FOUND)
    with caplog.at_level("WARNING", logger="video_cleaner.runner"):
        runner_module._Runner._subtitle_gone(
            SimpleNamespace(state=state),  # type: ignore[arg-type]
            sub,
            SubtitleCacheStale("missing"),
        )
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "\u202e" not in text and "\u200b" not in text
    assert "evil" in text
