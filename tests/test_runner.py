"""run_queue orchestration, against a fake pipeline.

Encoding, muxing and VMAF are on hold (no real encode may run), so every test injects a
`Pipeline` of fakes that write small dummy files; the real copy code copies small dummy sources.
No ffmpeg, ffprobe or mkvmerge process is started here.
"""

from __future__ import annotations

import dataclasses
import os
import re
import stat
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from video_cleaner import cli as cli_module
from video_cleaner import doctor as doctor_module
from video_cleaner import paths as paths_module
from video_cleaner import probe as probe_module
from video_cleaner import runner as runner_module
from video_cleaner import scratch as scratch_module
from video_cleaner import verify as verify_module
from video_cleaner.config import Settings
from video_cleaner.encode import CrfSearchResult
from video_cleaner.held import HELD_REASON
from video_cleaner.instancelock import AlreadyRunning, instance_lock, lock_path_for
from video_cleaner.manifest import ManifestEntry, ManifestError, read_manifest, write_manifest
from video_cleaner.planning import build_plan, plan_hash
from video_cleaner.probe import AudioStream, MediaInfo, ProbeError, VideoStream
from video_cleaner.progress import Progress, Reporter
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
from video_cleaner.state import FileRecord, State, Status
from video_cleaner.tools import ToolFailed, ToolNotFound, ToolOutputError, ToolResult, Tools
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

    def stage(self, label: str) -> None:
        self.stages.append(label)

    def update(self, p: Progress) -> None:
        self.updates.append(p)

    def heartbeat(self, label: str, elapsed_s: float) -> None:
        pass

    def file_done(self, summary: str) -> None:
        self.done.append(summary)


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
        return _info(path, st.st_size, st.st_mtime, duration, self.codec)

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
    ) -> None:
        self.calls.append(("mux", list(sources), new_video, output))
        self._check("mux", output)
        size = sum(p.stat().st_size for p in sources)
        if new_video is not None:
            size = int(size * self.ratio)
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

    monkeypatch.setattr(doctor_module, "prune_logs", fake_prune)
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


def test_held_stub_stops_run_and_leaves_pending(settings: Settings, state: State) -> None:
    a = _source(settings, "a.avi")
    b = _source(settings, "b.avi")
    _plan(state, a, est=900)
    _plan(state, b, est=100)
    fakes = Fakes()
    pipeline = fakes.pipeline()

    def held_mid_run(*args: Any) -> CrfSearchResult:
        raise NotImplementedError(HELD_REASON)  # not marked held: only found when called

    pipeline.search_crf = held_mid_run
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
    assert summary.stop_reason == HELD_REASON == "held: Task 10b"
    assert summary.flagged == 0 and summary.retry == 0 and summary.completed == 0
    assert _status(state, a).status is Status.PENDING
    assert _status(state, b).status is Status.PENDING
    assert _scratch_files(settings) == []


def test_default_pipeline_post_encode_scores_are_held() -> None:
    with pytest.raises(NotImplementedError, match="held: Task 10b"):
        Pipeline().window_scores(
            TOOLS,
            _info(Path("x"), 1, 1.0),
            Path("y"),
            _settings_stub(),
            Path("w"),
            RecordingReporter(),
        )


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
        "held: Task 10b",
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


def test_default_pipeline_is_refused_before_anything(
    settings: Settings, state: State, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_tools(monkeypatch)
    enc = _source(settings, "a.avi")
    remux = _source(settings, "b.mp4")
    pt1 = _source(settings, "C pt1.avi")
    pt2 = _source(settings, "C pt2.avi")
    stuck = _source(settings, "d.avi")
    _plan(state, enc, est=900)
    _plan(state, remux, decision="remux", est=0)
    _plan(state, pt1, decision="remux", parts=(pt1, pt2))
    _plan(state, stuck)
    state.set_status(stuck, Status.ENCODING)
    _ack(state)
    before = {r.path: (r.status, r.note) for r in state.records()}
    events = len(state.events(10_000))

    for only in (None, remux, pt1):
        summary = run_queue(
            settings, TOOLS, state, RecordingReporter(), deadline=None, only=only, now=Clock()
        )
        assert summary == RunSummary(0, 0, 0, 0, 0, False, HELD_REASON)

    assert {r.path: (r.status, r.note) for r in state.records()} == before
    assert len(state.events(10_000)) == events  # not even reset_in_progress ran
    assert not settings.scratch_dir.exists()
    assert not settings.completed_dir.exists()


def test_stage_marked_held_is_refused_up_front(settings: Settings, state: State) -> None:
    _plan(state, _source(settings, "a.avi"))
    fakes = Fakes()
    pipeline = fakes.pipeline()

    def scores(*args: Any) -> Sequence[float]:
        raise NotImplementedError(HELD_REASON)

    scores.held = True  # type: ignore[attr-defined]
    pipeline.window_scores = scores
    _ack(state)
    summary = run_queue(
        settings, TOOLS, state, RecordingReporter(), deadline=None, now=Clock(), pipeline=pipeline
    )
    assert summary.stop_reason == HELD_REASON
    assert fakes.calls == []


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


# --- calibration approval (Task 14, ruling R3) ---------------------------------------------


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
        # The held check comes first (fix list item 20): the default pipeline says "held".
        held = run_queue(settings, TOOLS, state, RecordingReporter(), deadline=None, now=Clock())
        assert held.stop_reason == HELD_REASON
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


# --- plan acknowledgement must match the current plan (Task 14 fix, ruling R15) -------------


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
    assert fakes.calls == []
    assert len(state.events(10_000)) == events
    assert not settings.scratch_dir.exists()
    assert not settings.completed_dir.exists()


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


# --- delivery order and crash recovery (plan section 6) ------------------------------------


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
    settings = _settings(tmp_path, scratch_cap_bytes=3000)
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


def test_held_is_reported_before_plan_ack_and_calibration(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with State.open(settings.state_db) as state:  # no ack, no calibration
        summary = run_queue(settings, TOOLS, state, RecordingReporter(), deadline=None, now=Clock())
    assert summary.stop_reason == HELD_REASON


def test_cli_run_without_state_database_exits_2_and_creates_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner_module, "run_queue", lambda *a, **k: _summary())
    config = _cli_config(tmp_path)
    result = CliRunner().invoke(cli_module.app, ["run", "--config", str(config)])
    assert result.exit_code == 2
    assert "survey" in result.output
    assert not (tmp_path / "appdata").exists()
