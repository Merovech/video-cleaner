"""Calibration sweep (fake engine only: no encode or VMAF may run) and `calibrate` CLI."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from helpers import add_record as _add
from helpers import tmp_settings, write_config
from video_cleaner import calibrate as calibrate_mod
from video_cleaner import scratch as scratch_mod
from video_cleaner.calibrate import (
    CALIBRATION_CRFS,
    CalibrationEngine,
    CalibrationError,
    CalibrationRow,
    SampleEncode,
    build_sidebyside_cmd,
    format_calibration_table,
    run_calibration,
    select_samples,
)
from video_cleaner.cli import app
from video_cleaner.config import Settings
from video_cleaner.instancelock import AlreadyRunning, instance_lock, lock_path_for
from video_cleaner.progress import Progress, Reporter
from video_cleaner.state import FileRecord, State, Status
from video_cleaner.tools import Tools

TOOLS = Tools(ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe"), mkvmerge=Path("mkvmerge"))


class NullReporter:
    def stage(self, label: str) -> None:
        pass

    def update(self, p: Progress) -> None:
        pass

    def heartbeat(self, label: str, elapsed_s: float) -> None:
        pass

    def file_done(self, summary: str) -> None:
        pass


def _rec(
    name: str, size: int, decision: str = "encode", status: Status = Status.PENDING
) -> FileRecord:
    return FileRecord(
        path=Path("/lib") / name,
        size=size,
        mtime=1.0,
        status=status,
        decision=decision,
        reasons=("x",),
        est_saving_bytes=0,
        parts=(),
        note="",
    )


# --- select_samples ------------------------------------------------------------------------


def test_select_samples_spreads_over_the_size_range() -> None:
    records = [_rec(f"f{i}.avi", size) for i, size in enumerate([50, 10, 90, 30, 70, 20, 80])]
    picked = select_samples(records, 3)
    assert [r.size for r in picked] == [10, 50, 90]
    assert select_samples(list(reversed(records)), 3) == picked  # input order does not matter
    assert [r.size for r in select_samples(records, 1)] == [50]  # the median
    assert len(select_samples(records, 99)) == 7


def test_select_samples_only_live_encode_candidates() -> None:
    records = [
        _rec("a.avi", 10),
        _rec("b.mkv", 20, decision="leave_alone", status=Status.LEFT_ALONE),
        _rec("c.mkv", 30, decision="flag", status=Status.FLAGGED),
        _rec("d.avi", 40, status=Status.STALE),
        _rec("e.mp4", 50, decision="remux"),
        _rec("f.avi", 60, status=Status.COMPLETED),
    ]
    assert [r.path.name for r in select_samples(records, 10)] == ["a.avi", "f.avi"]


def test_select_samples_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        select_samples([_rec("a.avi", 1)], 0)


# --- side-by-side command ------------------------------------------------------------------


def test_build_sidebyside_cmd() -> None:
    cmd = build_sidebyside_cmd(
        TOOLS, Path("/s/orig [1].mkv"), Path("/w/enc.mkv"), Path("/c/out.mp4"), 600.0, 20.0
    )
    assert cmd[0] == "ffmpeg"
    first, second = (i for i, a in enumerate(cmd) if a == "-i")
    assert cmd[first + 1] == str(Path("/s/orig [1].mkv"))
    assert cmd[second + 1] == str(Path("/w/enc.mkv"))
    # seek and length before each input
    assert cmd[first - 4 : first] == ["-ss", "600.000", "-t", "20.000"]
    assert cmd[second - 4 : second] == ["-ss", "600.000", "-t", "20.000"]
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "hstack" in graph and graph.count("scale=") == 2
    assert "drawtext" not in graph
    assert "-an" in cmd
    assert cmd[cmd.index("-c:v") + 1] == "libx264"
    assert cmd[cmd.index("-crf") + 1] == "18"
    assert cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"
    assert cmd[-2:] == ["-y", str(Path("/c/out.mp4"))]
    assert all(isinstance(a, str) for a in cmd)


@pytest.mark.parametrize(
    ("start", "length"), [(-1.0, 10.0), (0.0, 0.0), (float("nan"), 10.0), (0.0, float("inf"))]
)
def test_build_sidebyside_cmd_rejects_bad_window(start: float, length: float) -> None:
    with pytest.raises(ValueError):
        build_sidebyside_cmd(TOOLS, Path("a"), Path("b"), Path("c"), start, length)


# --- table ---------------------------------------------------------------------------------


def test_format_calibration_table() -> None:
    rows = [
        CalibrationRow(Path("/lib/Heat.avi"), 17, 61.25, 97.5, 95.0),
        CalibrationRow(Path("/lib/Heat.avi"), 26, 22.5, 90.25, 84.0),
    ]
    table = format_calibration_table(rows)
    assert "Heat.avi" in table
    assert "61.2" in table and "97.50" in table and "95.00" in table
    assert "22.5" in table and "90.25" in table and "84.00" in table
    assert "CRF" in table and "size %" in table and "VMAF" in table


def test_format_calibration_table_empty() -> None:
    assert "no samples" in format_calibration_table([])


# --- the sweep (fake engine) ---------------------------------------------------------------


class FakeEngine:
    """Writes tiny dummy files; sizes and scores fall as the CRF rises."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def encode_sample(
        self,
        tools: Tools,
        settings: Settings,
        sample: FileRecord,
        crf: int,
        workdir: Path,
        reporter: Reporter,
    ) -> SampleEncode:
        self.calls.append(("encode", sample.path, crf))
        encoded = workdir / f"crf{crf}.mkv"
        encoded.parent.mkdir(parents=True, exist_ok=True)
        encoded.write_bytes(b"e" * 8)
        return SampleEncode(encoded, 1000, 1000 - 25 * crf, 600.0, 20.0)

    def score_windows(
        self,
        tools: Tools,
        settings: Settings,
        sample: FileRecord,
        encoded: SampleEncode,
        workdir: Path,
        reporter: Reporter,
    ) -> Sequence[float]:
        self.calls.append(("score", sample.path, encoded.encoded.name))
        crf = int(encoded.encoded.stem[3:])
        return [110.0 - crf, 108.0 - crf, 106.0 - crf]

    def make_sidebyside(
        self,
        tools: Tools,
        sample: FileRecord,
        encoded: SampleEncode,
        dst: Path,
        reporter: Reporter,
    ) -> None:
        self.calls.append(("sidebyside", sample.path, dst))
        dst.write_bytes(b"s")

    def engine(self) -> CalibrationEngine:
        return CalibrationEngine(
            encode_sample=self.encode_sample,
            score_windows=self.score_windows,
            make_sidebyside=self.make_sidebyside,
        )


def _settings(tmp_path: Path) -> Settings:
    settings = tmp_settings(tmp_path)
    settings.library_root.mkdir(parents=True)
    return settings


def _library_snapshot(root: Path) -> list[tuple[str, int]]:
    return sorted(
        (str(Path(d, n).relative_to(root)), os.stat(Path(d, n)).st_mtime_ns)
        for d, dirs, names in os.walk(root)
        for n in dirs + names
    )


def _populate(settings: Settings, state: State) -> None:
    for name, size in (("Small.avi", 1000), ("Mid.avi", 5000), ("Big.avi", 9000)):
        path = settings.library_root / name
        path.write_bytes(b"x")
        _add(state, path, "encode", ("h264",), size=size)
    _add(state, settings.library_root / "Hevc.mkv", "leave_alone", ("already hevc",), size=7000)


def test_calibrate_table_has_size_and_vmaf_per_crf(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    fake = FakeEngine()
    with State.open(settings.state_db) as state:
        _populate(settings, state)
        before = _library_snapshot(settings.library_root)
        events = state.events(1000)
        rows = run_calibration(
            settings, TOOLS, state, NullReporter(), samples=2, engine=fake.engine()
        )
        assert state.events(1000) == events  # the sweep stores no event; approval is separate
        assert state.calibration() is None
        assert state.calibration_swept() is not None  # what `calibrate --approve` requires
    assert CALIBRATION_CRFS == (17, 20, 23, 26)
    assert [(r.sample.name, r.crf) for r in rows] == [
        (name, crf) for name in ("Small.avi", "Big.avi") for crf in CALIBRATION_CRFS
    ]
    row = rows[0]
    assert row.size_pct == pytest.approx(100.0 * (1000 - 25 * 17) / 1000)
    assert row.vmaf_mean == pytest.approx(108.0 - 17)
    assert row.vmaf_worst == 106.0 - 17

    table = (settings.state_db.parent / "calibration.txt").read_text(encoding="utf-8")
    for crf in CALIBRATION_CRFS:
        assert f"{crf:>3}" in table
    assert "57.5" in table  # size % at CRF 17
    assert "91.00" in table and "89.00" in table  # mean and worst VMAF at CRF 17
    (run_folder,) = (settings.completed_dir / "calibration").iterdir()  # one folder per sweep
    clips = sorted(p.name for p in run_folder.iterdir())
    assert len(clips) == 8
    assert all(c.endswith(".mp4") for c in clips)
    assert _library_snapshot(settings.library_root) == before
    ours = settings.scratch_dir / scratch_mod.TOOL_FOLDER / "calibration"
    assert not list(ours.glob("**/*.mkv"))  # work cleaned


def test_default_engine_is_held_and_touches_nothing(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    engine = calibrate_mod.default_engine()
    assert engine.held_stages() == ["encode_sample", "score_windows", "make_sidebyside"]
    with State.open(settings.state_db) as state:
        _populate(settings, state)
        with pytest.raises(NotImplementedError, match="held: Task 10b"):
            run_calibration(settings, TOOLS, state, NullReporter(), samples=2, engine=engine)
    assert not settings.scratch_dir.exists()
    assert not settings.completed_dir.exists()
    assert not (settings.state_db.parent / "calibration.txt").exists()


def test_held_default_stages_raise() -> None:
    engine = CalibrationEngine()
    with pytest.raises(NotImplementedError, match="held: Task 10b"):
        engine.encode_sample(TOOLS, None, None, 17, Path("w"), NullReporter())  # type: ignore[arg-type]


def test_no_encode_candidates_is_an_error(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with State.open(settings.state_db) as state:
        _add(state, settings.library_root / "a.mkv", "leave_alone", ("already hevc",))
        with pytest.raises(CalibrationError, match="survey"):
            run_calibration(
                settings, TOOLS, state, NullReporter(), samples=2, engine=FakeEngine().engine()
            )


def test_empty_scores_are_an_error(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    fake = FakeEngine()
    engine = fake.engine()
    engine.score_windows = lambda *a: []
    with State.open(settings.state_db) as state:
        _populate(settings, state)
        with pytest.raises(CalibrationError, match="VMAF"):
            run_calibration(settings, TOOLS, state, NullReporter(), samples=1, engine=engine)


# --- CLI -----------------------------------------------------------------------------------


def _invoke(*args: str) -> Any:
    return CliRunner().invoke(app, list(args))


def test_cli_calibrate_held_exits_1_and_touches_nothing(tmp_path: Path) -> None:
    cfg = write_config(tmp_path)
    result = _invoke("calibrate", "--samples", "3", "--config", str(cfg))
    assert result.exit_code == 1
    assert "held: Task 10b: calibration needs real encodes" in result.output
    assert "once the hold is lifted" in result.output
    assert not (tmp_path / "appdata").exists()
    assert not (tmp_path / "completed").exists()
    assert not (tmp_path / "scratch").exists()


def test_cli_calibrate_runs_with_an_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = write_config(tmp_path)
    settings = tmp_settings(tmp_path)
    with State.open(settings.state_db) as state:
        _populate(settings, state)
    monkeypatch.setattr(calibrate_mod, "default_engine", lambda: FakeEngine().engine())
    monkeypatch.setattr("video_cleaner.cli.locate_tools", lambda: TOOLS)
    result = _invoke("calibrate", "--samples", "2", "--config", str(cfg), "-q")
    assert result.exit_code == 0, result.output
    assert "Small.avi" in result.output and "Big.avi" in result.output
    assert str(settings.state_db.parent / "calibration.txt") in result.output
    assert "calibrate --approve" in result.output


def _swept(tmp_path: Path) -> None:
    """A state database in which a calibration sweep has run."""
    with State.open(tmp_settings(tmp_path).state_db) as state:
        state.record_calibration_sweep()


def test_cli_calibrate_approve_stores_values(tmp_path: Path) -> None:
    cfg = write_config(tmp_path)
    _swept(tmp_path)
    result = _invoke(
        "calibrate", "--approve", "--vmaf-target", "94.5", "--crf-max", "23", "--config", str(cfg)
    )
    assert result.exit_code == 0, result.output
    assert "94.5" in result.output and "23" in result.output
    with State.open(tmp_path / "appdata" / "state.db") as state:
        assert state.calibration() == (94.5, 23)


@pytest.mark.parametrize(
    ("vmaf", "crf"),
    [("49.9", "23"), ("100.1", "23"), ("nan", "23"), ("95", "15"), ("95", "52")],
)
def test_cli_calibrate_approve_validates(tmp_path: Path, vmaf: str, crf: str) -> None:
    cfg = write_config(tmp_path)
    result = _invoke(
        "calibrate", "--approve", "--vmaf-target", vmaf, "--crf-max", crf, "--config", str(cfg)
    )
    assert result.exit_code == 2
    assert not (tmp_path / "appdata" / "state.db").exists()


# The lower bound is the guard's mean floor (default 93), not 50, while the guard is above 50.
@pytest.mark.parametrize("vmaf", ["93", "100"])
def test_cli_calibrate_approve_accepts_bounds(tmp_path: Path, vmaf: str) -> None:
    cfg = write_config(tmp_path)
    _swept(tmp_path)
    for crf in ("16", "51"):
        result = _invoke(
            "calibrate", "--approve", "--vmaf-target", vmaf, "--crf-max", crf, "--config", str(cfg)
        )
        assert result.exit_code == 0, result.output


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--approve"],
        ["--approve", "--vmaf-target", "95"],
        ["--approve", "--crf-max", "23"],
        ["--samples", "2", "--approve", "--vmaf-target", "95", "--crf-max", "23"],
        ["--samples", "0"],
        ["--vmaf-target", "95", "--crf-max", "23"],
    ],
)
def test_cli_calibrate_bad_combinations_exit_2(tmp_path: Path, args: list[str]) -> None:
    cfg = write_config(tmp_path)
    result = _invoke("calibrate", *args, "--config", str(cfg))
    assert result.exit_code == 2, result.output
    assert not (tmp_path / "appdata").exists()


def test_cli_calibrate_refuses_outputs_in_library(tmp_path: Path) -> None:
    library = tmp_path / "library"
    cfg = write_config(tmp_path, completed_dir=library / "done")
    for args in (["--samples", "2"], ["--approve", "--vmaf-target", "95", "--crf-max", "23"]):
        result = _invoke("calibrate", *args, "--config", str(cfg))
        assert result.exit_code == 2
        assert "library root" in result.output
    assert os.listdir(library) == []
    assert not (tmp_path / "appdata").exists()


@pytest.mark.parametrize("height", [0, 719])
def test_build_sidebyside_cmd_rejects_odd_height(height: int) -> None:
    with pytest.raises(ValueError, match="height"):
        build_sidebyside_cmd(TOOLS, Path("a"), Path("b"), Path("c"), 0.0, 1.0, height=height)


def test_all_held_stubs_raise() -> None:
    engine = CalibrationEngine()
    rep = NullReporter()
    sample = _rec("a.avi", 1)
    enc = SampleEncode(Path("e"), 1, 1, 0.0, 1.0)
    with pytest.raises(NotImplementedError, match="held: Task 10b"):
        engine.score_windows(TOOLS, tmp_settings(Path("t")), sample, enc, Path("w"), rep)
    with pytest.raises(NotImplementedError, match="held: Task 10b"):
        engine.make_sidebyside(TOOLS, sample, enc, Path("d"), rep)


def test_stage_marked_held_counts_as_held() -> None:
    fake = FakeEngine()
    engine = fake.engine()

    def scores(*args: Any) -> Sequence[float]:
        return []

    scores.held = True  # type: ignore[attr-defined]
    engine.score_windows = scores
    assert engine.held_stages() == ["score_windows"]


def test_cli_calibrate_errors_exit_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = write_config(tmp_path)
    settings = tmp_settings(tmp_path)
    with State.open(settings.state_db):
        pass  # a state without encode candidates
    monkeypatch.setattr(calibrate_mod, "default_engine", lambda: FakeEngine().engine())
    monkeypatch.setattr("video_cleaner.cli.locate_tools", lambda: TOOLS)
    result = _invoke("calibrate", "--samples", "2", "--config", str(cfg), "-q")
    assert result.exit_code == 2
    assert "survey" in result.output


def test_cli_calibrate_without_state_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = write_config(tmp_path)
    monkeypatch.setattr(calibrate_mod, "default_engine", lambda: FakeEngine().engine())
    result = _invoke("calibrate", "--samples", "2", "--config", str(cfg), "-q")
    assert result.exit_code == 2
    assert "survey" in result.output
    assert not (tmp_path / "appdata").exists()


def test_unwritable_table_is_a_calibration_error(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with State.open(settings.state_db) as state:
        _populate(settings, state)
    (settings.state_db.parent / "calibration.txt").mkdir()  # a folder in the way
    with (
        State.open(settings.state_db) as state,
        pytest.raises(CalibrationError, match="cannot write"),
    ):
        run_calibration(
            settings, TOOLS, state, NullReporter(), samples=1, engine=FakeEngine().engine()
        )


def test_cli_calibrate_approve_refuses_target_below_guard_floor(tmp_path: Path) -> None:
    cfg = write_config(tmp_path)  # guard_mean_floor is the default 93.0
    result = _invoke(
        "calibrate", "--approve", "--vmaf-target", "92.5", "--crf-max", "23", "--config", str(cfg)
    )
    assert result.exit_code == 2
    assert "guard_mean_floor" in result.output
    assert not (tmp_path / "appdata" / "state.db").exists()


def test_calibration_is_refused_while_a_run_holds_the_lock(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    fake = FakeEngine()
    with State.open(settings.state_db) as state:
        _populate(settings, state)
        with instance_lock(lock_path_for(settings.state_db)), pytest.raises(AlreadyRunning):
            run_calibration(settings, TOOLS, state, NullReporter(), samples=1, engine=fake.engine())
    assert fake.calls == []
    assert not settings.scratch_dir.exists()


def test_cli_calibrate_already_running_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = write_config(tmp_path)
    settings = tmp_settings(tmp_path)
    with State.open(settings.state_db) as state:
        _populate(settings, state)
    monkeypatch.setattr(calibrate_mod, "default_engine", lambda: FakeEngine().engine())
    monkeypatch.setattr("video_cleaner.cli.locate_tools", lambda: TOOLS)
    with instance_lock(lock_path_for(settings.state_db)):
        result = _invoke("calibrate", "--samples", "1", "--config", str(cfg), "-q")
    assert result.exit_code == 2
    assert str(os.getpid()) in result.output


def test_calibration_never_removes_a_preexisting_scratch_folder(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    foreign = settings.scratch_dir / "calibration" / "01"
    foreign.mkdir(parents=True)
    (foreign / "mine.txt").write_bytes(b"not the tool's")
    with State.open(settings.state_db) as state:
        _populate(settings, state)
        run_calibration(
            settings, TOOLS, state, NullReporter(), samples=1, engine=FakeEngine().engine()
        )
    assert (foreign / "mine.txt").exists()


def test_cli_calibrate_approve_needs_a_state_database(tmp_path: Path) -> None:
    cfg = write_config(tmp_path)
    result = _invoke(
        "calibrate", "--approve", "--vmaf-target", "95", "--crf-max", "23", "--config", str(cfg)
    )
    assert result.exit_code == 2
    assert "survey" in result.output
    assert not (tmp_path / "appdata" / "state.db").exists()  # never creates an empty one


def test_cli_calibrate_approve_needs_a_recorded_sweep(tmp_path: Path) -> None:
    cfg = write_config(tmp_path)
    with State.open(tmp_settings(tmp_path).state_db):
        pass  # surveyed, never calibrated
    result = _invoke(
        "calibrate", "--approve", "--vmaf-target", "95", "--crf-max", "23", "--config", str(cfg)
    )
    assert result.exit_code == 2
    assert "calibrate --samples" in result.output
    with State.open(tmp_settings(tmp_path).state_db) as state:
        assert state.calibration() is None


def test_state_records_the_calibration_sweep(tmp_path: Path) -> None:
    with State.open(tmp_path / "s.db") as state:
        assert state.calibration_swept() is None
        state.record_calibration_sweep()
        swept = state.calibration_swept()
    assert swept is not None and swept[:2] == "20"  # an ISO timestamp


def test_each_sweep_gets_its_own_clips_folder(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    stamps = iter(["20261116-010000", "20261116-020000"])
    with State.open(settings.state_db) as state:
        _populate(settings, state)
        for _ in range(2):
            run_calibration(
                settings,
                TOOLS,
                state,
                NullReporter(),
                samples=1,
                engine=FakeEngine().engine(),
                stamp=lambda: next(stamps),
            )
    folders = sorted(p.name for p in (settings.completed_dir / "calibration").iterdir())
    assert folders == ["20261116-010000", "20261116-020000"]


def test_unwritable_clips_folder_is_a_calibration_error(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.completed_dir.mkdir(parents=True)
    (settings.completed_dir / "calibration").write_bytes(b"a file where the folder goes")
    with State.open(settings.state_db) as state:
        _populate(settings, state)
        with pytest.raises(CalibrationError, match="cannot create"):
            run_calibration(
                settings, TOOLS, state, NullReporter(), samples=1, engine=FakeEngine().engine()
            )
