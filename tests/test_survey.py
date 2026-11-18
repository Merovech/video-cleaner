"""Survey: walk the library read-only, probe, decide, record in State and write a CSV."""

from __future__ import annotations

import csv
import dataclasses
import os
import re
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from video_cleaner import survey as survey_module
from video_cleaner.cli import app
from video_cleaner.config import Settings
from video_cleaner.decide import Action
from video_cleaner.multipart import PartGroup
from video_cleaner.probe import AudioStream, MediaInfo, ProbeError, VideoStream
from video_cleaner.progress import Progress
from video_cleaner.state import State, Status
from video_cleaner.survey import (
    CSV_COLUMNS,
    EST_SAVING_FRACTION,
    SurveyError,
    SurveyResult,
    discover_files,
    run_survey,
)
from video_cleaner.tools import ToolNotFound, Tools, locate_tools

GIB = 1024**3
MKV = "matroska,webm"
FAKE_TOOLS = Tools(ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe"), mkvmerge=Path("mkvmerge"))
ODD_NAME = "The First Omen (2024).en.MX].mkv"
UNICODE_NAME = "[Group] Ünïcode 映画 (1999).mkv"


class RecordingReporter:
    def __init__(self) -> None:
        self.stages: list[str] = []
        self.updates: list[Progress] = []
        self._lock = threading.Lock()

    def stage(self, label: str) -> None:
        self.stages.append(label)

    def update(self, p: Progress) -> None:
        with self._lock:
            self.updates.append(p)

    def heartbeat(self, label: str, elapsed_s: float) -> None:
        pass

    def file_done(self, summary: str) -> None:
        pass


def _info(
    path: Path,
    *,
    codec: str = "h264",
    width: int = 1920,
    height: int = 1080,
    size: int = 6 * GIB,
    duration_s: float = 3600.0,
    has_video: bool = True,
) -> MediaInfo:
    video = (
        VideoStream(
            codec=codec,
            width=width,
            height=height,
            pix_fmt="yuv420p",
            fps=23.976,
            bit_rate=None,
            transfer="bt709",
            dv_profile=None,
            field_order="progressive",
        )
        if has_video
        else None
    )
    return MediaInfo(
        path=path,
        size=size,
        mtime=1_700_000_000.0,
        duration_s=duration_s,
        container=MKV,
        video=video,
        audio=(AudioStream(index=1, codec="ac3", language="eng", title=""),),
        subtitles=(),
        chapters=0,
        cover_art=0,
    )


ProbeFn = Callable[[Path, Tools, object], MediaInfo]


def _fake_probe(rules: dict[str, Callable[[Path], MediaInfo]]) -> ProbeFn:
    """probe_fn keyed on the file name; the default is an encodable 1080p h264 file."""

    def probe_fn(path: Path, tools: Tools, reporter: object) -> MediaInfo:
        rule = rules.get(path.name)
        return rule(path) if rule is not None else _info(path)

    return probe_fn


def _settings(tmp_path: Path, library: Path | None = None) -> Settings:
    return Settings(
        library_root=library if library is not None else tmp_path / "library",
        scratch_dir=tmp_path / "scratch",
        completed_dir=tmp_path / "completed",
        state_db=tmp_path / "appdata" / "state.db",
        log_dir=tmp_path / "appdata" / "logs",
    )


def _touch(path: Path, data: bytes = b"\x00" * 64) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _run(
    settings: Settings,
    probe_fn: ProbeFn,
    *,
    reporter: RecordingReporter | None = None,
    root: Path | None = None,
    workers: int = 4,
    hash_fn: Callable[[Path], str | None] | None = None,
) -> tuple[SurveyResult, State]:
    state = State.open(settings.state_db)
    extra = {} if hash_fn is None else {"hash_fn": hash_fn}
    result = run_survey(
        settings,
        FAKE_TOOLS,
        state,
        reporter if reporter is not None else RecordingReporter(),
        root=root,
        workers=workers,
        probe_fn=probe_fn,
        **extra,
    )
    return result, state


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        assert tuple(reader.fieldnames or ()) == CSV_COLUMNS
        return list(reader)


def _by_name(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {Path(r["path"]).name: r for r in rows}


# --- discovery -------------------------------------------------------------------------------


def test_discover_files_filters_extensions_case_insensitively_and_skips_part(
    tmp_path: Path,
) -> None:
    lib = tmp_path / "library"
    wanted = [
        _touch(lib / name)
        for name in ("a.mkv", "b.MP4", "c.avi", "d.M4V", "e.ts", "f.wmv", "g.MOV")
    ]
    wanted.append(_touch(lib / "deep" / "er" / ODD_NAME))
    _touch(lib / "notes.txt")
    _touch(lib / "poster.jpg")
    _touch(lib / "download.mkv.part")
    _touch(lib / "x.part")
    (lib / "folder.mkv").mkdir()  # a directory with a video extension is not a file

    assert discover_files(lib) == sorted(wanted)


def test_discover_files_of_missing_root_is_empty(tmp_path: Path) -> None:
    assert discover_files(tmp_path / "nope") == []


# --- core survey -----------------------------------------------------------------------------


def test_survey_counts_actions_and_writes_csv(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    lib = settings.library_root
    _touch(lib / "Encode Me (2001).mkv")
    _touch(lib / "Already (2010).MP4")
    _touch(lib / "sub" / "Audio Only.avi")
    _touch(lib / "sub" / ODD_NAME)
    _touch(lib / UNICODE_NAME)
    _touch(lib / "readme.txt")
    probe_fn = _fake_probe(
        {
            "Already (2010).MP4": lambda p: _info(p, codec="hevc", size=3 * GIB),
            "Audio Only.avi": lambda p: _info(p, has_video=False, size=GIB),
        }
    )

    result, state = _run(settings, probe_fn)

    assert result.counts == {
        Action.ENCODE: 3,
        Action.LEAVE_ALONE: 1,
        Action.REMUX: 0,
        Action.FLAG: 1,
    }
    assert result.total_bytes == 3 * 6 * GIB + 3 * GIB + GIB
    assert result.est_saving_bytes == 3 * int(6 * GIB * EST_SAVING_FRACTION)
    assert result.csv_path == settings.state_db.parent / "survey.csv"
    assert result.unprobeable == ()
    assert result.unrepresentable == ()
    assert result.groups == ()

    rows = _by_name(_csv_rows(result.csv_path))
    assert set(rows) == {
        "Encode Me (2001).mkv",
        "Already (2010).MP4",
        "Audio Only.avi",
        ODD_NAME,
        UNICODE_NAME,
    }
    enc = rows["Encode Me (2001).mkv"]
    assert enc["action"] == "encode"
    assert enc["size"] == str(6 * GIB)
    assert enc["codec"] == "h264"
    assert (enc["width"], enc["height"]) == ("1920", "1080")
    assert enc["container"] == MKV
    assert float(enc["duration_s"]) == pytest.approx(3600.0)
    assert float(enc["bitrate_kbps"]) == pytest.approx(6 * GIB * 8 / 3600 / 1000, rel=1e-3)
    assert enc["est_saving_bytes"] == str(int(6 * GIB * EST_SAVING_FRACTION))
    assert enc["parts"] == ""
    assert "h264 1080p" in enc["reasons"]
    assert rows["Already (2010).MP4"]["action"] == "leave_alone"
    assert rows["Already (2010).MP4"]["reasons"] == "already hevc"
    assert rows["Audio Only.avi"]["action"] == "flag"
    assert rows["Audio Only.avi"]["codec"] == ""
    assert rows["Audio Only.avi"]["reasons"] == "no video stream"
    assert rows[UNICODE_NAME]["action"] == "encode"

    records = {r.path.name: r for r in state.records()}
    assert set(records) == set(rows)
    assert records["Encode Me (2001).mkv"].status is Status.PENDING
    assert records["Already (2010).MP4"].status is Status.LEFT_ALONE
    assert records["Audio Only.avi"].status is Status.FLAGGED
    assert records[ODD_NAME].decision == "encode"
    state.close()


def test_survey_reasons_are_joined_with_semicolons(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _touch(settings.library_root / "uhd.mkv")
    # 4K HEVC: two no-encode reasons
    probe_fn = _fake_probe({"uhd.mkv": lambda p: _info(p, codec="hevc", width=3840, height=2160)})

    result, state = _run(settings, probe_fn)
    state.close()

    row = _csv_rows(result.csv_path)[0]
    assert row["reasons"] == "already hevc; 4K/UHD (3840x2160)"


def test_survey_uses_explicit_root_over_settings(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _touch(settings.library_root / "not-this.mkv")
    other = tmp_path / "other"
    _touch(other / "this.mkv")

    result, state = _run(settings, _fake_probe({}), root=other)
    state.close()

    assert [Path(r["path"]).name for r in _csv_rows(result.csv_path)] == ["this.mkv"]


def test_survey_of_empty_library_writes_header_only_csv(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.library_root.mkdir()

    result, state = _run(settings, _fake_probe({}))
    state.close()

    assert result.counts == dict.fromkeys(Action, 0)
    assert result.total_bytes == 0
    assert _csv_rows(result.csv_path) == []


def test_survey_is_read_only(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    lib = settings.library_root
    files = [
        _touch(lib / "a.mkv", b"alpha" * 100),
        _touch(lib / "Movie pt1.mkv", b"part one"),
        _touch(lib / "Movie pt2.mkv", b"part two"),
        _touch(lib / "x" / ODD_NAME, b"odd"),
        _touch(lib / "x" / "garbage.mkv", b"not a video"),
        _touch(lib / "x" / "notes.txt", b"text"),
    ]

    def snapshot() -> dict[Path, tuple[bytes, int]]:
        return {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in lib.rglob("*") if p.is_file()}

    dirs_before = {p: p.stat().st_mtime_ns for p in [lib, *lib.rglob("*")] if p.is_dir()}
    before = snapshot()
    assert set(before) == set(files)

    def boom(path: Path) -> MediaInfo:
        raise ProbeError(path, "Invalid data found when processing input")

    result, state = _run(settings, _fake_probe({"garbage.mkv": boom}))
    state.close()

    assert snapshot() == before
    assert {p: p.stat().st_mtime_ns for p in [lib, *lib.rglob("*")] if p.is_dir()} == dirs_before
    assert not result.csv_path.is_relative_to(lib)


def test_survey_refuses_output_under_the_library_root(tmp_path: Path) -> None:
    lib = tmp_path / "library"
    _touch(lib / "a.mkv")
    settings = Settings(
        library_root=lib,
        scratch_dir=tmp_path / "scratch",
        completed_dir=tmp_path / "completed",
        state_db=lib / "state" / "state.db",
        log_dir=tmp_path / "logs",
    )
    probe_calls: list[Path] = []

    def probe_fn(path: Path, tools: Tools, reporter: object) -> MediaInfo:
        probe_calls.append(path)
        return _info(path)

    state = State.open(tmp_path / "elsewhere.db")
    with pytest.raises(SurveyError, match="library root"):
        run_survey(settings, FAKE_TOOLS, state, RecordingReporter(), probe_fn=probe_fn)
    state.close()
    assert probe_calls == []
    assert not (lib / "state" / "survey.csv").exists()


def test_check_outputs_outside_covers_logs_and_configured_root(tmp_path: Path) -> None:
    lib = tmp_path / "library"
    logs_inside = Settings(
        library_root=lib,
        scratch_dir=tmp_path / "s",
        completed_dir=tmp_path / "c",
        state_db=tmp_path / "state.db",
        log_dir=lib / "logs",
    )
    with pytest.raises(SurveyError, match="log folder"):
        survey_module.check_outputs_outside(logs_inside, lib)
    # --root on a sub-folder does not excuse writing elsewhere under library_root
    with pytest.raises(SurveyError, match="log folder"):
        survey_module.check_outputs_outside(logs_inside, tmp_path / "subset")
    fine = _settings(tmp_path)
    assert survey_module.check_outputs_outside(fine, fine.library_root) == (
        tmp_path / "appdata" / "survey.csv"
    )


def test_survey_flags_unprobeable_file_and_continues(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    lib = settings.library_root
    _touch(lib / "good.mkv")
    garbage = _touch(lib / "garbage.mkv", b"this is not a video at all")
    weird = _touch(lib / "weird.mp4", b"x")

    def probe_error(path: Path) -> MediaInfo:
        raise ProbeError(path, "ffprobe exited with code 1: Invalid data")

    def crash(path: Path) -> MediaInfo:
        raise RuntimeError("something unexpected")

    result, state = _run(settings, _fake_probe({"garbage.mkv": probe_error, "weird.mp4": crash}))

    assert dict(result.unprobeable) == {
        garbage: "ffprobe exited with code 1: Invalid data",
        weird: "RuntimeError: something unexpected",
    }
    assert result.counts[Action.ENCODE] == 1
    assert result.counts[Action.FLAG] == 2
    assert result.total_bytes == 6 * GIB + garbage.stat().st_size + weird.stat().st_size

    rows = _by_name(_csv_rows(result.csv_path))
    assert rows["garbage.mkv"]["action"] == "flag"
    assert rows["garbage.mkv"]["reasons"] == "unprobeable: ffprobe exited with code 1: Invalid data"
    assert rows["garbage.mkv"]["size"] == str(garbage.stat().st_size)
    assert rows["garbage.mkv"]["codec"] == ""
    assert rows["garbage.mkv"]["est_saving_bytes"] == "0"
    assert rows["weird.mp4"]["reasons"] == "unprobeable: RuntimeError: something unexpected"
    assert rows["good.mkv"]["action"] == "encode"

    # State needs a MediaInfo, so unprobeable files are only in the CSV and the result.
    assert [r.path.name for r in state.records()] == ["good.mkv"]
    state.close()


def test_survey_progress_shows_n_of_m(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    for name in ("a.mkv", "b.mkv", "c.mkv"):
        _touch(settings.library_root / name)
    reporter = RecordingReporter()

    result, state = _run(settings, _fake_probe({}), reporter=reporter)
    state.close()

    assert reporter.stages[0] == "survey"
    assert [u.note for u in reporter.updates] == ["1 of 3 files", "2 of 3 files", "3 of 3 files"]
    percents = [u.percent for u in reporter.updates]
    assert percents == pytest.approx([100 / 3, 200 / 3, 100.0])
    assert sum(result.counts.values()) == 3


def test_survey_runs_probes_on_several_workers(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    for i in range(4):
        _touch(settings.library_root / f"f{i}.mkv")
    barrier = threading.Barrier(2, timeout=5)

    def probe_fn(path: Path, tools: Tools, reporter: object) -> MediaInfo:
        if path.name in ("f0.mkv", "f1.mkv"):
            barrier.wait()  # deadlocks (BrokenBarrierError) unless two probes run at once
        return _info(path)

    result, state = _run(settings, probe_fn, workers=2)
    state.close()
    assert result.unprobeable == ()
    assert result.counts[Action.ENCODE] == 4


def test_survey_rejects_zero_workers(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    state = State.open(settings.state_db)
    with pytest.raises(ValueError, match="workers"):
        run_survey(settings, FAKE_TOOLS, state, RecordingReporter(), workers=0)
    state.close()


def test_survey_csv_is_replaced_atomically_on_rerun(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _touch(settings.library_root / "a.mkv")
    first, state = _run(settings, _fake_probe({}))
    state.close()
    _touch(settings.library_root / "b.mkv")

    second, state = _run(settings, _fake_probe({}))
    state.close()

    assert first.csv_path == second.csv_path
    assert sorted(Path(r["path"]).name for r in _csv_rows(second.csv_path)) == ["a.mkv", "b.mkv"]
    assert sorted(p.name for p in second.csv_path.parent.iterdir() if "survey" in p.name) == [
        "survey.csv"
    ]


# --- multi-part ------------------------------------------------------------------------------


def test_survey_compatible_multipart_group_is_one_record(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    lib = settings.library_root
    pt1 = _touch(lib / "Mabuse" / "Dr. Mabuse pt1.mkv")
    pt2 = _touch(lib / "Mabuse" / "Dr. Mabuse pt2.mkv")
    probe_fn = _fake_probe(
        {
            "Dr. Mabuse pt1.mkv": lambda p: _info(p, size=4 * GIB),
            "Dr. Mabuse pt2.mkv": lambda p: _info(p, size=3 * GIB),
        }
    )

    result, state = _run(settings, probe_fn)

    assert result.groups == (PartGroup(output_name="Dr. Mabuse.mkv", parts=(pt1, pt2)),)
    first = state.get(pt1)
    assert first is not None
    assert first.decision == "encode"
    assert first.parts == (Path(pt1.as_posix()), Path(pt2.as_posix()))
    assert "multi-part file to join" in first.reasons
    assert first.est_saving_bytes == int(7 * GIB * EST_SAVING_FRACTION)
    second = state.get(pt2)
    assert second is not None
    assert second.decision == "leave_alone"
    assert second.reasons == ("part of group Dr. Mabuse.mkv",)
    assert second.est_saving_bytes == 0
    state.close()

    assert result.counts[Action.ENCODE] == 1
    assert result.counts[Action.LEAVE_ALONE] == 1
    assert result.est_saving_bytes == int(7 * GIB * EST_SAVING_FRACTION)
    rows = _by_name(_csv_rows(result.csv_path))
    assert rows["Dr. Mabuse pt1.mkv"]["parts"] == f"{pt1}; {pt2}"
    assert rows["Dr. Mabuse pt1.mkv"]["action"] == "encode"
    assert rows["Dr. Mabuse pt2.mkv"]["action"] == "leave_alone"
    assert rows["Dr. Mabuse pt2.mkv"]["reasons"] == "part of group Dr. Mabuse.mkv"


def test_survey_csv_has_hdr_dv_interlace_and_subtitle_columns(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    lib = settings.library_root
    _touch(lib / "Plain.mkv")
    _touch(lib / "Dv.mkv")

    def dv5(path: Path) -> MediaInfo:
        base = _info(path, codec="hevc", width=3840, height=2160)
        assert base.video is not None
        video = dataclasses.replace(base.video, dv_profile=5, transfer=None)
        return dataclasses.replace(base, video=video)

    result, state = _run(settings, _fake_probe({"Dv.mkv": dv5}))
    state.close()
    rows = _by_name(_csv_rows(result.csv_path))
    plain, dv = rows["Plain.mkv"], rows["Dv.mkv"]
    assert (plain["hdr"], plain["dv_profile"], plain["interlaced"]) == ("False", "", "False")
    assert plain["subtitle_streams"] == "0"
    assert (dv["hdr"], dv["dv_profile"]) == ("False", "5")
    assert "Dolby Vision profile 5: replace (no HDR10 fallback)" in dv["reasons"]


def _name_hash(path: Path) -> str:
    return f"{sum(path.name.encode()):016x}"


def test_survey_stores_movie_hash_per_file_and_per_part(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    lib = settings.library_root
    single = _touch(lib / "Heat (1995).mkv")
    pt1 = _touch(lib / "Mabuse" / "Dr. Mabuse pt1.mkv")
    pt2 = _touch(lib / "Mabuse" / "Dr. Mabuse pt2.mkv")
    result, state = _run(settings, _fake_probe({}), hash_fn=_name_hash)
    try:
        for path in (single, pt1, pt2):
            rec = state.get(path)
            assert rec is not None and rec.movie_hash == _name_hash(path)
    finally:
        state.close()
    rows = _by_name(_csv_rows(result.csv_path))
    assert rows["Heat (1995).mkv"]["movie_hash"] == _name_hash(single)


def test_survey_default_hash_reads_the_real_file(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    film = _touch(settings.library_root / "Big.mkv", b"\x00" * (128 * 1024))
    _result, state = _run(settings, _fake_probe({}))
    try:
        rec = state.get(film)
        assert rec is not None and rec.movie_hash == "0000000000020000"
    finally:
        state.close()


def test_survey_hash_failure_never_stops_the_survey(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    lib = settings.library_root
    bad = _touch(lib / "Bad.mkv")
    good = _touch(lib / "Good.mkv")

    def hash_fn(path: Path) -> str | None:
        if path.name == "Bad.mkv":
            raise RuntimeError("boom")
        return "abcdefabcdefabcd"

    result, state = _run(settings, _fake_probe({}), hash_fn=hash_fn)
    try:
        bad_rec, good_rec = state.get(bad), state.get(good)
        assert bad_rec is not None and bad_rec.movie_hash is None
        assert bad_rec.decision == "encode"  # the file is still surveyed normally
        assert good_rec is not None and good_rec.movie_hash == "abcdefabcdefabcd"
    finally:
        state.close()
    assert result.unprobeable == ()


def test_survey_compatible_hevc_multipart_group_is_remuxed(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _touch(settings.library_root / "Film cd1.mkv")
    _touch(settings.library_root / "Film cd2.mkv")
    probe_fn = _fake_probe(
        {
            "Film cd1.mkv": lambda p: _info(p, codec="hevc"),
            "Film cd2.mkv": lambda p: _info(p, codec="hevc"),
        }
    )

    result, state = _run(settings, probe_fn)
    state.close()

    assert result.counts[Action.REMUX] == 1
    assert result.est_saving_bytes == 0


def test_survey_incompatible_multipart_parts_are_flagged(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    pt1 = _touch(settings.library_root / "Movie pt1.mkv")
    pt2 = _touch(settings.library_root / "Movie pt2.mkv")
    probe_fn = _fake_probe({"Movie pt2.mkv": lambda p: _info(p, width=1280, height=720)})

    result, state = _run(settings, probe_fn)

    reason = ("multi-part parts have different stream layouts",)
    for part in (pt1, pt2):
        record = state.get(part)
        assert record is not None
        assert record.decision == "flag"
        assert record.status is Status.FLAGGED
        assert record.reasons == reason
        assert record.parts == ()
    state.close()
    assert result.counts[Action.FLAG] == 2
    assert result.est_saving_bytes == 0
    assert len(result.groups) == 1


def test_survey_multipart_with_unprobeable_part_flags_the_rest(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    pt1 = _touch(settings.library_root / "Movie pt1.mkv")
    pt2 = _touch(settings.library_root / "Movie pt2.mkv")

    def broken(path: Path) -> MediaInfo:
        raise ProbeError(path, "truncated")

    result, state = _run(settings, _fake_probe({"Movie pt2.mkv": broken}))

    record = state.get(pt1)
    assert record is not None
    assert record.decision == "flag"
    assert record.reasons == ("multi-part group Movie.mkv has unprobeable parts",)
    assert state.get(pt2) is None
    state.close()
    assert dict(result.unprobeable) == {pt2: "truncated"}
    assert result.counts[Action.FLAG] == 2


# --- unrepresentable paths -------------------------------------------------------------------


def test_survey_records_unrepresentable_path_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    lib = settings.library_root
    good = _touch(lib / "good.mkv")
    # A lone surrogate (an undecodable byte in a NAS file name) may not be creatable on
    # Windows, so the walk is replaced by a path list containing one.
    bad = lib / "bad \udcff name.mkv"
    monkeypatch.setattr(survey_module, "discover_files", lambda root: [bad, good])

    result, state = _run(settings, _fake_probe({}))

    assert result.unrepresentable == (bad,)
    assert result.counts[Action.ENCODE] == 1
    assert result.counts[Action.FLAG] == 1
    assert result.est_saving_bytes == int(6 * GIB * EST_SAVING_FRACTION)
    assert [r.path.name for r in state.records()] == ["good.mkv"]
    state.close()

    rows = _csv_rows(result.csv_path)
    flagged = [r for r in rows if r["action"] == "flag"]
    assert len(flagged) == 1
    assert "\\udcff" in flagged[0]["path"]
    assert flagged[0]["reasons"].startswith("path cannot be stored in the state database")
    assert flagged[0]["est_saving_bytes"] == "0"


# --- robustness: locked CSV, unwritable output, interrupts, missing tools --------------------


def test_survey_falls_back_to_timestamped_csv_when_target_is_locked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    _touch(settings.library_root / "a.mkv")
    canonical = settings.state_db.parent / "survey.csv"
    real_replace = os.replace

    def locked_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        if Path(dst) == canonical:
            raise PermissionError(13, "The process cannot access the file", str(dst))
        real_replace(src, dst)

    monkeypatch.setattr("video_cleaner.survey.os.replace", locked_replace)

    result, state = _run(settings, _fake_probe({}))
    state.close()

    assert result.csv_path != canonical
    assert result.csv_path.parent == canonical.parent
    assert re.fullmatch(r"survey-\d{8}-\d{6}\.csv", result.csv_path.name)
    assert [Path(r["path"]).name for r in _csv_rows(result.csv_path)] == ["a.mkv"]
    assert not canonical.exists()
    assert [p.name for p in canonical.parent.iterdir() if p.suffix == ".tmp"] == []
    assert result.counts[Action.ENCODE] == 1


def test_survey_unwritable_output_directory_fails_before_probing(tmp_path: Path) -> None:
    lib = tmp_path / "library"
    _touch(lib / "a.mkv")
    blocker = _touch(tmp_path / "not-a-dir")  # the CSV folder is a regular file
    settings = Settings(
        library_root=lib,
        scratch_dir=tmp_path / "s",
        completed_dir=tmp_path / "c",
        state_db=blocker / "state.db",
        log_dir=tmp_path / "logs",
    )
    calls: list[Path] = []

    def probe_fn(path: Path, tools: Tools, reporter: object) -> MediaInfo:
        calls.append(path)
        return _info(path)

    state = State.open(tmp_path / "elsewhere.db")
    with pytest.raises(SurveyError, match="cannot write the survey CSV"):
        run_survey(settings, FAKE_TOOLS, state, RecordingReporter(), probe_fn=probe_fn)
    state.close()
    assert calls == []


def test_survey_csv_write_failure_is_a_survey_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    _touch(settings.library_root / "a.mkv")

    def no_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("video_cleaner.survey.os.replace", no_replace)
    state = State.open(settings.state_db)
    with pytest.raises(SurveyError, match="cannot write the survey CSV"):
        run_survey(settings, FAKE_TOOLS, state, RecordingReporter(), probe_fn=_fake_probe({}))
    state.close()
    assert [p.name for p in settings.state_db.parent.iterdir() if "survey" in p.name] == []


def test_survey_interrupt_cancels_queued_probes(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    for i in range(10):
        _touch(settings.library_root / f"f{i}.mkv")
    release = threading.Event()
    started: list[str] = []
    lock = threading.Lock()

    def probe_fn(path: Path, tools: Tools, reporter: object) -> MediaInfo:
        with lock:
            started.append(path.name)
        if path.name == "f0.mkv":
            raise KeyboardInterrupt
        release.wait(timeout=1)
        return _info(path)

    state = State.open(settings.state_db)
    try:
        with pytest.raises(KeyboardInterrupt):
            run_survey(
                settings, FAKE_TOOLS, state, RecordingReporter(), workers=1, probe_fn=probe_fn
            )
    finally:
        release.set()
    time.sleep(0.3)  # any probe still queued would start now
    state.close()
    assert started[0] == "f0.mkv"
    assert len(started) <= 2  # at most the one the worker had already picked up


def test_survey_tool_not_found_aborts(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    for name in ("a.mkv", "b.mkv"):
        _touch(settings.library_root / name)

    def probe_fn(path: Path, tools: Tools, reporter: object) -> MediaInfo:
        raise ToolNotFound("ffprobe", "gone")

    state = State.open(settings.state_db)
    with pytest.raises(ToolNotFound):
        run_survey(settings, FAKE_TOOLS, state, RecordingReporter(), probe_fn=probe_fn)
    assert state.records() == []
    state.close()
    assert not (settings.state_db.parent / "survey.csv").exists()


# --- CLI ------------------------------------------------------------------------------------


def _cli_with_failing_survey(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exc: BaseException
) -> tuple[int, str]:
    lib = tmp_path / "library"
    lib.mkdir()
    cfg = _write_config(tmp_path, lib)
    monkeypatch.setattr("video_cleaner.cli.locate_tools", lambda: FAKE_TOOLS)

    def failing(*args: object, **kwargs: object) -> SurveyResult:
        raise exc

    monkeypatch.setattr("video_cleaner.survey.run_survey", failing)
    result = CliRunner().invoke(app, ["survey", "--config", str(cfg), "-q"])
    return result.exit_code, result.output


def test_cli_survey_ctrl_c_exits_130(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code, output = _cli_with_failing_survey(tmp_path, monkeypatch, KeyboardInterrupt())
    assert code == 130
    assert "interrupted" in output
    assert "stay recorded" not in output  # nothing is recorded while files are being probed
    assert "nothing from this survey was recorded" in " ".join(output.split())
    assert "Traceback" not in output


def test_cli_survey_csv_error_exits_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code, output = _cli_with_failing_survey(
        tmp_path, monkeypatch, SurveyError("cannot write the survey CSV in X: denied")
    )
    assert code == 2
    assert "cannot write the survey CSV" in output


def test_cli_survey_unwritable_csv_dir_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lib = tmp_path / "library"
    lib.mkdir()
    cfg = _write_config(tmp_path, lib)
    monkeypatch.setattr("video_cleaner.cli.locate_tools", lambda: FAKE_TOOLS)

    def denied(*args: object, **kwargs: object) -> tuple[int, str]:
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr("video_cleaner.survey.tempfile.mkstemp", denied)
    result = CliRunner().invoke(app, ["survey", "--config", str(cfg), "-q"])

    assert result.exit_code == 2
    assert "cannot write the survey CSV" in result.output


def test_cli_survey_tool_lost_mid_run_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, output = _cli_with_failing_survey(tmp_path, monkeypatch, ToolNotFound("ffprobe", "gone"))
    assert code == 2
    assert "tool not found: ffprobe" in output


def test_cli_survey_help_shows_config_default_path() -> None:
    result = CliRunner().invoke(app, ["survey", "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert r"%LOCALAPPDATA%\video-cleaner\config.toml" in result.stdout
    assert "\v" not in result.stdout  # the old help string turned "\v" into a vertical tab


def _write_config(tmp_path: Path, library: Path) -> Path:
    cfg = tmp_path / "config.toml"
    values = {
        "library_root": library,
        "scratch_dir": tmp_path / "scratch",
        "completed_dir": tmp_path / "completed",
        "state_db": tmp_path / "appdata" / "state.db",
        "log_dir": tmp_path / "appdata" / "logs",
    }
    cfg.write_text(
        "".join(f"{k} = '{v.as_posix()}'\n" for k, v in values.items()), encoding="utf-8"
    )
    return cfg


def test_cli_survey_of_empty_library_prints_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lib = tmp_path / "library"
    lib.mkdir()
    cfg = _write_config(tmp_path, lib)
    monkeypatch.setattr("video_cleaner.cli.locate_tools", lambda: FAKE_TOOLS)

    result = CliRunner().invoke(app, ["survey", "--config", str(cfg), "-q"])

    assert result.exit_code == 0, result.output
    assert "0 files" in result.stdout
    assert "survey.csv" in result.stdout
    assert (tmp_path / "appdata" / "survey.csv").is_file()


def test_cli_survey_prints_counts_groups_and_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lib = tmp_path / "library"
    lib.mkdir()
    cfg = _write_config(tmp_path, lib)
    other_root = tmp_path / "subset"
    monkeypatch.setattr("video_cleaner.cli.locate_tools", lambda: FAKE_TOOLS)
    seen: dict[str, object] = {}
    pt1, pt2 = lib / "Movie pt1.mkv", lib / "Movie pt2.mkv"

    def fake_run_survey(
        settings: Settings, tools: Tools, state: State, reporter: object, *, root: Path | None
    ) -> SurveyResult:
        seen["root"] = root
        seen["tools"] = tools
        return SurveyResult(
            counts={Action.ENCODE: 7, Action.REMUX: 1, Action.LEAVE_ALONE: 3, Action.FLAG: 2},
            total_bytes=10 * GIB,
            est_saving_bytes=3 * GIB,
            csv_path=settings.state_db.parent / "survey.csv",
            unprobeable=((lib / ODD_NAME, "truncated"),),
            unrepresentable=(lib / "bad.mkv",),
            groups=(PartGroup(output_name="Movie.mkv", parts=(pt1, pt2)),),
        )

    monkeypatch.setattr("video_cleaner.survey.run_survey", fake_run_survey)

    result = CliRunner().invoke(app, ["survey", "--config", str(cfg), "--root", str(other_root)])

    assert result.exit_code == 0, result.output
    assert seen == {"root": other_root, "tools": FAKE_TOOLS}
    out = result.stdout
    assert "13 files" in out
    assert "10.0 GiB" in out
    assert "encode" in out and "7" in out
    assert "3.0 GiB" in out
    assert "survey.csv" in out
    assert "Flagged: 2 (unprobeable: 1, unrepresentable: 1)" in out
    assert ODD_NAME in out
    assert "truncated" in out
    assert "Multi-part groups: 1" in out
    assert "Movie.mkv" in out and "Movie pt1.mkv" in out and "Movie pt2.mkv" in out


def test_cli_survey_config_error_exits_2(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["survey", "--config", str(tmp_path / "missing.toml")])
    assert result.exit_code == 2
    assert "config file not found" in result.output


def test_cli_survey_output_under_library_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lib = tmp_path / "library"
    lib.mkdir()
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f"library_root = '{lib.as_posix()}'\n"
        f"scratch_dir = '{(tmp_path / 's').as_posix()}'\n"
        f"completed_dir = '{(tmp_path / 'c').as_posix()}'\n"
        f"state_db = '{(lib / 'state.db').as_posix()}'\n"
        f"log_dir = '{(tmp_path / 'logs').as_posix()}'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("video_cleaner.cli.locate_tools", lambda: FAKE_TOOLS)

    result = CliRunner().invoke(app, ["survey", "--config", str(cfg), "-q"])

    assert result.exit_code == 2
    assert "library root" in result.output
    assert not (lib / "state.db").exists()


# --- integration: one tiny real clip ---------------------------------------------------------


@pytest.mark.integration
def test_survey_real_clip(tmp_path: Path) -> None:
    tools = locate_tools()
    settings = _settings(tmp_path)
    lib = settings.library_root
    lib.mkdir()
    clip = lib / ODD_NAME
    subprocess.run(  # noqa: S603 - fixed argument list
        [
            str(tools.ffmpeg), "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=s=160x120:r=24:d=2",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(clip),
        ],
        check=True,
    )  # fmt: skip
    _touch(lib / "garbage.mkv", b"definitely not matroska")
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in lib.iterdir()}

    state = State.open(settings.state_db)
    result = run_survey(settings, tools, state, RecordingReporter())
    state.close()

    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in lib.iterdir()} == before
    assert [p.name for p, _ in result.unprobeable] == ["garbage.mkv"]
    rows = _by_name(_csv_rows(result.csv_path))
    assert rows[ODD_NAME]["codec"] == "h264"
    assert (rows[ODD_NAME]["width"], rows[ODD_NAME]["height"]) == ("160", "120")
    # A tiny clip is far below the bit rate floor: left alone, not encoded.
    assert rows[ODD_NAME]["action"] == "leave_alone"
    assert rows["garbage.mkv"]["action"] == "flag"
