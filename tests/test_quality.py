"""Quality guard arithmetic (pure) and the VMAF scorer (fake run_tool, then real libvmaf)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from media import RecordingReporter, make_clip, vmaf_tools
from video_cleaner import quality as quality_module
from video_cleaner.quality import (
    GuardResult,
    VmafError,
    build_vmaf_cmd,
    evaluate_guard,
    parse_vmaf_log,
    vmaf_mean,
    vmaf_threads,
)
from video_cleaner.tools import ToolFailed, ToolResult, Tools

TOOLS = Tools(ffmpeg=Path("C:/t/ffmpeg.exe"), ffprobe=Path("C:/t/ffprobe.exe"), mkvmerge=Path("m"))


def test_guard_passes_when_all_windows_are_good() -> None:
    result = evaluate_guard([95.0, 94.0, 96.0], mean_floor=93.0, worst_floor=88.0)
    assert result.ok
    assert result.mean == pytest.approx(95.0)
    assert result.worst == 94.0
    assert result.reason == ""


def test_guard_fails_when_mean_below_93() -> None:
    result = evaluate_guard([92.0, 92.5, 92.9], mean_floor=93.0, worst_floor=88.0)
    assert not result.ok
    assert "mean" in result.reason
    assert "92.47" in result.reason
    assert "93.00" in result.reason


def test_guard_fails_when_one_window_below_worst_floor_even_if_mean_ok() -> None:
    result = evaluate_guard([99.0, 99.0, 85.0], mean_floor=93.0, worst_floor=88.0)
    assert result.mean == pytest.approx(94.3333, abs=1e-3)
    assert result.mean >= 93.0
    assert not result.ok
    assert result.worst == 85.0
    assert "worst" in result.reason
    assert "85.00" in result.reason
    assert "88.00" in result.reason


def test_guard_reports_both_failures() -> None:
    result = evaluate_guard([80.0, 85.0], mean_floor=93.0, worst_floor=88.0)
    assert not result.ok
    assert "mean" in result.reason
    assert "worst" in result.reason


def test_guard_boundaries_are_inclusive() -> None:
    assert evaluate_guard([93.0, 93.0], mean_floor=93.0, worst_floor=93.0).ok


def test_guard_empty_input_fails() -> None:
    result = evaluate_guard([], mean_floor=93.0, worst_floor=88.0)
    assert not result.ok
    assert "no window scores" in result.reason


def test_guard_non_finite_score_fails() -> None:
    for bad in (float("nan"), float("inf")):
        result = evaluate_guard([95.0, bad], mean_floor=93.0, worst_floor=88.0)
        assert not result.ok
        assert "finite" in result.reason


def test_guard_result_is_frozen() -> None:
    result = GuardResult(ok=True, mean=95.0, worst=94.0, reason="")
    with pytest.raises(AttributeError):
        result.ok = False  # type: ignore[misc]


# --- the libvmaf command ---------------------------------------------------------------------


def _graph(cmd: list[str]) -> str:
    return cmd[cmd.index("-lavfi") + 1]


def test_vmaf_cmd_distorted_first_reference_second_normalised() -> None:
    cmd = build_vmaf_cmd(TOOLS, Path("D:/w/enc.mkv"), Path("D:/w/src.mkv"), "vmaf-1.json", 8)
    assert cmd[0] == str(TOOLS.ffmpeg)
    assert "-nostdin" in cmd
    inputs = [cmd[i + 1] for i, part in enumerate(cmd) if part == "-i"]
    assert inputs == [str(Path("D:/w/enc.mkv")), str(Path("D:/w/src.mkv"))]
    graph = _graph(cmd)
    # both start at zero and share one pixel format; the distorted is scaled to the reference
    assert graph.count("setpts=PTS-STARTPTS") == 2
    assert graph.count("format=yuv420p10le") == 2
    assert "scale=w=rw:h=rh" in graph
    assert "libvmaf=" in graph
    assert "log_fmt=json" in graph
    assert "n_threads=8" in graph
    assert "shortest=1" in graph
    # only video is scored, nothing is written but the log
    assert cmd[-3:] == ["-f", "null", "-"]
    for flag in ("-an", "-sn", "-dn"):
        assert flag in cmd
    assert cmd[cmd.index("-progress") + 1] == "pipe:1"


def test_vmaf_cmd_log_path_is_relative_without_a_drive_colon() -> None:
    cmd = build_vmaf_cmd(TOOLS, Path("D:/w/enc.mkv"), Path("D:/w/src.mkv"), "vmaf-1.json", 4)
    assert "log_path=vmaf-1.json:" in _graph(cmd)
    for bad in ("D:/w/vmaf.json", "sub/vmaf.json", "sub\\vmaf.json", "v:x.json", ""):
        with pytest.raises(ValueError, match="log"):
            build_vmaf_cmd(TOOLS, Path("a"), Path("b"), bad, 4)


def test_vmaf_cmd_windows_seek_each_input_before_it() -> None:
    cmd = build_vmaf_cmd(
        TOOLS,
        Path("enc.mkv"),
        Path("src.mkv"),
        "v.json",
        4,
        reference_start_s=600.0,
        reference_length_s=30.0,
        distorted_start_s=12.5,
        distorted_length_s=20.0,
    )
    first, second = (i for i, part in enumerate(cmd) if part == "-i")
    assert cmd[first - 4 : first] == ["-ss", "12.500", "-t", "20.000"]
    assert cmd[second - 4 : second] == ["-ss", "600.000", "-t", "30.000"]


def test_vmaf_cmd_without_windows_does_not_seek() -> None:
    cmd = build_vmaf_cmd(TOOLS, Path("enc.mkv"), Path("src.mkv"), "v.json", 4)
    assert "-ss" not in cmd and "-t" not in cmd


@pytest.mark.parametrize(
    "window",
    [
        {"reference_start_s": -1.0},
        {"reference_length_s": 0.0},
        {"distorted_start_s": float("nan")},
        {"distorted_length_s": float("inf")},
    ],
)
def test_vmaf_cmd_rejects_bad_windows(window: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="_s"):
        build_vmaf_cmd(TOOLS, Path("a"), Path("b"), "v.json", 4, **window)


@pytest.mark.parametrize(("cpus", "threads"), [(None, 1), (1, 1), (8, 8), (28, 16), (64, 16)])
def test_vmaf_threads(cpus: int | None, threads: int) -> None:
    assert vmaf_threads(cpus) == threads


# --- the JSON log ----------------------------------------------------------------------------


def _log(tmp_path: Path, data: Any) -> Path:
    path = tmp_path / "vmaf.json"
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return path


def test_parse_vmaf_log_reads_the_pooled_mean(tmp_path: Path) -> None:
    log = _log(tmp_path, {"pooled_metrics": {"vmaf": {"min": 90.0, "mean": 95.25}}})
    assert parse_vmaf_log(log) == 95.25


@pytest.mark.parametrize(
    "data",
    [
        "",
        "not json",
        "[]",
        {},
        {"pooled_metrics": {}},
        {"pooled_metrics": {"vmaf": {"min": 3}}},
        {"pooled_metrics": {"vmaf": {"mean": "95"}}},
        {"pooled_metrics": {"vmaf": {"mean": True}}},
        '{"pooled_metrics": {"vmaf": {"mean": NaN}}}',
        '{"pooled_metrics": {"vmaf": {"mean": Infinity}}}',
    ],
)
def test_parse_vmaf_log_rejects_unusable_logs(tmp_path: Path, data: Any) -> None:
    with pytest.raises(VmafError):
        parse_vmaf_log(_log(tmp_path, data))


def test_parse_vmaf_log_missing_is_a_vmaf_error(tmp_path: Path) -> None:
    with pytest.raises(VmafError, match="missing"):
        parse_vmaf_log(tmp_path / "nope.json")


# --- vmaf_mean with a fake run_tool ----------------------------------------------------------


class FakeRun:
    """Stands in for run_tool: records the call and writes the JSON log libvmaf would."""

    def __init__(self, mean: Any = 96.5, *, write: bool = True, fail: bool = False) -> None:
        self.mean = mean
        self.write = write
        self.fail = fail
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, cmd: list[str], **kw: Any) -> ToolResult:
        self.calls.append((cmd, kw))
        if self.fail:
            raise ToolFailed(1, "ffmpeg ...", kw.get("stderr_log"))
        progress = ["out_time_us=1500000\n", "speed=2.0x\n", "progress=continue\n"]
        if kw.get("parser") is not None:
            for item in kw["parser"](progress):
                kw["reporter"].update(item)
        if self.write:
            graph = _graph(cmd)
            name = graph.split("log_path=")[1].split(":")[0]
            body = {"pooled_metrics": {"vmaf": {"mean": self.mean}}}
            (kw["cwd"] / name).write_text(json.dumps(body), encoding="utf-8")
        return ToolResult(returncode=0, stdout="", stderr_tail="")


def test_vmaf_mean_runs_in_workdir_and_returns_the_mean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeRun(97.25)
    monkeypatch.setattr(quality_module, "run_tool", fake)
    reporter = RecordingReporter()
    work = tmp_path / "work"
    score = vmaf_mean(
        TOOLS,
        Path("D:/x/enc.mkv"),
        Path("D:/x/src.mkv"),
        work,
        reporter,
        reference_start_s=10.0,
        reference_length_s=3.0,
    )
    assert score == 97.25
    (_cmd, kw) = fake.calls[0]
    assert kw["cwd"] == work
    assert "enc.mkv" in kw["label"]
    # progress against the 3 s window: 1.5 s done is 50%
    assert reporter.updates and reporter.updates[0].percent == pytest.approx(50.0)
    assert list(work.glob("*.json")) == []  # the log is removed once read


def test_vmaf_mean_positional_signature_still_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(quality_module, "run_tool", FakeRun(99.0))
    assert vmaf_mean(TOOLS, Path("a.mkv"), Path("b.mkv"), tmp_path, RecordingReporter()) == 99.0


@pytest.mark.parametrize("mean", [float("nan"), float("inf"), None])
def test_vmaf_mean_rejects_non_finite_or_missing_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mean: Any
) -> None:
    monkeypatch.setattr(quality_module, "run_tool", FakeRun(mean))
    with pytest.raises(VmafError):
        vmaf_mean(TOOLS, Path("a.mkv"), Path("b.mkv"), tmp_path, RecordingReporter())


def test_vmaf_mean_missing_log_is_a_vmaf_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(quality_module, "run_tool", FakeRun(write=False))
    with pytest.raises(VmafError, match="missing"):
        vmaf_mean(TOOLS, Path("a.mkv"), Path("b.mkv"), tmp_path, RecordingReporter())


def test_vmaf_mean_ffmpeg_failure_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(quality_module, "run_tool", FakeRun(fail=True))
    with pytest.raises(ToolFailed):
        vmaf_mean(TOOLS, Path("a.mkv"), Path("b.mkv"), tmp_path, RecordingReporter())


# --- integration: real libvmaf ---------------------------------------------------------------


@pytest.mark.integration
def test_real_vmaf_identical_is_about_100_and_degraded_is_much_lower(tmp_path: Path) -> None:
    tools = vmaf_tools()
    ref = make_clip(tools, tmp_path / "ref [a] (1).mkv", duration_s=4.0)
    degraded = tmp_path / "degraded.mkv"
    subprocess.run(  # noqa: S603 - fixed argument list
        [
            str(tools.ffmpeg), "-nostdin", "-v", "error", "-y", "-i", str(ref),
            "-vf", "scale=80:60", "-c:v", "libx264", "-crf", "45", "-an", str(degraded),
        ],
        check=True,
    )  # fmt: skip
    work = tmp_path / "work dir"
    same = vmaf_mean(tools, ref, ref, work, RecordingReporter())
    worse = vmaf_mean(tools, degraded, ref, work, RecordingReporter())
    assert same > 99.0
    assert worse < same - 20.0
    # a window of the reference against the same window of itself
    windowed = vmaf_mean(
        tools,
        ref,
        ref,
        work,
        RecordingReporter(),
        reference_start_s=1.0,
        reference_length_s=2.0,
        distorted_start_s=1.0,
        distorted_length_s=2.0,
    )
    assert windowed > 99.0
    assert not any(p.suffix == ".json" for p in work.iterdir())
    shutil.rmtree(work)
