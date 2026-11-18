"""Pure helpers of the encode stage: windows, CRF interpolation, saving, x265 command."""

from __future__ import annotations

import dataclasses
import itertools
import numbers
import os
from pathlib import Path
from typing import Any

import pytest

from helpers import make_info
from media import RecordingReporter, make_clip, vmaf_tools
from video_cleaner import encode as encode_module
from video_cleaner import quality as quality_module
from video_cleaner.config import Settings
from video_cleaner.encode import (
    CrfSearchResult,
    EncodeError,
    build_packet_sizes_cmd,
    build_x265_cmd,
    choose_crf,
    encode_full,
    fit_log_size,
    interpolate_crf,
    parse_packet_bytes,
    pick_windows,
    project_file_saving,
    projected_saving,
    sample_windows,
    search_crf,
)
from video_cleaner.probe import MediaInfo, probe
from video_cleaner.tools import ToolFailed, ToolResult, Tools

FFMPEG = Path("C:/tools/ffmpeg.exe")


def _tools() -> Tools:
    return Tools(
        ffmpeg=FFMPEG, ffprobe=Path("C:/tools/ffprobe.exe"), mkvmerge=Path("C:/tools/mkvmerge.exe")
    )


def _settings(preset: str = "slow") -> Settings:
    return Settings(
        library_root=Path("//nas/media"),
        scratch_dir=Path("D:/scratch"),
        completed_dir=Path("D:/completed"),
        state_db=Path("D:/state.db"),
        log_dir=Path("D:/logs"),
        preset=preset,
    )


def _contains_run(cmd: list[str], run: list[str]) -> bool:
    return any(cmd[i : i + len(run)] == run for i in range(len(cmd) - len(run) + 1))


# --- pick_windows ---------------------------------------------------------------------------


def test_pick_windows_even_and_inside_bounds() -> None:
    starts = pick_windows(7200.0)  # 2 h film
    assert len(starts) == 3
    gaps = [b - a for a, b in itertools.pairwise(starts)]
    assert gaps[0] == pytest.approx(gaps[1])
    for start in starts:
        assert start >= 0.10 * 7200.0
        assert start + 30.0 <= 0.90 * 7200.0
    assert starts == sorted(starts)


def test_pick_windows_skips_first_and_last_ten_percent() -> None:
    duration = 3000.0
    for count in (1, 3, 4, 7):
        starts = pick_windows(duration, count=count, length_s=20.0)
        assert len(starts) == count
        assert min(starts) >= 300.0
        assert max(starts) + 20.0 <= 2700.0


def test_pick_windows_single_window_is_centred() -> None:
    (start,) = pick_windows(1000.0, count=1, length_s=30.0)
    assert start + 15.0 == pytest.approx(500.0)


def test_pick_windows_short_film_stays_inside_the_middle() -> None:
    # 100 s: usable region 10..90 is 80 s, three 30 s windows must overlap.
    starts = pick_windows(100.0)
    assert starts
    assert all(s >= 10.0 and s + 30.0 <= 90.0 for s in starts)


def test_pick_windows_film_shorter_than_a_window_returns_one_start_at_ten_percent() -> None:
    assert pick_windows(20.0, count=3, length_s=30.0) == [pytest.approx(2.0)]


def test_pick_windows_no_duplicates_when_clamped() -> None:
    starts = pick_windows(40.0, count=3, length_s=30.0)
    assert len(starts) == len(set(starts))


@pytest.mark.parametrize("duration", [0.0, -5.0, float("nan"), float("inf")])
def test_pick_windows_unusable_duration_gives_no_windows(duration: float) -> None:
    assert pick_windows(duration) == []


def test_pick_windows_zero_count_gives_no_windows() -> None:
    assert pick_windows(3600.0, count=0) == []


def test_pick_windows_rejects_non_positive_length() -> None:
    with pytest.raises(ValueError, match="length_s"):
        pick_windows(3600.0, length_s=0.0)


# --- interpolate_crf ------------------------------------------------------------------------


def test_interpolate_crf_between_points() -> None:
    assert interpolate_crf([(20, 96.0), (24, 94.0)], 95.0) == 22


def test_interpolate_crf_rounds_to_nearest() -> None:
    # 20 + 4 * (96 - 95.6) / 2 = 20.8 -> 21
    assert interpolate_crf([(20, 96.0), (24, 94.0)], 95.6) == 21


def test_interpolate_crf_clamps_high_and_low() -> None:
    points = [(20, 96.0), (24, 94.0)]
    assert interpolate_crf(points, 80.0) == 24
    assert interpolate_crf(points, 99.0) == 16


def test_interpolate_crf_extrapolates_before_clamping() -> None:
    # slope is -0.5 VMAF per CRF step; with wide bounds the extrapolated value is visible.
    points = [(20, 96.0), (24, 94.0)]
    assert interpolate_crf(points, 93.0, lo=16, hi=30) == 26
    assert interpolate_crf(points, 98.0, lo=10, hi=30) == 16


def test_interpolate_crf_respects_custom_bounds() -> None:
    points = [(20, 96.0), (24, 94.0)]
    assert interpolate_crf(points, 95.0, lo=23, hi=26) == 23


def test_interpolate_crf_point_order_does_not_matter() -> None:
    assert interpolate_crf([(24, 94.0), (20, 96.0)], 95.0) == 22


def test_interpolate_crf_degenerate_equal_vmaf_returns_middle() -> None:
    assert interpolate_crf([(20, 95.0), (24, 95.0)], 95.0) == 22
    assert interpolate_crf([(20, 95.0), (24, 95.0)], 99.0) == 22


def test_interpolate_crf_degenerate_middle_is_clamped() -> None:
    assert interpolate_crf([(26, 95.0), (30, 95.0)], 95.0, lo=16, hi=24) == 24


def test_interpolate_crf_rising_vmaf_with_crf_still_returns_a_clamped_crf() -> None:
    # Noisy scores can invert the curve; the function must still return a clamped CRF.
    assert 16 <= interpolate_crf([(20, 94.0), (24, 96.0)], 95.0) <= 24


def test_interpolate_crf_overflowing_slope_clamps_to_the_extrapolated_side() -> None:
    # A denormal VMAF difference makes the extrapolation infinite: it is clamped, not hidden.
    assert interpolate_crf([(20, 0.0), (24, 5e-324)], 1.0) == 24
    assert interpolate_crf([(20, 5e-324), (24, 0.0)], 1.0) == 16


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_interpolate_crf_rejects_non_finite_vmaf(bad: float) -> None:
    with pytest.raises(ValueError, match="VMAF"):
        interpolate_crf([(20, bad), (24, 94.0)], 95.0)


def test_interpolate_crf_rejects_non_finite_target() -> None:
    with pytest.raises(ValueError, match="target"):
        interpolate_crf([(20, 96.0), (24, 94.0)], float("nan"))


@pytest.mark.parametrize("bad_crf", [20.5, True, False, "20", float("nan"), None])
def test_interpolate_crf_rejects_non_int_crf(bad_crf: object) -> None:
    with pytest.raises(ValueError, match="CRF"):
        interpolate_crf([(bad_crf, 96.0), (24, 94.0)], 95.0)  # type: ignore[list-item]


class _IntLike:
    """An integer that is not an ``int`` (as numpy's int64 is), registered as Integral."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __index__(self) -> int:
        return self.value

    def __int__(self) -> int:
        return self.value


numbers.Integral.register(_IntLike)


class _IntSubclass(int):
    pass


@pytest.mark.parametrize("make", [_IntLike, _IntSubclass])
def test_interpolate_crf_accepts_integer_like_crfs(make: type) -> None:
    """R21: numpy-style integers (any numbers.Integral except bool) are accepted."""
    points = [(make(20), 96.0), (make(24), 94.0)]
    result = interpolate_crf(points, 95.0)
    assert result == 22
    assert type(result) is int


def test_interpolate_crf_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError, match="lo"):
        interpolate_crf([(20, 96.0), (24, 94.0)], 95.0, lo=24, hi=16)


def test_interpolate_crf_needs_points() -> None:
    with pytest.raises(ValueError, match="points"):
        interpolate_crf([], 95.0)


def test_interpolate_crf_single_point_returns_its_crf_clamped() -> None:
    assert interpolate_crf([(22, 95.0)], 95.0) == 22
    assert interpolate_crf([(30, 95.0)], 95.0) == 24


# --- projected_saving -----------------------------------------------------------------------


def test_projected_saving() -> None:
    assert projected_saving(100, 40) == pytest.approx(0.60)


def test_projected_saving_zero_input_is_zero() -> None:
    assert projected_saving(0, 0) == 0.0
    assert projected_saving(0, 50) == 0.0


def test_projected_saving_output_larger_is_negative() -> None:
    assert projected_saving(100, 120) == pytest.approx(-0.20)


# --- build_x265_cmd -------------------------------------------------------------------------


def test_x265_cmd_has_required_flags() -> None:
    cmd = build_x265_cmd(_tools(), Path("in.mkv"), Path("out.mkv"), 22, _settings())
    assert cmd[0] == str(FFMPEG)
    assert _contains_run(cmd, ["-c:v", "libx265", "-preset", "slow", "-crf", "22"])
    assert _contains_run(cmd, ["-pix_fmt", "yuv420p10le"])
    assert _contains_run(cmd, ["-map", "0:V:0"])
    assert _contains_run(cmd, ["-an", "-sn"])
    assert _contains_run(cmd, ["-progress", "pipe:1", "-nostats"])
    assert "-y" in cmd
    assert cmd[-1] == str(Path("out.mkv"))
    assert all(isinstance(part, str) for part in cmd)


def test_x265_cmd_uses_the_configured_preset() -> None:
    cmd = build_x265_cmd(_tools(), Path("in.mkv"), Path("out.mkv"), 20, _settings("medium"))
    assert _contains_run(cmd, ["-preset", "medium"])


def test_x265_cmd_input_precedes_output_and_options() -> None:
    cmd = build_x265_cmd(_tools(), Path("in.mkv"), Path("out.mkv"), 22, _settings())
    assert cmd[cmd.index("-i") + 1] == str(Path("in.mkv"))
    assert cmd.index("-i") < cmd.index("-map") < cmd.index("-c:v") < len(cmd) - 1


def test_x265_cmd_without_window_has_no_seek_or_duration() -> None:
    cmd = build_x265_cmd(_tools(), Path("in.mkv"), Path("out.mkv"), 22, _settings())
    assert "-ss" not in cmd
    assert "-t" not in cmd


def test_x265_cmd_window_options_come_before_input() -> None:
    cmd = build_x265_cmd(
        _tools(), Path("in.mkv"), Path("out.mkv"), 22, _settings(), start_s=1234.5, length_s=30.0
    )
    assert _contains_run(cmd, ["-ss", "1234.5"])
    assert _contains_run(cmd, ["-t", "30.0"])
    assert cmd.index("-ss") < cmd.index("-i")
    assert cmd.index("-t") < cmd.index("-i")


def test_x265_cmd_start_only_and_length_only() -> None:
    only_start = build_x265_cmd(
        _tools(), Path("in.mkv"), Path("o.mkv"), 22, _settings(), start_s=10.0
    )
    assert "-ss" in only_start
    assert "-t" not in only_start
    only_len = build_x265_cmd(
        _tools(), Path("in.mkv"), Path("o.mkv"), 22, _settings(), length_s=30.0
    )
    assert "-t" in only_len
    assert "-ss" not in only_len


def test_x265_cmd_start_zero_is_still_emitted() -> None:
    cmd = build_x265_cmd(_tools(), Path("in.mkv"), Path("o.mkv"), 22, _settings(), start_s=0.0)
    assert _contains_run(cmd, ["-ss", "0.0"])


def test_x265_cmd_never_maps_cover_art() -> None:
    cmd = build_x265_cmd(_tools(), Path("in.mkv"), Path("out.mkv"), 22, _settings())
    maps = [cmd[i + 1] for i, part in enumerate(cmd) if part == "-map"]
    # Capital V: the first real video stream, never an attached picture (lower-case v matches
    # attached pics too). Exactly one map: no 0:v, 0, 0:t, 0:d, 0:s, 0:a.
    assert maps == ["0:V:0"]
    for forbidden in ("-attach", "-c", "-c:a", "-c:s", "-c:t", "-c:d", "copy"):
        assert forbidden not in cmd
    assert "-an" in cmd
    assert "-sn" in cmd


def test_x265_cmd_drops_chapters_and_metadata_from_the_encode() -> None:
    # The mux step takes chapters and tags from the source; the encode must carry none, or
    # mkvmerge writes a duplicate chapter edition and stale track statistics.
    cmd = build_x265_cmd(
        _tools(), Path("in.mkv"), Path("out.mkv"), 22, _settings(), start_s=5.0, length_s=30.0
    )
    assert _contains_run(cmd, ["-map_chapters", "-1"])
    assert _contains_run(cmd, ["-map_metadata", "-1"])
    # output options: after the input, before the output path
    assert cmd.index("-i") < cmd.index("-map_chapters") < len(cmd) - 1
    assert cmd.index("-i") < cmd.index("-map_metadata") < len(cmd) - 1


def test_x265_cmd_keeps_odd_paths_as_single_arguments() -> None:
    src = Path("D:/scratch/The First Omen (2024).en.MX].mkv")
    dst = Path("D:/scratch/out [x].mkv")
    cmd = build_x265_cmd(_tools(), src, dst, 18, _settings())
    assert str(src) in cmd
    assert cmd[-1] == str(dst)


@pytest.mark.parametrize("crf", [-1, 52])
def test_x265_cmd_rejects_crf_outside_x265_range(crf: int) -> None:
    with pytest.raises(ValueError, match="crf"):
        build_x265_cmd(_tools(), Path("in.mkv"), Path("out.mkv"), crf, _settings())


def test_x265_cmd_rejects_bad_window() -> None:
    with pytest.raises(ValueError, match="start_s"):
        build_x265_cmd(_tools(), Path("i.mkv"), Path("o.mkv"), 22, _settings(), start_s=-1.0)
    with pytest.raises(ValueError, match="length_s"):
        build_x265_cmd(_tools(), Path("i.mkv"), Path("o.mkv"), 22, _settings(), length_s=0.0)


def test_x265_cmd_never_reads_stdin() -> None:
    cmd = build_x265_cmd(_tools(), Path("in.mkv"), Path("o.mkv"), 22, _settings())
    assert cmd[1] == "-nostdin"


# --- CrfSearchResult ------------------------------------------------------------------------


def test_crf_search_result_is_frozen() -> None:
    result = CrfSearchResult(crf=22, projected_saving=0.5, points=((20, 96.0), (24, 94.0)))
    assert result.points == ((20, 96.0), (24, 94.0))
    assert result.extrapolated is False
    with pytest.raises(AttributeError):
        result.crf = 20  # type: ignore[misc]


# --- pure pieces of the search --------------------------------------------------------------


def test_sample_windows_cut_to_the_end_of_the_film() -> None:
    # rounded to milliseconds, as the commands carry them
    assert sample_windows(1000.0, 3, 30.0) == [(round(s, 3), 30.0) for s in pick_windows(1000.0)]
    # 20 s film: the middle 80% (2..18 s) is shorter than one window; one window, cut short
    ((start, length),) = sample_windows(20.0, 3, 30.0)
    assert start == pytest.approx(2.0)
    assert length == pytest.approx(18.0)
    assert start + length <= 20.0
    assert sample_windows(0.0, 3, 30.0) == []
    assert sample_windows(float("nan"), 3, 30.0) == []


def test_packet_sizes_cmd_reads_only_the_window_of_the_first_video() -> None:
    cmd = build_packet_sizes_cmd(_tools(), Path("D:/s/a [1].mkv"), 600.0, 30.0)
    assert cmd[0] == str(_tools().ffprobe)
    assert _contains_run(cmd, ["-select_streams", "V:0"])
    # an absolute end: "%+30" would count from the keyframe ffprobe seeks to
    assert _contains_run(cmd, ["-read_intervals", "600.000%630.000"])
    assert _contains_run(cmd, ["-show_entries", "packet=pts_time,dts_time,size"])
    assert cmd[-1] == str(Path("D:/s/a [1].mkv"))


def test_parse_packet_bytes_sums_the_window_only() -> None:
    text = "\n".join(
        [
            "9.958000,9.958000,500",  # before the window (read from the keyframe before it)
            "10.000000,10.000000,1000",
            "N/A,10.042000,2000",  # no pts: the dts is used
            "15.000000,N/A,3000",
            "",
            "garbage",
            "12.0,12.0,notanumber",
            "40.000000,40.000000,9999",  # at the end: excluded
        ]
    )
    assert parse_packet_bytes(text, 10.0, 30.0) == 6000


@pytest.mark.parametrize(
    ("points", "crf"),
    [
        (((20, 97.0), (24, 96.0)), 24),  # above the target at both: the ceiling
        (((20, 100.0), (24, 100.0)), 24),
        (((20, 90.0), (24, 85.0)), 16),  # below at both: the floor
        (((20, 96.0), (24, 94.0)), 22),  # bracketing: interpolated
        (((20, 95.0), (24, 93.0)), 20),  # exactly on the target at CRF 20
    ],
)
def test_choose_crf(points: tuple[tuple[int, float], ...], crf: int) -> None:
    assert choose_crf(points, 95.0, 16, 24) == crf


def test_fit_log_size_is_exact_at_the_samples_and_geometric_between() -> None:
    sizes = ((20, 4000), (24, 1000))
    assert fit_log_size(sizes, 20) == pytest.approx(4000.0)
    assert fit_log_size(sizes, 24) == pytest.approx(1000.0)
    assert fit_log_size(sizes, 22) == pytest.approx(2000.0)
    assert fit_log_size(sizes, 16) == pytest.approx(16000.0)  # extrapolated


@pytest.mark.parametrize("sizes", [((20, 0), (24, 10)), ((20, 10),), ((20, 10), (20, 5))])
def test_fit_log_size_rejects_unusable_samples(sizes: tuple[tuple[int, int], ...]) -> None:
    with pytest.raises(ValueError, match="sample"):
        fit_log_size(sizes, 22)


def test_project_file_saving_keeps_the_non_video_bytes() -> None:
    # 1000 s film, 10 GB; 90 s sampled: 450 MB of video in, 90 MB out (20% of the video).
    # Video in the whole file ~ 5 GB, out ~ 1 GB; audio etc. 5 GB untouched -> 6 GB: 40% saved.
    saving = project_file_saving(10_000_000_000, 1000.0, 90.0, 450_000_000, 90_000_000)
    assert saving == pytest.approx(0.4)


def test_project_file_saving_never_counts_more_video_than_the_file() -> None:
    saving = project_file_saving(1_000, 100.0, 10.0, 1_000, 100)  # estimate 10x the file
    assert saving == pytest.approx(0.9)
    assert project_file_saving(0, 100.0, 10.0, 1, 1) == 0.0


# --- search_crf and encode_full with a fake run_tool ----------------------------------------


class FakeTools:
    """Stands in for run_tool (ffmpeg sample/full encodes, ffprobe packet sizes) and VMAF."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.kwargs: list[dict[str, Any]] = []
        # encoded bytes per second of window, per CRF (CRF 20 is twice the size of CRF 24)
        self.rate = {20: 200, 22: 141, 24: 100}
        self.vmaf = {20: 96.0, 24: 94.0}
        self.source_rate = 1000  # source video bytes per second
        self.fail_crf: int | None = None
        self.vmaf_calls: list[dict[str, Any]] = []

    def run_tool(self, cmd: list[str], **kw: Any) -> ToolResult:
        self.calls.append(cmd)
        self.kwargs.append(kw)
        if cmd[0] == str(FFPROBE):
            start, _, end = cmd[cmd.index("-read_intervals") + 1].partition("%")
            first, span = float(start), float(end) - float(start)
            lines = [f"{first - 0.5:.6f},{first - 0.5:.6f},{self.source_rate}"]
            lines += [
                f"{first + i:.6f},{first + i:.6f},{self.source_rate}" for i in range(round(span))
            ]
            return ToolResult(0, "\n".join(lines) + "\n", "")
        crf = int(cmd[cmd.index("-crf") + 1])
        if crf == self.fail_crf:
            raise ToolFailed(1, "ffmpeg ...", kw.get("stderr_log"))
        length = float(cmd[cmd.index("-t") + 1]) if "-t" in cmd else 100.0
        out = Path(cmd[-1])
        out.write_bytes(b"e" * int(self.rate.get(crf, 0) * length))
        if kw.get("parser") is not None:
            lines = [f"out_time_us={int(length * 500_000)}\n", "progress=continue\n"]
            for item in kw["parser"](lines):
                kw["reporter"].update(item)
        return ToolResult(0, "", "")

    def vmaf_mean(
        self,
        tools: Tools,
        distorted: Path,
        reference: Path,
        workdir: Path,
        reporter: Any,
        **kw: Any,
    ) -> float:
        self.vmaf_calls.append({"distorted": distorted, "reference": reference, **kw})
        crf = int(distorted.name.split("crf")[1].split("-")[0])
        return self.vmaf[crf]


FFPROBE = Path("C:/tools/ffprobe.exe")


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeTools:
    fakes = FakeTools()
    monkeypatch.setattr(encode_module, "run_tool", fakes.run_tool)
    monkeypatch.setattr(quality_module, "vmaf_mean", fakes.vmaf_mean)
    return fakes


def _media(path: Path, duration_s: float, size: int = 10_000_000) -> MediaInfo:
    return dataclasses.replace(make_info(path, size), duration_s=duration_s)


def test_search_crf_samples_two_crfs_and_interpolates(fake: FakeTools, tmp_path: Path) -> None:
    info = _media(tmp_path / "copy [x].mkv", 1000.0, size=1_000_000)
    reporter = RecordingReporter()
    work = tmp_path / "work"
    result = search_crf(info, _tools(), _settings(), work, reporter)

    assert result.points == ((20, 96.0), (24, 94.0))
    assert result.crf == 22
    assert result.extrapolated is False
    assert result.windows == tuple((round(s, 3), 30.0) for s in pick_windows(1000.0))
    # 3 windows x 30 s: 90 s sampled, 90 000 source video bytes
    assert result.source_bytes == 90_000
    assert result.sample_bytes == ((20, 18_000), (24, 9_000))
    # ~1 MB of video in the 1 MB file; at CRF 22 ~141 bytes/s -> ~0.1414 of the video
    assert result.projected_saving == pytest.approx(1 - 0.1414, abs=0.002)

    encodes = [c for c in fake.calls if c[0] == str(FFMPEG)]
    assert len(encodes) == 6
    for cmd in encodes:
        assert cmd[1] == "-nostdin"
        assert cmd[cmd.index("-i") + 1] == str(info.path)
        assert _contains_run(cmd, ["-preset", "slow"])
        assert _contains_run(cmd, ["-pix_fmt", "yuv420p10le"])
        assert Path(cmd[-1]).parent == work
    assert len([c for c in fake.calls if c[0] == str(FFPROBE)]) == 3  # once per window
    # each encoded window is scored against the same window of the source
    for call in fake.vmaf_calls:
        assert call["reference"] == info.path
        assert call["reference_length_s"] == 30.0
        assert call["reference_start_s"] in [s for s, _ in result.windows]
    assert any("CRF 20" in s for s in reporter.stages)
    assert reporter.updates and reporter.updates[0].percent == pytest.approx(50.0)
    assert not list(work.glob("*.mkv"))  # the samples are removed once measured


def test_search_crf_short_film_uses_fewer_shorter_windows(fake: FakeTools, tmp_path: Path) -> None:
    info = _media(tmp_path / "short.mkv", 20.0)
    result = search_crf(info, _tools(), _settings(), tmp_path / "w", RecordingReporter())
    ((_start, length),) = result.windows
    assert length == pytest.approx(18.0)
    encodes = [c for c in fake.calls if c[0] == str(FFMPEG)]
    assert len(encodes) == 2
    assert all(_contains_run(c, ["-t", str(length)]) for c in encodes)


def test_search_crf_short_windows_parameter(fake: FakeTools, tmp_path: Path) -> None:
    info = _media(tmp_path / "m.mkv", 1000.0)
    result = search_crf(
        info, _tools(), _settings(), tmp_path / "w", RecordingReporter(), window_length_s=2.0
    )
    assert [length for _, length in result.windows] == [2.0, 2.0, 2.0]


def test_search_crf_vmaf_below_target_at_both_clamps_to_crf_min(
    fake: FakeTools, tmp_path: Path
) -> None:
    fake.vmaf = {20: 90.0, 24: 85.0}
    info = _media(tmp_path / "m.mkv", 1000.0)
    result = search_crf(info, _tools(), _settings(), tmp_path / "w", RecordingReporter())
    assert result.crf == 16
    assert result.extrapolated is True  # 16 is outside the measured 20..24
    # the size is extrapolated to CRF 16 (twice CRF 20's: 40% of the video, 6% of the file)
    assert result.projected_saving == pytest.approx(0.06, abs=0.001)


def test_search_crf_vmaf_above_target_at_both_clamps_to_crf_max(
    fake: FakeTools, tmp_path: Path
) -> None:
    fake.vmaf = {20: 99.0, 24: 98.0}
    info = _media(tmp_path / "m.mkv", 1000.0)
    settings = dataclasses.replace(_settings(), crf_max=23)
    result = search_crf(info, _tools(), settings, tmp_path / "w", RecordingReporter())
    assert result.crf == 23
    assert result.extrapolated is False


def test_search_crf_without_windows_is_an_encode_error(fake: FakeTools, tmp_path: Path) -> None:
    info = _media(tmp_path / "m.mkv", 0.0)
    with pytest.raises(EncodeError, match="no sample windows"):
        search_crf(info, _tools(), _settings(), tmp_path / "w", RecordingReporter())
    assert fake.calls == []


def test_search_crf_empty_sample_is_an_encode_error(fake: FakeTools, tmp_path: Path) -> None:
    fake.rate[24] = 0
    info = _media(tmp_path / "m.mkv", 1000.0)
    with pytest.raises(EncodeError, match="empty"):
        search_crf(info, _tools(), _settings(), tmp_path / "w", RecordingReporter())


def test_search_crf_window_without_video_is_an_encode_error(
    fake: FakeTools, tmp_path: Path
) -> None:
    fake.source_rate = 0
    info = _media(tmp_path / "m.mkv", 1000.0)
    with pytest.raises(EncodeError, match="no video"):
        search_crf(info, _tools(), _settings(), tmp_path / "w", RecordingReporter())


def test_search_crf_ffmpeg_failure_propagates(fake: FakeTools, tmp_path: Path) -> None:
    fake.fail_crf = 24
    info = _media(tmp_path / "m.mkv", 1000.0)
    with pytest.raises(ToolFailed):
        search_crf(info, _tools(), _settings(), tmp_path / "w", RecordingReporter())


def test_encode_full_encodes_the_whole_video(fake: FakeTools, tmp_path: Path) -> None:
    info = _media(tmp_path / "copy.mkv", 100.0)
    dst = tmp_path / "work" / "video.mkv"
    dst.parent.mkdir()
    reporter = RecordingReporter()
    encode_full(info, _tools(), _settings(), 22, dst, reporter)
    (cmd,) = fake.calls
    assert cmd[1] == "-nostdin"
    assert "-ss" not in cmd and "-t" not in cmd
    assert _contains_run(cmd, ["-crf", "22"])
    assert cmd[-1] == str(dst)
    kw = fake.kwargs[0]
    assert kw["label"] == "encode copy.mkv"
    assert kw["stderr_log"] == dst.with_name("video.mkv.stderr.log")
    assert reporter.updates[0].percent == pytest.approx(50.0)
    assert dst.stat().st_size > 0


def test_encode_full_empty_output_is_an_encode_error(fake: FakeTools, tmp_path: Path) -> None:
    fake.rate[22] = 0
    info = _media(tmp_path / "copy.mkv", 100.0)
    with pytest.raises(EncodeError, match="empty"):
        encode_full(info, _tools(), _settings(), 22, tmp_path / "video.mkv", RecordingReporter())


# --- integration: real x265 + libvmaf on a tiny clip ----------------------------------------


@pytest.mark.integration
def test_real_search_crf_and_encode_full(tmp_path: Path) -> None:
    tools = vmaf_tools()
    clip = make_clip(tools, tmp_path / "src" / "Odd [name] (2024).mkv", duration_s=8.0)
    info = probe(clip, tools, RecordingReporter())
    settings = Settings(
        library_root=tmp_path / "lib",
        scratch_dir=tmp_path / "scratch",
        completed_dir=tmp_path / "done",
        state_db=tmp_path / "s.db",
        log_dir=tmp_path / "logs",
    )
    work = tmp_path / "work"
    result = search_crf(
        info, tools, settings, work, RecordingReporter(), window_count=2, window_length_s=2.0
    )
    assert settings.crf_min <= result.crf <= settings.crf_max
    assert [crf for crf, _ in result.points] == [20, 24]
    assert all(50.0 < vmaf <= 100.0 for _, vmaf in result.points)
    assert 0.0 < result.projected_saving < 1.0  # a CRF 12 x264 source shrinks a lot
    assert len(result.windows) == 2

    dst = work / "video.mkv"
    encode_full(info, tools, settings, result.crf, dst, RecordingReporter())
    out = probe(dst, tools, RecordingReporter())
    assert out.video is not None
    assert (out.video.codec, out.video.pix_fmt) == ("hevc", "yuv420p10le")
    assert out.audio == () and out.chapters == 0
    assert out.duration_s == pytest.approx(info.duration_s, abs=0.5)


# --- local: a real film named by VC_LOCAL_SAMPLE --------------------------------------------


@pytest.mark.local
def test_local_search_crf_on_a_real_film(tmp_path: Path) -> None:
    """``VC_LOCAL_SAMPLE`` names a real video (a scratch copy: it is only read)."""
    sample = os.environ.get("VC_LOCAL_SAMPLE")
    if not sample:
        pytest.skip("set VC_LOCAL_SAMPLE to a real video file to run this test")
    tools = vmaf_tools()
    info = probe(Path(sample), tools, RecordingReporter())
    settings = Settings(
        library_root=tmp_path / "lib",
        scratch_dir=tmp_path / "scratch",
        completed_dir=tmp_path / "done",
        state_db=tmp_path / "s.db",
        log_dir=tmp_path / "logs",
    )
    result = search_crf(info, tools, settings, tmp_path / "work", RecordingReporter())
    assert settings.crf_min <= result.crf <= settings.crf_max
    assert -1.0 <= result.projected_saving <= 1.0
    assert not list((tmp_path / "work").glob("*.mkv"))
