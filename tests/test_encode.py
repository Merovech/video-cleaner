"""Pure helpers of the encode stage: windows, CRF interpolation, saving, x265 command."""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from video_cleaner.config import Settings
from video_cleaner.encode import (
    CrfSearchResult,
    build_x265_cmd,
    encode_full,
    interpolate_crf,
    pick_windows,
    projected_saving,
    search_crf,
)
from video_cleaner.probe import MediaInfo
from video_cleaner.tools import Tools

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


@pytest.mark.parametrize("bad_crf", [20.5, True, "20"])
def test_interpolate_crf_rejects_non_int_crf(bad_crf: object) -> None:
    with pytest.raises(ValueError, match="CRF"):
        interpolate_crf([(bad_crf, 96.0), (24, 94.0)], 95.0)  # type: ignore[list-item]


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


# --- CrfSearchResult and the held stubs -----------------------------------------------------


def test_crf_search_result_is_frozen() -> None:
    result = CrfSearchResult(crf=22, projected_saving=0.5, points=((20, 96.0), (24, 94.0)))
    assert result.points == ((20, 96.0), (24, 94.0))
    with pytest.raises(AttributeError):
        result.crf = 20  # type: ignore[misc]


def test_search_crf_is_held_until_task_10b() -> None:
    with pytest.raises(NotImplementedError, match="Task 10b"):
        search_crf(None, _tools(), _settings(), Path("w"), None)  # type: ignore[arg-type]


def test_encode_full_is_held_until_task_10b() -> None:
    info: MediaInfo = None  # type: ignore[assignment]
    with pytest.raises(NotImplementedError, match="Task 10b"):
        encode_full(info, _tools(), _settings(), 22, Path("o.mkv"), None)  # type: ignore[arg-type]
