"""Quality guard arithmetic (pure) and the held VMAF stub."""

from __future__ import annotations

from pathlib import Path

import pytest

from video_cleaner.quality import GuardResult, evaluate_guard, vmaf_mean
from video_cleaner.tools import Tools


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


def test_vmaf_mean_is_held_until_task_10b() -> None:
    tools = Tools(ffmpeg=Path("f"), ffprobe=Path("p"), mkvmerge=Path("m"))
    with pytest.raises(NotImplementedError, match="Task 10b"):
        vmaf_mean(tools, Path("d.mkv"), Path("r.mkv"), Path("w"), None)  # type: ignore[arg-type]
