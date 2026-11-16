"""Low system impact: tool processes at below-normal priority, and keep-awake during a run.

Nothing here changes the real power state or a real process priority: the system calls are
replaced (see also the autouse fixture in conftest.py).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from video_cleaner import power
from video_cleaner.tools import run_tool


class _Reporter:
    def stage(self, label: str) -> None:
        pass

    def update(self, p: Any) -> None:
        pass

    def heartbeat(self, label: str, elapsed_s: float) -> None:
        pass

    def file_done(self, summary: str) -> None:
        pass


def test_keep_awake_sets_and_clears_the_execution_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(power, "_set_execution_state", calls.append)
    monkeypatch.setattr(power, "_WINDOWS", True)
    with power.keep_awake():
        assert calls == [power.ES_CONTINUOUS | power.ES_SYSTEM_REQUIRED]
    assert calls[-1] == power.ES_CONTINUOUS


def test_keep_awake_clears_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(power, "_set_execution_state", calls.append)
    monkeypatch.setattr(power, "_WINDOWS", True)
    with pytest.raises(RuntimeError), power.keep_awake():
        raise RuntimeError("boom")
    assert calls[-1] == power.ES_CONTINUOUS


def test_keep_awake_is_a_no_op_elsewhere(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(power, "_set_execution_state", calls.append)
    monkeypatch.setattr(power, "_WINDOWS", False)
    with power.keep_awake():
        pass
    assert calls == []


def test_popen_priority_on_windows_and_elsewhere(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(power, "_WINDOWS", True)
    assert power.creation_flags() == power.BELOW_NORMAL_PRIORITY_CLASS
    monkeypatch.setattr(power, "_WINDOWS", False)
    assert power.creation_flags() == 0


def test_lower_priority_after_spawn_only_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[int, int]] = []
    monkeypatch.setattr(power, "_renice", lambda pid, inc: seen.append((pid, inc)))
    monkeypatch.setattr(power, "_WINDOWS", True)
    power.lower_priority(123)
    assert seen == []
    monkeypatch.setattr(power, "_WINDOWS", False)
    power.lower_priority(123)
    assert seen == [(123, power.NICE_INCREMENT)]


def test_lower_priority_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(pid: int, inc: int) -> None:
        raise PermissionError(1, "not permitted")

    monkeypatch.setattr(power, "_renice", denied)
    monkeypatch.setattr(power, "_WINDOWS", False)
    power.lower_priority(123)  # best effort: logged, not raised


def test_run_tool_starts_tools_at_below_normal_priority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}
    real_popen = subprocess.Popen

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", spy)
    run_tool([sys.executable, "-c", "pass"], label="prio", reporter=_Reporter())
    expected = power.BELOW_NORMAL_PRIORITY_CLASS if sys.platform == "win32" else 0
    assert seen["creationflags"] == expected
