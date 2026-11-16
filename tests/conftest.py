"""Shared fixtures: no test may change the machine's real power state."""

from __future__ import annotations

import pytest

from video_cleaner import power


@pytest.fixture(autouse=True)
def _no_real_power_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """`keep_awake` never reaches SetThreadExecutionState in tests (tests may re-patch it)."""
    monkeypatch.setattr(power, "_set_execution_state", lambda flags: None)
