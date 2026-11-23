"""Shared fixtures: no test may change the machine's real power state."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

from video_cleaner import power
from video_cleaner.logging_setup import LOGGER_NAME, _reset_registered_secrets


@pytest.fixture(autouse=True)
def _no_real_power_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """`keep_awake` never reaches SetThreadExecutionState in tests (tests may re-patch it)."""
    monkeypatch.setattr(power, "_set_execution_state", lambda flags: None)


@pytest.fixture(autouse=True)
def _close_tool_log_handlers() -> Iterator[None]:
    """Close the run-log handlers a test installed, so its log file is not left open.

    `setup_logging` (called by every CLI command) adds a file handler to the tool's logger. An
    open handle stops pytest deleting the test's temp folder on Windows.
    """
    yield
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


@pytest.fixture(autouse=True)
def _no_registered_secrets() -> Iterator[None]:
    """Secrets registered by one test (a process-wide set) never reach the next."""
    _reset_registered_secrets()
    yield
    _reset_registered_secrets()
