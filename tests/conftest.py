"""Shared fixtures: no test may change the machine's real power state."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator

import keyring
import keyring.backends.fail
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


@pytest.fixture(autouse=True)
def _no_real_keyring(monkeypatch: pytest.MonkeyPatch) -> None:
    """The keyring library reports "no usable backend", so no test reads or writes the real
    Windows Credential Manager (tests inject fake backends and stores)."""
    monkeypatch.setattr(keyring, "get_keyring", keyring.backends.fail.Keyring)


@pytest.fixture(autouse=True)
def _private_subtitle_cache(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every test gets its own subtitle cache folder through the environment.

    `subtitle_cache_dir` has no default of its own: without LOCALAPPDATA (Linux CI) it must be
    given, and with LOCALAPPDATA it would point into the user's real profile. Nothing is created
    on disk unless a test writes there. A test that needs another value, or none, changes the
    variable itself (``monkeypatch.setenv`` / ``delenv``).
    """
    folder = tmp_path_factory.getbasetemp() / "subtitle-cache" / uuid.uuid4().hex
    monkeypatch.setenv("VC_SUBTITLE_CACHE_DIR", str(folder))
