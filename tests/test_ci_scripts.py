"""The retrying Chocolatey installer used by CI, run against a fake `choco`."""

from __future__ import annotations

import os
import shutil
import subprocess  # nosec B404
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / ".github" / "scripts" / "choco-install.sh"
BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(BASH is None, reason="bash is not available")

# The fake feed is down for the first FAIL_INSTALLS installs: `choco install` then prints a 503
# and still exits 0 (as the real one does); afterwards the packages show in `choco list`.
FAKE_CHOCO = """#!/usr/bin/env bash
state="$FAKE_CHOCO_STATE"
case "$1" in
  install)
    count=$(cat "$state" 2>/dev/null || echo 0)
    echo $((count + 1)) > "$state"
    if [ "$count" -lt "$FAKE_CHOCO_FAIL_INSTALLS" ]; then
      echo "Failed to fetch results from V2 feed: 503 (Service Unavailable)"
    fi
    exit 0 ;;
  list)
    count=$(cat "$state" 2>/dev/null || echo 0)
    if [ "$count" -gt "$FAKE_CHOCO_FAIL_INSTALLS" ]; then
      echo "ffmpeg-full|9.0.2"
      echo "mkvtoolnix|103.0.0"
    fi
    exit 0 ;;
esac
"""


def _run(tmp_path: Path, fail_installs: int, attempts: int = 3) -> tuple[int, int, str]:
    """(exit code, number of install calls, stderr) of the script against the fake feed."""
    assert BASH is not None
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "choco").write_text(FAKE_CHOCO, encoding="utf-8", newline="\n")
    (fake / "choco").chmod(0o755)
    state = tmp_path / "installs"
    env = {
        **os.environ,
        "PATH": f"{fake}{os.pathsep}{os.environ['PATH']}",
        "FAKE_CHOCO_STATE": str(state),
        "FAKE_CHOCO_FAIL_INSTALLS": str(fail_installs),
        "CHOCO_ATTEMPTS": str(attempts),
        "CHOCO_RETRY_DELAY_S": "0",
    }
    done = subprocess.run(  # noqa: S603  # nosec B603
        [BASH, str(SCRIPT), "ffmpeg-full", "mkvtoolnix"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    installs = int(state.read_text(encoding="utf-8")) if state.exists() else 0
    return done.returncode, installs, done.stderr


def test_installs_once_when_the_feed_is_up(tmp_path: Path) -> None:
    code, installs, _ = _run(tmp_path, fail_installs=0)
    assert (code, installs) == (0, 1)


def test_retries_after_a_feed_outage_that_still_exits_zero(tmp_path: Path) -> None:
    code, installs, stderr = _run(tmp_path, fail_installs=2)
    assert (code, installs) == (0, 3)
    assert "attempt 1/3: not installed: ffmpeg-full mkvtoolnix" in stderr


def test_fails_when_the_packages_never_install(tmp_path: Path) -> None:
    code, installs, stderr = _run(tmp_path, fail_installs=99, attempts=2)
    assert (code, installs) == (1, 2)
    assert "could not install: ffmpeg-full mkvtoolnix" in stderr
