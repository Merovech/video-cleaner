"""Single-instance lock: one `run` or `calibrate` at a time per state database."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from video_cleaner.instancelock import (
    LOCK_NAME,
    AlreadyRunning,
    instance_lock,
    lock_path_for,
    subs_lock_path_for,
)


def test_lock_file_lives_beside_the_state_db(tmp_path: Path) -> None:
    assert lock_path_for(tmp_path / "a" / "state.db") == tmp_path / "a" / LOCK_NAME


def test_second_acquire_in_same_process_is_refused_with_pid(tmp_path: Path) -> None:
    lock = tmp_path / LOCK_NAME
    with instance_lock(lock), pytest.raises(AlreadyRunning) as info, instance_lock(lock):
        pass  # pragma: no cover - never reached
    assert info.value.pid == os.getpid()
    assert str(os.getpid()) in str(info.value)


def test_the_run_lock_message_is_unchanged_and_the_subs_lock_names_its_holder(
    tmp_path: Path,
) -> None:
    run_lock = lock_path_for(tmp_path / "state.db")
    with (
        instance_lock(run_lock),
        pytest.raises(AlreadyRunning) as run_info,
        instance_lock(run_lock),
    ):
        pass  # pragma: no cover
    assert "another video-cleaner run or calibration is active" in str(run_info.value)
    subs_lock = subs_lock_path_for(tmp_path / "state.db")
    with (
        instance_lock(subs_lock),
        pytest.raises(AlreadyRunning) as subs_info,
        instance_lock(subs_lock),
    ):
        pass  # pragma: no cover
    text = str(subs_info.value)
    assert "another `subs resolve` is running" in text
    assert "run or calibration" not in text
    assert str(os.getpid()) in text and "only one may run at a time" in text


def test_lock_is_released_on_exit_and_on_error(tmp_path: Path) -> None:
    lock = tmp_path / "sub" / LOCK_NAME
    with instance_lock(lock):
        pass
    with pytest.raises(RuntimeError), instance_lock(lock):
        raise RuntimeError("boom")
    with instance_lock(lock):  # free again
        assert lock.read_text(encoding="ascii").strip() == str(os.getpid())


def test_stale_lock_file_from_a_dead_process_does_not_block(tmp_path: Path) -> None:
    lock = tmp_path / LOCK_NAME
    lock.write_text("999999\n", encoding="ascii")  # left behind by a crash: no OS lock
    with instance_lock(lock):
        assert lock.read_text(encoding="ascii").strip() == str(os.getpid())


def test_lock_held_by_another_process_is_refused(tmp_path: Path) -> None:
    lock = tmp_path / LOCK_NAME
    src = Path(__file__).resolve().parents[1] / "src"
    script = textwrap.dedent(
        f"""
        import sys, time
        sys.path.insert(0, {str(src)!r})
        from pathlib import Path
        from video_cleaner.instancelock import instance_lock
        with instance_lock(Path({str(lock)!r})):
            print("locked", flush=True)
            sys.stdin.readline()
        """
    )
    child = subprocess.Popen(  # noqa: S603  # nosec B603
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None and child.stdout.readline().strip() == "locked"
        with pytest.raises(AlreadyRunning) as info, instance_lock(lock):
            pass  # pragma: no cover - never reached
        # The venv launcher on Windows may run the interpreter as a child of child.pid.
        assert info.value.pid is not None and info.value.pid != os.getpid()
    finally:
        assert child.stdin is not None
        child.stdin.write("\n")
        child.stdin.close()
        child.wait(timeout=30)
        if child.stdout is not None:
            child.stdout.close()
    with instance_lock(lock):  # the OS dropped the lock with the process
        pass
