"""Low system impact while the tool works: below-normal priority and keep-awake.

Tool processes (ffmpeg, ffprobe, mkvmerge) start at below-normal priority so the PC stays usable
during a night of encoding: ``BELOW_NORMAL_PRIORITY_CLASS`` on Windows, ``nice`` +10 elsewhere
(applied right after the spawn: ``preexec_fn`` is unsafe with the survey's worker threads).
During `run` the machine is kept from sleeping (``SetThreadExecutionState`` on Windows, a no-op
elsewhere); the state is restored when the run ends, however it ends.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager

_log = logging.getLogger(__name__)

_WINDOWS = sys.platform == "win32"
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
# Unix "niceness": a process's scheduling priority, from -20 (most favoured) to 19 (least).
# The name is because a higher value makes the process "nicer" to the others, which then get
# the CPU first. Raising it by 10 is the Unix counterpart of Windows' below-normal priority.
NICE_INCREMENT = 10
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def creation_flags() -> int:
    """`subprocess.Popen` ``creationflags`` that start the child at below-normal priority."""
    return BELOW_NORMAL_PRIORITY_CLASS if _WINDOWS else 0


def _renice(pid: int, increment: int) -> None:  # pragma: no cover - POSIX only
    current = os.getpriority(os.PRIO_PROCESS, pid)  # type: ignore[attr-defined,unused-ignore]
    os.setpriority(os.PRIO_PROCESS, pid, current + increment)  # type: ignore[attr-defined,unused-ignore]


def lower_priority(pid: int) -> None:
    """Off Windows: nice the freshly started *pid*. Best effort: a failure is only logged."""
    if _WINDOWS:
        return
    try:
        _renice(pid, NICE_INCREMENT)
    except OSError as exc:
        _log.debug("could not lower the priority of process %d: %s", pid, exc)


def _set_execution_state(flags: int) -> None:  # pragma: no cover - replaced in every test
    import ctypes

    ctypes.windll.kernel32.SetThreadExecutionState(flags)  # type: ignore[attr-defined,unused-ignore]


@contextmanager
def keep_awake() -> Iterator[None]:
    """Keep the system from sleeping for the duration of the block (Windows; no-op elsewhere)."""
    if not _WINDOWS:
        yield
        return
    _set_execution_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        yield
    finally:
        _set_execution_state(ES_CONTINUOUS)
