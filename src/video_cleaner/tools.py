"""External tool discovery and a subprocess runner that is never silent.

`locate_tools` finds ffmpeg, ffprobe and mkvmerge. `run_tool` runs one command (argument list,
never a shell), parses progress from its stdout, drains stderr concurrently, emits a heartbeat
while it runs and guarantees the process is gone when it returns or raises.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import subprocess  # nosec B404
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from video_cleaner import power
from video_cleaner.logging_setup import redact
from video_cleaner.progress import Progress, Reporter, parse_mkvmerge_progress

_log = logging.getLogger("video_cleaner.tools")

STDOUT_CAP_CHARS = 1_000_000
STDERR_TAIL_CHARS = 4_096
_TERMINATE_GRACE_S = 5.0
# `Popen.wait()` with no timeout blocks in one uninterruptible call on Windows, so Ctrl+C would
# only be seen when the child exits; waiting in short slices lets KeyboardInterrupt through.
_WAIT_SLICE_S = 0.2
_DEFAULT_JOIN_TIMEOUT_S = 5.0
_EXECUTABLE_SUFFIXES = ("", ".exe")

_ENV_VARS = {"ffmpeg": "VC_FFMPEG", "ffprobe": "VC_FFPROBE", "mkvmerge": "VC_MKVMERGE"}


class ToolNotFound(Exception):
    """A required external tool could not be located."""

    def __init__(self, name: str, detail: str = "") -> None:
        self.name = name
        self.detail = detail
        suffix = f": {detail}" if detail else ""
        super().__init__(f"tool not found: {name}{suffix}")


class ToolFailed(Exception):
    """A tool exited non-zero. *cmd_text* is the redacted command line."""

    def __init__(self, returncode: int, cmd_text: str, log_path: Path | None) -> None:
        self.returncode = returncode
        self.cmd_text = cmd_text
        self.log_path = log_path
        where = f"; stderr in {log_path}" if log_path is not None else ""
        super().__init__(f"command exited with code {returncode}: {cmd_text}{where}")


class ToolTimedOut(Exception):
    """A tool ran longer than the *timeout_s* given to `run_tool`; it has been stopped."""

    def __init__(self, label: str, timeout_s: float) -> None:
        self.label = label
        self.timeout_s = timeout_s
        super().__init__(f"{label}: stopped after {timeout_s:g}s without finishing")


class ToolOutputError(Exception):
    """Output handling for a tool run broke.

    Raised when the progress parser or reporter raised, a reader thread crashed, or a pipe
    stayed open after the process exited (a grandchild holding it). `__cause__` is the
    underlying exception; if the tool also exited non-zero it is the `ToolFailed`, whose own
    `__cause__` is the underlying exception.
    """

    def __init__(self, label: str, stream: str, detail: str) -> None:
        self.label = label
        self.stream = stream
        self.detail = detail
        super().__init__(f"{label}: {stream}: {detail}")


@dataclass(frozen=True)
class Tools:
    ffmpeg: Path
    ffprobe: Path
    mkvmerge: Path


@dataclass(frozen=True)
class ToolResult:
    returncode: int
    stdout: str
    stderr_tail: str


def _version_key(path: Path) -> tuple[int, ...]:
    return tuple(int(n) for n in re.findall(r"\d+", path.name))


def _in_dir(directory: Path, name: str) -> Path | None:
    for suffix in _EXECUTABLE_SUFFIXES:
        candidate = directory / f"{name}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _fallback_dirs(name: str, env: Mapping[str, str]) -> list[Path]:
    """Well-known install locations tried after PATH, best candidate first."""
    if name == "mkvmerge":
        program_files = env.get("ProgramFiles", r"C:\Program Files")
        return [Path(program_files) / "MKVToolNix"]
    local = env.get("LOCALAPPDATA")
    if not local:
        return []
    packages = Path(local) / "Microsoft" / "WinGet" / "Packages"
    builds = [p for p in packages.glob("Gyan.FFmpeg_*/ffmpeg-*/bin") if p.is_dir()]
    return sorted(builds, key=lambda p: _version_key(p.parent), reverse=True)


def locate_tool(name: str, env: Mapping[str, str]) -> Path:
    """Find one tool (``ffmpeg``, ``ffprobe`` or ``mkvmerge``); see `locate_tools`."""
    var = _ENV_VARS[name]
    explicit = env.get(var)
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return path.absolute()
        # An explicit override that points nowhere is a configuration error, not a hint.
        raise ToolNotFound(name, f"{var}={explicit} is not a file")
    for entry in env.get("PATH", "").split(os.pathsep):
        # A relative entry ("." or "bin") would depend on the current folder, where a planted
        # executable could be picked up: only absolute entries are searched.
        absolute = bool(entry) and Path(entry).is_absolute()
        if absolute and (found := _in_dir(Path(entry), name)) is not None:
            return found.absolute()
    for directory in _fallback_dirs(name, env):
        if (found := _in_dir(directory, name)) is not None:
            return found.absolute()
    raise ToolNotFound(name, f"not on PATH or in the default install locations; set {var}")


def locate_tools(env: Mapping[str, str] | None = None) -> Tools:
    """Find ffmpeg, ffprobe and mkvmerge.

    Order per tool: ``VC_FFMPEG`` / ``VC_FFPROBE`` / ``VC_MKVMERGE`` (full path), PATH, the
    WinGet ``Gyan.FFmpeg`` package ``bin`` directory (newest build) and
    ``%ProgramFiles%\\MKVToolNix``. Raises `ToolNotFound` naming the first missing tool.
    """
    source = os.environ if env is None else env
    return Tools(
        ffmpeg=locate_tool("ffmpeg", source),
        ffprobe=locate_tool("ffprobe", source),
        mkvmerge=locate_tool("mkvmerge", source),
    )


def mkvmerge_parser(lines: Iterable[str]) -> Iterator[Progress]:
    """Adapt `parse_mkvmerge_progress` to the `run_tool` parser signature (percent only)."""
    for line in lines:
        percent = parse_mkvmerge_progress(line)
        if percent is not None:
            yield Progress(percent=percent, fps=None, speed=None, eta_s=None)


class _Capture:
    """Base for a thread that drains one pipe to EOF no matter what goes wrong.

    The child must never block on a full pipe, so after any failure the stream is still
    drained. The first failure is recorded in `error` / `error_what` for `run_tool` to raise
    once the process has exited and the thread has finished. Buffer access is lock-protected,
    so a snapshot taken while the thread is still running is consistent.
    """

    stream_name = ""

    def __init__(self, stream: IO[str]) -> None:
        self._stream = stream
        self._lock = threading.Lock()
        self.error: Exception | None = None
        self.error_what = ""

    def _fail(self, what: str, exc: Exception) -> None:
        if self.error is None:
            self.error, self.error_what = exc, what
        _log.error("%s: %s", self.stream_name, what, exc_info=exc)

    def _read(self) -> None:
        raise NotImplementedError

    def run(self) -> None:
        try:
            self._read()
        except Exception as exc:  # a reader thread must never die silently
            self._fail("reader thread crashed", exc)
            try:
                for _ in self._stream:
                    pass
            except Exception:  # pragma: no cover - the stream itself is broken
                _log.exception("%s: draining after a crash failed", self.stream_name)


class _StdoutCapture(_Capture):
    """Reads stdout line by line, keeps the last ~1 MB, optionally feeds a progress parser."""

    stream_name = "stdout"

    def __init__(
        self,
        stream: IO[str],
        parser: Callable[[Iterable[str]], Iterator[Progress]] | None,
        reporter: Reporter,
    ) -> None:
        super().__init__(stream)
        self._parser = parser
        self._reporter = reporter
        self._lines: deque[str] = deque()
        self._size = 0

    def _tee(self) -> Iterator[str]:
        for line in self._stream:
            with self._lock:
                self._lines.append(line)
                self._size += len(line)
                while self._size > STDOUT_CAP_CHARS and len(self._lines) > 1:
                    self._size -= len(self._lines.popleft())
            yield line

    def _read(self) -> None:
        lines = self._tee()
        if self._parser is not None:
            try:
                for progress in self._parser(lines):
                    self._reporter.update(progress)
            except Exception as exc:
                self._fail("progress parser or reporter failed", exc)
        for _ in lines:  # drain whatever the parser did not consume
            pass

    @property
    def text(self) -> str:
        with self._lock:
            return "".join(self._lines)


class _StderrCapture(_Capture):
    """Drains stderr: keeps the last ~4 KB and streams every line to *log_path* if given.

    Every line is redacted before it is kept or written, so a credential a tool echoes never
    reaches the kept log or the returned tail.
    """

    stream_name = "stderr"

    def __init__(self, stream: IO[str], log_path: Path | None) -> None:
        super().__init__(stream)
        self._log_path = log_path
        self._tail: deque[str] = deque()
        self._size = 0

    def _open_log(self) -> IO[str] | None:
        if self._log_path is None:
            return None
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            return self._log_path.open("w", encoding="utf-8", errors="replace")
        except OSError:
            _log.exception("cannot write stderr log %s", self._log_path)
            return None

    def _read(self) -> None:
        sink = self._open_log()
        try:
            for raw in self._stream:
                line = redact(raw)
                with self._lock:
                    self._tail.append(line)
                    self._size += len(line)
                    while self._size > STDERR_TAIL_CHARS and len(self._tail) > 1:
                        self._size -= len(self._tail.popleft())
                if sink is not None:
                    try:
                        sink.write(line)
                    except OSError:
                        _log.exception("stderr log write failed; disabling it")
                        sink = None
        finally:
            if sink is not None:
                sink.close()

    @property
    def text(self) -> str:
        with self._lock:
            return "".join(self._tail)[-STDERR_TAIL_CHARS:]


def _heartbeat_loop(
    stop: threading.Event, reporter: Reporter, label: str, interval_s: float, started: float
) -> None:
    # Sent every interval regardless of parsed progress: rate-limited reporters may print
    # `update` less often than the heartbeat, and the console must never go quiet.
    while not stop.wait(interval_s):
        try:
            reporter.heartbeat(label, time.monotonic() - started)
        except Exception:
            _log.exception("heartbeat failed")


@contextlib.contextmanager
def heartbeat(
    reporter: Reporter,
    label: str,
    interval_s: float,
    *,
    started: float | None = None,
    join_timeout_s: float = _DEFAULT_JOIN_TIMEOUT_S,
) -> Iterator[None]:
    """Call ``reporter.heartbeat(label, elapsed)`` every *interval_s* while the block runs.

    *started* is the `time.monotonic` reading the elapsed time counts from (the block's start
    by default). The beating thread is stopped when the block ends or raises, so nothing beats
    afterwards. A reporter that raises is logged and the beat goes on.
    """
    stop = threading.Event()
    beat = threading.Thread(
        target=_heartbeat_loop,
        args=(stop, reporter, label, interval_s, time.monotonic() if started is None else started),
        name="vc-heartbeat",
        daemon=True,
    )
    beat.start()
    try:
        yield
    finally:
        stop.set()
        beat.join(timeout=join_timeout_s)


def _wait_interruptibly(proc: subprocess.Popen[str], label: str, timeout_s: float | None) -> int:
    """`proc.wait()` in slices of `_WAIT_SLICE_S`, so a KeyboardInterrupt is never stuck.

    Raises `ToolTimedOut` once *timeout_s* seconds have passed (never, if it is None); the
    caller stops the process.
    """
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    while True:
        try:
            return proc.wait(timeout=_WAIT_SLICE_S)
        except subprocess.TimeoutExpired:
            if timeout_s is not None and deadline is not None and time.monotonic() >= deadline:
                raise ToolTimedOut(label, timeout_s) from None


def _stop_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=_TERMINATE_GRACE_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _close_pipes_when_readers_finish(
    proc: subprocess.Popen[str], readers: Sequence[threading.Thread]
) -> None:
    """Close the pipes once the stuck readers see EOF.

    Closing a pipe another thread is blocked reading can itself block, so it is deferred to
    a daemon thread that waits for the readers; nothing leaks once the holder of the pipe exits.
    """

    def reap() -> None:
        for thread in readers:
            thread.join()
        for pipe in (proc.stdout, proc.stderr):
            if pipe is not None:
                pipe.close()

    threading.Thread(target=reap, name="vc-reaper", daemon=True).start()


def _surface_output_errors(
    label: str,
    captures: Sequence[_Capture],
    stuck: Sequence[str],
    join_timeout_s: float,
    failed: ToolFailed | None,
) -> None:
    """Raise `ToolOutputError` for the first reader problem, chained as documented there."""
    problem: tuple[str, str, Exception | None] | None = None
    for capture in captures:
        if capture.error is not None:
            problem = (capture.stream_name, capture.error_what, capture.error)
            break
    if problem is None and stuck:
        detail = (
            f"pipe still open {join_timeout_s:g}s after the process exited "
            "(a child process may still hold it)"
        )
        problem = (stuck[0], detail, None)
    if problem is None:
        return
    stream, detail, cause = problem
    if cause is not None and failed is not None:
        failed.__cause__ = cause
    raise ToolOutputError(label, stream, detail) from (failed or cause)


def run_tool(
    cmd: Sequence[str],
    *,
    label: str,
    reporter: Reporter,
    heartbeat_s: float = 15.0,
    parser: Callable[[Iterable[str]], Iterator[Progress]] | None = None,
    stderr_log: Path | None = None,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
    join_timeout_s: float = _DEFAULT_JOIN_TIMEOUT_S,
    timeout_s: float | None = None,
) -> ToolResult:
    """Run *cmd* (an argument list, never a shell) and return its result.

    stdout is read by one thread (fed to *parser*, whose `Progress` items go to
    ``reporter.update``; the last ~1 MB is returned), stderr by another (last ~4 KB returned,
    everything written to *stderr_log*), and a third calls ``reporter.heartbeat(label, elapsed)``
    every *heartbeat_s* seconds. The child is terminated, then killed, if this call is
    interrupted or fails.

    *env* is the child's whole environment; None (the default) inherits this process's. It is
    copied, and this process's environment is never changed. With *timeout_s* the child is
    stopped (terminated, then killed) once it has run that long and `ToolTimedOut` is raised;
    None waits for as long as the tool takes.

    Raises `ToolFailed` on a non-zero exit when *check* is true, `ToolTimedOut` when *timeout_s*
    passes, `ToolNotFound` if the executable does not exist, and `ToolOutputError` if the
    parser, the reporter or a reader thread failed, or a reader had not finished
    *join_timeout_s* after the process exited (typically a grandchild holding the pipe). Output
    errors are raised after the process has exited and the readers are done; the pipes are always
    drained meanwhile.
    """
    cmd_text = redact(subprocess.list2cmdline([str(part) for part in cmd]))
    _log.debug("run [%s]: %s", label, cmd_text)
    started = time.monotonic()
    try:
        # Argument list, shell=False.
        proc = subprocess.Popen(  # noqa: S603  # nosec B603
            [str(part) for part in cmd],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            cwd=cwd,
            env=None if env is None else dict(env),
            shell=False,
            creationflags=power.creation_flags(),  # below normal: the PC stays usable
        )
    except FileNotFoundError as exc:
        raise ToolNotFound(str(cmd[0]), "executable does not exist") from exc
    power.lower_priority(proc.pid)

    if proc.stdout is None or proc.stderr is None:  # pragma: no cover - PIPE always sets them
        raise RuntimeError("subprocess pipes were not created")
    out = _StdoutCapture(proc.stdout, parser, reporter)
    err = _StderrCapture(proc.stderr, stderr_log)
    readers = [
        threading.Thread(target=out.run, name="vc-stdout", daemon=True),
        threading.Thread(target=err.run, name="vc-stderr", daemon=True),
    ]
    stuck: list[str] = []
    try:
        for thread in readers:
            thread.start()
        with heartbeat(
            reporter, label, heartbeat_s, started=started, join_timeout_s=join_timeout_s
        ):
            returncode = _wait_interruptibly(proc, label, timeout_s)
    except BaseException:
        _stop_process(proc)
        raise
    finally:
        # One shared deadline, so two stuck readers cost one timeout, not two.
        deadline = time.monotonic() + join_timeout_s
        for thread, capture in zip(readers, (out, err), strict=True):
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                stuck.append(capture.stream_name)
        if stuck:
            _close_pipes_when_readers_finish(proc, readers)
        else:
            proc.stdout.close()
            proc.stderr.close()

    duration = time.monotonic() - started
    _log.debug("exit [%s]: code %d after %.1fs", label, returncode, duration)
    failed = ToolFailed(returncode, cmd_text, stderr_log) if check and returncode != 0 else None
    _surface_output_errors(label, (out, err), stuck, join_timeout_s, failed)
    if failed is not None:
        raise failed
    return ToolResult(returncode=returncode, stdout=out.text, stderr_tail=err.text)
