"""External tool discovery and a subprocess runner that is never silent (spec section 4.4).

`locate_tools` finds ffmpeg, ffprobe and mkvmerge. `run_tool` runs one command (argument list,
never a shell), parses progress from its stdout, drains stderr concurrently, emits a heartbeat
while it runs and guarantees the process is gone when it returns or raises.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from video_cleaner.logging_setup import redact
from video_cleaner.progress import Progress, Reporter, parse_mkvmerge_progress

_log = logging.getLogger("video_cleaner.tools")

STDOUT_CAP_CHARS = 1_000_000
STDERR_TAIL_CHARS = 4_096
_TERMINATE_GRACE_S = 5.0
_THREAD_JOIN_S = 5.0
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


def _locate(name: str, env: Mapping[str, str]) -> Path:
    var = _ENV_VARS[name]
    explicit = env.get(var)
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return path
        # An explicit override that points nowhere is a configuration error, not a hint.
        raise ToolNotFound(name, f"{var}={explicit} is not a file")
    for entry in env.get("PATH", "").split(os.pathsep):
        if entry and (found := _in_dir(Path(entry), name)) is not None:
            return found
    for directory in _fallback_dirs(name, env):
        if (found := _in_dir(directory, name)) is not None:
            return found
    raise ToolNotFound(name, f"not on PATH or in the default install locations; set {var}")


def locate_tools(env: Mapping[str, str] | None = None) -> Tools:
    """Find ffmpeg, ffprobe and mkvmerge.

    Order per tool: ``VC_FFMPEG`` / ``VC_FFPROBE`` / ``VC_MKVMERGE`` (full path), PATH, the
    WinGet ``Gyan.FFmpeg`` package ``bin`` directory (newest build) and
    ``%ProgramFiles%\\MKVToolNix``. Raises `ToolNotFound` naming the first missing tool.
    """
    source = os.environ if env is None else env
    return Tools(
        ffmpeg=_locate("ffmpeg", source),
        ffprobe=_locate("ffprobe", source),
        mkvmerge=_locate("mkvmerge", source),
    )


def mkvmerge_parser(lines: Iterable[str]) -> Iterator[Progress]:
    """Adapt `parse_mkvmerge_progress` to the `run_tool` parser signature (percent only)."""
    for line in lines:
        percent = parse_mkvmerge_progress(line)
        if percent is not None:
            yield Progress(percent=percent, fps=None, speed=None, eta_s=None)


class _StdoutCapture:
    """Reads stdout line by line, keeps the last ~1 MB, optionally feeds a progress parser."""

    def __init__(
        self,
        stream: IO[str],
        parser: Callable[[Iterable[str]], Iterator[Progress]] | None,
        reporter: Reporter,
    ) -> None:
        self._stream = stream
        self._parser = parser
        self._reporter = reporter
        self._lines: deque[str] = deque()
        self._size = 0

    def _tee(self) -> Iterator[str]:
        for line in self._stream:
            self._lines.append(line)
            self._size += len(line)
            while self._size > STDOUT_CAP_CHARS and len(self._lines) > 1:
                self._size -= len(self._lines.popleft())
            yield line

    def run(self) -> None:
        lines = self._tee()
        if self._parser is not None:
            try:
                for progress in self._parser(lines):
                    self._reporter.update(progress)
            except Exception:
                # A parser or reporter bug must not stop the pipe being drained (the child
                # would block on a full pipe); the run itself is judged by the exit code.
                _log.exception("progress handling failed; continuing to drain output")
        for _ in lines:  # drain whatever the parser did not consume
            pass

    @property
    def text(self) -> str:
        return "".join(self._lines)


class _StderrCapture:
    """Drains stderr: keeps the last ~4 KB and streams every line to *log_path* if given."""

    def __init__(self, stream: IO[str], log_path: Path | None) -> None:
        self._stream = stream
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

    def run(self) -> None:
        sink = self._open_log()
        try:
            for line in self._stream:
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


def _stop_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=_TERMINATE_GRACE_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def run_tool(
    cmd: Sequence[str],
    *,
    label: str,
    reporter: Reporter,
    heartbeat_s: float = 15.0,
    parser: Callable[[Iterable[str]], Iterator[Progress]] | None = None,
    stderr_log: Path | None = None,
    cwd: Path | None = None,
    check: bool = True,
) -> ToolResult:
    """Run *cmd* (an argument list, never a shell) and return its result.

    stdout is read by one thread (fed to *parser*, whose `Progress` items go to
    ``reporter.update``; the last ~1 MB is returned), stderr by another (last ~4 KB returned,
    everything written to *stderr_log*), and a third calls ``reporter.heartbeat(label, elapsed)``
    every *heartbeat_s* seconds. The child is terminated, then killed, if this call is
    interrupted or fails. Raises `ToolFailed` on a non-zero exit when *check* is true and
    `ToolNotFound` if the executable does not exist.
    """
    cmd_text = redact(subprocess.list2cmdline([str(part) for part in cmd]))
    _log.debug("run [%s]: %s", label, cmd_text)
    started = time.monotonic()
    try:
        proc = subprocess.Popen(  # noqa: S603 - argument list, shell=False, by design
            [str(part) for part in cmd],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            cwd=cwd,
            shell=False,
        )
    except FileNotFoundError as exc:
        raise ToolNotFound(str(cmd[0]), "executable does not exist") from exc

    if proc.stdout is None or proc.stderr is None:  # pragma: no cover - PIPE always sets them
        raise RuntimeError("subprocess pipes were not created")
    out = _StdoutCapture(proc.stdout, parser, reporter)
    err = _StderrCapture(proc.stderr, stderr_log)
    stop = threading.Event()
    threads = [
        threading.Thread(target=out.run, name="vc-stdout", daemon=True),
        threading.Thread(target=err.run, name="vc-stderr", daemon=True),
        threading.Thread(
            target=_heartbeat_loop,
            args=(stop, reporter, label, heartbeat_s, started),
            name="vc-heartbeat",
            daemon=True,
        ),
    ]
    try:
        for thread in threads:
            thread.start()
        returncode = proc.wait()
    except BaseException:
        _stop_process(proc)
        raise
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=_THREAD_JOIN_S)
        proc.stdout.close()
        proc.stderr.close()

    duration = time.monotonic() - started
    _log.debug("exit [%s]: code %d after %.1fs", label, returncode, duration)
    if check and returncode != 0:
        raise ToolFailed(returncode, cmd_text, stderr_log)
    return ToolResult(returncode=returncode, stdout=out.text, stderr_tail=err.text)
