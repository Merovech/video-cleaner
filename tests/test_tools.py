from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pytest

from video_cleaner import tools
from video_cleaner.progress import Progress, parse_ffmpeg_progress
from video_cleaner.tools import (
    ToolFailed,
    ToolNotFound,
    ToolOutputError,
    ToolResult,
    Tools,
    locate_tools,
    mkvmerge_parser,
    run_tool,
)

PY = sys.executable


class FakeReporter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.heartbeats: list[tuple[str, float]] = []
        self.updates: list[Progress] = []

    def stage(self, label: str) -> None:  # pragma: no cover - unused by run_tool
        pass

    def update(self, p: Progress) -> None:
        with self._lock:
            self.updates.append(p)

    def heartbeat(self, label: str, elapsed_s: float) -> None:
        with self._lock:
            self.heartbeats.append((label, elapsed_s))

    def file_done(self, summary: str) -> None:  # pragma: no cover - unused by run_tool
        pass


def _py(code: str, *args: str) -> list[str]:
    return [PY, "-c", code, *args]


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    path.chmod(0o755)
    return path


# --- run_tool -------------------------------------------------------------------------------


def test_run_tool_never_silent_longer_than_heartbeat() -> None:
    reporter = FakeReporter()
    result = run_tool(
        _py("import time; time.sleep(3)"), label="sleeper", reporter=reporter, heartbeat_s=0.5
    )
    assert result.returncode == 0
    assert len(reporter.heartbeats) >= 5
    assert all(label == "sleeper" for label, _ in reporter.heartbeats)
    elapsed = [e for _, e in reporter.heartbeats]
    assert elapsed == sorted(elapsed)


def test_run_tool_decodes_utf8_output() -> None:
    code = (
        "import sys; "
        "sys.stdout.buffer.write('h\\u00e9llo \\u2603 \\u65e5\\u672c\\n'.encode('utf-8'))"
    )
    result = run_tool(_py(code), label="utf8", reporter=FakeReporter())
    assert result.stdout == "h\u00e9llo \u2603 \u65e5\u672c\n"


def test_run_tool_decodes_utf8_stderr_tail() -> None:
    code = "import sys; sys.stderr.buffer.write('caf\\u00e9\\n'.encode('utf-8')); sys.exit(0)"
    result = run_tool(_py(code), label="utf8err", reporter=FakeReporter())
    assert "caf\u00e9" in result.stderr_tail


def test_run_tool_failure_keeps_stderr_in_log_and_raises(tmp_path: Path) -> None:
    log = tmp_path / "logs" / "stderr.log"
    code = "import sys; sys.stderr.write('boom: bad input\\n'); sys.exit(3)"
    with pytest.raises(ToolFailed) as excinfo:
        run_tool(_py(code), label="failing", reporter=FakeReporter(), stderr_log=log)
    assert excinfo.value.returncode == 3
    assert excinfo.value.log_path == log
    assert PY in excinfo.value.cmd_text
    assert "boom: bad input" in log.read_text(encoding="utf-8")


def test_run_tool_failure_without_log_path_has_none() -> None:
    with pytest.raises(ToolFailed) as excinfo:
        run_tool(_py("import sys; sys.exit(2)"), label="f", reporter=FakeReporter())
    assert excinfo.value.returncode == 2
    assert excinfo.value.log_path is None


def test_run_tool_check_false_returns_nonzero_result() -> None:
    code = "import sys; sys.stderr.write('oops'); sys.exit(5)"
    result = run_tool(_py(code), label="f", reporter=FakeReporter(), check=False)
    assert isinstance(result, ToolResult)
    assert result.returncode == 5
    assert "oops" in result.stderr_tail


def test_run_tool_never_uses_shell() -> None:
    tricky = "C:\\tmp dir\\a & b ; echo pwned.mkv"
    result = run_tool(
        _py("import sys; print(len(sys.argv)); print(sys.argv[1])", tricky),
        label="args",
        reporter=FakeReporter(),
    )
    assert result.stdout.splitlines() == ["2", tricky]


def test_run_tool_missing_executable_raises_tool_not_found(tmp_path: Path) -> None:
    with pytest.raises(ToolNotFound):
        run_tool([str(tmp_path / "does-not-exist.exe")], label="x", reporter=FakeReporter())


def test_run_tool_feeds_parser_and_reports_updates() -> None:
    code = (
        "print('fps=30.0'); print('out_time_us=30000000'); print('speed=2.0x');"
        "print('progress=continue');"
        "print('fps=30.0'); print('out_time_us=60000000'); print('speed=2.0x');"
        "print('progress=end')"
    )
    reporter = FakeReporter()

    def parser(lines: Iterable[str]) -> Iterator[Progress]:
        return parse_ffmpeg_progress(lines, duration_s=60.0)

    result = run_tool(_py(code), label="enc", reporter=reporter, parser=parser)
    assert [round(p.percent or 0) for p in reporter.updates] == [50, 100]
    assert reporter.updates[-1].note == "end"
    # Parsed lines are still returned as captured stdout.
    assert "progress=end" in result.stdout


def test_mkvmerge_parser_yields_percent_only() -> None:
    lines = [
        "#GUI#begin_scanning_playlists\n",
        "#GUI#progress 37%\n",
        "noise\n",
        "#GUI#progress 100%",
    ]
    assert list(mkvmerge_parser(lines)) == [
        Progress(percent=37.0, fps=None, speed=None, eta_s=None),
        Progress(percent=100.0, fps=None, speed=None, eta_s=None),
    ]


def test_run_tool_with_mkvmerge_parser() -> None:
    reporter = FakeReporter()
    code = "print('#GUI#progress 10%'); print('#GUI#progress 90%')"
    run_tool(_py(code), label="mux", reporter=reporter, parser=mkvmerge_parser)
    assert [p.percent for p in reporter.updates] == [10.0, 90.0]


_BIG_STDOUT = "for _ in range(20000): print('y' * 100)\n"  # far more than a pipe buffer


def _bad_parser(lines: Iterable[str]) -> Iterator[Progress]:
    for _ in lines:
        raise ValueError("parser bug")
    yield from ()  # pragma: no cover


def test_run_tool_parser_error_is_surfaced_after_child_finishes() -> None:
    # The child writes far more than a pipe buffer: if the reader stopped draining after the
    # parser failed, the child would block forever and this test would hang.
    with pytest.raises(ToolOutputError) as excinfo:
        run_tool(_py(_BIG_STDOUT), label="p", reporter=FakeReporter(), parser=_bad_parser)
    assert excinfo.value.label == "p"
    assert excinfo.value.stream == "stdout"
    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "parser bug" in str(excinfo.value.__cause__)


def test_run_tool_reporter_update_error_is_surfaced() -> None:
    class BrokenReporter(FakeReporter):
        def update(self, p: Progress) -> None:
            raise RuntimeError("reporter bug")

    code = "print('#GUI#progress 10%'); print('#GUI#progress 20%')\n" + _BIG_STDOUT
    with pytest.raises(ToolOutputError) as excinfo:
        run_tool(_py(code), label="mux", reporter=BrokenReporter(), parser=mkvmerge_parser)
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_run_tool_parser_error_is_chained_onto_tool_failed() -> None:
    code = _BIG_STDOUT + "import sys; sys.exit(4)\n"
    with pytest.raises(ToolOutputError) as excinfo:
        run_tool(_py(code), label="p", reporter=FakeReporter(), parser=_bad_parser)
    failed = excinfo.value.__cause__
    assert isinstance(failed, ToolFailed)
    assert failed.returncode == 4
    assert isinstance(failed.__cause__, ValueError)


def test_run_tool_parser_error_surfaces_even_with_check_false() -> None:
    with pytest.raises(ToolOutputError):
        run_tool(
            _py("print('x')"),
            label="p",
            reporter=FakeReporter(),
            parser=_bad_parser,
            check=False,
        )


def test_run_tool_reader_thread_crash_is_surfaced(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_read(self: object) -> None:
        raise RuntimeError("reader bug")

    monkeypatch.setattr(tools._StderrCapture, "_read", broken_read)
    code = "import sys\nfor _ in range(20000): sys.stderr.write('e' * 99 + '\\n')\n"
    with pytest.raises(ToolOutputError) as excinfo:
        run_tool(_py(code), label="enc", reporter=FakeReporter())
    assert excinfo.value.stream == "stderr"
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_run_tool_reader_that_outlives_the_process_is_an_error(tmp_path: Path) -> None:
    # The child exits at once but leaves a long-lived grandchild that inherits its stdout and
    # stderr, so the pipes never reach EOF and the reader threads cannot finish.
    pid_file = tmp_path / "grandchild.pid"
    code = (
        "import subprocess, sys\n"
        "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(g.pid))\n"
    )
    started = time.monotonic()
    try:
        with pytest.raises(ToolOutputError) as excinfo:
            run_tool(
                _py(code),
                label="leaky",
                reporter=FakeReporter(),
                heartbeat_s=0.2,
                join_timeout_s=1.0,
            )
        elapsed = time.monotonic() - started
    finally:
        pid = int(pid_file.read_text()) if pid_file.exists() else None
        if pid is not None:
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGTERM)
        _wait_for_reader_threads()
    assert elapsed < 4.0  # one shared deadline, not one timeout per thread
    assert excinfo.value.label == "leaky"
    assert excinfo.value.stream in {"stdout", "stderr"}
    assert "still open" in str(excinfo.value)


def _wait_for_reader_threads(timeout_s: float = 10.0) -> None:
    """Let reader threads and the pipe-closing reaper finish once the grandchild is gone."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not [t for t in threading.enumerate() if t.name.startswith("vc-")]:
            return
        time.sleep(0.05)


def test_run_tool_drains_large_stdout_and_stderr_without_deadlock(tmp_path: Path) -> None:
    code = (
        "import sys\n"
        "for i in range(20000):\n"
        "    sys.stdout.write('o' * 99 + '\\n'); sys.stderr.write('e' * 99 + '\\n')\n"
    )
    log = tmp_path / "err.log"
    result = run_tool(_py(code), label="big", reporter=FakeReporter(), stderr_log=log)
    assert result.returncode == 0
    # stdout is capped to the last ~1 MB, stderr_tail to the last ~4 KB.
    assert 0 < len(result.stdout) <= tools.STDOUT_CAP_CHARS
    assert result.stdout.endswith("o" * 99 + "\n")
    assert 0 < len(result.stderr_tail) <= tools.STDERR_TAIL_CHARS
    # The full stderr is in the log file.
    assert log.stat().st_size >= 20000 * 100


def test_run_tool_passes_cwd(tmp_path: Path) -> None:
    result = run_tool(
        _py("import os; print(os.getcwd())"), label="cwd", reporter=FakeReporter(), cwd=tmp_path
    )
    assert Path(result.stdout.strip()).resolve() == tmp_path.resolve()


def test_run_tool_logs_redacted_command_exit_code_and_duration() -> None:
    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger("video_cleaner.tools")
    handler = Capture(level=logging.DEBUG)
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        run_tool(_py("pass", "--api-key=SUPERSECRET123"), label="log", reporter=FakeReporter())
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    messages = [r.getMessage() for r in records]
    assert not any("SUPERSECRET123" in m for m in messages)
    assert any("api-key=***" in m for m in messages)
    assert any(r.levelno == logging.DEBUG for r in records)
    assert any("exit" in m and "0" in m for m in messages)


def test_run_tool_command_log_is_redacted_through_logging_filter(tmp_path: Path) -> None:
    from video_cleaner.logging_setup import RedactingFilter

    out: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            out.append(self.format(record))

    handler = Capture(level=logging.DEBUG)
    handler.addFilter(RedactingFilter(["SUPERSECRET123"]))
    logger = logging.getLogger("video_cleaner.tools")
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        run_tool(_py("pass", "SUPERSECRET123"), label="log", reporter=FakeReporter())
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    assert out
    assert not any("SUPERSECRET123" in line for line in out)


def test_run_tool_failure_message_redacts_command() -> None:
    with pytest.raises(ToolFailed) as excinfo:
        run_tool(
            _py("import sys; sys.exit(1)", "--password=hunter2hunter2"),
            label="f",
            reporter=FakeReporter(),
        )
    assert "hunter2hunter2" not in excinfo.value.cmd_text
    assert "hunter2hunter2" not in str(excinfo.value)


def test_run_tool_kills_process_on_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[Any] = []

    class InterruptingPopen(subprocess.Popen[str]):
        _first = True

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            started.append(self)

        def wait(self, timeout: float | None = None) -> int:
            if InterruptingPopen._first:
                InterruptingPopen._first = False
                raise KeyboardInterrupt
            return super().wait(timeout)

    monkeypatch.setattr(subprocess, "Popen", InterruptingPopen)
    with pytest.raises(KeyboardInterrupt):
        run_tool(
            _py("import time; time.sleep(60)"),
            label="sleeper",
            reporter=FakeReporter(),
            heartbeat_s=0.2,
        )
    assert len(started) == 1
    assert started[0].poll() is not None  # no orphan left running


def test_run_tool_stops_heartbeat_thread_after_return() -> None:
    before = {t.name for t in threading.enumerate()}
    run_tool(_py("pass"), label="x", reporter=FakeReporter(), heartbeat_s=0.1)
    leftover = [
        t for t in threading.enumerate() if t.name not in before and t.name.startswith("vc-")
    ]
    assert leftover == []


# --- locate_tools ---------------------------------------------------------------------------


def _fake_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    base = {
        "PATH": str(tmp_path / "emptybin"),
        "PATHEXT": ".EXE;.CMD",
        "LOCALAPPDATA": str(tmp_path / "localappdata"),
        "ProgramFiles": str(tmp_path / "programfiles"),
    }
    (tmp_path / "emptybin").mkdir(exist_ok=True)
    base.update(extra)
    return base


def test_locate_tools_raises_for_missing(tmp_path: Path) -> None:
    with pytest.raises(ToolNotFound) as excinfo:
        locate_tools(_fake_env(tmp_path))
    assert excinfo.value.name == "ffmpeg"


def test_locate_tools_explicit_env_vars_win_over_path(tmp_path: Path) -> None:
    pathdir = tmp_path / "bin"
    for n in ("ffmpeg", "ffprobe", "mkvmerge"):
        _touch(pathdir / f"{n}.exe")
    explicit = {n: _touch(tmp_path / "explicit" / f"{n}.exe") for n in ("ffmpeg", "ffprobe")}
    mkv = _touch(tmp_path / "explicit" / "mkvmerge.exe")
    env = _fake_env(
        tmp_path,
        PATH=str(pathdir),
        VC_FFMPEG=str(explicit["ffmpeg"]),
        VC_FFPROBE=str(explicit["ffprobe"]),
        VC_MKVMERGE=str(mkv),
    )
    found = locate_tools(env)
    assert found == Tools(ffmpeg=explicit["ffmpeg"], ffprobe=explicit["ffprobe"], mkvmerge=mkv)


def test_locate_tools_explicit_env_var_pointing_nowhere_raises(tmp_path: Path) -> None:
    env = _fake_env(tmp_path, VC_FFMPEG=str(tmp_path / "nope" / "ffmpeg.exe"))
    with pytest.raises(ToolNotFound) as excinfo:
        locate_tools(env)
    assert excinfo.value.name == "ffmpeg"
    assert "VC_FFMPEG" in str(excinfo.value)


def test_locate_tools_uses_path(tmp_path: Path) -> None:
    pathdir = tmp_path / "bin"
    exes = {n: _touch(pathdir / f"{n}.exe") for n in ("ffmpeg", "ffprobe", "mkvmerge")}
    found = locate_tools(_fake_env(tmp_path, PATH=str(pathdir)))
    assert found.ffmpeg.samefile(exes["ffmpeg"])
    assert found.ffprobe.samefile(exes["ffprobe"])
    assert found.mkvmerge.samefile(exes["mkvmerge"])


def test_locate_tools_falls_back_to_winget_and_program_files(tmp_path: Path) -> None:
    winget_bin = (
        tmp_path
        / "localappdata"
        / "Microsoft"
        / "WinGet"
        / "Packages"
        / "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
        / "ffmpeg-9.0.2-full_build"
        / "bin"
    )
    ffmpeg = _touch(winget_bin / "ffmpeg.exe")
    ffprobe = _touch(winget_bin / "ffprobe.exe")
    mkv = _touch(tmp_path / "programfiles" / "MKVToolNix" / "mkvmerge.exe")
    assert locate_tools(_fake_env(tmp_path)) == Tools(ffmpeg=ffmpeg, ffprobe=ffprobe, mkvmerge=mkv)


def test_locate_tools_picks_newest_winget_build(tmp_path: Path) -> None:
    packages = tmp_path / "localappdata" / "Microsoft" / "WinGet" / "Packages" / "Gyan.FFmpeg_x"
    _touch(packages / "ffmpeg-8.0-full_build" / "bin" / "ffmpeg.exe")
    _touch(packages / "ffmpeg-8.0-full_build" / "bin" / "ffprobe.exe")
    new_ffmpeg = _touch(packages / "ffmpeg-9.0.2-full_build" / "bin" / "ffmpeg.exe")
    _touch(packages / "ffmpeg-9.0.2-full_build" / "bin" / "ffprobe.exe")
    _touch(tmp_path / "programfiles" / "MKVToolNix" / "mkvmerge.exe")
    assert locate_tools(_fake_env(tmp_path)).ffmpeg == new_ffmpeg


def test_locate_tools_names_the_first_missing_tool(tmp_path: Path) -> None:
    pathdir = tmp_path / "bin"
    _touch(pathdir / "ffmpeg.exe")
    _touch(pathdir / "ffprobe.exe")
    with pytest.raises(ToolNotFound) as excinfo:
        locate_tools(_fake_env(tmp_path, PATH=str(pathdir)))
    assert excinfo.value.name == "mkvmerge"


# --- integration (real machine) -------------------------------------------------------------

_REAL_FFMPEG_BIN = Path(
    "C:/Users/user/AppData/Local/Microsoft/WinGet/Packages/"
    "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.2-full_build/bin"
)
_REAL_MKVMERGE = Path("C:/Program Files/MKVToolNix/mkvmerge.exe")


@pytest.mark.integration
def test_integration_locate_and_run_real_tools() -> None:
    if not (_REAL_FFMPEG_BIN / "ffmpeg.exe").exists() or not _REAL_MKVMERGE.exists():
        pytest.skip("real ffmpeg (WinGet Gyan.FFmpeg 9.0.2) or MKVToolNix not installed here")
    env = dict(os.environ)
    env["VC_FFMPEG"] = str(_REAL_FFMPEG_BIN / "ffmpeg.exe")
    env["VC_FFPROBE"] = str(_REAL_FFMPEG_BIN / "ffprobe.exe")
    env["VC_MKVMERGE"] = str(_REAL_MKVMERGE)
    found = locate_tools(env)
    reporter = FakeReporter()
    out = run_tool(
        [str(found.ffmpeg), "-hide_banner", "-version"], label="ffmpeg", reporter=reporter
    )
    assert out.stdout.startswith("ffmpeg version")
    out = run_tool([str(found.mkvmerge), "--version"], label="mkvmerge", reporter=reporter)
    assert "mkvmerge" in out.stdout


def test_locate_tools_skips_relative_path_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A relative PATH entry ("." or "bin") would resolve against whatever folder the command
    # runs in: a planted ffmpeg.exe there must never be picked.
    monkeypatch.chdir(tmp_path)
    for n in ("ffmpeg", "ffprobe", "mkvmerge"):
        _touch(tmp_path / "bin" / f"{n}.exe")
    with pytest.raises(ToolNotFound):
        locate_tools(_fake_env(tmp_path, PATH=os.pathsep.join(["bin", "."])))


def test_located_tools_are_absolute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    for n in ("ffmpeg", "ffprobe", "mkvmerge"):
        _touch(tmp_path / "x" / f"{n}.exe")
    env = _fake_env(
        tmp_path,
        VC_FFMPEG=os.path.join("x", "ffmpeg.exe"),  # relative override: made absolute
        PATH=str(tmp_path / "x"),
    )
    found = locate_tools(env)
    assert all(p.is_absolute() for p in (found.ffmpeg, found.ffprobe, found.mkvmerge))
    assert found.ffmpeg == tmp_path / "x" / "ffmpeg.exe"


def test_locate_tool_is_public(tmp_path: Path) -> None:
    from video_cleaner.tools import locate_tool

    exe = _touch(tmp_path / "bin" / "ffmpeg.exe")
    assert locate_tool("ffmpeg", _fake_env(tmp_path, PATH=str(tmp_path / "bin"))) == exe
