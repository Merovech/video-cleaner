"""Progress parsing and console reporting.

The tool is never silent: percentages where a tool reports them, a heartbeat where it does
not, an in-place `rich` display on a TTY and rate-limited plain lines otherwise.
"""

from __future__ import annotations

import re
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol, TextIO

from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    SpinnerColumn,
    TaskID,
    TextColumn,
)
from rich.progress import Progress as RichProgress

from video_cleaner.config import Settings
from video_cleaner.consolefmt import (
    Level,
    clear_active_console,
    encodable,
    set_active_console,
    tag,
    tagged,
)


@dataclass(frozen=True)
class Progress:
    percent: float | None
    fps: float | None
    speed: float | None
    eta_s: float | None
    note: str = ""


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value.strip())
    except ValueError:
        return None


def _parse_speed(value: str | None) -> float | None:
    if value is None:
        return None
    return _to_float(value.strip().removesuffix("x"))


def _out_time_seconds(fields: dict[str, str]) -> float | None:
    # ffmpeg quirk: out_time_ms is also in microseconds.
    for key in ("out_time_us", "out_time_ms"):
        micros = _to_float(fields.get(key))
        if micros is not None and micros >= 0:
            return micros / 1_000_000
    return None


def _progress_from_fields(fields: dict[str, str], duration_s: float) -> Progress:
    out_s = _out_time_seconds(fields)
    speed = _parse_speed(fields.get("speed"))
    percent: float | None = None
    eta: float | None = None
    if out_s is not None and duration_s > 0:
        percent = min(100.0, max(0.0, out_s / duration_s * 100.0))
        if speed is not None and speed > 0:
            eta = max(0.0, (duration_s - out_s) / speed)
    note = "end" if fields.get("progress") == "end" else ""
    if note and percent is not None:
        # The last out_time can fall short of the nominal duration (a filter graph that drops
        # the final partial frame, say); a finished tool is at 100%, whatever it last reported.
        percent, eta = 100.0, 0.0
    return Progress(
        percent=percent, fps=_to_float(fields.get("fps")), speed=speed, eta_s=eta, note=note
    )


def parse_ffmpeg_progress(lines: Iterable[str], duration_s: float) -> Iterator[Progress]:
    """Yield one `Progress` per ``progress=`` terminator of ``-progress pipe:1`` output."""
    fields: dict[str, str] = {}
    for raw in lines:
        key, sep, value = raw.strip().partition("=")
        if not sep or not key:
            continue
        fields[key] = value
        if key == "progress":
            yield _progress_from_fields(fields, duration_s)
            fields = {}


_MKVMERGE_PROGRESS = re.compile(r"^#GUI#progress\s+(\d+(?:\.\d+)?)%")


def parse_mkvmerge_progress(line: str) -> float | None:
    """``"#GUI#progress 37%"`` -> ``37.0``; any other line -> None."""
    match = _MKVMERGE_PROGRESS.match(line.strip())
    return float(match.group(1)) if match else None


def format_eta(eta_s: float | None) -> str:
    if eta_s is None:
        return "?"
    total = int(eta_s)
    hours, rem = divmod(total, 3600)
    minutes = rem // 60
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m"
    return f"{total}s"


def format_heartbeat(label: str, elapsed_s: float) -> str:
    minutes, seconds = divmod(int(elapsed_s), 60)
    return f"{label}, {minutes}m{seconds:02d}s elapsed, still working"


def format_progress(p: Progress) -> str:
    """Compact detail text such as ``43.2%  38 fps  1.6x  ETA 52m``."""
    parts: list[str] = []
    if p.percent is not None:
        parts.append(f"{p.percent:.1f}%")
    if p.fps is not None:
        parts.append(f"{p.fps:.0f} fps")
    if p.speed is not None:
        parts.append(f"{p.speed:.1f}x")
    if p.eta_s is not None:
        parts.append(f"ETA {format_eta(p.eta_s)}")
    return "  ".join(parts) if parts else "working"


class Reporter(Protocol):
    def stage(self, label: str) -> None: ...

    def update(self, p: Progress) -> None: ...

    def heartbeat(self, label: str, elapsed_s: float) -> None: ...

    def file_done(self, summary: str, level: Level = Level.INFO) -> None: ...


class PlainReporter:
    """Tagged, timestamped plain lines; `update` is rate-limited to one line per *interval_s*.

    Thread-safe: all state and writes are guarded by one re-entrant lock.
    """

    def __init__(
        self,
        interval_s: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        out: TextIO | None = None,
        now: Callable[[], datetime] = datetime.now,
    ) -> None:
        self._interval_s = interval_s
        self._clock = clock
        self._out = out
        self._now = now
        self._label = ""
        self._last_update: float | None = None
        self._lock = threading.RLock()

    def _emit(self, text: str, level: Level = Level.INFO) -> None:
        out = self._out if self._out is not None else sys.stdout
        line = f"{tag(level)} {self._now():%H:%M:%S} {text}\n"
        try:
            out.write(encodable(line, out))
        except UnicodeEncodeError:
            out.write(line.encode("ascii", errors="backslashreplace").decode("ascii"))
        out.flush()

    def stage(self, label: str) -> None:
        with self._lock:
            self._label = label
            self._emit(label)

    def update(self, p: Progress) -> None:
        with self._lock:
            now = self._clock()
            if self._last_update is not None and now - self._last_update < self._interval_s:
                return
            self._last_update = now
            prefix = f"{self._label}  " if self._label else ""
            self._emit(f"{prefix}{format_progress(p)}")

    def heartbeat(self, label: str, elapsed_s: float) -> None:
        with self._lock:
            self._emit(format_heartbeat(label, elapsed_s))

    def file_done(self, summary: str, level: Level = Level.INFO) -> None:
        with self._lock:
            self._emit(summary, level)


class RichReporter:
    """In-place `rich` progress bar for interactive terminals.

    Thread-safe: all state and console writes are guarded by one re-entrant lock. Text sent
    to the console is made encodable first (see `encodable`).
    """

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console(emoji=False)
        self._progress: RichProgress | None = None
        self._task: TaskID | None = None
        self._lock = threading.RLock()

    def _text(self, text: str) -> str:
        return encodable(text, self._console.file)

    def _ensure_started(self) -> RichProgress:
        if self._progress is None:
            self._progress = RichProgress(
                # ASCII spinner on narrow-encoding consoles (the braille frames cannot be encoded).
                SpinnerColumn("line" if self._console.options.ascii_only else "dots"),
                TextColumn("{task.description}"),
                BarColumn(),
                TextColumn("{task.fields[detail]}"),
                console=self._console,
                transient=True,
            )
            self._progress.start()
            # Log lines printed from now on go through this console, above the bar.
            set_active_console(self._console)
        return self._progress

    def stage(self, label: str) -> None:
        with self._lock:
            progress = self._ensure_started()
            if self._task is not None:
                progress.remove_task(self._task)
            self._task = progress.add_task(self._text(escape(label)), total=None, detail="")

    def update(self, p: Progress) -> None:
        with self._lock:
            if self._progress is None or self._task is None:
                return
            detail = format_progress(p)
            # The bar redraws ten times a second; the final update must not wait for that, or
            # the next line can replace the bar while it still shows an earlier percentage.
            refresh = p.note == "end"
            if p.percent is None:
                self._progress.update(self._task, total=None, detail=detail, refresh=refresh)
            else:
                self._progress.update(
                    self._task, total=100.0, completed=p.percent, detail=detail, refresh=refresh
                )

    def heartbeat(self, label: str, elapsed_s: float) -> None:
        """Nothing to print: the spinner on the bar already shows that the tool is alive."""

    def file_done(self, summary: str, level: Level = Level.INFO) -> None:
        with self._lock:
            self.close()
            self._console.print(tagged(level, self._text(summary)), soft_wrap=True)

    def close(self) -> None:
        """Stop the live display; safe to call repeatedly."""
        with self._lock:
            if self._progress is not None:
                self._progress.stop()
                clear_active_console(self._console)
                self._progress = None
                self._task = None


class ProgressMode(StrEnum):
    """How long commands report progress on the console."""

    AUTO = "auto"  # the live bar on an interactive terminal, plain lines otherwise
    RICH = "rich"  # always the live bar, even when output is piped (the terminal must show it)
    PLAIN = "plain"  # always plain lines, one every ``plain_progress_s``


def make_reporter(
    settings: Settings,
    *,
    is_tty: bool,
    mode: ProgressMode = ProgressMode.AUTO,
    clock: Callable[[], float] = time.monotonic,
    out: TextIO | None = None,
) -> Reporter:
    """A `rich` in-place reporter or plain lines every ``plain_progress_s``, chosen by *mode*.

    ``AUTO`` picks the live bar when *is_tty*; ``RICH`` and ``PLAIN`` ignore *is_tty*.
    """
    live = mode is ProgressMode.RICH or (mode is ProgressMode.AUTO and is_tty)
    if not live:
        return PlainReporter(settings.plain_progress_s, clock=clock, out=out)
    if mode is not ProgressMode.RICH:
        return RichReporter(Console(file=out, emoji=False))
    # Explicit rich mode: draw the bar although stdout may not be a console (a pipe, or a
    # terminal such as Git Bash's that Windows does not report as one). Rich would then fall
    # back to a Windows console API that does not exist there and print no control codes, so
    # ask for ANSI codes directly unless stdout really is a console.
    real_console = out is None and sys.stdout.isatty()
    return RichReporter(
        Console(
            file=out,
            emoji=False,
            force_terminal=True,
            legacy_windows=None if real_console else False,
        )
    )
