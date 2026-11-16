"""Progress parsing and console reporting (spec section 4.4).

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
from typing import IO, Protocol, TextIO

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

    def file_done(self, summary: str) -> None: ...


def _encodable(text: str, stream: IO[str] | None) -> str:
    """Make *text* writable to *stream* even if its encoding (e.g. cp1252) lacks characters.

    File names reach the reporters, so a redirected, narrow-encoding stream must never crash a
    long run: unencodable characters become backslash escapes.
    """
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return text
    try:
        return text.encode(encoding, errors="backslashreplace").decode(encoding)
    except (LookupError, UnicodeError):
        return text.encode("ascii", errors="backslashreplace").decode("ascii")


class PlainReporter:
    """Timestamped plain lines; `update` is rate-limited to one line per *interval_s*.

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

    def _emit(self, text: str) -> None:
        out = self._out if self._out is not None else sys.stdout
        line = f"[{self._now():%H:%M:%S}] {text}\n"
        try:
            out.write(_encodable(line, out))
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

    def file_done(self, summary: str) -> None:
        with self._lock:
            self._emit(summary)


class RichReporter:
    """In-place `rich` progress bar for interactive terminals.

    Thread-safe: all state and console writes are guarded by one re-entrant lock. Text sent
    to the console is made encodable first (see `_encodable`).
    """

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()
        self._progress: RichProgress | None = None
        self._task: TaskID | None = None
        self._lock = threading.RLock()

    def _text(self, text: str) -> str:
        return _encodable(text, self._console.file)

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
            if p.percent is None:
                self._progress.update(self._task, total=None, detail=detail)
            else:
                self._progress.update(self._task, total=100.0, completed=p.percent, detail=detail)

    def heartbeat(self, label: str, elapsed_s: float) -> None:
        with self._lock:
            self._console.print(self._text(escape(format_heartbeat(label, elapsed_s))))

    def file_done(self, summary: str) -> None:
        with self._lock:
            self.close()
            self._console.print(self._text(escape(summary)))

    def close(self) -> None:
        """Stop the live display; safe to call repeatedly."""
        with self._lock:
            if self._progress is not None:
                self._progress.stop()
                self._progress = None
                self._task = None


def make_reporter(
    settings: Settings,
    *,
    is_tty: bool,
    clock: Callable[[], float] = time.monotonic,
    out: TextIO | None = None,
) -> Reporter:
    """A `rich` in-place reporter on a TTY, otherwise plain lines every ``plain_progress_s``."""
    if is_tty:
        return RichReporter(Console(file=out) if out is not None else None)
    return PlainReporter(settings.plain_progress_s, clock=clock, out=out)
