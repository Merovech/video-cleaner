"""How the tool writes lines to the console: a level tag, colour, and one place to print.

Every line the tool prints about its own work starts with a four-character tag in brackets,
``[ OK ]`` (green), ``[INFO]`` (white), ``[WARN]`` (yellow), ``[FAIL]`` (red) or ``[VERB]``
(grey, debug detail shown by ``-v``); only the
text inside the brackets is coloured. The console log handler, the reporters and the command
summaries all use `tagged` / `print_tagged`, so the format lives here only.

While a live progress bar is on screen, lines must be printed through *its* console, which
clears the bar, prints, and redraws it. Writing to ``stderr`` behind its back leaves the bar's
last frame in the scrollback with the next line glued to it. `RichReporter` therefore registers
its console with `set_active_console`, and everything here prints through the registered
console while there is one.
"""

from __future__ import annotations

import logging
import threading
from enum import StrEnum
from typing import IO

from rich.console import Console
from rich.text import Text


class Level(StrEnum):
    OK = "ok"
    INFO = "info"
    WARN = "warn"
    FAIL = "fail"
    VERB = "verb"


# Four characters inside the brackets, so message text lines up whatever the level.
_TAGS = {
    Level.OK: " OK ",
    Level.INFO: "INFO",
    Level.WARN: "WARN",
    Level.FAIL: "FAIL",
    Level.VERB: "VERB",
}
_STYLES = {
    Level.OK: "green",
    Level.INFO: "white",
    Level.WARN: "yellow",
    Level.FAIL: "red",
    Level.VERB: "bright_black",
}


def tag(level: Level) -> str:
    """``"[ OK ]"``, ``"[INFO]"``, ``"[WARN]"``, ``"[FAIL]"`` or ``"[VERB]"`` (uncoloured)."""
    return f"[{_TAGS[level]}]"


def tagged(level: Level, message: str) -> Text:
    """*message* behind its level tag; *message* is taken literally (never rich markup)."""
    return Text.assemble("[", (_TAGS[level], _STYLES[level]), "] ", message)


def level_for_record(levelno: int) -> Level:
    """Log level to tag: ERROR and above FAIL, WARNING WARN, INFO INFO, DEBUG VERB."""
    if levelno >= logging.ERROR:
        return Level.FAIL
    if levelno >= logging.WARNING:
        return Level.WARN
    return Level.INFO if levelno >= logging.INFO else Level.VERB


def encodable(text: str, stream: IO[str] | None) -> str:
    """Make *text* writable to *stream* even if its encoding (e.g. cp1252) lacks characters.

    File names reach the console, so a redirected, narrow-encoding stream must never crash a
    long run: unencodable characters become backslash escapes.
    """
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return text
    try:
        return text.encode(encoding, errors="backslashreplace").decode(encoding)
    except (LookupError, UnicodeError):
        return text.encode("ascii", errors="backslashreplace").decode("ascii")


_lock = threading.Lock()
_active: Console | None = None
_fallback: dict[bool, Console] = {}


def set_active_console(console: Console | None) -> None:
    """Route console lines through *console* (the one drawing a live bar), or back to default."""
    global _active
    with _lock:
        _active = console


def clear_active_console(console: Console) -> None:
    """Stop routing through *console*; a no-op if another console has been registered since."""
    global _active
    with _lock:
        if _active is console:
            _active = None


def _console(*, stderr: bool) -> Console:
    with _lock:
        if _active is not None:
            return _active
        if stderr not in _fallback:
            # Resolves sys.stdout / sys.stderr at print time, so redirection and test capture work.
            _fallback[stderr] = Console(stderr=stderr, emoji=False, highlight=False)
        return _fallback[stderr]


def print_tagged(level: Level, message: str, *, stderr: bool = False) -> None:
    """Print one tagged line, through the live bar's console while there is one."""
    console = _console(stderr=stderr)
    # soft_wrap: the terminal wraps long paths; rich must not insert hard line breaks into them.
    console.print(tagged(level, encodable(message, console.file)), soft_wrap=True)


class ConsoleLogHandler(logging.Handler):
    """Console half of the run log: ``[LEVEL] message`` lines, coloured on a terminal.

    Writes to stderr unless a live bar is active (see `set_active_console`), in which case
    the line appears above the bar.
    """

    def __init__(self, level: int = logging.NOTSET) -> None:
        super().__init__(level)
        # message plus any traceback, the way the stdlib renders a record
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            print_tagged(level_for_record(record.levelno), self.format(record), stderr=True)
        except Exception:
            self.handleError(record)
