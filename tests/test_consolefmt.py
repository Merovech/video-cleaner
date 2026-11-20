from __future__ import annotations

import io
import logging
from collections.abc import Iterator

import pytest
from rich.console import Console

from video_cleaner.consolefmt import (
    ConsoleLogHandler,
    Level,
    clear_active_console,
    level_for_record,
    print_tagged,
    set_active_console,
    tag,
    tagged,
)
from video_cleaner.progress import Progress, RichReporter

ESC = "\x1b["


@pytest.fixture(autouse=True)
def _no_active_console() -> Iterator[None]:
    set_active_console(None)
    yield
    set_active_console(None)


def _colour_console(buf: io.StringIO) -> Console:
    return Console(file=buf, force_terminal=True, width=120, color_system="standard")


@pytest.mark.parametrize(
    ("level", "text"),
    [
        (Level.OK, "[ OK ]"),
        (Level.INFO, "[INFO]"),
        (Level.WARN, "[WARN]"),
        (Level.ERR, "[ERR ]"),
    ],
)
def test_tags_are_four_characters_inside_brackets(level: Level, text: str) -> None:
    assert tag(level) == text
    assert tagged(level, "hello").plain == f"{text} hello"


@pytest.mark.parametrize(
    ("level", "code"),
    [(Level.OK, "32"), (Level.INFO, "37"), (Level.WARN, "33"), (Level.ERR, "31")],
)
def test_only_the_text_inside_the_brackets_is_coloured(level: Level, code: str) -> None:
    buf = io.StringIO()
    _colour_console(buf).print(tagged(level, "message"))
    inner = tag(level)[1:-1]
    assert f"[{ESC}{code}m{inner}{ESC}0m] message" in buf.getvalue()


def test_message_is_never_parsed_as_markup() -> None:
    assert tagged(Level.INFO, "[bold]x[/bold] [red]").plain == "[INFO] [bold]x[/bold] [red]"


@pytest.mark.parametrize(
    ("levelno", "level"),
    [
        (logging.DEBUG, Level.INFO),
        (logging.INFO, Level.INFO),
        (logging.WARNING, Level.WARN),
        (logging.ERROR, Level.ERR),
        (logging.CRITICAL, Level.ERR),
    ],
)
def test_log_levels_map_to_tags(levelno: int, level: Level) -> None:
    assert level_for_record(levelno) is level


def _logger(handler: logging.Handler, name: str) -> logging.Logger:
    log = logging.getLogger(f"consolefmt-test.{name}")
    log.handlers = [handler]
    log.propagate = False
    log.setLevel(logging.DEBUG)
    return log


def test_log_handler_writes_tagged_lines_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    log = _logger(ConsoleLogHandler(), "stderr")
    log.info("one")
    log.warning("two")
    log.error("three")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.splitlines() == ["[INFO] one", "[WARN] two", "[ERR ] three"]


def test_log_handler_appends_the_traceback(capsys: pytest.CaptureFixture[str]) -> None:
    log = _logger(ConsoleLogHandler(), "exc")
    try:
        raise RuntimeError("kaput")
    except RuntimeError:
        log.exception("failed")
    err = capsys.readouterr().err
    assert err.startswith("[ERR ] failed")
    assert "RuntimeError: kaput" in err


def test_log_handler_prints_message_text_literally(capsys: pytest.CaptureFixture[str]) -> None:
    log = _logger(ConsoleLogHandler(), "literal")
    log.info("Smile :smile: [red]x[/red] %s", "映画")
    assert capsys.readouterr().err.strip() == "[INFO] Smile :smile: [red]x[/red] 映画"


def test_log_handler_does_not_wrap_long_lines(capsys: pytest.CaptureFixture[str]) -> None:
    log = _logger(ConsoleLogHandler(), "long")
    log.info("x" * 500)
    assert len(capsys.readouterr().err.splitlines()) == 1


def test_lines_go_through_the_active_console_instead_of_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    buf = io.StringIO()
    console = Console(file=buf, width=120, color_system=None)
    set_active_console(console)
    _logger(ConsoleLogHandler(), "active").warning("via the bar's console")
    print_tagged(Level.OK, "also via it")
    assert capsys.readouterr().err == ""
    assert buf.getvalue().splitlines() == ["[WARN] via the bar's console", "[ OK ] also via it"]


def test_clear_active_console_only_clears_its_own_registration() -> None:
    mine = Console(file=io.StringIO())
    other = Console(file=io.StringIO())
    set_active_console(other)
    clear_active_console(mine)  # not the registered one: nothing changes
    buf = io.StringIO()
    other.file = buf
    print_tagged(Level.INFO, "still routed")
    assert "still routed" in buf.getvalue()
    clear_active_console(other)


def test_rich_reporter_registers_its_console_while_the_bar_is_live(
    capsys: pytest.CaptureFixture[str],
) -> None:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=120, color_system=None)
    rep = RichReporter(console=console)
    log = _logger(ConsoleLogHandler(), "live")

    rep.stage("encode film.mkv")
    rep.update(Progress(percent=10.0, fps=1.0, speed=1.0, eta_s=1.0))
    log.info("while the bar is up")
    assert capsys.readouterr().err == ""  # not behind the bar's back
    assert "[INFO] while the bar is up" in buf.getvalue()

    rep.close()
    log.info("after the bar")
    assert "[INFO] after the bar" in capsys.readouterr().err
