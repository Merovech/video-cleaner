"""Run logging: a DEBUG file per run, a console handler by verbosity, and secret redaction.

Every record that reaches either handler passes through `RedactingFilter`, so keys, tokens
and passwords never reach the console or the log file.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from video_cleaner.consolefmt import ConsoleLogHandler

LOGGER_NAME = "video_cleaner"
MASK = "***"
_HANDLER_TAG = "_video_cleaner_handler"

# Each pattern keeps the label and replaces only the value. A key may carry a prefix
# (access_token, x_api_key) and a closing quote before the separator (dict/JSON style
# ``{"password": "x"}``). Values are quoted strings (spaces allowed, masked through the
# closing quote) or a bare run up to whitespace, a quote or a delimiter.
_VALUE = r"(?:\"[^\"\n]*\"?|'[^'\n]*'?|[^\s\"'&,;]+)"
_SEP = r"[\"']?\s*[:=]\s*"
_GENERIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(rf"(?i)(api[-_]?key{_SEP}){_VALUE}"),
    re.compile(rf"(?i)(password{_SEP}){_VALUE}"),
    re.compile(rf"(?i)(token{_SEP}){_VALUE}"),
    re.compile(r"(?i)(\bbearer\s+)[^\s\"',;]+"),
)


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    """Mask the explicit *secrets* and generic Api-Key / password / token / Bearer values."""
    # Longest first so a secret that contains another is masked whole.
    for secret in sorted({s for s in secrets if s}, key=len, reverse=True):
        text = text.replace(secret, MASK)
    for pattern in _GENERIC_PATTERNS:
        text = pattern.sub(rf"\g<1>{MASK}", text)
    return text


class RedactingFilter(logging.Filter):
    """Redacts the rendered message, traceback and stack of every record."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        super().__init__()
        self._secrets = tuple(s for s in secrets if s)

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            # A malformed log call must never abort the run (the stdlib would only print
            # "Logging error"); fall back to the raw message plus the arguments.
            message = f"{record.msg} {record.args!r}" if record.args else str(record.msg)
        record.msg = redact(message, self._secrets)
        record.args = None
        if record.exc_info and not record.exc_text:
            # Render now (Formatter would do the same) so the text can be redacted.
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = redact(record.exc_text, self._secrets)
        if record.stack_info:
            record.stack_info = redact(record.stack_info, self._secrets)
        return True


def _console_level(verbosity: int) -> int:
    if verbosity < 0:
        return logging.WARNING
    return logging.DEBUG if verbosity > 0 else logging.INFO


def setup_logging(
    log_dir: Path,
    *,
    verbosity: int,
    secrets: Iterable[str] = (),
    log_file: Path | None = None,
    file_level: int = logging.DEBUG,
) -> Path:
    """Configure the ``video_cleaner`` logger and return the path of this run's log file.

    The log file is ``run-YYYYMMDD-HHMMSS.log`` in *log_dir*, or *log_file* when given
    (``--log-file``; its folder is created, *log_dir* is then not). The file handler logs at
    *file_level* (DEBUG unless the ``log_file_level`` setting says otherwise). The console
    handler (`ConsoleLogHandler`: tagged lines on stderr, or above a live progress bar) is
    WARNING for ``verbosity < 0`` (-q), INFO for 0 and DEBUG for ``> 0`` (-v). Calling this
    again replaces the handlers installed by the previous call. The logger does not propagate,
    so records never reach un-redacted handlers on the root logger.
    """
    secret_list = [s for s in secrets if s]
    default = log_dir / f"run-{datetime.now():%Y%m%d-%H%M%S}.log"
    log_path = default if log_file is None else log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    for old in [h for h in logger.handlers if getattr(h, _HANDLER_TAG, False)]:
        logger.removeHandler(old)
        old.close()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    console_handler = ConsoleLogHandler(_console_level(verbosity))

    for handler in (file_handler, console_handler):
        handler.addFilter(RedactingFilter(secret_list))
        setattr(handler, _HANDLER_TAG, True)
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return log_path
