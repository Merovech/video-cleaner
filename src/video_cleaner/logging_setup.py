"""Run logging: a DEBUG file per run, a console handler by verbosity, and secret redaction.

Every record that reaches either handler passes through `RedactingFilter`, so keys, tokens
and passwords never reach the console or the log file (spec section 4.4).
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

LOGGER_NAME = "video_cleaner"
MASK = "***"
_HANDLER_TAG = "_video_cleaner_handler"

# Each pattern keeps the label and replaces only the value.
_GENERIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)(\bapi[-_]?key\s*[:=]\s*)[\"']?[^\s\"'&,;]+"),
    re.compile(r"(?i)(\bpassword\s*[:=]\s*)[\"']?[^\s\"'&,;]+"),
    re.compile(r"(?i)(\btoken\s*[:=]\s*)[\"']?[^\s\"'&,;]+"),
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
        record.msg = redact(record.getMessage(), self._secrets)
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


def setup_logging(log_dir: Path, *, verbosity: int, secrets: Iterable[str] = ()) -> Path:
    """Configure the ``video_cleaner`` logger and return the path of this run's log file.

    The file handler is always DEBUG. The console handler (stderr) is WARNING for
    ``verbosity < 0`` (-q), INFO for 0 and DEBUG for ``> 0`` (-v). Calling this again replaces
    the handlers installed by the previous call. The logger does not propagate, so records
    never reach un-redacted handlers on the root logger.
    """
    secret_list = [s for s in secrets if s]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run-{datetime.now():%Y%m%d-%H%M%S}.log"

    logger = logging.getLogger(LOGGER_NAME)
    for old in [h for h in logger.handlers if getattr(h, _HANDLER_TAG, False)]:
        logger.removeHandler(old)
        old.close()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(_console_level(verbosity))
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    for handler in (file_handler, console_handler):
        handler.addFilter(RedactingFilter(secret_list))
        setattr(handler, _HANDLER_TAG, True)
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return log_path
