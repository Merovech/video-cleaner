"""Run logging: a DEBUG file per run, a console handler by verbosity, and secret redaction.

Every record that reaches either handler passes through `RedactingFilter`, so keys, tokens
and passwords never reach the console or the log file.
"""

from __future__ import annotations

import logging
import re
import threading
import unicodedata
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from video_cleaner.consolefmt import ConsoleLogHandler

LOGGER_NAME = "video_cleaner"
MASK = "***"
_HANDLER_TAG = "_video_cleaner_handler"

MIN_SECRET_LENGTH = 4

# Each pattern keeps the label and replaces only the value. A key may carry a prefix
# (access_token, x_api_key) and a closing quote before the separator (dict/JSON style
# ``{"password": "x"}``, or the same inside a JSON string: ``{\"password\": \"x\"}``).
# Values are quoted strings (spaces allowed, masked through the closing quote; a backslash
# escapes a quote inside a plain-quoted value) or a bare run up to whitespace, a quote or a
# delimiter. A value may start with an auth scheme (``token: Bearer abc``), which is masked
# along with the credential.
_SCHEME = r"(?:(?:bearer|basic|digest)\s+)?"
_VALUE = (
    r"(?:\\\"(?:(?!\\\").)*(?:\\\")?"  # escaped quotes: \"value\"
    r"|\"(?:[^\"\\\n]|\\.)*\"?"  # "value", with \" allowed inside
    r"|'[^'\n]*'?"
    r"|[^\s\"'&,;]+)"
)
_SEP = r"\\?[\"']?\s*[:=]\s*"
_LABELLED_VALUE = rf"{_SCHEME}{_VALUE}"
_GENERIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(rf"(?i)(api[-_]?key{_SEP}){_LABELLED_VALUE}"),
    re.compile(rf"(?i)(password{_SEP}){_LABELLED_VALUE}"),
    re.compile(rf"(?i)(token{_SEP}){_LABELLED_VALUE}"),
    re.compile(r"(?i)(\bbearer\s+)[^\s\"',;]+"),
    # Basic credentials are base64. After an Authorization / Proxy-Authorization header any
    # base64 token is masked (base64 of "user:pass" can have no digit, '+', '/' or '=').
    re.compile(
        r"(?i)((?:proxy-)?authorization\s*[:=]\s*basic\s+)[A-Za-z0-9+/=_-]{4,}",
    ),
    # Elsewhere a bare "basic <token>" is masked only when the token looks encoded (a digit,
    # '+' or '=') so prose such as "basic setup/teardown" is left alone.
    re.compile(r"(?i)(\bbasic\s+)(?=[A-Za-z0-9+/_-]*[0-9+=])[A-Za-z0-9+/=_-]{6,}"),
    re.compile(r"(?i)(\bdigest\s+)(?=\w+\s*=)[^\n]+"),
)

# Process-wide secrets (values the run knows are credentials), masked by every `redact` call
# and every `RedactingFilter` in addition to the ones passed explicitly. An immutable snapshot,
# replaced under a lock on every write, so a thread that redacts (the tool's stderr drain) never
# iterates a set another thread is changing.
_registered: frozenset[str] = frozenset()
_registered_lock = threading.Lock()


def register_secrets(secrets: Iterable[str]) -> None:
    """Add *secrets* to the process-wide set that `redact` always masks.

    Values shorter than `MIN_SECRET_LENGTH` characters are ignored: masking a one- or
    two-letter value would garble ordinary text.
    """
    global _registered
    new = {s for s in secrets if len(s) >= MIN_SECRET_LENGTH}
    with _registered_lock:
        _registered = _registered | new


def _reset_registered_secrets() -> None:
    """Forget every registered secret (for tests, so one test cannot leak into the next)."""
    global _registered
    with _registered_lock:
        _registered = frozenset()


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    """Mask the explicit and registered *secrets* and generic credential values.

    The generic ones are Api-Key / password / token labels, Bearer / Basic / Digest credentials.
    """
    # Longest first so a secret that contains another is masked whole.
    registered = _registered  # one snapshot for the whole call
    for secret in sorted({s for s in (*secrets, *registered) if s}, key=len, reverse=True):
        text = text.replace(secret, MASK)
    for pattern in _GENERIC_PATTERNS:
        text = pattern.sub(rf"\g<1>{MASK}", text)
    return text


def printable(text: object, limit: int = 120, *, escape: bool = True) -> str:
    """*text* as one short line that is safe to show or log: credentials masked, and control,
    format and separator characters (terminal codes, C1 controls, bidi overrides, zero-width
    characters, line separators, lone surrogates) written as escapes, or dropped (and the result
    stripped) with ``escape=False``. At most *limit* characters; a longer one ends in "...".

    The one sanitiser for text a file name, a tool or a server wrote, on its way to the console,
    the log or a stored reason.
    """
    out: list[str] = []
    for ch in redact(str(text)):  # the whole text first: a secret must not be cut in two
        category = unicodedata.category(ch)
        if category[0] != "C" and category not in ("Zl", "Zp"):
            out.append(ch)
        elif escape:
            out.append(ascii(ch)[1:-1])
        if len(out) >= limit * 2:  # more than enough to fill *limit* (a long text costs no more)
            break
    line = "".join(out)
    if not escape:
        return line.strip()[:limit]
    return line if len(line) <= limit else line[: limit - 3] + "..."


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
