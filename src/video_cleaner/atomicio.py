"""Write a file so readers only ever see the old content or the complete new content.

The text goes to a temporary file beside the target (same folder, so the final move is a
rename on one volume) and is moved over the target with `os.replace`. On any failure the
temporary file is removed, the previous file stays as it was, and the error propagates
unchanged: callers turn it into their own error (or, like the survey, retry another name when
the target is locked).
"""

from __future__ import annotations

import contextlib
import csv
import io
import os
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path


def write_text_atomic(
    path: Path, text: str, *, fsync: bool = False, errors: str = "strict"
) -> None:
    """Write *text* as UTF-8 (no BOM, no newline translation) to *path*, atomically.

    The parent folder is created when missing. *fsync* flushes the data to disk before the
    move (for files whose loss would matter after a crash, such as the manifest). *errors* is
    the codec error handler: ``"backslashreplace"`` keeps file names with undecodable bytes
    writable; the default ``"strict"`` raises `UnicodeEncodeError` instead.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", errors=errors, newline="") as fh:
            fh.write(text)
            if fsync:
                fh.flush()
                os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


# A spreadsheet treats a cell starting with one of these as a formula (CSV injection, CWE-1236).
_FORMULA_TRIGGERS = ("=", "+", "-", "@", "\t", "\r")
_ESCAPE = "'"


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def csv_cell(value: object) -> object:
    """*value* made safe to open in a spreadsheet: a text cell that would be read as a formula
    gets a leading apostrophe. Numbers (``-3``, ``"-1.5"``) stay as they are. A text cell that
    already starts with an apostrophe gets one more, so `csv_cell_value` reverses it exactly.
    """
    if not isinstance(value, str) or not value:
        return value
    if value.startswith(_ESCAPE) or (value.startswith(_FORMULA_TRIGGERS) and not _is_number(value)):
        return _ESCAPE + value
    return value


def csv_cell_value(text: str) -> str:
    """The original text of a cell written through `csv_cell` (for readers of our own CSVs)."""
    return text[1:] if text.startswith(_ESCAPE) else text


def write_csv_atomic(
    path: Path,
    header: Sequence[str],
    rows: Iterable[Sequence[object]],
    *,
    fsync: bool = False,
    errors: str = "strict",
) -> None:
    """Write a CSV (csv module quoting) with *header* and *rows*; see `write_text_atomic`.

    Every cell goes through `csv_cell`, so no file name can turn into a spreadsheet formula.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow([csv_cell(cell) for cell in header])
    writer.writerows([csv_cell(cell) for cell in row] for row in rows)
    write_text_atomic(path, buffer.getvalue(), fsync=fsync, errors=errors)
