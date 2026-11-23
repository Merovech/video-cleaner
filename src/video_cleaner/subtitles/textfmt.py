"""Turn untrusted subtitle bytes into clean cues and UTF-8 SRT text.

Subtitle bytes come from downloads, files next to the video and extracted tracks, so they are
treated as hostile: the size is checked before anything is decoded, only a short list of plain
text formats is parsed (never XML or JSON), the content is never evaluated, and nothing taken
from it is used as a name. A `SubtitleRejected` message is safe to log: it carries counts and
sizes, never subtitle text.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import pysubs2
from pysubs2.formats import autodetect_format

#: Timestamps past this are not a real subtitle (and usually mean a mangled file).
MAX_TIMESTAMP_MS = 100 * 3600 * 1000
#: More cues than any film or episode has; stops a crafted file from costing unbounded work.
MAX_CUES = 50_000
#: Longer than any real subtitle line, even an ASS line carrying many override tags.
MAX_LINE_CHARS = 20_000

# Only plain-text subtitle formats are parsed. XML based ones (TTML, SAMI) and JSON are left
# out on purpose: the XML parser is open to entity-expansion attacks and nobody needs them here.
_SUPPORTED_FORMATS = frozenset({"srt", "ass", "ssa", "vtt"})

# Characters that do not belong in text: C0 controls except tab, line feed and carriage return,
# plus DEL and the C1 range.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
# A binary file has many control characters; a text file with a stray one or two is tolerated.
_BINARY_MIN_CONTROLS = 10
_BINARY_MAX_RATIO = 0.01

_HTML_PAGE_RE = re.compile(r"<\s*(?:!doctype\s+html|html|head|body)\b", re.IGNORECASE)
_HTML_SNIFF_CHARS = 4096

# Hours with three or more digits: the parsers read only the last two, so a timestamp such as
# 100:00:01 would silently turn into 00:00:01 and has to be caught first.
_LONG_HOURS_RE = re.compile(r"(?<!\d)(\d{3,12}):(\d{1,2}):(\d{1,2})[.,](\d{1,3})(?!\d)")

_ASS_OVERRIDE_RE = re.compile(r"\{[^{}]*\}")
_ASS_DRAWING_RE = re.compile(r"\{[^{}]*\\p\d")
_ASS_BREAK_RE = re.compile(r"\\[Nn]")
_ASS_HARD_SPACE_RE = re.compile(r"\\h")
# HTML/WebVTT tags (<i>, </font>, <c.yellow>, <v Anna>, <00:01.000>). A tag must start with a
# letter, so "I <3 you" and "2 < 3" are left alone.
_MARKUP_TAG_RE = re.compile(r"</?[A-Za-z][^<>\n]{0,200}>|<\d{1,2}:\d{2}(?::\d{2})?\.\d{3}>")

_UTF8_BOM = b"\xef\xbb\xbf"
_UTF16_LE_BOM = b"\xff\xfe"
_UTF16_BE_BOM = b"\xfe\xff"


@dataclass(frozen=True)
class Cue:
    """One subtitle line shown from *start_ms* to *end_ms*; *text* may hold line breaks."""

    start_ms: int
    end_ms: int
    text: str


class SubtitleRejected(Exception):
    """The subtitle cannot be used. The message is the reason and is safe to log."""


@dataclass(frozen=True)
class NormalisedSubtitle:
    """A subtitle reduced to plain cues, plus the same cues as UTF-8 SRT text."""

    srt: str
    cues: tuple[Cue, ...]
    source_format: str


def decode_subtitle_bytes(raw: bytes) -> str:
    """Decode subtitle bytes to text with LF line endings and without a byte-order mark.

    Tries, in order: a UTF-16 byte-order mark, strict UTF-8 (with or without its mark), cp1252,
    and latin-1, which accepts every byte, so this never raises.
    """
    if raw.startswith(_UTF16_LE_BOM):
        text = raw[2:].decode("utf-16-le", errors="replace")
    elif raw.startswith(_UTF16_BE_BOM):
        text = raw[2:].decode("utf-16-be", errors="replace")
    else:
        body = raw[len(_UTF8_BOM) :] if raw.startswith(_UTF8_BOM) else raw
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = body.decode("cp1252")
            except UnicodeDecodeError:
                text = body.decode("latin-1")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalise_subtitle(raw: bytes, *, max_bytes: int, min_cues: int) -> NormalisedSubtitle:
    """Decode, detect the format by content, parse and clean *raw* into cues and SRT.

    Raises `SubtitleRejected` when the input is too large, empty, binary, an HTML page, in an
    unsupported or unrecognised format, has too few cues, or has impossible timestamps.
    """
    if len(raw) > max_bytes:
        raise SubtitleRejected(f"subtitle is larger than {max_bytes} bytes ({len(raw)} bytes)")

    text = decode_subtitle_bytes(raw)
    if not text.strip():
        raise SubtitleRejected("subtitle is empty")
    _reject_binary(text)
    if _HTML_PAGE_RE.search(text[:_HTML_SNIFF_CHARS]):
        raise SubtitleRejected("subtitle is an HTML page, not a subtitle")
    _reject_long_lines(text)
    _reject_long_hours(text)

    source_format, events = _parse(text)

    cues: list[Cue] = []
    for event in events:
        if event.type == "Comment":
            continue
        cleaned = _clean_text(event.text)
        if cleaned is None:
            continue
        start, end = int(event.start), int(event.end)
        if start < 0 or end < 0 or max(start, end) > MAX_TIMESTAMP_MS:
            raise SubtitleRejected("subtitle has a timestamp beyond 100 hours or below zero")
        if end < start:
            raise SubtitleRejected("subtitle has a cue with its end before its start")
        cues.append(Cue(start, end, cleaned))
        if len(cues) > MAX_CUES:
            raise SubtitleRejected(f"subtitle has too many cues (more than {MAX_CUES})")

    cues.sort(key=lambda cue: cue.start_ms)
    if len(cues) < min_cues:
        raise SubtitleRejected(f"subtitle has {len(cues)} cue(s), fewer than {min_cues}")
    return NormalisedSubtitle(srt=to_srt(cues), cues=tuple(cues), source_format=source_format)


def cues_per_hour(cues: Sequence[Cue], duration_s: float) -> float:
    """How many cues there are per hour of a video lasting *duration_s* seconds."""
    if not duration_s > 0:  # also true for NaN
        raise ValueError("duration must be positive")
    return len(cues) / duration_s * 3600.0


def to_srt(cues: Sequence[Cue]) -> str:
    """Render *cues* as SRT text, numbered from 1, with LF line endings."""
    blocks = []
    for number, cue in enumerate(cues, start=1):
        # A blank line would end the cue early, so empty lines are dropped.
        body = "\n".join(line for line in cue.text.splitlines() if line.strip())
        stamps = f"{_srt_time(cue.start_ms)} --> {_srt_time(cue.end_ms)}"
        blocks.append(f"{number}\n{stamps}\n{body}\n")
    return "\n".join(blocks)


def _srt_time(ms: int) -> str:
    seconds, millis = divmod(max(ms, 0), 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _reject_binary(text: str) -> None:
    controls = len(_CONTROL_RE.findall(text))
    if controls >= _BINARY_MIN_CONTROLS and controls > len(text) * _BINARY_MAX_RATIO:
        raise SubtitleRejected("subtitle looks like binary data")


def _reject_long_lines(text: str) -> None:
    longest = max((len(line) for line in text.split("\n")), default=0)
    if longest > MAX_LINE_CHARS:
        raise SubtitleRejected(f"subtitle has a line longer than {MAX_LINE_CHARS} characters")


def _reject_long_hours(text: str) -> None:
    for match in _LONG_HOURS_RE.finditer(text):
        hours, minutes, seconds, fraction = match.groups()
        millis = int(fraction.ljust(3, "0"))
        total = ((int(hours) * 60 + int(minutes)) * 60 + int(seconds)) * 1000 + millis
        if total > MAX_TIMESTAMP_MS:
            raise SubtitleRejected("subtitle has a timestamp beyond 100 hours")


def _parse(text: str) -> tuple[str, list[pysubs2.SSAEvent]]:
    """Detect the format from the content and parse it; returns (format name, events)."""
    try:
        detected = autodetect_format(text)
    except pysubs2.FormatAutodetectionError:
        raise SubtitleRejected("subtitle is not a recognised subtitle format") from None
    except Exception:
        raise SubtitleRejected("subtitle could not be parsed") from None
    if detected not in _SUPPORTED_FORMATS:
        raise SubtitleRejected(f"subtitle format '{detected}' is not supported")
    try:
        subs = pysubs2.SSAFile.from_string(text, detected)
        return detected, list(subs.events)
    except Exception:
        raise SubtitleRejected("subtitle could not be parsed") from None


def _clean_text(text: str) -> str | None:
    """Plain text of one event, or None when nothing is left (or it is a vector drawing)."""
    if _ASS_DRAWING_RE.search(text):
        return None
    text = _ASS_OVERRIDE_RE.sub("", text)
    text = _ASS_BREAK_RE.sub(lambda m: "\n" if m.group(0) == "\\N" else " ", text)
    text = _ASS_HARD_SPACE_RE.sub(" ", text)
    text = _MARKUP_TAG_RE.sub("", text)
    text = _CONTROL_RE.sub("", text)
    lines = [line.strip() for line in text.split("\n")]
    cleaned = "\n".join(line for line in lines if line)
    return cleaned or None
