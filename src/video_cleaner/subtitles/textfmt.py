"""Turn untrusted subtitle bytes into clean cues and UTF-8 SRT text.

Subtitle bytes come from downloads, files next to the video and extracted tracks, so they are
treated as hostile. The size is checked before anything is decoded, only plain-text formats are
read (SRT, WebVTT, ASS/SSA; never XML or JSON), the content is never evaluated, and nothing taken
from it is used as a name. A `SubtitleRejected` message is safe to log: it carries counts and
sizes, never subtitle text.

The formats are read by the small line-based parsers in this module rather than by a subtitle
library. Library parsers lean on regular expressions with nested or optional quantifiers, and
attacker-shaped text (a "<" followed by a long run of spaces, a "[" heading line with a long run
of letters) makes those backtrack quadratically; a file under the size cap could then cost
minutes of CPU. Here every regular expression is applied to a single line, which is capped at
`MAX_LINE_CHARS`, and is either anchored with plain linear quantifiers or bounded in length.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

#: Timestamps past this are not a real subtitle (and usually mean a mangled file).
MAX_TIMESTAMP_MS = 100 * 3600 * 1000
#: More cues than any film or episode has; stops a crafted file from costing unbounded work.
MAX_CUES = 50_000
#: Longer than any real subtitle line, even an ASS line carrying many override tags.
MAX_LINE_CHARS = 20_000

# Characters that do not belong in text: C0 controls except tab, line feed and carriage return,
# plus DEL and the C1 range.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
# A binary file has many control characters; a text file with a stray one or two is tolerated.
_BINARY_MIN_CONTROLS = 10
_BINARY_MAX_RATIO = 0.01

_HTML_PAGE_RE = re.compile(r"<\s*(?:!doctype\s+html|html|head|body)\b", re.IGNORECASE)
_HTML_SNIFF_CHARS = 4096

# Format detection: ASS and SSA name their styles section, WebVTT starts with a signature.
_ASS_MARK_RE = re.compile(r"V4\+ Styles", re.IGNORECASE)
_SSA_MARK_RE = re.compile(r"V4 Styles", re.IGNORECASE)
_VTT_SIGNATURE = "WEBVTT"

# One timestamp: [hours:]minutes:seconds[,.]fraction. The digit runs are plain greedy runs that
# the next literal ends, so a line is matched in linear time, and their values are judged as
# text (see `_timestamp_ms`) because Python refuses to convert very long digit strings.
_TIMESTAMP = r"(?:(\d+):)?(\d+):(\d{2})[,.](\d{1,3})"
_CUE_LINE_RE = re.compile(rf"\s*{_TIMESTAMP}\s*-->\s*{_TIMESTAMP}")
_ASS_TIME_RE = re.compile(r"(\d+):(\d{1,2}):(\d{1,2})[.,](\d{1,3})")
# ASS event fields when the section has no Format line (the SSA "Marked" column sits where the
# ASS "Layer" column does).
_ASS_DEFAULT_FIELDS = (
    "layer",
    "start",
    "end",
    "style",
    "name",
    "marginl",
    "marginr",
    "marginv",
    "effect",
    "text",
)

_ASS_OVERRIDE_RE = re.compile(r"\{[^{}]*\}")
_ASS_DRAWING_RE = re.compile(r"\{[^{}]*\\p\d")
_ASS_BREAK_RE = re.compile(r"\\[Nn]")
_ASS_HARD_SPACE_RE = re.compile(r"\\h")
# HTML/WebVTT tags (<i>, </font>, <c.yellow>, <v Anna>, <00:01.000>). A tag must start with a
# letter, so "I <3 you" and "2 < 3" are left alone. At most 200 characters are looked at per "<".
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


@dataclass(frozen=True)
class _RawCue:
    """A cue as read from the file, before its text is cleaned."""

    start_ms: int
    end_ms: int
    text: str


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
    unrecognised format, has too few cues, or has impossible timestamps.
    """
    if len(raw) > max_bytes:
        raise SubtitleRejected(f"subtitle is larger than {max_bytes} bytes ({len(raw)} bytes)")

    text = decode_subtitle_bytes(raw)
    if not text.strip():
        raise SubtitleRejected("subtitle is empty")
    _reject_binary(text)
    if _HTML_PAGE_RE.search(text[:_HTML_SNIFF_CHARS]):
        raise SubtitleRejected("subtitle is an HTML page, not a subtitle")
    lines = text.split("\n")
    _reject_long_lines(lines)

    try:
        source_format, raw_cues = _parse(text, lines)
    except SubtitleRejected:
        raise
    except Exception:
        raise SubtitleRejected("subtitle could not be parsed") from None

    cues: list[Cue] = []
    for raw_cue in raw_cues:
        cleaned = _clean_text(raw_cue.text, ass=source_format in {"ass", "ssa"})
        if cleaned is None:
            continue
        if raw_cue.end_ms < raw_cue.start_ms:
            raise SubtitleRejected("subtitle has a cue with its end before its start")
        cues.append(Cue(raw_cue.start_ms, raw_cue.end_ms, cleaned))

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


def _reject_long_lines(lines: Sequence[str]) -> None:
    if max(map(len, lines), default=0) > MAX_LINE_CHARS:
        raise SubtitleRejected(f"subtitle has a line longer than {MAX_LINE_CHARS} characters")


def _timestamp_ms(hours: str | None, minutes: str, seconds: str, fraction: str) -> int:
    """Milliseconds for the digit strings of one timestamp; rejects 100 hours or more.

    The digit strings are compared as text first, so a field of thousands of digits is rejected
    without being converted to an int.
    """
    if hours is not None and len(hours.lstrip("0")) >= 3:
        raise SubtitleRejected("subtitle has a timestamp beyond 100 hours")
    if len(minutes.lstrip("0")) >= 3:
        raise SubtitleRejected("subtitle has a malformed timestamp (minutes field too long)")
    total = int(hours.lstrip("0") or "0") if hours is not None else 0
    total = total * 60 + int(minutes.lstrip("0") or "0")
    total = total * 60 + int(seconds)
    total = total * 1000 + int(fraction.ljust(3, "0"))
    if total > MAX_TIMESTAMP_MS:
        raise SubtitleRejected("subtitle has a timestamp beyond 100 hours")
    return total


def _parse(text: str, lines: Sequence[str]) -> tuple[str, list[_RawCue]]:
    """Detect the format from the content and read it; returns (format name, raw cues)."""
    if _ASS_MARK_RE.search(text):
        return "ass", _parse_ass(lines)
    if _SSA_MARK_RE.search(text):
        return "ssa", _parse_ass(lines)
    if text.lstrip().startswith(_VTT_SIGNATURE):
        return "vtt", _parse_cue_blocks(lines, vtt=True)
    cues = _parse_cue_blocks(lines, vtt=False)
    if not cues:
        raise SubtitleRejected("subtitle is not a recognised subtitle format")
    return "srt", cues


def _check_cue_count(count: int) -> None:
    if count > MAX_CUES:
        raise SubtitleRejected(f"subtitle has too many cues (more than {MAX_CUES})")


def _parse_cue_blocks(lines: Sequence[str], *, vtt: bool) -> list[_RawCue]:
    """Read SRT or WebVTT: a timestamp line starts a cue, the lines after it are its text.

    A WebVTT cue ends at the first blank line, which also makes the identifier and NOTE blocks
    that follow it harmless. An SRT cue is read more leniently: text after a blank line still
    belongs to the cue unless it is the bare number that opens the next one.
    """
    cues: list[_RawCue] = []
    bodies: list[list[str]] = []
    current: list[str] | None = None
    after_blank = False
    for line in lines:
        match = _CUE_LINE_RE.match(line)
        if match:
            start = _timestamp_ms(*match.group(1, 2, 3, 4))
            end = _timestamp_ms(*match.group(5, 6, 7, 8))
            current = []
            bodies.append(current)
            cues.append(_RawCue(start, end, ""))
            _check_cue_count(len(cues))
            after_blank = False
            continue
        if current is None:
            continue
        stripped = line.strip()
        if not stripped:
            if vtt:
                current = None
            else:
                after_blank = True
            continue
        if after_blank:
            if stripped.isascii() and stripped.isdigit():
                current = None
                continue
            after_blank = False
        current.append(line)
    return [
        _RawCue(cue.start_ms, cue.end_ms, "\n".join(body))
        for cue, body in zip(cues, bodies, strict=True)
    ]


def _ass_time_ms(field: str) -> int:
    match = _ASS_TIME_RE.fullmatch(field.strip())
    if match is None:
        raise SubtitleRejected("subtitle has an unreadable timestamp")
    hours, minutes, seconds, fraction = match.groups()
    return _timestamp_ms(hours, minutes, seconds, fraction)


def _parse_ass(lines: Sequence[str]) -> list[_RawCue]:
    """Read the Dialogue events of an ASS/SSA file; Comment events are skipped."""
    cues: list[_RawCue] = []
    in_events = False
    fields: tuple[str, ...] = _ASS_DEFAULT_FIELDS
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_events = stripped[1:-1].strip().lower() == "events"
            continue
        if not in_events:
            continue
        head, _, rest = stripped.partition(":")
        kind = head.strip().lower()
        if kind == "format":
            fields = tuple(name.strip().lower() for name in rest.split(","))
        elif kind == "dialogue":
            if fields[-1:] != ("text",) or "start" not in fields or "end" not in fields:
                raise SubtitleRejected("subtitle has an unusable event format line")
            parts = rest.lstrip().split(",", len(fields) - 1)
            if len(parts) != len(fields):
                continue
            by_name = dict(zip(fields, parts, strict=True))
            cues.append(
                _RawCue(
                    _ass_time_ms(by_name["start"]), _ass_time_ms(by_name["end"]), by_name["text"]
                )
            )
            _check_cue_count(len(cues))
    return cues


def _clean_text(text: str, *, ass: bool) -> str | None:
    """Plain text of one cue, or None when nothing is left (or it is an ASS vector drawing)."""
    if ass and _ASS_DRAWING_RE.search(text):
        return None
    text = _ASS_OVERRIDE_RE.sub("", text)
    if ass:
        text = _ASS_BREAK_RE.sub(lambda m: "\n" if m.group(0) == "\\N" else " ", text)
        text = _ASS_HARD_SPACE_RE.sub(" ", text)
    text = _MARKUP_TAG_RE.sub("", text)
    text = _CONTROL_RE.sub("", text)
    lines = [line.strip() for line in text.split("\n")]
    cleaned = "\n".join(line for line in lines if line)
    return cleaned or None
