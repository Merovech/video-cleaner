"""Hostile subtitle text must cost bounded time, whatever its shape.

The cost of parsing must not depend on regular expressions with nested or optional quantifiers
meeting attacker-shaped text, so these tests sweep families of shapes (a special character
followed by long runs of another, many short or few long lines, in SRT, WebVTT and ASS bodies)
at close to the default size cap, and require each to return, accepted or rejected, in a
generous time bound.
"""

from __future__ import annotations

import contextlib
import itertools
import time
from collections.abc import Callable

import pytest

from video_cleaner.subtitles.textfmt import SubtitleRejected, normalise_subtitle, to_srt

SIZE_CAP = 2_000_000
TIME_BOUND_S = 2.0
SRT_HEAD = b"1\n00:00:01,000 --> 00:00:02,000\n"
VTT_HEAD = b"WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n"
ASS_HEAD = (
    b"[Script Info]\nScriptType: v4.00+\n\n[V4+ Styles]\n"
    b"Format: Name, Fontname\nStyle: Default,Arial\n\n[Events]\n"
    b"Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
)
ASS_DIALOGUE = b"Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,"


def _timed(raw: bytes) -> float:
    started = time.perf_counter()
    with contextlib.suppress(SubtitleRejected):
        normalise_subtitle(raw, max_bytes=SIZE_CAP, min_cues=1)
    return time.perf_counter() - started


def _fill(unit: bytes, line_chars: int, total: int) -> bytes:
    """Lines of about *line_chars* characters made from *unit*, about *total* bytes in all."""
    line = (unit * max(1, line_chars // len(unit)))[:line_chars] + b"\n"
    return line * max(1, total // len(line))


def test_a_lt_followed_by_a_long_space_run_in_every_line_is_bounded() -> None:
    # One "<" per line followed by 19,000 spaces and a letter: 10 lines took 38 s to parse when
    # the SRT tag-stripping regular expression of the parsing library ran on such text.
    raw = SRT_HEAD + (b"<" + b" " * 19_000 + b"a\n") * 10
    assert _timed(raw) < TIME_BOUND_S


def test_the_same_shape_at_the_size_cap_is_bounded() -> None:
    raw = SRT_HEAD + (b"<" + b" " * 19_000 + b"a\n") * (SIZE_CAP // 19_002)
    assert _timed(raw) < TIME_BOUND_S


def test_the_same_shape_without_a_letter_is_bounded() -> None:
    raw = SRT_HEAD + (b"<" + b" " * 19_000 + b"\n") * (SIZE_CAP // 19_002)
    assert _timed(raw) < TIME_BOUND_S


def test_an_ass_section_heading_shape_is_bounded() -> None:
    # A line opening with "[" and a long run of letters: a section-heading pattern of the parsing
    # library backtracks quadratically on it.
    raw = ASS_HEAD + (b"[" + b"a" * 19_000 + b"\n") * (SIZE_CAP // 19_002)
    assert _timed(raw) < TIME_BOUND_S


OPENERS = [b"<", b"{", b"&", b"\\", b"["]
FILLERS = [b" ", b"\t", b"a", b"/", b"1", b"<", b"{", b"\\N", b"-->", b":"]


def _bodies(unit: bytes, line_chars: int) -> list[bytes]:
    text = _fill(unit, line_chars, SIZE_CAP // 2)
    return [
        SRT_HEAD + text,
        VTT_HEAD + text,
        ASS_HEAD + ASS_DIALOGUE + text.replace(b"\n", b"\\N") + b"\n",
        ASS_HEAD + text,
    ]


@pytest.mark.parametrize(
    ("opener", "filler"),
    list(itertools.product(OPENERS, FILLERS)),
    ids=lambda b: repr(b)[2:-1],
)
def test_shape_family_sweep_is_bounded(opener: bytes, filler: bytes) -> None:
    slowest = 0.0
    for line_chars in (19_000, 24):
        for unit in (opener + filler * 9_000 + b"a", (opener + filler) * 9_000):
            for raw in _bodies(unit, line_chars):
                slowest = max(slowest, _timed(raw))
    assert slowest < TIME_BOUND_S


def test_ordinary_tags_are_stripped_the_same_way_as_before() -> None:
    raw = (
        b'1\n00:00:01,000 --> 00:00:02,000\n<i>It\'s</i> <font color="#ff0000">red</font>'
        b" and <b>bold</b>\n<u>under</u>{\\an8} line\n\n"
        b"2\n00:00:03,000 --> 00:00:04,000\n- Hi <3\n- 5 < 6 > 4\n"
    )
    result = normalise_subtitle(raw, max_bytes=SIZE_CAP, min_cues=2)
    assert [c.text for c in result.cues] == [
        "It's red and bold\nunder line",
        "- Hi <3\n- 5 < 6 > 4",
    ]


def test_thousands_of_tag_rich_cues_are_accepted_quickly() -> None:
    blocks = []
    for i in range(1, 5_001):
        stamp = f"{i // 3600:02d}:{i // 60 % 60:02d}:{i % 60:02d}"
        blocks.append(
            f"{i}\n{stamp},000 --> {stamp},900\n"
            f'<i>Line {i}</i> <font color="#ffffff"><b>bold</b></font>\n<u>second</u>\n'
        )
    raw = "\n".join(blocks).encode()
    started = time.perf_counter()
    result = normalise_subtitle(raw, max_bytes=SIZE_CAP, min_cues=5_000)
    assert time.perf_counter() - started < TIME_BOUND_S
    assert len(result.cues) == 5_000
    assert result.cues[4_999].text == "Line 5000 bold\nsecond"


# --- the normalised text reads back as the same cues ----------------------------------------

HOSTILE_TEXTS = [
    "<i>00:59:00,000 --> 00:59:05,000</i>",
    "before\n<b>00:00:09,000 --> 00:00:10,000</b>\nafter",
    "<i>100:00:00,000 --> 100:00:01,000</i>",
    "<i>99999999999999:00:00,000 --> 00:00:01,000</i>",
    "7\n<i>00:00:09,000 --> 00:00:10,000</i>",
    "7\n<i>00:00:09,000 --> 00:00:10,000</i>\n8",
    "<i> 00:59:00.000  -->  00:59:05.000 </i> trailing words",
    "59:00,000 --> 59:05,000",
    "00:59:00,000 --> 00:59:05,000 and then words",
    "a --> b",
    "-->",
    "dash ---> arrow --> done",
    "line one\n   \n\t\nline two",
    "line one\u2028line two\u2029line three",
    "5",
    "5\n\n6",
    "<<i>b>",
    "<i<b>>text",
    "{\\{\\a}}x",
    "{{\\an8}name}",
    "<\x01i>gone</i>",
    "<" * 6 + "i>" * 6 + "x",
    "{\\" * 6 + "a}" * 6 + "x",
    "I <3 you, 2 < 3 and {name}",
]


def _srt_with_cue_text(text: str) -> bytes:
    cues = [
        "1\n00:00:01,000 --> 00:00:02,000\nfirst\n",
        f"2\n00:00:03,000 --> 00:00:04,000\n{text}\n",
        "3\n00:00:05,000 --> 00:00:06,000\nlast\n",
    ]
    return "\n".join(cues).encode()


def _vtt_with_cue_text(text: str) -> bytes:
    # A blank line ends a WebVTT cue, so the hostile text carries no blank lines here.
    text = "\n".join(line for line in text.split("\n") if line.strip())
    return (
        b"WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nfirst\n\n"
        + f"00:00:03.000 --> 00:00:04.000\n{text}\n\n".encode()
        + b"00:00:05.000 --> 00:00:06.000\nlast\n"
    )


def _ass_with_cue_text(text: str) -> bytes:
    one_line = text.replace("\n", "\\N")
    return (
        ASS_HEAD
        + b"Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,first\n"
        + b"Dialogue: 0,0:00:03.00,0:00:04.00,Default,,0,0,0,,"
        + one_line.encode()
        + b"\n"
        + b"Dialogue: 0,0:00:05.00,0:00:06.00,Default,,0,0,0,,last\n"
    )


@pytest.mark.parametrize("build", [_srt_with_cue_text, _vtt_with_cue_text, _ass_with_cue_text])
@pytest.mark.parametrize("text", HOSTILE_TEXTS, ids=lambda t: repr(t)[:40])
def test_normalised_text_reads_back_as_the_same_cues(
    text: str, build: Callable[[str], bytes]
) -> None:
    first = normalise_subtitle(build(text), max_bytes=SIZE_CAP, min_cues=1)
    again = normalise_subtitle(to_srt(first.cues).encode(), max_bytes=SIZE_CAP, min_cues=1)
    assert again.cues == first.cues
    assert again.srt == first.srt
    # no cue the file did not have: first and last are always there, the hostile one at most
    assert len(first.cues) <= 3


def test_markup_nested_deeper_than_the_passes_loses_its_brackets_in_bounded_time() -> None:
    nested = "<" * 6_000 + "i>" * 6_000
    started = time.perf_counter()
    first = normalise_subtitle(_srt_with_cue_text(nested), max_bytes=SIZE_CAP, min_cues=1)
    assert time.perf_counter() - started < TIME_BOUND_S
    assert all("<" not in cue.text and ">" not in cue.text for cue in first.cues)
    again = normalise_subtitle(first.srt.encode(), max_bytes=SIZE_CAP, min_cues=1)
    assert again.cues == first.cues


def test_a_timestamp_line_built_by_tag_removal_is_not_written_as_a_timing_line() -> None:
    first = normalise_subtitle(
        _srt_with_cue_text("<i>00:59:00,000 --> 00:59:05,000</i>"), max_bytes=SIZE_CAP, min_cues=1
    )
    assert [c.text for c in first.cues] == ["first", "last"]
    assert "00:59:00" not in first.srt
