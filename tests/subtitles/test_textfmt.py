"""Subtitle bytes in, clean UTF-8 SRT out; hostile or odd input is rejected, never executed."""

from __future__ import annotations

import pysubs2
import pytest

from video_cleaner.subtitles import textfmt
from video_cleaner.subtitles.textfmt import (
    Cue,
    SubtitleRejected,
    cues_per_hour,
    decode_subtitle_bytes,
    normalise_subtitle,
    to_srt,
)

BIG = 1_000_000

SRT = (
    b"1\r\n00:00:01,000 --> 00:00:02,500\r\nHello there\r\n\r\n"
    b"2\n00:00:03,000 --> 00:00:04,000\nGeneral Kenobi\n\n"
)

ASS = (
    b"[Script Info]\nScriptType: v4.00+\n\n"
    b"[V4+ Styles]\n"
    b"Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
    b"BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
    b"BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
    b"Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,"
    b"100,100,0,0,1,2,2,2,10,10,10,1\n\n"
    b"[Events]\n"
    b"Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    b"Dialogue: 0,0:00:03.00,0:00:04.00,Default,,0,0,0,,{\\an8\\i1}Second{\\i0} line\\Nwraps\n"
    b"Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,First\n"
)

VTT = (
    b"WEBVTT\n\n"
    b"intro\n00:01.000 --> 00:02.000\n<v Anna>Hello <i>there</i></v>\n\n"
    b"00:00:03.000 --> 00:00:04.000 align:start\n<c.yellow>Second</c>\n\n"
)


def srt_with(*cues: tuple[str, str, str]) -> bytes:
    blocks = [f"{i}\n{s} --> {e}\n{t}\n" for i, (s, e, t) in enumerate(cues, start=1)]
    return "\n".join(blocks).encode()


# --- decode_subtitle_bytes ----------------------------------------------------------------


def test_decode_plain_utf8() -> None:
    assert decode_subtitle_bytes("café".encode()) == "café"


def test_decode_utf8_bom_is_removed() -> None:
    assert decode_subtitle_bytes(b"\xef\xbb\xbfcaf\xc3\xa9") == "café"


def test_decode_utf16_little_endian_bom() -> None:
    assert decode_subtitle_bytes(b"\xff\xfe" + "café".encode("utf-16-le")) == "café"


def test_decode_utf16_big_endian_bom() -> None:
    assert decode_subtitle_bytes(b"\xfe\xff" + "café".encode("utf-16-be")) == "café"


def test_decode_truncated_utf16_does_not_raise() -> None:
    assert decode_subtitle_bytes(b"\xff\xfeA\x00B").startswith("A")


def test_decode_cp1252_fallback() -> None:
    assert decode_subtitle_bytes(b"caf\xe9") == "café"


def test_decode_cp1252_specific_characters() -> None:
    assert decode_subtitle_bytes(b"\x93quoted\x94") == "“quoted”"


def test_decode_latin1_last_resort_never_raises() -> None:
    # 0x81 is undefined in cp1252, so the latin-1 step must catch it.
    assert decode_subtitle_bytes(b"a\x81b") == "a\x81b"


def test_decode_mixed_line_endings_become_lf() -> None:
    assert decode_subtitle_bytes(b"a\r\nb\nc\rd") == "a\nb\nc\nd"


def test_decode_empty_bytes() -> None:
    assert decode_subtitle_bytes(b"") == ""


# --- normalise_subtitle: formats --------------------------------------------------------


def test_srt_is_detected_and_returned_in_order() -> None:
    result = normalise_subtitle(SRT, max_bytes=BIG, min_cues=2)
    assert result.source_format == "srt"
    assert result.cues == (Cue(1000, 2500, "Hello there"), Cue(3000, 4000, "General Kenobi"))
    assert result.srt == (
        "1\n00:00:01,000 --> 00:00:02,500\nHello there\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nGeneral Kenobi\n"
    )


def test_ass_is_detected_sorted_and_stripped_of_override_tags() -> None:
    result = normalise_subtitle(ASS, max_bytes=BIG, min_cues=2)
    assert result.source_format == "ass"
    assert result.cues == (Cue(1000, 2000, "First"), Cue(3000, 4000, "Second line\nwraps"))


def test_vtt_is_detected_and_its_markup_stripped() -> None:
    result = normalise_subtitle(VTT, max_bytes=BIG, min_cues=2)
    assert result.source_format == "vtt"
    assert [c.text for c in result.cues] == ["Hello there", "Second"]
    assert result.cues[0].start_ms == 1000
    assert result.cues[1].end_ms == 4000


def test_html_tags_are_stripped_from_srt_text() -> None:
    raw = srt_with(
        ("00:00:01,000", "00:00:02,000", '<i>Hello</i> <font color="red">world</font>'),
        ("00:00:03,000", "00:00:04,000", "<b>Bold</b> &amp; plain"),
    )
    result = normalise_subtitle(raw, max_bytes=BIG, min_cues=2)
    assert result.cues[0].text == "Hello world"
    assert result.cues[1].text == "Bold &amp; plain"


def test_a_lone_angle_bracket_is_not_a_tag() -> None:
    raw = srt_with(("00:00:01,000", "00:00:02,000", "I <3 you, 2 < 3"))
    result = normalise_subtitle(raw, max_bytes=BIG, min_cues=1)
    assert result.cues[0].text == "I <3 you, 2 < 3"


def test_leading_credit_cue_is_kept_unchanged() -> None:
    raw = srt_with(
        ("00:00:00,500", "00:00:02,000", "Subtitles by Some Group - www.example.org"),
        ("00:00:03,000", "00:00:04,000", "Real dialogue"),
    )
    result = normalise_subtitle(raw, max_bytes=BIG, min_cues=2)
    assert result.cues[0] == Cue(500, 2000, "Subtitles by Some Group - www.example.org")


def test_utf16_encoded_srt_is_normalised_to_utf8_srt() -> None:
    raw = b"\xff\xfe" + SRT.decode().encode("utf-16-le")
    result = normalise_subtitle(raw, max_bytes=BIG, min_cues=2)
    assert result.cues[0].text == "Hello there"
    assert result.srt.encode("utf-8").decode("utf-8") == result.srt


def test_cp1252_srt_text_is_decoded() -> None:
    raw = b"1\n00:00:01,000 --> 00:00:02,000\ncaf\xe9\n"
    assert normalise_subtitle(raw, max_bytes=BIG, min_cues=1).cues[0].text == "café"


def test_cues_are_sorted_by_start_time() -> None:
    raw = srt_with(
        ("00:00:05,000", "00:00:06,000", "later"),
        ("00:00:01,000", "00:00:02,000", "earlier"),
    )
    result = normalise_subtitle(raw, max_bytes=BIG, min_cues=2)
    assert [c.text for c in result.cues] == ["earlier", "later"]


def test_cues_with_no_text_left_are_dropped() -> None:
    raw = srt_with(
        ("00:00:01,000", "00:00:02,000", "<i></i>"),
        ("00:00:03,000", "00:00:04,000", "kept"),
    )
    result = normalise_subtitle(raw, max_bytes=BIG, min_cues=1)
    assert [c.text for c in result.cues] == ["kept"]


def test_ass_comment_and_drawing_events_are_dropped() -> None:
    body = ASS.rsplit(b"Dialogue:", 2)[0]
    raw = (
        body
        + b"Comment: 0,0:00:00.50,0:00:00.90,Default,,0,0,0,,not shown\n"
        + b"Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{\\p1}m 0 0 l 10 10{\\p0}\n"
        + b"Dialogue: 0,0:00:03.00,0:00:04.00,Default,,0,0,0,,Spoken\n"
    )
    result = normalise_subtitle(raw, max_bytes=BIG, min_cues=1)
    assert [c.text for c in result.cues] == ["Spoken"]


def test_blank_lines_inside_a_cue_do_not_survive() -> None:
    raw = b"1\n00:00:01,000 --> 00:00:02,000\nline one\n   \nline two\n"
    result = normalise_subtitle(raw, max_bytes=BIG, min_cues=1)
    assert result.cues[0].text == "line one\nline two"


def test_normalised_srt_parses_back_to_the_same_cues() -> None:
    first = normalise_subtitle(ASS, max_bytes=BIG, min_cues=2)
    again = normalise_subtitle(first.srt.encode(), max_bytes=BIG, min_cues=2)
    assert again.cues == first.cues
    assert again.source_format == "srt"


def test_format_comes_from_content_not_a_name() -> None:
    # The API takes bytes only; an ASS body is recognised as ASS whatever it was called.
    assert normalise_subtitle(ASS, max_bytes=BIG, min_cues=1).source_format == "ass"
    assert normalise_subtitle(VTT, max_bytes=BIG, min_cues=1).source_format == "vtt"


# --- normalise_subtitle: rejections ---------------------------------------------------


def test_rejects_more_than_max_bytes() -> None:
    with pytest.raises(SubtitleRejected, match="larger than"):
        normalise_subtitle(SRT, max_bytes=len(SRT) - 1, min_cues=1)


def test_size_is_checked_before_decoding(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(raw: bytes) -> str:
        raise AssertionError("decoded an oversized input")

    monkeypatch.setattr(textfmt, "decode_subtitle_bytes", boom)
    with pytest.raises(SubtitleRejected, match="larger than"):
        normalise_subtitle(b"x" * 100, max_bytes=10, min_cues=1)


def test_input_exactly_max_bytes_is_accepted() -> None:
    assert normalise_subtitle(SRT, max_bytes=len(SRT), min_cues=2).cues


@pytest.mark.parametrize("raw", [b"", b"   \r\n\t\n", b"\xef\xbb\xbf"])
def test_rejects_empty_input(raw: bytes) -> None:
    with pytest.raises(SubtitleRejected, match="empty"):
        normalise_subtitle(raw, max_bytes=BIG, min_cues=1)


def test_rejects_binary_content_with_nul_bytes() -> None:
    raw = bytes(range(256)) * 8
    with pytest.raises(SubtitleRejected, match="binary"):
        normalise_subtitle(raw, max_bytes=BIG, min_cues=1)


def test_rejects_content_with_many_control_characters() -> None:
    raw = (b"\x01\x02\x03\x04" * 50) + SRT
    with pytest.raises(SubtitleRejected, match="binary"):
        normalise_subtitle(raw, max_bytes=BIG, min_cues=1)


def test_a_lone_stray_control_character_is_tolerated() -> None:
    raw = b"1\n00:00:01,000 --> 00:00:02,000\nhello\x07there\n"
    assert normalise_subtitle(raw, max_bytes=BIG, min_cues=1).cues


@pytest.mark.parametrize(
    "raw",
    [
        b"<!DOCTYPE html><html><body>Not found</body></html>",
        b"  \n<html lang='en'><head></head><body>Sign in</body></html>",
        b"<HTML><BODY>Rate limit exceeded</BODY></HTML>",
    ],
)
def test_rejects_an_html_page(raw: bytes) -> None:
    with pytest.raises(SubtitleRejected, match="HTML"):
        normalise_subtitle(raw, max_bytes=BIG, min_cues=1)


def test_rejects_fewer_than_min_cues() -> None:
    with pytest.raises(SubtitleRejected, match=r"2 cue\(s\), fewer than 3"):
        normalise_subtitle(SRT, max_bytes=BIG, min_cues=3)


def test_rejects_cues_whose_end_precedes_start() -> None:
    raw = srt_with(("00:00:05,000", "00:00:02,000", "backwards"))
    with pytest.raises(SubtitleRejected, match="end before"):
        normalise_subtitle(raw, max_bytes=BIG, min_cues=1)


def test_rejects_timestamps_beyond_100_hours() -> None:
    raw = srt_with(
        ("00:00:01,000", "00:00:02,000", "fine"),
        ("100:00:01,000", "100:00:02,000", "absurd"),
    )
    with pytest.raises(SubtitleRejected, match="100 hours"):
        normalise_subtitle(raw, max_bytes=BIG, min_cues=1)


def test_timestamps_up_to_100_hours_are_accepted() -> None:
    raw = srt_with(("99:59:58,000", "99:59:59,999", "edge"))
    assert normalise_subtitle(raw, max_bytes=BIG, min_cues=1).cues[0].end_ms == 359_999_999


def test_rejects_text_that_is_not_a_subtitle_format() -> None:
    with pytest.raises(SubtitleRejected, match="not a recognised subtitle format"):
        normalise_subtitle(b"just some prose\nwith two lines\n", max_bytes=BIG, min_cues=1)


@pytest.mark.parametrize(
    "raw",
    [
        b'<?xml version="1.0"?><tt xmlns="http://www.w3.org/ns/ttml"><div/></tt>',
        b"{1}{25}hello\n{30}{50}there\n",
        b"[1000][2000]hello\n",
        b"<SAMI><SYNC Start=1000><P>x</SYNC></SAMI>",
    ],
)
def test_rejects_formats_outside_the_supported_set(raw: bytes) -> None:
    with pytest.raises(SubtitleRejected, match="not supported"):
        normalise_subtitle(raw, max_bytes=BIG, min_cues=1)


def test_rejects_json_that_is_not_a_subtitle() -> None:
    with pytest.raises(SubtitleRejected, match="not a recognised subtitle format"):
        normalise_subtitle(b'{"segments": [{"start": 1.0}]}', max_bytes=BIG, min_cues=1)


def test_a_detector_failure_becomes_a_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(text: str) -> str:
        raise RuntimeError("detector exploded: secret text")

    monkeypatch.setattr(textfmt, "autodetect_format", boom)
    with pytest.raises(SubtitleRejected, match="could not be parsed") as info:
        normalise_subtitle(SRT, max_bytes=BIG, min_cues=1)
    assert "secret" not in str(info.value)


@pytest.mark.parametrize(
    ("start", "end"), [(-1, 500), (0, 360_000_001), (360_000_001, 360_000_002)]
)
def test_rejects_parsed_timestamps_out_of_range(
    monkeypatch: pytest.MonkeyPatch, start: int, end: int
) -> None:
    subs = pysubs2.SSAFile()
    subs.events.append(pysubs2.SSAEvent(start=start, end=end, text="x"))
    monkeypatch.setattr(pysubs2.SSAFile, "from_string", lambda *args, **kwargs: subs)
    with pytest.raises(SubtitleRejected, match="100 hours or below zero"):
        normalise_subtitle(SRT, max_bytes=BIG, min_cues=1)


def test_rejects_too_many_cues(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(textfmt, "MAX_CUES", 1)
    with pytest.raises(SubtitleRejected, match="too many cues"):
        normalise_subtitle(SRT, max_bytes=BIG, min_cues=1)


def test_rejects_an_overlong_line_before_parsing() -> None:
    raw = b"1\n00:00:01,000 --> 00:00:02,000\n" + b"a" * (textfmt.MAX_LINE_CHARS + 1) + b"\n"
    with pytest.raises(SubtitleRejected, match="line longer than"):
        normalise_subtitle(raw, max_bytes=BIG * 10, min_cues=1)


def test_rejection_messages_never_echo_subtitle_text() -> None:
    secret = b"TOPSECRETDIALOGUE"
    raw = b"1\n00:00:05,000 --> 00:00:02,000\n" + secret + b"\n"
    with pytest.raises(SubtitleRejected) as info:
        normalise_subtitle(raw, max_bytes=BIG, min_cues=1)
    assert "TOPSECRET" not in str(info.value)


def test_hostile_text_is_never_evaluated() -> None:
    raw = srt_with(("00:00:01,000", "00:00:02,000", "__import__('os').system('echo hi')"))
    result = normalise_subtitle(raw, max_bytes=BIG, min_cues=1)
    assert result.cues[0].text == "__import__('os').system('echo hi')"


def test_a_parser_failure_becomes_a_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("parser exploded: secret text")

    monkeypatch.setattr(pysubs2.SSAFile, "from_string", boom)
    with pytest.raises(SubtitleRejected, match="could not be parsed") as info:
        normalise_subtitle(SRT, max_bytes=BIG, min_cues=1)
    assert "secret" not in str(info.value)


# --- cues_per_hour ---------------------------------------------------------------------


def test_cues_per_hour() -> None:
    cues = [Cue(i * 1000, i * 1000 + 500, "x") for i in range(300)]
    assert cues_per_hour(cues, 1800.0) == 600.0


def test_cues_per_hour_of_no_cues_is_zero() -> None:
    assert cues_per_hour([], 3600.0) == 0.0


@pytest.mark.parametrize("duration", [0.0, -5.0, float("nan")])
def test_cues_per_hour_rejects_a_non_positive_duration(duration: float) -> None:
    with pytest.raises(ValueError, match="duration"):
        cues_per_hour([Cue(0, 1, "x")], duration)


# --- to_srt ----------------------------------------------------------------------------


def test_to_srt_formats_timestamps_and_numbers_cues() -> None:
    cues = [Cue(3_723_004, 3_725_000, "a\nb"), Cue(3_726_000, 3_727_999, "c")]
    assert to_srt(cues) == (
        "1\n01:02:03,004 --> 01:02:05,000\na\nb\n\n2\n01:02:06,000 --> 01:02:07,999\nc\n"
    )


def test_to_srt_of_nothing_is_empty() -> None:
    assert to_srt([]) == ""


def test_to_srt_cannot_be_split_by_blank_lines_in_text() -> None:
    assert to_srt([Cue(0, 1000, "a\n\n\nb")]) == "1\n00:00:00,000 --> 00:00:01,000\na\nb\n"
