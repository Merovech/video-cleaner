"""verify_output: probe, duration, streams, decode check and size.

The unit tests inject the probe and decode functions; the ``integration`` tests at the end run
the real decode check and a real encode + mux on generated clips.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from media import make_clip
from video_cleaner import runner as runner_module
from video_cleaner import verify as verify_module
from video_cleaner.config import Settings
from video_cleaner.consolefmt import Level
from video_cleaner.encode import encode_full
from video_cleaner.mux import SubtitleInput
from video_cleaner.probe import (
    AudioStream,
    MediaInfo,
    ProbeError,
    SubtitleStream,
    VideoStream,
    probe,
)
from video_cleaner.progress import Progress, Reporter
from video_cleaner.tools import ToolResult, Tools, locate_tools
from video_cleaner.verify import TEN_BIT_PIX_FMTS, VerifyResult, decode_errors, verify_output

TOOLS = Tools(ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe"), mkvmerge=Path("mkvmerge"))
GIB = 1024**3


class NullReporter:
    def stage(self, label: str) -> None:
        pass

    def update(self, p: Progress) -> None:
        pass

    def heartbeat(self, label: str, elapsed_s: float) -> None:
        pass

    def file_done(self, summary: str, level: Level = Level.INFO) -> None:
        pass


REPORTER = NullReporter()


def _info(
    path: Path,
    *,
    size: int = 4 * GIB,
    duration_s: float = 6000.0,
    codec: str = "h264",
    pix_fmt: str = "yuv420p",
    video: bool = True,
    audio: int = 2,
    subtitles: int = 1,
    chapters: int = 12,
    cover_art: int = 0,
    extra_video_streams: int = 0,
) -> MediaInfo:
    return MediaInfo(
        path=path,
        size=size,
        mtime=1_700_000_000.0,
        duration_s=duration_s,
        container="matroska,webm",
        video=VideoStream(
            codec=codec,
            width=1920,
            height=1080,
            pix_fmt=pix_fmt,
            fps=23.976,
            bit_rate=None,
            transfer="bt709",
            dv_profile=None,
            field_order="progressive",
        )
        if video
        else None,
        audio=tuple(
            AudioStream(index=1 + i, codec="ac3", language="eng", title="") for i in range(audio)
        ),
        subtitles=tuple(
            SubtitleStream(
                index=10 + i,
                codec="subrip",
                language="eng",
                forced=False,
                sdh=False,
                is_image=False,
            )
            for i in range(subtitles)
        ),
        chapters=chapters,
        cover_art=cover_art,
        extra_video_streams=extra_video_streams,
    )


ORIGINAL = _info(Path("D:/scratch/movie.mkv"))
OUTPUT = Path("D:/scratch/work/movie.mkv")


def _probe_returning(info: MediaInfo) -> Any:
    def probe_fn(path: Path, tools: Tools, reporter: Reporter) -> MediaInfo:
        assert path == OUTPUT
        return info

    return probe_fn


def _clean_decode(tools: Tools, output: Path, reporter: Reporter) -> str:
    return ""


def _verify(out: MediaInfo, *, encoded: bool = True, **kwargs: Any) -> VerifyResult:
    return verify_output(
        TOOLS,
        ORIGINAL,
        OUTPUT,
        REPORTER,
        encoded=encoded,
        min_saving=0.10,
        probe_fn=_probe_returning(out),
        decode_fn=kwargs.pop("decode_fn", _clean_decode),
        **kwargs,
    )


def _good_output(**changes: Any) -> MediaInfo:
    base = _info(OUTPUT, size=2 * GIB, codec="hevc", pix_fmt="yuv420p10le")
    return dataclasses.replace(base, **changes)


def test_verify_accepts_good_encode() -> None:
    result = _verify(_good_output())
    assert result == VerifyResult(ok=True, reason="")


def test_verify_rejects_output_not_10_percent_smaller() -> None:
    almost = int(ORIGINAL.size * 0.91)  # only 9% smaller
    result = _verify(_good_output(size=almost))
    assert not result.ok
    assert result.too_large
    assert "9.0% smaller" in result.reason
    assert "10%" in result.reason


def test_verify_accepts_exactly_10_percent_smaller() -> None:
    assert _verify(_good_output(size=int(ORIGINAL.size * 0.90))).ok


def test_verify_size_not_checked_for_remux() -> None:
    result = _verify(_good_output(size=ORIGINAL.size + 1000), encoded=False)
    assert result.ok


def test_verify_rejects_truncated_output() -> None:
    result = _verify(_good_output(duration_s=3000.0))
    assert not result.ok
    assert not result.too_large
    assert "duration" in result.reason
    assert "3000.0" in result.reason and "6000.0" in result.reason


def test_verify_duration_tolerance_is_one_percent_plus_two_seconds() -> None:
    tolerance = 6000.0 * 0.01 + 2.0
    assert _verify(_good_output(duration_s=6000.0 - tolerance + 0.01)).ok
    assert _verify(_good_output(duration_s=6000.0 + tolerance - 0.01)).ok
    assert not _verify(_good_output(duration_s=6000.0 - tolerance - 0.01)).ok
    assert not _verify(_good_output(duration_s=6000.0 + tolerance + 0.01)).ok


def test_verify_multi_part_uses_duration_and_size_overrides() -> None:
    combined = _good_output(duration_s=9000.0, size=5 * GIB)
    result = _verify(combined, original_duration_s=9000.0, original_size=8 * GIB)
    assert result.ok
    # the override size is what the saving is measured against
    result = _verify(combined, original_duration_s=9000.0, original_size=int(5.2 * GIB))
    assert not result.ok and result.too_large


@pytest.mark.parametrize(
    ("changes", "words"),
    [
        ({"video": None}, "no video stream"),
        ({"audio": ()}, "audio streams: 0 in the output, 2 in the original"),
        ({"subtitles": ()}, "subtitle streams: 0 in the output, 1 in the original"),
        ({"chapters": 0}, "chapters: 0 in the output, 12 in the original"),
    ],
)
def test_verify_rejects_missing_streams(changes: dict[str, Any], words: str) -> None:
    result = _verify(_good_output(**changes))
    assert not result.ok
    assert not result.too_large
    assert words in result.reason


NEW_TITLE = "English (OpenSubtitles)"


def _with_new_subtitle(out: MediaInfo, **changes: Any) -> MediaInfo:
    """*out* with one more subtitle stream (the added track) after its existing ones."""
    values: dict[str, Any] = {
        "index": 99,
        "codec": "subrip",
        "language": "eng",
        "forced": False,
        "sdh": False,
        "is_image": False,
        "default": False,
        "title": NEW_TITLE,
    }
    values.update(changes)
    return dataclasses.replace(out, subtitles=(*out.subtitles, SubtitleStream(**values)))


def test_verify_accepts_one_new_subtitle() -> None:
    out = _with_new_subtitle(_good_output())
    assert len(out.subtitles) == 2
    result = _verify(out, expected_new_subtitles=1, expected_subtitle_title=NEW_TITLE)
    assert result == VerifyResult(ok=True, reason="")


def test_verify_accepts_one_new_subtitle_on_a_remux() -> None:
    out = _with_new_subtitle(_info(OUTPUT, size=4 * GIB))
    assert _verify(out, encoded=False, expected_new_subtitles=1).ok


def test_verify_rejects_the_missing_new_subtitle_naming_the_streams() -> None:
    result = _verify(_good_output(), expected_new_subtitles=1)
    assert not result.ok
    assert (
        "subtitle streams: 1 in the output, 2 expected (1 in the original + 1 new)" in result.reason
    )


def test_verify_without_expected_new_subtitles_rejects_an_extra_one() -> None:
    result = _verify(_with_new_subtitle(_good_output()))
    assert not result.ok and "subtitle streams: 2 in the output, 1 in the original" in result.reason


def test_verify_rejects_two_new_subtitles_when_one_was_expected() -> None:
    out = _with_new_subtitle(_with_new_subtitle(_good_output()))
    result = _verify(out, expected_new_subtitles=1)
    assert not result.ok and "subtitle streams: 3 in the output" in result.reason


@pytest.mark.parametrize(
    ("changes", "words"),
    [
        ({"language": "fre"}, "language fre, expected eng"),
        ({"language": "und"}, "language und, expected eng"),
        ({"default": True}, "default"),
        ({"forced": True}, "forced"),
    ],
)
def test_verify_rejects_a_new_track_with_the_wrong_language_or_flags(
    changes: dict[str, Any], words: str
) -> None:
    out = _with_new_subtitle(_good_output(), **changes)
    result = _verify(out, expected_new_subtitles=1, expected_subtitle_title=NEW_TITLE)
    assert not result.ok
    assert "new subtitle stream" in result.reason and words in result.reason


@pytest.mark.parametrize("title", ["", "Some other title"])
def test_verify_rejects_a_new_track_with_a_missing_or_wrong_title(title: str) -> None:
    out = _with_new_subtitle(_good_output(), title=title)
    result = _verify(out, expected_new_subtitles=1, expected_subtitle_title=NEW_TITLE)
    assert not result.ok
    assert f"title {title!r}, expected {NEW_TITLE!r}" in result.reason


def test_verify_skips_the_title_check_when_the_probe_does_not_report_titles() -> None:
    out = _with_new_subtitle(_good_output(), title=None)
    assert _verify(out, expected_new_subtitles=1, expected_subtitle_title=NEW_TITLE).ok


def test_verify_checks_the_last_subtitle_stream_as_the_new_one() -> None:
    # an original eng track with default=True first is fine; only the last one is "new"
    first = SubtitleStream(
        index=5,
        codec="subrip",
        language="fre",
        forced=True,
        sdh=False,
        is_image=False,
        default=True,
    )
    out = _good_output()
    out = dataclasses.replace(out, subtitles=(first,))
    original = dataclasses.replace(ORIGINAL, subtitles=(first,))
    out = _with_new_subtitle(out)
    result = verify_output(
        TOOLS,
        original,
        OUTPUT,
        REPORTER,
        encoded=True,
        min_saving=0.10,
        probe_fn=_probe_returning(out),
        decode_fn=_clean_decode,
        expected_new_subtitles=1,
        expected_subtitle_title=NEW_TITLE,
    )
    assert result.ok, result.reason


@pytest.mark.parametrize("bad", [-1, 2, 5])
def test_verify_refuses_other_expected_counts(bad: int) -> None:
    with pytest.raises(ValueError, match="expected_new_subtitles"):
        _verify(_good_output(), expected_new_subtitles=bad)


def test_verify_a_subtitle_problem_is_not_a_too_large_result() -> None:
    result = _verify(_good_output(size=3 * GIB), expected_new_subtitles=1)
    assert not result.ok and not result.too_large


def test_verify_no_chapters_needed_when_original_had_none() -> None:
    original = dataclasses.replace(ORIGINAL, chapters=0)
    result = verify_output(
        TOOLS,
        original,
        OUTPUT,
        REPORTER,
        encoded=True,
        min_saving=0.10,
        probe_fn=_probe_returning(_good_output(chapters=0)),
        decode_fn=_clean_decode,
    )
    assert result.ok


def test_verify_rejects_lost_cover_art_and_extra_video() -> None:
    original = dataclasses.replace(ORIGINAL, cover_art=1, extra_video_streams=1)
    result = verify_output(
        TOOLS,
        original,
        OUTPUT,
        REPORTER,
        encoded=False,
        min_saving=0.10,
        probe_fn=_probe_returning(_good_output(cover_art=0, extra_video_streams=0)),
        decode_fn=_clean_decode,
    )
    assert not result.ok
    assert "attachments" in result.reason
    assert "extra video streams" in result.reason


def test_verify_reports_every_problem_at_once() -> None:
    result = _verify(_good_output(duration_s=100.0, audio=(), size=ORIGINAL.size))
    assert not result.ok
    assert not result.too_large  # size is not the only problem
    assert "duration" in result.reason and "audio" in result.reason and "smaller" in result.reason


def test_verify_rejects_decode_errors() -> None:
    def broken(tools: Tools, output: Path, reporter: Reporter) -> str:
        return "Invalid NAL unit size"

    result = _verify(_good_output(), decode_fn=broken)
    assert not result.ok
    assert "decode check failed: Invalid NAL unit size" in result.reason


def test_verify_skips_decode_when_probe_fails() -> None:
    calls: list[Path] = []

    def probe_fn(path: Path, tools: Tools, reporter: Reporter) -> MediaInfo:
        raise ProbeError(path, "zero duration")

    def decode(tools: Tools, output: Path, reporter: Reporter) -> str:
        calls.append(output)
        return ""

    result = verify_output(
        TOOLS,
        ORIGINAL,
        OUTPUT,
        REPORTER,
        encoded=True,
        min_saving=0.10,
        probe_fn=probe_fn,
        decode_fn=decode,
    )
    assert not result.ok
    assert "cannot probe" in result.reason and "zero duration" in result.reason
    assert calls == []


def test_decode_errors_runs_ffmpeg_null_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Sequence[str]] = []

    def fake_run_tool(cmd: Sequence[str], **kwargs: Any) -> ToolResult:
        seen.append(cmd)
        assert kwargs["check"] is False
        return ToolResult(returncode=0, stdout="", stderr_tail="  \n")

    monkeypatch.setattr(verify_module, "run_tool", fake_run_tool)
    assert decode_errors(TOOLS, OUTPUT, REPORTER) == ""
    assert list(seen[0]) == [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(OUTPUT),
        "-map",
        "0:V",
        "-map",
        "0:a?",
        "-sn",
        "-dn",
        "-f",
        "null",
        "-",
    ]


def test_decode_errors_reports_stderr_and_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    results = iter(
        [
            ToolResult(returncode=0, stdout="", stderr_tail="error while decoding MB 1 2\n"),
            ToolResult(returncode=1, stdout="", stderr_tail=""),
        ]
    )
    monkeypatch.setattr(verify_module, "run_tool", lambda cmd, **kw: next(results))
    assert decode_errors(TOOLS, OUTPUT, REPORTER) == "error while decoding MB 1 2"
    assert decode_errors(TOOLS, OUTPUT, REPORTER) == "ffmpeg exited with code 1"


@pytest.mark.parametrize(
    ("codec", "pix_fmt", "problem"),
    [
        ("h264", "yuv420p10le", "codec h264"),
        ("hevc", "yuv420p", "pixel format yuv420p"),
        # "10" appears in yuv410p but it is an 8-bit format: no substring test
        ("hevc", "yuv410p", "pixel format yuv410p"),
        ("hevc", "yuv420p12le", "pixel format yuv420p12le"),
    ],
)
def test_verify_requires_10_bit_hevc_for_an_encode(codec: str, pix_fmt: str, problem: str) -> None:
    out = _info(OUTPUT, size=2 * GIB, codec=codec, pix_fmt=pix_fmt)
    result = _verify(out)
    assert not result.ok and not result.too_large
    assert problem in result.reason


@pytest.mark.parametrize(
    "pix_fmt", ["yuv420p10le", "yuv420p10be", "yuv422p10le", "yuv444p10le", "p010le"]
)
def test_verify_accepts_each_10_bit_format(pix_fmt: str) -> None:
    out = _info(OUTPUT, size=2 * GIB, codec="hevc", pix_fmt=pix_fmt)
    assert _verify(out).ok


def test_verify_remux_keeps_whatever_video_the_source_had() -> None:
    out = _info(OUTPUT, size=4 * GIB, codec="h264", pix_fmt="yuv420p")
    assert _verify(out, encoded=False).ok


# --- integration: real decode check and a real encode ---------------------------------------


@pytest.mark.integration
def test_real_decode_check_passes_on_mkv_with_srt_and_font(tmp_path: Path) -> None:
    tools = locate_tools()
    clip = make_clip(tools, tmp_path / "Rich [x] (2024).mkv", duration_s=3.0, rich=True)
    info = probe(clip, tools, REPORTER)
    assert len(info.subtitles) == 1
    assert info.cover_art >= 1
    assert info.chapters == 2
    assert decode_errors(tools, clip, REPORTER) == ""
    assert verify_output(tools, info, clip, REPORTER, encoded=False, min_saving=0.1).ok


@pytest.mark.integration
def test_real_encode_probes_as_10_bit_hevc_and_verifies(tmp_path: Path) -> None:
    tools = locate_tools()
    settings = Settings(
        library_root=tmp_path / "lib",
        scratch_dir=tmp_path / "scratch",
        completed_dir=tmp_path / "done",
        state_db=tmp_path / "s.db",
        log_dir=tmp_path / "logs",
    )
    clip = make_clip(tools, tmp_path / "src.mkv", duration_s=3.0, rich=True)
    info = probe(clip, tools, REPORTER)
    video = tmp_path / "video.mkv"
    encode_full(info, tools, settings, 24, video, REPORTER)
    output = tmp_path / "out.mkv"
    runner_module.mux(tools, [clip], video, output, REPORTER)
    out = probe(output, tools, REPORTER)
    assert out.video is not None
    assert out.video.codec == "hevc"
    assert out.video.pix_fmt in TEN_BIT_PIX_FMTS
    assert (len(out.audio), len(out.subtitles), out.chapters) == (1, 1, 2)
    assert out.cover_art == info.cover_art
    result = verify_output(tools, info, output, REPORTER, encoded=True, min_saving=0.1)
    assert result.ok, result.reason


@pytest.mark.integration
@pytest.mark.parametrize("title", ["English (OpenSubtitles)", '-He said "hi" & left'])
def test_real_mux_with_a_subtitle_adds_one_english_track_that_verifies(
    tmp_path: Path, title: str
) -> None:
    tools = locate_tools()
    clip = make_clip(tools, tmp_path / "Rich [x] (2024).mkv", duration_s=3.0, rich=True)
    srt = tmp_path / "staged.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello there.\n", encoding="utf-8")
    out = tmp_path / "out.mkv"
    runner_module.mux(tools, [clip], None, out, REPORTER, SubtitleInput(path=srt, title=title))

    info = probe(clip, tools, REPORTER)
    added = probe(out, tools, REPORTER)
    assert len(added.subtitles) == len(info.subtitles) + 1
    last = added.subtitles[-1]
    assert (last.language, last.default, last.forced, last.title) == ("eng", False, False, title)
    assert added.chapters == info.chapters and added.cover_art == info.cover_art
    verdict = verify_output(
        tools,
        info,
        out,
        REPORTER,
        encoded=False,
        min_saving=0.1,
        expected_new_subtitles=1,
        expected_subtitle_title=title,
    )
    assert verdict.ok, verdict.reason
    # without being told about the new track, verify rejects the output
    assert not verify_output(tools, info, out, REPORTER, encoded=False, min_saving=0.1).ok
