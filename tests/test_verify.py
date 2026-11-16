"""verify_output: probe, duration, streams, decode check and size, all against fakes.

No ffmpeg/ffprobe runs here (encoding is on hold): the probe and decode functions are injected.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from video_cleaner import verify as verify_module
from video_cleaner.probe import AudioStream, MediaInfo, ProbeError, SubtitleStream, VideoStream
from video_cleaner.progress import Progress, Reporter
from video_cleaner.tools import ToolResult, Tools
from video_cleaner.verify import VerifyResult, decode_errors, verify_output

TOOLS = Tools(ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe"), mkvmerge=Path("mkvmerge"))
GIB = 1024**3


class NullReporter:
    def stage(self, label: str) -> None:
        pass

    def update(self, p: Progress) -> None:
        pass

    def heartbeat(self, label: str, elapsed_s: float) -> None:
        pass

    def file_done(self, summary: str) -> None:
        pass


REPORTER = NullReporter()


def _info(
    path: Path,
    *,
    size: int = 4 * GIB,
    duration_s: float = 6000.0,
    codec: str = "h264",
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
            pix_fmt="yuv420p",
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
    base = _info(OUTPUT, size=2 * GIB, codec="hevc")
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
