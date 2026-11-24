"""Deciding from a file's own subtitle tracks whether it still needs an English subtitle."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from helpers import make_info
from media import RecordingReporter, make_clip
from video_cleaner.probe import MediaInfo, SubtitleStream, probe
from video_cleaner.subtitles import embedded
from video_cleaner.subtitles.embedded import (
    TEXT_SUBTITLE_CODECS,
    EmbeddedResult,
    EmbeddedVerdict,
    classify_embedded,
    density_candidates,
    is_english,
    measure_cue_density,
)
from video_cleaner.tools import ToolOutputError, ToolResult, Tools, locate_tools

THRESHOLD = 250.0
TOOLS = Tools(ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe"), mkvmerge=Path("mkvmerge"))
HOUR_S = 3600.0


def track(
    index: int,
    *,
    language: str = "eng",
    codec: str = "subrip",
    forced: bool = False,
    sdh: bool = False,
) -> SubtitleStream:
    return SubtitleStream(
        index=index,
        codec=codec,
        language=language,
        forced=forced,
        sdh=sdh,
        is_image=codec in {"hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle", "xsub"},
    )


def media(*tracks: SubtitleStream) -> MediaInfo:
    return replace(make_info(Path("movie.mkv"), 1000), subtitles=tuple(tracks))


def classify(*tracks: SubtitleStream, densities: dict[int, float] | None = None) -> EmbeddedResult:
    return classify_embedded(media(*tracks), densities or {}, threshold=THRESHOLD)


# --- is_english -------------------------------------------------------------------------------


@pytest.mark.parametrize("language", ["eng", "en", "english", "ENG", "En", "English", " eng "])
def test_english_language_tags_are_recognised(language: str) -> None:
    assert is_english(language)


@pytest.mark.parametrize("language", ["und", "fre", "fra", "spa", "", "engl", "enx"])
def test_other_language_tags_are_not_english(language: str) -> None:
    assert not is_english(language)


def test_text_codecs_are_the_plain_text_formats() -> None:
    assert set(TEXT_SUBTITLE_CODECS) == {
        "subrip",
        "srt",
        "ass",
        "ssa",
        "webvtt",
        "mov_text",
        "text",
        "subviewer",
    }


# --- classify_embedded ------------------------------------------------------------------------


def test_english_text_track_means_has_english() -> None:
    result = classify(track(2))
    assert result.verdict is EmbeddedVerdict.HAS_ENGLISH
    assert result.reclassified == ()


def test_english_image_track_means_has_english() -> None:
    result = classify(track(2, codec="hdmv_pgs_subtitle"))
    assert result.verdict is EmbeddedVerdict.HAS_ENGLISH


def test_english_sdh_track_means_has_english() -> None:
    result = classify(track(2, sdh=True))
    assert result.verdict is EmbeddedVerdict.HAS_ENGLISH


def test_english_full_track_beats_an_undetermined_one() -> None:
    result = classify(track(2), track(3, language="und"))
    assert result.verdict is EmbeddedVerdict.HAS_ENGLISH


def test_forced_only_english_text_track_needs_english_until_measured() -> None:
    forced = track(2, forced=True)
    assert classify(forced).verdict is EmbeddedVerdict.NEEDS_ENGLISH
    assert density_candidates(media(forced)) == (forced,)


def test_forced_track_with_a_full_density_is_reclassified_as_full() -> None:
    result = classify(track(2, forced=True), densities={2: 320.0})
    assert result.verdict is EmbeddedVerdict.HAS_ENGLISH
    assert result.reclassified == (2,)
    assert "320" in result.reason and "250" in result.reason


def test_forced_track_at_exactly_the_threshold_is_reclassified() -> None:
    result = classify(track(2, forced=True), densities={2: THRESHOLD})
    assert result.verdict is EmbeddedVerdict.HAS_ENGLISH
    assert result.reclassified == (2,)


def test_forced_track_with_a_low_density_still_needs_english() -> None:
    result = classify(track(2, forced=True), densities={2: 40.0})
    assert result.verdict is EmbeddedVerdict.NEEDS_ENGLISH
    assert result.reclassified == ()
    assert "40" in result.reason


def test_every_dense_forced_track_is_reclassified() -> None:
    result = classify(
        track(2, forced=True),
        track(3, forced=True),
        track(4, forced=True),
        densities={2: 300.0, 3: 10.0, 4: 900.0},
    )
    assert result.verdict is EmbeddedVerdict.HAS_ENGLISH
    assert result.reclassified == (2, 4)


def test_forced_image_track_needs_english_and_is_not_measured() -> None:
    forced = track(2, codec="hdmv_pgs_subtitle", forced=True)
    assert classify(forced, densities={2: 999.0}).verdict is EmbeddedVerdict.NEEDS_ENGLISH
    assert density_candidates(media(forced)) == ()


def test_only_foreign_tracks_need_english() -> None:
    result = classify(track(2, language="fre"), track(3, language="spa", forced=True))
    assert result.verdict is EmbeddedVerdict.NEEDS_ENGLISH


def test_no_subtitle_tracks_need_english() -> None:
    result = classify()
    assert result.verdict is EmbeddedVerdict.NEEDS_ENGLISH
    assert result.reclassified == ()


def test_undetermined_track_without_english_is_unclear() -> None:
    result = classify(track(2, language="und"), track(3, language="fre"))
    assert result.verdict is EmbeddedVerdict.UNCLEAR


def test_undetermined_track_makes_a_sparse_forced_english_track_unclear() -> None:
    result = classify(track(2, forced=True), track(3, language="und"), densities={2: 10.0})
    assert result.verdict is EmbeddedVerdict.UNCLEAR


@pytest.mark.parametrize("codec", ["eia_608", "eia_708", "cc_dec"])
def test_english_closed_caption_track_is_unclear(codec: str) -> None:
    result = classify(track(2, codec=codec))
    assert result.verdict is EmbeddedVerdict.UNCLEAR
    assert codec in result.reason


def test_closed_caption_track_does_not_count_as_a_full_english_track() -> None:
    cc = track(2, codec="eia_608")
    assert classify(cc, track(3, language="fre")).verdict is EmbeddedVerdict.UNCLEAR
    assert classify(cc, track(3)).verdict is EmbeddedVerdict.HAS_ENGLISH


def test_dense_forced_track_settles_a_file_that_also_has_an_unclear_track() -> None:
    result = classify(track(2, forced=True), track(3, language="und"), densities={2: 400.0})
    assert result.verdict is EmbeddedVerdict.HAS_ENGLISH
    assert result.reclassified == (2,)


# --- density_candidates -----------------------------------------------------------------------


def test_no_candidates_when_a_full_english_track_exists() -> None:
    info = media(track(2, forced=True), track(3))
    assert density_candidates(info) == ()


def test_candidates_are_the_forced_english_text_tracks_in_order() -> None:
    first = track(2, forced=True)
    second = track(5, language="en", codec="ass", forced=True)
    info = media(
        first,
        track(3, language="fre", forced=True),
        track(4, codec="dvd_subtitle", forced=True),
        second,
    )
    assert density_candidates(info) == (first, second)


def test_a_closed_caption_track_does_not_hide_the_forced_candidates() -> None:
    forced = track(3, forced=True)
    assert density_candidates(media(track(2, codec="eia_608"), forced)) == (forced,)


# --- measure_cue_density ----------------------------------------------------------------------


def srt(cue_count: int) -> str:
    return "".join(
        f"{n}\n00:00:{n % 60:02d},000 --> 00:00:{n % 60:02d},500\nLine number {n}\n\n"
        for n in range(1, cue_count + 1)
    )


class FakeRunTool:
    def __init__(self, result: ToolResult | Exception) -> None:
        self.result = result
        self.calls: list[tuple[Sequence[str], dict[str, Any]]] = []

    def __call__(self, cmd: Sequence[str], **kwargs: Any) -> ToolResult:
        self.calls.append((cmd, kwargs))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def patch_run_tool(monkeypatch: pytest.MonkeyPatch, result: ToolResult | Exception) -> FakeRunTool:
    fake = FakeRunTool(result)
    monkeypatch.setattr(embedded, "run_tool", fake)
    return fake


def ok(stdout: str, returncode: int = 0) -> ToolResult:
    return ToolResult(returncode=returncode, stdout=stdout, stderr_tail="")


def test_density_is_cues_per_hour_of_the_extracted_text(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_run_tool(monkeypatch, ok(srt(30)))
    reporter = RecordingReporter()
    density = measure_cue_density(
        TOOLS, Path("movie.mkv"), track(2, forced=True), HOUR_S / 2, reporter
    )
    assert density == pytest.approx(60.0)


def test_extraction_command_is_an_argument_list(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = patch_run_tool(monkeypatch, ok(srt(5)))
    measure_cue_density(
        TOOLS, Path("D:/lib/movie.mkv"), track(7, forced=True), HOUR_S, RecordingReporter()
    )
    cmd, kwargs = fake.calls[0]
    assert list(cmd) == [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(Path("D:/lib/movie.mkv")),
        "-map",
        "0:7",
        "-f",
        "srt",
        "-",
    ]
    assert kwargs["check"] is False
    assert "7" in kwargs["label"]


def test_rejected_text_gives_zero_density(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_run_tool(monkeypatch, ok("<html><body>not a subtitle</body></html>"))
    density = measure_cue_density(
        TOOLS, Path("movie.mkv"), track(2, forced=True), HOUR_S, RecordingReporter()
    )
    assert density == 0.0


def test_empty_output_gives_zero_density(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_run_tool(monkeypatch, ok(""))
    density = measure_cue_density(
        TOOLS, Path("movie.mkv"), track(2, forced=True), HOUR_S, RecordingReporter()
    )
    assert density == 0.0


def test_oversized_output_gives_zero_density(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_run_tool(monkeypatch, ok(srt(30)))
    density = measure_cue_density(
        TOOLS, Path("movie.mkv"), track(2, forced=True), HOUR_S, RecordingReporter(), max_bytes=100
    )
    assert density == 0.0


def test_failed_extraction_gives_zero_density(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_run_tool(monkeypatch, ok(srt(30), returncode=1))
    density = measure_cue_density(
        TOOLS, Path("movie.mkv"), track(2, forced=True), HOUR_S, RecordingReporter()
    )
    assert density == 0.0


def test_broken_output_handling_gives_zero_density(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_run_tool(monkeypatch, ToolOutputError("extract", "stdout", "reader crashed"))
    density = measure_cue_density(
        TOOLS, Path("movie.mkv"), track(2, forced=True), HOUR_S, RecordingReporter()
    )
    assert density == 0.0


def test_unusable_duration_gives_zero_density(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = patch_run_tool(monkeypatch, ok(srt(30)))
    density = measure_cue_density(
        TOOLS, Path("movie.mkv"), track(2, forced=True), 0.0, RecordingReporter()
    )
    assert density == 0.0
    assert fake.calls == []


@pytest.mark.integration
def test_real_ffmpeg_extracts_the_embedded_track_and_counts_its_cues(tmp_path: Path) -> None:
    tools = locate_tools()
    clip = make_clip(tools, tmp_path / "clip.mkv", duration_s=3.0, rich=True)
    reporter = RecordingReporter()
    info = probe(clip, tools, reporter)
    density = measure_cue_density(tools, clip, info.subtitles[0], info.duration_s, reporter)
    # The generated track has two cues.
    assert density == pytest.approx(2 / info.duration_s * 3600.0)
