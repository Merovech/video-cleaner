"""Aligning a subtitle to a video's audio with ffsubsync, and judging the result."""

from __future__ import annotations

import importlib.util
import logging
import math
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from media import RecordingReporter
from video_cleaner.config import Settings
from video_cleaner.subtitles import align
from video_cleaner.subtitles.align import (
    CLAMP_EPSILON_S,
    MIN_REFERENCE_SPEECH_S,
    AlignResult,
    accept_alignment,
    accept_reference_speech,
    align_subtitle,
    parse_ffsubsync_log,
    parse_reference_speech_s,
)
from video_cleaner.tools import (
    ToolFailed,
    ToolNotFound,
    ToolOutputError,
    ToolResult,
    Tools,
    ToolTimedOut,
    locate_tool,
    run_tool,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "ffsubsync"


def fixture_text(name: str) -> str:
    return (FIXTURES / f"{name}.txt").read_text(encoding="utf-8")


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    return Settings(
        library_root=tmp_path / "library",
        scratch_dir=tmp_path / "scratch",
        completed_dir=tmp_path / "completed",
        state_db=tmp_path / "state.db",
        log_dir=tmp_path / "logs",
        **overrides,
    )


def srt(cue_count: int) -> str:
    return "".join(
        f"{n}\n00:{n // 60:02d}:{n % 60:02d},000 --> 00:{n // 60:02d}:{n % 60:02d},500\n"
        f"Invented line {n}\n\n"
        for n in range(1, cue_count + 1)
    )


GOOD_LOG = (
    "           INFO     total of speech segments: 3022.0 speech_transformers.py:763\n"
    "           INFO     computing alignments...                    ffsubsync.py:234\n"
    "           INFO     score: 5256.000                            ffsubsync.py:255\n"
    "           INFO     offset seconds: -3.000                     ffsubsync.py:256\n"
    "           INFO     framerate scale factor: 1.000              ffsubsync.py:257\n"
)

# --- parse_ffsubsync_log ------------------------------------------------------------------------


def test_parses_the_numbers_of_a_recorded_successful_run() -> None:
    assert parse_ffsubsync_log(fixture_text("success")) == (-3.0, 1.0, 5256.0)


def test_parses_a_recorded_run_that_found_a_framerate_change() -> None:
    offset, scale, score = parse_ffsubsync_log(fixture_text("silent_audio"))
    assert offset == pytest.approx(4.78)
    assert scale == pytest.approx(1.043)
    assert score == pytest.approx(324.167)


def test_parses_windows_line_endings() -> None:
    text = fixture_text("success").replace("\n", "\r\n")
    assert parse_ffsubsync_log(text) == (-3.0, 1.0, 5256.0)


def test_parses_progress_redraws_separated_by_carriage_returns() -> None:
    text = "  0%|          | 0/60.0 [00:00<?, ?it/s]\r100%|##########| 60/60.0\r\n" + GOOD_LOG
    assert parse_ffsubsync_log(text) == (-3.0, 1.0, 5256.0)


def test_a_recorded_run_that_died_before_aligning_has_no_numbers() -> None:
    assert parse_ffsubsync_log(fixture_text("no_audio_stream")) == (None, None, None)


def test_the_numbers_of_a_run_that_then_raised_are_still_parsed() -> None:
    # Whether the run is trusted is decided from the traceback, not from the numbers.
    assert parse_ffsubsync_log(fixture_text("failed_to_sync")) == (-3.0, 1.0, 5256.0)


@pytest.mark.parametrize(
    "line_prefix",
    [
        "           INFO     ",
        "[12:00:00] INFO     ",
        "INFO     ",
        "INFO:ffsubsync.ffsubsync:",
        "",
    ],
)
def test_parses_the_other_log_layouts_ffsubsync_could_print(line_prefix: str) -> None:
    text = (
        f"{line_prefix}score: 12.500\n"
        f"{line_prefix}offset seconds: 7.250\n"
        f"{line_prefix}framerate scale factor: 0.960\n"
    )
    assert parse_ffsubsync_log(text) == (7.25, 0.96, 12.5)


@pytest.mark.parametrize(
    ("offset_text", "expected"),
    [("-120.000", -120.0), ("0.000", 0.0), ("+3.5", 3.5), ("12", 12.0), (".5", 0.5)],
)
def test_parses_signed_and_short_numbers(offset_text: str, expected: float) -> None:
    text = f"INFO score: 1.0\nINFO offset seconds: {offset_text}\nINFO framerate scale factor: 1\n"
    assert parse_ffsubsync_log(text)[0] == expected


def test_a_line_wrapped_after_the_label_is_not_read() -> None:
    # ffsubsync's console wraps at 80 columns; a number on a continuation line is not trusted.
    text = "           INFO     score:\n                    99.000\n"
    assert parse_ffsubsync_log(text) == (None, None, None)


def test_the_last_report_wins() -> None:
    text = GOOD_LOG + GOOD_LOG.replace("-3.000", "9.000")
    assert parse_ffsubsync_log(text)[0] == 9.0


def test_text_echoed_from_the_subtitle_cannot_spoof_a_report() -> None:
    # A warning can echo the beginning of a malformed subtitle line; the console indents what
    # it wraps, and the real report comes after it in any case.
    echoed = (
        "           WARNING  Skipped unparseable SRT data: 'x'      srt.py:434\n"
        "                    score: 99999.000\n"
        "                    offset seconds: 0.000\n"
        "                    framerate scale factor: 1.000\n"
    )
    assert parse_ffsubsync_log(echoed + GOOD_LOG) == (-3.0, 1.0, 5256.0)
    assert parse_ffsubsync_log(echoed) == (None, None, None)


@pytest.mark.parametrize("text", ["", "   \n", "random output\n", "score: \noffset seconds: \n"])
def test_text_without_a_report_has_no_numbers(text: str) -> None:
    assert parse_ffsubsync_log(text) == (None, None, None)


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "1e999"])
def test_numbers_that_are_not_finite_are_not_numbers(bad: str) -> None:
    text = f"INFO score: {bad}\nINFO offset seconds: {bad}\nINFO framerate scale factor: {bad}\n"
    assert parse_ffsubsync_log(text) == (None, None, None)


def test_a_partial_report_gives_the_numbers_it_has() -> None:
    assert parse_ffsubsync_log("INFO offset seconds: 2.000\n") == (2.0, None, None)


# --- the reference speech total ----------------------------------------------------------------


def test_the_speech_total_of_a_recorded_run_is_in_seconds() -> None:
    # ffsubsync prints the count of 10 ms frames it found speech in.
    assert parse_reference_speech_s(fixture_text("success")) == pytest.approx(30.22)


def test_a_recorded_run_over_a_silent_track_found_no_speech() -> None:
    assert parse_reference_speech_s(fixture_text("silent_audio")) == 0.0


def test_a_recorded_run_that_died_early_has_no_speech_total() -> None:
    assert parse_reference_speech_s(fixture_text("no_audio_stream")) is None


def test_the_speech_total_is_parsed_from_windows_line_endings_and_other_layouts() -> None:
    crlf = fixture_text("success").replace("\n", "\r\n")
    assert parse_reference_speech_s(crlf) == pytest.approx(30.22)
    assert parse_reference_speech_s("INFO:ffsubsync:total of speech segments: 500\n") == 5.0
    assert parse_reference_speech_s("total of speech segments: 250.0\n") == 2.5


def test_the_last_speech_total_wins() -> None:
    text = "INFO total of speech segments: 1.0\nINFO total of speech segments: 700.0\n"
    assert parse_reference_speech_s(text) == 7.0


@pytest.mark.parametrize(
    "text",
    [
        "",
        "INFO total of speech segments: nan\n",
        "INFO total of speech segments: inf\n",
        "                    total of speech segments: 99999.0\n",
    ],
)
def test_an_unreadable_or_echoed_speech_total_is_none(text: str) -> None:
    assert parse_reference_speech_s(text) is None


def test_enough_speech_is_accepted() -> None:
    assert accept_reference_speech(MIN_REFERENCE_SPEECH_S)[0]
    assert accept_reference_speech(1800.0)[0]


@pytest.mark.parametrize("seconds", [0.0, 0.5, MIN_REFERENCE_SPEECH_S - 0.01])
def test_too_little_speech_is_rejected_with_the_number_and_the_floor(seconds: float) -> None:
    accepted, reason = accept_reference_speech(seconds)
    assert not accepted
    assert "speech" in reason
    assert f"{seconds:.1f}" in reason
    assert f"{MIN_REFERENCE_SPEECH_S:g}" in reason


@pytest.mark.parametrize("seconds", [None, math.nan, math.inf])
def test_an_unknown_speech_total_is_rejected(seconds: float | None) -> None:
    accepted, reason = accept_reference_speech(seconds)
    assert not accepted
    assert "did not report" in reason


def test_the_floor_is_a_few_seconds() -> None:
    assert 2.0 <= MIN_REFERENCE_SPEECH_S <= 30.0


# --- accept_alignment ---------------------------------------------------------------------------


def settings_with(**overrides: Any) -> Settings:
    return make_settings(Path("."), **overrides)


def test_accepts_a_small_offset_with_no_framerate_change() -> None:
    accepted, reason = accept_alignment(3.2, 1.0, 120.0, settings_with())
    assert accepted
    assert "3.2" in reason


def test_accepts_a_negative_offset_within_the_limit() -> None:
    assert accept_alignment(-119.9, 1.0, 5.0, settings_with())[0]


def test_accepts_a_scale_exactly_at_the_tolerance() -> None:
    settings = settings_with(align_scale_tolerance=0.05)
    assert accept_alignment(1.0, 1.05, 1.0, settings)[0]
    assert accept_alignment(1.0, 0.95, 1.0, settings)[0]


def test_accepts_an_offset_just_inside_the_search_window() -> None:
    settings = settings_with(align_max_offset_s=10.0)
    assert accept_alignment(10.0 - 2 * CLAMP_EPSILON_S, 1.0, 1.0, settings)[0]
    assert accept_alignment(-10.0 + 2 * CLAMP_EPSILON_S, 1.0, 1.0, settings)[0]


@pytest.mark.parametrize("offset", [10.0, -10.0, 10.0 - CLAMP_EPSILON_S / 2, -9.99])
def test_rejects_an_offset_at_the_edge_of_the_search_window_as_clamped(offset: float) -> None:
    # ffsubsync only looks inside the window, so a best match at its edge means the real
    # offset is probably further out.
    accepted, reason = accept_alignment(offset, 1.0, 100.0, settings_with(align_max_offset_s=10.0))
    assert not accepted
    assert "search limit" in reason
    assert f"{offset:+.3f}" in reason


def test_the_clamp_margin_is_a_few_samples_of_ffsubsyncs_hundred_hertz_grid() -> None:
    assert 0.01 <= CLAMP_EPSILON_S <= 0.1


@pytest.mark.parametrize("score", [-0.001, -5.0, -324.167])
def test_rejects_an_anti_correlated_score_even_when_the_minimum_is_off(score: float) -> None:
    accepted, reason = accept_alignment(1.0, 1.0, score, settings_with(align_min_score=0.0))
    assert not accepted
    assert f"{score:.3f}" in reason
    assert "match" in reason


def test_rejects_a_score_of_zero_which_is_no_evidence_of_a_match() -> None:
    accepted, reason = accept_alignment(1.0, 1.0, 0.0, settings_with(align_min_score=0.0))
    assert not accepted
    assert "0.000" in reason


def test_a_small_positive_score_passes_when_the_minimum_is_off() -> None:
    assert accept_alignment(1.0, 1.0, 0.001, settings_with(align_min_score=0.0))[0]


def test_rejects_an_offset_beyond_the_limit_and_names_both_numbers() -> None:
    accepted, reason = accept_alignment(400.0, 1.0, 100.0, settings_with(align_max_offset_s=120.0))
    assert not accepted
    assert "400" in reason
    assert "120" in reason


def test_rejects_a_large_negative_offset() -> None:
    accepted, reason = accept_alignment(-400.0, 1.0, 100.0, settings_with())
    assert not accepted
    assert "-400" in reason


def test_rejects_a_framerate_change_beyond_the_tolerance_and_names_both_numbers() -> None:
    accepted, reason = accept_alignment(1.0, 1.12, 100.0, settings_with(align_scale_tolerance=0.05))
    assert not accepted
    assert "1.12" in reason
    assert "0.05" in reason


def test_rejects_a_framerate_change_below_one() -> None:
    assert not accept_alignment(1.0, 0.88, 100.0, settings_with())[0]


@pytest.mark.parametrize(
    "values",
    [(None, 1.0, 5.0), (1.0, None, 5.0), (1.0, 1.0, None), (None, None, None)],
)
def test_rejects_a_missing_number(values: tuple[float | None, float | None, float | None]) -> None:
    accepted, reason = accept_alignment(*values, settings_with())
    assert not accepted
    assert reason


def test_names_what_is_missing() -> None:
    _, reason = accept_alignment(1.0, None, None, settings_with())
    assert "scale" in reason
    assert "score" in reason
    assert "offset" not in reason


def test_rejects_a_score_below_a_minimum_and_names_both_numbers() -> None:
    accepted, reason = accept_alignment(1.0, 1.0, 40.0, settings_with(align_min_score=100.0))
    assert not accepted
    assert "40" in reason
    assert "100" in reason


def test_accepts_a_score_at_the_minimum() -> None:
    assert accept_alignment(1.0, 1.0, 100.0, settings_with(align_min_score=100.0))[0]


def test_a_zero_minimum_score_disables_the_score_check() -> None:
    assert accept_alignment(1.0, 1.0, 0.5, settings_with(align_min_score=0.0))[0]


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_rejects_numbers_that_are_not_finite(bad: float) -> None:
    assert not accept_alignment(bad, 1.0, 5.0, settings_with())[0]
    assert not accept_alignment(1.0, bad, 5.0, settings_with())[0]
    assert not accept_alignment(1.0, 1.0, bad, settings_with())[0]


# --- align_subtitle with a fake run_tool --------------------------------------------------------


def make_tools(tmp_path: Path) -> Tools:
    folder = tmp_path / "ffbin"
    return Tools(
        ffmpeg=folder / "ffmpeg.exe", ffprobe=folder / "ffprobe.exe", mkvmerge=folder / "mkvmerge"
    )


def option_value(cmd: Sequence[str], name: str) -> str:
    """The value of ``--name=value`` in *cmd*."""
    prefix = f"{name}="
    matches = [part[len(prefix) :] for part in cmd if part.startswith(prefix)]
    assert len(matches) == 1, f"{name} appears {len(matches)} times in {list(cmd)}"
    return matches[0]


class FakeFfsubsync:
    """Stands in for `run_tool`: records the call, writes what ffsubsync would, may raise."""

    def __init__(
        self,
        *,
        output: str | bytes | None = None,
        log: str = GOOD_LOG,
        tail: str = "",
        returncode: int = 0,
        raises: Exception | None = None,
    ) -> None:
        self.output = srt(25) if output is None else output
        self.log = log
        self.tail = tail
        self.returncode = returncode
        self.raises = raises
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.input_seen: bytes | None = None
        self.env_seen: dict[str, str] | None = None
        self.work_dir: Path | None = None
        self.write_output = True
        self.write_log = True

    def __call__(self, cmd: Sequence[str], **kwargs: Any) -> ToolResult:
        self.calls.append((list(cmd), kwargs))
        env = kwargs.get("env")
        self.env_seen = None if env is None else dict(env)
        self.input_seen = Path(option_value(cmd, "--srtin")).read_bytes()
        target = Path(option_value(cmd, "--srtout"))
        self.work_dir = target.parent
        stderr_log = kwargs.get("stderr_log")
        if self.write_log and stderr_log is not None:
            Path(stderr_log).parent.mkdir(parents=True, exist_ok=True)
            Path(stderr_log).write_text(self.log, encoding="utf-8")
        if self.raises is not None:
            raise self.raises
        if self.write_output:
            data = self.output.encode("utf-8") if isinstance(self.output, str) else self.output
            target.write_bytes(data)
        return ToolResult(returncode=self.returncode, stdout="", stderr_tail=self.tail)


def patch_run(monkeypatch: pytest.MonkeyPatch, fake: FakeFfsubsync) -> FakeFfsubsync:
    monkeypatch.setattr(align, "run_tool", fake)
    return fake


class Case:
    """A media file, an input subtitle and an output path in a temporary folder."""

    def __init__(self, tmp_path: Path, media_name: str = "movie.mkv") -> None:
        self.tmp_path = tmp_path
        self.tools = make_tools(tmp_path)
        self.media = tmp_path / "library" / media_name
        self.media.parent.mkdir(parents=True, exist_ok=True)
        self.media.write_bytes(b"not really a video")
        self.subtitle = tmp_path / "cache" / "in.srt"
        self.subtitle.parent.mkdir(parents=True, exist_ok=True)
        self.subtitle.write_text(srt(25), encoding="utf-8")
        self.out = tmp_path / "scratch" / "movie" / "aligned.en.srt"
        self.settings = make_settings(tmp_path)
        self.reporter = RecordingReporter()

    def run(self, **kwargs: Any) -> AlignResult:
        return align_subtitle(
            self.tools, self.media, self.subtitle, self.out, self.settings, self.reporter, **kwargs
        )


def test_a_good_alignment_is_ok_and_carries_the_numbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    patch_run(monkeypatch, FakeFfsubsync())
    result = case.run()
    assert result.ok
    assert result.offset_s == -3.0
    assert result.scale == 1.0
    assert result.score == 5256.0
    assert result.output == case.out
    assert "3.000" in result.reason


def test_the_aligned_subtitle_is_the_validated_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    # ffsubsync's output may carry a BOM and CRLF line ends; what lands at `out` is clean SRT.
    raw = "﻿" + srt(25).replace("\n", "\r\n")
    patch_run(monkeypatch, FakeFfsubsync(output=raw))
    result = case.run()
    assert result.ok
    data = case.out.read_bytes()
    assert not data.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in data
    assert data.decode("utf-8").startswith("1\n00:00:01,000 --> 00:00:01,500\nInvented line 1\n")


def test_the_command_is_an_argument_list_that_runs_ffsubsync_with_the_audio_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    fake = patch_run(monkeypatch, FakeFfsubsync())
    case.run()
    cmd, kwargs = fake.calls[0]
    assert all(isinstance(part, str) for part in cmd)
    assert cmd[0] == sys.executable
    assert cmd[cmd.index("-m") + 1] == "ffsubsync.ffsubsync"
    assert cmd[cmd.index("--reference-stream") + 1] == "a:0"
    assert cmd[cmd.index("--max-offset-seconds") + 1] == "120.0"
    assert kwargs["check"] is True


def test_the_offset_limit_comes_from_the_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    case.settings = replace(case.settings, align_max_offset_s=45.5)
    fake = patch_run(monkeypatch, FakeFfsubsync())
    case.run()
    cmd = fake.calls[0][0]
    assert cmd[cmd.index("--max-offset-seconds") + 1] == "45.5"


def test_the_command_names_a_webrtc_detector_and_never_the_gpl_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    fake = patch_run(monkeypatch, FakeFfsubsync())
    case.run()
    cmd = fake.calls[0][0]
    assert cmd[cmd.index("--vad") + 1] == "webrtc"
    assert not any("auditok" in part.lower() for part in cmd)


def test_the_command_downloads_and_loads_nothing_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    fake = patch_run(monkeypatch, FakeFfsubsync())
    case.run()
    cmd = fake.calls[0][0]
    for flag in ("--gss", "--whisper-weights", "--make-test-case", "--serialize-speech"):
        assert flag not in cmd
    assert not any(part.startswith("--whisper") for part in cmd)


def test_ffsubsync_is_told_the_subtitle_is_utf8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    fake = patch_run(monkeypatch, FakeFfsubsync())
    case.run()
    cmd = fake.calls[0][0]
    assert cmd[cmd.index("--encoding") + 1] == "utf-8"


def test_the_media_path_is_the_last_argument_after_a_separator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path, "-weird [1] üé 日本.mkv")
    fake = patch_run(monkeypatch, FakeFfsubsync())
    case.run()
    cmd = fake.calls[0][0]
    assert cmd[-2:] == ["--", str(case.media)]
    # The media path appears nowhere else, and never inside a longer shell-style string.
    assert [part for part in cmd if str(case.media) in part] == [str(case.media)]


def test_the_subtitle_given_to_ffsubsync_is_a_copy_named_by_us(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    case.subtitle = tmp_path / "cache" / "-odd [x] ü.sub"
    case.subtitle.write_text(srt(25), encoding="utf-8")
    fake = patch_run(monkeypatch, FakeFfsubsync())
    case.run()
    cmd = fake.calls[0][0]
    given = Path(option_value(cmd, "--srtin"))
    assert given.suffix == ".srt"
    assert given != case.subtitle
    assert fake.input_seen == case.subtitle.read_bytes()
    assert not any(str(case.subtitle) in part for part in cmd)


def test_options_whose_values_may_start_with_a_dash_are_attached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    case.out = tmp_path / "scratch" / "-out [1].srt"
    fake = patch_run(monkeypatch, FakeFfsubsync())
    case.run()
    cmd = fake.calls[0][0]
    assert "-i" not in cmd
    assert "-o" not in cmd
    assert option_value(cmd, "--srtin")
    assert option_value(cmd, "--srtout")


def test_ffsubsync_writes_beside_our_output_never_beside_the_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    fake = patch_run(monkeypatch, FakeFfsubsync())
    case.run()
    cmd, kwargs = fake.calls[0]
    for option in ("--srtin", "--srtout"):
        assert Path(option_value(cmd, option)).parent.parent == case.out.parent
    assert kwargs["cwd"] is not None
    assert Path(kwargs["cwd"]).parent == case.out.parent


def test_the_child_gets_the_ffmpeg_folder_first_on_path_and_this_environment_is_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    monkeypatch.setenv("VC_ALIGN_MARKER", "kept")
    before = dict(os.environ)
    fake = patch_run(monkeypatch, FakeFfsubsync())
    case.run()
    assert dict(os.environ) == before
    env = fake.env_seen
    assert env is not None
    path_key = next(key for key in env if key.upper() == "PATH")
    entries = env[path_key].split(os.pathsep)
    assert entries[0] == str(case.tools.ffmpeg.parent)
    assert entries[1:] == before.get(path_key, before.get("PATH", "")).split(os.pathsep)
    assert env["VC_ALIGN_MARKER"] == "kept"
    assert env["PYTHONIOENCODING"] == "utf-8"


def test_ffprobe_in_another_folder_is_also_put_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    other = tmp_path / "probe-bin"
    case.tools = Tools(
        ffmpeg=case.tools.ffmpeg, ffprobe=other / "ffprobe", mkvmerge=case.tools.mkvmerge
    )
    fake = patch_run(monkeypatch, FakeFfsubsync())
    case.run()
    env = fake.env_seen
    assert env is not None
    path_key = next(key for key in env if key.upper() == "PATH")
    entries = env[path_key].split(os.pathsep)
    assert entries[:2] == [str(case.tools.ffmpeg.parent), str(other)]


def test_the_work_files_are_gone_and_only_the_output_remains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    fake = patch_run(monkeypatch, FakeFfsubsync())
    case.run()
    assert fake.work_dir is not None
    assert not fake.work_dir.exists()
    assert [p.name for p in case.out.parent.iterdir()] == [case.out.name]


def test_the_media_folder_is_not_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = Case(tmp_path)
    patch_run(monkeypatch, FakeFfsubsync())
    case.run()
    assert [p.name for p in case.media.parent.iterdir()] == [case.media.name]
    assert case.media.read_bytes() == b"not really a video"


def test_the_run_is_labelled_and_given_a_generous_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    fake = patch_run(monkeypatch, FakeFfsubsync())
    case.run()
    kwargs = fake.calls[0][1]
    assert "movie.mkv" in kwargs["label"]
    assert kwargs["reporter"] is case.reporter
    assert kwargs["heartbeat_s"] == case.settings.heartbeat_s
    assert kwargs["timeout_s"] == align.ALIGN_TIMEOUT_S >= 1800.0


def test_the_timeout_can_be_chosen_by_the_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    fake = patch_run(monkeypatch, FakeFfsubsync())
    case.run(timeout_s=12.5)
    assert fake.calls[0][1]["timeout_s"] == 12.5


def test_the_output_folder_is_created(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = Case(tmp_path)
    case.out = tmp_path / "does" / "not" / "exist" / "out.srt"
    patch_run(monkeypatch, FakeFfsubsync())
    assert case.run().ok
    assert case.out.is_file()


def test_an_output_that_cannot_be_read_is_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    patch_run(monkeypatch, FakeFfsubsync())
    real_open = Path.open

    def refuse(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self.name == "aligned.srt" and args and args[0] == "rb":
            raise PermissionError("locked by another program")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", refuse)
    result = case.run()
    assert not result.ok
    assert "could not be read" in result.reason
    assert not case.out.exists()


def test_the_numbers_are_read_from_the_whole_log_not_the_last_few_kilobytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    # ffsubsync warns about every skipped cue while it reads the subtitle, before it reports the
    # numbers; the numbers must be found wherever they are, while the tail holds only warnings.
    skipped = "           INFO     Skipped subtitle at index 1: Start time < 0      srt.py:305\n"
    patch_run(monkeypatch, FakeFfsubsync(log=skipped * 5000 + GOOD_LOG, tail=skipped * 20))
    result = case.run()
    assert result.ok
    assert result.offset_s == -3.0


def test_the_stderr_tail_is_used_when_there_is_no_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    fake = FakeFfsubsync(tail=GOOD_LOG)
    fake.write_log = False
    patch_run(monkeypatch, fake)
    result = case.run()
    assert result.ok
    assert result.score == 5256.0


def test_a_rejected_alignment_is_not_ok_keeps_its_numbers_and_leaves_no_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    log = GOOD_LOG.replace("-3.000", "400.000")
    patch_run(monkeypatch, FakeFfsubsync(log=log))
    result = case.run()
    assert not result.ok
    assert "400" in result.reason
    assert result.offset_s == 400.0
    assert result.scale == 1.0
    assert result.score == 5256.0
    assert result.output is None
    assert not case.out.exists()


def test_a_framerate_change_beyond_the_tolerance_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    log = GOOD_LOG.replace("scale factor: 1.000", "scale factor: 1.120")
    patch_run(monkeypatch, FakeFfsubsync(log=log))
    result = case.run()
    assert not result.ok
    assert "1.12" in result.reason


def test_a_score_below_the_configured_minimum_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    case.settings = replace(case.settings, align_min_score=10_000.0)
    patch_run(monkeypatch, FakeFfsubsync())
    result = case.run()
    assert not result.ok
    assert "5256" in result.reason


def test_a_log_without_numbers_is_not_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = Case(tmp_path)
    patch_run(monkeypatch, FakeFfsubsync(log="INFO     nothing useful\n"))
    result = case.run()
    assert not result.ok
    assert result.offset_s is None
    assert result.output is None
    assert not case.out.exists()


def test_a_failed_run_does_not_touch_an_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    case.out.parent.mkdir(parents=True, exist_ok=True)
    case.out.write_text("earlier result", encoding="utf-8")
    patch_run(monkeypatch, FakeFfsubsync(log=GOOD_LOG.replace("-3.000", "400.000")))
    assert not case.run().ok
    assert case.out.read_text(encoding="utf-8") == "earlier result"


def test_the_recorded_successful_run_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    patch_run(monkeypatch, FakeFfsubsync(log=fixture_text("success")))
    result = case.run()
    assert result.ok
    assert result.output == case.out


def test_the_recorded_run_over_a_silent_track_is_rejected_for_lack_of_speech(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    patch_run(monkeypatch, FakeFfsubsync(log=fixture_text("silent_audio")))
    result = case.run()
    assert not result.ok
    assert "speech" in result.reason
    # ffsubsync's own numbers are kept for the record, but nothing is written.
    assert result.score == pytest.approx(324.167)
    assert result.output is None
    assert not case.out.exists()


def test_a_log_without_a_speech_total_is_rejected_as_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    log = "".join(line for line in GOOD_LOG.splitlines(keepends=True) if "total of" not in line)
    patch_run(monkeypatch, FakeFfsubsync(log=log))
    result = case.run()
    assert not result.ok
    assert "did not report" in result.reason
    assert not case.out.exists()


def test_a_negative_score_in_the_log_is_rejected_with_the_default_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    patch_run(monkeypatch, FakeFfsubsync(log=GOOD_LOG.replace("5256.000", "-12.000")))
    result = case.run()
    assert not result.ok
    assert result.score == -12.0
    assert not case.out.exists()


def test_an_offset_pinned_at_the_search_limit_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    log = GOOD_LOG.replace("-3.000", "-120.000")
    patch_run(monkeypatch, FakeFfsubsync(log=log))
    result = case.run()
    assert not result.ok
    assert "search limit" in result.reason


# --- the child's environment and the log excerpt -----------------------------------------------


def test_the_child_console_is_pinned_whatever_the_caller_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in (
        ("FORCE_COLOR", "1"),
        ("CLICOLOR_FORCE", "1"),
        ("COLUMNS", "20"),
        ("TERM", "xterm-256color"),
    ):
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("NO_COLOR", raising=False)
    case = Case(tmp_path)
    fake = patch_run(monkeypatch, FakeFfsubsync())
    case.run()
    env = fake.env_seen
    assert env is not None
    assert "FORCE_COLOR" not in env
    assert "CLICOLOR_FORCE" not in env
    assert env["NO_COLOR"] == "1"
    assert env["TERM"] == "dumb"
    assert int(env["COLUMNS"]) >= 1000


def test_the_child_does_not_inherit_credentials_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secrets = {
        "VC_OPENSUBTITLES_API_KEY": "k",
        "VC_OPENSUBTITLES_USERNAME": "u",
        "VC_OPENSUBTITLES_PASSWORD": "p",
        "vc_opensubtitles_other": "o",
        "SOME_SERVICE_API_KEY": "k",
        "my_password": "p",
        "GITHUB_TOKEN": "t",
        "Client_Secret": "s",
    }
    for name, value in secrets.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("VC_ALIGN_MARKER", "kept")
    case = Case(tmp_path)
    fake = patch_run(monkeypatch, FakeFfsubsync())
    case.run()
    env = fake.env_seen
    assert env is not None
    present = {name.upper() for name in env}
    assert not present & {name.upper() for name in secrets}
    assert env["VC_ALIGN_MARKER"] == "kept"
    assert "PYTHONIOENCODING" in env


_RICH_LOGGER = (
    "import logging, sys\n"
    "from rich.console import Console\n"
    "from rich.logging import RichHandler\n"
    "logging.basicConfig(level=logging.INFO, format='%(message)s', datefmt='[%X]',\n"
    "    handlers=[RichHandler(console=Console(file=sys.stderr))])\n"
    "log = logging.getLogger('x')\n"
    "log.info('score: %.3f', 5256.0)\n"
    "log.info('offset seconds: %.3f', -3.0)\n"
    "log.info('framerate scale factor: %.3f', 1.0)\n"
    "log.info('total of speech segments: %s', 3022.0)\n"
)


def test_a_hostile_caller_environment_cannot_garble_the_log_ffsubsync_prints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The logging setup ffsubsync uses (rich on stderr), run with and without our pinned child
    # environment while the caller's environment asks for colour and a narrow console.
    for name, value in (("FORCE_COLOR", "1"), ("CLICOLOR_FORCE", "1"), ("COLUMNS", "12")):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("TERM", "xterm-256color")
    command = [sys.executable, "-c", _RICH_LOGGER]
    hostile = run_tool(command, label="rich", reporter=RecordingReporter())
    assert parse_ffsubsync_log(hostile.stderr_tail) != (-3.0, 1.0, 5256.0)  # the premise
    pinned = run_tool(
        command,
        label="rich",
        reporter=RecordingReporter(),
        env=align._child_env(make_tools(tmp_path)),
    )
    assert parse_ffsubsync_log(pinned.stderr_tail) == (-3.0, 1.0, 5256.0)
    assert parse_reference_speech_s(pinned.stderr_tail) == pytest.approx(30.22)
    assert "\x1b" not in pinned.stderr_tail


def failure_log(lines: int = 30) -> str:
    return "".join(f"           INFO     log line number {n}\n" for n in range(1, lines + 1))


def warnings_of(caplog: pytest.LogCaptureFixture) -> str:
    return " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)


def test_a_failed_run_logs_the_end_of_ffsubsyncs_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    case = Case(tmp_path)
    patch_run(monkeypatch, FakeFfsubsync(raises=failed(1), log=failure_log()))
    with caplog.at_level(logging.WARNING, logger="video_cleaner.subtitles.align"):
        result = case.run()
    assert not result.ok
    warnings = warnings_of(caplog)
    assert "log line number 30" in warnings
    assert "log line number 21" in warnings
    assert "log line number 20" not in warnings
    assert "stderr in" not in warnings  # the work folder, and the log in it, is gone by then


@pytest.mark.parametrize("what", ["traceback", "timeout", "exit"])
def test_every_kind_of_failure_logs_the_excerpt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    what: str,
) -> None:
    case = Case(tmp_path)
    log = failure_log(5) + (fixture_text("failed_to_sync") if what == "traceback" else "")
    fake = FakeFfsubsync(
        log=log,
        raises=ToolTimedOut("align", 5.0) if what == "timeout" else None,
        returncode=3 if what == "exit" else 0,
    )
    patch_run(monkeypatch, fake)
    with caplog.at_level(logging.WARNING, logger="video_cleaner.subtitles.align"):
        assert not case.run().ok
    warnings = warnings_of(caplog)
    assert "log line number 5" in warnings or "FileNotFoundError" in warnings


def test_the_logged_excerpt_is_redacted_and_each_line_is_capped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    case = Case(tmp_path)
    log = (
        "           INFO     started with --api-key=SUPERSECRET123 now\n"
        f"           INFO     {'x' * 5000}\n"
    )
    patch_run(monkeypatch, FakeFfsubsync(raises=failed(1), log=log))
    with caplog.at_level(logging.WARNING, logger="video_cleaner.subtitles.align"):
        case.run()
    warnings = warnings_of(caplog)
    assert "SUPERSECRET123" not in warnings
    assert "x" * 301 not in warnings
    assert "x" * 200 in warnings


def test_a_successful_run_logs_no_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    case = Case(tmp_path)
    patch_run(monkeypatch, FakeFfsubsync())
    with caplog.at_level(logging.WARNING, logger="video_cleaner.subtitles.align"):
        assert case.run().ok
    assert warnings_of(caplog) == ""


# --- failures never raise -----------------------------------------------------------------------


def failed(code: int = 2) -> ToolFailed:
    return ToolFailed(code, "python -m ffsubsync.ffsubsync ...", None)


def test_a_failed_tool_is_a_failed_result_not_an_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    fake = FakeFfsubsync(raises=failed(2), log=fixture_text("no_audio_stream"))
    patch_run(monkeypatch, fake)
    result = case.run()
    assert result == AlignResult(ok=False, reason=result.reason)
    assert "2" in result.reason
    assert not case.out.exists()


def test_a_failed_tool_reason_names_no_speech_when_the_log_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    patch_run(monkeypatch, FakeFfsubsync(raises=failed(1), log=fixture_text("no_audio_stream")))
    assert "speech" in case.run().reason


def test_the_reason_never_carries_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = Case(tmp_path, "secret-title (2020).mkv")
    error = ToolFailed(1, f"python -m ffsubsync.ffsubsync -- {case.media}", None)
    patch_run(monkeypatch, FakeFfsubsync(raises=error))
    reason = case.run().reason
    assert "secret-title" not in reason
    assert str(tmp_path) not in reason


def test_a_timeout_is_a_failed_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = Case(tmp_path)
    patch_run(monkeypatch, FakeFfsubsync(raises=ToolTimedOut("align", 3600.0)))
    result = case.run()
    assert not result.ok
    assert "3600" in result.reason
    assert not case.out.exists()


def test_broken_output_handling_is_a_failed_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    patch_run(monkeypatch, FakeFfsubsync(raises=ToolOutputError("align", "stderr", "boom")))
    assert not case.run().ok


def test_a_missing_tool_is_not_swallowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = Case(tmp_path)
    patch_run(monkeypatch, FakeFfsubsync(raises=ToolNotFound("python", "gone")))
    with pytest.raises(ToolNotFound):
        case.run()


def test_the_work_files_are_removed_when_the_tool_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    fake = patch_run(monkeypatch, FakeFfsubsync(raises=failed()))
    case.run()
    assert fake.work_dir is not None
    assert not fake.work_dir.exists()


def test_the_work_files_are_removed_when_a_missing_tool_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    fake = patch_run(monkeypatch, FakeFfsubsync(raises=ToolNotFound("python")))
    with pytest.raises(ToolNotFound):
        case.run()
    assert fake.work_dir is not None
    assert not fake.work_dir.exists()


@pytest.mark.parametrize("name", ["failed_to_sync", "no_audio_stream"])
def test_a_traceback_in_the_log_is_a_failure_even_with_exit_code_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    case = Case(tmp_path)
    patch_run(monkeypatch, FakeFfsubsync(log=fixture_text(name)))
    result = case.run()
    assert not result.ok
    assert "error" in result.reason.lower()
    assert result.output is None
    assert not case.out.exists()


def test_a_traceback_in_the_stderr_tail_is_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    fake = FakeFfsubsync(tail=GOOD_LOG + fixture_text("failed_to_sync"))
    fake.write_log = False
    patch_run(monkeypatch, fake)
    assert not case.run().ok


def test_a_nonzero_exit_that_was_not_raised_is_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    patch_run(monkeypatch, FakeFfsubsync(returncode=3))
    result = case.run()
    assert not result.ok
    assert "3" in result.reason


def test_no_output_file_is_a_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = Case(tmp_path)
    fake = FakeFfsubsync()
    fake.write_output = False
    patch_run(monkeypatch, fake)
    result = case.run()
    assert not result.ok
    assert "no subtitle" in result.reason
    assert not case.out.exists()


@pytest.mark.parametrize(
    ("output", "word"),
    [
        ("", "empty"),
        ("   \n\n", "empty"),
        ("<html><body>nope</body></html>", "HTML"),
        (b"\x00\x01\x02" * 400, "binary"),
        ("just some words\n" * 30, "recognised"),
        (srt(5), "fewer than"),
    ],
)
def test_an_empty_or_invalid_output_is_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output: str | bytes, word: str
) -> None:
    case = Case(tmp_path)
    patch_run(monkeypatch, FakeFfsubsync(output=output))
    result = case.run()
    assert not result.ok
    assert word in result.reason
    # The numbers were fine, so they are still reported.
    assert result.offset_s == -3.0
    assert result.output is None
    assert not case.out.exists()


def test_an_output_larger_than_the_configured_limit_is_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    case.settings = replace(case.settings, subtitle_max_bytes=200)
    patch_run(monkeypatch, FakeFfsubsync(output=srt(25)))
    result = case.run()
    assert not result.ok
    assert "larger" in result.reason


def test_the_configured_minimum_cue_count_applies_to_the_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    case.settings = replace(case.settings, subtitle_min_cues=3)
    patch_run(monkeypatch, FakeFfsubsync(output=srt(3)))
    assert case.run().ok


def test_cues_that_ffsubsync_dropped_can_leave_too_few(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    patch_run(monkeypatch, FakeFfsubsync(output=srt(19)))
    result = case.run()
    assert not result.ok
    assert "19" in result.reason


# --- hash_exact ---------------------------------------------------------------------------------


def test_a_hash_exact_match_runs_nothing_and_keeps_the_subtitle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = Case(tmp_path)
    fake = patch_run(monkeypatch, FakeFfsubsync())
    result = case.run(hash_exact=True)
    assert result.ok
    assert result.output == case.subtitle
    assert result.offset_s is None
    assert result.reason
    assert fake.calls == []
    assert not case.out.exists()
    assert not case.out.parent.exists()


def test_hash_exact_does_not_even_look_at_the_media(tmp_path: Path) -> None:
    case = Case(tmp_path)
    case.media.unlink()
    assert case.run(hash_exact=True).ok


# --- the real ffsubsync -------------------------------------------------------------------------

# Voice-like audio in invented windows: a harmonic buzz with a syllable-rate envelope, so the
# voice detector sees speech. The subtitle's cues lie three seconds after the sound they match.
_SPEECH_WINDOWS = [(4, 7), (10, 15), (18, 21), (25, 31), (34, 37), (40, 46), (50, 54)]


def _speech_expression() -> str:
    gate = "+".join(f"between(t,{a},{b})" for a, b in _SPEECH_WINDOWS)
    harmonics = "+".join(f"{w}*sin(2*PI*{130 * k}*t)" for k, w in enumerate((1, 0.6, 0.5, 0.4), 1))
    return f"({gate})*(0.5+0.5*sin(2*PI*4*t))*0.3*({harmonics})"


def _timestamp(seconds: int) -> str:
    return f"00:{seconds // 60:02d}:{seconds % 60:02d},000"


def real_tools() -> Tools:
    """ffmpeg and ffprobe plus the ffsubsync modules, or skip the test when one is missing."""
    for module in ("ffsubsync", "webrtcvad"):
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"the {module} module is not installed")
    try:
        ffmpeg = locate_tool("ffmpeg", os.environ)
        ffprobe = locate_tool("ffprobe", os.environ)
    except ToolNotFound as exc:
        pytest.skip(f"ffmpeg and ffprobe are required: {exc}")
    # mkvmerge is not used here.
    return Tools(ffmpeg=ffmpeg, ffprobe=ffprobe, mkvmerge=Path("mkvmerge"))


def test_the_integration_test_skips_when_ffmpeg_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str, env: Any) -> Path:
        raise ToolNotFound(name, "not installed")

    monkeypatch.setattr(sys.modules[__name__], "locate_tool", missing)
    with pytest.raises(pytest.skip.Exception, match="ffmpeg"):
        real_tools()


def test_the_integration_test_skips_when_ffsubsync_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args: None if name == "ffsubsync" else real_find_spec(name, *args),
    )
    with pytest.raises(pytest.skip.Exception, match="ffsubsync"):
        real_tools()


@pytest.mark.integration
def test_the_real_ffsubsync_runs_the_command_on_generated_media(tmp_path: Path) -> None:
    tools = real_tools()
    media_dir = tmp_path / "library" / "film [2020] üé"
    media_dir.mkdir(parents=True)
    media = media_dir / "-clip [1] ü.mkv"
    _run_ffmpeg(
        tools,
        [
            "-f", "lavfi", "-i", "color=c=black:s=160x120:r=10:d=60",
            "-f", "lavfi", "-i", f"aevalsrc='{_speech_expression()}':s=16000:d=60",
            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(media),
        ],
    )  # fmt: skip
    cues = "".join(
        f"{n}\n{_timestamp(a + 3)} --> {_timestamp(b + 3)}\nInvented line {n}.\n\n"
        for n, (a, b) in enumerate(_SPEECH_WINDOWS, 1)
    )
    subtitle = tmp_path / "cache" / "input.srt"
    subtitle.parent.mkdir()
    subtitle.write_text(cues, encoding="utf-8")
    out = tmp_path / "scratch" / "aligned.srt"
    settings = make_settings(tmp_path, align_max_offset_s=10.0, subtitle_min_cues=1)
    before = sorted(p.name for p in media_dir.iterdir())

    result = align_subtitle(tools, media, subtitle, out, settings, RecordingReporter())

    assert result.ok, result.reason
    assert result.offset_s is not None
    assert result.scale is not None
    assert result.score is not None
    assert result.output == out
    assert out.read_text(encoding="utf-8").startswith("1\n")
    # The media folder was only read and the scratch folder holds just the result.
    assert sorted(p.name for p in media_dir.iterdir()) == before
    assert [p.name for p in out.parent.iterdir()] == ["aligned.srt"]


def _run_ffmpeg(tools: Tools, args: list[str]) -> None:
    subprocess.run(  # noqa: S603
        [str(tools.ffmpeg), "-nostdin", "-v", "error", "-y", *args],
        check=True,
        stdin=subprocess.DEVNULL,
    )


# --- the ffsubsync probe -------------------------------------------------------------------

FFSUBSYNC_START = [
    sys.executable,
    "-W",
    "ignore::RuntimeWarning:runpy",
    "-m",
    "ffsubsync.ffsubsync",
]


class ProbeRun:
    def __init__(self, outcome: tuple[int, str] | BaseException = (0, "usage: ffsubsync.py\n")):
        self.outcome = outcome
        self.calls: list[tuple[list[str], dict[str, str]]] = []

    def __call__(self, cmd: Sequence[str], env: Any) -> tuple[int, str]:
        self.calls.append((list(cmd), dict(env)))
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def test_the_aligner_and_the_probe_start_ffsubsync_the_same_way(tmp_path: Path) -> None:
    assert align.ffsubsync_command() == FFSUBSYNC_START
    case = Case(tmp_path)
    cmd = align._command(case.media, tmp_path / "in.srt", tmp_path / "out.srt", case.settings)
    assert cmd[: len(FFSUBSYNC_START)] == FFSUBSYNC_START


def test_the_probe_asks_for_help_and_passes_when_it_works() -> None:
    run = ProbeRun()
    result = align.probe_ffsubsync(run, {"PATH": "p"})
    assert result.ok is True
    assert [cmd for cmd, _ in run.calls] == [[*FFSUBSYNC_START, "--help"]]


def test_the_probe_reports_the_first_line_of_a_failure() -> None:
    run = ProbeRun((1, "\nModuleNotFoundError: No module named 'ffsubsync'\nmore\n"))
    result = align.probe_ffsubsync(run, {})
    assert result.ok is False
    assert result.timed_out is False
    assert result.detail == "ModuleNotFoundError: No module named 'ffsubsync'"


def test_the_probe_names_the_exit_code_when_there_is_no_output() -> None:
    assert align.probe_ffsubsync(ProbeRun((3, "")), {}).detail == "exit 3"


def test_the_probe_reports_a_start_failure() -> None:
    result = align.probe_ffsubsync(ProbeRun(OSError("cannot start")), {})
    assert result.ok is False
    assert "OSError" in result.detail


def test_the_probe_reports_a_timeout_without_the_command_line() -> None:
    timeout = subprocess.TimeoutExpired([*FFSUBSYNC_START, "--help"], 20)
    result = align.probe_ffsubsync(ProbeRun(timeout), {})
    assert result.ok is False
    assert result.timed_out is True
    assert result.detail == f"did not answer within {align.PROBE_TIMEOUT_S:g} s"
    assert "ffsubsync.ffsubsync" not in result.detail


def test_the_probe_child_gets_the_environment_the_aligner_gets() -> None:
    run = ProbeRun()
    environ = {
        "PATH": "p",
        "VC_OPENSUBTITLES_API_KEY": "k",
        "VC_OPENSUBTITLES_PASSWORD": "pw",
        "SOME_SERVICE_TOKEN": "t",
    }
    align.probe_ffsubsync(run, environ)
    (_, env) = run.calls[0]
    assert env == {"PATH": "p"}


def test_the_default_probe_runs_a_real_command_with_the_given_environment() -> None:
    code, output = align._run_probe(
        [sys.executable, "-c", "import os; print(os.environ.get('VC_PROBE_TEST', 'unset'))"],
        {**os.environ, "VC_PROBE_TEST": "seen"},
    )
    assert code == 0
    assert "seen" in output
