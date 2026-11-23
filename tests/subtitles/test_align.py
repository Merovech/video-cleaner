"""Aligning a subtitle to a video's audio with ffsubsync, and judging the result."""

from __future__ import annotations

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
    AlignResult,
    accept_alignment,
    align_subtitle,
    parse_ffsubsync_log,
)
from video_cleaner.tools import (
    ToolFailed,
    ToolNotFound,
    ToolOutputError,
    ToolResult,
    Tools,
    ToolTimedOut,
    locate_tools,
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


# --- accept_alignment ---------------------------------------------------------------------------


def settings_with(**overrides: Any) -> Settings:
    return make_settings(Path("."), **overrides)


def test_accepts_a_small_offset_with_no_framerate_change() -> None:
    accepted, reason = accept_alignment(3.2, 1.0, 120.0, settings_with())
    assert accepted
    assert "3.2" in reason


def test_accepts_a_negative_offset_within_the_limit() -> None:
    assert accept_alignment(-119.9, 1.0, 5.0, settings_with())[0]


def test_accepts_exactly_the_limits() -> None:
    settings = settings_with(align_max_offset_s=10.0, align_scale_tolerance=0.05)
    assert accept_alignment(10.0, 1.05, 1.0, settings)[0]
    assert accept_alignment(-10.0, 0.95, 1.0, settings)[0]


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
    # ffsubsync reports the numbers, then prints one line per skipped cue.
    skipped = "           INFO     Skipped subtitle at index 1: Start time < 0      srt.py:305\n"
    patch_run(monkeypatch, FakeFfsubsync(log=GOOD_LOG + skipped * 5000, tail=skipped * 20))
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


@pytest.mark.integration
def test_the_real_ffsubsync_runs_the_command_on_generated_media(tmp_path: Path) -> None:
    tools = locate_tools()
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
