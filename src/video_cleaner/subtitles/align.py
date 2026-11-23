"""Aligning a subtitle to a video's audio with ffsubsync, and judging the result.

`align_subtitle` runs ffsubsync, which listens to the video's first audio stream for speech and
finds the shift (and the frame-rate change, if any) that makes the subtitle fit it. ffsubsync
reports the numbers only in its log, and it exits with 0 even when it failed to sync, so the
log is read (`parse_ffsubsync_log`) and the result is judged against the user's limits
(`accept_alignment`) before the output is trusted. Whatever ffsubsync writes is checked like any
other subtitle (`normalise_subtitle`) before it is returned.

The video is only read; ffsubsync's files go to a work folder beside the output and are removed.
Nothing here raises for a tool that fails, times out or writes rubbish: that is a failed
`AlignResult`. A tool that is not installed still raises `ToolNotFound`, like everywhere else.
"""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from video_cleaner.atomicio import write_text_atomic
from video_cleaner.cmdline import flatten
from video_cleaner.config import Settings
from video_cleaner.progress import Reporter
from video_cleaner.subtitles.textfmt import SubtitleRejected, normalise_subtitle
from video_cleaner.tools import (
    ToolFailed,
    ToolOutputError,
    ToolResult,
    Tools,
    ToolTimedOut,
    run_tool,
)

_log = logging.getLogger("video_cleaner.subtitles.align")

#: Overall limit for one ffsubsync run. It reads the whole audio track (possibly over the
#: network), which takes minutes for a film; an hour is far past any healthy run.
ALIGN_TIMEOUT_S = 3600.0
# ffsubsync's log is mostly one line per skipped cue; the numbers are near its start.
_LOG_READ_CAP_BYTES = 8 * 1024 * 1024
_SCALE_SLACK = 1e-9
_WORK_PREFIX = ".align-"
_STAGED_INPUT = "input.srt"
_STAGED_OUTPUT = "aligned.srt"
_STAGED_LOG = "ffsubsync.log"

_TRACEBACK_MARK = "Traceback (most recent call last)"
_NO_SPEECH_MARK = "Unable to detect speech"

# A number as ffsubsync prints it ("%.3f"), plus the exponent and non-finite forms a different
# version could print; non-finite values are turned into None afterwards.
_NUMBER = r"[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf(?:inity)?)(?![\w.])"
# The label starts the message. The message follows a log level ("INFO", optionally after a
# bracketed time, or "INFO:logger:") or starts the line. The console wraps at 80 columns and
# indents what it wraps, so text echoed from a subtitle on a continuation line never matches.
_LINE_START = r"^(?:[ \t]*(?:\[[^\]\n]*\][ \t]*)?INFO(?:[ \t]+|:[\w.]+:)|)"


def _report_pattern(label: str) -> re.Pattern[str]:
    return re.compile(rf"{_LINE_START}{label}:[ \t]*({_NUMBER})", re.MULTILINE | re.IGNORECASE)


_SCORE_RE = _report_pattern("score")
_OFFSET_RE = _report_pattern("offset seconds")
_SCALE_RE = _report_pattern("framerate scale factor")


@dataclass(frozen=True)
class AlignResult:
    """The outcome of aligning one subtitle.

    *reason* is short and safe to log or show (no paths, no subtitle text). The numbers are
    ffsubsync's, when it got far enough to report them; *output* is the aligned subtitle, only
    when *ok*.
    """

    ok: bool
    reason: str
    offset_s: float | None = None
    scale: float | None = None
    score: float | None = None
    output: Path | None = None


def _last_number(pattern: re.Pattern[str], text: str) -> float | None:
    matches = pattern.findall(text)
    if not matches:
        return None
    value = float(matches[-1])
    return value if math.isfinite(value) else None


def parse_ffsubsync_log(text: str) -> tuple[float | None, float | None, float | None]:
    """The (offset in seconds, framerate scale factor, score) ffsubsync reported in *text*.

    ffsubsync logs ``score: 5256.000``, ``offset seconds: -3.000`` and
    ``framerate scale factor: 1.000`` as INFO messages on stderr. Each is None when it is not
    there or not a finite number; when a number is reported more than once the last one wins.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n")
    return (
        _last_number(_OFFSET_RE, lines),
        _last_number(_SCALE_RE, lines),
        _last_number(_SCORE_RE, lines),
    )


def accept_alignment(
    offset_s: float | None, scale: float | None, score: float | None, settings: Settings
) -> tuple[bool, str]:
    """Whether an alignment is believable, with a reason carrying the numbers.

    Rejected: a missing or non-finite number, an offset of more than ``align_max_offset_s``
    seconds either way, a framerate scale further than ``align_scale_tolerance`` from 1, and,
    when ``align_min_score`` is above zero, a score below it.
    """
    missing = [
        name
        for name, value in (("offset", offset_s), ("scale", scale), ("score", score))
        if value is None or not math.isfinite(value)
    ]
    if missing or offset_s is None or scale is None or score is None:
        return False, f"ffsubsync did not report a usable {', '.join(missing)}"
    if abs(offset_s) > settings.align_max_offset_s:
        return False, (
            f"offset {offset_s:+.3f} s is beyond the {settings.align_max_offset_s:g} s limit"
        )
    # The small slack keeps a scale exactly at the tolerance (1.05 - 1.0 is not exactly 0.05).
    if abs(scale - 1.0) > settings.align_scale_tolerance + _SCALE_SLACK:
        return False, (
            f"framerate scale {scale:.3f} differs from 1 by more than "
            f"{settings.align_scale_tolerance:g}"
        )
    if settings.align_min_score > 0 and score < settings.align_min_score:
        return False, f"score {score:.3f} is below the minimum {settings.align_min_score:g}"
    return True, f"offset {offset_s:+.3f} s, framerate scale {scale:.3f}, score {score:.3f}"


def _child_env(tools: Tools) -> dict[str, str]:
    """A copy of this environment with the ffmpeg (and ffprobe) folders first on PATH.

    ffsubsync finds ffmpeg only through PATH. Python's stdio is set to UTF-8 so the log, which
    names the video, is decoded the way `run_tool` reads it.
    """
    env = dict(os.environ)
    key = next((name for name in env if name.upper() == "PATH"), "PATH")
    folders = list(dict.fromkeys(str(tool.parent) for tool in (tools.ffmpeg, tools.ffprobe)))
    existing = env.get(key, "")
    env[key] = os.pathsep.join([*folders, existing] if existing else folders)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _command(media: Path, staged_in: Path, staged_out: Path, settings: Settings) -> list[str]:
    return [
        *flatten(
            (sys.executable,),
            # ffsubsync has no __main__ module, so it is run as ffsubsync.ffsubsync, which
            # makes Python warn on every start that the module was imported twice; silence
            # only that warning.
            ("-W", "ignore::RuntimeWarning:runpy"),
            ("-m", "ffsubsync.ffsubsync"),
            # The values are attached with "=", so a path that starts with "-" is a value and
            # never an option.
            (f"--srtin={staged_in}",),
            (f"--srtout={staged_out}",),
            # The staged subtitle is validated UTF-8; detecting the encoding could guess wrong.
            ("--encoding", "utf-8"),
            # How far the subtitle may be moved: the user's limit.
            ("--max-offset-seconds", repr(float(settings.align_max_offset_s))),
            # Listen to the first audio track.
            ("--reference-stream", "a:0"),
            # webrtc voice detection only. "auditok" is GPLv3 and is never used.
            ("--vad", "webrtc"),
            # The video, after "--" so a name that starts with "-" is not read as an option.
            ("--", str(media)),
        )
    ]


def _read_log(path: Path, fallback: str) -> str:
    """The whole stderr log ffsubsync wrote (up to a cap), or *fallback* if there is none."""
    try:
        with path.open("rb") as handle:
            return handle.read(_LOG_READ_CAP_BYTES).decode("utf-8", errors="replace")
    except OSError:
        return fallback


def _failure_reason(text: str, what: str) -> str:
    if _NO_SPEECH_MARK in text:
        return f"{what}: no speech found in the audio"
    return what


def _validated_srt(path: Path, settings: Settings) -> tuple[str | None, str]:
    """(clean SRT text, "") for ffsubsync's output, or (None, why it is unusable)."""
    try:
        with path.open("rb") as handle:
            # One byte past the limit is enough for `normalise_subtitle` to refuse it.
            raw = handle.read(settings.subtitle_max_bytes + 1)
    except FileNotFoundError:
        return None, "ffsubsync produced no subtitle"
    except OSError:
        return None, "the subtitle ffsubsync produced could not be read"
    if not raw:
        return None, "ffsubsync wrote an empty subtitle"
    try:
        cleaned = normalise_subtitle(
            raw, max_bytes=settings.subtitle_max_bytes, min_cues=settings.subtitle_min_cues
        )
    except SubtitleRejected as exc:
        return None, f"ffsubsync output rejected: {exc}"
    return cleaned.srt, ""


def _run(
    tools: Tools,
    media: Path,
    work: Path,
    out_label: str,
    settings: Settings,
    reporter: Reporter,
    timeout_s: float,
) -> tuple[ToolResult | None, str, str]:
    """Run ffsubsync: (its result, the log text, "") or (None, the log text, why it failed)."""
    log_path = work / _STAGED_LOG
    cmd = _command(media, work / _STAGED_INPUT, work / _STAGED_OUTPUT, settings)
    label = f"align subtitle to {out_label}"
    try:
        result = run_tool(
            cmd,
            label=label,
            reporter=reporter,
            heartbeat_s=settings.heartbeat_s,
            stderr_log=log_path,
            cwd=work,
            env=_child_env(tools),
            timeout_s=timeout_s,
            check=True,
        )
    except ToolFailed as exc:
        _log.warning("%s: %s", label, exc)
        text = _read_log(log_path, "")
        return None, text, _failure_reason(text, f"ffsubsync exited with code {exc.returncode}")
    except ToolTimedOut as exc:
        _log.warning("%s: %s", label, exc)
        return None, _read_log(log_path, ""), f"ffsubsync timed out after {exc.timeout_s:g} s"
    except ToolOutputError as exc:
        _log.warning("%s: %s", label, exc)
        return None, _read_log(log_path, ""), "ffsubsync output could not be read"
    return result, _read_log(log_path, result.stderr_tail), ""


def align_subtitle(
    tools: Tools,
    media: Path,
    subtitle: Path,
    out: Path,
    settings: Settings,
    reporter: Reporter,
    *,
    hash_exact: bool = False,
    timeout_s: float = ALIGN_TIMEOUT_S,
) -> AlignResult:
    """Align *subtitle* (already SRT) to the audio of *media*, writing the result to *out*.

    *media* is only read and may be on a read-only share; ffsubsync's work files live in a
    folder created beside *out* and removed afterwards, and *out* is only written (atomically)
    once the alignment is believable and the output is a valid subtitle. A failed alignment
    leaves *out* as it was. With *hash_exact* (the subtitle was matched to this very file by
    its hash, so it already fits) nothing is run and the subtitle itself is the output.
    """
    if hash_exact:
        return AlignResult(
            ok=True, reason="matched this file by hash; no alignment needed", output=subtitle
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=_WORK_PREFIX, dir=out.parent))
    try:
        shutil.copyfile(subtitle, work / _STAGED_INPUT)
        result, text, why = _run(tools, media, work, media.name, settings, reporter, timeout_s)
        if result is None:
            return AlignResult(ok=False, reason=why)
        if result.returncode != 0:
            return AlignResult(
                ok=False,
                reason=_failure_reason(text, f"ffsubsync exited with code {result.returncode}"),
            )
        if _TRACEBACK_MARK in text:
            return AlignResult(
                ok=False, reason=_failure_reason(text, "ffsubsync reported an error")
            )
        offset_s, scale, score = parse_ffsubsync_log(text)
        accepted, reason = accept_alignment(offset_s, scale, score, settings)
        if not accepted:
            return AlignResult(False, reason, offset_s, scale, score)
        srt, problem = _validated_srt(work / _STAGED_OUTPUT, settings)
        if srt is None:
            return AlignResult(False, problem, offset_s, scale, score)
        write_text_atomic(out, srt)
        return AlignResult(True, reason, offset_s, scale, score, out)
    finally:
        shutil.rmtree(work, ignore_errors=True)
