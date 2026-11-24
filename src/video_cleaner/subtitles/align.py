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

from video_cleaner.atomicio import read_bytes_capped, write_text_atomic
from video_cleaner.cmdline import flatten
from video_cleaner.config import Settings
from video_cleaner.logging_setup import redact
from video_cleaner.progress import Reporter
from video_cleaner.subtitles.textfmt import (
    SubtitleRejected,
    normalise_newlines,
    normalise_subtitle,
)
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
# ffsubsync's log is mostly one line per skipped cue; those warnings come before the report of
# the numbers and a traceback, if any, after it. A log past this cap is cut off at its end, so
# the report can be missing: that fails safe (no numbers, no alignment).
_LOG_READ_CAP_BYTES = 8 * 1024 * 1024
_SCALE_SLACK = 1e-9
#: ffsubsync works on a 100 Hz grid and only looks for offsets inside the window it is given
#: (`--max-offset-seconds`, our limit). When the true offset lies outside, the best candidate
#: inside is at or next to the edge, so an offset within this margin (five samples) of the
#: limit is a clamped search, not a result.
CLAMP_EPSILON_S = 0.05
#: Less speech than this in the video's audio (the "total of speech segments" ffsubsync logs,
#: counted in 10 ms frames) cannot anchor an alignment: a silent or music-only track gives
#: arbitrary numbers that look like a match. A real film has minutes of dialogue.
MIN_REFERENCE_SPEECH_S = 5.0
_SPEECH_FRAMES_PER_S = 100.0
# The excerpt of ffsubsync's output that goes to the log when a run fails.
_EXCERPT_LINES = 10
_CONSOLE_COLUMNS = "1000"
_EXCERPT_LINE_CHARS = 300
_WORK_PREFIX = ".align-"
# Environment variables that are not passed to ffsubsync: this tool's OpenSubtitles credentials
# and, as a net for others, any name with one of these words in it (compared in upper case).
_OPENSUBTITLES_ENV_PREFIX = "VC_OPENSUBTITLES_"
_SECRET_NAME_MARKERS = ("API_KEY", "PASSWORD", "TOKEN", "SECRET")
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
_SPEECH_RE = _report_pattern("total of speech segments")


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
    lines = normalise_newlines(text)
    return (
        _last_number(_OFFSET_RE, lines),
        _last_number(_SCALE_RE, lines),
        _last_number(_SCORE_RE, lines),
    )


def parse_reference_speech_s(text: str) -> float | None:
    """Seconds of speech ffsubsync found in the video's audio, or None if it did not say.

    It logs ``total of speech segments: 3022.0``, the number of 10 ms frames with speech.
    """
    frames = _last_number(_SPEECH_RE, normalise_newlines(text))
    return None if frames is None else frames / _SPEECH_FRAMES_PER_S


def accept_reference_speech(speech_s: float | None) -> tuple[bool, str]:
    """Whether the video's audio held enough speech (`MIN_REFERENCE_SPEECH_S`) to align to.

    An unknown total is rejected: ffsubsync always logs it for a video, so its absence means
    the log is not what this code understands.
    """
    if speech_s is None or not math.isfinite(speech_s):
        return False, "ffsubsync did not report how much speech it found"
    if speech_s < MIN_REFERENCE_SPEECH_S:
        return False, (
            f"only {speech_s:.1f} s of speech found in the audio "
            f"(at least {MIN_REFERENCE_SPEECH_S:g} s needed)"
        )
    return True, f"{speech_s:.1f} s of speech"


def accept_alignment(
    offset_s: float | None, scale: float | None, score: float | None, settings: Settings
) -> tuple[bool, str]:
    """Whether an alignment is believable, with a reason carrying the numbers.

    Rejected: a missing or non-finite number; an offset beyond ``align_max_offset_s`` seconds
    either way, or within `CLAMP_EPSILON_S` of it (the search was cut off there); a framerate
    scale further than ``align_scale_tolerance`` from 1; a score of zero or less, whatever
    ``align_min_score`` says (ffsubsync scores agreement minus disagreement, so zero is no
    evidence and a negative score is an anti-correlated, plainly wrong match); and, when
    ``align_min_score`` is above zero, a score below it.
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
    if abs(offset_s) >= settings.align_max_offset_s - CLAMP_EPSILON_S:
        return False, (
            f"offset {offset_s:+.3f} s is at the {settings.align_max_offset_s:g} s search limit, "
            "so the real offset is probably beyond it"
        )
    # ffsubsync tries only fixed ratios (24/23.976, 25/23.976, 25/24 and their inverses, at
    # most about 0.043 from 1) and one inferred from the two lengths, so only the inferred one
    # can exceed a tolerance of 0.05; it is the one that points at a different cut of the film.
    # The small slack keeps a scale exactly at the tolerance (1.05 - 1.0 is not exactly 0.05).
    if abs(scale - 1.0) > settings.align_scale_tolerance + _SCALE_SLACK:
        return False, (
            f"framerate scale {scale:.3f} differs from 1 by more than "
            f"{settings.align_scale_tolerance:g}"
        )
    if score <= 0:
        return False, f"score {score:.3f} is not a positive match"
    if settings.align_min_score > 0 and score < settings.align_min_score:
        return False, f"score {score:.3f} is below the minimum {settings.align_min_score:g}"
    return True, f"offset {offset_s:+.3f} s, framerate scale {scale:.3f}, score {score:.3f}"


def _is_secret_name(name: str) -> bool:
    """Whether an environment variable may hold a credential (compared in upper case)."""
    upper = name.upper()
    return upper.startswith(_OPENSUBTITLES_ENV_PREFIX) or any(
        marker in upper for marker in _SECRET_NAME_MARKERS
    )


def _child_env(tools: Tools) -> dict[str, str]:
    """A copy of this environment with the ffmpeg (and ffprobe) folders first on PATH.

    ffsubsync finds ffmpeg only through PATH. Python's stdio is set to UTF-8 so the log, which
    names the video, is decoded the way `run_tool` reads it, and the console is pinned to plain,
    wide output. ffsubsync needs no credentials, so the OpenSubtitles variables and any variable
    whose name looks like a secret are not passed on.
    """
    env = {name: value for name, value in os.environ.items() if not _is_secret_name(name)}
    key = next((name for name in env if name.upper() == "PATH"), "PATH")
    folders = list(dict.fromkeys(str(tool.parent) for tool in (tools.ffmpeg, tools.ffprobe)))
    existing = env.get(key, "")
    env[key] = os.pathsep.join([*folders, existing] if existing else folders)
    env["PYTHONIOENCODING"] = "utf-8"
    # ffsubsync logs through rich, which follows these variables: colour codes or a narrow
    # console that wraps a line would make the numbers unreadable.
    for name in ("FORCE_COLOR", "CLICOLOR_FORCE"):
        env.pop(name, None)
    env["NO_COLOR"] = "1"
    env["TERM"] = "dumb"
    env["COLUMNS"] = _CONSOLE_COLUMNS
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


def _log_excerpt(label: str, text: str) -> None:
    """Log the last lines of ffsubsync's output (redacted, each capped) at WARNING.

    The work folder and the log in it are removed when the run ends, so this is the only place
    the cause (a traceback, a missing module) is kept.
    """
    lines = [line.rstrip() for line in normalise_newlines(text).split("\n") if line.strip()]
    kept = [redact(line)[:_EXCERPT_LINE_CHARS] for line in lines[-_EXCERPT_LINES:]]
    if kept:
        _log.warning("%s: the end of ffsubsync's output:\n%s", label, "\n".join(kept))


def _failure_reason(text: str, what: str) -> str:
    if _NO_SPEECH_MARK in text:
        return f"{what}: no speech found in the audio"
    return what


def _validated_srt(path: Path, settings: Settings) -> tuple[str | None, str]:
    """(clean SRT text, "") for ffsubsync's output, or (None, why it is unusable)."""
    try:
        raw = read_bytes_capped(path, settings.subtitle_max_bytes)
    except FileNotFoundError:
        return None, "ffsubsync produced no subtitle"
    except OSError:
        return None, "the subtitle ffsubsync produced could not be read"
    if raw is None:
        limit = settings.subtitle_max_bytes
        return None, f"ffsubsync output rejected: subtitle is larger than {limit} bytes"
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
        text = _read_log(log_path, "")
        _log.warning("%s: ffsubsync exited with code %d", label, exc.returncode)
        _log_excerpt(label, text)
        return None, text, _failure_reason(text, f"ffsubsync exited with code {exc.returncode}")
    except ToolTimedOut as exc:
        text = _read_log(log_path, "")
        _log.warning("%s: %s", label, exc)
        _log_excerpt(label, text)
        return None, text, f"ffsubsync timed out after {exc.timeout_s:g} s"
    except ToolOutputError as exc:
        text = _read_log(log_path, "")
        _log.warning("%s: %s", label, exc)
        _log_excerpt(label, text)
        return None, text, "ffsubsync output could not be read"
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
        label = f"align subtitle to {media.name}"
        if result.returncode != 0:
            _log.warning("%s: ffsubsync exited with code %d", label, result.returncode)
            _log_excerpt(label, text)
            return AlignResult(
                ok=False,
                reason=_failure_reason(text, f"ffsubsync exited with code {result.returncode}"),
            )
        if _TRACEBACK_MARK in text:
            _log.warning("%s: ffsubsync reported an error", label)
            _log_excerpt(label, text)
            return AlignResult(
                ok=False, reason=_failure_reason(text, "ffsubsync reported an error")
            )
        offset_s, scale, score = parse_ffsubsync_log(text)
        enough_speech, speech_reason = accept_reference_speech(parse_reference_speech_s(text))
        if not enough_speech:
            return AlignResult(False, speech_reason, offset_s, scale, score)
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
