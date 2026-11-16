"""Output verification: the last gate before a file is moved to ``completed``.

`verify_output` probes the produced file and checks it against the original: duration within
1% + 2 s, a video stream, the same number of audio and subtitle streams, chapters, attachments
(cover art) and extra video streams kept, a clean full decode, and, for an encode, a size at
least ``min_saving`` smaller than the original. Every failed check is named in the reason.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from video_cleaner.probe import MediaInfo, ProbeError, probe
from video_cleaner.progress import Reporter
from video_cleaner.tools import Tools, run_tool

DURATION_TOLERANCE_FRACTION = 0.01
DURATION_TOLERANCE_S = 2.0

ProbeFn = Callable[[Path, Tools, Reporter], MediaInfo]
# Returns the decode errors as text; "" means the file decoded cleanly.
DecodeFn = Callable[[Tools, Path, Reporter], str]


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    reason: str
    # True when the size saving was the only failed check: the encode worked but is not worth
    # keeping, so the caller leaves the original alone instead of flagging it.
    too_large: bool = False


def decode_errors(tools: Tools, output: Path, reporter: Reporter) -> str:
    """Decode the video and audio of *output* to nowhere (``-f null``); "" if clean.

    Only what has to play to the end is decoded: real video streams (``0:V``, not cover art)
    and any audio. Subtitles, data and attachments are left out (``-sn -dn``): the null muxer
    has no subtitle codec, so mapping them would fail every file that has subtitles or fonts.
    """
    cmd = [
        str(tools.ffmpeg),
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(output),
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
    result = run_tool(cmd, label=f"decode check {output.name}", reporter=reporter, check=False)
    errors = result.stderr_tail.strip()
    if result.returncode != 0:
        exited = f"ffmpeg exited with code {result.returncode}"
        return f"{exited}: {errors}" if errors else exited
    return errors


def _count_problems(original: MediaInfo, out: MediaInfo) -> list[str]:
    problems: list[str] = []
    if out.video is None:
        problems.append("no video stream in the output")
    pairs = (
        ("audio streams", len(out.audio), len(original.audio)),
        ("subtitle streams", len(out.subtitles), len(original.subtitles)),
    )
    for label, got, want in pairs:
        if got != want:
            problems.append(f"{label}: {got} in the output, {want} in the original")
    kept = (
        ("chapters", out.chapters, original.chapters),
        ("attachments (cover art)", out.cover_art, original.cover_art),
        ("extra video streams", out.extra_video_streams, original.extra_video_streams),
    )
    for label, got, want in kept:
        if got < want:
            problems.append(f"{label}: {got} in the output, {want} in the original")
    return problems


def verify_output(
    tools: Tools,
    original: MediaInfo,
    output: Path,
    reporter: Reporter,
    *,
    encoded: bool,
    min_saving: float,
    original_duration_s: float | None = None,
    original_size: int | None = None,
    probe_fn: ProbeFn = probe,
    decode_fn: DecodeFn = decode_errors,
) -> VerifyResult:
    """Check *output* against *original*; see the module docstring for the checks.

    For a multi-part movie pass the summed duration and size of the parts as
    *original_duration_s* and *original_size*; the stream checks still use *original*.
    """
    try:
        out = probe_fn(output, tools, reporter)
    except ProbeError as exc:
        return VerifyResult(ok=False, reason=f"output check failed: {exc}")

    want_duration = original.duration_s if original_duration_s is None else original_duration_s
    want_size = original.size if original_size is None else original_size

    problems: list[str] = []
    tolerance = want_duration * DURATION_TOLERANCE_FRACTION + DURATION_TOLERANCE_S
    if abs(out.duration_s - want_duration) > tolerance:
        problems.append(
            f"duration {out.duration_s:.1f} s differs from the original {want_duration:.1f} s "
            f"by more than {tolerance:.1f} s"
        )
    problems.extend(_count_problems(original, out))
    if not problems:
        errors = decode_fn(tools, output, reporter)
        if errors:
            problems.append(f"decode check failed: {errors}")

    size_problem = ""
    if encoded:
        limit = want_size * (1.0 - min_saving)
        if out.size > limit:
            saved = 1.0 - out.size / want_size if want_size > 0 else 0.0
            size_problem = (
                f"output is only {saved:.1%} smaller than the original "
                f"({out.size} vs {want_size} bytes; needs at least {min_saving:.0%})"
            )

    if not problems and not size_problem:
        return VerifyResult(ok=True, reason="")
    reasons = [*problems, size_problem] if size_problem else problems
    return VerifyResult(
        ok=False, reason="; ".join(reasons), too_large=bool(size_problem) and not problems
    )
