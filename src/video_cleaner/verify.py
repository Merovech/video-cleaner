"""Output verification: the last gate before a file is moved to ``completed``.

`verify_output` probes the produced file and checks it against the original: duration within
1% + 2 s, a video stream, the same number of audio streams and of subtitle streams (plus the one
added subtitle track, when the caller muxed one in), chapters, attachments
(cover art) and extra video streams kept, a clean full decode, and, for an encode, 10-bit HEVC
video and a size at least ``min_saving`` smaller than the original. Every failed check is
named in the reason.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from video_cleaner.probe import MediaInfo, ProbeError, SubtitleStream, probe
from video_cleaner.progress import Reporter
from video_cleaner.tools import Tools, run_tool

DURATION_TOLERANCE_FRACTION = 0.01
DURATION_TOLERANCE_S = 2.0
ENCODED_CODEC = "hevc"
NEW_SUBTITLE_LANGUAGE = "eng"
# What build_x265_cmd writes is yuv420p10le; these 10-bit formats are accepted. An explicit set,
# not a substring test: "10" also occurs in the 8-bit yuv410p.
TEN_BIT_PIX_FMTS = frozenset({"yuv420p10le", "yuv420p10be", "yuv422p10le", "yuv444p10le", "p010le"})

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


def _new_subtitle_problems(stream: SubtitleStream, title: str | None) -> list[str]:
    """What is wrong with the added track: not English, flagged default or forced, misnamed.

    mkvmerge appends the new track after every source track, so it is the last subtitle stream.
    The title is compared only when the probe reports titles (``stream.title`` is not None) and
    the caller says which one it muxed.
    """
    name = f"new subtitle stream (index {stream.index})"
    problems: list[str] = []
    if stream.language.lower() != NEW_SUBTITLE_LANGUAGE:
        problems.append(f"{name} has language {stream.language}, expected {NEW_SUBTITLE_LANGUAGE}")
    if stream.default:
        problems.append(f"{name} is flagged default")
    if stream.forced:
        problems.append(f"{name} is flagged forced")
    if title is not None and stream.title is not None and stream.title != title:
        problems.append(f"{name} has title {stream.title!r}, expected {title!r}")
    return problems


def _count_problems(
    original: MediaInfo, out: MediaInfo, new_subtitles: int, title: str | None
) -> list[str]:
    problems: list[str] = []
    if out.video is None:
        problems.append("no video stream in the output")
    if len(out.audio) != len(original.audio):
        problems.append(
            f"audio streams: {len(out.audio)} in the output, {len(original.audio)} in the original"
        )
    want_subtitles = len(original.subtitles) + new_subtitles
    if len(out.subtitles) != want_subtitles:
        if new_subtitles:
            problems.append(
                f"subtitle streams: {len(out.subtitles)} in the output, {want_subtitles} expected "
                f"({len(original.subtitles)} in the original + {new_subtitles} new)"
            )
        else:
            problems.append(
                f"subtitle streams: {len(out.subtitles)} in the output, "
                f"{len(original.subtitles)} in the original"
            )
    elif new_subtitles:
        problems.extend(_new_subtitle_problems(out.subtitles[-1], title))
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
    expected_new_subtitles: int = 0,
    expected_subtitle_title: str | None = None,
) -> VerifyResult:
    """Check *output* against *original*; see the module docstring for the checks.

    *expected_new_subtitles* is 0, or 1 when a subtitle was muxed in: the output must then have
    one more subtitle stream than the original, and the last one (the new track) must be
    English, neither default nor forced, and carry *expected_subtitle_title* when that is given.
    Any other value raises `ValueError`.

    For a multi-part movie pass the summed duration and size of the parts as
    *original_duration_s* and *original_size*; the stream checks still use *original*.
    """
    if expected_new_subtitles not in (0, 1):
        raise ValueError(f"expected_new_subtitles must be 0 or 1, got {expected_new_subtitles}")
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
    problems.extend(_count_problems(original, out, expected_new_subtitles, expected_subtitle_title))
    if encoded and out.video is not None:
        if out.video.codec.lower() != ENCODED_CODEC:
            problems.append(f"encoded video is codec {out.video.codec}, expected {ENCODED_CODEC}")
        if out.video.pix_fmt not in TEN_BIT_PIX_FMTS:
            problems.append(f"encoded video has pixel format {out.video.pix_fmt}, expected 10-bit")
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
