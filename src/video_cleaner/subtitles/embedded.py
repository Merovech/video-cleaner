"""Does a video's own subtitle set already cover English?

`classify_embedded` reads the subtitle tracks `probe` found. A forced-only English track (just the
foreign-dialogue signs) is told apart from a full one by how many cues it has per hour:
`density_candidates` names the tracks worth measuring, `measure_cue_density` extracts one with
ffmpeg and counts, and `classify_embedded` takes the measured densities back. Extracted text
comes out of a media file the user does not control, so it only ever goes through
`normalise_subtitle`; anything it rejects counts as no cues.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from video_cleaner.cmdline import flatten
from video_cleaner.config import Settings
from video_cleaner.probe import MediaInfo, SubtitleStream
from video_cleaner.progress import Reporter
from video_cleaner.subtitles.textfmt import SubtitleRejected, cues_per_hour, normalise_subtitle
from video_cleaner.tools import ToolOutputError, Tools, run_tool

_log = logging.getLogger("video_cleaner.subtitles.embedded")

TEXT_SUBTITLE_CODECS = frozenset(
    {"subrip", "srt", "ass", "ssa", "webvtt", "mov_text", "text", "subviewer"}
)
# Closed captions riding in the video stream: their language tag says little about the content
# and they cannot be extracted as a normal subtitle track, so they are never a full track.
_CLOSED_CAPTION_CODECS = frozenset({"eia_608", "eia_708", "cc_dec"})
_ENGLISH_TAGS = frozenset({"eng", "en", "english"})
_UNDETERMINED_TAG = "und"


class EmbeddedVerdict(StrEnum):
    HAS_ENGLISH = "has_english"
    NEEDS_ENGLISH = "needs_english"
    UNCLEAR = "unclear"


@dataclass(frozen=True)
class EmbeddedResult:
    verdict: EmbeddedVerdict
    reason: str
    # Stream indexes of forced-only English tracks that turned out to be full ones.
    reclassified: tuple[int, ...] = ()


def is_english(language: str) -> bool:
    """Whether a language tag means English ("und", the undetermined tag, does not)."""
    return language.strip().lower() in _ENGLISH_TAGS


def _is_closed_caption(stream: SubtitleStream) -> bool:
    return stream.codec in _CLOSED_CAPTION_CODECS


def _full_english(info: MediaInfo) -> tuple[SubtitleStream, ...]:
    return tuple(
        s
        for s in info.subtitles
        if is_english(s.language) and not s.forced and not _is_closed_caption(s)
    )


def density_candidates(info: MediaInfo) -> tuple[SubtitleStream, ...]:
    """The forced English text tracks whose cue density is worth measuring.

    Empty when the file already has a full English track. Image tracks cannot be measured
    without OCR, so a forced image track is simply forced-only.
    """
    if _full_english(info):
        return ()
    return tuple(
        s
        for s in info.subtitles
        if is_english(s.language) and s.forced and s.codec in TEXT_SUBTITLE_CODECS
    )


def classify_embedded(
    info: MediaInfo, densities: Mapping[int, float], *, threshold: float
) -> EmbeddedResult:
    """Decide whether *info* already has English subtitles.

    *densities* maps a stream index to its measured cues per hour (see `measure_cue_density`);
    a forced track at or above *threshold* is a full track marked forced by mistake, and is
    reported in `reclassified`. A file with an undetermined track, or English only as closed
    captions, and nothing better is `UNCLEAR`: it might have English, so it should not be
    searched for automatically.
    """
    full = _full_english(info)
    if full:
        return EmbeddedResult(
            EmbeddedVerdict.HAS_ENGLISH,
            f"has a full English subtitle track (stream {full[0].index})",
        )

    forced = tuple(s for s in info.subtitles if is_english(s.language) and s.forced)
    dense = tuple(
        s
        for s in forced
        if s.codec in TEXT_SUBTITLE_CODECS and densities.get(s.index, 0.0) >= threshold
    )
    if dense:
        best = max(densities[s.index] for s in dense)
        return EmbeddedResult(
            EmbeddedVerdict.HAS_ENGLISH,
            f"forced English track has {best:g} cues/hour, at least {threshold:g}: a full track",
            reclassified=tuple(s.index for s in dense),
        )

    captions = [s for s in info.subtitles if is_english(s.language) and _is_closed_caption(s)]
    if captions:
        return EmbeddedResult(
            EmbeddedVerdict.UNCLEAR,
            f"English only as closed captions ({captions[0].codec}, stream {captions[0].index})",
        )
    undetermined = [s for s in info.subtitles if s.language.strip().lower() == _UNDETERMINED_TAG]
    if undetermined:
        return EmbeddedResult(
            EmbeddedVerdict.UNCLEAR,
            f"subtitle track with no language tag (stream {undetermined[0].index})",
        )

    if forced:
        measured = [f"{densities[s.index]:g}" for s in forced if s.index in densities]
        detail = (
            f"{', '.join(measured)} cues/hour, below {threshold:g}"
            if measured
            else "not measured as full"
        )
        return EmbeddedResult(
            EmbeddedVerdict.NEEDS_ENGLISH, f"only forced English subtitles ({detail})"
        )
    return EmbeddedResult(EmbeddedVerdict.NEEDS_ENGLISH, "no English subtitle track")


def measure_cue_density(
    tools: Tools,
    path: Path,
    stream: SubtitleStream,
    duration_s: float,
    reporter: Reporter,
    *,
    max_bytes: int = Settings.subtitle_max_bytes,
    min_cues: int = 1,
) -> float:
    """Cues per hour of subtitle track *stream* in *path*, by extracting it as SRT.

    Returns 0.0 when nothing usable comes out: ffmpeg fails, the text is rejected (too large,
    empty, not a subtitle), or *duration_s* is not positive. A missing ffmpeg still raises
    `ToolNotFound`. A caller should log a 0.0 it did not expect (the survey does).

    Limitation: `run_tool` keeps only the last ~1 MB of stdout, so a track whose SRT text is
    larger than that is cut to its tail before it is parsed; its cue count, and so its density,
    is then an undercount (cues are counted over the full *duration_s*).
    """
    if not duration_s > 0:  # also true for NaN
        return 0.0
    cmd = [
        str(tools.ffmpeg),
        *flatten(
            # Never read the keyboard: ffmpeg would wait for input in a batch run.
            ("-nostdin",),
            # Errors only on stderr; the subtitle itself goes to stdout.
            ("-v", "error"),
            # The video file.
            ("-i", str(path)),
            # Just the one subtitle track, by its stream index.
            ("-map", f"0:{stream.index}"),
            # Convert whatever text format it is to SRT.
            ("-f", "srt"),
            # Write to stdout.
            ("-",),
        ),
    ]
    label = f"extract subtitle stream {stream.index} of {path.name}"
    try:
        result = run_tool(cmd, label=label, reporter=reporter, check=False)
    except ToolOutputError as exc:
        _log.warning("%s: output handling failed: %s", label, exc)
        return 0.0
    if result.returncode != 0:
        _log.warning("%s: ffmpeg exited with code %d", label, result.returncode)
        return 0.0
    try:
        subtitle = normalise_subtitle(
            result.stdout.encode("utf-8"), max_bytes=max_bytes, min_cues=min_cues
        )
    except SubtitleRejected as exc:
        _log.warning("%s: extracted text rejected: %s", label, exc)
        return 0.0
    return cues_per_hour(subtitle.cues, duration_s)
