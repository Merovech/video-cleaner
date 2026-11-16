"""ffprobe wrapper and the media model every later stage consumes.

`probe` runs ffprobe through `run_tool`; `parse_ffprobe` turns its JSON into an immutable
`MediaInfo`. Anything that makes a file unusable (ffprobe error, bad JSON, no format section,
zero or missing duration) raises `ProbeError`, so the caller can flag the file and move on.
Cover art (attached pictures and attachment streams) is counted, never treated as the video.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from video_cleaner.progress import Reporter
from video_cleaner.tools import ToolOutputError, Tools, run_tool

_log = logging.getLogger("video_cleaner.probe")

HDR_TRANSFERS = frozenset({"smpte2084", "arib-std-b67"})
INTERLACED_FIELD_ORDERS = frozenset({"tt", "bb", "tb", "bt"})
IMAGE_SUBTITLE_CODECS = frozenset({"hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle", "xsub"})
UNKNOWN_FIELD_ORDER = "unknown"
UNDETERMINED_LANGUAGE = "und"

_FFPROBE_ARGS = (
    "-v",
    "error",
    "-print_format",
    "json",
    "-show_format",
    "-show_streams",
    "-show_chapters",
)


class ProbeError(Exception):
    """A file could not be probed into a usable `MediaInfo`."""

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"cannot probe {path}: {reason}")


@dataclass(frozen=True)
class VideoStream:
    codec: str
    width: int
    height: int
    pix_fmt: str
    fps: float
    bit_rate: int | None
    transfer: str | None
    dv_profile: int | None
    field_order: str


@dataclass(frozen=True)
class AudioStream:
    index: int
    codec: str
    language: str
    title: str


@dataclass(frozen=True)
class SubtitleStream:
    index: int
    codec: str
    language: str
    forced: bool
    sdh: bool
    is_image: bool


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    size: int
    mtime: float
    duration_s: float
    container: str
    video: VideoStream | None
    audio: tuple[AudioStream, ...]
    subtitles: tuple[SubtitleStream, ...]
    chapters: int
    cover_art: int
    # Container-level bit rate in bit/s as ffprobe reported it; None when it did not.
    format_bit_rate: int | None = None

    @property
    def bitrate_kbps(self) -> float:
        """Overall bit rate: the container's figure if present, else size over duration."""
        if self.format_bit_rate is not None and self.format_bit_rate > 0:
            return self.format_bit_rate / 1000
        return self.size * 8 / self.duration_s / 1000

    @property
    def is_hdr(self) -> bool:
        return self.video is not None and self.video.transfer in HDR_TRANSFERS

    @property
    def is_interlaced(self) -> bool:
        return self.video is not None and self.video.field_order in INTERLACED_FIELD_ORDERS


def _tag(stream: Mapping[str, Any], name: str) -> str:
    """Case-insensitive stream tag lookup ('' when absent)."""
    tags = stream.get("tags")
    if not isinstance(tags, Mapping):
        return ""
    for key, value in tags.items():
        if isinstance(key, str) and key.lower() == name:
            return str(value)
    return ""


def _flag(stream: Mapping[str, Any], name: str) -> bool:
    disposition = stream.get("disposition")
    return isinstance(disposition, Mapping) and bool(disposition.get(name))


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_rate(value: Any) -> float:
    """ffprobe rates are 'num/den' strings; '0/0' and junk give 0.0."""
    if not isinstance(value, str):
        return 0.0
    num, _, den = value.partition("/")
    try:
        numerator = float(num)
        denominator = float(den) if den else 1.0
    except ValueError:
        return 0.0
    if denominator == 0:
        return 0.0
    rate = numerator / denominator
    return rate if math.isfinite(rate) else 0.0


def _dv_profile(stream: Mapping[str, Any]) -> int | None:
    side_data = stream.get("side_data_list")
    if not isinstance(side_data, list):
        return None
    for entry in side_data:
        if isinstance(entry, Mapping) and "dv_profile" in entry:
            return _int_or_none(entry["dv_profile"])
    return None


def _video_stream(stream: Mapping[str, Any]) -> VideoStream:
    return VideoStream(
        codec=str(stream.get("codec_name", "")),
        width=_int_or_none(stream.get("width")) or 0,
        height=_int_or_none(stream.get("height")) or 0,
        pix_fmt=str(stream.get("pix_fmt", "")),
        fps=_parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate")),
        bit_rate=_int_or_none(stream.get("bit_rate")),
        transfer=str(stream["color_transfer"]) if stream.get("color_transfer") else None,
        dv_profile=_dv_profile(stream),
        field_order=str(stream.get("field_order") or UNKNOWN_FIELD_ORDER),
    )


def _is_sdh(stream: Mapping[str, Any]) -> bool:
    if _flag(stream, "hearing_impaired"):
        return True
    title = _tag(stream, "title").lower()
    return "sdh" in title or "hearing impaired" in title


def _is_forced(stream: Mapping[str, Any]) -> bool:
    return _flag(stream, "forced") or "forced" in _tag(stream, "title").lower()


def _subtitle_stream(stream: Mapping[str, Any]) -> SubtitleStream:
    codec = str(stream.get("codec_name", ""))
    return SubtitleStream(
        index=int(stream["index"]),
        codec=codec,
        language=_tag(stream, "language") or UNDETERMINED_LANGUAGE,
        forced=_is_forced(stream),
        sdh=_is_sdh(stream),
        is_image=codec in IMAGE_SUBTITLE_CODECS,
    )


def _audio_stream(stream: Mapping[str, Any]) -> AudioStream:
    return AudioStream(
        index=int(stream["index"]),
        codec=str(stream.get("codec_name", "")),
        language=_tag(stream, "language") or UNDETERMINED_LANGUAGE,
        title=_tag(stream, "title"),
    )


def _duration(fmt: Mapping[str, Any], path: Path) -> float:
    raw = fmt.get("duration")
    try:
        duration = float(raw) if raw is not None else math.nan
    except (TypeError, ValueError):
        raise ProbeError(path, f"missing or unreadable duration ({raw!r})") from None
    if not math.isfinite(duration) or duration <= 0:
        raise ProbeError(path, f"zero or invalid duration ({raw!r})")
    return duration


def _build(data: Mapping[str, Any], path: Path, size: int, mtime: float) -> MediaInfo:
    fmt = data.get("format")
    if not isinstance(fmt, Mapping):
        raise ProbeError(path, "ffprobe output has no format section")
    duration = _duration(fmt, path)
    streams = data.get("streams")
    chapters = data.get("chapters")

    video: VideoStream | None = None
    audio: list[AudioStream] = []
    subtitles: list[SubtitleStream] = []
    cover_art = 0
    for stream in streams if isinstance(streams, list) else []:
        kind = stream.get("codec_type")
        if kind == "attachment" or (kind == "video" and _flag(stream, "attached_pic")):
            cover_art += 1
        elif kind == "video":
            if video is None:
                video = _video_stream(stream)
        elif kind == "audio":
            audio.append(_audio_stream(stream))
        elif kind == "subtitle":
            subtitles.append(_subtitle_stream(stream))
    return MediaInfo(
        path=path,
        size=size,
        mtime=mtime,
        duration_s=duration,
        container=str(fmt.get("format_name", "")),
        video=video,
        audio=tuple(audio),
        subtitles=tuple(subtitles),
        chapters=len(chapters) if isinstance(chapters, list) else 0,
        cover_art=cover_art,
        format_bit_rate=_int_or_none(fmt.get("bit_rate")),
    )


def parse_ffprobe(data: dict[str, Any], path: Path, size: int, mtime: float) -> MediaInfo:
    """Turn ffprobe's JSON (``-show_format -show_streams -show_chapters``) into `MediaInfo`.

    Attached pictures (``disposition.attached_pic``) and ``attachment`` streams are counted in
    ``cover_art`` and never become ``video``; ``video`` is the first other video stream.
    Raises `ProbeError` for a missing format section, a zero or missing duration, or streams
    too malformed to read.
    """
    try:
        return _build(data, path, size, mtime)
    except ProbeError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ProbeError(path, f"unexpected ffprobe output: {exc!r}") from exc


def probe(path: Path, tools: Tools, reporter: Reporter) -> MediaInfo:
    """Probe *path* with ffprobe. Raises `ProbeError` (never a raw tool error) on any failure.

    ``ToolNotFound`` still propagates: a missing ffprobe is a setup problem, not a bad file.
    """
    try:
        st = os.stat(path)
    except OSError as exc:
        raise ProbeError(path, f"cannot stat file: {exc}") from exc
    cmd = [str(tools.ffprobe), *_FFPROBE_ARGS, "-i", str(path)]
    try:
        result = run_tool(cmd, label=f"probe {path.name}", reporter=reporter, check=False)
    except ToolOutputError as exc:
        raise ProbeError(path, f"ffprobe output handling failed: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr_tail.strip() or "no error output"
        raise ProbeError(path, f"ffprobe exited with code {result.returncode}: {detail}")
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ProbeError(path, f"ffprobe output is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ProbeError(path, "ffprobe output is not a JSON object")
    info = parse_ffprobe(data, path, st.st_size, st.st_mtime)
    _log.debug("probed %s: %d bytes, %.1fs", path.name, info.size, info.duration_s)
    return info
