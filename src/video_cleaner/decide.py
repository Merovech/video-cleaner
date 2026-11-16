"""Decision engine: what to do with one probed file.

Pure function over `MediaInfo` and `Settings`; no I/O. The order of the checks matters:

1. No video stream: flag.
2. Not eligible for encoding (already HEVC/AV1, 4K, below the bit rate floor or the minimum
   file size): never encoded and never flagged, even when HDR, Dolby Vision or interlaced.
   Remuxed only when new subtitles or a multi-part join call for it, otherwise left alone
   (MP4 and AVI included: no container conversion for its own sake).
3. Eligible but Dolby Vision, HDR or interlaced: flag, because the encode would damage it.
4. Otherwise encode; the new video is muxed with the untouched streams, so remux is set too.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from video_cleaner.config import Settings
from video_cleaner.probe import MediaInfo, VideoStream

EFFICIENT_CODECS = frozenset({"hevc", "av1"})
UHD_MIN_HEIGHT = 2000


class Action(StrEnum):
    LEAVE_ALONE = "leave_alone"
    ENCODE = "encode"
    REMUX = "remux"
    FLAG = "flag"


@dataclass(frozen=True)
class Decision:
    action: Action
    reasons: tuple[str, ...]
    encode: bool
    remux: bool


def _no_encode_reasons(info: MediaInfo, video: VideoStream, settings: Settings) -> list[str]:
    reasons: list[str] = []
    codec = video.codec.lower()
    if codec in EFFICIENT_CODECS:
        reasons.append(f"already {codec}")
    if video.height >= UHD_MIN_HEIGHT:
        reasons.append(f"4K (height {video.height} >= {UHD_MIN_HEIGHT})")
    kbps = info.bitrate_kbps
    if kbps < settings.bitrate_floor_kbps:
        reasons.append(f"bitrate {kbps:.0f} kbps below floor {settings.bitrate_floor_kbps} kbps")
    if settings.min_file_size > 0 and info.size < settings.min_file_size:
        reasons.append(f"size {info.size} bytes below minimum {settings.min_file_size} bytes")
    return reasons


def _remux_reasons(needs_subs: bool, multipart: bool) -> list[str]:
    reasons: list[str] = []
    if needs_subs:
        reasons.append("new subtitles to mux in")
    if multipart:
        reasons.append("multi-part file to join")
    return reasons


def _flag_reasons(info: MediaInfo, video: VideoStream) -> list[str]:
    reasons: list[str] = []
    # Checked on its own: Dolby Vision profile 5 carries no HDR transfer characteristic.
    if video.dv_profile is not None:
        reasons.append(f"Dolby Vision (profile {video.dv_profile})")
    if info.is_hdr:
        reasons.append(f"HDR (transfer {video.transfer})")
    if info.is_interlaced:
        reasons.append(f"interlaced (field order {video.field_order})")
    return reasons


def decide(
    info: MediaInfo, settings: Settings, *, needs_subs: bool = False, multipart: bool = False
) -> Decision:
    """Choose the action for *info*; `Decision.reasons` says why, with the numbers."""
    video = info.video
    if video is None:
        return Decision(Action.FLAG, ("no video stream",), encode=False, remux=False)

    skip = _no_encode_reasons(info, video, settings)
    if skip:
        remux = _remux_reasons(needs_subs, multipart)
        action = Action.REMUX if remux else Action.LEAVE_ALONE
        return Decision(action, (*skip, *remux), encode=False, remux=bool(remux))

    flagged = _flag_reasons(info, video)
    if flagged:
        return Decision(Action.FLAG, tuple(flagged), encode=False, remux=False)

    reason = (
        f"{video.codec.lower()} {video.height}p at {info.bitrate_kbps:.0f} kbps "
        f"(floor {settings.bitrate_floor_kbps} kbps)"
    )
    return Decision(
        Action.ENCODE, (reason, *_remux_reasons(needs_subs, multipart)), encode=True, remux=True
    )
