"""Encode stage: sample windows, CRF interpolation, saving projection and the x265 command.

Everything here is pure (arithmetic and command lists as data) except `search_crf` and
`encode_full`, which are signature stubs held until Task 10b because no encode may run yet.
The x265 command encodes the first video stream only: audio, subtitles, chapters,
attachments and cover art are never mapped here; the mux step carries them over untouched.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from video_cleaner.config import Settings
from video_cleaner.probe import MediaInfo
from video_cleaner.progress import Reporter
from video_cleaner.tools import Tools

EDGE_SKIP_FRACTION = 0.10
X265_CRF_MIN = 0
X265_CRF_MAX = 51


def pick_windows(duration_s: float, count: int = 3, length_s: float = 30.0) -> list[float]:
    """Start times (seconds) of *count* sample windows of *length_s*, evenly spaced.

    Windows are centred in *count* equal slices of the middle 80% of the film, so the first
    and last 10% (titles, credits) are never sampled. When the middle is too short for the
    windows, starts are clamped into it (windows may overlap) and duplicates are dropped, so
    a very short film yields fewer than *count* starts. An unusable duration (zero, negative,
    NaN, infinite) or a *count* below 1 yields no windows.
    """
    if not length_s > 0:
        raise ValueError(f"length_s must be positive, got {length_s!r}")
    if count < 1 or not math.isfinite(duration_s) or duration_s <= 0:
        return []
    lo = EDGE_SKIP_FRACTION * duration_s
    hi = (1.0 - EDGE_SKIP_FRACTION) * duration_s
    latest = max(lo, hi - length_s)
    slice_len = (hi - lo) / count
    starts: list[float] = []
    for i in range(count):
        centre = lo + (i + 0.5) * slice_len
        start = min(max(centre - length_s / 2, lo), latest)
        if start not in starts:
            starts.append(start)
    return starts


def _round_half_up(value: float) -> int:
    return math.floor(value + 0.5)


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _bracketing_pair(
    ordered: Sequence[tuple[int, float]], target: float
) -> tuple[tuple[int, float], tuple[int, float]]:
    """The adjacent pair whose VMAFs bracket *target*, else the pair at the nearer end."""
    for left, right in itertools.pairwise(ordered):
        if min(left[1], right[1]) <= target <= max(left[1], right[1]):
            return left, right
    first_gap = abs(ordered[0][1] - target)
    last_gap = abs(ordered[-1][1] - target)
    if first_gap <= last_gap:
        return ordered[0], ordered[1]
    return ordered[-2], ordered[-1]


def interpolate_crf(
    points: Sequence[tuple[int, float]], target: float, lo: int = 16, hi: int = 24
) -> int:
    """The CRF expected to reach *target* VMAF, from sampled ``(crf, vmaf)`` points.

    Linear interpolation of CRF against VMAF between the two points that bracket the target
    (the nearest-end pair, extrapolated, when the target lies outside the sampled range),
    rounded half up and clamped to [*lo*, *hi*]. A pair with equal VMAF (or equal CRF) has no
    usable slope, so the middle CRF of the pair is returned, clamped. A single point returns
    its own CRF, clamped.
    """
    if not points:
        raise ValueError("points must not be empty")
    if lo > hi:
        raise ValueError(f"lo ({lo}) must not exceed hi ({hi})")
    ordered = sorted(points)
    if len(ordered) == 1:
        return _clamp(ordered[0][0], lo, hi)
    (crf_a, vmaf_a), (crf_b, vmaf_b) = _bracketing_pair(ordered, target)
    middle = _clamp(_round_half_up((crf_a + crf_b) / 2), lo, hi)
    if vmaf_a == vmaf_b or crf_a == crf_b:
        return middle
    estimate = crf_a + (target - vmaf_a) * (crf_b - crf_a) / (vmaf_b - vmaf_a)
    if not math.isfinite(estimate):
        return middle
    return _clamp(_round_half_up(estimate), lo, hi)


def projected_saving(sample_in_bytes: int, sample_out_bytes: int) -> float:
    """Fraction of the size saved: ``projected_saving(100, 40) == 0.60``.

    Zero input size gives 0.0 (nothing to save, and no division by zero). An output larger
    than the input gives a negative saving.
    """
    if sample_in_bytes <= 0:
        return 0.0
    return 1.0 - sample_out_bytes / sample_in_bytes


def build_x265_cmd(
    tools: Tools,
    src: Path,
    dst: Path,
    crf: int,
    settings: Settings,
    *,
    start_s: float | None = None,
    length_s: float | None = None,
) -> list[str]:
    """ffmpeg argument list encoding the first video stream of *src* to 10-bit x265.

    ``-ss`` / ``-t`` (when given) are placed before ``-i`` so ffmpeg seeks the input and
    limits it, which is how sample windows are cut. Only ``0:v:0`` is mapped and audio and
    subtitles are disabled, so cover art, attachments, data streams and chapters never reach
    the output. Progress goes to stdout as ``-progress pipe:1`` key/value blocks.
    """
    if not X265_CRF_MIN <= crf <= X265_CRF_MAX:
        raise ValueError(f"crf must be within {X265_CRF_MIN}..{X265_CRF_MAX}, got {crf}")
    if start_s is not None and not (math.isfinite(start_s) and start_s >= 0):
        raise ValueError(f"start_s must be a non-negative number, got {start_s!r}")
    if length_s is not None and not (math.isfinite(length_s) and length_s > 0):
        raise ValueError(f"length_s must be a positive number, got {length_s!r}")
    cmd = [str(tools.ffmpeg)]
    if start_s is not None:
        cmd += ["-ss", str(start_s)]
    if length_s is not None:
        cmd += ["-t", str(length_s)]
    cmd += [
        "-i",
        str(src),
        "-map",
        "0:v:0",
        "-c:v",
        "libx265",
        "-preset",
        settings.preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p10le",
        "-an",
        "-sn",
        "-progress",
        "pipe:1",
        "-nostats",
        "-y",
        str(dst),
    ]
    return cmd


@dataclass(frozen=True)
class CrfSearchResult:
    crf: int
    projected_saving: float
    points: tuple[tuple[int, float], ...]


def search_crf(
    info: MediaInfo, tools: Tools, settings: Settings, workdir: Path, reporter: Reporter
) -> CrfSearchResult:
    """Find the per-file CRF that reaches the VMAF target. HELD until Task 10b.

    Intended behaviour: encode the `pick_windows` sample windows of the scratch copy at CRF
    20 and CRF 24 with the same settings (preset, pixel format), score each encoded window
    against its source cut with `vmaf_mean`, average the scores per CRF, and feed the two
    ``(crf, mean_vmaf)`` points to `interpolate_crf` (target ``settings.vmaf_target``, bounds
    ``settings.crf_min..crf_max``). The projected saving comes from the sample sizes at the
    chosen CRF, re-estimated by a linear fit of log(size) against CRF between the two
    samples. Progress is reported through *reporter*; intermediates live in *workdir*.
    """
    raise NotImplementedError("held: Task 10b")


def encode_full(
    info: MediaInfo, tools: Tools, settings: Settings, crf: int, dst: Path, reporter: Reporter
) -> None:
    """Encode the whole video stream of ``info.path`` to *dst* at *crf*. HELD until Task 10b.

    Intended behaviour: run `build_x265_cmd` (no window) through `run_tool` with the x265
    progress parser and the reporter. Video only: ``0:v:0`` is mapped, so audio, subtitles,
    chapters, attachments and cover art are left for the mux step to copy untouched.
    """
    raise NotImplementedError("held: Task 10b")
