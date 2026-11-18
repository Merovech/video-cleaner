"""Encode stage: sample windows, CRF search, saving projection and the x265 encode.

The arithmetic and the command lists are pure. `measure_windows`, `encode_windows`,
`search_crf` and `encode_full` run ffprobe / ffmpeg through `run_tool`. The x265 command
encodes the first real video stream only (``0:V:0``, never an attached picture) and strips
chapters and metadata: audio, subtitles, chapters, tags, attachments and cover art are left
for the mux step, which copies them from the source untouched.
"""

from __future__ import annotations

import itertools
import logging
import math
import numbers
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from video_cleaner import quality
from video_cleaner.cmdline import PROGRESS_OPTIONS, Option, flatten
from video_cleaner.config import Settings
from video_cleaner.probe import MediaInfo
from video_cleaner.progress import Progress, Reporter, parse_ffmpeg_progress
from video_cleaner.tools import Tools, run_tool

_log = logging.getLogger(__name__)

# The CRF search samples these two CRFs ...
SEARCH_CRFS: tuple[int, int] = (20, 24)
# ... over this many windows of this length.
SAMPLE_WINDOW_COUNT = 3
SAMPLE_WINDOW_S = 30.0
# ffprobe prints packet times with 6 decimals; window bounds are compared with this slack.
_TIME_EPSILON_S = 1e-6

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
    its own CRF, clamped. An extrapolation that overflows to infinity is clamped to the bound
    on its side.

    A failed measurement is never hidden: a non-finite VMAF or target, or a CRF that is not
    an integer (any `numbers.Integral` such as a numpy integer is accepted; ``bool`` is not),
    raises ValueError.
    """
    if not points:
        raise ValueError("points must not be empty")
    if lo > hi:
        raise ValueError(f"lo ({lo}) must not exceed hi ({hi})")
    if not math.isfinite(target):
        raise ValueError(f"target VMAF must be a finite number, got {target!r}")
    normalised: list[tuple[int, float]] = []
    for crf, vmaf in points:
        if isinstance(crf, bool) or not isinstance(crf, numbers.Integral):
            raise ValueError(f"CRF must be an integer, got {crf!r}")
        if not math.isfinite(vmaf):
            raise ValueError(f"VMAF of CRF {crf} must be a finite number, got {vmaf!r}")
        normalised.append((int(crf), float(vmaf)))
    ordered = sorted(normalised)
    if len(ordered) == 1:
        return _clamp(ordered[0][0], lo, hi)
    (crf_a, vmaf_a), (crf_b, vmaf_b) = _bracketing_pair(ordered, target)
    middle = _clamp(_round_half_up((crf_a + crf_b) / 2), lo, hi)
    if vmaf_a == vmaf_b or crf_a == crf_b:
        return middle
    estimate = crf_a + (target - vmaf_a) * (crf_b - crf_a) / (vmaf_b - vmaf_a)
    if math.isinf(estimate):  # finite inputs cannot give NaN, only an overflowing slope
        return hi if estimate > 0 else lo
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
    limits it, which is how sample windows are cut. Only ``0:V:0`` is mapped (capital V:
    the first real video stream, never an attached cover picture), audio and subtitles are
    disabled, and ``-map_chapters -1`` / ``-map_metadata -1`` stop ffmpeg's default copy of
    chapters and global/stream tags, so the output holds the bare video track for the mux.
    Progress goes to stdout as ``-progress pipe:1`` key/value blocks.
    """
    if not X265_CRF_MIN <= crf <= X265_CRF_MAX:
        raise ValueError(f"crf must be within {X265_CRF_MIN}..{X265_CRF_MAX}, got {crf}")
    if start_s is not None and not (math.isfinite(start_s) and start_s >= 0):
        raise ValueError(f"start_s must be a non-negative number, got {start_s!r}")
    if length_s is not None and not (math.isfinite(length_s) and length_s > 0):
        raise ValueError(f"length_s must be a positive number, got {length_s!r}")
    window: list[Option] = []
    if start_s is not None:
        window.append(("-ss", str(start_s)))  # seek the input to the window start (fast)
    if length_s is not None:
        window.append(("-t", str(length_s)))  # and stop reading after the window length
    return [
        str(tools.ffmpeg),
        *flatten(
            ("-nostdin",),  # never wait for keyboard input
            *window,
            ("-i", str(src)),  # the input file
            ("-map", "0:V:0"),  # only the first real video stream (not cover art)
            ("-map_chapters", "-1"),  # drop chapters: the mux takes them from the source
            ("-map_metadata", "-1"),  # drop global and stream tags, same reason
            ("-c:v", "libx265"),  # the encoder
            ("-preset", settings.preset),  # speed against compression (slower = smaller)
            ("-crf", str(crf)),  # constant quality: lower is better and bigger
            ("-pix_fmt", "yuv420p10le"),  # 10-bit 4:2:0, which avoids banding in dark scenes
            ("-an",),  # no audio
            ("-sn",),  # no subtitles
            *PROGRESS_OPTIONS,
            ("-y",),  # overwrite the work file of an interrupted earlier attempt
            (str(dst),),  # the output file
        ),
    ]


class EncodeError(Exception):
    """A sample or full encode produced nothing usable; the runner flags the file."""


@dataclass(frozen=True)
class CrfSearchResult:
    crf: int
    projected_saving: float
    points: tuple[tuple[int, float], ...]
    # True when *crf* lies outside the two sampled CRFs, so its size was extrapolated.
    extrapolated: bool = False
    # (start_s, length_s) of each sample window.
    windows: tuple[tuple[float, float], ...] = ()
    # (crf, encoded bytes of all windows together) per sampled CRF.
    sample_bytes: tuple[tuple[int, int], ...] = ()
    # Source video bytes of all windows together.
    source_bytes: int = 0


@dataclass(frozen=True)
class Window:
    """A sample window of a source and the bytes of its first video stream inside it."""

    start_s: float
    length_s: float
    source_bytes: int


@dataclass(frozen=True)
class WindowEncode:
    """One sample window encoded at one CRF."""

    start_s: float
    length_s: float
    source_bytes: int
    encoded: Path
    encoded_bytes: int


def sample_windows(duration_s: float, count: int, length_s: float) -> list[tuple[float, float]]:
    """``(start_s, length_s)`` of the `pick_windows` windows, each cut at the end of the film.

    A film shorter than one window yields one shorter window; an unusable duration none.
    """
    spans: list[tuple[float, float]] = []
    for start in pick_windows(duration_s, count, length_s):
        length = min(length_s, duration_s - start)
        if length > 0:
            # Milliseconds: what the commands carry, so measuring and cutting agree.
            spans.append((round(start, 3), round(length, 3)))
    return spans


def build_packet_sizes_cmd(tools: Tools, src: Path, start_s: float, length_s: float) -> list[str]:
    """ffprobe argument list listing the packets of the first video stream in a window.

    ``-read_intervals`` makes ffprobe seek to the window instead of reading the whole file;
    reading starts at the keyframe before *start_s*, so `parse_packet_bytes` filters by time.
    """
    return [
        str(tools.ffprobe),
        *flatten(
            ("-v", "error"),  # print nothing but errors
            ("-select_streams", "V:0"),  # the first real video stream
            # An absolute end: a relative ``%+length`` counts from the keyframe ffprobe seeks to.
            ("-read_intervals", f"{start_s:.3f}%{start_s + length_s:.3f}"),
            ("-show_entries", "packet=pts_time,dts_time,size"),  # per packet: time and bytes
            ("-of", "csv=p=0"),  # plain comma-separated values, no header or section names
            (str(src),),
        ),
    ]


def _number(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def parse_packet_bytes(text: str, start_s: float, length_s: float) -> int:
    """Sum the sizes of the ``pts_time,dts_time,size`` packet lines inside the window.

    A packet belongs to the window when its time (pts, else dts) is in
    ``[start_s, start_s + length_s)``; lines that do not parse are ignored.
    """
    total = 0
    end = start_s + length_s
    for line in text.splitlines():
        fields = line.strip().split(",")
        if len(fields) < 3:
            continue
        when = _number(fields[0])
        if when is None:
            when = _number(fields[1])
        size = _number(fields[2])
        if when is None or size is None:
            continue
        if start_s - _TIME_EPSILON_S <= when < end - _TIME_EPSILON_S:
            total += int(size)
    return total


def measure_windows(
    tools: Tools, src: Path, spans: Sequence[tuple[float, float]], reporter: Reporter
) -> tuple[Window, ...]:
    """The source video bytes in each window of *src* (read-only ffprobe).

    Raises `EncodeError` for a window without video packets: no saving can be projected.
    """
    windows: list[Window] = []
    for start, length in spans:
        result = run_tool(
            build_packet_sizes_cmd(tools, src, start, length),
            label=f"measure {src.name}",
            reporter=reporter,
        )
        source_bytes = parse_packet_bytes(result.stdout, start, length)
        if source_bytes <= 0:
            raise EncodeError(f"{src.name}: no video packets in the sample window at {start:.1f} s")
        windows.append(Window(start, length, source_bytes))
    return tuple(windows)


def _ffmpeg_parser(duration_s: float) -> Callable[[Iterable[str]], Iterator[Progress]]:
    def parser(lines: Iterable[str]) -> Iterator[Progress]:
        return parse_ffmpeg_progress(lines, duration_s)

    return parser


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def encode_windows(
    tools: Tools,
    settings: Settings,
    src: Path,
    crf: int,
    windows: Sequence[Window],
    workdir: Path,
    reporter: Reporter,
) -> tuple[WindowEncode, ...]:
    """Encode each window of *src* at *crf* into *workdir* with `build_x265_cmd`.

    The same preset and pixel format as the full encode. Raises `EncodeError` for an empty
    or missing sample and `ToolFailed` when ffmpeg fails.
    """
    encodes: list[WindowEncode] = []
    for index, window in enumerate(windows, start=1):
        what = f"CRF {crf} window {index}/{len(windows)}"
        dst = workdir / f"sample-crf{crf}-w{index}.mkv"
        reporter.stage(f"sample {what}: {src.name}")
        cmd = build_x265_cmd(
            tools, src, dst, crf, settings, start_s=window.start_s, length_s=window.length_s
        )
        run_tool(
            cmd,
            label=f"sample {what} {src.name}",
            reporter=reporter,
            parser=_ffmpeg_parser(window.length_s),
            stderr_log=dst.with_name(dst.name + ".stderr.log"),
        )
        encoded_bytes = _size(dst)
        if encoded_bytes <= 0:
            raise EncodeError(f"{src.name}: the {what} sample is empty")
        encodes.append(
            WindowEncode(window.start_s, window.length_s, window.source_bytes, dst, encoded_bytes)
        )
    return tuple(encodes)


def choose_crf(points: Sequence[tuple[int, float]], target: float, lo: int, hi: int) -> int:
    """The CRF for *target* from the sampled ``(crf, vmaf)`` points, within [*lo*, *hi*].

    VMAF at or above the target at every point gives *hi* (the size ceiling); below it at
    every point gives *lo*; otherwise `interpolate_crf`.
    """
    if all(vmaf >= target for _, vmaf in points):
        return hi
    if all(vmaf < target for _, vmaf in points):
        return lo
    return interpolate_crf(points, target, lo, hi)


def fit_log_size(sizes: Sequence[tuple[int, int]], crf: int) -> float:
    """Size at *crf* from a straight line through log(size) against CRF of two samples.

    Outside the two sampled CRFs the line is extrapolated. Raises ValueError for anything
    but two positive samples at different CRFs.
    """
    if len(sizes) != 2:
        raise ValueError(f"need exactly two sample sizes, got {len(sizes)}")
    (crf_a, size_a), (crf_b, size_b) = sizes
    if size_a <= 0 or size_b <= 0 or crf_a == crf_b:
        raise ValueError(f"unusable sample sizes {sizes!r}")
    slope = (math.log(size_b) - math.log(size_a)) / (crf_b - crf_a)
    return math.exp(math.log(size_a) + slope * (crf - crf_a))


def project_file_saving(
    file_size: int,
    duration_s: float,
    sample_s: float,
    source_sample_bytes: int,
    encoded_sample_bytes: float,
) -> float:
    """Projected saving of the whole file, from the video bytes of the sample windows.

    The windows' source and encoded video bytes are scaled to the film's duration. The video
    estimate is capped at the file size; everything else in the file (audio, subtitles,
    attachments) is assumed to be copied unchanged.
    """
    if file_size <= 0 or sample_s <= 0:
        return 0.0
    if source_sample_bytes <= 0:
        return 0.0
    video_in = min(float(file_size), source_sample_bytes * duration_s / sample_s)
    video_out = video_in * encoded_sample_bytes / source_sample_bytes
    return projected_saving(file_size, round(file_size - video_in + video_out))


def search_crf(
    info: MediaInfo,
    tools: Tools,
    settings: Settings,
    workdir: Path,
    reporter: Reporter,
    *,
    window_count: int = SAMPLE_WINDOW_COUNT,
    window_length_s: float = SAMPLE_WINDOW_S,
) -> CrfSearchResult:
    """Find the per-file CRF that reaches ``settings.vmaf_target``.

    The `sample_windows` of ``info.path`` (the scratch copy) are encoded at each of
    `SEARCH_CRFS` with the real encode's preset and pixel format; each encoded window is scored
    with `quality.vmaf_mean` against the same window of the source and the scores are averaged
    per CRF. `choose_crf` turns the two points into the CRF (clamped to
    ``settings.crf_min..crf_max``). The projected saving comes from the windows' byte totals: a
    straight line through log(size) against CRF (`fit_log_size`) evaluated at the chosen CRF,
    scaled to the file by `project_file_saving`; ``extrapolated`` says when the CRF lies
    outside the sampled ones. *window_count* and *window_length_s* exist for tests.

    Raises `EncodeError` when there is no window, a window has no video or a sample is empty,
    `VmafError` for an unusable score and `ToolFailed` when a tool fails. Samples are deleted
    from *workdir* once measured.
    """
    spans = sample_windows(info.duration_s, window_count, window_length_s)
    if not spans:
        raise EncodeError(f"{info.path.name}: no sample windows (duration {info.duration_s!r} s)")
    workdir.mkdir(parents=True, exist_ok=True)
    windows = measure_windows(tools, info.path, spans, reporter)
    points: list[tuple[int, float]] = []
    sizes: list[tuple[int, int]] = []
    for crf in SEARCH_CRFS:
        encodes = encode_windows(tools, settings, info.path, crf, windows, workdir, reporter)
        try:
            scores = [
                quality.vmaf_mean(
                    tools,
                    enc.encoded,
                    info.path,
                    workdir,
                    reporter,
                    reference_start_s=enc.start_s,
                    reference_length_s=enc.length_s,
                )
                for enc in encodes
            ]
        finally:
            for enc in encodes:
                enc.encoded.unlink(missing_ok=True)
        points.append((crf, sum(scores) / len(scores)))
        sizes.append((crf, sum(enc.encoded_bytes for enc in encodes)))
    chosen = choose_crf(points, settings.vmaf_target, settings.crf_min, settings.crf_max)
    source_bytes = sum(w.source_bytes for w in windows)
    saving = project_file_saving(
        info.size,
        info.duration_s,
        sum(w.length_s for w in windows),
        source_bytes,
        fit_log_size(sizes, chosen),
    )
    sampled = [crf for crf, _ in sizes]
    result = CrfSearchResult(
        crf=chosen,
        projected_saving=saving,
        points=tuple(points),
        extrapolated=not min(sampled) <= chosen <= max(sampled),
        windows=tuple(spans),
        sample_bytes=tuple(sizes),
        source_bytes=source_bytes,
    )
    _log.info(
        "%s: CRF search %s -> CRF %d%s, projected saving %.1f%%",
        info.path.name,
        ", ".join(f"CRF {c}: VMAF {v:.2f}" for c, v in points),
        chosen,
        " (size extrapolated)" if result.extrapolated else "",
        saving * 100,
    )
    return result


def encode_full(
    info: MediaInfo, tools: Tools, settings: Settings, crf: int, dst: Path, reporter: Reporter
) -> None:
    """Encode the whole first video stream of ``info.path`` to *dst* at *crf*.

    `build_x265_cmd` (no window) through `run_tool` with the ffmpeg progress parser, labelled
    ``encode <name>``, stderr kept in ``<dst>.stderr.log``. Video only: ``0:V:0`` is mapped and
    chapters and metadata are stripped, so audio, subtitles, chapters, tags, attachments and
    cover art are left for the mux step to copy from the source untouched. Raises `EncodeError`
    when the output is missing or empty and `ToolFailed` when ffmpeg fails.
    """
    reporter.stage(f"encode CRF {crf}: {info.path.name}")
    run_tool(
        build_x265_cmd(tools, info.path, dst, crf, settings),
        label=f"encode {info.path.name}",
        reporter=reporter,
        parser=_ffmpeg_parser(info.duration_s),
        stderr_log=dst.with_name(dst.name + ".stderr.log"),
    )
    if _size(dst) <= 0:
        raise EncodeError(f"{info.path.name}: the encode at CRF {crf} is empty")
