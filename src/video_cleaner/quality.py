"""Quality: the post-encode guard (pure arithmetic) and the VMAF scorer (ffmpeg ``libvmaf``)."""

from __future__ import annotations

import json
import math
import os
import uuid
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from video_cleaner.progress import Progress, Reporter, parse_ffmpeg_progress
from video_cleaner.tools import Tools, run_tool

# Both inputs are converted to this before scoring, whatever the source and encode carry.
VMAF_PIX_FMT = "yuv420p10le"
# libvmaf scales well up to about this many threads; the i7-14700K has 28.
MAX_VMAF_THREADS = 16


class VmafError(Exception):
    """VMAF produced no usable score (log missing or unreadable, NaN, infinite)."""


@dataclass(frozen=True)
class GuardResult:
    ok: bool
    mean: float
    worst: float
    reason: str


def evaluate_guard(
    window_scores: Sequence[float], mean_floor: float, worst_floor: float
) -> GuardResult:
    """Pass only if the mean score reaches *mean_floor* and no window is below *worst_floor*.

    Both floors are inclusive. Empty input or a non-finite score fails (nothing was measured,
    so nothing can be trusted). *reason* is empty on success and names each failed check
    with the numbers otherwise.
    """
    if not window_scores:
        return GuardResult(ok=False, mean=0.0, worst=0.0, reason="no window scores to evaluate")
    if not all(math.isfinite(score) for score in window_scores):
        return GuardResult(
            ok=False, mean=0.0, worst=0.0, reason="a window score is not a finite number"
        )
    mean = sum(window_scores) / len(window_scores)
    worst = min(window_scores)
    problems: list[str] = []
    if mean < mean_floor:
        problems.append(f"mean VMAF {mean:.2f} is below the floor {mean_floor:.2f}")
    if worst < worst_floor:
        problems.append(f"worst window VMAF {worst:.2f} is below the floor {worst_floor:.2f}")
    return GuardResult(ok=not problems, mean=mean, worst=worst, reason="; ".join(problems))


def vmaf_threads(cpu_count: int | None = None) -> int:
    """libvmaf ``n_threads``: every logical CPU up to `MAX_VMAF_THREADS` (1 if unknown)."""
    return max(1, min(MAX_VMAF_THREADS, cpu_count or 1))


def _window(start_s: float | None, length_s: float | None, name: str) -> list[str]:
    """``-ss`` / ``-t`` input options for one input (placed before its ``-i``)."""
    args: list[str] = []
    if start_s is not None:
        if not (math.isfinite(start_s) and start_s >= 0):
            raise ValueError(f"{name}_start_s must be a non-negative number, got {start_s!r}")
        args += ["-ss", f"{start_s:.3f}"]
    if length_s is not None:
        if not (math.isfinite(length_s) and length_s > 0):
            raise ValueError(f"{name}_length_s must be a positive number, got {length_s!r}")
        args += ["-t", f"{length_s:.3f}"]
    return args


def build_vmaf_cmd(
    tools: Tools,
    distorted: Path,
    reference: Path,
    log_name: str,
    n_threads: int,
    *,
    reference_start_s: float | None = None,
    reference_length_s: float | None = None,
    distorted_start_s: float | None = None,
    distorted_length_s: float | None = None,
) -> list[str]:
    """ffmpeg argument list scoring the first real video stream of *distorted* against *reference*.

    Each input may be cut to a window (``-ss``/``-t`` before its ``-i``). Both are reset to
    start at timestamp zero and converted to `VMAF_PIX_FMT`, and the distorted one is scaled
    to the reference's size, so a different size, pixel format or start time never skews or
    breaks the score. Frames are paired by the nearest timestamp (``ts_sync_mode=nearest``):
    Matroska stores milliseconds, so a 23.976 fps source and its encode can round a frame's
    time differently by 1 ms, and libvmaf's default (the reference frame at or before) then
    scores that frame against its predecessor; measured on a real film, a near-lossless encode
    scored 85.9 instead of 99.

    ``libvmaf`` writes a JSON log to *log_name*, which must be a bare file name: the command is
    run with the work folder as ``cwd``, because an absolute Windows path carries a drive colon
    that the filtergraph parser splits on.
    """
    if not log_name or any(ch in log_name for ch in "/\\:"):
        raise ValueError(f"the VMAF log must be a bare file name, got {log_name!r}")
    fmt = VMAF_PIX_FMT
    graph = (
        f"[0:V:0]setpts=PTS-STARTPTS,format={fmt}[dist];"
        f"[1:V:0]setpts=PTS-STARTPTS,format={fmt},split[ref_size][ref];"
        "[dist][ref_size]scale=w=rw:h=rh:flags=bicubic[dist_sized];"
        f"[dist_sized][ref]libvmaf=log_fmt=json:log_path={log_name}:"
        f"n_threads={n_threads}:shortest=1:ts_sync_mode=nearest"
    )
    return [
        str(tools.ffmpeg),
        "-nostdin",
        "-hide_banner",
        *_window(distorted_start_s, distorted_length_s, "distorted"),
        "-i",
        str(distorted),
        *_window(reference_start_s, reference_length_s, "reference"),
        "-i",
        str(reference),
        "-lavfi",
        graph,
        "-an",
        "-sn",
        "-dn",
        "-progress",
        "pipe:1",
        "-nostats",
        "-f",
        "null",
        "-",
    ]


def parse_vmaf_log(path: Path) -> float:
    """The pooled VMAF mean of a libvmaf JSON log; `VmafError` if there is no finite one."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise VmafError(f"the VMAF log {path.name} is missing") from None
    except OSError as exc:
        raise VmafError(f"the VMAF log {path.name} cannot be read: {exc}") from exc
    try:
        mean = json.loads(text)["pooled_metrics"]["vmaf"]["mean"]
    except (ValueError, KeyError, TypeError) as exc:
        raise VmafError(f"the VMAF log {path.name} has no pooled mean: {exc!r}") from None
    if isinstance(mean, bool) or not isinstance(mean, int | float) or not math.isfinite(mean):
        raise VmafError(f"the VMAF log {path.name} has an unusable mean {mean!r}")
    return float(mean)


def vmaf_mean(
    tools: Tools,
    distorted: Path,
    reference: Path,
    workdir: Path,
    reporter: Reporter,
    *,
    reference_start_s: float | None = None,
    reference_length_s: float | None = None,
    distorted_start_s: float | None = None,
    distorted_length_s: float | None = None,
) -> float:
    """Mean VMAF of *distorted* against *reference*, optionally over windows of each.

    Runs `build_vmaf_cmd` through `run_tool` (progress, heartbeat, stderr log in *workdir*)
    with ``cwd=workdir`` and a relative log name, reads the pooled mean and removes the log.
    Raises `VmafError` when the log is missing or holds no finite mean, and `ToolFailed`
    when ffmpeg fails.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    log_name = f"vmaf-{uuid.uuid4().hex[:12]}.json"
    cmd = build_vmaf_cmd(
        tools,
        distorted,
        reference,
        log_name,
        vmaf_threads(os.cpu_count()),
        reference_start_s=reference_start_s,
        reference_length_s=reference_length_s,
        distorted_start_s=distorted_start_s,
        distorted_length_s=distorted_length_s,
    )
    length = reference_length_s or distorted_length_s or 0.0

    def parser(lines: Iterable[str]) -> Iterator[Progress]:
        return parse_ffmpeg_progress(lines, length)

    log = workdir / log_name
    try:
        run_tool(
            cmd,
            label=f"VMAF {distorted.name}",
            reporter=reporter,
            parser=parser,
            stderr_log=workdir / f"{log_name}.stderr.log",
            cwd=workdir,
        )
        return parse_vmaf_log(log)
    finally:
        log.unlink(missing_ok=True)
