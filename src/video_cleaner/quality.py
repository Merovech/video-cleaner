"""Post-encode quality guard: pure arithmetic over per-window VMAF scores, plus a held scorer."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from video_cleaner.progress import Reporter
from video_cleaner.tools import Tools


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


def vmaf_mean(
    tools: Tools, distorted: Path, reference: Path, workdir: Path, reporter: Reporter
) -> float:
    """Mean VMAF of *distorted* against *reference*. HELD until Task 10b.

    Intended behaviour: run ffmpeg with the ``libvmaf`` filter (distorted first, reference
    second), writing its JSON log to a RELATIVE ``log_path`` and running with ``cwd=workdir``:
    on Windows an absolute ``log_path`` contains a drive-letter colon that the filtergraph
    parser mis-handles, so it must be escaped away by never passing one. The mean is read
    from the pooled metrics of that log.
    """
    raise NotImplementedError("held: Task 10b")
