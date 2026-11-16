"""Calibration: encode sample windows at several CRFs and tabulate size against VMAF.

A few encode candidates spread over the size range are encoded at CRF 17, 20, 23 and 26; each
encode is scored per window with VMAF, and a side-by-side clip (original left, encode right) is
written so Ari can also judge by eye. The table goes to ``calibration.txt`` beside the state
database and the clips to ``<completed>/calibration``. Ari then approves a VMAF target and a CRF
ceiling with ``calibrate --approve``; `run` refuses to start until he has.

The real work (encode, VMAF, side-by-side) is injected through `CalibrationEngine`. Its defaults
are held until Task 10b and raise ``NotImplementedError("held: Task 10b")``; `run_calibration`
refuses a held engine before touching anything.
"""

from __future__ import annotations

import logging
import math
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from video_cleaner.atomicio import write_text_atomic
from video_cleaner.config import Settings
from video_cleaner.decide import Action
from video_cleaner.instancelock import instance_lock, lock_path_for
from video_cleaner.progress import Reporter
from video_cleaner.runner import HELD_REASON, refuse_outputs_in_library
from video_cleaner.scratch import ScratchNotOurs, prepare_tool_scratch
from video_cleaner.state import FileRecord, State, Status
from video_cleaner.tools import Tools

_log = logging.getLogger(__name__)

CALIBRATION_CRFS: tuple[int, ...] = (17, 20, 23, 26)
CALIBRATION_TXT = "calibration.txt"
CLIPS_FOLDER = "calibration"
SIDEBYSIDE_HEIGHT = 720
SIDEBYSIDE_CRF = 18
VMAF_TARGET_RANGE = (50.0, 100.0)


class CalibrationError(Exception):
    """The calibration cannot run or produced unusable numbers."""


@dataclass(frozen=True)
class SampleEncode:
    """One sample encoded at one CRF: the encode, the bytes in and out, the window used."""

    encoded: Path
    source_bytes: int
    encoded_bytes: int
    start_s: float
    length_s: float


@dataclass(frozen=True)
class CalibrationRow:
    sample: Path
    crf: int
    size_pct: float
    vmaf_mean: float
    vmaf_worst: float


EncodeSample = Callable[[Tools, Settings, FileRecord, int, Path, Reporter], SampleEncode]
ScoreWindows = Callable[
    [Tools, Settings, FileRecord, SampleEncode, Path, Reporter], Sequence[float]
]
MakeSideBySide = Callable[[Tools, FileRecord, SampleEncode, Path, Reporter], None]


def held_encode_sample(
    tools: Tools,
    settings: Settings,
    sample: FileRecord,
    crf: int,
    workdir: Path,
    reporter: Reporter,
) -> SampleEncode:
    """Encode the sample windows of *sample* at *crf* into *workdir*. HELD until Task 10b."""
    raise NotImplementedError(HELD_REASON)


def held_score_windows(
    tools: Tools,
    settings: Settings,
    sample: FileRecord,
    encoded: SampleEncode,
    workdir: Path,
    reporter: Reporter,
) -> Sequence[float]:
    """VMAF of each encoded window against its source cut. HELD until Task 10b."""
    raise NotImplementedError(HELD_REASON)


def held_make_sidebyside(
    tools: Tools, sample: FileRecord, encoded: SampleEncode, dst: Path, reporter: Reporter
) -> None:
    """Run `build_sidebyside_cmd` for *encoded* into *dst*. HELD until Task 10b."""
    raise NotImplementedError(HELD_REASON)


_HELD: tuple[object, ...] = (held_encode_sample, held_score_windows, held_make_sidebyside)


@dataclass
class CalibrationEngine:
    """The real work of a calibration; tests inject fakes."""

    encode_sample: EncodeSample = held_encode_sample
    score_windows: ScoreWindows = held_score_windows
    make_sidebyside: MakeSideBySide = held_make_sidebyside

    def held_stages(self) -> list[str]:
        """Names of the stages that are still held stubs."""
        stages = {
            "encode_sample": self.encode_sample,
            "score_windows": self.score_windows,
            "make_sidebyside": self.make_sidebyside,
        }
        return [
            name
            for name, stage in stages.items()
            if any(stage is held for held in _HELD) or getattr(stage, "held", False)
        ]


def default_engine() -> CalibrationEngine:
    """The engine the `calibrate` command uses (held until Task 10b)."""
    return CalibrationEngine()


def select_samples(records: Sequence[FileRecord], n: int) -> list[FileRecord]:
    """At most *n* live encode candidates spread evenly over the size range.

    Deterministic: candidates are sorted by (size, path) and the picks are evenly spaced from
    the smallest to the largest; one pick is the median.
    """
    if n < 1:
        raise ValueError(f"the number of samples must be at least 1, got {n}")
    candidates = sorted(
        (r for r in records if r.decision == Action.ENCODE.value and r.status is not Status.STALE),
        key=lambda r: (r.size, r.path.as_posix()),
    )
    if len(candidates) <= n:
        return candidates
    if n == 1:
        return [candidates[(len(candidates) - 1) // 2]]
    last = len(candidates) - 1
    indices = sorted({round(i * last / (n - 1)) for i in range(n)})
    return [candidates[i] for i in indices]


def _seconds(value: float) -> str:
    return f"{value:.3f}"


def build_sidebyside_cmd(
    tools: Tools,
    original: Path,
    encoded: Path,
    dst: Path,
    start_s: float,
    length_s: float,
    *,
    height: int = SIDEBYSIDE_HEIGHT,
) -> list[str]:
    """ffmpeg argument list for a side-by-side clip: *original* left, *encoded* right.

    Both inputs are cut with ``-ss``/``-t`` before their ``-i``, scaled to the same *height*
    and stacked. No audio. The clip is 8-bit H.264 (CRF 18) so any player can show it.
    """
    if not (math.isfinite(start_s) and start_s >= 0):
        raise ValueError(f"start_s must be a non-negative number, got {start_s!r}")
    if not (math.isfinite(length_s) and length_s > 0):
        raise ValueError(f"length_s must be a positive number, got {length_s!r}")
    if height < 2 or height % 2:
        raise ValueError(f"height must be an even number of at least 2, got {height}")
    window = ["-ss", _seconds(start_s), "-t", _seconds(length_s)]
    graph = (
        f"[0:V:0]scale=-2:{height}[left];[1:V:0]scale=-2:{height}[right];"
        "[left][right]hstack=inputs=2[v]"
    )
    return [
        str(tools.ffmpeg),
        *window,
        "-i",
        str(original),
        *window,
        "-i",
        str(encoded),
        "-filter_complex",
        graph,
        "-map",
        "[v]",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        str(SIDEBYSIDE_CRF),
        "-pix_fmt",
        "yuv420p",
        "-y",
        str(dst),
    ]


def format_calibration_table(rows: Sequence[CalibrationRow]) -> str:
    """The per-sample table, then the average per CRF over all samples."""
    if not rows:
        return "Calibration: no samples\n"
    header = f"{'CRF':>3}  {'size %':>7}  {'VMAF mean':>9}  {'VMAF worst':>10}"
    lines = ["Calibration: size of the encode (% of the source) and VMAF per CRF", ""]
    samples: dict[Path, list[CalibrationRow]] = {}
    for row in rows:
        samples.setdefault(row.sample, []).append(row)
    for sample, sample_rows in samples.items():
        lines += [str(sample), header]
        lines += [
            f"{r.crf:>3}  {r.size_pct:>7.1f}  {r.vmaf_mean:>9.2f}  {r.vmaf_worst:>10.2f}"
            for r in sample_rows
        ]
        lines.append("")
    lines += [f"All samples (mean of {len(samples)}; worst = lowest window)", header]
    for crf in sorted({r.crf for r in rows}):
        at = [r for r in rows if r.crf == crf]
        size = sum(r.size_pct for r in at) / len(at)
        mean = sum(r.vmaf_mean for r in at) / len(at)
        worst = min(r.vmaf_worst for r in at)
        lines.append(f"{crf:>3}  {size:>7.1f}  {mean:>9.2f}  {worst:>10.2f}")
    return "\n".join(lines) + "\n"


def _row(
    sample: FileRecord, crf: int, encoded: SampleEncode, scores: Sequence[float]
) -> CalibrationRow:
    if not scores or not all(math.isfinite(s) for s in scores):
        raise CalibrationError(f"no usable VMAF scores for {sample.path} at CRF {crf}")
    if encoded.source_bytes <= 0:
        raise CalibrationError(f"the sample of {sample.path} at CRF {crf} has no source bytes")
    return CalibrationRow(
        sample=sample.path,
        crf=crf,
        size_pct=100.0 * encoded.encoded_bytes / encoded.source_bytes,
        vmaf_mean=sum(scores) / len(scores),
        vmaf_worst=min(scores),
    )


def run_calibration(
    settings: Settings,
    tools: Tools,
    state: State,
    reporter: Reporter,
    *,
    samples: int,
    engine: CalibrationEngine,
) -> list[CalibrationRow]:
    """Sweep `CALIBRATION_CRFS` over `select_samples` and write the table and the clips.

    Raises `NotImplementedError` (held) before touching anything while a stage of *engine*
    is still held, `RunRefused` if an output folder is under the library root, `AlreadyRunning`
    while a run or another calibration holds the single-instance lock, and
    `CalibrationError` when there is nothing to sample or a score is unusable. Work files go to
    ``<scratch>/.video-cleaner/calibration`` (the tool-owned scratch folder) and are removed
    after each sample; nothing is stored in *state*.
    """
    held = engine.held_stages()
    if held:
        raise NotImplementedError(HELD_REASON)
    refuse_outputs_in_library(settings)
    with instance_lock(lock_path_for(settings.state_db)):
        return _calibrate_locked(settings, tools, state, reporter, samples, engine)


def _calibrate_locked(
    settings: Settings,
    tools: Tools,
    state: State,
    reporter: Reporter,
    samples: int,
    engine: CalibrationEngine,
) -> list[CalibrationRow]:
    picked = select_samples(state.records(), samples)
    if not picked:
        raise CalibrationError("no encode candidates to sample: run `survey` first")
    try:
        work_root = prepare_tool_scratch(settings.scratch_dir) / CLIPS_FOLDER
    except ScratchNotOurs as exc:
        raise CalibrationError(str(exc)) from None
    clips = settings.completed_dir / CLIPS_FOLDER
    clips.mkdir(parents=True, exist_ok=True)
    rows: list[CalibrationRow] = []
    for index, sample in enumerate(picked, start=1):
        workdir = work_root / f"{index:02d}"
        try:
            for crf in CALIBRATION_CRFS:
                reporter.stage(f"calibrate {index}/{len(picked)} CRF {crf}: {sample.path.name}")
                workdir.mkdir(parents=True, exist_ok=True)
                encoded = engine.encode_sample(tools, settings, sample, crf, workdir, reporter)
                scores = engine.score_windows(tools, settings, sample, encoded, workdir, reporter)
                rows.append(_row(sample, crf, encoded, scores))
                clip = clips / f"{index:02d}-{sample.path.stem}-crf{crf}.mp4"
                engine.make_sidebyside(tools, sample, encoded, clip, reporter)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
    table = settings.state_db.parent / CALIBRATION_TXT
    try:
        write_text_atomic(table, format_calibration_table(rows))
    except OSError as exc:
        raise CalibrationError(f"cannot write {table}: {exc}") from exc
    _log.info("calibration of %d sample(s) written", len(picked))
    return rows
