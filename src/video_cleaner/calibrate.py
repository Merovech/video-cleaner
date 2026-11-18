"""Calibration: encode sample windows at several CRFs and tabulate size against VMAF.

A few encode candidates spread over the size range are encoded at CRF 17, 20, 23 and 26; each
encode is scored per window with VMAF, and a side-by-side clip (original left, encode right) is
written so the user can also judge by eye. The table goes to ``calibration.txt`` beside the
state database and the clips to ``<completed>/calibration``. The user then approves a VMAF
target and a CRF ceiling with ``calibrate --approve``; `run` refuses to start until they have.

The real work (encode, VMAF, side-by-side) is injected through `CalibrationEngine`; its defaults
are the real `encode_sample`, `score_windows` and `make_sidebyside`. Calibration reads each
sample's library file directly and read-only (ffprobe and ``-ss`` input seeks, no copy
through scratch); it writes only to ``<completed>/calibration`` and the tool scratch folder.
"""

from __future__ import annotations

import logging
import math
import shutil
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from video_cleaner import quality
from video_cleaner.atomicio import write_text_atomic
from video_cleaner.cmdline import PROGRESS_OPTIONS, flatten
from video_cleaner.config import Settings
from video_cleaner.decide import Action
from video_cleaner.encode import (
    SAMPLE_WINDOW_COUNT,
    SAMPLE_WINDOW_S,
    EncodeError,
    WindowEncode,
    encode_windows,
    measure_windows,
    sample_windows,
)
from video_cleaner.instancelock import instance_lock, lock_path_for
from video_cleaner.probe import ProbeError, probe
from video_cleaner.progress import Progress, Reporter, parse_ffmpeg_progress
from video_cleaner.quality import VmafError
from video_cleaner.runner import refuse_outputs_in_library
from video_cleaner.scratch import ScratchNotOurs, prepare_tool_scratch
from video_cleaner.state import FileRecord, State
from video_cleaner.tools import ToolFailed, ToolNotFound, ToolOutputError, Tools, run_tool

_log = logging.getLogger(__name__)

CALIBRATION_CRFS: tuple[int, ...] = (17, 20, 23, 26)
CALIBRATION_TXT = "calibration.txt"
CLIPS_FOLDER = "calibration"
SIDEBYSIDE_HEIGHT = 720
SIDEBYSIDE_CRF = 18
VMAF_TARGET_RANGE = (50.0, 100.0)
# What the real engine raises for one sample; `run_calibration` turns them into CalibrationError.
_ENGINE_ERRORS = (
    EncodeError,
    VmafError,
    ProbeError,
    ToolFailed,
    ToolOutputError,
    ToolNotFound,
    OSError,
)


class CalibrationError(Exception):
    """The calibration cannot run or produced unusable numbers."""


@dataclass(frozen=True)
class SampleEncode:
    """One sample encoded at one CRF: one record per sample window."""

    crf: int
    windows: tuple[WindowEncode, ...]

    @property
    def source_bytes(self) -> int:
        """Source video bytes of all windows together."""
        return sum(w.source_bytes for w in self.windows)

    @property
    def encoded_bytes(self) -> int:
        """Encoded bytes of all windows together."""
        return sum(w.encoded_bytes for w in self.windows)


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


def encode_sample(
    tools: Tools,
    settings: Settings,
    sample: FileRecord,
    crf: int,
    workdir: Path,
    reporter: Reporter,
    *,
    window_count: int = SAMPLE_WINDOW_COUNT,
    window_length_s: float = SAMPLE_WINDOW_S,
) -> SampleEncode:
    """Encode the sample windows of the library file *sample* at *crf* into *workdir*.

    The library file is probed and read in place, read-only: ffprobe for the duration
    and the windows' source bytes, ffmpeg ``-ss`` input seeks for the encodes, which use the
    run's preset and pixel format (`encode.encode_windows`). The windows are those of the CRF
    search (`encode.sample_windows`); *window_count* and *window_length_s* exist for tests.
    """
    info = probe(sample.path, tools, reporter)
    spans = sample_windows(info.duration_s, window_count, window_length_s)
    if not spans:
        raise CalibrationError(
            f"{sample.path.name}: no sample windows (duration {info.duration_s!r} s)"
        )
    windows = measure_windows(tools, sample.path, spans, reporter)
    encodes = encode_windows(tools, settings, sample.path, crf, windows, workdir, reporter)
    return SampleEncode(crf, encodes)


def score_windows(
    tools: Tools,
    settings: Settings,
    sample: FileRecord,
    encoded: SampleEncode,
    workdir: Path,
    reporter: Reporter,
) -> list[float]:
    """VMAF of each encoded window against the same window of the library file, in order."""
    return [
        quality.vmaf_mean(
            tools,
            window.encoded,
            sample.path,
            workdir,
            reporter,
            reference_start_s=window.start_s,
            reference_length_s=window.length_s,
        )
        for window in encoded.windows
    ]


def make_sidebyside(
    tools: Tools, sample: FileRecord, encoded: SampleEncode, dst: Path, reporter: Reporter
) -> None:
    """Write the side-by-side clip of the middle window of *encoded* to *dst*.

    The original window is cut from the library file; the encoded window already is that
    window. ffmpeg's stderr log stays in the scratch work folder, beside the encode, so
    nothing but the clip lands in the completed folder.
    """
    window = encoded.windows[len(encoded.windows) // 2]
    cmd = build_sidebyside_cmd(
        tools, sample.path, window.encoded, dst, window.start_s, window.length_s
    )

    def parser(lines: Iterable[str]) -> Iterator[Progress]:
        return parse_ffmpeg_progress(lines, window.length_s)

    run_tool(
        cmd,
        label=f"side-by-side {dst.name}",
        reporter=reporter,
        parser=parser,
        stderr_log=window.encoded.with_name(f"{dst.stem}.stderr.log"),
    )


@dataclass
class CalibrationEngine:
    """The real work of a calibration; tests inject fakes."""

    encode_sample: EncodeSample = encode_sample
    score_windows: ScoreWindows = score_windows
    make_sidebyside: MakeSideBySide = make_sidebyside


def default_engine() -> CalibrationEngine:
    """The engine the `calibrate` command uses: the real encode, VMAF and side-by-side."""
    return CalibrationEngine()


def select_samples(records: Sequence[FileRecord], n: int) -> list[FileRecord]:
    """At most *n* encode candidates of *records* spread evenly over the size range.

    *records* are live records (`State.records()` leaves STALE ones out). Deterministic:
    candidates are sorted by (size, path) and the picks are evenly spaced from the smallest to
    the largest; one pick is the median.
    """
    if n < 1:
        raise ValueError(f"the number of samples must be at least 1, got {n}")
    candidates = sorted(
        (r for r in records if r.decision == Action.ENCODE.value),
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

    *original* is cut to the window with ``-ss``/``-t`` before its ``-i``; *encoded* is the
    encoded sample of that window, so it starts at zero and is only limited to the same length.
    Both are scaled to the same *height* and stacked. No audio. The clip is 8-bit H.264
    (CRF 18) so any player can show it.
    """
    if not (math.isfinite(start_s) and start_s >= 0):
        raise ValueError(f"start_s must be a non-negative number, got {start_s!r}")
    if not (math.isfinite(length_s) and length_s > 0):
        raise ValueError(f"length_s must be a positive number, got {length_s!r}")
    if height < 2 or height % 2:
        raise ValueError(f"height must be an even number of at least 2, got {height}")
    length = ("-t", _seconds(length_s))
    # Scale both to the same height, then put the original on the left and the encode on the
    # right.
    graph = (
        f"[0:V:0]scale=-2:{height}[left];[1:V:0]scale=-2:{height}[right];"
        "[left][right]hstack=inputs=2[v]"
    )
    return [
        str(tools.ffmpeg),
        *flatten(
            ("-nostdin",),  # never wait for keyboard input
            ("-ss", _seconds(start_s)),  # seek the original to the window start (fast)
            length,  # and read only the window length of it
            ("-i", str(original)),  # input 0: the original, left
            length,  # the encoded sample already is the window; limit it to the same length
            ("-i", str(encoded)),  # input 1: the encode, right
            ("-filter_complex", graph),  # the scaling and stacking above
            ("-map", "[v]"),  # write the stacked picture
            ("-an",),  # no audio
            ("-c:v", "libx264"),  # 8-bit H.264 so any player can show the clip
            ("-crf", str(SIDEBYSIDE_CRF)),  # high quality: this clip is for judging by eye
            ("-pix_fmt", "yuv420p"),  # the pixel format every player supports
            *PROGRESS_OPTIONS,
            ("-y",),  # overwrite an earlier clip
            (str(dst),),  # the output file
        ),
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
    if len(scores) != len(encoded.windows):
        raise CalibrationError(
            f"{len(scores)} VMAF score(s) for {len(encoded.windows)} windows of {sample.path} "
            f"at CRF {crf}: one per window is required"
        )
    if encoded.source_bytes <= 0:
        raise CalibrationError(f"the sample of {sample.path} at CRF {crf} has no source bytes")
    return CalibrationRow(
        sample=sample.path,
        crf=crf,
        size_pct=100.0 * encoded.encoded_bytes / encoded.source_bytes,
        vmaf_mean=sum(scores) / len(scores),
        vmaf_worst=min(scores),
    )


def _make_folder(folder: Path) -> None:
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CalibrationError(f"cannot create {folder}: {exc}") from exc


def run_calibration(
    settings: Settings,
    tools: Tools,
    state: State,
    reporter: Reporter,
    *,
    samples: int,
    engine: CalibrationEngine,
    stamp: Callable[[], str] = lambda: f"{datetime.now():%Y%m%d-%H%M%S}",
) -> list[CalibrationRow]:
    """Sweep `CALIBRATION_CRFS` over `select_samples` and write the table and the clips.

    The clips of one sweep go to their own ``<completed>/calibration/<stamp>`` folder, so a
    sweep never mixes with (or deletes) an earlier one. A folder that cannot be created is a
    `CalibrationError`.

    Raises `RunRefused` if an output folder is under the library root, `AlreadyRunning` while a
    run or another calibration holds the single-instance lock, and `CalibrationError` when there
    is nothing to sample, the scratch folder cannot be used, a score is unusable or the engine
    fails (an encode, probe, VMAF or tool error, chained as the cause). A finished sweep
    is recorded in *state* (`State.record_calibration_sweep`, no event): `calibrate --approve`
    requires one. Work files go to
    ``<scratch>/.video-cleaner/calibration`` (the tool-owned scratch folder) and are removed
    after each sample; nothing is stored in *state*.
    """
    refuse_outputs_in_library(settings)
    with instance_lock(lock_path_for(settings.state_db)):
        return _calibrate_locked(settings, tools, state, reporter, samples, engine, stamp())


def _calibrate_locked(
    settings: Settings,
    tools: Tools,
    state: State,
    reporter: Reporter,
    samples: int,
    engine: CalibrationEngine,
    run_name: str,
) -> list[CalibrationRow]:
    picked = select_samples(state.records(), samples)
    if not picked:
        raise CalibrationError("no encode candidates to sample: run `survey` first")
    try:
        work_root = prepare_tool_scratch(settings.scratch_dir) / CLIPS_FOLDER
    except ScratchNotOurs as exc:
        raise CalibrationError(str(exc)) from None
    except OSError as exc:
        raise CalibrationError(
            f"cannot use the scratch folder {settings.scratch_dir}: {exc}"
        ) from exc
    clips = settings.completed_dir / CLIPS_FOLDER / run_name
    _make_folder(clips)
    rows: list[CalibrationRow] = []
    for index, sample in enumerate(picked, start=1):
        workdir = work_root / f"{index:02d}"
        try:
            for crf in CALIBRATION_CRFS:
                reporter.stage(f"calibrate {index}/{len(picked)} CRF {crf}: {sample.path.name}")
                _make_folder(workdir)
                clip = clips / f"{index:02d}-{sample.path.stem}-crf{crf}.mp4"
                try:
                    encoded = engine.encode_sample(tools, settings, sample, crf, workdir, reporter)
                    scores = engine.score_windows(
                        tools, settings, sample, encoded, workdir, reporter
                    )
                    rows.append(_row(sample, crf, encoded, scores))
                    engine.make_sidebyside(tools, sample, encoded, clip, reporter)
                except _ENGINE_ERRORS as exc:
                    raise CalibrationError(f"{sample.path.name} at CRF {crf}: {exc}") from exc
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
    table = settings.state_db.parent / CALIBRATION_TXT
    try:
        write_text_atomic(table, format_calibration_table(rows))
    except OSError as exc:
        raise CalibrationError(f"cannot write {table}: {exc}") from exc
    state.record_calibration_sweep()
    _log.info("calibration of %d sample(s) written", len(picked))
    return rows
