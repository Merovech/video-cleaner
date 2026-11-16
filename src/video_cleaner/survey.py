"""Survey: walk the library read-only, probe every video file, decide, record and report.

Nothing is ever written under the library root: files are only opened by ffprobe, and the CSV
goes next to the state database (refused if that lies under the root). A file that cannot be
probed, or whose path cannot be stored in the state database, is flagged in the CSV and the
result and the survey carries on.
"""

from __future__ import annotations

import csv
import logging
import os
import tempfile
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from video_cleaner.config import Settings
from video_cleaner.decide import Action, decide
from video_cleaner.multipart import PartGroup, find_part_groups, parts_compatible
from video_cleaner.probe import MediaInfo, ProbeError, probe
from video_cleaner.progress import Progress, Reporter
from video_cleaner.state import State, UnrepresentablePath
from video_cleaner.tools import Tools

_log = logging.getLogger("video_cleaner.survey")

VIDEO_EXTENSIONS = frozenset({".mkv", ".mp4", ".avi", ".m4v", ".ts", ".wmv", ".mov"})
PARTIAL_SUFFIX = ".part"
CSV_NAME = "survey.csv"
CSV_COLUMNS = (
    "path",
    "size",
    "duration_s",
    "container",
    "codec",
    "width",
    "height",
    "bitrate_kbps",
    "action",
    "reasons",
    "est_saving_bytes",
    "parts",
)
LIST_SEPARATOR = "; "
# Placeholder until calibration: an encode is assumed to save half of the source size.
EST_SAVING_FRACTION = 0.5

INCOMPATIBLE_PARTS_REASON = "multi-part parts have different stream layouts"

ProbeFn = Callable[[Path, Tools, Reporter], MediaInfo]


class SurveyError(Exception):
    """The survey cannot run safely (for example its output would land in the library)."""


@dataclass(frozen=True)
class SurveyResult:
    counts: dict[Action, int]
    total_bytes: int
    est_saving_bytes: int
    csv_path: Path
    unprobeable: tuple[tuple[Path, str], ...] = ()
    unrepresentable: tuple[Path, ...] = ()
    groups: tuple[PartGroup, ...] = ()


@dataclass(frozen=True)
class _Row:
    """One CSV line: a file, what was decided for it and why."""

    path: Path
    size: int | None
    info: MediaInfo | None
    action: Action
    reasons: tuple[str, ...]
    est_saving_bytes: int = 0
    parts: tuple[Path, ...] = ()
    unrepresentable: bool = False


def discover_files(root: Path) -> list[Path]:
    """Video files under *root* (recursive, extension case-insensitive), sorted.

    Names ending in ``.part`` (unfinished downloads) are skipped. Unreadable directories are
    logged and skipped; a missing root gives an empty list.
    """

    def on_error(exc: OSError) -> None:
        _log.warning("cannot list %s: %s", exc.filename, exc.strerror or exc)

    found: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root, onerror=on_error):
        for name in filenames:
            lowered = name.lower()
            if lowered.endswith(PARTIAL_SUFFIX):
                continue
            if os.path.splitext(lowered)[1] in VIDEO_EXTENSIONS:
                found.append(Path(dirpath) / name)
    return sorted(found)


def _is_under(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def check_outputs_outside(settings: Settings, root: Path) -> Path:
    """Raise `SurveyError` if anything the survey writes would land under *root*.

    Checks the state database, the survey CSV and the log directory, and also
    ``settings.library_root`` when *root* is a different folder. Returns the CSV path.
    """
    csv_path = settings.state_db.parent / CSV_NAME
    targets = (
        ("state database", settings.state_db),
        ("survey CSV", csv_path),
        ("log directory", settings.log_dir),
    )
    for base in dict.fromkeys((root, settings.library_root)):
        for label, target in targets:
            if _is_under(target, base):
                raise SurveyError(
                    f"{label} {target} is under the library root {base}; "
                    "the survey never writes to the library"
                )
    return csv_path


def _probe_all(
    paths: Sequence[Path],
    tools: Tools,
    reporter: Reporter,
    workers: int,
    probe_fn: ProbeFn,
) -> tuple[dict[Path, MediaInfo], dict[Path, str]]:
    infos: dict[Path, MediaInfo] = {}
    failures: dict[Path, str] = {}
    total = len(paths)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="survey") as pool:
        futures: dict[Future[MediaInfo], Path] = {
            pool.submit(probe_fn, path, tools, reporter): path for path in paths
        }
        for done, future in enumerate(as_completed(futures), start=1):
            path = futures[future]
            try:
                infos[path] = future.result()
            except ProbeError as exc:
                failures[path] = exc.reason
            except Exception as exc:  # one bad file must never stop the survey
                failures[path] = f"{type(exc).__name__}: {exc}"
            if path in failures:
                _log.warning("unprobeable %s: %s", path, failures[path])
            reporter.update(
                Progress(
                    percent=done / total * 100.0,
                    fps=None,
                    speed=None,
                    eta_s=None,
                    note=f"{done} of {total} files",
                )
            )
    return infos, failures


def _encode_saving(action: Action, size: int) -> int:
    return int(size * EST_SAVING_FRACTION) if action is Action.ENCODE else 0


def _plan_group(group: PartGroup, infos: dict[Path, MediaInfo], settings: Settings) -> list[_Row]:
    probed = [infos[p] for p in group.parts if p in infos]
    if len(probed) < len(group.parts):
        reason = f"multi-part group {group.output_name} has unprobeable parts"
        return [_Row(i.path, i.size, i, Action.FLAG, (reason,)) for i in probed]
    first, *others = probed
    if not all(parts_compatible(first, other) for other in others):
        return [_Row(i.path, i.size, i, Action.FLAG, (INCOMPATIBLE_PARTS_REASON,)) for i in probed]
    decision = decide(first, settings, multipart=True)
    group_size = sum(i.size for i in probed)
    rows = [
        _Row(
            first.path,
            first.size,
            first,
            decision.action,
            decision.reasons,
            _encode_saving(decision.action, group_size),
            group.parts,
        )
    ]
    member = (f"part of group {group.output_name}",)
    rows.extend(_Row(i.path, i.size, i, Action.LEAVE_ALONE, member) for i in others)
    return rows


def _record(row: _Row, state: State) -> _Row:
    """Store *row* in *state*; an unrepresentable path becomes a flag row instead."""
    if row.info is None:
        return row
    try:
        state.upsert_file(
            row.info,
            decision=row.action.value,
            reasons=row.reasons,
            est_saving_bytes=row.est_saving_bytes,
            parts=row.parts,
        )
    except UnrepresentablePath as exc:
        _log.warning("%s", exc)
        return _Row(row.path, row.size, row.info, Action.FLAG, (str(exc),), unrepresentable=True)
    return row


def _size_or_none(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except (OSError, ValueError):
        return None


def _cell(value: object) -> str:
    return "" if value is None else str(value)


def _csv_line(row: _Row) -> list[str]:
    info = row.info
    video = info.video if info is not None else None
    return [
        str(row.path),
        _cell(row.size),
        f"{info.duration_s:.3f}" if info is not None else "",
        info.container if info is not None else "",
        video.codec if video is not None else "",
        _cell(video.width if video is not None else None),
        _cell(video.height if video is not None else None),
        f"{info.bitrate_kbps:.0f}" if info is not None else "",
        row.action.value,
        LIST_SEPARATOR.join(row.reasons),
        str(row.est_saving_bytes),
        LIST_SEPARATOR.join(str(p) for p in row.parts),
    ]


def _write_csv(path: Path, rows: Sequence[_Row]) -> None:
    """Write the CSV to a temporary file beside *path*, then replace *path* atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        # backslashreplace: an undecodable file name (lone surrogate) still yields valid UTF-8.
        with os.fdopen(fd, "w", encoding="utf-8", errors="backslashreplace", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(CSV_COLUMNS)
            writer.writerows(_csv_line(r) for r in rows)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def run_survey(
    settings: Settings,
    tools: Tools,
    state: State,
    reporter: Reporter,
    *,
    root: Path | None = None,
    workers: int = 4,
    probe_fn: ProbeFn = probe,
) -> SurveyResult:
    """Probe every video file under *root* (default ``settings.library_root``), read-only.

    Each file is decided and upserted into *state*; a compatible multi-part group becomes one
    record on its first part. Writes ``survey.csv`` beside the state database. Raises
    `SurveyError` (before touching anything) if an output would lie under the library.
    """
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    root = settings.library_root if root is None else root
    csv_path = check_outputs_outside(settings, root)

    reporter.stage("survey")
    paths = discover_files(root)
    _log.info("survey of %s: %d video files", root, len(paths))
    infos, failures = _probe_all(paths, tools, reporter, workers, probe_fn)

    groups = tuple(find_part_groups(paths))
    rows: list[_Row] = []
    grouped: set[Path] = set()
    for group in groups:
        grouped.update(group.parts)
        rows.extend(_record(r, state) for r in _plan_group(group, infos, settings))
    for path in paths:
        if path in failures:
            reason = f"unprobeable: {failures[path]}"
            rows.append(_Row(path, _size_or_none(path), None, Action.FLAG, (reason,)))
        elif path not in grouped:
            info = infos[path]
            d = decide(info, settings)
            saving = _encode_saving(d.action, info.size)
            rows.append(_record(_Row(path, info.size, info, d.action, d.reasons, saving), state))
    rows.sort(key=lambda r: r.path)

    _write_csv(csv_path, rows)
    counts = dict.fromkeys(Action, 0)
    for row in rows:
        counts[row.action] += 1
    return SurveyResult(
        counts=counts,
        total_bytes=sum(r.size or 0 for r in rows),
        est_saving_bytes=sum(r.est_saving_bytes for r in rows),
        csv_path=csv_path,
        unprobeable=tuple((p, failures[p]) for p in paths if p in failures),
        unrepresentable=tuple(r.path for r in rows if r.unrepresentable),
        groups=groups,
    )
