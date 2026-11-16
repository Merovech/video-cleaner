"""Survey: walk the library read-only, probe every video file, decide, record and report.

Nothing is ever written under the library root: files are only opened by ffprobe, and the CSV
goes next to the state database (refused if that lies under the root). A file that cannot be
probed, or whose path cannot be stored in the state database, is flagged in the CSV and the
result and the survey carries on.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from video_cleaner.atomicio import write_csv_atomic
from video_cleaner.config import Settings
from video_cleaner.decide import Action, decide
from video_cleaner.moviehash import movie_hash
from video_cleaner.multipart import PartGroup, find_part_groups, parts_compatible
from video_cleaner.paths import OutputInLibrary, outputs_outside_library
from video_cleaner.probe import MediaInfo, ProbeError, probe
from video_cleaner.progress import Progress, Reporter
from video_cleaner.state import State, UnrepresentablePath
from video_cleaner.tools import ToolNotFound, Tools

_log = logging.getLogger("video_cleaner.survey")

VIDEO_EXTENSIONS = frozenset({".mkv", ".mp4", ".avi", ".m4v", ".ts", ".wmv", ".mov"})
PARTIAL_SUFFIX = ".part"
CSV_NAME = "survey.csv"
# Used when survey.csv is locked (open in Excel, say): survey-YYYYmmdd-HHMMSS.csv beside it.
FALLBACK_CSV_PATTERN = "survey-{:%Y%m%d-%H%M%S}.csv"
CSV_COLUMNS = (
    "path",
    "size",
    "duration_s",
    "container",
    "codec",
    "width",
    "height",
    "bitrate_kbps",
    "hdr",
    "dv_profile",
    "interlaced",
    "subtitle_streams",
    "action",
    "reasons",
    "est_saving_bytes",
    "parts",
    "movie_hash",
)
LIST_SEPARATOR = "; "
# Placeholder until calibration: an encode is assumed to save half of the source size.
EST_SAVING_FRACTION = 0.5

INCOMPATIBLE_PARTS_REASON = "multi-part parts have different stream layouts"

ProbeFn = Callable[[Path, Tools, Reporter], MediaInfo]
HashFn = Callable[[Path], str | None]


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
    movie_hash: str | None = None


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


def check_outputs_outside(settings: Settings, root: Path) -> Path:
    """Raise `SurveyError` if any output (the survey CSV included) overlaps *root* or
    ``settings.library_root``, in either direction (`paths.outputs_outside_library`).

    Returns the CSV path.
    """
    csv_path = settings.state_db.parent / CSV_NAME
    try:
        outputs_outside_library(settings, extra=(("survey CSV", csv_path),), roots=(root,))
    except OutputInLibrary as exc:
        raise SurveyError(str(exc)) from None
    return csv_path


def _safe_hash(path: Path, hash_fn: HashFn) -> str | None:
    """The movie hash of *path*; any failure is a warning and None, never a survey stop."""
    try:
        return hash_fn(path)
    except Exception as exc:  # a hash is a bonus: the file is still surveyed without it
        _log.warning("no movie hash for %s: %s: %s", path, type(exc).__name__, exc)
        return None


def _probe_all(
    paths: Sequence[Path],
    tools: Tools,
    reporter: Reporter,
    workers: int,
    probe_fn: ProbeFn,
    hash_fn: HashFn,
) -> tuple[dict[Path, MediaInfo], dict[Path, str], dict[Path, str | None]]:
    infos: dict[Path, MediaInfo] = {}
    failures: dict[Path, str] = {}
    hashes: dict[Path, str | None] = {}
    total = len(paths)

    def work(path: Path) -> tuple[MediaInfo, str | None]:
        info = probe_fn(path, tools, reporter)
        return info, _safe_hash(path, hash_fn)

    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="survey")
    try:
        futures: dict[Future[tuple[MediaInfo, str | None]], Path] = {
            pool.submit(work, path): path for path in paths
        }
        for done, future in enumerate(as_completed(futures), start=1):
            path = futures[future]
            try:
                infos[path], hashes[path] = future.result()
            except ProbeError as exc:
                failures[path] = exc.reason
            except ToolNotFound:
                raise  # a missing ffprobe is a setup problem, not a bad file
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
    except BaseException:
        # Ctrl+C or an abort: start no more probes; the ones already running finish alone.
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return infos, failures, hashes


def _encode_saving(action: Action, size: int) -> int:
    return int(size * EST_SAVING_FRACTION) if action is Action.ENCODE else 0


def _plan_group(
    group: PartGroup,
    infos: dict[Path, MediaInfo],
    hashes: dict[Path, str | None],
    settings: Settings,
) -> list[_Row]:
    probed = [infos[p] for p in group.parts if p in infos]

    def flag_all(reason: str) -> list[_Row]:
        return [
            _Row(i.path, i.size, i, Action.FLAG, (reason,), movie_hash=hashes.get(i.path))
            for i in probed
        ]

    if len(probed) < len(group.parts):
        return flag_all(f"multi-part group {group.output_name} has unprobeable parts")
    first, *others = probed
    if not all(parts_compatible(first, other) for other in others):
        return flag_all(INCOMPATIBLE_PARTS_REASON)
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
            movie_hash=hashes.get(first.path),
        )
    ]
    member = (f"part of group {group.output_name}",)
    rows.extend(
        _Row(i.path, i.size, i, Action.LEAVE_ALONE, member, movie_hash=hashes.get(i.path))
        for i in others
    )
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
            movie_hash=row.movie_hash,
        )
    except UnrepresentablePath as exc:
        _log.warning("%s", exc)
        return _Row(
            row.path,
            row.size,
            row.info,
            Action.FLAG,
            (str(exc),),
            unrepresentable=True,
            movie_hash=row.movie_hash,
        )
    return row


def _size_or_none(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except (OSError, ValueError):
        return None


def _cell(value: object) -> str:
    return "" if value is None else str(value)


def _yes_no(flag: bool) -> str:
    return "yes" if flag else "no"


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
        _yes_no(info.is_hdr) if info is not None else "",
        _cell(video.dv_profile if video is not None else None),
        _yes_no(info.is_interlaced) if info is not None else "",
        str(len(info.subtitles)) if info is not None else "",
        row.action.value,
        LIST_SEPARATOR.join(row.reasons),
        str(row.est_saving_bytes),
        LIST_SEPARATOR.join(str(p) for p in row.parts),
        _cell(row.movie_hash),
    ]


def _csv_error(path: Path, exc: OSError) -> SurveyError:
    return SurveyError(f"cannot write the survey CSV in {path.parent}: {exc}")


def check_csv_writable(path: Path) -> None:
    """Fail fast, before a long probe pass, if the CSV folder cannot be written.

    Raises `SurveyError` when the folder cannot be created or a file cannot be created in it.
    An existing *path* that is locked (open in Excel) only logs a warning: `_write_csv` then
    falls back to a timestamped name.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        probe_file = Path(name)
        os.close(fd)
        probe_file.unlink()
    except OSError as exc:
        raise _csv_error(path, exc) from exc
    if path.exists():
        try:
            with path.open("ab"):
                pass
        except OSError as exc:
            _log.warning("%s is not writable (%s); a timestamped CSV will be written", path, exc)


def _write_csv(path: Path, rows: Sequence[_Row]) -> Path:
    """Write the CSV atomically (`atomicio`) and return the path written.

    If *path* is locked, the file goes to ``survey-<timestamp>.csv`` in the same folder. Any
    other failure raises `SurveyError`; the previous CSV and no temporary file stay behind.
    """
    lines = [_csv_line(r) for r in rows]

    def write(target: Path) -> None:
        # backslashreplace: an undecodable file name (lone surrogate) still yields valid UTF-8.
        write_csv_atomic(target, CSV_COLUMNS, lines, errors="backslashreplace")

    try:
        try:
            write(path)
            return path
        except PermissionError as exc:
            fallback = path.with_name(FALLBACK_CSV_PATTERN.format(datetime.now()))
            _log.warning("cannot replace %s (%s); writing %s instead", path, exc, fallback)
            write(fallback)
            return fallback
    except OSError as exc:
        raise _csv_error(path, exc) from exc


def run_survey(
    settings: Settings,
    tools: Tools,
    state: State,
    reporter: Reporter,
    *,
    root: Path | None = None,
    workers: int = 4,
    probe_fn: ProbeFn = probe,
    hash_fn: HashFn = movie_hash,
) -> SurveyResult:
    """Probe every video file under *root* (default ``settings.library_root``), read-only.

    Each file is decided and upserted into *state*; a compatible multi-part group becomes one
    record on its first part. Writes ``survey.csv`` beside the state database. Raises
    `SurveyError` (before probing) if an output would lie under the library or the CSV
    folder is not writable, and if the CSV cannot be written at all. `ToolNotFound` from the
    probe and `KeyboardInterrupt` abort the survey; queued probes are cancelled and records
    already upserted stay. Every probed file also gets its OpenSubtitles movie hash
    (`moviehash`, read-only, 128 KiB per file), stored on its record; a file whose hash cannot
    be computed is surveyed without one.
    """
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    root = settings.library_root if root is None else root
    csv_path = check_outputs_outside(settings, root)
    check_csv_writable(csv_path)

    reporter.stage("survey")
    paths = discover_files(root)
    _log.info("survey of %s: %d video files", root, len(paths))
    infos, failures, hashes = _probe_all(paths, tools, reporter, workers, probe_fn, hash_fn)

    groups = tuple(find_part_groups(paths))
    rows: list[_Row] = []
    grouped: set[Path] = set()
    for group in groups:
        grouped.update(group.parts)
        rows.extend(_record(r, state) for r in _plan_group(group, infos, hashes, settings))
    for path in paths:
        if path in failures:
            reason = f"unprobeable: {failures[path]}"
            rows.append(_Row(path, _size_or_none(path), None, Action.FLAG, (reason,)))
        elif path not in grouped:
            info = infos[path]
            d = decide(info, settings)
            saving = _encode_saving(d.action, info.size)
            row = _Row(path, info.size, info, d.action, d.reasons, saving, movie_hash=hashes[path])
            rows.append(_record(row, state))
    rows.sort(key=lambda r: r.path)

    csv_path = _write_csv(csv_path, rows)
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
