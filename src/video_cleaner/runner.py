"""The resumable run queue: copy, encode or remux, guard, verify and deliver, file by file.

Files are taken PENDING first, then RETRY, biggest estimated saving first. For each one the
sources are copied to scratch (the next file is prefetched in the background meanwhile), the
original is re-checked against its state record, and then:

* encode: CRF search, a projected saving under ``min_saving`` leaves the file alone; otherwise
  full encode, mux of the new video with the source's untouched streams, the VMAF window guard
  and `verify_output`;
* remux only: mkvmerge with the sources and no new video (every source video stream survives);
* multi-part (Ruling R11): the parts are appended into one combined file first, which is then
  decided, encoded or delivered like a single file.

A verified output is moved to the completed folder and recorded in ``manifest.csv``; scratch is
released after every file, whatever happened. The library is never written: every output path
lives under ``scratch_dir`` or ``completed_dir``, and a run refuses to start if any of them is
under ``library_root``.

The stages are injectable through `Pipeline`. Its defaults are the real implementations; the
encode, CRF search and VMAF scoring defaults are still held (Task 10b) and raise
``NotImplementedError("held: Task 10b")``, which stops the run and puts the file back to
PENDING.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import logging
import math
import os
import re
import shutil
import subprocess  # nosec B404
import tempfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from video_cleaner import doctor, encode, planning, quality
from video_cleaner.config import Settings
from video_cleaner.decide import Action, decide
from video_cleaner.encode import CrfSearchResult
from video_cleaner.instancelock import instance_lock, lock_path_for
from video_cleaner.logging_setup import redact
from video_cleaner.manifest import (
    ManifestEntry,
    ManifestError,
    check_unchanged,
    completed_path_for,
    read_manifest,
    write_manifest,
)
from video_cleaner.multipart import find_part_groups
from video_cleaner.mux import build_mux_cmd
from video_cleaner.paths import is_under, path_key
from video_cleaner.probe import MediaInfo, ProbeError, probe
from video_cleaner.progress import Progress, Reporter
from video_cleaner.scratch import (
    CopyResult,
    Prefetcher,
    ScratchFull,
    ScratchManager,
    SourceChanged,
    copy_with_progress,
)
from video_cleaner.state import IN_PROGRESS, FileRecord, StaleRecord, State, Status
from video_cleaner.tools import (
    ToolFailed,
    ToolNotFound,
    ToolOutputError,
    Tools,
    mkvmerge_parser,
    run_tool,
)
from video_cleaner.verify import VerifyResult, verify_output

_log = logging.getLogger(__name__)

HELD_REASON = "held: Task 10b"
STOP_SCRATCH_FULL = "scratch full"
STOP_LIBRARY_UNREACHABLE = "library unreachable"
STOP_MANIFEST_UNREADABLE = "manifest unreadable"
STOP_MANIFEST_NOT_WRITABLE = "manifest not writable"
STOP_COMPLETED_NOT_WRITABLE = "completed folder not writable"
STALE_NOTE = "source changed"
MANIFEST_NAME = "manifest.csv"
# A delivery in flight: ``<target>.part`` in the completed folder (targets are always .mkv).
DELIVERY_SUFFIX = ".part"
_DELIVERY_GLOB = "*.mkv" + DELIVERY_SUFFIX
RECOVERED_NOTE = "recovered after an interrupted delivery"
# mkvmerge exits 1 for warnings (the output is fine) and 2 for errors.
MKVMERGE_ERROR_EXIT = 2

_WORK_DECISIONS = frozenset({Action.ENCODE.value, Action.REMUX.value})
_HHMM = re.compile(r"(\d{1,2}):(\d{2})")
# Names this module (and the Prefetcher) create in scratch; the startup sweep touches no others.
_COPY_NAME = re.compile(r"[0-9a-f]{8}-.+")
_WORK_NAME = re.compile(r"work-[0-9a-f]{12}")


class PlanNotAcknowledged(Exception):
    """`run` was asked for before the plan was acknowledged."""


class CalibrationNotApproved(Exception):
    """`run` was asked for before a calibration was approved (or the approval no longer fits)."""


class RunRefused(Exception):
    """The configuration would make the run write under the library root."""


class _SourceStale(Exception):
    """The original changed (or vanished from a reachable library) since it was planned."""


class _Refused(Exception):
    """The file cannot be delivered safely; it is flagged with this reason."""


class _ManifestTrouble(Exception):
    """The manifest cannot be read or written: no file can be delivered, so the run stops."""


# --- pipeline ------------------------------------------------------------------------------


def _run_mkvmerge(tools: Tools, cmd: Sequence[str], output: Path, reporter: Reporter) -> None:
    log = output.with_name(output.name + ".stderr.log")
    result = run_tool(
        cmd,
        label=f"mkvmerge {output.name}",
        reporter=reporter,
        parser=mkvmerge_parser,
        stderr_log=log,
        check=False,
    )
    if result.returncode >= MKVMERGE_ERROR_EXIT:
        cmd_text = redact(subprocess.list2cmdline([str(part) for part in cmd]))
        raise ToolFailed(result.returncode, cmd_text, log)
    if result.returncode != 0:
        _log.info("mkvmerge finished with warnings (exit %d) for %s", result.returncode, output)


def append_parts(tools: Tools, parts: Sequence[Path], output: Path, reporter: Reporter) -> None:
    """mkvmerge-append *parts* into *output* (no new video: every stream is kept)."""
    _run_mkvmerge(tools, build_mux_cmd(tools, output, parts, None), output, reporter)


def mux(
    tools: Tools,
    sources: Sequence[Path],
    new_video: Path | None,
    output: Path,
    reporter: Reporter,
) -> None:
    """mkvmerge *sources* (with *new_video* swapped in, when given) into *output*."""
    _run_mkvmerge(tools, build_mux_cmd(tools, output, sources, new_video), output, reporter)


def held_window_scores(
    tools: Tools,
    source: MediaInfo,
    output: Path,
    settings: Settings,
    workdir: Path,
    reporter: Reporter,
) -> Sequence[float]:
    """VMAF of each post-encode window of *output* against *source*. HELD until Task 10b."""
    raise NotImplementedError(HELD_REASON)


# The stages that are still signature stubs. A replacement can also declare itself held with
# a truthy ``held`` attribute. `run_queue` refuses to start while any of them is in the pipeline.
_HELD_STAGES: tuple[object, ...] = (encode.search_crf, encode.encode_full, held_window_scores)


@dataclass
class Pipeline:
    """The stages `run_queue` calls; tests replace them with fakes.

    ``copy`` is the foreground copy. The background copy of the next file is done by the
    Prefetcher that ``prefetcher`` builds (its own cancellable copy loop); tests that must not
    copy for real inject a different ``prefetcher`` here.
    """

    copy: Callable[[Path, Path, Reporter], CopyResult] = copy_with_progress
    search_crf: Callable[[MediaInfo, Tools, Settings, Path, Reporter], CrfSearchResult] = (
        encode.search_crf
    )
    encode_full: Callable[[MediaInfo, Tools, Settings, int, Path, Reporter], None] = (
        encode.encode_full
    )
    append_parts: Callable[[Tools, Sequence[Path], Path, Reporter], None] = append_parts
    mux: Callable[[Tools, Sequence[Path], Path | None, Path, Reporter], None] = mux
    window_scores: Callable[[Tools, MediaInfo, Path, Settings, Path, Reporter], Sequence[float]] = (
        held_window_scores
    )
    verify: Callable[..., VerifyResult] = verify_output
    probe: Callable[[Path, Tools, Reporter], MediaInfo] = probe
    prefetcher: Callable[[ScratchManager, Reporter], Prefetcher] = Prefetcher

    def held_stages(self) -> list[str]:
        """Names of the stages that are still held stubs."""
        stages = {
            "search_crf": self.search_crf,
            "encode_full": self.encode_full,
            "window_scores": self.window_scores,
        }
        return [
            name
            for name, stage in stages.items()
            if any(stage is held for held in _HELD_STAGES) or getattr(stage, "held", False)
        ]


# --- deadline and summary ------------------------------------------------------------------


@dataclass(frozen=True)
class RunSummary:
    completed: int
    left_alone: int
    flagged: int
    retry: int
    bytes_saved: int
    stopped_by_deadline: bool
    stop_reason: str = ""


def deadline_from(until: str | None, max_hours: float | None, now: datetime) -> datetime | None:
    """When the run must stop starting new files; None for no limit.

    *until* is a wall-clock ``HH:MM`` that rolls to the next day when it is not after *now*;
    *max_hours* counts from *now*. With both, the earlier wins. Raises ValueError for a
    malformed time or a non-positive or non-finite number of hours.
    """
    candidates: list[datetime] = []
    if until is not None:
        match = _HHMM.fullmatch(until)
        hour, minute = (int(match.group(1)), int(match.group(2))) if match else (-1, -1)
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError(f"--until must be HH:MM between 00:00 and 23:59, got {until!r}")
        at = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if at <= now:
            at += timedelta(days=1)
        candidates.append(at)
    if max_hours is not None:
        if not (math.isfinite(max_hours) and max_hours > 0):
            raise ValueError(f"--max-hours must be a positive number of hours, got {max_hours!r}")
        candidates.append(now + timedelta(hours=max_hours))
    return min(candidates) if candidates else None


# --- helpers -------------------------------------------------------------------------------


def _clean(note: str) -> str:
    """A note SQLite can store even if a file name carries undecodable bytes."""
    return note.encode("utf-8", "backslashreplace").decode("utf-8")


class _BackgroundReporter:
    """Reporter for the background prefetch: never touches the console, logs at DEBUG."""

    def stage(self, label: str) -> None:
        _log.debug("prefetch: %s", label)

    def update(self, p: Progress) -> None:
        pass

    def heartbeat(self, label: str, elapsed_s: float) -> None:
        pass

    def file_done(self, summary: str) -> None:
        _log.debug("prefetch: %s", summary)


@dataclass
class _Outcome:
    status: Status
    note: str
    bytes_saved: int = 0


@dataclass
class _Work:
    """Scratch owned by the file in progress, released whatever happens."""

    workdir: Path
    reserved: list[Path] = field(default_factory=list)


def _delivery_part(target: Path) -> Path:
    return target.with_name(target.name + DELIVERY_SUFFIX)


def _file_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _same_file(a: Path, b: Path) -> bool:
    return _file_key(a) == _file_key(b)


def refuse_outputs_in_library(settings: Settings) -> None:
    """Raise `RunRefused` if anything a run writes would land under the library root, or if
    the library root lies inside a folder a run writes to or cleans."""
    for label, path in (
        ("scratch folder", settings.scratch_dir),
        ("completed folder", settings.completed_dir),
        ("state database", settings.state_db),
        ("log folder", settings.log_dir),
    ):
        if is_under(path, settings.library_root):
            raise RunRefused(
                f"{label} {path} is under the library root {settings.library_root}; "
                "a run never writes to the library"
            )
    # The other way round is as bad: completed paths mirror the library's relative folders,
    # so a library inside the completed folder can be written to, and scratch sweeps delete.
    for label, path in (
        ("scratch folder", settings.scratch_dir),
        ("completed folder", settings.completed_dir),
        ("log folder", settings.log_dir),
    ):
        if is_under(settings.library_root, path):
            raise RunRefused(
                f"the library root {settings.library_root} is inside the {label} {path}; "
                "a run never writes to the library"
            )


def apply_calibration(settings: Settings, state: State) -> Settings:
    """*settings* with the approved ``vmaf_target`` and ``crf_max`` (ruling R3).

    Raises `CalibrationNotApproved` when nothing was approved, when the approved ceiling is
    below the configured ``crf_min``, or when the approved VMAF target is below
    ``guard_mean_floor`` (the guard would reject every encode); the last two mean the config
    changed after the approval.
    """
    approved = state.calibration()
    if approved is None:
        raise CalibrationNotApproved(
            "no calibration has been approved: run `calibrate --samples N`, review the table "
            "and approve it with `calibrate --approve --vmaf-target X --crf-max Y`"
        )
    vmaf_target, crf_max = approved
    if crf_max < settings.crf_min:
        raise CalibrationNotApproved(
            f"the approved crf_max {crf_max} is below the configured crf_min "
            f"{settings.crf_min}: approve the calibration again with `calibrate --approve`"
        )
    if vmaf_target < settings.guard_mean_floor:
        raise CalibrationNotApproved(
            f"the approved vmaf_target {vmaf_target:g} is below the configured guard_mean_floor "
            f"{settings.guard_mean_floor:g}: the quality guard would reject every encode; "
            "approve the calibration again with `calibrate --approve`"
        )
    return dataclasses.replace(settings, vmaf_target=vmaf_target, crf_max=crf_max)


# --- the run -------------------------------------------------------------------------------


class _Runner:
    def __init__(
        self,
        settings: Settings,
        tools: Tools,
        state: State,
        reporter: Reporter,
        pipeline: Pipeline,
        deadline: datetime | None,
        only: Path | None,
        now: Callable[[], datetime],
    ) -> None:
        self.settings = settings
        self.tools = tools
        self.state = state
        self.reporter = reporter
        self.pipeline = pipeline
        self.deadline = deadline
        self.only = only
        self.now = now
        self.scratch = ScratchManager(settings.scratch_dir, settings.scratch_cap_bytes)
        self.prefetcher = pipeline.prefetcher(self.scratch, _BackgroundReporter())
        self.prefetched: Path | None = None
        self.attempted: set[tuple[str, int, float]] = set()
        self.claims: dict[str, list[Path]] = {}
        self.counts = dict.fromkeys(
            (Status.COMPLETED, Status.LEFT_ALONE, Status.FLAGGED, Status.RETRY), 0
        )
        self.bytes_saved = 0

    # -- selection ------------------------------------------------------------------------

    def _in_scope(self, rec: FileRecord) -> bool:
        if self.only is None:
            return True
        return any(is_under(p, self.only) for p in (rec.path, *rec.parts))

    def _wanted(self, rec: FileRecord) -> bool:
        return (
            rec.decision in _WORK_DECISIONS
            and (path_key(rec.path), rec.size, rec.mtime) not in self.attempted
            and self._in_scope(rec)
        )

    def _next(self) -> FileRecord | None:
        for status in (Status.PENDING, Status.RETRY):
            records = [r for r in self.state.records(status) if self._wanted(r)]
            if records:
                # records() is ordered by path, so equal savings keep a stable order
                return max(records, key=lambda r: r.est_saving_bytes)
        return None

    # -- targets and collisions -----------------------------------------------------------

    def _target(self, rec: FileRecord) -> Path:
        if not rec.parts:
            return completed_path_for(self.settings, rec.path)
        groups = find_part_groups(list(rec.parts))
        if len(groups) != 1 or len(groups[0].parts) != len(rec.parts):
            raise _Refused(f"the parts of {rec.path} no longer form one multi-part group")
        return completed_path_for(self.settings, rec.parts[0], new_name=groups[0].output_name)

    def claim_targets(self) -> None:
        """Map every planned output name to the records that would write it."""
        for rec in self.state.records(Status.PENDING, Status.RETRY, Status.COMPLETED):
            if rec.decision not in _WORK_DECISIONS:
                continue
            try:
                target = self._target(rec)
            except (ValueError, _Refused):
                continue  # flagged with the reason when the record comes up
            self.claims.setdefault(str(target).casefold(), []).append(rec.path)

    def _library_twin(self, rec: FileRecord, target: Path) -> Path | None:
        """The library file that *target* would later replace, unless it is one of *rec*'s own.

        The completed folder mirrors the library, so ``completed/a/Film.mkv`` stands for
        ``library/a/Film.mkv``; an unrelated file there must not be overwritten one day.
        """
        relative = target.relative_to(self.settings.completed_dir)
        folder = self.settings.library_root / relative.parent
        try:
            names = os.listdir(folder)
        except OSError:
            return None
        wanted = relative.name.casefold()
        own = {path_key(p) for p in (rec.path, *rec.parts)}
        for name in names:
            if name.casefold() == wanted:
                twin = folder / name
                if path_key(twin) not in own:
                    return twin
        return None

    def _precheck(self, rec: FileRecord) -> Path:
        """The completed path for *rec*; raises `_Refused` if delivering there is unsafe."""
        try:
            target = self._target(rec)
        except ValueError as exc:
            raise _Refused(str(exc)) from None
        others = [
            p
            for p in self.claims.get(str(target).casefold(), [])
            if path_key(p) != path_key(rec.path)
        ]
        if others:
            raise _Refused(f"output name collides with {', '.join(str(p) for p in others)}")
        twin = self._library_twin(rec, target)
        if twin is not None:
            raise _Refused(f"output name collides with {twin}")
        if target.exists():
            raise _Refused(f"output {target} already exists in the completed folder")
        return target

    # -- status ---------------------------------------------------------------------------

    def _set(self, rec: FileRecord, status: Status, note: str = "") -> None:
        """Advance *rec*; raises `StaleRecord` if it is no longer the live record."""
        self.state.set_status(rec.path, status, _clean(note), size=rec.size, mtime=rec.mtime)
        self.reporter.stage(f"{status.value}: {rec.path.name}")

    def _finish(self, rec: FileRecord, outcome: _Outcome) -> None:
        try:
            self.state.set_status(
                rec.path, outcome.status, _clean(outcome.note), size=rec.size, mtime=rec.mtime
            )
        except StaleRecord:
            _log.warning("%s: record superseded; outcome %s dropped", rec.path, outcome.status)
            return
        if outcome.status in self.counts:
            self.counts[outcome.status] += 1
        self.bytes_saved += outcome.bytes_saved
        detail = f" ({outcome.note})" if outcome.note else ""
        _log.info("%s: %s%s", rec.path, outcome.status.value, detail)
        self.reporter.file_done(f"{rec.path.name}: {outcome.status.value}{detail}")

    # -- scratch --------------------------------------------------------------------------

    def _library_reachable(self) -> bool:
        return os.path.isdir(self.settings.library_root)

    def _reserve(self, size: int, path: Path, work: _Work) -> None:
        if not self.scratch.reserve(size, path):
            raise ScratchFull(
                f"no room in scratch for {path.name} ({size} bytes, cap "
                f"{self.settings.scratch_cap_bytes} bytes)"
            )
        work.reserved.append(path)

    def _copy(self, src: Path, size: int, work: _Work) -> Path:
        if self.prefetched is not None and path_key(self.prefetched) == path_key(src):
            self.prefetched = None
            self.reporter.stage(f"waiting for prefetch: {src.name}")
            result = self.prefetcher.result()
            work.reserved.append(result.dest)
            return result.dest
        dest = self.prefetcher.dest_for(src)
        self._reserve(size, dest, work)
        self.reporter.stage(f"copy: {src.name}")
        self.pipeline.copy(src, dest, self.reporter)
        return dest

    def _start_prefetch(self) -> None:
        if self.prefetched is not None:
            return
        if self.deadline is not None and self.now() >= self.deadline:
            return  # nothing more will start
        nxt = self._next()
        if nxt is None:
            return
        try:
            self._precheck(nxt)
        except _Refused:
            return
        src = (nxt.parts or (nxt.path,))[0]
        self.prefetcher.start(src)
        self.prefetched = src

    def drop_prefetch(self) -> None:
        if self.prefetched is not None:
            self.prefetched = None
            self.prefetcher.cancel()

    def _release(self, work: _Work) -> None:
        for path in work.reserved:
            try:
                self.scratch.release(path)
            except OSError as exc:
                _log.warning("could not remove scratch file %s: %s", path, exc)
        try:
            shutil.rmtree(work.workdir)
        except FileNotFoundError:
            pass
        except OSError as exc:
            _log.warning("could not remove scratch folder %s: %s", work.workdir, exc)

    def sweep(self) -> None:
        """Remove what a killed run left in scratch: partial copies, work folders, orphans."""
        keep: set[str] = set()
        for rec in self.state.records(Status.PENDING, Status.RETRY):
            keep.update(path_key(self.prefetcher.dest_for(p)) for p in (rec.parts or (rec.path,)))
        for entry in os.scandir(self.settings.scratch_dir):
            try:
                if entry.is_dir(follow_symlinks=False):
                    if _WORK_NAME.fullmatch(entry.name):
                        shutil.rmtree(entry.path)
                        _log.info("removed leftover scratch folder %s", entry.path)
                elif _COPY_NAME.fullmatch(entry.name) and (
                    entry.name.endswith(".part") or path_key(entry.path) not in keep
                ):
                    os.unlink(entry.path)
                    _log.info("removed leftover scratch file %s", entry.path)
            except OSError as exc:
                _log.warning("could not remove leftover %s: %s", entry.path, exc)

    @property
    def _manifest_path(self) -> Path:
        return self.settings.completed_dir / MANIFEST_NAME

    def _read_manifest(self) -> list[ManifestEntry]:
        try:
            return read_manifest(self._manifest_path)
        except FileNotFoundError:
            return []
        except (ManifestError, OSError) as exc:
            raise _ManifestTrouble(f"{STOP_MANIFEST_UNREADABLE}: {exc}") from exc

    def check_delivery(self) -> str:
        """Before any copy: can results be delivered at all? Returns a stop reason or ""."""
        folder = self.settings.completed_dir
        try:
            folder.mkdir(parents=True, exist_ok=True)
            fd, probe_name = tempfile.mkstemp(dir=folder, prefix=".write-check-")
            os.close(fd)
            os.unlink(probe_name)
        except OSError as exc:
            return f"{STOP_COMPLETED_NOT_WRITABLE}: {exc}"
        try:
            self._read_manifest()
        except _ManifestTrouble as exc:
            return str(exc)
        if self._manifest_path.exists():
            try:
                # Opening for append changes nothing but fails if the file is locked/read-only.
                with open(self._manifest_path, "a", encoding="utf-8"):
                    pass
            except OSError as exc:
                return f"{STOP_MANIFEST_NOT_WRITABLE}: {exc}"
        return ""

    def _drop_manifest_row(self, target: Path) -> None:
        """Remove *target*'s row again after its delivery failed (best effort)."""
        try:
            entries = self._read_manifest()
            write_manifest(
                self._manifest_path, [e for e in entries if not _same_file(e.completed, target)]
            )
        except (_ManifestTrouble, ManifestError, OSError) as exc:
            _log.warning("could not remove the manifest row of %s: %s", target, exc)

    def _recover(self, entry: ManifestEntry) -> None:
        """Mark the record of a delivered *entry* COMPLETED if a killed run left it unfinished."""
        rec = self.state.get(entry.original[0])
        if rec is None or rec.status not in (Status.RETRY, *IN_PROGRESS):
            return
        if (rec.size, rec.mtime) != (entry.original_sizes[0], entry.original_mtimes[0]):
            return
        try:
            if entry.completed.stat().st_size != entry.completed_size:
                return
        except OSError:
            return
        saved = sum(entry.original_sizes) - entry.completed_size
        note = f"{entry.action}: saved {saved} bytes ({RECOVERED_NOTE})"
        self._finish(rec, _Outcome(Status.COMPLETED, note, saved))

    def _delivered_earlier(self, entry: ManifestEntry) -> bool:
        rec = self.state.get(entry.original[0])
        return rec is not None and rec.status is Status.COMPLETED

    def reconcile(self) -> str:
        """Repair what a run killed during delivery left behind. Returns a stop reason or "".

        A manifest row whose target is missing but whose ``.part`` exists gets the rename
        finished; a row with neither file is dropped unless its record is already COMPLETED (a
        delivery the owner has since moved away); a delivered target whose record a killed run
        left in progress or RETRY is marked COMPLETED; and a ``*.mkv.part`` with no row is
        removed. Nothing else in the completed folder is touched.
        """
        try:
            entries = self._read_manifest()
        except _ManifestTrouble as exc:
            return str(exc)
        kept: list[ManifestEntry] = []
        for entry in entries:
            target = entry.completed
            part = _delivery_part(target)
            if not target.exists() and part.exists():
                try:
                    os.rename(part, target)
                    _log.info("finished the interrupted delivery of %s", target)
                except OSError as exc:
                    _log.warning("could not finish the delivery of %s: %s", target, exc)
            if target.exists() or part.exists() or self._delivered_earlier(entry):
                kept.append(entry)
            else:
                _log.warning("dropped the manifest row of %s: no output was delivered", target)
        if len(kept) != len(entries):
            try:
                write_manifest(self._manifest_path, kept)
            except (ManifestError, OSError) as exc:
                return f"{STOP_MANIFEST_NOT_WRITABLE}: {exc}"
        for entry in kept:
            if entry.completed.exists():
                self._recover(entry)
        delivered = {_file_key(e.completed) for e in kept}
        for part in self.settings.completed_dir.rglob(_DELIVERY_GLOB):
            if _file_key(part.with_name(part.name[: -len(DELIVERY_SUFFIX)])) in delivered:
                continue
            try:
                part.unlink()
                _log.info("removed unfinished delivery %s", part)
            except OSError as exc:
                _log.warning("could not remove unfinished delivery %s: %s", part, exc)
        return ""

    def _keep_tool_log(self, log: Path | None) -> Path | None:
        """Copy a failed tool's stderr log out of scratch (which is about to be released)."""
        if log is None or not log.is_file() or not is_under(log, self.settings.scratch_dir):
            return log
        stamp = f"{self.now():%Y%m%d-%H%M%S}"
        tag = hashlib.sha1(str(log).encode("utf-8", "surrogatepass"), usedforsecurity=False)
        name = log.name if log.name.endswith(".log") else log.name + ".log"
        kept = self.settings.log_dir / f"run-{stamp}-{tag.hexdigest()[:8]}-{name}"
        try:
            self.settings.log_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(log, kept)
        except OSError as exc:
            _log.warning("could not keep tool log %s: %s", log, exc)
            return None
        return kept

    # -- one file -------------------------------------------------------------------------

    def _expected(self, rec: FileRecord, sources: Sequence[Path]) -> list[tuple[int, float]]:
        """The (size, mtime) each source had when it was planned."""
        out: list[tuple[int, float]] = []
        for src in sources:
            if path_key(src) == path_key(rec.path):
                out.append((rec.size, rec.mtime))
                continue
            member = self.state.get(src)
            if member is not None and member.status is not Status.STALE:
                out.append((member.size, member.mtime))
            else:
                st = os.stat(src)
                out.append((st.st_size, st.st_mtime))
        return out

    def _assert_unchanged(
        self, sources: Sequence[Path], expected: Sequence[tuple[int, float]]
    ) -> None:
        for src, (size, mtime) in zip(sources, expected, strict=True):
            try:
                st = os.stat(src)
            except (FileNotFoundError, NotADirectoryError):
                if not self._library_reachable():
                    raise
                raise _SourceStale(f"{src} is gone") from None
            if (st.st_size, st.st_mtime) != (size, mtime):
                raise _SourceStale(
                    f"{src}: size {size} -> {st.st_size}, mtime {mtime} -> {st.st_mtime}"
                )

    def _still_live(self, rec: FileRecord) -> None:
        live = self.state.get(rec.path)
        same = live is not None and (live.size, live.mtime) == (rec.size, rec.mtime)
        if live is None or live.status is Status.STALE or not same:
            raise StaleRecord(f"{rec.path} was re-planned during the run")

    def _deliver(
        self,
        rec: FileRecord,
        sources: Sequence[Path],
        expected: Sequence[tuple[int, float]],
        output: Path,
        target: Path,
        *,
        action: str,
        crf: int | None,
        vmaf: tuple[float, float] | None,
    ) -> _Outcome:
        completed_size = output.stat().st_size
        entry = ManifestEntry(
            original=tuple(sources),
            original_sizes=tuple(size for size, _ in expected),
            original_mtimes=tuple(mtime for _, mtime in expected),
            completed=target,
            completed_size=completed_size,
            vmaf_mean=None if vmaf is None else vmaf[0],
            vmaf_worst=None if vmaf is None else vmaf[1],
            crf=crf,
            verified=True,
            action=action,
        )
        # Second line of defence: the original may have changed during the encode.
        if not check_unchanged(entry):
            if not self._library_reachable():
                raise OSError(f"library root {self.settings.library_root} is unreachable")
            raise _SourceStale(f"{rec.path} changed during processing")
        self._still_live(rec)

        manifest_path = self._manifest_path
        entries = self._read_manifest()
        key = str(target).casefold()
        if any(str(e.completed).casefold() == key for e in entries):
            raise _Refused(f"output {target} is already in the manifest")
        if target.exists():
            raise _Refused(f"output {target} already exists in the completed folder")

        # Order (plan section 6, crash safety): output -> <target>.part, manifest row, then
        # rename .part -> target. A kill at any point is repaired by `reconcile` at the next
        # start: a .part without a row is removed, a row whose .part is still there is finished.
        # Cleanup below runs for errors only; a kill leaves exactly that state behind.
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = _delivery_part(target)
        tmp.unlink(missing_ok=True)  # an orphan of ours (reconcile normally removed it)
        try:
            shutil.move(output, tmp)  # a rename on one volume, copy and delete across volumes
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        try:
            write_manifest(manifest_path, [*entries, entry])
        except Exception as exc:
            tmp.unlink(missing_ok=True)  # no output without its manifest row
            if isinstance(exc, ManifestError | OSError):
                raise _ManifestTrouble(f"{STOP_MANIFEST_NOT_WRITABLE}: {exc}") from exc
            raise
        try:
            # os.rename refuses an existing target on Windows: delivery never overwrites.
            os.rename(tmp, target)
        except OSError as exc:
            _log.error("%s: delivered as %s but not renamed: %s", rec.path, tmp, exc)
            self._drop_manifest_row(target)
            tmp.unlink(missing_ok=True)
            raise _Refused(f"output {target} could not be put in place: {exc}") from exc
        saved = sum(entry.original_sizes) - completed_size
        return _Outcome(Status.COMPLETED, f"{action}: saved {saved} bytes", saved)

    def _work(self, rec: FileRecord, target: Path, work: _Work) -> _Outcome:
        s, tools, rep, pipe = self.settings, self.tools, self.reporter, self.pipeline
        self._set(rec, Status.QUEUED)
        self._set(rec, Status.COPYING)
        sources = list(rec.parts) or [rec.path]
        expected = self._expected(rec, sources)
        copies = [
            self._copy(src, size, work) for src, (size, _) in zip(sources, expected, strict=True)
        ]
        self._assert_unchanged(sources, expected)
        self._start_prefetch()
        work.workdir.mkdir(parents=True, exist_ok=True)

        duration_s: float | None = None
        original_size: int | None = None
        if rec.parts:
            combined = work.workdir / "combined.mkv"
            original_size = sum(size for size, _ in expected)
            self._reserve(original_size, combined, work)
            rep.stage(f"append parts: {rec.path.name}")
            pipe.append_parts(tools, copies, combined, rep)
            duration_s = sum(pipe.probe(c, tools, rep).duration_s for c in copies)
            info = pipe.probe(combined, tools, rep)
            decision = decide(info, s, multipart=True)
            if decision.action is Action.FLAG:
                return _Outcome(Status.FLAGGED, "; ".join(decision.reasons))
            source = combined
            action = Action.ENCODE if decision.encode else Action.REMUX
        else:
            info = pipe.probe(copies[0], tools, rep)
            source = copies[0]
            action = Action(rec.decision)

        verify_args = {
            "min_saving": s.min_saving,
            "original_duration_s": duration_s,
            "original_size": original_size,
        }
        if action is Action.REMUX:
            if rec.parts:
                output = source  # the append already is the remux
            else:
                self._set(rec, Status.MUXING)
                output = work.workdir / target.name
                self._reserve(rec.size, output, work)
                pipe.mux(tools, [source], None, output, rep)
            self._set(rec, Status.VERIFYING)
            verdict = pipe.verify(tools, info, output, rep, encoded=False, **verify_args)
            if not verdict.ok:
                return _Outcome(Status.FLAGGED, f"verify failed: {verdict.reason}")
            return self._deliver(
                rec, sources, expected, output, target, action="remux", crf=None, vmaf=None
            )

        self._set(rec, Status.ENCODING)
        search = pipe.search_crf(info, tools, s, work.workdir, rep)
        if search.projected_saving < s.min_saving:
            return _Outcome(
                Status.LEFT_ALONE,
                f"projected saving {search.projected_saving:.1%} at CRF {search.crf} is below "
                f"the minimum {s.min_saving:.0%}",
            )
        encoded = work.workdir / "video.mkv"
        pipe.encode_full(info, tools, s, search.crf, encoded, rep)
        self._set(rec, Status.MUXING)
        output = work.workdir / target.name
        pipe.mux(tools, [source], encoded, output, rep)
        self._set(rec, Status.VERIFYING)
        scores = pipe.window_scores(tools, info, output, s, work.workdir, rep)
        guard = quality.evaluate_guard(scores, s.guard_mean_floor, s.guard_worst_floor)
        if not guard.ok:
            return _Outcome(Status.LEFT_ALONE, f"quality guard failed: {guard.reason}")
        verdict = pipe.verify(tools, info, output, rep, encoded=True, **verify_args)
        if not verdict.ok:
            if verdict.too_large:
                return _Outcome(Status.LEFT_ALONE, f"output not kept: {verdict.reason}")
            return _Outcome(Status.FLAGGED, f"verify failed: {verdict.reason}")
        return self._deliver(
            rec,
            sources,
            expected,
            output,
            target,
            action="encode",
            crf=search.crf,
            vmaf=(guard.mean, guard.worst),
        )

    def process(self, rec: FileRecord) -> str:
        """Handle one record; returns a stop reason, or "" to carry on with the next file."""
        self.attempted.add((path_key(rec.path), rec.size, rec.mtime))
        sources = rec.parts or (rec.path,)
        if self.prefetched is not None and path_key(self.prefetched) not in {
            path_key(p) for p in sources
        }:
            self.drop_prefetch()
        try:
            target = self._precheck(rec)
        except _Refused as exc:
            self._finish(rec, _Outcome(Status.FLAGGED, str(exc)))
            return ""
        digest = hashlib.sha1(
            f"{rec.path}|{rec.size}|{rec.mtime}".encode("utf-8", "surrogatepass"),
            usedforsecurity=False,
        ).hexdigest()[:12]
        work = _Work(workdir=self.settings.scratch_dir / f"work-{digest}")
        try:
            outcome = self._work(rec, target, work)
        except (_SourceStale, SourceChanged) as exc:
            _log.warning("%s: %s; result discarded", rec.path, exc)
            outcome = _Outcome(Status.STALE, STALE_NOTE)
        except StaleRecord as exc:
            _log.warning("%s; result discarded", exc)
            return ""
        except _Refused as exc:
            outcome = _Outcome(Status.FLAGGED, str(exc))
        except ScratchFull as exc:
            self._finish(rec, _Outcome(Status.RETRY, f"scratch full: {exc}"))
            return STOP_SCRATCH_FULL
        except _ManifestTrouble as exc:
            self._finish(rec, _Outcome(Status.RETRY, str(exc)))
            return str(exc)
        except ToolNotFound as exc:
            self._finish(rec, _Outcome(Status.RETRY, str(exc)))
            return f"tool not found: {exc.name}"
        except ToolOutputError as exc:
            outcome = _Outcome(Status.RETRY, f"tool output handling failed: {exc}")
        except ToolFailed as exc:
            kept = self._keep_tool_log(exc.log_path)
            where = f"; log {kept}" if kept is not None else ""
            outcome = _Outcome(Status.RETRY, f"tool failed with exit code {exc.returncode}{where}")
        except NotImplementedError as exc:
            if str(exc) != HELD_REASON:
                outcome = _Outcome(Status.FLAGGED, f"NotImplementedError: {exc}")
            else:
                with contextlib.suppress(StaleRecord):
                    self.state.set_status(
                        rec.path, Status.PENDING, "", size=rec.size, mtime=rec.mtime
                    )
                return HELD_REASON
        except ProbeError as exc:
            outcome = _Outcome(Status.FLAGGED, str(exc))
        except OSError as exc:
            if not self._library_reachable():
                self._finish(rec, _Outcome(Status.RETRY, f"library unreachable: {exc}"))
                return STOP_LIBRARY_UNREACHABLE
            outcome = _Outcome(Status.RETRY, f"I/O error: {exc}")
        except Exception as exc:
            _log.exception("%s: unexpected error", rec.path)
            outcome = _Outcome(Status.FLAGGED, f"{type(exc).__name__}: {exc}")
        finally:
            self._release(work)
        self._finish(rec, outcome)
        return ""

    def summary(self, stopped_by_deadline: bool, stop_reason: str) -> RunSummary:
        return RunSummary(
            completed=self.counts[Status.COMPLETED],
            left_alone=self.counts[Status.LEFT_ALONE],
            flagged=self.counts[Status.FLAGGED],
            retry=self.counts[Status.RETRY],
            bytes_saved=self.bytes_saved,
            stopped_by_deadline=stopped_by_deadline,
            stop_reason=stop_reason,
        )

    def records(self) -> Iterator[FileRecord]:
        while (rec := self._next()) is not None:
            yield rec


def run_queue(
    settings: Settings,
    tools: Tools,
    state: State,
    reporter: Reporter,
    *,
    deadline: datetime | None,
    only: Path | None = None,
    now: Callable[[], datetime] = datetime.now,
    pipeline: Pipeline | None = None,
    current_log: Path | None = None,
) -> RunSummary:
    """Work through the queue until it is empty, the deadline passes or a stop condition hits.

    Raises `PlanNotAcknowledged` before touching anything if the plan was not acknowledged or
    the current plan (`planning.plan_hash`) differs from the acknowledged one,
    `CalibrationNotApproved` if no calibration was approved (or the approved CRF ceiling is now
    below the configured ``crf_min``) and `RunRefused` if an output location is under the
    library root. The approved VMAF target and CRF ceiling replace the configured ones for the
    whole run. Raises `AlreadyRunning` (before resetting, sweeping or copying anything) while
    another run or calibration holds the single-instance lock. The deadline is checked
    before each file; the file in progress always finishes. KeyboardInterrupt leaves the
    current record in its in-progress status (the next run retries it), cleans its scratch and
    propagates.
    """
    acknowledged = state.plan_acknowledged()
    if acknowledged is None:
        raise PlanNotAcknowledged(
            "the plan has not been acknowledged: review and acknowledge it with `plan` first"
        )
    if planning.plan_hash(planning.build_plan(state)) != acknowledged:
        raise PlanNotAcknowledged(
            "plan changed since it was acknowledged: run plan, review, plan --ack"
        )
    settings = apply_calibration(settings, state)
    refuse_outputs_in_library(settings)
    pipeline = pipeline or Pipeline()
    held = pipeline.held_stages()
    if held:
        _log.warning("run refused: %s (%s still held)", HELD_REASON, ", ".join(held))
        return RunSummary(0, 0, 0, 0, 0, False, HELD_REASON)
    with instance_lock(lock_path_for(settings.state_db)):
        return _run_locked(
            settings, tools, state, reporter, pipeline, deadline, only, now, current_log
        )


def _run_locked(
    settings: Settings,
    tools: Tools,
    state: State,
    reporter: Reporter,
    pipeline: Pipeline,
    deadline: datetime | None,
    only: Path | None,
    now: Callable[[], datetime],
    current_log: Path | None,
) -> RunSummary:
    """The body of `run_queue`, run while this process holds the single-instance lock."""
    reset = state.reset_in_progress()
    if reset:
        _log.info("%d file(s) left in progress by an earlier run were reset to retry", reset)
    removed = doctor.prune_logs(settings.log_dir, exclude=current_log)
    if removed:
        _log.info("pruned %d old log file(s)", len(removed))

    runner = _Runner(settings, tools, state, reporter, pipeline, deadline, only, now)
    runner.sweep()
    if not os.path.isdir(settings.library_root):
        _log.error("library root %s is unreachable; nothing started", settings.library_root)
        return runner.summary(False, STOP_LIBRARY_UNREACHABLE)
    problem = runner.check_delivery() or runner.reconcile()
    if problem:
        _log.error("run stopped before any file: %s", problem)
        return runner.summary(False, problem)
    runner.claim_targets()

    stopped_by_deadline = False
    stop_reason = ""
    try:
        for rec in runner.records():
            if deadline is not None and now() >= deadline:
                stopped_by_deadline = True
                _log.info("deadline %s reached; stopping before %s", deadline, rec.path)
                break
            stop_reason = runner.process(rec)
            if stop_reason:
                _log.warning("run stopped: %s", stop_reason)
                break
    finally:
        runner.drop_prefetch()
    return runner.summary(stopped_by_deadline, stop_reason)
