"""The plan the user reviews before a run, its hash, and the results report after one.

`build_plan` turns the live state records into one row per file to review: what will happen,
why, how big it is. A multi-part movie is one row (its first part, with every part listed) and
one entry in ``groups``; the member parts are not rows of their own, so nothing is counted
twice. `plan_hash` fingerprints what was decided (not how far a run got), so acknowledging a
plan means acknowledging exactly the reviewed decisions, and a survey that changes any of them
gives a new hash.

`build_results` reads the completed manifest and the state for the `report` command.

Nothing here writes to the library: the CSV and text files go beside the state database (the
CLI checks that before calling), always via a temporary file moved into place.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from video_cleaner.atomicio import write_csv_atomic, write_text_atomic
from video_cleaner.decide import DV5_REPLACE_REASON, Action
from video_cleaner.manifest import ManifestEntry, check_unchanged
from video_cleaner.multipart import find_part_groups
from video_cleaner.state import FileRecord, State, Status, SubtitleStatus

PLAN_CSV = "plan.csv"
PLAN_HASH = "plan.hash"
REPORT_CSV = "report.csv"
LIST_SEPARATOR = "; "
MEMBER_REASON_PREFIX = "part of group "
PLAN_COLUMNS = (
    "path",
    "action",
    "status",
    "subtitle",
    "size",
    "est_saving_bytes",
    "group",
    "parts",
    "reasons",
)
REPORT_COLUMNS = (
    "original",
    "completed",
    "original_size",
    "completed_size",
    "saving_pct",
    "vmaf_mean",
    "vmaf_worst",
    "crf",
    "action",
    "original_unchanged",
)
UNCHANGED_YES = "yes"
UNCHANGED_NO = "no"
UNCHANGED_UNKNOWN = "unknown"
UNCHANGED_VERDICTS = (UNCHANGED_YES, UNCHANGED_NO, UNCHANGED_UNKNOWN)
# Said for a record that somehow carries no reason, so every row explains itself.
_DEFAULT_REASON = {
    Action.ENCODE.value: "encode candidate",
    Action.REMUX.value: "remux only",
    Action.LEAVE_ALONE.value: "nothing to do",
    Action.FLAG.value: "flagged (no reason recorded)",
}
# Work not finished yet: what the estimated saving still stands for.
_OUTSTANDING = frozenset(
    {
        Status.PENDING,
        Status.RETRY,
        Status.QUEUED,
        Status.COPYING,
        Status.ENCODING,
        Status.MUXING,
        Status.VERIFYING,
    }
)


class PlanError(Exception):
    """A plan or report file cannot be written."""


@dataclass(frozen=True)
class PlanRow:
    path: Path
    action: str
    reasons: tuple[str, ...]
    size: int
    est_saving_bytes: int
    parts: tuple[Path, ...]
    status: Status
    # The file's subtitle status ("wanted", "found", ...), or "" when it has no subtitle record.
    subtitle: str = ""
    # The sha256 of the subtitle to mux in (FOUND only): another file is another plan.
    subtitle_sha256: str = ""


@dataclass(frozen=True)
class PlanGroup:
    """One multi-part movie: the parts that will be joined into ``output_name``."""

    output_name: str
    parts: tuple[Path, ...]
    action: str


@dataclass(frozen=True)
class PlanReport:
    rows: tuple[PlanRow, ...]
    counts: dict[str, int]
    total_bytes: int
    est_saving_bytes: int
    groups: tuple[PlanGroup, ...]
    notes: tuple[str, ...]
    # Files whose subtitle is still to be looked up (a WANTED record of a live file).
    waiting_on_subtitles: int = 0


# --- multi-part members --------------------------------------------------------------------


def group_members(records: Sequence[FileRecord]) -> set[Path]:
    """Paths of member parts that a live group record already stands for.

    A member is a later part of some record's ``parts`` whose own record is the survey's
    "part of group" left-alone placeholder. A member no group lists is not hidden: nothing
    else would explain it.
    """
    listed = {p for rec in records if rec.parts for p in rec.parts[1:]}
    return {
        rec.path
        for rec in records
        if rec.path in listed and rec.decision == Action.LEAVE_ALONE.value
    }


def _group_name(rec: FileRecord, by_path: dict[Path, FileRecord]) -> str:
    groups = find_part_groups(list(rec.parts))
    if len(groups) == 1 and len(groups[0].parts) == len(rec.parts):
        return groups[0].output_name
    for part in rec.parts[1:]:
        member = by_path.get(part)
        for reason in member.reasons if member is not None else ():
            if reason.startswith(MEMBER_REASON_PREFIX):
                return reason[len(MEMBER_REASON_PREFIX) :]
    return f"{rec.path.stem} (parts)"


# --- the plan ------------------------------------------------------------------------------


def build_plan(state: State) -> PlanReport:
    """The plan for every live record (STALE history and group members excluded)."""
    records = state.records()
    by_path = {rec.path: rec for rec in records}
    members = group_members(records)
    rows: list[PlanRow] = []
    groups: list[PlanGroup] = []
    for rec in records:
        if rec.path in members:
            continue
        size = rec.size
        if rec.parts:
            size += sum(by_path[p].size for p in rec.parts[1:] if p in by_path)
            groups.append(PlanGroup(_group_name(rec, by_path), rec.parts, rec.decision))
        reasons = tuple(r for r in rec.reasons if r.strip()) or (
            _DEFAULT_REASON.get(rec.decision, rec.decision),
        )
        sub = state.subtitle(rec.path, rec.size, rec.mtime)
        rows.append(
            PlanRow(
                rec.path,
                rec.decision,
                reasons,
                size,
                rec.est_saving_bytes,
                rec.parts,
                rec.status,
                subtitle="" if sub is None else sub.status.value,
                subtitle_sha256="" if sub is None else sub.sha256,
            )
        )

    counts = dict.fromkeys((a.value for a in Action), 0)
    for row in rows:
        counts[row.action] = counts.get(row.action, 0) + 1
    waiting = sum(1 for row in rows if row.subtitle == SubtitleStatus.WANTED.value)
    notes: list[str] = []
    if not rows:
        notes.append("no files recorded: run `survey` first")
    if waiting:
        notes.append(f"{waiting} {'file' if waiting == 1 else 'files'} waiting on subtitle lookups")
    if groups:
        notes.append(
            f"{len(groups)} multi-part group(s) will each be joined into one file: check that "
            "every group really is one movie split in parts (titles such as 'Part 1' / "
            "'Part 2' can be separate films)"
        )
    if counts[Action.FLAG.value]:
        notes.append(
            f"{counts[Action.FLAG.value]} file(s) flagged: never touched; see their reasons"
        )
    dv5 = sum(1 for row in rows if DV5_REPLACE_REASON in row.reasons)
    if dv5:
        notes.append(
            f"{dv5} file(s) are Dolby Vision profile 5 with no HDR10 fallback: left alone; "
            "replace them with another release (see the CSV)"
        )
    if rows:
        notes.append("estimated savings are survey placeholders until calibration")
    return PlanReport(
        rows=tuple(rows),
        counts=counts,
        total_bytes=sum(r.size for r in rows),
        est_saving_bytes=sum(r.est_saving_bytes for r in rows if r.status in _OUTSTANDING),
        groups=tuple(groups),
        notes=tuple(notes),
        waiting_on_subtitles=waiting,
    )


def _hash_row(row: PlanRow) -> list[object]:
    entry: list[object] = [
        row.path.as_posix(),
        row.action,
        list(row.reasons),
        row.size,
        row.est_saving_bytes,
        [p.as_posix() for p in row.parts],
    ]
    if row.subtitle:  # nothing is added for a file without a subtitle record
        entry.append([row.subtitle, row.subtitle_sha256])
    return entry


def plan_hash(report: PlanReport) -> str:
    """sha256 of the decisions in *report*: path, action, reasons, size, saving, parts and, for a
    file with a subtitle record, the subtitle's status and sha256.

    A subtitle turning FOUND, or being replaced by another one, is a new plan to acknowledge;
    the provider, reason and attempts of a subtitle record are not part of it. The run status
    and note are not part of it either, so a run that only advances or finishes files keeps the
    acknowledgement valid. The hash does change when a survey upserts a different size,
    decision, reason, saving or parts list, when a new record appears, and when a record turns
    STALE (it drops out of the live records): the runner marks a record STALE itself when it
    finds the source changed, so the next run is refused until `plan` is reviewed and
    acknowledged again. The mtime is intentionally not part of the hash: a touch that leaves
    every decision the same needs no new review.
    """
    canonical = [_hash_row(row) for row in sorted(report.rows, key=lambda r: r.path.as_posix())]
    text = json.dumps(canonical, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("ascii")).hexdigest()


# --- files ---------------------------------------------------------------------------------


@contextmanager
def _writing(path: Path) -> Iterator[None]:
    """Turn a failed write of *path* into `PlanError`; the old file is left as it was."""
    try:
        yield
    except PermissionError as exc:
        raise PlanError(
            f"cannot write {path}: {exc}; close it if it is open in another program"
        ) from exc
    except OSError as exc:
        raise PlanError(f"cannot write {path}: {exc}") from exc


def _write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    # backslashreplace: an undecodable file name still yields valid UTF-8.
    with _writing(path):
        write_csv_atomic(path, header, rows, errors="backslashreplace")


def _plan_line(row: PlanRow, group_names: dict[Path, str]) -> list[object]:
    return [
        str(row.path),
        row.action,
        row.status.value,
        row.subtitle,
        row.size,
        row.est_saving_bytes,
        group_names.get(row.path, ""),
        LIST_SEPARATOR.join(str(p) for p in row.parts),
        LIST_SEPARATOR.join(row.reasons),
    ]


def write_plan_report(report: PlanReport, path: Path) -> str:
    """Write *report* as a CSV at *path* and return its `plan_hash`. Stores nothing in State."""
    group_names = {g.parts[0]: g.output_name for g in report.groups if g.parts}
    _write_csv(path, PLAN_COLUMNS, (_plan_line(r, group_names) for r in report.rows))
    return plan_hash(report)


def write_plan_hash(path: Path, digest: str) -> None:
    with _writing(path):
        write_text_atomic(path, digest + "\n")


def read_plan_hash(path: Path) -> str | None:
    """The hash recorded for the last written plan, or None if there is none."""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return text or None


# --- status --------------------------------------------------------------------------------


def status_counts(state: State) -> tuple[dict[Status, int], int]:
    """Live records per status (every status but STALE, zeros included), and the number of
    multi-part member records left out because their group record stands for them."""
    records = state.records()
    members = group_members(records)
    counts: dict[Status, int] = {s: 0 for s in Status if s is not Status.STALE}
    for rec in records:
        if rec.path not in members:
            counts[rec.status] += 1
    return counts, len(members)


# --- results report ------------------------------------------------------------------------


@dataclass(frozen=True)
class ResultsReport:
    entries: tuple[ManifestEntry, ...]
    bytes_saved: int
    left_alone: int
    flagged: tuple[tuple[Path, str], ...]
    retry: tuple[tuple[Path, str], ...]
    # Per entry: are its originals still exactly as recorded? "yes", "no" or "unknown".
    unchanged: tuple[str, ...] = ()
    # Dolby Vision profile 5 files to replace with another release (left alone).
    replace: tuple[Path, ...] = ()


def _saving_pct(entry: ManifestEntry) -> float:
    original = sum(entry.original_sizes)
    return 0.0 if original <= 0 else 100.0 * (original - entry.completed_size) / original


def originals_unchanged(entries: Sequence[ManifestEntry], library_root: Path) -> tuple[str, ...]:
    """Per entry, whether its originals still have the recorded size and mtime (read-only).

    "yes" means the completed file may replace them; "no" means an original changed or is gone
    (the completed file is stale and must not be placed); "unknown" means it could not be
    checked (the library root is unreachable, or a stat failed other than "not found").
    """
    if not library_root.is_dir():
        return tuple(UNCHANGED_UNKNOWN for _ in entries)
    verdicts: list[str] = []
    for entry in entries:
        try:
            verdicts.append(UNCHANGED_YES if check_unchanged(entry) else UNCHANGED_NO)
        except OSError:
            verdicts.append(UNCHANGED_UNKNOWN)
    return tuple(verdicts)


def build_results(
    entries: Sequence[ManifestEntry], state: State, *, library_root: Path | None = None
) -> ResultsReport:
    """Totals of what was completed (from the manifest) and what was not (from the state).

    Multi-part member records and STALE history are not counted as left-alone files. With
    *library_root*, every entry also gets its `originals_unchanged` verdict.
    """
    records = state.records()
    members = group_members(records)
    live = [r for r in records if r.path not in members]

    def why(rec: FileRecord) -> str:
        return rec.note or LIST_SEPARATOR.join(rec.reasons) or rec.decision

    return ResultsReport(
        entries=tuple(entries),
        bytes_saved=sum(sum(e.original_sizes) - e.completed_size for e in entries),
        left_alone=sum(1 for r in live if r.status is Status.LEFT_ALONE),
        flagged=tuple((r.path, why(r)) for r in live if r.status is Status.FLAGGED),
        retry=tuple((r.path, why(r)) for r in live if r.status is Status.RETRY),
        unchanged=() if library_root is None else originals_unchanged(entries, library_root),
        replace=tuple(r.path for r in live if DV5_REPLACE_REASON in r.reasons),
    )


def _optional(value: float | int | None) -> object:
    return "" if value is None else value


def write_results_csv(results: ResultsReport, path: Path) -> None:
    """One row per completed file: sizes, saving %, VMAF, CRF, action, original unchanged."""
    verdicts = results.unchanged or tuple("" for _ in results.entries)
    _write_csv(
        path,
        REPORT_COLUMNS,
        (
            [
                LIST_SEPARATOR.join(str(p) for p in e.original),
                str(e.completed),
                sum(e.original_sizes),
                e.completed_size,
                f"{_saving_pct(e):.1f}",
                _optional(e.vmaf_mean),
                _optional(e.vmaf_worst),
                _optional(e.crf),
                e.action,
                verdict,
            ]
            for e, verdict in zip(results.entries, verdicts, strict=True)
        ),
    )
