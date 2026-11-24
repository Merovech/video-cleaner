"""The subtitle job: look up a subtitle for every file that is waiting for one.

`run_resolve_job` walks the WANTED subtitle rows in queue order and gives each one to the
resolver (`resolve.resolve_subtitle`: sidecar, local folders, OpenSubtitles). It is written for
a free OpenSubtitles account (20 downloads a day) and for hostile input, so:

- The stored allowance is checked before every file; the job stops as soon as it is spent, and
  the allowance the client reports is stored after every file, also when the file ended in a stop
  or a crash. The subtitle row is always written before the allowance, so a kill in between
  costs at most a stale (too high) remaining count, never a subtitle that was paid for and lost.
- Every outcome the resolver reports is stored, including the partial record of a stop (its
  tried ids and attempts), before the job acts on the stop.
- A FOUND row is never replaced by a weaker one without `downgrade_stale_subtitle` first
  (`persist_subtitle`), because `State.set_subtitle` silently drops such a write.
- One bad file (a probe that fails, a bug on a hostile subtitle) is logged and skipped; only a
  missing tool, an unwritable cache, a login or service that keeps failing, a spent allowance,
  the deadline, the limit and Ctrl+C end the job.
- Only the files that turned FOUND are decided again (`survey.redecide_found`); the video is
  already probed for the resolver, so nothing else is probed twice.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path

from video_cleaner.config import Settings
from video_cleaner.consolefmt import Level
from video_cleaner.logging_setup import redact
from video_cleaner.paths import name_key, name_under
from video_cleaner.probe import MediaInfo, ProbeError, probe
from video_cleaner.progress import Reporter
from video_cleaner.state import (
    IN_PROGRESS,
    FileRecord,
    State,
    StateError,
    Status,
    SubtitleRecord,
    SubtitleStatus,
)
from video_cleaner.subtitles.cache import check_cache_dir, sweep_align_workdirs
from video_cleaner.subtitles.models import QuotaInfo
from video_cleaner.subtitles.quota import next_reset, quota_usable
from video_cleaner.subtitles.resolve import (
    CacheUnwritable,
    OnlineFailure,
    QuotaStop,
    ResolveContext,
    ResolveStop,
    resolve_subtitle,
)
from video_cleaner.survey import redecide_found
from video_cleaner.tools import ToolNotFound, Tools

__all__ = [
    "DEADLINE_REACHED",
    "INTERRUPTED",
    "LIMIT_REACHED",
    "NOT_ANSWERING",
    "JobSummary",
    "is_failure",
    "persist_subtitle",
    "retry_subtitle",
    "run_resolve_job",
]

_log = logging.getLogger("video_cleaner.subtitles.job")

# Why a job stopped (`JobSummary.stopped_reason`); "" means every wanted subtitle was looked up.
DEADLINE_REACHED = "deadline reached"
LIMIT_REACHED = "file limit reached"
INTERRUPTED = "interrupted"
NOT_ANSWERING = "OpenSubtitles is not answering"
QUOTA_PREFIX = "quota exhausted until "
CACHE_PREFIX = "cache folder is not writable: "
CANNOT_START_PREFIX = "cannot start: "

#: This many failed OpenSubtitles calls in a row end the job.
MAX_CONSECUTIVE_FAILURES = 3
#: A reset time further away than this is not believed (a hostile or broken reply could lock the
#: job out for good); the next midnight UTC is assumed instead.
MAX_RESET_AHEAD = timedelta(hours=48)
_MAX_LINE_CHARS = 120
_MOVIEHASH = re.compile(r"[0-9a-f]{16}")
_SOURCES = {"sidecar": "sidecar", "local": "local", "opensubtitles": "OpenSubtitles"}
NO_SUBTITLE = "no subtitle found"

ResolveFn = Callable[
    [FileRecord, MediaInfo, str | None, SubtitleRecord, ResolveContext], SubtitleRecord
]
ProbeFn = Callable[[Path, Tools, Reporter], MediaInfo]


@dataclass(frozen=True)
class JobSummary:
    """What a job did. *skipped* counts wanted rows it could not or would not work on (no live
    file, delivered, being processed, decided by a run, changed or unreadable file, an error in
    one file); *deferred_quota* the files left waiting for the allowance to renew; UNCLEAR
    outcomes count as *unavailable*. *stopped_reason* is "" when the job ran through."""

    resolved_found: int
    unavailable: int
    deferred_quota: int
    skipped: int
    stopped_reason: str


def is_failure(reason: str) -> bool:
    """Whether a `JobSummary.stopped_reason` is a problem to fix (not the end of a normal day:
    finished, limit, deadline, spent allowance, Ctrl+C)."""
    normal = ("", LIMIT_REACHED, DEADLINE_REACHED, INTERRUPTED)
    return not (reason in normal or reason.startswith(QUOTA_PREFIX))


def persist_subtitle(
    state: State, before: SubtitleRecord, after: SubtitleRecord, reason: str = ""
) -> None:
    """Store *after*, the outcome for the row that was *before*.

    When *before* was FOUND and *after* is not, the row is downgraded first
    (`State.downgrade_stale_subtitle`): `State.set_subtitle` never replaces a FOUND row with
    WANTED, UNAVAILABLE or UNCLEAR, so without the downgrade the new tried ids would be lost.
    """
    if before.status is SubtitleStatus.FOUND and after.status is not SubtitleStatus.FOUND:
        state.downgrade_stale_subtitle(
            before.path,
            before.size,
            before.mtime,
            reason or after.reason or "the cached subtitle is gone or changed",
        )
    state.set_subtitle(after)


def _record_for(records: list[FileRecord], path: Path) -> FileRecord | None:
    """The live record whose subtitle row stands for *path*: the group record of a multi-part
    film when *path* is one of its parts, else the record of the file itself."""
    key = name_key(path)
    for rec in records:
        if len(rec.parts) > 1 and any(name_key(part) == key for part in rec.parts):
            return rec
    return next((rec for rec in records if name_key(rec.path) == key), None)


def retry_subtitle(state: State, path: Path) -> tuple[bool, str]:
    """Make the UNAVAILABLE or UNCLEAR subtitle of *path* WANTED again (forgetting what was
    tried); *path* may be a part of a multi-part film. Returns (changed, a line to show)."""
    shown = _printable(path, 300)
    rec = _record_for(state.records(), path)
    if rec is None:
        return False, f"{shown}: no surveyed file with this path: run `survey` first"
    sub = state.subtitle(rec.path, rec.size, rec.mtime)
    if sub is None:
        return False, f"{shown}: no subtitle record for this file"
    if sub.status not in (SubtitleStatus.UNAVAILABLE, SubtitleStatus.UNCLEAR):
        return False, (
            f"{shown}: the subtitle is {sub.status.value}; only unavailable or unclear ones "
            "are retried"
        )
    state.reset_subtitle(rec.path, rec.size, rec.mtime)
    return True, (
        f"{shown}: subtitle {sub.status.value} -> wanted; `subs resolve` looks it up again "
        "(a file waiting for a subtitle is skipped by `run`)"
    )


def _printable(text: object, limit: int = _MAX_LINE_CHARS) -> str:
    """*text* as one short line for the console: credentials masked, control, format and
    separator characters (terminal codes, bidi overrides, lone surrogates) written as escapes."""
    out: list[str] = []
    for ch in redact(str(text)):
        category = unicodedata.category(ch)
        out.append(ascii(ch)[1:-1] if category[0] == "C" or category in ("Zl", "Zp") else ch)
    line = "".join(out)
    return line if len(line) <= limit else line[: limit - 3] + "..."


def _format_reset(moment: datetime) -> str:
    return f"{moment:%Y-%m-%d %H:%M} UTC"


class _Job:
    """One run of `run_resolve_job`."""

    def __init__(
        self,
        settings: Settings,
        state: State,
        ctx: ResolveContext,
        *,
        limit: int | None,
        deadline: datetime | None,
        only: Path | None,
        resolve: ResolveFn,
        probe_fn: ProbeFn,
        clock: Callable[[], datetime],
    ) -> None:
        self.settings = settings
        self.state = state
        self.ctx = ctx
        self.limit = limit
        self.deadline = deadline
        self.only = only
        self.resolve = resolve
        self.probe_fn = probe_fn
        self.clock = clock
        self.reporter = ctx.reporter
        # The client's own record of the allowance; kept when the client is dropped.
        self._quota_source = ctx.client
        self._forced_quota: QuotaInfo | None = None
        self._stored_quota: QuotaInfo | None = None
        self._reported_quota: QuotaInfo | None = None
        self.found = 0
        self.unavailable = 0
        self.deferred = 0
        self.skipped = 0
        self.attempted = 0
        self.failures = 0

    # -- the loop ---------------------------------------------------------------------------

    def run(self) -> JobSummary:
        try:
            check_cache_dir(self.settings, self.settings.library_root)
        except ValueError as exc:
            _log.error("the subtitle cache folder is not usable: %s", _printable(exc, 300))
            return self._summary(CANNOT_START_PREFIX + _printable(exc, 300))
        swept = sweep_align_workdirs(self.settings.subtitle_cache_dir)
        if swept:
            _log.info("removed %d align work folder(s) left by a stopped run", swept)
        queue = self._select()
        try:
            reason = self._loop(queue)
        except KeyboardInterrupt:
            _log.warning("interrupted; what was stored stays")
            reason = INTERRUPTED
        return self._summary(reason)

    def _summary(self, reason: str) -> JobSummary:
        return JobSummary(self.found, self.unavailable, self.deferred, self.skipped, reason)

    def _loop(self, queue: list[tuple[FileRecord, SubtitleRecord]]) -> str:
        total = len(queue)
        for index, (rec, sub) in enumerate(queue, start=1):
            reason = self._stop_before(total - index + 1)
            if reason:
                return reason
            reason = self._process(f"[{index}/{total}]", total - index + 1, rec, sub)
            if reason:
                return reason
        return ""

    def _past(self, deadline: datetime) -> bool:
        # The command line gives a local wall-clock time (naive); anything aware is compared
        # with the context's UTC clock.
        if deadline.tzinfo is None or deadline.utcoffset() is None:
            return self.clock() >= deadline
        return self.ctx.now() >= deadline

    def _stop_before(self, remaining: int) -> str:
        """Why no further file may start ("" to go on). *remaining* counts the next one."""
        if self.deadline is not None and self._past(self.deadline):
            return DEADLINE_REACHED
        if self.limit is not None and self.attempted >= self.limit:
            return LIMIT_REACHED
        quota = self.state.quota()
        if (
            self.ctx.client is not None
            and quota is not None
            and not quota_usable(quota, self.ctx.now())
        ):
            self.deferred += remaining
            return QUOTA_PREFIX + _format_reset(quota.reset_at)
        return ""

    # -- selection --------------------------------------------------------------------------

    def _in_scope(self, *paths: Path) -> bool:
        return self.only is None or any(name_under(p, self.only) for p in paths)

    @staticmethod
    def _unsuitable(rec: FileRecord | None) -> str:
        """Why a wanted row is not worked on ("" if it is)."""
        if rec is None:
            return "no live file record"
        if rec.status is Status.COMPLETED:
            return "already delivered"
        if rec.status in IN_PROGRESS:
            return "being processed by a run"
        if rec.status in (Status.LEFT_ALONE, Status.FLAGGED) and rec.note:
            return "already decided by a run (`plan --retry` puts it back)"
        return ""

    def _select(self) -> list[tuple[FileRecord, SubtitleRecord]]:
        live = {(r.path, r.size, r.mtime): r for r in self.state.records()}
        queue: list[tuple[FileRecord, SubtitleRecord]] = []
        for sub in self.state.wanted_subtitles():
            rec = live.get((sub.path, sub.size, sub.mtime))
            if not self._in_scope(*((sub.path,) if rec is None else (rec.path, *rec.parts))):
                continue
            why = self._unsuitable(rec)
            if rec is None or why:
                self.skipped += 1
                _log.info("skipping the wanted subtitle of %s: %s", _printable(sub.path.name), why)
                continue
            queue.append((rec, sub))
        return queue

    # -- one file ---------------------------------------------------------------------------

    def _probe(self, tag: str, rec: FileRecord) -> MediaInfo | None:
        """The probed file, or None (counted as skipped) when it cannot be used. `ToolNotFound`
        propagates: a missing ffprobe is a setup problem, not a bad file."""
        try:
            info = self.probe_fn(rec.path, self.ctx.tools, self.reporter)
        except ProbeError as exc:
            self.skipped += 1
            self._say(tag, rec, f"cannot be probed ({_printable(exc.reason, 80)})", Level.WARN)
            return None
        if info.size != rec.size or info.mtime != rec.mtime:
            self.skipped += 1
            self._say(tag, rec, "changed since the survey: run `survey` again", Level.WARN)
            return None
        return info

    def _say(self, tag: str, rec: FileRecord, text: str, level: Level = Level.INFO) -> None:
        line = f"{tag} {_printable(rec.path.name)}: {text}"
        _log.log(logging.WARNING if level is Level.WARN else logging.INFO, "%s", line)
        self.reporter.file_done(line, level)

    def _process(self, tag: str, remaining: int, rec: FileRecord, sub: SubtitleRecord) -> str:
        """Resolve one file and store the result; the reason to stop the job, or ""."""
        self.reporter.stage(f"{tag} {_printable(rec.path.name)}")
        info = self._probe(tag, rec)
        if info is None:
            return ""
        # Another command may have changed the row since the listing: work on what is stored.
        current = self.state.subtitle(rec.path, rec.size, rec.mtime)
        if current is None or current.status not in (SubtitleStatus.WANTED, SubtitleStatus.FOUND):
            self.skipped += 1
            return ""
        self.attempted += 1
        movie_hash = rec.movie_hash if _valid_hash(rec.movie_hash) else None
        try:
            try:
                outcome = self.resolve(rec, info, movie_hash, current, self.ctx)
            except ResolveStop as stop:
                return self._stopped(tag, remaining, rec, current, stop)
            except ToolNotFound:
                raise  # a missing ffsubsync or ffprobe is a setup problem: end the job
            except Exception as exc:  # one hostile file must never end the job
                self.skipped += 1
                _log.error("%s: unexpected %s; skipped", tag, type(exc).__name__)
                _log.debug("details", exc_info=True)
                self._say(tag, rec, f"skipped after an error ({type(exc).__name__})", Level.WARN)
                return ""
            self._finished(tag, rec, info, current, outcome)
            return ""
        finally:
            self._store_quota()

    def _finished(
        self,
        tag: str,
        rec: FileRecord,
        info: MediaInfo,
        current: SubtitleRecord,
        outcome: SubtitleRecord,
    ) -> None:
        persist_subtitle(self.state, current, outcome)
        plan = ""
        if outcome.status is SubtitleStatus.FOUND:
            self.found += 1
            plan = self._redecide(rec, info)
        elif outcome.status in (SubtitleStatus.UNAVAILABLE, SubtitleStatus.UNCLEAR):
            self.unavailable += 1
        if outcome.provider == "opensubtitles" or outcome.status is SubtitleStatus.UNAVAILABLE:
            self.failures = 0  # the server answered
        self._say(tag, rec, self._describe(outcome, plan), self._level(outcome))

    @staticmethod
    def _level(outcome: SubtitleRecord) -> Level:
        return Level.OK if outcome.status is SubtitleStatus.FOUND else Level.INFO

    def _describe(self, outcome: SubtitleRecord, plan: str) -> str:
        if outcome.status is SubtitleStatus.FOUND:
            source = _SOURCES.get(outcome.provider, "unknown source")
            text = f"{source} -> found"
        else:
            reason = f" ({_printable(outcome.reason, 100)})" if outcome.reason else ""
            text = f"{NO_SUBTITLE} -> {outcome.status.value}{reason}"
        if plan:
            text += f" ({plan})"
        left = self._allowance_note()
        return f"{text}; {left}" if left else text

    def _allowance_note(self) -> str:
        """ "OpenSubtitles: N downloads left" when a download changed the allowance, else ""."""
        quota = None if self._quota_source is None else self._quota_source.last_quota()
        if quota is None or quota == self._reported_quota:
            return ""
        self._reported_quota = quota
        return f"OpenSubtitles: {quota.remaining} downloads left"

    def _redecide(self, rec: FileRecord, info: MediaInfo) -> str:
        """Decide the file again now that it has a subtitle; a note for the progress line."""
        if len(rec.parts) > 1:
            return "multi-part group: plan unchanged"
        try:
            redecide_found(info, self.settings, self.state, movie_hash=rec.movie_hash)
        except StateError as exc:
            _log.warning("%s: the file could not be decided again: %s", rec.path.name, exc)
            return "plan unchanged"
        again = self.state.get(rec.path)
        if again is None:
            return ""
        return f"plan: {again.decision}, {again.status.value}"

    # -- stops ------------------------------------------------------------------------------

    def _stopped(
        self,
        tag: str,
        remaining: int,
        rec: FileRecord,
        current: SubtitleRecord,
        stop: ResolveStop,
    ) -> str:
        """Store the partial record of *stop*, then act on it; the reason to end the job."""
        persist_subtitle(self.state, current, stop.record, stop.reason)
        if isinstance(stop, QuotaStop):
            reset = self._trusted_reset(stop.reset_at)
            self._forced_quota = QuotaInfo(0, reset)
            self.deferred += remaining
            self._say(
                tag, rec, "stopped: the OpenSubtitles download allowance is used up", Level.WARN
            )
            return QUOTA_PREFIX + _format_reset(reset)
        if isinstance(stop, CacheUnwritable):
            self._say(tag, rec, f"stopped: {_printable(stop.reason)}", Level.WARN)
            return CACHE_PREFIX + _printable(stop.reason, 300)
        if isinstance(stop, OnlineFailure):
            return self._online_failure(tag, rec, stop)
        self._say(tag, rec, f"stopped: {_printable(stop.reason)}", Level.WARN)
        return ""

    def _online_failure(self, tag: str, rec: FileRecord, stop: OnlineFailure) -> str:
        if stop.fatal:
            self.ctx = replace(self.ctx, client=None)
            _log.warning(
                "OpenSubtitles refused the login: online lookups are off for the rest of this "
                "job (sidecar and local files are still used)"
            )
            self._say(tag, rec, f"{_printable(stop.reason)}", Level.WARN)
            return ""
        self.failures += 1
        self._say(tag, rec, f"OpenSubtitles failed ({_printable(stop.reason, 80)})", Level.WARN)
        if self.failures >= MAX_CONSECUTIVE_FAILURES:
            return NOT_ANSWERING
        return ""

    # -- the allowance ----------------------------------------------------------------------

    def _trusted_reset(self, reset_at: datetime | None) -> datetime:
        """The reset time of a spent allowance: *reset_at* if it lies in the near future, else
        the next midnight UTC (the free account's allowance renews daily)."""
        now = self.ctx.now()
        if reset_at is None or reset_at <= now:
            return next_reset(now)
        return self._near(reset_at)

    def _near(self, reset_at: datetime) -> datetime:
        """*reset_at*, unless it is too far away to be believed (then the next midnight UTC)."""
        now = self.ctx.now()
        return next_reset(now) if reset_at > now + MAX_RESET_AHEAD else reset_at

    def _store_quota(self) -> None:
        """Store the allowance (a stop's, else the client's last report) unless it is unchanged."""
        quota = self._forced_quota
        self._forced_quota = None
        if quota is None and self._quota_source is not None:
            quota = self._quota_source.last_quota()
            if quota is not None:
                quota = replace(quota, reset_at=self._near(quota.reset_at))
        if quota is None or quota == self._stored_quota:
            return
        self.state.set_quota(quota)
        self._stored_quota = quota


def _valid_hash(value: str | None) -> bool:
    return value is not None and _MOVIEHASH.fullmatch(value) is not None


def run_resolve_job(
    settings: Settings,
    state: State,
    ctx: ResolveContext,
    *,
    limit: int | None,
    deadline: datetime | None,
    only: Path | None,
    resolve: ResolveFn = resolve_subtitle,
    probe_fn: ProbeFn = probe,
    clock: Callable[[], datetime] = datetime.now,
) -> JobSummary:
    """Look up the subtitle of every WANTED row, quota-safe; see the module docstring.

    *limit* is the most files to attempt, *deadline* the time after which no new file starts
    (a naive datetime is local wall-clock time, compared with *clock*; an aware one with
    ``ctx.now()``), *only* a file, a multi-part group's part or a folder to restrict the job to.
    *resolve*, *probe_fn* and *clock* are replaceable for tests. The job never writes to the
    library. `ToolNotFound` (ffprobe or ffsubsync missing) propagates after what was done is
    stored; Ctrl+C ends the job with ``stopped_reason == INTERRUPTED``.
    """
    return _Job(
        settings,
        state,
        ctx,
        limit=limit,
        deadline=deadline,
        only=only,
        resolve=resolve,
        probe_fn=probe_fn,
        clock=clock,
    ).run()
