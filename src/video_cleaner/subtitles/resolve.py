"""Resolving one subtitle for one file: sidecar, then local folders, then OpenSubtitles.

`resolve_subtitle` takes a file's `SubtitleRecord` and returns the new one; it does not touch
the state database (the caller stores the record). Every candidate, wherever it comes from,
goes through the same gauntlet: it is normalised to plain cues, its text must look English,
it is aligned to the video's audio (skipped only for a hash match, which says the subtitle was
made for this very file), and the result is read back and checked again (still English, at
least 95% of the cues, not longer than the video can hold). The text that is cached is always
our own normalised SRT of the accepted result, never a file as it was downloaded or as
ffsubsync wrote it. A wrong subtitle is worse than none, so every doubt is a rejection.

OpenSubtitles downloads are the scarce thing (the allowance is counted on the download
request), so an online candidate is handled in a quota-safe order: a finished cache file or the
download of a killed run is reused before anything is requested, and the download is written to
disk before the slow alignment starts. Nothing a server or a subtitle file says reaches a path,
a command line or a log unescaped: cache names are built from a hash and a number only.

A download is spent once per file and never twice: the ids that were downloaded and rejected are
remembered, and `subtitle_candidates_per_file` bounds the downloads of the file in total, across
calls (the record's `attempts` counts them). A download kept on disk is reused for free: a
candidate rejected for a reason a setting can undo (the alignment score, the length, the cue
count) keeps its normalised download, so that after `subs retry` and a looser setting it is
checked again without another request; one rejected for its content (not English, not a
subtitle) does not. Adopting a kept download or a finished cache file never counts.

Failure surface of `resolve_subtitle`:

- `QuotaStop`: the allowance is spent.
- `OnlineFailure`: a login or service failure; the record stays WANTED. The id of a download
  that failed after the server may have counted it is remembered; one that provably never
  reached the server (`DownloadNotCounted`) is not.
- `CacheUnwritable`: the cache folder cannot be written (before a download, or when the
  download cannot be kept), or any disk error after a download (the aligner's folder, its
  output, the result): nothing more may be downloaded, the download stays on disk and its id is
  not blamed.
- `ToolNotFound` from the aligner.
- `ValueError` for a bad argument or a cache folder that overlaps the library (bugs).

A problem with the candidate itself (a rejected file, a sidecar that cannot be read) is a
rejection of that candidate, not an exception.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, TypeVar

from video_cleaner.atomicio import read_bytes_capped, write_text_atomic
from video_cleaner.config import Settings
from video_cleaner.logging_setup import redact
from video_cleaner.probe import MediaInfo
from video_cleaner.progress import Reporter
from video_cleaner.state import FileRecord, SubtitleRecord, SubtitleStatus
from video_cleaner.subtitles.align import AlignResult, align_subtitle
from video_cleaner.subtitles.cache import (
    CacheNames,
    cache_key,
    cache_names,
    cached_result_is_valid,
    check_cache_dir,
    discard,
    probe_writable,
    prune_kept_downloads,
    reusable_ids,
    sidecar_candidates,
    sweep_align_workdirs,
)
from video_cleaner.subtitles.language import looks_english
from video_cleaner.subtitles.local import (
    LocalSubtitle,
    match_local,
    title_year_confidence,
)
from video_cleaner.subtitles.models import QuotaInfo
from video_cleaner.subtitles.names import TitleYear, normalise_title, parse_library_name
from video_cleaner.subtitles.opensubtitles import (
    AuthError,
    DownloadNotCounted,
    ProviderError,
    QuotaExhausted,
    SubtitleResult,
)
from video_cleaner.subtitles.quota import quota_usable
from video_cleaner.subtitles.textfmt import NormalisedSubtitle, SubtitleRejected, normalise_subtitle
from video_cleaner.tools import Tools

__all__ = [
    "CacheUnwritable",
    "OnlineFailure",
    "QuotaStop",
    "ResolveContext",
    "ResolveStop",
    "SubtitleClient",
    "resolve_subtitle",
    "sidecar_candidates",
    "sweep_align_workdirs",
]

_log = logging.getLogger("video_cleaner.subtitles.resolve")
_T = TypeVar("_T")

#: The most ids kept in a record's `tried_ids` (the state database reads back a longer list as
#: empty), keeping the latest.
MAX_TRIED_IDS = 50
_MAX_FILE_ID = 2**63 - 1
#: An aligned subtitle must keep at least this share of the cues it was given.
MIN_CUES_KEPT_PERCENT = 95
#: A subtitle may end this much (factor, plus seconds) after the video does, for credits and
#: for a cut that differs a little; a hash match on the wrong cut is caught by this alone.
DURATION_FACTOR = 1.10
DURATION_SLACK_S = 120.0
MAX_REASON_CHARS = 300
_SHORT_REASON_CHARS = 70
_PROVENANCE = {
    "sidecar": "English (sidecar)",
    "local": "English (local file)",
    "opensubtitles": "English (OpenSubtitles)",
}
NO_CREDENTIALS = "unavailable: no OpenSubtitles credentials"
#: What every reason the resolver gives a multi-part film starts with; a survey never resets such
#: a row (it tells it from its own UNCLEAR rows by this).
MULTI_PART_PREFIX = "multi-part film"
MULTI_PART = f"{MULTI_PART_PREFIX}: subtitle lookup needs the parts combined (not supported yet)"


class SubtitleClient(Protocol):
    """What the resolver uses of `OpenSubtitlesClient` (a fake can stand in for it)."""

    def search(
        self, *, moviehash: str | None, title: str | None, year: int | None
    ) -> list[SubtitleResult]: ...

    def download(self, result: SubtitleResult, *, max_bytes: int) -> bytes: ...

    def last_quota(self) -> QuotaInfo | None: ...


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class ResolveContext:
    """Everything `resolve_subtitle` needs besides the file: settings, tools, the console,
    the indexed local subtitle folders, the OpenSubtitles client (None when there are no
    credentials: the online step is skipped), the aligner and the clock (both replaceable)."""

    settings: Settings
    tools: Tools
    reporter: Reporter
    local_index: Sequence[LocalSubtitle]
    client: SubtitleClient | None
    align: Callable[..., AlignResult] = align_subtitle
    now: Callable[[], datetime] = field(default=_utc_now)


class ResolveStop(Exception):
    """The resolution stopped before an outcome. *record* is the partial `SubtitleRecord`
    (still WANTED, with the `tried_ids` and `attempts` so far) and *reason* says why."""

    def __init__(self, record: SubtitleRecord, reason: str) -> None:
        super().__init__(reason)
        self.record = record
        self.reason = reason


class QuotaStop(ResolveStop):
    """The OpenSubtitles download allowance is spent; *reset_at* is when it renews, if known."""

    def __init__(self, record: SubtitleRecord, reset_at: datetime | None) -> None:
        when = f"; it renews at {reset_at:%Y-%m-%d %H:%M} UTC" if reset_at else ""
        super().__init__(record, f"the OpenSubtitles download allowance is used up{when}")
        self.reset_at = reset_at


class CacheUnwritable(ResolveStop):
    """The cache folder cannot be written (disk full, no permission). The job should stop
    resolving subtitles: every further download would be spent for nothing."""


class OnlineFailure(ResolveStop):
    """OpenSubtitles failed. *fatal* is a refused login: the job must stop using the client."""

    def __init__(self, record: SubtitleRecord, reason: str, fatal: bool) -> None:
        super().__init__(record, reason)
        self.fatal = fatal


def _safe(value: object, limit: int = _SHORT_REASON_CHARS) -> str:
    """*value* as one short line of printable ASCII with credentials masked, for logs and
    reasons: the server's and the subtitle's text is never trusted to be tidy."""
    text = ascii(redact(str(value)))[1:-1]
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _bounded_ids(ids: Iterable[int]) -> list[int]:
    """*ids* without repeats, in order of last use, at most `MAX_TRIED_IDS` of the latest."""
    last = {item: position for position, item in enumerate(ids)}
    return sorted(last, key=lambda item: last[item])[-MAX_TRIED_IDS:]


class _PerFileLimit(Exception):
    """The file's downloads have reached `subtitle_candidates_per_file` (internal)."""


@dataclass(frozen=True)
class _Vetted:
    """The outcome of vetting a candidate: its accepted cues, or why it was rejected.

    *keep* says a rejection is one a setting can undo (alignment score, length, cue count), so
    that a download of it is kept for a later retry; a content rejection is not kept.
    """

    cues: NormalisedSubtitle | None
    detail: str
    keep: bool = False


class _Resolution:
    """One run of `resolve_subtitle` for one file."""

    def __init__(
        self,
        rec: FileRecord,
        info: MediaInfo,
        movie_hash: str | None,
        sub: SubtitleRecord,
        ctx: ResolveContext,
        cache_dir: Path,
    ) -> None:
        self.rec = rec
        self.info = info
        self.movie_hash = movie_hash
        self.ctx = ctx
        self.settings = ctx.settings
        self.cache_dir = cache_dir
        self.key = cache_key(sub)
        self.label = _safe(rec.path.name, 80)
        self.base = replace(
            sub,
            status=SubtitleStatus.WANTED,
            provider="",
            cache_path=None,
            sha256="",
            provenance="",
            hash_exact=False,
            reason="",
        )
        self.tried = _bounded_ids(sub.tried_ids)
        # The downloads of this file so far, across calls: the per-file cap counts them all.
        self.attempts = sub.attempts
        self.online_tried = 0
        self._probed = False
        self.notes: dict[str, list[str]] = {"sidecar": [], "local": [], "online": []}
        self._target: TitleYear | None = None

    def run(self) -> SubtitleRecord:
        if len(self.rec.parts) > 1:
            _log.info("%s: a multi-part film; no subtitle lookup", self.label)
            return self._record(SubtitleStatus.UNCLEAR, MULTI_PART)
        for path in sidecar_candidates(self.rec.path):
            found = self._file_candidate(path, "sidecar")
            if found is not None:
                return found
        local = match_local(self.ctx.local_index, self.target)
        if local is not None:
            _log.info("%s: a local file matches (confidence %.2f)", self.label, local.confidence)
            found = self._file_candidate(local.path, "local")
            if found is not None:
                return found
        client = self.ctx.client
        if client is None:
            _log.info("%s: no OpenSubtitles client; the online step is skipped", self.label)
            return self._record(SubtitleStatus.WANTED, NO_CREDENTIALS)
        return self._online(client)

    @property
    def target(self) -> TitleYear:
        if self._target is None:
            self._target = parse_library_name(self.rec.path)
        return self._target

    def _record(self, status: SubtitleStatus, reason: str) -> SubtitleRecord:
        return replace(
            self.base,
            status=status,
            reason=reason,
            tried_ids=tuple(self.tried),
            attempts=self.attempts,
        )

    def _note(self, source: str, reason: str) -> None:
        self.notes[source].append(_safe(reason))

    def _names(self, provider: str, ident: str) -> CacheNames:
        return cache_names(self.cache_dir, self.key, provider, ident)

    def _found(
        self, provider: str, names: CacheNames, data: bytes, hash_exact: bool, detail: str
    ) -> SubtitleRecord:
        _log.info("%s: accepted the %s subtitle (%s)", self.label, provider, _safe(detail))
        return replace(
            self._record(SubtitleStatus.FOUND, _safe(detail, 200)),
            provider=provider,
            cache_path=names.final,
            sha256=hashlib.sha256(data).hexdigest(),
            provenance=_PROVENANCE[provider],
            hash_exact=hash_exact,
        )

    def _normalise(self, raw: bytes) -> NormalisedSubtitle:
        """Normalise *raw*; `SubtitleRejected` unless the result is usable and writable."""
        norm = normalise_subtitle(
            raw,
            max_bytes=self.settings.subtitle_max_bytes,
            min_cues=self.settings.subtitle_min_cues,
        )
        try:
            size = len(norm.srt.encode("utf-8"))
        except UnicodeEncodeError:
            raise SubtitleRejected("subtitle text cannot be written as UTF-8") from None
        if size > self.settings.subtitle_max_bytes:
            raise SubtitleRejected(
                f"normalised subtitle is larger than {self.settings.subtitle_max_bytes} bytes"
            )
        return norm

    def _length_problem(self, norm: NormalisedSubtitle) -> str:
        """Why the subtitle cannot belong to this video by its length, or "" if it can."""
        duration = self.info.duration_s
        if not (math.isfinite(duration) and duration > 0):
            return "the video's length is unknown, so the subtitle's length cannot be checked"
        limit_s = duration * DURATION_FACTOR + DURATION_SLACK_S
        end_s = max((cue.end_ms for cue in norm.cues), default=0) / 1000.0
        if end_s > limit_s:
            return f"the subtitle runs to {end_s:.0f} s but the video allows {limit_s:.0f} s"
        return ""

    def _reread(self, path: Path) -> tuple[NormalisedSubtitle | None, str]:
        """The aligner's output, read back and normalised again, or (None, why not)."""
        try:
            raw = read_bytes_capped(path, self.settings.subtitle_max_bytes)
        except FileNotFoundError:
            return None, "no output file"
        if raw is None:
            return None, f"larger than {self.settings.subtitle_max_bytes} bytes"
        try:
            norm = self._normalise(raw)
        except SubtitleRejected as exc:
            return None, str(exc)
        if not looks_english(norm.cues):
            return None, "does not look English"
        return norm, ""

    def _vet(self, norm: NormalisedSubtitle, names: CacheNames, hash_exact: bool) -> _Vetted:
        """Language, alignment and the checks on its result. An `OSError` and `ToolNotFound`
        propagate."""
        if not looks_english(norm.cues):
            return _Vetted(None, "does not look English")
        if hash_exact:
            final, detail = norm, "matched by hash; not aligned"
        else:
            if not names.staged.exists():
                self._cache_write(names.staged, norm.srt, fsync=True)
            names.aligned.unlink(missing_ok=True)
            aligned = self.ctx.align(
                self.ctx.tools,
                self.info.path,
                names.staged,
                names.aligned,
                self.settings,
                self.ctx.reporter,
                hash_exact=False,
            )
            _log.info(
                "%s: alignment %s: %s",
                self.label,
                "accepted" if aligned.ok else "rejected",
                _safe(aligned.reason, 200),
            )
            if not aligned.ok or aligned.output is None:
                return _Vetted(None, f"alignment failed: {aligned.reason}", keep=True)
            reread, why = self._reread(aligned.output)
            if reread is None:
                return _Vetted(None, f"aligned output rejected: {why}")
            final = reread
            if len(final.cues) * 100 < len(norm.cues) * MIN_CUES_KEPT_PERCENT:
                return _Vetted(
                    None, f"alignment kept only {len(final.cues)} of {len(norm.cues)} cues", True
                )
            detail = aligned.reason
        problem = self._length_problem(final)
        return _Vetted(None, problem, keep=True) if problem else _Vetted(final, detail)

    def _adopt_final(
        self, provider: str, names: CacheNames, hash_exact: bool
    ) -> SubtitleRecord | None:
        """The record for a finished cache file of this candidate, if it is still our own
        normalised output and still fits the video; None otherwise (it will be overwritten)."""
        try:
            data = read_bytes_capped(names.final, self.settings.subtitle_max_bytes)
        except OSError:
            return None
        if data is None:
            return None
        try:
            norm = self._normalise(data)
        except SubtitleRejected:
            return None
        if norm.srt.encode("utf-8") != data or not looks_english(norm.cues):
            return None
        if self._length_problem(norm):
            return None
        return self._found(provider, names, data, hash_exact, "reused an earlier result")

    def _accept(
        self,
        provider: str,
        names: CacheNames,
        final: NormalisedSubtitle,
        hash_exact: bool,
        detail: str,
    ) -> SubtitleRecord:
        self._cache_write(names.final, final.srt)
        return self._found(provider, names, final.srt.encode("utf-8"), hash_exact, detail)

    def _file_candidate(self, path: Path, provider: str) -> SubtitleRecord | None:
        self.ctx.reporter.stage(f"subtitle: checking {provider} file {_safe(path.name, 60)}")
        names: CacheNames | None = None
        try:
            raw = read_bytes_capped(path, self.settings.subtitle_max_bytes)
            if raw is None:
                self._reject_file(provider, "larger than the size limit")
                return None
            norm = self._normalise(raw)
            ident = hashlib.sha256(norm.srt.encode("utf-8")).hexdigest()[:12]
            names = self._names(provider, ident)
            adopted = self._adopt_final(provider, names, False)
            if adopted is not None:
                return adopted
            vetted = self._vet(norm, names, False)
            if vetted.cues is None:
                self._reject_file(provider, vetted.detail)
                return None
            return self._accept(provider, names, vetted.cues, False, vetted.detail)
        except SubtitleRejected as exc:
            self._reject_file(provider, str(exc))
            return None
        except OSError as exc:
            self._reject_file(provider, f"file error ({type(exc).__name__})")
            return None
        finally:
            if names is not None:
                discard(names.staged, names.aligned)

    def _reject_file(self, provider: str, reason: str) -> None:
        _log.info("%s: rejected the %s file: %s", self.label, provider, _safe(reason))
        self._note(provider, reason)

    def _cache_write(self, path: Path, text: str, *, fsync: bool = False) -> None:
        """Write a cache file; a failure stops the resolution (`CacheUnwritable`)."""
        try:
            write_text_atomic(path, text, fsync=fsync)
        except OSError as exc:
            raise self._unwritable(exc) from None

    def _unwritable(self, exc: OSError) -> CacheUnwritable:
        reason = f"the subtitle cache folder {_safe(self.cache_dir, 150)} cannot be written"
        _log.warning("%s: %s (%s)", self.label, reason, type(exc).__name__)
        return CacheUnwritable(
            self._record(SubtitleStatus.WANTED, f"{reason} ({type(exc).__name__})"),
            f"{reason} ({type(exc).__name__})",
        )

    def _remember(self, file_id: int) -> None:
        """Remember that *file_id* was downloaded (and so possibly counted) and is not wanted
        again for this file."""
        self.tried = _bounded_ids([*self.tried, file_id])

    def _count_download(self) -> None:
        """Count one request to the download endpoint against the file's cap."""
        self.attempts += 1

    def _call(self, call: Callable[[], _T], spent: int | None = None) -> _T:
        """Run a client call, turning its errors into the resolver's stops. *spent* is the id
        of a download: if it fails in a way that the server may have counted, the id is blamed
        (and the download counts); a failure that never reached the server (`DownloadNotCounted`)
        blames nothing."""
        try:
            return call()
        except QuotaExhausted as exc:
            record = self._record(SubtitleStatus.WANTED, "OpenSubtitles quota used up")
            raise QuotaStop(record, exc.reset_at) from None
        except AuthError as exc:
            reason = f"OpenSubtitles refused the login: {_safe(exc, 150)}"
            raise OnlineFailure(self._record(SubtitleStatus.WANTED, reason), reason, True) from None
        except ProviderError as exc:
            if spent is not None and not isinstance(exc, DownloadNotCounted):
                self._remember(spent)
                self._count_download()
            reason = f"OpenSubtitles failed: {_safe(exc, 150)}"
            raise OnlineFailure(
                self._record(SubtitleStatus.WANTED, reason), reason, False
            ) from None

    def _capped(self, seen: set[int]) -> bool:
        """Whether the file's downloads are used up and no kept or finished file of an id not
        yet seen could still be adopted for free: a search could not be acted on."""
        if self.attempts < self.settings.subtitle_candidates_per_file:
            return False
        return not (reusable_ids(self.cache_dir, self.key, "opensubtitles") - seen)

    def _candidates(
        self, client: SubtitleClient, seen: set[int]
    ) -> Iterator[tuple[SubtitleResult, bool]]:
        """The results to try in order, each with whether it came from the hash search. The
        title search only runs when the hash search's results are used up."""
        if self.movie_hash is not None and not self._capped(seen):
            digest = self.movie_hash
            _log.info("%s: searching OpenSubtitles by file hash", self.label)
            for item in self._call(lambda: client.search(moviehash=digest, title=None, year=None)):
                yield item, True
        target = self.target
        if self._capped(seen):
            return  # the downloads are used up: a second search could not be acted on
        if not target.is_episode and normalise_title(target.title):
            _log.info("%s: searching OpenSubtitles by title", self.label)
            for item in self._call(
                lambda: client.search(moviehash=None, title=target.title, year=target.year)
            ):
                yield item, False

    def _drop_reason(self, item: SubtitleResult, from_hash: bool, seen: set[int]) -> str:
        """Why *item* is not downloaded (it costs nothing and is not remembered), or ""."""
        if not 0 <= item.file_id <= _MAX_FILE_ID:
            return "unusable id"
        if item.file_id in seen:
            return "already tried"
        if item.foreign_parts_only:
            return "foreign parts only"
        if not (from_hash and item.hash_match):
            if self.target.is_episode:
                return "an episode is only trusted on a hash match"
            as_found = TitleYear(item.title, item.year, False, ())
            if title_year_confidence(as_found, self.target) is None:
                return "title or year does not match the file"
        return ""

    def _online(self, client: SubtitleClient) -> SubtitleRecord:
        seen = set(self.tried)
        for item, from_hash in self._candidates(client, seen):
            problem = self._drop_reason(item, from_hash, seen)
            if problem:
                _log.debug(
                    "%s: skipped OpenSubtitles file %d (%s): release %r",
                    self.label,
                    item.file_id,
                    problem,
                    _safe(item.release, 100),
                )
                continue
            seen.add(item.file_id)
            self.online_tried += 1
            try:
                found = self._online_candidate(client, item, from_hash and item.hash_match)
            except _PerFileLimit:
                self.online_tried -= 1
                break
            except OSError as exc:
                _log.warning("%s: a file error stopped one candidate: %s", self.label, _safe(exc))
                self._note("online", f"file error ({type(exc).__name__})")
                continue
            if found is not None:
                return found
        return self._record(SubtitleStatus.UNAVAILABLE, self._unavailable_reason())

    def _download(self, client: SubtitleClient, item: SubtitleResult) -> bytes:
        if self.attempts >= self.settings.subtitle_candidates_per_file:
            raise _PerFileLimit
        quota = client.last_quota()
        if quota is not None and not quota_usable(quota, self.ctx.now()):
            raise QuotaStop(
                self._record(SubtitleStatus.WANTED, "OpenSubtitles quota used up"), quota.reset_at
            )
        if not self._probed:
            try:
                probe_writable(self.cache_dir)
            except OSError as exc:
                raise self._unwritable(exc) from None
            self._probed = True
        data = self._call(
            lambda: client.download(item, max_bytes=self.settings.subtitle_max_bytes),
            spent=item.file_id,
        )
        self._count_download()
        return data

    def _online_candidate(
        self, client: SubtitleClient, item: SubtitleResult, hash_exact: bool
    ) -> SubtitleRecord | None:
        names = self._names("opensubtitles", str(item.file_id))
        _log.info(
            "%s: OpenSubtitles file %d (%s)",
            self.label,
            item.file_id,
            "hash match" if hash_exact else "title match",
        )
        self.ctx.reporter.stage(f"subtitle: trying OpenSubtitles file {item.file_id}")
        adopted = self._adopt_final("opensubtitles", names, hash_exact)
        if adopted is not None:
            return adopted
        raw: bytes | None
        try:
            raw = read_bytes_capped(names.staged, self.settings.subtitle_max_bytes)
        except FileNotFoundError:
            raw = None
        else:
            if raw is None:
                self._reject_online(item.file_id, names, "kept download is too large", keep=False)
                return None
            _log.info("%s: reusing the download a stopped run kept (no request)", self.label)
        downloaded = raw is None
        if raw is None:
            raw = self._download(client, item)
        try:
            norm = self._normalise(raw)
        except SubtitleRejected as exc:
            self._reject_online(item.file_id, names, str(exc), keep=False)
            return None
        if downloaded:
            try:
                write_text_atomic(names.staged, norm.srt, fsync=True)
            except OSError as exc:
                # The download is spent and cannot be resumed, so it must not be spent again.
                self._remember(item.file_id)
                raise self._unwritable(exc) from None
        try:
            vetted = self._vet(norm, names, hash_exact)
            if vetted.cues is None:
                self._reject_online(item.file_id, names, vetted.detail, keep=vetted.keep)
                return None
            record = self._accept("opensubtitles", names, vetted.cues, hash_exact, vetted.detail)
        except OSError as exc:
            # A disk error after a download: the download stays on disk, the id is not blamed,
            # and nothing more is downloaded in this call.
            raise self._unwritable(exc) from None
        finally:
            discard(names.aligned)
        discard(names.staged)
        return record

    def _reject_online(self, file_id: int, names: CacheNames, reason: str, *, keep: bool) -> None:
        """Remember *file_id* as rejected. Its download is kept for a retry when *keep* (the
        reason is one a setting can undo) and the file's kept downloads stay within the cap."""
        self._remember(file_id)
        self._note("online", reason)
        kept = keep and names.staged.exists()
        if not kept:
            discard(names.staged)
        discard(names.aligned)
        if kept:
            prune_kept_downloads(
                self.cache_dir,
                self.key,
                "opensubtitles",
                keep=self.settings.subtitle_candidates_per_file,
                protect=names.staged,
            )
        _log.info(
            "%s: rejected OpenSubtitles file %d: %s%s",
            self.label,
            file_id,
            _safe(reason),
            " (download kept)" if kept else "",
        )

    def _unavailable_reason(self) -> str:
        def part(name: str) -> str:
            reasons = self.notes[name]
            return f"{name} rejected ({'; '.join(reasons)})" if reasons else f"{name} none"

        online = f"OpenSubtitles tried {self.online_tried}"
        if self.notes["online"]:
            online += f" (rejected: {'; '.join(self.notes['online'])})"
        text = f"no English subtitle found: {part('sidecar')}, {part('local')}, {online}"
        limit = self.settings.subtitle_candidates_per_file
        if self.attempts >= limit:
            text = f"per-file download limit ({limit}) reached (`subs retry` resets it); {text}"
        if len(text) > MAX_REASON_CHARS:
            text = text[: MAX_REASON_CHARS - 3] + "..."
        _log.info("%s: %s", self.label, text)
        return text


def resolve_subtitle(
    rec: FileRecord,
    info: MediaInfo,
    movie_hash: str | None,
    sub: SubtitleRecord,
    ctx: ResolveContext,
) -> SubtitleRecord:
    """The new subtitle record for *rec*, found in the order sidecar, local, OpenSubtitles.

    A record that is not WANTED is returned unchanged; a FOUND one whose cache file is still
    there and unchanged too (a missing or changed file makes it WANTED again, keeping its
    `tried_ids`). Nothing accepted gives UNAVAILABLE with the reasons, except without a client
    (no credentials), which leaves it WANTED so that adding credentials later retries. A
    multi-part film is UNCLEAR. See the module docstring for the exceptions.
    """
    if sub.status not in (SubtitleStatus.WANTED, SubtitleStatus.FOUND):
        return sub
    cache_dir = check_cache_dir(ctx.settings, rec.path)
    if sub.status is SubtitleStatus.FOUND:
        if cached_result_is_valid(sub, cache_dir, ctx.settings.subtitle_max_bytes):
            return sub
        _log.warning(
            "%s: the cached subtitle is gone or changed; resolving again", _safe(rec.path.name)
        )
    return _Resolution(rec, info, movie_hash, sub, ctx, cache_dir).run()
