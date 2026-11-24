"""Fakes for the subtitle resolver tests: a client, an aligner and a ready-made case.

No network, no media and no real ffmpeg/ffsubsync: the client answers from lists, the aligner
copies or fails on request, and the video is an empty file in a temporary library folder.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from helpers import MTIME, make_info
from media import RecordingReporter
from video_cleaner.config import Settings
from video_cleaner.probe import MediaInfo
from video_cleaner.progress import Reporter
from video_cleaner.state import FileRecord, Status, SubtitleRecord, SubtitleStatus
from video_cleaner.subtitles.align import AlignResult
from video_cleaner.subtitles.local import LocalSubtitle
from video_cleaner.subtitles.models import QuotaInfo
from video_cleaner.subtitles.opensubtitles import SubtitleResult
from video_cleaner.subtitles.resolve import ResolveContext, resolve_subtitle
from video_cleaner.tools import Tools

NOW = datetime(2030, 1, 1, 12, 0, tzinfo=UTC)
MOVIE_NAME = "The Great Movie (2010).mkv"
MOVIE_HASH = "0123456789abcdef"

_ENGLISH = (
    "I don't know what you are talking about, but we can get that from him.",
    "She said that they would be there when you have the time to go.",
    "What do you want me to do with all of this, and why are we here?",
    "It was not the one that I had in my hand, but it is just as good.",
    "They have been out for a while and we are not sure who it was.",
)
_GERMAN = (
    "Ich weiss nicht, wovon du sprichst, aber wir koennen das von ihm bekommen.",
    "Sie sagte, dass sie dort sein wuerden, wenn du die Zeit dafuer hast.",
    "Was soll ich mit alledem machen, und warum sind wir ueberhaupt hier?",
    "Es war nicht das, was ich in der Hand hatte, aber es ist genauso gut.",
    "Sie sind schon eine Weile weg und wir wissen nicht, wer es gewesen ist.",
)


def _stamp(seconds: float) -> str:
    ms = round(seconds * 1000)
    hours, rest = divmod(ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def srt_text(
    count: int = 30,
    *,
    tag: str = "a",
    start_s: float = 1.0,
    step_s: float = 4.0,
    german: bool = False,
) -> str:
    """An SRT of *count* cues; *tag* makes the text (and so its hash) differ between files."""
    pool = _GERMAN if german else _ENGLISH
    blocks = []
    for n in range(count):
        start = start_s + n * step_s
        blocks.append(
            f"{n + 1}\n{_stamp(start)} --> {_stamp(start + 2)}\n{pool[n % len(pool)]} {tag}\n"
        )
    return "\n".join(blocks)


def srt_bytes(count: int = 30, **kwargs: Any) -> bytes:
    return srt_text(count, **kwargs).encode("utf-8")


def result(
    file_id: int = 1001,
    *,
    title: str = "The Great Movie",
    year: int | None = 2010,
    hash_match: bool = False,
    foreign_parts_only: bool = False,
    release: str = "The.Great.Movie.2010.1080p.BluRay",
    download_count: int = 10,
) -> SubtitleResult:
    return SubtitleResult(
        file_id=file_id,
        release=release,
        title=title,
        year=year,
        download_count=download_count,
        hash_match=hash_match,
        hearing_impaired=False,
        foreign_parts_only=foreign_parts_only,
    )


class FakeClient:
    """Answers searches from lists and downloads from a mapping; records every call."""

    def __init__(
        self,
        *,
        hash_results: Sequence[SubtitleResult] = (),
        title_results: Sequence[SubtitleResult] = (),
        downloads: Mapping[int, bytes | Exception] | None = None,
        hash_error: Exception | None = None,
        title_error: Exception | None = None,
        quota: QuotaInfo | None = None,
    ) -> None:
        self.hash_results = list(hash_results)
        self.title_results = list(title_results)
        self.downloads = dict(downloads or {})
        self.hash_error = hash_error
        self.title_error = title_error
        self.quota = quota
        self.search_calls: list[dict[str, Any]] = []
        self.download_calls: list[int] = []

    def search(
        self, *, moviehash: str | None, title: str | None, year: int | None
    ) -> list[SubtitleResult]:
        self.search_calls.append({"moviehash": moviehash, "title": title, "year": year})
        if moviehash is not None:
            if self.hash_error is not None:
                raise self.hash_error
            return list(self.hash_results)
        if self.title_error is not None:
            raise self.title_error
        return list(self.title_results)

    def download(self, result: SubtitleResult, *, max_bytes: int) -> bytes:
        self.download_calls.append(result.file_id)
        item = self.downloads[result.file_id]
        if isinstance(item, Exception):
            raise item
        return item

    def last_quota(self) -> QuotaInfo | None:
        return self.quota

    @property
    def calls(self) -> int:
        return len(self.search_calls) + len(self.download_calls)


Behaviour = Callable[[Path, Path], AlignResult]


def passthrough() -> Behaviour:
    """An alignment that succeeds and writes the input as the output."""

    def run(subtitle: Path, out: Path) -> AlignResult:
        shutil.copyfile(subtitle, out)
        return AlignResult(
            True, "offset +1.000 s, framerate scale 1.000, score 500.000", 1.0, 1.0, 500.0, out
        )

    return run


def failing(reason: str = "score 0.000 is not a positive match") -> Behaviour:
    def run(subtitle: Path, out: Path) -> AlignResult:
        return AlignResult(False, reason, 1.0, 1.0, 0.0)

    return run


def writing(text: str) -> Behaviour:
    """An alignment that succeeds and writes *text* as the output."""

    def run(subtitle: Path, out: Path) -> AlignResult:
        out.write_text(text, encoding="utf-8")
        return AlignResult(
            True, "offset +1.000 s, framerate scale 1.000, score 500.000", 1.0, 1.0, 500.0, out
        )

    return run


def raising(exc: BaseException) -> Behaviour:
    def run(subtitle: Path, out: Path) -> AlignResult:
        raise exc

    return run


@dataclass(frozen=True)
class AlignCall:
    media: Path
    subtitle: Path
    out: Path
    subtitle_text: str
    hash_exact: bool


class FakeAligner:
    """Stands in for `align_subtitle`: runs the next behaviour (pass-through when none is left)."""

    def __init__(self, *behaviours: Behaviour) -> None:
        self._behaviours = list(behaviours)
        self.calls: list[AlignCall] = []

    def __call__(
        self,
        tools: Tools,
        media: Path,
        subtitle: Path,
        out: Path,
        settings: Settings,
        reporter: Reporter,
        *,
        hash_exact: bool = False,
        timeout_s: float = 0.0,
    ) -> AlignResult:
        self.calls.append(
            AlignCall(media, subtitle, out, subtitle.read_text(encoding="utf-8"), hash_exact)
        )
        behaviour = self._behaviours.pop(0) if self._behaviours else passthrough()
        return behaviour(subtitle, out)


@dataclass
class Case:
    tmp_path: Path
    settings: Settings
    rec: FileRecord
    info: MediaInfo
    sub: SubtitleRecord
    ctx: ResolveContext
    client: FakeClient | None
    aligner: FakeAligner
    movie_hash: str | None
    reporter: RecordingReporter = field(default_factory=RecordingReporter)

    @property
    def cache_dir(self) -> Path:
        return self.settings.subtitle_cache_dir

    @property
    def video(self) -> Path:
        return self.rec.path

    def cache_files(self) -> list[str]:
        if not self.cache_dir.exists():
            return []
        return sorted(p.name for p in self.cache_dir.iterdir())

    def resolve(self, sub: SubtitleRecord | None = None) -> SubtitleRecord:
        return resolve_subtitle(self.rec, self.info, self.movie_hash, sub or self.sub, self.ctx)


def make_case(
    tmp_path: Path,
    *,
    client: FakeClient | None = None,
    aligner: FakeAligner | None = None,
    local_index: Sequence[LocalSubtitle] = (),
    movie_hash: str | None = MOVIE_HASH,
    name: str = MOVIE_NAME,
    parts: tuple[Path, ...] = (),
    duration_s: float = 100.0,
    english_audio: bool = False,
    **overrides: Any,
) -> Case:
    library = tmp_path / "library"
    library.mkdir(parents=True, exist_ok=True)
    video = library / name
    video.write_bytes(b"not a real video")
    values: dict[str, Any] = {
        "library_root": library,
        "scratch_dir": tmp_path / "scratch",
        "completed_dir": tmp_path / "completed",
        "state_db": tmp_path / "state.db",
        "log_dir": tmp_path / "logs",
        "subtitle_cache_dir": tmp_path / "cache",
        "subtitle_candidates_per_file": 3,
    }
    values.update(overrides)
    settings = Settings(**values)
    size = video.stat().st_size
    rec = FileRecord(
        path=video,
        size=size,
        mtime=MTIME,
        status=Status.PENDING,
        decision="encode",
        reasons=(),
        est_saving_bytes=0,
        parts=parts,
        note="",
        movie_hash=movie_hash,
    )
    info = replace(make_info(video, size), duration_s=duration_s)
    sub = SubtitleRecord(
        path=video,
        size=size,
        mtime=MTIME,
        status=SubtitleStatus.WANTED,
        english_audio=english_audio,
    )
    fake_aligner = aligner or FakeAligner()
    reporter = RecordingReporter()
    ctx = ResolveContext(
        settings=settings,
        tools=Tools(Path("ffmpeg"), Path("ffprobe"), Path("mkvmerge")),
        reporter=reporter,
        local_index=tuple(local_index),
        client=client,
        align=fake_aligner,
        now=lambda: NOW,
    )
    return Case(tmp_path, settings, rec, info, sub, ctx, client, fake_aligner, movie_hash, reporter)
