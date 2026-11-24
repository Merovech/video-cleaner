"""The resolver's OpenSubtitles step: candidates, filters, quota, kill-safety and failures."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from resolve_fakes import (
    NOW,
    Behaviour,
    Case,
    FakeAligner,
    FakeClient,
    failing,
    make_case,
    passthrough,
    raising,
    result,
    srt_bytes,
    srt_text,
    writing,
)
from video_cleaner.atomicio import write_text_atomic
from video_cleaner.logging_setup import register_secrets
from video_cleaner.state import SubtitleRecord, SubtitleStatus
from video_cleaner.subtitles.align import AlignResult
from video_cleaner.subtitles.models import QuotaInfo
from video_cleaner.subtitles.opensubtitles import AuthError, ProviderError, QuotaExhausted
from video_cleaner.subtitles.resolve import (
    CacheUnwritable,
    OnlineFailure,
    QuotaStop,
    ResolveStop,
)
from video_cleaner.subtitles.textfmt import normalise_subtitle
from video_cleaner.tools import ToolNotFound

CACHE_NAME = re.compile(r"^[0-9a-f]{16}-(sidecar|local|opensubtitles)-[0-9a-z]+\.srt$")
HTML = b"<html><body>Download limit reached</body></html>"


def normalised(raw: bytes | str) -> str:
    data = raw.encode("utf-8") if isinstance(raw, str) else raw
    return normalise_subtitle(data, max_bytes=2_000_000, min_cues=20).srt


def key_of(case: Case) -> str:
    text = f"{case.sub.path}|{case.sub.size}|{case.sub.mtime}"
    return hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def online(
    tmp_path: Path,
    client: FakeClient,
    *behaviours: Behaviour,
    movie_hash: str | None = "0123456789abcdef",
    **overrides: Any,
) -> Case:
    aligner = FakeAligner(*behaviours)
    return make_case(tmp_path, client=client, aligner=aligner, movie_hash=movie_hash, **overrides)


# --- the happy paths ------------------------------------------------------------------------


def test_a_hash_exact_result_skips_the_aligner_and_caches_the_normalised_text(
    tmp_path: Path,
) -> None:
    raw = srt_bytes(tag="os").replace(b"\n", b"\r\n")
    client = FakeClient(hash_results=[result(7, hash_match=True)], downloads={7: raw})
    case = online(tmp_path, client)
    out = case.resolve()
    assert out.status is SubtitleStatus.FOUND
    assert out.provider == "opensubtitles"
    assert out.provenance == "English (OpenSubtitles)"
    assert out.hash_exact is True
    assert case.aligner.calls == []
    assert out.cache_path == case.cache_dir / f"{key_of(case)}-opensubtitles-7.srt"
    assert out.cache_path.read_text(encoding="utf-8") == normalised(raw)
    assert out.sha256 == hashlib.sha256(out.cache_path.read_bytes()).hexdigest()
    assert case.cache_files() == [f"{key_of(case)}-opensubtitles-7.srt"]


def test_a_title_result_is_aligned_and_the_cache_holds_the_aligned_normalised_text(
    tmp_path: Path,
) -> None:
    aligned = srt_text(30, tag="shifted", start_s=3.0)
    client = FakeClient(title_results=[result(8)], downloads={8: srt_bytes(tag="os")})
    case = online(tmp_path, client, writing(aligned), movie_hash=None)
    out = case.resolve()
    assert out.status is SubtitleStatus.FOUND
    assert out.hash_exact is False
    assert out.cache_path is not None
    assert out.cache_path.read_text(encoding="utf-8") == normalised(aligned)
    assert case.aligner.calls[0].hash_exact is False
    assert case.cache_files() == [out.cache_path.name]  # the intermediates are gone


def test_the_hash_search_comes_first_and_a_hash_hit_needs_no_title_search(
    tmp_path: Path,
) -> None:
    client = FakeClient(
        hash_results=[result(1, hash_match=True, title="Something Else")],
        title_results=[result(2)],
        downloads={1: srt_bytes(tag="one"), 2: srt_bytes(tag="two")},
    )
    out = online(tmp_path, client).resolve()
    assert out.status is SubtitleStatus.FOUND
    assert client.search_calls == [{"moviehash": "0123456789abcdef", "title": None, "year": None}]
    assert client.download_calls == [1]


def test_without_a_hash_only_the_title_search_runs_with_the_year(tmp_path: Path) -> None:
    client = FakeClient(title_results=[result(2)], downloads={2: srt_bytes()})
    online(tmp_path, client, movie_hash=None).resolve()
    assert client.search_calls == [{"moviehash": None, "title": "The Great Movie", "year": 2010}]


def test_without_a_searchable_title_the_title_search_is_skipped(tmp_path: Path) -> None:
    client = FakeClient(hash_results=[], title_results=[result(2)], downloads={2: srt_bytes()})
    case = online(tmp_path, client, name="2010.mkv")
    out = case.resolve()
    assert [c["title"] for c in client.search_calls] == [None]
    assert out.status is SubtitleStatus.UNAVAILABLE


def test_an_episode_is_never_searched_by_title(tmp_path: Path) -> None:
    client = FakeClient(title_results=[result(2)], downloads={2: srt_bytes()})
    case = online(tmp_path, client, movie_hash=None, name="Some Show S01E02.mkv")
    assert case.resolve().status is SubtitleStatus.UNAVAILABLE
    assert client.search_calls == []


def test_the_same_id_in_both_searches_is_downloaded_once(tmp_path: Path) -> None:
    client = FakeClient(
        hash_results=[result(1)],
        title_results=[result(1), result(2)],
        downloads={1: srt_bytes(tag="one"), 2: srt_bytes(tag="two")},
    )
    case = online(tmp_path, client, failing(), failing())
    out = case.resolve()
    assert client.download_calls == [1, 2]
    assert out.status is SubtitleStatus.UNAVAILABLE


# --- rejections and fall-through ------------------------------------------------------------


def test_the_first_candidate_fails_alignment_and_the_second_is_accepted(tmp_path: Path) -> None:
    client = FakeClient(
        title_results=[result(1), result(2)],
        downloads={1: srt_bytes(tag="one"), 2: srt_bytes(tag="two")},
    )
    case = online(tmp_path, client, failing(), passthrough(), movie_hash=None)
    out = case.resolve()
    assert out.status is SubtitleStatus.FOUND
    assert out.tried_ids == (1,)
    assert out.attempts == 1
    assert out.cache_path is not None and out.cache_path.name.endswith("-opensubtitles-2.srt")
    assert case.cache_files() == [out.cache_path.name]


def test_candidates_exhausted_gives_unavailable_with_a_reason(tmp_path: Path) -> None:
    client = FakeClient(
        title_results=[result(1), result(2)],
        downloads={1: srt_bytes(tag="one"), 2: srt_bytes(tag="two")},
    )
    case = online(
        tmp_path, client, failing("score 0.000 is not a positive match"), failing(), movie_hash=None
    )
    out = case.resolve()
    assert out.status is SubtitleStatus.UNAVAILABLE
    assert out.tried_ids == (1, 2)
    assert out.attempts == 2
    assert out.reason.startswith(
        "no English subtitle found: sidecar none, local none, OpenSubtitles tried 2"
    )
    assert "not a positive match" in out.reason
    assert len(out.reason) < 300
    assert case.cache_files() == []


def test_no_result_at_all_is_unavailable(tmp_path: Path) -> None:
    out = online(tmp_path, FakeClient()).resolve()
    assert out.status is SubtitleStatus.UNAVAILABLE
    assert "OpenSubtitles tried 0" in out.reason


def test_a_hostile_download_is_rejected_and_the_next_is_tried(tmp_path: Path) -> None:
    client = FakeClient(
        title_results=[result(1), result(2)], downloads={1: HTML, 2: srt_bytes(tag="two")}
    )
    case = online(tmp_path, client, movie_hash=None)
    out = case.resolve()
    assert out.status is SubtitleStatus.FOUND
    assert out.tried_ids == (1,)
    assert client.download_calls == [1, 2]


def test_a_non_english_download_is_rejected_even_when_hash_exact(tmp_path: Path) -> None:
    client = FakeClient(
        hash_results=[result(1, hash_match=True)], downloads={1: srt_bytes(german=True)}
    )
    out = online(tmp_path, client).resolve()
    assert out.status is SubtitleStatus.UNAVAILABLE
    assert out.tried_ids == (1,)
    assert case_files_empty(tmp_path)


def case_files_empty(tmp_path: Path) -> bool:
    cache = tmp_path / "cache"
    return not cache.exists() or not any(cache.iterdir())


def test_a_hash_exact_file_of_the_wrong_length_is_rejected(tmp_path: Path) -> None:
    client = FakeClient(
        hash_results=[result(1, hash_match=True)], downloads={1: srt_bytes(30, start_s=5000.0)}
    )
    out = online(tmp_path, client).resolve()
    assert out.status is SubtitleStatus.UNAVAILABLE
    assert out.tried_ids == (1,)


def test_a_cut_just_inside_the_length_limit_is_accepted(tmp_path: Path) -> None:
    # 100 s of video: the limit is 100 * 1.10 + 120 = 230 s.
    inside = FakeClient(
        hash_results=[result(1, hash_match=True)],
        downloads={1: srt_bytes(30, start_s=1.0, step_s=7.5)},
    )
    assert online(tmp_path, inside).resolve().status is SubtitleStatus.FOUND


def test_a_video_of_unknown_duration_cannot_vouch_for_a_subtitle(tmp_path: Path) -> None:
    client = FakeClient(hash_results=[result(1, hash_match=True)], downloads={1: srt_bytes()})
    out = online(tmp_path, client, duration_s=0.0).resolve()
    assert out.status is SubtitleStatus.UNAVAILABLE


def test_an_aligned_output_with_too_few_cues_is_rejected(tmp_path: Path) -> None:
    client = FakeClient(title_results=[result(1)], downloads={1: srt_bytes(30)})
    out = online(tmp_path, client, writing(srt_text(25)), movie_hash=None).resolve()
    assert out.status is SubtitleStatus.UNAVAILABLE
    assert out.tried_ids == (1,)


def test_an_aligned_output_that_keeps_95_percent_of_the_cues_is_accepted(tmp_path: Path) -> None:
    client = FakeClient(title_results=[result(1)], downloads={1: srt_bytes(30)})
    out = online(tmp_path, client, writing(srt_text(29)), movie_hash=None).resolve()
    assert out.status is SubtitleStatus.FOUND


@pytest.mark.parametrize(
    "output",
    [srt_text(30, german=True), "<html><body>no</body></html>", "", "garbage"],
    ids=["german", "html", "empty", "garbage"],
)
def test_an_aligned_output_that_is_not_a_good_english_subtitle_is_rejected(
    tmp_path: Path, output: str
) -> None:
    client = FakeClient(title_results=[result(1)], downloads={1: srt_bytes(30)})
    out = online(tmp_path, client, writing(output), movie_hash=None).resolve()
    assert out.status is SubtitleStatus.UNAVAILABLE
    assert out.tried_ids == (1,)


def test_an_alignment_that_claims_success_without_an_output_is_rejected(tmp_path: Path) -> None:
    def no_output(subtitle: Path, out: Path) -> AlignResult:
        return AlignResult(True, "fine", 1.0, 1.0, 5.0, None)

    client = FakeClient(title_results=[result(1)], downloads={1: srt_bytes(30)})
    out = online(tmp_path, client, no_output, movie_hash=None).resolve()
    assert out.status is SubtitleStatus.UNAVAILABLE


def test_an_aligned_output_over_the_size_limit_is_rejected(tmp_path: Path) -> None:
    client = FakeClient(title_results=[result(1)], downloads={1: srt_bytes(30)})
    big = srt_text(30) + "\n" * 4000
    case = online(tmp_path, client, writing(big), movie_hash=None, subtitle_max_bytes=3000)
    assert case.resolve().status is SubtitleStatus.UNAVAILABLE


# --- candidate filters ----------------------------------------------------------------------


def test_foreign_parts_only_results_are_never_downloaded(tmp_path: Path) -> None:
    client = FakeClient(
        title_results=[result(1, foreign_parts_only=True), result(2)],
        downloads={1: srt_bytes(tag="one"), 2: srt_bytes(tag="two")},
    )
    out = online(tmp_path, client, movie_hash=None).resolve()
    assert client.download_calls == [2]
    assert out.tried_ids == ()


@pytest.mark.parametrize(
    "bad",
    [
        result(1, title="A Different Film"),
        result(1, year=2015),
        result(1, title=""),
        result(1, title="The Great Movie 2"),
        result(1, year=None, title="The Great Movie Returns"),
    ],
    ids=["title", "year", "empty-title", "sequel", "no-year-different"],
)
def test_a_result_whose_title_or_year_does_not_match_is_not_downloaded(
    tmp_path: Path, bad: object
) -> None:
    client = FakeClient(title_results=[bad], downloads={1: srt_bytes()})  # type: ignore[list-item]
    out = online(tmp_path, client, movie_hash=None).resolve()
    assert client.download_calls == []
    assert out.tried_ids == ()
    assert out.status is SubtitleStatus.UNAVAILABLE


def test_a_hash_search_result_without_a_hash_match_needs_the_title_to_match(
    tmp_path: Path,
) -> None:
    client = FakeClient(
        hash_results=[result(1, title="Other"), result(2, title="The Great Movie")],
        downloads={1: srt_bytes(tag="one"), 2: srt_bytes(tag="two")},
    )
    case = online(tmp_path, client)
    out = case.resolve()
    assert client.download_calls == [2]
    assert out.hash_exact is False
    assert case.aligner.calls  # not trusted by hash, so it was aligned


def test_a_title_search_result_is_never_hash_exact_even_if_it_says_so(tmp_path: Path) -> None:
    client = FakeClient(title_results=[result(1, hash_match=True)], downloads={1: srt_bytes()})
    case = online(tmp_path, client, movie_hash=None)
    out = case.resolve()
    assert out.hash_exact is False
    assert case.aligner.calls


def test_ids_already_tried_are_not_downloaded_again(tmp_path: Path) -> None:
    client = FakeClient(
        title_results=[result(1), result(2)],
        downloads={1: srt_bytes(tag="one"), 2: srt_bytes(tag="two")},
    )
    case = online(tmp_path, client, movie_hash=None)
    out = case.resolve(replace(case.sub, tried_ids=(1,)))
    assert client.download_calls == [2]
    assert out.tried_ids == (1,)


def test_at_most_the_configured_number_of_downloads_are_made(tmp_path: Path) -> None:
    results = [result(n) for n in range(1, 8)]
    client = FakeClient(
        title_results=[result(100, foreign_parts_only=True), result(101, title="Other"), *results],
        downloads={n: srt_bytes(tag=str(n)) for n in range(1, 8)},
    )
    case = online(
        tmp_path,
        client,
        *[failing() for _ in range(7)],
        movie_hash=None,
        subtitle_candidates_per_file=2,
    )
    out = case.resolve()
    assert client.download_calls == [1, 2]
    assert out.status is SubtitleStatus.UNAVAILABLE
    assert out.tried_ids == (1, 2)


def test_no_title_search_is_made_once_the_downloads_are_used_up(tmp_path: Path) -> None:
    client = FakeClient(
        hash_results=[result(1)], title_results=[result(2)], downloads={1: srt_bytes()}
    )
    case = online(tmp_path, client, failing(), subtitle_candidates_per_file=1)
    out = case.resolve()
    assert out.status is SubtitleStatus.UNAVAILABLE
    assert [c["title"] for c in client.search_calls] == [None]


def test_tried_ids_are_deduplicated_and_capped_at_fifty_keeping_the_latest(
    tmp_path: Path,
) -> None:
    client = FakeClient(title_results=[result(999)], downloads={999: srt_bytes()})
    case = online(tmp_path, client, failing(), movie_hash=None)
    old = (*range(1, 61), 5, 6)
    out = case.resolve(replace(case.sub, tried_ids=old))
    assert len(out.tried_ids) == 50
    assert len(set(out.tried_ids)) == 50
    assert out.tried_ids[-1] == 999
    assert out.tried_ids[-2:-1] == (6,)


def test_foreign_audio_still_searches_online(tmp_path: Path) -> None:
    client = FakeClient(hash_results=[result(1, hash_match=True)], downloads={1: srt_bytes()})
    out = online(tmp_path, client, english_audio=False).resolve()
    assert out.status is SubtitleStatus.FOUND
    assert out.english_audio is False
    assert client.download_calls == [1]


def test_english_audio_is_carried_through_every_outcome(tmp_path: Path) -> None:
    case = online(tmp_path, FakeClient(), english_audio=True)
    assert case.resolve().english_audio is True


# --- names and logs -------------------------------------------------------------------------


def test_nothing_the_server_sends_reaches_a_path_or_an_info_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    hostile = "..\\..\\evil\n‮gnp.srt SECRET-TOKEN-XYZ"
    register_secrets(["SECRET-TOKEN-XYZ"])
    client = FakeClient(
        hash_results=[result(7, hash_match=True, release=hostile, title=hostile)],
        downloads={7: srt_bytes()},
    )
    case = online(tmp_path, client)
    with caplog.at_level(logging.DEBUG, logger="video_cleaner.subtitles.resolve"):
        out = case.resolve()
    assert out.status is SubtitleStatus.FOUND
    assert all(CACHE_NAME.match(name) for name in case.cache_files())
    for record in caplog.records:
        message = record.getMessage()
        assert "SECRET-TOKEN-XYZ" not in message
        assert "\n" not in message and "‮" not in message
        if record.levelno >= logging.INFO:
            assert "evil" not in message
    assert not any("evil" in text for text in case.reporter.stages)
    assert not (tmp_path / "evil").exists()


# --- quota ----------------------------------------------------------------------------------


def test_quota_exhausted_at_the_first_download_stops_and_leaves_the_record_wanted(
    tmp_path: Path,
) -> None:
    reset = NOW + timedelta(hours=5)
    client = FakeClient(title_results=[result(1)], downloads={1: QuotaExhausted("used up", reset)})
    case = online(tmp_path, client, movie_hash=None)
    with pytest.raises(QuotaStop) as caught:
        case.resolve()
    stop = caught.value
    assert isinstance(stop, ResolveStop)
    assert stop.reset_at == reset
    assert stop.record.status is SubtitleStatus.WANTED
    assert stop.record.tried_ids == ()
    assert stop.reason


def test_quota_exhausted_after_a_rejected_candidate_keeps_its_id(tmp_path: Path) -> None:
    client = FakeClient(
        title_results=[result(1), result(2)],
        downloads={1: srt_bytes(), 2: QuotaExhausted("used up", None)},
    )
    case = online(tmp_path, client, failing(), movie_hash=None)
    with pytest.raises(QuotaStop) as caught:
        case.resolve()
    assert caught.value.reset_at is None
    assert caught.value.record.tried_ids == (1,)
    assert caught.value.record.attempts == 1
    assert caught.value.record.status is SubtitleStatus.WANTED


def test_quota_exhausted_at_search_stops(tmp_path: Path) -> None:
    client = FakeClient(hash_error=QuotaExhausted("used up", NOW))
    with pytest.raises(QuotaStop):
        online(tmp_path, client).resolve()


def test_a_spent_allowance_known_from_the_last_reply_stops_before_any_request(
    tmp_path: Path,
) -> None:
    reset = NOW + timedelta(hours=3)
    client = FakeClient(
        title_results=[result(1)], downloads={1: srt_bytes()}, quota=QuotaInfo(0, reset)
    )
    case = online(tmp_path, client, movie_hash=None)
    with pytest.raises(QuotaStop) as caught:
        case.resolve()
    assert client.download_calls == []
    assert caught.value.reset_at == reset


def test_an_allowance_whose_reset_has_passed_is_usable(tmp_path: Path) -> None:
    client = FakeClient(
        title_results=[result(1)],
        downloads={1: srt_bytes()},
        quota=QuotaInfo(0, NOW - timedelta(hours=1)),
    )
    assert online(tmp_path, client, movie_hash=None).resolve().status is SubtitleStatus.FOUND


# --- failures -------------------------------------------------------------------------------


def test_a_login_failure_is_a_fatal_online_failure_and_the_record_stays_wanted(
    tmp_path: Path,
) -> None:
    client = FakeClient(hash_error=AuthError("refused"))
    with pytest.raises(OnlineFailure) as caught:
        online(tmp_path, client).resolve()
    assert caught.value.fatal is True
    assert caught.value.record.status is SubtitleStatus.WANTED
    assert "refused" in caught.value.reason


def test_an_auth_failure_at_download_is_fatal(tmp_path: Path) -> None:
    client = FakeClient(title_results=[result(1)], downloads={1: AuthError("refused")})
    with pytest.raises(OnlineFailure) as caught:
        online(tmp_path, client, movie_hash=None).resolve()
    assert caught.value.fatal is True


def test_a_provider_error_after_a_rejected_candidate_keeps_the_rejected_id(
    tmp_path: Path,
) -> None:
    client = FakeClient(
        title_results=[result(1), result(2)],
        downloads={1: HTML, 2: ProviderError("HTTP 500")},
    )
    with pytest.raises(OnlineFailure) as caught:
        online(tmp_path, client, movie_hash=None).resolve()
    stop = caught.value
    assert stop.fatal is False
    assert stop.record.status is SubtitleStatus.WANTED
    # The failed id may have been counted by the server, so it is not asked for again.
    assert stop.record.tried_ids == (1, 2)
    assert stop.record.attempts == 2


def test_a_provider_error_at_search_is_not_fatal(tmp_path: Path) -> None:
    client = FakeClient(title_error=ProviderError("HTTP 503"))
    with pytest.raises(OnlineFailure) as caught:
        online(tmp_path, client, movie_hash=None).resolve()
    assert caught.value.fatal is False


def test_a_bad_argument_is_a_bug_and_propagates(tmp_path: Path) -> None:
    client = FakeClient(hash_error=ValueError("moviehash must be 16 lower-case hex digits"))
    with pytest.raises(ValueError, match="moviehash"):
        online(tmp_path, client).resolve()


def test_a_missing_aligner_tool_propagates_and_keeps_the_download(tmp_path: Path) -> None:
    client = FakeClient(title_results=[result(1)], downloads={1: srt_bytes()})
    case = online(tmp_path, client, raising(ToolNotFound("ffsubsync")), movie_hash=None)
    with pytest.raises(ToolNotFound):
        case.resolve()
    assert case.cache_files() == [f"{key_of(case)}-opensubtitles-1.dl.srt"]


# --- kill safety ---------------------------------------------------------------------------


def test_the_download_is_on_disk_before_the_alignment_starts(tmp_path: Path) -> None:
    raw = srt_bytes(tag="os").replace(b"\n", b"\r\n")
    client = FakeClient(title_results=[result(1)], downloads={1: raw})
    case = online(tmp_path, client, movie_hash=None)
    case.resolve()
    call = case.aligner.calls[0]
    assert call.subtitle.name == f"{key_of(case)}-opensubtitles-1.dl.srt"
    assert call.subtitle_text == normalised(raw)


def test_a_download_left_by_a_killed_run_is_adopted_without_another_download(
    tmp_path: Path,
) -> None:
    client = FakeClient(title_results=[result(1)], downloads={})
    case = online(tmp_path, client, movie_hash=None)
    case.cache_dir.mkdir()
    (case.cache_dir / f"{key_of(case)}-opensubtitles-1.dl.srt").write_text(
        normalised(srt_bytes(tag="left")), encoding="utf-8"
    )
    out = case.resolve()
    assert out.status is SubtitleStatus.FOUND
    assert client.download_calls == []
    assert case.cache_files() == [f"{key_of(case)}-opensubtitles-1.srt"]


def test_a_left_download_that_fails_alignment_is_rejected_and_removed(tmp_path: Path) -> None:
    client = FakeClient(title_results=[result(1)], downloads={})
    case = online(tmp_path, client, failing(), movie_hash=None)
    case.cache_dir.mkdir()
    (case.cache_dir / f"{key_of(case)}-opensubtitles-1.dl.srt").write_text(
        normalised(srt_bytes(tag="left")), encoding="utf-8"
    )
    out = case.resolve()
    assert out.status is SubtitleStatus.UNAVAILABLE
    assert out.tried_ids == (1,)
    assert client.download_calls == []
    assert case.cache_files() == []


def test_a_finished_cache_file_is_adopted_with_no_download_and_no_alignment(
    tmp_path: Path,
) -> None:
    first = FakeClient(title_results=[result(1)], downloads={1: srt_bytes(tag="os")})
    case = online(tmp_path, first, movie_hash=None)
    done = case.resolve()
    second = FakeClient(title_results=[result(1)], downloads={})
    case.ctx = replace(case.ctx, client=second, align=FakeAligner(raising(AssertionError("no"))))
    again = case.resolve()
    assert again.status is SubtitleStatus.FOUND
    assert again.cache_path == done.cache_path
    assert again.sha256 == done.sha256
    assert second.download_calls == []


def test_a_cache_file_that_is_not_our_normalised_output_is_not_adopted(tmp_path: Path) -> None:
    client = FakeClient(title_results=[result(1)], downloads={1: srt_bytes(tag="fresh")})
    case = online(tmp_path, client, movie_hash=None)
    case.cache_dir.mkdir()
    final = case.cache_dir / f"{key_of(case)}-opensubtitles-1.srt"
    final.write_bytes(srt_bytes(tag="tampered").replace(b"\n", b"\r\n"))
    out = case.resolve()
    assert client.download_calls == [1]
    assert out.cache_path is not None
    assert out.cache_path.read_text(encoding="utf-8") == normalised(srt_bytes(tag="fresh"))


def test_the_intermediates_are_removed_after_a_rejection(tmp_path: Path) -> None:
    client = FakeClient(title_results=[result(1)], downloads={1: srt_bytes()})
    case = online(tmp_path, client, failing(), movie_hash=None)
    case.resolve()
    assert case.cache_files() == []


def _failing_writes(
    monkeypatch: pytest.MonkeyPatch, ends_with: str, seen: list[tuple[str, bool]] | None = None
) -> None:
    def broken(path: Path, text: str, **kwargs: Any) -> None:
        if seen is not None:
            seen.append((path.name, bool(kwargs.get("fsync", False))))
        if path.name.endswith(ends_with):
            raise OSError("disk full")
        write_text_atomic(path, text, **kwargs)

    monkeypatch.setattr("video_cleaner.subtitles.resolve.write_text_atomic", broken)


def test_a_failure_to_keep_the_download_stops_before_another_download_is_spent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _failing_writes(monkeypatch, ".dl.srt")
    client = FakeClient(
        title_results=[result(1), result(2), result(3)],
        downloads={n: srt_bytes(tag=str(n)) for n in (1, 2, 3)},
    )
    case = online(tmp_path, client, movie_hash=None)
    with pytest.raises(CacheUnwritable) as caught:
        case.resolve()
    assert isinstance(caught.value, ResolveStop)
    assert client.download_calls == [1]
    # It was downloaded and cannot be resumed from, so it is not spent again; 2 and 3 were
    # never asked for and are not blamed.
    assert caught.value.record.tried_ids == (1,)
    assert caught.value.record.status is SubtitleStatus.WANTED
    assert case.aligner.calls == []


def test_the_kept_download_is_flushed_to_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, bool]] = []
    _failing_writes(monkeypatch, "never-matches", seen)
    client = FakeClient(title_results=[result(1)], downloads={1: srt_bytes()})
    online(tmp_path, client, movie_hash=None).resolve()
    assert [fsync for name, fsync in seen if name.endswith(".dl.srt")] == [True]


def test_a_failure_to_write_the_result_keeps_the_download_for_the_next_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _failing_writes(monkeypatch, "-opensubtitles-1.srt")
    client = FakeClient(
        title_results=[result(1), result(2)], downloads={1: srt_bytes(), 2: srt_bytes(tag="b")}
    )
    case = online(tmp_path, client, movie_hash=None)
    with pytest.raises(CacheUnwritable) as caught:
        case.resolve()
    assert client.download_calls == [1]
    assert caught.value.record.tried_ids == ()
    assert case.cache_files() == [f"{key_of(case)}-opensubtitles-1.dl.srt"]


def test_an_unwritable_cache_folder_is_found_before_any_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(folder: Path) -> None:
        raise OSError("read-only")

    monkeypatch.setattr("video_cleaner.subtitles.resolve.probe_writable", broken)
    client = FakeClient(title_results=[result(1)], downloads={1: srt_bytes()})
    case = online(tmp_path, client, movie_hash=None)
    with pytest.raises(CacheUnwritable) as caught:
        case.resolve()
    assert client.download_calls == []
    assert caught.value.record.status is SubtitleStatus.WANTED
    assert caught.value.record.tried_ids == ()


def test_a_cache_path_that_is_a_file_is_unwritable_before_any_download(tmp_path: Path) -> None:
    client = FakeClient(title_results=[result(1)], downloads={1: srt_bytes()})
    case = online(tmp_path, client, movie_hash=None)
    case.cache_dir.write_text("in the way")
    with pytest.raises(CacheUnwritable):
        case.resolve()
    assert client.download_calls == []


def test_the_cache_folder_is_probed_once_per_call_and_only_when_a_download_is_due(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probes: list[Path] = []
    monkeypatch.setattr("video_cleaner.subtitles.resolve.probe_writable", probes.append)
    client = FakeClient(
        title_results=[result(1), result(2)],
        downloads={1: srt_bytes(tag="a"), 2: srt_bytes(tag="b")},
    )
    online(tmp_path / "a", client, failing(), failing(), movie_hash=None).resolve()
    assert len(probes) == 1
    probes.clear()
    online(tmp_path / "b", FakeClient(), movie_hash=None).resolve()
    assert probes == []


@pytest.mark.parametrize(
    "message",
    [
        "OpenSubtitles download: the downloaded file is empty",
        "download: the reply names a link outside opensubtitles.com",
        "OpenSubtitles download failed: HTTP 503; not retried, the server may have counted it",
    ],
    ids=["empty", "foreign-link", "transient"],
)
def test_a_provider_error_at_download_is_never_asked_for_again(
    tmp_path: Path, message: str
) -> None:
    client = FakeClient(title_results=[result(1)], downloads={1: ProviderError(message)})
    case = online(tmp_path, client, movie_hash=None)
    with pytest.raises(OnlineFailure) as caught:
        case.resolve()
    stop = caught.value
    assert stop.fatal is False
    assert stop.record.tried_ids == (1,)
    assert stop.record.attempts == 1
    assert stop.record.status is SubtitleStatus.WANTED
    again = case.resolve(stop.record)
    assert client.download_calls == [1]
    assert again.status is SubtitleStatus.UNAVAILABLE
    assert again.tried_ids == (1,)


def test_a_refused_login_or_a_spent_allowance_does_not_blame_the_id(tmp_path: Path) -> None:
    auth = FakeClient(title_results=[result(1)], downloads={1: AuthError("refused")})
    with pytest.raises(OnlineFailure) as refused:
        online(tmp_path / "a", auth, movie_hash=None).resolve()
    assert refused.value.record.tried_ids == ()
    quota = FakeClient(title_results=[result(1)], downloads={1: QuotaExhausted("used up", None)})
    with pytest.raises(QuotaStop) as spent:
        online(tmp_path / "b", quota, movie_hash=None).resolve()
    assert spent.value.record.tried_ids == ()


def test_an_episode_accepts_only_a_hash_match(tmp_path: Path) -> None:
    show = result(1, title="Some Show", year=None)
    client = FakeClient(hash_results=[show], downloads={1: srt_bytes()})
    out = online(tmp_path, client, name="Some Show S01E02.mkv").resolve()
    assert client.download_calls == []
    assert out.status is SubtitleStatus.UNAVAILABLE
    exact = FakeClient(hash_results=[replace(show, hash_match=True)], downloads={1: srt_bytes()})
    found = online(tmp_path / "b", exact, name="Some Show S01E02.mkv").resolve()
    assert found.status is SubtitleStatus.FOUND


def test_a_stored_record_is_unchanged_in_its_key_fields(tmp_path: Path) -> None:
    client = FakeClient(hash_results=[result(1, hash_match=True)], downloads={1: srt_bytes()})
    case = online(tmp_path, client)
    out = case.resolve()
    assert isinstance(out, SubtitleRecord)
    assert (out.path, out.size, out.mtime) == (case.sub.path, case.sub.size, case.sub.mtime)
