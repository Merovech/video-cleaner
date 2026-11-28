"""Survey and subtitles: classify each file's own subtitles, write the subtitle record, and turn a
file that was only flagged into a remux once a subtitle has been found for it."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from pathlib import Path

import pytest

from test_survey import (
    FAKE_TOOLS,
    GIB,
    RecordingReporter,
    _by_name,
    _csv_rows,
    _fake_probe,
    _info,
    _settings,
    _touch,
)
from video_cleaner import survey as survey_module
from video_cleaner.config import Settings
from video_cleaner.decide import Action
from video_cleaner.probe import AudioStream, MediaInfo, ProbeError, SubtitleStream
from video_cleaner.progress import Reporter
from video_cleaner.state import State, Status, SubtitleRecord, SubtitleStatus
from video_cleaner.survey import CSV_COLUMNS, SurveyResult, run_survey
from video_cleaner.tools import ToolNotFound, Tools

HDR_REASON = "HDR (transfer smpte2084)"
FOUND_REASONS = (HDR_REASON, "video left as is", "new subtitles to mux in")


def _sub(
    index: int, language: str = "eng", *, forced: bool = False, codec: str = "subrip"
) -> SubtitleStream:
    return SubtitleStream(
        index=index,
        codec=codec,
        language=language,
        forced=forced,
        sdh=False,
        is_image=codec in {"hdmv_pgs_subtitle", "dvd_subtitle"},
    )


def _with(
    path: Path,
    *subs: SubtitleStream,
    audio_language: str = "eng",
    hdr: bool = False,
    has_video: bool = True,
    size: int = 6 * GIB,
) -> MediaInfo:
    base = _info(path, has_video=has_video, size=size)
    video = base.video
    if hdr and video is not None:
        video = dataclasses.replace(video, transfer="smpte2084")
    return dataclasses.replace(
        base,
        video=video,
        audio=(AudioStream(index=1, codec="ac3", language=audio_language, title=""),),
        subtitles=subs,
    )


MediaInfoRule = Callable[[Path], MediaInfo]


class FakeDensity:
    """Stands in for `measure_cue_density`; remembers every call."""

    def __init__(self, densities: dict[int, float] | None = None) -> None:
        self.densities = densities or {}
        self.calls: list[tuple[Path, int, float, int, int]] = []

    def __call__(
        self,
        tools: Tools,
        path: Path,
        stream: SubtitleStream,
        duration_s: float,
        reporter: Reporter,
        *,
        max_bytes: int,
        min_cues: int,
    ) -> float:
        self.calls.append((path, stream.index, duration_s, max_bytes, min_cues))
        return self.densities.get(stream.index, 0.0)


def _survey(
    settings: Settings,
    rules: dict[str, MediaInfoRule],
    density: FakeDensity | None = None,
    state: State | None = None,
) -> tuple[SurveyResult, State]:
    state = State.open(settings.state_db) if state is None else state
    result = run_survey(
        settings,
        FAKE_TOOLS,
        state,
        RecordingReporter(),
        probe_fn=_fake_probe(rules),
        hash_fn=lambda _p: None,
        density_fn=density if density is not None else FakeDensity(),
    )
    return result, state


def _sub_row(state: State, path: Path) -> SubtitleRecord | None:
    rec = state.get(path)
    assert rec is not None
    return state.subtitle(path, rec.size, rec.mtime)


# --- what the file's own subtitles say ---------------------------------------------------------


def test_a_full_english_track_needs_nothing(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    film = _touch(settings.library_root / "Has Subs.mkv")
    density = FakeDensity()

    result, state = _survey(settings, {"Has Subs.mkv": lambda p: _with(p, _sub(2))}, density)
    try:
        sub = _sub_row(state, film)
        assert sub is not None
        assert sub.status is SubtitleStatus.NOT_NEEDED and sub.provider == "embedded"
        rec = state.get(film)
        assert rec is not None and rec.decision == "encode"  # the decision is unchanged
    finally:
        state.close()
    row = _by_name(_csv_rows(result.csv_path))["Has Subs.mkv"]
    assert (row["english_subtitle"], row["subtitle_status"]) == ("embedded", "not_needed")
    assert density.calls == []  # a full English track needs no measuring


def test_forced_english_track_with_many_cues_is_reclassified_as_full(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    film = _touch(settings.library_root / "Mislabelled.mkv")
    density = FakeDensity({3: 400.0})
    forced = _sub(3, forced=True)

    result, state = _survey(
        settings,
        {"Mislabelled.mkv": lambda p: _with(p, _sub(2, "fra"), forced)},
        density,
    )
    try:
        sub = _sub_row(state, film)
        assert sub is not None and sub.status is SubtitleStatus.NOT_NEEDED
        assert sub.provider == "embedded"
    finally:
        state.close()
    row = _by_name(_csv_rows(result.csv_path))["Mislabelled.mkv"]
    assert (row["english_subtitle"], row["subtitle_status"]) == ("reclassified", "not_needed")
    # Measured once, for the forced English track only, with the configured size limit. The
    # minimum of cues for a usable subtitle does not apply to counting a track's cues.
    assert density.calls == [(film, 3, 3600.0, settings.subtitle_max_bytes, 1)]


def test_forced_english_track_with_few_cues_stays_wanted(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    film = _touch(settings.library_root / "Signs Only.mkv")
    density = FakeDensity({3: 40.0})

    result, state = _survey(
        settings, {"Signs Only.mkv": lambda p: _with(p, _sub(3, forced=True))}, density
    )
    try:
        sub = _sub_row(state, film)
        assert sub is not None and sub.status is SubtitleStatus.WANTED
        assert sub.english_audio is True
    finally:
        state.close()
    row = _by_name(_csv_rows(result.csv_path))["Signs Only.mkv"]
    assert (row["english_subtitle"], row["subtitle_status"]) == ("none", "wanted")
    assert len(density.calls) == 1


def test_density_is_measured_only_for_forced_english_text_tracks(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _touch(settings.library_root / "Mixed.mkv")
    density = FakeDensity({4: 10.0})
    mixed = _with(
        Path("Mixed.mkv"),
        _sub(2, "fra", forced=True),  # forced, but not English
        _sub(3, "eng", forced=True, codec="hdmv_pgs_subtitle"),  # an image track: cannot be counted
        _sub(4, "eng", forced=True),  # the only one worth measuring
    )

    _result, state = _survey(
        settings, {"Mixed.mkv": lambda p: dataclasses.replace(mixed, path=p)}, density
    )
    state.close()

    assert [call[1] for call in density.calls] == [4]


def test_a_failed_density_measurement_is_logged_with_the_file_name(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = _settings(tmp_path)
    _touch(settings.library_root / "Broken Track.mkv")
    density = FakeDensity()  # every stream measures 0.0

    with caplog.at_level(logging.WARNING, logger="video_cleaner.survey"):
        _result, state = _survey(
            settings, {"Broken Track.mkv": lambda p: _with(p, _sub(3, forced=True))}, density
        )
    state.close()

    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "Broken Track.mkv" in m and "cue density could not be measured" in m for m in messages
    )


def test_a_forced_track_with_only_a_few_cues_is_measured_without_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    from video_cleaner.subtitles import embedded
    from video_cleaner.tools import ToolResult

    signs = "".join(
        f"{n}\n00:0{n}:00,000 --> 00:0{n}:02,000\nA sign on a door.\n\n" for n in range(1, 8)
    )
    monkeypatch.setattr(
        embedded,
        "run_tool",
        lambda *args, **kwargs: ToolResult(returncode=0, stdout=signs, stderr_tail=""),
    )
    settings = _settings(tmp_path)
    assert settings.subtitle_min_cues > 7
    film = _touch(settings.library_root / "Signs.mkv")
    state = State.open(settings.state_db)
    try:
        with caplog.at_level(logging.WARNING):
            run_survey(
                settings,
                FAKE_TOOLS,
                state,
                RecordingReporter(),
                probe_fn=_fake_probe({"Signs.mkv": lambda p: _with(p, _sub(3, forced=True))}),
                hash_fn=lambda _p: None,
                density_fn=embedded.measure_cue_density,
            )
        sub = _sub_row(state, film)
        assert sub is not None and sub.status is SubtitleStatus.WANTED
    finally:
        state.close()
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_a_hung_density_extraction_does_not_hang_the_survey(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    from video_cleaner.subtitles import embedded
    from video_cleaner.tools import ToolTimedOut

    def hung(*args: object, **kwargs: object) -> object:
        assert kwargs["timeout_s"] == embedded.DENSITY_TIMEOUT_S
        raise ToolTimedOut("extract subtitle stream 3", embedded.DENSITY_TIMEOUT_S)

    monkeypatch.setattr(embedded, "run_tool", hung)
    settings = _settings(tmp_path)
    film = _touch(settings.library_root / "Hung.mkv")
    state = State.open(settings.state_db)
    try:
        with caplog.at_level(logging.WARNING, logger="video_cleaner.survey"):
            run_survey(
                settings,
                FAKE_TOOLS,
                state,
                RecordingReporter(),
                probe_fn=_fake_probe({"Hung.mkv": lambda p: _with(p, _sub(3, forced=True))}),
                hash_fn=lambda _p: None,
                density_fn=embedded.measure_cue_density,
            )
        sub = _sub_row(state, film)
        assert sub is not None and sub.status is SubtitleStatus.WANTED
    finally:
        state.close()
    assert any("cue density could not be measured" in r.getMessage() for r in caplog.records)


def test_a_file_with_no_english_subtitles_is_wanted(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    film = _touch(settings.library_root / "Plain.mkv")
    density = FakeDensity()

    result, state = _survey(
        settings,
        {"Plain.mkv": lambda p: _with(p, _sub(2, "fra"), audio_language="fra")},
        density,
    )
    try:
        sub = _sub_row(state, film)
        assert sub is not None and sub.status is SubtitleStatus.WANTED
        assert sub.english_audio is False
        rec = state.get(film)
        assert rec is not None and rec.decision == "encode"
    finally:
        state.close()
    row = _by_name(_csv_rows(result.csv_path))["Plain.mkv"]
    assert (row["english_subtitle"], row["subtitle_status"]) == ("none", "wanted")
    assert density.calls == []


def test_english_audio_is_true_when_any_audio_track_is_english(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    film = _touch(settings.library_root / "Dub.mkv")

    def two_tracks(path: Path) -> MediaInfo:
        info = _with(path, audio_language="fra")
        extra = AudioStream(index=2, codec="ac3", language="en", title="")
        return dataclasses.replace(info, audio=(*info.audio, extra))

    _result, state = _survey(settings, {"Dub.mkv": two_tracks})
    try:
        sub = _sub_row(state, film)
        assert sub is not None and sub.english_audio is True
    finally:
        state.close()


def test_an_undetermined_track_is_unclear_and_flags_the_file(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    film = _touch(settings.library_root / "Und.mkv")

    result, state = _survey(settings, {"Und.mkv": lambda p: _with(p, _sub(2, "und"))})
    try:
        sub = _sub_row(state, film)
        assert sub is not None and sub.status is SubtitleStatus.UNCLEAR
        assert "no language tag" in sub.reason
        rec = state.get(film)
        assert rec is not None
        assert (rec.decision, rec.status) == ("flag", Status.FLAGGED)
        assert rec.reasons == (sub.reason,)
    finally:
        state.close()
    row = _by_name(_csv_rows(result.csv_path))["Und.mkv"]
    assert (row["english_subtitle"], row["subtitle_status"]) == ("unclear", "unclear")
    assert row["action"] == "flag" and "no language tag" in row["reasons"]
    assert result.counts[Action.FLAG] == 1


def test_an_unclear_file_that_was_already_flagged_keeps_its_reasons(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    film = _touch(settings.library_root / "Hdr Und.mkv")

    _result, state = _survey(
        settings, {"Hdr Und.mkv": lambda p: _with(p, _sub(2, "und"), hdr=True)}
    )
    try:
        rec = state.get(film)
        assert rec is not None and rec.decision == "flag"
        assert rec.reasons[0] == HDR_REASON and "no language tag" in rec.reasons[1]
    finally:
        state.close()


# --- flagged files ------------------------------------------------------------------------------


def test_an_hdr_file_without_english_subtitles_is_wanted_and_stays_flagged(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    film = _touch(settings.library_root / "Hdr.mkv")

    result, state = _survey(settings, {"Hdr.mkv": lambda p: _with(p, hdr=True)})
    try:
        sub = _sub_row(state, film)
        assert sub is not None and sub.status is SubtitleStatus.WANTED
        rec = state.get(film)
        assert rec is not None
        assert (rec.decision, rec.reasons) == ("flag", (HDR_REASON,))
    finally:
        state.close()
    assert _by_name(_csv_rows(result.csv_path))["Hdr.mkv"]["subtitle_status"] == "wanted"


def test_a_found_subtitle_turns_the_flagged_file_into_a_remux_on_resurvey(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    film = _touch(settings.library_root / "Hdr.mkv")
    rules = {"Hdr.mkv": lambda p: _with(p, hdr=True)}
    _result, state = _survey(settings, rules)
    try:
        rec = state.get(film)
        assert rec is not None and rec.decision == "flag"
        cache = tmp_path / "cache" / "Hdr.srt"
        found = SubtitleRecord(
            film,
            rec.size,
            rec.mtime,
            SubtitleStatus.FOUND,
            provider="opensubtitles",
            cache_path=cache,
            sha256="ab" * 32,
            english_audio=True,
            tried_ids=(7,),
        )
        state.set_subtitle(found)

        density = FakeDensity()
        result, _ = _survey(settings, rules, density, state=state)

        again = state.get(film)
        assert again is not None
        assert (again.decision, again.status) == ("remux", Status.PENDING)
        assert again.reasons == FOUND_REASONS
        assert _sub_row(state, film) == found  # untouched
        assert density.calls == []
    finally:
        state.close()
    row = _by_name(_csv_rows(result.csv_path))["Hdr.mkv"]
    assert (row["action"], row["subtitle_status"]) == ("remux", "found")
    assert row["english_subtitle"] == "none"


def test_a_found_row_without_a_cache_file_does_not_make_a_remux(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    film = _touch(settings.library_root / "Hdr.mkv")
    rules = {"Hdr.mkv": lambda p: _with(p, hdr=True)}
    _result, state = _survey(settings, rules)
    try:
        rec = state.get(film)
        assert rec is not None
        state.set_subtitle(SubtitleRecord(film, rec.size, rec.mtime, SubtitleStatus.FOUND))
        _survey(settings, rules, state=state)
        again = state.get(film)
        assert again is not None and again.decision == "flag"
    finally:
        state.close()


def test_resurvey_keeps_what_the_resolver_decided(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    gone = _touch(settings.library_root / "Gone.mkv")
    wait = _touch(settings.library_root / "Wait.mkv")
    unclear = _touch(settings.library_root / "Unclear.mkv")
    rules = {"Unclear.mkv": lambda p: _with(p, _sub(2, "und"))}
    _result, state = _survey(settings, rules)
    try:
        rec = state.get(gone)
        assert rec is not None
        state.set_subtitle(
            SubtitleRecord(
                gone,
                rec.size,
                rec.mtime,
                SubtitleStatus.UNAVAILABLE,
                reason="none",
                tried_ids=(3, 4),
            )
        )
        before = {p: _sub_row(state, p) for p in (gone, wait, unclear)}
        _survey(settings, rules, state=state)
        assert {p: _sub_row(state, p) for p in (gone, wait, unclear)} == before
        gone_before = before[gone]
        assert gone_before is not None and gone_before.tried_ids == (3, 4)
    finally:
        state.close()


def test_a_threshold_calibrated_up_sends_not_needed_back_to_wanted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = _settings(tmp_path)
    film = _touch(settings.library_root / "Forced.mkv")
    rules = {"Forced.mkv": lambda p: _with(p, _sub(3, forced=True))}
    first, state = _survey(settings, rules, FakeDensity({3: 300.0}))
    try:
        before = _sub_row(state, film)
        assert before is not None and before.status is SubtitleStatus.NOT_NEEDED
        row = _by_name(_csv_rows(first.csv_path))["Forced.mkv"]
        assert (row["english_subtitle"], row["subtitle_status"]) == ("reclassified", "not_needed")

        calibrated = dataclasses.replace(settings, forced_full_cues_per_hour=600.0)
        with caplog.at_level(logging.INFO, logger="video_cleaner.survey"):
            result, _ = _survey(calibrated, rules, FakeDensity({3: 300.0}), state=state)

        after = _sub_row(state, film)
        assert after is not None and after.status is SubtitleStatus.WANTED
        assert after.provider == "" and after.reason == "" and after.english_audio is True
    finally:
        state.close()
    row = _by_name(_csv_rows(result.csv_path))["Forced.mkv"]
    assert (row["english_subtitle"], row["subtitle_status"]) == ("none", "wanted")
    assert any("Forced.mkv" in r.getMessage() and r.levelno == logging.INFO for r in caplog.records)


def test_an_unclear_survey_row_goes_back_to_wanted_when_the_track_no_longer_classifies_so(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    film = _touch(settings.library_root / "Und.mkv")
    _result, state = _survey(settings, {"Und.mkv": lambda p: _with(p, _sub(2, "und"))})
    try:
        unclear = _sub_row(state, film)
        assert unclear is not None and unclear.status is SubtitleStatus.UNCLEAR
        result, _ = _survey(settings, {"Und.mkv": lambda p: _with(p, _sub(2, "fra"))}, state=state)
        again = _sub_row(state, film)
        assert again is not None and again.status is SubtitleStatus.WANTED
    finally:
        state.close()
    row = _by_name(_csv_rows(result.csv_path))["Und.mkv"]
    assert (row["english_subtitle"], row["subtitle_status"]) == ("none", "wanted")


def test_the_resolvers_multipart_unclear_row_is_not_reset_by_a_survey(tmp_path: Path) -> None:
    from video_cleaner.subtitles.resolve import MULTI_PART

    settings = _settings(tmp_path)
    pt1 = _touch(settings.library_root / "Mabuse pt1.mkv")
    _touch(settings.library_root / "Mabuse pt2.mkv")
    _result, state = _survey(settings, {})
    try:
        rec = state.get(pt1)
        assert rec is not None
        resolver_row = SubtitleRecord(
            pt1, rec.size, rec.mtime, SubtitleStatus.UNCLEAR, reason=MULTI_PART
        )
        state.set_subtitle(resolver_row)
        result, _ = _survey(settings, {}, state=state)
        assert _sub_row(state, pt1) == resolver_row
    finally:
        state.close()
    row = _by_name(_csv_rows(result.csv_path))["Mabuse pt1.mkv"]
    assert row["subtitle_status"] == "unclear"


def test_a_wanted_row_keeps_its_tried_ids_and_a_found_row_stays_found_on_resurvey(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    wanted = _touch(settings.library_root / "Wanted.mkv")
    found = _touch(settings.library_root / "Found.mkv")
    unavailable = _touch(settings.library_root / "Unavailable.mkv")
    _result, state = _survey(settings, {})
    try:
        stored = {}
        for path, status in (
            (wanted, SubtitleStatus.WANTED),
            (found, SubtitleStatus.FOUND),
            (unavailable, SubtitleStatus.UNAVAILABLE),
        ):
            rec = state.get(path)
            assert rec is not None
            row = SubtitleRecord(
                path,
                rec.size,
                rec.mtime,
                status,
                english_audio=True,
                tried_ids=(5, 6),
                attempts=2,
                reason="x",
            )
            state.set_subtitle(row)
            stored[path] = _sub_row(state, path)
        result, _ = _survey(settings, {}, state=state)
        assert {p: _sub_row(state, p) for p in stored} == stored
    finally:
        state.close()
    rows = _by_name(_csv_rows(result.csv_path))
    assert rows["Found.mkv"]["subtitle_status"] == "found"
    assert rows["Wanted.mkv"]["subtitle_status"] == "wanted"
    assert rows["Unavailable.mkv"]["subtitle_status"] == "unavailable"


def test_the_csv_never_contradicts_a_kept_found_row(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    film = _touch(settings.library_root / "Und.mkv")
    rules = {"Und.mkv": lambda p: _with(p, _sub(2, "und"))}
    _result, state = _survey(settings, {})
    try:
        rec = state.get(film)
        assert rec is not None
        state.set_subtitle(SubtitleRecord(film, rec.size, rec.mtime, SubtitleStatus.FOUND))
        result, _ = _survey(settings, rules, state=state)  # no cache file: classified again
        stored = _sub_row(state, film)
        assert stored is not None and stored.status is SubtitleStatus.FOUND
    finally:
        state.close()
    row = _by_name(_csv_rows(result.csv_path))["Und.mkv"]
    assert (row["english_subtitle"], row["subtitle_status"]) == ("none", "found")


def test_a_changed_file_gets_a_new_subtitle_row(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    film = _touch(settings.library_root / "Film.mkv")
    _result, state = _survey(settings, {})
    try:
        state.set_subtitle(
            SubtitleRecord(
                film, 6 * GIB, 1_700_000_000.0, SubtitleStatus.FOUND, cache_path=tmp_path
            )
        )
        # The file was replaced: another size, so another key.
        _survey(settings, {"Film.mkv": lambda p: _with(p, size=5 * GIB)}, state=state)
        counts = state.subtitle_counts()
        assert counts[SubtitleStatus.FOUND] == 1 and counts[SubtitleStatus.WANTED] == 1
    finally:
        state.close()


# --- files that no output can come from ---------------------------------------------------------


def test_blocked_flags_get_no_subtitle_row_and_spend_no_quota(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    lib = settings.library_root
    audio_only = _touch(lib / "Audio Only.mkv")
    _touch(lib / "Garbage.mkv")
    density = FakeDensity({3: 400.0})

    def unreadable(path: Path) -> MediaInfo:
        raise ProbeError(path, "Invalid data found when processing input")

    result, state = _survey(
        settings,
        {
            "Audio Only.mkv": lambda p: _with(p, _sub(3, forced=True), has_video=False),
            "Garbage.mkv": unreadable,
        },
        density,
    )
    try:
        assert sum(state.subtitle_counts().values()) == 0
        rec = state.get(audio_only)
        assert rec is not None and rec.reasons == ("no video stream",)
    finally:
        state.close()
    assert density.calls == []  # not even measured
    rows = _by_name(_csv_rows(result.csv_path))
    for name in ("Audio Only.mkv", "Garbage.mkv"):
        assert (rows[name]["english_subtitle"], rows[name]["subtitle_status"]) == ("", "")


def test_a_group_that_cannot_be_joined_gets_no_subtitle_rows(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    lib = settings.library_root
    _touch(lib / "Mismatch pt1.mkv")
    _touch(lib / "Mismatch pt2.mkv")
    _touch(lib / "Broken pt1.mkv")
    _touch(lib / "Broken pt2.mkv")

    def broken(path: Path) -> MediaInfo:
        raise ProbeError(path, "truncated")

    _result, state = _survey(
        settings,
        {
            "Mismatch pt2.mkv": lambda p: dataclasses.replace(_with(p), audio=()),
            "Broken pt2.mkv": broken,
        },
    )
    try:
        assert sum(state.subtitle_counts().values()) == 0
    finally:
        state.close()


def test_unrepresentable_path_gets_no_subtitle_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    good = _touch(settings.library_root / "good.mkv")
    bad = settings.library_root / "bad \udcff name.mkv"
    monkeypatch.setattr(survey_module, "discover_files", lambda root: [bad, good])

    result, state = _survey(settings, {})
    try:
        assert result.unrepresentable == (bad,)
        assert state.subtitle_counts()[SubtitleStatus.WANTED] == 1  # the good file only
    finally:
        state.close()


# --- multi-part groups --------------------------------------------------------------------------


def test_a_multipart_group_gets_one_subtitle_row_on_its_first_part(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    pt1 = _touch(settings.library_root / "Mabuse pt1.mkv")
    pt2 = _touch(settings.library_root / "Mabuse pt2.mkv")
    density = FakeDensity({3: 400.0})
    rules = {
        "Mabuse pt1.mkv": lambda p: _with(p, _sub(3, forced=True)),
        "Mabuse pt2.mkv": lambda p: _with(p, _sub(3, forced=True)),
    }

    result, state = _survey(settings, rules, density)
    try:
        first = _sub_row(state, pt1)
        assert first is not None and first.status is SubtitleStatus.NOT_NEEDED
        rec2 = state.get(pt2)
        assert rec2 is not None
        assert state.subtitle(pt2, rec2.size, rec2.mtime) is None
        assert sum(state.subtitle_counts().values()) == 1
    finally:
        state.close()
    assert [c[0] for c in density.calls] == [pt1]  # the first part stands for the group
    rows = _by_name(_csv_rows(result.csv_path))
    assert rows["Mabuse pt1.mkv"]["english_subtitle"] == "reclassified"
    assert rows["Mabuse pt2.mkv"]["english_subtitle"] == ""
    assert rows["Mabuse pt2.mkv"]["subtitle_status"] == ""


def test_a_flagged_multipart_group_becomes_a_remux_with_its_subtitle(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    pt1 = _touch(settings.library_root / "Hdr pt1.mkv")
    _touch(settings.library_root / "Hdr pt2.mkv")
    rules = {
        "Hdr pt1.mkv": lambda p: _with(p, hdr=True),
        "Hdr pt2.mkv": lambda p: _with(p, hdr=True),
    }
    _result, state = _survey(settings, rules)
    try:
        first = state.get(pt1)
        assert first is not None and first.decision == "flag"
        state.set_subtitle(
            SubtitleRecord(
                pt1, first.size, first.mtime, SubtitleStatus.FOUND, cache_path=tmp_path / "s.srt"
            )
        )
        _survey(settings, rules, state=state)
        again = state.get(pt1)
        assert again is not None and again.decision == "remux"
        assert again.reasons[-2:] == ("new subtitles to mux in", "multi-part file to join")
    finally:
        state.close()


# --- the CSV and the run -----------------------------------------------------------------------


def test_csv_ends_with_the_two_subtitle_columns() -> None:
    assert CSV_COLUMNS[-2:] == ("english_subtitle", "subtitle_status")
    assert CSV_COLUMNS[:17] == (
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


def test_a_missing_ffmpeg_aborts_the_survey(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _touch(settings.library_root / "Signs.mkv")

    class NoFfmpeg(FakeDensity):
        def __call__(self, *args: object, **kwargs: object) -> float:
            raise ToolNotFound("ffmpeg")

    state = State.open(settings.state_db)
    try:
        with pytest.raises(ToolNotFound):
            _survey(
                settings,
                {"Signs.mkv": lambda p: _with(p, _sub(3, forced=True))},
                NoFfmpeg(),
                state=state,
            )
    finally:
        state.close()


def test_the_default_density_function_is_the_embedded_measurement() -> None:
    from video_cleaner.subtitles.embedded import measure_cue_density

    assert survey_module.DEFAULT_DENSITY_FN is measure_cue_density


# --- re-deciding a file whose subtitle was just found --------------------------------------------


def _hevc(path: Path) -> MediaInfo:
    return _info(path, codec="hevc")


def test_an_hevc_file_without_english_subtitles_is_wanted_and_becomes_a_remux_once_found(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    film = _touch(settings.library_root / "Efficient.mkv")
    rules: dict[str, MediaInfoRule] = {"Efficient.mkv": _hevc}
    result, state = _survey(settings, rules)
    try:
        sub = _sub_row(state, film)
        assert sub is not None and sub.status is SubtitleStatus.WANTED
        rec = state.get(film)
        assert rec is not None and (rec.decision, rec.status) == ("leave_alone", Status.LEFT_ALONE)
        row = _by_name(_csv_rows(result.csv_path))["Efficient.mkv"]
        assert (row["action"], row["english_subtitle"], row["subtitle_status"]) == (
            "leave_alone",
            "none",
            "wanted",
        )

        found = SubtitleRecord(
            film,
            rec.size,
            rec.mtime,
            SubtitleStatus.FOUND,
            provider="opensubtitles",
            cache_path=tmp_path / "cache" / "Efficient.srt",
            sha256="cd" * 32,
            english_audio=True,
        )
        state.set_subtitle(found)
        again, _ = _survey(settings, rules, state=state)

        rec = state.get(film)
        assert rec is not None and (rec.decision, rec.status) == ("remux", Status.PENDING)
        assert "new subtitles to mux in" in rec.reasons
        assert _sub_row(state, film) == found
        row = _by_name(_csv_rows(again.csv_path))["Efficient.mkv"]
        assert (row["action"], row["subtitle_status"]) == ("remux", "found")
    finally:
        state.close()


def test_redecide_found_turns_a_left_alone_file_into_a_pending_remux(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    film = _touch(settings.library_root / "Done.mkv")
    _result, state = _survey(settings, {"Done.mkv": _hevc})
    try:
        rec = state.get(film)
        assert rec is not None and (rec.decision, rec.status) == ("leave_alone", Status.LEFT_ALONE)

        decision = survey_module.redecide_found(
            _hevc(film), settings, state, movie_hash="0123456789abcdef"
        )

        assert decision.action is Action.REMUX
        again = state.get(film)
        assert again is not None
        assert (again.decision, again.status) == ("remux", Status.PENDING)
        assert "new subtitles to mux in" in again.reasons
        assert again.movie_hash == "0123456789abcdef"
    finally:
        state.close()


def test_redecide_found_turns_a_flagged_hdr_file_into_a_remux(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    film = _touch(settings.library_root / "Hdr.mkv")
    rules = {"Hdr.mkv": lambda p: _with(p, hdr=True)}
    _result, state = _survey(settings, rules)
    try:
        survey_module.redecide_found(_with(film, hdr=True), settings, state, movie_hash=None)
        again = state.get(film)
        assert again is not None
        assert (again.decision, again.status) == ("remux", Status.PENDING)
        assert again.reasons == FOUND_REASONS
    finally:
        state.close()


def test_redecide_found_keeps_an_encode_an_encode(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    film = _touch(settings.library_root / "Big.mkv")
    _result, state = _survey(settings, {})
    try:
        decision = survey_module.redecide_found(_info(film), settings, state, movie_hash=None)
        assert decision.action is Action.ENCODE
        again = state.get(film)
        assert again is not None
        assert (again.decision, again.status) == ("encode", Status.PENDING)
        assert again.reasons[-1] == "new subtitles to mux in"
        assert again.est_saving_bytes > 0
    finally:
        state.close()


def test_redecide_found_leaves_a_record_a_run_has_decided_on(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    film = _touch(settings.library_root / "Done.mkv")
    _result, state = _survey(settings, {"Done.mkv": _hevc})
    try:
        state.set_status(film, Status.LEFT_ALONE, "output not smaller")
        survey_module.redecide_found(_hevc(film), settings, state, movie_hash=None)
        again = state.get(film)
        assert again is not None
        assert (again.decision, again.status, again.note) == (
            "leave_alone",
            Status.LEFT_ALONE,
            "output not smaller",
        )
    finally:
        state.close()


@pytest.mark.parametrize(
    ("cues_per_hour", "status"),
    [(186.0, SubtitleStatus.NOT_NEEDED), (92.0, SubtitleStatus.WANTED)],
)
def test_the_default_threshold_tells_intertitles_from_signs(
    tmp_path: Path, cues_per_hour: float, status: SubtitleStatus
) -> None:
    settings = _settings(tmp_path)
    film = _touch(settings.library_root / "Silent.mkv")

    _result, state = _survey(
        settings,
        {"Silent.mkv": lambda p: _with(p, _sub(3, forced=True))},
        FakeDensity({3: cues_per_hour}),
    )
    try:
        sub = _sub_row(state, film)
        assert sub is not None and sub.status is status
    finally:
        state.close()
