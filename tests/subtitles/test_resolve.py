"""The subtitle resolver: sidecars, local folders, cached records, guards and the sweep."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import re
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from resolve_fakes import (
    Case,
    FakeAligner,
    FakeClient,
    failing,
    make_case,
    raising,
    result,
    srt_bytes,
)
from video_cleaner.state import SubtitleRecord, SubtitleStatus
from video_cleaner.subtitles.local import index_local_folders
from video_cleaner.subtitles.resolve import (
    sidecar_candidates,
    sweep_align_workdirs,
)
from video_cleaner.subtitles.textfmt import normalise_subtitle
from video_cleaner.tools import ToolNotFound

CACHE_NAME = re.compile(r"^[0-9a-f]{16}-(sidecar|local|opensubtitles)-[0-9a-z]+\.srt$")
NO_CREDENTIALS = "unavailable: no OpenSubtitles credentials"


def normalised(raw: bytes) -> str:
    return normalise_subtitle(raw, max_bytes=2_000_000, min_cues=20).srt


def write_sidecar(case: Case, suffix: str, data: bytes) -> Path:
    path = case.video.with_name(case.video.stem + suffix)
    path.write_bytes(data)
    return path


def local_case(
    tmp_path: Path, data: bytes, *, name: str = "The Great Movie 2010.srt", **kw: object
) -> Case:
    folder = tmp_path / "local-subs"
    folder.mkdir()
    (folder / name).write_bytes(data)
    index = index_local_folders([folder], max_bytes=2_000_000)
    assert index
    return make_case(tmp_path, local_index=index, **kw)  # type: ignore[arg-type]


def cached_text(case: Case, rec: SubtitleRecord) -> str:
    assert rec.cache_path is not None
    return rec.cache_path.read_text(encoding="utf-8")


# --- sidecar_candidates ---------------------------------------------------------------------


def test_sidecars_are_the_same_stem_with_an_optional_english_part(tmp_path: Path) -> None:
    folder = tmp_path / "v"
    folder.mkdir()
    for name in ("Movie.en.srt", "Movie.eng.vtt", "Movie.ass", "Movie.fr.srt", "Movie2.srt"):
        (folder / name).write_text("x")
    (folder / "Movie.sub").write_text("x")
    (folder / "Movie.en.fr.srt").write_text("x")
    found = sidecar_candidates(folder / "Movie.mkv")
    assert [p.name for p in found] == ["Movie.ass", "Movie.en.srt", "Movie.eng.vtt"]
    assert all(p.parent == folder for p in found)


def test_sidecars_match_the_stem_case_insensitively_and_are_sorted(tmp_path: Path) -> None:
    folder = tmp_path / "v"
    folder.mkdir()
    for name in ("MOVIE.EN.SRT", "movie.srt"):
        (folder / name).write_text("x")
    found = sidecar_candidates(folder / "Movie.mkv")
    assert [p.name for p in found] == sorted(p.name for p in found)
    assert len(found) == 2


def test_sidecars_are_at_most_three(tmp_path: Path) -> None:
    folder = tmp_path / "v"
    folder.mkdir()
    for name in ("Movie.srt", "Movie.ass", "Movie.vtt", "Movie.en.srt", "Movie.eng.srt"):
        (folder / name).write_text("x")
    assert len(sidecar_candidates(folder / "Movie.mkv")) == 3


def test_sidecars_skip_directories_and_links(tmp_path: Path) -> None:
    folder = tmp_path / "v"
    folder.mkdir()
    (folder / "Movie.srt").mkdir()
    target = tmp_path / "elsewhere.srt"
    target.write_text("x")
    try:
        (folder / "Movie.en.srt").symlink_to(target)
    except OSError:
        pytest.skip("cannot create symlinks here")
    assert sidecar_candidates(folder / "Movie.mkv") == []


def test_sidecars_of_an_unreadable_folder_are_none(tmp_path: Path) -> None:
    assert sidecar_candidates(tmp_path / "no-such-folder" / "Movie.mkv") == []


# --- the order: sidecar, local, online ------------------------------------------------------


def test_a_sidecar_wins_over_local_and_online(tmp_path: Path) -> None:
    client = FakeClient(
        hash_results=[result(1, hash_match=True)], downloads={1: srt_bytes(tag="os")}
    )
    case = local_case(tmp_path, srt_bytes(tag="local"), client=client)
    write_sidecar(case, ".en.srt", srt_bytes(tag="sidecar"))
    out = case.resolve()
    assert out.status is SubtitleStatus.FOUND
    assert out.provider == "sidecar"
    assert out.provenance == "English (sidecar)"
    assert client.calls == 0
    assert "sidecar" in cached_text(case, out)


def test_local_wins_over_online_and_makes_no_client_call(tmp_path: Path) -> None:
    client = FakeClient(
        hash_results=[result(1, hash_match=True)], downloads={1: srt_bytes(tag="os")}
    )
    case = local_case(tmp_path, srt_bytes(tag="local"), client=client)
    out = case.resolve()
    assert out.status is SubtitleStatus.FOUND
    assert out.provider == "local"
    assert out.provenance == "English (local file)"
    assert client.calls == 0
    assert len(case.aligner.calls) == 1


def test_a_found_record_is_complete_and_the_cache_holds_the_normalised_text(
    tmp_path: Path,
) -> None:
    ass = (
        "[Script Info]\nScriptType: v4.00+\n\n[V4+ Styles]\nFormat: Name\n\n[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        + "".join(
            f"Dialogue: 0,0:00:{n:02d}.00,0:00:{n:02d}.90,Default,,0,0,0,,"
            f"{{\\i1}}I don't know what you are talking about, but we can get that from him. {n}\n"
            for n in range(1, 31)
        )
    ).encode()
    case = make_case(tmp_path, english_audio=True)
    write_sidecar(case, ".ass", ass)
    out = case.resolve()
    assert out.status is SubtitleStatus.FOUND
    assert out.english_audio is True
    assert out.hash_exact is False
    assert out.cache_path is not None
    assert out.cache_path.parent == case.cache_dir
    assert CACHE_NAME.match(out.cache_path.name)
    text = cached_text(case, out)
    assert "-->" in text and "Dialogue" not in text and "{" not in text
    assert out.sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert (out.path, out.size, out.mtime) == (case.sub.path, case.sub.size, case.sub.mtime)
    # The aligner was given the normalised SRT, not the ASS file.
    assert "Dialogue" not in case.aligner.calls[0].subtitle_text
    assert case.aligner.calls[0].media == case.info.path
    assert case.aligner.calls[0].hash_exact is False


def test_a_misaligned_local_candidate_falls_through_to_online(tmp_path: Path) -> None:
    client = FakeClient(
        hash_results=[result(7, hash_match=True)], downloads={7: srt_bytes(tag="os")}
    )
    case = local_case(
        tmp_path, srt_bytes(tag="local"), client=client, aligner=FakeAligner(failing())
    )
    out = case.resolve()
    assert out.provider == "opensubtitles"
    assert client.download_calls == [7]


def test_a_non_english_local_file_is_rejected_by_its_content(tmp_path: Path) -> None:
    client = FakeClient(
        hash_results=[result(7, hash_match=True)], downloads={7: srt_bytes(tag="os")}
    )
    case = local_case(tmp_path, srt_bytes(german=True), client=client)
    out = case.resolve()
    assert out.provider == "opensubtitles"
    assert case.aligner.calls == []  # the hash-exact download needs no alignment either


def test_a_non_english_sidecar_is_rejected_and_the_next_source_is_tried(tmp_path: Path) -> None:
    case = local_case(tmp_path, srt_bytes(tag="local"))
    write_sidecar(case, ".srt", srt_bytes(german=True))
    out = case.resolve()
    assert out.provider == "local"


def test_a_sidecar_over_the_size_limit_is_rejected(tmp_path: Path) -> None:
    case = make_case(tmp_path, subtitle_max_bytes=1000)
    write_sidecar(case, ".srt", srt_bytes(30))
    out = case.resolve()
    assert out.status is SubtitleStatus.WANTED
    assert case.aligner.calls == []


def test_an_unreadable_local_file_rejects_that_candidate_only(tmp_path: Path) -> None:
    case = local_case(tmp_path, srt_bytes(tag="local"))
    (tmp_path / "local-subs" / "The Great Movie 2010.srt").unlink()
    out = case.resolve()
    assert out.status is SubtitleStatus.WANTED
    assert out.reason == NO_CREDENTIALS


def test_a_failing_aligner_on_a_sidecar_leaves_nothing_behind(tmp_path: Path) -> None:
    case = make_case(tmp_path, aligner=FakeAligner(failing()))
    write_sidecar(case, ".srt", srt_bytes())
    out = case.resolve()
    assert out.status is SubtitleStatus.WANTED
    assert case.cache_files() == []


def test_a_missing_tool_propagates_and_the_job_aborts(tmp_path: Path) -> None:
    case = make_case(tmp_path, aligner=FakeAligner(raising(ToolNotFound("ffsubsync"))))
    write_sidecar(case, ".srt", srt_bytes())
    with pytest.raises(ToolNotFound):
        case.resolve()


# --- no client ------------------------------------------------------------------------------


def test_without_a_client_the_record_stays_wanted_with_the_credentials_reason(
    tmp_path: Path,
) -> None:
    case = make_case(tmp_path, client=None)
    out = case.resolve()
    assert out.status is SubtitleStatus.WANTED
    assert out.reason == NO_CREDENTIALS
    assert case.cache_files() == []


def test_without_a_client_a_local_result_is_still_used(tmp_path: Path) -> None:
    case = local_case(tmp_path, srt_bytes(tag="local"), client=None)
    assert case.resolve().status is SubtitleStatus.FOUND


# --- records already decided ----------------------------------------------------------------


def test_a_second_call_with_a_cached_found_record_does_nothing(tmp_path: Path) -> None:
    client = FakeClient()
    case = local_case(tmp_path, srt_bytes(tag="local"), client=client)
    first = case.resolve()
    calls = len(case.aligner.calls)
    again = case.resolve(first)
    assert again == first
    assert len(case.aligner.calls) == calls
    assert client.calls == 0


@pytest.mark.parametrize("damage", ["missing", "changed", "outside"])
def test_a_found_record_whose_cache_file_is_gone_or_changed_is_resolved_again(
    tmp_path: Path, damage: str
) -> None:
    case = local_case(tmp_path, srt_bytes(tag="local"), client=FakeClient())
    first = case.resolve()
    assert first.cache_path is not None
    stale = replace(first, tried_ids=(5,), attempts=2)
    if damage == "missing":
        first.cache_path.unlink()
    elif damage == "changed":
        first.cache_path.write_text("tampered", encoding="utf-8")
    else:
        elsewhere = tmp_path / "elsewhere.srt"
        elsewhere.write_bytes(first.cache_path.read_bytes())
        stale = replace(stale, cache_path=elsewhere)
    again = case.resolve(stale)
    assert again.status is SubtitleStatus.FOUND
    assert again.cache_path is not None
    assert again.cache_path.parent == case.cache_dir
    assert again.sha256 == hashlib.sha256(again.cache_path.read_bytes()).hexdigest()
    assert again.tried_ids == (5,)
    assert again.attempts == 2


@pytest.mark.parametrize(
    "status", [SubtitleStatus.NOT_NEEDED, SubtitleStatus.UNAVAILABLE, SubtitleStatus.UNCLEAR]
)
def test_statuses_other_than_wanted_are_returned_unchanged(
    tmp_path: Path, status: SubtitleStatus
) -> None:
    client = FakeClient()
    case = make_case(tmp_path, client=client)
    sub = replace(case.sub, status=status, reason="decided earlier")
    assert case.resolve(sub) == sub
    assert client.calls == 0
    assert case.cache_files() == []


def test_a_multi_part_film_is_unclear_and_nothing_is_looked_up(tmp_path: Path) -> None:
    client = FakeClient()
    parts = (tmp_path / "a-pt1.mkv", tmp_path / "a-pt2.mkv")
    case = local_case(tmp_path, srt_bytes(tag="local"), client=client, parts=parts)
    out = case.resolve()
    assert out.status is SubtitleStatus.UNCLEAR
    assert (
        out.reason
        == "multi-part film: subtitle lookup needs the parts combined (not supported yet)"
    )
    assert client.calls == 0
    assert case.aligner.calls == []
    assert case.cache_files() == []


# --- the cache folder guard ----------------------------------------------------------------


def test_a_cache_folder_inside_the_library_raises_and_writes_nothing(tmp_path: Path) -> None:
    case = make_case(tmp_path, subtitle_cache_dir=tmp_path / "library" / "subs")
    write_sidecar(case, ".srt", srt_bytes())
    with pytest.raises(ValueError, match="library"):
        case.resolve()
    assert not (tmp_path / "library" / "subs").exists()
    assert case.aligner.calls == []


def test_a_library_inside_the_cache_folder_raises(tmp_path: Path) -> None:
    case = make_case(tmp_path, subtitle_cache_dir=tmp_path)
    with pytest.raises(ValueError, match="library"):
        case.resolve()


def test_a_relative_cache_folder_raises(tmp_path: Path) -> None:
    case = make_case(tmp_path, subtitle_cache_dir=Path("relative-cache"))
    write_sidecar(case, ".srt", srt_bytes())
    with pytest.raises(ValueError, match="absolute"):
        case.resolve()
    assert not Path("relative-cache").exists()


# --- logging --------------------------------------------------------------------------------


def test_steps_are_logged_and_a_hostile_file_name_is_escaped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    case = make_case(tmp_path)
    write_sidecar(case, ".srt", srt_bytes())
    with caplog.at_level(logging.DEBUG, logger="video_cleaner.subtitles.resolve"):
        case.resolve()
    messages = [
        r.getMessage() for r in caplog.records if r.name == "video_cleaner.subtitles.resolve"
    ]
    assert any("sidecar" in m and "accept" in m.lower() for m in messages)
    assert all("\n" not in m and "\r" not in m for m in messages)
    assert case.reporter.stages


# --- sweep_align_workdirs -------------------------------------------------------------------


def test_the_sweep_removes_only_align_work_folders(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / ".align-abc123").mkdir()
    (cache / ".align-abc123" / "input.srt").write_text("x")
    (cache / ".align-second").mkdir()
    (cache / ".align-file").write_text("a file, not a folder")
    (cache / "keep-folder").mkdir()
    (cache / "0123456789abcdef-local-abc.srt").write_text("final")
    target = tmp_path / "linked-target"
    target.mkdir()
    (target / "precious.txt").write_text("keep")
    with contextlib.suppress(OSError):
        (cache / ".align-link").symlink_to(target, target_is_directory=True)
    removed = sweep_align_workdirs(cache)
    assert removed == 2
    assert not (cache / ".align-abc123").exists()
    assert not (cache / ".align-second").exists()
    assert (cache / ".align-file").is_file()
    assert (cache / "keep-folder").is_dir()
    assert (cache / "0123456789abcdef-local-abc.srt").is_file()
    assert (target / "precious.txt").is_file()
    if os.path.lexists(cache / ".align-link"):
        assert (cache / ".align-link").is_symlink()


@pytest.mark.skipif(os.name != "nt", reason="junctions exist on Windows only")
def test_the_sweep_never_follows_a_junction(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    target = tmp_path / "junction-target"
    target.mkdir()
    (target / "precious.txt").write_text("keep")
    link = cache / ".align-junction"
    made = subprocess.run(  # noqa: S603 - fixed arguments, test paths
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],  # noqa: S607
        capture_output=True,
        check=False,
    )
    if made.returncode != 0:
        pytest.skip("cannot create a junction here")
    assert sweep_align_workdirs(cache) == 0
    assert (target / "precious.txt").is_file()


def test_the_sweep_of_a_missing_folder_is_zero(tmp_path: Path) -> None:
    assert sweep_align_workdirs(tmp_path / "nope") == 0


def test_the_sweep_skips_a_folder_it_cannot_remove(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    (cache / ".align-a").mkdir(parents=True)
    (cache / ".align-b").mkdir()

    real = shutil.rmtree

    def flaky(path: object, *args: object, **kwargs: object) -> None:
        if str(path).endswith(".align-a"):
            raise PermissionError("locked")
        real(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("video_cleaner.subtitles.resolve.shutil.rmtree", flaky)
    assert sweep_align_workdirs(cache) == 1
    assert (cache / ".align-a").exists()
    assert not (cache / ".align-b").exists()


def test_the_resolver_itself_does_not_sweep(tmp_path: Path) -> None:
    case = make_case(tmp_path)
    case.cache_dir.mkdir()
    (case.cache_dir / ".align-old").mkdir()
    case.resolve()
    assert (case.cache_dir / ".align-old").is_dir()
