"""Multi-part movie detection: grouping by file name and stream-layout compatibility."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from video_cleaner.multipart import PartGroup, find_part_groups, parts_compatible
from video_cleaner.probe import AudioStream, MediaInfo, SubtitleStream, VideoStream

FOLDER = Path("//nas/media/Movies/Dr. Mabuse the Gambler (1922)")


def _video(**changes: object) -> VideoStream:
    base = VideoStream(
        codec="h264",
        width=1920,
        height=1080,
        pix_fmt="yuv420p",
        fps=23.976,
        bit_rate=None,
        transfer=None,
        dv_profile=None,
        field_order="progressive",
    )
    return replace(base, **changes)  # type: ignore[arg-type]


def _audio(index: int = 1, codec: str = "ac3", language: str = "eng") -> AudioStream:
    return AudioStream(index=index, codec=codec, language=language, title="")


def _sub(index: int = 2, codec: str = "subrip", language: str = "eng") -> SubtitleStream:
    return SubtitleStream(
        index=index, codec=codec, language=language, forced=False, sdh=False, is_image=False
    )


def _info(
    name: str = "a.mkv",
    video: VideoStream | None = None,
    audio: tuple[AudioStream, ...] = (_audio(),),
    subtitles: tuple[SubtitleStream, ...] = (_sub(),),
    has_video: bool = True,
) -> MediaInfo:
    return MediaInfo(
        path=Path(name),
        size=1000,
        mtime=0.0,
        duration_s=60.0,
        container="matroska,webm",
        video=(video or _video()) if has_video else None,
        audio=audio,
        subtitles=subtitles,
        chapters=0,
        cover_art=0,
    )


def test_mabuse_pt1_pt2_grouped_with_combined_name() -> None:
    pt1 = FOLDER / "Dr. Mabuse the Gambler (1922) - pt1.mkv"
    pt2 = FOLDER / "Dr. Mabuse the Gambler (1922) - pt2.mkv"

    groups = find_part_groups([pt2, pt1])

    assert groups == [PartGroup(output_name="Dr. Mabuse the Gambler (1922).mkv", parts=(pt1, pt2))]


def test_part_numbers_must_start_at_1_and_be_consecutive() -> None:
    folder = Path("lib")
    gap = [folder / "Gap pt1.mkv", folder / "Gap pt3.mkv"]
    late_start = [folder / "Late pt2.mkv", folder / "Late pt3.mkv"]
    duplicate = [folder / "Dup pt1.mkv", folder / "Dup pt1.mp4", folder / "Dup pt2.mkv"]
    zero = [folder / "Zero pt0.mkv", folder / "Zero pt1.mkv"]
    good = [folder / "Good pt3.mkv", folder / "Good pt1.mkv", folder / "Good pt2.mkv"]

    groups = find_part_groups([*gap, *late_start, *duplicate, *zero, *good])

    assert groups == [
        PartGroup(
            output_name="Good.mkv",
            parts=(folder / "Good pt1.mkv", folder / "Good pt2.mkv", folder / "Good pt3.mkv"),
        )
    ]


def test_lone_pt2_is_not_grouped() -> None:
    assert find_part_groups([Path("lib/Movie pt2.mkv")]) == []
    assert find_part_groups([Path("lib/Movie pt1.mkv")]) == []


def test_unrelated_files_not_grouped() -> None:
    files = [
        Path("lib/Alien (1979).mkv"),
        Path("lib/Aliens (1986).mkv"),
        Path("lib/Concept1.mkv"),  # "pt1" inside a word is not a marker
        Path("lib/Concept2.mkv"),
        Path("lib/Apart 1.mkv"),
        Path("lib/Apart 2.mkv"),
        Path("lib/One pt1.mkv"),  # different titles
        Path("lib/Two pt2.mkv"),
        Path("lib/a/Split pt1.mkv"),  # same title, different folders
        Path("lib/b/Split pt2.mkv"),
        Path("lib/pt1.mkv"),  # marker only, no title
        Path("lib/pt2.mkv"),
        Path("lib/Rocky 1.mkv"),  # bare numbers are not part markers
        Path("lib/Rocky 2.mkv"),
        Path("lib/Big pt100.mkv"),
        Path("lib/Big pt101.mkv"),
    ]

    assert find_part_groups(files) == []
    assert find_part_groups([]) == []


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ("Film pt1.mkv", "Film pt2.mkv", "Film.mkv"),
        ("Film PT1.mkv", "Film Pt2.mkv", "Film.mkv"),
        ("Film part1.mkv", "Film part2.mkv", "Film.mkv"),
        ("Film Part 1.mkv", "Film Part 2.mkv", "Film.mkv"),
        ("Film.cd1.avi", "Film.cd2.avi", "Film.mkv"),
        ("Film_CD_1.mkv", "Film_CD_2.mkv", "Film.mkv"),
        ("Film (2001) - part.01.mkv", "Film (2001) - part.02.mkv", "Film (2001).mkv"),
        ("Film (2001)pt1.mkv", "Film (2001)pt2.mkv", "Film (2001).mkv"),
        ("Film pt1.avi", "Film pt2.mp4", "Film.mkv"),  # mixed extensions
        ("film pt1.mkv", "Film pt2.mkv", "film.mkv"),  # base compared case-insensitively
        ("Amélie (2001) - pt1.mkv", "Amélie (2001) - pt2.mkv", "Amélie (2001).mkv"),
        ("[Grp] Film [1080p] cd1.mkv", "[Grp] Film [1080p] cd2.mkv", "[Grp] Film [1080p].mkv"),
    ],
)
def test_marker_variants(first: str, second: str, expected: str) -> None:
    a, b = Path("lib") / first, Path("lib") / second

    assert find_part_groups([a, b]) == [PartGroup(output_name=expected, parts=(a, b))]


def test_odd_names_never_raise() -> None:
    files = [
        Path("lib/The First Omen (2024).en.MX]"),
        Path("lib/The First Omen (2024).mkv"),
        Path("lib/[[(("),
        Path("lib/日本語 pt1.mkv"),
        Path("lib/日本語 pt2.mkv"),
        Path("lib/trailing dots pt1..."),
        Path("lib/.hidden"),
        Path("lib/no extension part 1"),
        Path("lib/no extension part 2"),
        Path("//nas/media/" + "x" * 300 + " pt1.mkv"),
        Path("pt1"),
        Path("."),
    ]

    groups = find_part_groups(files)

    assert [g.output_name for g in groups] == ["no extension.mkv", "日本語.mkv"]


def test_groups_are_ordered_by_folder_then_name() -> None:
    files = [
        Path("lib/b/Zed pt1.mkv"),
        Path("lib/b/Zed pt2.mkv"),
        Path("lib/b/Abc cd2.mkv"),
        Path("lib/b/Abc cd1.mkv"),
        Path("lib/a/Mid part 2.mkv"),
        Path("lib/a/Mid part 1.mkv"),
    ]

    forward = find_part_groups(files)

    assert [(g.parts[0].parent, g.output_name) for g in forward] == [
        (Path("lib/a"), "Mid.mkv"),
        (Path("lib/b"), "Abc.mkv"),
        (Path("lib/b"), "Zed.mkv"),
    ]
    assert find_part_groups(list(reversed(files))) == forward


def test_ten_or_more_parts_sort_numerically() -> None:
    files = [Path(f"lib/Long pt{n}.mkv") for n in range(11, 0, -1)]

    (group,) = find_part_groups(files)

    assert group.parts == tuple(Path(f"lib/Long pt{n}.mkv") for n in range(1, 12))


def test_compatible_parts_return_true() -> None:
    a = _info("a.mkv", video=_video(fps=24000 / 1001))
    b = _info("b.mkv", video=_video(fps=23.976, bit_rate=5_000_000))

    assert parts_compatible(a, b) is True
    assert parts_compatible(_info(audio=(), subtitles=()), _info(audio=(), subtitles=())) is True


@pytest.mark.parametrize(
    "other",
    [
        _info(video=_video(codec="mpeg4")),
        _info(video=_video(width=1280)),
        _info(video=_video(height=800)),
        _info(video=_video(pix_fmt="yuv420p10le")),
        _info(video=_video(fps=25.0)),
        _info(video=_video(fps=23.98)),
        _info(has_video=False),
        _info(audio=(_audio(), _audio(index=2))),
        _info(audio=()),
        _info(audio=(_audio(codec="dts"),)),
        _info(audio=(_audio(language="deu"),)),
        _info(subtitles=()),
        _info(subtitles=(_sub(codec="hdmv_pgs_subtitle"),)),
        _info(subtitles=(_sub(language="fra"),)),
    ],
)
def test_incompatible_parts_return_false(other: MediaInfo) -> None:
    reference = _info()

    assert parts_compatible(reference, other) is False
    assert parts_compatible(other, reference) is False


def test_stream_order_matters_for_compatibility() -> None:
    a = _info(audio=(_audio(1, "ac3", "eng"), _audio(2, "aac", "deu")))
    b = _info(audio=(_audio(1, "aac", "deu"), _audio(2, "ac3", "eng")))

    assert parts_compatible(a, b) is False


def test_two_files_without_video_are_incompatible() -> None:
    assert parts_compatible(_info(has_video=False), _info(has_video=False)) is False
