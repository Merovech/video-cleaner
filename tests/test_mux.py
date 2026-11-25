"""mkvmerge command construction (pure)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from video_cleaner.mux import SubtitleInput, build_mux_cmd
from video_cleaner.tools import Tools

MKVMERGE = Path("C:/Program Files/MKVToolNix/mkvmerge.exe")


def _tools() -> Tools:
    return Tools(ffmpeg=Path("f"), ffprobe=Path("p"), mkvmerge=MKVMERGE)


def test_mux_cmd_uses_new_video_and_source_without_video() -> None:
    cmd = build_mux_cmd(_tools(), Path("out.mkv"), [Path("src.mkv")], Path("video.mkv"))
    assert cmd[0] == str(MKVMERGE)
    assert cmd[cmd.index("-o") + 1] == str(Path("out.mkv"))
    assert "--gui-mode" in cmd
    # new video first, then --no-video applying to the source that follows it
    assert cmd.index(str(Path("video.mkv"))) < cmd.index("--no-video")
    assert cmd[cmd.index("--no-video") + 1] == str(Path("src.mkv"))
    assert cmd.count("--no-video") == 1


def test_mux_cmd_new_video_contributes_only_its_video_track() -> None:
    cmd = build_mux_cmd(_tools(), Path("out.mkv"), [Path("src.mkv")], Path("video.mkv"))
    video_at = cmd.index(str(Path("video.mkv")))
    # mkvmerge options apply to the file that follows them
    assert cmd[video_at - 4 : video_at] == [
        "--no-chapters",
        "--no-global-tags",
        "--no-track-tags",
        "--no-attachments",
    ]
    # the intermediate's track statistics tags never reach the output
    for flag in ("--no-chapters", "--no-global-tags", "--no-track-tags", "--no-attachments"):
        assert cmd.count(flag) == 1


def test_mux_cmd_keeps_everything_else_from_the_source() -> None:
    cmd = build_mux_cmd(_tools(), Path("out.mkv"), [Path("src.mkv")], Path("video.mkv"))
    after_source = cmd[cmd.index("--no-video") :]
    assert after_source == ["--no-video", str(Path("src.mkv"))]
    cmd = build_mux_cmd(_tools(), Path("out.mkv"), [Path("a.mkv"), Path("b.mkv")], None)
    for dropping in (
        "--no-audio",
        "--no-subtitles",
        "--no-chapters",
        "--no-attachments",
        "--no-global-tags",
        "--no-track-tags",
    ):
        assert dropping not in cmd


def test_mux_cmd_with_new_video_exact() -> None:
    cmd = build_mux_cmd(_tools(), Path("out.mkv"), [Path("src.mkv")], Path("video.mkv"))
    assert cmd == [
        str(MKVMERGE),
        "--gui-mode",
        "-o",
        str(Path("out.mkv")),
        "--no-chapters",
        "--no-global-tags",
        "--no-track-tags",
        "--no-attachments",
        str(Path("video.mkv")),
        "--no-video",
        str(Path("src.mkv")),
    ]


def test_mux_cmd_appends_parts_with_plus() -> None:
    parts = [Path("a.mkv"), Path("b.mkv"), Path("c.mkv")]
    cmd = build_mux_cmd(_tools(), Path("out.mkv"), parts, None)
    assert cmd[cmd.index("-o") + 1] == str(Path("out.mkv"))
    assert "--gui-mode" in cmd
    assert "--no-video" not in cmd
    assert cmd[cmd.index(str(Path("a.mkv"))) :] == [
        str(Path("a.mkv")),
        "+" + str(Path("b.mkv")),
        "+" + str(Path("c.mkv")),
    ]


def test_mux_cmd_single_source_without_new_video_is_plain_remux() -> None:
    cmd = build_mux_cmd(_tools(), Path("out.mkv"), [Path("a.mkv")], None)
    assert cmd == [
        str(MKVMERGE),
        "--gui-mode",
        "-o",
        str(Path("out.mkv")),
        str(Path("a.mkv")),
    ]


def test_mux_cmd_output_flags_precede_inputs() -> None:
    cmd = build_mux_cmd(_tools(), Path("out.mkv"), [Path("a.mkv"), Path("b.mkv")], None)
    assert cmd.index("--gui-mode") < cmd.index(str(Path("a.mkv")))
    assert cmd.index("-o") < cmd.index(str(Path("a.mkv")))


def test_mux_cmd_odd_paths_stay_single_arguments() -> None:
    src = Path("D:/scratch/The First Omen (2024).en.MX].mkv")
    out = Path("D:/out/a b.mkv")
    cmd = build_mux_cmd(_tools(), out, [src], Path("D:/scratch/v [1].mkv"))
    assert str(src) in cmd
    assert str(out) in cmd
    assert all(isinstance(part, str) for part in cmd)


def test_mux_cmd_needs_a_source() -> None:
    with pytest.raises(ValueError, match="source"):
        build_mux_cmd(_tools(), Path("out.mkv"), [], None)
    with pytest.raises(ValueError, match="source"):
        build_mux_cmd(_tools(), Path("out.mkv"), [], Path("video.mkv"))


def test_mux_cmd_new_video_with_several_sources_is_refused() -> None:
    with pytest.raises(ValueError, match="new_video"):
        build_mux_cmd(_tools(), Path("o.mkv"), [Path("a.mkv"), Path("b.mkv")], Path("v.mkv"))


# --- subtitle ------------------------------------------------------------------------------

SRT = Path("D:/scratch/work-1/subtitle.srt")


def _sub(title: str = "English (OpenSubtitles)") -> SubtitleInput:
    return SubtitleInput(path=SRT, title=title)


def _sub_options(title: str = "English (OpenSubtitles)") -> list[str]:
    return [
        "--language",
        "0:eng",
        "--track-name",
        f"0:{title}",
        "--default-track-flag",
        "0:0",
        "--forced-display-flag",
        "0:0",
        str(SRT),
    ]


def test_mux_cmd_remux_with_subtitle_exact() -> None:
    cmd = build_mux_cmd(_tools(), Path("out.mkv"), [Path("src.mkv")], None, _sub())
    assert cmd == [
        str(MKVMERGE),
        "--gui-mode",
        "-o",
        str(Path("out.mkv")),
        str(Path("src.mkv")),
        *_sub_options(),
    ]


def test_mux_cmd_new_video_with_subtitle_exact() -> None:
    cmd = build_mux_cmd(_tools(), Path("out.mkv"), [Path("src.mkv")], Path("video.mkv"), _sub())
    assert cmd == [
        str(MKVMERGE),
        "--gui-mode",
        "-o",
        str(Path("out.mkv")),
        "--no-chapters",
        "--no-global-tags",
        "--no-track-tags",
        "--no-attachments",
        str(Path("video.mkv")),
        "--no-video",
        str(Path("src.mkv")),
        *_sub_options(),
    ]


def test_mux_cmd_append_of_two_parts_with_subtitle_exact() -> None:
    cmd = build_mux_cmd(_tools(), Path("out.mkv"), [Path("a.mkv"), Path("b.mkv")], None, _sub())
    assert cmd == [
        str(MKVMERGE),
        "--gui-mode",
        "-o",
        str(Path("out.mkv")),
        str(Path("a.mkv")),
        "+" + str(Path("b.mkv")),
        *_sub_options(),
    ]


def test_mux_cmd_without_subtitle_is_unchanged_by_the_new_parameter() -> None:
    plain = build_mux_cmd(_tools(), Path("out.mkv"), [Path("src.mkv")], None)
    assert plain == build_mux_cmd(_tools(), Path("out.mkv"), [Path("src.mkv")], None, None)
    assert "--track-name" not in plain


@pytest.mark.parametrize(
    "title",
    [
        'He said "hi"',
        "-leading dash",
        "two  spaces and 'quotes'",
        "--no-video",
        "@file.json",
        "a=b",
    ],
)
def test_mux_cmd_subtitle_title_stays_one_argument(title: str) -> None:
    cmd = build_mux_cmd(_tools(), Path("out.mkv"), [Path("src.mkv")], None, _sub(title))
    assert cmd[cmd.index("--track-name") + 1] == f"0:{title}"
    assert cmd.count(f"0:{title}") == 1
    # the hostile text never became a separate argument
    assert title not in cmd


def test_mux_cmd_subtitle_options_are_the_last_arguments_and_precede_their_file() -> None:
    cmd = build_mux_cmd(_tools(), Path("out.mkv"), [Path("src.mkv")], Path("v.mkv"), _sub())
    assert cmd[-1] == str(SRT)
    assert cmd.index("--language") < cmd.index(str(SRT))
    assert cmd.index(str(Path("src.mkv"))) < cmd.index("--language")
    assert cmd.count(str(SRT)) == 1
    assert all(isinstance(part, str) for part in cmd)


def test_subtitle_input_defaults_to_english() -> None:
    assert _sub().language == "eng"


@pytest.mark.parametrize("language", ["", "en", "ENG", "engl", "e1g", "eng ", "--x"])
def test_subtitle_input_rejects_a_bad_language(language: str) -> None:
    with pytest.raises(ValueError, match="language"):
        SubtitleInput(path=SRT, title="English", language=language)


@pytest.mark.parametrize(
    "title",
    [
        "",
        "x" * 201,
        "line\nbreak",
        "tab\there",
        "nul\x00",
        "bell\x07",
        "del\x7f",
        "c1\x85",
        "line\u2028sep",  # Zl
        "para\u2029sep",  # Zp
        "rtl\u202eoverride",  # Cf
        "zero\u200bwidth",  # Cf
        "\ufeffbom",  # Cf
        "   ",
        "\u3000\u3000",  # only Zs
        "\u00a0",
    ],
)
def test_subtitle_input_rejects_a_bad_title(title: str) -> None:
    with pytest.raises(ValueError, match="title"):
        SubtitleInput(path=SRT, title=title)


def test_subtitle_input_accepts_the_longest_title() -> None:
    assert SubtitleInput(path=SRT, title="x" * 200).title == "x" * 200


def test_subtitle_input_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        _sub().title = "other"  # type: ignore[misc]
