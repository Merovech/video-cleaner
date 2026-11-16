"""mkvmerge command construction (pure)."""

from __future__ import annotations

from pathlib import Path

import pytest

from video_cleaner.mux import build_mux_cmd
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
    assert cmd[video_at - 3 : video_at] == ["--no-chapters", "--no-global-tags", "--no-attachments"]
    for flag in ("--no-chapters", "--no-global-tags", "--no-attachments"):
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
