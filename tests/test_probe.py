from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from video_cleaner import probe as probe_module
from video_cleaner.consolefmt import Level
from video_cleaner.probe import (
    MediaInfo,
    ProbeError,
    VideoStream,
    parse_ffprobe,
    probe,
)
from video_cleaner.progress import Progress
from video_cleaner.tools import ToolOutputError, ToolResult, Tools, locate_tools

FIXTURES = Path(__file__).parent / "fixtures" / "ffprobe"
SIZE = 8_000_000_000
MTIME = 1_700_000_000.5


class FakeReporter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.heartbeats: list[tuple[str, float]] = []

    def stage(self, label: str) -> None:  # pragma: no cover - unused
        pass

    def update(self, p: Progress) -> None:  # pragma: no cover - unused
        pass

    def heartbeat(self, label: str, elapsed_s: float) -> None:  # pragma: no cover - unused
        with self._lock:
            self.heartbeats.append((label, elapsed_s))

    def file_done(
        self, summary: str, level: Level = Level.INFO
    ) -> None:  # pragma: no cover - unused
        pass


def load_fixture(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    return data


def parse_fixture(name: str, path: Path | None = None) -> MediaInfo:
    return parse_ffprobe(load_fixture(name), path or Path("/lib/movie.mkv"), SIZE, MTIME)


def video_of(info: MediaInfo) -> VideoStream:
    assert info.video is not None
    return info.video


# --- parse_ffprobe -----------------------------------------------------------------------


def test_h264_1080p_sdr() -> None:
    path = Path("/lib/movie.mkv")
    info = parse_fixture("h264_1080p_sdr_mkv", path)
    video = video_of(info)
    assert (info.path, info.size, info.mtime) == (path, SIZE, MTIME)
    assert info.container == "matroska,webm"
    assert info.duration_s == pytest.approx(7928.9205)
    assert (video.codec, video.width, video.height) == ("h264", 1920, 1080)
    assert video.pix_fmt == "yuv420p"
    assert video.fps == pytest.approx(23.976, abs=1e-3)
    assert video.transfer == "bt709"
    assert video.dv_profile is None
    assert video.field_order == "progressive"
    assert video.bit_rate is None  # mkv does not carry a per-stream bit rate
    assert not info.is_hdr
    assert not info.is_interlaced
    assert info.chapters == 16
    assert info.cover_art == 0
    assert info.subtitles == ()
    assert [(a.index, a.codec, a.language) for a in info.audio] == [(1, "ac3", "eng")]
    assert info.bitrate_kbps == pytest.approx(8643.609)


def test_hevc_4k_hdr10_is_hdr() -> None:
    info = parse_fixture("hevc_4k_hdr10_mkv")
    video = video_of(info)
    assert (video.codec, video.width, video.height) == ("hevc", 3840, 2160)
    assert video.transfer == "smpte2084"
    assert video.dv_profile is None
    assert info.is_hdr
    assert [a.codec for a in info.audio] == ["truehd", "ac3", "ac3"]
    assert [a.title for a in info.audio][2] == "Commentary"


def test_hlg_is_hdr() -> None:
    data = load_fixture("h264_1080p_sdr_mkv")
    data["streams"][0]["color_transfer"] = "arib-std-b67"
    assert parse_ffprobe(data, Path("x.mkv"), SIZE, MTIME).is_hdr


def test_missing_transfer_is_not_hdr() -> None:
    data = load_fixture("h264_1080p_sdr_mkv")
    del data["streams"][0]["color_transfer"]
    info = parse_ffprobe(data, Path("x.mkv"), SIZE, MTIME)
    assert video_of(info).transfer is None
    assert not info.is_hdr


def test_dolby_vision_profile_detected() -> None:
    info = parse_fixture("hevc_dv_profile8_mkv")
    assert video_of(info).dv_profile == 8
    assert info.is_hdr  # DV profile 8 is HDR10-compatible and signals smpte2084


@pytest.mark.parametrize("order", ["tt", "bb", "tb", "bt"])
def test_interlaced_field_orders(order: str) -> None:
    data = load_fixture("h264_1080p_sdr_mkv")
    data["streams"][0]["field_order"] = order
    info = parse_ffprobe(data, Path("x.mkv"), SIZE, MTIME)
    assert info.is_interlaced
    assert video_of(info).field_order == order


def test_progressive_and_unknown_field_order_not_interlaced() -> None:
    assert not parse_fixture("h264_1080p_sdr_mkv").is_interlaced
    unknown = parse_fixture("hevc_4k_hdr10_mkv")  # hevc: ffprobe omits field_order
    assert video_of(unknown).field_order == "unknown"
    assert not unknown.is_interlaced


def test_cover_art_not_counted_as_video_stream() -> None:
    info = parse_fixture("h264_with_cover_art_mkv")
    assert info.cover_art == 1
    assert video_of(info).codec == "h264"
    assert len(info.audio) == 1


def test_cover_art_first_in_stream_order_is_still_skipped() -> None:
    data = load_fixture("h264_with_cover_art_mkv")
    data["streams"].reverse()
    info = parse_ffprobe(data, Path("x.mkv"), SIZE, MTIME)
    assert info.cover_art == 1
    assert video_of(info).codec == "h264"


def test_attachment_streams_count_as_cover_art_not_video() -> None:
    data = load_fixture("h264_1080p_sdr_mkv")
    data["streams"].append({"index": 2, "codec_type": "attachment", "codec_name": "ttf"})
    info = parse_ffprobe(data, Path("x.mkv"), SIZE, MTIME)
    assert info.cover_art == 1
    assert video_of(info).codec == "h264"


def test_second_real_video_stream_counted_as_extra() -> None:
    data = load_fixture("h264_1080p_sdr_mkv")
    second = copy.deepcopy(data["streams"][0])
    second["index"] = len(data["streams"])
    second["codec_name"] = "mjpeg"
    second["width"], second["height"] = 640, 360
    data["streams"].append(second)
    info = parse_ffprobe(data, Path("x.mkv"), SIZE, MTIME)
    assert info.extra_video_streams == 1
    assert info.cover_art == 0
    assert video_of(info).codec == "h264"
    assert video_of(info).width == 1920


def test_two_extra_real_video_streams_counted() -> None:
    data = load_fixture("h264_1080p_sdr_mkv")
    for n in range(2):
        extra = copy.deepcopy(data["streams"][0])
        extra["index"] = 10 + n
        data["streams"].append(extra)
    assert parse_ffprobe(data, Path("x.mkv"), SIZE, MTIME).extra_video_streams == 2


def test_attached_pic_cover_is_not_an_extra_video_stream() -> None:
    info = parse_fixture("h264_with_cover_art_mkv")
    assert info.cover_art == 1
    assert info.extra_video_streams == 0


def test_single_video_stream_has_no_extras() -> None:
    assert parse_fixture("h264_1080p_sdr_mkv").extra_video_streams == 0


def test_only_attached_pic_means_no_video() -> None:
    data = load_fixture("h264_with_cover_art_mkv")
    data["streams"] = [s for s in data["streams"] if s["codec_name"] != "h264"]
    info = parse_ffprobe(data, Path("x.mkv"), SIZE, MTIME)
    assert info.video is None
    assert info.cover_art == 1


def test_forced_and_sdh_flags() -> None:
    pgs = parse_fixture("h264_pgs_forced_sdh_mkv")
    by_index = {s.index: s for s in pgs.subtitles}
    assert by_index[2].forced and not by_index[2].sdh  # disposition forced
    assert not by_index[3].forced and not by_index[3].sdh  # plain English
    assert by_index[4].sdh and not by_index[4].forced  # only the title says SDH
    assert [s.language for s in pgs.subtitles] == ["eng", "eng", "eng", "fre", "spa"]

    srt = parse_fixture("hevc_subrip_sdh_disposition_mkv")
    assert [(s.index, s.forced, s.sdh) for s in srt.subtitles] == [
        (2, False, False),
        (3, False, True),  # hearing_impaired disposition
    ]


def test_pgs_is_image_subtitle() -> None:
    pgs = parse_fixture("h264_pgs_forced_sdh_mkv")
    assert pgs.subtitles
    assert all(s.codec == "hdmv_pgs_subtitle" and s.is_image for s in pgs.subtitles)
    srt = parse_fixture("hevc_subrip_sdh_disposition_mkv")
    assert srt.subtitles
    assert all(s.codec == "subrip" and not s.is_image for s in srt.subtitles)


@pytest.mark.parametrize("codec", ["dvd_subtitle", "dvb_subtitle", "xsub"])
def test_other_bitmap_subtitles_are_image(codec: str) -> None:
    data = load_fixture("hevc_subrip_sdh_disposition_mkv")
    data["streams"][2]["codec_name"] = codec
    info = parse_ffprobe(data, Path("x.mkv"), SIZE, MTIME)
    assert info.subtitles[0].is_image


def test_missing_language_is_und_and_tag_case_is_ignored() -> None:
    data = load_fixture("hevc_subrip_sdh_disposition_mkv")
    del data["streams"][2]["tags"]["language"]
    data["streams"][3]["tags"] = {"LANGUAGE": "fra", "TITLE": "Forced"}
    info = parse_ffprobe(data, Path("x.mkv"), SIZE, MTIME)
    assert info.subtitles[0].language == "und"
    assert info.subtitles[1].language == "fra"
    assert info.subtitles[1].forced


def test_no_video_stream_gives_video_none() -> None:
    info = parse_fixture("synthetic_no_video_mka")
    assert info.video is None
    assert not info.is_hdr
    assert not info.is_interlaced
    assert [a.codec for a in info.audio] == ["flac"]


def test_zero_duration_raises_probe_error() -> None:
    path = Path("/lib/broken.mkv")
    with pytest.raises(ProbeError) as excinfo:
        parse_ffprobe(load_fixture("synthetic_zero_duration_mkv"), path, SIZE, MTIME)
    assert excinfo.value.path == path
    assert "duration" in excinfo.value.reason
    assert str(path) in str(excinfo.value)


@pytest.mark.parametrize("value", [None, "N/A", "nan", "inf", "-3", ""])
def test_missing_or_invalid_duration_raises_probe_error(value: str | None) -> None:
    data = load_fixture("h264_1080p_sdr_mkv")
    if value is None:
        del data["format"]["duration"]
    else:
        data["format"]["duration"] = value
    with pytest.raises(ProbeError, match="duration"):
        parse_ffprobe(data, Path("x.mkv"), SIZE, MTIME)


@pytest.mark.parametrize("data", [{}, {"streams": []}, {"format": None}, {"format": "x"}])
def test_missing_format_raises_probe_error(data: dict[str, Any]) -> None:
    with pytest.raises(ProbeError, match="format"):
        parse_ffprobe(data, Path("x.mkv"), SIZE, MTIME)


def test_malformed_stream_raises_probe_error() -> None:
    data = load_fixture("h264_1080p_sdr_mkv")
    data["streams"].append("not a stream")
    with pytest.raises(ProbeError, match="unexpected ffprobe output"):
        parse_ffprobe(data, Path("x.mkv"), SIZE, MTIME)


def test_bitrate_falls_back_to_size_over_duration() -> None:
    data = load_fixture("h264_1080p_sdr_mkv")
    del data["format"]["bit_rate"]
    info = parse_ffprobe(data, Path("x.mkv"), 1_000_000_000, MTIME)
    assert info.bitrate_kbps == pytest.approx(1_000_000_000 * 8 / 7928.9205 / 1000)
    data["format"]["bit_rate"] = "N/A"
    assert parse_ffprobe(data, Path("x.mkv"), 1_000_000_000, MTIME).bitrate_kbps == pytest.approx(
        info.bitrate_kbps
    )


def test_unicode_and_bracket_filename() -> None:
    path = Path("/movies/The First Omen (2024).en.MX]")
    info = parse_fixture("h264_1080p_sdr_mkv", path)
    assert info.path == path
    assert info.path.name == "The First Omen (2024).en.MX]"
    unicode_path = Path("/lib/Amélie [2001] 千と千尋.mkv")
    assert parse_fixture("h264_1080p_sdr_mkv", unicode_path).path == unicode_path


def test_mp4_and_avi_fixtures() -> None:
    mp4 = parse_fixture("h264_mp4", Path("/lib/a.mp4"))
    assert mp4.container.split(",")[0] == "mov"
    assert video_of(mp4).bit_rate == 2_498_996
    assert video_of(mp4).fps == pytest.approx(23.976, abs=1e-3)
    avi = parse_fixture("mpeg4_avi", Path("/lib/a.avi"))
    assert avi.container == "avi"
    assert video_of(avi).codec == "mpeg4"
    assert avi.audio[0].language == "und"


def test_fps_falls_back_to_r_frame_rate_then_zero() -> None:
    data = load_fixture("h264_1080p_sdr_mkv")
    data["streams"][0]["avg_frame_rate"] = "0/0"
    data["streams"][0]["r_frame_rate"] = "25/1"
    assert video_of(parse_ffprobe(data, Path("x.mkv"), SIZE, MTIME)).fps == 25.0
    del data["streams"][0]["r_frame_rate"]
    data["streams"][0]["avg_frame_rate"] = "garbage"
    assert video_of(parse_ffprobe(data, Path("x.mkv"), SIZE, MTIME)).fps == 0.0


def test_parse_does_not_mutate_input() -> None:
    data = load_fixture("h264_pgs_forced_sdh_mkv")
    before = copy.deepcopy(data)
    parse_ffprobe(data, Path("x.mkv"), SIZE, MTIME)
    assert data == before


# --- probe() with run_tool stubbed ------------------------------------------------------


TOOLS = Tools(ffmpeg=Path("ffmpeg"), ffprobe=Path("C:/tools/ffprobe.exe"), mkvmerge=Path("m"))


class StubRun:
    def __init__(self, result: ToolResult | Exception) -> None:
        self.result = result
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, cmd: list[str], **kwargs: Any) -> ToolResult:
        self.calls.append((list(cmd), kwargs))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def ok_result(name: str = "h264_1080p_sdr_mkv") -> ToolResult:
    text = (FIXTURES / f"{name}.json").read_text(encoding="utf-8")
    return ToolResult(returncode=0, stdout=text, stderr_tail="")


def test_probe_runs_ffprobe_with_json_args_and_stats_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    movie = tmp_path / "movie.mkv"
    movie.write_bytes(b"x" * 1234)
    os.utime(movie, (1_600_000_000, 1_600_000_123))
    stub = StubRun(ok_result())
    monkeypatch.setattr(probe_module, "run_tool", stub)
    reporter = FakeReporter()

    info = probe(movie, TOOLS, reporter)

    cmd, kwargs = stub.calls[0]
    assert cmd[0] == str(TOOLS.ffprobe)
    assert cmd[1:8] == [
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        "-show_chapters",
    ]
    assert cmd[-1] == str(movie)
    assert kwargs["reporter"] is reporter
    assert kwargs["check"] is False
    assert (info.size, info.mtime) == (1234, 1_600_000_123)
    assert info.path == movie


def test_probe_passes_odd_filename_as_single_argument(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    movie = tmp_path / "The First Omen (2024).en.MX]"
    movie.write_bytes(b"abc")
    stub = StubRun(ok_result())
    monkeypatch.setattr(probe_module, "run_tool", stub)
    info = probe(movie, TOOLS, FakeReporter())
    assert stub.calls[0][0][-1] == str(movie)
    assert info.path.name == "The First Omen (2024).en.MX]"
    assert info.size == 3


def test_probe_nonzero_exit_raises_probe_error_with_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    movie = tmp_path / "bad.mkv"
    movie.write_bytes(b"junk")
    result = ToolResult(returncode=1, stdout="", stderr_tail="bad.mkv: Invalid data found\n")
    monkeypatch.setattr(probe_module, "run_tool", StubRun(result))
    with pytest.raises(ProbeError) as excinfo:
        probe(movie, TOOLS, FakeReporter())
    assert excinfo.value.path == movie
    assert "code 1" in excinfo.value.reason
    assert "Invalid data found" in excinfo.value.reason


@pytest.mark.parametrize("stdout", ["", "not json", '{"format": ', "[1, 2]"])
def test_probe_invalid_json_raises_probe_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stdout: str
) -> None:
    movie = tmp_path / "m.mkv"
    movie.write_bytes(b"x")
    monkeypatch.setattr(
        probe_module, "run_tool", StubRun(ToolResult(returncode=0, stdout=stdout, stderr_tail=""))
    )
    with pytest.raises(ProbeError, match="JSON"):
        probe(movie, TOOLS, FakeReporter())


def test_probe_zero_duration_raises_probe_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    movie = tmp_path / "m.mkv"
    movie.write_bytes(b"x")
    monkeypatch.setattr(probe_module, "run_tool", StubRun(ok_result("synthetic_zero_duration_mkv")))
    with pytest.raises(ProbeError, match="duration"):
        probe(movie, TOOLS, FakeReporter())


def test_probe_missing_file_raises_probe_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = StubRun(ok_result())
    monkeypatch.setattr(probe_module, "run_tool", stub)
    with pytest.raises(ProbeError, match="stat"):
        probe(tmp_path / "gone.mkv", TOOLS, FakeReporter())
    assert stub.calls == []


def test_probe_wraps_tool_output_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    movie = tmp_path / "m.mkv"
    movie.write_bytes(b"x")
    monkeypatch.setattr(
        probe_module, "run_tool", StubRun(ToolOutputError("probe", "stdout", "reader crashed"))
    )
    with pytest.raises(ProbeError, match="reader crashed"):
        probe(movie, TOOLS, FakeReporter())


def test_probe_decodes_utf8_stdout_end_to_end(tmp_path: Path) -> None:
    """Run the real run_tool with a stand-in ffprobe to prove UTF-8 JSON survives the pipe."""
    movie = tmp_path / "Amélie 千と千尋 [2001].mkv"
    movie.write_bytes(b"x" * 10)
    data = load_fixture("hevc_subrip_sdh_disposition_mkv")
    data["streams"][2]["tags"]["title"] = "Français - 千と千尋"
    payload = tmp_path / "payload.json"
    payload.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    script = tmp_path / "fake_ffprobe.py"
    script.write_text(
        f"import sys\nsys.stdout.buffer.write(open({str(payload)!r}, 'rb').read())\n",
        encoding="utf-8",
    )
    # run_tool executes tools.ffprobe directly, so use a launcher that runs the script.
    launcher = tmp_path / ("fake_ffprobe.cmd" if os.name == "nt" else "fake_ffprobe.sh")
    if os.name == "nt":
        launcher.write_text(f'@"{Path(sys.executable)}" "{script}"\n', encoding="utf-8")
    else:  # pragma: no cover - Windows is the target platform
        launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}"\n', encoding="utf-8")
        launcher.chmod(0o755)
    tools = Tools(ffmpeg=Path("ffmpeg"), ffprobe=launcher, mkvmerge=Path("m"))
    info = probe(movie, tools, FakeReporter())
    assert info.path == movie
    assert info.container == "matroska,webm"
    assert len(info.subtitles) == 2


# --- integration: real ffmpeg + ffprobe on a generated clip ------------------------------


@pytest.mark.integration
def test_probe_real_clip_with_chapters_and_cover(tmp_path: Path) -> None:
    tools = locate_tools()
    cover = tmp_path / "cover.jpg"
    subprocess.run(  # noqa: S603 - fixed argument list
        [
            str(tools.ffmpeg), "-v", "error", "-y",
            "-f", "lavfi", "-i", "color=c=red:s=64x64:d=1",
            "-frames:v", "1", str(cover),
        ],
        check=True,
    )  # fmt: skip
    chapters = tmp_path / "chapters.ffmeta"
    chapters.write_text(
        ";FFMETADATA1\n"
        "[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=1000\ntitle=One\n"
        "[CHAPTER]\nTIMEBASE=1/1000\nSTART=1000\nEND=2000\ntitle=Two\n",
        encoding="utf-8",
    )
    clip = tmp_path / "The First Omen (2024).en.MX]"
    clip = clip.with_name(clip.name + ".mkv")
    subprocess.run(  # noqa: S603 - fixed argument list
        [
            str(tools.ffmpeg), "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=s=160x120:r=24:d=2",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-i", str(chapters),
            "-map", "0:v", "-map", "1:a", "-map_metadata", "2", "-map_chapters", "2",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-metadata:s:a:0", "language=eng",
            # A Matroska attachment named cover.* is what ffprobe reports as an attached_pic
            # mjpeg "video" stream, exactly like the real library files.
            "-attach", str(cover), "-metadata:s:t", "mimetype=image/jpeg",
            str(clip),
        ],
        check=True,
    )  # fmt: skip

    info = probe(clip, tools, FakeReporter())

    video = video_of(info)
    assert (video.codec, video.width, video.height) == ("h264", 160, 120)
    assert video.fps == pytest.approx(24.0)
    assert info.duration_s == pytest.approx(2.0, abs=0.2)
    assert info.chapters == 2
    assert info.cover_art == 1
    assert [(a.codec, a.language) for a in info.audio] == [("aac", "eng")]
    assert info.size == clip.stat().st_size
    assert info.container == "matroska,webm"
    assert info.bitrate_kbps > 0
    assert not info.is_hdr
    assert not info.is_interlaced


def test_side_data_without_dv_profile_gives_none() -> None:
    data = load_fixture("hevc_4k_hdr10_mkv")
    data["streams"][0]["side_data_list"] = [{"side_data_type": "Mastering display metadata"}]
    assert video_of(parse_ffprobe(data, Path("x.mkv"), SIZE, MTIME)).dv_profile is None
