"""Tiny generated media for the integration tests (lavfi clips, no media files committed)."""

from __future__ import annotations

import functools
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from video_cleaner.progress import Progress
from video_cleaner.tools import Tools, locate_tools

SRT = "1\n00:00:00,500 --> 00:00:01,500\nHello.\n\n2\n00:00:02,000 --> 00:00:03,000\nGoodbye.\n"
# Not a real font: mkvmerge stores attachments as opaque bytes with the given MIME type.
FONT_BYTES = b"\x00\x01\x00\x00" + b"\x00" * 60


class RecordingReporter:
    def __init__(self) -> None:
        self.stages: list[str] = []
        self.updates: list[Progress] = []
        self.heartbeats: list[str] = []
        self.done: list[str] = []

    def stage(self, label: str) -> None:
        self.stages.append(label)

    def update(self, p: Progress) -> None:
        self.updates.append(p)

    def heartbeat(self, label: str, elapsed_s: float) -> None:
        self.heartbeats.append(label)

    def file_done(self, summary: str) -> None:
        self.done.append(summary)


@functools.cache
def _has_libvmaf(ffmpeg: Path) -> bool:
    result = subprocess.run(  # noqa: S603 - fixed argument list
        [str(ffmpeg), "-hide_banner", "-filters"],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        stdin=subprocess.DEVNULL,
    )
    return "libvmaf" in result.stdout


def vmaf_tools() -> Tools:
    """`locate_tools`, skipping the test when this ffmpeg has no ``libvmaf`` (Ubuntu's apt build).

    CI fails the Windows job when libvmaf is missing, so the VMAF tests always run there.
    """
    tools = locate_tools()
    if not _has_libvmaf(tools.ffmpeg):
        pytest.skip("this ffmpeg build has no libvmaf")
    return tools


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)  # noqa: S603 - fixed argument list


def make_clip(
    tools: Tools,
    path: Path,
    *,
    duration_s: float = 6.0,
    size: str = "320x240",
    rate: str = "24",
    rich: bool = False,
) -> Path:
    """Write an H.264 + AAC MKV of `testsrc2` and a sine to *path*, with two chapters.

    *rate* is the frame rate; ``24000/1001`` gives the jittery 1 ms Matroska timestamps of
    real 23.976 fps films.

    The video is high-bitrate (x264 CRF 12) so an x265 encode of it is far smaller. With
    *rich*, mkvmerge adds an SRT subtitle, an mjpeg cover attachment (``cover.jpg``) and a font
    attachment, like the real library files. Work files go to a temporary folder beside *path*
    and are removed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".clip-", dir=path.parent))
    try:
        chapters = work / "chapters.ffmeta"
        half = int(duration_s * 500)
        chapters.write_text(
            ";FFMETADATA1\n"
            f"[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND={half}\ntitle=One\n"
            f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={half}\nEND={half * 2}\ntitle=Two\n",
            encoding="utf-8",
        )
        base = work / "base.mkv" if rich else path
        _run(
            [
                str(tools.ffmpeg), "-nostdin", "-v", "error", "-y",
                "-f", "lavfi", "-i", f"testsrc2=s={size}:r={rate}:d={duration_s}",
                "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration_s}",
                "-i", str(chapters),
                "-map", "0:v", "-map", "1:a", "-map_metadata", "2", "-map_chapters", "2",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "12", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "64k", "-metadata:s:a:0", "language=eng",
                str(base),
            ]
        )  # fmt: skip
        if rich:
            cover = work / "cover.jpg"
            _run(
                [
                    str(tools.ffmpeg), "-nostdin", "-v", "error", "-y",
                    "-f", "lavfi", "-i", "color=c=red:s=64x64:d=1",
                    "-frames:v", "1", str(cover),
                ]
            )  # fmt: skip
            subs = work / "subs.srt"
            subs.write_text(SRT, encoding="utf-8")
            font = work / "font.ttf"
            font.write_bytes(FONT_BYTES)
            _run(
                [
                    str(tools.mkvmerge), "-q", "-o", str(path), str(base),
                    "--language", "0:eng", str(subs),
                    "--attachment-mime-type", "image/jpeg", "--attachment-name", "cover.jpg",
                    "--attach-file", str(cover),
                    "--attachment-mime-type", "font/ttf", "--attach-file", str(font),
                ]
            )  # fmt: skip
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return path
