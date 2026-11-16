"""Small helpers shared by the plan, calibrate and status/report tests (no media, no tools)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from video_cleaner.config import Settings
from video_cleaner.probe import MediaInfo, VideoStream
from video_cleaner.state import State

MTIME = 1_700_000_000.0


def make_info(path: Path, size: int, mtime: float = MTIME) -> MediaInfo:
    return MediaInfo(
        path=path,
        size=size,
        mtime=mtime,
        duration_s=100.0,
        container="matroska,webm",
        video=VideoStream(
            codec="h264",
            width=1920,
            height=1080,
            pix_fmt="yuv420p",
            fps=23.976,
            bit_rate=None,
            transfer="bt709",
            dv_profile=None,
            field_order="progressive",
        ),
        audio=(),
        subtitles=(),
        chapters=0,
        cover_art=0,
    )


def add_record(
    state: State,
    path: Path,
    decision: str,
    reasons: Sequence[str],
    *,
    size: int = 1000,
    est: int = 0,
    parts: Sequence[Path] = (),
    mtime: float = MTIME,
) -> None:
    """Upsert a planned record the way survey would."""
    state.upsert_file(
        make_info(path, size, mtime),
        decision=decision,
        reasons=reasons,
        est_saving_bytes=est,
        parts=parts,
    )


def tmp_settings(tmp_path: Path) -> Settings:
    """Settings whose folders all live in *tmp_path* (the same ones `write_config` writes)."""
    return Settings(
        library_root=tmp_path / "library",
        scratch_dir=tmp_path / "scratch",
        completed_dir=tmp_path / "completed",
        state_db=tmp_path / "appdata" / "state.db",
        log_dir=tmp_path / "appdata" / "logs",
    )


def write_config(tmp_path: Path, **changes: Path) -> Path:
    """A config file for `tmp_settings` (outputs outside the library), with *changes* applied.

    Creates the library folder.
    """
    base = tmp_settings(tmp_path)
    values: dict[str, Path] = {
        name: getattr(base, name)
        for name in ("library_root", "scratch_dir", "completed_dir", "state_db", "log_dir")
    }
    values.update(changes)
    values["library_root"].mkdir(parents=True, exist_ok=True)
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        "".join(f"{k} = '{v.as_posix()}'\n" for k, v in values.items()), encoding="utf-8"
    )
    return cfg
