"""mkvmerge command construction (pure): swap in a new video track, or append multi-part files."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from video_cleaner.tools import Tools


def build_mux_cmd(
    tools: Tools, output: Path, sources: Sequence[Path], new_video: Path | None
) -> list[str]:
    """mkvmerge argument list (no shell). ``--gui-mode`` makes progress print ``#GUI#progress N%``.

    With *new_video*: it is preceded by ``--no-chapters --no-global-tags --no-attachments``
    (mkvmerge options apply to the file that follows them), so it contributes only its video
    track; the single source follows with ``--no-video``, so every audio and subtitle track,
    the chapters, the global tags and all attachments (cover art included) come from the
    source by mkvmerge's defaults, with exactly one chapter edition. Several sources together
    with *new_video* are refused rather than guessed at.

    Precondition: when *new_video* is given the source must have exactly one real video
    stream, because ``--no-video`` drops every video track of the source; enforced by
    ``decide()``.

    Without *new_video*: the sources are appended in order; mkvmerge's append syntax prefixes
    the second and later files with ``+``. The first source keeps its chapters and
    attachments.
    """
    if not sources:
        raise ValueError("at least one source file is required")
    cmd = [str(tools.mkvmerge), "--gui-mode", "-o", str(output)]
    if new_video is not None:
        if len(sources) > 1:
            raise ValueError("new_video can be combined with exactly one source file")
        return [
            *cmd,
            "--no-chapters",
            "--no-global-tags",
            "--no-attachments",
            str(new_video),
            "--no-video",
            str(sources[0]),
        ]
    first, *rest = sources
    return [*cmd, str(first), *(f"+{part}" for part in rest)]
