"""mkvmerge command construction (pure): swap in a new video track, or append multi-part files."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from video_cleaner.cmdline import flatten
from video_cleaner.tools import Tools

MAX_TITLE_CHARS = 200
_LANGUAGE = re.compile(r"[a-z]{3}")


@dataclass(frozen=True)
class SubtitleInput:
    """One subtitle file to add as a new track: its path, track name and ISO 639-2 language.

    The title and language reach mkvmerge as command-line arguments (never a shell string), but
    they are still checked here: a title is 1 to 200 characters without control characters (the
    resolver writes constants, so anything else is a damaged record), a language is three
    lowercase letters.
    """

    path: Path
    title: str
    language: str = "eng"

    def __post_init__(self) -> None:
        if not _LANGUAGE.fullmatch(self.language):
            raise ValueError(
                f"subtitle language must be three lowercase letters: {self.language!r}"
            )
        if not 1 <= len(self.title) <= MAX_TITLE_CHARS:
            raise ValueError(f"subtitle title must be 1 to {MAX_TITLE_CHARS} characters")
        if any(unicodedata.category(ch) == "Cc" for ch in self.title):
            raise ValueError("subtitle title must not contain control characters")


def _subtitle_args(subtitle: SubtitleInput) -> tuple[str, ...]:
    """The options for the subtitle file, then the file: every other track stays untouched.

    mkvmerge options apply to the file that follows them. The new track is neither default nor
    forced, so a player keeps choosing its audio and subtitles as it did before. ``0:<title>``
    is one argument whatever the title holds, a leading ``-`` included.
    """
    return (
        "--language",
        f"0:{subtitle.language}",
        "--track-name",
        f"0:{subtitle.title}",
        "--default-track-flag",
        "0:0",
        "--forced-display-flag",
        "0:0",
        str(subtitle.path),
    )


def build_mux_cmd(
    tools: Tools,
    output: Path,
    sources: Sequence[Path],
    new_video: Path | None,
    subtitle: SubtitleInput | None = None,
) -> list[str]:
    """mkvmerge argument list (no shell). ``--gui-mode`` makes progress print ``#GUI#progress N%``.

    With *new_video*: it is preceded by ``--no-chapters --no-global-tags --no-track-tags
    --no-attachments`` (mkvmerge options apply to the file that follows them), so it contributes
    only its video track, without the encoder's statistics tags; the single source follows
    with ``--no-video``, so every audio and subtitle track, the chapters, the global tags and
    all attachments (cover art included) come from the
    source by mkvmerge's defaults, with exactly one chapter edition. Several sources together
    with *new_video* are refused rather than guessed at.

    Precondition: when *new_video* is given the source must have exactly one real video
    stream, because ``--no-video`` drops every video track of the source; enforced by
    ``decide()``.

    Without *new_video*: the sources are appended in order; mkvmerge's append syntax prefixes
    the second and later files with ``+``. The first source keeps its chapters and
    attachments.

    With *subtitle*, its options and file come after the source options, so it is added as one
    new last track (language, name, no default or forced flag) and nothing else changes.
    """
    if not sources:
        raise ValueError("at least one source file is required")
    cmd = [str(tools.mkvmerge), "--gui-mode", "-o", str(output)]
    extra = () if subtitle is None else _subtitle_args(subtitle)
    if new_video is not None:
        if len(sources) > 1:
            raise ValueError("new_video can be combined with exactly one source file")
        return [
            *cmd,
            *flatten(
                # The next file contributes only its video track:
                ("--no-chapters",),  # not its chapters
                ("--no-global-tags",),  # nor its global tags
                ("--no-track-tags",),  # nor the encoder's statistics tags
                ("--no-attachments",),  # nor attachments
                (str(new_video),),  # the new video track
                # The original contributes everything but its video:
                ("--no-video",),
                (str(sources[0]),),
                extra,
            ),
        ]
    first, *rest = sources
    return [*cmd, str(first), *(f"+{part}" for part in rest), *extra]
