from __future__ import annotations

from video_cleaner.cmdline import PROGRESS_OPTIONS, flatten


def test_flatten_joins_flags_and_values_in_order() -> None:
    assert flatten(("-crf", "20"), ("-an",), ("-i", "in.mkv")) == [
        "-crf",
        "20",
        "-an",
        "-i",
        "in.mkv",
    ]


def test_flatten_of_nothing_is_empty() -> None:
    assert flatten() == []


def test_progress_options_report_machine_readable_progress_without_stats() -> None:
    assert flatten(*PROGRESS_OPTIONS) == ["-progress", "pipe:1", "-nostats"]
