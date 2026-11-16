from __future__ import annotations

import io
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pytest
from rich.console import Console

from video_cleaner.config import Settings
from video_cleaner.progress import (
    PlainReporter,
    Progress,
    RichReporter,
    format_eta,
    format_heartbeat,
    make_reporter,
    parse_ffmpeg_progress,
    parse_mkvmerge_progress,
)

BLOCK = """\
frame=1440
fps=38.50
stream_0_0_q=-0.0
bitrate=N/A
total_size=N/A
out_time_us=60000000
out_time_ms=60000000
out_time=00:01:00.000000
dup_frames=0
drop_frames=0
speed=1.6x
progress=continue
"""


def _lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def test_parse_ffmpeg_progress_percent_and_eta() -> None:
    (p,) = list(parse_ffmpeg_progress(_lines(BLOCK), duration_s=120.0))
    assert p.percent == pytest.approx(50.0)
    assert p.fps == pytest.approx(38.5)
    assert p.speed == pytest.approx(1.6)
    assert p.eta_s == pytest.approx(60.0 / 1.6)


def test_parse_ffmpeg_progress_one_progress_per_terminator() -> None:
    second = BLOCK.replace("60000000", "90000000").replace("progress=continue", "progress=end")
    out = list(parse_ffmpeg_progress(_lines(BLOCK + second), duration_s=120.0))
    assert [round(p.percent or 0) for p in out] == [50, 75]
    assert out[-1].note == "end"
    assert out[0].note == ""


def test_parse_ffmpeg_progress_handles_N_A_values() -> None:
    block = "fps=0.00\nout_time_us=N/A\nout_time_ms=N/A\nspeed=N/A\nprogress=continue\n"
    (p,) = list(parse_ffmpeg_progress(_lines(block), duration_s=120.0))
    assert p.percent is None
    assert p.speed is None
    assert p.eta_s is None
    assert p.fps == pytest.approx(0.0)


def test_parse_ffmpeg_progress_fps_na_and_zero_speed() -> None:
    block = "fps=N/A\nout_time_us=1000000\nspeed=0x\nprogress=continue\n"
    (p,) = list(parse_ffmpeg_progress(_lines(block), duration_s=100.0))
    assert p.fps is None
    assert p.percent == pytest.approx(1.0)
    assert p.eta_s is None  # speed 0 -> no ETA


def test_parse_ffmpeg_progress_uses_out_time_ms_when_us_missing() -> None:
    block = "out_time_ms=30000000\nspeed=2x\nprogress=continue\n"
    (p,) = list(parse_ffmpeg_progress(_lines(block), duration_s=120.0))
    assert p.percent == pytest.approx(25.0)
    assert p.eta_s == pytest.approx(45.0)


def test_parse_ffmpeg_progress_clamps_and_ignores_negative_times() -> None:
    over = "out_time_us=999000000\nspeed=1x\nprogress=continue\n"
    neg = "out_time_us=-9223372036854775807\nspeed=1x\nprogress=continue\n"
    a, b = list(parse_ffmpeg_progress(_lines(over + neg), duration_s=120.0))
    assert a.percent == 100.0
    assert a.eta_s == 0.0
    assert b.percent is None


def test_parse_ffmpeg_progress_zero_duration_gives_no_percent() -> None:
    (p,) = list(parse_ffmpeg_progress(_lines(BLOCK), duration_s=0.0))
    assert p.percent is None
    assert p.eta_s is None


def test_parse_ffmpeg_progress_skips_garbage_lines() -> None:
    text = "garbage line\n\n" + BLOCK
    assert len(list(parse_ffmpeg_progress(_lines(text), duration_s=120.0))) == 1


def test_parse_mkvmerge_progress() -> None:
    assert parse_mkvmerge_progress("#GUI#progress 37%") == 37.0
    assert parse_mkvmerge_progress("#GUI#progress 100%\n") == 100.0
    assert parse_mkvmerge_progress("#GUI#progress 12.5%") == 12.5
    assert parse_mkvmerge_progress("#GUI#begin_scanning_playlists") is None
    assert parse_mkvmerge_progress("Progress: 37%") is None
    assert parse_mkvmerge_progress("") is None


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


FIXED_WALL = datetime(2026, 11, 15, 7, 30, 5)


def _plain(clock: FakeClock, out: io.StringIO, interval: float = 30.0) -> PlainReporter:
    return PlainReporter(interval, clock=clock, out=out, now=lambda: FIXED_WALL)


def test_plain_reporter_rate_limits_to_30s() -> None:
    clock, out = FakeClock(), io.StringIO()
    rep = _plain(clock, out)
    rep.stage("encode")
    out.seek(0)
    out.truncate()
    for i in range(100):
        clock.now = 1000.0 + i * 0.29  # 100 updates spanning 28.71 s
        rep.update(Progress(percent=i / 2, fps=30.0, speed=1.5, eta_s=600.0))
    assert len(out.getvalue().splitlines()) == 1
    clock.now = 1000.0 + 30.0
    rep.update(Progress(percent=60.0, fps=30.0, speed=1.5, eta_s=300.0))
    assert len(out.getvalue().splitlines()) == 2


def test_plain_reporter_line_has_timestamp_label_and_values() -> None:
    clock, out = FakeClock(), io.StringIO()
    rep = _plain(clock, out)
    rep.stage("encode")
    rep.update(Progress(percent=43.2, fps=38.0, speed=1.6, eta_s=3120.0))
    lines = out.getvalue().splitlines()
    assert lines[0] == "[07:30:05] encode"
    assert "[07:30:05]" in lines[1]
    for fragment in ("encode", "43.2%", "38 fps", "1.6x", "ETA 52m"):
        assert fragment in lines[1]


def test_plain_reporter_update_with_unknown_values() -> None:
    clock, out = FakeClock(), io.StringIO()
    rep = _plain(clock, out)
    rep.update(Progress(percent=None, fps=None, speed=None, eta_s=None))
    assert out.getvalue().strip().endswith("working")


def test_plain_reporter_heartbeat_and_file_done_always_print() -> None:
    clock, out = FakeClock(), io.StringIO()
    rep = _plain(clock, out)
    rep.heartbeat("ffsubsync", 15.0)
    rep.heartbeat("ffsubsync", 30.0)
    rep.file_done("Movie (2018).mkv: 12.0 GB -> 4.1 GB")
    lines = out.getvalue().splitlines()
    assert len(lines) == 3
    assert "Movie (2018).mkv" in lines[2]


def test_heartbeat_line_contains_label_and_elapsed() -> None:
    assert format_heartbeat("ffsubsync", 192.0) == "ffsubsync, 3m12s elapsed, still working"
    assert format_heartbeat("hashing", 5.0) == "hashing, 0m05s elapsed, still working"
    assert format_heartbeat("verify", 3725.0) == "verify, 62m05s elapsed, still working"
    clock, out = FakeClock(), io.StringIO()
    _plain(clock, out).heartbeat("ffsubsync", 192.0)
    assert "ffsubsync, 3m12s elapsed, still working" in out.getvalue()


@pytest.mark.parametrize(
    ("eta", "expected"),
    [
        (None, "?"),
        (0.0, "0s"),
        (45.0, "45s"),
        (3120.0, "52m"),
        (3 * 3600 + 5 * 60, "3h05m"),
    ],
)
def test_format_eta(eta: float | None, expected: str) -> None:
    assert format_eta(eta) == expected


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        library_root=tmp_path / "lib",
        scratch_dir=tmp_path / "scratch",
        completed_dir=tmp_path / "completed",
        state_db=tmp_path / "state.db",
        log_dir=tmp_path / "logs",
        plain_progress_s=12.0,
    )


def test_make_reporter_picks_by_tty(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    assert isinstance(make_reporter(s, is_tty=False), PlainReporter)
    rich_rep = make_reporter(s, is_tty=True, out=io.StringIO())
    assert isinstance(rich_rep, RichReporter)
    rich_rep.close()


def test_make_reporter_plain_uses_configured_interval(tmp_path: Path) -> None:
    clock, out = FakeClock(), io.StringIO()
    rep = make_reporter(_settings(tmp_path), is_tty=False, clock=clock, out=out)
    p = Progress(percent=1.0, fps=None, speed=None, eta_s=None)
    rep.update(p)
    clock.now += 11.0
    rep.update(p)
    clock.now += 2.0
    rep.update(p)
    assert len(out.getvalue().splitlines()) == 2


def test_rich_reporter_renders_stage_update_heartbeat_and_summary() -> None:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=120, color_system=None)
    rep = RichReporter(console=console)
    rep.stage("encode")
    rep.update(Progress(percent=43.2, fps=38.0, speed=1.6, eta_s=3120.0))
    rep.update(Progress(percent=None, fps=None, speed=None, eta_s=None))
    rep.heartbeat("ffsubsync", 192.0)
    rep.file_done("Movie (2018).mkv done")
    rep.update(Progress(percent=5.0, fps=None, speed=None, eta_s=None))  # no stage: harmless
    rep.close()
    rep.close()  # idempotent
    text = buf.getvalue()
    assert "ffsubsync, 3m12s elapsed, still working" in text
    assert "Movie (2018).mkv done" in text


def _cp1252_stream() -> tuple[io.BytesIO, io.TextIOWrapper]:
    raw = io.BytesIO()
    return raw, io.TextIOWrapper(raw, encoding="cp1252", errors="strict")


def test_plain_reporter_survives_unencodable_file_names() -> None:
    raw, stream = _cp1252_stream()
    rep = _plain(FakeClock(), stream)  # type: ignore[arg-type]
    rep.stage("encode 映画 (2018).mkv")
    rep.update(Progress(percent=1.0, fps=None, speed=None, eta_s=None))
    rep.heartbeat("映画", 15.0)
    rep.file_done("映画 (2018).mkv done")
    stream.flush()
    text = raw.getvalue().decode("cp1252")
    assert "(2018).mkv done" in text
    assert len(text.splitlines()) == 4


def test_rich_reporter_survives_unencodable_file_names() -> None:
    raw, stream = _cp1252_stream()
    console = Console(file=stream, force_terminal=True, width=120, color_system=None)
    rep = RichReporter(console=console)
    rep.stage("encode 映画.mkv")
    rep.update(Progress(percent=10.0, fps=1.0, speed=1.0, eta_s=1.0))
    rep.heartbeat("映画", 15.0)
    rep.file_done("映画 (2018).mkv done")
    rep.close()
    stream.flush()
    assert "(2018).mkv done" in raw.getvalue().decode("cp1252")


class _SlowClock:
    """Constant clock whose read yields to other threads, widening any check-then-set race."""

    def __call__(self) -> float:
        time.sleep(0.005)
        return 1000.0


def _run_threads(target: Callable[[int], None], n: int = 8) -> list[BaseException]:
    errors: list[BaseException] = []
    barrier = threading.Barrier(n)

    def runner(i: int) -> None:
        try:
            barrier.wait()
            target(i)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=runner, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


def test_plain_reporter_rate_limit_holds_across_threads() -> None:
    out = io.StringIO()
    rep = PlainReporter(30.0, clock=_SlowClock(), out=out, now=lambda: FIXED_WALL)
    p = Progress(percent=1.0, fps=None, speed=None, eta_s=None)
    assert _run_threads(lambda _i: rep.update(p)) == []
    assert len(out.getvalue().splitlines()) == 1


def test_plain_reporter_is_thread_safe() -> None:
    out = io.StringIO()
    rep = PlainReporter(0.0001, out=out, now=lambda: FIXED_WALL)
    p = Progress(percent=1.0, fps=None, speed=None, eta_s=None)

    def work(i: int) -> None:
        for n in range(50):
            rep.stage(f"t{i}-{n}")
            rep.update(p)
            rep.heartbeat("x", 1.0)
            rep.file_done("done")

    assert _run_threads(work) == []
    assert all(line.startswith("[07:30:05] ") for line in out.getvalue().splitlines())


def test_rich_reporter_is_thread_safe() -> None:
    console = Console(file=io.StringIO(), force_terminal=True, width=120, color_system=None)
    rep = RichReporter(console=console)
    p = Progress(percent=50.0, fps=1.0, speed=1.0, eta_s=1.0)

    def work(i: int) -> None:
        for n in range(60):
            rep.stage(f"t{i}-{n}")
            rep.update(p)
            rep.heartbeat("x", 1.0)
            rep.update(p)
            rep.file_done("done")
            rep.update(p)

    assert _run_threads(work) == []
    rep.close()


def test_rich_reporter_prints_emoji_codes_literally() -> None:
    """A file name such as "Smile :smile: (2020).mkv" must not turn into an emoji."""
    from video_cleaner.progress import make_reporter as _make

    buf = io.StringIO()
    settings = Settings(
        library_root=Path("/l"),
        scratch_dir=Path("/s"),
        completed_dir=Path("/c"),
        state_db=Path("/d/s.db"),
        log_dir=Path("/d/logs"),
    )
    rep = _make(settings, is_tty=True, out=buf)
    rep.file_done("Smile :smile: (2020).mkv done")
    assert ":smile:" in buf.getvalue()


def test_encodable_is_public() -> None:
    from video_cleaner.progress import encodable

    assert encodable("abc", None) == "abc"
