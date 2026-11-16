"""OpenSubtitles movie hash (size + 64-bit LE word sums of the first and last 64 KiB)."""

from __future__ import annotations

import logging
import struct
from pathlib import Path

import pytest

from video_cleaner.moviehash import CHUNK, MIN_SIZE, movie_hash


def _write(path: Path, size: int, words: dict[int, int] | None = None) -> Path:
    data = bytearray(size)
    for offset, value in (words or {}).items():
        data[offset : offset + 8] = struct.pack("<Q", value)
    path.write_bytes(bytes(data))
    return path


def test_constants_match_the_algorithm() -> None:
    assert CHUNK == 65536
    assert MIN_SIZE == 2 * CHUNK


def test_all_zero_file_hashes_to_its_size(tmp_path: Path) -> None:
    assert movie_hash(_write(tmp_path / "z.mkv", MIN_SIZE)) == "0000000000020000"


def test_sum_wraps_modulo_2_64(tmp_path: Path) -> None:
    f = _write(tmp_path / "w.mkv", MIN_SIZE, {0: 2**64 - 1})
    # (2^64 - 1) + 131072 wraps to 131071
    assert movie_hash(f) == "000000000001ffff"


def test_head_and_tail_words_are_little_endian(tmp_path: Path) -> None:
    size = 200_000
    tail = size - CHUNK
    f = _write(tmp_path / "le.mkv", size, {8: 0x0102030405060708, tail: 0x10})
    expected = (size + 0x0102030405060708 + 0x10) % 2**64
    assert movie_hash(f) == f"{expected:016x}"


def test_bytes_between_head_and_tail_are_ignored(tmp_path: Path) -> None:
    size = 300_000
    a = _write(tmp_path / "a.mkv", size)
    b = _write(tmp_path / "b.mkv", size, {CHUNK + 8: 12345})
    assert movie_hash(a) == movie_hash(b)


def test_unaligned_size_reads_tail_from_its_own_start(tmp_path: Path) -> None:
    size = MIN_SIZE + 3
    raw = bytearray(size)
    raw[size - CHUNK + 1] = 1  # second byte of the tail chunk: word value 256
    f = tmp_path / "u.mkv"
    f.write_bytes(bytes(raw))
    assert movie_hash(f) == f"{size + 256:016x}"


def test_independent_reference_implementation(tmp_path: Path) -> None:
    size = 150_001
    raw = bytes((i * 7 + 3) % 256 for i in range(size))
    f = tmp_path / "r.mkv"
    f.write_bytes(raw)
    total = size
    for chunk in (raw[:CHUNK], raw[-CHUNK:]):
        total += sum(struct.unpack(f"<{CHUNK // 8}Q", chunk))
    assert movie_hash(f) == f"{total % 2**64:016x}"


def test_file_smaller_than_two_chunks_has_no_hash(tmp_path: Path) -> None:
    assert movie_hash(_write(tmp_path / "s.mkv", MIN_SIZE - 1)) is None
    assert movie_hash(_write(tmp_path / "e.mkv", 0)) is None


def test_read_error_gives_none_and_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    logger = logging.getLogger("video_cleaner")
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="video_cleaner"):
            assert movie_hash(tmp_path / "missing.mkv") is None
    finally:
        logger.removeHandler(caplog.handler)
    assert "movie hash" in caplog.text


def test_file_shrinking_while_read_gives_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = _write(tmp_path / "t.mkv", MIN_SIZE)
    real_stat = Path.stat

    def bigger(self: Path, **kwargs: bool) -> object:
        st = real_stat(self, **kwargs)
        return type("S", (), {"st_size": st.st_size + 10})()

    monkeypatch.setattr(Path, "stat", bigger)
    assert movie_hash(f) is None
