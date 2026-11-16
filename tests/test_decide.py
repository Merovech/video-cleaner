"""Decision engine: which files are encoded, remuxed, flagged or left alone."""

from __future__ import annotations

from pathlib import Path

import pytest

from video_cleaner.config import Settings
from video_cleaner.decide import Action, Decision, decide
from video_cleaner.probe import MediaInfo, VideoStream

MKV = "matroska,webm"
MP4 = "mov,mp4,m4a,3gp,3g2,mj2"
GIB = 1024**3


def _settings(*, bitrate_floor_kbps: int = 2500, min_file_size: int = 0) -> Settings:
    return Settings(
        library_root=Path("//nas/media"),
        scratch_dir=Path("D:/scratch"),
        completed_dir=Path("D:/completed"),
        state_db=Path("D:/state.db"),
        log_dir=Path("D:/logs"),
        bitrate_floor_kbps=bitrate_floor_kbps,
        min_file_size=min_file_size,
    )


def _info(
    *,
    codec: str = "h264",
    height: int = 1080,
    kbps: int = 8000,
    size: int = 6 * GIB,
    container: str = MKV,
    transfer: str | None = "bt709",
    dv_profile: int | None = None,
    field_order: str = "progressive",
    has_video: bool = True,
) -> MediaInfo:
    video = (
        VideoStream(
            codec=codec,
            width=height * 16 // 9,
            height=height,
            pix_fmt="yuv420p",
            fps=23.976,
            bit_rate=None,
            transfer=transfer,
            dv_profile=dv_profile,
            field_order=field_order,
        )
        if has_video
        else None
    )
    return MediaInfo(
        path=Path("//nas/media/Film (2020)/Film (2020).mkv"),
        size=size,
        mtime=1_700_000_000.0,
        duration_s=6000.0,
        container=container,
        video=video,
        audio=(),
        subtitles=(),
        chapters=0,
        cover_art=0,
        format_bit_rate=kbps * 1000,
    )


def _assert_shape(d: Decision, action: Action, *, encode: bool, remux: bool) -> None:
    assert (d.action, d.encode, d.remux) == (action, encode, remux)
    assert d.reasons
    assert all(isinstance(r, str) and r.strip() for r in d.reasons)


def test_h264_1080p_high_bitrate_encodes() -> None:
    d = decide(_info(), _settings())
    _assert_shape(d, Action.ENCODE, encode=True, remux=True)


def test_hevc_4k_left_alone() -> None:
    d = decide(_info(codec="hevc", height=2160, kbps=20000), _settings())
    _assert_shape(d, Action.LEAVE_ALONE, encode=False, remux=False)


def test_hevc_4k_needing_subs_is_remux() -> None:
    d = decide(_info(codec="hevc", height=2160, kbps=20000), _settings(), needs_subs=True)
    _assert_shape(d, Action.REMUX, encode=False, remux=True)


def test_low_bitrate_skips_encode() -> None:
    d = decide(_info(kbps=1800), _settings())
    _assert_shape(d, Action.LEAVE_ALONE, encode=False, remux=False)
    assert "bitrate 1800 kbps below floor 2500 kbps" in d.reasons


def test_bitrate_exactly_at_floor_encodes() -> None:
    d = decide(_info(kbps=2500), _settings())
    assert d.action is Action.ENCODE


def test_mp4_h264_below_floor_left_alone_not_remuxed() -> None:
    d = decide(_info(kbps=1800, container=MP4), _settings())
    _assert_shape(d, Action.LEAVE_ALONE, encode=False, remux=False)


def test_avi_below_floor_left_alone_not_remuxed() -> None:
    d = decide(_info(codec="mpeg4", height=480, kbps=1200, container="avi"), _settings())
    _assert_shape(d, Action.LEAVE_ALONE, encode=False, remux=False)


def test_hdr_h264_flagged() -> None:
    d = decide(_info(transfer="smpte2084"), _settings())
    _assert_shape(d, Action.FLAG, encode=False, remux=False)
    assert any("HDR" in r for r in d.reasons)


def test_interlaced_flagged() -> None:
    d = decide(_info(field_order="tt"), _settings())
    _assert_shape(d, Action.FLAG, encode=False, remux=False)
    assert any("interlaced" in r for r in d.reasons)


def test_dolby_vision_profile5_without_hdr_transfer_flagged() -> None:
    info = _info(dv_profile=5, transfer=None)
    assert not info.is_hdr
    d = decide(info, _settings())
    _assert_shape(d, Action.FLAG, encode=False, remux=False)
    assert any("Dolby Vision" in r and "5" in r for r in d.reasons)


def test_flag_reasons_name_every_cause() -> None:
    d = decide(_info(dv_profile=8, transfer="smpte2084", field_order="bb"), _settings())
    assert d.action is Action.FLAG
    text = " | ".join(d.reasons)
    assert "Dolby Vision" in text and "HDR" in text and "interlaced" in text


def test_flag_wins_over_needs_subs_and_multipart() -> None:
    d = decide(_info(transfer="smpte2084"), _settings(), needs_subs=True, multipart=True)
    _assert_shape(d, Action.FLAG, encode=False, remux=False)


def test_no_video_stream_flagged() -> None:
    d = decide(_info(has_video=False), _settings())
    _assert_shape(d, Action.FLAG, encode=False, remux=False)
    assert d.reasons == ("no video stream",)


def test_no_video_stream_flagged_even_below_floor() -> None:
    d = decide(_info(has_video=False, kbps=100), _settings(), needs_subs=True)
    assert d.action is Action.FLAG


def test_hevc_4k_hdr_left_alone_not_flagged() -> None:
    info = _info(codec="hevc", height=2160, kbps=25000, transfer="smpte2084", dv_profile=8)
    d = decide(info, _settings())
    _assert_shape(d, Action.LEAVE_ALONE, encode=False, remux=False)


@pytest.mark.parametrize(
    "info",
    [
        _info(codec="hevc", transfer="smpte2084"),  # HEVC 1080p HDR
        _info(codec="av1", transfer="arib-std-b67"),  # AV1 HLG
        _info(height=2160, kbps=40000, transfer="smpte2084"),  # H.264 4K HDR
        _info(kbps=1800, field_order="tt"),  # interlaced below floor
        _info(kbps=1800, dv_profile=5, transfer=None),  # DV below floor
    ],
)
def test_ineligible_files_are_never_flagged(info: MediaInfo) -> None:
    assert decide(info, _settings()).action is Action.LEAVE_ALONE
    assert decide(info, _settings(), needs_subs=True).action is Action.REMUX


def test_height_boundary_for_4k() -> None:
    assert decide(_info(height=1999), _settings()).action is Action.ENCODE
    d = decide(_info(height=2000), _settings())
    assert d.action is Action.LEAVE_ALONE
    assert any("2000" in r for r in d.reasons)


def test_av1_left_alone() -> None:
    d = decide(_info(codec="av1"), _settings())
    _assert_shape(d, Action.LEAVE_ALONE, encode=False, remux=False)


def test_codec_match_is_case_insensitive() -> None:
    assert decide(_info(codec="HEVC"), _settings()).action is Action.LEAVE_ALONE


def test_multipart_forces_remux_even_if_hevc() -> None:
    d = decide(_info(codec="hevc"), _settings(), multipart=True)
    _assert_shape(d, Action.REMUX, encode=False, remux=True)
    assert any("multi-part" in r for r in d.reasons)


def test_multipart_h264_still_encodes() -> None:
    d = decide(_info(), _settings(), multipart=True, needs_subs=True)
    _assert_shape(d, Action.ENCODE, encode=True, remux=True)


def test_min_file_size_skips_small_files() -> None:
    d = decide(_info(size=500_000_000), _settings(min_file_size=1_000_000_000))
    _assert_shape(d, Action.LEAVE_ALONE, encode=False, remux=False)
    assert any("500000000" in r and "1000000000" in r for r in d.reasons)


def test_min_file_size_boundary_is_exclusive() -> None:
    d = decide(_info(size=1_000_000_000), _settings(min_file_size=1_000_000_000))
    assert d.action is Action.ENCODE


def test_min_file_size_zero_means_no_floor() -> None:
    d = decide(_info(size=1), _settings(min_file_size=0))
    _assert_shape(d, Action.ENCODE, encode=True, remux=True)


def test_reasons_explain_each_decision() -> None:
    s = _settings()
    cases = {
        "encode": decide(_info(), s),
        "leave": decide(_info(codec="hevc", height=2160), s),
        "remux": decide(_info(codec="hevc"), s, needs_subs=True),
        "flag": decide(_info(field_order="tb"), s),
        "novideo": decide(_info(has_video=False), s),
    }
    for name, d in cases.items():
        assert isinstance(d.reasons, tuple), name
        assert d.reasons, name
        assert all(r.strip() for r in d.reasons), name
    assert any("hevc" in r for r in cases["leave"].reasons)
    assert any("2160" in r for r in cases["leave"].reasons)
    assert any("subtitles" in r for r in cases["remux"].reasons)
    assert any("8000 kbps" in r for r in cases["encode"].reasons)


def test_decision_is_immutable_and_action_is_str_enum() -> None:
    d = decide(_info(), _settings())
    with pytest.raises(AttributeError):
        d.action = Action.FLAG  # type: ignore[misc]
    assert isinstance(Action.ENCODE, str) and Action.ENCODE.value == "encode"
    assert {a.name for a in Action} == {"LEAVE_ALONE", "ENCODE", "REMUX", "FLAG"}
