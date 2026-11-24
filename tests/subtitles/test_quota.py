"""When the download allowance is spent and when it comes back."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from video_cleaner.subtitles.models import QuotaInfo
from video_cleaner.subtitles.quota import format_utc, next_reset, quota_usable


def utc(
    year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0
) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)


def test_next_reset_is_the_coming_midnight_utc() -> None:
    assert next_reset(utc(2026, 11, 23, 14, 30, 5)) == utc(2026, 11, 24)


def test_next_reset_at_exactly_midnight_is_the_following_midnight() -> None:
    assert next_reset(utc(2026, 11, 24)) == utc(2026, 11, 25)


def test_next_reset_across_a_month_end() -> None:
    assert next_reset(utc(2026, 11, 30, 23, 59, 59)) == utc(2026, 12, 1)
    assert next_reset(utc(2028, 2, 28, 12)) == utc(2028, 2, 29)  # leap year
    assert next_reset(utc(2028, 2, 29, 12)) == utc(2028, 3, 1)


def test_next_reset_across_a_year_end() -> None:
    assert next_reset(utc(2026, 12, 31, 23, 59, 59)) == utc(2027, 1, 1)


def test_next_reset_uses_the_utc_day_of_a_local_time() -> None:
    plus_two = timezone(timedelta(hours=2))
    # 01:00 on the 24th at +02:00 is still 23:00 on the 23rd in UTC.
    assert next_reset(datetime(2026, 11, 24, 1, 0, tzinfo=plus_two)) == utc(2026, 11, 24)


def test_next_reset_result_is_utc_aware() -> None:
    got = next_reset(utc(2026, 11, 23, 5))
    assert got.tzinfo is not None
    assert got.utcoffset() == timedelta(0)


def test_next_reset_refuses_a_naive_time() -> None:
    with pytest.raises(ValueError, match="timezone"):
        next_reset(datetime(2026, 11, 23, 5))


def test_no_quota_information_means_usable() -> None:
    assert quota_usable(None, utc(2026, 11, 23, 12))


def test_remaining_downloads_are_usable() -> None:
    assert quota_usable(QuotaInfo(remaining=1, reset_at=utc(2026, 11, 24)), utc(2026, 11, 23, 12))


def test_spent_quota_is_unusable_until_the_reset() -> None:
    q = QuotaInfo(remaining=0, reset_at=utc(2026, 11, 24))
    assert not quota_usable(q, utc(2026, 11, 23, 23, 59, 59))
    assert quota_usable(q, utc(2026, 11, 24, 0, 0, 1))


def test_quota_is_usable_exactly_at_the_reset() -> None:
    assert quota_usable(QuotaInfo(remaining=0, reset_at=utc(2026, 11, 24)), utc(2026, 11, 24))


def test_spent_quota_stays_unusable_for_the_whole_day() -> None:
    q = QuotaInfo(remaining=0, reset_at=utc(2026, 11, 24))
    assert not quota_usable(q, utc(2026, 11, 23, 0, 0, 1))


def test_usable_compares_in_utc_for_a_local_now() -> None:
    q = QuotaInfo(remaining=0, reset_at=utc(2026, 11, 24))
    plus_two = timezone(timedelta(hours=2))
    assert not quota_usable(q, datetime(2026, 11, 24, 1, 59, tzinfo=plus_two))  # 23:59 UTC
    assert quota_usable(q, datetime(2026, 11, 24, 2, 0, 1, tzinfo=plus_two))  # 00:00:01 UTC


def test_usable_refuses_a_naive_now() -> None:
    q = QuotaInfo(remaining=0, reset_at=utc(2026, 11, 24))
    with pytest.raises(ValueError, match="timezone"):
        quota_usable(q, datetime(2026, 11, 23, 12))
    with pytest.raises(ValueError, match="timezone"):
        quota_usable(None, datetime(2026, 11, 23, 12))


def test_format_utc_converts_before_it_says_utc() -> None:
    plus_two = timezone(timedelta(hours=2))
    assert format_utc(datetime(2030, 1, 2, 1, 30, tzinfo=plus_two)) == "2030-01-01 23:30 UTC"
    assert format_utc(datetime(2030, 1, 2, 0, 0, tzinfo=UTC)) == "2030-01-02 00:00 UTC"
    assert format_utc(datetime(2030, 1, 2, 0, 0)) == "2030-01-02 00:00 UTC"  # naive: taken as UTC
