from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from video_cleaner.subtitles.models import QuotaInfo


def test_quota_info_holds_remaining_and_reset_time() -> None:
    reset = datetime(2026, 11, 23, 0, 0, tzinfo=UTC)
    quota = QuotaInfo(remaining=17, reset_at=reset)
    assert quota.remaining == 17
    assert quota.reset_at == reset
    assert quota.reset_at.utcoffset() == timedelta(0)


def test_quota_info_is_frozen() -> None:
    quota = QuotaInfo(remaining=1, reset_at=datetime(2026, 1, 1, tzinfo=UTC))
    with pytest.raises(AttributeError):
        quota.remaining = 2  # type: ignore[misc]


def test_a_naive_reset_time_is_refused() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        QuotaInfo(remaining=1, reset_at=datetime(2026, 1, 1))


def test_an_aware_reset_time_in_another_zone_is_normalised_to_utc() -> None:
    plus_two = timezone(timedelta(hours=2))
    quota = QuotaInfo(remaining=1, reset_at=datetime(2026, 1, 1, 2, 0, tzinfo=plus_two))
    assert quota.reset_at.utcoffset() == timedelta(0)
    assert quota.reset_at == datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
