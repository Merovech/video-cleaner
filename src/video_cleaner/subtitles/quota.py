"""When a provider's download allowance is spent and when it comes back."""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta

from video_cleaner.subtitles.models import QuotaInfo


def _as_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("a timezone-aware time is required")
    return moment.astimezone(UTC)


def next_reset(now: datetime) -> datetime:
    """The next 00:00 UTC after *now*, which is when the daily allowance comes back.

    *now* must be timezone-aware; its UTC day decides the answer. At exactly 00:00 UTC the
    answer is the midnight a day later.
    """
    today = _as_utc(now).date()
    return datetime.combine(today + timedelta(days=1), time.min, tzinfo=UTC)


def quota_usable(q: QuotaInfo | None, now: datetime) -> bool:
    """Whether a download may be attempted at *now*.

    Nothing known about the allowance counts as usable (the provider will say when it is
    spent); so does a reset time that has passed. Otherwise the answer is whether any
    downloads remain. *now* must be timezone-aware.
    """
    moment = _as_utc(now)
    if q is None:
        return True
    return moment >= q.reset_at or q.remaining > 0
