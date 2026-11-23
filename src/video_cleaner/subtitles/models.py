"""Plain data shared by the subtitle providers and the code that consumes them."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class QuotaInfo:
    """What a provider reports about its download allowance.

    *remaining* downloads are left until *reset_at*, which must be timezone-aware and is stored
    in UTC; a naive datetime raises `ValueError` because it cannot be compared with another
    provider's reset time without guessing its zone.
    """

    remaining: int
    reset_at: datetime

    def __post_init__(self) -> None:
        if self.reset_at.tzinfo is None or self.reset_at.utcoffset() is None:
            raise ValueError("reset_at must be timezone-aware (UTC)")
        object.__setattr__(self, "reset_at", self.reset_at.astimezone(UTC))
