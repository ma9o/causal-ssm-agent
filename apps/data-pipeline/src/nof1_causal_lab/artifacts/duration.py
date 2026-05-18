"""Shared duration parsing helpers for stage artifacts and compilation."""

from __future__ import annotations

import re

_DURATION_UNIT_HOURS: dict[str, float] = {
    "s": 1 / 3600,
    "m": 1 / 60,
    "h": 1.0,
    "d": 24.0,
    "w": 168.0,
    "mo": 720.0,
    "q": 2160.0,
    "y": 8760.0,
}

_DURATION_RE = re.compile(r"^(\d+)(s|m|h|d|w|mo|q|y)$")


def parse_duration_to_hours(duration: str) -> float:
    """Parse a Polars-compatible duration string to hours."""
    match = _DURATION_RE.match(duration)
    if not match:
        raise ValueError(
            f"Invalid duration: {duration!r}. "
            f"Expected format: <int><unit> where unit is one of "
            f"{', '.join(_DURATION_UNIT_HOURS)}"
        )
    count = int(match.group(1))
    unit = match.group(2)
    if count == 0:
        raise ValueError("Duration must be positive (got 0)")
    return count * _DURATION_UNIT_HOURS[unit]


__all__ = ["parse_duration_to_hours"]
