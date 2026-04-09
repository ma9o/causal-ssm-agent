"""Tests for support-window extraction infrastructure.

Covers: bucket_by_clock (data.py),
chunk_windows, format_window_chunk (workers/windows.py).
"""

from datetime import datetime

import polars as pl
import pytest

from causal_ssm_agent.utils.data import bucket_by_clock
from causal_ssm_agent.workers.windows import chunk_windows, format_window_chunk

# =============================================================================
# bucket_by_clock
# =============================================================================


def _make_events_df(timestamps: list[str], values: list[int] | None = None) -> pl.DataFrame:
    """Create a simple events DataFrame with a timestamp column."""
    n = len(timestamps)
    return pl.DataFrame(
        {
            "timestamp": [datetime.fromisoformat(t) for t in timestamps],
            "value": values or list(range(n)),
        }
    )


class TestBucketByClock:
    def test_daily_bucketing(self):
        df = _make_events_df(
            [
                "2024-01-01T08:00:00",
                "2024-01-01T12:00:00",
                "2024-01-01T18:00:00",
                "2024-01-02T09:00:00",
                "2024-01-02T15:00:00",
            ]
        )
        ticks = bucket_by_clock(df, "1d", "timestamp")
        assert len(ticks) == 2
        assert ticks[0][0] == "2024-01-01T00:00:00"
        assert len(ticks[0][1]) == 3  # 3 events on day 1
        assert len(ticks[1][1]) == 2  # 2 events on day 2

    def test_hourly_bucketing(self):
        df = _make_events_df(
            [
                "2024-01-01T08:15:00",
                "2024-01-01T08:45:00",
                "2024-01-01T09:30:00",
            ]
        )
        ticks = bucket_by_clock(df, "1h", "timestamp")
        assert len(ticks) == 2  # 08:xx and 09:xx
        assert len(ticks[0][1]) == 2
        assert len(ticks[1][1]) == 1

    def test_string_timestamp_column(self):
        """String timestamps should be auto-parsed."""
        df = pl.DataFrame(
            {
                "timestamp": ["2024-01-01T10:00:00", "2024-01-02T10:00:00"],
                "value": [1, 2],
            }
        )
        ticks = bucket_by_clock(df, "1d", "timestamp")
        assert len(ticks) == 2

    def test_timezone_aware_string_timestamp_column(self):
        """UTC-suffixed string timestamps should bucket without parse errors."""
        df = pl.DataFrame(
            {
                "timestamp": ["2025-03-03T10:00:00Z", "2025-03-04T10:00:00Z"],
                "value": [1, 2],
            }
        )

        ticks = bucket_by_clock(df, "1d", "timestamp")

        assert [tick_id for tick_id, _ in ticks] == [
            "2025-03-03T00:00:00+00:00",
            "2025-03-04T00:00:00+00:00",
        ]

    def test_chronological_ordering(self):
        """Ticks should be sorted chronologically."""
        df = _make_events_df(
            [
                "2024-01-03T10:00:00",
                "2024-01-01T10:00:00",
                "2024-01-02T10:00:00",
            ]
        )
        ticks = bucket_by_clock(df, "1d", "timestamp")
        tick_ids = [t[0] for t in ticks]
        assert tick_ids == sorted(tick_ids)

    def test_empty_df(self):
        df = pl.DataFrame(schema={"timestamp": pl.Datetime, "value": pl.Int64})
        ticks = bucket_by_clock(df, "1d", "timestamp")
        assert ticks == []

    def test_tick_id_is_iso_format(self):
        df = _make_events_df(["2024-06-15T14:30:00"])
        ticks = bucket_by_clock(df, "1d", "timestamp")
        assert ticks[0][0] == "2024-06-15T00:00:00"

    def test_no_tick_column_in_output(self):
        """The internal __tick__ column should not appear in output DataFrames."""
        df = _make_events_df(["2024-01-01T10:00:00", "2024-01-01T15:00:00"])
        ticks = bucket_by_clock(df, "1d", "timestamp")
        assert "__tick__" not in ticks[0][1].columns


# =============================================================================
# chunk_windows
# =============================================================================


def _make_windows(n: int) -> list[tuple[str, pl.DataFrame]]:
    """Create N dummy support windows."""
    return [(f"2024-01-{i + 1:02d}", pl.DataFrame({"value": [i]})) for i in range(n)]


class TestChunkWindows:
    @pytest.mark.parametrize(
        ("n_windows", "windows_per_chunk", "chunk_sizes"),
        [
            (6, 3, [3, 3]),
            (7, 3, [3, 3, 1]),
            (1, 7, [1]),
        ],
    )
    def test_chunk_sizes(self, n_windows, windows_per_chunk, chunk_sizes):
        windows = _make_windows(n_windows)
        chunks = chunk_windows(windows, windows_per_chunk)
        assert [len(chunk) for chunk in chunks] == chunk_sizes

    def test_empty_input(self):
        assert chunk_windows([], 7) == []

    def test_preserves_order(self):
        windows = _make_windows(5)
        chunks = chunk_windows(windows, 2)
        flat = [window_start for chunk in chunks for window_start, _ in chunk]
        assert flat == [window[0] for window in windows]


# =============================================================================
# format_window_chunk
# =============================================================================


class TestFormatWindowChunk:
    def test_basic_formatting(self):
        events = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 8, 0), datetime(2024, 1, 1, 12, 30)],
                "action": ["searched python", "viewed stackoverflow"],
            }
        )
        chunk = [("2024-01-01", events)]
        text = format_window_chunk(chunk, "timestamp", ["action"])
        assert "## Window Start: 2024-01-01" in text
        assert "searched python" in text
        assert "viewed stackoverflow" in text
        assert "08:00" in text
        assert "12:30" in text

    def test_multiple_windows(self):
        events1 = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 8, 0)],
                "action": ["event1"],
            }
        )
        events2 = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 2, 9, 0)],
                "action": ["event2"],
            }
        )
        chunk = [("2024-01-01", events1), ("2024-01-02", events2)]
        text = format_window_chunk(chunk, "timestamp", ["action"])
        assert "## Window Start: 2024-01-01" in text
        assert "## Window Start: 2024-01-02" in text
        assert "event1" in text
        assert "event2" in text

    def test_truncation(self):
        """Events exceeding max_events_per_window should be truncated."""
        n = 50
        events = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, h % 24, m)
                    for h, m in zip(range(n), range(n), strict=True)
                ],
                "action": [f"event_{i}" for i in range(n)],
            }
        )
        chunk = [("2024-01-01", events)]
        text = format_window_chunk(chunk, "timestamp", ["action"], max_events_per_window=20)
        assert "showing 20 of 50 events" in text

    def test_truncation_is_deterministic(self):
        n = 50
        events = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, h % 24, m)
                    for h, m in zip(range(n), range(n), strict=True)
                ],
                "action": [f"event_{i}" for i in range(n)],
            }
        )
        chunk = [("2024-01-01", events)]

        text_a = format_window_chunk(chunk, "timestamp", ["action"], max_events_per_window=20)
        text_b = format_window_chunk(chunk, "timestamp", ["action"], max_events_per_window=20)

        assert text_a == text_b

    def test_no_truncation_under_limit(self):
        events = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, i, 0) for i in range(5)],
                "action": [f"event_{i}" for i in range(5)],
            }
        )
        chunk = [("2024-01-01", events)]
        text = format_window_chunk(chunk, "timestamp", ["action"], max_events_per_window=300)
        assert "showing" not in text

    def test_multiple_display_columns(self):
        events = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 8, 0)],
                "col1": ["val1"],
                "col2": ["val2"],
            }
        )
        chunk = [("2024-01-01", events)]
        text = format_window_chunk(chunk, "timestamp", ["col1", "col2"])
        assert "val1" in text
        assert "val2" in text

    def test_none_values_excluded(self):
        events = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 8, 0)],
                "action": [None],
            }
        )
        chunk = [("2024-01-01", events)]
        text = format_window_chunk(chunk, "timestamp", ["action"])
        # Should still produce a support-window header
        assert "## Window Start: 2024-01-01" in text
