"""Tests for utils/data.py pure utility functions.

Covers: chunk_lines, pivot_to_wide.
"""

from datetime import datetime, timedelta

import polars as pl

from causal_ssm_agent.utils.data import chunk_lines

# =============================================================================
# chunk_lines
# =============================================================================


class TestChunkLines:
    def test_exact_division(self):
        """Lines divide evenly into chunks."""
        lines = ["a", "b", "c", "d"]
        result = chunk_lines(lines, 2)
        assert result == ["a\nb", "c\nd"]

    def test_remainder(self):
        """Last chunk can have fewer lines."""
        lines = ["a", "b", "c", "d", "e"]
        result = chunk_lines(lines, 2)
        assert result == ["a\nb", "c\nd", "e"]

    def test_single_chunk(self):
        """All lines fit in one chunk."""
        lines = ["a", "b"]
        result = chunk_lines(lines, 10)
        assert result == ["a\nb"]

    def test_empty_input(self):
        """Empty input produces empty output."""
        assert chunk_lines([], 5) == []

    def test_chunk_size_one(self):
        """Chunk size 1 keeps each line separate."""
        lines = ["a", "b", "c"]
        result = chunk_lines(lines, 1)
        assert result == ["a", "b", "c"]


# =============================================================================
# pivot_to_wide
# =============================================================================


class TestPivotToWide:
    def test_basic_pivot(self):
        """Simple long-to-wide conversion."""
        df = pl.DataFrame({
            "timestamp": [1.0, 2.0, 1.0, 2.0],
            "indicator": ["x", "x", "y", "y"],
            "value": [10.0, 20.0, 30.0, 40.0],
        })
        from causal_ssm_agent.utils.data import pivot_to_wide

        wide = pivot_to_wide(df)
        assert "time" in wide.columns
        assert "x" in wide.columns
        assert "y" in wide.columns
        assert wide.height == 2

    def test_empty_dataframe(self):
        """Empty input returns empty output."""
        from causal_ssm_agent.utils.data import pivot_to_wide

        df = pl.DataFrame({"timestamp": [], "indicator": [], "value": []})
        result = pivot_to_wide(df)
        assert result.is_empty()

    def test_time_bucket_column(self):
        """Uses time_bucket column when present."""
        from causal_ssm_agent.utils.data import pivot_to_wide

        df = pl.DataFrame({
            "time_bucket": [1.0, 2.0],
            "indicator": ["x", "x"],
            "value": [10.0, 20.0],
        })
        wide = pivot_to_wide(df)
        assert "time" in wide.columns

    def test_datetime_to_fractional_days(self):
        """Datetime timestamps are converted to fractional days from t0."""
        from causal_ssm_agent.utils.data import pivot_to_wide

        t0 = datetime(2024, 1, 1)
        t1 = t0 + timedelta(days=1)
        t2 = t0 + timedelta(days=2)
        df = pl.DataFrame({
            "timestamp": [t0, t1, t2],
            "indicator": ["x", "x", "x"],
            "value": [1.0, 2.0, 3.0],
        })
        wide = pivot_to_wide(df)
        assert "time" in wide.columns
        times = wide["time"].to_list()
        assert abs(times[0]) < 0.001  # t0 should be 0
        assert abs(times[1] - 1.0) < 0.001  # t1 should be ~1 day
        assert abs(times[2] - 2.0) < 0.001  # t2 should be ~2 days

    def test_sorted_by_time(self):
        """Output should be sorted by time."""
        from causal_ssm_agent.utils.data import pivot_to_wide

        df = pl.DataFrame({
            "timestamp": [3.0, 1.0, 2.0],
            "indicator": ["x", "x", "x"],
            "value": [30.0, 10.0, 20.0],
        })
        wide = pivot_to_wide(df)
        times = wide["time"].to_list()
        assert times == sorted(times)

    def test_missing_values_as_null(self):
        """Indicators without values at certain times should be null."""
        from causal_ssm_agent.utils.data import pivot_to_wide

        df = pl.DataFrame({
            "timestamp": [1.0, 2.0, 2.0],
            "indicator": ["x", "x", "y"],
            "value": [10.0, 20.0, 30.0],
        })
        wide = pivot_to_wide(df)
        # y has no value at t=1, so it should be null
        y_at_t1 = wide.filter(pl.col("time") == 1.0)["y"].to_list()
        assert y_at_t1[0] is None

    def test_string_timestamps_parsed(self):
        """String timestamps should be parsed to datetime."""
        from causal_ssm_agent.utils.data import pivot_to_wide

        df = pl.DataFrame({
            "timestamp": ["2024-01-01", "2024-01-02"],
            "indicator": ["x", "x"],
            "value": [1.0, 2.0],
        })
        wide = pivot_to_wide(df)
        assert "time" in wide.columns
        times = wide["time"].to_list()
        assert abs(times[0]) < 0.001
        assert abs(times[1] - 1.0) < 0.001
