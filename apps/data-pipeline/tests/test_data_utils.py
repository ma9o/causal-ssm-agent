"""Tests for utils/data.py pure utility functions.

Covers: chunk_lines, pivot_to_wide, load_lines, sample_chunks,
        get_latest_preprocessed_file, resolve_query_path, load_query, list_queries.
"""

import time
from datetime import datetime, timedelta

import polars as pl

from causal_ssm_agent.utils.data import (
    chunk_lines,
    get_latest_preprocessed_file,
    load_lines,
    sample_chunks,
)

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
# load_lines
# =============================================================================


class TestLoadLines:
    def test_basic_load(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("line1\nline2\nline3\n")
        assert load_lines(f) == ["line1", "line2", "line3"]

    def test_strips_whitespace(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("  hello  \n  world  \n")
        assert load_lines(f) == ["hello", "world"]

    def test_skips_blank_lines(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("a\n\n\nb\n  \nc\n")
        assert load_lines(f) == ["a", "b", "c"]

    def test_empty_file(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("")
        assert load_lines(f) == []


# =============================================================================
# sample_chunks
# =============================================================================


class TestSampleChunks:
    def _write_lines(self, tmp_path, n):
        """Write n lines to a temp file and return the path."""
        f = tmp_path / "data.txt"
        f.write_text("\n".join(f"line{i}" for i in range(n)))
        return f

    def test_n_zero_returns_empty(self, tmp_path):
        f = self._write_lines(tmp_path, 10)
        assert sample_chunks(f, 0, chunk_size=1) == []

    def test_n_negative_returns_empty(self, tmp_path):
        f = self._write_lines(tmp_path, 10)
        assert sample_chunks(f, -5, chunk_size=1) == []

    def test_n_equals_total(self, tmp_path):
        f = self._write_lines(tmp_path, 5)
        result = sample_chunks(f, 5, chunk_size=1)
        assert len(result) == 5

    def test_n_exceeds_total(self, tmp_path):
        f = self._write_lines(tmp_path, 3)
        result = sample_chunks(f, 100, chunk_size=1)
        assert len(result) == 3

    def test_deterministic_with_seed(self, tmp_path):
        f = self._write_lines(tmp_path, 20)
        r1 = sample_chunks(f, 5, seed=42, chunk_size=1)
        r2 = sample_chunks(f, 5, seed=42, chunk_size=1)
        assert r1 == r2

    def test_samples_span_dataset(self, tmp_path):
        """Evenly-spaced samples should cover early and late chunks."""
        f = self._write_lines(tmp_path, 100)
        result = sample_chunks(f, 3, seed=0, chunk_size=1)
        assert len(result) == 3
        # First sample from early lines, last from late lines
        first_num = int(result[0].replace("line", ""))
        last_num = int(result[-1].replace("line", ""))
        assert first_num < 34  # first third
        assert last_num >= 66  # last third


# =============================================================================
# get_latest_preprocessed_file
# =============================================================================


class TestGetLatestPreprocessedFile:
    def test_returns_most_recent(self, tmp_path):
        old = tmp_path / "old.txt"
        old.write_text("old")
        time.sleep(0.05)  # ensure different mtime
        new = tmp_path / "new.txt"
        new.write_text("new")
        result = get_latest_preprocessed_file(tmp_path)
        assert result == new

    def test_excludes_specified_files(self, tmp_path):
        keep = tmp_path / "keep.txt"
        keep.write_text("keep")
        time.sleep(0.05)
        skip = tmp_path / "skip.txt"
        skip.write_text("skip")
        result = get_latest_preprocessed_file(tmp_path, exclude={"skip.txt"})
        assert result == keep

    def test_empty_directory(self, tmp_path):
        assert get_latest_preprocessed_file(tmp_path) is None

    def test_ignores_non_txt_files(self, tmp_path):
        (tmp_path / "data.csv").write_text("csv")
        assert get_latest_preprocessed_file(tmp_path) is None

    def test_single_file(self, tmp_path):
        f = tmp_path / "only.txt"
        f.write_text("only")
        assert get_latest_preprocessed_file(tmp_path) == f


# =============================================================================
# resolve_query_path / load_query / list_queries
# =============================================================================


class TestQueryFunctions:
    def test_resolve_exact_match(self, tmp_path, monkeypatch):
        monkeypatch.setattr("causal_ssm_agent.utils.data.QUERIES_DIR", tmp_path)
        f = tmp_path / "test.txt"
        f.write_text("query content")
        from causal_ssm_agent.utils.data import resolve_query_path

        assert resolve_query_path("test.txt") == f

    def test_resolve_adds_txt_extension(self, tmp_path, monkeypatch):
        monkeypatch.setattr("causal_ssm_agent.utils.data.QUERIES_DIR", tmp_path)
        f = tmp_path / "test.txt"
        f.write_text("content")
        from causal_ssm_agent.utils.data import resolve_query_path

        assert resolve_query_path("test") == f

    def test_resolve_adds_md_extension(self, tmp_path, monkeypatch):
        monkeypatch.setattr("causal_ssm_agent.utils.data.QUERIES_DIR", tmp_path)
        f = tmp_path / "test.md"
        f.write_text("content")
        from causal_ssm_agent.utils.data import resolve_query_path

        assert resolve_query_path("test") == f

    def test_resolve_not_found(self, tmp_path, monkeypatch):
        monkeypatch.setattr("causal_ssm_agent.utils.data.QUERIES_DIR", tmp_path)
        import pytest

        from causal_ssm_agent.utils.data import resolve_query_path

        with pytest.raises(FileNotFoundError):
            resolve_query_path("nonexistent")

    def test_load_query(self, tmp_path, monkeypatch):
        monkeypatch.setattr("causal_ssm_agent.utils.data.QUERIES_DIR", tmp_path)
        f = tmp_path / "q.txt"
        f.write_text("  Does X cause Y?  \n")
        from causal_ssm_agent.utils.data import load_query

        assert load_query("q.txt") == "Does X cause Y?"

    def test_list_queries(self, tmp_path, monkeypatch):
        monkeypatch.setattr("causal_ssm_agent.utils.data.QUERIES_DIR", tmp_path)
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.md").write_text("b")
        (tmp_path / ".gitkeep").write_text("")
        from causal_ssm_agent.utils.data import list_queries

        names = list_queries()
        assert "a.txt" in names
        assert "b.md" in names
        assert ".gitkeep" not in names


# =============================================================================
# pivot_to_wide
# =============================================================================


class TestPivotToWide:
    def test_basic_pivot(self):
        """Simple long-to-wide conversion."""
        df = pl.DataFrame(
            {
                "timestamp": [1.0, 2.0, 1.0, 2.0],
                "indicator": ["x", "x", "y", "y"],
                "value": [10.0, 20.0, 30.0, 40.0],
            }
        )
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

        df = pl.DataFrame(
            {
                "time_bucket": [1.0, 2.0],
                "indicator": ["x", "x"],
                "value": [10.0, 20.0],
            }
        )
        wide = pivot_to_wide(df)
        assert "time" in wide.columns

    def test_datetime_to_fractional_days(self):
        """Datetime timestamps are converted to fractional days from t0."""
        from causal_ssm_agent.utils.data import pivot_to_wide

        t0 = datetime(2024, 1, 1)
        t1 = t0 + timedelta(days=1)
        t2 = t0 + timedelta(days=2)
        df = pl.DataFrame(
            {
                "timestamp": [t0, t1, t2],
                "indicator": ["x", "x", "x"],
                "value": [1.0, 2.0, 3.0],
            }
        )
        wide = pivot_to_wide(df)
        assert "time" in wide.columns
        times = wide["time"].to_list()
        assert abs(times[0]) < 0.001  # t0 should be 0
        assert abs(times[1] - 1.0) < 0.001  # t1 should be ~1 day
        assert abs(times[2] - 2.0) < 0.001  # t2 should be ~2 days

    def test_sorted_by_time(self):
        """Output should be sorted by time."""
        from causal_ssm_agent.utils.data import pivot_to_wide

        df = pl.DataFrame(
            {
                "timestamp": [3.0, 1.0, 2.0],
                "indicator": ["x", "x", "x"],
                "value": [30.0, 10.0, 20.0],
            }
        )
        wide = pivot_to_wide(df)
        times = wide["time"].to_list()
        assert times == sorted(times)

    def test_missing_values_as_null(self):
        """Indicators without values at certain times should be null."""
        from causal_ssm_agent.utils.data import pivot_to_wide

        df = pl.DataFrame(
            {
                "timestamp": [1.0, 2.0, 2.0],
                "indicator": ["x", "x", "y"],
                "value": [10.0, 20.0, 30.0],
            }
        )
        wide = pivot_to_wide(df)
        # y has no value at t=1, so it should be null
        y_at_t1 = wide.filter(pl.col("time") == 1.0)["y"].to_list()
        assert y_at_t1[0] is None

    def test_string_timestamps_parsed(self):
        """String timestamps should be parsed to datetime."""
        from causal_ssm_agent.utils.data import pivot_to_wide

        df = pl.DataFrame(
            {
                "timestamp": ["2024-01-01", "2024-01-02"],
                "indicator": ["x", "x"],
                "value": [1.0, 2.0],
            }
        )
        wide = pivot_to_wide(df)
        assert "time" in wide.columns
        times = wide["time"].to_list()
        assert abs(times[0]) < 0.001
        assert abs(times[1] - 1.0) < 0.001

    def test_string_values_cast_to_float(self):
        """String values should be cast to Float64."""
        from causal_ssm_agent.utils.data import pivot_to_wide

        df = pl.DataFrame(
            {
                "timestamp": [1.0, 2.0],
                "indicator": ["x", "x"],
                "value": ["10.5", "20.3"],
            }
        )
        wide = pivot_to_wide(df)
        assert wide["x"].dtype == pl.Float64
        assert abs(wide["x"][0] - 10.5) < 0.001

    def test_duplicate_values_aggregated_with_mean(self):
        """Multiple values at same time for same indicator should be averaged."""
        from causal_ssm_agent.utils.data import pivot_to_wide

        df = pl.DataFrame(
            {
                "timestamp": [1.0, 1.0, 2.0],
                "indicator": ["x", "x", "x"],
                "value": [10.0, 20.0, 30.0],
            }
        )
        wide = pivot_to_wide(df)
        assert wide.height == 2
        # At t=1, mean of 10 and 20 is 15
        x_at_t1 = wide.filter(pl.col("time") == 1.0)["x"][0]
        assert abs(x_at_t1 - 15.0) < 0.001

    def test_single_indicator(self):
        """Minimal case with just one indicator."""
        from causal_ssm_agent.utils.data import pivot_to_wide

        df = pl.DataFrame(
            {
                "timestamp": [1.0],
                "indicator": ["x"],
                "value": [42.0],
            }
        )
        wide = pivot_to_wide(df)
        assert wide.height == 1
        assert "x" in wide.columns
        assert wide["x"][0] == 42.0
