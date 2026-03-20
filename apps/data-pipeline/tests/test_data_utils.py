"""Tests for utils/data.py pure utility functions.

Covers: chunk_lines, pivot_to_wide, load_lines, sample_chunks,
        get_latest_preprocessed_file.
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
# pivot_to_wide
# =============================================================================


class TestAnnotateObservationRows:
    def test_adds_observation_metadata_from_indicator_specs(self):
        from causal_ssm_agent.utils.data import annotate_observation_rows

        raw = pl.DataFrame(
            {
                "indicator": ["stress_score"],
                "value": ["4.0"],
                "timestamp": ["2024-01-01T00:00:00Z"],
            }
        )
        causal_spec = {
            "measurement": {
                "model_clock": "1d",
                "indicators": [
                    {
                        "name": "stress_score",
                        "measurement_dtype": "continuous",
                        "aggregation": "mean",
                    }
                ],
            }
        }

        annotated = annotate_observation_rows(raw, causal_spec)

        assert "timestamp" not in annotated.columns
        assert annotated["anchor_time"][0] == "2024-01-02T00:00:00"
        assert annotated["support_kind"][0] == "interval"
        assert annotated["summary_operator"][0] == "mean"
        assert annotated["anchor_policy"][0] == "support_end"
        assert annotated["support_start"][0] == "2024-01-01T00:00:00"
        assert annotated["support_end"][0] == "2024-01-02T00:00:00"

    def test_uses_indicator_specific_observation_window_when_present(self):
        from causal_ssm_agent.utils.data import annotate_observation_rows

        raw = pl.DataFrame(
            {
                "indicator": ["monthly_stress_score"],
                "value": ["4.0"],
                "timestamp": ["2024-01-01T00:00:00Z"],
            }
        )
        causal_spec = {
            "measurement": {
                "model_clock": "1d",
                "indicators": [
                    {
                        "name": "monthly_stress_score",
                        "measurement_dtype": "continuous",
                        "aggregation": "mean",
                        "observation_window": "1mo",
                    }
                ],
            }
        }

        annotated = annotate_observation_rows(raw, causal_spec)

        assert "timestamp" not in annotated.columns
        assert annotated["anchor_time"][0] == "2024-02-01T00:00:00"
        assert annotated["observation_window"][0] == "1mo"
        assert annotated["support_start"][0] == "2024-01-01T00:00:00"
        assert annotated["support_end"][0] == "2024-02-01T00:00:00"

    def test_point_last_observations_anchor_at_window_end(self):
        from causal_ssm_agent.utils.data import annotate_observation_rows

        raw = pl.DataFrame(
            {
                "indicator": ["closing_mood"],
                "value": ["4.0"],
                "timestamp": ["2024-01-01T00:00:00Z"],
            }
        )
        causal_spec = {
            "measurement": {
                "model_clock": "1d",
                "indicators": [
                    {
                        "name": "closing_mood",
                        "measurement_dtype": "continuous",
                        "aggregation": "last",
                    }
                ],
            }
        }

        annotated = annotate_observation_rows(raw, causal_spec)

        assert annotated["support_kind"][0] == "point"
        assert annotated["summary_operator"][0] == "last"
        assert annotated["anchor_policy"][0] == "support_end"
        assert annotated["anchor_time"][0] == "2024-01-02T00:00:00"
        assert annotated["support_start"][0] == "2024-01-01T00:00:00"
        assert annotated["support_end"][0] == "2024-01-02T00:00:00"


class TestPivotToWide:
    def test_basic_pivot(self):
        """Simple long-to-wide conversion."""
        df = pl.DataFrame(
            {
                "anchor_time": [1.0, 2.0, 1.0, 2.0],
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

        df = pl.DataFrame({"anchor_time": [], "indicator": [], "value": []})
        result = pivot_to_wide(df)
        assert result.is_empty()

    def test_anchor_time_column(self):
        """Uses anchor_time as the canonical observation time."""
        from causal_ssm_agent.utils.data import pivot_to_wide

        df = pl.DataFrame(
            {
                "anchor_time": [1.0, 2.0],
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
                "anchor_time": [t0, t1, t2],
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
                "anchor_time": [3.0, 1.0, 2.0],
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
                "anchor_time": [1.0, 2.0, 2.0],
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
                "anchor_time": ["2024-01-01", "2024-01-02"],
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
                "anchor_time": [1.0, 2.0],
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
                "anchor_time": [1.0, 1.0, 2.0],
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
                "anchor_time": [1.0],
                "indicator": ["x"],
                "value": [42.0],
            }
        )
        wide = pivot_to_wide(df)
        assert wide.height == 1
        assert "x" in wide.columns
        assert wide["x"][0] == 42.0


class TestPivotToWideSparsity:
    """Test post-pivot sparsity detection (moved from test_stage4.py)."""

    def test_pivot_warns_on_sparse_matrix(self, caplog):
        """Sparse multi-granularity data triggers a warning."""
        import logging

        from causal_ssm_agent.utils.data import pivot_to_wide

        rows = []
        for h in range(24):
            rows.append({"indicator": "hourly_var", "value": float(h), "anchor_time": h})
        rows.append({"indicator": "daily_b", "value": 5.0, "anchor_time": 0})
        rows.append({"indicator": "daily_c", "value": 9.0, "anchor_time": 0})

        raw = pl.DataFrame(rows)
        logger = logging.getLogger("causal_ssm_agent.utils.data")
        logger.propagate = True
        with caplog.at_level(logging.WARNING, logger="causal_ssm_agent.utils.data"):
            wide = pivot_to_wide(raw)

        assert wide.height == 24
        assert any("Sparse observation matrix" in msg for msg in caplog.messages)

    def test_pivot_no_warning_on_complete_matrix(self, caplog):
        """Complete data should not trigger sparsity warning."""
        import logging

        from causal_ssm_agent.utils.data import pivot_to_wide

        rows = []
        for t in range(10):
            rows.append({"indicator": "A", "value": float(t), "anchor_time": t})
            rows.append({"indicator": "B", "value": float(t * 2), "anchor_time": t})

        raw = pl.DataFrame(rows)
        with caplog.at_level(logging.WARNING, logger="causal_ssm_agent.utils.data"):
            pivot_to_wide(raw)

        assert not any("Sparse" in msg for msg in caplog.messages)


class TestPivotToWideTimezoneStrings:
    def test_utc_string_timestamps_parsed(self):
        """UTC timestamps with timezone suffix should parse to fractional days."""
        from causal_ssm_agent.utils.data import pivot_to_wide

        df = pl.DataFrame(
            {
                "anchor_time": ["2024-01-01T00:00:00Z", "2024-01-02T12:00:00Z"],
                "indicator": ["x", "x"],
                "value": [1.0, 2.0],
            }
        )
        wide = pivot_to_wide(df)

        assert wide.schema["time"] == pl.Float64
        times = wide["time"].to_list()
        assert abs(times[0]) < 0.001
        assert abs(times[1] - 1.5) < 0.001
