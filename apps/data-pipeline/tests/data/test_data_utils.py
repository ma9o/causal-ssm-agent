"""Tests for utils/data.py dataframe utility functions."""

from datetime import datetime, timedelta

import polars as pl

# =============================================================================
# pivot_to_wide
# =============================================================================


class TestAnnotateObservationRows:
    def test_adds_observation_metadata_from_indicator_specs(self):
        from nof1_causal_lab.utils.data import annotate_observation_rows

        raw = pl.DataFrame(
            {
                "indicator": ["stress_score"],
                "value": ["4.0"],
                "timestamp": ["2024-01-01T00:00:00Z"],
            }
        )
        measurement_structure = {
            "model_clock": "1d",
            "indicators": [
                {
                    "name": "stress_score",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                }
            ],
        }

        annotated = annotate_observation_rows(raw, measurement_structure)

        assert "timestamp" not in annotated.columns
        assert annotated["anchor_time"][0] == "2024-01-02T00:00:00"
        assert annotated["support_kind"][0] == "interval"
        assert annotated["summary_operator"][0] == "mean"
        assert annotated["anchor_policy"][0] == "support_end"
        assert annotated["support_start"][0] == "2024-01-01T00:00:00"
        assert annotated["support_end"][0] == "2024-01-02T00:00:00"

    def test_uses_indicator_specific_observation_window_when_present(self):
        from nof1_causal_lab.utils.data import annotate_observation_rows

        raw = pl.DataFrame(
            {
                "indicator": ["monthly_stress_score"],
                "value": ["4.0"],
                "timestamp": ["2024-01-01T00:00:00Z"],
            }
        )
        measurement_structure = {
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

        annotated = annotate_observation_rows(raw, measurement_structure)

        assert "timestamp" not in annotated.columns
        assert annotated["anchor_time"][0] == "2024-02-01T00:00:00"
        assert annotated["observation_window"][0] == "1mo"
        assert annotated["support_start"][0] == "2024-01-01T00:00:00"
        assert annotated["support_end"][0] == "2024-02-01T00:00:00"

    def test_point_last_observations_anchor_at_window_end(self):
        from nof1_causal_lab.utils.data import annotate_observation_rows

        raw = pl.DataFrame(
            {
                "indicator": ["closing_mood"],
                "value": ["4.0"],
                "timestamp": ["2024-01-01T00:00:00Z"],
            }
        )
        measurement_structure = {
            "model_clock": "1d",
            "indicators": [
                {
                    "name": "closing_mood",
                    "measurement_dtype": "continuous",
                    "aggregation": "last",
                }
            ],
        }

        annotated = annotate_observation_rows(raw, measurement_structure)

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
        from nof1_causal_lab.utils.data import pivot_to_wide

        wide = pivot_to_wide(df)
        assert "time" in wide.columns
        assert "x" in wide.columns
        assert "y" in wide.columns
        assert wide.height == 2

    def test_empty_dataframe(self):
        """Empty input returns empty output."""
        from nof1_causal_lab.utils.data import pivot_to_wide

        df = pl.DataFrame({"anchor_time": [], "indicator": [], "value": []})
        result = pivot_to_wide(df)
        assert result.is_empty()

    def test_anchor_time_column(self):
        """Uses anchor_time as the canonical observation time."""
        from nof1_causal_lab.utils.data import pivot_to_wide

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
        from nof1_causal_lab.utils.data import pivot_to_wide

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
        from nof1_causal_lab.utils.data import pivot_to_wide

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
        from nof1_causal_lab.utils.data import pivot_to_wide

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
        from nof1_causal_lab.utils.data import pivot_to_wide

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
        from nof1_causal_lab.utils.data import pivot_to_wide

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
        from nof1_causal_lab.utils.data import pivot_to_wide

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
        from nof1_causal_lab.utils.data import pivot_to_wide

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

        from nof1_causal_lab.utils.data import pivot_to_wide

        rows = []
        for h in range(24):
            rows.append({"indicator": "hourly_var", "value": float(h), "anchor_time": h})
        rows.append({"indicator": "daily_b", "value": 5.0, "anchor_time": 0})
        rows.append({"indicator": "daily_c", "value": 9.0, "anchor_time": 0})

        raw = pl.DataFrame(rows)
        logger = logging.getLogger("nof1_causal_lab.utils.data")
        logger.propagate = True
        with caplog.at_level(logging.WARNING, logger="nof1_causal_lab.utils.data"):
            wide = pivot_to_wide(raw)

        assert wide.height == 24
        assert any("Sparse observation matrix" in msg for msg in caplog.messages)

    def test_pivot_no_warning_on_complete_matrix(self, caplog):
        """Complete data should not trigger sparsity warning."""
        import logging

        from nof1_causal_lab.utils.data import pivot_to_wide

        rows = []
        for t in range(10):
            rows.append({"indicator": "A", "value": float(t), "anchor_time": t})
            rows.append({"indicator": "B", "value": float(t * 2), "anchor_time": t})

        raw = pl.DataFrame(rows)
        with caplog.at_level(logging.WARNING, logger="nof1_causal_lab.utils.data"):
            pivot_to_wide(raw)

        assert not any("Sparse" in msg for msg in caplog.messages)


class TestPivotToWideTimezoneStrings:
    def test_utc_string_timestamps_parsed(self):
        """UTC timestamps with timezone suffix should parse to fractional days."""
        from nof1_causal_lab.utils.data import pivot_to_wide

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
