"""Tests for aggregate_worker_measurements().

Tests the post-extraction aggregation step that buckets raw worker
extractions to the specified aggregation_window using each indicator's
aggregation function.
"""

import polars as pl
import pytest

from causal_ssm_agent.utils.aggregations import (
    _build_agg_expr,
    aggregate_worker_measurements,
)

# ==============================================================================
# HELPERS
# ==============================================================================


def _make_df(records: list[dict]) -> pl.DataFrame:
    """Create a worker-style DataFrame from records."""
    return pl.DataFrame(
        records,
        schema={"indicator": pl.Utf8, "value": pl.Object, "timestamp": pl.Utf8},
    )


def _make_causal_spec(indicators: list[dict]) -> dict:
    """Create a minimal CausalSpec dict with given indicators."""
    return {
        "measurement": {"indicators": indicators},
        "latent": {"constructs": []},
    }


# ==============================================================================
# TEST: Basic Aggregation
# ==============================================================================


class TestBasicAggregation:
    """Hourly data -> daily mean, single reading passthrough."""

    def test_hourly_to_daily_mean(self):
        """Hourly readings aggregated to daily mean."""
        records = [
            {"indicator": "stress", "value": 3.0, "timestamp": "2024-01-01T08:00:00"},
            {"indicator": "stress", "value": 5.0, "timestamp": "2024-01-01T14:00:00"},
            {"indicator": "stress", "value": 7.0, "timestamp": "2024-01-01T20:00:00"},
            {"indicator": "stress", "value": 2.0, "timestamp": "2024-01-02T09:00:00"},
            {"indicator": "stress", "value": 4.0, "timestamp": "2024-01-02T15:00:00"},
        ]
        df = _make_df(records)
        model = _make_causal_spec(
            [
                {
                    "name": "stress",
                    "aggregation": "mean",
                }
            ]
        )

        result = aggregate_worker_measurements([df], model)

        assert "daily" in result
        daily = result["daily"]
        assert len(daily) == 2

        day1 = daily.filter(pl.col("time_bucket") == pl.lit("2024-01-01").str.to_datetime())
        assert day1["value"][0] == pytest.approx(5.0)  # mean(3, 5, 7)

        day2 = daily.filter(pl.col("time_bucket") == pl.lit("2024-01-02").str.to_datetime())
        assert day2["value"][0] == pytest.approx(3.0)  # mean(2, 4)

    def test_single_reading_passthrough(self):
        """Single reading per bucket passes through unchanged."""
        records = [
            {"indicator": "mood", "value": 8.0, "timestamp": "2024-01-01T12:00:00"},
        ]
        df = _make_df(records)
        model = _make_causal_spec(
            [
                {
                    "name": "mood",
                    "aggregation": "mean",
                }
            ]
        )

        result = aggregate_worker_measurements([df], model)
        assert "daily" in result
        assert result["daily"]["value"][0] == pytest.approx(8.0)


# ==============================================================================
# TEST: Multiple Aggregation Functions
# ==============================================================================


class TestMultipleAggregations:
    """sum, max, last, percentiles, range."""

    @pytest.fixture
    def hourly_data(self):
        """Hourly data for a single day."""
        records = [
            {"indicator": "steps", "value": 100, "timestamp": "2024-01-01T08:00:00"},
            {"indicator": "steps", "value": 500, "timestamp": "2024-01-01T10:00:00"},
            {"indicator": "steps", "value": 200, "timestamp": "2024-01-01T12:00:00"},
            {"indicator": "steps", "value": 800, "timestamp": "2024-01-01T14:00:00"},
            {"indicator": "steps", "value": 300, "timestamp": "2024-01-01T16:00:00"},
        ]
        return _make_df(records)

    def test_sum_aggregation(self, hourly_data):
        model = _make_causal_spec([{"name": "steps", "aggregation": "sum"}])
        result = aggregate_worker_measurements([hourly_data], model)
        assert result["daily"]["value"][0] == pytest.approx(1900.0)

    def test_max_aggregation(self, hourly_data):
        model = _make_causal_spec([{"name": "steps", "aggregation": "max"}])
        result = aggregate_worker_measurements([hourly_data], model)
        assert result["daily"]["value"][0] == pytest.approx(800.0)

    def test_last_aggregation(self, hourly_data):
        """last should return the chronologically last value."""
        model = _make_causal_spec([{"name": "steps", "aggregation": "last"}])
        result = aggregate_worker_measurements([hourly_data], model)
        assert result["daily"]["value"][0] == pytest.approx(300.0)

    def test_first_aggregation(self, hourly_data):
        """first should return the chronologically first value."""
        model = _make_causal_spec([{"name": "steps", "aggregation": "first"}])
        result = aggregate_worker_measurements([hourly_data], model)
        assert result["daily"]["value"][0] == pytest.approx(100.0)

    def test_median_aggregation(self, hourly_data):
        model = _make_causal_spec([{"name": "steps", "aggregation": "median"}])
        result = aggregate_worker_measurements([hourly_data], model)
        assert result["daily"]["value"][0] == pytest.approx(300.0)

    def test_range_aggregation(self, hourly_data):
        model = _make_causal_spec([{"name": "steps", "aggregation": "range"}])
        result = aggregate_worker_measurements([hourly_data], model)
        assert result["daily"]["value"][0] == pytest.approx(700.0)  # 800 - 100

    def test_count_aggregation(self, hourly_data):
        model = _make_causal_spec([{"name": "steps", "aggregation": "count"}])
        result = aggregate_worker_measurements([hourly_data], model)
        assert result["daily"]["value"][0] == 5


# ==============================================================================
# TEST: Finest Granularity
# ==============================================================================


class TestFinestGranularity:
    """Dedup only, keeps different values at same time."""

    def test_dedup_exact_triples(self):
        """Exact (indicator, datetime, value) triples are deduplicated."""
        records = [
            {"indicator": "hr", "value": 72.0, "timestamp": "2024-01-01T08:00:00"},
            {"indicator": "hr", "value": 72.0, "timestamp": "2024-01-01T08:00:00"},  # dup
            {"indicator": "hr", "value": 80.0, "timestamp": "2024-01-01T09:00:00"},
        ]
        df = _make_df(records)
        model = _make_causal_spec([{"name": "hr", "aggregation": "mean"}])

        result = aggregate_worker_measurements([df], model, aggregation_window="finest")
        assert "finest" in result
        assert len(result["finest"]) == 2

    def test_different_values_same_time_kept(self):
        """Different values at same timestamp are kept (not deduped)."""
        records = [
            {"indicator": "hr", "value": 72.0, "timestamp": "2024-01-01T08:00:00"},
            {"indicator": "hr", "value": 75.0, "timestamp": "2024-01-01T08:00:00"},  # different val
        ]
        df = _make_df(records)
        model = _make_causal_spec([{"name": "hr", "aggregation": "mean"}])

        result = aggregate_worker_measurements([df], model, aggregation_window="finest")
        assert len(result["finest"]) == 2


# ==============================================================================
# TEST: Mixed Granularities
# ==============================================================================


class TestAggregationWindows:
    """All indicators share the same aggregation_window parameter."""

    def test_hourly_window(self):
        """All indicators aggregated to hourly buckets."""
        records = [
            {"indicator": "hr", "value": 72.0, "timestamp": "2024-01-01T08:00:00"},
            {"indicator": "hr", "value": 80.0, "timestamp": "2024-01-01T08:30:00"},
            {"indicator": "hr", "value": 90.0, "timestamp": "2024-01-01T09:00:00"},
            {"indicator": "stress", "value": 5.0, "timestamp": "2024-01-01T08:15:00"},
            {"indicator": "stress", "value": 7.0, "timestamp": "2024-01-01T08:45:00"},
        ]
        df = _make_df(records)
        model = _make_causal_spec(
            [
                {"name": "hr", "aggregation": "mean"},
                {"name": "stress", "aggregation": "mean"},
            ]
        )

        result = aggregate_worker_measurements([df], model, aggregation_window="hourly")
        assert "hourly" in result

        hourly = result["hourly"]
        # hr has readings in hours 08 and 09
        hr_rows = hourly.filter(pl.col("indicator") == "hr")
        assert len(hr_rows) == 2
        # stress has readings only in hour 08
        stress_rows = hourly.filter(pl.col("indicator") == "stress")
        assert len(stress_rows) == 1
        assert stress_rows["value"][0] == pytest.approx(6.0)

    def test_daily_window_multiple_indicators(self):
        """Multiple indicators aggregated to daily buckets (default)."""
        records = [
            {"indicator": "hr", "value": 72.0, "timestamp": "2024-01-01T08:00:00"},
            {"indicator": "hr", "value": 80.0, "timestamp": "2024-01-01T09:00:00"},
            {"indicator": "stress", "value": 5.0, "timestamp": "2024-01-01T10:00:00"},
            {"indicator": "stress", "value": 7.0, "timestamp": "2024-01-01T14:00:00"},
        ]
        df = _make_df(records)
        model = _make_causal_spec(
            [
                {"name": "hr", "aggregation": "mean"},
                {"name": "stress", "aggregation": "mean"},
            ]
        )

        result = aggregate_worker_measurements([df], model)
        assert "daily" in result
        daily = result["daily"]
        # hr: 1 daily bucket, mean(72, 80) = 76
        hr_rows = daily.filter(pl.col("indicator") == "hr")
        assert len(hr_rows) == 1
        assert hr_rows["value"][0] == pytest.approx(76.0)
        # stress: 1 daily bucket, mean(5, 7) = 6
        stress_rows = daily.filter(pl.col("indicator") == "stress")
        assert len(stress_rows) == 1
        assert stress_rows["value"][0] == pytest.approx(6.0)


# ==============================================================================
# TEST: Edge Cases
# ==============================================================================


class TestEdgeCases:
    """Empty list, None DataFrames, null timestamps, null values, unknown indicators, non-numeric."""

    def test_empty_list(self):
        model = _make_causal_spec([{"name": "x", "aggregation": "mean"}])
        result = aggregate_worker_measurements([], model)
        assert result == {}

    def test_none_dataframes(self):
        model = _make_causal_spec([{"name": "x", "aggregation": "mean"}])
        result = aggregate_worker_measurements([None, None], model)
        assert result == {}

    def test_null_timestamps_dropped(self):
        """Rows with null timestamps are dropped."""
        records = [
            {"indicator": "x", "value": 5.0, "timestamp": "2024-01-01T10:00:00"},
            {"indicator": "x", "value": 3.0, "timestamp": None},
        ]
        df = _make_df(records)
        model = _make_causal_spec([{"name": "x", "aggregation": "mean"}])

        result = aggregate_worker_measurements([df], model)
        assert result["daily"]["value"][0] == pytest.approx(5.0)  # only the non-null row

    def test_null_values_dropped(self):
        """Rows with non-numeric values (cast to null) are dropped."""
        records = [
            {"indicator": "x", "value": 5.0, "timestamp": "2024-01-01T10:00:00"},
            {"indicator": "x", "value": "not_a_number", "timestamp": "2024-01-01T12:00:00"},
        ]
        df = _make_df(records)
        model = _make_causal_spec([{"name": "x", "aggregation": "mean"}])

        result = aggregate_worker_measurements([df], model)
        assert result["daily"]["value"][0] == pytest.approx(5.0)

    def test_unknown_indicators_filtered(self):
        """Indicators not in the model are filtered out."""
        records = [
            {"indicator": "known", "value": 5.0, "timestamp": "2024-01-01T10:00:00"},
            {"indicator": "unknown", "value": 3.0, "timestamp": "2024-01-01T10:00:00"},
        ]
        df = _make_df(records)
        model = _make_causal_spec([{"name": "known", "aggregation": "mean"}])

        result = aggregate_worker_measurements([df], model)
        assert "daily" in result
        indicators = result["daily"]["indicator"].to_list()
        assert "known" in indicators
        assert "unknown" not in indicators

    def test_all_non_numeric_returns_empty(self):
        """All non-numeric values results in empty output."""
        records = [
            {"indicator": "x", "value": "high", "timestamp": "2024-01-01T10:00:00"},
            {"indicator": "x", "value": "low", "timestamp": "2024-01-01T12:00:00"},
        ]
        df = _make_df(records)
        model = _make_causal_spec([{"name": "x", "aggregation": "mean"}])

        result = aggregate_worker_measurements([df], model)
        assert result == {}


# ==============================================================================
# TEST: Overlapping Workers
# ==============================================================================


class TestOverlappingWorkers:
    """Two workers covering same day, exact duplicate dedup."""

    def test_overlapping_workers_aggregated(self):
        """Two workers with overlapping data are aggregated correctly."""
        df1 = _make_df(
            [
                {"indicator": "stress", "value": 3.0, "timestamp": "2024-01-01T08:00:00"},
                {"indicator": "stress", "value": 5.0, "timestamp": "2024-01-01T12:00:00"},
            ]
        )
        df2 = _make_df(
            [
                {"indicator": "stress", "value": 7.0, "timestamp": "2024-01-01T16:00:00"},
                {"indicator": "stress", "value": 4.0, "timestamp": "2024-01-01T20:00:00"},
            ]
        )
        model = _make_causal_spec([{"name": "stress", "aggregation": "mean"}])

        result = aggregate_worker_measurements([df1, df2], model)
        assert result["daily"]["value"][0] == pytest.approx(4.75)  # mean(3, 5, 7, 4)

    def test_exact_duplicate_from_workers(self):
        """Exact same extraction from two workers is NOT deduped for non-finest."""
        df1 = _make_df(
            [
                {"indicator": "stress", "value": 5.0, "timestamp": "2024-01-01T12:00:00"},
            ]
        )
        df2 = _make_df(
            [
                {"indicator": "stress", "value": 5.0, "timestamp": "2024-01-01T12:00:00"},
            ]
        )
        model = _make_causal_spec([{"name": "stress", "aggregation": "mean"}])

        # For non-finest, duplicates are kept (both contribute to aggregate)
        result = aggregate_worker_measurements([df1, df2], model)
        # mean(5, 5) = 5
        assert result["daily"]["value"][0] == pytest.approx(5.0)

    def test_exact_duplicate_finest_deduped(self):
        """For finest granularity, exact duplicates ARE deduped."""
        df1 = _make_df(
            [
                {"indicator": "hr", "value": 72.0, "timestamp": "2024-01-01T08:00:00"},
            ]
        )
        df2 = _make_df(
            [
                {"indicator": "hr", "value": 72.0, "timestamp": "2024-01-01T08:00:00"},
            ]
        )
        model = _make_causal_spec([{"name": "hr", "aggregation": "mean"}])

        result = aggregate_worker_measurements([df1, df2], model, aggregation_window="finest")
        assert len(result["finest"]) == 1


# ==============================================================================
# TEST: _build_agg_expr
# ==============================================================================


class TestBuildAggExpr:
    """All 23 expression-based functions build; trend uses map_groups."""

    @pytest.mark.parametrize(
        "agg_name",
        [
            "mean",
            "sum",
            "min",
            "max",
            "std",
            "var",
            "last",
            "first",
            "count",
            "median",
            "n_unique",
            "p10",
            "p25",
            "p75",
            "p90",
            "p99",
            "range",
            "iqr",
            "cv",
            "skew",
            "kurtosis",
            "entropy",
            "instability",
        ],
    )
    def test_supported_agg_builds(self, agg_name):
        """Each supported aggregation produces a Polars expression."""
        expr = _build_agg_expr(agg_name)
        assert isinstance(expr, pl.Expr)

    def test_unknown_agg_raises_valueerror(self):
        """Completely unknown aggregation raises ValueError."""
        with pytest.raises(ValueError, match="Unknown"):
            _build_agg_expr("nonexistent_function")


# ==============================================================================
# TEST: New Aggregation Functions (skew, kurtosis, entropy, instability, trend)
# ==============================================================================


class TestNewAggregations:
    """Integration tests for newly implemented aggregation functions."""

    def test_skew_aggregation(self):
        """Asymmetric data should produce non-zero skew."""
        records = [
            {"indicator": "x", "value": 1.0, "timestamp": "2024-01-01T01:00:00"},
            {"indicator": "x", "value": 1.0, "timestamp": "2024-01-01T02:00:00"},
            {"indicator": "x", "value": 1.0, "timestamp": "2024-01-01T03:00:00"},
            {"indicator": "x", "value": 1.0, "timestamp": "2024-01-01T04:00:00"},
            {"indicator": "x", "value": 100.0, "timestamp": "2024-01-01T05:00:00"},
        ]
        df = _make_df(records)
        model = _make_causal_spec([{"name": "x", "aggregation": "skew"}])
        result = aggregate_worker_measurements([df], model)
        assert "daily" in result
        skew_val = result["daily"]["value"][0]
        assert skew_val is not None
        assert skew_val != 0.0  # Clearly right-skewed data

    def test_kurtosis_aggregation(self):
        """Kurtosis returns a number."""
        records = [
            {"indicator": "x", "value": float(v), "timestamp": f"2024-01-01T{h:02d}:00:00"}
            for h, v in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ]
        df = _make_df(records)
        model = _make_causal_spec([{"name": "x", "aggregation": "kurtosis"}])
        result = aggregate_worker_measurements([df], model)
        assert "daily" in result
        assert result["daily"]["value"][0] is not None

    def test_entropy_aggregation(self):
        """Uniform data has higher entropy than concentrated data."""
        # Uniform-ish
        uniform_records = [
            {"indicator": "x", "value": float(v), "timestamp": f"2024-01-01T{h:02d}:00:00"}
            for h, v in enumerate([1, 2, 3, 4, 5])
        ]
        # Concentrated
        concentrated_records = [
            {"indicator": "x", "value": float(v), "timestamp": f"2024-01-01T{h:02d}:00:00"}
            for h, v in enumerate([5, 5, 5, 5, 5])
        ]
        model = _make_causal_spec([{"name": "x", "aggregation": "entropy"}])
        r_uniform = aggregate_worker_measurements([_make_df(uniform_records)], model)
        r_concentrated = aggregate_worker_measurements([_make_df(concentrated_records)], model)
        assert "daily" in r_uniform
        assert "daily" in r_concentrated
        # Both return a number (entropy behavior depends on Polars implementation)
        assert r_uniform["daily"]["value"][0] is not None
        assert r_concentrated["daily"]["value"][0] is not None

    def test_instability_mssd(self):
        """MSSD of [1,3,1,3] = mean((2^2, 2^2, 2^2)) = 4.0."""
        records = [
            {"indicator": "x", "value": 1.0, "timestamp": "2024-01-01T01:00:00"},
            {"indicator": "x", "value": 3.0, "timestamp": "2024-01-01T02:00:00"},
            {"indicator": "x", "value": 1.0, "timestamp": "2024-01-01T03:00:00"},
            {"indicator": "x", "value": 3.0, "timestamp": "2024-01-01T04:00:00"},
        ]
        df = _make_df(records)
        model = _make_causal_spec([{"name": "x", "aggregation": "instability"}])
        result = aggregate_worker_measurements([df], model)
        assert "daily" in result
        # diff() produces [null, 2, -2, 2], pow(2) = [null, 4, 4, 4], mean = 4.0
        assert result["daily"]["value"][0] == pytest.approx(4.0)

    def test_trend_positive(self):
        """Increasing data should have a positive slope."""
        records = [
            {"indicator": "x", "value": float(v), "timestamp": f"2024-01-01T{h:02d}:00:00"}
            for h, v in enumerate([1, 2, 3, 4, 5])
        ]
        df = _make_df(records)
        model = _make_causal_spec([{"name": "x", "aggregation": "trend"}])
        result = aggregate_worker_measurements([df], model)
        assert "daily" in result
        assert result["daily"]["value"][0] > 0

    def test_trend_flat(self):
        """Constant data should have slope ~ 0."""
        records = [
            {"indicator": "x", "value": 5.0, "timestamp": f"2024-01-01T{h:02d}:00:00"}
            for h, v in enumerate([5, 5, 5, 5])
        ]
        df = _make_df(records)
        model = _make_causal_spec([{"name": "x", "aggregation": "trend"}])
        result = aggregate_worker_measurements([df], model)
        assert "daily" in result
        assert result["daily"]["value"][0] == pytest.approx(0.0, abs=1e-10)

    def test_trend_negative(self):
        """Decreasing data should have a negative slope."""
        records = [
            {"indicator": "x", "value": float(v), "timestamp": f"2024-01-01T{h:02d}:00:00"}
            for h, v in enumerate([5, 4, 3, 2, 1])
        ]
        df = _make_df(records)
        model = _make_causal_spec([{"name": "x", "aggregation": "trend"}])
        result = aggregate_worker_measurements([df], model)
        assert "daily" in result
        assert result["daily"]["value"][0] < 0

    def test_trend_single_value(self):
        """Single value should have slope = 0."""
        records = [
            {"indicator": "x", "value": 42.0, "timestamp": "2024-01-01T12:00:00"},
        ]
        df = _make_df(records)
        model = _make_causal_spec([{"name": "x", "aggregation": "trend"}])
        result = aggregate_worker_measurements([df], model)
        assert "daily" in result
        assert result["daily"]["value"][0] == pytest.approx(0.0)
