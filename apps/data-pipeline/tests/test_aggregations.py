"""Tests for aggregation utility functions.

Covers: _build_agg_expr, _build_map_groups_fn, _encode_non_continuous,
flatten_aggregated_data, aggregate_worker_measurements.
"""

import polars as pl
import pytest

from causal_ssm_agent.utils.aggregations import (
    _build_agg_expr,
    _build_map_groups_fn,
    _encode_non_continuous,
    aggregate_worker_measurements,
    flatten_aggregated_data,
)


def _make_df(values: list[float]) -> pl.DataFrame:
    """Create a simple DataFrame with a 'value' column."""
    return pl.DataFrame({"value": values})


# =============================================================================
# _build_agg_expr
# =============================================================================


class TestBuildAggExpr:
    def test_mean(self):
        df = _make_df([1.0, 2.0, 3.0])
        result = df.select(_build_agg_expr("mean"))
        assert abs(result["value"][0] - 2.0) < 1e-10

    def test_sum(self):
        df = _make_df([1.0, 2.0, 3.0])
        result = df.select(_build_agg_expr("sum"))
        assert abs(result["value"][0] - 6.0) < 1e-10

    def test_min(self):
        df = _make_df([3.0, 1.0, 2.0])
        result = df.select(_build_agg_expr("min"))
        assert abs(result["value"][0] - 1.0) < 1e-10

    def test_max(self):
        df = _make_df([3.0, 1.0, 2.0])
        result = df.select(_build_agg_expr("max"))
        assert abs(result["value"][0] - 3.0) < 1e-10

    def test_std(self):
        df = _make_df([1.0, 2.0, 3.0])
        result = df.select(_build_agg_expr("std"))
        assert result["value"][0] > 0

    def test_count(self):
        df = _make_df([1.0, 2.0, 3.0])
        result = df.select(_build_agg_expr("count"))
        assert result["value"][0] == 3

    def test_median(self):
        df = _make_df([1.0, 5.0, 3.0])
        result = df.select(_build_agg_expr("median"))
        assert abs(result["value"][0] - 3.0) < 1e-10

    def test_first(self):
        df = _make_df([7.0, 2.0, 3.0])
        result = df.select(_build_agg_expr("first"))
        assert abs(result["value"][0] - 7.0) < 1e-10

    def test_last(self):
        df = _make_df([7.0, 2.0, 9.0])
        result = df.select(_build_agg_expr("last"))
        assert abs(result["value"][0] - 9.0) < 1e-10

    def test_range(self):
        df = _make_df([1.0, 5.0, 3.0])
        result = df.select(_build_agg_expr("range"))
        assert abs(result["value"][0] - 4.0) < 1e-10

    def test_p25(self):
        df = _make_df([1.0, 2.0, 3.0, 4.0])
        result = df.select(_build_agg_expr("p25"))
        assert result["value"][0] is not None

    def test_p75(self):
        df = _make_df([1.0, 2.0, 3.0, 4.0])
        result = df.select(_build_agg_expr("p75"))
        assert result["value"][0] is not None

    def test_iqr(self):
        df = _make_df([1.0, 2.0, 3.0, 4.0])
        result = df.select(_build_agg_expr("iqr"))
        assert result["value"][0] >= 0

    def test_cv_nonzero_mean(self):
        df = _make_df([10.0, 12.0, 8.0])
        result = df.select(_build_agg_expr("cv"))
        assert result["value"][0] is not None
        assert result["value"][0] > 0

    def test_instability(self):
        df = _make_df([1.0, 3.0, 2.0, 4.0])
        result = df.select(_build_agg_expr("instability"))
        assert result["value"][0] > 0

    def test_output_alias_is_value(self):
        """All expressions should alias to 'value'."""
        for agg in ["mean", "sum", "min", "max", "range", "p25", "cv", "instability"]:
            df = _make_df([1.0, 2.0, 3.0])
            result = df.select(_build_agg_expr(agg))
            assert "value" in result.columns

    def test_single_value(self):
        """Aggregating a single value works for all functions."""
        df = _make_df([42.0])
        for agg in ["mean", "sum", "min", "max", "count", "median", "first", "last"]:
            result = df.select(_build_agg_expr(agg))
            assert result["value"][0] is not None, f"{agg} failed on single value"

    def test_cv_zero_mean(self):
        """CV with zero mean should handle division safely."""
        df = _make_df([-1.0, 1.0])  # mean = 0
        result = df.select(_build_agg_expr("cv"))
        # Should not crash; result may be inf or null
        assert result.shape == (1, 1)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown aggregation"):
            _build_agg_expr("nonexistent_agg")


# =============================================================================
# _build_map_groups_fn
# =============================================================================


class TestBuildMapGroupsFn:
    def test_trend_positive_slope(self):
        """Increasing values should give positive slope."""
        fn = _build_map_groups_fn("trend")
        df = pl.DataFrame({"value": [1.0, 2.0, 3.0, 4.0], "group": ["a"] * 4})
        result = fn(df)
        assert result["value"][0] > 0

    def test_trend_zero_slope(self):
        """Constant values should give zero slope."""
        fn = _build_map_groups_fn("trend")
        df = pl.DataFrame({"value": [5.0, 5.0, 5.0], "group": ["a"] * 3})
        result = fn(df)
        assert abs(result["value"][0]) < 1e-10

    def test_trend_single_point(self):
        """Single data point should give zero slope."""
        fn = _build_map_groups_fn("trend")
        df = pl.DataFrame({"value": [5.0], "group": ["a"]})
        result = fn(df)
        assert abs(result["value"][0]) < 1e-10

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown map_groups"):
            _build_map_groups_fn("nonexistent")


# =============================================================================
# _encode_non_continuous
# =============================================================================


class TestEncodeNonContinuous:
    def test_binary_true_false(self):
        df = pl.DataFrame({
            "indicator": ["mood", "mood", "mood"],
            "value": ["true", "false", "yes"],
        })
        result = _encode_non_continuous(df, {"mood": "binary"})
        values = result.sort("value")["value"].to_list()
        # "false" -> "0.0", "true" -> "1.0", "yes" -> "1.0"
        assert "0.0" in values
        assert "1.0" in values

    def test_ordinal_encoding(self):
        df = pl.DataFrame({
            "indicator": ["pain", "pain", "pain"],
            "value": ["low", "medium", "high"],
        })
        result = _encode_non_continuous(
            df,
            {"pain": "ordinal"},
            ordinal_levels_lookup={"pain": ["low", "medium", "high"]},
        )
        # low=0, medium=1, high=2
        vals = sorted(float(v) for v in result["value"].to_list())
        assert vals == [0.0, 1.0, 2.0]

    def test_continuous_passthrough(self):
        df = pl.DataFrame({
            "indicator": ["weight", "weight"],
            "value": [70.5, 80.2],
        })
        result = _encode_non_continuous(df, {"weight": "continuous"})
        assert len(result) == 2

    def test_empty_dtype_lookup(self):
        df = pl.DataFrame({"indicator": ["x"], "value": [1.0]})
        result = _encode_non_continuous(df, {})
        assert len(result) == 1

    def test_mixed_indicators(self):
        """Only non-continuous indicators should be encoded."""
        df = pl.DataFrame({
            "indicator": ["mood", "weight"],
            "value": ["true", "70.5"],
        })
        result = _encode_non_continuous(df, {"mood": "binary", "weight": "continuous"})
        assert len(result) == 2


# =============================================================================
# flatten_aggregated_data
# =============================================================================


class TestFlattenAggregatedData:
    def test_empty_dict(self):
        result = flatten_aggregated_data({})
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0
        assert set(result.columns) == {"indicator", "value", "time_bucket"}

    def test_single_granularity(self):
        df = pl.DataFrame({
            "indicator": ["mood", "mood"],
            "value": [3.0, 4.0],
            "time_bucket": ["2024-01-01", "2024-01-02"],
        })
        result = flatten_aggregated_data({"daily": df})
        assert len(result) == 2
        assert set(result.columns) == {"indicator", "value", "time_bucket"}

    def test_multiple_granularities_combined(self):
        df1 = pl.DataFrame({
            "indicator": ["mood"],
            "value": [3.0],
            "time_bucket": ["2024-01-01"],
        })
        df2 = pl.DataFrame({
            "indicator": ["mood"],
            "value": [4.0],
            "time_bucket": ["2024-01-08"],
        })
        result = flatten_aggregated_data({"daily": df1, "weekly": df2})
        assert len(result) == 2

    def test_sorted_output(self):
        df = pl.DataFrame({
            "indicator": ["sleep", "mood", "mood"],
            "value": [8.0, 3.0, 4.0],
            "time_bucket": ["2024-01-01", "2024-01-02", "2024-01-01"],
        })
        result = flatten_aggregated_data({"daily": df})
        indicators = result["indicator"].to_list()
        assert indicators[0] == "mood"  # sorted by indicator first


# =============================================================================
# aggregate_worker_measurements
# =============================================================================


def _worker_df(rows):
    """Build a worker DataFrame from list of (indicator, value, timestamp) tuples."""
    return pl.DataFrame(
        {
            "indicator": [r[0] for r in rows],
            "value": [str(r[1]) for r in rows],
            "timestamp": [r[2] for r in rows],
        },
        schema={"indicator": pl.Utf8, "value": pl.Utf8, "timestamp": pl.Utf8},
    )


def _causal_spec_for_agg(*indicators):
    """Build a causal spec for aggregation tests. Each indicator: (name, dtype, aggregation)."""
    return {
        "measurement": {
            "indicators": [
                {
                    "name": name,
                    "measurement_dtype": dtype,
                    "aggregation": agg,
                }
                for name, dtype, agg in indicators
            ]
        }
    }


class TestAggregateWorkerMeasurements:
    def test_empty_input(self):
        spec = _causal_spec_for_agg(("mood", "continuous", "mean"))
        result = aggregate_worker_measurements([], spec)
        assert result == {}

    def test_all_none_input(self):
        spec = _causal_spec_for_agg(("mood", "continuous", "mean"))
        result = aggregate_worker_measurements([None, None], spec)
        assert result == {}

    def test_basic_daily_aggregation(self):
        df = _worker_df([
            ("mood", 3.0, "2024-01-01T10:00:00"),
            ("mood", 5.0, "2024-01-01T14:00:00"),
            ("mood", 7.0, "2024-01-02T10:00:00"),
        ])
        spec = _causal_spec_for_agg(("mood", "continuous", "mean"))
        result = aggregate_worker_measurements([df], spec, "daily")
        assert "daily" in result
        agged = result["daily"]
        assert len(agged) == 2  # 2 days
        # Day 1 should be mean(3, 5) = 4
        day1 = agged.filter(pl.col("time_bucket") == agged["time_bucket"][0])
        assert abs(day1["value"][0] - 4.0) < 1e-6

    def test_multiple_workers_combined(self):
        df1 = _worker_df([("mood", 3.0, "2024-01-01T10:00:00")])
        df2 = _worker_df([("mood", 5.0, "2024-01-01T14:00:00")])
        spec = _causal_spec_for_agg(("mood", "continuous", "mean"))
        result = aggregate_worker_measurements([df1, df2], spec, "daily")
        assert "daily" in result
        agged = result["daily"]
        assert len(agged) == 1
        assert abs(agged["value"][0] - 4.0) < 1e-6

    def test_none_dfs_filtered(self):
        df = _worker_df([("mood", 3.0, "2024-01-01T10:00:00")])
        spec = _causal_spec_for_agg(("mood", "continuous", "mean"))
        result = aggregate_worker_measurements([None, df, None], spec, "daily")
        assert "daily" in result

    def test_finest_no_truncation(self):
        df = _worker_df([
            ("mood", 3.0, "2024-01-01T10:00:00"),
            ("mood", 5.0, "2024-01-01T14:00:00"),
        ])
        spec = _causal_spec_for_agg(("mood", "continuous", "mean"))
        result = aggregate_worker_measurements([df], spec, "finest")
        assert "finest" in result
        assert len(result["finest"]) == 2  # no aggregation

    def test_unknown_aggregation_window_raises(self):
        df = _worker_df([("mood", 3.0, "2024-01-01T10:00:00")])
        spec = _causal_spec_for_agg(("mood", "continuous", "mean"))
        with pytest.raises(ValueError, match="Unknown aggregation_window"):
            aggregate_worker_measurements([df], spec, "invalid_window")

    def test_single_row_df(self):
        """A DataFrame with one row still aggregates correctly."""
        df = _worker_df([("mood", 3.0, "2024-01-01T10:00:00")])
        spec = _causal_spec_for_agg(("mood", "continuous", "mean"))
        result = aggregate_worker_measurements([df], spec, "daily")
        assert "daily" in result
        assert len(result["daily"]) == 1
        assert abs(result["daily"]["value"][0] - 3.0) < 1e-6

    def test_unknown_indicators_filtered(self):
        df = _worker_df([
            ("mood", 3.0, "2024-01-01T10:00:00"),
            ("unknown_ind", 5.0, "2024-01-01T10:00:00"),
        ])
        spec = _causal_spec_for_agg(("mood", "continuous", "mean"))
        result = aggregate_worker_measurements([df], spec, "daily")
        assert "daily" in result
        indicators = result["daily"]["indicator"].unique().to_list()
        assert "unknown_ind" not in indicators
