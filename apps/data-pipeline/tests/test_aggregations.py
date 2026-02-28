"""Tests for aggregation utility functions.

Covers: _build_agg_expr, _build_map_groups_fn, _encode_non_continuous.
"""

import polars as pl
import pytest

from causal_ssm_agent.utils.aggregations import (
    _build_agg_expr,
    _build_map_groups_fn,
    _encode_non_continuous,
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
