"""Tests for aggregation utility functions.

Covers: _build_agg_expr, _build_map_groups_fn, _encode_non_continuous, compute_indicators.
"""

from datetime import datetime

import polars as pl
import pytest

from nof1_causal_lab.utils.aggregations import (
    _build_agg_expr,
    _build_map_groups_fn,
    _encode_non_continuous,
    compute_indicators,
)


def _make_df(values: list[float]) -> pl.DataFrame:
    """Create a simple DataFrame with a 'value' column."""
    return pl.DataFrame({"value": values})


# =============================================================================
# _build_agg_expr
# =============================================================================


class TestBuildAggExpr:
    @pytest.mark.parametrize(
        ("agg_name", "values", "expected"),
        [
            ("mean", [1.0, 2.0, 3.0], 2.0),
            ("sum", [1.0, 2.0, 3.0], 6.0),
            ("min", [3.0, 1.0, 2.0], 1.0),
            ("max", [3.0, 1.0, 2.0], 3.0),
            ("count", [1.0, 2.0, 3.0], 3.0),
            ("median", [1.0, 5.0, 3.0], 3.0),
            ("first", [7.0, 2.0, 3.0], 7.0),
            ("last", [7.0, 2.0, 9.0], 9.0),
            ("range", [1.0, 5.0, 3.0], 4.0),
            ("p25", [1.0, 2.0, 3.0, 4.0], 2.0),
            ("p75", [1.0, 2.0, 3.0, 4.0], 3.0),
            ("iqr", [1.0, 2.0, 3.0, 4.0], 1.0),
            ("cv", [10.0, 12.0, 8.0], 0.2),
            ("instability", [1.0, 3.0, 2.0, 4.0], 3.0),
        ],
    )
    def test_supported_aggregations(self, agg_name, values, expected):
        df = _make_df(values)
        result = df.select(_build_agg_expr(agg_name))
        assert result["value"][0] == pytest.approx(expected)

    def test_std(self):
        df = _make_df([1.0, 2.0, 3.0])
        result = df.select(_build_agg_expr("std"))
        assert result["value"][0] == pytest.approx(1.0)  # sample std (ddof=1)

    @pytest.mark.parametrize(
        ("agg_name", "expected"),
        [
            ("mean", 42.0),
            ("sum", 42.0),
            ("min", 42.0),
            ("max", 42.0),
            ("count", 1.0),
            ("median", 42.0),
            ("first", 42.0),
            ("last", 42.0),
        ],
    )
    def test_single_value(self, agg_name, expected):
        """Aggregating a single value works for the basic scalar reducers."""
        result = _make_df([42.0]).select(_build_agg_expr(agg_name))
        assert result["value"][0] == pytest.approx(expected), f"{agg_name} failed on single value"

    def test_cv_zero_mean(self):
        """CV with zero mean returns null (guarded by abs(mean) > 1e-15)."""
        df = _make_df([-1.0, 1.0])  # mean = 0
        result = df.select(_build_agg_expr("cv"))
        assert result["value"][0] is None

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
        df = pl.DataFrame(
            {
                "indicator": ["mood", "mood", "mood"],
                "value": ["true", "false", "yes"],
            }
        )
        result = _encode_non_continuous(df, {"mood": "binary"})
        values = result.sort("value")["value"].to_list()
        # "false" -> "0.0", "true" -> "1.0", "yes" -> "1.0"
        assert "0.0" in values
        assert "1.0" in values

    def test_ordinal_numeric_codes_passthrough(self):
        df = pl.DataFrame(
            {
                "indicator": ["pain", "pain", "pain"],
                "value": ["0", "1", "2"],
            }
        )
        result = _encode_non_continuous(
            df,
            {"pain": "ordinal"},
            ordinal_levels_lookup={"pain": ["low", "medium", "high"]},
        )
        vals = sorted(float(v) for v in result["value"].to_list())
        assert vals == [0.0, 1.0, 2.0]

    def test_ordinal_out_of_range_code_becomes_null(self):
        df = pl.DataFrame(
            {
                "indicator": ["pain", "pain"],
                "value": ["2", "3"],
            }
        )
        result = _encode_non_continuous(
            df,
            {"pain": "ordinal"},
            ordinal_levels_lookup={"pain": ["low", "medium", "high"]},
        ).sort("value", nulls_last=True)
        assert result["value"].to_list() == ["2.0", None]

    def test_continuous_passthrough(self):
        df = pl.DataFrame(
            {
                "indicator": ["weight", "weight"],
                "value": [70.5, 80.2],
            }
        )
        result = _encode_non_continuous(df, {"weight": "continuous"})
        assert result["value"].to_list() == [70.5, 80.2]

    def test_empty_dtype_lookup(self):
        df = pl.DataFrame({"indicator": ["x"], "value": [1.0]})
        result = _encode_non_continuous(df, {})
        assert result["value"].to_list() == [1.0]

    def test_mixed_indicators(self):
        """Only non-continuous indicators should be encoded; continuous left unchanged."""
        df = pl.DataFrame(
            {
                "indicator": ["mood", "weight"],
                "value": ["true", "70.5"],
            }
        )
        result = _encode_non_continuous(df, {"mood": "binary", "weight": "continuous"})
        mood_row = result.filter(pl.col("indicator") == "mood")
        weight_row = result.filter(pl.col("indicator") == "weight")
        assert float(mood_row["value"][0]) == 1.0
        assert weight_row["value"][0] == "70.5"


# =============================================================================
# compute_indicators
# =============================================================================


def _make_raw_df() -> pl.DataFrame:
    """Create a raw DataFrame spanning 3 days with heart_rate and steps."""
    return pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 8, 0),
                datetime(2024, 1, 1, 12, 0),
                datetime(2024, 1, 1, 18, 0),
                datetime(2024, 1, 2, 9, 0),
                datetime(2024, 1, 2, 15, 0),
                datetime(2024, 1, 3, 10, 0),
            ],
            "heart_rate": [72.0, 85.0, 68.0, 74.0, 90.0, 70.0],
            "steps": [1000, 3000, 500, 2000, 4000, 1500],
        }
    )


class TestComputeIndicators:
    def test_single_mean(self):
        """Mean of heart_rate across 3 daily ticks."""
        df = _make_raw_df()
        indicators = [
            {"name": "avg_hr", "source_columns": ["heart_rate"], "aggregation": "mean"},
        ]
        result = compute_indicators(df, indicators, "1d", "timestamp")
        assert result.columns == ["indicator", "value", "timestamp"]
        assert result["indicator"].to_list() == ["avg_hr"] * 3
        # Day 1: mean(72, 85, 68) = 75.0
        values = [float(v) for v in result["value"].to_list()]
        assert abs(values[0] - 75.0) < 0.01

    def test_sum_aggregation(self):
        """Sum of steps across daily ticks."""
        df = _make_raw_df()
        indicators = [
            {"name": "total_steps", "source_columns": ["steps"], "aggregation": "sum"},
        ]
        result = compute_indicators(df, indicators, "1d", "timestamp")
        values = [float(v) for v in result["value"].to_list()]
        # Day 1: 1000+3000+500=4500, Day 2: 2000+4000=6000, Day 3: 1500
        assert values == [4500.0, 6000.0, 1500.0]

    def test_multiple_indicators(self):
        """Two computed indicators in one call."""
        df = _make_raw_df()
        indicators = [
            {"name": "avg_hr", "source_columns": ["heart_rate"], "aggregation": "mean"},
            {"name": "total_steps", "source_columns": ["steps"], "aggregation": "sum"},
        ]
        result = compute_indicators(df, indicators, "1d", "timestamp")
        # 3 ticks * 2 indicators = 6 rows
        assert len(result) == 6
        assert set(result["indicator"].to_list()) == {"avg_hr", "total_steps"}

    def test_output_schema(self):
        """Output columns are exactly {indicator, value, timestamp} as Utf8."""
        df = _make_raw_df()
        indicators = [
            {"name": "avg_hr", "source_columns": ["heart_rate"], "aggregation": "mean"},
        ]
        result = compute_indicators(df, indicators, "1d", "timestamp")
        assert result.columns == ["indicator", "value", "timestamp"]
        assert result.schema["indicator"] == pl.Utf8
        assert result.schema["value"] == pl.Utf8
        assert result.schema["timestamp"] == pl.Utf8

    def test_empty_indicators(self):
        """Empty indicator list returns empty DataFrame with correct schema."""
        df = _make_raw_df()
        result = compute_indicators(df, [], "1d", "timestamp")
        assert result.columns == ["indicator", "value", "timestamp"]
        assert len(result) == 0

    def test_trend_aggregation(self):
        """Trend (map_groups path) computes OLS slope."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 1, 12, 0),
                    datetime(2024, 1, 1, 18, 0),
                ],
                "hr": [70.0, 75.0, 80.0],  # increasing → positive slope
            }
        )
        indicators = [{"name": "hr_trend", "source_columns": ["hr"], "aggregation": "trend"}]
        result = compute_indicators(df, indicators, "1d", "timestamp")
        assert len(result) == 1
        assert float(result["value"][0]) > 0  # positive slope

    def test_missing_source_column(self):
        """Missing source column is skipped with warning, not crash."""
        df = _make_raw_df()
        indicators = [
            {"name": "missing", "source_columns": ["nonexistent_col"], "aggregation": "mean"},
        ]
        result = compute_indicators(df, indicators, "1d", "timestamp")
        assert len(result) == 0

    def test_null_source_values(self):
        """Source column with nulls aggregates correctly."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 1, 12, 0),
                    datetime(2024, 1, 1, 18, 0),
                ],
                "hr": [72.0, None, 68.0],
            }
        )
        indicators = [{"name": "avg_hr", "source_columns": ["hr"], "aggregation": "mean"}]
        result = compute_indicators(df, indicators, "1d", "timestamp")
        assert len(result) == 1
        # mean(72, 68) = 70.0 (null ignored)
        assert abs(float(result["value"][0]) - 70.0) < 0.01

    def test_first_ignores_leading_nulls(self):
        """Point aggregations should use the first observed value, not the first row."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 1, 9, 0),
                    datetime(2024, 1, 1, 10, 0),
                ],
                "care_setting": [None, "home", "clinic"],
            }
        )
        indicators = [
            {
                "name": "first_setting",
                "source_columns": ["care_setting"],
                "measurement_dtype": "categorical",
                "aggregation": "first",
            }
        ]

        result = compute_indicators(df, indicators, "1d", "timestamp")

        assert result["value"].to_list() == ["home"]

    def test_categorical_last_preserves_string_value(self):
        """Direct categorical computed indicators should preserve raw labels."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 1, 12, 0),
                ],
                "care_setting": ["home", "clinic"],
            }
        )
        indicators = [
            {
                "name": "last_setting",
                "source_columns": ["care_setting"],
                "measurement_dtype": "categorical",
                "aggregation": "last",
            }
        ]

        result = compute_indicators(df, indicators, "1d", "timestamp")

        assert result["value"].to_list() == ["clinic"]

    def test_ordinal_last_encodes_label_to_numeric_code(self):
        """Direct ordinal computed indicators should emit canonical integer codes."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 1, 12, 0),
                ],
                "mood_label": ["bad", "good"],
            }
        )
        indicators = [
            {
                "name": "closing_mood",
                "source_columns": ["mood_label"],
                "measurement_dtype": "ordinal",
                "aggregation": "last",
                "ordinal_levels": ["bad", "ok", "good"],
            }
        ]

        result = compute_indicators(df, indicators, "1d", "timestamp")

        assert result["value"].to_list() == ["2"]

    def test_count_aggregation_counts_non_null_string_values(self):
        """Count aggregations should not null out string source columns before counting."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 1, 9, 0),
                    datetime(2024, 1, 1, 10, 0),
                ],
                "message_text": ["alpha", None, "beta"],
            }
        )
        indicators = [
            {
                "name": "text_events",
                "source_columns": ["message_text"],
                "measurement_dtype": "count",
                "aggregation": "count",
            }
        ]

        result = compute_indicators(df, indicators, "1d", "timestamp")

        assert result["value"].to_list() == ["2"]

    def test_computed_rule_multi_column_formula(self):
        """Computed rules can deterministically derive window values from multiple columns."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 1, 12, 0),
                    datetime(2024, 1, 2, 9, 0),
                ],
                "systolic_bp": [120.0, 150.0, 110.0],
                "diastolic_bp": [80.0, 90.0, 70.0],
            }
        )
        indicators = [
            {
                "name": "map",
                "source_columns": ["systolic_bp", "diastolic_bp"],
                "measurement_dtype": "continuous",
                "aggregation": "mean",
                "computed_rule": {
                    "window_expr": "mean(diastolic_bp + (systolic_bp - diastolic_bp) / 3)"
                },
            }
        ]

        result = compute_indicators(df, indicators, "1d", "timestamp")

        values = [float(value) for value in result["value"].to_list()]
        assert values[0] == pytest.approx((80 + (120 - 80) / 3 + 90 + (150 - 90) / 3) / 2)
        assert values[1] == pytest.approx(70 + (110 - 70) / 3)

    def test_computed_rule_filtered_count_preserves_zero_vs_null(self):
        """Filtered deterministic counts should distinguish observed zero from no observation."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 1, 12, 0),
                    datetime(2024, 1, 1, 14, 0),
                    datetime(2024, 1, 2, 9, 0),
                    datetime(2024, 1, 3, 10, 0),
                ],
                "event_type": ["med_admin", "med_admin", "note", "med_admin", "note"],
                "admin_status": ["missed", "taken", None, "taken", None],
            }
        )
        indicators = [
            {
                "name": "missed_doses",
                "source_columns": ["event_type", "admin_status"],
                "measurement_dtype": "count",
                "aggregation": "sum",
                "computed_rule": {
                    "window_expr": 'None if count_true(event_type == "med_admin") == 0 else sum(1 if (event_type == "med_admin" and admin_status == "missed") else 0)'
                },
            }
        ]

        result = compute_indicators(df, indicators, "1d", "timestamp")

        assert result["timestamp"].to_list() == [
            "2024-01-01T00:00:00",
            "2024-01-02T00:00:00",
            "2024-01-03T00:00:00",
        ]
        assert result["value"].to_list() == ["1", "0", None]

    def test_computed_rule_binary_flag_preserves_zero_vs_null(self):
        """Binary deterministic window flags should keep observed negative distinct from missing."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 1, 12, 0),
                    datetime(2024, 1, 2, 8, 0),
                    datetime(2024, 1, 2, 12, 0),
                    datetime(2024, 1, 3, 8, 0),
                    datetime(2024, 1, 3, 12, 0),
                ],
                "spo2_pct": [95.0, 94.0, 91.0, 95.0, None, None],
            }
        )
        indicators = [
            {
                "name": "low_spo2",
                "source_columns": ["spo2_pct"],
                "measurement_dtype": "binary",
                "aggregation": "last",
                "computed_rule": {
                    "window_expr": "1 if any(spo2_pct < 92) else (0 if count_non_null(spo2_pct) > 0 else None)"
                },
            }
        ]

        result = compute_indicators(df, indicators, "1d", "timestamp")

        assert result["value"].to_list() == ["0", "1", None]

    def test_computed_rule_contains_any_literal_list(self):
        """contains_any() should accept literal string lists in computed rules."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 1, 12, 0),
                    datetime(2024, 1, 2, 9, 0),
                ],
                "title_url": [
                    "https://facebook.com/some-post",
                    "https://example.com",
                    "https://reddit.com/r/polars",
                ],
            }
        )
        indicators = [
            {
                "name": "social_media_hits",
                "source_columns": ["title_url"],
                "measurement_dtype": "count",
                "aggregation": "count",
                "computed_rule": {
                    "window_expr": 'count_true(contains_any(title_url, ["facebook.com", "reddit.com"]))'
                },
            }
        ]

        result = compute_indicators(df, indicators, "1d", "timestamp")

        assert result["timestamp"].to_list() == ["2024-01-01T00:00:00", "2024-01-02T00:00:00"]
        assert result["value"].to_list() == ["1", "1"]

    def test_computed_rule_nested_contains_any_with_if_else(self):
        """Nested computed rules should handle contains_any() list literals inside bool expressions."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 1, 12, 0),
                    datetime(2024, 1, 2, 9, 0),
                    datetime(2024, 1, 2, 14, 0),
                    datetime(2024, 1, 3, 10, 0),
                ],
                "title": [
                    "Stress management",
                    None,
                    "ordinary browsing",
                    None,
                    "nothing relevant",
                ],
                "title_url": [
                    None,
                    "https://example.com",
                    "https://burnout.example/article",
                    "https://example.com/other",
                    None,
                ],
            }
        )
        indicators = [
            {
                "name": "stress_content_count",
                "source_columns": ["timestamp", "title", "title_url"],
                "measurement_dtype": "count",
                "aggregation": "count",
                "computed_rule": {
                    "window_expr": 'None if count_non_null(timestamp) == 0 else count_true(contains_any(lower(coalesce(title, "")), ["stress", "burnout"]) or contains_any(lower(coalesce(title_url, "")), ["stress", "burnout"]))'
                },
            }
        ]

        result = compute_indicators(df, indicators, "1d", "timestamp")

        assert result["timestamp"].to_list() == [
            "2024-01-01T00:00:00",
            "2024-01-02T00:00:00",
            "2024-01-03T00:00:00",
        ]
        assert result["value"].to_list() == ["1", "1", "0"]

    def test_timestamp_format_matches_bucket_by_clock(self):
        """Computed timestamps match the ISO format from bucket_by_clock."""
        df = _make_raw_df()
        indicators = [
            {"name": "avg_hr", "source_columns": ["heart_rate"], "aggregation": "mean"},
        ]
        result = compute_indicators(df, indicators, "1d", "timestamp")
        # Should be ISO format: YYYY-MM-DDTHH:MM:SS
        assert result["timestamp"][0] == "2024-01-01T00:00:00"

    def test_timezone_aware_string_timestamps(self):
        """UTC-suffixed string timestamps should aggregate without parse errors."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    "2025-03-03T08:00:00Z",
                    "2025-03-03T12:00:00Z",
                    "2025-03-04T09:00:00Z",
                ],
                "heart_rate": [72.0, 84.0, 90.0],
            }
        )
        indicators = [
            {"name": "avg_hr", "source_columns": ["heart_rate"], "aggregation": "mean"},
        ]

        result = compute_indicators(df, indicators, "1d", "timestamp")

        assert result["timestamp"].to_list() == ["2025-03-03T00:00:00", "2025-03-04T00:00:00"]
        assert [float(v) for v in result["value"].to_list()] == [78.0, 90.0]

    def test_hourly_clock(self):
        """Hourly model_clock produces hourly ticks."""
        df = _make_raw_df()
        indicators = [
            {"name": "avg_hr", "source_columns": ["heart_rate"], "aggregation": "mean"},
        ]
        result = compute_indicators(df, indicators, "1h", "timestamp")
        # 6 events at 6 different hours → 6 ticks (each with 1 event)
        assert len(result) == 6

    def test_indicator_specific_observation_window_overrides_model_clock(self):
        """Computed indicators bucket by their own support window, not the global clock."""
        df = _make_raw_df()
        indicators = [
            {
                "name": "weekly_steps",
                "source_columns": ["steps"],
                "aggregation": "sum",
                "observation_window": "1w",
            },
        ]

        result = compute_indicators(df, indicators, "1d", "timestamp")

        assert len(result) == 1
        assert result["timestamp"].to_list() == ["2024-01-01T00:00:00"]
        assert float(result["value"][0]) == 12000.0

    def test_col_name_parameter(self):
        """_build_agg_expr with custom col_name works correctly."""
        df = pl.DataFrame({"heart_rate": [72.0, 85.0, 68.0]})
        expr = _build_agg_expr("mean", "heart_rate")
        result = df.select(expr)
        assert abs(result["value"][0] - 75.0) < 0.01
