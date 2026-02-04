"""Tests for Stage 3: Transform + Validate.

This module tests:
1. validate_extraction() - semantic checks (variance, sample size)
2. Integration test - full Stage 3 pipeline with mock Stage 2 output
"""

from dataclasses import dataclass

import polars as pl
import pytest

from dsem_agent.flows.stages.stage3_validation import (
    MIN_OBSERVATIONS,
    _get_indicator_granularity,
    aggregate_measurements,
    validate_extraction,
)
from dsem_agent.utils.aggregations import aggregate_worker_measurements

# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def simple_dsem_model():
    """Simple DSEM model with daily granularity constructs."""
    return {
        "latent": {
            "constructs": [
                {"name": "stress", "causal_granularity": "daily"},
                {"name": "sleep", "causal_granularity": "daily"},
            ],
            "edges": [{"cause": "stress", "effect": "sleep"}],
        },
        "measurement": {
            "indicators": [
                {
                    "name": "stress_score",
                    "construct": "stress",
                    "how_to_measure": "Extract stress level",
                    "aggregation": "mean",
                },
                {
                    "name": "sleep_hours",
                    "construct": "sleep",
                    "how_to_measure": "Extract sleep duration",
                    "aggregation": "mean",
                },
            ],
        },
    }


@pytest.fixture
def multi_granularity_model():
    """DSEM model with multiple granularities."""
    return {
        "latent": {
            "constructs": [
                {"name": "activity", "causal_granularity": "hourly"},
                {"name": "mood", "causal_granularity": "daily"},
                {"name": "health", "causal_granularity": "weekly"},
                {"name": "demographics", "causal_granularity": None},  # time-invariant
            ],
            "edges": [],
        },
        "measurement": {
            "indicators": [
                {"name": "steps", "construct": "activity", "aggregation": "sum"},
                {"name": "heart_rate", "construct": "activity", "aggregation": "mean"},
                {"name": "mood_score", "construct": "mood", "aggregation": "mean"},
                {"name": "weight", "construct": "health", "aggregation": "last"},
                {"name": "age", "construct": "demographics", "aggregation": "first"},
            ],
        },
    }


@dataclass
class MockWorkerResult:
    """Mock WorkerResult for testing aggregate_measurements."""

    dataframe: pl.DataFrame


# ==============================================================================
# UNIT TESTS: _get_indicator_granularity
# ==============================================================================


class TestGetIndicatorGranularity:
    """Test helper function for getting indicator granularity."""

    def test_finds_granularity_with_construct_key(self):
        """Find granularity when indicator uses 'construct' key."""
        indicator = {"name": "test", "construct": "stress"}
        constructs = [
            {"name": "stress", "causal_granularity": "daily"},
            {"name": "sleep", "causal_granularity": "weekly"},
        ]
        assert _get_indicator_granularity(indicator, constructs) == "daily"

    def test_finds_granularity_with_construct_name_key(self):
        """Find granularity when indicator uses 'construct_name' key."""
        indicator = {"name": "test", "construct_name": "sleep"}
        constructs = [
            {"name": "stress", "causal_granularity": "daily"},
            {"name": "sleep", "causal_granularity": "weekly"},
        ]
        assert _get_indicator_granularity(indicator, constructs) == "weekly"

    def test_returns_none_for_time_invariant(self):
        """Return None for time-invariant constructs."""
        indicator = {"name": "age", "construct": "demographics"}
        constructs = [{"name": "demographics", "causal_granularity": None}]
        assert _get_indicator_granularity(indicator, constructs) is None

    def test_returns_none_for_unknown_construct(self):
        """Return None when construct not found."""
        indicator = {"name": "test", "construct": "unknown"}
        constructs = [{"name": "stress", "causal_granularity": "daily"}]
        assert _get_indicator_granularity(indicator, constructs) is None


# ==============================================================================
# UNIT TESTS: validate_extraction
# ==============================================================================


class TestValidateExtraction:
    """Test validate_extraction semantic checks."""

    # --------------------------------------------------------------------------
    # Basic cases
    # --------------------------------------------------------------------------

    def test_empty_extracted_data(self, simple_dsem_model):
        """Empty extracted_data returns no issues."""
        result = validate_extraction.fn(simple_dsem_model, {})
        assert result["is_valid"] is True
        assert result["issues"] == []

    def test_valid_data_no_issues(self, simple_dsem_model):
        """Valid data with sufficient variance and sample size passes."""
        # 14+ days of daily data with variance
        dates = [f"2024-01-{d:02d}" for d in range(1, 20)]
        extracted = {
            "daily": pl.DataFrame(
                {
                    "time_bucket": pl.Series(dates).str.to_datetime(),
                    "stress_score": [float(i % 5 + 1) for i in range(19)],  # 1-5 varying
                    "sleep_hours": [6.0 + (i % 3) for i in range(19)],  # 6-8 varying
                }
            )
        }
        result = validate_extraction.fn(simple_dsem_model, extracted)
        assert result["is_valid"] is True
        assert result["issues"] == []

    def test_missing_indicator_in_dataframe(self, simple_dsem_model):
        """Missing indicator column is not flagged (structural issue)."""
        extracted = {
            "daily": pl.DataFrame(
                {
                    "time_bucket": pl.Series(["2024-01-01"] * 20).str.to_datetime(),
                    "stress_score": [float(i) for i in range(20)],
                    # sleep_hours is missing
                }
            )
        }
        result = validate_extraction.fn(simple_dsem_model, extracted)
        # Only issues for stress_score if any, not for missing sleep_hours
        assert all(i["indicator"] != "sleep_hours" for i in result["issues"])

    def test_all_nulls_no_issue(self, simple_dsem_model):
        """Indicator with all nulls is skipped (no data to check)."""
        extracted = {
            "daily": pl.DataFrame(
                {
                    "time_bucket": pl.Series(["2024-01-01"] * 5).str.to_datetime(),
                    "stress_score": [None] * 5,
                    "sleep_hours": [7.0, 7.5, 8.0, 7.0, 6.5],
                }
            )
        }
        result = validate_extraction.fn(simple_dsem_model, extracted)
        # stress_score should not generate issue since all nulls
        stress_issues = [i for i in result["issues"] if i["indicator"] == "stress_score"]
        assert len(stress_issues) == 0

    # --------------------------------------------------------------------------
    # Variance checks (severity: error)
    # --------------------------------------------------------------------------

    def test_zero_variance_is_error(self, simple_dsem_model):
        """Constant values (zero variance) should return error."""
        extracted = {
            "daily": pl.DataFrame(
                {
                    "time_bucket": pl.Series(
                        [f"2024-01-{d:02d}" for d in range(1, 20)]
                    ).str.to_datetime(),
                    "stress_score": [5.0] * 19,  # Constant!
                    "sleep_hours": [6.0 + (i % 3) for i in range(19)],
                }
            )
        }
        result = validate_extraction.fn(simple_dsem_model, extracted)

        assert result["is_valid"] is False

        error_issues = [i for i in result["issues"] if i["severity"] == "error"]
        assert len(error_issues) == 1
        assert error_issues[0]["indicator"] == "stress_score"
        assert error_issues[0]["issue_type"] == "no_variance"
        assert "constant value = 5" in error_issues[0]["message"]

    def test_nonzero_variance_no_error(self, simple_dsem_model):
        """Non-constant values should not generate variance error."""
        extracted = {
            "daily": pl.DataFrame(
                {
                    "time_bucket": pl.Series(
                        [f"2024-01-{d:02d}" for d in range(1, 20)]
                    ).str.to_datetime(),
                    "stress_score": [float(i) for i in range(19)],  # 0-18
                    "sleep_hours": [7.0, 8.0] * 9 + [7.5],  # Varying
                }
            )
        }
        result = validate_extraction.fn(simple_dsem_model, extracted)

        variance_errors = [i for i in result["issues"] if i["issue_type"] == "no_variance"]
        assert len(variance_errors) == 0

    def test_single_value_variance_edge_case(self, simple_dsem_model):
        """Single non-null value may have undefined/zero variance."""
        extracted = {
            "daily": pl.DataFrame(
                {
                    "time_bucket": pl.Series(["2024-01-01"]).str.to_datetime(),
                    "stress_score": [5.0],
                    "sleep_hours": [7.0],
                }
            )
        }
        result = validate_extraction.fn(simple_dsem_model, extracted)
        # Single value - variance is undefined/zero, but also low_n
        # Implementation may or may not flag this as no_variance
        # At minimum, low_n should be flagged
        assert any(i["issue_type"] == "low_n" for i in result["issues"])

    def test_multiple_zero_variance_indicators(self, simple_dsem_model):
        """Multiple constant indicators all flagged as errors."""
        extracted = {
            "daily": pl.DataFrame(
                {
                    "time_bucket": pl.Series(
                        [f"2024-01-{d:02d}" for d in range(1, 20)]
                    ).str.to_datetime(),
                    "stress_score": [3.0] * 19,  # Constant
                    "sleep_hours": [7.0] * 19,  # Constant
                }
            )
        }
        result = validate_extraction.fn(simple_dsem_model, extracted)

        assert result["is_valid"] is False

        variance_errors = [i for i in result["issues"] if i["issue_type"] == "no_variance"]
        assert len(variance_errors) == 2
        indicators_with_errors = {i["indicator"] for i in variance_errors}
        assert indicators_with_errors == {"stress_score", "sleep_hours"}

    # --------------------------------------------------------------------------
    # Sample size checks (severity: warning)
    # --------------------------------------------------------------------------

    def test_low_n_daily_is_warning(self, simple_dsem_model):
        """Fewer than 14 daily observations should return warning."""
        # Only 10 days (less than MIN_OBSERVATIONS["daily"] = 14)
        extracted = {
            "daily": pl.DataFrame(
                {
                    "time_bucket": pl.Series(
                        [f"2024-01-{d:02d}" for d in range(1, 11)]
                    ).str.to_datetime(),
                    "stress_score": [float(i) for i in range(10)],
                    "sleep_hours": [7.0 + i * 0.1 for i in range(10)],
                }
            )
        }
        result = validate_extraction.fn(simple_dsem_model, extracted)

        # Should be valid (only warnings, no errors)
        assert result["is_valid"] is True

        low_n_warnings = [i for i in result["issues"] if i["issue_type"] == "low_n"]
        assert len(low_n_warnings) == 2  # Both indicators
        assert all(i["severity"] == "warning" for i in low_n_warnings)

    def test_sufficient_n_no_warning(self, simple_dsem_model):
        """Sufficient observations should not generate warning."""
        # 14+ days
        extracted = {
            "daily": pl.DataFrame(
                {
                    "time_bucket": pl.Series(
                        [f"2024-01-{d:02d}" for d in range(1, 20)]
                    ).str.to_datetime(),
                    "stress_score": [float(i) for i in range(19)],
                    "sleep_hours": [7.0 + i * 0.1 for i in range(19)],
                }
            )
        }
        result = validate_extraction.fn(simple_dsem_model, extracted)

        low_n_warnings = [i for i in result["issues"] if i["issue_type"] == "low_n"]
        assert len(low_n_warnings) == 0

    @pytest.mark.parametrize(
        "granularity,min_n",
        [
            ("hourly", 48),
            ("daily", 14),
            ("weekly", 8),
            ("monthly", 6),
            ("yearly", 3),
            ("time_invariant", 1),
        ],
    )
    def test_sample_size_thresholds(self, granularity, min_n):
        """Test each granularity has correct minimum threshold."""
        assert MIN_OBSERVATIONS[granularity] == min_n

    def test_hourly_low_n_threshold(self, multi_granularity_model):
        """Hourly needs 48+ observations."""
        # Only 30 hourly observations
        extracted = {
            "hourly": pl.DataFrame(
                {
                    "time_bucket": pl.Series(
                        [f"2024-01-01T{h:02d}:00" for h in range(24)]
                        + [f"2024-01-02T{h:02d}:00" for h in range(6)]
                    ).str.to_datetime(),
                    "steps": [float(1000 + i * 10) for i in range(30)],
                    "heart_rate": [float(70 + i % 10) for i in range(30)],
                }
            ),
            "daily": pl.DataFrame(
                {
                    "time_bucket": pl.Series(
                        [f"2024-01-{d:02d}" for d in range(1, 20)]
                    ).str.to_datetime(),
                    "mood_score": [float(i % 5) for i in range(19)],
                }
            ),
            "weekly": pl.DataFrame(
                {
                    "time_bucket": pl.Series(
                        [f"2024-{w:02d}-01" for w in range(1, 12)]
                    ).str.to_datetime(),
                    "weight": [float(70 + i * 0.1) for i in range(11)],
                }
            ),
            "time_invariant": pl.DataFrame(
                {
                    "age": [35.0],
                }
            ),
        }
        result = validate_extraction.fn(multi_granularity_model, extracted)

        hourly_warnings = [
            i
            for i in result["issues"]
            if i["issue_type"] == "low_n" and i["indicator"] in ("steps", "heart_rate")
        ]
        assert len(hourly_warnings) == 2

    # --------------------------------------------------------------------------
    # Combined scenarios
    # --------------------------------------------------------------------------

    def test_error_and_warning_together(self, simple_dsem_model):
        """Indicator can have both variance error and sample size warning."""
        # 5 constant values
        extracted = {
            "daily": pl.DataFrame(
                {
                    "time_bucket": pl.Series(
                        [f"2024-01-{d:02d}" for d in range(1, 6)]
                    ).str.to_datetime(),
                    "stress_score": [5.0] * 5,  # Constant AND low N
                    "sleep_hours": [7.0 + i * 0.1 for i in range(5)],  # Only low N
                }
            )
        }
        result = validate_extraction.fn(simple_dsem_model, extracted)

        assert result["is_valid"] is False  # Has error

        # stress_score should have both issues
        stress_issues = [i for i in result["issues"] if i["indicator"] == "stress_score"]
        issue_types = {i["issue_type"] for i in stress_issues}
        assert "no_variance" in issue_types
        assert "low_n" in issue_types

    def test_only_warnings_is_valid(self, simple_dsem_model):
        """is_valid=True when only warnings exist (no errors)."""
        extracted = {
            "daily": pl.DataFrame(
                {
                    "time_bucket": pl.Series(
                        [f"2024-01-{d:02d}" for d in range(1, 6)]
                    ).str.to_datetime(),
                    "stress_score": [float(i) for i in range(5)],  # Varying but low N
                    "sleep_hours": [7.0 + i * 0.1 for i in range(5)],  # Varying but low N
                }
            )
        }
        result = validate_extraction.fn(simple_dsem_model, extracted)

        assert result["is_valid"] is True
        assert len(result["issues"]) > 0
        assert all(i["severity"] == "warning" for i in result["issues"])

    def test_mixed_issues_across_indicators(self, multi_granularity_model):
        """Different indicators can have different issues."""
        extracted = {
            "hourly": pl.DataFrame(
                {
                    "time_bucket": pl.Series(
                        [f"2024-01-01T{h:02d}:00" for h in range(24)]
                        + [f"2024-01-02T{h:02d}:00" for h in range(24)]
                    ).str.to_datetime(),
                    "steps": [1000.0] * 48,  # Constant - error
                    "heart_rate": [float(70 + i % 20) for i in range(48)],  # Good
                }
            ),
            "daily": pl.DataFrame(
                {
                    "time_bucket": pl.Series(
                        [f"2024-01-{d:02d}" for d in range(1, 5)]
                    ).str.to_datetime(),
                    "mood_score": [float(i) for i in range(4)],  # Low N - warning
                }
            ),
            "weekly": pl.DataFrame(
                {
                    "time_bucket": pl.Series(
                        [f"2024-{w:02d}-01" for w in range(1, 12)]
                    ).str.to_datetime(),
                    "weight": [float(70 + i * 0.1) for i in range(11)],  # Good
                }
            ),
            "time_invariant": pl.DataFrame(
                {
                    "age": [35.0],  # Good
                }
            ),
        }
        result = validate_extraction.fn(multi_granularity_model, extracted)

        assert result["is_valid"] is False  # steps has error

        # Check specific issues
        steps_issues = [i for i in result["issues"] if i["indicator"] == "steps"]
        assert any(i["issue_type"] == "no_variance" for i in steps_issues)

        mood_issues = [i for i in result["issues"] if i["indicator"] == "mood_score"]
        assert any(i["issue_type"] == "low_n" for i in mood_issues)

        # heart_rate and weight should have no issues
        hr_issues = [i for i in result["issues"] if i["indicator"] == "heart_rate"]
        assert len(hr_issues) == 0
        weight_issues = [i for i in result["issues"] if i["indicator"] == "weight"]
        assert len(weight_issues) == 0


# ==============================================================================
# INTEGRATION TEST: Full Stage 3 Pipeline
# ==============================================================================


class TestStage3Integration:
    """Integration tests for the full Stage 3 pipeline.

    Tests the flow: mock Stage 2 output -> aggregate_measurements -> validate_extraction
    """

    @pytest.fixture
    def realistic_dsem_model(self):
        """Realistic DSEM model for integration testing."""
        return {
            "latent": {
                "constructs": [
                    {
                        "name": "physical_activity",
                        "description": "Daily physical activity level",
                        "role": "exogenous",
                        "causal_granularity": "daily",
                    },
                    {
                        "name": "sleep_quality",
                        "description": "Quality of sleep",
                        "role": "endogenous",
                        "causal_granularity": "daily",
                    },
                    {
                        "name": "stress",
                        "description": "Stress level",
                        "role": "endogenous",
                        "is_outcome": True,
                        "causal_granularity": "daily",
                    },
                    {
                        "name": "demographics",
                        "description": "Demographic factors",
                        "role": "exogenous",
                        "causal_granularity": None,  # time-invariant
                    },
                ],
                "edges": [
                    {"cause": "physical_activity", "effect": "sleep_quality"},
                    {"cause": "sleep_quality", "effect": "stress"},
                    {"cause": "physical_activity", "effect": "stress"},
                ],
            },
            "measurement": {
                "indicators": [
                    {
                        "name": "step_count",
                        "construct": "physical_activity",
                        "how_to_measure": "Extract daily step count",
                        "measurement_granularity": "daily",
                        "measurement_dtype": "continuous",
                        "aggregation": "sum",
                    },
                    {
                        "name": "active_minutes",
                        "construct": "physical_activity",
                        "how_to_measure": "Extract active minutes",
                        "measurement_granularity": "daily",
                        "measurement_dtype": "continuous",
                        "aggregation": "sum",
                    },
                    {
                        "name": "sleep_duration",
                        "construct": "sleep_quality",
                        "how_to_measure": "Extract sleep duration in hours",
                        "measurement_granularity": "daily",
                        "measurement_dtype": "continuous",
                        "aggregation": "mean",
                    },
                    {
                        "name": "sleep_score",
                        "construct": "sleep_quality",
                        "how_to_measure": "Extract sleep quality score 1-10",
                        "measurement_granularity": "daily",
                        "measurement_dtype": "continuous",
                        "aggregation": "mean",
                    },
                    {
                        "name": "stress_level",
                        "construct": "stress",
                        "how_to_measure": "Extract stress level 1-10",
                        "measurement_granularity": "daily",
                        "measurement_dtype": "continuous",
                        "aggregation": "mean",
                    },
                    {
                        "name": "age",
                        "construct": "demographics",
                        "how_to_measure": "Extract participant age",
                        "measurement_granularity": "once",
                        "measurement_dtype": "continuous",
                        "aggregation": "first",
                    },
                ],
            },
        }

    def _create_mock_worker_dataframes(self, n_days: int = 30) -> list[pl.DataFrame]:
        """Create mock worker output DataFrames simulating Stage 2 output.

        Args:
            n_days: Number of days of data to generate

        Returns:
            List of DataFrames in worker output format (indicator, value, timestamp)
        """
        import random

        random.seed(42)

        worker_dfs = []

        # Simulate 3 workers processing different chunks
        for worker_id in range(3):
            records = []

            # Each worker covers a subset of days with some overlap
            start_day = worker_id * 8 + 1
            end_day = min(start_day + 15, n_days + 1)

            for day in range(start_day, end_day):
                date_str = f"2024-01-{day:02d}"

                # Multiple readings per day (simulating raw extractions)
                for reading in range(random.randint(1, 3)):
                    hour = random.randint(6, 22)
                    timestamp = f"{date_str} {hour:02d}:{random.randint(0, 59):02d}"

                    # step_count: 3000-15000 per reading
                    records.append(
                        {
                            "indicator": "step_count",
                            "value": float(random.randint(3000, 15000)),
                            "timestamp": timestamp,
                        }
                    )

                    # active_minutes: 10-60 per reading
                    records.append(
                        {
                            "indicator": "active_minutes",
                            "value": float(random.randint(10, 60)),
                            "timestamp": timestamp,
                        }
                    )

                    # sleep_duration: 5-9 hours
                    if reading == 0:  # Only one sleep reading per day
                        records.append(
                            {
                                "indicator": "sleep_duration",
                                "value": round(random.uniform(5.0, 9.0), 1),
                                "timestamp": f"{date_str} 07:00",
                            }
                        )

                        # sleep_score: 1-10
                        records.append(
                            {
                                "indicator": "sleep_score",
                                "value": float(random.randint(4, 10)),
                                "timestamp": f"{date_str} 07:00",
                            }
                        )

                        # stress_level: 1-10
                        records.append(
                            {
                                "indicator": "stress_level",
                                "value": float(random.randint(1, 8)),
                                "timestamp": f"{date_str} 20:00",
                            }
                        )

            # Time-invariant: age appears once per worker (they may extract it multiple times)
            if worker_id == 0:
                records.append(
                    {
                        "indicator": "age",
                        "value": 35,
                        "timestamp": "2024-01-01",
                    }
                )

            df = pl.DataFrame(
                records,
                schema={"indicator": pl.Utf8, "value": pl.Object, "timestamp": pl.Utf8},
            )
            worker_dfs.append(df)

        return worker_dfs

    def test_full_pipeline_valid_data(self, realistic_dsem_model):
        """Full pipeline with sufficient valid data passes all checks."""
        # Create mock worker output
        worker_dfs = self._create_mock_worker_dataframes(n_days=30)

        # Transform: aggregate worker measurements
        aggregated = aggregate_worker_measurements(worker_dfs, realistic_dsem_model)

        # Verify aggregation structure
        assert "daily" in aggregated
        assert "time_invariant" in aggregated

        daily_df = aggregated["daily"]
        assert "time_bucket" in daily_df.columns
        assert "step_count" in daily_df.columns
        assert "active_minutes" in daily_df.columns
        assert "sleep_duration" in daily_df.columns
        assert "sleep_score" in daily_df.columns
        assert "stress_level" in daily_df.columns

        # Verify aggregation worked (sum for step_count, mean for others)
        assert daily_df.height >= 14  # At least 14 days

        ti_df = aggregated["time_invariant"]
        assert "age" in ti_df.columns
        assert ti_df.height == 1

        # Validate
        validation = validate_extraction.fn(realistic_dsem_model, aggregated)

        assert validation["is_valid"] is True
        # May have warnings but no errors
        errors = [i for i in validation["issues"] if i["severity"] == "error"]
        assert len(errors) == 0

    def test_full_pipeline_low_sample_size(self, realistic_dsem_model):
        """Pipeline with insufficient data raises warnings."""
        # Only 5 days of data
        worker_dfs = self._create_mock_worker_dataframes(n_days=5)

        aggregated = aggregate_worker_measurements(worker_dfs, realistic_dsem_model)
        validation = validate_extraction.fn(realistic_dsem_model, aggregated)

        # Should be valid (warnings only) but have low_n warnings
        assert validation["is_valid"] is True

        low_n_warnings = [i for i in validation["issues"] if i["issue_type"] == "low_n"]
        assert len(low_n_warnings) > 0
        # All time-varying indicators should have low_n warning
        warned_indicators = {i["indicator"] for i in low_n_warnings}
        assert "step_count" in warned_indicators
        assert "stress_level" in warned_indicators

    def test_full_pipeline_constant_indicator(self, realistic_dsem_model):
        """Pipeline with constant indicator values raises error."""
        # Create data with constant stress_level
        worker_dfs = []

        records = []
        for day in range(1, 20):
            date_str = f"2024-01-{day:02d}"

            records.extend(
                [
                    {
                        "indicator": "step_count",
                        "value": float(5000 + day * 100),
                        "timestamp": f"{date_str} 10:00",
                    },
                    {
                        "indicator": "active_minutes",
                        "value": float(30 + day),
                        "timestamp": f"{date_str} 10:00",
                    },
                    {
                        "indicator": "sleep_duration",
                        "value": 7.0 + (day % 3) * 0.5,
                        "timestamp": f"{date_str} 07:00",
                    },
                    {
                        "indicator": "sleep_score",
                        "value": float(5 + day % 4),
                        "timestamp": f"{date_str} 07:00",
                    },
                    {
                        "indicator": "stress_level",
                        "value": 5.0,
                        "timestamp": f"{date_str} 20:00",
                    },  # CONSTANT!
                    {"indicator": "age", "value": 35, "timestamp": "2024-01-01"},
                ]
            )

        df = pl.DataFrame(
            records,
            schema={"indicator": pl.Utf8, "value": pl.Object, "timestamp": pl.Utf8},
        )
        worker_dfs.append(df)

        aggregated = aggregate_worker_measurements(worker_dfs, realistic_dsem_model)
        validation = validate_extraction.fn(realistic_dsem_model, aggregated)

        assert validation["is_valid"] is False

        stress_errors = [
            i
            for i in validation["issues"]
            if i["indicator"] == "stress_level" and i["issue_type"] == "no_variance"
        ]
        assert len(stress_errors) == 1
        assert stress_errors[0]["severity"] == "error"

    def test_full_pipeline_with_mock_worker_results(self, realistic_dsem_model):
        """Test using MockWorkerResult to simulate aggregate_measurements task."""
        worker_dfs = self._create_mock_worker_dataframes(n_days=20)

        # Wrap in MockWorkerResult (simulating what Stage 2 returns)
        mock_results = [MockWorkerResult(dataframe=df) for df in worker_dfs]

        # Use the Prefect task directly (with .fn to bypass task wrapper)
        aggregated = aggregate_measurements.fn(mock_results, realistic_dsem_model)

        # Verify structure
        assert "daily" in aggregated
        daily_df = aggregated["daily"]
        assert daily_df.height >= 14

        # Validate
        validation = validate_extraction.fn(realistic_dsem_model, aggregated)
        assert validation["is_valid"] is True

    def test_full_pipeline_multiple_granularities(self, multi_granularity_model):
        """Test pipeline handles multiple granularities correctly."""
        import random

        random.seed(123)

        records = []

        # Hourly data (need 48+ observations)
        for hour in range(72):  # 3 days of hourly
            day = hour // 24 + 1
            h = hour % 24
            timestamp = f"2024-01-{day:02d} {h:02d}:00"
            records.append(
                {
                    "indicator": "steps",
                    "value": float(random.randint(100, 500)),
                    "timestamp": timestamp,
                }
            )
            records.append(
                {
                    "indicator": "heart_rate",
                    "value": float(random.randint(60, 100)),
                    "timestamp": timestamp,
                }
            )

        # Daily data (need 14+ observations)
        for day in range(1, 20):
            timestamp = f"2024-01-{day:02d} 20:00"
            records.append(
                {
                    "indicator": "mood_score",
                    "value": float(random.randint(1, 10)),
                    "timestamp": timestamp,
                }
            )

        # Weekly data (need 8+ observations)
        for week in range(1, 12):
            timestamp = f"2024-{week:02d}-01 12:00"
            records.append(
                {"indicator": "weight", "value": float(70 + week * 0.1), "timestamp": timestamp}
            )

        # Time-invariant
        records.append({"indicator": "age", "value": 30, "timestamp": "2024-01-01"})

        df = pl.DataFrame(
            records,
            schema={"indicator": pl.Utf8, "value": pl.Object, "timestamp": pl.Utf8},
        )

        aggregated = aggregate_worker_measurements([df], multi_granularity_model)

        # Verify all granularities present
        assert "hourly" in aggregated
        assert "daily" in aggregated
        assert "weekly" in aggregated
        assert "time_invariant" in aggregated

        validation = validate_extraction.fn(multi_granularity_model, aggregated)

        # All indicators should have sufficient data
        assert validation["is_valid"] is True
        # Hourly may have low_n warning since we have 72 obs but grouped by hour
        # Actually we have 72 hourly buckets which is > 48

    def test_full_pipeline_preserves_aggregation_functions(self, realistic_dsem_model):
        """Verify correct aggregation functions are applied per indicator."""
        records = [
            # step_count: aggregation="sum"
            {"indicator": "step_count", "value": 1000.0, "timestamp": "2024-01-01 08:00"},
            {"indicator": "step_count", "value": 2000.0, "timestamp": "2024-01-01 12:00"},
            {"indicator": "step_count", "value": 3000.0, "timestamp": "2024-01-01 18:00"},
            # sleep_duration: aggregation="mean"
            {"indicator": "sleep_duration", "value": 7.0, "timestamp": "2024-01-01 07:00"},
            {"indicator": "sleep_duration", "value": 8.0, "timestamp": "2024-01-01 07:30"},
        ]

        # Add more days to avoid low_n warnings
        for day in range(2, 20):
            records.extend(
                [
                    {
                        "indicator": "step_count",
                        "value": 5000.0,
                        "timestamp": f"2024-01-{day:02d} 12:00",
                    },
                    {
                        "indicator": "sleep_duration",
                        "value": 7.5,
                        "timestamp": f"2024-01-{day:02d} 07:00",
                    },
                    {
                        "indicator": "sleep_score",
                        "value": 7.0,
                        "timestamp": f"2024-01-{day:02d} 07:00",
                    },
                    {
                        "indicator": "stress_level",
                        "value": float(day % 5 + 1),
                        "timestamp": f"2024-01-{day:02d} 20:00",
                    },
                    {
                        "indicator": "active_minutes",
                        "value": 30.0,
                        "timestamp": f"2024-01-{day:02d} 12:00",
                    },
                ]
            )

        records.append({"indicator": "age", "value": 40, "timestamp": "2024-01-01"})

        df = pl.DataFrame(
            records,
            schema={"indicator": pl.Utf8, "value": pl.Object, "timestamp": pl.Utf8},
        )

        aggregated = aggregate_worker_measurements([df], realistic_dsem_model)
        daily_df = aggregated["daily"].sort("time_bucket")

        # Day 1: step_count should be SUM of 1000+2000+3000 = 6000
        jan1 = daily_df.filter(pl.col("time_bucket").dt.day() == 1)
        assert jan1["step_count"][0] == pytest.approx(6000.0)

        # Day 1: sleep_duration should be MEAN of 7.0 and 8.0 = 7.5
        assert jan1["sleep_duration"][0] == pytest.approx(7.5)


# ==============================================================================
# AGGREGATION EDGE CASES
# ==============================================================================


class TestAggregationEdgeCases:
    """Edge-case coverage for aggregate_worker_measurements helper."""

    def test_coerces_numeric_like_values(self):
        """String/boolean values should be coerced to numeric before aggregation."""
        model = {
            "latent": {
                "constructs": [
                    {"name": "activity", "causal_granularity": "daily"},
                ],
                "edges": [],
            },
            "measurement": {
                "indicators": [
                    {"name": "steps", "construct": "activity", "aggregation": "sum"},
                    {"name": "medication_taken", "construct": "activity", "aggregation": "mean"},
                ],
            },
        }

        df = pl.DataFrame(
            [
                {"indicator": "steps", "value": "1000", "timestamp": "2024-02-01 09:00"},
                {"indicator": "steps", "value": "2000", "timestamp": "2024-02-01 18:00"},
                {"indicator": "medication_taken", "value": True, "timestamp": "2024-02-01 08:00"},
                {"indicator": "medication_taken", "value": False, "timestamp": "2024-02-01 20:00"},
            ],
            schema={"indicator": pl.Utf8, "value": pl.Object, "timestamp": pl.Utf8},
        )

        aggregated = aggregate_worker_measurements([df], model)
        daily = aggregated["daily"]
        feb1 = daily.filter(pl.col("time_bucket").dt.day() == 1)

        assert feb1["steps"][0] == pytest.approx(3000.0)
        assert feb1["medication_taken"][0] == pytest.approx(0.5)

    def test_fallback_to_mean_for_unknown_aggregation(self):
        """Unknown aggregation names should fall back to mean."""
        model = {
            "latent": {
                "constructs": [
                    {"name": "mood", "causal_granularity": "daily"},
                ],
                "edges": [],
            },
            "measurement": {
                "indicators": [
                    {"name": "mood_score", "construct": "mood", "aggregation": "not-a-real-agg"},
                ],
            },
        }

        df = pl.DataFrame(
            [
                {"indicator": "mood_score", "value": 4.0, "timestamp": "2024-03-10 08:00"},
                {"indicator": "mood_score", "value": 6.0, "timestamp": "2024-03-10 12:00"},
                {"indicator": "mood_score", "value": 8.0, "timestamp": "2024-03-10 20:00"},
            ],
            schema={"indicator": pl.Utf8, "value": pl.Float64, "timestamp": pl.Utf8},
        )

        aggregated = aggregate_worker_measurements([df], model)
        daily = aggregated["daily"]
        mar10 = daily.filter(pl.col("time_bucket").dt.day() == 10)

        # Fallback mean of (4+6+8)/3 = 6
        assert mar10["mood_score"][0] == pytest.approx(6.0)
