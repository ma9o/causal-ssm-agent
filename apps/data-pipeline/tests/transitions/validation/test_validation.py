"""Tests for validation: Validate extracted data.

This module tests:
1. validate_extraction() - semantic checks (variance, sample size)
2. Validation checks - timestamps, dtype, coverage, gaps, hallucination, correlations
"""

from datetime import datetime
from typing import Any

import polars as pl
import pytest

from nof1_causal_lab.flows.transitions.validation.flow import (
    derive_validation_status,
    validate_extraction,
)


@pytest.fixture
def simple_causal_design():
    """Simple CausalDesign with daily granularity constructs."""
    return {
        "latent": {
            "constructs": [
                {"name": "stress"},
                {"name": "sleep"},
            ],
            "edges": [{"cause": "stress", "effect": "sleep"}],
        },
        "measurement": {
            "model_clock": "1d",
            "indicators": [
                {
                    "name": "stress_score",
                    "construct_name": "stress",
                    "how_to_measure": "Extract stress level",
                },
                {
                    "name": "sleep_hours",
                    "construct_name": "sleep",
                    "how_to_measure": "Extract sleep duration",
                },
            ],
        },
    }


def _create_worker_dfs(records: list[dict[str, Any]]) -> list[pl.DataFrame]:
    """Create DataFrames for validate_extraction from records."""
    df = pl.DataFrame(
        records,
        schema={"indicator": pl.Utf8, "value": pl.Utf8, "anchor_time": pl.Utf8},
    )
    return [df]


def _all_issues(result: dict[str, Any]) -> list[dict[str, Any]]:
    indicator_issues = [
        issue
        for audit in result.get("indicators", {}).values()
        for issue in audit.get("validation", {}).get("issues", [])
    ]
    return [*indicator_issues, *result.get("dataset_issues", [])]


def _issues_for_indicator(
    result: dict[str, Any],
    indicator: str,
) -> list[dict[str, Any]]:
    return result["indicators"][indicator]["validation"]["issues"]


def _make_spec(
    indicator_name="stress_score",
    construct_name="stress",
    dtype="continuous",
    model_clock="1d",
    temporal_status="time_varying",
    extra_indicators=None,
):
    """Create a minimal causal design for testing individual checks."""
    indicators = [
        {
            "name": indicator_name,
            "construct_name": construct_name,
            "measurement_dtype": dtype,
            "how_to_measure": f"Extract {indicator_name}",
        },
    ]
    if extra_indicators:
        indicators.extend(extra_indicators)

    constructs = [
        {
            "name": construct_name,
            "temporal_status": temporal_status,
        },
    ]
    # Add constructs for extra indicators if they reference different constructs
    if extra_indicators:
        seen = {construct_name}
        for ind in extra_indicators:
            cn = ind.get("construct_name", "")
            if cn and cn not in seen:
                seen.add(cn)
                constructs.append(
                    {
                        "name": cn,
                        "temporal_status": temporal_status,
                    }
                )

    measurement: dict[str, Any] = {"indicators": indicators}
    if model_clock is not None:
        measurement["model_clock"] = model_clock

    return {
        "latent": {"constructs": constructs},
        "measurement": measurement,
    }


# ==============================================================================
# UNIT TESTS: validate_extraction
# ==============================================================================


class TestValidateExtraction:
    """Test validate_extraction semantic checks."""

    def test_derive_validation_status_maps_issue_severity_to_validity(self):
        """Stage-level status should reduce directly from local issue severities."""
        assert derive_validation_status([]) == {
            "is_valid": True,
            "has_warnings": False,
        }
        assert derive_validation_status(
            [
                {
                    "indicator": "stress_score",
                    "issue_type": "low_n",
                    "severity": "warning",
                    "message": "Only 3 observations",
                }
            ]
        ) == {
            "is_valid": True,
            "has_warnings": True,
        }
        assert derive_validation_status(
            [
                {
                    "indicator": "stress_score",
                    "issue_type": "no_numeric",
                    "severity": "error",
                    "message": "No numeric values extracted",
                }
            ]
        ) == {
            "is_valid": False,
            "has_warnings": False,
        }

    def test_empty_results_returns_error(self, simple_causal_design):
        """Empty worker results returns error."""
        result = validate_extraction(simple_causal_design, [])
        assert result["is_valid"] is False
        assert any(i["issue_type"] == "no_data" for i in _all_issues(result))

    def test_valid_data_no_issues(self, simple_causal_design):
        """Valid data with sufficient variance and sample size passes."""
        records = []
        for i in range(20):
            records.append(
                {
                    "indicator": "stress_score",
                    "value": str(float(i % 5 + 1)),  # 1-5 varying
                    "anchor_time": f"2024-01-{i + 1:02d} 10:00",
                }
            )
            records.append(
                {
                    "indicator": "sleep_hours",
                    "value": str(6.0 + (i % 3)),  # 6-8 varying
                    "anchor_time": f"2024-01-{i + 1:02d} 08:00",
                }
            )

        worker_results = _create_worker_dfs(records)
        result = validate_extraction(simple_causal_design, worker_results)

        assert result["is_valid"] is True
        # May have warnings but no errors
        errors = [i for i in _all_issues(result) if i["severity"] == "error"]
        assert len(errors) == 0

    def test_missing_indicator_is_warning(self, simple_causal_design):
        """Missing indicator generates warning."""
        records = [
            {"indicator": "stress_score", "value": "5.0", "anchor_time": "2024-01-01 10:00"},
            # sleep_hours is missing
        ]
        worker_results = _create_worker_dfs(records)
        result = validate_extraction(simple_causal_design, worker_results)

        # Should have warning for missing sleep_hours
        missing_issues = [i for i in _all_issues(result) if i["issue_type"] == "missing"]
        assert any(i["indicator"] == "sleep_hours" for i in missing_issues)

    def test_zero_variance_is_error(self, simple_causal_design):
        """Constant values (zero variance) returns error."""
        records = []
        for i in range(20):
            records.append(
                {
                    "indicator": "stress_score",
                    "value": "5.0",  # Constant!
                    "anchor_time": f"2024-01-{i + 1:02d} 10:00",
                }
            )
            records.append(
                {
                    "indicator": "sleep_hours",
                    "value": str(6.0 + (i % 3)),  # Varying
                    "anchor_time": f"2024-01-{i + 1:02d} 08:00",
                }
            )

        worker_results = _create_worker_dfs(records)
        result = validate_extraction(simple_causal_design, worker_results)

        assert result["is_valid"] is False

        error_issues = [i for i in _all_issues(result) if i["severity"] == "error"]
        assert len(error_issues) == 1
        assert error_issues[0]["indicator"] == "stress_score"
        assert error_issues[0]["issue_type"] == "no_variance"

    def test_time_invariant_skips_variance(self):
        """Time-invariant indicators are constant by definition; no_variance should not fire."""
        spec = _make_spec(model_clock=None, temporal_status="time_invariant")
        records = [
            {
                "indicator": "stress_score",
                "value": "1.0",
                "anchor_time": f"2024-01-{i + 1:02d} 10:00",
            }
            for i in range(20)
        ]
        result = validate_extraction(spec, _create_worker_dfs(records))
        no_var = [i for i in _all_issues(result) if i["issue_type"] == "no_variance"]
        assert len(no_var) == 0
        assert result["is_valid"] is True

    def test_low_sample_size_is_warning(self, simple_causal_design):
        """Low sample size generates warning."""
        records = [
            {"indicator": "stress_score", "value": "3.0", "anchor_time": "2024-01-01 10:00"},
            {"indicator": "stress_score", "value": "4.0", "anchor_time": "2024-01-02 10:00"},
            {"indicator": "stress_score", "value": "5.0", "anchor_time": "2024-01-03 10:00"},
            {"indicator": "sleep_hours", "value": "7.0", "anchor_time": "2024-01-01 08:00"},
            {"indicator": "sleep_hours", "value": "7.5", "anchor_time": "2024-01-02 08:00"},
            {"indicator": "sleep_hours", "value": "8.0", "anchor_time": "2024-01-03 08:00"},
        ]

        worker_results = _create_worker_dfs(records)
        result = validate_extraction(simple_causal_design, worker_results)

        # Should be valid (warnings only)
        assert result["is_valid"] is True

        # But should have low_n warnings
        low_n_warnings = [i for i in _all_issues(result) if i["issue_type"] == "low_n"]
        assert len(low_n_warnings) == 2  # Both indicators

    def test_non_numeric_values_are_errors(self, simple_causal_design):
        """Non-numeric values that can't be cast generate error."""
        records = [
            {"indicator": "stress_score", "value": "high", "anchor_time": "2024-01-01 10:00"},
            {"indicator": "stress_score", "value": "medium", "anchor_time": "2024-01-02 10:00"},
            {"indicator": "sleep_hours", "value": "7.0", "anchor_time": "2024-01-01 08:00"},
        ]

        worker_results = _create_worker_dfs(records)
        result = validate_extraction(simple_causal_design, worker_results)

        # stress_score should have no_numeric error
        stress_issues = _issues_for_indicator(result, "stress_score")
        assert any(i["issue_type"] == "no_numeric" for i in stress_issues)

    def test_combined_error_and_warning(self, simple_causal_design):
        """Indicator can have multiple issues."""
        records = [
            # stress_score: constant AND low N
            {"indicator": "stress_score", "value": "5.0", "anchor_time": "2024-01-01 10:00"},
            {"indicator": "stress_score", "value": "5.0", "anchor_time": "2024-01-02 10:00"},
            {"indicator": "stress_score", "value": "5.0", "anchor_time": "2024-01-03 10:00"},
            # sleep_hours: varying but low N
            {"indicator": "sleep_hours", "value": "7.0", "anchor_time": "2024-01-01 08:00"},
            {"indicator": "sleep_hours", "value": "8.0", "anchor_time": "2024-01-02 08:00"},
            {"indicator": "sleep_hours", "value": "7.5", "anchor_time": "2024-01-03 08:00"},
        ]

        worker_results = _create_worker_dfs(records)
        result = validate_extraction(simple_causal_design, worker_results)

        assert result["is_valid"] is False  # Has error

        # stress_score should have both issues
        stress_issues = _issues_for_indicator(result, "stress_score")
        issue_types = {i["issue_type"] for i in stress_issues}
        assert "no_variance" in issue_types
        assert "low_n" in issue_types

    def test_only_warnings_is_valid(self, simple_causal_design):
        """is_valid=True when only warnings exist."""
        records = []
        # Only 5 observations but varying
        for i in range(5):
            records.append(
                {
                    "indicator": "stress_score",
                    "value": str(float(i + 1)),
                    "anchor_time": f"2024-01-{i + 1:02d} 10:00",
                }
            )
            records.append(
                {
                    "indicator": "sleep_hours",
                    "value": str(6.0 + i * 0.5),
                    "anchor_time": f"2024-01-{i + 1:02d} 08:00",
                }
            )

        worker_results = _create_worker_dfs(records)
        result = validate_extraction(simple_causal_design, worker_results)

        assert result["is_valid"] is True
        issues = _all_issues(result)
        assert len(issues) > 0
        assert all(i["severity"] == "warning" for i in issues)


# ==============================================================================
# UNIT TESTS: _check_timestamps
# ==============================================================================


class TestCheckTimestamps:
    """Test timestamp parseability checks."""

    def test_all_parseable_no_issue(self):
        """All parseable timestamps produce no issues."""
        spec = _make_spec()
        records = [
            {
                "indicator": "stress_score",
                "value": str(float(i)),
                "anchor_time": f"2024-01-{i + 1:02d} 10:00",
            }
            for i in range(20)
        ]
        result = validate_extraction(spec, _create_worker_dfs(records))
        ts_issues = [i for i in _all_issues(result) if i["issue_type"] == "unparseable_timestamps"]
        assert len(ts_issues) == 0

    def test_native_datetime_timestamps_are_parseable(self):
        """Already-normalized datetime timestamps should remain parseable."""
        spec = _make_spec()
        df = pl.DataFrame(
            {
                "indicator": ["stress_score"] * 20,
                "value": [float(i) for i in range(20)],
                "anchor_time": [datetime(2024, 1, i + 1, 10, 0, 0) for i in range(20)],
            }
        )
        result = validate_extraction(spec, [df])
        ts_issues = [i for i in _all_issues(result) if i["issue_type"] == "unparseable_timestamps"]
        assert len(ts_issues) == 0

    def test_all_unparseable_is_error(self):
        """100% unparseable timestamps → error."""
        spec = _make_spec()
        records = [
            {"indicator": "stress_score", "value": str(float(i)), "anchor_time": "not-a-date"}
            for i in range(20)
        ]
        result = validate_extraction(spec, _create_worker_dfs(records))
        ts_issues = [i for i in _all_issues(result) if i["issue_type"] == "unparseable_timestamps"]
        assert len(ts_issues) == 1
        assert ts_issues[0]["severity"] == "error"

    def test_majority_unparseable_is_warning(self):
        """Over 50% unparseable timestamps → warning."""
        spec = _make_spec()
        records = []
        for i in range(20):
            ts = f"2024-01-{i + 1:02d} 10:00" if i < 8 else "garbage"
            records.append({"indicator": "stress_score", "value": str(float(i)), "anchor_time": ts})
        result = validate_extraction(spec, _create_worker_dfs(records))
        ts_issues = [i for i in _all_issues(result) if i["issue_type"] == "unparseable_timestamps"]
        assert len(ts_issues) == 1
        assert ts_issues[0]["severity"] == "warning"

    def test_minority_unparseable_no_issue(self):
        """Under 50% unparseable timestamps → no timestamp issue."""
        spec = _make_spec()
        records = []
        for i in range(20):
            ts = "garbage" if i < 5 else f"2024-01-{i + 1:02d} 10:00"
            records.append({"indicator": "stress_score", "value": str(float(i)), "anchor_time": ts})
        result = validate_extraction(spec, _create_worker_dfs(records))
        ts_issues = [i for i in _all_issues(result) if i["issue_type"] == "unparseable_timestamps"]
        assert len(ts_issues) == 0


# ==============================================================================
# UNIT TESTS: _check_dtype_range
# ==============================================================================


class TestCheckDtypeRange:
    """Test dtype range conformance checks."""

    def test_binary_valid(self):
        """Binary values in {0, 1} produce no dtype issues."""
        spec = _make_spec(dtype="binary")
        records = [
            {"indicator": "stress_score", "value": v, "anchor_time": f"2024-01-{i + 1:02d} 10:00"}
            for i, v in enumerate(["0", "1", "0", "1", "1", "0", "1", "0", "1", "0"] * 2)
        ]
        result = validate_extraction(spec, _create_worker_dfs(records))
        dtype_issues = [i for i in _all_issues(result) if i["issue_type"] == "dtype_violation"]
        assert len(dtype_issues) == 0

    def test_binary_violation_is_error(self):
        """Binary values outside {0, 1} → error."""
        spec = _make_spec(dtype="binary")
        records = [
            {"indicator": "stress_score", "value": v, "anchor_time": f"2024-01-{i + 1:02d} 10:00"}
            for i, v in enumerate(["0", "1", "2", "0.5", "1", "0", "1", "0", "1", "0"])
        ]
        result = validate_extraction(spec, _create_worker_dfs(records))
        dtype_issues = [i for i in _all_issues(result) if i["issue_type"] == "dtype_violation"]
        assert len(dtype_issues) == 1
        assert dtype_issues[0]["severity"] == "error"

    def test_count_negative_is_error(self):
        """Count indicator with negative values → error."""
        spec = _make_spec(dtype="count")
        records = [
            {"indicator": "stress_score", "value": v, "anchor_time": f"2024-01-{i + 1:02d} 10:00"}
            for i, v in enumerate(["3", "5", "-1", "2", "4", "0", "1", "6", "3", "2"])
        ]
        result = validate_extraction(spec, _create_worker_dfs(records))
        dtype_issues = [i for i in _all_issues(result) if i["issue_type"] == "dtype_violation"]
        assert any(i["severity"] == "error" for i in dtype_issues)
        assert any("negative" in i["message"] for i in dtype_issues)

    def test_count_fractional_is_error(self):
        """Count indicator with fractional values → error."""
        spec = _make_spec(dtype="count")
        records = [
            {"indicator": "stress_score", "value": v, "anchor_time": f"2024-01-{i + 1:02d} 10:00"}
            for i, v in enumerate(["3", "5", "2.5", "2", "4", "0", "1", "6", "3", "2"])
        ]
        result = validate_extraction(spec, _create_worker_dfs(records))
        dtype_issues = [i for i in _all_issues(result) if i["issue_type"] == "dtype_violation"]
        assert any(i["severity"] == "error" for i in dtype_issues)
        assert any("fractional" in i["message"] for i in dtype_issues)

    def test_continuous_outlier_warning(self):
        """Continuous data with extreme outlier → warning."""
        spec = _make_spec(dtype="continuous")
        values = [str(float(i)) for i in range(20)]
        values[-1] = "1000.0"  # Extreme outlier
        records = [
            {"indicator": "stress_score", "value": v, "anchor_time": f"2024-01-{i + 1:02d} 10:00"}
            for i, v in enumerate(values)
        ]
        result = validate_extraction(spec, _create_worker_dfs(records))
        dtype_issues = [i for i in _all_issues(result) if i["issue_type"] == "dtype_violation"]
        assert len(dtype_issues) == 1
        assert dtype_issues[0]["severity"] == "warning"

    def test_continuous_no_outlier(self):
        """Continuous data without outliers produces no dtype issues."""
        spec = _make_spec(dtype="continuous")
        records = [
            {
                "indicator": "stress_score",
                "value": str(float(i % 10)),
                "anchor_time": f"2024-01-{i + 1:02d} 10:00",
            }
            for i in range(20)
        ]
        result = validate_extraction(spec, _create_worker_dfs(records))
        dtype_issues = [i for i in _all_issues(result) if i["issue_type"] == "dtype_violation"]
        assert len(dtype_issues) == 0


# ==============================================================================
# UNIT TESTS: _check_time_coverage
# ==============================================================================


class TestCheckTimeCoverage:
    """Test time coverage checks."""

    def test_sufficient_coverage_no_issue(self):
        """Enough time span produces no coverage issue."""
        spec = _make_spec(model_clock="1d")
        records = [
            {
                "indicator": "stress_score",
                "value": str(float(i % 5)),
                "anchor_time": f"2024-01-{i + 1:02d} 10:00",
            }
            for i in range(20)
        ]
        result = validate_extraction(spec, _create_worker_dfs(records))
        cov_issues = [i for i in _all_issues(result) if i["issue_type"] == "insufficient_coverage"]
        assert len(cov_issues) == 0

    def test_insufficient_coverage_is_warning(self):
        """Short time span → insufficient_coverage warning."""
        spec = _make_spec(model_clock="1d")
        # Only 3 days of data, need 10 * 24h = 240h
        records = [
            {
                "indicator": "stress_score",
                "value": str(float(i)),
                "anchor_time": f"2024-01-{i + 1:02d} 10:00",
            }
            for i in range(3)
        ]
        result = validate_extraction(spec, _create_worker_dfs(records))
        cov_issues = [i for i in _all_issues(result) if i["issue_type"] == "insufficient_coverage"]
        assert len(cov_issues) == 1
        assert cov_issues[0]["severity"] == "warning"

    def test_time_invariant_skips_coverage(self):
        """Time-invariant constructs skip coverage check."""
        spec = _make_spec(model_clock=None, temporal_status="time_invariant")
        records = [
            {
                "indicator": "stress_score",
                "value": str(float(i)),
                "anchor_time": f"2024-01-0{i + 1} 10:00",
            }
            for i in range(3)
        ]
        result = validate_extraction(spec, _create_worker_dfs(records))
        cov_issues = [i for i in _all_issues(result) if i["issue_type"] == "insufficient_coverage"]
        assert len(cov_issues) == 0

    def test_weekly_granularity_needs_more_span(self):
        """Weekly granularity requires 10 * 168h = 1680h of coverage."""
        spec = _make_spec(model_clock="1w")
        # 20 days < 70 days needed
        records = [
            {
                "indicator": "stress_score",
                "value": str(float(i % 5)),
                "anchor_time": f"2024-01-{i + 1:02d} 10:00",
            }
            for i in range(20)
        ]
        result = validate_extraction(spec, _create_worker_dfs(records))
        cov_issues = [i for i in _all_issues(result) if i["issue_type"] == "insufficient_coverage"]
        assert len(cov_issues) == 1


# ==============================================================================
# UNIT TESTS: _check_timestamp_gaps
# ==============================================================================


class TestCheckTimestampGaps:
    """Test timestamp gap detection."""

    def test_no_large_gaps(self):
        """Regular daily data has no large gaps."""
        spec = _make_spec(model_clock="1d")
        records = [
            {
                "indicator": "stress_score",
                "value": str(float(i % 5)),
                "anchor_time": f"2024-01-{i + 1:02d} 10:00",
            }
            for i in range(20)
        ]
        result = validate_extraction(spec, _create_worker_dfs(records))
        gap_issues = [i for i in _all_issues(result) if i["issue_type"] == "large_timestamp_gap"]
        assert len(gap_issues) == 0

    def test_large_gap_warning(self):
        """Gap > 5x granularity → warning."""
        spec = _make_spec(model_clock="1d")
        # 3 observations with a 10-day gap (>5x daily=120h)
        records = [
            {"indicator": "stress_score", "value": "1.0", "anchor_time": "2024-01-01 10:00"},
            {"indicator": "stress_score", "value": "2.0", "anchor_time": "2024-01-02 10:00"},
            {"indicator": "stress_score", "value": "3.0", "anchor_time": "2024-01-03 10:00"},
            {"indicator": "stress_score", "value": "4.0", "anchor_time": "2024-01-04 10:00"},
            {"indicator": "stress_score", "value": "5.0", "anchor_time": "2024-01-20 10:00"},
            {"indicator": "stress_score", "value": "6.0", "anchor_time": "2024-01-21 10:00"},
            {"indicator": "stress_score", "value": "7.0", "anchor_time": "2024-01-22 10:00"},
            {"indicator": "stress_score", "value": "8.0", "anchor_time": "2024-01-23 10:00"},
            {"indicator": "stress_score", "value": "9.0", "anchor_time": "2024-01-24 10:00"},
            {"indicator": "stress_score", "value": "10.0", "anchor_time": "2024-01-25 10:00"},
        ]
        result = validate_extraction(spec, _create_worker_dfs(records))
        gap_issues = [i for i in _all_issues(result) if i["issue_type"] == "large_timestamp_gap"]
        assert len(gap_issues) == 1
        assert gap_issues[0]["severity"] == "warning"

    def test_skips_with_few_timestamps(self):
        """Fewer than 3 timestamps skips gap check."""
        spec = _make_spec(model_clock="1d")
        records = [
            {"indicator": "stress_score", "value": "1.0", "anchor_time": "2024-01-01 10:00"},
            {"indicator": "stress_score", "value": "2.0", "anchor_time": "2024-06-01 10:00"},
        ]
        result = validate_extraction(spec, _create_worker_dfs(records))
        gap_issues = [i for i in _all_issues(result) if i["issue_type"] == "large_timestamp_gap"]
        assert len(gap_issues) == 0


# ==============================================================================
# UNIT TESTS: _check_hallucination_signals
# ==============================================================================


class TestCheckHallucinationSignals:
    """Test hallucination signal detection."""

    def test_clean_data_no_warning(self):
        """Normal data produces no hallucination warnings."""
        spec = _make_spec(dtype="continuous")
        records = [
            {
                "indicator": "stress_score",
                "value": str(float(i % 7 + 1)),
                "anchor_time": f"2024-01-{i + 1:02d} 10:00",
            }
            for i in range(20)
        ]
        result = validate_extraction(spec, _create_worker_dfs(records))
        hall_issues = [i for i in _all_issues(result) if i["issue_type"] == "suspicious_pattern"]
        assert len(hall_issues) == 0

    def test_excessive_duplicates_warning(self):
        """Over 50% same value in continuous data → warning."""
        spec = _make_spec(dtype="continuous")
        # 15 out of 20 are 5.0
        values = ["5.0"] * 15 + ["1.0", "2.0", "3.0", "4.0", "6.0"]
        records = [
            {"indicator": "stress_score", "value": v, "anchor_time": f"2024-01-{i + 1:02d} 10:00"}
            for i, v in enumerate(values)
        ]
        result = validate_extraction(spec, _create_worker_dfs(records))
        hall_issues = [i for i in _all_issues(result) if i["issue_type"] == "suspicious_pattern"]
        assert any("5.0" in i["message"] for i in hall_issues)

    def test_arithmetic_sequence_warning(self):
        """Perfect arithmetic sequence → warning."""
        spec = _make_spec(dtype="continuous")
        records = [
            {
                "indicator": "stress_score",
                "value": str(float(i * 2)),
                "anchor_time": f"2024-01-{i + 1:02d} 10:00",
            }
            for i in range(20)
        ]
        result = validate_extraction(spec, _create_worker_dfs(records))
        hall_issues = [i for i in _all_issues(result) if i["issue_type"] == "suspicious_pattern"]
        assert any("arithmetic sequence" in i["message"] for i in hall_issues)

    def test_binary_exempt_from_duplicates(self):
        """Binary data with >50% same value is natural, not flagged."""
        spec = _make_spec(dtype="binary")
        # 15 out of 20 are 1.0 — normal for binary
        values = ["1"] * 15 + ["0"] * 5
        records = [
            {"indicator": "stress_score", "value": v, "anchor_time": f"2024-01-{i + 1:02d} 10:00"}
            for i, v in enumerate(values)
        ]
        result = validate_extraction(spec, _create_worker_dfs(records))
        hall_issues = [
            i
            for i in _all_issues(result)
            if i["issue_type"] == "suspicious_pattern"
            and "duplicate" in i.get("message", "").lower()
        ]
        # No duplicate-based hallucination warning for binary
        assert len(hall_issues) == 0

    def test_count_exempt_from_duplicates(self):
        """Count data with >50% same value is natural, not flagged."""
        spec = _make_spec(dtype="count")
        # Lots of zeros is typical for count data
        values = ["0"] * 15 + ["1", "2", "3", "4", "5"]
        records = [
            {"indicator": "stress_score", "value": v, "anchor_time": f"2024-01-{i + 1:02d} 10:00"}
            for i, v in enumerate(values)
        ]
        result = validate_extraction(spec, _create_worker_dfs(records))
        hall_issues = [
            i
            for i in _all_issues(result)
            if i["issue_type"] == "suspicious_pattern"
            and "duplicate" in i.get("message", "").lower()
        ]
        assert len(hall_issues) == 0


# ==============================================================================
# UNIT TESTS: _check_construct_correlations
# ==============================================================================


class TestCheckConstructCorrelations:
    """Test cross-indicator construct correlation checks."""

    def test_positive_correlation_no_issue(self):
        """Positively correlated indicators within a construct pass."""
        spec = _make_spec(
            indicator_name="stress_score",
            construct_name="stress",
            extra_indicators=[
                {
                    "name": "stress_self_report",
                    "construct_name": "stress",
                    "measurement_dtype": "continuous",
                    "how_to_measure": "Self reported stress",
                },
            ],
        )
        records = []
        for i in range(20):
            val = float(i % 5 + 1)
            records.append(
                {
                    "indicator": "stress_score",
                    "value": str(val),
                    "anchor_time": f"2024-01-{i + 1:02d} 10:00",
                }
            )
            records.append(
                {
                    "indicator": "stress_self_report",
                    "value": str(val + 0.5),  # Positively correlated
                    "anchor_time": f"2024-01-{i + 1:02d} 10:00",
                }
            )
        result = validate_extraction(spec, _create_worker_dfs(records))
        corr_issues = [
            i for i in _all_issues(result) if i["issue_type"] == "low_construct_correlation"
        ]
        assert len(corr_issues) == 0

    def test_negative_correlation_warns(self):
        """Negatively correlated indicators → warning."""
        spec = _make_spec(
            indicator_name="stress_score",
            construct_name="stress",
            extra_indicators=[
                {
                    "name": "stress_self_report",
                    "construct_name": "stress",
                    "measurement_dtype": "continuous",
                    "how_to_measure": "Self reported stress",
                },
            ],
        )
        records = []
        for i in range(20):
            val = float(i % 5 + 1)
            records.append(
                {
                    "indicator": "stress_score",
                    "value": str(val),
                    "anchor_time": f"2024-01-{i + 1:02d} 10:00",
                }
            )
            records.append(
                {
                    "indicator": "stress_self_report",
                    "value": str(10.0 - val),  # Negatively correlated
                    "anchor_time": f"2024-01-{i + 1:02d} 10:00",
                }
            )
        result = validate_extraction(spec, _create_worker_dfs(records))
        corr_issues = [
            i for i in _all_issues(result) if i["issue_type"] == "low_construct_correlation"
        ]
        assert len(corr_issues) == 1
        assert corr_issues[0]["severity"] == "warning"

    def test_single_indicator_skipped(self):
        """Constructs with only one indicator skip correlation check."""
        spec = _make_spec()
        records = [
            {
                "indicator": "stress_score",
                "value": str(float(i % 5)),
                "anchor_time": f"2024-01-{i + 1:02d} 10:00",
            }
            for i in range(20)
        ]
        result = validate_extraction(spec, _create_worker_dfs(records))
        corr_issues = [
            i for i in _all_issues(result) if i["issue_type"] == "low_construct_correlation"
        ]
        assert len(corr_issues) == 0

    def test_insufficient_aligned_skipped(self):
        """Fewer than MIN_ALIGNED_FOR_CFA aligned observations skips check."""
        spec = _make_spec(
            indicator_name="stress_score",
            construct_name="stress",
            extra_indicators=[
                {
                    "name": "stress_self_report",
                    "construct_name": "stress",
                    "measurement_dtype": "continuous",
                    "how_to_measure": "Self reported stress",
                },
            ],
        )
        # Different timestamps so nothing aligns
        records = []
        for i in range(20):
            records.append(
                {
                    "indicator": "stress_score",
                    "value": str(float(i)),
                    "anchor_time": f"2024-01-{i + 1:02d} 10:00",
                }
            )
            records.append(
                {
                    "indicator": "stress_self_report",
                    "value": str(float(20 - i)),
                    "anchor_time": f"2024-02-{i + 1:02d} 10:00",  # Different month
                }
            )
        result = validate_extraction(spec, _create_worker_dfs(records))
        corr_issues = [
            i for i in _all_issues(result) if i["issue_type"] == "low_construct_correlation"
        ]
        assert len(corr_issues) == 0
