"""Tests for Stage 3: Validate extracted data.

This module tests:
1. validate_extraction() - semantic checks (variance, sample size)
2. combine_worker_results() - combining raw worker data
"""

from dataclasses import dataclass

import polars as pl
import pytest

from dsem_agent.flows.stages.stage3_validation import (
    MIN_OBSERVATIONS,
    combine_worker_results,
    validate_extraction,
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@dataclass
class MockWorkerResult:
    """Mock WorkerResult for testing."""

    dataframe: pl.DataFrame


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
                },
                {
                    "name": "sleep_hours",
                    "construct": "sleep",
                    "how_to_measure": "Extract sleep duration",
                },
            ],
        },
    }


def _create_worker_results(records: list[dict]) -> list[MockWorkerResult]:
    """Create mock worker results from records."""
    df = pl.DataFrame(
        records,
        schema={"indicator": pl.Utf8, "value": pl.Utf8, "timestamp": pl.Utf8},
    )
    return [MockWorkerResult(dataframe=df)]


# ==============================================================================
# UNIT TESTS: combine_worker_results
# ==============================================================================


class TestCombineWorkerResults:
    """Test combining raw worker results."""

    def test_combines_multiple_workers(self):
        """Combines DataFrames from multiple workers."""
        df1 = pl.DataFrame({
            "indicator": ["stress_score"],
            "value": ["5.0"],
            "timestamp": ["2024-01-01 10:00"],
        })
        df2 = pl.DataFrame({
            "indicator": ["sleep_hours"],
            "value": ["7.5"],
            "timestamp": ["2024-01-01 08:00"],
        })

        results = [MockWorkerResult(dataframe=df1), MockWorkerResult(dataframe=df2)]
        combined = combine_worker_results.fn(results)

        assert len(combined) == 2
        assert set(combined["indicator"].to_list()) == {"stress_score", "sleep_hours"}

    def test_handles_empty_results(self):
        """Returns empty DataFrame for no results."""
        combined = combine_worker_results.fn([])
        assert combined.is_empty()

    def test_handles_none_dataframes(self):
        """Skips None DataFrames."""
        df = pl.DataFrame({
            "indicator": ["stress_score"],
            "value": ["5.0"],
            "timestamp": ["2024-01-01 10:00"],
        })
        results = [
            MockWorkerResult(dataframe=df),
            MockWorkerResult(dataframe=None),
        ]
        combined = combine_worker_results.fn(results)

        assert len(combined) == 1


# ==============================================================================
# UNIT TESTS: validate_extraction
# ==============================================================================


class TestValidateExtraction:
    """Test validate_extraction semantic checks."""

    def test_empty_results_returns_error(self, simple_dsem_model):
        """Empty worker results returns error."""
        result = validate_extraction.fn(simple_dsem_model, [])
        assert result["is_valid"] is False
        assert any(i["issue_type"] == "no_data" for i in result["issues"])

    def test_valid_data_no_issues(self, simple_dsem_model):
        """Valid data with sufficient variance and sample size passes."""
        records = []
        for i in range(20):
            records.append({
                "indicator": "stress_score",
                "value": str(float(i % 5 + 1)),  # 1-5 varying
                "timestamp": f"2024-01-{i+1:02d} 10:00",
            })
            records.append({
                "indicator": "sleep_hours",
                "value": str(6.0 + (i % 3)),  # 6-8 varying
                "timestamp": f"2024-01-{i+1:02d} 08:00",
            })

        worker_results = _create_worker_results(records)
        result = validate_extraction.fn(simple_dsem_model, worker_results)

        assert result["is_valid"] is True
        # May have warnings but no errors
        errors = [i for i in result["issues"] if i["severity"] == "error"]
        assert len(errors) == 0

    def test_missing_indicator_is_warning(self, simple_dsem_model):
        """Missing indicator generates warning."""
        records = [
            {"indicator": "stress_score", "value": "5.0", "timestamp": "2024-01-01 10:00"},
            # sleep_hours is missing
        ]
        worker_results = _create_worker_results(records)
        result = validate_extraction.fn(simple_dsem_model, worker_results)

        # Should have warning for missing sleep_hours
        missing_issues = [i for i in result["issues"] if i["issue_type"] == "missing"]
        assert any(i["indicator"] == "sleep_hours" for i in missing_issues)

    def test_zero_variance_is_error(self, simple_dsem_model):
        """Constant values (zero variance) returns error."""
        records = []
        for i in range(20):
            records.append({
                "indicator": "stress_score",
                "value": "5.0",  # Constant!
                "timestamp": f"2024-01-{i+1:02d} 10:00",
            })
            records.append({
                "indicator": "sleep_hours",
                "value": str(6.0 + (i % 3)),  # Varying
                "timestamp": f"2024-01-{i+1:02d} 08:00",
            })

        worker_results = _create_worker_results(records)
        result = validate_extraction.fn(simple_dsem_model, worker_results)

        assert result["is_valid"] is False

        error_issues = [i for i in result["issues"] if i["severity"] == "error"]
        assert len(error_issues) == 1
        assert error_issues[0]["indicator"] == "stress_score"
        assert error_issues[0]["issue_type"] == "no_variance"

    def test_low_sample_size_is_warning(self, simple_dsem_model):
        """Low sample size generates warning."""
        records = [
            {"indicator": "stress_score", "value": "3.0", "timestamp": "2024-01-01 10:00"},
            {"indicator": "stress_score", "value": "4.0", "timestamp": "2024-01-02 10:00"},
            {"indicator": "stress_score", "value": "5.0", "timestamp": "2024-01-03 10:00"},
            {"indicator": "sleep_hours", "value": "7.0", "timestamp": "2024-01-01 08:00"},
            {"indicator": "sleep_hours", "value": "7.5", "timestamp": "2024-01-02 08:00"},
            {"indicator": "sleep_hours", "value": "8.0", "timestamp": "2024-01-03 08:00"},
        ]

        worker_results = _create_worker_results(records)
        result = validate_extraction.fn(simple_dsem_model, worker_results)

        # Should be valid (warnings only)
        assert result["is_valid"] is True

        # But should have low_n warnings
        low_n_warnings = [i for i in result["issues"] if i["issue_type"] == "low_n"]
        assert len(low_n_warnings) == 2  # Both indicators

    def test_minimum_observations_threshold(self):
        """Verify MIN_OBSERVATIONS is set reasonably."""
        assert MIN_OBSERVATIONS >= 5  # At least a few observations needed

    def test_non_numeric_values_are_errors(self, simple_dsem_model):
        """Non-numeric values that can't be cast generate error."""
        records = [
            {"indicator": "stress_score", "value": "high", "timestamp": "2024-01-01 10:00"},
            {"indicator": "stress_score", "value": "medium", "timestamp": "2024-01-02 10:00"},
            {"indicator": "sleep_hours", "value": "7.0", "timestamp": "2024-01-01 08:00"},
        ]

        worker_results = _create_worker_results(records)
        result = validate_extraction.fn(simple_dsem_model, worker_results)

        # stress_score should have no_numeric error
        stress_issues = [i for i in result["issues"] if i["indicator"] == "stress_score"]
        assert any(i["issue_type"] == "no_numeric" for i in stress_issues)

    def test_combined_error_and_warning(self, simple_dsem_model):
        """Indicator can have multiple issues."""
        records = [
            # stress_score: constant AND low N
            {"indicator": "stress_score", "value": "5.0", "timestamp": "2024-01-01 10:00"},
            {"indicator": "stress_score", "value": "5.0", "timestamp": "2024-01-02 10:00"},
            {"indicator": "stress_score", "value": "5.0", "timestamp": "2024-01-03 10:00"},
            # sleep_hours: varying but low N
            {"indicator": "sleep_hours", "value": "7.0", "timestamp": "2024-01-01 08:00"},
            {"indicator": "sleep_hours", "value": "8.0", "timestamp": "2024-01-02 08:00"},
            {"indicator": "sleep_hours", "value": "7.5", "timestamp": "2024-01-03 08:00"},
        ]

        worker_results = _create_worker_results(records)
        result = validate_extraction.fn(simple_dsem_model, worker_results)

        assert result["is_valid"] is False  # Has error

        # stress_score should have both issues
        stress_issues = [i for i in result["issues"] if i["indicator"] == "stress_score"]
        issue_types = {i["issue_type"] for i in stress_issues}
        assert "no_variance" in issue_types
        assert "low_n" in issue_types

    def test_only_warnings_is_valid(self, simple_dsem_model):
        """is_valid=True when only warnings exist."""
        records = []
        # Only 5 observations but varying
        for i in range(5):
            records.append({
                "indicator": "stress_score",
                "value": str(float(i + 1)),
                "timestamp": f"2024-01-{i+1:02d} 10:00",
            })
            records.append({
                "indicator": "sleep_hours",
                "value": str(6.0 + i * 0.5),
                "timestamp": f"2024-01-{i+1:02d} 08:00",
            })

        worker_results = _create_worker_results(records)
        result = validate_extraction.fn(simple_dsem_model, worker_results)

        assert result["is_valid"] is True
        assert len(result["issues"]) > 0
        assert all(i["severity"] == "warning" for i in result["issues"])
