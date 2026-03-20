"""Tests for worker schema validation.

Covers: _check_dtype_match, validate_worker_output, WorkerOutput.to_dataframe.
"""

import polars as pl

from causal_ssm_agent.workers.schemas import (
    WindowExtraction,
    WorkerOutput,
    _check_dtype_match,
    validate_worker_output,
)


def _causal_spec(*indicators):
    """Build a minimal CausalSpec dict with given indicator tuples (name, dtype)."""
    default_aggregations = {
        "continuous": "mean",
        "binary": "last",
        "count": "count",
        "ordinal": "last",
        "categorical": "last",
    }
    return {
        "measurement": {
            "model_clock": "1d",
            "indicators": [
                {
                    "name": name,
                    "measurement_dtype": dtype,
                    "aggregation": default_aggregations.get(dtype, "last"),
                    **({"ordinal_levels": ["low", "medium", "high"]} if dtype == "ordinal" else {}),
                }
                for name, dtype in indicators
            ],
        }
    }


# =============================================================================
# _check_dtype_match
# =============================================================================


class TestCheckDtypeMatch:
    def test_none_always_valid(self):
        assert _check_dtype_match(None, "continuous") is True
        assert _check_dtype_match(None, "binary") is True
        assert _check_dtype_match(None, "count") is True

    def test_continuous_int(self):
        assert _check_dtype_match(42, "continuous") is True

    def test_continuous_float(self):
        assert _check_dtype_match(3.14, "continuous") is True

    def test_continuous_string_rejected(self):
        assert _check_dtype_match("hello", "continuous") is False

    def test_binary_bool(self):
        assert _check_dtype_match(True, "binary") is True
        assert _check_dtype_match(False, "binary") is True

    def test_binary_int_01(self):
        assert _check_dtype_match(0, "binary") is True
        assert _check_dtype_match(1, "binary") is True

    def test_binary_string_01(self):
        assert _check_dtype_match("0", "binary") is True
        assert _check_dtype_match("1", "binary") is True

    def test_binary_string_true_false(self):
        assert _check_dtype_match("true", "binary") is True
        assert _check_dtype_match("false", "binary") is True
        assert _check_dtype_match("True", "binary") is True
        assert _check_dtype_match("False", "binary") is True

    def test_binary_other_rejected(self):
        assert _check_dtype_match("maybe", "binary") is False

    def test_count_positive_int(self):
        assert _check_dtype_match(5, "count") is True

    def test_count_zero(self):
        assert _check_dtype_match(0, "count") is True

    def test_count_float_whole(self):
        assert _check_dtype_match(3.0, "count") is True

    def test_count_negative_int_accepted(self):
        # dtype check validates type, not value range; -1 is still an int
        assert _check_dtype_match(-1, "count") is True

    def test_count_float_fractional_rejected(self):
        assert _check_dtype_match(3.5, "count") is False

    def test_ordinal_accepts_int(self):
        assert _check_dtype_match(3, "ordinal") is True

    def test_ordinal_rejects_string_label(self):
        assert _check_dtype_match("moderate", "ordinal") is False

    def test_categorical_string(self):
        assert _check_dtype_match("category_a", "categorical") is True

    def test_categorical_int_rejected(self):
        assert _check_dtype_match(42, "categorical") is False

    def test_unknown_dtype_always_valid(self):
        assert _check_dtype_match("anything", "unknown_dtype") is True


# =============================================================================
# validate_worker_output
# =============================================================================


class TestValidateWorkerOutput:
    def test_valid_single_extraction(self):
        spec = _causal_spec(("mood", "continuous"))
        data = {"extractions": [{"window_start": "2024-01-01", "indicator": "mood", "value": 3.5}]}
        output, errors = validate_worker_output(data, spec)
        assert output is not None
        assert errors == []
        assert len(output.extractions) == 1

    def test_empty_extractions_valid(self):
        spec = _causal_spec(("mood", "continuous"))
        data = {"extractions": []}
        output, errors = validate_worker_output(data, spec)
        assert output is not None
        assert errors == []

    def test_missing_extractions_defaults_empty(self):
        spec = _causal_spec(("mood", "continuous"))
        data = {}
        output, errors = validate_worker_output(data, spec)
        assert output is not None
        assert errors == []

    def test_not_dict_returns_error(self):
        spec = _causal_spec(("mood", "continuous"))
        output, errors = validate_worker_output("not a dict", spec)
        assert output is None
        assert len(errors) == 1
        assert "dictionary" in errors[0].lower()

    def test_extractions_not_list(self):
        spec = _causal_spec(("mood", "continuous"))
        data = {"extractions": "bad"}
        output, errors = validate_worker_output(data, spec)
        assert output is None
        assert any("list" in e.lower() for e in errors)

    def test_extraction_not_dict(self):
        spec = _causal_spec(("mood", "continuous"))
        data = {"extractions": [42]}
        output, errors = validate_worker_output(data, spec)
        assert output is None
        assert any("dictionary" in e.lower() for e in errors)

    def test_unknown_indicator_error(self):
        spec = _causal_spec(("mood", "continuous"))
        data = {
            "extractions": [
                {"window_start": "2024-01-01", "indicator": "nonexistent", "value": 1.0}
            ]
        }
        output, errors = validate_worker_output(data, spec)
        assert output is None
        assert any("nonexistent" in e for e in errors)
        assert any("mood" in e for e in errors)  # suggests valid indicators

    def test_dtype_mismatch_error(self):
        spec = _causal_spec(("mood", "continuous"))
        data = {
            "extractions": [
                {"window_start": "2024-01-01", "indicator": "mood", "value": "not_a_number"}
            ]
        }
        output, errors = validate_worker_output(data, spec)
        assert output is None
        assert any("dtype" in e for e in errors)

    def test_multiple_errors_collected(self):
        spec = _causal_spec(("mood", "continuous"))
        data = {
            "extractions": [
                {"window_start": "2024-01-01", "indicator": "bad1", "value": 1.0},
                {"window_start": "2024-01-01", "indicator": "bad2", "value": 2.0},
            ]
        }
        output, errors = validate_worker_output(data, spec)
        assert output is None
        assert len(errors) >= 2

    def test_window_start_preserved(self):
        spec = _causal_spec(("mood", "continuous"))
        data = {"extractions": [{"window_start": "2024-01-01", "indicator": "mood", "value": 5.0}]}
        output, errors = validate_worker_output(data, spec)
        assert output is not None
        assert errors == []
        assert output.extractions[0].window_start == "2024-01-01"

    def test_null_value_accepted(self):
        spec = _causal_spec(("mood", "continuous"))
        data = {"extractions": [{"window_start": "2024-01-01", "indicator": "mood", "value": None}]}
        output, errors = validate_worker_output(data, spec)
        assert output is not None
        assert errors == []

    def test_binary_valid(self):
        spec = _causal_spec(("is_smoking", "binary"))
        data = {
            "extractions": [
                {"window_start": "2024-01-01", "indicator": "is_smoking", "value": True}
            ]
        }
        output, errors = validate_worker_output(data, spec)
        assert output is not None
        assert errors == []

    def test_multiple_indicators(self):
        spec = _causal_spec(("mood", "continuous"), ("is_smoking", "binary"))
        data = {
            "extractions": [
                {"window_start": "2024-01-01", "indicator": "mood", "value": 7.0},
                {"window_start": "2024-01-01", "indicator": "is_smoking", "value": False},
            ]
        }
        output, errors = validate_worker_output(data, spec)
        assert output is not None
        assert errors == []
        assert len(output.extractions) == 2

    def test_duplicate_window_start_indicator_rejected(self):
        spec = _causal_spec(("mood", "continuous"))
        data = {
            "extractions": [
                {"window_start": "2024-01-01", "indicator": "mood", "value": 3.0},
                {"window_start": "2024-01-01", "indicator": "mood", "value": 4.0},
            ]
        }
        output, errors = validate_worker_output(data, spec)
        assert output is None
        assert any("duplicate" in e.lower() for e in errors)

    def test_unexpected_window_start_rejected_when_expected_window_starts_given(self):
        spec = _causal_spec(("mood", "continuous"))
        data = {
            "extractions": [
                {"window_start": "2024-01-99", "indicator": "mood", "value": 3.0},
            ]
        }
        output, errors = validate_worker_output(
            data, spec, expected_window_starts=["2024-01-01", "2024-01-02"]
        )
        assert output is None
        assert any("not in expected support windows" in e for e in errors)

    def test_ordinal_requires_numeric_code(self):
        spec = _causal_spec(("severity", "ordinal"))
        data = {
            "extractions": [
                {"window_start": "2024-01-01", "indicator": "severity", "value": "high"}
            ]
        }
        output, errors = validate_worker_output(data, spec)
        assert output is None
        assert any("expected dtype 'ordinal'" in e for e in errors)

    def test_ordinal_code_must_be_in_range(self):
        spec = _causal_spec(("severity", "ordinal"))
        data = {
            "extractions": [{"window_start": "2024-01-01", "indicator": "severity", "value": 3}]
        }
        output, errors = validate_worker_output(data, spec)
        assert output is None
        assert any("must be in 0..2" in e for e in errors)

    def test_ordinal_value_normalized_to_int(self):
        spec = _causal_spec(("severity", "ordinal"))
        data = {
            "extractions": [{"window_start": "2024-01-01", "indicator": "severity", "value": 2.0}]
        }
        output, errors = validate_worker_output(data, spec)
        assert errors == []
        assert output is not None
        assert output.extractions[0].value == 2


# =============================================================================
# WorkerOutput.to_dataframe
# =============================================================================


class TestWorkerOutputToDataframe:
    def test_empty_extractions(self):
        output = WorkerOutput(extractions=[])
        df = output.to_dataframe()
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0
        assert set(df.columns) == {"indicator", "value", "timestamp"}

    def test_basic_conversion(self):
        output = WorkerOutput(
            extractions=[
                WindowExtraction(window_start="2024-01-01", indicator="mood", value=7.5),
            ]
        )
        df = output.to_dataframe()
        assert len(df) == 1
        assert df["indicator"][0] == "mood"
        assert df["value"][0] == "7.5"
        assert df["timestamp"][0] == "2024-01-01"

    def test_none_value_preserved(self):
        output = WorkerOutput(
            extractions=[WindowExtraction(window_start="2024-01-01", indicator="mood", value=None)]
        )
        df = output.to_dataframe()
        assert df["value"][0] is None

    def test_bool_converted_to_string(self):
        output = WorkerOutput(
            extractions=[WindowExtraction(window_start="2024-01-01", indicator="smoke", value=True)]
        )
        df = output.to_dataframe()
        assert df["value"][0] == "True"

    def test_multiple_rows(self):
        output = WorkerOutput(
            extractions=[
                WindowExtraction(window_start="2024-01-01", indicator="mood", value=7.0),
                WindowExtraction(window_start="2024-01-01", indicator="sleep", value=8.0),
            ]
        )
        df = output.to_dataframe()
        assert len(df) == 2
