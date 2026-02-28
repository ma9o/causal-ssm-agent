"""Tests for worker schema validation.

Covers: _check_dtype_match, validate_worker_output, WorkerOutput.to_dataframe.
"""

import polars as pl

from causal_ssm_agent.workers.schemas import (
    Extraction,
    WorkerOutput,
    _check_dtype_match,
    validate_worker_output,
)


def _causal_spec(*indicators):
    """Build a minimal CausalSpec dict with given indicator tuples (name, dtype)."""
    return {
        "measurement": {
            "indicators": [
                {"name": name, "measurement_dtype": dtype}
                for name, dtype in indicators
            ]
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

    def test_ordinal_accepts_string(self):
        assert _check_dtype_match("moderate", "ordinal") is True

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
        data = {"extractions": [{"indicator": "mood", "value": 3.5}]}
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
        data = {"extractions": [{"indicator": "nonexistent", "value": 1.0}]}
        output, errors = validate_worker_output(data, spec)
        assert output is None
        assert any("nonexistent" in e for e in errors)
        assert any("mood" in e for e in errors)  # suggests valid indicators

    def test_dtype_mismatch_error(self):
        spec = _causal_spec(("mood", "continuous"))
        data = {"extractions": [{"indicator": "mood", "value": "not_a_number"}]}
        output, errors = validate_worker_output(data, spec)
        assert output is None
        assert any("dtype" in e for e in errors)

    def test_multiple_errors_collected(self):
        spec = _causal_spec(("mood", "continuous"))
        data = {
            "extractions": [
                {"indicator": "bad1", "value": 1.0},
                {"indicator": "bad2", "value": 2.0},
            ]
        }
        output, errors = validate_worker_output(data, spec)
        assert output is None
        assert len(errors) >= 2

    def test_timestamp_preserved(self):
        spec = _causal_spec(("mood", "continuous"))
        data = {
            "extractions": [
                {"indicator": "mood", "value": 5.0, "timestamp": "2024-01-01T10:00:00"}
            ]
        }
        output, errors = validate_worker_output(data, spec)
        assert output is not None
        assert errors == []
        assert output.extractions[0].timestamp == "2024-01-01T10:00:00"

    def test_null_value_accepted(self):
        spec = _causal_spec(("mood", "continuous"))
        data = {"extractions": [{"indicator": "mood", "value": None}]}
        output, errors = validate_worker_output(data, spec)
        assert output is not None
        assert errors == []

    def test_binary_valid(self):
        spec = _causal_spec(("is_smoking", "binary"))
        data = {"extractions": [{"indicator": "is_smoking", "value": True}]}
        output, errors = validate_worker_output(data, spec)
        assert output is not None
        assert errors == []

    def test_multiple_indicators(self):
        spec = _causal_spec(("mood", "continuous"), ("is_smoking", "binary"))
        data = {
            "extractions": [
                {"indicator": "mood", "value": 7.0},
                {"indicator": "is_smoking", "value": False},
            ]
        }
        output, errors = validate_worker_output(data, spec)
        assert output is not None
        assert errors == []
        assert len(output.extractions) == 2


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
                Extraction(indicator="mood", value=7.5, timestamp="2024-01-01"),
            ]
        )
        df = output.to_dataframe()
        assert len(df) == 1
        assert df["indicator"][0] == "mood"
        assert df["value"][0] == "7.5"
        assert df["timestamp"][0] == "2024-01-01"

    def test_none_value_preserved(self):
        output = WorkerOutput(
            extractions=[Extraction(indicator="mood", value=None)]
        )
        df = output.to_dataframe()
        assert df["value"][0] is None

    def test_bool_converted_to_string(self):
        output = WorkerOutput(
            extractions=[Extraction(indicator="smoke", value=True)]
        )
        df = output.to_dataframe()
        assert df["value"][0] == "True"

    def test_multiple_rows(self):
        output = WorkerOutput(
            extractions=[
                Extraction(indicator="mood", value=7.0),
                Extraction(indicator="sleep", value=8.0),
            ]
        )
        df = output.to_dataframe()
        assert len(df) == 2
