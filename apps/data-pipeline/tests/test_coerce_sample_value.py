"""Tests for pipeline._coerce_sample_value helper."""

from causal_ssm_agent.flows.pipeline import _coerce_sample_value


class TestCoerceSampleValue:
    def test_none_passthrough(self):
        assert _coerce_sample_value(None) is None

    def test_bool_passthrough(self):
        assert _coerce_sample_value(True) is True
        assert _coerce_sample_value(False) is False

    def test_int_passthrough(self):
        assert _coerce_sample_value(42) == 42
        assert _coerce_sample_value(-1) == -1

    def test_float_passthrough(self):
        assert _coerce_sample_value(3.14) == 3.14

    def test_string_true(self):
        assert _coerce_sample_value("true") is True
        assert _coerce_sample_value("True") is True
        assert _coerce_sample_value("TRUE") is True
        assert _coerce_sample_value("  true  ") is True

    def test_string_false(self):
        assert _coerce_sample_value("false") is False
        assert _coerce_sample_value("False") is False

    def test_string_int(self):
        assert _coerce_sample_value("42") == 42
        assert _coerce_sample_value("-7") == -7
        assert _coerce_sample_value("0") == 0

    def test_string_float(self):
        result = _coerce_sample_value("3.14")
        assert isinstance(result, float)
        assert abs(result - 3.14) < 1e-10

    def test_string_inf_kept_as_string(self):
        assert _coerce_sample_value("inf") == "inf"
        assert _coerce_sample_value("-inf") == "-inf"

    def test_plain_string_passthrough(self):
        assert _coerce_sample_value("hello") == "hello"
        assert _coerce_sample_value("  hello  ") == "hello"

    def test_non_string_non_scalar(self):
        """Non-scalar objects get str() conversion."""
        assert _coerce_sample_value([1, 2]) == "[1, 2]"

    def test_empty_string(self):
        assert _coerce_sample_value("") == ""
        assert _coerce_sample_value("   ") == ""
