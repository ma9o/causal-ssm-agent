"""Tests for LLM tool functions (pure logic, no API calls).

Covers: calculate, parse_date tool executors.
"""

import asyncio

import pytest

from causal_ssm_agent.utils.llm import calculate, parse_date


def _run(coro):
    """Run an async function synchronously."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    return loop.run_until_complete(coro)


@pytest.fixture
def calc():
    """Get the calculate tool executor."""
    tool_instance = calculate()
    return tool_instance


@pytest.fixture
def date_parser():
    """Get the parse_date tool executor."""
    tool_instance = parse_date()
    return tool_instance


# =============================================================================
# calculate
# =============================================================================


class TestCalculate:
    def test_addition(self, calc):
        result = _run(calc("2 + 3"))
        assert result == "5"

    def test_multiplication(self, calc):
        result = _run(calc("6 * 7"))
        assert result == "42"

    def test_division(self, calc):
        result = _run(calc("100 / 4"))
        assert result == "25.0"

    def test_modulo(self, calc):
        result = _run(calc("10 % 3"))
        assert result == "1"

    def test_power(self, calc):
        result = _run(calc("2 ** 8"))
        assert result == "256"

    def test_parentheses(self, calc):
        result = _run(calc("(10 + 5) * 2"))
        assert result == "30"

    def test_complex_expression(self, calc):
        result = _run(calc("3.14 * 2"))
        assert result == "6.28"

    def test_invalid_chars_rejected(self, calc):
        result = _run(calc("import os"))
        assert "Error" in result
        assert "invalid characters" in result

    def test_semicolon_rejected(self, calc):
        result = _run(calc("1; 2"))
        assert "Error" in result

    def test_division_by_zero(self, calc):
        result = _run(calc("1 / 0"))
        assert "Error" in result

    def test_empty_expression(self, calc):
        result = _run(calc(""))
        assert "Error" in result


# =============================================================================
# parse_date
# =============================================================================


class TestParseDate:
    def test_iso_date(self, date_parser):
        result = _run(date_parser("2024-03-15"))
        assert "March" in result
        assert "2024" in result
        assert "15" in result

    def test_iso_datetime(self, date_parser):
        result = _run(date_parser("2024-03-15T10:30:00"))
        assert "March" in result
        assert "2024" in result

    def test_iso_datetime_with_z(self, date_parser):
        result = _run(date_parser("2024-03-15T10:30:00Z"))
        assert "March" in result

    def test_slash_format(self, date_parser):
        result = _run(date_parser("2024/03/15"))
        assert "March" in result

    def test_day_month_year(self, date_parser):
        result = _run(date_parser("15-03-2024"))
        assert "March" in result
        assert "2024" in result

    def test_unparseable_date(self, date_parser):
        result = _run(date_parser("not-a-date"))
        assert "Could not parse" in result

    def test_whitespace_trimmed(self, date_parser):
        result = _run(date_parser("  2024-03-15  "))
        assert "March" in result

    def test_day_name_included(self, date_parser):
        # 2024-03-15 was a Friday
        result = _run(date_parser("2024-03-15"))
        assert "Friday" in result

    def test_datetime_with_milliseconds(self, date_parser):
        result = _run(date_parser("2024-03-15T10:30:00.123"))
        assert "March" in result
