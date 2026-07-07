"""Shared indicator observation semantics.

This module is the single source of truth for how an indicator's aggregation
maps to downstream measurement semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

SUPPORTED_SUMMARY_OPERATORS: tuple[str, ...] = ("first", "last", "sum", "count", "mean", "std")
_POINT_START_OPERATORS = frozenset({"first"})
_POINT_END_OPERATORS = frozenset({"last"})


class SupportKind(StrEnum):
    """Whether the measurement equation is point-local or interval-summary."""

    POINT = "point"
    INTERVAL = "interval"


class AnchorPolicy(StrEnum):
    """Which support boundary receives the observation anchor."""

    SUPPORT_START = "support_start"
    SUPPORT_END = "support_end"


class SummaryOperator(StrEnum):
    """Supported summary operators for indicator observations."""

    FIRST = "first"
    LAST = "last"
    SUM = "sum"
    COUNT = "count"
    MEAN = "mean"
    STD = "std"


@dataclass(frozen=True)
class IndicatorObservationSemantics:
    """Canonical semantics derived from one indicator definition."""

    support_kind: SupportKind
    summary_operator: SummaryOperator
    anchor_policy: AnchorPolicy


def supported_summary_operators_text() -> str:
    """Return a stable human-readable list of supported operators."""
    return ", ".join(SUPPORTED_SUMMARY_OPERATORS)


def _coerce_summary_operator(aggregation: str) -> SummaryOperator:
    try:
        return SummaryOperator(aggregation)
    except ValueError as exc:
        raise ValueError(
            "aggregation "
            f"'{aggregation}' is not yet supported by the measurement structure. "
            f"Supported operators: {supported_summary_operators_text()}."
        ) from exc


def validate_indicator_observation_semantics(
    aggregation: str,
    measurement_dtype: str,
) -> str | None:
    """Return a user-facing validation error for unsupported semantics."""
    if aggregation not in SUPPORTED_SUMMARY_OPERATORS:
        return (
            f"aggregation '{aggregation}' is not yet supported by the measurement structure. "
            f"Supported operators: {supported_summary_operators_text()}."
        )

    if measurement_dtype == "ordinal" and aggregation not in _POINT_START_OPERATORS.union(
        _POINT_END_OPERATORS
    ):
        return "ordinal indicators currently support only first/last point measurements."

    if aggregation == SummaryOperator.COUNT.value and measurement_dtype != "count":
        return "aggregation 'count' requires measurement_dtype='count'."

    if (
        aggregation in {SummaryOperator.MEAN.value, SummaryOperator.STD.value}
        and measurement_dtype != "continuous"
    ):
        return f"aggregation '{aggregation}' requires measurement_dtype='continuous'."

    if aggregation == SummaryOperator.SUM.value and measurement_dtype not in {
        "continuous",
        "count",
    }:
        return "aggregation 'sum' requires measurement_dtype='continuous' or 'count'."

    return None


def derive_indicator_observation_semantics(
    aggregation: str,
    measurement_dtype: str,
) -> IndicatorObservationSemantics:
    """Derive canonical semantics from aggregation and measurement dtype."""
    error = validate_indicator_observation_semantics(aggregation, measurement_dtype)
    if error is not None:
        raise ValueError(error)

    summary_operator = _coerce_summary_operator(aggregation)
    if summary_operator in {SummaryOperator.FIRST, SummaryOperator.LAST}:
        support_kind = SupportKind.POINT
    elif summary_operator in {
        SummaryOperator.SUM,
        SummaryOperator.COUNT,
        SummaryOperator.MEAN,
        SummaryOperator.STD,
    }:
        support_kind = SupportKind.INTERVAL
    else:
        raise ValueError(
            "Unhandled summary operator "
            f"'{summary_operator.value}'. Supported operators: {supported_summary_operators_text()}."
        )

    anchor_policy = (
        AnchorPolicy.SUPPORT_START
        if summary_operator == SummaryOperator.FIRST
        else AnchorPolicy.SUPPORT_END
    )
    return IndicatorObservationSemantics(
        support_kind=support_kind,
        summary_operator=summary_operator,
        anchor_policy=anchor_policy,
    )


def get_observation_semantics(indicator: dict) -> IndicatorObservationSemantics:
    """Derive canonical observation semantics for an indicator dict."""
    return derive_indicator_observation_semantics(
        indicator["aggregation"],
        indicator["measurement_dtype"],
    )
