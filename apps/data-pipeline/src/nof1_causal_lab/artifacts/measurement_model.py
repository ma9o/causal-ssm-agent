"""Measurement-model artifact models and validation."""

from __future__ import annotations

import ast
import re
from enum import StrEnum
from typing import Any, get_args

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    computed_field,
    field_validator,
    model_validator,
)

from nof1_causal_lab.flows import get_prefect_logger
from nof1_causal_lab.measurement_types import AggregationFunction, MeasurementDtype
from nof1_causal_lab.utils.aggregations import COMPUTED_RULE_FUNCTIONS
from nof1_causal_lab.utils.observation_semantics import (
    AnchorPolicy,
    IndicatorObservationSemantics,
    SummaryOperator,
    SupportKind,
    derive_indicator_observation_semantics,
    supported_summary_operators_text,
)

from .duration import parse_duration_to_hours
from .latent_model import LatentModel  # noqa: TC001

logger = get_prefect_logger(__name__)

VALID_AGGREGATIONS: set[str] = set(get_args(AggregationFunction))
VALID_MEASUREMENT_DTYPES: set[str] = set(get_args(MeasurementDtype))

_SEMANTIC_COLLISIONS: list[tuple[str, set[str], str]] = [
    (
        r"\bcount\b|\bnumber of\b|\bhow many\b",
        {"mean", "median", "std", "var"},
        "how_to_measure implies counting but aggregation computes a statistic",
    ),
    (
        r"\baverage\b|\bmean\b",
        {"sum", "first", "last", "count"},
        "how_to_measure implies averaging but aggregation is not mean/median",
    ),
    (
        r"\btotal\b|\bcumulative\b|\bsum\b",
        {"mean", "median", "first", "last"},
        "how_to_measure implies summing but aggregation is not sum",
    ),
    (
        r"\blast\b|\bmost recent\b|\bcurrent\b",
        {"mean", "sum", "median"},
        "how_to_measure implies point-in-time but aggregation is a window statistic",
    ),
]


def check_semantic_collisions(
    how_to_measure: str,
    aggregation: str,
) -> list[str]:
    """Check for inconsistencies between how_to_measure text and aggregation."""
    warnings: list[str] = []
    text_lower = how_to_measure.lower()
    for pattern, conflict_aggs, explanation in _SEMANTIC_COLLISIONS:
        match = re.search(pattern, text_lower)
        if aggregation in conflict_aggs and match:
            warnings.append(
                f"Semantic collision: {explanation}. "
                f"how_to_measure contains '{match.group()}' "
                f"but aggregation='{aggregation}'."
            )
    return warnings


class IndicatorPolarity(StrEnum):
    """Direction of an indicator relative to its construct."""

    POSITIVE = "positive"
    NEGATIVE = "negative"


def _parse_computed_rule_expr(expr: str) -> ast.Expression:
    """Parse a computed-rule expression and surface a stable error."""
    try:
        parsed = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid computed_rule.window_expr: {exc.msg}") from exc
    return parsed


def _computed_rule_source_names(expr: str) -> set[str]:
    """Collect source-column references from a computed-rule expression."""
    parsed = _parse_computed_rule_expr(expr)
    names: set[str] = set()

    class _NameCollector(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            if not isinstance(node.func, ast.Name):
                raise ValueError("computed_rule.window_expr only supports simple function calls")
            if node.func.id not in COMPUTED_RULE_FUNCTIONS:
                available = ", ".join(sorted(COMPUTED_RULE_FUNCTIONS))
                raise ValueError(
                    f"Unsupported computed_rule function '{node.func.id}'. Available: {available}"
                )
            if node.keywords:
                raise ValueError("computed_rule.window_expr does not support keyword arguments")
            for arg in node.args:
                self.visit(arg)

        def visit_Attribute(self, node: ast.Attribute) -> Any:
            _ = node
            raise ValueError("computed_rule.window_expr does not support attribute access")

        def visit_Name(self, node: ast.Name) -> None:
            if node.id not in COMPUTED_RULE_FUNCTIONS:
                names.add(node.id)

    _NameCollector().visit(parsed.body)
    return names


class ComputedRule(BaseModel):
    """Deterministic per-window expression for computed indicators."""

    window_expr: str = Field(
        description=(
            "Deterministic support-window expression that returns one scalar per window. "
            "Use Python-like syntax over source_columns with arithmetic, comparisons, "
            "if/else, and helper functions such as any(), sum(), mean(), std(), "
            "first(), last(), count_true(), count_non_null(), lower(), contains(), "
            "and contains_any(). Use None for missing values."
        )
    )


class Indicator(BaseModel):
    """An observed variable that reflects a construct."""

    name: str = Field(description="Indicator name (e.g., 'hrv', 'self_reported_stress')")
    construct_name: str = Field(description="Which construct this indicator measures")
    how_to_measure: str = Field(
        description="Instructions for workers on how to extract this from data"
    )
    construct_polarity: IndicatorPolarity = Field(
        description=(
            "Whether higher indicator values move in the same direction as the construct "
            "(`positive`) or the opposite direction (`negative`)."
        )
    )
    measurement_dtype: str = Field(
        description="'continuous', 'binary', 'count', 'ordinal', 'categorical'"
    )
    aggregation: str = Field(
        description=(
            "Aggregation function applied when bucketing raw extractions within the "
            "indicator support window. Measurement-model support is currently limited to: "
            f"{supported_summary_operators_text()}. Available parser operators: {', '.join(sorted(VALID_AGGREGATIONS))}"
        ),
    )
    observation_window: str | None = Field(
        default=None,
        description=(
            "Optional duration string describing the support window summarized by this "
            "indicator (for example '1mo' for a monthly average on a daily model clock). "
            "If omitted, the support window defaults to the global model_clock."
        ),
    )
    ordinal_levels: list[str] | None = Field(
        default=None,
        description=(
            "Ordered list of level labels from lowest to highest for ordinal indicators "
            "(e.g., ['low', 'medium', 'high']). Required when measurement_dtype='ordinal' "
            "to ensure correct numeric encoding."
        ),
    )
    source_columns: list[str] = Field(
        default_factory=list,
        description=(
            "Raw data column names referenced by how_to_measure. "
            "Used to project chunks to only relevant columns before extraction."
        ),
    )
    computed_rule: ComputedRule | None = Field(
        default=None,
        description=(
            "Optional deterministic support-window expression for extraction_mode='computed'. "
            "Use this when a computed indicator needs formulas, thresholds, or multiple "
            "source columns instead of a direct single-column aggregation. "
            "The expression must return one scalar per support window."
        ),
    )
    extraction_mode: str = Field(
        default="semantic",
        description=(
            "'computed' (deterministic pipeline extraction) or 'semantic' (LLM extraction). "
            "Use 'computed' when the indicator can be derived deterministically either from "
            "a direct source-column aggregation or from a computed_rule support-window "
            "expression over the declared source_columns."
        ),
    )

    @field_validator("extraction_mode")
    @classmethod
    def validate_extraction_mode(cls, value: str) -> str:
        if value not in ("computed", "semantic"):
            raise ValueError(f"extraction_mode must be 'computed' or 'semantic', got '{value}'")
        return value

    @field_validator("aggregation")
    @classmethod
    def validate_aggregation(cls, value: str) -> str:
        if value not in VALID_AGGREGATIONS:
            available = ", ".join(sorted(VALID_AGGREGATIONS))
            raise ValueError(f"Unknown aggregation '{value}'. Available: {available}")
        return value

    @field_validator("observation_window")
    @classmethod
    def validate_observation_window(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parse_duration_to_hours(value)
        return value

    @field_validator("measurement_dtype")
    @classmethod
    def validate_measurement_dtype(cls, value: str) -> str:
        if value not in VALID_MEASUREMENT_DTYPES:
            raise ValueError(
                f"Invalid measurement_dtype '{value}'. Must be one of: {', '.join(sorted(VALID_MEASUREMENT_DTYPES))}"
            )
        return value

    @field_validator("computed_rule")
    @classmethod
    def validate_computed_rule(cls, value: ComputedRule | None) -> ComputedRule | None:
        if value is None:
            return None
        _computed_rule_source_names(value.window_expr)
        return value

    @model_validator(mode="after")
    def validate_ordinal_levels(self) -> Indicator:
        """Ensure ordinal_levels is valid when measurement_dtype is 'ordinal'."""
        if self.measurement_dtype == "ordinal":
            if not self.ordinal_levels:
                raise ValueError(
                    "ordinal_levels is required when measurement_dtype='ordinal' "
                    "(provide at least 2 ordered level labels)"
                )
            if len(self.ordinal_levels) < 2:
                raise ValueError(
                    f"ordinal_levels must have at least 2 items, got {len(self.ordinal_levels)}"
                )
            if len(self.ordinal_levels) != len(set(self.ordinal_levels)):
                raise ValueError("ordinal_levels must not contain duplicate labels")
        return self

    @model_validator(mode="after")
    def warn_semantic_collisions(self) -> Indicator:
        """Log warnings when how_to_measure text conflicts with aggregation."""
        collisions = check_semantic_collisions(self.how_to_measure, self.aggregation)
        for warning in collisions:
            logger.warning("Indicator '%s': %s", self.name, warning)
        return self

    @model_validator(mode="after")
    def validate_computed_mode(self) -> Indicator:
        """Enforce constraints when extraction_mode='computed'."""
        if self.computed_rule is not None and self.extraction_mode != "computed":
            raise ValueError(
                f"Indicator '{self.name}' sets computed_rule but extraction_mode is "
                f"'{self.extraction_mode}'. computed_rule is only valid for "
                "extraction_mode='computed'."
            )
        if self.extraction_mode != "computed":
            return self
        if self.computed_rule is None and len(self.source_columns) != 1:
            raise ValueError(
                f"Computed indicator '{self.name}' currently requires exactly 1 direct "
                f"source_column, got {len(self.source_columns)}: {self.source_columns}"
            )
        if self.computed_rule is not None:
            if not self.source_columns:
                raise ValueError(
                    f"Computed indicator '{self.name}' with computed_rule must declare "
                    "at least 1 source_column."
                )
            referenced = _computed_rule_source_names(self.computed_rule.window_expr)
            if not referenced:
                raise ValueError(
                    f"Computed indicator '{self.name}' has computed_rule.window_expr "
                    "that does not reference any source_columns."
                )
            unknown = sorted(referenced - set(self.source_columns))
            if unknown:
                raise ValueError(
                    f"Computed indicator '{self.name}' computed_rule.window_expr "
                    f"references undeclared source_columns: {unknown}. "
                    f"Declared source_columns: {self.source_columns}"
                )
        return self

    @model_validator(mode="after")
    def validate_observation_semantics(self) -> Indicator:
        """Reject aggregation/dtype combinations the measurement stack cannot model."""
        derive_indicator_observation_semantics(self.aggregation, self.measurement_dtype)
        return self

    def _observation_semantics(self) -> IndicatorObservationSemantics:
        return derive_indicator_observation_semantics(self.aggregation, self.measurement_dtype)

    @computed_field
    @property
    def support_kind(self) -> SupportKind:
        """Whether this indicator is point-local or interval-summary."""
        return self._observation_semantics().support_kind

    @computed_field
    @property
    def summary_operator(self) -> SummaryOperator:
        """Canonical summary operator used by extraction and likelihoods."""
        return self._observation_semantics().summary_operator

    @computed_field
    @property
    def anchor_policy(self) -> AnchorPolicy:
        """Which support boundary receives the observation anchor."""
        return self._observation_semantics().anchor_policy

    @property
    def requires_interval_summary_measurement(self) -> bool:
        """Whether this indicator requires an interval-summary measurement equation."""
        return self.support_kind == SupportKind.INTERVAL


class MeasurementModel(BaseModel):
    """Operationalization of constructs into observed indicators."""

    indicators: list[Indicator] = Field(
        description="Observed indicators, each measuring a construct"
    )
    model_clock: str = Field(
        description=(
            "Observation window width for extraction and SSM discretization. "
            "Any Polars-compatible duration string (e.g. '1h', '4h', '1d', '1w'). "
            "Choose based on data density: need enough events per support window."
        )
    )

    @field_validator("model_clock")
    @classmethod
    def validate_model_clock(cls, value: str) -> str:
        parse_duration_to_hours(value)
        return value

    @property
    def model_clock_hours(self) -> float:
        return parse_duration_to_hours(self.model_clock)

    @property
    def model_clock_days(self) -> float:
        return self.model_clock_hours / 24.0

    def get_indicators_for_construct(self, construct_name: str) -> list[Indicator]:
        return [
            indicator for indicator in self.indicators if indicator.construct_name == construct_name
        ]


def validate_measurement_model(
    data: dict,
    latent: LatentModel,
) -> tuple[MeasurementModel | None, list[str]]:
    """Validate a measurement model dict against a latent model."""
    errors: list[str] = []

    if not isinstance(data, dict):
        return None, ["Input must be a dictionary"]

    indicators = data.get("indicators", [])
    if not isinstance(indicators, list):
        errors.append("'indicators' must be a list")
        indicators = []

    model_clock = data.get("model_clock")
    if model_clock is None:
        errors.append("'model_clock' is required")
    elif not isinstance(model_clock, str):
        errors.append("'model_clock' must be a string")
        model_clock = None
    else:
        try:
            parse_duration_to_hours(model_clock)
        except ValueError as exc:
            errors.append(f"model_clock: {exc}")
            model_clock = None

    construct_names = {construct.name for construct in latent.constructs}
    valid_indicators: list[Indicator] = []
    indicator_names: set[str] = set()

    for index, indicator_data in enumerate(indicators):
        if not isinstance(indicator_data, dict):
            errors.append(f"indicators[{index}]: must be a dictionary")
            continue

        name = indicator_data.get("name", f"<unnamed_{index}>")
        if name in indicator_names:
            errors.append(f"Duplicate indicator name: '{name}'")
        indicator_names.add(name)

        try:
            indicator = Indicator.model_validate(indicator_data)
        except ValidationError as exc:
            error_msg = str(exc)
            if "validation error" in error_msg.lower():
                for line in error_msg.split("\n")[1:]:
                    line = line.strip()
                    if line and not line.startswith("For further"):
                        errors.append(f"indicators[{index}] ({name}): {line}")
            else:
                errors.append(f"indicators[{index}] ({name}): {error_msg}")
            continue

        if indicator.construct_name not in construct_names:
            errors.append(
                f"indicators[{index}] ({name}): references unknown construct '{indicator.construct_name}'"
            )
            continue

        valid_indicators.append(indicator)

    if not errors:
        try:
            if model_clock is None:
                errors.append("Measurement model is missing required model_clock")
                return None, errors
            model = MeasurementModel(indicators=valid_indicators, model_clock=model_clock)
            return model, []
        except ValidationError as exc:
            errors.append(f"Final validation failed: {exc}")

    return None, errors


__all__ = [
    "ComputedRule",
    "Indicator",
    "IndicatorPolarity",
    "MeasurementModel",
    "check_semantic_collisions",
    "validate_measurement_model",
]
