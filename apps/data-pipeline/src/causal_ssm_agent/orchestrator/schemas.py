"""Causal model schemas following Anderson & Gerbing two-step approach.

Separates:
1. LatentModel - theoretical constructs + causal edges (theory-driven)
2. MeasurementModel - observed indicators that reflect constructs (data-driven)
"""

import ast
import re
from enum import StrEnum
from typing import Literal, get_args

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.models.ssm.schemas_inference import AggregationFunction, MeasurementDtype
from causal_ssm_agent.utils.observation_semantics import (
    AnchorPolicy,
    IndicatorObservationSemantics,
    SummaryOperator,
    SupportKind,
    derive_indicator_observation_semantics,
    supported_summary_operators_text,
)

logger = get_prefect_logger(__name__)

# Derived from the canonical Literal types in schemas_inference.py
VALID_AGGREGATIONS: set[str] = set(get_args(AggregationFunction))
VALID_MEASUREMENT_DTYPES: set[str] = set(get_args(MeasurementDtype))

# Aggregation keywords that conflict with how_to_measure text
_SEMANTIC_COLLISIONS: list[tuple[str, set[str], str]] = [
    # (regex pattern in how_to_measure, conflicting aggregations, explanation)
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
    """Check for inconsistencies between how_to_measure text and aggregation.

    Returns list of warning messages (empty if no collisions found).
    """
    warnings = []
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


class Role(StrEnum):
    """Whether a variable is modeled (endogenous) or given (exogenous)."""

    ENDOGENOUS = "endogenous"  # Has inbound edges, is modeled
    EXOGENOUS = "exogenous"  # No inbound edges, given/external


class TemporalStatus(StrEnum):
    """Whether a variable changes over time."""

    TIME_VARYING = "time_varying"  # Changes within person over time
    TIME_INVARIANT = "time_invariant"  # Fixed for each person


_COMPUTED_RULE_FUNCTIONS = {
    "abs",
    "all",
    "any",
    "coalesce",
    "contains",
    "contains_any",
    "count_non_null",
    "count_true",
    "first",
    "last",
    "lower",
    "max",
    "mean",
    "min",
    "std",
    "sum",
}


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
            if node.func.id not in _COMPUTED_RULE_FUNCTIONS:
                available = ", ".join(sorted(_COMPUTED_RULE_FUNCTIONS))
                raise ValueError(
                    f"Unsupported computed_rule function '{node.func.id}'. Available: {available}"
                )
            if node.keywords:
                raise ValueError("computed_rule.window_expr does not support keyword arguments")
            for arg in node.args:
                self.visit(arg)

        def visit_Attribute(self, _node: ast.Attribute) -> None:
            raise ValueError("computed_rule.window_expr does not support attribute access")

        def visit_Name(self, node: ast.Name) -> None:
            if node.id not in _COMPUTED_RULE_FUNCTIONS:
                names.add(node.id)

    _NameCollector().visit(parsed.body)
    return names


# ══════════════════════════════════════════════════════════════════════════════
# DURATION PARSING (Polars-compatible duration strings)
# ══════════════════════════════════════════════════════════════════════════════

# Polars duration units → hours conversion factor
_DURATION_UNIT_HOURS: dict[str, float] = {
    "s": 1 / 3600,
    "m": 1 / 60,
    "h": 1.0,
    "d": 24.0,
    "w": 168.0,
    "mo": 720.0,  # 30 days
    "q": 2160.0,  # 90 days
    "y": 8760.0,  # 365 days
}

# Match <integer><unit> where unit is a Polars duration suffix
_DURATION_RE = re.compile(r"^(\d+)(s|m|h|d|w|mo|q|y)$")


def parse_duration_to_hours(duration: str) -> float:
    """Parse a Polars-compatible duration string to hours.

    Accepts any string of the form ``<int><unit>`` where unit is one of:
    s (seconds), m (minutes), h (hours), d (days), w (weeks),
    mo (months/30d), q (quarters/90d), y (years/365d).

    >>> parse_duration_to_hours("1d")
    24.0
    >>> parse_duration_to_hours("4h")
    4.0
    >>> parse_duration_to_hours("1w")
    168.0
    """
    match = _DURATION_RE.match(duration)
    if not match:
        raise ValueError(
            f"Invalid duration: {duration!r}. "
            f"Expected format: <int><unit> where unit is one of "
            f"{', '.join(_DURATION_UNIT_HOURS)}"
        )
    n = int(match.group(1))
    unit = match.group(2)
    if n == 0:
        raise ValueError("Duration must be positive (got 0)")
    return n * _DURATION_UNIT_HOURS[unit]


# ══════════════════════════════════════════════════════════════════════════════
# LATENT MODEL (theoretical - what exists and how it relates)
# ══════════════════════════════════════════════════════════════════════════════


class Construct(BaseModel):
    """A theoretical entity in the causal model.

    Constructs are conceptually 'latent' - they represent theoretical entities
    that may be measured by one or more observed indicators.
    """

    name: str = Field(description="Construct name (e.g., 'stress', 'sleep_quality')")
    description: str = Field(description="What this theoretical construct represents")
    role: Role = Field(description="'endogenous' (modeled) or 'exogenous' (given)")
    is_outcome: bool = Field(
        default=False,
        description="True if this is the primary outcome variable Y implied by the question",
    )
    temporal_status: TemporalStatus = Field(
        description="'time_varying' (changes over time) or 'time_invariant' (fixed)"
    )

    @model_validator(mode="after")
    def validate_construct(self):
        """Validate construct field consistency."""
        # Outcomes must be endogenous
        if self.is_outcome and self.role != Role.ENDOGENOUS:
            raise ValueError(
                f"Outcome construct '{self.name}' must be endogenous, got {self.role.value}"
            )

        return self


class CausalEdge(BaseModel):
    """A directed causal relationship between constructs."""

    cause: str = Field(description="Name of cause construct")
    effect: str = Field(description="Name of effect construct")
    description: str = Field(description="Theoretical justification for this causal link")
    lagged: bool = Field(
        default=True,
        description=(
            "If True, effect at t is caused by cause at t-1 (one model_clock tick delay). "
            "If False (contemporaneous), effect at t is caused by cause at t."
        ),
    )


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


# ══════════════════════════════════════════════════════════════════════════════
# SHARED VALIDATION PREDICATES
# Used by both the Pydantic model validator (raise-on-first) and the
# validate_latent_model() function (collect-all-errors).
# ══════════════════════════════════════════════════════════════════════════════


def _check_edge_constraint(
    edge: "CausalEdge",
    construct_map: dict[str, "Construct"],
) -> str | None:
    """Check a single edge against cross-cutting constraints.

    Returns an error message string if violated, or None if valid.
    Assumes edge endpoints have already been checked for existence.
    """
    cause_construct = construct_map[edge.cause]
    effect_construct = construct_map[edge.effect]

    if effect_construct.role == Role.EXOGENOUS:
        return f"Exogenous construct '{edge.effect}' cannot be an effect"

    if (
        cause_construct.temporal_status == TemporalStatus.TIME_VARYING
        and effect_construct.temporal_status == TemporalStatus.TIME_INVARIANT
    ):
        return (
            f"Time-varying construct '{edge.cause}' cannot be a cause of "
            f"time-invariant construct '{edge.effect}'. Time-invariant constructs "
            "are fixed within person and cannot have time-varying parents."
        )

    both_time_varying = (
        cause_construct.temporal_status == TemporalStatus.TIME_VARYING
        and effect_construct.temporal_status == TemporalStatus.TIME_VARYING
    )

    both_endogenous = (
        cause_construct.role == Role.ENDOGENOUS and effect_construct.role == Role.ENDOGENOUS
    )
    if not edge.lagged and both_time_varying and both_endogenous:
        return (
            f"Directed contemporaneous edge '{edge.cause}' -> '{edge.effect}' "
            "between endogenous time-varying latent constructs is excluded by the "
            "latent-model contract. Represent directed effects between evolving "
            "latent states with lagged=True; reserve same-time dependence for "
            "explicit confounding or diffusion covariance."
        )

    return None


def _check_global_constraints(
    constructs: list["Construct"],
    edges: list["CausalEdge"],
) -> list[str]:
    """Check global constraints across all constructs and edges.

    Returns a list of error messages (empty if all valid).
    """
    errors = []

    # Exactly one outcome required
    outcomes = [c for c in constructs if c.is_outcome]
    if len(outcomes) == 0:
        errors.append("Exactly one construct must have is_outcome=true")
    elif len(outcomes) > 1:
        names = [c.name for c in outcomes]
        errors.append(f"Only one outcome allowed, got {len(outcomes)}: {names}")

    # Outcome must have at least one incoming edge
    if len(outcomes) == 1:
        outcome_name = outcomes[0].name
        incoming_to_outcome = [e for e in edges if e.effect == outcome_name]
        if not incoming_to_outcome:
            errors.append(
                f"Outcome construct '{outcome_name}' has no incoming causal edges. "
                "The model must include at least one cause of the outcome."
            )

    # Check acyclicity within time slice (contemporaneous edges only)
    contemporaneous_edges = [(e.cause, e.effect) for e in edges if not e.lagged]
    if contemporaneous_edges:
        import networkx as nx

        G = nx.DiGraph(contemporaneous_edges)
        if not nx.is_directed_acyclic_graph(G):
            cycles = list(nx.simple_cycles(G))
            errors.append(
                f"Contemporaneous edges form cycle(s) within time slice: {cycles}. "
                "Use lagged=true for feedback loops across time."
            )

    return errors


class LatentModel(BaseModel):
    """Theoretical causal structure over constructs (the latent model).

    This is the output of Stage 1a - proposed based on domain knowledge alone,
    without seeing data. Defines the topological structure among latent constructs.
    """

    constructs: list[Construct] = Field(description="Theoretical constructs in the model")
    edges: list[CausalEdge] = Field(description="Causal edges between constructs")

    @model_validator(mode="after")
    def validate_latent_model(self):
        """Validate latent model constraints."""
        construct_map = {c.name: c for c in self.constructs}

        for edge in self.edges:
            if edge.cause not in construct_map:
                raise ValueError(f"Edge cause '{edge.cause}' not in constructs")
            if edge.effect not in construct_map:
                raise ValueError(f"Edge effect '{edge.effect}' not in constructs")

            error = _check_edge_constraint(edge, construct_map)
            if error:
                raise ValueError(error)

        global_errors = _check_global_constraints(self.constructs, self.edges)
        if global_errors:
            raise ValueError(global_errors[0])

        return self


# ══════════════════════════════════════════════════════════════════════════════
# MEASUREMENT MODEL (operational - how constructs are observed)
# ══════════════════════════════════════════════════════════════════════════════


class Indicator(BaseModel):
    """An observed variable that reflects a construct.

    Following the reflective measurement model (A1), causality flows from
    construct to indicator: the latent construct causes the observed values.
    """

    name: str = Field(description="Indicator name (e.g., 'hrv', 'self_reported_stress')")
    construct_name: str = Field(
        description="Which construct this indicator measures",
    )
    how_to_measure: str = Field(
        description="Instructions for workers on how to extract this from data"
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
    computed_rule: "ComputedRule | None" = Field(
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
    def validate_extraction_mode(cls, v: str) -> str:
        if v not in ("computed", "semantic"):
            raise ValueError(f"extraction_mode must be 'computed' or 'semantic', got '{v}'")
        return v

    @field_validator("aggregation")
    @classmethod
    def validate_aggregation(cls, v: str) -> str:
        if v not in VALID_AGGREGATIONS:
            available = ", ".join(sorted(VALID_AGGREGATIONS))
            raise ValueError(f"Unknown aggregation '{v}'. Available: {available}")
        return v

    @field_validator("observation_window")
    @classmethod
    def validate_observation_window(cls, v: str | None) -> str | None:
        if v is None:
            return None
        parse_duration_to_hours(v)
        return v

    @field_validator("measurement_dtype")
    @classmethod
    def validate_measurement_dtype(cls, v: str) -> str:
        if v not in VALID_MEASUREMENT_DTYPES:
            raise ValueError(
                f"Invalid measurement_dtype '{v}'. Must be one of: {', '.join(sorted(VALID_MEASUREMENT_DTYPES))}"
            )
        return v

    @field_validator("computed_rule")
    @classmethod
    def validate_computed_rule(cls, v: "ComputedRule | None") -> "ComputedRule | None":
        if v is None:
            return None
        _computed_rule_source_names(v.window_expr)
        return v

    @model_validator(mode="after")
    def validate_ordinal_levels(self) -> "Indicator":
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
    def warn_semantic_collisions(self) -> "Indicator":
        """Log warnings when how_to_measure text conflicts with aggregation."""
        collisions = check_semantic_collisions(self.how_to_measure, self.aggregation)
        for warning in collisions:
            logger.warning("Indicator '%s': %s", self.name, warning)
        return self

    @model_validator(mode="after")
    def validate_computed_mode(self) -> "Indicator":
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
                f"source_column, "
                f"got {len(self.source_columns)}: {self.source_columns}"
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
    def validate_observation_semantics(self) -> "Indicator":
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
    """Operationalization of constructs into observed indicators.

    This is the output of Stage 1b - proposed after seeing data sample,
    given the latent model from Stage 1a.

    Each construct from the latent model must have at least one indicator.
    """

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
    def validate_model_clock(cls, v: str) -> str:
        """Check that model_clock is a valid Polars duration string."""
        parse_duration_to_hours(v)
        return v

    @property
    def model_clock_hours(self) -> float:
        """Model clock duration in hours."""
        return parse_duration_to_hours(self.model_clock)

    @property
    def model_clock_days(self) -> float:
        """Model clock duration in fractional days (for SSM discretization)."""
        return self.model_clock_hours / 24.0

    def get_indicators_for_construct(self, construct_name: str) -> list[Indicator]:
        """Get all indicators that measure a given construct."""
        return [i for i in self.indicators if i.construct_name == construct_name]


# ══════════════════════════════════════════════════════════════════════════════
# CAUSAL SPEC (composition of latent + measurement)
# ══════════════════════════════════════════════════════════════════════════════


class IdentifiedTreatmentStatus(BaseModel):
    """Details on how a treatment effect is identified."""

    method: str = Field(
        description="Identification strategy (e.g., do_calculus, instrumental_variable)"
    )
    estimand: str = Field(description="Closed-form estimand or IV placeholder")
    marginalized_confounders: list[str] = Field(
        default_factory=list,
        description="Unobserved confounders the estimand integrates out",
    )
    instruments: list[str] = Field(
        default_factory=list,
        description="Instrumental variables used (if method=instrumental_variable)",
    )


class NonIdentifiableTreatmentStatus(BaseModel):
    """Context on why a treatment effect is not identifiable."""

    confounders: list[str] = Field(
        default_factory=list,
        description="Unobserved constructs blocking identification",
    )
    notes: str | None = Field(
        default=None,
        description="Optional explanation if confounders cannot be enumerated",
    )


class IdentifiabilityStatus(BaseModel):
    """Status of causal effect identifiability."""

    identifiable_treatments: dict[str, IdentifiedTreatmentStatus] = Field(
        default_factory=dict,
        description="Treatments with identifiable effects and how to estimate them",
    )
    non_identifiable_treatments: dict[str, NonIdentifiableTreatmentStatus] = Field(
        default_factory=dict,
        description="Treatments whose effects are currently not identifiable",
    )


class InducedDependency(BaseModel):
    """Dependence induced among retained states after marginalizing latent roots."""

    between: tuple[str, str] = Field(
        description="Pair of retained states whose joint dependence is induced"
    )
    kind: Literal["innovation_correlation", "initial_state_correlation"] = Field(
        description="Which covariance block the induced dependence belongs to"
    )
    source_confounders: list[str] = Field(
        default_factory=list,
        description="Marginalized source constructs that induce this dependence",
    )


class EstimationSpec(BaseModel):
    """Deterministic estimation-time projection of the user-facing latent DAG."""

    state_order: list[str] = Field(
        description="Retained latent states in canonical array order for compilation"
    )
    edges: list[CausalEdge] = Field(
        default_factory=list,
        description="Directed estimation graph over retained states",
    )
    induced_dependencies: list[InducedDependency] = Field(
        default_factory=list,
        description="Dependencies induced after marginalizing latent root confounders",
    )


class CausalSpec(BaseModel):
    """Complete causal specification combining latent and measurement models.

    This is the full model after both Stage 1a (latent) and Stage 1b (measurement).
    Includes identifiability status for target causal effects.
    """

    latent: LatentModel = Field(description="Theoretical causal structure (topological)")
    measurement: MeasurementModel = Field(description="Operationalization into indicators")
    identifiability: IdentifiabilityStatus | None = Field(
        default=None, description="Identifiability status of target causal effects"
    )
    estimation: EstimationSpec | None = Field(
        default=None,
        description="Deterministic estimation-time projection consumed by downstream fitting",
    )

    @model_validator(mode="after")
    def validate_causal_spec(self):
        """Validate measurement model covers all constructs."""
        construct_names = {c.name for c in self.latent.constructs}

        # Check all indicator references are valid
        for indicator in self.measurement.indicators:
            if indicator.construct_name not in construct_names:
                raise ValueError(
                    f"Indicator '{indicator.name}' references unknown construct '{indicator.construct_name}'"
                )

        estimation = self.estimation
        if estimation is not None:
            if len(estimation.state_order) != len(set(estimation.state_order)):
                raise ValueError("Estimation state_order contains duplicate construct names")

            state_names = set(estimation.state_order)
            unknown_states = state_names - construct_names
            if unknown_states:
                raise ValueError(
                    "Estimation state_order references unknown constructs: "
                    f"{sorted(unknown_states)}"
                )

            for edge in estimation.edges:
                if edge.cause not in state_names or edge.effect not in state_names:
                    raise ValueError(
                        "Estimation edge must reference retained states: "
                        f"{edge.cause!r} -> {edge.effect!r}"
                    )

            for dependency in estimation.induced_dependencies:
                state_1, state_2 = dependency.between
                if state_1 not in state_names or state_2 not in state_names:
                    raise ValueError(
                        "Induced dependency must reference retained states: "
                        f"{dependency.between!r}"
                    )
                unknown_sources = set(dependency.source_confounders) - construct_names
                if unknown_sources:
                    raise ValueError(
                        "Induced dependency references unknown source confounders: "
                        f"{sorted(unknown_sources)}"
                    )

        return self

    def get_edge_lag_hours(self, edge: CausalEdge) -> float:
        """Compute lag in hours for a causal edge."""
        return self.measurement.model_clock_hours if edge.lagged else 0


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION FUNCTIONS (for LLM tool use)
# ══════════════════════════════════════════════════════════════════════════════


def validate_latent_model(data: dict) -> tuple[LatentModel | None, list[str]]:
    """Validate a latent model dict, collecting ALL errors.

    Args:
        data: Dictionary to validate as LatentModel

    Returns:
        Tuple of (validated model or None, list of error messages)
    """
    errors = []

    if not isinstance(data, dict):
        return None, ["Input must be a dictionary"]

    constructs = data.get("constructs", [])
    edges = data.get("edges", [])

    if not isinstance(constructs, list):
        errors.append("'constructs' must be a list")
        constructs = []
    if not isinstance(edges, list):
        errors.append("'edges' must be a list")
        edges = []

    # Validate each construct individually
    valid_constructs = []
    construct_names = set()

    for i, construct_data in enumerate(constructs):
        if not isinstance(construct_data, dict):
            errors.append(f"constructs[{i}]: must be a dictionary")
            continue

        name = construct_data.get("name", f"<unnamed_{i}>")

        if name in construct_names:
            errors.append(f"Duplicate construct name: '{name}'")
        construct_names.add(name)

        try:
            construct = Construct.model_validate(construct_data)
            valid_constructs.append(construct)
        except Exception as e:
            error_msg = str(e)
            if "validation error" in error_msg.lower():
                for line in error_msg.split("\n")[1:]:
                    line = line.strip()
                    if line and not line.startswith("For further"):
                        errors.append(f"constructs[{i}] ({name}): {line}")
            else:
                errors.append(f"constructs[{i}] ({name}): {error_msg}")

    # Build construct map for edge validation
    construct_map = {c.name: c for c in valid_constructs}

    # Validate each edge individually
    valid_edges = []
    for i, edge_data in enumerate(edges):
        if not isinstance(edge_data, dict):
            errors.append(f"edges[{i}]: must be a dictionary")
            continue

        cause = edge_data.get("cause", "<missing>")
        effect = edge_data.get("effect", "<missing>")
        edge_label = f"edges[{i}] ({cause} -> {effect})"

        try:
            edge = CausalEdge.model_validate(edge_data)
        except Exception as e:
            errors.append(f"{edge_label}: {e}")
            continue

        if edge.cause not in construct_map:
            errors.append(f"{edge_label}: cause '{edge.cause}' not in constructs")
            continue
        if edge.effect not in construct_map:
            errors.append(f"{edge_label}: effect '{edge.effect}' not in constructs")
            continue

        constraint_error = _check_edge_constraint(edge, construct_map)
        if constraint_error:
            errors.append(f"{edge_label}: {constraint_error}")
            continue

        valid_edges.append(edge)

    # Check global constraints (outcome count, incoming edges, acyclicity)
    errors.extend(_check_global_constraints(valid_constructs, valid_edges))

    if not errors:
        try:
            model = LatentModel(constructs=valid_constructs, edges=valid_edges)
            return model, []
        except Exception as e:
            errors.append(f"Final validation failed: {e}")

    return None, errors


def validate_measurement_model(
    data: dict,
    latent: LatentModel,
) -> tuple[MeasurementModel | None, list[str]]:
    """Validate a measurement model dict against a latent model.

    Args:
        data: Dictionary to validate as MeasurementModel
        latent: The latent model this measurement model operationalizes

    Returns:
        Tuple of (validated model or None, list of error messages)
    """
    errors = []

    if not isinstance(data, dict):
        return None, ["Input must be a dictionary"]

    indicators = data.get("indicators", [])

    if not isinstance(indicators, list):
        errors.append("'indicators' must be a list")
        indicators = []

    # Validate model_clock
    model_clock = data.get("model_clock")
    if model_clock is None:
        errors.append("'model_clock' is required")
    elif not isinstance(model_clock, str):
        errors.append("'model_clock' must be a string")
        model_clock = None
    else:
        try:
            parse_duration_to_hours(model_clock)
        except ValueError as e:
            errors.append(f"model_clock: {e}")
            model_clock = None

    construct_names = {c.name for c in latent.constructs}

    # Validate each indicator
    valid_indicators = []
    indicator_names = set()

    for i, indicator_data in enumerate(indicators):
        if not isinstance(indicator_data, dict):
            errors.append(f"indicators[{i}]: must be a dictionary")
            continue

        name = indicator_data.get("name", f"<unnamed_{i}>")

        if name in indicator_names:
            errors.append(f"Duplicate indicator name: '{name}'")
        indicator_names.add(name)

        try:
            indicator = Indicator.model_validate(indicator_data)
        except Exception as e:
            error_msg = str(e)
            if "validation error" in error_msg.lower():
                for line in error_msg.split("\n")[1:]:
                    line = line.strip()
                    if line and not line.startswith("For further"):
                        errors.append(f"indicators[{i}] ({name}): {line}")
            else:
                errors.append(f"indicators[{i}] ({name}): {error_msg}")
            continue

        # Check construct reference
        if indicator.construct_name not in construct_names:
            errors.append(
                f"indicators[{i}] ({name}): references unknown construct '{indicator.construct_name}'"
            )
            continue

        valid_indicators.append(indicator)

    if not errors:
        try:
            model = MeasurementModel(indicators=valid_indicators, model_clock=model_clock)
            return model, []
        except Exception as e:
            errors.append(f"Final validation failed: {e}")

    return None, errors


def validate_causal_spec(
    latent_data: dict,
    measurement_data: dict,
) -> tuple[CausalSpec | None, list[str]]:
    """Validate both latent and measurement models together.

    Args:
        latent_data: Dictionary to validate as LatentModel
        measurement_data: Dictionary to validate as MeasurementModel

    Returns:
        Tuple of (validated CausalSpec or None, list of error messages)
    """
    latent, latent_errors = validate_latent_model(latent_data)
    if latent is None:
        return None, ["Latent model errors:", *latent_errors]

    measurement, measurement_errors = validate_measurement_model(measurement_data, latent)
    if measurement is None:
        return None, ["Measurement model errors:", *measurement_errors]

    try:
        from causal_ssm_agent.utils.estimation_projection import build_estimation_projection

        latent_payload = latent.model_dump(mode="json")
        measurement_payload = measurement.model_dump(mode="json")
        model = CausalSpec(
            latent=latent,
            measurement=measurement,
            estimation=build_estimation_projection(
                latent_payload,
                measurement_payload,
                identifiability_result=None,
            ),
        )
        return model, []
    except Exception as e:
        return None, [f"CausalSpec validation failed: {e}"]
