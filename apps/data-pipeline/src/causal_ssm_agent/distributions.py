"""Central distribution catalog for observation models and priors."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Literal


class DistributionFamily(StrEnum):
    """Distribution families for observation and process noise."""

    GAUSSIAN = "gaussian"
    STUDENT_T = "student_t"
    POISSON = "poisson"
    GAMMA = "gamma"
    BERNOULLI = "bernoulli"
    NEGATIVE_BINOMIAL = "negative_binomial"
    BETA = "beta"
    ORDERED_LOGISTIC = "ordered_logistic"
    CATEGORICAL = "categorical"

    @property
    def is_discrete(self) -> bool:
        """Whether this family has discrete (integer) support."""
        return self in {
            DistributionFamily.BERNOULLI,
            DistributionFamily.POISSON,
            DistributionFamily.NEGATIVE_BINOMIAL,
            DistributionFamily.ORDERED_LOGISTIC,
            DistributionFamily.CATEGORICAL,
        }

    @property
    def support_interior_point(self) -> float:
        """A scalar strictly inside this family's support (for dummy observations)."""
        if self == DistributionFamily.GAMMA:
            return 1.0
        if self == DistributionFamily.BETA:
            return 0.5
        return 0.0


@dataclass(frozen=True)
class ObservationFamilySpec:
    """Central observation-family metadata shared across prompts and validation."""

    family: DistributionFamily
    summary: str
    links: tuple[str, ...]


class PriorDistributionFamily(StrEnum):
    """Distribution families allowed in Stage 4 prior proposals."""

    NORMAL = "Normal"
    HALF_NORMAL = "HalfNormal"
    BETA = "Beta"
    UNIFORM = "Uniform"
    TRUNCATED_NORMAL = "TruncatedNormal"
    GAMMA = "Gamma"
    LOG_NORMAL = "LogNormal"
    EXPONENTIAL = "Exponential"


class PriorRuntimeKind(StrEnum):
    """Executable encoding for prior families."""

    NORMAL = "normal"
    HALF_NORMAL = "half_normal"
    BETA = "beta"
    UNIFORM = "uniform"
    TRUNCATED_NORMAL = "truncated_normal"
    GAMMA = "gamma"
    LOG_NORMAL = "log_normal"
    EXPONENTIAL = "exponential"


@dataclass(frozen=True)
class PriorFamilySpec:
    """Central prior-family metadata shared across prompts, docs, and runtime."""

    family: PriorDistributionFamily
    signature: str
    summary: str
    support: Literal["real", "positive", "unit_interval", "bounded"]
    runtime_kind: PriorRuntimeKind


@dataclass(frozen=True)
class PriorConstraintGuidance:
    """Constraint-level prior guidance derived from the prior catalog."""

    constraint: str
    domain: str
    typical_families: str


@dataclass(frozen=True)
class PriorParameterGuidanceRow:
    """Parameter-level prior heuristics reused across Stage 4 prompts."""

    parameter_type: str
    typical_distribution: str
    typical_range: str
    scale: str


OBSERVATION_FAMILY_SPECS: Final[tuple[ObservationFamilySpec, ...]] = (
    ObservationFamilySpec(
        family=DistributionFamily.GAUSSIAN,
        summary="Continuous unbounded data, approximately symmetric.",
        links=("identity",),
    ),
    ObservationFamilySpec(
        family=DistributionFamily.STUDENT_T,
        summary="Continuous data with heavy tails or outliers.",
        links=("identity",),
    ),
    ObservationFamilySpec(
        family=DistributionFamily.POISSON,
        summary="Count data with variance roughly tracking the mean.",
        links=("log",),
    ),
    ObservationFamilySpec(
        family=DistributionFamily.GAMMA,
        summary="Positive continuous data such as durations or reaction times.",
        links=("log", "inverse"),
    ),
    ObservationFamilySpec(
        family=DistributionFamily.BERNOULLI,
        summary="Binary outcomes with two possible states.",
        links=("logit", "probit"),
    ),
    ObservationFamilySpec(
        family=DistributionFamily.NEGATIVE_BINOMIAL,
        summary="Overdispersed count data where variance exceeds the mean.",
        links=("log",),
    ),
    ObservationFamilySpec(
        family=DistributionFamily.BETA,
        summary="Proportions or rates strictly inside the unit interval.",
        links=("logit", "probit"),
    ),
    ObservationFamilySpec(
        family=DistributionFamily.ORDERED_LOGISTIC,
        summary="Ordered categorical outcomes with ranked levels.",
        links=("cumulative_logit",),
    ),
    ObservationFamilySpec(
        family=DistributionFamily.CATEGORICAL,
        summary="Unordered multi-class outcomes.",
        links=("softmax",),
    ),
)

PRIOR_FAMILY_SPECS: Final[tuple[PriorFamilySpec, ...]] = (
    PriorFamilySpec(
        family=PriorDistributionFamily.NORMAL,
        signature="Normal(mu, sigma)",
        summary="Unconstrained effects that can be positive or negative.",
        support="real",
        runtime_kind=PriorRuntimeKind.NORMAL,
    ),
    PriorFamilySpec(
        family=PriorDistributionFamily.HALF_NORMAL,
        signature="HalfNormal(sigma)",
        summary="Positive-only parameters such as standard deviations and scales.",
        support="positive",
        runtime_kind=PriorRuntimeKind.HALF_NORMAL,
    ),
    PriorFamilySpec(
        family=PriorDistributionFamily.BETA,
        signature="Beta(alpha, beta)",
        summary="Parameters constrained to the unit interval [0, 1].",
        support="unit_interval",
        runtime_kind=PriorRuntimeKind.BETA,
    ),
    PriorFamilySpec(
        family=PriorDistributionFamily.UNIFORM,
        signature="Uniform(lower, upper)",
        summary="Hard-bounded parameters when only plausible limits are known.",
        support="bounded",
        runtime_kind=PriorRuntimeKind.UNIFORM,
    ),
    PriorFamilySpec(
        family=PriorDistributionFamily.TRUNCATED_NORMAL,
        signature="TruncatedNormal(mu, sigma, lower, upper)",
        summary="Bounded parameters when both a center and hard limits are meaningful.",
        support="bounded",
        runtime_kind=PriorRuntimeKind.TRUNCATED_NORMAL,
    ),
    PriorFamilySpec(
        family=PriorDistributionFamily.GAMMA,
        signature="Gamma(concentration, rate)",
        summary="Positive-only parameters when right-skewed uncertainty is plausible.",
        support="positive",
        runtime_kind=PriorRuntimeKind.GAMMA,
    ),
    PriorFamilySpec(
        family=PriorDistributionFamily.LOG_NORMAL,
        signature="LogNormal(mu, sigma)",
        summary="Positive-only parameters when uncertainty is multiplicative on the log scale.",
        support="positive",
        runtime_kind=PriorRuntimeKind.LOG_NORMAL,
    ),
    PriorFamilySpec(
        family=PriorDistributionFamily.EXPONENTIAL,
        signature="Exponential(rate)",
        summary="Positive-only parameters with mass near zero and a single decay rate.",
        support="positive",
        runtime_kind=PriorRuntimeKind.EXPONENTIAL,
    ),
)


PRIOR_FAMILY_REGISTRY: Final[dict[PriorDistributionFamily, PriorFamilySpec]] = {
    spec.family: spec for spec in PRIOR_FAMILY_SPECS
}

OBSERVATION_LINK_VALUES_BY_DISTRIBUTION: Final[dict[DistributionFamily, tuple[str, ...]]] = {
    spec.family: spec.links for spec in OBSERVATION_FAMILY_SPECS
}

PRIOR_CONSTRAINT_GUIDANCE: Final[tuple[PriorConstraintGuidance, ...]] = (
    PriorConstraintGuidance("none", "(-inf, +inf)", "Normal"),
    PriorConstraintGuidance("positive", "(0, +inf)", "HalfNormal, Gamma, LogNormal, Exponential"),
    PriorConstraintGuidance("unit_interval", "[0, 1]", "Beta, Uniform(0, 1)"),
    PriorConstraintGuidance(
        "correlation",
        "[-1, 1]",
        "Uniform(-1, 1), TruncatedNormal(0, sigma, -1, 1)",
    ),
)

PRIOR_PARAMETER_GUIDANCE_ROWS: Final[tuple[PriorParameterGuidanceRow, ...]] = (
    PriorParameterGuidanceRow("beta (causal effect)", "Normal(0, 0.5)", "[-2, 2]", "Discrete-time"),
    PriorParameterGuidanceRow(
        "rho (AR coefficient)",
        "Beta(2, 2) or Uniform(0, 1)",
        "[0, 1]",
        "Discrete-time persistence",
    ),
    PriorParameterGuidanceRow("sigma (residual SD)", "HalfNormal(1)", "[0, 5]", "Data scale"),
    PriorParameterGuidanceRow("lambda (loading)", "HalfNormal(1)", "[0, 3]", "Data scale"),
    PriorParameterGuidanceRow(
        "cor (correlation)",
        "Uniform(-1, 1) or TruncatedNormal(0, 0.3, -1, 1)",
        "[-1, 1]",
        "Innovation correlation",
    ),
    PriorParameterGuidanceRow("tau (random SD)", "HalfNormal(0.5)", "[0, 2]", "Data scale"),
)

# Pure-JAX real-support runtime family indices used by parameterization.py.
REAL_RUNTIME_FAMILY_INDEX: Final[dict[PriorRuntimeKind, int]] = {
    PriorRuntimeKind.NORMAL: 0,
    PriorRuntimeKind.TRUNCATED_NORMAL: 1,
    PriorRuntimeKind.UNIFORM: 2,
}

PRIMARY_REAL_RUNTIME_KIND_BY_INDEX: Final[dict[int, PriorRuntimeKind]] = {
    index: kind for kind, index in REAL_RUNTIME_FAMILY_INDEX.items()
}

# Pure-JAX positive-support runtime family indices used by parameterization.py.
POSITIVE_RUNTIME_FAMILY_INDEX: Final[dict[PriorRuntimeKind, int]] = {
    PriorRuntimeKind.HALF_NORMAL: 0,
    PriorRuntimeKind.GAMMA: 1,
    PriorRuntimeKind.LOG_NORMAL: 2,
    PriorRuntimeKind.EXPONENTIAL: 3,
}

PRIMARY_POSITIVE_RUNTIME_KIND_BY_INDEX: Final[dict[int, PriorRuntimeKind]] = {
    index: kind for kind, index in POSITIVE_RUNTIME_FAMILY_INDEX.items()
}


def get_prior_family_spec(family: PriorDistributionFamily | str) -> PriorFamilySpec:
    """Return the catalog entry for a prior family."""
    return PRIOR_FAMILY_REGISTRY[PriorDistributionFamily(family)]


def get_real_runtime_family_index(runtime_kind: PriorRuntimeKind) -> int:
    """Return the executable real-support family index for a runtime kind."""
    try:
        return REAL_RUNTIME_FAMILY_INDEX[runtime_kind]
    except KeyError as exc:
        raise ValueError(
            f"Runtime kind {runtime_kind!r} is not a real-support executable family."
        ) from exc


def get_real_runtime_kind_from_index(index: int) -> PriorRuntimeKind:
    """Return the runtime kind for a serialized real-support family index."""
    try:
        return PRIMARY_REAL_RUNTIME_KIND_BY_INDEX[index]
    except KeyError as exc:
        raise ValueError(f"Unsupported serialized real prior family index {index}") from exc


def get_positive_runtime_family_index(runtime_kind: PriorRuntimeKind) -> int:
    """Return the executable positive-support family index for a runtime kind."""
    try:
        return POSITIVE_RUNTIME_FAMILY_INDEX[runtime_kind]
    except KeyError as exc:
        raise ValueError(
            f"Runtime kind {runtime_kind!r} is not a positive-support executable family."
        ) from exc


def get_positive_runtime_kind_from_index(index: int) -> PriorRuntimeKind:
    """Return the primary runtime kind for a serialized positive family index."""
    try:
        return PRIMARY_POSITIVE_RUNTIME_KIND_BY_INDEX[index]
    except KeyError as exc:
        raise ValueError(f"Unsupported serialized positive prior family index {index}") from exc


def format_prior_distribution_choice_list(separator: str = "|") -> str:
    """Render the enum values in catalog order for machine-readable prompts."""
    return separator.join(spec.family.value for spec in PRIOR_FAMILY_SPECS)


def format_prior_distribution_name_list(
    *,
    quote: str = "",
    separator: str = ", ",
) -> str:
    """Render the prior family names in catalog order for prose or schema text."""
    return separator.join(f"{quote}{spec.family.value}{quote}" for spec in PRIOR_FAMILY_SPECS)


def render_prior_distribution_guidance_bullets() -> str:
    """Render the authoritative prompt bullet list for prior family guidance."""
    return "\n".join(f"- **{spec.signature}**: {spec.summary}" for spec in PRIOR_FAMILY_SPECS)


def render_observation_distribution_guidance_bullets() -> str:
    """Render the authoritative prompt bullet list for observation-family guidance."""
    return "\n".join(
        f"- `{spec.family.value}`: {spec.summary}" for spec in OBSERVATION_FAMILY_SPECS
    )


def render_observation_link_guidance_bullets() -> str:
    """Render prompt bullets for observation families with multiple valid links."""
    lines: list[str] = []
    for spec in OBSERVATION_FAMILY_SPECS:
        if len(spec.links) <= 1:
            continue
        default_link, *other_links = spec.links
        other_links_str = " or ".join(f"`{link}`" for link in other_links)
        lines.append(
            f"- **{spec.family.value}**: `{default_link}` (default)"
            + (f" or {other_links_str}" if other_links_str else "")
        )
    return "\n".join(lines)


def render_prior_distribution_markdown_table() -> str:
    """Render a markdown table describing supported prior families."""
    lines = [
        "| Family | Signature | Support | Use When |",
        "|---|---|---|---|",
    ]
    for spec in PRIOR_FAMILY_SPECS:
        lines.append(
            f"| `{spec.family.value}` | `{spec.signature}` | `{spec.support}` | {spec.summary} |"
        )
    return "\n".join(lines)


def render_prior_constraint_guidance_markdown_table() -> str:
    """Render a markdown table for constraint-level prior guidance."""
    lines = [
        "| Constraint | Domain | Typical prior families |",
        "|---|---|---|",
    ]
    for row in PRIOR_CONSTRAINT_GUIDANCE:
        lines.append(f"| `{row.constraint}` | `{row.domain}` | {row.typical_families} |")
    return "\n".join(lines)


def render_prior_parameter_guidance_markdown_table() -> str:
    """Render a markdown table for common parameter-level prior defaults."""
    lines = [
        "| Type | Typical Distribution | Typical Range | Scale |",
        "|---|---|---|---|",
    ]
    for row in PRIOR_PARAMETER_GUIDANCE_ROWS:
        lines.append(
            f"| {row.parameter_type} | {row.typical_distribution} | {row.typical_range} | {row.scale} |"
        )
    return "\n".join(lines)
