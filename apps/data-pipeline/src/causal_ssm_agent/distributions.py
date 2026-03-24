"""Central distribution catalog for observation models and priors."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Final, Literal


def _normalize_distribution_token(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


class _CatalogBackedStrEnum(StrEnum):
    """StrEnum with case-insensitive alias lookup from a catalog."""

    _ALIASES: ClassVar[dict[str, str]]

    @classmethod
    def _missing_(cls, value: object):
        if not isinstance(value, str):
            return None

        normalized = _normalize_distribution_token(value)
        canonical = _normalize_distribution_token(cls._ALIASES.get(normalized, normalized))
        for member in cls:
            if _normalize_distribution_token(member.value) == canonical:
                return member
        return None


OBSERVATION_DISTRIBUTION_ALIASES: Final[dict[str, str]] = {
    "normal": "gaussian",
    "negativebinomial": "negative_binomial",
    "orderedlogistic": "ordered_logistic",
}


class DistributionFamily(_CatalogBackedStrEnum):
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


PRIOR_DISTRIBUTION_ALIASES: Final[dict[str, str]] = {
    "normal": "Normal",
    "halfnormal": "HalfNormal",
    "half_normal": "HalfNormal",
    "beta": "Beta",
    "uniform": "Uniform",
    "truncatednormal": "TruncatedNormal",
    "truncated_normal": "TruncatedNormal",
    "gamma": "Gamma",
    "lognormal": "LogNormal",
    "log_normal": "LogNormal",
    "exponential": "Exponential",
    "exp": "Exponential",
}


class PriorDistributionFamily(_CatalogBackedStrEnum):
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

# Pure-JAX positive-support runtime family indices used by parameterization.py.
POSITIVE_RUNTIME_FAMILY_INDEX: Final[dict[PriorRuntimeKind, int]] = {
    PriorRuntimeKind.HALF_NORMAL: 0,
    PriorRuntimeKind.GAMMA: 1,
    PriorRuntimeKind.LOG_NORMAL: 2,
    PriorRuntimeKind.EXPONENTIAL: 1,
}

PRIMARY_POSITIVE_RUNTIME_KIND_BY_INDEX: Final[dict[int, PriorRuntimeKind]] = {
    index: kind
    for kind, index in POSITIVE_RUNTIME_FAMILY_INDEX.items()
    if kind != PriorRuntimeKind.EXPONENTIAL
}


def get_prior_family_spec(family: PriorDistributionFamily | str) -> PriorFamilySpec:
    """Return the catalog entry for a prior family."""
    return PRIOR_FAMILY_REGISTRY[PriorDistributionFamily(family)]


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


DistributionFamily._ALIASES = OBSERVATION_DISTRIBUTION_ALIASES
PriorDistributionFamily._ALIASES = PRIOR_DISTRIBUTION_ALIASES
