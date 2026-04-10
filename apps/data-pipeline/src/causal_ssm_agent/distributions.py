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
class ObservationFamilyCatalogEntry:
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


@dataclass(frozen=True)
class PriorFamilySpec:
    """Central prior-family metadata shared across prompts, docs, and runtime."""

    family: PriorDistributionFamily
    signature: str
    summary: str
    support: Literal["real", "positive", "unit_interval", "bounded"]


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


LAGGED_BETA_AUTHORED_INTERVAL_SCALE: Final[str] = (
    "Authored interval effect (defaults to model interval; use "
    "`reference_interval_days` when evidence is on another interval)"
)


def render_dynamic_prior_scale_guidance() -> str:
    """Render the shared authored-scale contract for dynamic priors."""
    return (
        "AR coefficients (`rho_*`) should be authored as discrete-time persistence "
        "per observation interval. `beta_*` priors should be authored on the "
        "interval they mean. For lagged `beta_*`, set `reference_interval_days` "
        "when the evidence is on a different interval; otherwise the model interval "
        "is assumed. The compiler handles interval normalization and CT conversion. "
        "`t0_mean_*` and `t0_sd_*` live on the latent state scale: do not set them "
        "to raw reference-indicator means or `log(mean(indicator))` unless the "
        "construct is explicitly identified on that observed scale."
    )


def render_lagged_beta_authored_interval_guidance() -> str:
    """Render the shared Stage 4 guidance for lagged beta priors."""
    return "\n".join(
        [
            "- Author `params` on the interval you actually mean.",
            "- If you omit `reference_interval_days`, the system assumes `params` "
            "are already expressed on the model interval shown in the fixed-effect "
            "prior card.",
            "- If the literature estimate comes from a different study interval, set "
            "`reference_interval_days` to that interval and keep `params` on that "
            "authored interval scale. The compiler will rescale it before CT compilation.",
            "- Do not manually pre-convert lagged `beta_*` priors into continuous-time "
            "or one-step drift units.",
            "- If `Feedback Loop` is `yes`, use a more conservative interval-scale "
            "effect than a cross-sectional or multi-day association would suggest.",
        ]
    )


# ---------------------------------------------------------------------------
# Parameter role catalog — authoritative source for docs codegen and the
# EXPECTED_CONSTRAINT_FOR_ROLE dict in artifacts/model_spec.py.
# Uses plain strings (not enum references) so this module stays import-clean.
# ---------------------------------------------------------------------------

_CONSTRAINT_DOMAINS: Final[dict[str, str]] = {
    "unit_interval": "[0, 1]",
    "none": "(-inf, +inf)",
    "positive": "(0, +inf)",
    "negative": "(-inf, 0)",
    "correlation": "[-1, 1]",
}


@dataclass(frozen=True)
class ParameterRoleSpec:
    """Metadata for a parameter role used by docs codegen and validation."""

    role: str  # matches ParameterRole enum value
    symbol: str
    count: str
    constraint: str  # matches ParameterConstraint enum value
    ssm_location: str
    note: str = ""

    @property
    def domain(self) -> str:
        return _CONSTRAINT_DOMAINS[self.constraint]


PARAMETER_ROLE_SPECS: Final[tuple[ParameterRoleSpec, ...]] = (
    ParameterRoleSpec(
        role="ar_coefficient",
        symbol="rho",
        count="One per endogenous time-varying construct",
        constraint="unit_interval",
        ssm_location="Drift diagonal",
        note="Stage 4 elicits discrete-time persistence magnitude; "
        "[compilation](../compilation.md) converts to continuous-time drift",
    ),
    ParameterRoleSpec(
        role="fixed_effect",
        symbol="beta",
        count="One per causal edge",
        constraint="none",
        ssm_location="Drift off-diagonal",
        note="Causal effects can be positive or negative; unconstrained",
    ),
    ParameterRoleSpec(
        role="residual_sd",
        symbol="sigma",
        count="One per construct",
        constraint="positive",
        ssm_location="Diffusion diagonal",
    ),
    ParameterRoleSpec(
        role="state_intercept",
        symbol="cint",
        count="One per eligible dynamic construct when equilibrium forcing is enabled",
        constraint="none",
        ssm_location="Continuous-time state intercept",
    ),
    ParameterRoleSpec(
        role="observation_intercept",
        symbol="manifest_mean",
        count="One per manifest channel whose observation family requires a baseline intercept",
        constraint="none",
        ssm_location="Manifest intercept vector",
    ),
    ParameterRoleSpec(
        role="initial_state_mean",
        symbol="t0_mean",
        count="One per latent construct",
        constraint="none",
        ssm_location="Initial-state mean vector",
    ),
    ParameterRoleSpec(
        role="initial_state_sd",
        symbol="t0_sd",
        count="One per latent construct",
        constraint="positive",
        ssm_location="Initial-state covariance diagonal",
    ),
    ParameterRoleSpec(
        role="static_state_sd",
        symbol="tau",
        count="One per compiled baseline factor induced by marginalized time-invariant confounders",
        constraint="positive",
        ssm_location="Static baseline-factor covariance",
        note="Used to build low-rank initial-state covariance contributions of the form "
        "`B diag(tau^2) B^T`.",
    ),
    ParameterRoleSpec(
        role="loading",
        symbol="lambda",
        count="One per non-reference indicator in multi-indicator constructs",
        constraint="positive",
        ssm_location="Measurement model",
        note="Stage 1b indicator polarity fixes each loading sign as either "
        "`positive` or `negative`; Stage 4 no longer chooses loading orientation",
    ),
    ParameterRoleSpec(
        role="measurement_error_sd",
        symbol="obs_sd",
        count="One per free manifest measurement-error SD",
        constraint="positive",
        ssm_location="Manifest variance diagonal",
        note="Surfaced only when measurement error is separately estimated "
        "(multi-indicator constructs).",
    ),
    ParameterRoleSpec(
        role="observation_hyperparameter",
        symbol="obs_*",
        count="One per active real-valued observation-family hyperparameter site",
        constraint="none",
        ssm_location="Observation-family auxiliary site",
        note="Examples include ordered-threshold bases and categorical logit offsets.",
    ),
    ParameterRoleSpec(
        role="observation_hyperparameter_positive",
        symbol="obs_*",
        count="One per active positive observation-family hyperparameter site",
        constraint="positive",
        ssm_location="Observation-family auxiliary site",
        note="Examples include Student-t degrees of freedom, Gamma shape, and NB dispersion.",
    ),
    ParameterRoleSpec(
        role="correlation",
        symbol="cor",
        count="One per construct-pair with marginalized confounder",
        constraint="correlation",
        ssm_location="Diffusion covariance",
    ),
)


OBSERVATION_FAMILY_SPECS: Final[tuple[ObservationFamilyCatalogEntry, ...]] = (
    ObservationFamilyCatalogEntry(
        family=DistributionFamily.GAUSSIAN,
        summary="Continuous unbounded data, approximately symmetric.",
        links=("identity",),
    ),
    ObservationFamilyCatalogEntry(
        family=DistributionFamily.STUDENT_T,
        summary="Continuous data with heavy tails or outliers.",
        links=("identity",),
    ),
    ObservationFamilyCatalogEntry(
        family=DistributionFamily.POISSON,
        summary="Count data with variance roughly tracking the mean.",
        links=("log",),
    ),
    ObservationFamilyCatalogEntry(
        family=DistributionFamily.GAMMA,
        summary="Positive continuous data such as durations or reaction times.",
        links=("log", "inverse"),
    ),
    ObservationFamilyCatalogEntry(
        family=DistributionFamily.BERNOULLI,
        summary="Binary outcomes with two possible states.",
        links=("logit", "probit"),
    ),
    ObservationFamilyCatalogEntry(
        family=DistributionFamily.NEGATIVE_BINOMIAL,
        summary="Overdispersed count data where variance exceeds the mean.",
        links=("log",),
    ),
    ObservationFamilyCatalogEntry(
        family=DistributionFamily.BETA,
        summary="Proportions or rates strictly inside the unit interval.",
        links=("logit", "probit"),
    ),
    ObservationFamilyCatalogEntry(
        family=DistributionFamily.ORDERED_LOGISTIC,
        summary="Ordered categorical outcomes with ranked levels.",
        links=("cumulative_logit",),
    ),
    ObservationFamilyCatalogEntry(
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
    ),
    PriorFamilySpec(
        family=PriorDistributionFamily.HALF_NORMAL,
        signature="HalfNormal(sigma)",
        summary="Positive-only parameters such as standard deviations and scales.",
        support="positive",
    ),
    PriorFamilySpec(
        family=PriorDistributionFamily.BETA,
        signature="Beta(alpha, beta)",
        summary="Parameters constrained to the unit interval [0, 1].",
        support="unit_interval",
    ),
    PriorFamilySpec(
        family=PriorDistributionFamily.UNIFORM,
        signature="Uniform(lower, upper)",
        summary="Hard-bounded parameters when only plausible limits are known.",
        support="bounded",
    ),
    PriorFamilySpec(
        family=PriorDistributionFamily.TRUNCATED_NORMAL,
        signature="TruncatedNormal(mu, sigma, lower, upper)",
        summary="Bounded parameters when both a center and hard limits are meaningful.",
        support="bounded",
    ),
    PriorFamilySpec(
        family=PriorDistributionFamily.GAMMA,
        signature="Gamma(concentration, rate)",
        summary="Positive-only parameters when right-skewed uncertainty is plausible.",
        support="positive",
    ),
    PriorFamilySpec(
        family=PriorDistributionFamily.LOG_NORMAL,
        signature="LogNormal(mu, sigma)",
        summary="Positive-only parameters when uncertainty is multiplicative on the log scale.",
        support="positive",
    ),
    PriorFamilySpec(
        family=PriorDistributionFamily.EXPONENTIAL,
        signature="Exponential(rate)",
        summary="Positive-only parameters with mass near zero and a single decay rate.",
        support="positive",
    ),
)


PRIOR_FAMILY_REGISTRY: Final[dict[PriorDistributionFamily, PriorFamilySpec]] = {
    spec.family: spec for spec in PRIOR_FAMILY_SPECS
}

OBSERVATION_LINK_VALUES_BY_DISTRIBUTION: Final[dict[DistributionFamily, tuple[str, ...]]] = {
    spec.family: spec.links for spec in OBSERVATION_FAMILY_SPECS
}

# Ordered dtype → valid distributions (default first).  Authoritative source
# for both the validation logic and the generated likelihoods docs.
VALID_LIKELIHOODS_FOR_DTYPE: Final[dict[str, tuple[DistributionFamily, ...]]] = {
    "continuous": (
        DistributionFamily.GAUSSIAN,
        DistributionFamily.STUDENT_T,
        DistributionFamily.GAMMA,
        DistributionFamily.BETA,
    ),
    "binary": (DistributionFamily.BERNOULLI,),
    "count": (DistributionFamily.POISSON, DistributionFamily.NEGATIVE_BINOMIAL),
    "ordinal": (DistributionFamily.ORDERED_LOGISTIC,),
    "categorical": (DistributionFamily.CATEGORICAL, DistributionFamily.ORDERED_LOGISTIC),
}

# Per-alternative notes for the generated likelihoods table.
_DTYPE_ALTERNATIVE_NOTES: Final[dict[tuple[str, DistributionFamily], str]] = {
    ("categorical", DistributionFamily.ORDERED_LOGISTIC): (
        "when categories are substantively ordered"
    ),
}

PRIOR_CONSTRAINT_GUIDANCE: Final[tuple[PriorConstraintGuidance, ...]] = (
    PriorConstraintGuidance("none", "(-inf, +inf)", "Normal"),
    PriorConstraintGuidance("positive", "(0, +inf)", "HalfNormal, Gamma, LogNormal, Exponential"),
    PriorConstraintGuidance(
        "negative",
        "(-inf, 0)",
        "TruncatedNormal(mu<0, sigma, lower, 0), Uniform(lower, 0)",
    ),
    PriorConstraintGuidance("unit_interval", "[0, 1]", "Beta, Uniform(0, 1)"),
    PriorConstraintGuidance(
        "correlation",
        "[-1, 1]",
        "Uniform(-1, 1), TruncatedNormal(0, sigma, -1, 1)",
    ),
)

PRIOR_PARAMETER_GUIDANCE_ROWS: Final[tuple[PriorParameterGuidanceRow, ...]] = (
    PriorParameterGuidanceRow(
        "beta (causal effect)",
        "Normal(0, 0.5)",
        "[-2, 2]",
        LAGGED_BETA_AUTHORED_INTERVAL_SCALE,
    ),
    PriorParameterGuidanceRow(
        "rho (AR coefficient)",
        "Beta(2, 2) or Uniform(0, 1)",
        "[0, 1]",
        "Discrete-time persistence",
    ),
    PriorParameterGuidanceRow("sigma (residual SD)", "HalfNormal(1)", "[0, 5]", "Data scale"),
    PriorParameterGuidanceRow(
        "t0_mean (initial-state mean)",
        "Normal(0, 1)",
        "[-3, 3]",
        "Latent state scale; do not copy raw indicator means or log-means unless the construct is explicitly identified on that observed scale",
    ),
    PriorParameterGuidanceRow(
        "t0_sd (initial-state SD)",
        "HalfNormal(1)",
        "[0, 3]",
        "Latent state scale",
    ),
    PriorParameterGuidanceRow(
        "lambda (loading)",
        "HalfNormal(1) if positive, TruncatedNormal(-1, 0.5, -5, 0) if negative",
        "[-3, 3]",
        "Data scale with sign fixed by indicator polarity",
    ),
    PriorParameterGuidanceRow(
        "obs_sd (measurement error SD)",
        "HalfNormal(0.5) or HalfNormal(1)",
        "[0, 3]",
        "Manifest observation-noise scale; larger values attribute more variation to indicator noise instead of the latent state",
    ),
    PriorParameterGuidanceRow(
        "obs_df (Student-t tails)",
        "Gamma(5, 1) or LogNormal(log(5), 0.3)",
        "[2, 30]",
        "Observation-tail heaviness; smaller values mean heavier tails",
    ),
    PriorParameterGuidanceRow(
        "obs_shape (Gamma shape)",
        "Gamma(2, 1)",
        "[0.5, 10]",
        "Observation overdispersion/shape for Gamma-family emissions",
    ),
    PriorParameterGuidanceRow(
        "obs_r (negative-binomial dispersion)",
        "Gamma(2, 0.5)",
        "[0.5, 20]",
        "Observation overdispersion; smaller values imply heavier count overdispersion",
    ),
    PriorParameterGuidanceRow(
        "obs_concentration (Beta concentration)",
        "Gamma(5, 0.5)",
        "[1, 50]",
        "Observation concentration around the latent mean on (0, 1)",
    ),
    PriorParameterGuidanceRow(
        "obs_ordered_base (ordered thresholds)",
        "Normal(0, 1)",
        "[-3, 3]",
        "Ordered-logistic threshold location on the latent predictor scale",
    ),
    PriorParameterGuidanceRow(
        "obs_ordered_gaps (ordered threshold gaps)",
        "HalfNormal(1)",
        "[0, 3]",
        "Positive spacing between adjacent ordered-logistic thresholds",
    ),
    PriorParameterGuidanceRow(
        "obs_cat_intercepts (categorical logits)",
        "Normal(0, 1)",
        "[-4, 4]",
        "Baseline category-logit offsets on the latent predictor scale",
    ),
    PriorParameterGuidanceRow(
        "obs_cat_slopes (categorical logits)",
        "Normal(0, 1)",
        "[-4, 4]",
        "Category-specific slope adjustments on the latent predictor scale",
    ),
    PriorParameterGuidanceRow(
        "cor (correlation)",
        "Uniform(-1, 1) or TruncatedNormal(0, 0.3, -1, 1)",
        "[-1, 1]",
        "Innovation correlation",
    ),
    PriorParameterGuidanceRow("tau (random SD)", "HalfNormal(0.5)", "[0, 2]", "Data scale"),
)

# Pure-JAX real-support runtime family indices used by parameterization.py.
REAL_RUNTIME_FAMILY_INDEX: Final[dict[PriorDistributionFamily, int]] = {
    PriorDistributionFamily.NORMAL: 0,
    PriorDistributionFamily.TRUNCATED_NORMAL: 1,
    PriorDistributionFamily.UNIFORM: 2,
}

PRIMARY_REAL_RUNTIME_KIND_BY_INDEX: Final[dict[int, PriorDistributionFamily]] = {
    index: kind for kind, index in REAL_RUNTIME_FAMILY_INDEX.items()
}

# Pure-JAX positive-support runtime family indices used by parameterization.py.
POSITIVE_RUNTIME_FAMILY_INDEX: Final[dict[PriorDistributionFamily, int]] = {
    PriorDistributionFamily.HALF_NORMAL: 0,
    PriorDistributionFamily.GAMMA: 1,
    PriorDistributionFamily.LOG_NORMAL: 2,
    PriorDistributionFamily.EXPONENTIAL: 3,
}

PRIMARY_POSITIVE_RUNTIME_KIND_BY_INDEX: Final[dict[int, PriorDistributionFamily]] = {
    index: kind for kind, index in POSITIVE_RUNTIME_FAMILY_INDEX.items()
}


def get_prior_family_spec(family: PriorDistributionFamily | str) -> PriorFamilySpec:
    """Return the catalog entry for a prior family."""
    return PRIOR_FAMILY_REGISTRY[PriorDistributionFamily(family)]


def get_real_runtime_family_index(family: PriorDistributionFamily) -> int:
    """Return the executable real-support family index."""
    try:
        return REAL_RUNTIME_FAMILY_INDEX[family]
    except KeyError as exc:
        raise ValueError(f"{family!r} is not a real-support executable family.") from exc


def get_real_runtime_kind_from_index(index: int) -> PriorDistributionFamily:
    """Return the prior family for a serialized real-support family index."""
    try:
        return PRIMARY_REAL_RUNTIME_KIND_BY_INDEX[index]
    except KeyError as exc:
        raise ValueError(f"Unsupported serialized real prior family index {index}") from exc


def get_positive_runtime_family_index(family: PriorDistributionFamily) -> int:
    """Return the executable positive-support family index."""
    try:
        return POSITIVE_RUNTIME_FAMILY_INDEX[family]
    except KeyError as exc:
        raise ValueError(f"{family!r} is not a positive-support executable family.") from exc


def get_positive_runtime_kind_from_index(index: int) -> PriorDistributionFamily:
    """Return the prior family for a serialized positive family index."""
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


# ---------------------------------------------------------------------------
# Likelihood doc renderers
# ---------------------------------------------------------------------------


def _format_dist_with_links(dist: DistributionFamily) -> str:
    """Format a distribution with its valid links for the alternatives column."""
    links = OBSERVATION_LINK_VALUES_BY_DISTRIBUTION[dist]
    if len(links) == 1:
        return f"`{dist.value}` (`{links[0]}`)"
    link_str = " or ".join(f"`{lk}`" for lk in links)
    return f"`{dist.value}` ({link_str})"


def render_dtype_likelihood_markdown_table() -> str:
    """Render the dtype-to-distribution mapping table for likelihoods docs."""
    lines = [
        "| `measurement_dtype` | Default distribution | Link | Alternatives |",
        "|---|---|---|---|",
    ]
    for dtype, valid_dists in VALID_LIKELIHOODS_FOR_DTYPE.items():
        default_dist = valid_dists[0]
        default_links = OBSERVATION_LINK_VALUES_BY_DISTRIBUTION[default_dist]
        default_link = default_links[0]

        alternatives: list[str] = []
        # Other links for the default distribution
        for link in default_links[1:]:
            alternatives.append(f"`{default_dist.value}` with `{link}`")
        # Other distributions
        for dist in valid_dists[1:]:
            entry = _format_dist_with_links(dist)
            note = _DTYPE_ALTERNATIVE_NOTES.get((dtype, dist))
            if note:
                entry = f"{entry} {note}"
            alternatives.append(entry)

        alt_str = ", ".join(alternatives) if alternatives else "None"
        lines.append(f"| `{dtype}` | `{default_dist.value}` | `{default_link}` | {alt_str} |")
    return "\n".join(lines)


def render_distribution_families_prose() -> str:
    """Render the DistributionFamily enumeration as a prose sentence."""
    names = [f"`{spec.family.value}`" for spec in OBSERVATION_FAMILY_SPECS]
    return (
        f"`DistributionFamily` enumerates the valid likelihood distribution names: "
        f"{', '.join(names[:-1])}, and {names[-1]}."
    )


def render_link_functions_prose() -> str:
    """Render the LinkFunction enumeration as a prose sentence."""
    all_links: list[str] = []
    seen: set[str] = set()
    for spec in OBSERVATION_FAMILY_SPECS:
        for link in spec.links:
            if link not in seen:
                seen.add(link)
                all_links.append(f"`{link}`")
    return (
        f"`LinkFunction` enumerates the valid link function names: "
        f"{', '.join(all_links[:-1])}, and {all_links[-1]}."
    )


# ---------------------------------------------------------------------------
# Parameter role doc renderers
# ---------------------------------------------------------------------------


def render_parameter_roles_markdown_table() -> str:
    """Render the Parameter Roles table for docs codegen."""
    lines = [
        "| Role | Symbol | Count | Constraint | SSM location |",
        "|---|---|---|---|---|",
    ]
    for spec in PARAMETER_ROLE_SPECS:
        constraint_cell = f"`{spec.constraint}` `{spec.domain}`"
        if spec.role == "loading":
            constraint_cell = "`positive` or `negative`"
        lines.append(
            f"| `{spec.role}` | `{spec.symbol}` "
            f"| {spec.count} | {constraint_cell} | {spec.ssm_location} |"
        )
    return "\n".join(lines)


def render_parameter_constraint_notes() -> str:
    """Render the constraint notes bullets for docs codegen."""
    notes = [spec for spec in PARAMETER_ROLE_SPECS if spec.note]
    return "\n".join(f"- `{spec.role}`: {spec.note}" for spec in notes)
