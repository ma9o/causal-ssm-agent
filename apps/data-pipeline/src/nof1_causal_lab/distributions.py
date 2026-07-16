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

    @property
    def uses_manifest_noise(self) -> bool:
        """Whether this family's emission log-prob reads per-channel manifest noise R.

        Only Gaussian and Student-t emissions read R in the ``emission_log_prob_*``
        functions and the posterior-predictive switch branches. All other
        families determine observation variance from family-level hyperparameters
        (``obs_r``, ``obs_shape``, ``obs_concentration``, ...) and mark R as
        unused with the ``_R`` / ``_std`` naming convention. Emitting a free
        ``obs_sd_<indicator>`` parameter for a non-{Gaussian, Student-t} channel
        therefore creates a disconnected parameter that contributes nothing to
        the likelihood.
        """
        return self in {DistributionFamily.GAUSSIAN, DistributionFamily.STUDENT_T}


@dataclass(frozen=True)
class ObservationFamilyCatalogEntry:
    """Central observation-family metadata shared across prompts and validation."""

    family: DistributionFamily
    summary: str
    links: tuple[str, ...]
    hyperparameters: tuple[str, ...] = ()


class PriorDistributionFamily(StrEnum):
    """Distribution families allowed in model-spec prior proposals."""

    NORMAL = "Normal"
    HALF_NORMAL = "HalfNormal"
    BETA = "Beta"
    UNIFORM = "Uniform"
    TRUNCATED_NORMAL = "TruncatedNormal"
    GAMMA = "Gamma"
    LOG_NORMAL = "LogNormal"
    EXPONENTIAL = "Exponential"
    DELTA = "Delta"


@dataclass(frozen=True)
class PriorFamilySpec:
    """Central prior-family metadata shared across prompts, docs, and runtime."""

    family: PriorDistributionFamily
    signature: str
    summary: str
    support: Literal["real", "positive", "unit_interval", "bounded"]


@dataclass(frozen=True)
class PriorParameterGuidanceRow:
    """Parameter-level prior heuristics reused across model-spec prompts."""

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
        "AR coefficients (`rho_*`) should be authored as a baseline discrete-time "
        "persistence per observation interval, absent feedback from incoming "
        "causes: support on [0, 1] with the prior's location (`mu`) strictly "
        "inside (0, 1) — 0 and 1 are degenerate (the CT decay is -ln(mu)/dt). "
        "`beta_*` priors should be authored on the interval they mean. For lagged "
        "`beta_*`, set `reference_interval_days` when the evidence is on a different "
        "interval; otherwise the model interval is assumed. The compiler handles "
        "interval normalization, CT conversion, and the realised diagonal damping "
        "needed to keep the drift stable. "
        "`t0_mean_*` and `t0_sd_*` live on the latent state scale: do not set them "
        "to raw reference-indicator means or `log(mean(indicator))` unless the "
        "construct is explicitly identified on that observed scale."
    )


# ---------------------------------------------------------------------------
# Parameter role catalog — authoritative source for docs codegen and the
# EXPECTED_CONSTRAINT_FOR_ROLE dict in artifacts/statistical_model_spec.py.
# Uses plain strings (not enum references) so this module stays import-clean.
# ---------------------------------------------------------------------------

_CONSTRAINT_DOMAINS: Final[dict[str, str]] = {
    "unit_interval": "[0, 1]",
    "none": "(-inf, +inf)",
    "positive": "(0, +inf)",
    "negative": "(-inf, 0)",
    "correlation": "[-1, 1]",
}


def constraint_domain(constraint: str) -> str:
    """The numeric support domain (e.g. ``[0, 1]``) for a ``ParameterConstraint``.

    ``constraint`` is a ``ParameterConstraint`` enum *value* — kept as a plain str
    here so this module stays import-clean (see the note above).
    """
    return _CONSTRAINT_DOMAINS[constraint]


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
        ssm_location="State-decay dynamics site",
        note="model-spec elicits baseline discrete-time persistence absent feedback; "
        "[compilation](../compilation.md) binds it to the owning decay component "
        "and converts to continuous-time decay scale",
    ),
    ParameterRoleSpec(
        role="fixed_effect",
        symbol="beta",
        count="One per causal edge",
        constraint="none",
        ssm_location="Dynamics edge or input-effect site",
        note="Causal effects can be positive or negative; compiler binds each "
        "coefficient to the owning edge component or known-input effect site",
    ),
    ParameterRoleSpec(
        role="dynamics_parameter",
        symbol="theta",
        count="One per real-valued component-owned dynamics parameter",
        constraint="none",
        ssm_location="Component dynamics site",
        note="Used for component-owned dynamics parameters that are not authored "
        "as interval-scale effect coefficients.",
    ),
    ParameterRoleSpec(
        role="dynamics_parameter_positive",
        symbol="theta+",
        count="One per positive component-owned dynamics parameter",
        constraint="positive",
        ssm_location="Component dynamics site",
        note="Used for positive component-owned dynamics parameters such as Hill Emax and EC50.",
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
        ssm_location="Observation model",
        note="measurement-structure indicator polarity fixes each loading sign as either "
        "`positive` or `negative`; model-spec no longer chooses loading orientation",
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
        hyperparameters=("obs_df",),
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
        hyperparameters=("obs_shape",),
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
        hyperparameters=("obs_r",),
    ),
    ObservationFamilyCatalogEntry(
        family=DistributionFamily.BETA,
        summary="Proportions or rates strictly inside the unit interval.",
        links=("logit", "probit"),
        hyperparameters=("obs_concentration",),
    ),
    ObservationFamilyCatalogEntry(
        family=DistributionFamily.ORDERED_LOGISTIC,
        summary=(
            "Ordered categorical outcomes with ranked levels. Keeps a loading on the "
            "latent (fixed logistic scale), unlike `categorical`."
        ),
        links=("cumulative_logit",),
        hyperparameters=("obs_ordered_base", "obs_ordered_gaps"),
    ),
    ObservationFamilyCatalogEntry(
        family=DistributionFamily.CATEGORICAL,
        summary=(
            "Unordered multi-class outcomes. Choosing it removes the channel's loading "
            "(the class slopes are exactly redundant with it, so the compiler pins it); "
            "discrimination moves into `obs_cat_slopes`."
        ),
        links=("softmax",),
        hyperparameters=("obs_cat_intercepts", "obs_cat_slopes"),
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
    PriorFamilySpec(
        family=PriorDistributionFamily.DELTA,
        signature="Delta(value)",
        summary="Fixed positive value inserted by compiler-owned deterministic repairs.",
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
        "Baseline discrete-time persistence absent feedback",
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
        "Category-specific slope adjustments on the latent predictor scale; when every "
        "indicator of a construct is categorical, the reference channel's first "
        "non-baseline slope is compiler-pinned to +1 as the scale/sign anchor and the "
        "prior applies to the remaining slopes",
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
    PriorDistributionFamily.DELTA: 4,
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


def _prompt_prior_family_specs(
    *,
    include_delta: bool,
) -> tuple[PriorFamilySpec, ...]:
    if include_delta:
        return PRIOR_FAMILY_SPECS
    return tuple(
        spec for spec in PRIOR_FAMILY_SPECS if spec.family != PriorDistributionFamily.DELTA
    )


def format_prior_distribution_choice_list(
    separator: str = "|",
    *,
    include_delta: bool = False,
) -> str:
    """Render the enum values in catalog order for machine-readable prompts."""
    return separator.join(
        spec.family.value for spec in _prompt_prior_family_specs(include_delta=include_delta)
    )


def format_prior_distribution_name_list(
    *,
    quote: str = "",
    separator: str = ", ",
    include_delta: bool = False,
) -> str:
    """Render the prior family names in catalog order for prose or schema text."""
    return separator.join(
        f"{quote}{spec.family.value}{quote}"
        for spec in _prompt_prior_family_specs(include_delta=include_delta)
    )


def render_prior_distribution_guidance_bullets(*, include_delta: bool = False) -> str:
    """Render the authoritative prompt bullet list for prior family guidance."""
    return "\n".join(
        f"- **{spec.signature}**: {spec.summary}"
        for spec in _prompt_prior_family_specs(include_delta=include_delta)
    )


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
