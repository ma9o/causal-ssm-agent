"""Typed deterministic candidates used to assemble the model-spec decision surface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.statistical_model_spec import (
        InitializationPolicy,
        LinkFunction,
        ParameterConstraint,
    )
    from nof1_causal_lab.distributions import DistributionFamily
    from nof1_causal_lab.models.ssm.structure.sites import (
        PriorAuthoringTransform,
        SiteKind,
    )

type TemporalStatus = Literal["time_varying", "time_invariant"]


class ObservationSemanticsFields(TypedDict):
    """Support semantics copied from a causal-design indicator."""

    support_kind: str | None
    summary_operator: str | None


class ResolvedLikelihoodCandidate(ObservationSemanticsFields):
    """Likelihood whose family and link are fixed deterministically."""

    variable: str
    construct_name: str | None
    distribution: DistributionFamily
    link: LinkFunction
    reasoning: str


class FixedDistributionLikelihoodCandidate(ObservationSemanticsFields):
    """Likelihood with a fixed family and a remaining link choice."""

    variable: str
    construct_name: str | None
    dtype: str
    fixed_distribution: DistributionFamily
    valid_links: list[LinkFunction]


class OpenLikelihoodCandidate(ObservationSemanticsFields):
    """Likelihood with both family and family-specific link choices open."""

    variable: str
    construct_name: str | None
    dtype: str
    valid_distributions: list[DistributionFamily]
    link_options: dict[DistributionFamily, list[LinkFunction]]


type AmbiguousLikelihoodCandidate = FixedDistributionLikelihoodCandidate | OpenLikelihoodCandidate


class CandidateBindingMetadata(TypedDict, total=False):
    """Compiler-owned metadata attached after semantic parameter binding."""

    construct_names: list[str]
    indicator_names: list[str]
    compiled_site_name: str
    compiled_prior_field: str | None
    compiled_flat_index: int
    compiled_site_kind: SiteKind
    prior_transform: PriorAuthoringTransform
    component_index: int | None
    component_parameter: str | None
    temporal_status: TemporalStatus | None
    conditional_prior_surface: bool
    activation_equilibrium_forcing: bool
    activation_initialization_policies: list[InitializationPolicy]
    activation_indicator_names: list[str]
    activation_distribution_families: list[DistributionFamily]


class ConstructParameterCandidate(CandidateBindingMetadata):
    """Parameter owned by one latent construct."""

    name: str
    role: Literal[
        "ar_coefficient",
        "residual_sd",
        "state_intercept",
        "initial_state_mean",
        "initial_state_sd",
    ]
    constraint: ParameterConstraint
    description: str
    construct: str


class EdgeParameterCandidate(CandidateBindingMetadata):
    """Directed causal edge coefficient candidate."""

    name: str
    role: Literal["fixed_effect"]
    constraint: ParameterConstraint
    description: str
    cause: str
    effect: str
    lagged: bool


class LoadingParameterCandidate(CandidateBindingMetadata):
    """Non-reference measurement loading candidate."""

    name: str
    role: Literal["loading"]
    constraint: ParameterConstraint
    description: str
    indicator: str
    construct: str
    reference_indicator: str | None
    indicator_polarity: ParameterConstraint


class ObservationParameterCandidate(CandidateBindingMetadata, total=False):
    """Observation-channel intercept, noise, or family-specific parameter."""

    name: str
    role: Literal[
        "measurement_error_sd",
        "observation_intercept",
        "observation_hyperparameter",
        "observation_hyperparameter_positive",
    ]
    constraint: ParameterConstraint
    description: str
    construct: str | None
    indicator: str


class CorrelationParameterCandidate(CandidateBindingMetadata):
    """Innovation or initial-state dependency parameter."""

    name: str
    role: Literal["correlation", "initial_state_correlation", "static_state_sd"]
    constraint: ParameterConstraint
    description: str
    dependency_kind: Literal["innovation_correlation", "initial_state_correlation"]
    source_confounders: list[str]


class ComponentParameterCandidate(CandidateBindingMetadata):
    """Compiler-owned nonlinear dynamics component parameter."""

    name: str
    role: Literal["dynamics_parameter", "dynamics_parameter_positive"]
    constraint: ParameterConstraint
    description: str


type ModelParameterCandidate = (
    ConstructParameterCandidate
    | EdgeParameterCandidate
    | LoadingParameterCandidate
    | ObservationParameterCandidate
    | CorrelationParameterCandidate
    | ComponentParameterCandidate
)
