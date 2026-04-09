"""Typed Stage 4 parameter surfaces derived from the compiler inventory."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from causal_ssm_agent.models.ssm_spec_translation import get_construct_dt_days
from causal_ssm_agent.utils.causal_spec import get_estimation_edges

if TYPE_CHECKING:
    from .stage4_skeleton import Stage4Skeleton

Stage4ParameterBlockKind = Literal[
    "measurement_prior",
    "observation_prior",
    "dynamics_prior",
    "effect_prior",
    "correlation_prior",
]

_DYNAMICS_PARAMETER_ROLES = frozenset(
    {
        "ar_coefficient",
        "residual_sd",
        "initial_state_mean",
        "initial_state_sd",
    }
)
_OBSERVATION_PARAMETER_ROLES = frozenset(
    {
        "observation_hyperparameter",
        "observation_hyperparameter_positive",
    }
)
_CORRELATION_PARAMETER_ROLES = frozenset(
    {
        "correlation",
        "initial_state_correlation",
    }
)


@dataclass(frozen=True)
class Stage4ParameterSurface:
    """Semantic Stage 4 parameter surface with one owning prior-block family."""

    parameter: dict[str, Any]
    block_kind: Stage4ParameterBlockKind
    owner_key: str
    construct_names: tuple[str, ...] = ()
    indicator_names: tuple[str, ...] = ()
    structural_context: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Return the compiler-authoritative semantic parameter name."""
        return str(self.parameter["name"])

    @property
    def role(self) -> str:
        """Return the semantic role for this Stage 4 parameter."""
        return str(self.parameter["role"])

    @property
    def constraint(self) -> str:
        """Return the public parameter constraint."""
        return str(self.parameter["constraint"])

    @property
    def description(self) -> str:
        """Return the prompt-facing description for this parameter."""
        return str(self.parameter.get("description") or self.name)

    @property
    def effect_edge(self) -> tuple[str, str] | None:
        """Return the `(cause, effect)` edge when this is a fixed-effect surface."""
        if self.role != "fixed_effect" or len(self.construct_names) != 2:
            return None
        return self.construct_names

    def to_prior_card(self) -> dict[str, Any]:
        """Render the prompt-local prior card for this semantic surface."""
        return {
            "parameter": self.name,
            "role": self.role,
            "constraint": self.constraint,
            "structural_context": deepcopy(self.structural_context),
        }


@dataclass(frozen=True)
class Stage4ParameterSurfaceIndex:
    """Ordered Stage 4 parameter surfaces plus derived lookup indexes."""

    surfaces: tuple[Stage4ParameterSurface, ...]
    by_name: dict[str, Stage4ParameterSurface]
    by_block_kind: dict[Stage4ParameterBlockKind, tuple[Stage4ParameterSurface, ...]]

    @property
    def ordered_names(self) -> tuple[str, ...]:
        """Return semantic parameter names in deterministic Stage 4 order."""
        return tuple(surface.name for surface in self.surfaces)

    def for_block_kind(self, kind: Stage4ParameterBlockKind) -> tuple[Stage4ParameterSurface, ...]:
        """Return ordered surfaces owned by one Stage 4 prior-block family."""
        return self.by_block_kind.get(kind, ())


def parameter_is_active_for_likelihoods(
    parameter: dict[str, Any],
    chosen_distribution_by_variable: dict[str, str],
) -> bool:
    """Return whether a Stage 4 parameter survives the locked likelihood choices."""
    activation_families = parameter.get("activation_distribution_families")
    if not isinstance(activation_families, list) or not activation_families:
        return True

    relevant_variables = parameter.get("activation_indicator_names")
    if not isinstance(relevant_variables, list) or not relevant_variables:
        relevant_variables = parameter.get("indicator_names")
    if not isinstance(relevant_variables, list) or not relevant_variables:
        relevant_variables = list(chosen_distribution_by_variable)

    allowed_families = {str(family) for family in activation_families}
    return any(
        chosen_distribution_by_variable.get(str(variable)) in allowed_families
        for variable in relevant_variables
    )


def build_stage4_parameter_surface_index(
    causal_spec: dict,
    skeleton: Stage4Skeleton,
) -> Stage4ParameterSurfaceIndex:
    """Build the typed Stage 4 parameter surfaces from the compiler inventory."""
    model_interval_days = get_construct_dt_days(causal_spec)
    lagged_edges = {
        (edge["cause"], edge["effect"])
        for edge in get_estimation_edges(causal_spec)
        if edge.get("lagged", True)
    }
    surfaces = tuple(
        _build_parameter_surface(
            parameter,
            model_interval_days=model_interval_days,
            lagged_edges=lagged_edges,
        )
        for parameter in skeleton.all_params
    )
    return Stage4ParameterSurfaceIndex(
        surfaces=surfaces,
        by_name={surface.name: surface for surface in surfaces},
        by_block_kind={
            kind: tuple(surface for surface in surfaces if surface.block_kind == kind)
            for kind in (
                "measurement_prior",
                "observation_prior",
                "dynamics_prior",
                "effect_prior",
                "correlation_prior",
            )
        },
    )


def _build_parameter_surface(
    parameter: dict[str, Any],
    *,
    model_interval_days: float,
    lagged_edges: set[tuple[str, str]],
) -> Stage4ParameterSurface:
    """Interpret one compiler-authoritative parameter row as a Stage 4 surface."""
    role = str(parameter["role"])

    if role in _DYNAMICS_PARAMETER_ROLES:
        construct_name = str(parameter["construct"])
        return Stage4ParameterSurface(
            parameter=parameter,
            block_kind="dynamics_prior",
            owner_key=construct_name,
            construct_names=(construct_name,),
            structural_context={"construct": construct_name},
        )

    if role == "measurement_error_sd":
        construct_name = str(parameter["construct"])
        indicator_name = str(parameter["indicator"])
        return Stage4ParameterSurface(
            parameter=parameter,
            block_kind="measurement_prior",
            owner_key=construct_name,
            construct_names=(construct_name,),
            indicator_names=(indicator_name,),
            structural_context={
                "construct": construct_name,
                "indicator": indicator_name,
            },
        )

    if role == "fixed_effect":
        cause = str(parameter["cause"])
        effect = str(parameter["effect"])
        lagged = bool(parameter.get("lagged", True))
        return Stage4ParameterSurface(
            parameter=parameter,
            block_kind="effect_prior",
            owner_key=effect,
            construct_names=(cause, effect),
            structural_context={
                "cause": cause,
                "effect": effect,
                "lagged": lagged,
                "expected_lag_days": model_interval_days if lagged else 0.0,
                "feedback_loop": lagged and (effect, cause) in lagged_edges,
            },
        )

    if role == "loading":
        construct_name = str(parameter["construct"])
        indicator_name = str(parameter["indicator"])
        return Stage4ParameterSurface(
            parameter=parameter,
            block_kind="measurement_prior",
            owner_key=construct_name,
            construct_names=(construct_name,),
            indicator_names=(indicator_name,),
            structural_context={
                "construct": construct_name,
                "indicator": indicator_name,
                "reference_indicator": parameter.get("reference_indicator"),
                "indicator_polarity": parameter.get("indicator_polarity"),
            },
        )

    if role in _OBSERVATION_PARAMETER_ROLES:
        construct_names = tuple(
            construct_name
            for construct_name in (parameter.get("construct_names") or ())
            if isinstance(construct_name, str)
        )
        indicator_names = tuple(
            indicator_name
            for indicator_name in (parameter.get("indicator_names") or ())
            if isinstance(indicator_name, str)
        )
        return Stage4ParameterSurface(
            parameter=parameter,
            block_kind="observation_prior",
            owner_key=str(parameter["name"]),
            construct_names=construct_names,
            indicator_names=indicator_names,
            structural_context={
                "indicator_names": list(indicator_names),
                "construct_names": list(construct_names),
                "activation_distribution_families": list(
                    parameter.get("activation_distribution_families") or ()
                ),
            },
        )

    if role in _CORRELATION_PARAMETER_ROLES:
        construct_names = tuple(
            name
            for name in (parameter.get("construct_1"), parameter.get("construct_2"))
            if isinstance(name, str)
        )
        return Stage4ParameterSurface(
            parameter=parameter,
            block_kind="correlation_prior",
            owner_key=str(parameter["name"]),
            construct_names=construct_names,
            structural_context={
                "construct_1": parameter["construct_1"],
                "construct_2": parameter["construct_2"],
                "dependency_kind": parameter["dependency_kind"],
                "source_confounders": parameter["source_confounders"],
            },
        )

    raise ValueError(f"Unsupported Stage 4 parameter role {role!r}")
