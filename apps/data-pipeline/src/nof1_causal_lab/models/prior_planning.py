"""Pure authoring-prior planning before SSM compilation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.prior import (
    ExecutablePrior,
    PriorPlan,
    prior_params_model,
)
from nof1_causal_lab.artifacts.statistical_model_spec import (
    ParameterConstraint,
    ParameterRole,
    ParameterSpec,
    StatisticalModelSpec,
)
from nof1_causal_lab.distributions import PriorDistributionFamily

if TYPE_CHECKING:
    from collections.abc import Iterable


def default_executable_prior(parameter: ParameterSpec) -> ExecutablePrior:
    """Choose the explicit authoring default for one semantic parameter."""
    if parameter.role == ParameterRole.AR_COEFFICIENT:
        distribution = PriorDistributionFamily.BETA
        params = {"alpha": 2.0, "beta": 2.0}
    elif parameter.role == ParameterRole.LOADING:
        distribution = PriorDistributionFamily.NORMAL
        params = {
            "mu": -0.5 if parameter.constraint == ParameterConstraint.NEGATIVE else 0.5,
            "sigma": 0.5,
        }
    elif parameter.constraint == ParameterConstraint.POSITIVE:
        distribution = PriorDistributionFamily.HALF_NORMAL
        params = {"sigma": 1.0}
    elif parameter.constraint == ParameterConstraint.NEGATIVE:
        distribution = PriorDistributionFamily.TRUNCATED_NORMAL
        params = {"mu": -1.0, "sigma": 0.5, "lower": -5.0, "upper": 0.0}
    elif parameter.constraint == ParameterConstraint.UNIT_INTERVAL:
        distribution = PriorDistributionFamily.BETA
        params = {"alpha": 2.0, "beta": 2.0}
    elif parameter.constraint == ParameterConstraint.CORRELATION:
        distribution = PriorDistributionFamily.UNIFORM
        params = {"lower": -1.0, "upper": 1.0}
    else:
        distribution = PriorDistributionFamily.NORMAL
        params = {"mu": 0.0, "sigma": 0.5}

    if parameter.role in (ParameterRole.RESIDUAL_SD, ParameterRole.STATIC_STATE_SD):
        distribution = PriorDistributionFamily.HALF_NORMAL
        params = {"sigma": 1.0}

    return ExecutablePrior(
        parameter=parameter.name,
        distribution=distribution,
        params=prior_params_model(distribution, params),
    )


def build_default_prior_plan(statistical_model_spec: StatisticalModelSpec) -> PriorPlan:
    """Build a complete explicit plan from compiler-independent authoring defaults."""
    return PriorPlan(
        priors={
            parameter.name: default_executable_prior(parameter)
            for parameter in statistical_model_spec.parameters
        }
    )


def build_prior_plan(
    statistical_model_spec: StatisticalModelSpec,
    authored_priors: Iterable[ExecutablePrior],
) -> PriorPlan:
    """Overlay typed authored priors on a complete explicit default plan."""
    planned = dict(build_default_prior_plan(statistical_model_spec).priors)
    parameter_names = set(planned)
    for prior in authored_priors:
        if prior.parameter not in parameter_names:
            raise ValueError(
                f"Prior {prior.parameter!r} does not correspond to StatisticalModelSpec."
            )
        planned[prior.parameter] = prior
    return PriorPlan(priors=planned)


__all__ = [
    "build_default_prior_plan",
    "build_prior_plan",
    "default_executable_prior",
]
