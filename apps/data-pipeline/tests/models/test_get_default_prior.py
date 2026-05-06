"""Tests for get_default_prior fallback logic in prior_research."""

import pytest

from causal_ssm_agent.artifacts import (
    ParameterConstraint,
    ParameterRole,
    ParameterSpec,
)
from causal_ssm_agent.distributions import PriorDistributionFamily
from causal_ssm_agent.workers.prior_research import get_default_prior


def _make_param(
    name: str = "beta_x",
    role: ParameterRole = ParameterRole.FIXED_EFFECT,
    constraint: ParameterConstraint = ParameterConstraint.NONE,
) -> ParameterSpec:
    return ParameterSpec(
        name=name,
        role=role,
        constraint=constraint,
        description="test param",
    )


class TestGetDefaultPrior:
    @pytest.mark.parametrize(
        (
            "role",
            "constraint",
            "expected_distribution",
            "expected_params",
        ),
        [
            (
                ParameterRole.FIXED_EFFECT,
                ParameterConstraint.NONE,
                PriorDistributionFamily.NORMAL,
                {"mu": 0.0, "sigma": 0.5},
            ),
            (
                ParameterRole.FIXED_EFFECT,
                ParameterConstraint.POSITIVE,
                PriorDistributionFamily.HALF_NORMAL,
                {"sigma": 1.0},
            ),
            (
                ParameterRole.FIXED_EFFECT,
                ParameterConstraint.UNIT_INTERVAL,
                PriorDistributionFamily.BETA,
                {"alpha": 2.0, "beta": 2.0},
            ),
            (
                ParameterRole.FIXED_EFFECT,
                ParameterConstraint.CORRELATION,
                PriorDistributionFamily.UNIFORM,
                {"lower": -1.0, "upper": 1.0},
            ),
            (
                ParameterRole.RESIDUAL_SD,
                ParameterConstraint.NONE,
                PriorDistributionFamily.HALF_NORMAL,
                {"sigma": 1.0},
            ),
            (
                ParameterRole.STATIC_STATE_SD,
                ParameterConstraint.NONE,
                PriorDistributionFamily.HALF_NORMAL,
                {"sigma": 1.0},
            ),
            (
                ParameterRole.AR_COEFFICIENT,
                ParameterConstraint.CORRELATION,
                PriorDistributionFamily.BETA,
                {"alpha": 2.0, "beta": 2.0},
            ),
        ],
        ids=[
            "unconstrained-normal",
            "positive-half-normal",
            "unit-interval-beta",
            "correlation-uniform",
            "residual-sd-role",
            "static-state-sd-role",
            "ar-role",
        ],
    )
    def test_distribution_selection(
        self,
        role: ParameterRole,
        constraint: ParameterConstraint,
        expected_distribution: PriorDistributionFamily,
        expected_params: dict[str, float],
    ):
        p = _make_param(role=role, constraint=constraint)
        result = get_default_prior(p)
        assert result.distribution == expected_distribution
        assert result.params == expected_params

    def test_parameter_name_propagated(self):
        p = _make_param(name="sigma_residual")
        result = get_default_prior(p)
        assert result.parameter == "sigma_residual"

    def test_sources_empty(self):
        p = _make_param()
        result = get_default_prior(p)
        assert result.sources == []

    def test_reasoning_includes_role(self):
        p = _make_param(role=ParameterRole.AR_COEFFICIENT)
        result = get_default_prior(p)
        assert "ar_coefficient" in result.reasoning
