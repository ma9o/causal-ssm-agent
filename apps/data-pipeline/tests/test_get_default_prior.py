"""Tests for get_default_prior fallback logic in prior_research."""

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
    def test_unconstrained_returns_normal(self):
        p = _make_param(constraint=ParameterConstraint.NONE)
        result = get_default_prior(p)
        assert result.distribution == PriorDistributionFamily.NORMAL
        assert result.params == {"mu": 0.0, "sigma": 0.5}

    def test_positive_constraint_returns_half_normal(self):
        p = _make_param(constraint=ParameterConstraint.POSITIVE)
        result = get_default_prior(p)
        assert result.distribution == PriorDistributionFamily.HALF_NORMAL
        assert result.params == {"sigma": 1.0}

    def test_unit_interval_returns_beta(self):
        p = _make_param(constraint=ParameterConstraint.UNIT_INTERVAL)
        result = get_default_prior(p)
        assert result.distribution == PriorDistributionFamily.BETA
        assert result.params == {"alpha": 2.0, "beta": 2.0}

    def test_correlation_constraint_returns_uniform(self):
        p = _make_param(constraint=ParameterConstraint.CORRELATION)
        result = get_default_prior(p)
        assert result.distribution == PriorDistributionFamily.UNIFORM
        assert result.params == {"lower": -1.0, "upper": 1.0}

    def test_residual_sd_role_overrides_constraint(self):
        # Even with NONE constraint, RESIDUAL_SD role forces HalfNormal
        p = _make_param(
            role=ParameterRole.RESIDUAL_SD,
            constraint=ParameterConstraint.NONE,
        )
        result = get_default_prior(p)
        assert result.distribution == PriorDistributionFamily.HALF_NORMAL
        assert result.params == {"sigma": 1.0}

    def test_static_state_sd_role_overrides_constraint(self):
        p = _make_param(
            role=ParameterRole.STATIC_STATE_SD,
            constraint=ParameterConstraint.NONE,
        )
        result = get_default_prior(p)
        assert result.distribution == PriorDistributionFamily.HALF_NORMAL
        assert result.params == {"sigma": 1.0}

    def test_ar_role_overrides_correlation_constraint(self):
        p = _make_param(
            role=ParameterRole.AR_COEFFICIENT,
            constraint=ParameterConstraint.CORRELATION,
        )
        result = get_default_prior(p)
        assert result.distribution == PriorDistributionFamily.BETA
        assert result.params == {"alpha": 2.0, "beta": 2.0}

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
