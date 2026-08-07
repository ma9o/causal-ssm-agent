"""Tests for explicit compiler-independent prior defaults."""

import pytest
from pydantic import ValidationError

from nof1_causal_lab.artifacts import (
    ParameterConstraint,
    ParameterRole,
    ParameterSpec,
)
from nof1_causal_lab.artifacts.prior import ExecutablePrior, ScalePriorParams
from nof1_causal_lab.distributions import PriorDistributionFamily
from nof1_causal_lab.models.prior_planning import default_executable_prior
from nof1_causal_lab.workers.schemas_prior import PriorProposal


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


class TestDefaultExecutablePrior:
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
            (
                ParameterRole.LOADING,
                ParameterConstraint.POSITIVE,
                PriorDistributionFamily.NORMAL,
                {"mu": 0.5, "sigma": 0.5},
            ),
            (
                ParameterRole.LOADING,
                ParameterConstraint.NEGATIVE,
                PriorDistributionFamily.NORMAL,
                {"mu": -0.5, "sigma": 0.5},
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
            "positive-loading-pooled-family",
            "negative-loading-pooled-family",
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
        result = default_executable_prior(p)
        assert result.distribution == expected_distribution
        assert result.params.model_dump() == expected_params

    def test_parameter_name_propagated(self):
        p = _make_param(name="sigma_residual")
        result = default_executable_prior(p)
        assert result.parameter == "sigma_residual"

    def test_returns_compiler_facing_prior(self):
        p = _make_param()
        result = default_executable_prior(p)
        assert isinstance(result, ExecutablePrior)

    def test_prior_parameters_must_match_the_declared_family(self):
        with pytest.raises(ValidationError, match="LocationScalePriorParams"):
            PriorProposal(
                parameter="beta_x",
                distribution=PriorDistributionFamily.NORMAL,
                params=ScalePriorParams(sigma=1.0),
                reasoning="invalid family/parameter pairing",
            )
