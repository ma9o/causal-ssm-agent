"""Tests for model spec merging, dict validation, and DistributionFamily."""

import pytest

from causal_ssm_agent.orchestrator.schemas_model import (
    DistributionChoice,
    DistributionFamily,
    LinkFunction,
    ModelSpecDecisions,
    ParameterConstraint,
    merge_decisions_to_spec,
    validate_model_spec_decisions_dict,
    validate_model_spec_dict,
)


def _valid_spec_dict():
    """Minimal valid ModelSpec dict."""
    return {
        "likelihoods": [
            {
                "variable": "mood_score",
                "distribution": "gaussian",
                "link": "identity",
                "reasoning": "Continuous variable",
            }
        ],
        "parameters": [
            {
                "name": "beta_stress_mood",
                "role": "fixed_effect",
                "constraint": "none",
                "description": "Effect of stress on mood",
            }
        ],
        "reasoning": "Standard linear model",
    }


# =============================================================================
# validate_model_spec_dict
# =============================================================================


class TestValidateModelSpecDict:
    def test_valid_dict(self):
        spec, errors = validate_model_spec_dict(_valid_spec_dict())
        assert spec is not None
        assert errors == []

    def test_not_dict(self):
        spec, errors = validate_model_spec_dict("not a dict")
        assert spec is None
        assert any("dictionary" in e.lower() for e in errors)

    def test_invalid_distribution(self):
        d = _valid_spec_dict()
        d["likelihoods"][0]["distribution"] = "nonexistent"
        spec, errors = validate_model_spec_dict(d)
        assert spec is None
        assert any("distribution" in e and "nonexistent" in e for e in errors)

    def test_invalid_link(self):
        d = _valid_spec_dict()
        d["likelihoods"][0]["link"] = "bad_link"
        spec, errors = validate_model_spec_dict(d)
        assert spec is None
        assert any("link" in e and "bad_link" in e for e in errors)

    def test_distribution_link_incompatible(self):
        d = _valid_spec_dict()
        d["likelihoods"][0]["distribution"] = "gaussian"
        d["likelihoods"][0]["link"] = "log"
        spec, errors = validate_model_spec_dict(d)
        assert spec is None
        assert any("link" in e and "invalid" in e for e in errors)

    def test_invalid_role(self):
        d = _valid_spec_dict()
        d["parameters"][0]["role"] = "bad_role"
        spec, errors = validate_model_spec_dict(d)
        assert spec is None
        assert any("role" in e and "bad_role" in e for e in errors)

    def test_invalid_constraint(self):
        d = _valid_spec_dict()
        d["parameters"][0]["constraint"] = "bad_constraint"
        spec, errors = validate_model_spec_dict(d)
        assert spec is None
        assert any("constraint" in e and "bad_constraint" in e for e in errors)

    def test_duplicate_likelihood(self):
        d = _valid_spec_dict()
        d["likelihoods"].append(d["likelihoods"][0].copy())
        spec, errors = validate_model_spec_dict(d)
        assert spec is None
        assert any("duplicate" in e for e in errors)

    def test_duplicate_parameter(self):
        d = _valid_spec_dict()
        d["parameters"].append(d["parameters"][0].copy())
        spec, errors = validate_model_spec_dict(d)
        assert spec is None
        assert any("duplicate" in e for e in errors)

    def test_dtype_check_with_indicators(self):
        d = _valid_spec_dict()
        indicators = [{"name": "mood_score", "measurement_dtype": "binary"}]
        spec, errors = validate_model_spec_dict(d, indicators=indicators)
        assert spec is None
        assert any("dtype" in e for e in errors)

    def test_missing_indicator_coverage(self):
        d = _valid_spec_dict()
        indicators = [
            {"name": "mood_score", "measurement_dtype": "continuous"},
            {"name": "extra_var", "measurement_dtype": "continuous"},
        ]
        spec, errors = validate_model_spec_dict(d, indicators=indicators)
        assert spec is None
        assert any("extra_var" in e for e in errors)

    def test_likelihoods_not_list(self):
        d = _valid_spec_dict()
        d["likelihoods"] = "bad"
        spec, errors = validate_model_spec_dict(d)
        assert spec is None
        assert any("list" in e for e in errors)

    def test_parameters_not_list(self):
        d = _valid_spec_dict()
        d["parameters"] = "bad"
        spec, errors = validate_model_spec_dict(d)
        assert spec is None
        assert any("list" in e for e in errors)

    def test_likelihood_not_dict(self):
        d = _valid_spec_dict()
        d["likelihoods"].append(42)
        spec, errors = validate_model_spec_dict(d)
        assert spec is None
        assert any("dictionary" in e.lower() for e in errors)

    def test_parameter_not_dict(self):
        d = _valid_spec_dict()
        d["parameters"].append("bad")
        spec, errors = validate_model_spec_dict(d)
        assert spec is None
        assert any("dictionary" in e.lower() for e in errors)

    def test_role_constraint_mismatch(self):
        d = _valid_spec_dict()
        d["parameters"][0]["role"] = "residual_sd"
        d["parameters"][0]["constraint"] = "none"  # should be positive
        spec, errors = validate_model_spec_dict(d)
        assert spec is None
        assert any("constraint" in e and "unexpected" in e for e in errors)

    def test_ar_role_requires_unit_interval(self):
        d = _valid_spec_dict()
        d["parameters"][0] = {
            "name": "rho_mood",
            "role": "ar_coefficient",
            "constraint": "correlation",
            "description": "AR(1) persistence for mood",
        }
        spec, errors = validate_model_spec_dict(d)
        assert spec is None
        assert any("expected 'unit_interval'" in e for e in errors)

    def test_initial_state_correlation_requires_canonical_prefix(self):
        d = _valid_spec_dict()
        d["parameters"][0] = {
            "name": "init_cor_mood_sleep",
            "role": "initial_state_correlation",
            "constraint": "correlation",
            "description": "legacy alias should be rejected",
        }
        spec, errors = validate_model_spec_dict(d)
        assert spec is None
        assert any("must use canonical names starting with 'cor0_'" in e for e in errors)

    def test_loading_role_accepts_negative_constraint(self):
        d = _valid_spec_dict()
        d["parameters"][0] = {
            "name": "lambda_sleep_problem_search_count_sleep_quality",
            "role": "loading",
            "constraint": "negative",
            "description": "Inverse proxy loading for sleep problems on sleep quality",
        }
        spec, errors = validate_model_spec_dict(d)
        assert errors == []
        assert spec is not None
        assert spec.parameters[0].constraint == ParameterConstraint.NEGATIVE

    def test_initial_state_mean_role_accepts_none_constraint(self):
        d = _valid_spec_dict()
        d["parameters"][0] = {
            "name": "t0_mean_mood",
            "role": "initial_state_mean",
            "constraint": "none",
            "description": "Initial state mean for mood",
        }
        spec, errors = validate_model_spec_dict(d)
        assert errors == []
        assert spec is not None
        assert spec.parameters[0].constraint == ParameterConstraint.NONE

    def test_initial_state_sd_role_requires_positive_constraint(self):
        d = _valid_spec_dict()
        d["parameters"][0] = {
            "name": "t0_sd_mood",
            "role": "initial_state_sd",
            "constraint": "none",
            "description": "Initial state SD for mood",
        }
        spec, errors = validate_model_spec_dict(d)
        assert spec is None
        assert any("expected 'positive'" in e for e in errors)


# =============================================================================
# merge_decisions_to_spec
# =============================================================================


class TestMergeDecisionsToSpec:
    def test_basic_merge(self):
        resolved = [
            {
                "variable": "mood_score",
                "distribution": "gaussian",
                "link": "identity",
                "reasoning": "continuous",
            },
        ]
        params = [
            {
                "name": "beta_x",
                "role": "fixed_effect",
                "constraint": "none",
                "description": "Effect of X",
            },
        ]
        decisions = ModelSpecDecisions(
            distribution_choices=[],
            reasoning="Standard model",
        )
        spec, errors = merge_decisions_to_spec(resolved, params, decisions)
        assert errors == []
        assert spec is not None
        assert len(spec.likelihoods) == 1
        assert spec.likelihoods[0].variable == "mood_score"

    def test_with_distribution_choices(self):
        resolved = []
        params = [
            {
                "name": "beta_x",
                "role": "fixed_effect",
                "constraint": "none",
                "description": "test",
            },
        ]
        decisions = ModelSpecDecisions(
            distribution_choices=[
                DistributionChoice(
                    variable="steps",
                    distribution=DistributionFamily.POISSON,
                    link=LinkFunction.LOG,
                    reasoning="Count data",
                ),
            ],
            reasoning="model reason",
        )
        spec, errors = merge_decisions_to_spec(resolved, params, decisions)
        assert errors == []
        assert spec is not None
        assert len(spec.likelihoods) == 1
        assert spec.likelihoods[0].distribution == DistributionFamily.POISSON

    def test_resolved_and_choices_combined(self):
        resolved = [
            {
                "variable": "mood",
                "distribution": "gaussian",
                "link": "identity",
                "reasoning": "continuous",
            },
        ]
        params = [
            {
                "name": "p1",
                "role": "fixed_effect",
                "constraint": "none",
                "description": "d1",
            },
        ]
        decisions = ModelSpecDecisions(
            distribution_choices=[
                DistributionChoice(
                    variable="steps",
                    distribution=DistributionFamily.POISSON,
                    link=LinkFunction.LOG,
                    reasoning="count",
                ),
            ],
            reasoning="test",
        )
        spec, errors = merge_decisions_to_spec(resolved, params, decisions)
        assert errors == []
        assert spec is not None
        assert len(spec.likelihoods) == 2
        variables = {lik.variable for lik in spec.likelihoods}
        assert variables == {"mood", "steps"}

    def test_merge_filters_inactive_conditional_observation_parameters(self):
        resolved = [
            {
                "variable": "steps",
                "distribution": "gaussian",
                "link": "identity",
                "reasoning": "resolved for test",
            },
        ]
        params = [
            {
                "name": "obs_df",
                "role": "observation_hyperparameter_positive",
                "constraint": "positive",
                "description": "Student-t observation degrees of freedom",
                "indicator_names": ["steps"],
                "activation_indicator_names": ["steps"],
                "activation_distribution_families": ["student_t"],
            },
            {
                "name": "obs_r",
                "role": "observation_hyperparameter_positive",
                "constraint": "positive",
                "description": "Negative-binomial observation dispersion",
                "indicator_names": ["steps"],
                "activation_indicator_names": ["steps"],
                "activation_distribution_families": ["negative_binomial"],
            },
            {
                "name": "beta_x",
                "role": "fixed_effect",
                "constraint": "none",
                "description": "Effect of X",
            },
        ]
        decisions = ModelSpecDecisions(
            distribution_choices=[],
            reasoning="test",
        )

        spec, errors = merge_decisions_to_spec(resolved, params, decisions)

        assert errors == []
        assert spec is not None
        assert [parameter.name for parameter in spec.parameters] == ["beta_x"]


# =============================================================================
# validate_model_spec_decisions_dict
# =============================================================================


class TestValidateModelSpecDecisionsDict:
    def _ambiguous(self):
        return [{"variable": "steps", "dtype": "count"}]

    def _resolved(self):
        return [
            {
                "variable": "mood",
                "distribution": "gaussian",
                "link": "identity",
                "reasoning": "continuous",
            }
        ]

    def _params(self):
        return [
            {
                "name": "p1",
                "role": "fixed_effect",
                "constraint": "none",
                "description": "d",
            }
        ]

    def test_valid_decisions(self):
        data = {
            "distribution_choices": [
                {
                    "variable": "steps",
                    "distribution": "poisson",
                    "link": "log",
                    "reasoning": "count",
                },
            ],
            "reasoning": "test",
        }
        spec, errors = validate_model_spec_decisions_dict(
            data, self._resolved(), self._ambiguous(), self._params()
        )
        assert errors == []
        assert spec is not None

    def test_not_dict(self):
        spec, errors = validate_model_spec_decisions_dict(
            "bad", self._resolved(), self._ambiguous(), self._params()
        )
        assert spec is None
        assert any("dictionary" in e.lower() for e in errors)

    def test_missing_ambiguous_decision(self):
        data = {
            "distribution_choices": [],  # missing decision for 'steps'
            "reasoning": "test",
        }
        spec, errors = validate_model_spec_decisions_dict(
            data, self._resolved(), self._ambiguous(), self._params()
        )
        assert spec is None
        assert any("steps" in e for e in errors)

    def test_invalid_distribution_in_choices(self):
        data = {
            "distribution_choices": [
                {"variable": "steps", "distribution": "bad_dist", "link": "log", "reasoning": "r"},
            ],
            "reasoning": "test",
        }
        spec, errors = validate_model_spec_decisions_dict(
            data, self._resolved(), self._ambiguous(), self._params()
        )
        assert spec is None
        assert any("bad_dist" in e for e in errors)

    def test_no_ambiguous_indicators(self):
        """When no ambiguous indicators, no distribution_choices needed."""
        data = {
            "distribution_choices": [],
            "reasoning": "test",
        }
        spec, errors = validate_model_spec_decisions_dict(
            data, self._resolved(), [], self._params()
        )
        assert errors == []
        assert spec is not None


# =============================================================================
# DistributionFamily exact construction
# =============================================================================


class TestDistributionFamilyConstruction:
    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            DistributionFamily(42)
        with pytest.raises(ValueError):
            DistributionFamily("")
        with pytest.raises(ValueError):
            DistributionFamily("not_a_distribution")
        with pytest.raises(ValueError):
            DistributionFamily("GAUSSIAN")
        with pytest.raises(ValueError):
            DistributionFamily("Normal")
        with pytest.raises(ValueError):
            DistributionFamily("negative binomial")
