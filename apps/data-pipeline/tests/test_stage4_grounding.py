"""Focused tests for stage4_grounding in stage_tools.py.

The broad Stage 4 grounding contract lives in ``test_stage4.py``. This file
keeps only the direct call-path branches that are not already covered there:
missing input, missing state, and merge behavior across incremental updates.
"""

import pytest

from causal_ssm_agent.flows.stages.stage_tools import stage4_grounding

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_causal_spec() -> dict:
    """Minimal causal spec: stress→sleep, two continuous indicators."""
    return {
        "latent": {
            "constructs": [
                {
                    "name": "stress",
                    "role": "exogenous",
                    "temporal_status": "time_varying",
                    "temporal_scale": "daily",
                },
                {
                    "name": "sleep",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                    "temporal_scale": "daily",
                    "is_outcome": True,
                },
            ],
            "edges": [{"cause": "stress", "effect": "sleep"}],
        },
        "measurement": {
            "indicators": [
                {
                    "name": "pss_score",
                    "construct_name": "stress",
                    "measurement_dtype": "continuous",
                    "how_to_measure": "PSS score",
                    "aggregation": "mean",
                },
                {
                    "name": "sleep_quality",
                    "construct_name": "sleep",
                    "measurement_dtype": "continuous",
                    "how_to_measure": "Sleep quality rating",
                    "aggregation": "mean",
                },
            ],
        },
    }


def _make_model_spec() -> dict:
    """Minimal model spec matching the causal spec above."""
    return {
        "likelihoods": [
            {
                "variable": "pss_score",
                "distribution": "gaussian",
                "link": "identity",
                "reasoning": "Continuous score",
            },
            {
                "variable": "sleep_quality",
                "distribution": "gaussian",
                "link": "identity",
                "reasoning": "Continuous score",
            },
        ],
        "parameters": [
            {
                "name": "rho_stress",
                "role": "ar_coefficient",
                "constraint": "unit_interval",
                "description": "AR(1) for stress",
            },
            {
                "name": "rho_sleep",
                "role": "ar_coefficient",
                "constraint": "unit_interval",
                "description": "AR(1) for sleep",
            },
            {
                "name": "beta_stress_sleep",
                "role": "fixed_effect",
                "constraint": "none",
                "description": "Effect of stress on sleep",
            },
            {
                "name": "sigma_stress",
                "role": "residual_sd",
                "constraint": "positive",
                "description": "Residual SD for stress",
            },
            {
                "name": "sigma_sleep",
                "role": "residual_sd",
                "constraint": "positive",
                "description": "Residual SD for sleep",
            },
        ],
    }


def _make_priors(model_spec: dict) -> dict[str, dict]:
    """Default priors for each parameter in the model spec."""
    priors = {}
    for p in model_spec["parameters"]:
        name = p["name"]
        if "rho" in name:
            priors[name] = {
                "parameter": name,
                "distribution": "Beta",
                "params": {"alpha": 2.0, "beta": 2.0},
                "sources": [],
                "reasoning": "Weakly informative AR prior",
            }
        elif "sigma" in name:
            priors[name] = {
                "parameter": name,
                "distribution": "HalfNormal",
                "params": {"sigma": 1.0},
                "sources": [],
                "reasoning": "Weakly informative SD prior",
            }
        else:
            priors[name] = {
                "parameter": name,
                "distribution": "Normal",
                "params": {"mu": 0.0, "sigma": 0.5},
                "sources": [],
                "reasoning": "Weakly informative effect prior",
            }
    return priors


@pytest.fixture
def causal_spec():
    return _make_causal_spec()


@pytest.fixture
def model_spec():
    return _make_model_spec()


@pytest.fixture
def priors(model_spec):
    return _make_priors(model_spec)


# ---------------------------------------------------------------------------
# Check 0: input validation
# ---------------------------------------------------------------------------


class TestStage4GroundingInputValidation:
    """Neither model_spec nor priors → error."""

    def test_empty_data_returns_error(self, causal_spec):
        output, feedback = stage4_grounding({}, causal_spec)
        assert output is None
        assert "model_spec" in feedback and "priors" in feedback


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestStage4GroundingSchemaValidation:
    """Schema branches that are only exercised through stage4_grounding itself."""

    def test_invalid_prior_distribution_returns_error(self, causal_spec, model_spec):
        """Unknown prior family should fail schema validation immediately."""
        output, feedback = stage4_grounding(
            {
                "priors": {
                    "beta_stress_sleep": {
                        "parameter": "beta_stress_sleep",
                        "distribution": "Cauchy",
                        "params": {"mu": 0.0, "sigma": 1.0},
                        "sources": [],
                        "reasoning": "bad family",
                    }
                }
            },
            causal_spec,
            current={"model_spec": model_spec},
        )
        assert output is None
        assert "SCHEMA ERRORS" in feedback


# ---------------------------------------------------------------------------
# Missing state
# ---------------------------------------------------------------------------


class TestStage4GroundingMissingState:
    """Compile guidance when incremental updates arrive before model state exists."""

    def test_priors_without_model_spec_returns_compile_error(self, causal_spec, priors):
        """Priors without model_spec in current state should fail with guidance."""
        _output, feedback = stage4_grounding(
            {"priors": {"beta_stress_sleep": priors["beta_stress_sleep"]}},
            causal_spec,
            current=None,
        )
        assert "COMPILE ERROR" in feedback
        assert "model_spec" in feedback.lower()


# ---------------------------------------------------------------------------
# State merging
# ---------------------------------------------------------------------------


class TestStage4GroundingStateMerging:
    """Priors merge with current state (accumulate during refinement)."""

    def test_partial_priors_merge_with_current(self, causal_spec, model_spec, priors):
        """Submitting one prior merges with existing priors in current."""
        # Start with full priors in current
        current = {"model_spec": model_spec, "authored_priors": dict(priors)}

        # Submit a changed single prior
        new_beta = {
            "parameter": "beta_stress_sleep",
            "distribution": "Normal",
            "params": {"mu": -0.3, "sigma": 0.1},
            "sources": [],
            "reasoning": "Updated based on evidence",
        }
        output, feedback = stage4_grounding(
            {"priors": {"beta_stress_sleep": new_beta}},
            causal_spec,
            current=current,
        )
        assert output is not None
        assert feedback == "VALID"

        # Output should have ALL priors (merged), not just the submitted one
        assert len(output["authored_priors"]) == len(priors)
        # The submitted prior should be updated
        assert output["authored_priors"]["beta_stress_sleep"]["params"]["mu"] == -0.3

    def test_new_model_spec_replaces_current(self, causal_spec, model_spec):
        """Submitting model_spec replaces the one in current."""
        old_spec = {**model_spec, "extra_field": "old"}
        current = {"model_spec": old_spec}

        output, feedback = stage4_grounding(
            {"model_spec": model_spec},
            causal_spec,
            current=current,
        )
        assert output is not None
        # Model spec accepted but priors still needed
        assert "MODEL STATE SAVED" in feedback or "missing priors" in feedback.lower()
        assert "extra_field" not in output["model_spec"]
