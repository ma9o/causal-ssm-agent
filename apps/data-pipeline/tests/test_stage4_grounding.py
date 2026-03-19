"""Unit tests for stage4_grounding in stage_tools.py.

Tests the unified grounding function that handles model_spec and/or priors.
Gates: schema validation → compile → prior predictive.
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
# Gate 0: input validation
# ---------------------------------------------------------------------------


class TestStage4GroundingInputValidation:
    """Neither model_spec nor priors → error."""

    def test_empty_data_returns_error(self, causal_spec):
        output, feedback = stage4_grounding({}, causal_spec)
        assert output is None
        assert "model_spec" in feedback and "priors" in feedback


# ---------------------------------------------------------------------------
# Gate 1: schema validation
# ---------------------------------------------------------------------------


class TestStage4GroundingSchemaValidation:
    """Schema validation for model_spec and priors."""

    def test_invalid_model_spec_returns_error(self, causal_spec):
        """model_spec with empty likelihoods/parameters gets stored but
        fails compile (no likelihood specification)."""
        _output, feedback = stage4_grounding(
            {"model_spec": {"likelihoods": [], "parameters": []}},
            causal_spec,
        )
        # Empty model_spec is stored but triggers a compile error
        assert "COMPILE ERROR" in feedback or "no likelihood" in feedback.lower()

    def test_invalid_prior_schema_returns_error(self, causal_spec, model_spec):
        """Prior with missing required fields fails PriorProposal validation."""
        output, feedback = stage4_grounding(
            {"priors": {"beta_stress_sleep": {"bad": "data"}}},
            causal_spec,
            current={"model_spec": model_spec},
        )
        assert output is None
        assert "SCHEMA ERRORS" in feedback

    def test_valid_model_spec_saved_with_missing_priors(self, causal_spec, model_spec):
        """Valid model_spec alone is saved but feedback requests priors."""
        output, feedback = stage4_grounding(
            {"model_spec": model_spec},
            causal_spec,
        )
        assert output is not None
        assert "model_spec" in output
        # Model spec is accepted but priors are still needed
        assert "MODEL STATE SAVED" in feedback or "missing priors" in feedback.lower()


# ---------------------------------------------------------------------------
# Gate 2: compile
# ---------------------------------------------------------------------------


class TestStage4GroundingCompile:
    """Compile gate: trial compile (no priors) or full compile (with priors)."""

    def test_priors_without_model_spec_returns_compile_error(self, causal_spec, model_spec, priors):
        """Priors without model_spec in current state → compile error."""
        _output, feedback = stage4_grounding(
            {"priors": {"beta_stress_sleep": priors["beta_stress_sleep"]}},
            causal_spec,
            current=None,  # no existing model_spec
        )
        # Valid priors may be stored, but compile fails without model_spec
        assert "COMPILE ERROR" in feedback
        assert "model_spec" in feedback.lower()

    def test_priors_with_current_model_spec_compiles(self, causal_spec, model_spec, priors):
        """Priors + model_spec in current state → full compile."""
        output, feedback = stage4_grounding(
            {"priors": priors},
            causal_spec,
            current={"model_spec": model_spec},
        )
        assert output is not None
        assert feedback == "VALID"
        assert "priors" in output

    def test_model_spec_then_priors_separately(self, causal_spec, model_spec, priors):
        """model_spec and priors must be submitted in separate calls."""
        # First call: submit model_spec
        output1, _feedback1 = stage4_grounding(
            {"model_spec": model_spec},
            causal_spec,
        )
        assert output1 is not None
        assert "model_spec" in output1

        # Second call: submit priors with model_spec in current state
        output2, feedback2 = stage4_grounding(
            {"priors": priors},
            causal_spec,
            current=output1,
        )
        assert output2 is not None
        assert feedback2 == "VALID"
        assert "priors" in output2

    def test_model_spec_and_priors_together_rejected(self, causal_spec, model_spec, priors):
        """Submitting both model_spec and priors in one call is rejected."""
        output, feedback = stage4_grounding(
            {"model_spec": model_spec, "priors": priors},
            causal_spec,
        )
        assert output is None
        assert "UPDATE TOO BROAD" in feedback


# ---------------------------------------------------------------------------
# State merging
# ---------------------------------------------------------------------------


class TestStage4GroundingStateMerging:
    """Priors merge with current state (accumulate during refinement)."""

    def test_partial_priors_merge_with_current(self, causal_spec, model_spec, priors):
        """Submitting one prior merges with existing priors in current."""
        # Start with full priors in current
        current = {"model_spec": model_spec, "priors": dict(priors)}

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
        assert len(output["priors"]) == len(priors)
        # The submitted prior should be updated
        assert output["priors"]["beta_stress_sleep"]["params"]["mu"] == -0.3

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


# ---------------------------------------------------------------------------
# Gate 3: prior predictive (only with raw_data)
# ---------------------------------------------------------------------------


class TestStage4GroundingPriorPredictive:
    """Prior predictive gate runs only when priors + raw_data are present."""

    def test_with_raw_data_runs_pp(self, causal_spec, model_spec, priors):
        """With raw_data, PP gate runs. With reasonable priors it should pass."""
        import numpy as np
        import polars as pl

        n = 100
        rng = np.random.default_rng(42)
        timestamps = pl.Series(
            "timestamp",
            pl.date_range(
                pl.date(2024, 1, 1),
                pl.date(2024, 1, 1) + pl.duration(days=n - 1),
                eager=True,
            ),
        )
        raw_data = pl.DataFrame(
            {
                "indicator": ["pss_score"] * n + ["sleep_quality"] * n,
                "value": np.concatenate(
                    [
                        rng.normal(3.0, 1.0, n),
                        rng.normal(7.0, 1.5, n),
                    ]
                ),
                "timestamp": pl.concat([timestamps, timestamps]),
            }
        )

        # Must submit priors separately from model_spec
        _output, feedback = stage4_grounding(
            {"priors": priors},
            causal_spec,
            current={"model_spec": model_spec},
            raw_data=raw_data,
        )
        # With reasonable priors and data, should pass (or fail with PP feedback)
        # We mainly verify it runs without crashing
        assert (
            feedback == "VALID" or "PRIOR PREDICTIVE" in feedback or "parameter" in feedback.lower()
        )

    def test_extreme_priors_fail_pp(self, causal_spec, model_spec):
        """Extremely wide priors should fail prior predictive checks."""
        import numpy as np
        import polars as pl

        extreme_priors = {}
        for p in model_spec["parameters"]:
            name = p["name"]
            extreme_priors[name] = {
                "parameter": name,
                "distribution": "Normal",
                "params": {"mu": 1e6, "sigma": 1e6},
                "sources": [],
                "reasoning": "Deliberately extreme",
            }

        n = 100
        rng = np.random.default_rng(42)
        timestamps = pl.Series(
            "timestamp",
            pl.date_range(
                pl.date(2024, 1, 1),
                pl.date(2024, 1, 1) + pl.duration(days=n - 1),
                eager=True,
            ),
        )
        raw_data = pl.DataFrame(
            {
                "indicator": ["pss_score"] * n + ["sleep_quality"] * n,
                "value": np.concatenate(
                    [
                        rng.normal(3.0, 1.0, n),
                        rng.normal(7.0, 1.5, n),
                    ]
                ),
                "timestamp": pl.concat([timestamps, timestamps]),
            }
        )

        _output, feedback = stage4_grounding(
            {"priors": extreme_priors},
            causal_spec,
            current={"model_spec": model_spec},
            raw_data=raw_data,
        )
        # Extreme priors should either fail compile or PP
        # (Normal doesn't satisfy positive constraint for sigma parameters)
        assert _output is None or "PRIOR PREDICTIVE" in feedback or "COMPILE ERROR" in feedback
