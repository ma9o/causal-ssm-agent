"""Tests for Stage 4 grounding helpers in ``stage_tools.py``.

This file owns the direct grounding call paths:
- ``stage4_grounding`` validation, merge, and feedback behavior
- ``_agentic_stage4_grounding`` delta application behavior
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
        "estimation": {
            "state_order": ["stress", "sleep"],
            "edges": [
                {
                    "cause": "stress",
                    "effect": "sleep",
                    "description": "Stress affects sleep",
                }
            ],
            "induced_dependencies": [],
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


class TestStage4GroundingCompileOwnership:
    """Grounding should surface compile and global validation failures clearly."""

    def test_compile_failure_surfaces_in_grounding(self, monkeypatch):
        from causal_ssm_agent.flows.stages.stage4_assembly import AssemblyValidation

        def stub_validate_assembly(
            model_spec, priors, data_for_model, indicator_audits, causal_spec
        ):
            return AssemblyValidation(
                normalized_model_spec=model_spec,
                compile_ok=False,
                compile_error="dimension mismatch in drift matrix",
            )

        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4_assembly.validate_assembly",
            stub_validate_assembly,
        )

        data = {
            "priors": {
                "rho_outcome": {
                    "parameter": "rho_outcome",
                    "distribution": "Beta",
                    "params": {"alpha": 2, "beta": 2},
                    "sources": [],
                    "reasoning": "test prior",
                }
            },
        }
        output, feedback = stage4_grounding(
            data,
            causal_spec={},
            current={"model_spec": {"likelihoods": [], "parameters": []}},
            data_for_model=None,
        )

        assert output is not None
        assert output["authored_priors"]["rho_outcome"]["distribution"] == "Beta"
        assert output["validation"].compile_ok is False
        assert "COMPILE ERROR" in feedback
        assert "dimension mismatch" in feedback
        assert "Resubmit only the fields you changed" in feedback

    def test_schema_error_keeps_valid_priors_and_model_state(self):
        current = {
            "model_spec": {
                "likelihoods": [
                    {"variable": "mood_score", "distribution": "gaussian", "link": "identity"}
                ],
                "parameters": [
                    {"name": "rho_mood", "role": "ar_coefficient", "constraint": "unit_interval"},
                    {"name": "sigma_mood", "role": "residual_sd", "constraint": "positive"},
                ],
            }
        }
        data = {
            "priors": {
                "rho_mood": {
                    "parameter": "rho_mood",
                    "distribution": "Beta",
                    "params": {"alpha": 2, "beta": 2},
                    "sources": [],
                    "reasoning": "valid prior",
                },
                "sigma_mood": {
                    "parameter": "sigma_mood",
                    "distribution": "HalfNormal",
                    "params": {"sigma": 1.0},
                    "sources": ["not a structured source"],
                    "reasoning": "invalid source payload",
                },
            },
        }

        output, feedback = stage4_grounding(
            data, causal_spec={}, current=current, data_for_model=None
        )

        assert output is not None
        assert sorted(output["authored_priors"]) == ["rho_mood"]
        assert "SCHEMA ERRORS for prior 'sigma_mood'" in feedback

    def test_model_spec_can_be_saved_before_all_priors_arrive(self, monkeypatch, model_spec):
        from causal_ssm_agent.flows.stages.stage4_assembly import AssemblyValidation

        def stub_validate_assembly(model_spec, *_args, **_kwargs):
            return AssemblyValidation(
                normalized_model_spec=model_spec,
                compile_ok=True,
            )

        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4_assembly.validate_assembly",
            stub_validate_assembly,
        )

        output, feedback = stage4_grounding(
            {"model_spec": model_spec},
            causal_spec={},
            current=None,
            data_for_model=None,
        )

        assert output is not None
        assert output["model_spec"]["parameters"][0]["name"] == "rho_stress"
        assert output["validation"].compile_ok is True
        assert "MODEL STATE SAVED" in feedback
        assert "missing priors" in feedback

    def test_rejects_mixed_model_and_prior_updates(self):
        output, feedback = stage4_grounding(
            {
                "model_spec": {"likelihoods": [], "parameters": []},
                "priors": {
                    "rho_mood": {
                        "parameter": "rho_mood",
                        "distribution": "Beta",
                        "params": {"alpha": 2, "beta": 2},
                        "sources": [],
                        "reasoning": "test prior",
                    }
                },
            },
            causal_spec={},
            current=None,
            data_for_model=None,
        )

        assert output is None
        assert "UPDATE TOO BROAD" in feedback
        assert "separate calls" in feedback

    def test_accepts_large_prior_batches(self, monkeypatch):
        from causal_ssm_agent.flows.stages.stage4_assembly import AssemblyValidation

        def stub_validate_assembly(model_spec, *_args, **_kwargs):
            return AssemblyValidation(
                normalized_model_spec=model_spec,
                compile_ok=True,
                compiled_ssm={"compiled": True},
            )

        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4_assembly.validate_assembly",
            stub_validate_assembly,
        )
        monkeypatch.setattr(
            "causal_ssm_agent.models.ssm_compiler.resolve_prior_proposals",
            lambda *_args, **_kwargs: [{"parameter": "resolved"}],
        )

        current = {
            "model_spec": {
                "likelihoods": [],
                "parameters": [
                    {"name": f"rho_{idx}", "role": "ar_coefficient", "constraint": "unit_interval"}
                    for idx in range(9)
                ],
            }
        }
        priors = {
            f"rho_{idx}": {
                "parameter": f"rho_{idx}",
                "distribution": "Beta",
                "params": {"alpha": 2, "beta": 2},
                "sources": [],
                "reasoning": "test prior",
            }
            for idx in range(9)
        }

        output, feedback = stage4_grounding(
            {"priors": priors},
            causal_spec={},
            current=current,
            data_for_model=None,
        )

        assert output is not None
        assert feedback == "VALID"
        assert len(output["authored_priors"]) == 9
        assert output["resolved_priors"] == [{"parameter": "resolved"}]

    def test_rejects_redundant_prior_updates(self):
        current = {
            "model_spec": {
                "likelihoods": [],
                "parameters": [
                    {"name": "rho_mood", "role": "ar_coefficient", "constraint": "unit_interval"}
                ],
            },
            "authored_priors": {
                "rho_mood": {
                    "parameter": "rho_mood",
                    "distribution": "Beta",
                    "params": {"alpha": 2, "beta": 2},
                    "sources": [],
                    "reasoning": "test prior",
                }
            },
        }

        output, feedback = stage4_grounding(
            {"priors": dict(current["authored_priors"])},
            causal_spec={},
            current=current,
            data_for_model=None,
        )

        assert output is None
        assert "REDUNDANT PRIORS UPDATE" in feedback
        assert "`rho_mood`" in feedback

    def test_global_validation_failure_produces_correct_feedback(self, monkeypatch):
        from causal_ssm_agent.flows.stages.stage4_assembly import (
            AssemblyValidation,
            build_validation_payload,
        )
        from causal_ssm_agent.workers.schemas_prior import PriorValidationResult

        global_failure = PriorValidationResult(
            parameter="model_build",
            is_valid=False,
            issue=(
                "Model build failed:\n"
                "Observation support check failed:\n"
                "- 'outcome_score' uses gamma emission but 1/10 observations are outside support"
            ),
            suggested_adjustment="Fix model_spec or priors to enable model construction",
        )

        model_spec = {
            "likelihoods": [
                {
                    "variable": "outcome_score",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "test",
                }
            ],
            "parameters": [
                {
                    "name": "rho_outcome",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "AR coefficient",
                }
            ],
        }

        validation = AssemblyValidation(
            normalized_model_spec=model_spec,
            compile_ok=True,
            pp_checked=True,
            pp_valid=False,
            pp_results=[global_failure],
        )

        payload = build_validation_payload(validation, model_spec)
        assert payload["is_valid"] is False
        assert len(payload["issues"]) == 1
        assert "global issue" in payload["issues"][0]
        assert "model_spec issue" in payload["issues"][0]


class TestAgenticStage4Grounding:
    def test_merges_distribution_choice_delta(self, monkeypatch):
        from causal_ssm_agent.flows.stages.stage_tools import _agentic_stage4_grounding

        forwarded: dict[str, dict] = {}

        def fake_stage4_grounding(data, causal_spec, current, data_for_model, indicator_audits):
            del causal_spec, current, data_for_model, indicator_audits
            forwarded.update(data)
            return {"model_spec": data["model_spec"]}, "MODEL STATE SAVED"

        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage_tools.stage4_grounding",
            fake_stage4_grounding,
        )

        output, feedback = _agentic_stage4_grounding(
            data={
                "distribution_choices": [
                    {
                        "variable": "ide_focus_gaps",
                        "distribution": "student_t",
                        "link": "identity",
                        "reasoning": "Updated choice for zero-heavy duration data.",
                    }
                ]
            },
            causal_spec={},
            current={
                "model_spec": {
                    "likelihoods": [
                        {
                            "variable": "dev_platform_activity",
                            "distribution": "bernoulli",
                            "link": "logit",
                            "reasoning": "Deterministic.",
                        },
                        {
                            "variable": "ide_focus_gaps",
                            "distribution": "gamma",
                            "link": "log",
                            "reasoning": "Old choice.",
                        },
                        {
                            "variable": "advanced_tech_searches",
                            "distribution": "bernoulli",
                            "link": "logit",
                            "reasoning": "Retained choice.",
                        },
                    ],
                    "parameters": [
                        {
                            "name": "lambda_stackoverflow_visits_productivity",
                            "role": "loading",
                            "constraint": "positive",
                            "description": "loading",
                        }
                    ],
                }
            },
            data_for_model=None,
            indicator_audits=None,
            resolved_likelihoods=[
                {
                    "variable": "dev_platform_activity",
                    "distribution": "bernoulli",
                    "link": "logit",
                }
            ],
            ambiguous_indicators=[
                {"variable": "ide_focus_gaps"},
                {"variable": "advanced_tech_searches"},
            ],
            all_params=[
                {
                    "name": "lambda_stackoverflow_visits_productivity",
                    "role": "loading",
                    "constraint": "none",
                    "description": "loading",
                }
            ],
        )

        assert feedback == "MODEL STATE SAVED"
        assert output is not None
        merged_likelihoods = {
            likelihood["variable"]: likelihood for likelihood in forwarded["model_spec"]["likelihoods"]
        }
        assert merged_likelihoods["ide_focus_gaps"]["distribution"] == "student_t"
        assert merged_likelihoods["advanced_tech_searches"]["distribution"] == "bernoulli"

    def test_rejects_mixed_updates(self):
        from causal_ssm_agent.flows.stages.stage_tools import _agentic_stage4_grounding

        output, feedback = _agentic_stage4_grounding(
            data={
                "distribution_choices": [
                    {
                        "variable": "ide_focus_gaps",
                        "distribution": "student_t",
                        "link": "identity",
                        "reasoning": "Changed likelihood.",
                    }
                ],
                "priors": {
                    "rho_mood": {
                        "parameter": "rho_mood",
                        "distribution": "Beta",
                        "params": {"alpha": 2, "beta": 2},
                        "sources": [],
                        "reasoning": "test prior",
                    }
                },
            },
            causal_spec={},
            current={},
            data_for_model=None,
            indicator_audits=None,
            resolved_likelihoods=[],
            ambiguous_indicators=[{"variable": "ide_focus_gaps"}],
            all_params=[],
        )

        assert output is None
        assert "UPDATE TOO BROAD" in feedback

    def test_rejects_redundant_decision_update(self):
        from causal_ssm_agent.flows.stages.stage_tools import _agentic_stage4_grounding

        output, feedback = _agentic_stage4_grounding(
            data={
                "distribution_choices": [
                    {
                        "variable": "ide_focus_gaps",
                        "distribution": "student_t",
                        "link": "identity",
                        "reasoning": "Same decision, different words.",
                    }
                ]
            },
            causal_spec={},
            current={
                "model_spec": {
                    "likelihoods": [
                        {
                            "variable": "ide_focus_gaps",
                            "distribution": "student_t",
                            "link": "identity",
                            "reasoning": "Accepted choice.",
                        }
                    ],
                    "parameters": [],
                }
            },
            data_for_model=None,
            indicator_audits=None,
            resolved_likelihoods=[],
            ambiguous_indicators=[{"variable": "ide_focus_gaps"}],
            all_params=[],
        )

        assert output is None
        assert "REDUNDANT MODEL DECISIONS UPDATE" in feedback

    def test_ignores_redundant_decisions_when_cleanup_needed(self, monkeypatch):
        from causal_ssm_agent.flows.stages.stage_tools import _agentic_stage4_grounding

        forwarded: dict[str, dict] = {}

        def fake_stage4_grounding(data, causal_spec, current, data_for_model, indicator_audits):
            del causal_spec, current, data_for_model, indicator_audits
            forwarded.update(data)
            return {"model_spec": data["model_spec"]}, "MODEL STATE SAVED"

        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage_tools.stage4_grounding",
            fake_stage4_grounding,
        )

        output, feedback = _agentic_stage4_grounding(
            data={
                "distribution_choices": [
                    {
                        "variable": "daily_event_count",
                        "distribution": "negative_binomial",
                        "link": "log",
                        "reasoning": "Already accepted and unchanged.",
                    },
                    {
                        "variable": "sleep_issue_searches",
                        "distribution": "negative_binomial",
                        "link": "log",
                        "reasoning": "Revert to the valid count likelihood.",
                    },
                ]
            },
            causal_spec={},
            current={
                "model_spec": {
                    "likelihoods": [
                        {
                            "variable": "daily_event_count",
                            "distribution": "negative_binomial",
                            "link": "log",
                            "reasoning": "Accepted choice.",
                        },
                        {
                            "variable": "sleep_issue_searches",
                            "distribution": "poisson",
                            "link": "log",
                            "reasoning": "Invalid temporary choice.",
                        },
                        {
                            "variable": "chronotype",
                            "distribution": "gaussian",
                            "link": "identity",
                            "reasoning": "Stale invalid state that should be dropped.",
                        },
                    ],
                    "parameters": [],
                }
            },
            data_for_model=None,
            indicator_audits=None,
            resolved_likelihoods=[],
            ambiguous_indicators=[
                {"variable": "daily_event_count"},
                {"variable": "sleep_issue_searches"},
            ],
            all_params=[],
        )

        assert feedback == "MODEL STATE SAVED"
        assert output is not None
        merged_likelihoods = {
            likelihood["variable"]: likelihood for likelihood in forwarded["model_spec"]["likelihoods"]
        }
        assert merged_likelihoods["daily_event_count"]["distribution"] == "negative_binomial"
        assert merged_likelihoods["sleep_issue_searches"]["distribution"] == "negative_binomial"
        assert "chronotype" not in merged_likelihoods

    def test_accepts_loading_constraint_delta(self, monkeypatch):
        from causal_ssm_agent.flows.stages.stage_tools import _agentic_stage4_grounding

        forwarded: dict[str, dict] = {}

        def fake_stage4_grounding(data, causal_spec, current, data_for_model, indicator_audits):
            del causal_spec, current, data_for_model, indicator_audits
            forwarded.update(data)
            return {"model_spec": data["model_spec"]}, "MODEL STATE SAVED"

        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage_tools.stage4_grounding",
            fake_stage4_grounding,
        )

        output, feedback = _agentic_stage4_grounding(
            data={
                "loading_constraints": [
                    {
                        "parameter": "lambda_stackoverflow_visits_productivity",
                        "constraint": "positive",
                        "reasoning": "Updated loading sign.",
                    }
                ]
            },
            causal_spec={},
            current={
                "model_spec": {
                    "likelihoods": [
                        {
                            "variable": "ide_focus_gaps",
                            "distribution": "student_t",
                            "link": "identity",
                            "reasoning": "Accepted choice.",
                        }
                    ],
                    "parameters": [
                        {
                            "name": "lambda_stackoverflow_visits_productivity",
                            "role": "loading",
                            "constraint": "none",
                            "description": "loading",
                        }
                    ],
                }
            },
            data_for_model=None,
            indicator_audits=None,
            resolved_likelihoods=[],
            ambiguous_indicators=[{"variable": "ide_focus_gaps"}],
            all_params=[
                {
                    "name": "lambda_stackoverflow_visits_productivity",
                    "role": "loading",
                    "constraint": "none",
                    "description": "loading",
                }
            ],
        )

        assert feedback == "MODEL STATE SAVED"
        assert output is not None
        merged_params = {
            parameter["name"]: parameter for parameter in forwarded["model_spec"]["parameters"]
        }
        assert merged_params["lambda_stackoverflow_visits_productivity"]["constraint"] == "positive"
