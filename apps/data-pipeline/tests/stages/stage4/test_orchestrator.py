"""Tests for Stage 4 deterministic skeleton and prior-card derivation."""

import polars as pl

from causal_ssm_agent.flows.stages.stage4.agentic.stage4_block_specs import (
    get_stage4_prompt_scope_policy,
)
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_cards import (
    build_construct_scale_cards,
    build_distribution_cards,
    build_model_topology,
    build_prior_cards,
)
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_feedback import (
    make_stage4_grounding_result,
    make_stage4_validation_packet,
)
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_navigation import (
    _set_block_cursor,
    make_stage4_runtime,
)
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_orchestrator import (
    Stage4FrontierBlock,
    Stage4Plan,
    build_stage4_plan,
)
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_parameter_surfaces import (
    build_stage4_parameter_surface_index,
)
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_prompt_context import Stage4Messages
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_session import Stage4Session
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_skeleton import (
    Stage4Skeleton,
    derive_deterministic_spec,
)
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_state import (
    Stage4AcceptedArtifacts,
    Stage4Runtime,
)
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_types import Stage4Deps
from tests.helpers import make_stage4_plan as _make_plan
from tests.stages.stage4._support import make_causal_spec_dict as _make_causal_spec


def _simple_spec():
    """Two-construct, one-edge, binary + continuous indicators."""
    return _make_causal_spec(
        constructs=[
            {
                "name": "stress",
                "role": "exogenous",
                "temporal_status": "time_varying",
            },
            {
                "name": "sleep",
                "role": "endogenous",
                "temporal_status": "time_varying",
                "is_outcome": True,
            },
        ],
        edges=[{"cause": "stress", "effect": "sleep"}],
        indicators=[
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
    )


def _noop_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
    """Minimal grounding stub for message-rewriter tests."""
    if current:
        return make_stage4_grounding_result(
            stage_output=dict(current),
            status="accepted",
            feedback="VALID",
            retain_for_next_prompt=False,
            capture_stage_output=True,
        )
    return make_stage4_grounding_result(
        stage_output=data,
        status="accepted",
        feedback="VALID",
        retain_for_next_prompt=False,
        capture_stage_output=True,
    )


def _make_session(
    *,
    messages: Stage4Messages,
    plan: Stage4Plan,
    runtime: Stage4Runtime,
) -> Stage4Session:
    """Build a minimal Stage 4 session for prompt/turn tests."""
    return Stage4Session(
        plan=plan,
        prompt_context=messages,
        deps=Stage4Deps(
            skeleton=Stage4Skeleton(),
            causal_spec={},
            data_for_model=pl.DataFrame(),
            indicator_audits={},
            grounding_fn=_noop_stage4_grounding,
        ),
        runtime=runtime,
    )


# =============================================================================
# derive_deterministic_spec
# =============================================================================


class TestDeriveDeterministicSpec:
    def test_binary_resolves_to_bernoulli(self):
        """Binary dtype should resolve to bernoulli."""
        spec = _make_causal_spec(
            constructs=[
                {
                    "name": "mood",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                    "is_outcome": True,
                },
            ],
            edges=[],
            indicators=[
                {
                    "name": "happy",
                    "construct_name": "mood",
                    "measurement_dtype": "binary",
                    "how_to_measure": "Happy?",
                    "aggregation": "last",
                },
            ],
        )
        skeleton = derive_deterministic_spec(spec)
        # Binary has single dist (bernoulli) but multiple links (logit, probit)
        # So it should go to ambiguous
        all_vars = [r["variable"] for r in skeleton.resolved_likelihoods] + [
            a["variable"] for a in skeleton.ambiguous_indicators
        ]
        assert "happy" in all_vars

    def test_count_is_ambiguous(self):
        """Count dtype has multiple valid distributions (poisson, negative_binomial)."""
        spec = _make_causal_spec(
            constructs=[
                {
                    "name": "activity",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                    "is_outcome": True,
                },
            ],
            edges=[],
            indicators=[
                {
                    "name": "steps",
                    "construct_name": "activity",
                    "measurement_dtype": "count",
                    "how_to_measure": "Steps",
                    "aggregation": "sum",
                },
            ],
        )
        skeleton = derive_deterministic_spec(spec)
        step_ambig = [a for a in skeleton.ambiguous_indicators if a["variable"] == "steps"]
        assert len(step_ambig) == 1
        assert "valid_distributions" in step_ambig[0] or "fixed_distribution" in step_ambig[0]
        assert step_ambig[0]["support_kind"] == "interval"
        assert step_ambig[0]["summary_operator"] == "sum"

    def test_ar_params_for_time_varying_constructs(self):
        """Time-varying constructs should get base-persistence params."""
        skeleton = derive_deterministic_spec(_simple_spec())
        ar_params = [p for p in skeleton.parameters if p["role"] == "ar_coefficient"]
        assert {p["name"] for p in ar_params} == {"rho_stress", "rho_sleep"}
        assert all(p["constraint"] == "unit_interval" for p in ar_params)

    def test_beta_params_for_edges(self):
        """Each edge should produce a beta (fixed effect) parameter."""
        skeleton = derive_deterministic_spec(_simple_spec())
        beta_params = [p for p in skeleton.parameters if p["role"] == "fixed_effect"]
        assert len(beta_params) == 1
        assert "stress" in beta_params[0]["name"]
        assert "sleep" in beta_params[0]["name"]
        assert beta_params[0]["cause"] == "stress"
        assert beta_params[0]["effect"] == "sleep"

    def test_sigma_params_for_all_constructs(self):
        """Each construct should get a residual SD parameter."""
        skeleton = derive_deterministic_spec(_simple_spec())
        sigma_params = [p for p in skeleton.parameters if p["role"] == "residual_sd"]
        assert len(sigma_params) == 2

    def test_time_invariant_targets_do_not_expose_drift_surfaces(self):
        """Static states should not expose fixed-effect or diffusion surfaces."""
        spec = _make_causal_spec(
            constructs=[
                {
                    "name": "stable_trait",
                    "role": "exogenous",
                    "temporal_status": "time_invariant",
                },
                {
                    "name": "mood",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                    "is_outcome": True,
                },
                {
                    "name": "baseline_severity",
                    "role": "endogenous",
                    "temporal_status": "time_invariant",
                },
            ],
            edges=[
                {"cause": "stable_trait", "effect": "mood"},
                {"cause": "stable_trait", "effect": "baseline_severity"},
            ],
            indicators=[
                {
                    "name": "trait_score",
                    "construct_name": "stable_trait",
                    "measurement_dtype": "continuous",
                    "how_to_measure": "Trait score",
                    "aggregation": "mean",
                },
                {
                    "name": "mood_rating",
                    "construct_name": "mood",
                    "measurement_dtype": "continuous",
                    "how_to_measure": "Mood rating",
                    "aggregation": "mean",
                },
                {
                    "name": "severity_score",
                    "construct_name": "baseline_severity",
                    "measurement_dtype": "continuous",
                    "how_to_measure": "Severity score",
                    "aggregation": "mean",
                },
            ],
        )

        skeleton = derive_deterministic_spec(spec)
        parameter_names = {parameter["name"] for parameter in skeleton.parameters}

        assert "beta_stable_trait_mood" in parameter_names
        assert "beta_stable_trait_baseline_severity" not in parameter_names
        assert "sigma_mood" in parameter_names
        assert "sigma_stable_trait" not in parameter_names
        assert "sigma_baseline_severity" not in parameter_names

    def test_compiler_derived_initial_state_params_are_exposed(self):
        """Compiler-owned initial-state priors should appear in the Stage 4 inventory."""
        skeleton = derive_deterministic_spec(_simple_spec())
        parameter_names = {parameter["name"] for parameter in skeleton.parameters}
        assert {"t0_mean_stress", "t0_sd_stress", "t0_mean_sleep", "t0_sd_sleep"} <= parameter_names

    def test_parameter_surface_index_owns_block_grouping_and_context(self):
        """Typed surfaces should be the single semantic owner for block grouping."""
        spec = _make_causal_spec(
            constructs=[
                {
                    "name": "stress",
                    "role": "exogenous",
                    "temporal_status": "time_varying",
                },
                {
                    "name": "sleep",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                    "is_outcome": True,
                },
            ],
            edges=[{"cause": "stress", "effect": "sleep"}],
            indicators=[
                {
                    "name": "pss",
                    "construct_name": "stress",
                    "measurement_dtype": "continuous",
                    "how_to_measure": "PSS score",
                    "aggregation": "mean",
                },
                {
                    "name": "vas",
                    "construct_name": "stress",
                    "measurement_dtype": "continuous",
                    "how_to_measure": "Stress VAS",
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
        )
        skeleton = derive_deterministic_spec(spec)
        surface_index = build_stage4_parameter_surface_index(spec, skeleton)

        loading_surface = surface_index.by_name["lambda_vas_stress"]
        assert loading_surface.block_kind == "measurement_prior"
        assert loading_surface.owner_key == "stress"
        assert loading_surface.structural_context["reference_indicator"] == "pss"

        effect_surface = surface_index.by_name["beta_stress_sleep"]
        assert effect_surface.block_kind == "effect_prior"
        assert effect_surface.owner_key == "sleep"
        assert effect_surface.effect_edge == ("stress", "sleep")
        assert effect_surface.structural_context["expected_lag_days"] == 1.0

    def test_stage4_inventory_matches_compiler_public_prior_rows(self):
        """Stage 4 should expose compiler rows plus conditional likelihood extras."""
        from causal_ssm_agent.models.ssm_compiler import (
            compile_ssm_artifact,
            resolve_prior_proposals,
        )

        spec = _simple_spec()
        skeleton = derive_deterministic_spec(spec)
        compiled_ssm = compile_ssm_artifact(
            {
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
                        "description": (
                            "Baseline discrete-time persistence absent incoming feedback for stress"
                        ),
                    },
                    {
                        "name": "rho_sleep",
                        "role": "ar_coefficient",
                        "constraint": "unit_interval",
                        "description": (
                            "Baseline discrete-time persistence absent incoming feedback for sleep"
                        ),
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
                        "description": "Residual/innovation SD for stress",
                    },
                    {
                        "name": "sigma_sleep",
                        "role": "residual_sd",
                        "constraint": "positive",
                        "description": "Residual/innovation SD for sleep",
                    },
                ],
            },
            {},
            causal_spec=spec,
        )

        compiler_parameter_names = {
            row["parameter"] for row in resolve_prior_proposals(compiled_ssm, authored_priors={})
        }

        final_parameter_names = set(skeleton.final_parameter_names)
        assert compiler_parameter_names <= final_parameter_names
        assert final_parameter_names - compiler_parameter_names == {
            "manifest_mean_pss_score",
            "manifest_mean_sleep_quality",
            "obs_concentration",
            "obs_df",
            "obs_shape",
            "cint_stress",
            "cint_sleep",
            "t0_mean_stress",
            "t0_mean_sleep",
            "t0_sd_stress",
            "t0_sd_sleep",
        }

    def test_negative_binomial_candidate_obs_r_is_surfaced_to_stage4(self):
        """Stage 4 should expose the negative-binomial dispersion prior surface."""
        from causal_ssm_agent.models.ssm_compiler import (
            compile_ssm_artifact,
            resolve_prior_proposals,
        )

        spec = _make_causal_spec(
            constructs=[
                {
                    "name": "activity",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                    "is_outcome": True,
                },
            ],
            edges=[],
            indicators=[
                {
                    "name": "steps",
                    "construct_name": "activity",
                    "measurement_dtype": "count",
                    "how_to_measure": "Count steps",
                    "aggregation": "sum",
                }
            ],
        )
        skeleton = derive_deterministic_spec(spec)
        compiled_ssm = compile_ssm_artifact(
            {
                "likelihoods": [
                    {
                        "variable": "steps",
                        "distribution": "negative_binomial",
                        "link": "log",
                        "reasoning": "Count outcome with overdispersion support",
                    }
                ],
                "parameters": [
                    {
                        "name": "obs_r",
                        "role": "observation_hyperparameter_positive",
                        "constraint": "positive",
                        "description": "Negative-binomial observation dispersion",
                    },
                    {
                        "name": "rho_activity",
                        "role": "ar_coefficient",
                        "constraint": "unit_interval",
                        "description": "AR(1) discrete-time persistence for activity",
                    },
                    {
                        "name": "sigma_activity",
                        "role": "residual_sd",
                        "constraint": "positive",
                        "description": "Residual/innovation SD for activity",
                    },
                ],
            },
            {},
            causal_spec=spec,
        )

        compiler_parameter_names = {
            row["parameter"] for row in resolve_prior_proposals(compiled_ssm, authored_priors={})
        }

        assert "obs_r" in compiler_parameter_names
        assert "obs_r" in skeleton.final_parameter_names

    def test_multi_indicator_loadings(self):
        """Multi-indicator constructs should get loading params for non-reference indicators."""
        spec = _make_causal_spec(
            constructs=[
                {
                    "name": "stress",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                    "is_outcome": True,
                },
            ],
            edges=[],
            indicators=[
                {
                    "name": "pss",
                    "construct_name": "stress",
                    "measurement_dtype": "continuous",
                    "how_to_measure": "PSS",
                    "aggregation": "mean",
                },
                {
                    "name": "vas",
                    "construct_name": "stress",
                    "measurement_dtype": "continuous",
                    "how_to_measure": "VAS",
                    "aggregation": "mean",
                },
            ],
        )
        skeleton = derive_deterministic_spec(spec)
        # First indicator is reference (no param), second gets loading
        assert len(skeleton.loading_params) == 1
        assert "vas" in skeleton.loading_params[0]["name"]
        assert skeleton.loading_params[0]["reference_indicator"] == "pss"

    def test_single_indicator_no_loadings(self):
        """Single-indicator constructs should not generate loading params."""
        skeleton = derive_deterministic_spec(_simple_spec())
        assert len(skeleton.loading_params) == 0

    def test_exogenous_time_varying_constructs_get_ar_params(self):
        """Exogenous time-varying constructs should get base-persistence params."""
        skeleton = derive_deterministic_spec(_simple_spec())
        ar_params = [p for p in skeleton.parameters if p["role"] == "ar_coefficient"]
        assert "rho_stress" in {p["name"] for p in ar_params}

    def test_marginalized_confounder_adds_correlation_param(self):
        spec = _make_causal_spec(
            constructs=[
                {
                    "name": "u_shared",
                    "role": "exogenous",
                    "temporal_status": "time_varying",
                    "description": "Shared unobserved cause",
                },
                {
                    "name": "stress",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                },
                {
                    "name": "sleep",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                    "is_outcome": True,
                },
            ],
            edges=[
                {"cause": "u_shared", "effect": "stress"},
                {"cause": "u_shared", "effect": "sleep"},
            ],
            indicators=[
                {
                    "name": "stress_score",
                    "construct_name": "stress",
                    "measurement_dtype": "continuous",
                    "how_to_measure": "Stress score",
                    "aggregation": "mean",
                },
                {
                    "name": "sleep_score",
                    "construct_name": "sleep",
                    "measurement_dtype": "continuous",
                    "how_to_measure": "Sleep score",
                    "aggregation": "mean",
                },
            ],
        )
        spec["identifiability"] = {
            "graph_info": {"unobserved_confounders": ["u_shared"]},
            "identifiable_treatments": {},
            "non_identifiable_treatments": {},
        }
        spec["estimation"]["induced_dependencies"] = [
            {
                "between": ["sleep", "stress"],
                "kind": "innovation_correlation",
                "source_confounders": ["u_shared"],
            }
        ]

        skeleton = derive_deterministic_spec(spec)
        correlation_params = [p for p in skeleton.parameters if p["role"] == "correlation"]
        assert len(correlation_params) == 1
        assert correlation_params[0]["name"] == "cor_sleep_stress"
        assert correlation_params[0]["constraint"] == "correlation"
        assert correlation_params[0]["source_confounders"] == ["u_shared"]


class TestBuildPriorCards:
    def test_prior_cards_reference_structural_context_only_once(self):
        spec = _simple_spec()
        skeleton = derive_deterministic_spec(spec)
        cards = build_prior_cards(spec, skeleton)
        beta_card = next(card for card in cards if card["parameter"] == "beta_stress_sleep")
        sigma_card = next(card for card in cards if card["parameter"] == "sigma_sleep")

        assert beta_card["structural_context"] == {
            "cause": "stress",
            "effect": "sleep",
            "lagged": True,
            "expected_lag_days": 1.0,
            "feedback_loop": False,
        }
        assert sigma_card["structural_context"] == {"construct": "sleep"}


class TestPromptContextBuilders:
    def test_model_topology_is_compact(self):
        topology = build_model_topology(_simple_spec())
        assert topology["model_clock"] == "1d"
        assert topology["model_interval_days"] == 1.0
        assert topology["outcome"] == "sleep"
        assert topology["latent_edges"][0]["cause"] == "stress"
        assert "constructs" not in topology

    def test_distribution_cards_merge_options_with_empirical_profile(self):
        spec = _make_causal_spec(
            constructs=[
                {
                    "name": "activity",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                    "is_outcome": True,
                }
            ],
            edges=[],
            indicators=[
                {
                    "name": "steps",
                    "construct_name": "activity",
                    "measurement_dtype": "count",
                    "how_to_measure": "Count steps",
                    "aggregation": "sum",
                }
            ],
        )
        skeleton = derive_deterministic_spec(spec)
        cards = build_distribution_cards(
            spec,
            {
                "steps": {
                    "profile": {
                        "n_obs": 50,
                        "mean": 12.0,
                        "std": 4.0,
                        "min": 0.0,
                        "max": 25.0,
                        "variance_to_mean_ratio": 1.4,
                        "is_nonnegative": True,
                        "looks_integer_valued": True,
                    },
                    "validation": {
                        "issues": [
                            {
                                "severity": "warning",
                                "issue_type": "large_timestamp_gap",
                                "message": "gap too large",
                            }
                        ]
                    },
                }
            },
            skeleton,
        )
        assert len(cards) == 1
        assert cards[0]["variable"] == "steps"
        assert cards[0]["profile"]["variance_to_mean_ratio"] == 1.4
        assert cards[0]["validation_issues"] == ["warning large_timestamp_gap"]

    def test_construct_scale_cards_factor_out_indicator_profiles(self):
        spec = _simple_spec()
        cards = build_construct_scale_cards(
            spec,
            {
                "pss_score": {
                    "profile": {
                        "n_obs": 40,
                        "mean": 12.0,
                        "std": 3.5,
                        "min": 3.0,
                        "max": 21.0,
                        "is_nonnegative": True,
                    }
                },
                "sleep_quality": {
                    "profile": {
                        "n_obs": 40,
                        "mean": 6.0,
                        "std": 1.2,
                        "min": 2.0,
                        "max": 8.0,
                    }
                },
            },
        )
        stress_card = next(card for card in cards if card["construct"] == "stress")
        assert stress_card["reference_indicator"] == "pss_score"
        assert stress_card["indicators"][0]["profile"]["std"] == 3.5
        assert stress_card["indicators"][0]["has_distribution_decision_card"] is False

    def test_construct_scale_cards_mark_ambiguous_indicators(self):
        spec = _make_causal_spec(
            constructs=[
                {
                    "name": "mood",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                    "is_outcome": True,
                }
            ],
            edges=[],
            indicators=[
                {
                    "name": "happy",
                    "construct_name": "mood",
                    "measurement_dtype": "binary",
                    "how_to_measure": "Happy?",
                    "aggregation": "last",
                }
            ],
        )
        skeleton = derive_deterministic_spec(spec)
        cards = build_construct_scale_cards(spec, {"happy": {"profile": {"n_obs": 20}}}, skeleton)
        mood_card = next(card for card in cards if card["construct"] == "mood")
        assert mood_card["indicators"][0]["has_distribution_decision_card"] is True

    def test_construct_scale_cards_include_support_window_and_extended_profile_fields(self):
        spec = _make_causal_spec(
            constructs=[
                {
                    "name": "mood",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                    "is_outcome": True,
                }
            ],
            edges=[],
            indicators=[
                {
                    "name": "happy",
                    "construct_name": "mood",
                    "measurement_dtype": "continuous",
                    "how_to_measure": "Average happiness this week",
                    "aggregation": "mean",
                    "observation_window": "1w",
                }
            ],
        )
        skeleton = derive_deterministic_spec(spec)
        cards = build_construct_scale_cards(
            spec,
            {
                "happy": {
                    "profile": {
                        "n_obs": 20,
                        "q25": 2.0,
                        "q50": 3.0,
                        "q75": 4.0,
                        "time_coverage_ratio": 0.75,
                        "max_gap_ratio": 2.5,
                        "duplicate_pct": 0.10,
                        "n_unparseable_timestamps": 1,
                    }
                }
            },
            skeleton,
        )
        indicator = cards[0]["indicators"][0]

        assert indicator["observation_window"] == "1w"
        assert indicator["effective_window"] == "1w"
        assert indicator["profile"]["q50"] == 3.0
        assert indicator["profile"]["time_coverage_ratio"] == 0.75
        assert indicator["profile"]["max_gap_ratio"] == 2.5
        assert indicator["profile"]["duplicate_pct"] == 0.10
        assert indicator["profile"]["n_unparseable_timestamps"] == 1


class TestStage4Plan:
    def test_plan_splits_model_and_prior_blocks(self):
        spec = _make_causal_spec(
            constructs=[
                {
                    "name": "activity",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                    "is_outcome": True,
                }
            ],
            edges=[],
            indicators=[
                {
                    "name": "steps",
                    "construct_name": "activity",
                    "measurement_dtype": "count",
                    "how_to_measure": "Count steps",
                    "aggregation": "sum",
                }
            ],
        )
        skeleton = derive_deterministic_spec(spec)
        plan = build_stage4_plan(spec, skeleton)

        assert [block.kind for block in plan.model_blocks] == [
            "model_configuration",
            "indicator_decision",
        ]
        assert plan.model_blocks[1].variable_names == ("steps",)
        assert plan.review_block is not None
        assert plan.review_block.kind == "global_review"
        assert plan.prior_review_block is not None
        assert plan.prior_review_block.kind == "global_prior_review"
        assert "review:prior_system" in {block.id for block in plan.all_blocks}
        assert [block.kind for block in plan.prior_blocks] == [
            "observation_prior",
            "observation_prior",
            "dynamics_prior",
        ]
        assert plan.prior_blocks[0].parameter_names == ("manifest_mean_steps",)
        assert plan.prior_blocks[1].parameter_names == ("obs_r",)
        assert set(plan.prior_blocks[2].parameter_names) == {
            "rho_activity",
            "sigma_activity",
            "cint_activity",
            "t0_mean_activity",
            "t0_sd_activity",
        }

    def test_plan_marks_dynamics_t0_priors_optional(self):
        spec = _make_causal_spec(
            constructs=[
                {
                    "name": "activity",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                    "is_outcome": True,
                }
            ],
            edges=[],
            indicators=[
                {
                    "name": "steps",
                    "construct_name": "activity",
                    "measurement_dtype": "count",
                    "how_to_measure": "Count steps",
                    "aggregation": "sum",
                }
            ],
        )
        skeleton = derive_deterministic_spec(spec)
        plan = build_stage4_plan(spec, skeleton)

        dynamics_block = next(
            block for block in plan.prior_blocks if block.kind == "dynamics_prior"
        )

        assert dynamics_block.coverage_policy == "all_required_parameters"
        assert set(dynamics_block.required_parameter_names) == {
            "rho_activity",
            "sigma_activity",
            "cint_activity",
        }
        assert set(dynamics_block.optional_parameter_names) == {
            "t0_mean_activity",
            "t0_sd_activity",
        }
        assert plan.prior_review_block is not None
        assert plan.prior_review_block.coverage_policy == "subset_allowed"

    def test_multi_indicator_measurement_block_includes_obs_sd_priors(self):
        """Free manifest-noise priors should surface alongside loading priors."""
        spec = _make_causal_spec(
            constructs=[
                {
                    "name": "stress",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                    "is_outcome": True,
                },
            ],
            edges=[],
            indicators=[
                {
                    "name": "pss",
                    "construct_name": "stress",
                    "measurement_dtype": "continuous",
                    "how_to_measure": "PSS",
                    "aggregation": "mean",
                },
                {
                    "name": "vas",
                    "construct_name": "stress",
                    "measurement_dtype": "continuous",
                    "how_to_measure": "VAS",
                    "aggregation": "mean",
                },
            ],
        )

        skeleton = derive_deterministic_spec(spec)
        plan = build_stage4_plan(spec, skeleton)
        measurement_block = next(
            block for block in plan.prior_blocks if block.kind == "measurement_prior"
        )

        assert set(measurement_block.parameter_names) == {
            "lambda_vas_stress",
            "obs_sd_pss",
            "obs_sd_vas",
        }

    def test_plan_groups_effect_priors_by_target_construct(self):
        spec = _make_causal_spec(
            constructs=[
                {
                    "name": "stress",
                    "role": "exogenous",
                    "temporal_status": "time_varying",
                },
                {
                    "name": "activity",
                    "role": "exogenous",
                    "temporal_status": "time_varying",
                },
                {
                    "name": "sleep",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                    "is_outcome": True,
                },
            ],
            edges=[
                {"cause": "stress", "effect": "sleep"},
                {"cause": "activity", "effect": "sleep"},
            ],
            indicators=[
                {
                    "name": "stress_score",
                    "construct_name": "stress",
                    "measurement_dtype": "continuous",
                    "how_to_measure": "Stress score",
                    "aggregation": "mean",
                },
                {
                    "name": "steps",
                    "construct_name": "activity",
                    "measurement_dtype": "count",
                    "how_to_measure": "Count steps",
                    "aggregation": "sum",
                },
                {
                    "name": "sleep_score",
                    "construct_name": "sleep",
                    "measurement_dtype": "continuous",
                    "how_to_measure": "Sleep score",
                    "aggregation": "mean",
                },
            ],
        )

        skeleton = derive_deterministic_spec(spec)
        plan = build_stage4_plan(spec, skeleton)
        effect_block = next(block for block in plan.prior_blocks if block.kind == "effect_prior")

        assert effect_block.id == "effects:sleep"
        assert effect_block.parameter_names == (
            "beta_stress_sleep",
            "beta_activity_sleep",
        )
        assert effect_block.construct_names == ("stress", "activity", "sleep")


class TestStage4TurnProjection:
    def test_messages_for_scope_render_only_active_block(self):
        spec = _make_causal_spec(
            constructs=[
                {
                    "name": "activity",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                    "is_outcome": True,
                },
                {
                    "name": "sleep",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                },
            ],
            edges=[{"cause": "activity", "effect": "sleep"}],
            indicators=[
                {
                    "name": "steps",
                    "construct_name": "activity",
                    "measurement_dtype": "count",
                    "how_to_measure": "Count steps",
                    "aggregation": "sum",
                },
                {
                    "name": "sleep_score",
                    "construct_name": "sleep",
                    "measurement_dtype": "continuous",
                    "how_to_measure": "Sleep score",
                    "aggregation": "mean",
                },
            ],
        )
        skeleton = derive_deterministic_spec(spec)
        plan = build_stage4_plan(spec, skeleton)
        messages = Stage4Messages(
            question="Does activity affect sleep?",
            model_topology=build_model_topology(spec),
            distribution_cards=build_distribution_cards(spec, {}, skeleton),
            loading_params=skeleton.loading_params,
            construct_scale_cards=build_construct_scale_cards(spec, {}, skeleton),
            prior_cards=build_prior_cards(spec, skeleton),
        )
        runtime = make_stage4_runtime(plan)
        session = _make_session(messages=messages, plan=plan, runtime=runtime)

        turn = session.current_turn()
        assert turn is not None
        prompt = turn.messages

        assert prompt[0]["role"] == "system"
        assert prompt[1]["role"] == "user"
        assert "`id`: `model:configuration`" in prompt[1]["content"]
        assert (
            "Choose the global initialization policy, observation-intercept policy, and whether equilibrium forcing is enabled."
            in prompt[1]["content"]
        )
        assert "allowed initialization_policy values: `stationary`, `free`" in prompt[1]["content"]
        assert (
            "allowed observation_intercept_policy values: `fixed`, `free`" in prompt[1]["content"]
        )
        assert "allowed equilibrium_forcing values: `true`, `false`" in prompt[1]["content"]
        assert (
            "centered-indicator constructs that can identify a latent baseline if forcing is enabled: `sleep`"
            in prompt[1]["content"]
        )
        assert "beta_activity_sleep" not in prompt[1]["content"]
        assert (
            "Use `submit_model_configuration` with exactly this argument object:"
            in prompt[1]["content"]
        )
        assert "### Parameter Prior Cards" not in prompt[1]["content"]

    def test_current_turn_moves_to_next_pending_prior_block(self):
        messages = Stage4Messages(
            question="How does stress affect sleep?",
            model_topology={"model_clock": "1d", "model_interval_days": 1.0, "outcome": "sleep"},
            construct_scale_cards=[],
            prior_cards=[
                {
                    "parameter": "beta_stress_sleep",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "structural_context": {
                        "cause": "stress",
                        "effect": "sleep",
                        "lagged": True,
                    },
                },
                {
                    "parameter": "rho_sleep",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "structural_context": {"construct": "sleep"},
                },
            ],
        )
        prior_blocks = (
            Stage4FrontierBlock(
                id="effects:sleep",
                kind="effect_prior",
                label="Effect prior",
                construct_names=("stress", "sleep"),
                parameter_names=("beta_stress_sleep",),
            ),
            Stage4FrontierBlock(
                id="dynamics:sleep",
                kind="dynamics_prior",
                label="Dynamics prior",
                construct_names=("sleep",),
                parameter_names=("rho_sleep",),
            ),
        )
        plan = _make_plan(prior_blocks=prior_blocks)
        runtime = make_stage4_runtime(plan)
        runtime.domain.block_status.update(
            {
                "effects:sleep": "accepted",
                "dynamics:sleep": "pending",
            }
        )
        runtime.domain.accepted = Stage4AcceptedArtifacts(
            model_spec={"parameters": [{"name": "beta_stress_sleep"}, {"name": "rho_sleep"}]},
            authored_priors={"beta_stress_sleep": {"distribution": "Normal"}},
        )
        active_block = plan.get_block("dynamics:sleep")
        assert active_block is not None
        _set_block_cursor(runtime, active_block)
        session = _make_session(messages=messages, plan=plan, runtime=runtime)

        turn = session.current_turn()
        assert turn is not None
        assert turn.block.id == "dynamics:sleep"
        content = turn.messages[1]["content"]
        assert "`id`: `dynamics:sleep`" in content
        assert "rho_sleep" in content
        assert "Submit the active block only" in content

    def test_current_turn_surfaces_runtime_feedback(self):
        prior_blocks = (
            Stage4FrontierBlock(
                id="effects:sleep",
                kind="effect_prior",
                label="Effect prior",
                construct_names=("stress", "sleep"),
                parameter_names=("beta_stress_sleep",),
            ),
        )
        plan = _make_plan(prior_blocks=prior_blocks)
        runtime = make_stage4_runtime(plan)
        runtime.domain.block_status["effects:sleep"] = "pending"
        runtime.domain.accepted = Stage4AcceptedArtifacts(
            model_spec={"parameters": [{"name": "beta_stress_sleep"}]}
        )
        runtime.interaction.last_validation_packet = make_stage4_validation_packet(
            status="info",
            feedback="submit priors",
            active_scope_id="effects:sleep",
            retain_for_next_prompt=True,
            capture_stage_output=False,
        )
        active_block = plan.get_block("effects:sleep")
        assert active_block is not None
        _set_block_cursor(runtime, active_block)
        session = _make_session(
            messages=Stage4Messages(
                question="How does stress affect sleep?",
                model_topology={},
                construct_scale_cards=[],
                prior_cards=[
                    {
                        "parameter": "beta_stress_sleep",
                        "role": "fixed_effect",
                        "constraint": "none",
                        "structural_context": {
                            "cause": "stress",
                            "effect": "sleep",
                            "lagged": True,
                        },
                    }
                ],
            ),
            plan=plan,
            runtime=runtime,
        )

        turn = session.current_turn()
        assert turn is not None
        assert turn.latest_feedback == "submit priors"
        assert "submit priors" in turn.messages[1]["content"]

    def test_current_turn_renders_global_prior_review_from_normal_block_registry(self):
        prior_review_block = Stage4FrontierBlock(
            id="review:prior_system",
            kind="global_prior_review",
            label="Repair full prior system",
            construct_names=("stress", "sleep"),
            parameter_names=("beta_stress_sleep",),
        )
        plan = _make_plan(prior_review_block=prior_review_block)
        runtime = make_stage4_runtime(plan)
        runtime.domain.block_status[prior_review_block.id] = "reopened"
        runtime.domain.accepted = Stage4AcceptedArtifacts(
            model_spec={"parameters": [{"name": "beta_stress_sleep"}]}
        )
        _set_block_cursor(runtime, prior_review_block)
        session = _make_session(
            messages=Stage4Messages(
                question="How does stress affect sleep?",
                model_topology={},
                construct_scale_cards=[],
                prior_cards=[
                    {
                        "parameter": "beta_stress_sleep",
                        "role": "fixed_effect",
                        "constraint": "none",
                        "structural_context": {
                            "cause": "stress",
                            "effect": "sleep",
                            "lagged": True,
                        },
                    }
                ],
            ),
            plan=plan,
            runtime=runtime,
        )

        turn = session.current_turn()

        assert turn is not None
        assert turn.block.id == "review:prior_system"
        assert turn.allowed_tool_names == ("submit_prior_block",)
        assert "`submit_prior_block`" in turn.messages[1]["content"]
        assert '"priors": {' in turn.messages[1]["content"]


class TestStage4PromptScopePolicy:
    def test_policy_is_looked_up_by_block_kind(self):
        policy = get_stage4_prompt_scope_policy("measurement_prior")

        assert policy.system_task.startswith("Propose full prior specifications")
        assert policy.user_task.startswith("Propose full prior specifications")
        assert policy.visible_sections == (
            "distribution_cards",
            "construct_scale_cards",
            "prior_cards",
        )
        assert policy.guidance_section_keys == (
            "prior_distribution_types",
            "parameter_guidance",
            "measurement_prior_guidance",
        )
        assert policy.parameter_guidance_prefixes == ("lambda", "obs_sd")
        assert policy.allowed_tool_names == ("submit_prior_block", "elicit_prior_gmm")

    def test_observation_policy_is_looked_up_by_block_kind(self):
        policy = get_stage4_prompt_scope_policy("observation_prior")

        assert policy.system_task.startswith("Propose full prior specifications")
        assert policy.user_task.startswith("Propose priors only for this active observation-family")
        assert policy.visible_sections == (
            "distribution_cards",
            "construct_scale_cards",
            "prior_cards",
        )
        assert policy.parameter_guidance_prefixes == ("obs_", "manifest_mean")
        assert policy.allowed_tool_names == ("submit_prior_block", "elicit_prior_gmm")

    def test_effect_policy_keeps_search_enabled(self):
        policy = get_stage4_prompt_scope_policy("effect_prior")

        assert policy.parameter_guidance_prefixes == ("beta",)
        assert policy.allowed_tool_names == (
            "submit_prior_block",
            "search_literature",
            "elicit_prior_gmm",
        )

    def test_dynamics_and_correlation_policies_disable_search(self):
        dynamics_policy = get_stage4_prompt_scope_policy("dynamics_prior")
        correlation_policy = get_stage4_prompt_scope_policy("correlation_prior")

        assert dynamics_policy.allowed_tool_names == ("submit_prior_block", "elicit_prior_gmm")
        assert correlation_policy.allowed_tool_names == ("submit_prior_block", "elicit_prior_gmm")
        assert correlation_policy.guidance_section_keys == (
            "prior_distribution_types",
            "parameter_guidance",
            "continuous_time_dynamics",
            "latent_initial_state_guidance",
        )

    def test_global_review_policy_is_validate_only(self):
        policy = get_stage4_prompt_scope_policy("global_review")

        assert policy.user_task.startswith("Review the locked model form")
        assert policy.visible_sections == (
            "distribution_cards",
            "loading_params",
            "construct_scale_cards",
        )
        assert policy.allowed_tool_names == ("submit_model_review",)

    def test_global_prior_review_policy_is_validate_only(self):
        policy = get_stage4_prompt_scope_policy("global_prior_review")

        assert policy.user_task.startswith("Review the full accepted prior system")
        assert policy.visible_sections == ("construct_scale_cards", "prior_cards")
        assert policy.parameter_guidance_prefixes == (
            "lambda",
            "obs_",
            "manifest_mean",
            "rho",
            "sigma",
            "cint",
            "beta",
            "cor",
            "tau",
            "t0_mean",
            "t0_sd",
        )
        assert policy.allowed_tool_names == ("submit_prior_block",)
