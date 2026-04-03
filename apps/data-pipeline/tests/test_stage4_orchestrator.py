"""Tests for Stage 4 deterministic skeleton and prior-card derivation."""

import polars as pl

import causal_ssm_agent.orchestrator.stage4 as stage4_module
from causal_ssm_agent.orchestrator.stage4 import (
    Stage4AcceptedState,
    Stage4Deps,
    Stage4Messages,
    Stage4Runtime,
    Stage4Session,
    make_stage4_runtime,
)
from causal_ssm_agent.orchestrator.stage4_orchestrator import (
    Stage4FrontierBlock,
    Stage4Plan,
    Stage4Skeleton,
    build_construct_scale_cards,
    build_distribution_cards,
    build_model_topology,
    build_prior_cards,
    build_stage4_plan,
    derive_deterministic_spec,
    get_stage4_prompt_scope_policy,
)
from tests.helpers import make_stage4_plan as _make_plan


def _make_causal_spec(
    constructs: list[dict],
    edges: list[dict],
    indicators: list[dict],
) -> dict:
    """Build a CausalSpec dict from components."""
    indicators = [
        {"construct_polarity": "positive", **indicator}
        if "construct_polarity" not in indicator
        else dict(indicator)
        for indicator in indicators
    ]
    retained_names = [construct["name"] for construct in constructs]
    return {
        "latent": {"constructs": constructs, "edges": edges},
        "measurement": {"model_clock": "1d", "indicators": indicators},
        "estimation": {
            "state_order": retained_names,
            "edges": edges,
            "induced_dependencies": [],
        },
    }


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
        return dict(current), "VALID"
    return data, "VALID"


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
                    "aggregation": "mean",
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

    def test_ar_params_for_endogenous(self):
        """Endogenous time-varying constructs should get AR params."""
        skeleton = derive_deterministic_spec(_simple_spec())
        ar_params = [p for p in skeleton.parameters if p["role"] == "ar_coefficient"]
        # Only "sleep" is endogenous
        assert len(ar_params) == 1
        assert "sleep" in ar_params[0]["name"]
        assert ar_params[0]["constraint"] == "unit_interval"

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

    def test_compiler_derived_initial_state_params_are_exposed(self):
        """Compiler-owned initial-state priors should appear in the Stage 4 inventory."""
        skeleton = derive_deterministic_spec(_simple_spec())
        parameter_names = {parameter["name"] for parameter in skeleton.parameters}
        assert {"t0_mean_stress", "t0_sd_stress", "t0_mean_sleep", "t0_sd_sleep"} <= parameter_names

    def test_stage4_inventory_matches_compiler_public_prior_rows(self):
        """Stage 4 should expose exactly the compiler's public prior rows."""
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
                        "name": "rho_sleep",
                        "role": "ar_coefficient",
                        "constraint": "unit_interval",
                        "description": "AR(1) discrete-time persistence for sleep",
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

        assert compiler_parameter_names == set(skeleton.final_parameter_names)

    def test_observation_defaults_stay_compiler_only(self):
        """Likelihood-extra defaults such as obs_r should stay hidden from Stage 4."""
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

        assert all(not name.startswith("obs_") for name in compiler_parameter_names)
        assert all(not name.startswith("obs_") for name in skeleton.final_parameter_names)
        assert "obs_r" not in compiler_parameter_names

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

    def test_no_ar_for_exogenous(self):
        """Exogenous constructs should NOT get AR parameters."""
        skeleton = derive_deterministic_spec(_simple_spec())
        ar_params = [p for p in skeleton.parameters if p["role"] == "ar_coefficient"]
        for p in ar_params:
            assert "stress" not in p["name"]  # stress is exogenous

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
                    "aggregation": "mean",
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

        assert [block.kind for block in plan.model_blocks] == ["indicator_decision"]
        assert plan.model_blocks[0].variable_names == ("steps",)
        assert plan.review_block is not None
        assert plan.review_block.kind == "global_review"
        assert plan.prior_review_block is not None
        assert plan.prior_review_block.kind == "global_prior_review"
        assert "review:prior_system" in {block.id for block in plan.all_blocks}
        assert [block.kind for block in plan.prior_blocks] == ["dynamics_prior"]
        assert set(plan.prior_blocks[0].parameter_names) == {
            "rho_activity",
            "sigma_activity",
            "t0_mean_activity",
            "t0_sd_activity",
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
        assert "`id`: `indicator:steps`" in prompt[1]["content"]
        assert (
            "Choose exactly one distribution/link pair for the active indicator."
            in prompt[1]["content"]
        )
        assert "| steps | activity | count | sum |" in prompt[1]["content"]
        assert "beta_activity_sleep" not in prompt[1]["content"]
        assert '"block_kind": "indicator_decision"' in prompt[1]["content"]
        assert "### Parameter Prior Cards" not in prompt[1]["content"]

    def test_current_turn_moves_to_next_pending_prior_block(self):
        messages = Stage4Messages(
            question="How does stress affect sleep?",
            model_topology={"model_clock": "1d", "model_interval_days": 1.0, "outcome": "sleep"},
            construct_scale_cards=[],
            prior_cards=[],
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
        runtime.block_status.update(
            {
                "effects:sleep": "accepted",
                "dynamics:sleep": "pending",
            }
        )
        runtime.accepted = Stage4AcceptedState(
            model_spec={"parameters": [{"name": "beta_stress_sleep"}, {"name": "rho_sleep"}]},
            authored_priors={"beta_stress_sleep": {"distribution": "Normal"}},
        )
        active_block = plan.get_block("dynamics:sleep")
        assert active_block is not None
        stage4_module._set_block_cursor(runtime, active_block)
        session = _make_session(messages=messages, plan=plan, runtime=runtime)

        turn = session.current_turn()
        assert turn is not None
        assert turn.block.id == "dynamics:sleep"
        assert "`id`: `dynamics:sleep`" in turn.messages[1]["content"]
        assert "rho_sleep" in turn.messages[1]["content"]
        assert "beta_stress_sleep" not in turn.messages[1]["content"]

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
        runtime.block_status["effects:sleep"] = "pending"
        runtime.accepted = Stage4AcceptedState(
            model_spec={"parameters": [{"name": "beta_stress_sleep"}]}
        )
        runtime.last_feedback = "submit priors"
        active_block = plan.get_block("effects:sleep")
        assert active_block is not None
        stage4_module._set_block_cursor(runtime, active_block)
        session = _make_session(
            messages=Stage4Messages(
                question="How does stress affect sleep?",
                model_topology={},
                construct_scale_cards=[],
                prior_cards=[],
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
        runtime.block_status[prior_review_block.id] = "reopened"
        runtime.accepted = Stage4AcceptedState(
            model_spec={"parameters": [{"name": "beta_stress_sleep"}]}
        )
        stage4_module._set_block_cursor(runtime, prior_review_block)
        session = _make_session(
            messages=Stage4Messages(
                question="How does stress affect sleep?",
                model_topology={},
                construct_scale_cards=[],
                prior_cards=[],
            ),
            plan=plan,
            runtime=runtime,
        )

        turn = session.current_turn()

        assert turn is not None
        assert turn.block.id == "review:prior_system"
        assert turn.allowed_tool_names == ("validate_model",)
        assert '"block_kind": "global_prior_review"' in turn.messages[1]["content"]
        assert '"block_id": "review:prior_system"' in turn.messages[1]["content"]


class TestStage4PromptScopePolicy:
    def test_policy_is_looked_up_by_block_kind(self):
        policy = get_stage4_prompt_scope_policy("measurement_prior")

        assert policy.system_task.startswith("Propose full prior specifications")
        assert policy.user_task.startswith("Propose full prior specifications")
        assert policy.visible_sections == ("construct_scale_cards", "prior_cards")
        assert policy.guidance_section_keys == (
            "prior_distribution_types",
            "parameter_guidance",
            "measurement_prior_guidance",
        )
        assert policy.parameter_guidance_prefixes == ("lambda",)
        assert policy.allowed_tool_names == ("validate_model", "elicit_prior_gmm")

    def test_effect_policy_keeps_search_enabled(self):
        policy = get_stage4_prompt_scope_policy("effect_prior")

        assert policy.parameter_guidance_prefixes == ("beta",)
        assert policy.allowed_tool_names == (
            "validate_model",
            "search_literature",
            "elicit_prior_gmm",
        )

    def test_dynamics_and_correlation_policies_disable_search(self):
        dynamics_policy = get_stage4_prompt_scope_policy("dynamics_prior")
        correlation_policy = get_stage4_prompt_scope_policy("correlation_prior")

        assert dynamics_policy.allowed_tool_names == ("validate_model", "elicit_prior_gmm")
        assert correlation_policy.allowed_tool_names == ("validate_model", "elicit_prior_gmm")

    def test_global_review_policy_is_validate_only(self):
        policy = get_stage4_prompt_scope_policy("global_review")

        assert policy.user_task.startswith("Review the locked model form")
        assert policy.visible_sections == (
            "distribution_cards",
            "loading_params",
            "construct_scale_cards",
        )
        assert policy.allowed_tool_names == ("validate_model",)

    def test_global_prior_review_policy_is_validate_only(self):
        policy = get_stage4_prompt_scope_policy("global_prior_review")

        assert policy.user_task.startswith("Review the full accepted prior system")
        assert policy.visible_sections == ("construct_scale_cards", "prior_cards")
        assert policy.parameter_guidance_prefixes == (
            "lambda",
            "rho",
            "sigma",
            "beta",
            "cor",
            "t0_mean",
            "t0_sd",
        )
        assert policy.allowed_tool_names == ("validate_model",)
