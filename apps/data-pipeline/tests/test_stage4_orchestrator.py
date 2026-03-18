"""Tests for Stage 4 deterministic skeleton and prior-card derivation."""

from causal_ssm_agent.orchestrator.stage4_orchestrator import (
    Stage4Skeleton,
    build_construct_scale_cards,
    build_distribution_cards,
    build_model_topology,
    build_prior_cards,
    derive_deterministic_spec,
)


def _make_causal_spec(
    constructs: list[dict],
    edges: list[dict],
    indicators: list[dict],
) -> dict:
    """Build a CausalSpec dict from components."""
    return {
        "latent": {"constructs": constructs, "edges": edges},
        "measurement": {"model_clock": "1d", "indicators": indicators},
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


# =============================================================================
# derive_deterministic_spec
# =============================================================================


class TestDeriveDeterministicSpec:
    def test_returns_stage4_skeleton(self):
        """Should return a typed deterministic skeleton."""
        result = derive_deterministic_spec(_simple_spec())
        assert isinstance(result, Stage4Skeleton)
        assert isinstance(result.resolved_likelihoods, list)
        assert isinstance(result.ambiguous_indicators, list)
        assert isinstance(result.parameters, list)
        assert isinstance(result.loading_params, list)

    def test_continuous_resolves_to_gaussian(self):
        """Continuous dtype should resolve to gaussian/identity."""
        skeleton = derive_deterministic_spec(_simple_spec())
        for lik in skeleton.resolved_likelihoods:
            assert lik["distribution"] == "gaussian"
            assert lik["link"] == "identity"

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

        skeleton = derive_deterministic_spec(spec)
        correlation_params = [p for p in skeleton.parameters if p["role"] == "correlation"]
        assert len(correlation_params) == 1
        assert correlation_params[0]["name"] == "cor_sleep_stress"
        assert correlation_params[0]["constraint"] == "correlation"
        assert correlation_params[0]["marginalized_confounder"] == "u_shared"


class TestBuildPriorCards:
    def test_prior_cards_reference_structural_context_only_once(self):
        skeleton = derive_deterministic_spec(_simple_spec())
        cards = build_prior_cards(skeleton)
        beta_card = next(card for card in cards if card["parameter"] == "beta_stress_sleep")
        sigma_card = next(card for card in cards if card["parameter"] == "sigma_sleep")

        assert beta_card["structural_context"] == {
            "cause": "stress",
            "effect": "sleep",
            "lagged": True,
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
