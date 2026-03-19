"""Tests for Stage 4 model proposal prompt formatting."""

from causal_ssm_agent.orchestrator.prompts.model_proposal import (
    format_construct_scale_cards,
    format_distribution_cards,
    format_loading_params,
    format_model_topology,
    format_prior_cards,
)


class TestFormatModelTopology:
    def test_empty_context(self):
        result = format_model_topology({})
        assert "none" in result.lower()

    def test_renders_model_metadata_and_edges(self):
        result = format_model_topology(
            {
                "model_clock": "1d",
                "model_interval_days": 1.0,
                "outcome": "sleep",
                "latent_edges": [
                    {
                        "cause": "stress",
                        "effect": "sleep",
                        "lagged": True,
                        "description": "Stress reduces subsequent sleep quality.",
                    }
                ],
            }
        )
        assert "model_clock" in result
        assert "model_interval_days" in result
        assert "stress" in result
        assert "sleep" in result
        assert "Stress reduces subsequent sleep quality." in result


class TestFormatDistributionCards:
    def test_empty_cards(self):
        result = format_distribution_cards([])
        assert "deterministic" in result.lower()

    def test_renders_indicator_options_profile_and_issues(self):
        result = format_distribution_cards(
            [
                {
                    "variable": "steps",
                    "construct": "activity",
                    "measurement_dtype": "count",
                    "aggregation": "sum",
                    "how_to_measure": "Count recorded steps per day.",
                    "options": [
                        {"distribution": "poisson", "links": ["log"], "distribution_fixed": False},
                        {
                            "distribution": "negative_binomial",
                            "links": ["log"],
                            "distribution_fixed": False,
                        },
                    ],
                    "profile": {
                        "n_obs": 100,
                        "mean": 12.4,
                        "std": 4.1,
                        "min": 0.0,
                        "max": 31.0,
                        "zero_fraction": 0.05,
                        "variance_to_mean_ratio": 1.7,
                        "is_nonnegative": True,
                        "looks_integer_valued": True,
                    },
                    "validation_issues": ["warning large_timestamp_gap"],
                }
            ]
        )
        assert "| Variable |" in result
        assert "steps" in result
        assert "Count recorded steps per day." in result
        assert "`poisson` → `log` (auto)" in result
        assert "`negative_binomial` → `log` (auto)" in result
        assert "var/mean=1.7" in result
        assert "warning large_timestamp_gap" in result


class TestFormatLoadingParams:
    def test_empty_list(self):
        result = format_loading_params([])
        assert "skip" in result.lower()

    def test_single_loading(self):
        result = format_loading_params(
            [{"name": "lambda_mood_pss", "indicator": "pss_score", "construct": "mood"}]
        )
        assert "lambda_mood_pss" in result
        assert "pss_score" in result
        assert "mood" in result
        assert "| Parameter |" not in result


class TestFormatConstructScaleCards:
    def test_empty_cards(self):
        result = format_construct_scale_cards([])
        assert "none" in result.lower()

    def test_renders_single_indicator_construct_inline(self):
        result = format_construct_scale_cards(
            [
                {
                    "construct": "stress",
                    "description": "Perceived psychological stress.",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                    "is_outcome": False,
                    "reference_indicator": "pss_score",
                    "indicators": [
                        {
                            "indicator": "pss_score",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                            "how_to_measure": "Use the pss column directly.",
                            "is_reference": True,
                            "has_distribution_decision_card": False,
                            "profile": {
                                "n_obs": 120,
                                "mean": 18.5,
                                "std": 7.1,
                                "min": 0.0,
                                "max": 40.0,
                            },
                        }
                    ],
                }
            ]
        )
        assert "### `stress`" in result
        assert "Perceived psychological stress." in result
        assert "`pss_score`" in result
        assert "Use the pss column directly." in result
        assert "mean=18.5" in result
        assert "| Indicator |" not in result

    def test_ambiguous_indicator_points_to_distribution_card(self):
        result = format_construct_scale_cards(
            [
                {
                    "construct": "stress",
                    "description": "Perceived psychological stress.",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                    "is_outcome": False,
                    "reference_indicator": "pss_score",
                    "indicators": [
                        {
                            "indicator": "pss_score",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                            "how_to_measure": "Use the pss column directly.",
                            "is_reference": True,
                            "has_distribution_decision_card": True,
                            "profile": {
                                "n_obs": 120,
                                "mean": 18.5,
                                "std": 7.1,
                            },
                        }
                    ],
                }
            ]
        )
        assert "see distribution decision card" in result
        assert "Use the pss column directly." not in result
        assert "mean=18.5" not in result

    def test_multi_indicator_construct_keeps_table(self):
        result = format_construct_scale_cards(
            [
                {
                    "construct": "stress",
                    "description": "Perceived psychological stress.",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                    "is_outcome": False,
                    "reference_indicator": "pss_score",
                    "indicators": [
                        {
                            "indicator": "pss_score",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                            "how_to_measure": "Use the pss column directly.",
                            "is_reference": True,
                            "has_distribution_decision_card": False,
                            "profile": {
                                "n_obs": 120,
                                "mean": 18.5,
                                "std": 7.1,
                            },
                        },
                        {
                            "indicator": "stress_diary",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                            "how_to_measure": "Use the diary score directly.",
                            "is_reference": False,
                            "has_distribution_decision_card": False,
                            "profile": {
                                "n_obs": 100,
                                "mean": 17.0,
                                "std": 5.0,
                            },
                        },
                    ],
                }
            ]
        )
        assert "| Indicator | Dtype | Aggregation | Reference | Details |" in result
        assert "stress_diary" in result


class TestFormatPriorCards:
    def test_empty_cards(self):
        result = format_prior_cards([])
        assert "none" in result.lower()

    def test_renders_compact_parameter_inventory(self):
        result = format_prior_cards(
            [
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
            ]
        )
        assert "#### Fixed Effects" in result
        assert "#### AR Coefficients" in result
        assert "beta_stress_sleep" in result
        assert "| beta_stress_sleep | stress | sleep | lagged | none |" in result
        assert "| rho_sleep | sleep | unit_interval |" in result
