"""Tests for model proposal prompt formatting functions.

Covers: format_resolved_likelihoods, format_ambiguous_indicators,
format_parameters, format_loading_params, format_constructs,
format_edges, format_indicators.
"""

from causal_ssm_agent.orchestrator.prompts.model_proposal import (
    format_ambiguous_indicators,
    format_constructs,
    format_edges,
    format_indicators,
    format_loading_params,
    format_parameters,
    format_resolved_likelihoods,
)

# =============================================================================
# format_resolved_likelihoods
# =============================================================================


class TestFormatResolvedLikelihoods:
    def test_empty_list(self):
        result = format_resolved_likelihoods([])
        assert "none" in result.lower()

    def test_single_likelihood(self):
        result = format_resolved_likelihoods(
            [
                {
                    "variable": "mood",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "continuous",
                }
            ]
        )
        assert "mood" in result
        assert "gaussian" in result
        assert "identity" in result
        assert "continuous" in result

    def test_markdown_table_header(self):
        result = format_resolved_likelihoods(
            [{"variable": "x", "distribution": "d", "link": "l", "reasoning": "r"}]
        )
        assert "Variable" in result
        assert "Distribution" in result
        assert "|" in result

    def test_multiple_likelihoods(self):
        result = format_resolved_likelihoods(
            [
                {
                    "variable": "mood",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "r1",
                },
                {
                    "variable": "smoke",
                    "distribution": "bernoulli",
                    "link": "logit",
                    "reasoning": "r2",
                },
            ]
        )
        assert "mood" in result
        assert "smoke" in result


# =============================================================================
# format_ambiguous_indicators
# =============================================================================


class TestFormatAmbiguousIndicators:
    def test_empty_list(self):
        result = format_ambiguous_indicators([])
        assert "none" in result.lower()

    def test_fixed_distribution_choose_link(self):
        result = format_ambiguous_indicators(
            [
                {
                    "variable": "smoke",
                    "dtype": "binary",
                    "fixed_distribution": "bernoulli",
                    "valid_links": ["logit", "probit"],
                }
            ]
        )
        assert "smoke" in result
        assert "bernoulli" in result
        assert "logit" in result
        assert "probit" in result

    def test_choose_distribution(self):
        result = format_ambiguous_indicators(
            [
                {
                    "variable": "steps",
                    "dtype": "count",
                    "valid_distributions": ["poisson", "negative_binomial"],
                    "link_options": {"poisson": ["log"], "negative_binomial": ["log"]},
                }
            ]
        )
        assert "steps" in result
        assert "poisson" in result
        assert "negative_binomial" in result

    def test_auto_link_shown(self):
        result = format_ambiguous_indicators(
            [
                {
                    "variable": "x",
                    "dtype": "count",
                    "valid_distributions": ["poisson"],
                    "link_options": {"poisson": ["log"]},
                }
            ]
        )
        assert "auto" in result


# =============================================================================
# format_parameters
# =============================================================================


class TestFormatParameters:
    def test_empty_list(self):
        result = format_parameters([])
        assert "none" in result.lower()

    def test_single_parameter(self):
        result = format_parameters(
            [
                {
                    "name": "beta_X_Y",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "description": "Effect of X on Y",
                }
            ]
        )
        assert "beta_X_Y" in result
        assert "fixed_effect" in result
        assert "Effect of X on Y" in result

    def test_loading_role_annotated(self):
        result = format_parameters(
            [
                {
                    "name": "lambda_mood",
                    "role": "loading",
                    "constraint": "positive",
                    "description": "Loading for mood",
                }
            ]
        )
        assert "you decide" in result

    def test_non_loading_no_annotation(self):
        result = format_parameters(
            [
                {
                    "name": "sigma_X",
                    "role": "residual_sd",
                    "constraint": "positive",
                    "description": "SD for X",
                }
            ]
        )
        assert "you decide" not in result

    def test_markdown_table(self):
        result = format_parameters(
            [{"name": "x", "role": "r", "constraint": "c", "description": "d"}]
        )
        assert "Name" in result
        assert "Role" in result
        assert "|" in result


# =============================================================================
# format_loading_params
# =============================================================================


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

    def test_markdown_table(self):
        result = format_loading_params([{"name": "l", "indicator": "i", "construct": "c"}])
        assert "Parameter" in result
        assert "Indicator" in result
        assert "|" in result


# =============================================================================
# format_constructs
# =============================================================================


class TestFormatConstructs:
    def test_basic_construct(self):
        spec = {
            "latent": {
                "constructs": [
                    {
                        "name": "stress",
                        "role": "exogenous",
                        "temporal_status": "time_varying",
                        "temporal_scale": "daily",
                    }
                ]
            }
        }
        result = format_constructs(spec)
        assert "stress" in result
        assert "exogenous" in result
        assert "time_varying" in result

    def test_outcome_annotated(self):
        spec = {
            "latent": {
                "constructs": [
                    {
                        "name": "sleep",
                        "role": "endogenous",
                        "temporal_status": "time_varying",
                        "is_outcome": True,
                    }
                ]
            }
        }
        result = format_constructs(spec)
        assert "OUTCOME" in result

    def test_empty_constructs(self):
        result = format_constructs({"latent": {"constructs": []}})
        assert result == ""


# =============================================================================
# format_edges
# =============================================================================


class TestFormatEdges:
    def test_basic_edge(self):
        spec = {
            "latent": {
                "edges": [
                    {"cause": "stress", "effect": "sleep", "description": "Stress disrupts sleep"}
                ]
            }
        }
        result = format_edges(spec)
        assert "stress" in result
        assert "sleep" in result

    def test_lagged_default(self):
        spec = {"latent": {"edges": [{"cause": "X", "effect": "Y", "lagged": True}]}}
        result = format_edges(spec)
        assert "lagged" in result

    def test_contemporaneous(self):
        spec = {"latent": {"edges": [{"cause": "X", "effect": "Y", "lagged": False}]}}
        result = format_edges(spec)
        assert "contemporaneous" in result

    def test_empty_edges(self):
        result = format_edges({"latent": {"edges": []}})
        assert result == ""


# =============================================================================
# format_indicators
# =============================================================================


class TestFormatIndicators:
    def test_basic_indicator(self):
        spec = {
            "measurement": {
                "indicators": [
                    {
                        "name": "pss_score",
                        "construct_name": "stress",
                        "measurement_dtype": "continuous",
                        "aggregation": "mean",
                    }
                ]
            }
        }
        result = format_indicators(spec)
        assert "pss_score" in result
        assert "stress" in result
        assert "continuous" in result
        assert "mean" in result

    def test_empty_indicators(self):
        result = format_indicators({"measurement": {"indicators": []}})
        assert result == ""

    def test_multiple_indicators(self):
        spec = {
            "measurement": {
                "indicators": [
                    {
                        "name": "pss",
                        "construct_name": "stress",
                        "measurement_dtype": "continuous",
                        "aggregation": "mean",
                    },
                    {
                        "name": "hrs",
                        "construct_name": "sleep",
                        "measurement_dtype": "continuous",
                        "aggregation": "mean",
                    },
                ]
            }
        }
        result = format_indicators(spec)
        assert "pss" in result
        assert "hrs" in result
