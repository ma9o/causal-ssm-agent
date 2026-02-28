"""Tests for identifiability analysis and formatting functions.

Covers: _validate_max_lag_one, format_identifiability_report,
format_marginalization_report, analyze_unobserved_constructs,
get_correlation_pairs_from_marginalization, inject_marginalized_correlations.
"""

import pytest

from causal_ssm_agent.utils.identifiability import (
    _validate_max_lag_one,
    analyze_unobserved_constructs,
    format_identifiability_report,
    format_marginalization_report,
    get_correlation_pairs_from_marginalization,
    inject_marginalized_correlations,
)

# =============================================================================
# Fixtures
# =============================================================================


def _latent_model():
    """X -> Y, U -> X, U -> Y (U is unobserved confounder)."""
    return {
        "constructs": [
            {"name": "X"},
            {"name": "Y", "is_outcome": True},
            {"name": "U"},
        ],
        "edges": [
            {"cause": "X", "effect": "Y"},
            {"cause": "U", "effect": "X"},
            {"cause": "U", "effect": "Y"},
        ],
    }


def _measurement_model():
    """X and Y observed, U unobserved."""
    return {
        "indicators": [
            {"name": "x1", "construct_name": "X", "measurement_dtype": "continuous"},
            {"name": "y1", "construct_name": "Y", "measurement_dtype": "continuous"},
        ]
    }


def _id_result_all_identifiable():
    """All treatments identifiable via do-calculus."""
    return {
        "identifiable_treatments": {
            "X": {"method": "do_calculus", "estimand": "P(Y|do(X))", "marginalized_confounders": ["U"]},
        },
        "non_identifiable_treatments": {},
        "graph_info": {
            "observed_constructs": ["X", "Y"],
            "total_constructs": 3,
            "unobserved_confounders": ["U"],
            "n_directed_edges": 8,
        },
    }


def _id_result_non_identifiable():
    """X is non-identifiable due to U."""
    return {
        "identifiable_treatments": {},
        "non_identifiable_treatments": {
            "X": {"confounders": ["U"]},
        },
        "graph_info": {
            "observed_constructs": ["X", "Y"],
            "total_constructs": 3,
            "unobserved_confounders": ["U"],
            "n_directed_edges": 8,
        },
    }


# =============================================================================
# _validate_max_lag_one
# =============================================================================


class TestValidateMaxLagOne:
    def test_boolean_lagged_passes(self):
        lm = {"edges": [{"cause": "X", "effect": "Y", "lagged": True}]}
        _validate_max_lag_one(lm)  # should not raise

    def test_default_no_lagged_passes(self):
        lm = {"edges": [{"cause": "X", "effect": "Y"}]}
        _validate_max_lag_one(lm)  # should not raise

    def test_false_lagged_passes(self):
        lm = {"edges": [{"cause": "X", "effect": "Y", "lagged": False}]}
        _validate_max_lag_one(lm)  # should not raise

    def test_non_boolean_lagged_raises(self):
        lm = {"edges": [{"cause": "X", "effect": "Y", "lagged": 2}]}
        with pytest.raises(AssertionError, match="non-boolean"):
            _validate_max_lag_one(lm)

    def test_empty_edges_passes(self):
        _validate_max_lag_one({"edges": []})

    def test_no_edges_key_passes(self):
        _validate_max_lag_one({})


# =============================================================================
# format_identifiability_report
# =============================================================================


class TestFormatIdentifiabilityReport:
    def test_all_identifiable(self):
        result = _id_result_all_identifiable()
        report = format_identifiability_report(result, "Y")
        assert "All" in report
        assert "identifiable" in report.lower()
        assert "X" in report

    def test_non_identifiable(self):
        result = _id_result_non_identifiable()
        report = format_identifiability_report(result, "Y")
        assert "non-identifiable" in report.lower() or "non_identifiable" in report.lower()
        assert "X" in report
        assert "U" in report

    def test_graph_info_shown(self):
        result = _id_result_all_identifiable()
        report = format_identifiability_report(result, "Y")
        assert "Graph" in report
        assert "2/3" in report or ("2" in report and "3" in report)

    def test_mixed_result(self):
        result = {
            "identifiable_treatments": {
                "A": {"method": "do_calculus", "estimand": "P(Y|do(A))"},
            },
            "non_identifiable_treatments": {
                "B": {"confounders": ["U"]},
            },
            "graph_info": {
                "observed_constructs": ["A", "B", "Y"],
                "total_constructs": 4,
                "unobserved_confounders": ["U"],
                "n_directed_edges": 6,
            },
        }
        report = format_identifiability_report(result, "Y")
        assert "A" in report
        assert "B" in report
        assert "1/2" in report  # 1 non-identifiable out of 2 total

    def test_iv_method_shown(self):
        result = {
            "identifiable_treatments": {
                "X": {"method": "instrumental_variable", "estimand": "IV(Z)"},
            },
            "non_identifiable_treatments": {},
            "graph_info": {
                "observed_constructs": ["X", "Y", "Z"],
                "total_constructs": 3,
                "unobserved_confounders": [],
                "n_directed_edges": 4,
            },
        }
        report = format_identifiability_report(result, "Y")
        assert "instrumental_variable" in report

    def test_notes_displayed(self):
        result = {
            "identifiable_treatments": {},
            "non_identifiable_treatments": {
                "X": {"notes": "outcome is unobserved"},
            },
            "graph_info": {
                "observed_constructs": [],
                "total_constructs": 2,
                "unobserved_confounders": [],
                "n_directed_edges": 1,
            },
        }
        report = format_identifiability_report(result, "Y")
        assert "unobserved" in report.lower()


# =============================================================================
# format_marginalization_report
# =============================================================================


class TestFormatMargReportFormat:
    def test_can_marginalize(self):
        analysis = {
            "can_marginalize": {"U"},
            "marginalize_reason": {"U": "confounding handled by identification strategy"},
            "blocking_details": {},
        }
        report = format_marginalization_report(analysis)
        assert "MARGINALIZE" in report
        assert "U" in report

    def test_needs_modeling(self):
        analysis = {
            "can_marginalize": set(),
            "marginalize_reason": {},
            "blocking_details": {"U": ["X"]},
        }
        report = format_marginalization_report(analysis)
        assert "NEEDS MODELING" in report
        assert "U" in report
        assert "X" in report

    def test_all_observed(self):
        analysis = {
            "can_marginalize": set(),
            "marginalize_reason": {},
            "blocking_details": {},
        }
        report = format_marginalization_report(analysis)
        assert "All constructs are observed" in report

    def test_mixed(self):
        analysis = {
            "can_marginalize": {"U1"},
            "marginalize_reason": {"U1": "single child"},
            "blocking_details": {"U2": ["X"]},
        }
        report = format_marginalization_report(analysis)
        assert "U1" in report
        assert "U2" in report


# =============================================================================
# analyze_unobserved_constructs
# =============================================================================


class TestAnalyzeUnobservedConstructs:
    def test_can_marginalize_when_identifiable(self):
        """If all effects are identifiable, confounders can be marginalized."""
        result = analyze_unobserved_constructs(
            _latent_model(),
            _measurement_model(),
            _id_result_all_identifiable(),
        )
        assert "U" in result["can_marginalize"]
        assert "U" not in result["blocking_details"]

    def test_blocking_when_not_identifiable(self):
        """If X is not identifiable due to U, U cannot be marginalized."""
        result = analyze_unobserved_constructs(
            _latent_model(),
            _measurement_model(),
            _id_result_non_identifiable(),
        )
        assert "U" not in result["can_marginalize"]
        assert "U" in result["blocking_details"]
        assert "X" in result["blocking_details"]["U"]

    def test_all_observed_empty(self):
        """With all constructs observed, nothing to marginalize."""
        lm = {
            "constructs": [{"name": "X"}, {"name": "Y"}],
            "edges": [{"cause": "X", "effect": "Y"}],
        }
        mm = {
            "indicators": [
                {"name": "x1", "construct_name": "X"},
                {"name": "y1", "construct_name": "Y"},
            ]
        }
        id_result = {
            "identifiable_treatments": {"X": {}},
            "non_identifiable_treatments": {},
            "graph_info": {
                "observed_constructs": ["X", "Y"],
                "total_constructs": 2,
                "unobserved_confounders": [],
                "n_directed_edges": 4,
            },
        }
        result = analyze_unobserved_constructs(lm, mm, id_result)
        assert result["can_marginalize"] == set()
        assert result["blocking_details"] == {}

    def test_marginalize_reason_provided(self):
        result = analyze_unobserved_constructs(
            _latent_model(),
            _measurement_model(),
            _id_result_all_identifiable(),
        )
        assert "U" in result["marginalize_reason"]
        assert len(result["marginalize_reason"]["U"]) > 0


# =============================================================================
# get_correlation_pairs_from_marginalization
# =============================================================================


class TestGetCorrelationPairs:
    def test_returns_pairs_for_marginalizable_confounder(self):
        """U -> X, U -> Y with U marginalizable should produce (X, Y, U)."""
        pairs = get_correlation_pairs_from_marginalization(
            _latent_model(),
            _measurement_model(),
            _id_result_all_identifiable(),
        )
        assert len(pairs) == 1
        s1, s2, confounder = pairs[0]
        assert confounder == "U"
        assert {s1, s2} == {"X", "Y"}

    def test_no_pairs_when_blocking(self):
        """Blocking confounders should not produce correlation pairs."""
        pairs = get_correlation_pairs_from_marginalization(
            _latent_model(),
            _measurement_model(),
            _id_result_non_identifiable(),
        )
        assert pairs == []

    def test_no_pairs_all_observed(self):
        lm = {
            "constructs": [{"name": "X"}, {"name": "Y"}],
            "edges": [{"cause": "X", "effect": "Y"}],
        }
        mm = {
            "indicators": [
                {"name": "x1", "construct_name": "X"},
                {"name": "y1", "construct_name": "Y"},
            ]
        }
        id_result = {
            "identifiable_treatments": {"X": {}},
            "non_identifiable_treatments": {},
            "graph_info": {
                "observed_constructs": ["X", "Y"],
                "total_constructs": 2,
                "unobserved_confounders": [],
                "n_directed_edges": 4,
            },
        }
        pairs = get_correlation_pairs_from_marginalization(lm, mm, id_result)
        assert pairs == []

    def test_pairs_sorted_lexicographically(self):
        """state1 < state2 within each tuple."""
        pairs = get_correlation_pairs_from_marginalization(
            _latent_model(),
            _measurement_model(),
            _id_result_all_identifiable(),
        )
        for s1, s2, _confounder in pairs:
            assert s1 < s2


# =============================================================================
# inject_marginalized_correlations
# =============================================================================


class TestInjectMarginalizedCorrelations:
    def test_adds_correlation_parameter(self):
        model_spec = {"parameters": []}
        causal_spec = {
            "latent": _latent_model(),
            "measurement": _measurement_model(),
            "identifiability": _id_result_all_identifiable(),
        }
        inject_marginalized_correlations(model_spec, causal_spec)
        params = model_spec["parameters"]
        assert len(params) == 1
        assert params[0]["name"] == "cor_X_Y"
        assert params[0]["role"] == "correlation"
        assert params[0]["constraint"] == "correlation"
        assert "U" in params[0]["description"]

    def test_no_duplicates(self):
        """Should not add the same parameter twice."""
        model_spec = {"parameters": [{"name": "cor_X_Y", "role": "correlation"}]}
        causal_spec = {
            "latent": _latent_model(),
            "measurement": _measurement_model(),
            "identifiability": _id_result_all_identifiable(),
        }
        inject_marginalized_correlations(model_spec, causal_spec)
        cor_params = [p for p in model_spec["parameters"] if p["name"] == "cor_X_Y"]
        assert len(cor_params) == 1

    def test_no_identifiability_noop(self):
        model_spec = {"parameters": []}
        causal_spec = {"latent": {}, "measurement": {}}
        inject_marginalized_correlations(model_spec, causal_spec)
        assert model_spec["parameters"] == []

    def test_non_identifiable_no_correlation(self):
        """Blocking confounders should not get correlation parameters."""
        model_spec = {"parameters": []}
        causal_spec = {
            "latent": _latent_model(),
            "measurement": _measurement_model(),
            "identifiability": _id_result_non_identifiable(),
        }
        inject_marginalized_correlations(model_spec, causal_spec)
        assert model_spec["parameters"] == []
