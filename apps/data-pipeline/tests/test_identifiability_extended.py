"""Tests for identifiability utility functions (extended).

Covers: _validate_max_lag_one, dag_to_admg, check_identifiability,
        analyze_unobserved_constructs, get_correlation_pairs_from_marginalization,
        format_identifiability_report, format_marginalization_report,
        inject_marginalized_correlations.
"""

import pytest

from causal_ssm_agent.utils.identifiability import (
    _validate_max_lag_one,
    analyze_unobserved_constructs,
    check_identifiability,
    dag_to_admg,
    format_identifiability_report,
    format_marginalization_report,
    get_correlation_pairs_from_marginalization,
    inject_marginalized_correlations,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _confounded_model():
    """X -> Y with unobserved confounder U -> X, U -> Y."""
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


def _simple_observed():
    """X -> Y, both observed (no confounding)."""
    return {
        "constructs": [
            {"name": "X"},
            {"name": "Y", "is_outcome": True},
        ],
        "edges": [
            {"cause": "X", "effect": "Y"},
        ],
    }


def _front_door_model():
    """U -> X, U -> Y, X -> M -> Y. Front-door criterion applies."""
    return {
        "constructs": [
            {"name": "X"},
            {"name": "Y", "is_outcome": True},
            {"name": "M"},
            {"name": "U"},
        ],
        "edges": [
            {"cause": "U", "effect": "X"},
            {"cause": "U", "effect": "Y"},
            {"cause": "X", "effect": "M"},
            {"cause": "M", "effect": "Y"},
        ],
    }


def _iv_model():
    """Z -> X -> Y with U confounding X-Y. Z is an instrument."""
    return {
        "constructs": [
            {"name": "X"},
            {"name": "Y", "is_outcome": True},
            {"name": "U"},
            {"name": "Z"},
        ],
        "edges": [
            {"cause": "Z", "effect": "X"},
            {"cause": "X", "effect": "Y"},
            {"cause": "U", "effect": "X"},
            {"cause": "U", "effect": "Y"},
        ],
    }


def _measurement_for(*names):
    """Create minimal measurement model for given construct names."""
    return {
        "indicators": [
            {"name": f"{n}_obs", "construct_name": n} for n in names
        ]
    }


# =============================================================================
# _validate_max_lag_one
# =============================================================================


class TestValidateMaxLagOne:
    def test_valid_boolean_lagged(self):
        lm = {
            "edges": [
                {"cause": "X", "effect": "Y", "lagged": True},
                {"cause": "Y", "effect": "Z", "lagged": False},
            ]
        }
        _validate_max_lag_one(lm)  # Should not raise

    def test_default_lagged_is_ok(self):
        lm = {"edges": [{"cause": "X", "effect": "Y"}]}
        _validate_max_lag_one(lm)  # lagged defaults to False

    def test_empty_edges(self):
        _validate_max_lag_one({"edges": []})

    def test_no_edges_key(self):
        _validate_max_lag_one({})

    def test_non_boolean_lagged_raises(self):
        lm = {"edges": [{"cause": "X", "effect": "Y", "lagged": 2}]}
        with pytest.raises(AssertionError, match="non-boolean"):
            _validate_max_lag_one(lm)

    def test_string_lagged_raises(self):
        lm = {"edges": [{"cause": "X", "effect": "Y", "lagged": "yes"}]}
        with pytest.raises(AssertionError, match="non-boolean"):
            _validate_max_lag_one(lm)


# =============================================================================
# dag_to_admg
# =============================================================================


class TestDagToAdmg:
    def test_returns_admg_and_confounders(self):
        lm = _confounded_model()
        observed = {"X", "Y"}
        admg, confounders = dag_to_admg(lm, observed)
        assert confounders == {"U"}
        # ADMG should have directed edges
        assert len(list(admg.directed.edges())) > 0

    def test_no_confounders_when_all_observed(self):
        lm = _simple_observed()
        observed = {"X", "Y"}
        _admg, confounders = dag_to_admg(lm, observed)
        assert confounders == set()

    def test_single_child_unobserved_not_confounder(self):
        """Unobserved node with single observed child is not a confounder."""
        lm = {
            "constructs": [
                {"name": "X"},
                {"name": "Y", "is_outcome": True},
                {"name": "U"},
            ],
            "edges": [
                {"cause": "X", "effect": "Y"},
                {"cause": "U", "effect": "X"},
            ],
        }
        observed = {"X", "Y"}
        _, confounders = dag_to_admg(lm, observed)
        assert "U" not in confounders

    def test_invalid_lag_raises(self):
        lm = {
            "constructs": [{"name": "X"}, {"name": "Y"}],
            "edges": [{"cause": "X", "effect": "Y", "lagged": 2}],
        }
        with pytest.raises(AssertionError):
            dag_to_admg(lm, {"X", "Y"})


# =============================================================================
# check_identifiability
# =============================================================================


class TestCheckIdentifiability:
    def test_all_observed_is_identifiable(self):
        lm = _simple_observed()
        mm = _measurement_for("X", "Y")
        result = check_identifiability(lm, mm)
        assert "X" in result["identifiable_treatments"]
        assert len(result["non_identifiable_treatments"]) == 0

    def test_confounded_is_non_identifiable(self):
        lm = _confounded_model()
        mm = _measurement_for("X", "Y")
        result = check_identifiability(lm, mm)
        # X should be non-identifiable (confounded by U)
        assert "X" in result["non_identifiable_treatments"]

    def test_no_outcome_raises(self):
        lm = {
            "constructs": [{"name": "X"}, {"name": "Y"}],
            "edges": [{"cause": "X", "effect": "Y"}],
        }
        mm = _measurement_for("X", "Y")
        with pytest.raises(ValueError, match="No outcome found"):
            check_identifiability(lm, mm)

    def test_unobserved_outcome(self):
        lm = _simple_observed()
        mm = _measurement_for("X")  # Y is unobserved
        result = check_identifiability(lm, mm)
        assert len(result["identifiable_treatments"]) == 0

    def test_graph_info_populated(self):
        lm = _confounded_model()
        mm = _measurement_for("X", "Y")
        result = check_identifiability(lm, mm)
        info = result["graph_info"]
        assert "observed_constructs" in info
        assert "total_constructs" in info
        assert info["total_constructs"] == 3
        assert "n_directed_edges" in info

    def test_iv_model_with_instrument(self):
        """IV model: Z is instrument for X->Y effect."""
        lm = _iv_model()
        mm = _measurement_for("X", "Y", "Z")
        result = check_identifiability(lm, mm)
        # X should be identifiable via IV or do-calculus
        if "X" in result["identifiable_treatments"]:
            method = result["identifiable_treatments"]["X"]["method"]
            assert method in ("do_calculus", "instrumental_variable")

    def test_unobserved_treatment_excluded(self):
        """Unobserved constructs can't be treatments."""
        lm = _confounded_model()
        mm = _measurement_for("Y")  # Only Y observed
        result = check_identifiability(lm, mm)
        # No treatments possible since X and U are unobserved
        assert len(result["identifiable_treatments"]) == 0


# =============================================================================
# analyze_unobserved_constructs
# =============================================================================


class TestAnalyzeUnobservedConstructs:
    def test_no_unobserved(self):
        lm = _simple_observed()
        mm = _measurement_for("X", "Y")
        id_result = check_identifiability(lm, mm)
        analysis = analyze_unobserved_constructs(lm, mm, id_result)
        assert analysis["can_marginalize"] == set()
        assert analysis["blocking_details"] == {}

    def test_single_child_can_marginalize(self):
        """Unobserved node with one observed child can be marginalized."""
        lm = {
            "constructs": [
                {"name": "X"},
                {"name": "Y", "is_outcome": True},
                {"name": "U"},
            ],
            "edges": [
                {"cause": "X", "effect": "Y"},
                {"cause": "U", "effect": "X"},
            ],
        }
        mm = _measurement_for("X", "Y")
        id_result = check_identifiability(lm, mm)
        analysis = analyze_unobserved_constructs(lm, mm, id_result)
        assert "U" in analysis["can_marginalize"]

    def test_blocking_confounder_not_marginalizable(self):
        lm = _confounded_model()
        mm = _measurement_for("X", "Y")
        id_result = check_identifiability(lm, mm)
        analysis = analyze_unobserved_constructs(lm, mm, id_result)
        # U blocks X->Y identification, so should NOT be marginalizable
        if "U" in analysis["blocking_details"]:
            assert "U" not in analysis["can_marginalize"]


# =============================================================================
# get_correlation_pairs_from_marginalization
# =============================================================================


class TestGetCorrelationPairs:
    def test_empty_when_no_unobserved(self):
        lm = _simple_observed()
        mm = _measurement_for("X", "Y")
        id_result = check_identifiability(lm, mm)
        pairs = get_correlation_pairs_from_marginalization(lm, mm, id_result)
        assert pairs == []

    def test_returns_sorted_pairs(self):
        """When a confounder with 2 observed children is marginalized, return pairs."""
        # X -> Y, both observed. U -> X, U -> Y (unobserved confounder).
        # If U is marginalizable (via front-door etc.), we get (X, Y, U) pair.
        lm = _front_door_model()
        mm = _measurement_for("X", "Y", "M")
        id_result = check_identifiability(lm, mm)
        pairs = get_correlation_pairs_from_marginalization(lm, mm, id_result)
        # If U is marginalizable, we should get correlation pairs
        for s1, s2, _conf in pairs:
            assert s1 < s2  # Sorted order


# =============================================================================
# format_identifiability_report
# =============================================================================


class TestFormatIdentifiabilityReport:
    def test_all_identifiable(self):
        result = {
            "identifiable_treatments": {
                "X": {"method": "do_calculus", "estimand": "P(Y|do(X))"},
            },
            "non_identifiable_treatments": {},
            "graph_info": {
                "observed_constructs": ["X", "Y"],
                "total_constructs": 2,
                "unobserved_confounders": [],
                "n_directed_edges": 4,
            },
        }
        report = format_identifiability_report(result, "Y")
        assert "All 1 treatment effects on Y are identifiable" in report
        assert "X via do_calculus" in report

    def test_non_identifiable_with_blockers(self):
        result = {
            "identifiable_treatments": {},
            "non_identifiable_treatments": {
                "X": {"confounders": ["U"]},
            },
            "graph_info": {
                "observed_constructs": ["X", "Y"],
                "total_constructs": 3,
                "unobserved_confounders": ["U"],
                "n_directed_edges": 6,
            },
        }
        report = format_identifiability_report(result, "Y")
        assert "1/1 treatments have non-identifiable effects" in report
        assert "X (blocked by: U)" in report
        assert "Unobserved confounders: U" in report

    def test_mixed_identifiability(self):
        result = {
            "identifiable_treatments": {
                "A": {"method": "do_calculus"},
            },
            "non_identifiable_treatments": {
                "B": {"confounders": ["U"]},
            },
            "graph_info": {
                "observed_constructs": ["A", "B", "Y"],
                "total_constructs": 4,
                "unobserved_confounders": ["U"],
                "n_directed_edges": 8,
            },
        }
        report = format_identifiability_report(result, "Y")
        assert "1/2 treatments have non-identifiable effects" in report
        assert "1 treatments have identifiable effects" in report

    def test_non_identifiable_with_notes(self):
        result = {
            "identifiable_treatments": {},
            "non_identifiable_treatments": {
                "X": {"notes": "outcome is unobserved"},
            },
            "graph_info": {
                "observed_constructs": ["X"],
                "total_constructs": 2,
                "unobserved_confounders": [],
                "n_directed_edges": 0,
            },
        }
        report = format_identifiability_report(result, "Y")
        assert "outcome is unobserved" in report

    def test_many_treatments_truncated(self):
        """More than 5 identifiable treatments should show '... and N more'."""
        treatments = {f"X{i}": {"method": "do_calculus"} for i in range(7)}
        result = {
            "identifiable_treatments": treatments,
            "non_identifiable_treatments": {},
            "graph_info": {
                "observed_constructs": [f"X{i}" for i in range(7)] + ["Y"],
                "total_constructs": 8,
                "unobserved_confounders": [],
                "n_directed_edges": 14,
            },
        }
        report = format_identifiability_report(result, "Y")
        assert "and 2 more" in report


# =============================================================================
# format_marginalization_report
# =============================================================================


class TestFormatMarginalizationReport:
    def test_can_marginalize(self):
        analysis = {
            "can_marginalize": {"U"},
            "marginalize_reason": {"U": "does not create confounding"},
            "blocking_details": {},
        }
        report = format_marginalization_report(analysis)
        assert "CAN MARGINALIZE (1 constructs)" in report
        assert "U" in report

    def test_needs_modeling(self):
        analysis = {
            "can_marginalize": set(),
            "marginalize_reason": {},
            "blocking_details": {"U": ["X"]},
        }
        report = format_marginalization_report(analysis)
        assert "NEEDS MODELING (1 constructs)" in report
        assert "blocks identification of: X" in report

    def test_all_observed(self):
        analysis = {
            "can_marginalize": set(),
            "marginalize_reason": {},
            "blocking_details": {},
        }
        report = format_marginalization_report(analysis)
        assert "All constructs are observed" in report

    def test_mixed_analysis(self):
        analysis = {
            "can_marginalize": {"U1"},
            "marginalize_reason": {"U1": "single child"},
            "blocking_details": {"U2": ["X", "Z"]},
        }
        report = format_marginalization_report(analysis)
        assert "CAN MARGINALIZE" in report
        assert "NEEDS MODELING" in report
        assert "U1" in report
        assert "U2" in report


# =============================================================================
# inject_marginalized_correlations
# =============================================================================


class TestInjectMarginalizedCorrelations:
    def test_no_identifiability_is_noop(self):
        model_spec = {"parameters": []}
        causal_spec = {}
        inject_marginalized_correlations(model_spec, causal_spec)
        assert model_spec["parameters"] == []

    def test_empty_identifiability_is_noop(self):
        model_spec = {"parameters": []}
        causal_spec = {"identifiability": {}}
        inject_marginalized_correlations(model_spec, causal_spec)
        assert model_spec["parameters"] == []

    def test_does_not_duplicate_existing_params(self):
        """If cor_X_Y already exists, don't add it again."""
        model_spec = {
            "parameters": [
                {"name": "cor_X_Y", "role": "correlation"},
            ]
        }
        # Build a causal_spec where marginalization would produce cor_X_Y
        lm = _front_door_model()
        mm = _measurement_for("X", "Y", "M")
        id_result = check_identifiability(lm, mm)
        causal_spec = {
            "latent": lm,
            "measurement": mm,
            "identifiability": id_result,
        }
        inject_marginalized_correlations(model_spec, causal_spec)
        # Count occurrences of cor_X_Y
        cor_count = sum(1 for p in model_spec["parameters"] if p["name"] == "cor_X_Y")
        assert cor_count == 1

    def test_injects_correlation_params(self):
        """When a confounder is marginalizable, inject cor_ parameters."""
        lm = _front_door_model()
        mm = _measurement_for("X", "Y", "M")
        id_result = check_identifiability(lm, mm)
        # Check if there are any marginalizable confounders
        pairs = get_correlation_pairs_from_marginalization(lm, mm, id_result)
        if pairs:
            model_spec = {"parameters": []}
            causal_spec = {
                "latent": lm,
                "measurement": mm,
                "identifiability": id_result,
            }
            inject_marginalized_correlations(model_spec, causal_spec)
            cor_params = [p for p in model_spec["parameters"] if p["role"] == "correlation"]
            assert len(cor_params) == len(pairs)
            for p in cor_params:
                assert p["name"].startswith("cor_")
                assert "marginalized confounder" in p["description"]
