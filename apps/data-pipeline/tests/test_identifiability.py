"""Tests for identifiability utility functions.

Covers: get_observed_constructs, _is_time_varying, _node_name,
        unroll_temporal_dag, find_blocking_confounders, find_instruments.
"""

import pytest

import networkx as nx

from causal_ssm_agent.utils.identifiability import (
    _is_time_varying,
    _node_name,
    find_blocking_confounders,
    find_instruments,
    get_observed_constructs,
    unroll_temporal_dag,
)


def _simple_latent():
    """Simple 3-node latent model: X -> Y, U -> X, U -> Y."""
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


# =============================================================================
# get_observed_constructs
# =============================================================================


class TestGetObservedConstructs:
    def test_basic(self):
        mm = {
            "indicators": [
                {"name": "x1", "construct_name": "X"},
                {"name": "y1", "construct_name": "Y"},
            ]
        }
        result = get_observed_constructs(mm)
        assert result == {"X", "Y"}

    def test_empty_indicators(self):
        assert get_observed_constructs({"indicators": []}) == set()

    def test_empty_dict(self):
        assert get_observed_constructs({}) == set()

    def test_missing_construct_name(self):
        mm = {"indicators": [{"name": "x1"}]}
        assert get_observed_constructs(mm) == set()

    def test_multiple_indicators_same_construct(self):
        mm = {
            "indicators": [
                {"name": "x1", "construct_name": "X"},
                {"name": "x2", "construct_name": "X"},
            ]
        }
        result = get_observed_constructs(mm)
        assert result == {"X"}


# =============================================================================
# _is_time_varying
# =============================================================================


class TestIsTimeVarying:
    def test_default_is_time_varying(self):
        lm = {"constructs": [{"name": "X"}]}
        assert _is_time_varying(lm, "X") is True

    def test_explicit_time_varying(self):
        lm = {"constructs": [{"name": "X", "temporal_status": "time_varying"}]}
        assert _is_time_varying(lm, "X") is True

    def test_time_invariant(self):
        lm = {"constructs": [{"name": "X", "temporal_status": "time_invariant"}]}
        assert _is_time_varying(lm, "X") is False

    def test_not_found_raises_value_error(self):
        lm = {"constructs": [{"name": "X"}]}
        with pytest.raises(ValueError, match="not found in latent model"):
            _is_time_varying(lm, "Z")


# =============================================================================
# _node_name
# =============================================================================


class TestNodeName:
    def test_current_timestep(self):
        assert _node_name("stress", "t") == "stress_t"

    def test_lagged_timestep(self):
        assert _node_name("X", "{t-1}") == "X_{t-1}"


# =============================================================================
# unroll_temporal_dag
# =============================================================================


class TestUnrollTemporalDag:
    def test_simple_two_node(self):
        """Two time-varying nodes, one edge: should get 4 nodes (2 per construct)."""
        lm = {
            "constructs": [{"name": "X"}, {"name": "Y"}],
            "edges": [{"cause": "X", "effect": "Y"}],
        }
        observed = {"X", "Y"}
        dag = unroll_temporal_dag(lm, observed)
        # 2 constructs * 2 timesteps = 4 nodes
        assert dag.number_of_nodes() == 4
        assert "X_t" in dag
        assert "X_{t-1}" in dag
        assert "Y_t" in dag
        assert "Y_{t-1}" in dag

    def test_ar1_edges_for_observed(self):
        """Observed time-varying constructs should get AR(1) edges."""
        lm = {
            "constructs": [{"name": "X"}, {"name": "Y"}],
            "edges": [{"cause": "X", "effect": "Y"}],
        }
        observed = {"X", "Y"}
        dag = unroll_temporal_dag(lm, observed)
        assert dag.has_edge("X_{t-1}", "X_t")
        assert dag.has_edge("Y_{t-1}", "Y_t")

    def test_no_ar1_for_hidden(self):
        """Unobserved constructs should NOT get AR(1) edges."""
        lm = {
            "constructs": [{"name": "X"}, {"name": "Y"}, {"name": "U"}],
            "edges": [
                {"cause": "X", "effect": "Y"},
                {"cause": "U", "effect": "X"},
            ],
        }
        observed = {"X", "Y"}
        dag = unroll_temporal_dag(lm, observed)
        assert not dag.has_edge("U_{t-1}", "U_t")

    def test_contemporaneous_edge(self):
        """Non-lagged edge should be X_t -> Y_t and X_{t-1} -> Y_{t-1}."""
        lm = {
            "constructs": [{"name": "X"}, {"name": "Y"}],
            "edges": [{"cause": "X", "effect": "Y", "lagged": False}],
        }
        observed = {"X", "Y"}
        dag = unroll_temporal_dag(lm, observed)
        assert dag.has_edge("X_t", "Y_t")
        assert dag.has_edge("X_{t-1}", "Y_{t-1}")

    def test_lagged_edge(self):
        """Lagged edge should be X_{t-1} -> Y_t only."""
        lm = {
            "constructs": [{"name": "X"}, {"name": "Y"}],
            "edges": [{"cause": "X", "effect": "Y", "lagged": True}],
        }
        observed = {"X", "Y"}
        dag = unroll_temporal_dag(lm, observed)
        assert dag.has_edge("X_{t-1}", "Y_t")
        assert not dag.has_edge("X_t", "Y_t")

    def test_hidden_nodes_labeled(self):
        """Unobserved constructs should have hidden=True."""
        lm = {
            "constructs": [{"name": "X"}, {"name": "U"}],
            "edges": [{"cause": "U", "effect": "X"}],
        }
        observed = {"X"}
        dag = unroll_temporal_dag(lm, observed)
        assert dag.nodes["X_t"]["hidden"] is False
        assert dag.nodes["U_t"]["hidden"] is True

    def test_time_invariant_single_node(self):
        """Time-invariant construct should get a single node (no timestep)."""
        lm = {
            "constructs": [
                {"name": "X"},
                {"name": "trait", "temporal_status": "time_invariant"},
            ],
            "edges": [{"cause": "trait", "effect": "X"}],
        }
        observed = {"X", "trait"}
        dag = unroll_temporal_dag(lm, observed)
        assert "trait" in dag
        assert "trait_t" not in dag
        assert "trait_{t-1}" not in dag

    def test_time_invariant_edges_to_both_timesteps(self):
        """Time-invariant cause should connect to both timesteps of effect."""
        lm = {
            "constructs": [
                {"name": "X"},
                {"name": "trait", "temporal_status": "time_invariant"},
            ],
            "edges": [{"cause": "trait", "effect": "X"}],
        }
        observed = {"X", "trait"}
        dag = unroll_temporal_dag(lm, observed)
        assert dag.has_edge("trait", "X_t")
        assert dag.has_edge("trait", "X_{t-1}")

    def test_is_dag(self):
        """Unrolled graph should be a DAG."""
        lm = _simple_latent()
        observed = {"X", "Y"}
        dag = unroll_temporal_dag(lm, observed)
        assert nx.is_directed_acyclic_graph(dag)


# =============================================================================
# find_blocking_confounders
# =============================================================================


class TestFindBlockingConfounders:
    def test_confounded_by_u(self):
        """U -> X, U -> Y with U unobserved should block."""
        lm = _simple_latent()
        observed = {"X", "Y"}
        blockers = find_blocking_confounders(lm, observed, "X", "Y")
        assert "U" in blockers

    def test_no_confounders_when_all_observed(self):
        """All observed = no blocking confounders."""
        lm = _simple_latent()
        observed = {"X", "Y", "U"}
        blockers = find_blocking_confounders(lm, observed, "X", "Y")
        assert len(blockers) == 0

    def test_no_confounding_on_independent_path(self):
        """U -> X only (no path to Y) should not be a blocker."""
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
        blockers = find_blocking_confounders(lm, observed, "X", "Y")
        assert "U" not in blockers

    def test_chain_confounding(self):
        """U -> X and U -> M -> Y: U is a confounder via backdoor through M."""
        lm = {
            "constructs": [
                {"name": "X"},
                {"name": "Y", "is_outcome": True},
                {"name": "M"},
                {"name": "U"},
            ],
            "edges": [
                {"cause": "X", "effect": "Y"},
                {"cause": "U", "effect": "X"},
                {"cause": "U", "effect": "M"},
                {"cause": "M", "effect": "Y"},
            ],
        }
        observed = {"X", "Y", "M"}
        blockers = find_blocking_confounders(lm, observed, "X", "Y")
        assert "U" in blockers


# =============================================================================
# find_instruments
# =============================================================================


class TestFindInstruments:
    def test_valid_instrument(self):
        """Z -> X -> Y with confounding: Z is a valid instrument."""
        lm = {
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
        observed = {"X", "Y", "Z"}
        instruments = find_instruments(lm, observed, "X", "Y")
        assert "Z" in instruments

    def test_no_instrument_unobserved(self):
        """Unobserved Z should not be returned as instrument."""
        lm = {
            "constructs": [
                {"name": "X"},
                {"name": "Y", "is_outcome": True},
                {"name": "Z"},
            ],
            "edges": [
                {"cause": "Z", "effect": "X"},
                {"cause": "X", "effect": "Y"},
            ],
        }
        observed = {"X", "Y"}  # Z is unobserved
        instruments = find_instruments(lm, observed, "X", "Y")
        assert "Z" not in instruments

    def test_invalid_instrument_direct_to_outcome(self):
        """Z -> X and Z -> Y: Z violates exclusion restriction."""
        lm = {
            "constructs": [
                {"name": "X"},
                {"name": "Y", "is_outcome": True},
                {"name": "Z"},
            ],
            "edges": [
                {"cause": "Z", "effect": "X"},
                {"cause": "Z", "effect": "Y"},
                {"cause": "X", "effect": "Y"},
            ],
        }
        observed = {"X", "Y", "Z"}
        instruments = find_instruments(lm, observed, "X", "Y")
        assert "Z" not in instruments

    def test_missing_treatment_returns_empty(self):
        """Missing treatment node should return empty list."""
        lm = {
            "constructs": [{"name": "Y"}],
            "edges": [],
        }
        instruments = find_instruments(lm, {"Y"}, "X", "Y")
        assert instruments == []

    def test_no_parents_no_instruments(self):
        """If treatment has no parents, no instruments possible."""
        lm = {
            "constructs": [
                {"name": "X"},
                {"name": "Y", "is_outcome": True},
            ],
            "edges": [
                {"cause": "X", "effect": "Y"},
            ],
        }
        observed = {"X", "Y"}
        instruments = find_instruments(lm, observed, "X", "Y")
        assert instruments == []
