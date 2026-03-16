"""Tests for utils/effects.py graph utilities.

Covers: build_digraph, get_outcome_name, get_all_treatments.
"""

from causal_ssm_agent.utils.causal_spec import (
    build_digraph,
    get_all_treatments,
    get_outcome_name,
)

# =============================================================================
# build_digraph
# =============================================================================


class TestBuildDigraph:
    def test_simple_chain(self):
        """A → B → C chain creates correct graph."""
        model = {
            "edges": [
                {"cause": "A", "effect": "B"},
                {"cause": "B", "effect": "C"},
            ]
        }
        G = build_digraph(model)
        assert set(G.nodes()) == {"A", "B", "C"}
        assert G.has_edge("A", "B")
        assert G.has_edge("B", "C")
        assert not G.has_edge("A", "C")

    def test_empty_edges(self):
        """Empty edges list creates empty graph."""
        G = build_digraph({"edges": []})
        assert len(G.nodes()) == 0

    def test_diamond_topology(self):
        """Diamond: A → B, A → C, B → D, C → D."""
        model = {
            "edges": [
                {"cause": "A", "effect": "B"},
                {"cause": "A", "effect": "C"},
                {"cause": "B", "effect": "D"},
                {"cause": "C", "effect": "D"},
            ]
        }
        G = build_digraph(model)
        assert set(G.nodes()) == {"A", "B", "C", "D"}
        assert len(G.edges()) == 4

    def test_self_loop(self):
        """Self-loop A → A is preserved in graph."""
        model = {"edges": [{"cause": "A", "effect": "A"}]}
        G = build_digraph(model)
        assert set(G.nodes()) == {"A"}
        assert G.has_edge("A", "A")

    def test_duplicate_edges(self):
        """Duplicate edges don't create multiple edges in digraph."""
        model = {
            "edges": [
                {"cause": "A", "effect": "B"},
                {"cause": "A", "effect": "B"},
            ]
        }
        G = build_digraph(model)
        assert len(G.edges()) == 1


# =============================================================================
# get_outcome_name
# =============================================================================


class TestGetOutcome:
    def test_finds_outcome(self):
        model = {
            "constructs": [
                {"name": "X", "is_outcome": False},
                {"name": "Y", "is_outcome": True},
            ]
        }
        assert get_outcome_name(model) == "Y"

    def test_no_outcome(self):
        model = {
            "constructs": [
                {"name": "X", "is_outcome": False},
                {"name": "Z"},
            ]
        }
        assert get_outcome_name(model) is None

    def test_empty_constructs(self):
        assert get_outcome_name({"constructs": []}) is None

    def test_missing_constructs_key(self):
        assert get_outcome_name({}) is None


# =============================================================================
# get_all_treatments
# =============================================================================


class TestGetAllTreatments:
    def test_chain_treatments(self):
        """In A → B → Y, both A and B are treatments for Y."""
        model = {
            "constructs": [
                {"name": "A"},
                {"name": "B"},
                {"name": "Y", "is_outcome": True},
            ],
            "edges": [
                {"cause": "A", "effect": "B"},
                {"cause": "B", "effect": "Y"},
            ],
        }
        treatments = get_all_treatments(model)
        assert treatments == ["A", "B"]

    def test_disconnected_not_treatment(self):
        """Disconnected node Z has no path to Y → not a treatment."""
        model = {
            "constructs": [
                {"name": "X"},
                {"name": "Y", "is_outcome": True},
                {"name": "Z"},
            ],
            "edges": [
                {"cause": "X", "effect": "Y"},
            ],
        }
        treatments = get_all_treatments(model)
        assert treatments == ["X"]

    def test_no_outcome_returns_empty(self):
        """If no outcome, return empty list."""
        model = {
            "constructs": [{"name": "A"}, {"name": "B"}],
            "edges": [{"cause": "A", "effect": "B"}],
        }
        assert get_all_treatments(model) == []

    def test_sorted_output(self):
        """Treatments are returned in sorted order."""
        model = {
            "constructs": [
                {"name": "Zebra"},
                {"name": "Apple"},
                {"name": "Outcome", "is_outcome": True},
            ],
            "edges": [
                {"cause": "Zebra", "effect": "Outcome"},
                {"cause": "Apple", "effect": "Outcome"},
            ],
        }
        treatments = get_all_treatments(model)
        assert treatments == ["Apple", "Zebra"]

    def test_fork_topology(self):
        """Fork: X → Y, X → Z. With Y as outcome, only X is treatment."""
        model = {
            "constructs": [
                {"name": "X"},
                {"name": "Y", "is_outcome": True},
                {"name": "Z"},
            ],
            "edges": [
                {"cause": "X", "effect": "Y"},
                {"cause": "X", "effect": "Z"},
            ],
        }
        treatments = get_all_treatments(model)
        assert treatments == ["X"]

    def test_diamond_all_treatments(self):
        """Diamond: A → B → D, A → C → D. All of A, B, C are treatments for D."""
        model = {
            "constructs": [
                {"name": "A"},
                {"name": "B"},
                {"name": "C"},
                {"name": "D", "is_outcome": True},
            ],
            "edges": [
                {"cause": "A", "effect": "B"},
                {"cause": "A", "effect": "C"},
                {"cause": "B", "effect": "D"},
                {"cause": "C", "effect": "D"},
            ],
        }
        treatments = get_all_treatments(model)
        assert treatments == ["A", "B", "C"]

    def test_empty_model(self):
        """Empty model returns empty treatments list."""
        model = {"constructs": [], "edges": []}
        assert get_all_treatments(model) == []

    def test_outcome_only(self):
        """Single outcome node with no edges has no treatments."""
        model = {
            "constructs": [{"name": "Y", "is_outcome": True}],
            "edges": [],
        }
        assert get_all_treatments(model) == []
