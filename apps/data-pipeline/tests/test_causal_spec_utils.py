"""Tests for ``utils.causal_spec`` helpers.

Trivial accessors are exercised through higher-level tests. This file covers
the helpers with real transformation or graph logic:
- ``make_extraction_context``
- ``build_digraph``
- ``get_outcome_name``
- ``get_all_treatments``
- ``get_estimation_constructs``
- ``get_estimable_treatments``
"""

from causal_ssm_agent.utils.causal_spec import (
    build_digraph,
    get_all_treatments,
    get_estimable_treatments,
    get_estimation_constructs,
    get_outcome_name,
    make_extraction_context,
)


def _full_spec():
    """Minimal valid CausalSpec dict."""
    return {
        "latent": {
            "constructs": [
                {"name": "stress", "role": "exogenous"},
                {
                    "name": "mood",
                    "role": "endogenous",
                    "is_outcome": True,
                },
            ],
            "edges": [
                {"cause": "stress", "effect": "mood", "description": "Stress affects mood"},
            ],
        },
        "measurement": {
            "model_clock": "1d",
            "indicators": [
                {
                    "name": "pss_score",
                    "construct_name": "stress",
                    "measurement_dtype": "continuous",
                    "how_to_measure": "Extract PSS score",
                    "aggregation": "mean",
                },
                {
                    "name": "mood_rating",
                    "construct_name": "mood",
                    "measurement_dtype": "ordinal",
                    "how_to_measure": "Rate mood 1-5",
                    "aggregation": "last",
                    "ordinal_levels": ["low", "medium", "high"],
                },
            ],
        },
        "estimation": {
            "state_order": ["stress", "mood"],
            "edges": [{"cause": "stress", "effect": "mood", "description": "Stress affects mood"}],
            "induced_dependencies": [],
        },
    }


# =============================================================================
# make_extraction_context
# =============================================================================


class TestMakeExtractionContext:
    def test_strips_to_worker_fields(self):
        spec = _full_spec()
        # Add extra fields that workers don't need
        spec["measurement"]["indicators"][0]["aggregation"] = "mean"
        spec["measurement"]["indicators"][0]["construct_name"] = "stress"
        spec["measurement"]["indicators"][0]["source_columns"] = ["pss_col"]
        ctx = make_extraction_context(spec)
        ind = ctx["measurement"]["indicators"][0]
        assert set(ind.keys()) == {
            "name",
            "measurement_dtype",
            "how_to_measure",
            "source_columns",
            "aggregation",
            "support_kind",
            "summary_operator",
            "anchor_policy",
            "observation_window",
        }
        assert "construct_name" not in ind
        assert "ordinal_levels" not in ind

    def test_outcome_slimmed_to_name_and_description(self):
        spec = _full_spec()
        ctx = make_extraction_context(spec)
        constructs = ctx["latent"]["constructs"]
        assert len(constructs) == 1
        outcome = constructs[0]
        assert set(outcome.keys()) == {"name", "description"}

    def test_no_outcome_gives_empty_constructs(self):
        spec = {
            "latent": {"constructs": [{"name": "x", "role": "exogenous"}]},
            "measurement": {
                "model_clock": "1d",
                "indicators": [
                    {
                        "name": "ind",
                        "measurement_dtype": "continuous",
                        "aggregation": "mean",
                    }
                ],
            },
        }
        ctx = make_extraction_context(spec)
        assert ctx["latent"]["constructs"] == []

    def test_source_columns_included_when_present(self):
        spec = _full_spec()
        spec["measurement"]["indicators"][0]["source_columns"] = ["col_a", "col_b"]
        ctx = make_extraction_context(spec)
        assert ctx["measurement"]["indicators"][0]["source_columns"] == ["col_a", "col_b"]

    def test_ordinal_levels_included_for_worker_codebook(self):
        spec = _full_spec()
        ctx = make_extraction_context(spec)
        assert ctx["measurement"]["indicators"][1]["ordinal_levels"] == ["low", "medium", "high"]


class TestBuildDigraph:
    def test_simple_chain(self):
        model = {
            "edges": [
                {"cause": "A", "effect": "B"},
                {"cause": "B", "effect": "C"},
            ]
        }
        graph = build_digraph(model)
        assert set(graph.nodes()) == {"A", "B", "C"}
        assert graph.has_edge("A", "B")
        assert graph.has_edge("B", "C")
        assert not graph.has_edge("A", "C")

    def test_empty_edges(self):
        assert len(build_digraph({"edges": []}).nodes()) == 0

    def test_diamond_topology(self):
        graph = build_digraph(
            {
                "edges": [
                    {"cause": "A", "effect": "B"},
                    {"cause": "A", "effect": "C"},
                    {"cause": "B", "effect": "D"},
                    {"cause": "C", "effect": "D"},
                ]
            }
        )
        assert set(graph.nodes()) == {"A", "B", "C", "D"}
        assert len(graph.edges()) == 4

    def test_self_loop(self):
        graph = build_digraph({"edges": [{"cause": "A", "effect": "A"}]})
        assert set(graph.nodes()) == {"A"}
        assert graph.has_edge("A", "A")

    def test_duplicate_edges(self):
        graph = build_digraph(
            {
                "edges": [
                    {"cause": "A", "effect": "B"},
                    {"cause": "A", "effect": "B"},
                ]
            }
        )
        assert len(graph.edges()) == 1


class TestGetOutcomeName:
    def test_finds_outcome(self):
        assert get_outcome_name(
            {
                "constructs": [
                    {"name": "X", "is_outcome": False},
                    {"name": "Y", "is_outcome": True},
                ]
            }
        ) == "Y"

    def test_no_outcome(self):
        assert get_outcome_name(
            {
                "constructs": [
                    {"name": "X", "is_outcome": False},
                    {"name": "Z"},
                ]
            }
        ) is None

    def test_empty_constructs(self):
        assert get_outcome_name({"constructs": []}) is None

    def test_missing_constructs_key(self):
        assert get_outcome_name({}) is None


class TestGetAllTreatments:
    def test_chain_treatments(self):
        treatments = get_all_treatments(
            {
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
        )
        assert treatments == ["A", "B"]

    def test_disconnected_not_treatment(self):
        treatments = get_all_treatments(
            {
                "constructs": [
                    {"name": "X"},
                    {"name": "Y", "is_outcome": True},
                    {"name": "Z"},
                ],
                "edges": [{"cause": "X", "effect": "Y"}],
            }
        )
        assert treatments == ["X"]

    def test_no_outcome_returns_empty(self):
        assert get_all_treatments(
            {
                "constructs": [{"name": "A"}, {"name": "B"}],
                "edges": [{"cause": "A", "effect": "B"}],
            }
        ) == []

    def test_sorted_output(self):
        treatments = get_all_treatments(
            {
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
        )
        assert treatments == ["Apple", "Zebra"]

    def test_fork_topology(self):
        treatments = get_all_treatments(
            {
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
        )
        assert treatments == ["X"]

    def test_diamond_all_treatments(self):
        treatments = get_all_treatments(
            {
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
        )
        assert treatments == ["A", "B", "C"]

    def test_empty_model(self):
        assert get_all_treatments({"constructs": [], "edges": []}) == []

    def test_outcome_only(self):
        assert get_all_treatments(
            {
                "constructs": [{"name": "Y", "is_outcome": True}],
                "edges": [],
            }
        ) == []


class TestEstimationAccessors:
    def test_get_estimation_constructs_preserves_retained_state_order(self):
        constructs = get_estimation_constructs(_full_spec())
        assert [construct["name"] for construct in constructs] == ["stress", "mood"]

    def test_get_estimable_treatments_ignores_theoretical_only_nodes(self):
        spec = _full_spec()
        spec["latent"]["constructs"].append(
            {
                "name": "baseline_trait",
                "role": "exogenous",
                "description": "Theoretical but marginalized cause",
            }
        )
        spec["latent"]["edges"].append(
            {
                "cause": "baseline_trait",
                "effect": "mood",
                "description": "Theoretical-only path",
            }
        )

        assert get_estimable_treatments(spec) == ["stress"]
