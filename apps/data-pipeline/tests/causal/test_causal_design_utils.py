"""Tests for ``utils.causal_design`` helpers.

Trivial accessors are exercised through higher-level tests. This file covers
the helpers with real transformation or graph logic:
- ``make_measurement_extraction_context``
- ``build_digraph``
- ``get_outcome_name``
- ``get_all_treatments``
- StructuralPlan state and marginalized-scale accessors
"""

from typing import Any

import pytest

from nof1_causal_lab.artifacts import StructuralPlan
from nof1_causal_lab.utils.causal_design import (
    build_digraph,
    get_all_treatments,
    get_outcome_name,
    make_measurement_extraction_context,
)
from nof1_causal_lab.utils.structural_plan import (
    get_marginalized_scales,
    get_state_names,
)
from tests.helpers import make_structural_plan


def _full_spec():
    """Minimal valid CausalDesign dict."""
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
                    "construct_polarity": "positive",
                    "measurement_dtype": "continuous",
                    "how_to_measure": "Extract PSS score",
                    "aggregation": "mean",
                },
                {
                    "name": "mood_rating",
                    "construct_name": "mood",
                    "construct_polarity": "positive",
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
# make_measurement_extraction_context
# =============================================================================


class TestMakeMeasurementExtractionContext:
    def test_strips_to_worker_fields(self):
        spec = _full_spec()
        # Add extra fields that workers don't need
        spec["measurement"]["indicators"][0]["aggregation"] = "mean"
        spec["measurement"]["indicators"][0]["construct_name"] = "stress"
        spec["measurement"]["indicators"][0]["source_columns"] = ["pss_col"]
        ctx = make_measurement_extraction_context(spec["measurement"])
        ind = ctx["indicators"][0]
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

    def test_source_columns_included_when_present(self):
        spec = _full_spec()
        spec["measurement"]["indicators"][0]["source_columns"] = ["col_a", "col_b"]
        ctx = make_measurement_extraction_context(spec["measurement"])
        assert ctx["indicators"][0]["source_columns"] == ["col_a", "col_b"]

    def test_ordinal_levels_included_for_worker_codebook(self):
        spec = _full_spec()
        ctx = make_measurement_extraction_context(spec["measurement"])
        assert ctx["indicators"][1]["ordinal_levels"] == ["low", "medium", "high"]


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
        assert (
            get_outcome_name(
                {
                    "constructs": [
                        {"name": "X", "is_outcome": False},
                        {"name": "Y", "is_outcome": True},
                    ]
                }
            )
            == "Y"
        )

    def test_no_outcome(self):
        assert (
            get_outcome_name(
                {
                    "constructs": [
                        {"name": "X", "is_outcome": False},
                        {"name": "Z"},
                    ]
                }
            )
            is None
        )

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
        assert (
            get_all_treatments(
                {
                    "constructs": [{"name": "A"}, {"name": "B"}],
                    "edges": [{"cause": "A", "effect": "B"}],
                }
            )
            == []
        )

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
        assert (
            get_all_treatments(
                {
                    "constructs": [{"name": "Y", "is_outcome": True}],
                    "edges": [],
                }
            )
            == []
        )


class TestStructuralPlanAccessors:
    def test_get_state_names_preserves_compiled_order(self):
        plan = make_structural_plan(["stress", "mood"], [("stress", "mood")])
        assert get_state_names(StructuralPlan.model_validate(plan)) == ["stress", "mood"]


class TestGetMarginalizedScales:
    @staticmethod
    def _spec(induced_dependencies: list[dict[str, Any]]) -> StructuralPlan:
        state_names = sorted(
            {str(state) for dependency in induced_dependencies for state in dependency["between"]}
        )
        source_names = sorted(
            {
                str(source)
                for dependency in induced_dependencies
                for source in dependency["source_confounders"]
            }
        )
        plan = make_structural_plan(state_names, [])
        construct_id_by_name = {
            construct["name"]: source_id
            for source_id, construct in plan["semantics"]["constructs"].items()
        }
        for source_name in source_names:
            source_id = f"construct:{len(construct_id_by_name):04d}"
            construct_id_by_name[source_name] = source_id
            plan["semantics"]["constructs"][source_id] = {
                "name": source_name,
                "description": source_name,
                "role": "exogenous",
                "is_outcome": False,
                "temporal_status": "time_invariant",
            }
            plan["dispositions"].append(
                {
                    "source_id": source_id,
                    "source_kind": "construct",
                    "disposition": "marginalized",
                    "reason": "test confounder",
                }
            )
        plan["induced_dependencies"] = [
            {
                "source_id": f"dependency:{index:04d}",
                "between": [construct_id_by_name[name] for name in dependency["between"]],
                "kind": dependency["kind"],
                "source_confounder_ids": [
                    construct_id_by_name[name] for name in dependency["source_confounders"]
                ],
            }
            for index, dependency in enumerate(induced_dependencies)
        ]
        return StructuralPlan.model_validate(plan)

    def test_golden_like_three_plus_one_confounders_yield_two_scales(self):
        spec = self._spec(
            [
                {
                    "between": ["screen_time", "sleep_quality"],
                    "kind": "initial_state_correlation",
                    "source_confounders": ["age", "living_situation", "personality_traits"],
                },
                {
                    "between": ["screen_time", "stress"],
                    "kind": "initial_state_correlation",
                    "source_confounders": ["occupation_demands"],
                },
            ]
        )
        scales = get_marginalized_scales(spec)

        assert [scale["parameter"] for scale in scales] == [
            "tau_age__living_situation__personality_traits",
            "tau_occupation_demands",
        ]
        merged, solo = scales
        assert merged["sources"] == ["age", "living_situation", "personality_traits"]
        assert merged["affected_states"] == ["screen_time", "sleep_quality"]
        assert merged["directions"] == [("screen_time", "sleep_quality")]
        assert merged["kind"] == "initial_state_correlation"
        assert solo["sources"] == ["occupation_demands"]
        assert solo["affected_states"] == ["screen_time", "stress"]

    def test_multi_scale_per_dep_when_footprints_differ(self):
        spec = self._spec(
            [
                {
                    "between": ["x", "y"],
                    "kind": "initial_state_correlation",
                    "source_confounders": ["c1", "c2"],
                },
                {
                    "between": ["x", "z"],
                    "kind": "initial_state_correlation",
                    "source_confounders": ["c2"],
                },
                {
                    "between": ["y", "z"],
                    "kind": "initial_state_correlation",
                    "source_confounders": ["c2"],
                },
            ]
        )
        scales = get_marginalized_scales(spec)

        assert len(scales) == 2
        by_name = {scale["parameter"]: scale for scale in scales}
        assert by_name["tau_c1"]["affected_states"] == ["x", "y"]
        assert by_name["tau_c1"]["directions"] == [("x", "y")]
        assert by_name["tau_c2"]["affected_states"] == ["x", "y", "z"]
        assert by_name["tau_c2"]["directions"] == [
            ("x", "y"),
            ("x", "z"),
            ("y", "z"),
        ]

    def test_confounder_with_inconsistent_kind_raises(self):
        spec = self._spec(
            [
                {
                    "between": ["x", "y"],
                    "kind": "initial_state_correlation",
                    "source_confounders": ["c"],
                },
                {
                    "between": ["y", "z"],
                    "kind": "innovation_correlation",
                    "source_confounders": ["c"],
                },
            ]
        )
        with pytest.raises(ValueError, match="inconsistent dependency kinds"):
            get_marginalized_scales(spec)

    def test_empty_dependencies_yield_empty_scales(self):
        assert get_marginalized_scales(self._spec([])) == []

    def test_canonical_name_is_sorted(self):
        spec = self._spec(
            [
                {
                    "between": ["x", "y"],
                    "kind": "initial_state_correlation",
                    "source_confounders": ["zebra", "apple", "mango"],
                }
            ]
        )
        (scale,) = get_marginalized_scales(spec)
        assert scale["parameter"] == "tau_apple__mango__zebra"
