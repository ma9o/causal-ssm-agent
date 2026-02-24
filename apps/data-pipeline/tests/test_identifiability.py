"""Test identifiability checking using y0."""

import pytest

from causal_ssm_agent.utils.identifiability import (
    analyze_unobserved_constructs,
    check_identifiability,
    dag_to_admg,
    format_identifiability_report,
    format_marginalization_report,
    get_correlation_pairs_from_marginalization,
    get_observed_constructs,
)


def _blockers(result: dict, treatment: str) -> list[str]:
    """Helper to extract blocking confounders for a treatment."""
    details = result.get("non_identifiable_treatments", {}).get(treatment, {})
    if isinstance(details, dict):
        return details.get("confounders", [])
    return []


def test_get_observed_constructs():
    """Test extraction of observed constructs from measurement model."""
    measurement_model = {
        "indicators": [
            {"name": "ind1", "construct_name": "A", "how_to_measure": "test"},
            {"name": "ind2", "construct_name": "B", "how_to_measure": "test"},
            {"name": "ind3", "construct_name": "A", "how_to_measure": "test"},  # Duplicate
            {"name": "ind4", "construct_name": "C", "how_to_measure": "test"},
        ]
    }

    observed = get_observed_constructs(measurement_model)
    assert observed == {"A", "B", "C"}


def test_identifiability_simple_chain():
    """Test identifiability on simple chain: A -> B -> C (all observed)."""
    latent_model = {
        "constructs": [
            {"name": "A", "role": "exogenous"},
            {"name": "B", "role": "endogenous"},
            {"name": "C", "role": "endogenous", "is_outcome": True},
        ],
        "edges": [
            {"cause": "A", "effect": "B", "description": "A causes B"},
            {"cause": "B", "effect": "C", "description": "B causes C"},
        ],
    }

    measurement_model = {
        "indicators": [
            {"name": "a_ind", "construct_name": "A", "how_to_measure": "test"},
            {"name": "b_ind", "construct_name": "B", "how_to_measure": "test"},
            {"name": "c_ind", "construct_name": "C", "how_to_measure": "test"},
        ]
    }

    result = check_identifiability(latent_model, measurement_model)

    # All effects should be identifiable (no unobserved confounders)
    assert len(result["non_identifiable_treatments"]) == 0
    assert "A" in result["identifiable_treatments"]
    assert "B" in result["identifiable_treatments"]


def test_identifiability_unobserved_treatment():
    """Test when the treatment itself is unobserved.

    Unobserved constructs are NOT considered as treatments because you can't
    do(X) on something you don't observe. They should not appear in
    non_identifiable_treatments - they simply aren't treatments at all.
    """
    latent_model = {
        "constructs": [
            {"name": "X", "role": "exogenous"},
            {"name": "Y", "role": "endogenous", "is_outcome": True},
        ],
        "edges": [
            {"cause": "X", "effect": "Y", "description": "X causes Y"},
        ],
    }

    # Only Y is observed
    measurement_model = {
        "indicators": [
            {"name": "y_ind", "construct_name": "Y", "how_to_measure": "test"},
        ]
    }

    result = check_identifiability(latent_model, measurement_model)

    # X is not listed as a treatment at all - unobserved constructs aren't treatments
    assert "X" not in result["non_identifiable_treatments"]
    assert "X" not in result["identifiable_treatments"]
    # No observed treatments with paths to outcome, so both should be empty
    assert len(result["non_identifiable_treatments"]) == 0
    assert len(result["identifiable_treatments"]) == 0


def test_lagged_confounding_blocks_identification():
    """Test that lagged confounding blocks identification.

    U_{t-1} -> X_t, U_{t-1} -> Y_t creates confounding that blocks X_t -> Y_t
    even though U is only connected via lagged edges.
    """
    latent_model = {
        "constructs": [
            {"name": "X", "role": "endogenous", "temporal_status": "time_varying"},
            {
                "name": "Y",
                "role": "endogenous",
                "temporal_status": "time_varying",
                "is_outcome": True,
            },
            {"name": "U", "role": "exogenous", "temporal_status": "time_varying"},
        ],
        "edges": [
            {"cause": "X", "effect": "Y", "lagged": False},  # X_t -> Y_t
            {"cause": "U", "effect": "X", "lagged": True},  # U_{t-1} -> X_t
            {"cause": "U", "effect": "Y", "lagged": True},  # U_{t-1} -> Y_t
        ],
    }

    # Only X and Y observed, U is unobserved
    measurement_model = {
        "indicators": [
            {"name": "x_ind", "construct_name": "X"},
            {"name": "y_ind", "construct_name": "Y"},
        ]
    }

    result = check_identifiability(latent_model, measurement_model)

    # X -> Y should NOT be identifiable due to lagged confounding from U
    assert "X" in result["non_identifiable_treatments"]
    assert "U" in _blockers(result, "X")


def test_dag_to_admg_unrolls_to_two_timesteps():
    """Test that dag_to_admg creates a 2-timestep unrolled graph."""
    latent_model = {
        "constructs": [
            {"name": "A", "role": "endogenous", "temporal_status": "time_varying"},
            {"name": "B", "role": "endogenous", "temporal_status": "time_varying"},
        ],
        "edges": [
            # Contemporaneous: A_t -> B_t
            {"cause": "A", "effect": "B", "lagged": False},
            # Lagged feedback: B_{t-1} -> A_t
            {"cause": "B", "effect": "A", "lagged": True},
        ],
    }

    observed = {"A", "B"}
    admg, _confounders = dag_to_admg(latent_model, observed)

    directed_edges = list(admg.directed.edges())
    edge_names = [(str(e[0]), str(e[1])) for e in directed_edges]

    # Should have:
    # - A_t -> B_t (contemporaneous)
    # - A_{t-1} -> B_{t-1} (mirrored previous timestep)
    # - B_{t-1} -> A_t (lagged)
    # - A_{t-1} -> A_t (AR(1) for endogenous)
    # - B_{t-1} -> B_t (AR(1) for endogenous)
    assert ("A_t", "B_t") in edge_names
    assert ("A_{t-1}", "B_{t-1}") in edge_names
    assert ("B_{t-1}", "A_t") in edge_names
    assert ("A_{t-1}", "A_t") in edge_names
    assert ("B_{t-1}", "B_t") in edge_names


def test_dag_to_admg_validates_max_lag_one():
    """Test that dag_to_admg asserts if lag > 1 (violates A3a)."""
    latent_model = {
        "constructs": [
            {"name": "A"},
            {"name": "B"},
        ],
        "edges": [
            # Invalid: non-boolean lagged value (simulating lag > 1)
            {"cause": "A", "effect": "B", "lagged": 2},
        ],
    }

    observed = {"A", "B"}

    with pytest.raises(AssertionError) as exc_info:
        dag_to_admg(latent_model, observed)

    assert "A3a" in str(exc_info.value) or "lagged" in str(exc_info.value)


def test_format_identifiability_report():
    """Test formatting of identifiability report."""
    result = {
        "identifiable_treatments": {
            "X": {
                "method": "do_calculus",
                "estimand": "P(Y|X)",
                "marginalized_confounders": [],
            }
        },
        "non_identifiable_treatments": {
            "A": {"confounders": ["U", "V"]},
        },
        "graph_info": {
            "observed_constructs": ["X", "Y", "A"],
            "total_constructs": 5,
            "unobserved_confounders": ["U", "V"],
            "n_directed_edges": 3,
            "n_bidirected_edges": 1,
        },
    }

    report = format_identifiability_report(result, outcome="Y")

    assert "1/2 treatments have non-identifiable effects on Y" in report
    assert "A (blocked by: U, V)" in report
    assert "1 treatments have identifiable effects" in report
    assert "X" in report
    assert "3/5 constructs observed" in report


def test_analyze_unobserved_all_observed():
    """When all constructs are observed, nothing needs marginalization analysis."""
    latent_model = {
        "constructs": [
            {"name": "A", "role": "exogenous"},
            {"name": "B", "role": "endogenous", "is_outcome": True},
        ],
        "edges": [
            {"cause": "A", "effect": "B", "description": "A causes B"},
        ],
    }

    measurement_model = {
        "indicators": [
            {"name": "a_ind", "construct_name": "A", "how_to_measure": "test"},
            {"name": "b_ind", "construct_name": "B", "how_to_measure": "test"},
        ]
    }

    id_result = check_identifiability(latent_model, measurement_model)
    analysis = analyze_unobserved_constructs(latent_model, measurement_model, id_result)

    assert len(analysis["can_marginalize"]) == 0
    assert analysis["blocking_details"] == {}


def test_format_marginalization_report():
    """Test formatting of marginalization report."""
    analysis = {
        "can_marginalize": {"U1", "U2"},
        "blocking_details": {"U3": ["Treatment"]},
        "marginalize_reason": {
            "U1": "does not create confounding (single child or no observed children)",
            "U2": "confounding handled by identification strategy (front-door or similar)",
        },
    }

    report = format_marginalization_report(analysis)

    assert "CAN MARGINALIZE (2 constructs)" in report
    assert "NEEDS MODELING (1 constructs)" in report
    assert "U1" in report
    assert "U2" in report
    assert "U3" in report
    assert "blocks identification" in report


def test_correlation_pairs_from_marginalized_confounder():
    """Front-door: U confounds X->Y but is marginalized. Should produce cor pair."""
    latent_model = {
        "constructs": [
            {"name": "X", "role": "exogenous", "temporal_status": "time_varying"},
            {"name": "M", "role": "endogenous", "temporal_status": "time_varying"},
            {
                "name": "Y",
                "role": "endogenous",
                "temporal_status": "time_varying",
                "is_outcome": True,
            },
            {"name": "U", "role": "exogenous", "temporal_status": "time_varying"},
        ],
        "edges": [
            {"cause": "X", "effect": "M", "lagged": False},
            {"cause": "M", "effect": "Y", "lagged": False},
            {"cause": "U", "effect": "X", "lagged": False},
            {"cause": "U", "effect": "Y", "lagged": False},
        ],
    }
    measurement_model = {
        "indicators": [
            {"name": "x_ind", "construct_name": "X"},
            {"name": "m_ind", "construct_name": "M"},
            {"name": "y_ind", "construct_name": "Y"},
        ]
    }

    id_result = check_identifiability(latent_model, measurement_model)
    pairs = get_correlation_pairs_from_marginalization(latent_model, measurement_model, id_result)

    # U confounds X and Y, both observed → should produce (X, Y, U) pair
    assert len(pairs) == 1
    s1, s2, confounder = pairs[0]
    assert {s1, s2} == {"X", "Y"}
    assert confounder == "U"


def test_correlation_pairs_no_marginalized_confounders():
    """All observed → no correlation pairs needed."""
    latent_model = {
        "constructs": [
            {"name": "A", "role": "exogenous"},
            {"name": "B", "role": "endogenous", "is_outcome": True},
        ],
        "edges": [{"cause": "A", "effect": "B"}],
    }
    measurement_model = {
        "indicators": [
            {"name": "a_ind", "construct_name": "A"},
            {"name": "b_ind", "construct_name": "B"},
        ]
    }

    id_result = check_identifiability(latent_model, measurement_model)
    pairs = get_correlation_pairs_from_marginalization(latent_model, measurement_model, id_result)
    assert pairs == []


def test_correlation_pairs_single_child_not_confounder():
    """Unobserved with single observed child is not a confounder → no pair."""
    latent_model = {
        "constructs": [
            {"name": "X", "role": "exogenous", "temporal_status": "time_varying"},
            {
                "name": "Y",
                "role": "endogenous",
                "temporal_status": "time_varying",
                "is_outcome": True,
            },
            {"name": "U", "role": "exogenous", "temporal_status": "time_varying"},
        ],
        "edges": [
            {"cause": "X", "effect": "Y", "lagged": False},
            {"cause": "U", "effect": "X", "lagged": False},  # U only affects X
        ],
    }
    measurement_model = {
        "indicators": [
            {"name": "x_ind", "construct_name": "X"},
            {"name": "y_ind", "construct_name": "Y"},
        ]
    }

    id_result = check_identifiability(latent_model, measurement_model)
    pairs = get_correlation_pairs_from_marginalization(latent_model, measurement_model, id_result)
    assert pairs == []


def test_correlation_pairs_multiple_confounders():
    """Two marginalized confounders produce independent correlation pairs.

    DAG: U1 -> X, U1 -> Y (front-door via M1)
         U2 -> Y, U2 -> Z (front-door via M2)
         X -> M1 -> Y -> M2 -> Z

    U1 confounds {X, Y} → cor_X_Y
    U2 confounds {Y, Z} → cor_Y_Z
    """
    latent_model = {
        "constructs": [
            {"name": "X", "role": "exogenous", "temporal_status": "time_varying"},
            {"name": "M1", "role": "endogenous", "temporal_status": "time_varying"},
            {
                "name": "Y",
                "role": "endogenous",
                "temporal_status": "time_varying",
            },
            {"name": "M2", "role": "endogenous", "temporal_status": "time_varying"},
            {
                "name": "Z",
                "role": "endogenous",
                "temporal_status": "time_varying",
                "is_outcome": True,
            },
            {"name": "U1", "role": "exogenous", "temporal_status": "time_varying"},
            {"name": "U2", "role": "exogenous", "temporal_status": "time_varying"},
        ],
        "edges": [
            # Causal chain with mediators
            {"cause": "X", "effect": "M1", "lagged": False},
            {"cause": "M1", "effect": "Y", "lagged": False},
            {"cause": "Y", "effect": "M2", "lagged": False},
            {"cause": "M2", "effect": "Z", "lagged": False},
            # Confounders
            {"cause": "U1", "effect": "X", "lagged": False},
            {"cause": "U1", "effect": "Y", "lagged": False},
            {"cause": "U2", "effect": "Y", "lagged": False},
            {"cause": "U2", "effect": "Z", "lagged": False},
        ],
    }
    measurement_model = {
        "indicators": [
            {"name": "x_ind", "construct_name": "X"},
            {"name": "m1_ind", "construct_name": "M1"},
            {"name": "y_ind", "construct_name": "Y"},
            {"name": "m2_ind", "construct_name": "M2"},
            {"name": "z_ind", "construct_name": "Z"},
        ]
    }

    id_result = check_identifiability(latent_model, measurement_model)
    pairs = get_correlation_pairs_from_marginalization(latent_model, measurement_model, id_result)

    # Extract just the state pairs (ignoring confounder name)
    state_pairs = {(s1, s2) for s1, s2, _ in pairs}
    confounders = {confounder for _, _, confounder in pairs}

    # U1 confounds X and Y → (X, Y) pair
    assert ("X", "Y") in state_pairs
    # U2 confounds Y and Z → (Y, Z) pair
    assert ("Y", "Z") in state_pairs
    # Both confounders should be referenced
    assert confounders == {"U1", "U2"}
    # No spurious pairs (X, Z) — X and Z don't share a confounder
    assert ("X", "Z") not in state_pairs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
