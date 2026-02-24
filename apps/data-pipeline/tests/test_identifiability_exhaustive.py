"""Exhaustive theoretical tests for identifiability logic.

This test suite verifies the theoretical soundness of our identification logic
by testing against classic causal inference scenarios, complex graph topologies,
and edge cases. All tests are derived from Pearl's do-calculus literature and
the y0 package's identification algorithms.

Organizational Structure:
1. Classic Pearl Graphs - Well-known identifiable/non-identifiable structures
2. Backdoor Criterion - Testing adjustment-based identification
3. Front-Door Criterion - Testing mediation-based identification
4. Instrumental Variables - Testing IV-based identification
5. Temporal Dynamics - Testing 2-timestep unrolling under AR(1)/A3a
6. Time-Invariant Constructs - Testing mixed temporal status
7. Complex Confounding Patterns - Multiple confounders, chains, M-bias
8. Collider Structures - Verifying colliders don't create confounding
9. Multiple Treatments - Testing simultaneous treatment identifiability
10. Edge Cases - Boundary conditions and degenerate graphs
11. Marginalization Analysis - Testing unobserved construct classification

References:
- Pearl, J. (2009). Causality: Models, Reasoning, and Inference
- Shpitser & Pearl (2006). Identification of Joint Interventional Distributions
- arXiv:2504.20172 - Jahn, Karnik & Schulman on bounded latent reach
"""

import pytest

from causal_ssm_agent.utils.identifiability import (
    analyze_unobserved_constructs,
    check_identifiability,
    dag_to_admg,
    unroll_temporal_dag,
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def make_latent_model(constructs: list[dict], edges: list[dict]) -> dict:
    """Create a latent model dict with sensible defaults."""
    processed_constructs = []
    for c in constructs:
        construct = {"name": c["name"]}
        construct["role"] = c.get("role", "endogenous")
        if "is_outcome" in c:
            construct["is_outcome"] = c["is_outcome"]
        if "temporal_status" in c:
            construct["temporal_status"] = c["temporal_status"]
        processed_constructs.append(construct)

    processed_edges = []
    for e in edges:
        edge = {
            "cause": e["cause"],
            "effect": e["effect"],
        }
        if "lagged" in e:
            edge["lagged"] = e["lagged"]
        processed_edges.append(edge)

    return {"constructs": processed_constructs, "edges": processed_edges}


def make_measurement_model(observed_constructs: list[str]) -> dict:
    """Create a measurement model with one indicator per observed construct."""
    return {
        "indicators": [
            {"name": f"{c.lower()}_ind", "construct_name": c, "how_to_measure": "test"}
            for c in observed_constructs
        ]
    }


def assert_identifiable(result: dict, treatment: str, msg: str = ""):
    """Assert that a treatment is identifiable."""
    assert treatment in result["identifiable_treatments"], (
        f"{treatment} should be identifiable. {msg}\nResult: {result}"
    )


def assert_not_identifiable(result: dict, treatment: str, msg: str = ""):
    """Assert that a treatment is NOT identifiable."""
    assert treatment in result["non_identifiable_treatments"], (
        f"{treatment} should NOT be identifiable. {msg}\nResult: {result}"
    )


def get_blockers(result: dict, treatment: str) -> list[str]:
    """Get confounders blocking a treatment."""
    details = result.get("non_identifiable_treatments", {}).get(treatment, {})
    if isinstance(details, dict):
        return details.get("confounders", [])
    return []


def get_estimand(result: dict, treatment: str) -> str:
    """Get estimand string for an identifiable treatment."""
    details = result.get("identifiable_treatments", {}).get(treatment, {})
    if isinstance(details, dict):
        return details.get("estimand", "")
    return ""


def assert_blocked_by(result: dict, treatment: str, blocker: str, msg: str = ""):
    """Assert that a treatment is blocked by a specific confounder."""
    details = result["non_identifiable_treatments"].get(treatment)
    assert details, f"{treatment} should have blocking confounders. {msg}"
    blockers = details.get("confounders", []) if isinstance(details, dict) else []
    assert blocker in blockers, (
        f"{treatment} should be blocked by {blocker}. {msg}\nBlockers: {blockers}"
    )


# =============================================================================
# 1. CLASSIC PEARL GRAPHS
# =============================================================================


class TestClassicPearlGraphs:
    """Test classic graphs from Pearl's causality literature."""

    def test_bow_graph_non_identifiable(self):
        """Bow graph: X -> Y with U -> X, U -> Y.

        This is the simplest non-identifiable structure.
        No adjustment, front-door, or IV applies.

             U
            / \\
           v   v
           X -> Y
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "X"},
                {"name": "Y", "is_outcome": True},
                {"name": "U"},
            ],
            edges=[
                {"cause": "X", "effect": "Y"},
                {"cause": "U", "effect": "X"},
                {"cause": "U", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["X", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        assert_not_identifiable(result, "X", "Bow graph is non-identifiable")
        assert_blocked_by(result, "X", "U")

    def test_confounded_chain_non_identifiable(self):
        """Chain with confounding at every step.

        X -> M -> Y with U1 -> X, U1 -> M and U2 -> M, U2 -> Y

        Neither front-door nor adjustment works.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "X"},
                {"name": "M"},
                {"name": "Y", "is_outcome": True},
                {"name": "U1"},
                {"name": "U2"},
            ],
            edges=[
                {"cause": "X", "effect": "M"},
                {"cause": "M", "effect": "Y"},
                {"cause": "U1", "effect": "X"},
                {"cause": "U1", "effect": "M"},
                {"cause": "U2", "effect": "M"},
                {"cause": "U2", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["X", "M", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        # X -> Y not identifiable: can't use M as mediator (U2 confounds M-Y)
        assert_not_identifiable(result, "X", "Chain with double confounding")

    def test_verma_constraint_graph(self):
        """Test a Verma-constraint scenario.

        Some graphs satisfy Verma constraints but aren't ID-identifiable.
        This tests y0's handling of such cases.

        W -> X -> Y -> Z
             ^        ^
             U1 ------+

        U1 confounds X and Z (but not Y directly).
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "W", "role": "exogenous"},
                {"name": "X"},
                {"name": "Y", "is_outcome": True},
                {"name": "Z"},
                {"name": "U1"},
            ],
            edges=[
                {"cause": "W", "effect": "X"},
                {"cause": "X", "effect": "Y"},
                {"cause": "Y", "effect": "Z"},
                {"cause": "U1", "effect": "X"},
                {"cause": "U1", "effect": "Z"},
            ],
        )
        measurement_model = make_measurement_model(["W", "X", "Y", "Z"])

        result = check_identifiability(latent_model, measurement_model)

        # X -> Y should be identifiable (W as instrument, or backdoor through W)
        assert_identifiable(result, "X", "W can serve as instrument/adjustment")
        assert_identifiable(result, "W", "W -> Y through X is identifiable")


# =============================================================================
# 2. BACKDOOR CRITERION
# =============================================================================


class TestBackdoorCriterion:
    """Test identification via backdoor adjustment."""

    def test_backdoor_with_observed_confounder(self):
        """Classic backdoor: adjust for observed Z.

            Z
           / \\
          v   v
          X -> Y

        If Z is observed, P(Y|do(X)) = sum_z P(Y|X,Z)P(Z).
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "Z", "role": "exogenous"},
                {"name": "X"},
                {"name": "Y", "is_outcome": True},
            ],
            edges=[
                {"cause": "Z", "effect": "X"},
                {"cause": "Z", "effect": "Y"},
                {"cause": "X", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["Z", "X", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        assert_identifiable(result, "X", "Backdoor through observed Z")
        assert_identifiable(result, "Z", "Direct cause Z is identifiable")

    def test_backdoor_blocked_by_unobserved(self):
        """Backdoor path exists but confounder is unobserved.

            U (unobserved)
           / \\
          v   v
          X -> Y
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "U", "role": "exogenous"},
                {"name": "X"},
                {"name": "Y", "is_outcome": True},
            ],
            edges=[
                {"cause": "U", "effect": "X"},
                {"cause": "U", "effect": "Y"},
                {"cause": "X", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["X", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        assert_not_identifiable(result, "X", "Can't adjust for unobserved U")
        assert_blocked_by(result, "X", "U")

    def test_backdoor_multiple_confounders_all_observed(self):
        """Multiple confounders, all observed - should be identifiable.

        Z1   Z2
         \\  /|
          v v v
          X -> Y
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "Z1", "role": "exogenous"},
                {"name": "Z2", "role": "exogenous"},
                {"name": "X"},
                {"name": "Y", "is_outcome": True},
            ],
            edges=[
                {"cause": "Z1", "effect": "X"},
                {"cause": "Z2", "effect": "X"},
                {"cause": "Z2", "effect": "Y"},
                {"cause": "X", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["Z1", "Z2", "X", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        assert_identifiable(result, "X", "Adjust for Z2 (Z1 not on backdoor)")

    def test_backdoor_one_confounder_unobserved_but_iv_available(self):
        """Unobserved confounder, but Z1 serves as instrument.

            Z1   U (unobserved)
             \\  /|
              v v v
              X -> Y

        Z1 is a valid instrument: Z1 -> X, Z1 has no path to Y except through X,
        and Z1 is not affected by U (U doesn't cause Z1).
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "Z1", "role": "exogenous"},
                {"name": "U", "role": "exogenous"},
                {"name": "X"},
                {"name": "Y", "is_outcome": True},
            ],
            edges=[
                {"cause": "Z1", "effect": "X"},
                {"cause": "U", "effect": "X"},
                {"cause": "U", "effect": "Y"},
                {"cause": "X", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["Z1", "X", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        # X is identifiable via IV using Z1
        assert_identifiable(result, "X", "Z1 is valid instrument")
        assert "IV(Z1)" in get_estimand(result, "X")

    def test_no_identification_strategy_available(self):
        """No identification strategy works: no backdoor, no front-door, no IV.

            U (unobserved)
           / \\
          v   v
          X -> Y

        - No backdoor: U is unobserved
        - No front-door: No mediator
        - No IV: No parent of X other than U
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "U", "role": "exogenous"},
                {"name": "X"},
                {"name": "Y", "is_outcome": True},
            ],
            edges=[
                {"cause": "U", "effect": "X"},
                {"cause": "U", "effect": "Y"},
                {"cause": "X", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["X", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        assert_not_identifiable(result, "X", "No identification strategy available")
        assert_blocked_by(result, "X", "U")

    def test_backdoor_chain_of_confounders(self):
        """Confounder chain: Z -> W, W affects both X and Y.

        Z -> W -> X
             \\
              v
          X -> Y

        Adjusting for W suffices (Z is not on backdoor after conditioning on W).
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "Z", "role": "exogenous"},
                {"name": "W"},
                {"name": "X"},
                {"name": "Y", "is_outcome": True},
            ],
            edges=[
                {"cause": "Z", "effect": "W"},
                {"cause": "W", "effect": "X"},
                {"cause": "W", "effect": "Y"},
                {"cause": "X", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["Z", "W", "X", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        assert_identifiable(result, "X", "Adjust for W blocks backdoor")


# =============================================================================
# 3. FRONT-DOOR CRITERION
# =============================================================================


class TestFrontDoorCriterion:
    """Test identification via front-door criterion."""

    def test_classic_front_door(self):
        """Classic front-door: X -> M -> Y with U -> X, U -> Y.

             U
            / \\
           v   v
           X -> M -> Y

        P(Y|do(X)) = sum_m P(M=m|X) sum_x' P(Y|M=m,X=x')P(X=x')
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "X"},
                {"name": "M"},
                {"name": "Y", "is_outcome": True},
                {"name": "U"},
            ],
            edges=[
                {"cause": "X", "effect": "M"},
                {"cause": "M", "effect": "Y"},
                {"cause": "U", "effect": "X"},
                {"cause": "U", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["X", "M", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        assert_identifiable(result, "X", "Front-door through M")
        # The estimand should mention M
        assert "M" in get_estimand(result, "X")

    def test_front_door_fails_if_mediator_confounded(self):
        """Front-door fails if U also affects M.

             U -----+
            / \\     |
           v   v   v
           X -> M -> Y

        U -> M breaks front-door criterion.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "X"},
                {"name": "M"},
                {"name": "Y", "is_outcome": True},
                {"name": "U"},
            ],
            edges=[
                {"cause": "X", "effect": "M"},
                {"cause": "M", "effect": "Y"},
                {"cause": "U", "effect": "X"},
                {"cause": "U", "effect": "Y"},
                {"cause": "U", "effect": "M"},  # Breaks front-door
            ],
        )
        measurement_model = make_measurement_model(["X", "M", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        assert_not_identifiable(result, "X", "U->M breaks front-door")

    def test_front_door_with_multiple_mediators(self):
        """Front-door with parallel mediators.

             U
            / \\
           v   v
           X -> M1 -> Y
           |         ^
           +-> M2 ---+

        Both paths X->M1->Y and X->M2->Y are unconfounded.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "X"},
                {"name": "M1"},
                {"name": "M2"},
                {"name": "Y", "is_outcome": True},
                {"name": "U"},
            ],
            edges=[
                {"cause": "X", "effect": "M1"},
                {"cause": "X", "effect": "M2"},
                {"cause": "M1", "effect": "Y"},
                {"cause": "M2", "effect": "Y"},
                {"cause": "U", "effect": "X"},
                {"cause": "U", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["X", "M1", "M2", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        # Front-door should still work with multiple unconfounded mediators
        assert_identifiable(result, "X", "Front-door with multiple mediators")


# =============================================================================
# 4. INSTRUMENTAL VARIABLES
# =============================================================================


class TestInstrumentalVariables:
    """Test identification via instrumental variables."""

    def test_classic_iv(self):
        """Classic IV: Z -> X -> Y with U -> X, U -> Y.

        Z -> X -> Y
             ^   ^
             U --+

        Z is an instrument: affects Y only through X, not confounded with Y.

        Under linearity (which our framework assumes), IV identification works.
        The estimand notes that linearity is required.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "Z", "role": "exogenous"},
                {"name": "X"},
                {"name": "Y", "is_outcome": True},
                {"name": "U"},
            ],
            edges=[
                {"cause": "Z", "effect": "X"},
                {"cause": "X", "effect": "Y"},
                {"cause": "U", "effect": "X"},
                {"cause": "U", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["Z", "X", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        # X -> Y is identifiable via IV (under linearity)
        assert_identifiable(result, "X", "IV identification under linearity")
        assert "IV(Z)" in get_estimand(result, "X")
        assert "linearity" in get_estimand(result, "X").lower()
        # Z -> Y is also identifiable (no confounding)
        assert_identifiable(result, "Z", "Z -> X -> Y is identifiable")

    def test_iv_fails_if_instrument_confounded(self):
        """IV fails if instrument is confounded with outcome.

        Z -> X -> Y
        ^    ^   ^
        +--- U --+

        U -> Z breaks exogeneity (Z shares unobserved common cause with Y).
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "Z", "role": "exogenous"},
                {"name": "X"},
                {"name": "Y", "is_outcome": True},
                {"name": "U"},
            ],
            edges=[
                {"cause": "Z", "effect": "X"},
                {"cause": "X", "effect": "Y"},
                {"cause": "U", "effect": "X"},
                {"cause": "U", "effect": "Y"},
                {"cause": "U", "effect": "Z"},  # Breaks IV exogeneity
            ],
        )
        measurement_model = make_measurement_model(["Z", "X", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        # Z is not a valid instrument (confounded with Y via U)
        # X -> Y is not identifiable
        assert_not_identifiable(result, "X", "U->Z breaks IV exogeneity")

    def test_iv_fails_if_direct_path_to_outcome(self):
        """IV fails if instrument has direct path to outcome.

        Z -> X -> Y
        |         ^
        +---------+

        Z -> Y directly violates exclusion restriction.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "Z", "role": "exogenous"},
                {"name": "X"},
                {"name": "Y", "is_outcome": True},
                {"name": "U"},
            ],
            edges=[
                {"cause": "Z", "effect": "X"},
                {"cause": "Z", "effect": "Y"},  # Direct effect
                {"cause": "X", "effect": "Y"},
                {"cause": "U", "effect": "X"},
                {"cause": "U", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["Z", "X", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        # Z->Y makes Z not a valid instrument for X (violates exclusion)
        # U confounds X-Y with no alternative identification strategy for X
        assert_not_identifiable(result, "X", "Z->Y violates exclusion, U confounds X-Y")

        # Z's effect on Y IS identifiable (Z is observed, no unobserved confounder of Z-Y)
        assert_identifiable(result, "Z", "Z->Y direct effect identifiable")


# =============================================================================
# 5. TEMPORAL DYNAMICS (AR(1) under A3a)
# =============================================================================


class TestTemporalDynamics:
    """Test 2-timestep unrolling under AR(1) assumption (A3a)."""

    def test_lagged_confounding_blocks_identification(self):
        """Lagged confounding: U_{t-1} -> X_t, U_{t-1} -> Y_t.

        This creates confounding in the unrolled graph even though
        U doesn't contemporaneously affect both X and Y.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "X", "temporal_status": "time_varying"},
                {"name": "Y", "temporal_status": "time_varying", "is_outcome": True},
                {"name": "U", "temporal_status": "time_varying"},
            ],
            edges=[
                {"cause": "X", "effect": "Y", "lagged": False},  # X_t -> Y_t
                {"cause": "U", "effect": "X", "lagged": True},  # U_{t-1} -> X_t
                {"cause": "U", "effect": "Y", "lagged": True},  # U_{t-1} -> Y_t
            ],
        )
        measurement_model = make_measurement_model(["X", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        assert_not_identifiable(result, "X", "Lagged U confounds X_t and Y_t")
        assert_blocked_by(result, "X", "U")

    def test_contemporaneous_confounding(self):
        """Contemporaneous confounding: U_t -> X_t, U_t -> Y_t."""
        latent_model = make_latent_model(
            constructs=[
                {"name": "X", "temporal_status": "time_varying"},
                {"name": "Y", "temporal_status": "time_varying", "is_outcome": True},
                {"name": "U", "temporal_status": "time_varying"},
            ],
            edges=[
                {"cause": "X", "effect": "Y", "lagged": False},
                {"cause": "U", "effect": "X", "lagged": False},  # U_t -> X_t
                {"cause": "U", "effect": "Y", "lagged": False},  # U_t -> Y_t
            ],
        )
        measurement_model = make_measurement_model(["X", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        assert_not_identifiable(result, "X", "Contemporaneous confounding")
        assert_blocked_by(result, "X", "U")

    def test_ar1_enables_identification_via_lagged_adjustment(self):
        """AR(1) enables identification by adjusting for lagged values.

        In the 2-timestep unrolled graph, conditioning on X_{t-1} and Y_{t-1}
        can block backdoor paths. This is similar to panel data fixed effects.

        U_t -> X_t (contemporaneous, mirrored to U_{t-1} -> X_{t-1})
        U_{t-1} -> Y_t (lagged)

        The backdoor path U_{t-1} -> X_{t-1} -> X_t is blocked by conditioning
        on X_{t-1}. y0 finds this identification strategy.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "X", "temporal_status": "time_varying"},
                {"name": "Y", "temporal_status": "time_varying", "is_outcome": True},
                {"name": "U", "temporal_status": "time_varying"},
            ],
            edges=[
                {"cause": "X", "effect": "Y", "lagged": False},  # X_t -> Y_t
                {
                    "cause": "U",
                    "effect": "X",
                    "lagged": False,
                },  # U_t -> X_t (also U_{t-1} -> X_{t-1})
                {"cause": "U", "effect": "Y", "lagged": True},  # U_{t-1} -> Y_t
            ],
        )
        measurement_model = make_measurement_model(["X", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        # y0 finds identification via adjustment for lagged values
        # The estimand involves P(Y_t | X_t, X_{t-1}, Y_{t-1})
        assert_identifiable(result, "X", "Lagged adjustment enables identification")

    def test_staggered_confounding_identifiable_via_lagged_adjustment(self):
        """Staggered temporal confounding can be identified via lagged adjustment.

        U_{t-1} -> X_t (lagged effect on X)
        U_t -> Y_t (contemporaneous effect on Y)

        With AR(1): U_{t-1} -> U_t creates a path, but conditioning on
        lagged observed values can block backdoor paths in the 2-timestep graph.
        y0 finds an identification strategy.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "X", "temporal_status": "time_varying"},
                {"name": "Y", "temporal_status": "time_varying", "is_outcome": True},
                {"name": "U", "temporal_status": "time_varying"},
            ],
            edges=[
                {"cause": "X", "effect": "Y", "lagged": False},
                {"cause": "U", "effect": "X", "lagged": True},  # U_{t-1} -> X_t
                {"cause": "U", "effect": "Y", "lagged": False},  # U_t -> Y_t
            ],
        )
        measurement_model = make_measurement_model(["X", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        # y0 finds identification via adjustment for lagged values
        assert_identifiable(result, "X", "Lagged adjustment blocks backdoor")

    def test_lagged_treatment_effect_observed(self):
        """Lagged treatment effect with observed past.

        X_{t-1} -> Y_t (lagged effect).
        If we observe X at both timesteps, this should be identifiable.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "X", "temporal_status": "time_varying"},
                {"name": "Y", "temporal_status": "time_varying", "is_outcome": True},
            ],
            edges=[
                {"cause": "X", "effect": "Y", "lagged": True},  # X_{t-1} -> Y_t
            ],
        )
        measurement_model = make_measurement_model(["X", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        # X is observed, no confounding - should be identifiable
        assert_identifiable(result, "X", "Lagged effect with no confounding")

    def test_mixed_lagged_and_contemporaneous(self):
        """Mixed lagged and contemporaneous edges.

        X -> Y (contemporaneous) and X -> Y (lagged) - modeling both effects.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "X", "temporal_status": "time_varying"},
                {"name": "Y", "temporal_status": "time_varying", "is_outcome": True},
            ],
            edges=[
                {"cause": "X", "effect": "Y", "lagged": False},  # X_t -> Y_t
                {"cause": "X", "effect": "Y", "lagged": True},  # X_{t-1} -> Y_t
            ],
        )
        measurement_model = make_measurement_model(["X", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        assert_identifiable(result, "X", "Both effects identifiable")

    def test_feedback_loop(self):
        """Feedback loop: X -> Y and Y -> X (at different times).

        X_t -> Y_t and Y_{t-1} -> X_t represents reciprocal causation.
        Under AR(1), the graph is acyclic when unrolled.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "X", "temporal_status": "time_varying"},
                {"name": "Y", "temporal_status": "time_varying", "is_outcome": True},
            ],
            edges=[
                {"cause": "X", "effect": "Y", "lagged": False},  # X_t -> Y_t
                {"cause": "Y", "effect": "X", "lagged": True},  # Y_{t-1} -> X_t
            ],
        )
        measurement_model = make_measurement_model(["X", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        # With both observed and no unobserved confounders, should be identifiable
        assert_identifiable(result, "X", "Feedback loop identifiable when all observed")


# =============================================================================
# 6. TIME-INVARIANT CONSTRUCTS
# =============================================================================


class TestTimeInvariantConstructs:
    """Test mixed time-invariant and time-varying constructs."""

    def test_time_invariant_confounder_observed(self):
        """Time-invariant trait confounds time-varying constructs.

        Trait -> X_t, Trait -> Y_t (stable trait affects both).
        If Trait is observed, should be identifiable by adjustment.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "Trait", "role": "exogenous", "temporal_status": "time_invariant"},
                {"name": "X", "temporal_status": "time_varying"},
                {"name": "Y", "temporal_status": "time_varying", "is_outcome": True},
            ],
            edges=[
                {"cause": "Trait", "effect": "X", "lagged": False},
                {"cause": "Trait", "effect": "Y", "lagged": False},
                {"cause": "X", "effect": "Y", "lagged": False},
            ],
        )
        measurement_model = make_measurement_model(["Trait", "X", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        assert_identifiable(result, "X", "Adjust for observed Trait")

    def test_time_invariant_confounder_unobserved(self):
        """Unobserved time-invariant trait confounds X and Y.

        Stable traits like personality, genetics, etc. that aren't measured.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "Trait", "role": "exogenous", "temporal_status": "time_invariant"},
                {"name": "X", "temporal_status": "time_varying"},
                {"name": "Y", "temporal_status": "time_varying", "is_outcome": True},
            ],
            edges=[
                {"cause": "Trait", "effect": "X", "lagged": False},
                {"cause": "Trait", "effect": "Y", "lagged": False},
                {"cause": "X", "effect": "Y", "lagged": False},
            ],
        )
        measurement_model = make_measurement_model(["X", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        assert_not_identifiable(result, "X", "Unobserved Trait confounds")
        assert_blocked_by(result, "X", "Trait")

    def test_time_invariant_treatment(self):
        """Time-invariant treatment affecting time-varying outcome.

        E.g., a one-time intervention or baseline characteristic.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "Treatment", "temporal_status": "time_invariant"},
                {"name": "Y", "temporal_status": "time_varying", "is_outcome": True},
            ],
            edges=[
                {"cause": "Treatment", "effect": "Y", "lagged": False},
            ],
        )
        measurement_model = make_measurement_model(["Treatment", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        assert_identifiable(result, "Treatment", "Time-invariant treatment")

    def test_mixed_time_status_chain(self):
        """Chain with mixed time statuses.

        Trait -> X_t -> Y_t where Trait is stable, X and Y vary.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "Trait", "role": "exogenous", "temporal_status": "time_invariant"},
                {"name": "X", "temporal_status": "time_varying"},
                {"name": "Y", "temporal_status": "time_varying", "is_outcome": True},
            ],
            edges=[
                {"cause": "Trait", "effect": "X", "lagged": False},
                {"cause": "X", "effect": "Y", "lagged": False},
            ],
        )
        measurement_model = make_measurement_model(["Trait", "X", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        # No confounding, all observed
        assert_identifiable(result, "X", "Chain with time-invariant observed")
        assert_identifiable(result, "Trait", "Trait effect identifiable")


# =============================================================================
# 7. COMPLEX CONFOUNDING PATTERNS
# =============================================================================


class TestComplexConfounding:
    """Test complex confounding structures."""

    def test_diamond_graph_all_observed(self):
        """Diamond: X -> A -> Y, X -> B -> Y (all observed).

        X -> A -> Y
        |         ^
        +-> B ----+
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "X"},
                {"name": "A"},
                {"name": "B"},
                {"name": "Y", "is_outcome": True},
            ],
            edges=[
                {"cause": "X", "effect": "A"},
                {"cause": "X", "effect": "B"},
                {"cause": "A", "effect": "Y"},
                {"cause": "B", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["X", "A", "B", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        assert_identifiable(result, "X", "Diamond with all observed")

    def test_chain_of_unobserved(self):
        """Chain of unobserved: U1 -> U2 -> X, U2 -> Y.

        Only the closest unobserved to both X and Y matters.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "U1"},
                {"name": "U2"},
                {"name": "X"},
                {"name": "Y", "is_outcome": True},
            ],
            edges=[
                {"cause": "U1", "effect": "U2"},
                {"cause": "U2", "effect": "X"},
                {"cause": "U2", "effect": "Y"},
                {"cause": "X", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["X", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        # U2 is the confounder (has observed children X and Y)
        assert_not_identifiable(result, "X", "U2 confounds X and Y")
        assert_blocked_by(result, "X", "U2")

    def test_find_blocking_confounders_latent_chain_to_outcome(self):
        """Test find_blocking_confounders catches confounders via latent chains.

        U -> X
        U -> V -> Y
        X -> Y

        Where U and V are unobserved. U is a confounder because it creates
        the backdoor path: X <- U -> V -> Y

        U has only ONE direct observed child (X), so the old heuristic of
        "2+ direct observed children" would miss it.

        This test directly checks find_blocking_confounders, not full identification.
        """
        from causal_ssm_agent.utils.identifiability import find_blocking_confounders

        latent_model = make_latent_model(
            constructs=[
                {"name": "U"},
                {"name": "V"},
                {"name": "X"},
                {"name": "Y", "is_outcome": True},
            ],
            edges=[
                {"cause": "U", "effect": "X"},
                {"cause": "U", "effect": "V"},
                {"cause": "V", "effect": "Y"},
                {"cause": "X", "effect": "Y"},
            ],
        )
        observed = {"X", "Y"}

        blockers = find_blocking_confounders(latent_model, observed, "X", "Y")

        # U creates backdoor path X <- U -> V -> Y, should be reported
        assert "U" in blockers, f"U should be a blocking confounder. Got: {blockers}"
        # V does NOT create a backdoor path (no path from V to X)
        assert "V" not in blockers, f"V should not be a blocking confounder. Got: {blockers}"

    def test_m_bias_structure(self):
        """M-bias (butterfly): Classic structure where naive adjustment fails.

        U1 -> A -> X
              |
        U2 -> B -> Y

        A and B are observed "pre-treatment" variables.
        Conditioning on A opens a path through U1.

        But in this simple form with no direct confounding of X-Y,
        X -> Y is actually identifiable without conditioning on A or B.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "U1"},
                {"name": "U2"},
                {"name": "A"},
                {"name": "B"},
                {"name": "X"},
                {"name": "Y", "is_outcome": True},
            ],
            edges=[
                {"cause": "U1", "effect": "A"},
                {"cause": "A", "effect": "X"},
                {"cause": "U2", "effect": "B"},
                {"cause": "B", "effect": "Y"},
                {"cause": "X", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["A", "B", "X", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        # No backdoor X <- ... -> Y, so identifiable without adjustment
        assert_identifiable(result, "X", "No X-Y backdoor in M-bias")

    def test_m_bias_with_direct_confounding_but_iv_available(self):
        """M-bias with direct confounding, but A serves as instrument.

        U1 -> A -> X <- U3 -> Y
              |              ^
        U2 -> B -------------+

        U3 confounds X and Y directly, but A is a valid instrument:
        - A -> X (relevance)
        - A has no path to Y except through X (exclusion)
        - A is not a descendant of U3 or any other cause of Y (exogeneity)
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "U1"},
                {"name": "U2"},
                {"name": "U3"},
                {"name": "A"},
                {"name": "B"},
                {"name": "X"},
                {"name": "Y", "is_outcome": True},
            ],
            edges=[
                {"cause": "U1", "effect": "A"},
                {"cause": "A", "effect": "X"},
                {"cause": "U2", "effect": "B"},
                {"cause": "B", "effect": "Y"},
                {"cause": "U3", "effect": "X"},
                {"cause": "U3", "effect": "Y"},
                {"cause": "X", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["A", "B", "X", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        # X is identifiable via IV using A
        assert_identifiable(result, "X", "A is valid instrument despite U3")
        assert "IV(A)" in get_estimand(result, "X")

    def test_multiple_disjoint_confounders(self):
        """Multiple independent unobserved confounders.

        U1 -> X, U1 -> Y
        U2 -> X, U2 -> Y
        X -> Y

        Both confound independently.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "U1"},
                {"name": "U2"},
                {"name": "X"},
                {"name": "Y", "is_outcome": True},
            ],
            edges=[
                {"cause": "U1", "effect": "X"},
                {"cause": "U1", "effect": "Y"},
                {"cause": "U2", "effect": "X"},
                {"cause": "U2", "effect": "Y"},
                {"cause": "X", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["X", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        assert_not_identifiable(result, "X", "Multiple confounders")
        # At least one should be listed as blocking
        blockers = get_blockers(result, "X")
        assert "U1" in blockers or "U2" in blockers


# =============================================================================
# 8. COLLIDER STRUCTURES
# =============================================================================


class TestColliderStructures:
    """Test that colliders don't create spurious confounding."""

    def test_simple_collider(self):
        """Simple collider: X -> C <- Y.

        X and Y are independent (no confounding).
        C is a collider - conditioning on it would create bias,
        but we don't condition on descendants.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "X"},
                {"name": "C"},
                {"name": "Y", "is_outcome": True},
            ],
            edges=[
                {"cause": "X", "effect": "C"},
                {"cause": "Y", "effect": "C"},
                {"cause": "X", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["X", "C", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        # No confounding - collider blocks the path X -> C <- Y
        assert_identifiable(result, "X", "Collider doesn't create confounding")

    def test_collider_with_descendant(self):
        """Collider with descendant: X -> C <- U -> Y, C -> D.

        Even if we observe D (descendant of collider),
        do-calculus doesn't condition on colliders/descendants.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "X"},
                {"name": "C"},
                {"name": "D"},
                {"name": "U"},
                {"name": "Y", "is_outcome": True},
            ],
            edges=[
                {"cause": "X", "effect": "C"},
                {"cause": "U", "effect": "C"},
                {"cause": "U", "effect": "Y"},
                {"cause": "C", "effect": "D"},
                {"cause": "X", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["X", "C", "D", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        # U confounds X-Y via X <- ... U -> Y? No, U -> C <- X, not U -> X
        # The path X -> C <- U -> Y is blocked at collider C
        assert_identifiable(result, "X", "Collider blocks U->Y path")


# =============================================================================
# 9. MULTIPLE TREATMENTS
# =============================================================================


class TestMultipleTreatments:
    """Test handling of multiple treatment variables."""

    def test_multiple_treatments_all_identifiable(self):
        """Multiple treatments, all with identifiable effects.

        X1 -> Y
        X2 -> Y
        (no confounding)
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "X1"},
                {"name": "X2"},
                {"name": "Y", "is_outcome": True},
            ],
            edges=[
                {"cause": "X1", "effect": "Y"},
                {"cause": "X2", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["X1", "X2", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        assert_identifiable(result, "X1", "X1 effect identifiable")
        assert_identifiable(result, "X2", "X2 effect identifiable")

    def test_multiple_treatments_some_identifiable(self):
        """Multiple treatments, only some identifiable.

        X1 -> Y (no confounding)
        X2 -> Y with U -> X2, U -> Y (confounded)
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "X1"},
                {"name": "X2"},
                {"name": "U"},
                {"name": "Y", "is_outcome": True},
            ],
            edges=[
                {"cause": "X1", "effect": "Y"},
                {"cause": "X2", "effect": "Y"},
                {"cause": "U", "effect": "X2"},
                {"cause": "U", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["X1", "X2", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        assert_identifiable(result, "X1", "X1 unconfounded")
        assert_not_identifiable(result, "X2", "X2 confounded by U")

    def test_treatment_chain(self):
        """Chain of treatments: X1 -> X2 -> Y.

        Both X1 and X2 are treatments for Y.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "X1"},
                {"name": "X2"},
                {"name": "Y", "is_outcome": True},
            ],
            edges=[
                {"cause": "X1", "effect": "X2"},
                {"cause": "X2", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["X1", "X2", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        assert_identifiable(result, "X1", "X1 -> X2 -> Y path")
        assert_identifiable(result, "X2", "Direct X2 -> Y")


# =============================================================================
# 10. EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Test boundary conditions and edge cases."""

    def test_outcome_only_no_treatments(self):
        """Graph with only outcome, no treatments."""
        latent_model = make_latent_model(
            constructs=[
                {"name": "Y", "is_outcome": True},
            ],
            edges=[],
        )
        measurement_model = make_measurement_model(["Y"])

        result = check_identifiability(latent_model, measurement_model)

        assert len(result["identifiable_treatments"]) == 0
        assert len(result["non_identifiable_treatments"]) == 0

    def test_unobserved_outcome(self):
        """Unobserved outcome - nothing should be identifiable."""
        latent_model = make_latent_model(
            constructs=[
                {"name": "X"},
                {"name": "Y", "is_outcome": True},
            ],
            edges=[
                {"cause": "X", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["X"])  # Y unobserved

        result = check_identifiability(latent_model, measurement_model)

        # Can't identify effects on unobserved outcome
        assert len(result["identifiable_treatments"]) == 0

    def test_all_constructs_unobserved_except_outcome(self):
        """All unobserved except outcome - no treatments."""
        latent_model = make_latent_model(
            constructs=[
                {"name": "X"},
                {"name": "Y", "is_outcome": True},
            ],
            edges=[
                {"cause": "X", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["Y"])  # Only Y observed

        result = check_identifiability(latent_model, measurement_model)

        # X is not a treatment (unobserved)
        assert "X" not in result["identifiable_treatments"]
        assert "X" not in result["non_identifiable_treatments"]

    def test_long_causal_chain(self):
        """Long chain: A -> B -> C -> D -> E -> Y (all observed)."""
        latent_model = make_latent_model(
            constructs=[
                {"name": "A"},
                {"name": "B"},
                {"name": "C"},
                {"name": "D"},
                {"name": "E"},
                {"name": "Y", "is_outcome": True},
            ],
            edges=[
                {"cause": "A", "effect": "B"},
                {"cause": "B", "effect": "C"},
                {"cause": "C", "effect": "D"},
                {"cause": "D", "effect": "E"},
                {"cause": "E", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["A", "B", "C", "D", "E", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        # All should be identifiable (no confounding)
        for treatment in ["A", "B", "C", "D", "E"]:
            assert_identifiable(result, treatment, f"{treatment} in long chain")

    def test_isolated_construct(self):
        """Construct with no edges (isolated in graph)."""
        latent_model = make_latent_model(
            constructs=[
                {"name": "X"},
                {"name": "Isolated"},  # No edges
                {"name": "Y", "is_outcome": True},
            ],
            edges=[
                {"cause": "X", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["X", "Isolated", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        # Isolated has no path to Y, so it's not a treatment
        assert "Isolated" not in result["identifiable_treatments"]
        assert "Isolated" not in result["non_identifiable_treatments"]
        assert_identifiable(result, "X")

    def test_no_path_to_outcome(self):
        """Treatment candidate with no path to outcome."""
        latent_model = make_latent_model(
            constructs=[
                {"name": "X"},
                {"name": "Z"},  # Points away from Y
                {"name": "Y", "is_outcome": True},
            ],
            edges=[
                {"cause": "X", "effect": "Y"},
                {"cause": "Z", "effect": "X"},  # Z -> X, no path Z -> Y
            ],
        )
        measurement_model = make_measurement_model(["X", "Z", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        # Z -> X -> Y, so Z does have a path to Y
        assert_identifiable(result, "Z", "Z -> X -> Y path exists")
        assert_identifiable(result, "X")


# =============================================================================
# 11. MARGINALIZATION ANALYSIS
# =============================================================================


class TestMarginalizationAnalysis:
    """Test classification of unobserved constructs for marginalization."""

    def test_marginalize_single_child_confounder(self):
        """Unobserved with single observed child can be marginalized."""
        latent_model = make_latent_model(
            constructs=[
                {"name": "U"},  # Only affects X
                {"name": "X"},
                {"name": "Y", "is_outcome": True},
            ],
            edges=[
                {"cause": "U", "effect": "X"},
                {"cause": "X", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["X", "Y"])

        id_result = check_identifiability(latent_model, measurement_model)
        analysis = analyze_unobserved_constructs(latent_model, measurement_model, id_result)

        assert "U" in analysis["can_marginalize"]
        assert "U" not in analysis["blocking_details"]

    def test_needs_modeling_blocking_confounder(self):
        """Confounder blocking identification needs modeling."""
        latent_model = make_latent_model(
            constructs=[
                {"name": "U"},
                {"name": "X"},
                {"name": "Y", "is_outcome": True},
            ],
            edges=[
                {"cause": "U", "effect": "X"},
                {"cause": "U", "effect": "Y"},
                {"cause": "X", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["X", "Y"])

        id_result = check_identifiability(latent_model, measurement_model)
        analysis = analyze_unobserved_constructs(latent_model, measurement_model, id_result)

        assert "U" in analysis["blocking_details"]
        assert analysis["blocking_details"]["U"] == ["X"]
        assert "U" not in analysis["can_marginalize"]

    def test_marginalize_front_door_handled(self):
        """Confounder handled by front-door can be marginalized."""
        latent_model = make_latent_model(
            constructs=[
                {"name": "X"},
                {"name": "M"},
                {"name": "Y", "is_outcome": True},
                {"name": "U"},
            ],
            edges=[
                {"cause": "X", "effect": "M"},
                {"cause": "M", "effect": "Y"},
                {"cause": "U", "effect": "X"},
                {"cause": "U", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["X", "M", "Y"])

        id_result = check_identifiability(latent_model, measurement_model)
        analysis = analyze_unobserved_constructs(latent_model, measurement_model, id_result)

        # X is identifiable via front-door, so U can be marginalized
        assert "U" in analysis["can_marginalize"]

    def test_mixed_marginalization(self):
        """Some unobserved can be marginalized, others can't.

        U1 -> X only (can marginalize)
        U2 -> X, U2 -> Y (needs modeling)
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "U1"},
                {"name": "U2"},
                {"name": "X"},
                {"name": "Y", "is_outcome": True},
            ],
            edges=[
                {"cause": "U1", "effect": "X"},
                {"cause": "U2", "effect": "X"},
                {"cause": "U2", "effect": "Y"},
                {"cause": "X", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["X", "Y"])

        id_result = check_identifiability(latent_model, measurement_model)
        analysis = analyze_unobserved_constructs(latent_model, measurement_model, id_result)

        assert "U1" in analysis["can_marginalize"]
        assert "U2" in analysis["blocking_details"]

    def test_chain_of_unobserved_marginalization(self):
        """Chain of unobserved: only the one creating confounding needs modeling.

        U1 -> U2 -> X, U2 -> Y

        U1 has no observed children (only U2), so it can be marginalized.
        U2 confounds X and Y, needs modeling.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "U1"},
                {"name": "U2"},
                {"name": "X"},
                {"name": "Y", "is_outcome": True},
            ],
            edges=[
                {"cause": "U1", "effect": "U2"},
                {"cause": "U2", "effect": "X"},
                {"cause": "U2", "effect": "Y"},
                {"cause": "X", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["X", "Y"])

        id_result = check_identifiability(latent_model, measurement_model)
        analysis = analyze_unobserved_constructs(latent_model, measurement_model, id_result)

        assert "U1" in analysis["can_marginalize"], "U1 has no observed children"
        assert "U2" in analysis["blocking_details"], "U2 confounds X and Y"


# =============================================================================
# 12. UNROLLING VERIFICATION
# =============================================================================


class TestUnrollingVerification:
    """Verify correct construction of unrolled temporal graphs."""

    def test_unroll_creates_two_timesteps(self):
        """Verify nodes at both t and t-1 are created."""
        latent_model = make_latent_model(
            constructs=[
                {"name": "X", "temporal_status": "time_varying"},
                {"name": "Y", "temporal_status": "time_varying", "is_outcome": True},
            ],
            edges=[
                {"cause": "X", "effect": "Y", "lagged": False},
            ],
        )
        observed = {"X", "Y"}

        dag = unroll_temporal_dag(latent_model, observed)

        nodes = set(dag.nodes())
        assert "X_t" in nodes
        assert "X_{t-1}" in nodes
        assert "Y_t" in nodes
        assert "Y_{t-1}" in nodes

    def test_unroll_ar1_edges(self):
        """Verify AR(1) edges are added for OBSERVED time-varying constructs."""
        latent_model = make_latent_model(
            constructs=[
                {"name": "X", "temporal_status": "time_varying"},
                {"name": "Y", "temporal_status": "time_varying", "is_outcome": True},
            ],
            edges=[
                {"cause": "X", "effect": "Y", "lagged": False},
            ],
        )
        observed = {"X", "Y"}

        dag = unroll_temporal_dag(latent_model, observed)

        edges = list(dag.edges())
        assert ("X_{t-1}", "X_t") in edges, "X AR(1)"
        assert ("Y_{t-1}", "Y_t") in edges, "Y AR(1)"

    def test_unroll_ar1_not_added_for_unobserved(self):
        """Verify AR(1) edges are NOT added for unobserved constructs.

        AR(1) on hidden nodes doesn't add confounding information for
        identification - it just models U's internal dynamics. What matters
        is the confounding edges from U to observed nodes.

        Including AR(1) for hidden nodes would cause y0's projection to
        incorrectly include hidden nodes in the ADMG.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "X", "temporal_status": "time_varying"},
                {"name": "U", "temporal_status": "time_varying"},  # Unobserved
                {"name": "Y", "temporal_status": "time_varying", "is_outcome": True},
            ],
            edges=[
                {"cause": "X", "effect": "Y", "lagged": False},
                {"cause": "U", "effect": "X", "lagged": False},
                {"cause": "U", "effect": "Y", "lagged": False},
            ],
        )
        observed = {"X", "Y"}  # U is unobserved

        dag = unroll_temporal_dag(latent_model, observed)

        edges = list(dag.edges())

        # Observed constructs should have AR(1) edges
        assert ("X_{t-1}", "X_t") in edges, "X AR(1) should exist"
        assert ("Y_{t-1}", "Y_t") in edges, "Y AR(1) should exist"

        # Unobserved construct U should NOT have AR(1) edge
        assert ("U_{t-1}", "U_t") not in edges, "U AR(1) should NOT exist"

        # But U nodes should still exist (for confounding edges)
        assert "U_t" in dag.nodes()
        assert "U_{t-1}" in dag.nodes()

    def test_unroll_mirrored_contemporaneous(self):
        """Verify contemporaneous edges appear at both timesteps."""
        latent_model = make_latent_model(
            constructs=[
                {"name": "X", "temporal_status": "time_varying"},
                {"name": "Y", "temporal_status": "time_varying", "is_outcome": True},
            ],
            edges=[
                {"cause": "X", "effect": "Y", "lagged": False},
            ],
        )
        observed = {"X", "Y"}

        dag = unroll_temporal_dag(latent_model, observed)

        edges = list(dag.edges())
        assert ("X_t", "Y_t") in edges, "Current timestep"
        assert ("X_{t-1}", "Y_{t-1}") in edges, "Mirrored previous timestep"

    def test_unroll_lagged_edges(self):
        """Verify lagged edges connect t-1 to t."""
        latent_model = make_latent_model(
            constructs=[
                {"name": "X", "temporal_status": "time_varying"},
                {"name": "Y", "temporal_status": "time_varying", "is_outcome": True},
            ],
            edges=[
                {"cause": "X", "effect": "Y", "lagged": True},
            ],
        )
        observed = {"X", "Y"}

        dag = unroll_temporal_dag(latent_model, observed)

        edges = list(dag.edges())
        assert ("X_{t-1}", "Y_t") in edges, "Lagged X_{t-1} -> Y_t"

    def test_unroll_time_invariant_single_node(self):
        """Verify time-invariant constructs get single node (no subscript)."""
        latent_model = make_latent_model(
            constructs=[
                {"name": "Trait", "temporal_status": "time_invariant"},
                {"name": "Y", "temporal_status": "time_varying", "is_outcome": True},
            ],
            edges=[
                {"cause": "Trait", "effect": "Y", "lagged": False},
            ],
        )
        observed = {"Trait", "Y"}

        dag = unroll_temporal_dag(latent_model, observed)

        nodes = set(dag.nodes())
        assert "Trait" in nodes, "Time-invariant has no subscript"
        assert "Trait_t" not in nodes
        assert "Trait_{t-1}" not in nodes

    def test_unroll_time_invariant_affects_both_timesteps(self):
        """Time-invariant constructs affect time-varying at both t and t-1."""
        latent_model = make_latent_model(
            constructs=[
                {"name": "Trait", "temporal_status": "time_invariant"},
                {"name": "Y", "temporal_status": "time_varying", "is_outcome": True},
            ],
            edges=[
                {"cause": "Trait", "effect": "Y", "lagged": False},
            ],
        )
        observed = {"Trait", "Y"}

        dag = unroll_temporal_dag(latent_model, observed)

        edges = list(dag.edges())
        assert ("Trait", "Y_t") in edges
        assert ("Trait", "Y_{t-1}") in edges

    def test_unroll_hidden_labels_correct(self):
        """Verify hidden labels are set correctly in unrolled graph."""
        latent_model = make_latent_model(
            constructs=[
                {"name": "X", "temporal_status": "time_varying"},
                {"name": "U", "temporal_status": "time_varying"},
                {"name": "Y", "temporal_status": "time_varying", "is_outcome": True},
            ],
            edges=[
                {"cause": "X", "effect": "Y", "lagged": False},
                {"cause": "U", "effect": "X", "lagged": False},
                {"cause": "U", "effect": "Y", "lagged": False},
            ],
        )
        observed = {"X", "Y"}  # U is unobserved

        dag = unroll_temporal_dag(latent_model, observed)

        # Observed nodes should have hidden=False
        assert dag.nodes["X_t"].get("hidden", False) is False
        assert dag.nodes["Y_t"].get("hidden", False) is False

        # Unobserved nodes should have hidden=True
        assert dag.nodes["U_t"].get("hidden", False) is True
        assert dag.nodes["U_{t-1}"].get("hidden", False) is True


# =============================================================================
# 13. ADMG PROJECTION VERIFICATION
# =============================================================================


class TestADMGProjection:
    """Verify correct projection from DAG to ADMG."""

    def test_bidirected_from_contemporaneous_confounder(self):
        """Contemporaneous confounder creates bidirected edge at t."""
        latent_model = make_latent_model(
            constructs=[
                {"name": "X", "temporal_status": "time_varying"},
                {"name": "Y", "temporal_status": "time_varying", "is_outcome": True},
                {"name": "U", "temporal_status": "time_varying"},
            ],
            edges=[
                {"cause": "X", "effect": "Y", "lagged": False},
                {"cause": "U", "effect": "X", "lagged": False},
                {"cause": "U", "effect": "Y", "lagged": False},
            ],
        )
        observed = {"X", "Y"}

        admg, confounders = dag_to_admg(latent_model, observed)

        assert "U" in confounders
        # Check for bidirected edge
        undirected = {tuple(sorted((str(e[0]), str(e[1])))) for e in admg.undirected.edges()}
        assert ("X_t", "Y_t") in undirected or ("X_{t-1}", "Y_{t-1}") in undirected

    def test_bidirected_from_lagged_confounder(self):
        """Lagged confounder creates bidirected edge."""
        latent_model = make_latent_model(
            constructs=[
                {"name": "X", "temporal_status": "time_varying"},
                {"name": "Y", "temporal_status": "time_varying", "is_outcome": True},
                {"name": "U", "temporal_status": "time_varying"},
            ],
            edges=[
                {"cause": "X", "effect": "Y", "lagged": False},
                {"cause": "U", "effect": "X", "lagged": True},
                {"cause": "U", "effect": "Y", "lagged": True},
            ],
        )
        observed = {"X", "Y"}

        admg, confounders = dag_to_admg(latent_model, observed)

        assert "U" in confounders
        # Should have at least one bidirected edge
        assert len(list(admg.undirected.edges())) > 0

    def test_no_bidirected_single_child(self):
        """Unobserved with single observed child creates no bidirected edge."""
        latent_model = make_latent_model(
            constructs=[
                {"name": "X", "temporal_status": "time_varying"},
                {"name": "Y", "temporal_status": "time_varying", "is_outcome": True},
                {"name": "U", "temporal_status": "time_varying"},
            ],
            edges=[
                {"cause": "X", "effect": "Y", "lagged": False},
                {"cause": "U", "effect": "X", "lagged": False},  # Only affects X
            ],
        )
        observed = {"X", "Y"}

        _admg, confounders = dag_to_admg(latent_model, observed)

        # U should not be listed as a confounder (only has 1 observed child)
        assert "U" not in confounders


# =============================================================================
# 14. COMPLEX HEDGE STRUCTURES (KNOWN NON-IDENTIFIABLE GRAPHS)
# =============================================================================


class TestComplexHedgeStructures:
    """Test complex non-identifiable graphs from causal inference literature.

    These are known hedge structures that test the limits of identification.
    The Shpitser-Pearl ID algorithm should correctly identify these as
    non-identifiable.
    """

    def test_napkin_like_identifiable(self):
        r"""Napkin-like graph that IS identifiable.

        This graph structure:
            U1      U2
           / \    / \
          v   v  v   v
          X   W1-W2  Y
           \        ^
            +------+

        Surprisingly, this IS identifiable! The key insight:
        - The backdoor X <- U1 -> W1 -> W2 <- U2 -> Y is blocked at W2 (collider)
        - The direct path X -> Y exists and is unconfounded
        - y0's algorithm finds an identification strategy

        Note: The classic "napkin" hedge requires the confounders to share
        a common child in a way that doesn't create a collider block.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "X"},
                {"name": "W1"},
                {"name": "W2"},
                {"name": "Y", "is_outcome": True},
                {"name": "U1"},
                {"name": "U2"},
            ],
            edges=[
                {"cause": "X", "effect": "Y"},
                {"cause": "W1", "effect": "W2"},
                {"cause": "U1", "effect": "X"},
                {"cause": "U1", "effect": "W1"},
                {"cause": "U2", "effect": "W2"},
                {"cause": "U2", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["X", "W1", "W2", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        # Actually identifiable! W2 is a collider that blocks the backdoor
        assert_identifiable(result, "X", "Napkin-like: collider W2 blocks backdoor")

    def test_kite_graph(self):
        r"""Kite graph: Another hedge structure.

        The kite has confounding that crosses through a mediator:

            U1 ----+
           / \     |
          v   v    |
          X   M    v
           \ / \   |
            v   v  |
            C   Y<-+

        Where C is a collider and U1 affects Y through a long path.
        Actually, let's do the standard kite:

            U
           /|\
          v v v
          X M Y
           \ ^
            \|
             Y

        Simplified kite with confounding triangle.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "X"},
                {"name": "M"},
                {"name": "Y", "is_outcome": True},
                {"name": "U1"},
                {"name": "U2"},
            ],
            edges=[
                {"cause": "X", "effect": "M"},
                {"cause": "M", "effect": "Y"},
                {"cause": "U1", "effect": "X"},
                {"cause": "U1", "effect": "M"},
                {"cause": "U2", "effect": "M"},
                {"cause": "U2", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["X", "M", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        # Kite: confounding at both X-M and M-Y breaks front-door
        assert_not_identifiable(result, "X", "Kite: double confounding on chain")

    def test_double_bow(self):
        """Double bow: Two stacked bow graphs.

        X -> M -> Y
        ^    ^    ^
        U1---+    |
             U2---+

        U1 confounds X-M, U2 confounds M-Y.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "X"},
                {"name": "M"},
                {"name": "Y", "is_outcome": True},
                {"name": "U1"},
                {"name": "U2"},
            ],
            edges=[
                {"cause": "X", "effect": "M"},
                {"cause": "M", "effect": "Y"},
                {"cause": "U1", "effect": "X"},
                {"cause": "U1", "effect": "M"},
                {"cause": "U2", "effect": "M"},
                {"cause": "U2", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["X", "M", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        # Same as confounded_chain - not identifiable
        assert_not_identifiable(result, "X", "Double bow blocks front-door")

    def test_w_structure_confounding(self):
        r"""W-structure: Creates complex backdoor pattern.

        W-graph (also called "butterfly" or "M-bias" in different contexts):

              U1    U2
             / \  / \
            v   vv   v
            A   XY   B
                 \  /
                  \/
                  Y

        Where X affects Y directly and U1, U2 create confounding through A, B.
        But if A and B are not on the X-Y path, this might be identifiable...

        Let's do a proper W:
        U1 -> X, U1 -> Z, U2 -> Z, U2 -> Y, X -> Y
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "X"},
                {"name": "Z"},
                {"name": "Y", "is_outcome": True},
                {"name": "U1"},
                {"name": "U2"},
            ],
            edges=[
                {"cause": "X", "effect": "Y"},
                {"cause": "U1", "effect": "X"},
                {"cause": "U1", "effect": "Z"},
                {"cause": "U2", "effect": "Z"},
                {"cause": "U2", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["X", "Z", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        # W-structure: Z is a collider between U1 and U2 paths
        # The backdoor path X <- U1 -> Z <- U2 -> Y is blocked at Z (collider)
        # So this should actually be identifiable!
        assert_identifiable(result, "X", "W-structure: collider Z blocks backdoor")

    def test_verma_graph_extended(self):
        """Extended Verma constraint graph.

        Tests a graph that satisfies Verma constraints but where
        standard ID may or may not work depending on the structure.

        W -> X -> Y -> Z
             ^        ^
             U1 ------+
             ^
        V ---+

        V -> U1 -> X and U1 -> Z creates complex identification scenario.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "V", "role": "exogenous"},
                {"name": "W", "role": "exogenous"},
                {"name": "X"},
                {"name": "Y", "is_outcome": True},
                {"name": "Z"},
                {"name": "U1"},
            ],
            edges=[
                {"cause": "V", "effect": "U1"},
                {"cause": "W", "effect": "X"},
                {"cause": "X", "effect": "Y"},
                {"cause": "Y", "effect": "Z"},
                {"cause": "U1", "effect": "X"},
                {"cause": "U1", "effect": "Z"},
            ],
        )
        measurement_model = make_measurement_model(["V", "W", "X", "Y", "Z"])

        result = check_identifiability(latent_model, measurement_model)

        # X -> Y is identifiable via W as instrument or backdoor adjustment
        assert_identifiable(result, "X", "W blocks backdoor, or IV via W")


# =============================================================================
# 15. CONDITIONAL INSTRUMENTS AND COMPLEX IV SCENARIOS
# =============================================================================


class TestConditionalInstruments:
    """Test IV scenarios requiring adjustment or with complex structure."""

    def test_iv_with_required_adjustment_not_found(self):
        """Conditional IV: Z is only valid when adjusting for C.

        Z -> X -> Y
        ^    ^   ^
        |    U1--+
        C -------+

        Z is an instrument, but C confounds Z-Y (C -> Z, C -> Y).
        Conditional on C, Z would be a valid instrument, but:
        - y0's nonparametric do-calculus doesn't find this
        - Our IV finder requires Z to be unconditionally exogenous

        This documents a known limitation: conditional instruments require
        specialized algorithms beyond standard do-calculus.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "Z", "role": "exogenous"},
                {"name": "C", "role": "exogenous"},
                {"name": "X"},
                {"name": "Y", "is_outcome": True},
                {"name": "U1"},
            ],
            edges=[
                {"cause": "Z", "effect": "X"},
                {"cause": "C", "effect": "Z"},
                {"cause": "C", "effect": "Y"},
                {"cause": "X", "effect": "Y"},
                {"cause": "U1", "effect": "X"},
                {"cause": "U1", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["Z", "C", "X", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        # Known limitation: conditional IV not detected
        # X is NOT identifiable with current algorithm
        assert_not_identifiable(result, "X", "Conditional IV not supported")
        # But Z and C should be identifiable
        assert_identifiable(result, "Z", "Z -> X -> Y identifiable")
        assert_identifiable(result, "C", "C -> Z -> X -> Y and C -> Y identifiable")

    def test_multiple_instruments(self):
        """Multiple potential instruments available.

        Z1 -> X -> Y
        Z2 -+  ^   ^
               U --+

        Both Z1 and Z2 are valid instruments.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "Z1", "role": "exogenous"},
                {"name": "Z2", "role": "exogenous"},
                {"name": "X"},
                {"name": "Y", "is_outcome": True},
                {"name": "U"},
            ],
            edges=[
                {"cause": "Z1", "effect": "X"},
                {"cause": "Z2", "effect": "X"},
                {"cause": "X", "effect": "Y"},
                {"cause": "U", "effect": "X"},
                {"cause": "U", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["Z1", "Z2", "X", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        assert_identifiable(result, "X", "Multiple valid instruments")
        # Should mention at least one instrument
        assert "IV" in get_estimand(result, "X")

    def test_weak_instrument_chain(self):
        """Instrument through a chain (weaker but valid).

        Z -> W -> X -> Y
              ^   ^   ^
              U1  U2--+

        Z affects X only through W. If W is confounded with X by U1,
        Z might still be valid if U1 doesn't affect Y.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "Z", "role": "exogenous"},
                {"name": "W"},
                {"name": "X"},
                {"name": "Y", "is_outcome": True},
                {"name": "U1"},
                {"name": "U2"},
            ],
            edges=[
                {"cause": "Z", "effect": "W"},
                {"cause": "W", "effect": "X"},
                {"cause": "X", "effect": "Y"},
                {"cause": "U1", "effect": "W"},
                {"cause": "U1", "effect": "X"},
                {"cause": "U2", "effect": "X"},
                {"cause": "U2", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["Z", "W", "X", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        # W serves as instrument for X -> Y:
        # - Relevance: W -> X
        # - Exclusion: W has no path to Y except through X
        # - Exogeneity: U1 -> W, U1 -> X but U1 doesn't affect Y (only U2 does)
        # So W is a valid instrument despite Z being the original exogenous source
        assert_identifiable(result, "X", "W is valid instrument despite indirect Z")
        assert "IV(W)" in get_estimand(result, "X")


# =============================================================================
# 16. TEMPORAL COMPLEXITY - CROSS-LAGGED PANELS AND FEEDBACK
# =============================================================================


class TestTemporalComplexity:
    """Test complex temporal structures common in panel data analysis."""

    def test_cross_lagged_panel_no_confounding(self):
        """Classic cross-lagged panel model without confounding.

        Cross-lagged structure where X and Y affect each other with lags:
        - X_{t-1} -> X_t (autoregressive)
        - Y_{t-1} -> Y_t (autoregressive)
        - X_{t-1} -> Y_t (cross-lagged)
        - Y_{t-1} -> X_t (cross-lagged)
        - X_t -> Y_t (contemporaneous)
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "X", "temporal_status": "time_varying"},
                {"name": "Y", "temporal_status": "time_varying", "is_outcome": True},
            ],
            edges=[
                # Contemporaneous
                {"cause": "X", "effect": "Y", "lagged": False},
                # Cross-lagged
                {"cause": "X", "effect": "Y", "lagged": True},
                {"cause": "Y", "effect": "X", "lagged": True},
            ],
        )
        measurement_model = make_measurement_model(["X", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        # All observed, no confounding - should be identifiable
        assert_identifiable(result, "X", "Cross-lagged panel without confounding")

    def test_cross_lagged_panel_with_trait_confounding(self):
        """Cross-lagged panel with time-invariant trait confounding.

        This is the CLPM vs RI-CLPM debate in psychology:
        A stable trait affects both X and Y at all times.

        Trait -> X_t, Trait -> Y_t (and X_t, Y_t cross-lagged)
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "Trait", "role": "exogenous", "temporal_status": "time_invariant"},
                {"name": "X", "temporal_status": "time_varying"},
                {"name": "Y", "temporal_status": "time_varying", "is_outcome": True},
            ],
            edges=[
                {"cause": "Trait", "effect": "X", "lagged": False},
                {"cause": "Trait", "effect": "Y", "lagged": False},
                {"cause": "X", "effect": "Y", "lagged": False},
                {"cause": "X", "effect": "Y", "lagged": True},
                {"cause": "Y", "effect": "X", "lagged": True},
            ],
        )
        measurement_model = make_measurement_model(["X", "Y"])  # Trait unobserved

        result = check_identifiability(latent_model, measurement_model)

        # Unobserved Trait confounds X and Y - not identifiable
        assert_not_identifiable(result, "X", "Trait confounding in CLPM")
        assert_blocked_by(result, "X", "Trait")

    def test_temporal_front_door(self):
        """Front-door criterion with mediator measured at current + lagged timesteps.

        U_t -> X_t and U_t -> Y_t create contemporaneous confounding.
        X_t -> M_t (current) and M_t -> Y_t ensure classic front-door conditions.
        M also has a lagged effect on next timestep outcome to verify unrolling.

        Even with temporal dynamics, the mediator remains unconfounded and the
        effect P(Y_t | do(X_t)) is identifiable via front-door adjustment.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "X", "temporal_status": "time_varying"},
                {"name": "M", "temporal_status": "time_varying"},
                {"name": "Y", "temporal_status": "time_varying", "is_outcome": True},
                {"name": "U", "temporal_status": "time_varying"},
            ],
            edges=[
                # Classic front-door path plus a lagged carry-over for M
                {"cause": "X", "effect": "M", "lagged": False},
                {"cause": "M", "effect": "Y", "lagged": False},
                {"cause": "M", "effect": "Y", "lagged": True},
                # Contemporaneous confounding on X and Y only
                {"cause": "U", "effect": "X", "lagged": False},
                {"cause": "U", "effect": "Y", "lagged": False},
            ],
        )
        measurement_model = make_measurement_model(["X", "M", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        # Mediator M_t satisfies front-door requirements despite temporal carry-over
        assert_identifiable(result, "X", "Temporal front-door via contemporaneous M_t")

    def test_bidirectional_contemporaneous_confounded(self):
        """Bidirectional contemporaneous effects with confounding.

        X_t <-> Y_t (contemporaneous bidirectional)
        U_t -> X_t, U_t -> Y_t

        This requires careful handling - contemporaneous cycles
        become acyclic when unrolled, but confounding persists.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "X", "temporal_status": "time_varying"},
                {"name": "Y", "temporal_status": "time_varying", "is_outcome": True},
                {"name": "U", "temporal_status": "time_varying"},
            ],
            edges=[
                # Bidirectional contemporaneous (one must be lagged to avoid cycle)
                {"cause": "X", "effect": "Y", "lagged": False},
                {"cause": "Y", "effect": "X", "lagged": True},  # Must be lagged
                # Confounding
                {"cause": "U", "effect": "X", "lagged": False},
                {"cause": "U", "effect": "Y", "lagged": False},
            ],
        )
        measurement_model = make_measurement_model(["X", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        # U confounds X_t and Y_t contemporaneously
        assert_not_identifiable(result, "X", "Contemporaneous confounding persists")

    def test_lagged_instrument_temporal(self):
        """Lagged variable as instrument in temporal setting.

        Z_{t-1} -> X_t -> Y_t
                   ^      ^
                   U_t ---+

        Z at t-1 affects X at t, with no path to Y except through X.
        This is a common temporal IV setup.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "Z", "temporal_status": "time_varying"},
                {"name": "X", "temporal_status": "time_varying"},
                {"name": "Y", "temporal_status": "time_varying", "is_outcome": True},
                {"name": "U", "temporal_status": "time_varying"},
            ],
            edges=[
                {"cause": "Z", "effect": "X", "lagged": True},  # Z_{t-1} -> X_t
                {"cause": "X", "effect": "Y", "lagged": False},  # X_t -> Y_t
                {"cause": "U", "effect": "X", "lagged": False},
                {"cause": "U", "effect": "Y", "lagged": False},
            ],
        )
        measurement_model = make_measurement_model(["Z", "X", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        # Z_{t-1} should serve as instrument for X_t -> Y_t
        # (assuming Z_{t-1} doesn't directly affect Y_t)
        assert_identifiable(result, "X", "Lagged Z as temporal instrument")


# =============================================================================
# 17. MULTIPLE OVERLAPPING CONFOUNDERS
# =============================================================================


class TestOverlappingConfounders:
    """Test scenarios with multiple confounders affecting overlapping sets."""

    def test_partial_confounding_coverage(self):
        """Each confounder affects different pairs.

        U1 -> {X, M}
        U2 -> {M, Y}
        X -> M -> Y

        U1 confounds X-M, U2 confounds M-Y.
        Front-door fails because M is confounded with both X and Y.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "X"},
                {"name": "M"},
                {"name": "Y", "is_outcome": True},
                {"name": "U1"},
                {"name": "U2"},
            ],
            edges=[
                {"cause": "X", "effect": "M"},
                {"cause": "M", "effect": "Y"},
                {"cause": "U1", "effect": "X"},
                {"cause": "U1", "effect": "M"},
                {"cause": "U2", "effect": "M"},
                {"cause": "U2", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["X", "M", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        # This is the same as double_bow / confounded_chain
        assert_not_identifiable(result, "X", "Partial confounding blocks front-door")

    def test_triangle_confounding(self):
        """Triangular confounding pattern.

        U1 -> {X, Y}
        U2 -> {X, Z}
        U3 -> {Y, Z}
        X -> Y, Y -> Z

        Every pair of observed variables shares an unobserved confounder.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "X"},
                {"name": "Y", "is_outcome": True},
                {"name": "Z"},
                {"name": "U1"},
                {"name": "U2"},
                {"name": "U3"},
            ],
            edges=[
                {"cause": "X", "effect": "Y"},
                {"cause": "Y", "effect": "Z"},
                {"cause": "U1", "effect": "X"},
                {"cause": "U1", "effect": "Y"},
                {"cause": "U2", "effect": "X"},
                {"cause": "U2", "effect": "Z"},
                {"cause": "U3", "effect": "Y"},
                {"cause": "U3", "effect": "Z"},
            ],
        )
        measurement_model = make_measurement_model(["X", "Y", "Z"])

        result = check_identifiability(latent_model, measurement_model)

        # U1 directly confounds X-Y
        assert_not_identifiable(result, "X", "U1 confounds X-Y directly")
        assert_blocked_by(result, "X", "U1")

    def test_four_node_complete_confounding(self):
        """Four observed nodes with pairwise unobserved confounding.

        A -> B -> C -> D (outcome)
        U_AB -> {A, B}
        U_BC -> {B, C}
        U_CD -> {C, D}
        U_AD -> {A, D}

        Every adjacent pair and the first-last pair are confounded.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "A"},
                {"name": "B"},
                {"name": "C"},
                {"name": "D", "is_outcome": True},
                {"name": "U_AB"},
                {"name": "U_BC"},
                {"name": "U_CD"},
                {"name": "U_AD"},
            ],
            edges=[
                {"cause": "A", "effect": "B"},
                {"cause": "B", "effect": "C"},
                {"cause": "C", "effect": "D"},
                {"cause": "U_AB", "effect": "A"},
                {"cause": "U_AB", "effect": "B"},
                {"cause": "U_BC", "effect": "B"},
                {"cause": "U_BC", "effect": "C"},
                {"cause": "U_CD", "effect": "C"},
                {"cause": "U_CD", "effect": "D"},
                {"cause": "U_AD", "effect": "A"},
                {"cause": "U_AD", "effect": "D"},
            ],
        )
        measurement_model = make_measurement_model(["A", "B", "C", "D"])

        result = check_identifiability(latent_model, measurement_model)

        # A -> D: U_AD confounds directly, no way to identify
        assert_not_identifiable(result, "A", "Dense confounding blocks all paths")

    def test_selective_confounding_some_identifiable(self):
        """Selective confounding where some effects are identifiable.

        A -> B -> C -> D (outcome)
        U -> {B, C}

        A -> D is identifiable (no confounding on A)
        B -> D is confounded at B-C link
        C -> D is confounded
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "A"},
                {"name": "B"},
                {"name": "C"},
                {"name": "D", "is_outcome": True},
                {"name": "U"},
            ],
            edges=[
                {"cause": "A", "effect": "B"},
                {"cause": "B", "effect": "C"},
                {"cause": "C", "effect": "D"},
                {"cause": "U", "effect": "B"},
                {"cause": "U", "effect": "C"},
            ],
        )
        measurement_model = make_measurement_model(["A", "B", "C", "D"])

        result = check_identifiability(latent_model, measurement_model)

        # A is not directly confounded with D
        # But B-C are confounded, and B -> C -> D, also A -> B -> C
        # So the question is: does U on B-C block A -> D?
        # The backdoor from A is A <- ... but there's no backdoor from A
        # (nothing causes A). So A -> D should be identifiable by front-door through B-C-D
        # But wait - is B-C a valid front-door? U confounds B-C...
        # Actually: A -> B -> C -> D. U -> B, U -> C.
        # For A -> D: backdoor? Nothing causes A. So no backdoor.
        # A should be identifiable.
        assert_identifiable(result, "A", "A has no backdoor paths")

        # For B -> D: U confounds B with C, and C is on the path to D
        # But the question is B -> D, not B -> C -> D
        # Actually B -> C -> D is the only path. U -> B, U -> C.
        # Backdoor from B: B <- U -> C -> D. This is a backdoor!
        # So B -> D is confounded.

        # For C -> D: U -> C, but U doesn't affect D directly
        # No backdoor C <- ... -> D unless U -> D, but U -> C only.
        # So C -> D should be identifiable!
        assert_identifiable(result, "C", "C -> D has no backdoor")


# =============================================================================
# 18. MEDIATOR-COLLIDER DUALITY AND COMPLEX PATH STRUCTURES
# =============================================================================


class TestMediatorColliderDuality:
    """Test nodes that serve as both mediators and colliders."""

    def test_node_is_mediator_and_collider(self):
        """Node M is mediator on one path, collider on another.

        A -> M -> Y
             ^
        B ---+
        (B -> M makes M a collider for A-B, but M mediates A -> Y)

        Since we care about A -> Y, and M is on the causal path,
        this should be identifiable if B doesn't confound A-Y.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "A"},
                {"name": "B"},
                {"name": "M"},
                {"name": "Y", "is_outcome": True},
            ],
            edges=[
                {"cause": "A", "effect": "M"},
                {"cause": "B", "effect": "M"},
                {"cause": "M", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["A", "B", "M", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        # A -> M -> Y is the causal path
        # M being a collider for A-B doesn't create confounding for A -> Y
        # (B doesn't affect Y)
        assert_identifiable(result, "A", "Collider property doesn't block causal path")
        assert_identifiable(result, "B", "B -> M -> Y is also identifiable")

    def test_mediator_collider_with_confounding(self):
        """Mediator-collider with unobserved confounding.

        A -> M -> Y
             ^   ^
        U ---+---+

        M is a collider for A-U and mediator for A -> Y.
        U confounds M-Y, breaking identification.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "A"},
                {"name": "M"},
                {"name": "Y", "is_outcome": True},
                {"name": "U"},
            ],
            edges=[
                {"cause": "A", "effect": "M"},
                {"cause": "U", "effect": "M"},
                {"cause": "U", "effect": "Y"},
                {"cause": "M", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["A", "M", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        # A -> M -> Y, with U -> M, U -> Y (M-Y confounded)
        #
        # For A -> Y:
        # - A -> M is unconfounded (U doesn't affect A)
        # - The temporal structure enables identification via y0
        #
        # For M -> Y:
        # - M <- U -> Y creates backdoor confounding
        # - But A serves as an instrument for M -> Y under linearity:
        #   * Relevance: A -> M
        #   * Exclusion: A has no path to Y except through M
        #   * Exogeneity: U doesn't affect A
        #
        # Both effects are identifiable via temporal/IV strategies
        assert_identifiable(result, "A", "A -> M -> Y identifiable via temporal structure")
        assert_identifiable(result, "M", "M -> Y identifiable (A as IV or temporal)")


# =============================================================================
# 19. NESTED AND HIERARCHICAL STRUCTURES
# =============================================================================


class TestNestedStructures:
    """Test nested and hierarchical graph structures."""

    def test_nested_front_door(self):
        """Nested front-door: front-door within front-door.

        U1 -> {X, M1}
        U2 -> {M1, M2}
        U3 -> {M2, Y}
        X -> M1 -> M2 -> Y

        Each link is confounded, but no single confounder spans X to Y.
        This tests whether y0 can find a complex identification strategy.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "X"},
                {"name": "M1"},
                {"name": "M2"},
                {"name": "Y", "is_outcome": True},
                {"name": "U1"},
                {"name": "U2"},
                {"name": "U3"},
            ],
            edges=[
                {"cause": "X", "effect": "M1"},
                {"cause": "M1", "effect": "M2"},
                {"cause": "M2", "effect": "Y"},
                {"cause": "U1", "effect": "X"},
                {"cause": "U1", "effect": "M1"},
                {"cause": "U2", "effect": "M1"},
                {"cause": "U2", "effect": "M2"},
                {"cause": "U3", "effect": "M2"},
                {"cause": "U3", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["X", "M1", "M2", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        # This is a chain of overlapping confounders
        # Similar to the "double bow" - each step is confounded
        # No identification strategy works
        assert_not_identifiable(result, "X", "Nested confounding blocks all strategies")

    def test_hierarchical_treatment_structure(self):
        """Hierarchical treatment: X -> A -> B with both affecting Y.

        X -> A -> Y
        |    |   ^
        +--> B --+

        X affects Y through A and through B (which also depends on A).
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "X"},
                {"name": "A"},
                {"name": "B"},
                {"name": "Y", "is_outcome": True},
            ],
            edges=[
                {"cause": "X", "effect": "A"},
                {"cause": "X", "effect": "B"},
                {"cause": "A", "effect": "B"},
                {"cause": "A", "effect": "Y"},
                {"cause": "B", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["X", "A", "B", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        # All observed, no confounding
        assert_identifiable(result, "X", "Hierarchical treatment, all observed")
        assert_identifiable(result, "A", "A -> Y identifiable")
        assert_identifiable(result, "B", "B -> Y identifiable")


# =============================================================================
# 20. SPARSE VS DENSE MEASUREMENT COVERAGE
# =============================================================================


class TestMeasurementCoverage:
    """Test how measurement coverage affects identification."""

    def test_same_structure_different_coverage_identifiable(self):
        """Same latent structure, with good measurement coverage.

        Z -> X -> M -> Y
             ^       ^
             U ------+

        With Z, X, M, Y observed: Z is instrument, identification works.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "Z", "role": "exogenous"},
                {"name": "X"},
                {"name": "M"},
                {"name": "Y", "is_outcome": True},
                {"name": "U"},
            ],
            edges=[
                {"cause": "Z", "effect": "X"},
                {"cause": "X", "effect": "M"},
                {"cause": "M", "effect": "Y"},
                {"cause": "U", "effect": "X"},
                {"cause": "U", "effect": "Y"},
            ],
        )

        # Good coverage: Z observed, enables IV
        measurement_model = make_measurement_model(["Z", "X", "M", "Y"])
        result = check_identifiability(latent_model, measurement_model)
        assert_identifiable(result, "X", "Good coverage: Z enables IV")

    def test_same_structure_different_coverage_non_identifiable(self):
        """Same latent structure, with poor measurement coverage.

        Z -> X -> M -> Y
             ^       ^
             U ------+

        Without Z observed: No instrument, bow-like confounding.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "Z", "role": "exogenous"},
                {"name": "X"},
                {"name": "M"},
                {"name": "Y", "is_outcome": True},
                {"name": "U"},
            ],
            edges=[
                {"cause": "Z", "effect": "X"},
                {"cause": "X", "effect": "M"},
                {"cause": "M", "effect": "Y"},
                {"cause": "U", "effect": "X"},
                {"cause": "U", "effect": "Y"},
            ],
        )

        # Poor coverage: Z unobserved, no IV available
        measurement_model = make_measurement_model(["X", "M", "Y"])
        result = check_identifiability(latent_model, measurement_model)

        # Without Z, can we use front-door through M?
        # X -> M -> Y with U -> X, U -> Y
        # Front-door: M intercepts all paths X -> Y? No! U -> Y is direct.
        # Wait, X -> M -> Y is the only directed path from X to Y
        # U -> X creates backdoor: X <- U -> Y
        # Front-door condition 1: M intercepts all directed X -> Y ✓
        # Front-door condition 2: No unblocked backdoor X <- ... -> M
        #   X <- U -> Y, M is not on this path. Check X <- ... -> M.
        #   U -> X, but U -> M? No. So no backdoor X <- U -> M.
        # Front-door condition 3: All backdoor M <- ... -> Y blocked by X
        #   Backdoor from M: nothing causes M except X. So no backdoor.
        # Front-door applies! This should be identifiable!
        assert_identifiable(result, "X", "Front-door through M works")

    def test_minimal_coverage_for_identification(self):
        """Test minimum measurement needed for identification.

        Full structure:
        Z -> X -> M1 -> M2 -> Y
             ^              ^
             U -------------+

        Question: Which measurements are necessary?
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "Z", "role": "exogenous"},
                {"name": "X"},
                {"name": "M1"},
                {"name": "M2"},
                {"name": "Y", "is_outcome": True},
                {"name": "U"},
            ],
            edges=[
                {"cause": "Z", "effect": "X"},
                {"cause": "X", "effect": "M1"},
                {"cause": "M1", "effect": "M2"},
                {"cause": "M2", "effect": "Y"},
                {"cause": "U", "effect": "X"},
                {"cause": "U", "effect": "Y"},
            ],
        )

        # Minimal: just X and Y
        # In a cross-sectional setting, this would be a bow graph (non-identifiable)
        # But with temporal unrolling, y0 finds identification via lagged adjustment!
        # This is because X_{t-1} and Y_{t-1} provide additional information
        measurement_model = make_measurement_model(["X", "Y"])
        result = check_identifiability(latent_model, measurement_model)
        # Surprisingly identifiable via temporal structure
        assert_identifiable(result, "X", "Temporal structure enables identification")

        # Add Z: IV enabled
        measurement_model = make_measurement_model(["Z", "X", "Y"])
        result = check_identifiability(latent_model, measurement_model)
        assert_identifiable(result, "X", "Z enables IV")

        # Add M1 instead of Z: front-door enabled
        measurement_model = make_measurement_model(["X", "M1", "Y"])
        result = check_identifiability(latent_model, measurement_model)
        assert_identifiable(result, "X", "M1 enables front-door")


# =============================================================================
# 21. REGRESSION DISCONTINUITY AND SPECIAL IV STRUCTURES
# =============================================================================


class TestSpecialIVStructures:
    """Test special instrumental variable scenarios."""

    def test_regression_discontinuity_like(self):
        """RD-like structure: threshold creates instrument.

        Running variable R affects treatment D via threshold.
        R -> D -> Y
             ^   ^
             U --+

        R is exogenous and only affects Y through D (exclusion).
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "R", "role": "exogenous"},  # Running variable
                {"name": "D"},  # Treatment (threshold-based)
                {"name": "Y", "is_outcome": True},
                {"name": "U"},
            ],
            edges=[
                {"cause": "R", "effect": "D"},
                {"cause": "D", "effect": "Y"},
                {"cause": "U", "effect": "D"},
                {"cause": "U", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["R", "D", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        assert_identifiable(result, "D", "R is valid instrument for D")
        assert "IV(R)" in get_estimand(result, "D")

    @pytest.mark.skip(reason="DID requires parallel trends assumption - not graph-identifiable")
    def test_diff_in_diff_like(self):
        """Diff-in-diff-like structure: group and time as instruments.

        Group -> D <- Time
        Group -> Y <- Time
        D -> Y
        U -> D, U -> Y

        This is actually confounded... Both Group and Time affect Y directly.
        Not a clean DID setup for our purposes.

        DID identification relies on the parallel trends assumption which is
        not testable from the graph structure alone. It's a substantive
        assumption about counterfactual trends, not a graphical criterion.
        """
        # DID is fundamentally different from graphical identification
        # It requires assuming parallel trends in potential outcomes
        # which cannot be encoded in a causal DAG
        pytest.skip("DID requires parallel trends assumption")

    def test_mendelian_randomization(self):
        """Mendelian randomization: genetic variant as instrument.

        G -> X -> Y
             ^   ^
             U --+

        G (genetic variant) is exogenous and only affects Y through X.
        Classic MR setup.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "G", "role": "exogenous"},  # Genetic variant
                {"name": "X"},  # Exposure
                {"name": "Y", "is_outcome": True},  # Outcome
                {"name": "U"},  # Confounders (lifestyle, environment)
            ],
            edges=[
                {"cause": "G", "effect": "X"},
                {"cause": "X", "effect": "Y"},
                {"cause": "U", "effect": "X"},
                {"cause": "U", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["G", "X", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        assert_identifiable(result, "X", "G is valid MR instrument")
        assert "IV(G)" in get_estimand(result, "X")


# =============================================================================
# 22. EDGE CASES IN TEMPORAL UNROLLING
# =============================================================================


class TestTemporalUnrollingEdgeCases:
    """Test edge cases specific to 2-timestep unrolling."""

    def test_only_lagged_effects_no_contemporaneous(self):
        """Graph with only lagged effects.

        X_{t-1} -> Y_t (no contemporaneous X_t -> Y_t)
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "X", "temporal_status": "time_varying"},
                {"name": "Y", "temporal_status": "time_varying", "is_outcome": True},
            ],
            edges=[
                {"cause": "X", "effect": "Y", "lagged": True},
            ],
        )
        measurement_model = make_measurement_model(["X", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        # Pure lagged effect, no confounding
        assert_identifiable(result, "X", "Pure lagged effect identifiable")

    def test_time_invariant_only(self):
        """All constructs are time-invariant (cross-sectional).

        This tests that time-invariant handling doesn't break.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "X", "temporal_status": "time_invariant"},
                {"name": "Y", "temporal_status": "time_invariant", "is_outcome": True},
            ],
            edges=[
                {"cause": "X", "effect": "Y"},
            ],
        )
        measurement_model = make_measurement_model(["X", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        assert_identifiable(result, "X", "Cross-sectional, no confounding")

    def test_mixed_time_status_complex_iv_available(self):
        """Complex mix of time-varying and time-invariant with IV.

        Trait (invariant, observed) -> X_t -> Y_t
        State (varying, unobserved) -> X_t, State -> Y_t

        Although State confounds X-Y, Trait serves as an instrument:
        - Trait -> X (relevance)
        - Trait has no path to Y except through X (exclusion)
        - Trait is not affected by State (exogeneity - Trait is time-invariant)
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "Trait", "temporal_status": "time_invariant", "role": "exogenous"},
                {"name": "State", "temporal_status": "time_varying"},
                {"name": "X", "temporal_status": "time_varying"},
                {"name": "Y", "temporal_status": "time_varying", "is_outcome": True},
            ],
            edges=[
                {"cause": "Trait", "effect": "X"},
                {"cause": "State", "effect": "X", "lagged": False},
                {"cause": "State", "effect": "Y", "lagged": False},
                {"cause": "X", "effect": "Y", "lagged": False},
            ],
        )
        measurement_model = make_measurement_model(["Trait", "X", "Y"])  # State unobserved

        result = check_identifiability(latent_model, measurement_model)

        # Trait serves as instrument for X -> Y
        assert_identifiable(result, "X", "Trait is valid instrument")
        assert "IV(Trait)" in get_estimand(result, "X")
        # Trait -> X -> Y is also identifiable
        assert_identifiable(result, "Trait", "Trait effect on Y via X")

    def test_mixed_time_status_no_instrument(self):
        """Mixed time status WITHOUT instrument available.

        State (varying, unobserved) -> X_t, State -> Y_t
        X_t -> Y_t

        No instrument available - pure bow graph.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "State", "temporal_status": "time_varying"},
                {"name": "X", "temporal_status": "time_varying"},
                {"name": "Y", "temporal_status": "time_varying", "is_outcome": True},
            ],
            edges=[
                {"cause": "State", "effect": "X", "lagged": False},
                {"cause": "State", "effect": "Y", "lagged": False},
                {"cause": "X", "effect": "Y", "lagged": False},
            ],
        )
        measurement_model = make_measurement_model(["X", "Y"])  # State unobserved

        result = check_identifiability(latent_model, measurement_model)

        # No instrument, contemporaneous confounding - NOT identifiable
        assert_not_identifiable(result, "X", "State confounds X-Y, no instrument")
        assert_blocked_by(result, "X", "State")

    def test_lagged_confounding_with_lagged_treatment(self):
        """Lagged confounding with lagged treatment effect.

        U_{t-1} -> X_t, U_{t-1} -> Y_t
        X_{t-1} -> Y_t

        The treatment effect is lagged, confounding is also lagged.
        """
        latent_model = make_latent_model(
            constructs=[
                {"name": "X", "temporal_status": "time_varying"},
                {"name": "Y", "temporal_status": "time_varying", "is_outcome": True},
                {"name": "U", "temporal_status": "time_varying"},
            ],
            edges=[
                {"cause": "X", "effect": "Y", "lagged": True},  # X_{t-1} -> Y_t
                {"cause": "U", "effect": "X", "lagged": True},  # U_{t-1} -> X_t
                {"cause": "U", "effect": "Y", "lagged": True},  # U_{t-1} -> Y_t
            ],
        )
        measurement_model = make_measurement_model(["X", "Y"])

        result = check_identifiability(latent_model, measurement_model)

        # In unrolled graph:
        # X_{t-1} -> Y_t (treatment effect)
        # U_{t-1} -> X_t (confounds X_t, not X_{t-1})
        # U_{t-1} -> Y_t (confounds Y_t)
        # The question is: does U_{t-1} confound X_{t-1} -> Y_t?
        # U_{t-1} -> X_t, and X_{t-1} -> X_t (AR1), so U_{t-1} affects X_t not X_{t-1}
        # But we care about X_{t-1} -> Y_t. U_{t-1} -> Y_t directly.
        # Is there a backdoor X_{t-1} <- ... -> Y_t?
        # U_{t-2} -> X_{t-1} and U_{t-2} -> Y_{t-1}, but U_{t-2} not in 2-timestep graph
        # In our 2-timestep: X_{t-1} has incoming only from X_{t-2} (not modeled) and...
        # Actually, the contemporaneous edges are mirrored: U_{t-2} -> X_{t-1}, U_{t-2} -> Y_{t-1}
        # Wait, let me check the unrolling logic...
        # Lagged edges: U_{t-1} -> X_t, U_{t-1} -> Y_t (not U_{t-2} -> X_{t-1})
        # So X_{t-1} is not affected by U in the 2-timestep model
        # Therefore X_{t-1} -> Y_t should be identifiable!
        assert_identifiable(result, "X", "Lagged treatment avoids lagged confounding")


# =============================================================================
# MAIN
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
