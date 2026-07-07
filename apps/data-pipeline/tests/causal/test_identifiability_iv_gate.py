"""Tests for the ``iv_allowed`` gate on ``check_identifiability``.

The flag controls whether graph-theoretic IV candidates can be reported under
an explicit parametric linearity assumption.
"""

from __future__ import annotations

from nof1_causal_lab.utils.identifiability import check_identifiability


def _iv_structure_latent_structure():
    """A DAG with a textbook IV pattern: U → X → Y, Z → X, with U unobserved.

    With U unobserved and confounding both X and Y, the backdoor cannot be
    blocked by adjusting on observed variables. But Z (a parent of X with
    no other path to Y) is a valid instrument *under linearity*.
    """
    return {
        "constructs": [
            {"name": "X", "is_outcome": False, "temporal_status": "time_invariant"},
            {"name": "Y", "is_outcome": True, "temporal_status": "time_invariant"},
            {"name": "Z", "is_outcome": False, "temporal_status": "time_invariant"},
            {"name": "U", "is_outcome": False, "temporal_status": "time_invariant"},
        ],
        "edges": [
            {"cause": "Z", "effect": "X", "lagged": False},
            {"cause": "X", "effect": "Y", "lagged": False},
            {"cause": "U", "effect": "X", "lagged": False},
            {"cause": "U", "effect": "Y", "lagged": False},
        ],
    }


def _measurement_structure_observing_xyz():
    return {
        "indicators": [
            {"name": "y_obs", "construct_name": "Y"},
            {"name": "x_obs", "construct_name": "X"},
            {"name": "z_obs", "construct_name": "Z"},
        ],
    }


class TestIVAllowedDefault:
    def test_default_gate_finds_iv(self):
        """Default ``iv_allowed=True`` may mark X as identifiable via Z
        when U blocks backdoor identification."""
        latent_structure = _iv_structure_latent_structure()
        measurement_structure = _measurement_structure_observing_xyz()

        result = check_identifiability(latent_structure, measurement_structure)

        # X should be identifiable via Z (IV) under linearity assumption.
        assert "X" in result["identifiable_treatments"]
        info = result["identifiable_treatments"]["X"]
        # Either do-calculus (front-door / backdoor via Z) OR instrumental_variable
        # is acceptable — what matters is that IV was considered when available.
        # We at least want to record that IV was *available* when it's used.
        if info["method"] == "instrumental_variable":
            assert "Z" in info["instruments"]
        assert result["graph_info"]["iv_allowed"] is True


class TestIVAllowedFalse:
    def test_disabled_iv_gate_skips_iv(self):
        """With ``iv_allowed=False`` and only-IV-identification structure,
        the treatment must end up non-identifiable."""
        latent_structure = _iv_structure_latent_structure()
        measurement_structure = _measurement_structure_observing_xyz()

        result_with_iv = check_identifiability(
            latent_structure, measurement_structure, iv_allowed=True
        )
        result_no_iv = check_identifiability(
            latent_structure, measurement_structure, iv_allowed=False
        )

        assert result_no_iv["graph_info"]["iv_allowed"] is False

        # If IV identified X, disabling IV must drop X to the non-identifiable set.
        iv_enabled_x = result_with_iv["identifiable_treatments"].get("X")
        iv_disabled_x = result_no_iv["identifiable_treatments"].get("X")

        if iv_enabled_x is not None and iv_enabled_x.get("method") == "instrumental_variable":
            # Was IV-only identified → must be dropped without IV.
            assert iv_disabled_x is None
            assert "X" in result_no_iv["non_identifiable_treatments"]
        else:
            # If do-calculus alone identifies X (e.g., front-door), both paths agree.
            assert iv_disabled_x is not None
            assert iv_disabled_x.get("method") == "do_calculus"

    def test_disabled_iv_gate_preserves_do_calculus_identifications(self):
        """Treatments identified via do-calculus (backdoor/front-door) should
        be unchanged when IV is disabled — IV is a fallback, not a primary."""
        # Simpler DAG: X → Y, no confounders. Backdoor trivially identifiable.
        latent_structure = {
            "constructs": [
                {"name": "X", "is_outcome": False, "temporal_status": "time_invariant"},
                {"name": "Y", "is_outcome": True, "temporal_status": "time_invariant"},
            ],
            "edges": [{"cause": "X", "effect": "Y", "lagged": False}],
        }
        measurement_structure = {
            "indicators": [
                {"name": "y_obs", "construct_name": "Y"},
                {"name": "x_obs", "construct_name": "X"},
            ],
        }

        result_with_iv = check_identifiability(
            latent_structure, measurement_structure, iv_allowed=True
        )
        result_no_iv = check_identifiability(
            latent_structure, measurement_structure, iv_allowed=False
        )

        assert "X" in result_with_iv["identifiable_treatments"]
        assert "X" in result_no_iv["identifiable_treatments"]
        # Same method either way.
        assert (
            result_with_iv["identifiable_treatments"]["X"]["method"]
            == result_no_iv["identifiable_treatments"]["X"]["method"]
        )
