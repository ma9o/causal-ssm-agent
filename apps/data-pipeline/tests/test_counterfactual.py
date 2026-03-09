"""Tests for do-operator and causal intervention functions.

Covers: steady_state, do, treatment_effect, forward_simulate_intervention,
        compute_interventions, _summarize_trajectory.
"""

import jax.numpy as jnp

from causal_ssm_agent.models.ssm.counterfactual import (
    _summarize_trajectory,
    compute_interventions,
    do,
    forward_simulate_intervention,
    steady_state,
    treatment_effect,
)

# =============================================================================
# steady_state
# =============================================================================


class TestSteadyState:
    def test_identity_drift(self):
        """With A=-I, c=[1,2]: eta* = -(-I)^-1 [1,2] = [1,2]."""
        A = -jnp.eye(2)
        c = jnp.array([1.0, 2.0])
        ss = steady_state(A, c)
        assert jnp.allclose(ss, jnp.array([1.0, 2.0]), atol=1e-6)

    def test_scaled_drift(self):
        """With A=-2I, c=[4,6]: eta* = [2,3]."""
        A = -2.0 * jnp.eye(2)
        c = jnp.array([4.0, 6.0])
        ss = steady_state(A, c)
        assert jnp.allclose(ss, jnp.array([2.0, 3.0]), atol=1e-6)

    def test_zero_intercept(self):
        """Zero intercept gives zero steady state."""
        A = -jnp.eye(3)
        c = jnp.zeros(3)
        ss = steady_state(A, c)
        assert jnp.allclose(ss, jnp.zeros(3), atol=1e-10)

    def test_satisfies_equilibrium(self):
        """Steady state should satisfy A*eta* + c = 0."""
        A = jnp.array([[-2.0, 0.5], [0.3, -1.5]])
        c = jnp.array([1.0, -0.5])
        ss = steady_state(A, c)
        residual = A @ ss + c
        assert jnp.allclose(residual, 0.0, atol=1e-5)

    def test_1d_system(self):
        """Scalar system: A=-3, c=6 -> eta*=2."""
        A = jnp.array([[-3.0]])
        c = jnp.array([6.0])
        ss = steady_state(A, c)
        assert jnp.isclose(ss[0], 2.0, atol=1e-6)


# =============================================================================
# do (intervention)
# =============================================================================


class TestDo:
    def test_clamped_value(self):
        """Intervened variable should equal the do-value."""
        A = -jnp.eye(3)
        c = jnp.array([1.0, 2.0, 3.0])
        result = do(A, c, do_idx=1, do_value=10.0)
        assert jnp.isclose(result[1], 10.0, atol=1e-5)

    def test_independent_variables_unchanged(self):
        """With diagonal A, non-intervened variables keep their steady state."""
        A = -jnp.eye(3)
        c = jnp.array([1.0, 2.0, 3.0])
        result = do(A, c, do_idx=1, do_value=10.0)
        # Variables 0 and 2 are independent (diagonal A), so unchanged
        assert jnp.isclose(result[0], 1.0, atol=1e-5)
        assert jnp.isclose(result[2], 3.0, atol=1e-5)

    def test_downstream_effect(self):
        """Intervention should propagate through causal structure."""
        # A with off-diagonal: variable 0 affects variable 1
        A = jnp.array([[-1.0, 0.0], [0.5, -1.0]])
        c = jnp.array([2.0, 1.0])
        baseline = steady_state(A, c)
        result = do(A, c, do_idx=0, do_value=5.0)
        assert jnp.isclose(result[0], 5.0, atol=1e-5)
        # Variable 1 should differ from baseline due to off-diagonal coupling
        assert not jnp.isclose(result[1], baseline[1], atol=0.1)

    def test_matches_steady_state_at_baseline(self):
        """do(eta_j = eta*_j) should give back the original steady state."""
        A = jnp.array([[-1.0, 0.3], [0.2, -1.5]])
        c = jnp.array([1.0, 0.5])
        baseline = steady_state(A, c)
        result = do(A, c, do_idx=0, do_value=float(baseline[0]))
        assert jnp.allclose(result, baseline, atol=1e-4)


# =============================================================================
# treatment_effect
# =============================================================================


class TestTreatmentEffect:
    def test_no_causal_path_zero_effect(self):
        """No causal path from treatment to outcome -> zero effect."""
        A = -jnp.eye(3)  # Diagonal = no cross-variable influence
        c = jnp.array([1.0, 2.0, 3.0])
        effect = treatment_effect(A, c, treat_idx=0, outcome_idx=2)
        assert jnp.isclose(effect, 0.0, atol=1e-5)

    def test_direct_effect(self):
        """Direct causal path should produce nonzero effect."""
        A = jnp.array([[-1.0, 0.0], [0.5, -1.0]])
        c = jnp.array([1.0, 0.5])
        effect = treatment_effect(A, c, treat_idx=0, outcome_idx=1)
        assert not jnp.isclose(effect, 0.0, atol=0.01)

    def test_effect_scales_with_shift(self):
        """Doubling shift_size should double the effect (linear system)."""
        A = jnp.array([[-1.0, 0.0], [0.5, -1.0]])
        c = jnp.array([1.0, 0.5])
        e1 = treatment_effect(A, c, treat_idx=0, outcome_idx=1, shift_size=1.0)
        e2 = treatment_effect(A, c, treat_idx=0, outcome_idx=1, shift_size=2.0)
        assert jnp.isclose(e2, 2.0 * e1, atol=1e-4)

    def test_zero_shift_zero_effect(self):
        """Zero shift should give zero effect."""
        A = jnp.array([[-1.0, 0.0], [0.5, -1.0]])
        c = jnp.array([1.0, 0.5])
        effect = treatment_effect(A, c, treat_idx=0, outcome_idx=1, shift_size=0.0)
        assert jnp.isclose(effect, 0.0, atol=1e-6)

    def test_sign_matches_coupling(self):
        """Positive coupling coefficient should give positive effect."""
        A = jnp.array([[-1.0, 0.0], [0.8, -1.0]])
        c = jnp.ones(2)
        effect = treatment_effect(A, c, treat_idx=0, outcome_idx=1, shift_size=1.0)
        assert effect > 0


# =============================================================================
# _summarize_trajectory
# =============================================================================


class TestSummarizeTrajectory:
    def test_keys_present(self):
        """Should return all expected keys."""
        traj = jnp.linspace(0, 1, 100)
        result = _summarize_trajectory(traj, dt=0.1)
        expected_keys = {"effect_1d", "effect_7d", "effect_30d", "peak_effect", "time_to_peak_days"}
        assert set(result.keys()) == expected_keys

    def test_monotonic_increasing_trajectory(self):
        """For monotonically increasing trajectory, 30d > 7d > 1d."""
        traj = jnp.linspace(0, 3, 300)
        result = _summarize_trajectory(traj, dt=0.1)
        assert result["effect_30d"] > result["effect_7d"]
        assert result["effect_7d"] > result["effect_1d"]

    def test_peak_effect_at_end_for_increasing(self):
        """Peak should be at end for monotonically increasing trajectory."""
        traj = jnp.linspace(0, 5, 50)
        result = _summarize_trajectory(traj, dt=1.0)
        assert result["peak_effect"] == float(traj[-1])

    def test_constant_trajectory(self):
        """Constant trajectory should have same value everywhere."""
        traj = jnp.ones(100) * 2.5
        result = _summarize_trajectory(traj, dt=0.1)
        assert abs(result["effect_1d"] - 2.5) < 1e-5
        assert abs(result["effect_7d"] - 2.5) < 1e-5
        assert abs(result["peak_effect"] - 2.5) < 1e-5

    def test_daily_dt(self):
        """With dt=1.0, 1d effect should be the first element."""
        traj = jnp.array([0.5, 1.0, 1.5, 2.0, 2.5])
        result = _summarize_trajectory(traj, dt=1.0)
        assert abs(result["effect_1d"] - 0.5) < 1e-5

    def test_time_to_peak_positive(self):
        """Time to peak should always be positive."""
        traj = jnp.array([3.0, 2.0, 1.0, 0.5])
        result = _summarize_trajectory(traj, dt=0.5)
        assert result["time_to_peak_days"] > 0


# =============================================================================
# forward_simulate_intervention
# =============================================================================


class TestForwardSimulateIntervention:
    def test_returns_correct_shape(self):
        """Should return trajectory of length horizon_steps."""
        A = jnp.array([[-1.0, 0.0], [0.5, -1.0]])
        c = jnp.array([1.0, 0.5])
        traj = forward_simulate_intervention(
            A,
            c,
            treat_idx=0,
            outcome_idx=1,
            shift_size=1.0,
            dt=0.1,
            horizon_steps=50,
        )
        assert traj.shape == (50,)

    def test_no_causal_path_flat_trajectory(self):
        """No coupling => trajectory should stay near zero."""
        A = -jnp.eye(2)
        c = jnp.array([1.0, 2.0])
        traj = forward_simulate_intervention(
            A,
            c,
            treat_idx=0,
            outcome_idx=1,
            shift_size=1.0,
            dt=0.1,
            horizon_steps=100,
        )
        assert jnp.allclose(traj, 0.0, atol=1e-3)

    def test_positive_coupling_produces_positive_trajectory(self):
        """Positive A[1,0] coupling should produce positive outcome effects."""
        A = jnp.array([[-1.0, 0.0], [0.5, -1.0]])
        c = jnp.array([1.0, 0.5])
        traj = forward_simulate_intervention(
            A,
            c,
            treat_idx=0,
            outcome_idx=1,
            shift_size=1.0,
            dt=0.1,
            horizon_steps=200,
        )
        # Later in the trajectory, effect should be positive
        assert float(traj[-1]) > 0

    def test_zero_shift_zero_trajectory(self):
        """Zero shift should give near-zero trajectory."""
        A = jnp.array([[-1.0, 0.0], [0.5, -1.0]])
        c = jnp.array([1.0, 0.5])
        traj = forward_simulate_intervention(
            A,
            c,
            treat_idx=0,
            outcome_idx=1,
            shift_size=0.0,
            dt=0.1,
            horizon_steps=50,
        )
        assert jnp.allclose(traj, 0.0, atol=1e-4)


# =============================================================================
# compute_interventions
# =============================================================================


class TestComputeInterventions:
    def _make_samples(self, n_draws=10, n_latent=3):
        """Create minimal posterior samples for testing."""
        drift = jnp.broadcast_to(-jnp.eye(n_latent), (n_draws, n_latent, n_latent))
        cint = jnp.broadcast_to(jnp.ones(n_latent), (n_draws, n_latent))
        return {"drift": drift, "cint": cint}

    def test_basic_output_structure(self):
        """Should return a list of dicts with required keys."""
        samples = self._make_samples()
        results = compute_interventions(
            samples,
            treatments=["A", "B"],
            outcome="C",
            latent_names=["A", "B", "C"],
        )
        assert len(results) == 2
        for r in results:
            assert "treatment" in r
            assert "effect_size" in r
            assert "identifiable" in r

    def test_outcome_not_in_latent_names(self):
        """Missing outcome returns skeleton entries."""
        samples = self._make_samples()
        results = compute_interventions(
            samples,
            treatments=["A"],
            outcome="MISSING",
            latent_names=["A", "B", "C"],
        )
        assert len(results) == 1
        assert results[0]["effect_size"] is None

    def test_treatment_not_in_latent_names(self):
        """Treatment not in latent names should produce a warning entry."""
        samples = self._make_samples()
        results = compute_interventions(
            samples,
            treatments=["UNKNOWN"],
            outcome="C",
            latent_names=["A", "B", "C"],
        )
        assert results[0]["effect_size"] is None
        assert "warning" in results[0]

    def test_no_drift_samples(self):
        """Missing drift returns skeletons."""
        results = compute_interventions(
            {},
            treatments=["A"],
            outcome="C",
            latent_names=["A", "B", "C"],
        )
        assert results[0]["effect_size"] is None

    def test_sorted_by_abs_effect(self):
        """Results should be sorted by |effect_size| descending."""
        # Use non-diagonal A to get different effect sizes
        n = 10
        A = jnp.array([[-1.0, 0.0, 0.0], [0.8, -1.0, 0.0], [0.1, 0.0, -1.0]])
        samples = {
            "drift": jnp.broadcast_to(A, (n, 3, 3)),
            "cint": jnp.broadcast_to(jnp.ones(3), (n, 3)),
        }
        results = compute_interventions(
            samples,
            treatments=["A", "B"],
            outcome="C",
            latent_names=["A", "B", "C"],
        )
        effects = [abs(r["effect_size"]) for r in results if r["effect_size"] is not None]
        assert effects == sorted(effects, reverse=True)

    def test_identifiability_flag(self):
        """Non-identifiable treatments should be flagged."""
        samples = self._make_samples()
        causal_spec = {
            "identifiability": {"non_identifiable_treatments": {"A": {"confounders": ["U"]}}}
        }
        results = compute_interventions(
            samples,
            treatments=["A", "B"],
            outcome="C",
            latent_names=["A", "B", "C"],
            causal_spec=causal_spec,
        )
        a_result = next(r for r in results if r["treatment"] == "A")
        b_result = next(r for r in results if r["treatment"] == "B")
        assert a_result["identifiable"] is False
        assert b_result["identifiable"] is True
        assert "warning" in a_result

    def test_missing_cint_defaults_to_zeros(self):
        """When cint is missing, should default to zeros."""
        n = 5
        samples = {"drift": jnp.broadcast_to(-jnp.eye(2), (n, 2, 2))}
        results = compute_interventions(
            samples,
            treatments=["A"],
            outcome="B",
            latent_names=["A", "B"],
        )
        # With diagonal drift and zero cint, steady state is zero
        # so treatment effect should be nonzero from the shift
        assert results[0]["effect_size"] is not None
