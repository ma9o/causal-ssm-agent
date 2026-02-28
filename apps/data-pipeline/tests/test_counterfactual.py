"""Tests for do-operator and causal intervention functions.

Covers: steady_state, do, treatment_effect, _summarize_trajectory.
"""

import jax.numpy as jnp

from causal_ssm_agent.models.ssm.counterfactual import (
    _summarize_trajectory,
    do,
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
