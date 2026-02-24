"""Tests for do-operator counterfactual module."""

import jax.numpy as jnp
import pytest

from causal_ssm_agent.models.ssm.counterfactual import (
    compute_interventions,
    do,
    forward_simulate_intervention,
    steady_state,
    treatment_effect,
)


@pytest.fixture
def two_var_system():
    """2-latent system: A = [[-1, β], [0, -1]], β = 0.5, c = [1, 1].

    Steady state: η* = -A⁻¹ c
    For A = [[-1, 0.5], [0, -1]], A⁻¹ = [[-1, 0.5], [0, -1]]
    (block triangular), so -A⁻¹ = [[1, -0.5], [0, 1]],
    η* = [[1, -0.5], [0, 1]] @ [1, 1] = [0.5, 1.0].
    """
    beta = 0.5
    drift = jnp.array([[-1.0, beta], [0.0, -1.0]])
    cint = jnp.array([1.0, 1.0])
    return drift, cint, beta


def test_steady_state_matches_analytic(two_var_system):
    """Steady state should match -A⁻¹ c."""
    drift, cint, _ = two_var_system
    eta = steady_state(drift, cint)
    expected = -jnp.linalg.solve(drift, cint)
    assert jnp.allclose(eta, expected, atol=1e-4)


def test_do_clamps_treatment(two_var_system):
    """do(η₁ = v) should clamp index 1 to v."""
    drift, cint, _ = two_var_system
    v = 3.0
    eta = do(drift, cint, do_idx=1, do_value=v)
    assert jnp.isclose(eta[1], v, atol=1e-4)


def test_treatment_effect_equals_beta(two_var_system):
    """Effect of do(η₁ += 1) on η₀ should equal β (cross-effect)."""
    drift, cint, beta = two_var_system
    # treat_idx=1 (treatment), outcome_idx=0 (outcome)
    effect = treatment_effect(drift, cint, treat_idx=1, outcome_idx=0)
    assert jnp.isclose(effect, beta, atol=1e-3), f"Expected {beta}, got {float(effect)}"


def test_diagonal_drift_zero_cross_effect():
    """With diagonal drift (no cross-effects), treatment effect should be zero."""
    drift = jnp.array([[-1.0, 0.0], [0.0, -1.0]])
    cint = jnp.array([1.0, 1.0])
    effect = treatment_effect(drift, cint, treat_idx=1, outcome_idx=0)
    assert jnp.isclose(effect, 0.0, atol=1e-4), f"Expected 0, got {float(effect)}"


def test_treatment_effect_self():
    """Effect of do(η₀ += 1) on η₀ itself should be ~1 (we clamp it)."""
    drift = jnp.array([[-1.0, 0.0], [0.0, -1.0]])
    cint = jnp.array([1.0, 1.0])
    effect = treatment_effect(drift, cint, treat_idx=0, outcome_idx=0)
    assert jnp.isclose(effect, 1.0, atol=1e-3), f"Expected 1.0, got {float(effect)}"


# --- Forward simulation tests ---


def test_forward_simulate_converges_to_steady_state(two_var_system):
    """Forward simulation trajectory should converge to steady-state do() effect."""
    drift, cint, beta = two_var_system
    traj = forward_simulate_intervention(
        drift,
        cint,
        treat_idx=1,
        outcome_idx=0,
        shift_size=1.0,
        dt=0.1,
        horizon_steps=500,
    )
    # Trajectory end should approach the steady-state cross-effect (beta)
    assert jnp.isclose(traj[-1], beta, atol=0.05), (
        f"Trajectory end {float(traj[-1]):.4f} != steady-state {beta}"
    )


def test_forward_simulate_near_singular_drift():
    """Near-singular drift (time-invariant construct) should not produce NaN/Inf."""
    drift = jnp.array([[-1e-6, 0.0], [0.0, -1.0]])
    cint = jnp.array([0.0, 1.0])
    traj = forward_simulate_intervention(
        drift,
        cint,
        treat_idx=1,
        outcome_idx=0,
        shift_size=1.0,
        dt=1.0,
        horizon_steps=50,
    )
    assert jnp.all(jnp.isfinite(traj)), "Trajectory contains NaN/Inf"


def test_compute_interventions_with_times():
    """compute_interventions with times adds temporal keys to results."""
    n_draws = 10
    drift = jnp.array([[-1.0, 0.5], [0.0, -1.0]])
    cint = jnp.array([1.0, 1.0])
    samples = {
        "drift": jnp.tile(drift, (n_draws, 1, 1)),
        "cint": jnp.tile(cint, (n_draws, 1)),
    }
    times = jnp.arange(0, 30, 1.0)  # daily for 30 days
    results = compute_interventions(
        samples=samples,
        treatments=["B"],
        outcome="A",
        latent_names=["A", "B"],
        times=times,
    )
    assert len(results) == 1
    assert "temporal" in results[0]
    temporal = results[0]["temporal"]
    assert "effect_1d" in temporal
    assert "effect_7d" in temporal
    assert "peak_effect" in temporal
    assert "time_to_peak_days" in temporal


def test_manifest_effects_through_lambda():
    """Lambda projection maps latent effect to manifest space."""
    n_draws = 10
    drift = jnp.array([[-1.0, 0.5], [0.0, -1.0]])
    cint = jnp.array([1.0, 1.0])
    # Lambda: 2 manifest variables, loading 1.0 and 0.8 on latent 0
    lambda_mat = jnp.array([[1.0, 0.0], [0.8, 0.0]])
    samples = {
        "drift": jnp.tile(drift, (n_draws, 1, 1)),
        "cint": jnp.tile(cint, (n_draws, 1)),
        "lambda": jnp.tile(lambda_mat, (n_draws, 1, 1)),
    }
    times = jnp.arange(0, 30, 1.0)
    results = compute_interventions(
        samples=samples,
        treatments=["B"],
        outcome="A",
        latent_names=["A", "B"],
        manifest_names=["obs_a1", "obs_a2"],
        times=times,
    )
    assert "manifest_effects" in results[0]
    me = results[0]["manifest_effects"]
    # obs_a1 loads 1.0 on latent A, obs_a2 loads 0.8
    assert "obs_a1" in me
    assert "obs_a2" in me
    assert abs(me["obs_a2"] / me["obs_a1"] - 0.8) < 0.05


def test_compute_interventions_attaches_ppc_warnings():
    """PPC warnings from stage output are propagated onto treatment entries."""
    n_draws = 8
    drift = jnp.array([[-1.0, 0.5], [0.0, -1.0]])
    cint = jnp.array([1.0, 1.0])
    samples = {
        "drift": jnp.tile(drift, (n_draws, 1, 1)),
        "cint": jnp.tile(cint, (n_draws, 1)),
    }
    ppc_result = {
        "checked": True,
        "per_variable_warnings": [
            {
                "variable": "obs_a",
                "check_type": "calibration",
                "message": "Synthetic warning",
                "value": 0.0,
                "passed": False,
            }
        ],
    }

    results = compute_interventions(
        samples=samples,
        treatments=["B"],
        outcome="A",
        latent_names=["A", "B"],
        ppc_result=ppc_result,
    )

    assert len(results) == 1
    assert "ppc_warnings" in results[0]
    assert len(results[0]["ppc_warnings"]) == 1
    assert results[0]["ppc_warnings"][0]["variable"] == "obs_a"
