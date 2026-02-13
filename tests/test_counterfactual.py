"""Tests for do-operator counterfactual module."""

import jax.numpy as jnp
import pytest

from dsem_agent.models.ssm.counterfactual import do, steady_state, treatment_effect


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
