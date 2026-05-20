"""Tests for SDE simulation mode in ``simulate``.

The ``simulate`` function defaults to the deterministic ODE path. When
both ``key`` and ``diffusion_cov`` are supplied it integrates the
additive-noise SDE ``dy = f(t,y) dt + L dW`` with a Heun solver, where
``L = chol(diffusion_cov + jitter·I)``. Tests verify:

- Same key + same diffusion → bit-identical trajectories (no hidden RNG).
- Different keys → different sample paths (Brownian noise is doing work).
- Zero diffusion → SDE path matches the deterministic ODE path.
- Averaging many SDE samples converges to the ODE mean (additive noise
  is mean-zero).
- Passing only one of ``key`` / ``diffusion_cov`` raises.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest

from nof1_causal_lab.models.ssm.counterfactual import linear_vector_field
from nof1_causal_lab.models.ssm.dynamics import (
    Intervention,
    SimulationConfig,
    simulate,
)


def _linear_setup(decay: float = 1.0, c: float = 0.5):
    A = jnp.array([[-decay]])
    cint = jnp.array([c])
    vf = linear_vector_field(n_latent=1)
    params = ({"drift": A, "cint": cint},)
    init_state = jnp.array([0.0])
    time_grid = jnp.linspace(0.0, 5.0, 21)
    return vf, params, init_state, time_grid


class TestSimulateSDEMode:
    def test_same_key_gives_identical_trajectory(self):
        vf, params, y0, time_grid = _linear_setup()
        diff = jnp.eye(1) * 0.1
        key = jr.PRNGKey(0)
        traj_a = simulate(
            vf, params, Intervention.none(), y0, time_grid,
            key=key, diffusion_cov=diff,
        )
        traj_b = simulate(
            vf, params, Intervention.none(), y0, time_grid,
            key=key, diffusion_cov=diff,
        )
        assert jnp.allclose(traj_a, traj_b, atol=1e-10), (
            "Same key must yield identical SDE samples"
        )

    def test_different_keys_give_different_trajectories(self):
        vf, params, y0, time_grid = _linear_setup()
        diff = jnp.eye(1) * 0.2
        traj_a = simulate(
            vf, params, Intervention.none(), y0, time_grid,
            key=jr.PRNGKey(0), diffusion_cov=diff,
        )
        traj_b = simulate(
            vf, params, Intervention.none(), y0, time_grid,
            key=jr.PRNGKey(1), diffusion_cov=diff,
        )
        assert not jnp.allclose(traj_a, traj_b, atol=1e-3), (
            "Different RNG keys must yield different sample paths"
        )

    def test_zero_diffusion_matches_ode(self):
        vf, params, y0, time_grid = _linear_setup()
        # Diffusion ≈ 0 → SDE samples should match ODE.
        det = simulate(vf, params, Intervention.none(), y0, time_grid)
        near_zero_diff = jnp.eye(1) * 1e-10
        sde = simulate(
            vf, params, Intervention.none(), y0, time_grid,
            key=jr.PRNGKey(42), diffusion_cov=near_zero_diff,
        )
        assert jnp.allclose(det, sde, atol=5e-3), (
            f"SDE with ~0 diffusion should match ODE; max diff "
            f"{float(jnp.max(jnp.abs(det - sde)))}"
        )

    def test_sample_mean_approaches_ode(self):
        """Average over many SDE samples should approach the ODE mean
        (additive Wiener noise is zero-mean)."""
        vf, params, y0, time_grid = _linear_setup()
        det = simulate(vf, params, Intervention.none(), y0, time_grid)
        diff = jnp.eye(1) * 0.05

        n_samples = 64
        keys = jr.split(jr.PRNGKey(7), n_samples)
        samples = jnp.stack(
            [
                simulate(
                    vf, params, Intervention.none(), y0, time_grid,
                    key=k, diffusion_cov=diff,
                )
                for k in keys
            ]
        )
        sample_mean = jnp.mean(samples, axis=0)
        # Mean of 64 stochastic samples should be within a generous
        # tolerance of the deterministic mean.
        max_diff = float(jnp.max(jnp.abs(sample_mean - det)))
        assert max_diff < 0.1, (
            f"Sample-mean of 64 SDE draws not close enough to ODE mean; "
            f"max diff {max_diff}"
        )

    def test_requires_both_key_and_diffusion(self):
        vf, params, y0, time_grid = _linear_setup()
        with pytest.raises(ValueError, match="SDE mode requires both"):
            simulate(
                vf, params, Intervention.none(), y0, time_grid,
                key=jr.PRNGKey(0),
            )
        with pytest.raises(ValueError, match="SDE mode requires both"):
            simulate(
                vf, params, Intervention.none(), y0, time_grid,
                diffusion_cov=jnp.eye(1) * 0.1,
            )

    def test_sde_outputs_finite_at_grid_points(self):
        vf, params, y0, time_grid = _linear_setup()
        traj = simulate(
            vf, params, Intervention.none(), y0, time_grid,
            key=jr.PRNGKey(0), diffusion_cov=jnp.eye(1) * 0.1,
        )
        assert traj.shape == (21, 1)
        assert bool(jnp.all(jnp.isfinite(traj)))

    def test_sde_config_overrides_step_size(self):
        """The SimulationConfig.sde_dt knob actually changes the step size
        — same key with different ``sde_dt`` should produce different
        trajectories (different Brownian discretisation)."""
        vf, params, y0, time_grid = _linear_setup()
        diff = jnp.eye(1) * 0.1
        key = jr.PRNGKey(0)
        traj_default = simulate(
            vf, params, Intervention.none(), y0, time_grid,
            key=key, diffusion_cov=diff,
        )
        traj_fine = simulate(
            vf, params, Intervention.none(), y0, time_grid,
            config=SimulationConfig(sde_dt=0.005),
            key=key, diffusion_cov=diff,
        )
        # They should differ — finer step size = different SDE samples.
        # (Both still finite + same shape.)
        assert traj_default.shape == traj_fine.shape
        assert bool(jnp.all(jnp.isfinite(traj_fine)))
        # Some difference somewhere.
        assert float(jnp.max(jnp.abs(traj_default - traj_fine))) > 1e-4
