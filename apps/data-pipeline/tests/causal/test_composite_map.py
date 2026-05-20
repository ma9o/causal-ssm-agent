"""Tests for the composite MAP estimator.

MAP estimation = L-BFGS on the marginal (Gaussian) or joint-at-fixed-x
(non-Gaussian) log-posterior, returned as an ``InferenceResult`` with
``method="map"``. Useful for fast model exploration, smoke checks, and
warm-starting MCMC.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro.distributions as ndist

from nof1_causal_lab.artifacts.model_spec import DistributionFamily, LinkFunction
from nof1_causal_lab.models.ssm.dynamics import (
    CompositeSpec,
    DiagonalDecaySpec,
    HillEdgeSpec,
    Intervention,
    compile_composite,
    runtime_from_composite,
    simulate,
)
from nof1_causal_lab.models.ssm.dynamics.priors import (
    diagonal_decay_prior,
    hill_ec50_prior,
    hill_emax_prior,
    hill_n_prior,
)
from nof1_causal_lab.models.ssm.inference.methods.composite_map import (
    fit_composite_map,
)
from nof1_causal_lab.models.ssm.inference.targets.kernels import (
    build_observation_kernel,
)
from nof1_causal_lab.models.ssm.inference.types import InferenceResult
from tests.ssm_test_utils import make_composite_ssm_model


def _build_synthetic_hill_problem():
    """Same scaffold as test_composite_aux_kalman uses for MCMC tests."""
    spec = CompositeSpec(
        n_latent=2,
        components=(
            DiagonalDecaySpec(decay_prior=diagonal_decay_prior()),
            HillEdgeSpec(
                source=0,
                target=1,
                emax_prior=hill_emax_prior(loc=0.0, scale=0.5),
                ec50_prior=hill_ec50_prior(loc=0.0, scale=0.5),
                n_prior=hill_n_prior(),
            ),
        ),
    )
    compiled = compile_composite(spec)
    true_params = (
        {"decay": jnp.array([0.3, 0.5])},
        {
            "Emax": jnp.asarray(1.5),
            "EC50": jnp.asarray(1.0),
            "n": jnp.asarray(2.0),
        },
    )
    init_mean = jnp.array([1.5, 0.0])
    H = jnp.array([[0.0, 1.0]])
    d_meas = jnp.array([0.0])
    R = jnp.array([[0.02]])
    T = 5
    runtime_times = jnp.linspace(0.5, 2.5, T)
    time_grid = jnp.concatenate([jnp.array([0.0]), runtime_times])
    traj_full = simulate(
        compiled.vector_field, true_params, Intervention.none(), init_mean, time_grid
    )
    true_x = traj_full[1:]
    obs_clean = jnp.einsum("ij,tj->ti", H, true_x) + d_meas
    obs = obs_clean + jr.normal(jr.PRNGKey(0), obs_clean.shape) * jnp.sqrt(R[0, 0])
    kernel = build_observation_kernel(
        DistributionFamily.GAUSSIAN, LinkFunction.IDENTITY, manifest_cov=np.asarray(R)
    )
    canonical = runtime_from_composite(
        compiled,
        init_mean=init_mean,
        init_cov=jnp.eye(2) * 0.1,
        diffusion_cov=jnp.eye(2) * 0.005,
        H=H,
        d_meas=d_meas,
        R=R,
        obs_kernel=kernel,
    )
    model = make_composite_ssm_model(
        spec,
        n_latent=2,
        n_manifest=1,
        H=H,
        d_meas=d_meas,
        init_mean=init_mean,
        init_cov=jnp.eye(2) * 0.1,
        diffusion_cov=jnp.eye(2) * 0.005,
        R=R,
    )
    return {
        "canonical": canonical,
        "model": model,
        "obs_kernel": kernel,
        "runtime_times": runtime_times,
        "observations": obs,
        "true_x": true_x,
        "true_params": true_params,
    }


class TestFitCompositeMap:
    def test_returns_inference_result_with_method_map(self):
        prob = _build_synthetic_hill_problem()
        result = fit_composite_map(
            prob["model"],
            prob["observations"],
            prob["runtime_times"],
            obs_kernel=prob["obs_kernel"],
            rng_seed=0,
            n_starts=2,
            maxiter=10,
        )
        assert isinstance(result, InferenceResult)
        assert result.method == "map"
        # Single MAP "draw" — samples have shape (1, *)
        samples = result.get_samples()
        for arr in samples.values():
            assert arr.shape[0] == 1
        # Composite-MAP-specific diagnostics
        assert "params_map" in result.diagnostics
        assert "log_post_at_map" in result.diagnostics
        assert result.diagnostics["objective"] == "marginal"

    def test_map_log_posterior_finite(self):
        prob = _build_synthetic_hill_problem()
        result = fit_composite_map(
            prob["model"],
            prob["observations"],
            prob["runtime_times"],
            obs_kernel=prob["obs_kernel"],
            rng_seed=1,
            n_starts=2,
            maxiter=10,
        )
        log_post = result.diagnostics["log_post_at_map"]
        assert isinstance(log_post, float)
        # Sanity bound — a 5-obs problem with reasonable priors shouldn't
        # blow up to absurd log-posterior values.
        assert -1000.0 < log_post < 1000.0

    def test_laplace_marginal_evidence_finite(self):
        """Round 28 — Laplace marginal likelihood from the pathfinder's
        local Gaussian approximation. Useful for Bayesian model
        comparison via Bayes factors; complements PSIS-LOO."""
        prob = _build_synthetic_hill_problem()
        result = fit_composite_map(
            prob["model"],
            prob["observations"],
            prob["runtime_times"],
            obs_kernel=prob["obs_kernel"],
            rng_seed=3,
            n_starts=2,
            maxiter=10,
        )
        laplace_log_z = result.diagnostics["laplace_log_evidence"]
        assert laplace_log_z is not None
        assert isinstance(laplace_log_z, float)
        # Sanity bound — for a 5-obs problem the marginal log-evidence
        # should be in a reasonable range.
        assert -200.0 < laplace_log_z < 200.0

    def test_non_gaussian_uses_joint_objective(self):
        """When the canonical's obs kernel is non-Gaussian, the MAP
        estimator falls back to the joint-at-fixed-trajectory log-
        posterior (marginal Kalman doesn't apply). Verifies the
        dispatch by checking the ``objective`` diagnostic field."""

        spec = CompositeSpec(
            n_latent=1,
            components=(DiagonalDecaySpec(decay_prior=ndist.LogNormal(0.0, 0.5)),),
        )
        beta_kernel = build_observation_kernel(
            DistributionFamily.BETA,
            LinkFunction.LOGIT,
            extra_params={"obs_concentration": jnp.asarray([50.0])},
            manifest_cov=np.asarray(jnp.eye(1) * 0.05),
        )
        beta_model = make_composite_ssm_model(
            spec,
            n_latent=1,
            n_manifest=1,
            H=jnp.array([[1.0]]),
            d_meas=jnp.zeros(1),
            init_mean=jnp.zeros(1),
            init_cov=jnp.eye(1) * 0.1,
            diffusion_cov=jnp.eye(1) * 0.01,
            R=jnp.eye(1) * 0.05,
        )
        # Beta-shaped observations
        obs = jnp.array([[0.4], [0.5], [0.6]])
        result = fit_composite_map(
            beta_model,
            obs,
            jnp.linspace(0.0, 1.0, 3),
            obs_kernel=beta_kernel,
            rng_seed=2,
            n_starts=2,
            maxiter=8,
        )
        assert result.method == "map"
        assert result.diagnostics["objective"] == "joint_at_fixed_trajectory"
