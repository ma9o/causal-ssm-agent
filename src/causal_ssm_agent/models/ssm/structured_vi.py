"""Structured Variational Inference with backward-factored Gaussian family.

Implements Method 2 from the algorithmic specification:

The variational family mirrors the true smoothing distribution's backward
factorization:
    q(z_{1:T} | phi) = q(z_T | phi_T) prod_{t=1}^{T-1} q(z_t | z_{t+1}, phi_t)

where each backward kernel is Gaussian:
    q(z_T | phi_T) = N(z_T; m_T, S_T)
    q(z_t | z_{t+1}, phi_t) = N(z_t; m_t + C_t(z_{t+1} - m_{t+1}), S_t)

The ELBO is optimized jointly over variational parameters phi and model
parameters theta using the reparameterization trick.

Can be initialized from Laplace-EM (Method 1) output for better convergence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import jax.random as random

from causal_ssm_agent.models.likelihoods.emissions import get_emission_fn
from causal_ssm_agent.models.ssm.discretization import discretize_system_batched
from causal_ssm_agent.models.ssm.tempered_core import run_tempered_smc

if TYPE_CHECKING:
    from causal_ssm_agent.models.likelihoods.base import (
        CTParams,
        InitialStateParams,
        MeasurementParams,
    )
    from causal_ssm_agent.models.ssm.inference import InferenceResult

# ---------------------------------------------------------------------------
# Variational parameters and sampling
# ---------------------------------------------------------------------------


def _init_variational_params(T, D, rng_key):
    """Initialize variational parameters for the backward-factored family.

    Args:
        T: number of time steps
        D: latent dimension

    Returns:
        phi: dict with keys 'm', 'log_S_diag', 'C'
            m: (T, D) variational means
            log_S_diag: (T, D) log-diagonal of Cholesky of S_t
            C: (T-1, D, D) backward coupling matrices
    """
    key1, _key2, _key3 = random.split(rng_key, 3)
    return {
        "m": random.normal(key1, (T, D)) * 0.01,
        "log_S_diag": jnp.full((T, D), -1.0),  # S_t ≈ 0.37 * I
        "C": jnp.zeros((max(T - 1, 1), D, D)),  # No backward coupling initially
    }


def _sample_trajectory(phi, rng_key):
    """Sample z_{1:T} from the backward-factored variational distribution.

    Sampling proceeds backward: z_T first, then z_{T-1}|z_T, etc.

    Args:
        phi: variational parameters
        rng_key: PRNG key

    Returns:
        z: (T, D) sampled trajectory
    """
    m = phi["m"]  # (T, D)
    log_S_diag = phi["log_S_diag"]  # (T, D)
    C = phi["C"]  # (T-1, D, D)

    T, D = m.shape
    S_diag = jnp.exp(log_S_diag)  # (T, D) - diagonal of Cholesky

    # Sample z_T ~ N(m_T, S_T)
    key_T, rng_key = random.split(rng_key)
    eps_T = random.normal(key_T, (D,))
    z_T = m[T - 1] + S_diag[T - 1] * eps_T

    if T == 1:
        return z_T[None]

    # Backward sampling: z_t | z_{t+1} ~ N(m_t + C_t(z_{t+1} - m_{t+1}), S_t)
    keys = random.split(rng_key, T - 1)

    def _backward_step(z_next, inputs):
        key, m_t, S_diag_t, C_t, m_tp1 = inputs
        eps = random.normal(key, (D,))
        mean = m_t + C_t @ (z_next - m_tp1)
        z_t = mean + S_diag_t * eps
        return z_t, z_t

    # Inputs for backward scan: t = T-2, T-3, ..., 0
    bwd_inputs = (
        keys[::-1],
        m[: T - 1][::-1],
        S_diag[: T - 1][::-1],
        C[::-1],
        m[1:][::-1],  # m_{t+1} for each t
    )

    _, z_rest_rev = jax.lax.scan(_backward_step, z_T, bwd_inputs)
    z_rest = z_rest_rev[::-1]  # (T-1, D)

    return jnp.concatenate([z_rest, z_T[None]], axis=0)


def _log_q_trajectory(z, phi):
    """Compute log q(z_{1:T} | phi) for the backward-factored family.

    Args:
        z: (T, D) trajectory
        phi: variational parameters

    Returns:
        scalar log q(z | phi)
    """
    m = phi["m"]
    log_S_diag = phi["log_S_diag"]
    C = phi["C"]
    T, _D = m.shape
    S_diag = jnp.exp(log_S_diag)

    # log q(z_T | phi_T) = log N(z_T; m_T, diag(S_T^2))
    from numpyro.distributions import Normal

    log_q_T = Normal(m[T - 1], S_diag[T - 1]).log_prob(z[T - 1]).sum()

    if T == 1:
        return log_q_T

    # log q(z_t | z_{t+1}, phi_t) for t = 0..T-2
    def _log_q_step(t):
        mean_t = m[t] + C[t] @ (z[t + 1] - m[t + 1])
        return Normal(mean_t, S_diag[t]).log_prob(z[t]).sum()

    log_q_rest = jax.vmap(_log_q_step)(jnp.arange(T - 1))
    return log_q_T + jnp.sum(log_q_rest)


# ---------------------------------------------------------------------------
# ELBO computation
# ---------------------------------------------------------------------------


def _compute_elbo(
    z,
    phi,
    observations,
    obs_mask,
    Ad,
    Qd,
    cd,
    H,
    d_meas,
    R,
    init_mean,
    init_cov,
    emission_log_prob_fn,
):
    """Compute the ELBO for a single sampled trajectory.

    ELBO = E_q[log p(y,z|theta) - log q(z|phi)]
         = sum_t log p(y_t|z_t) + log p(z_1) + sum_{t>=2} log p(z_t|z_{t-1}) - log q(z|phi)
    """
    T, D = z.shape
    jitter = jnp.eye(D) * 1e-6

    # 1. Emission log-probs
    mask_float = obs_mask.astype(jnp.float32)
    emission_lls = jax.vmap(lambda y, zz, mm: emission_log_prob_fn(y, zz, H, d_meas, R, mm))(
        observations, z, mask_float
    )
    total_emission_ll = jnp.sum(emission_lls)

    # 2. Transition log-probs (forward model)
    from numpyro.distributions import MultivariateNormal

    def _transition_ll(z_t, z_tm1, Ad_t, Qd_t, cd_t):
        mean = Ad_t @ z_tm1 + cd_t
        return MultivariateNormal(mean, covariance_matrix=Qd_t + jitter).log_prob(z_t)

    # Initial state: z_0 | prior
    z0_pred = Ad[0] @ init_mean + cd[0]
    P0_pred = Ad[0] @ init_cov @ Ad[0].T + Qd[0]
    P0_pred = 0.5 * (P0_pred + P0_pred.T) + jitter
    init_ll = MultivariateNormal(z0_pred, covariance_matrix=P0_pred).log_prob(z[0])

    if T > 1:
        trans_lls = jax.vmap(_transition_ll)(z[1:], z[:-1], Ad[1:], Qd[1:], cd[1:])
        total_trans_ll = init_ll + jnp.sum(trans_lls)
    else:
        total_trans_ll = init_ll

    # 3. Variational entropy: -log q(z | phi)
    log_q = _log_q_trajectory(z, phi)

    return total_emission_ll + total_trans_ll - log_q


# ---------------------------------------------------------------------------
# Structured VI likelihood backend
# ---------------------------------------------------------------------------


class StructuredVILikelihood:
    """Structured VI likelihood backend.

    Computes an ELBO lower bound on log p(y|theta) using the backward-factored
    Gaussian variational family. The variational parameters are optimized
    jointly with model parameters.
    """

    def __init__(
        self,
        n_latent: int,
        n_manifest: int,
        manifest_dist: str = "gaussian",
        n_vi_steps: int = 100,
        n_mc_samples: int = 4,
        vi_lr: float = 0.01,
    ):
        self.n_latent = n_latent
        self.n_manifest = n_manifest
        self.manifest_dist = manifest_dist
        self.n_vi_steps = n_vi_steps
        self.n_mc_samples = n_mc_samples
        self.vi_lr = vi_lr

    def compute_log_likelihood(
        self,
        ct_params: CTParams,
        measurement_params: MeasurementParams,
        initial_state: InitialStateParams,
        observations: jnp.ndarray,
        time_intervals: jnp.ndarray,
        obs_mask: jnp.ndarray | None = None,
        extra_params: dict | None = None,
    ) -> float:
        """Compute ELBO lower bound on log-likelihood."""
        import optax

        n = self.n_latent
        T = observations.shape[0]

        if obs_mask is None:
            obs_mask = ~jnp.isnan(observations)
        clean_obs = jnp.nan_to_num(observations, nan=0.0)

        Ad, Qd, cd = discretize_system_batched(
            ct_params.drift, ct_params.diffusion_cov, ct_params.cint, time_intervals
        )
        if cd is None:
            cd = jnp.zeros((T, n))

        emission_fn = get_emission_fn(self.manifest_dist, extra_params)

        H = measurement_params.lambda_mat
        d_meas = measurement_params.manifest_means
        R = measurement_params.manifest_cov

        # Initialize variational params
        rng_key = random.PRNGKey(0)
        phi = _init_variational_params(T, n, rng_key)

        # Optimize variational parameters for this theta
        optimizer = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(self.vi_lr))
        opt_state = optimizer.init(phi)

        def _mc_elbo(phi, rng_key):
            keys = random.split(rng_key, self.n_mc_samples)

            def _single_elbo(key):
                z = _sample_trajectory(phi, key)
                return _compute_elbo(
                    z,
                    phi,
                    clean_obs,
                    obs_mask,
                    Ad,
                    Qd,
                    cd,
                    H,
                    d_meas,
                    R,
                    initial_state.mean,
                    initial_state.cov,
                    emission_fn,
                )

            return jnp.mean(jax.vmap(_single_elbo)(keys))

        def _vi_body(carry, _):
            phi, opt_state, rng_key = carry
            rng_key, step_key = random.split(rng_key)
            _elbo, grads = jax.value_and_grad(_mc_elbo)(phi, step_key)
            updates, opt_state_new = optimizer.update(grads, opt_state)
            phi_new = optax.apply_updates(phi, updates)
            return (phi_new, opt_state_new, rng_key), None

        (phi, opt_state, rng_key), _ = jax.lax.scan(
            _vi_body, (phi, opt_state, rng_key), None, length=self.n_vi_steps
        )

        # Final ELBO estimate with more samples
        rng_key, eval_key = random.split(rng_key)
        final_elbo = _mc_elbo(phi, eval_key)
        return final_elbo


# ---------------------------------------------------------------------------
# fit_structured_vi: outer loop for parameter estimation
# ---------------------------------------------------------------------------


def fit_structured_vi(
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    n_outer: int = 100,
    n_csmc_particles: int = 20,
    n_mh_steps: int = 10,
    param_step_size: float = 0.1,
    n_warmup: int | None = None,
    target_accept: float | None = None,
    seed: int = 0,
    n_vi_steps: int = 50,
    n_mc_samples: int = 4,
    vi_lr: float = 0.01,
    n_leapfrog: int = 5,
    adaptive_tempering: bool = True,
    target_ess_ratio: float = 0.5,
    waste_free: bool = False,
    **kwargs: Any,  # noqa: ARG001
) -> InferenceResult:
    """Fit SSM via structured VI with tempered SMC outer loop.

    Uses the ELBO from structured VI as the log-density for a tempered SMC
    sampler over the parameter space.
    """
    manifest_dist = (
        model.spec.manifest_dist.value
        if hasattr(model.spec.manifest_dist, "value")
        else str(model.spec.manifest_dist)
    )
    backend = StructuredVILikelihood(
        n_latent=model.spec.n_latent,
        n_manifest=model.spec.n_manifest,
        manifest_dist=manifest_dist,
        n_vi_steps=n_vi_steps,
        n_mc_samples=n_mc_samples,
        vi_lr=vi_lr,
    )
    return run_tempered_smc(
        model,
        observations,
        times,
        n_outer=n_outer,
        n_csmc_particles=n_csmc_particles,
        n_mh_steps=n_mh_steps,
        param_step_size=param_step_size,
        n_warmup=n_warmup,
        target_accept=target_accept,
        seed=seed,
        adaptive_tempering=adaptive_tempering,
        target_ess_ratio=target_ess_ratio,
        waste_free=waste_free,
        n_leapfrog=n_leapfrog,
        method_name="structured_vi",
        likelihood_backend=backend,
        extra_diagnostics={
            "n_vi_steps": n_vi_steps,
            "n_mc_samples": n_mc_samples,
            "vi_lr": vi_lr,
        },
        print_prefix="Structured VI",
    )
