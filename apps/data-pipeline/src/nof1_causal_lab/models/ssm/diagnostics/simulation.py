"""Synthetic data generation for diagnostic workflows."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax

from causal_ssm_agent.artifacts.model_spec import DistributionFamily
from causal_ssm_agent.models.ssm.discretization import discretize_system_batched
from causal_ssm_agent.models.ssm.inference.targets.base import CHOL_JITTER
from causal_ssm_agent.models.ssm.parameterization import (
    assemble_deterministics_from_registry,
    build_site_registry,
)


def simulate_ssm(
    drift: jnp.ndarray,
    diffusion_chol: jnp.ndarray,
    lambda_mat: jnp.ndarray,
    manifest_chol: jnp.ndarray,
    t0_means: jnp.ndarray,
    t0_chol: jnp.ndarray,
    times: jnp.ndarray,
    rng_key: jnp.ndarray,
    cint: jnp.ndarray | None = None,
    manifest_means: jnp.ndarray | None = None,
    manifest_dists: list[DistributionFamily | str] | None = None,
) -> jnp.ndarray:
    """Generate synthetic observations from constrained SSM parameters."""
    n_latent = drift.shape[0]
    n_manifest = lambda_mat.shape[0]
    n_timepoints = times.shape[0]

    diffusion_cov = diffusion_chol @ diffusion_chol.T
    dt_array = jnp.diff(times)
    Ad, Qd, cd = discretize_system_batched(drift, diffusion_cov, cint, dt_array)

    t0_cov = t0_chol @ t0_chol.T
    manifest_cov = manifest_chol @ manifest_chol.T

    if manifest_means is None:
        manifest_means = jnp.zeros(n_manifest)

    resolved_manifest_dists = (
        [
            dist if isinstance(dist, DistributionFamily) else DistributionFamily(dist)
            for dist in manifest_dists
        ]
        if manifest_dists is not None
        else [DistributionFamily.GAUSSIAN] * n_manifest
    )
    if len(resolved_manifest_dists) != n_manifest:
        raise ValueError(
            "manifest_dists length must match n_manifest: "
            f"{len(resolved_manifest_dists)} vs {n_manifest}"
        )
    unsupported_manifest_dists = sorted(
        {
            dist.value
            for dist in resolved_manifest_dists
            if dist not in {DistributionFamily.GAUSSIAN, DistributionFamily.POISSON}
        }
    )
    if unsupported_manifest_dists:
        raise ValueError(
            "simulate_ssm only supports gaussian/poisson manifest_dists. "
            f"Got {unsupported_manifest_dists}."
        )
    poisson_mask = [dist == DistributionFamily.POISSON for dist in resolved_manifest_dists]
    poisson_mask_array = jnp.asarray(poisson_mask, dtype=bool)
    all_gaussian = not any(poisson_mask)

    rng_key, init_key = random.split(rng_key)
    t0_chol_safe = jnp.linalg.cholesky(t0_cov + jnp.eye(n_latent) * CHOL_JITTER)
    x_0 = t0_means + t0_chol_safe @ random.normal(init_key, (n_latent,))

    if all_gaussian:
        manifest_chol_safe = jnp.linalg.cholesky(manifest_cov + jnp.eye(n_manifest) * CHOL_JITTER)

        def _sample_observation(key: jnp.ndarray, mean: jnp.ndarray) -> jnp.ndarray:
            return mean + manifest_chol_safe @ random.normal(key, (n_manifest,))

    else:
        manifest_sd = jnp.sqrt(jnp.maximum(jnp.diag(manifest_cov), CHOL_JITTER))

        def _sample_observation(key: jnp.ndarray, mean: jnp.ndarray) -> jnp.ndarray:
            gaussian_key, poisson_key = random.split(key)
            gaussian_obs = mean + manifest_sd * random.normal(gaussian_key, (n_manifest,))
            poisson_obs = random.poisson(poisson_key, jax.nn.softplus(mean)).astype(jnp.float64)
            return jnp.where(poisson_mask_array, poisson_obs, gaussian_obs)

    rng_key, obs_key = random.split(rng_key)
    mu_0 = lambda_mat @ x_0 + manifest_means
    y_0 = _sample_observation(obs_key, mu_0)

    def scan_fn(carry, inputs):
        x_prev, rng = carry
        Ad_t, Qd_t, cd_t = inputs

        rng, state_key, obs_key = random.split(rng, 3)
        Qd_chol = jnp.linalg.cholesky(Qd_t + jnp.eye(n_latent) * CHOL_JITTER)
        mean_x = Ad_t @ x_prev + cd_t
        x_t = mean_x + Qd_chol @ random.normal(state_key, (n_latent,))

        mu_t = lambda_mat @ x_t + manifest_means
        y_t = _sample_observation(obs_key, mu_t)

        return (x_t, rng), y_t

    cd_scan = jnp.zeros((n_timepoints - 1, n_latent)) if cd is None else cd
    (_, _), y_rest = lax.scan(scan_fn, (x_0, rng_key), (Ad, Qd, cd_scan))
    return jnp.concatenate([y_0[None, :], y_rest], axis=0)


def _simulate_from_params(con_dict, spec, times, rng_key, *, registry=None):
    """Simulate observations from constrained parameter dict."""
    if registry is None:
        registry = build_site_registry(spec)
    det = assemble_deterministics_from_registry(
        {name: value[None, ...] for name, value in con_dict.items()},
        spec,
        registry,
    )
    det = {name: value[0] for name, value in det.items()}
    n_latent, n_manifest = spec.n_latent, spec.n_manifest
    return simulate_ssm(
        drift=det.get("drift", jnp.zeros((n_latent, n_latent))),
        diffusion_chol=det.get("diffusion", jnp.eye(n_latent)),
        lambda_mat=det.get("lambda", jnp.eye(n_manifest, n_latent)),
        manifest_chol=jnp.linalg.cholesky(
            det.get("manifest_cov", jnp.eye(n_manifest)) + jnp.eye(n_manifest) * 1e-8
        ),
        t0_means=det.get("t0_means", jnp.zeros(n_latent)),
        t0_chol=jnp.linalg.cholesky(
            det.get("t0_cov", jnp.eye(n_latent)) + jnp.eye(n_latent) * CHOL_JITTER
        ),
        times=times,
        rng_key=rng_key,
        cint=det.get("cint"),
        manifest_dists=[dist.value for dist in spec.manifest_dists],
    )
