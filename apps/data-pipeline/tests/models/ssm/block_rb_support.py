"""Shared block-RB particle-filter test builders."""

import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla

from causal_ssm_agent.models.ssm.inference.targets.base import (
    CTParams,
    InitialStateParams,
    MeasurementParams,
)

_CANONICAL_LINK = {
    "gaussian": "identity",
    "student_t": "identity",
    "poisson": "log",
    "gamma": "log",
    "negative_binomial": "log",
    "bernoulli": "logit",
    "beta": "logit",
}


def canonical_link(manifest_dist: str) -> str:
    return _CANONICAL_LINK.get(str(manifest_dist), "identity")


def make_mixed_params(n_g=1, n_s=1, n_manifest=2, cross_coupling=True):
    """Build test parameters for a mixed Gaussian/non-Gaussian model."""
    n = n_g + n_s

    drift = jnp.diag(jnp.full(n, -0.5))
    if cross_coupling and n_g > 0 and n_s > 0:
        drift = drift.at[n_g, 0].set(0.2)
        drift = drift.at[0, n_g].set(0.15)

    ct_params = CTParams(
        drift=drift,
        diffusion_cov=jnp.eye(n) * 0.1,
        cint=jnp.zeros(n),
    )
    meas_params = MeasurementParams(
        lambda_mat=jnp.eye(n_manifest, n),
        manifest_means=jnp.zeros(n_manifest),
        manifest_cov=jnp.eye(n_manifest) * 0.1,
    )
    init = InitialStateParams(
        mean=jnp.zeros(n),
        cov=jnp.eye(n),
    )
    return ct_params, meas_params, init


def simulate_data(key, ct_params, meas_params, init, T=30):
    """Simulate Gaussian observations from a linear-Gaussian latent process."""
    n = init.mean.shape[0]
    n_manifest = meas_params.lambda_mat.shape[0]

    k1, k2 = random.split(key)
    states = [init.mean]
    dt = 1.0
    for _t in range(T - 1):
        k1, k_step = random.split(k1)
        drift_effect = ct_params.drift @ states[-1] * dt
        noise = random.normal(k_step, (n,)) * jnp.sqrt(0.1 * dt)
        states.append(states[-1] + drift_effect + noise)
    states = jnp.stack(states)

    eta = states @ meas_params.lambda_mat.T + meas_params.manifest_means
    noise = random.normal(k2, (T, n_manifest)) * jnp.sqrt(0.1)
    observations = eta + noise

    time_intervals = jnp.ones(T)
    return observations, time_intervals


def simulate_data_exact(key, ct_params, meas_params, init, T=30):
    """Simulate Gaussian observations using exact CT-to-DT discretization."""
    from causal_ssm_agent.models.ssm.discretization import discretize_system

    n = init.mean.shape[0]
    n_manifest = meas_params.lambda_mat.shape[0]
    dt = 1.0

    Ad, Qd, cd = discretize_system(ct_params.drift, ct_params.diffusion_cov, ct_params.cint, dt)
    if cd is None:
        cd = jnp.zeros(n)
    cd = cd.flatten()
    chol_Qd = jla.cholesky(Qd + jnp.eye(n) * 1e-6, lower=True)

    k1, k2 = random.split(key)
    states = [init.mean]
    for _t in range(T - 1):
        k1, k_step = random.split(k1)
        mean = Ad @ states[-1] + cd
        states.append(mean + chol_Qd @ random.normal(k_step, (n,)))
    states = jnp.stack(states)

    eta = states @ meas_params.lambda_mat.T + meas_params.manifest_means
    chol_R = jla.cholesky(meas_params.manifest_cov + jnp.eye(n_manifest) * 1e-8, lower=True)
    noise = (chol_R @ random.normal(k2, (n_manifest, T))).T
    observations = eta + noise

    time_intervals = jnp.ones(T)
    return observations, time_intervals


def simulate_poisson_data(key, ct_params, meas_params, init, T=30):
    """Simulate Poisson count observations using exact CT-to-DT discretization."""
    from causal_ssm_agent.models.ssm.discretization import discretize_system

    n = init.mean.shape[0]
    dt = 1.0

    Ad, Qd, cd = discretize_system(ct_params.drift, ct_params.diffusion_cov, ct_params.cint, dt)
    if cd is None:
        cd = jnp.zeros(n)
    cd = cd.flatten()
    chol_Qd = jla.cholesky(Qd + jnp.eye(n) * 1e-6, lower=True)

    k1, k2 = random.split(key)
    states = [init.mean]
    for _t in range(T - 1):
        k1, k_step = random.split(k1)
        mean = Ad @ states[-1] + cd
        states.append(mean + chol_Qd @ random.normal(k_step, (n,)))
    states = jnp.stack(states)

    eta = states @ meas_params.lambda_mat.T + meas_params.manifest_means
    rates = jnp.exp(jnp.clip(eta, -5.0, 5.0))
    observations = random.poisson(k2, rates).astype(jnp.float32)

    time_intervals = jnp.ones(T)
    return observations, time_intervals


def run_block_rbpf(
    ct_params,
    meas_params,
    init,
    observations,
    time_intervals,
    diffusion_dists,
    manifest_dists=None,
    n_particles=200,
    rng_key=None,
    extra_params=None,
    manifest_links=None,
):
    """Run block RBPF with per-variable diffusion distributions."""
    from causal_ssm_agent.models.ssm.inference.targets.particle import ParticleLikelihood

    if rng_key is None:
        rng_key = random.PRNGKey(42)
    if manifest_dists is None:
        manifest_dists = ["gaussian"] * meas_params.lambda_mat.shape[0]
    if manifest_links is None:
        manifest_links = [canonical_link(dist) for dist in manifest_dists]

    backend = ParticleLikelihood(
        n_latent=init.mean.shape[0],
        n_manifest=meas_params.lambda_mat.shape[0],
        n_particles=n_particles,
        rng_key=rng_key,
        manifest_dists=manifest_dists,
        diffusion_dists=diffusion_dists,
        manifest_links=manifest_links,
    )
    return backend.compute_log_likelihood(
        ct_params,
        meas_params,
        init,
        observations,
        time_intervals,
        extra_params=extra_params,
    )[-1]


def run_full_rbpf(
    ct_params,
    meas_params,
    init,
    observations,
    time_intervals,
    manifest_dists=None,
    n_particles=200,
    rng_key=None,
    manifest_links=None,
):
    """Run full RBPF with all latent variables Rao-Blackwellized."""
    from causal_ssm_agent.models.ssm.inference.targets.particle import ParticleLikelihood

    if rng_key is None:
        rng_key = random.PRNGKey(42)
    if manifest_dists is None:
        manifest_dists = ["gaussian"] * meas_params.lambda_mat.shape[0]
    if manifest_links is None:
        manifest_links = [canonical_link(dist) for dist in manifest_dists]

    backend = ParticleLikelihood(
        n_latent=init.mean.shape[0],
        n_manifest=meas_params.lambda_mat.shape[0],
        n_particles=n_particles,
        rng_key=rng_key,
        manifest_dists=manifest_dists,
        diffusion_dists=["gaussian"] * init.mean.shape[0],
        manifest_links=manifest_links,
    )
    return backend.compute_log_likelihood(
        ct_params,
        meas_params,
        init,
        observations,
        time_intervals,
    )[-1]


def run_bootstrap_pf(
    ct_params,
    meas_params,
    init,
    observations,
    time_intervals,
    manifest_dists=None,
    diffusion_dists=None,
    n_particles=200,
    rng_key=None,
    extra_params=None,
    manifest_links=None,
):
    """Run bootstrap PF with all latent variables sampled."""
    from causal_ssm_agent.models.ssm.inference.targets.particle import ParticleLikelihood

    if rng_key is None:
        rng_key = random.PRNGKey(42)
    if manifest_dists is None:
        manifest_dists = ["gaussian"] * meas_params.lambda_mat.shape[0]
    if manifest_links is None:
        manifest_links = [canonical_link(dist) for dist in manifest_dists]

    ep = {"proc_df": 100.0}
    if extra_params:
        ep.update(extra_params)
    if diffusion_dists is None:
        diffusion_dists = ["student_t"] * init.mean.shape[0]

    backend = ParticleLikelihood(
        n_latent=init.mean.shape[0],
        n_manifest=meas_params.lambda_mat.shape[0],
        n_particles=n_particles,
        rng_key=rng_key,
        manifest_dists=manifest_dists,
        diffusion_dists=diffusion_dists,
        manifest_links=manifest_links,
    )
    return backend.compute_log_likelihood(
        ct_params,
        meas_params,
        init,
        observations,
        time_intervals,
        extra_params=ep,
    )[-1]
