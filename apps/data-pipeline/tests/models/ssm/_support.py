"""Shared SSM test builders, simulators, and reference models.

Consolidates what used to live in ``autoreparam_support.py``,
``prior_predictive_support.py``, ``posterior_predictive_support.py``, and
``block_rb_support.py``. Imported from both fast (``tests/models/ssm/``)
and slow (``tests/slow/models/ssm/``) SSM test suites.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla
import numpyro
import numpyro.distributions as dist

from causal_ssm_agent.artifacts import LinkFunction
from causal_ssm_agent.distributions import DistributionFamily
from causal_ssm_agent.models.ssm.inference.targets.base import (
    CTParams,
    InitialStateParams,
    MeasurementParams,
)
from tests.ssm_test_utils import make_ssm_spec

if TYPE_CHECKING:
    from causal_ssm_agent.models.ssm.model import SSMSpec

# ══════════════════════════════════════════════════════════════════════════════
# AUTOREPARAM
# ══════════════════════════════════════════════════════════════════════════════


def simple_normal_model():
    x = numpyro.sample("x", dist.Normal(0.0, 1.0))
    y = numpyro.sample("y", dist.Normal(x, 0.5))
    numpyro.sample("obs", dist.Normal(y, 0.1), obs=jnp.array(1.0))


# ══════════════════════════════════════════════════════════════════════════════
# POSTERIOR-PREDICTIVE SAMPLES
# ══════════════════════════════════════════════════════════════════════════════


def make_samples(
    n_draws: int = 20,
    n_latent: int = 2,
    n_manifest: int = 3,
    seed: int = 0,
    drift_diag: float = -0.3,
    diff_sd: float = 0.3,
    obs_sd: float = 0.5,
    with_cint: bool = False,
) -> dict[str, jnp.ndarray]:
    """Build synthetic posterior samples for testing."""
    key = random.PRNGKey(seed)

    k1, *_ = random.split(key, 6)
    drift_base = jnp.eye(n_latent) * drift_diag
    offdiag = random.normal(k1, (n_draws, n_latent, n_latent)) * 0.01
    drift_draws = jnp.broadcast_to(drift_base, (n_draws, n_latent, n_latent)) + offdiag
    diag_idx = jnp.arange(n_latent)
    drift_draws = drift_draws.at[:, diag_idx, diag_idx].set(
        -jnp.abs(drift_draws[:, diag_idx, diag_idx])
    )

    diff_chol = jnp.eye(n_latent) * diff_sd
    diffusion_draws = jnp.broadcast_to(diff_chol, (n_draws, n_latent, n_latent))

    lambda_mat = jnp.zeros((n_manifest, n_latent))
    for i in range(min(n_manifest, n_latent)):
        lambda_mat = lambda_mat.at[i, i].set(1.0)
    for i in range(n_latent, n_manifest):
        lambda_mat = lambda_mat.at[i, 0].set(0.5)

    samples = {
        "drift": drift_draws,
        "diffusion": diffusion_draws,
        "lambda": lambda_mat,
        "manifest_cov": jnp.eye(n_manifest) * obs_sd**2,
        "t0_means": jnp.zeros((n_draws, n_latent)),
        "t0_cov": jnp.eye(n_latent) * 1.0,
    }

    if with_cint:
        samples["cint"] = jnp.zeros((n_draws, n_latent))

    return samples


def complex_mixed_family_config() -> tuple[list[str], list[str], list[int], list[str]]:
    manifest_dists = [
        "gaussian",
        "bernoulli",
        "poisson",
        "student_t",
        "gamma",
        "beta",
        "ordered_logistic",
        "categorical",
        "negative_binomial",
        "gaussian",
    ]
    manifest_links = [
        LinkFunction.IDENTITY.value,
        LinkFunction.LOGIT.value,
        LinkFunction.LOG.value,
        LinkFunction.IDENTITY.value,
        LinkFunction.LOG.value,
        LinkFunction.LOGIT.value,
        LinkFunction.CUMULATIVE_LOGIT.value,
        LinkFunction.SOFTMAX.value,
        LinkFunction.LOG.value,
        LinkFunction.IDENTITY.value,
    ]
    manifest_level_counts = [0, 0, 0, 0, 0, 0, 4, 4, 0, 0]
    manifest_names = [
        "stress_cont",
        "adherence_flag",
        "steps_count",
        "fatigue_t",
        "screen_gap",
        "sleep_efficiency",
        "symptom_severity",
        "coping_style",
        "rumination_count",
        "focus_cont",
    ]
    return manifest_dists, manifest_links, manifest_level_counts, manifest_names


def make_complex_mixed_samples(
    *,
    n_draws: int = 12,
    n_latent: int = 4,
    seed: int = 0,
) -> dict[str, jnp.ndarray]:
    samples = make_samples(
        n_draws=n_draws,
        n_latent=n_latent,
        n_manifest=10,
        seed=seed,
        drift_diag=-0.35,
        diff_sd=0.2,
        obs_sd=0.15,
        with_cint=True,
    )
    samples["lambda"] = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.3, 0.4, 0.0, 0.0],
            [0.0, 0.8, 0.0, 0.0],
            [0.2, 0.6, 0.0, 0.0],
            [0.0, 0.0, 0.9, 0.0],
            [0.0, 0.0, 0.5, 0.3],
            [0.0, 0.0, 0.7, 0.0],
            [0.0, 0.2, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.9],
            [0.1, 0.0, 0.2, 0.8],
        ],
        dtype=jnp.float32,
    )
    samples["manifest_cov"] = jnp.diag(
        jnp.array([0.12, 0.08, 0.1, 0.18, 0.1, 0.05, 0.08, 0.08, 0.11, 0.12], dtype=jnp.float32)
        ** 2
    )
    samples["t0_cov"] = jnp.eye(n_latent, dtype=jnp.float32) * 0.25
    samples["manifest_means"] = jnp.broadcast_to(
        jnp.array([0.0, -0.3, 0.4, 0.0, -0.2, 0.0, 0.0, 0.1, 0.2, -0.1], dtype=jnp.float32),
        (n_draws, 10),
    )
    samples["obs_df"] = jnp.full((n_draws,), 6.0, dtype=jnp.float32)
    samples["obs_shape"] = jnp.full((n_draws,), 3.0, dtype=jnp.float32)
    samples["obs_r"] = jnp.full((n_draws,), 8.0, dtype=jnp.float32)
    samples["obs_concentration"] = jnp.full((n_draws,), 14.0, dtype=jnp.float32)

    ordered_cutpoints = jnp.zeros((10, 3), dtype=jnp.float32)
    ordered_cutpoints = ordered_cutpoints.at[6].set(jnp.array([-1.2, 0.0, 1.1], dtype=jnp.float32))
    samples["obs_ordered_cutpoints"] = ordered_cutpoints

    cat_intercepts = jnp.zeros((10, 3), dtype=jnp.float32)
    cat_intercepts = cat_intercepts.at[7].set(jnp.array([0.8, -0.1, -0.6], dtype=jnp.float32))
    samples["obs_cat_intercepts"] = cat_intercepts

    cat_slopes = jnp.zeros((10, 3), dtype=jnp.float32)
    cat_slopes = cat_slopes.at[7].set(jnp.array([0.5, -0.2, -0.4], dtype=jnp.float32))
    samples["obs_cat_slopes"] = cat_slopes

    return samples


# ══════════════════════════════════════════════════════════════════════════════
# PRIOR-PREDICTIVE RUNTIME SPEC
# ══════════════════════════════════════════════════════════════════════════════


def complex_mixed_runtime_spec() -> SSMSpec:
    return make_ssm_spec(
        n_latent=4,
        n_manifest=10,
        drift=jnp.array(
            [
                [-0.45, 0.0, 0.0, 0.0],
                [0.08, -0.35, 0.0, 0.0],
                [0.02, 0.06, -0.4, 0.0],
                [0.0, 0.03, 0.05, -0.3],
            ],
            dtype=jnp.float32,
        ),
        diffusion=jnp.diag(jnp.array([0.2, 0.18, 0.16, 0.14], dtype=jnp.float32)),
        cint=jnp.zeros(4, dtype=jnp.float32),
        lambda_mat=jnp.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.3, 0.4, 0.0, 0.0],
                [0.0, 0.8, 0.0, 0.0],
                [0.2, 0.6, 0.0, 0.0],
                [0.0, 0.0, 0.9, 0.0],
                [0.0, 0.0, 0.5, 0.3],
                [0.0, 0.0, 0.7, 0.0],
                [0.0, 0.2, 0.5, 0.0],
                [0.0, 0.0, 0.0, 0.9],
                [0.1, 0.0, 0.2, 0.8],
            ],
            dtype=jnp.float32,
        ),
        manifest_means=jnp.array(
            [0.0, -0.3, 0.4, 0.0, -0.2, 0.0, 0.0, 0.1, 0.2, -0.1],
            dtype=jnp.float32,
        ),
        manifest_var=jnp.diag(
            jnp.array([0.12, 0.08, 0.1, 0.18, 0.1, 0.05, 0.08, 0.08, 0.11, 0.12], dtype=jnp.float32)
            ** 2
        ),
        t0_means=jnp.zeros(4, dtype=jnp.float32),
        t0_var=jnp.eye(4, dtype=jnp.float32) * 0.25,
        manifest_dists=[
            DistributionFamily.GAUSSIAN,
            DistributionFamily.BERNOULLI,
            DistributionFamily.POISSON,
            DistributionFamily.STUDENT_T,
            DistributionFamily.GAMMA,
            DistributionFamily.BETA,
            DistributionFamily.ORDERED_LOGISTIC,
            DistributionFamily.CATEGORICAL,
            DistributionFamily.NEGATIVE_BINOMIAL,
            DistributionFamily.GAUSSIAN,
        ],
        manifest_links=[
            LinkFunction.IDENTITY,
            LinkFunction.LOGIT,
            LinkFunction.LOG,
            LinkFunction.IDENTITY,
            LinkFunction.LOG,
            LinkFunction.LOGIT,
            LinkFunction.CUMULATIVE_LOGIT,
            LinkFunction.SOFTMAX,
            LinkFunction.LOG,
            LinkFunction.IDENTITY,
        ],
        manifest_level_counts=[0, 0, 0, 0, 0, 0, 4, 4, 0, 0],
        latent_names=["stress", "adherence", "sleep", "focus"],
        manifest_names=[
            "stress_cont",
            "adherence_flag",
            "steps_count",
            "fatigue_t",
            "screen_gap",
            "sleep_efficiency",
            "symptom_severity",
            "coping_style",
            "rumination_count",
            "focus_cont",
        ],
    )


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK-RB / PARTICLE-FILTER BUILDERS
# ══════════════════════════════════════════════════════════════════════════════


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
