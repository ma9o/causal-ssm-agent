"""Posterior Predictive Checks (PPCs) for fitted CT-SSM models.

Forward-simulates observations from posterior parameter draws and compares
them to the real data, producing per-variable diagnostics that flag
calibration, autocorrelation, and variance issues.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import lax, vmap
from pydantic import BaseModel, Field

from causal_ssm_agent.models.ssm.constants import MIN_DT
from causal_ssm_agent.models.ssm.discretization import discretize_system_batched
from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily, LinkFunction

# Integer indices for jax.lax.switch dispatch (per-channel distribution + link)
# Default link entries (backward-compatible indices 0-6)
_DIST_IDX: dict[str, int] = {
    DistributionFamily.GAUSSIAN: 0,
    DistributionFamily.STUDENT_T: 1,
    DistributionFamily.POISSON: 2,
    DistributionFamily.GAMMA: 3,
    DistributionFamily.BERNOULLI: 4,
    DistributionFamily.NEGATIVE_BINOMIAL: 5,
    DistributionFamily.BETA: 6,
}

# Combined (dist, link) dispatch — adds non-default link variants
_DIST_LINK_IDX: dict[tuple[str, str], int] = {
    (DistributionFamily.GAUSSIAN, LinkFunction.IDENTITY): 0,
    (DistributionFamily.STUDENT_T, LinkFunction.IDENTITY): 1,
    (DistributionFamily.POISSON, LinkFunction.LOG): 2,
    (DistributionFamily.GAMMA, LinkFunction.LOG): 3,
    (DistributionFamily.BERNOULLI, LinkFunction.LOGIT): 4,
    (DistributionFamily.NEGATIVE_BINOMIAL, LinkFunction.LOG): 5,
    (DistributionFamily.BETA, LinkFunction.LOGIT): 6,
    # Non-default link variants
    (DistributionFamily.GAMMA, LinkFunction.INVERSE): 7,
    (DistributionFamily.BERNOULLI, LinkFunction.PROBIT): 8,
    (DistributionFamily.BETA, LinkFunction.PROBIT): 9,
}

# ---------------------------------------------------------------------------
# PPC models
# ---------------------------------------------------------------------------


class PPCWarning(BaseModel):
    """A single diagnostic warning for one manifest variable."""

    variable: str
    check_type: Literal["calibration", "autocorrelation", "variance"]
    message: str
    value: float
    passed: bool = True


class PPCOverlay(BaseModel):
    """Per-variable quantile bands for PPC ribbon/density overlay plots.

    Provides the data for Gabry's ppc_dens_overlay / ppc_ribbon plots:
    observed time series vs posterior predictive quantile bands.
    Optionally includes individual y_rep draw lines for spaghetti plots.
    """

    variable: str
    observed: list[float | None]
    q025: list[float]
    q25: list[float]
    median: list[float]
    q75: list[float]
    q975: list[float]
    spaghetti_draws: list[list[float]] = Field(default_factory=list)


class PPCTestStat(BaseModel):
    """Distribution of a test statistic across y_rep draws vs observed.

    Provides the data for Gabry's ppc_stat plots: histogram of T(y_rep)
    with a vertical line at T(y_observed).
    """

    variable: str
    stat_name: Literal["mean", "sd", "min", "max"]
    observed_value: float
    rep_values: list[float]


class PPCResult(BaseModel):
    """Aggregate PPC result."""

    per_variable_warnings: list[PPCWarning] = Field(default_factory=list)
    checked: bool = False
    n_subsample: int = 0
    overlays: list[PPCOverlay] = Field(default_factory=list)
    test_stats: list[PPCTestStat] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Forward simulation
# ---------------------------------------------------------------------------


def _simulate_one_draw_gaussian(
    drift: jnp.ndarray,
    diffusion_chol: jnp.ndarray,
    cint: jnp.ndarray | None,
    lambda_mat: jnp.ndarray,
    manifest_means: jnp.ndarray,
    manifest_chol: jnp.ndarray,
    t0_mean: jnp.ndarray,
    t0_chol: jnp.ndarray,
    dt_array: jnp.ndarray,
    rng_key: jax.Array,
) -> jnp.ndarray:
    """Simulate one trajectory for Gaussian observation noise.

    Returns:
        y_sim: (T, n_manifest) simulated observations
    """
    n_latent = drift.shape[0]
    n_manifest = lambda_mat.shape[0]
    T = dt_array.shape[0]

    diffusion_cov = diffusion_chol @ diffusion_chol.T
    Ad, Qd, cd = discretize_system_batched(drift, diffusion_cov, cint, dt_array)

    # If cd is None, use zeros
    if cd is None:
        cd = jnp.zeros((T, n_latent))

    # Split rng keys
    key_init, key_proc, key_obs = jax.random.split(rng_key, 3)

    # Initial state
    eta_0 = t0_mean + t0_chol @ jax.random.normal(key_init, (n_latent,))

    # Process noise keys
    proc_keys = jax.random.split(key_proc, T)
    obs_keys = jax.random.split(key_obs, T)

    def scan_fn(eta_prev, inputs):
        Ad_t, Qd_t, cd_t, pkey, okey = inputs
        # Cholesky of Qd_t (with jitter for PSD)
        Qd_t_safe = Qd_t + 1e-8 * jnp.eye(n_latent)
        Qd_chol = jnp.linalg.cholesky(Qd_t_safe)
        eps = jax.random.normal(pkey, (n_latent,))
        eta_t = Ad_t @ eta_prev + cd_t + Qd_chol @ eps

        # Observation
        delta = jax.random.normal(okey, (n_manifest,))
        y_t = lambda_mat @ eta_t + manifest_means + manifest_chol @ delta
        return eta_t, y_t

    _, y_sim = lax.scan(scan_fn, eta_0, (Ad, Qd, cd, proc_keys, obs_keys))
    return y_sim  # (T, n_manifest)


def _simulate_one_draw_nongaussian(
    drift: jnp.ndarray,
    diffusion_chol: jnp.ndarray,
    cint: jnp.ndarray | None,
    lambda_mat: jnp.ndarray,
    manifest_means: jnp.ndarray,
    manifest_cov: jnp.ndarray,
    t0_mean: jnp.ndarray,
    t0_chol: jnp.ndarray,
    dt_array: jnp.ndarray,
    rng_key: jax.Array,
    manifest_dist: str,
    obs_df: float | None = None,
    obs_shape: float | None = None,
    obs_r: float | None = None,
    obs_concentration: float | None = None,
    manifest_link: str | None = None,
) -> jnp.ndarray:
    """Simulate one trajectory for non-Gaussian observation noise.

    Returns:
        y_sim: (T, n_manifest) simulated observations
    """
    import numpyro.distributions as npdist

    n_latent = drift.shape[0]
    n_manifest = lambda_mat.shape[0]
    T = dt_array.shape[0]

    diffusion_cov = diffusion_chol @ diffusion_chol.T
    Ad, Qd, cd = discretize_system_batched(drift, diffusion_cov, cint, dt_array)
    if cd is None:
        cd = jnp.zeros((T, n_latent))

    key_init, key_proc, key_obs = jax.random.split(rng_key, 3)
    eta_0 = t0_mean + t0_chol @ jax.random.normal(key_init, (n_latent,))
    proc_keys = jax.random.split(key_proc, T)
    obs_keys = jax.random.split(key_obs, T)

    manifest_std = jnp.sqrt(jnp.diag(manifest_cov))

    def scan_fn(eta_prev, inputs):
        Ad_t, Qd_t, cd_t, pkey, okey = inputs
        Qd_t_safe = Qd_t + 1e-8 * jnp.eye(n_latent)
        Qd_chol = jnp.linalg.cholesky(Qd_t_safe)
        eps = jax.random.normal(pkey, (n_latent,))
        eta_t = Ad_t @ eta_prev + cd_t + Qd_chol @ eps

        loc = lambda_mat @ eta_t + manifest_means

        if manifest_dist == DistributionFamily.STUDENT_T:
            df = obs_df if obs_df is not None else 5.0
            y_t = npdist.StudentT(df=df, loc=loc, scale=manifest_std).sample(okey)
        elif manifest_dist == DistributionFamily.POISSON:
            rate = jnp.exp(loc)
            y_t = npdist.Poisson(rate=rate).sample(okey)
        elif manifest_dist == DistributionFamily.GAMMA:
            shape_param = obs_shape if obs_shape is not None else 2.0
            if manifest_link == LinkFunction.INVERSE:
                mean = 1.0 / jnp.clip(loc, 1e-6, None)
            else:
                mean = jnp.exp(loc)
            scale = mean / shape_param
            scale = jnp.maximum(scale, 1e-8)
            y_t = npdist.Gamma(concentration=shape_param, rate=1.0 / scale).sample(okey)
        elif manifest_dist == DistributionFamily.BERNOULLI:
            if manifest_link == LinkFunction.PROBIT:
                p = jax.scipy.stats.norm.cdf(loc)
            else:
                p = jax.nn.sigmoid(loc)
            y_t = npdist.Bernoulli(probs=p).sample(okey)
        elif manifest_dist == DistributionFamily.NEGATIVE_BINOMIAL:
            mu = jnp.exp(loc)
            r_val = obs_r if obs_r is not None else 5.0
            probs = mu / (mu + r_val)
            y_t = npdist.NegativeBinomialProbs(total_count=int(r_val), probs=probs).sample(okey)
        elif manifest_dist == DistributionFamily.BETA:
            if manifest_link == LinkFunction.PROBIT:
                mean = jax.scipy.stats.norm.cdf(loc)
            else:
                mean = jax.nn.sigmoid(loc)
            phi = obs_concentration if obs_concentration is not None else 10.0
            alpha = mean * phi
            beta_ = (1.0 - mean) * phi
            y_t = npdist.Beta(concentration1=alpha, concentration0=beta_).sample(okey)
        else:
            # Fallback to Gaussian
            manifest_chol = jnp.diag(manifest_std)
            delta = jax.random.normal(okey, (n_manifest,))
            y_t = loc + manifest_chol @ delta

        return eta_t, y_t

    _, y_sim = lax.scan(scan_fn, eta_0, (Ad, Qd, cd, proc_keys, obs_keys))
    return y_sim  # (T, n_manifest)


def _sample_channel(loc_j, key, dist_idx, std_j, df, shape_p, r_p, phi_p):
    """Sample one observation from a channel's distribution using jax.lax.switch.

    All 10 distribution+link combos are compiled but only the matching one executes.
    Uses raw JAX random functions for switch-compatibility.
    Indices 0-6: default link; 7: gamma+inverse; 8: bernoulli+probit; 9: beta+probit.
    """

    def _gauss(loc, k, s, _df, _sh, _r, _ph):
        return loc + s * jax.random.normal(k, ())

    def _student_t(loc, k, s, df_v, _sh, _r, _ph):
        # t-distribution via normal / sqrt(chi2/df); chi2(df) = 2*Gamma(df/2)
        k1, k2 = jax.random.split(k)
        z = jax.random.normal(k1, ())
        chi2 = 2.0 * jax.random.gamma(k2, df_v / 2.0)
        t_val = z * jnp.sqrt(df_v / jnp.maximum(chi2, 1e-10))
        return loc + s * t_val

    def _poisson(loc, k, _s, _df, _sh, _r, _ph):
        rate = jnp.exp(jnp.clip(loc, -20.0, 20.0))
        return jax.random.poisson(k, rate).astype(jnp.float32)

    def _gamma(loc, k, _s, _df, shape_v, _r, _ph):
        mean = jnp.exp(jnp.clip(loc, -20.0, 20.0))
        scale = jnp.maximum(mean / jnp.maximum(shape_v, 1e-8), 1e-8)
        return jax.random.gamma(k, shape_v) * scale

    def _bernoulli(loc, k, _s, _df, _sh, _r, _ph):
        p = jax.nn.sigmoid(loc)
        return jax.random.bernoulli(k, p).astype(jnp.float32)

    def _negbin(loc, k, _s, _df, _sh, r_v, _ph):
        # Gamma-Poisson mixture: g ~ Gamma(r, 1), y ~ Poisson(g * mu / r)
        mu = jnp.exp(jnp.clip(loc, -20.0, 20.0))
        k1, k2 = jax.random.split(k)
        g = jax.random.gamma(k1, r_v) * mu / jnp.maximum(r_v, 1e-8)
        return jax.random.poisson(k2, jnp.maximum(g, 1e-10)).astype(jnp.float32)

    def _beta(loc, k, _s, _df, _sh, _r, phi_v):
        mean = jax.nn.sigmoid(loc)
        alpha = jnp.maximum(mean * phi_v, 1e-4)
        beta_p = jnp.maximum((1.0 - mean) * phi_v, 1e-4)
        k1, k2 = jax.random.split(k)
        g1 = jax.random.gamma(k1, alpha)
        g2 = jax.random.gamma(k2, beta_p)
        return g1 / jnp.maximum(g1 + g2, 1e-10)

    def _gamma_inv(loc, k, _s, _df, shape_v, _r, _ph):
        mean = 1.0 / jnp.clip(loc, 1e-6, None)
        scale = jnp.maximum(mean / jnp.maximum(shape_v, 1e-8), 1e-8)
        return jax.random.gamma(k, shape_v) * scale

    def _bernoulli_probit(loc, k, _s, _df, _sh, _r, _ph):
        p = jax.scipy.stats.norm.cdf(loc)
        return jax.random.bernoulli(k, p).astype(jnp.float32)

    def _beta_probit(loc, k, _s, _df, _sh, _r, phi_v):
        mean = jax.scipy.stats.norm.cdf(loc)
        alpha = jnp.maximum(mean * phi_v, 1e-4)
        beta_p = jnp.maximum((1.0 - mean) * phi_v, 1e-4)
        k1, k2 = jax.random.split(k)
        g1 = jax.random.gamma(k1, alpha)
        g2 = jax.random.gamma(k2, beta_p)
        return g1 / jnp.maximum(g1 + g2, 1e-10)

    branches = [
        _gauss,
        _student_t,
        _poisson,
        _gamma,
        _bernoulli,
        _negbin,
        _beta,
        _gamma_inv,
        _bernoulli_probit,
        _beta_probit,
    ]
    return jax.lax.switch(dist_idx, branches, loc_j, key, std_j, df, shape_p, r_p, phi_p)


def _simulate_one_draw_mixed(
    drift: jnp.ndarray,
    diffusion_chol: jnp.ndarray,
    cint: jnp.ndarray | None,
    lambda_mat: jnp.ndarray,
    manifest_means: jnp.ndarray,
    manifest_cov: jnp.ndarray,
    t0_mean: jnp.ndarray,
    t0_chol: jnp.ndarray,
    dt_array: jnp.ndarray,
    rng_key: jax.Array,
    dist_indices: jnp.ndarray,
    obs_df: float = 5.0,
    obs_shape: float = 2.0,
    obs_r: float = 5.0,
    obs_concentration: float = 10.0,
) -> jnp.ndarray:
    """Simulate one trajectory with per-channel distribution types.

    Uses jax.lax.switch + vmap for per-channel dispatch, so each manifest
    variable can have a different observation noise distribution and link function.

    Args:
        dist_indices: (n_manifest,) integer array mapping each channel to
            a combined distribution+link type index (see _DIST_LINK_IDX).

    Returns:
        y_sim: (T, n_manifest) simulated observations
    """
    n_latent = drift.shape[0]
    n_manifest = lambda_mat.shape[0]
    T = dt_array.shape[0]

    diffusion_cov = diffusion_chol @ diffusion_chol.T
    Ad, Qd, cd = discretize_system_batched(drift, diffusion_cov, cint, dt_array)
    if cd is None:
        cd = jnp.zeros((T, n_latent))

    key_init, key_proc, key_obs = jax.random.split(rng_key, 3)
    eta_0 = t0_mean + t0_chol @ jax.random.normal(key_init, (n_latent,))
    proc_keys = jax.random.split(key_proc, T)
    obs_keys = jax.random.split(key_obs, T)

    manifest_std = jnp.sqrt(jnp.diag(manifest_cov))

    def scan_fn(eta_prev, inputs):
        Ad_t, Qd_t, cd_t, pkey, okey = inputs
        Qd_t_safe = Qd_t + 1e-8 * jnp.eye(n_latent)
        Qd_chol = jnp.linalg.cholesky(Qd_t_safe)
        eps = jax.random.normal(pkey, (n_latent,))
        eta_t = Ad_t @ eta_prev + cd_t + Qd_chol @ eps

        loc = lambda_mat @ eta_t + manifest_means
        channel_keys = jax.random.split(okey, n_manifest)
        y_t = vmap(_sample_channel)(
            loc,
            channel_keys,
            dist_indices,
            manifest_std,
            jnp.full(n_manifest, obs_df),
            jnp.full(n_manifest, obs_shape),
            jnp.full(n_manifest, obs_r),
            jnp.full(n_manifest, obs_concentration),
        )
        return eta_t, y_t

    _, y_sim = lax.scan(scan_fn, eta_0, (Ad, Qd, cd, proc_keys, obs_keys))
    return y_sim  # (T, n_manifest)


def simulate_posterior_predictive(
    samples: dict[str, jnp.ndarray],
    times: jnp.ndarray,
    manifest_dist: str = "gaussian",
    manifest_dists: list[str] | None = None,
    manifest_links: list[str] | None = None,
    n_subsample: int = 50,
    rng_seed: int = 42,
) -> jnp.ndarray:
    """Forward-simulate observations from posterior draws.

    Args:
        samples: Posterior samples dict from InferenceResult.get_samples().
            Expected keys: "drift", "diffusion", "lambda", "manifest_cov",
            "t0_means", "t0_cov". Optional: "cint", "manifest_means",
            "obs_df", "obs_shape".
        times: (T,) observation times.
        manifest_dist: Scalar noise family string (fallback when manifest_dists
            is None or all channels share the same distribution).
        manifest_dists: Per-channel noise families. When provided and channels
            have different distributions, uses per-channel dispatch via
            jax.lax.switch. Overrides manifest_dist.
        manifest_links: Per-channel link function strings. When provided,
            used together with manifest_dists for combined dispatch.
        n_subsample: Number of posterior draws to use.
        rng_seed: Random seed for simulation.

    Returns:
        y_sim: (n_subsample, T, n_manifest) simulated observations.
    """
    drift_draws = samples["drift"]  # (n_draws, n, n) or (n_draws, n_subj, n, n)
    diffusion_draws = samples["diffusion"]  # (n_draws, n, n) or cholesky
    lambda_mat = samples["lambda"]  # (n_draws, m, n) or (m, n)
    manifest_cov = samples["manifest_cov"]  # (n_draws, m, m) or (m, m)
    t0_means = samples["t0_means"]  # (n_draws, n) or (n_draws, n_subj, n)
    t0_cov = samples["t0_cov"]  # (n_draws, n, n) or (n, n)

    cint_draws = samples.get("cint")  # (n_draws, n) or None
    manifest_means_draws = samples.get("manifest_means")  # (n_draws, m) or None

    n_draws_total = drift_draws.shape[0]
    n_use = min(n_subsample, n_draws_total)

    # Subsample draws (evenly spaced)
    indices = jnp.linspace(0, n_draws_total - 1, n_use).astype(int)

    drift_sub = drift_draws[indices]
    diffusion_sub = diffusion_draws[indices]  # cholesky factor
    t0_means_sub = t0_means[indices]
    cint_sub = cint_draws[indices] if cint_draws is not None else None

    # Handle shared vs per-draw parameters
    if lambda_mat.ndim == 2:
        # Shared lambda — broadcast
        lambda_sub = jnp.broadcast_to(lambda_mat, (n_use, *lambda_mat.shape))
    else:
        lambda_sub = lambda_mat[indices]

    if manifest_cov.ndim == 2:
        manifest_cov_sub = jnp.broadcast_to(manifest_cov, (n_use, *manifest_cov.shape))
    else:
        manifest_cov_sub = manifest_cov[indices]

    if t0_cov.ndim == 2:
        t0_cov_sub = jnp.broadcast_to(t0_cov, (n_use, *t0_cov.shape))
    else:
        t0_cov_sub = t0_cov[indices]

    n_manifest = lambda_sub.shape[1]

    if manifest_means_draws is not None:
        if manifest_means_draws.ndim == 1:
            manifest_means_sub = jnp.broadcast_to(
                manifest_means_draws, (n_use, manifest_means_draws.shape[0])
            )
        else:
            manifest_means_sub = manifest_means_draws[indices]
    else:
        manifest_means_sub = jnp.zeros((n_use, n_manifest))

    # Compute dt array
    dt_array = jnp.diff(times, prepend=times[0])
    dt_array = jnp.maximum(dt_array, MIN_DT)

    rng = jax.random.PRNGKey(rng_seed)
    draw_keys = jax.random.split(rng, n_use)

    # Determine dispatch path: all-Gaussian, uniform non-Gaussian, or mixed
    effective_dists = manifest_dists if manifest_dists else None
    effective_links = manifest_links if manifest_links else None
    unique_dists = set(effective_dists) if effective_dists else {manifest_dist}
    all_gaussian = unique_dists == {DistributionFamily.GAUSSIAN} or unique_dists == {"gaussian"}

    # Check if link functions introduce mixing (e.g. all bernoulli but some probit)
    has_nondefault_link = False
    if effective_links is not None:
        has_nondefault_link = any(
            lk not in (LinkFunction.IDENTITY, "identity") for lk in effective_links
        )

    is_mixed = (effective_dists is not None and len(unique_dists) > 1) or (
        has_nondefault_link and not all_gaussian
    )

    # When links force mixed path but dists are scalar, broadcast to per-channel
    if is_mixed and effective_dists is None:
        effective_dists = [manifest_dist] * n_manifest

    if all_gaussian:
        # Fast path: correlated Gaussian observation noise via Cholesky
        manifest_chol_sub = vmap(
            lambda cov: jnp.linalg.cholesky(cov + 1e-8 * jnp.eye(cov.shape[0]))
        )(manifest_cov_sub)
        t0_chol_sub = vmap(lambda cov: jnp.linalg.cholesky(cov + 1e-8 * jnp.eye(cov.shape[0])))(
            t0_cov_sub
        )

        def sim_one(i):
            ci = cint_sub[i] if cint_sub is not None else None
            return _simulate_one_draw_gaussian(
                drift=drift_sub[i],
                diffusion_chol=diffusion_sub[i],
                cint=ci,
                lambda_mat=lambda_sub[i],
                manifest_means=manifest_means_sub[i],
                manifest_chol=manifest_chol_sub[i],
                t0_mean=t0_means_sub[i],
                t0_chol=t0_chol_sub[i],
                dt_array=dt_array,
                rng_key=draw_keys[i],
            )

        y_sim = vmap(sim_one)(jnp.arange(n_use))

    elif is_mixed:
        # Per-channel dispatch: different distributions/links for different channels
        t0_chol_sub = vmap(lambda cov: jnp.linalg.cholesky(cov + 1e-8 * jnp.eye(cov.shape[0])))(
            t0_cov_sub
        )

        # Build combined (dist, link) indices
        if effective_links is not None:
            assert effective_dists is not None
            dist_indices = jnp.array(
                [
                    _DIST_LINK_IDX.get((d, lk), _DIST_IDX.get(d, 0))
                    for d, lk in zip(effective_dists, effective_links)
                ]
            )
        else:
            assert effective_dists is not None
            dist_indices = jnp.array([_DIST_IDX.get(d, 0) for d in effective_dists])

        def _scalar(arr):
            if arr is None:
                return None
            return float(jnp.mean(arr)) if hasattr(arr, "ndim") and arr.ndim > 0 else arr

        obs_df_val = _scalar(samples.get("obs_df")) or 5.0
        obs_shape_val = _scalar(samples.get("obs_shape")) or 2.0
        obs_r_val = _scalar(samples.get("obs_r")) or 5.0
        obs_conc_val = _scalar(samples.get("obs_concentration")) or 10.0

        def sim_one(i):
            ci = cint_sub[i] if cint_sub is not None else None
            return _simulate_one_draw_mixed(
                drift=drift_sub[i],
                diffusion_chol=diffusion_sub[i],
                cint=ci,
                lambda_mat=lambda_sub[i],
                manifest_means=manifest_means_sub[i],
                manifest_cov=manifest_cov_sub[i],
                t0_mean=t0_means_sub[i],
                t0_chol=t0_chol_sub[i],
                dt_array=dt_array,
                rng_key=draw_keys[i],
                dist_indices=dist_indices,
                obs_df=obs_df_val,
                obs_shape=obs_shape_val,
                obs_r=obs_r_val,
                obs_concentration=obs_conc_val,
            )

        y_sim = vmap(sim_one)(jnp.arange(n_use))

    else:
        # Uniform non-Gaussian: all channels share the same non-Gaussian distribution
        t0_chol_sub = vmap(lambda cov: jnp.linalg.cholesky(cov + 1e-8 * jnp.eye(cov.shape[0])))(
            t0_cov_sub
        )

        # Resolve effective scalar distribution and link
        effective_dist = effective_dists[0] if effective_dists else manifest_dist
        effective_link = effective_links[0] if effective_links else None

        def _scalar(arr):
            """Collapse a posterior draw array to a scalar mean."""
            if arr is None:
                return None
            return float(jnp.mean(arr)) if hasattr(arr, "ndim") and arr.ndim > 0 else arr

        obs_df_val = _scalar(samples.get("obs_df"))
        obs_shape_val = _scalar(samples.get("obs_shape"))
        obs_r_val = _scalar(samples.get("obs_r"))
        obs_conc_val = _scalar(samples.get("obs_concentration"))

        def sim_one(i):
            ci = cint_sub[i] if cint_sub is not None else None
            return _simulate_one_draw_nongaussian(
                drift=drift_sub[i],
                diffusion_chol=diffusion_sub[i],
                cint=ci,
                lambda_mat=lambda_sub[i],
                manifest_means=manifest_means_sub[i],
                manifest_cov=manifest_cov_sub[i],
                t0_mean=t0_means_sub[i],
                t0_chol=t0_chol_sub[i],
                dt_array=dt_array,
                rng_key=draw_keys[i],
                manifest_dist=effective_dist,
                obs_df=obs_df_val,
                obs_shape=obs_shape_val,
                obs_r=obs_r_val,
                obs_concentration=obs_conc_val,
                manifest_link=effective_link,
            )

        y_sim = vmap(sim_one)(jnp.arange(n_use))

    return y_sim  # (n_subsample, T, n_manifest)


# ---------------------------------------------------------------------------
# Diagnostic checks
# ---------------------------------------------------------------------------


def _check_calibration(
    y_sim: jnp.ndarray,
    observations: jnp.ndarray,
    manifest_names: list[str],
    low_threshold: float = 0.70,
    high_threshold: float = 0.98,
) -> list[PPCWarning]:
    """Check calibration: % of timepoints where obs falls in [2.5th, 97.5th].

    Args:
        y_sim: (n_subsample, T, n_manifest)
        observations: (T, n_manifest)
        manifest_names: variable names
    """
    warnings = []
    n_manifest = observations.shape[1]

    q025 = jnp.percentile(y_sim, 2.5, axis=0)  # (T, m)
    q975 = jnp.percentile(y_sim, 97.5, axis=0)  # (T, m)

    for j in range(n_manifest):
        obs_j = observations[:, j]
        valid = ~jnp.isnan(obs_j)
        n_valid = jnp.sum(valid)
        if n_valid < 2:
            continue

        in_interval = valid & (obs_j >= q025[:, j]) & (obs_j <= q975[:, j])
        coverage = float(jnp.sum(in_interval) / n_valid)

        name = manifest_names[j] if j < len(manifest_names) else f"var_{j}"
        if coverage < low_threshold:
            warnings.append(
                PPCWarning(
                    variable=name,
                    check_type="calibration",
                    message=f"Undercoverage: {coverage:.0%} of observations fall in 95% PPC interval (expected ~95%)",
                    value=coverage,
                    passed=False,
                )
            )
        elif coverage > high_threshold:
            warnings.append(
                PPCWarning(
                    variable=name,
                    check_type="calibration",
                    message=f"Overcoverage: {coverage:.0%} of observations fall in 95% PPC interval (model may be too diffuse)",
                    value=coverage,
                    passed=False,
                )
            )
        else:
            warnings.append(
                PPCWarning(
                    variable=name,
                    check_type="calibration",
                    message=f"95% CI coverage: {coverage:.1%} (expected ~95%)",
                    value=coverage,
                    passed=True,
                )
            )

    return warnings


def _check_residual_autocorrelation(
    y_sim: jnp.ndarray,
    observations: jnp.ndarray,
    manifest_names: list[str],
    threshold: float = 0.3,
) -> list[PPCWarning]:
    """Check lag-1 autocorrelation of residuals (obs - posterior predictive mean).

    Args:
        y_sim: (n_subsample, T, n_manifest)
        observations: (T, n_manifest)
        manifest_names: variable names
    """
    warnings = []
    n_manifest = observations.shape[1]

    pp_mean = jnp.mean(y_sim, axis=0)  # (T, m)

    for j in range(n_manifest):
        obs_j = observations[:, j]
        valid = ~jnp.isnan(obs_j)

        # Build valid residuals
        residuals = jnp.where(valid, obs_j - pp_mean[:, j], 0.0)
        n_valid = int(jnp.sum(valid))
        if n_valid < 5:
            continue

        # Compute lag-1 autocorrelation on valid residuals
        # Extract valid residuals using masking
        valid_idx = jnp.where(valid, size=n_valid)[0]
        valid_res = residuals[valid_idx]

        mean_r = jnp.mean(valid_res)
        centered = valid_res - mean_r
        var_r = jnp.mean(centered**2)

        if var_r < 1e-12:
            continue

        autocov = jnp.mean(centered[:-1] * centered[1:])
        rho = float(autocov / var_r)

        name = manifest_names[j] if j < len(manifest_names) else f"var_{j}"
        passed = abs(rho) <= threshold
        warnings.append(
            PPCWarning(
                variable=name,
                check_type="autocorrelation",
                message=f"Residual autocorrelation at lag 1: {rho:.2f}"
                + ("" if passed else f" (|rho| > {threshold})"),
                value=rho,
                passed=passed,
            )
        )

    return warnings


def _check_variance_ratio(
    y_sim: jnp.ndarray,
    observations: jnp.ndarray,
    manifest_names: list[str],
    high_ratio: float = 3.0,
    low_ratio: float = 1.0 / 3.0,
) -> list[PPCWarning]:
    """Check posterior predictive std / observed std ratio.

    Args:
        y_sim: (n_subsample, T, n_manifest)
        observations: (T, n_manifest)
        manifest_names: variable names
    """
    warnings = []
    n_manifest = observations.shape[1]

    # Posterior predictive std: compute per-draw temporal std, then average across draws
    # This separates posterior uncertainty from temporal variation
    per_draw_std = jnp.std(y_sim, axis=1)  # (n_subsample, m) — temporal std per draw
    pp_std = jnp.mean(per_draw_std, axis=0)  # (m,) — average across draws

    for j in range(n_manifest):
        obs_j = observations[:, j]
        valid = ~jnp.isnan(obs_j)
        n_valid = int(jnp.sum(valid))
        if n_valid < 3:
            continue

        valid_idx = jnp.where(valid, size=n_valid)[0]
        obs_std = float(jnp.std(obs_j[valid_idx]))
        if obs_std < 1e-12:
            continue

        ratio = float(pp_std[j] / obs_std)
        name = manifest_names[j] if j < len(manifest_names) else f"var_{j}"

        if ratio > high_ratio:
            warnings.append(
                PPCWarning(
                    variable=name,
                    check_type="variance",
                    message=f"PPC variance too high: simulated std / observed std = {ratio:.1f}",
                    value=ratio,
                    passed=False,
                )
            )
        elif ratio < low_ratio:
            warnings.append(
                PPCWarning(
                    variable=name,
                    check_type="variance",
                    message=f"PPC variance too low: simulated std / observed std = {ratio:.1f}",
                    value=ratio,
                    passed=False,
                )
            )
        else:
            warnings.append(
                PPCWarning(
                    variable=name,
                    check_type="variance",
                    message=f"Predicted variance {float(pp_std[j]):.3f} vs observed {obs_std:.3f} (ratio {ratio:.2f})",
                    value=ratio,
                    passed=True,
                )
            )

    return warnings


# ---------------------------------------------------------------------------
# Overlay and test statistic computations
# ---------------------------------------------------------------------------


def _compute_overlays(
    y_sim: jnp.ndarray,
    observations: jnp.ndarray,
    manifest_names: list[str],
    n_spaghetti: int = 20,
) -> list[PPCOverlay]:
    """Compute per-variable quantile bands and spaghetti draws for PPC plots.

    Args:
        y_sim: (n_subsample, T, n_manifest)
        observations: (T, n_manifest)
        manifest_names: variable names
        n_spaghetti: number of individual y_rep draws to include for spaghetti plots
    """
    overlays = []
    n_manifest = observations.shape[1]
    n_draws = y_sim.shape[0]

    q025 = jnp.percentile(y_sim, 2.5, axis=0)  # (T, m)
    q25 = jnp.percentile(y_sim, 25.0, axis=0)
    q50 = jnp.percentile(y_sim, 50.0, axis=0)
    q75 = jnp.percentile(y_sim, 75.0, axis=0)
    q975 = jnp.percentile(y_sim, 97.5, axis=0)

    # Select evenly-spaced spaghetti draws
    n_spag = min(n_spaghetti, n_draws)
    spag_indices = jnp.linspace(0, n_draws - 1, n_spag).astype(int)

    for j in range(n_manifest):
        name = manifest_names[j] if j < len(manifest_names) else f"var_{j}"
        obs_j = observations[:, j]
        observed = [None if jnp.isnan(v) else float(v) for v in obs_j]

        # Spaghetti: individual draw trajectories for this variable
        spaghetti = [[float(v) for v in y_sim[int(idx), :, j]] for idx in spag_indices]

        overlays.append(
            PPCOverlay(
                variable=name,
                observed=observed,
                q025=[float(v) for v in q025[:, j]],
                q25=[float(v) for v in q25[:, j]],
                median=[float(v) for v in q50[:, j]],
                q75=[float(v) for v in q75[:, j]],
                q975=[float(v) for v in q975[:, j]],
                spaghetti_draws=spaghetti,
            )
        )

    return overlays


def _compute_test_stats(
    y_sim: jnp.ndarray,
    observations: jnp.ndarray,
    manifest_names: list[str],
) -> list[PPCTestStat]:
    """Compute test statistic distributions across y_rep draws.

    For each variable and each stat (mean, sd, min, max), computes the
    statistic across all y_rep draws and compares to the observed value.
    This is Gabry's ppc_stat plot data.

    Args:
        y_sim: (n_subsample, T, n_manifest)
        observations: (T, n_manifest)
        manifest_names: variable names
    """
    test_stats = []
    n_manifest = observations.shape[1]

    _StatName = Literal["mean", "sd", "min", "max"]
    stat_fns: dict[_StatName, Callable[..., jnp.ndarray]] = {
        "mean": jnp.nanmean,
        "sd": lambda x, **kw: jnp.nanstd(x, **kw),
        "min": jnp.nanmin,
        "max": jnp.nanmax,
    }

    for j in range(n_manifest):
        name = manifest_names[j] if j < len(manifest_names) else f"var_{j}"
        obs_j = observations[:, j]
        valid = ~jnp.isnan(obs_j)
        n_valid = int(jnp.sum(valid))
        if n_valid < 3:
            continue

        valid_idx = jnp.where(valid, size=n_valid)[0]
        obs_valid = obs_j[valid_idx]

        for stat_name, stat_fn in stat_fns.items():
            obs_stat = float(stat_fn(obs_valid))

            # Compute stat for each y_rep draw (over time axis)
            rep_stats = []
            for i in range(y_sim.shape[0]):
                y_rep_j = y_sim[i, :, j]
                # Use same valid mask as observed
                y_rep_valid = y_rep_j[valid_idx]
                rep_stats.append(float(stat_fn(y_rep_valid)))

            test_stats.append(
                PPCTestStat(
                    variable=name,
                    stat_name=stat_name,
                    observed_value=obs_stat,
                    rep_values=rep_stats,
                )
            )

    return test_stats


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def get_relevant_manifest_variables(
    lambda_mat: jnp.ndarray,
    treat_idx: int | None,
    outcome_idx: int | None,
    manifest_names: list[str],
    threshold: float = 0.01,
) -> set[str]:
    """Return manifest variable names with nonzero loadings on treatment or outcome.

    Args:
        lambda_mat: (n_manifest, n_latent) factor loading matrix
        treat_idx: index of treatment latent construct (or None)
        outcome_idx: index of outcome latent construct (or None)
        manifest_names: list of manifest variable names
        threshold: minimum absolute loading to be considered relevant

    Returns:
        Set of manifest variable names relevant to treatment/outcome.
    """
    relevant = set()
    n_manifest = lambda_mat.shape[0]

    for idx in (treat_idx, outcome_idx):
        if idx is None:
            continue
        for j in range(n_manifest):
            if abs(float(lambda_mat[j, idx])) >= threshold and j < len(manifest_names):
                relevant.add(manifest_names[j])

    return relevant


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_posterior_predictive_checks(
    samples: dict[str, jnp.ndarray],
    observations: jnp.ndarray,
    times: jnp.ndarray,
    manifest_names: list[str],
    manifest_dist: str = "gaussian",
    manifest_dists: list[str] | None = None,
    manifest_links: list[str] | None = None,
    n_subsample: int = 50,
    rng_seed: int = 42,
) -> PPCResult:
    """Run posterior predictive checks.

    Args:
        samples: Posterior samples from InferenceResult.get_samples()
        observations: (T, n_manifest) observed data
        times: (T,) observation times
        manifest_names: list of manifest variable names
        manifest_dist: scalar observation noise family (fallback)
        manifest_dists: per-channel noise families (overrides manifest_dist)
        manifest_links: per-channel link function strings
        n_subsample: number of posterior draws to forward-simulate
        rng_seed: random seed

    Returns:
        PPCResult with diagnostics
    """
    y_sim = simulate_posterior_predictive(
        samples=samples,
        times=times,
        manifest_dist=manifest_dist,
        manifest_dists=manifest_dists,
        manifest_links=manifest_links,
        n_subsample=n_subsample,
        rng_seed=rng_seed,
    )

    warnings: list[PPCWarning] = []
    warnings.extend(_check_calibration(y_sim, observations, manifest_names))
    warnings.extend(_check_residual_autocorrelation(y_sim, observations, manifest_names))
    warnings.extend(_check_variance_ratio(y_sim, observations, manifest_names))

    overlays = _compute_overlays(y_sim, observations, manifest_names)
    test_stats = _compute_test_stats(y_sim, observations, manifest_names)

    return PPCResult(
        per_variable_warnings=warnings,
        checked=True,
        n_subsample=int(y_sim.shape[0]),
        overlays=overlays,
        test_stats=test_stats,
    )
