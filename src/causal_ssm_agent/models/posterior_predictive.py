"""Posterior Predictive Checks (PPCs) for fitted CT-SSM models.

Forward-simulates observations from posterior parameter draws and compares
them to the real data, producing per-variable diagnostics that flag
calibration, autocorrelation, and variance issues.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax import lax, vmap

from causal_ssm_agent.models.ssm.discretization import discretize_system_batched
from causal_ssm_agent.models.ssm.model import NoiseFamily


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PPCWarning:
    """A single diagnostic warning for one manifest variable."""

    variable: str  # manifest variable name
    check: str  # "calibration" | "autocorrelation" | "variance"
    message: str  # human-readable
    value: float  # diagnostic statistic

    def to_dict(self) -> dict:
        return {
            "variable": self.variable,
            "check": self.check,
            "message": self.message,
            "value": self.value,
        }


@dataclass
class PPCResult:
    """Aggregate PPC result."""

    warnings: list[PPCWarning] = field(default_factory=list)
    checked: bool = False
    n_subsample: int = 0

    def to_dict(self) -> dict:
        return {
            "warnings": [w.to_dict() for w in self.warnings],
            "checked": self.checked,
            "n_subsample": self.n_subsample,
        }


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

        if manifest_dist == NoiseFamily.STUDENT_T:
            df = obs_df if obs_df is not None else 5.0
            y_t = npdist.StudentT(df=df, loc=loc, scale=manifest_std).sample(okey)
        elif manifest_dist == NoiseFamily.POISSON:
            rate = jnp.exp(loc)
            y_t = npdist.Poisson(rate=rate).sample(okey)
        elif manifest_dist == NoiseFamily.GAMMA:
            shape_param = obs_shape if obs_shape is not None else 2.0
            # mean = shape * scale => scale = exp(loc) / shape
            scale = jnp.exp(loc) / shape_param
            scale = jnp.maximum(scale, 1e-8)
            y_t = npdist.Gamma(concentration=shape_param, rate=1.0 / scale).sample(okey)
        else:
            # Fallback to Gaussian
            manifest_chol = jnp.diag(manifest_std)
            delta = jax.random.normal(okey, (n_manifest,))
            y_t = loc + manifest_chol @ delta

        return eta_t, y_t

    _, y_sim = lax.scan(scan_fn, eta_0, (Ad, Qd, cd, proc_keys, obs_keys))
    return y_sim  # (T, n_manifest)


def simulate_posterior_predictive(
    samples: dict[str, jnp.ndarray],
    times: jnp.ndarray,
    manifest_dist: str = "gaussian",
    n_subsample: int = 50,
    rng_seed: int = 42,
    subject_ids: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Forward-simulate observations from posterior draws.

    Args:
        samples: Posterior samples dict from InferenceResult.get_samples().
            Expected keys: "drift", "diffusion", "lambda", "manifest_cov",
            "t0_means", "t0_cov". Optional: "cint", "manifest_means",
            "obs_df", "obs_shape".
        times: (T,) observation times.
        manifest_dist: Noise family string (e.g. "gaussian", "student_t").
        n_subsample: Number of posterior draws to use.
        rng_seed: Random seed for simulation.
        subject_ids: (T,) subject indices for hierarchical models.

    Returns:
        y_sim: (n_subsample, T, n_manifest) simulated observations.
            For hierarchical models: (n_subsample, n_subjects, T_max, n_manifest).
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
    dt_array = dt_array.at[0].set(1e-6)

    rng = jax.random.PRNGKey(rng_seed)
    draw_keys = jax.random.split(rng, n_use)

    # Check if hierarchical (drift has subject dimension)
    is_hierarchical = drift_sub.ndim == 4  # (n_use, n_subj, n, n)

    if is_hierarchical and subject_ids is not None:
        return _simulate_hierarchical(
            drift_sub=drift_sub,
            diffusion_sub=diffusion_sub,
            cint_sub=cint_sub,
            lambda_sub=lambda_sub,
            manifest_means_sub=manifest_means_sub,
            manifest_cov_sub=manifest_cov_sub,
            t0_means_sub=t0_means_sub,
            t0_cov_sub=t0_cov_sub,
            times=times,
            subject_ids=subject_ids,
            manifest_dist=manifest_dist,
            obs_df=samples.get("obs_df"),
            obs_shape=samples.get("obs_shape"),
            draw_keys=draw_keys,
        )

    is_gaussian = manifest_dist in (NoiseFamily.GAUSSIAN, "gaussian")

    if is_gaussian:
        # Compute manifest cholesky from cov
        manifest_chol_sub = vmap(
            lambda cov: jnp.linalg.cholesky(cov + 1e-8 * jnp.eye(cov.shape[0]))
        )(manifest_cov_sub)
        t0_chol_sub = vmap(
            lambda cov: jnp.linalg.cholesky(cov + 1e-8 * jnp.eye(cov.shape[0]))
        )(t0_cov_sub)

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
    else:
        t0_chol_sub = vmap(
            lambda cov: jnp.linalg.cholesky(cov + 1e-8 * jnp.eye(cov.shape[0]))
        )(t0_cov_sub)

        obs_df_val = samples.get("obs_df")
        obs_shape_val = samples.get("obs_shape")
        # Extract scalar from draws if present
        if obs_df_val is not None and obs_df_val.ndim > 0:
            obs_df_val = float(jnp.mean(obs_df_val))
        if obs_shape_val is not None and obs_shape_val.ndim > 0:
            obs_shape_val = float(jnp.mean(obs_shape_val))

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
                manifest_dist=manifest_dist,
                obs_df=obs_df_val,
                obs_shape=obs_shape_val,
            )

        y_sim = vmap(sim_one)(jnp.arange(n_use))

    return y_sim  # (n_subsample, T, n_manifest)


def _simulate_hierarchical(
    drift_sub: jnp.ndarray,
    diffusion_sub: jnp.ndarray,
    cint_sub: jnp.ndarray | None,
    lambda_sub: jnp.ndarray,
    manifest_means_sub: jnp.ndarray,
    manifest_cov_sub: jnp.ndarray,
    t0_means_sub: jnp.ndarray,
    t0_cov_sub: jnp.ndarray,
    times: jnp.ndarray,
    subject_ids: jnp.ndarray,
    manifest_dist: str,
    obs_df: float | None,
    obs_shape: float | None,
    draw_keys: jnp.ndarray,
) -> jnp.ndarray:
    """Simulate posterior predictive for hierarchical models.

    Returns:
        y_sim: (n_subsample, n_subjects, T_max, n_manifest)
    """
    n_use = drift_sub.shape[0]
    n_subjects = drift_sub.shape[1]
    n_manifest = lambda_sub.shape[1]
    T = times.shape[0]

    dt_array = jnp.diff(times, prepend=times[0])
    dt_array = dt_array.at[0].set(1e-6)

    is_gaussian = manifest_dist in (NoiseFamily.GAUSSIAN, "gaussian")

    results = jnp.zeros((n_use, n_subjects, T, n_manifest))

    # For each draw, simulate each subject
    for i in range(n_use):
        subj_keys = jax.random.split(draw_keys[i], n_subjects)
        for s in range(n_subjects):
            drift_s = drift_sub[i, s]
            diff_s = diffusion_sub[i, s] if diffusion_sub.ndim == 4 else diffusion_sub[i]
            cint_s = cint_sub[i, s] if cint_sub is not None and cint_sub.ndim == 3 else (
                cint_sub[i] if cint_sub is not None else None
            )
            t0_mean_s = t0_means_sub[i, s] if t0_means_sub.ndim == 3 else t0_means_sub[i]
            t0_cov_s = t0_cov_sub[i]
            t0_chol_s = jnp.linalg.cholesky(t0_cov_s + 1e-8 * jnp.eye(t0_cov_s.shape[0]))

            if is_gaussian:
                mc = manifest_cov_sub[i]
                m_chol = jnp.linalg.cholesky(mc + 1e-8 * jnp.eye(mc.shape[0]))
                y_s = _simulate_one_draw_gaussian(
                    drift=drift_s,
                    diffusion_chol=diff_s,
                    cint=cint_s,
                    lambda_mat=lambda_sub[i],
                    manifest_means=manifest_means_sub[i],
                    manifest_chol=m_chol,
                    t0_mean=t0_mean_s,
                    t0_chol=t0_chol_s,
                    dt_array=dt_array,
                    rng_key=subj_keys[s],
                )
            else:
                y_s = _simulate_one_draw_nongaussian(
                    drift=drift_s,
                    diffusion_chol=diff_s,
                    cint=cint_s,
                    lambda_mat=lambda_sub[i],
                    manifest_means=manifest_means_sub[i],
                    manifest_cov=manifest_cov_sub[i],
                    t0_mean=t0_mean_s,
                    t0_chol=t0_chol_s,
                    dt_array=dt_array,
                    rng_key=subj_keys[s],
                    manifest_dist=manifest_dist,
                    obs_df=obs_df,
                    obs_shape=obs_shape,
                )
            results = results.at[i, s].set(y_s)

    return results  # (n_subsample, n_subjects, T, n_manifest)


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
                    check="calibration",
                    message=f"Undercoverage: {coverage:.0%} of observations fall in 95% PPC interval (expected ~95%)",
                    value=coverage,
                )
            )
        elif coverage > high_threshold:
            warnings.append(
                PPCWarning(
                    variable=name,
                    check="calibration",
                    message=f"Overcoverage: {coverage:.0%} of observations fall in 95% PPC interval (model may be too diffuse)",
                    value=coverage,
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
        if abs(rho) > threshold:
            warnings.append(
                PPCWarning(
                    variable=name,
                    check="autocorrelation",
                    message=f"Residual lag-1 autocorrelation = {rho:.2f} (|rho| > {threshold})",
                    value=rho,
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

    # Posterior predictive std across draws and time
    pp_std = jnp.std(y_sim, axis=(0, 1))  # (m,)

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
                    check="variance",
                    message=f"PPC variance too high: simulated std / observed std = {ratio:.1f}",
                    value=ratio,
                )
            )
        elif ratio < low_ratio:
            warnings.append(
                PPCWarning(
                    variable=name,
                    check="variance",
                    message=f"PPC variance too low: simulated std / observed std = {ratio:.1f}",
                    value=ratio,
                )
            )

    return warnings


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
            if abs(float(lambda_mat[j, idx])) >= threshold:
                if j < len(manifest_names):
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
    n_subsample: int = 50,
    rng_seed: int = 42,
    subject_ids: jnp.ndarray | None = None,
) -> PPCResult:
    """Run posterior predictive checks.

    Args:
        samples: Posterior samples from InferenceResult.get_samples()
        observations: (T, n_manifest) observed data
        times: (T,) observation times
        manifest_names: list of manifest variable names
        manifest_dist: observation noise family
        n_subsample: number of posterior draws to forward-simulate
        rng_seed: random seed
        subject_ids: (T,) subject indices for hierarchical models

    Returns:
        PPCResult with diagnostics
    """
    y_sim = simulate_posterior_predictive(
        samples=samples,
        times=times,
        manifest_dist=manifest_dist,
        n_subsample=n_subsample,
        rng_seed=rng_seed,
        subject_ids=subject_ids,
    )

    # For hierarchical: flatten subjects for aggregate diagnostics
    if y_sim.ndim == 4:
        # (n_sub, n_subjects, T, m) — average across subjects for diagnostics
        y_sim_flat = jnp.mean(y_sim, axis=1)  # (n_sub, T, m)
    else:
        y_sim_flat = y_sim

    warnings: list[PPCWarning] = []
    warnings.extend(_check_calibration(y_sim_flat, observations, manifest_names))
    warnings.extend(_check_residual_autocorrelation(y_sim_flat, observations, manifest_names))
    warnings.extend(_check_variance_ratio(y_sim_flat, observations, manifest_names))

    return PPCResult(
        warnings=warnings,
        checked=True,
        n_subsample=int(y_sim.shape[0]),
    )
