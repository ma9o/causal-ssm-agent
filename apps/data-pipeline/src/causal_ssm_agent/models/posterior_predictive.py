"""Posterior Predictive Checks (PPCs) for fitted CT-SSM models.

Forward-simulates observations from posterior parameter draws and compares
them to the real data, producing per-variable diagnostics that flag
calibration, autocorrelation, and variance issues.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

    from causal_ssm_agent.models.ssm_observation_metadata import ObservationSupportRuntime
    from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily

import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax, vmap
from pydantic import BaseModel, Field

from causal_ssm_agent.models.likelihoods.emissions import build_predictive_observation_sampler
from causal_ssm_agent.models.likelihoods.kernels import (
    build_composite_observation_kernel,
    build_observation_kernel,
    build_transition_kernel,
    compile_transition_semantics,
)
from causal_ssm_agent.models.likelihoods.observation_families import (
    any_family_needs_level_metadata,
    resolve_manifest_families_and_links,
)
from causal_ssm_agent.models.likelihoods.trajectory_observations import (
    compile_observation_operator,
)
from causal_ssm_agent.models.ssm.constants import MIN_DT
from causal_ssm_agent.models.ssm.covariance_utils import (
    INITIAL_STATE_COV_MIN_EIGENVALUE,
    stable_cholesky,
)
from causal_ssm_agent.models.ssm.discretization import discretize_system_batched

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


def _broadcast_draw_param(
    value: jnp.ndarray | None,
    n_use: int,
    indices: jnp.ndarray,
) -> jnp.ndarray | None:
    if value is None:
        return None
    if value.ndim == 0:
        return jnp.broadcast_to(value, (n_use,))
    if value.shape[0] == n_use:
        return value
    if value.shape[0] >= int(indices[-1]) + 1:
        return value[indices]
    return jnp.broadcast_to(value, (n_use, *value.shape))


def _linear_predictors_from_latent_trajectory(
    latent_trajectory: jnp.ndarray,
    lambda_mat: jnp.ndarray,
    manifest_means: jnp.ndarray,
) -> jnp.ndarray:
    """Map a latent trajectory to observation linear predictors."""
    return jax.vmap(lambda eta_t: lambda_mat @ eta_t + manifest_means)(latent_trajectory)


def _simulate_latent_trajectory(
    drift: jnp.ndarray,
    diffusion_chol: jnp.ndarray,
    cint: jnp.ndarray | None,
    t0_mean: jnp.ndarray,
    t0_chol: jnp.ndarray,
    transition_dt_array: jnp.ndarray,
    n_timepoints: int,
    rng_key: jax.Array,
    transition_semantics,
    proc_df: float | jnp.ndarray = 5.0,
) -> jnp.ndarray:
    """Simulate one latent trajectory from the continuous-time dynamics."""
    n_latent = drift.shape[0]
    T = n_timepoints

    key_init, key_proc = random.split(rng_key)
    eta_0 = t0_mean + t0_chol @ random.normal(key_init, (n_latent,))
    if T == 1:
        return eta_0[None, :]

    diffusion_cov = diffusion_chol @ diffusion_chol.T
    Ad, Qd, cd = discretize_system_batched(drift, diffusion_cov, cint, transition_dt_array)
    if cd is None:
        cd = jnp.zeros((T - 1, n_latent))

    proc_keys = random.split(key_proc, T - 1)
    transition_kernel = build_transition_kernel(transition_semantics, {"proc_df": proc_df})

    def scan_fn(eta_prev, inputs):
        Ad_t, Qd_t, cd_t, pkey = inputs
        Qd_chol = stable_cholesky(Qd_t)
        noise = transition_kernel.sample_noise_fn(pkey, Qd_chol)
        eta_t = Ad_t @ eta_prev + cd_t + noise
        return eta_t, eta_t

    _, eta_rest = lax.scan(scan_fn, eta_0, (Ad, Qd, cd, proc_keys))
    return jnp.concatenate((eta_0[None, :], eta_rest), axis=0)


def _build_response_kernel(
    manifest_dist: str,
    manifest_dists: list[str] | None,
    manifest_links: list[str] | None,
    n_manifest: int,
    extra_params: dict | None,
):
    """Build the response-space observation kernel for one posterior draw."""
    dists, links = resolve_manifest_families_and_links(
        manifest_dist,
        n_manifest,
        manifest_dists=manifest_dists,
        manifest_links=manifest_links,
    )
    if len(set(zip(dists, links))) == 1:
        return build_observation_kernel(dists[0], links[0], extra_params)
    return build_composite_observation_kernel(dists, links, extra_params)


def _slice_extra_params_for_indices(
    extra_params: dict | None,
    indices: list[int],
) -> dict | None:
    """Slice per-channel extra params down to a manifest subset."""
    if extra_params is None:
        return None

    sliced: dict = {}
    idx = jnp.asarray(indices, dtype=jnp.int32)
    for key, value in extra_params.items():
        if hasattr(value, "ndim") and hasattr(value, "shape") and value.ndim >= 1:
            try:
                if value.shape[0] >= len(indices):
                    sliced[key] = value[idx]
                    continue
            except TypeError:
                pass
        sliced[key] = value
    return sliced


def _resolve_effective_observation_mask(
    target_shape: tuple[int, ...],
    semantic_mask: jnp.ndarray | None,
    observation_mask: jnp.ndarray | None,
) -> jnp.ndarray:
    """Return the explicit emitted-observation mask for one simulated draw."""
    if len(target_shape) == 2:
        mask_shape = target_shape
    else:
        mask_shape = target_shape[1:]
    effective_mask = jnp.ones(mask_shape, dtype=bool)
    if semantic_mask is not None:
        effective_mask = effective_mask & (semantic_mask > 0.5)
    if observation_mask is not None:
        effective_mask = effective_mask & observation_mask.astype(bool)
    return effective_mask


def _apply_observation_mask(
    y_sim: jnp.ndarray,
    semantic_mask: jnp.ndarray | None,
    observation_mask: jnp.ndarray | None,
) -> jnp.ndarray:
    """Set structurally absent observations to NaN."""
    effective_mask = _resolve_effective_observation_mask(
        y_sim.shape,
        semantic_mask,
        observation_mask,
    )
    if y_sim.ndim == 2:
        return jnp.where(effective_mask, y_sim, jnp.nan)
    return jnp.where(effective_mask[None, :, :], y_sim, jnp.nan)


def _simulate_one_draw(
    drift: jnp.ndarray,
    diffusion_chol: jnp.ndarray,
    cint: jnp.ndarray | None,
    transition_semantics,
    proc_df: float | jnp.ndarray,
    lambda_mat: jnp.ndarray,
    manifest_means: jnp.ndarray,
    t0_mean: jnp.ndarray,
    t0_chol: jnp.ndarray,
    transition_dt_array: jnp.ndarray,
    n_timepoints: int,
    rng_key: jax.Array,
    *,
    manifest_dist: str,
    manifest_dists: list[str] | None,
    manifest_links: list[str] | None,
    point_sampler,
    interval_summary_sampler,
    observation_operator,
    observation_mask: jnp.ndarray | None,
    extra_params: dict | None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Simulate one draw using shared transition and observation operators."""
    key_latent, key_point, key_interval_summary = random.split(rng_key, 3)
    latent_trajectory = _simulate_latent_trajectory(
        drift,
        diffusion_chol,
        cint,
        t0_mean,
        t0_chol,
        transition_dt_array,
        n_timepoints,
        key_latent,
        transition_semantics,
        proc_df=proc_df,
    )
    linear_predictors = _linear_predictors_from_latent_trajectory(
        latent_trajectory,
        lambda_mat,
        manifest_means,
    )
    point_samples = point_sampler.sample_point_trajectory(key_point, linear_predictors)

    if not observation_operator.requires_interval_summary_handling:
        effective_mask = _resolve_effective_observation_mask(
            point_samples.shape,
            None,
            observation_mask,
        )
        return _apply_observation_mask(point_samples, None, observation_mask), effective_mask

    n_manifest = linear_predictors.shape[1]
    response_kernel = _build_response_kernel(
        manifest_dist,
        manifest_dists,
        manifest_links,
        n_manifest,
        extra_params,
    )
    responses = jax.vmap(response_kernel.response_fn)(linear_predictors)
    expected_means, semantic_mask = observation_operator.project_response_trajectory(responses)

    interval_summary_indices = list(observation_operator.interval_summary_indices)
    interval_summary_idx = jnp.asarray(interval_summary_indices, dtype=jnp.int32)
    assert interval_summary_sampler is not None
    sampled_interval_summary = interval_summary_sampler.sample_mean_trajectory(
        key_interval_summary,
        expected_means[:, interval_summary_idx],
    )
    point_samples = jax.vmap(lambda y_t, sampled_t: y_t.at[interval_summary_idx].set(sampled_t))(
        point_samples,
        sampled_interval_summary,
    )

    effective_mask = _resolve_effective_observation_mask(
        point_samples.shape,
        semantic_mask,
        observation_mask,
    )
    return _apply_observation_mask(point_samples, semantic_mask, observation_mask), effective_mask


def simulate_posterior_predictive(
    samples: dict[str, jnp.ndarray],
    times: jnp.ndarray,
    diffusion_dist: DistributionFamily | str = "gaussian",
    diffusion_dists: list[DistributionFamily | str] | None = None,
    manifest_dist: str = "gaussian",
    manifest_dists: list[str] | None = None,
    manifest_links: list[str] | None = None,
    manifest_level_counts: list[int] | None = None,
    observation_support: ObservationSupportRuntime | None = None,
    observation_mask: jnp.ndarray | None = None,
    n_subsample: int = 50,
    rng_seed: int = 42,
    return_mask: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    """Forward-simulate observations from posterior draws.

    Args:
        samples: Posterior samples dict from InferenceResult.get_samples().
            Expected keys: "drift", "diffusion", "lambda", "manifest_cov",
            "t0_means", "t0_cov". Optional: "cint", "manifest_means",
            "obs_df", "obs_shape".
        times: (T,) observation times.
        diffusion_dist: Scalar process-noise family (fallback).
        diffusion_dists: Per-latent process-noise families. When provided,
            mixed diffusion simulation uses the declared per-latent shock family.
        manifest_dist: Scalar noise family string (fallback when manifest_dists
            is None or all channels share the same distribution).
        manifest_dists: Per-channel noise families. When provided and channels
            have different distributions, uses per-channel dispatch via
            jax.lax.switch. Overrides manifest_dist.
        manifest_links: Per-channel link function strings. When provided,
            used together with manifest_dists for combined dispatch.
        manifest_level_counts: Per-channel encoded category counts for
            ordered-logistic/categorical emissions.
        observation_support: Optional compiled observation-window semantics
            aligned to ``times``. When provided, interval-summary manifests are
            emitted only on their anchor rows using aggregated mean semantics.
        observation_mask: Optional boolean template aligned to ``times`` and
            manifests. False entries are set to NaN in the simulated output.
        n_subsample: Number of posterior draws to use.
        rng_seed: Random seed for simulation.

    Returns:
        y_sim: (n_subsample, T, n_manifest) simulated observations.
        When ``return_mask`` is true, also returns the explicit emitted-observation
        mask with the same shape.
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

    # Handle shared vs per-draw parameters using _broadcast_draw_param
    lambda_sub = _broadcast_draw_param(lambda_mat, n_use, indices)
    manifest_cov_sub = _broadcast_draw_param(manifest_cov, n_use, indices)
    t0_cov_sub = _broadcast_draw_param(t0_cov, n_use, indices)

    n_manifest = lambda_sub.shape[1]

    if manifest_means_draws is not None:
        manifest_means_sub = _broadcast_draw_param(manifest_means_draws, n_use, indices)
    else:
        manifest_means_sub = jnp.zeros((n_use, n_manifest))

    ordered_cutpoints_sub = _broadcast_draw_param(
        samples.get("obs_ordered_cutpoints"), n_use, indices
    )
    cat_intercepts_sub = _broadcast_draw_param(samples.get("obs_cat_intercepts"), n_use, indices)
    cat_slopes_sub = _broadcast_draw_param(samples.get("obs_cat_slopes"), n_use, indices)
    obs_df_sub = _broadcast_draw_param(samples.get("obs_df"), n_use, indices)
    proc_df_sub = _broadcast_draw_param(samples.get("proc_df"), n_use, indices)
    obs_shape_sub = _broadcast_draw_param(samples.get("obs_shape"), n_use, indices)
    obs_r_sub = _broadcast_draw_param(samples.get("obs_r"), n_use, indices)
    obs_conc_sub = _broadcast_draw_param(samples.get("obs_concentration"), n_use, indices)
    level_counts = (
        jnp.asarray(manifest_level_counts, dtype=jnp.int32)
        if manifest_level_counts is not None
        else None
    )

    n_timepoints = int(times.shape[0])
    transition_dt_array = jnp.maximum(jnp.diff(times), MIN_DT)
    transition_semantics = compile_transition_semantics(
        diffusion_dists or diffusion_dist, drift_sub.shape[-1]
    )

    rng = jax.random.PRNGKey(rng_seed)
    draw_keys = jax.random.split(rng, n_use)

    resolved_dists, _resolved_links = resolve_manifest_families_and_links(
        manifest_dist,
        n_manifest,
        manifest_dists=manifest_dists,
        manifest_links=manifest_links,
    )
    unique_dists = {dist.value for dist in resolved_dists}
    observation_operator = compile_observation_operator(observation_support)

    if level_counts is None and any_family_needs_level_metadata(unique_dists):
        raise ValueError(
            "manifest_level_counts is required for ordered_logistic/categorical PPC simulation"
        )

    observation_mask_array = None
    if observation_mask is not None:
        observation_mask_array = jnp.asarray(observation_mask, dtype=bool)
        if observation_mask_array.shape != (n_timepoints, n_manifest):
            raise ValueError(
                "observation_mask must have shape (T, n_manifest) matching the predictive grid"
            )

    # Cholesky decomposition of t0_cov (needed by all paths)
    t0_chol_sub = vmap(
        lambda cov: stable_cholesky(cov, min_eigenvalue=INITIAL_STATE_COV_MIN_EIGENVALUE)
    )(t0_cov_sub)

    if observation_operator.requires_interval_summary_handling:
        support = observation_operator.observation_support
        assert support is not None
        if support.anchor_times.shape != times.shape or not bool(
            jnp.allclose(jnp.asarray(support.anchor_times), times)
        ):
            raise ValueError("observation_support is not aligned to the predictive time grid")

    def _draw_extra_params(i: int) -> dict[str, jnp.ndarray | float]:
        extra_params: dict[str, jnp.ndarray | float] = {}
        if obs_df_sub is not None:
            extra_params["obs_df"] = obs_df_sub[i]
        if obs_shape_sub is not None:
            extra_params["obs_shape"] = obs_shape_sub[i]
        if obs_r_sub is not None:
            extra_params["obs_r"] = obs_r_sub[i]
        if obs_conc_sub is not None:
            extra_params["obs_concentration"] = obs_conc_sub[i]
        if level_counts is not None:
            extra_params["obs_level_counts"] = level_counts
        if ordered_cutpoints_sub is not None:
            extra_params["obs_ordered_cutpoints"] = ordered_cutpoints_sub[i]
        if cat_intercepts_sub is not None:
            extra_params["obs_cat_intercepts"] = cat_intercepts_sub[i]
        if cat_slopes_sub is not None:
            extra_params["obs_cat_slopes"] = cat_slopes_sub[i]
        return extra_params

    def sim_one(i):
        ci = cint_sub[i] if cint_sub is not None else None
        extra_params = _draw_extra_params(i)
        point_sampler = build_predictive_observation_sampler(
            manifest_dist=manifest_dist,
            manifest_cov=manifest_cov_sub[i],
            manifest_dists=manifest_dists,
            manifest_links=manifest_links,
            extra_params=extra_params,
        )
        interval_summary_sampler = None
        if observation_operator.requires_interval_summary_handling:
            interval_summary_indices = list(observation_operator.interval_summary_indices)
            interval_summary_idx = jnp.asarray(interval_summary_indices, dtype=jnp.int32)
            interval_summary_sampler = build_predictive_observation_sampler(
                manifest_dist=manifest_dist,
                manifest_cov=manifest_cov_sub[i][
                    jnp.ix_(interval_summary_idx, interval_summary_idx)
                ],
                manifest_dists=[
                    point_sampler.manifest_dists[idx] for idx in interval_summary_indices
                ],
                extra_params=_slice_extra_params_for_indices(
                    extra_params, interval_summary_indices
                ),
            )
        return _simulate_one_draw(
            drift=drift_sub[i],
            diffusion_chol=diffusion_sub[i],
            cint=ci,
            transition_semantics=transition_semantics,
            proc_df=proc_df_sub[i] if proc_df_sub is not None else 5.0,
            lambda_mat=lambda_sub[i],
            manifest_means=manifest_means_sub[i],
            t0_mean=t0_means_sub[i],
            t0_chol=t0_chol_sub[i],
            transition_dt_array=transition_dt_array,
            n_timepoints=n_timepoints,
            rng_key=draw_keys[i],
            manifest_dist=manifest_dist,
            manifest_dists=manifest_dists,
            manifest_links=manifest_links,
            point_sampler=point_sampler,
            interval_summary_sampler=interval_summary_sampler,
            observation_operator=observation_operator,
            observation_mask=observation_mask_array,
            extra_params=extra_params,
        )

    y_sim, y_mask = vmap(sim_one)(jnp.arange(n_use))

    if return_mask:
        return y_sim, y_mask
    return y_sim


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
    diffusion_dist: DistributionFamily | str = "gaussian",
    diffusion_dists: list[DistributionFamily | str] | None = None,
    manifest_dist: str = "gaussian",
    manifest_dists: list[str] | None = None,
    manifest_links: list[str] | None = None,
    manifest_level_counts: list[int] | None = None,
    observation_support: ObservationSupportRuntime | None = None,
    observation_mask: jnp.ndarray | None = None,
    n_subsample: int = 50,
    rng_seed: int = 42,
) -> PPCResult:
    """Run posterior predictive checks.

    Args:
        samples: Posterior samples from InferenceResult.get_samples()
        observations: (T, n_manifest) observed data
        times: (T,) observation times
        manifest_names: list of manifest variable names
        diffusion_dist: scalar process-noise family (fallback)
        diffusion_dists: per-latent process-noise families (overrides diffusion_dist)
        manifest_dist: scalar observation noise family (fallback)
        manifest_dists: per-channel noise families (overrides manifest_dist)
        manifest_links: per-channel link function strings
        manifest_level_counts: per-channel encoded category counts
        observation_support: optional compiled interval-summary semantics
        observation_mask: optional boolean observation schedule mask
        n_subsample: number of posterior draws to forward-simulate
        rng_seed: random seed

    Returns:
        PPCResult with diagnostics
    """
    y_sim = simulate_posterior_predictive(
        samples=samples,
        times=times,
        diffusion_dist=diffusion_dist,
        diffusion_dists=diffusion_dists,
        manifest_dist=manifest_dist,
        manifest_dists=manifest_dists,
        manifest_links=manifest_links,
        manifest_level_counts=manifest_level_counts,
        observation_support=observation_support,
        observation_mask=observation_mask,
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
