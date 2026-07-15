"""Posterior Predictive Checks (PPCs) for fitted CT-SSM models.

Forward-simulates observations from posterior parameter draws and compares
them to the real data, producing per-variable diagnostics that flag
calibration, autocorrelation, and variance issues.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

    from nof1_causal_lab.models.ssm.model import SSMSpec
    from nof1_causal_lab.models.ssm.observation_support import ObservationSupportRuntime

import jax.numpy as jnp
from pydantic import BaseModel, Field

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
# Main entry point
# ---------------------------------------------------------------------------


def run_posterior_predictive_checks(
    samples: dict[str, jnp.ndarray],
    observations: jnp.ndarray,
    times: jnp.ndarray,
    manifest_names: list[str],
    spec: SSMSpec,
    *,
    observation_support: ObservationSupportRuntime | None = None,
    observation_mask: jnp.ndarray | None = None,
    transition_inputs: jnp.ndarray | None = None,
    n_subsample: int = 50,
    rng_seed: int = 42,
) -> PPCResult:
    """Run posterior predictive checks.

    Forward-simulates ``n_subsample`` posterior draws through the *exact* nonlinear
    vector field (the same Diffrax simulator as prior predictive — never a
    linearised drift matrix) and compares them to the observed data.

    Args:
        samples: Posterior samples from InferenceResult.get_samples()
        observations: (T, n_manifest) observed data
        times: (T,) observation times
        manifest_names: list of manifest variable names
        spec: compiled SSM spec — provides the vector field and emission families
        observation_support: optional compiled interval-summary semantics
        observation_mask: optional boolean observation schedule mask
        transition_inputs: optional exogenous input schedule (input-driven models)
        n_subsample: number of posterior draws to forward-simulate
        rng_seed: random seed

    Returns:
        PPCResult with diagnostics
    """
    from nof1_causal_lab.models.ssm.predictive.registry_runtime import (
        simulate_posterior_predictive_observations,
    )

    y_sim, _y_mask = simulate_posterior_predictive_observations(
        spec,
        samples,
        times,
        observation_support=observation_support,
        observation_mask=observation_mask,
        transition_inputs=transition_inputs,
        n_subsample=n_subsample,
        seed=rng_seed,
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
