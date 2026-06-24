"""Parameter-recovery extraction from posterior samples.

Lifted out of the synthetic-nonlinear benchmark CLI so the recovery math is
reusable by the evaluation registry's recovery scorer rather than buried in an
83 KB script. ``parameter_recovery`` and ``scalar_posterior_ess`` are the public
entry points; both read an ``InferenceResult`` and compute per-target coverage /
ESS against the fixture's ``RECOVERY_TARGETS``.
"""

from __future__ import annotations

from typing import Any

import jax
import numpy as np

from evaluation.fixtures.synthetic_nonlinear import (
    MEASUREMENT_MEANS_FREE_POSITIONS,
    RECOVERY_TARGETS,
    TRUE_MANIFEST_SD,
)


def autocorrelation_ess_1d(draws: np.ndarray) -> float:
    values = np.asarray(draws, dtype=np.float64).reshape(-1)
    n = int(values.size)
    if n <= 1:
        return float(n)
    centered = values - float(np.mean(values))
    variance = float(np.dot(centered, centered) / n)
    if not np.isfinite(variance) or variance <= 0.0:
        return float(n)
    max_lag = min(n - 1, 512)
    rho_sum = 0.0
    for lag in range(1, max_lag + 1):
        autocov = float(np.dot(centered[:-lag], centered[lag:]) / (n - lag))
        rho = autocov / variance
        if not np.isfinite(rho) or rho <= 0.0:
            break
        rho_sum += rho
    return float(max(1.0, min(n, n / (1.0 + 2.0 * rho_sum))))


def lag1_autocorrelation_1d(draws: np.ndarray) -> float | None:
    values = np.asarray(draws, dtype=np.float64).reshape(-1)
    n = int(values.size)
    if n <= 1:
        return None
    centered = values - float(np.mean(values))
    variance = float(np.dot(centered, centered) / n)
    if not np.isfinite(variance) or variance <= 0.0:
        return 0.0
    autocov = float(np.dot(centered[:-1], centered[1:]) / (n - 1))
    return autocov / variance


def _recovery_target_location(label: str, target: Any) -> tuple[str, tuple[int, ...], float]:
    if isinstance(target, dict):
        site = str(target["site"])
        raw_index = target["index"]
        if isinstance(raw_index, (list, tuple)):
            index = tuple(int(item) for item in raw_index)
        else:
            index = (int(raw_index),)
        return site, index, float(target["true"])
    return label, (), float(target)


def _recovery_family(label: str) -> str:
    if label.startswith("input_"):
        return "input_effect"
    if label.startswith("manifest_mean_"):
        return "manifest_mean"
    if label.startswith("loading_"):
        return "loading"
    if label.startswith("obs_"):
        return "observation"
    return "dynamics_or_nonlinear"


def _recovery_target_scale(label: str, target: Any, true_value: float) -> float:
    if isinstance(target, dict) and "scale" in target:
        return max(abs(float(target["scale"])), 1e-12)
    if label.startswith("manifest_mean_") and isinstance(target, dict):
        free_index = int(target["index"])
        manifest_index = int(MEASUREMENT_MEANS_FREE_POSITIONS[free_index])
        return max(float(TRUE_MANIFEST_SD[manifest_index]), 1e-12)
    return max(abs(float(true_value)), 1.0)


def _target_draws(grouped_samples: dict[str, Any], site: str, index: tuple[int, ...]):
    if site not in grouped_samples:
        return None
    draws = np.asarray(jax.device_get(grouped_samples[site]))
    if index:
        draws = draws[(..., *index)]
    return draws


def _summarize_recovery_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "target_count": 0,
            "mean_abs_error": None,
            "median_abs_error": None,
            "max_abs_error": None,
            "mean_scale_adjusted_abs_error": None,
            "median_scale_adjusted_abs_error": None,
            "max_scale_adjusted_abs_error": None,
            "coverage_90": None,
            "ess_approx_min": None,
            "ess_approx_median": None,
            "ess_approx_mean": None,
        }
    abs_errors = np.asarray([row["abs_error"] for row in rows], dtype=np.float64)
    scaled_errors = np.asarray(
        [row["scale_adjusted_abs_error"] for row in rows],
        dtype=np.float64,
    )
    ess_values = np.asarray([row["ess_approx"] for row in rows], dtype=np.float64)
    coverage = np.asarray([row["covered_90"] for row in rows], dtype=np.float64)
    return {
        "target_count": len(rows),
        "mean_abs_error": float(np.mean(abs_errors)),
        "median_abs_error": float(np.median(abs_errors)),
        "max_abs_error": float(np.max(abs_errors)),
        "mean_scale_adjusted_abs_error": float(np.mean(scaled_errors)),
        "median_scale_adjusted_abs_error": float(np.median(scaled_errors)),
        "max_scale_adjusted_abs_error": float(np.max(scaled_errors)),
        "coverage_90": float(np.mean(coverage)),
        "ess_approx_min": float(np.min(ess_values)),
        "ess_approx_median": float(np.median(ess_values)),
        "ess_approx_mean": float(np.mean(ess_values)),
    }


def parameter_recovery(result, *, elapsed_seconds: float) -> dict[str, Any]:
    grouped_samples = result.diagnostics["mcmc"].get_samples(group_by_chain=True)
    site_rows: dict[str, Any] = {}
    missing_targets: dict[str, Any] = {}
    by_family_rows: dict[str, list[dict[str, Any]]] = {}
    for label, target in RECOVERY_TARGETS.items():
        site, index, true_value = _recovery_target_location(label, target)
        draws = _target_draws(grouped_samples, site, index)
        if draws is None:
            missing_targets[label] = {"site": site, "index": index}
            continue
        if draws.size == 0:
            continue
        mean = float(np.mean(draws))
        std = float(np.std(draws))
        q05, q50, q95 = np.quantile(np.asarray(draws, dtype=np.float64), [0.05, 0.5, 0.95])
        abs_error = abs(mean - true_value)
        scale = _recovery_target_scale(label, target, true_value)
        family = _recovery_family(label)
        ess = autocorrelation_ess_1d(draws)
        lag1 = lag1_autocorrelation_1d(draws)
        row = {
            "family": family,
            "site": site,
            "index": index,
            "true": true_value,
            "target_scale": scale,
            "mean": mean,
            "median": float(q50),
            "std": std,
            "q05": float(q05),
            "q95": float(q95),
            "covered_90": float(q05 <= true_value <= q95),
            "abs_error": abs_error,
            "relative_abs_error": abs_error / max(abs(true_value), 1e-12),
            "scale_adjusted_abs_error": abs_error / scale,
            "posterior_z_abs_error": None if std <= 0.0 else abs_error / std,
            "ess_approx": ess,
            "ess_per_second": ess / max(float(elapsed_seconds), 1e-12),
            "lag1_autocorr": lag1,
        }
        site_rows[label] = row
        by_family_rows.setdefault(family, []).append(row)

    rows = list(site_rows.values())
    return {
        "note": (
            "Recovery is computed for every synthetic nonlinear target from retained "
            "posterior samples. target_scale is the observation-noise SD for manifest "
            "means and max(abs(true), 1) otherwise; use family summaries and "
            "scale_adjusted_abs_error when comparing heterogeneous parameter units."
        ),
        "target_count": len(RECOVERY_TARGETS),
        "site_count": len(site_rows),
        "missing_targets": missing_targets,
        "summary": _summarize_recovery_rows(rows),
        "by_family": {
            family: _summarize_recovery_rows(family_rows)
            for family, family_rows in sorted(by_family_rows.items())
        },
        "sites": site_rows,
    }


def scalar_posterior_ess(
    result,
    *,
    max_sites: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    grouped_samples = result.diagnostics["mcmc"].get_samples(group_by_chain=True)
    site_rows: dict[str, Any] = {}
    for label, target in list(RECOVERY_TARGETS.items())[:max_sites]:
        if isinstance(target, dict):
            site = str(target["site"])
            if site not in grouped_samples:
                continue
            draws = np.asarray(jax.device_get(grouped_samples[site]))[..., int(target["index"])]
        else:
            site = label
            if site not in grouped_samples:
                continue
            draws = np.asarray(jax.device_get(grouped_samples[site]))
        if draws.size == 0:
            continue
        ess = autocorrelation_ess_1d(draws)
        site_rows[label] = {
            "site": site,
            "ess_approx": ess,
            "ess_per_second": ess / max(float(elapsed_seconds), 1e-12),
            "mean": float(np.mean(draws)),
            "std": float(np.std(draws)),
        }

    ess_values = np.asarray(
        [row["ess_approx"] for row in site_rows.values()],
        dtype=np.float64,
    )
    return {
        "note": (
            "ess_approx is a simple initial-positive-sequence autocorrelation "
            "estimate over the retained posterior samples. It is a smoke-check "
            "metric, not a replacement for full ArviZ diagnostics."
        ),
        "site_count": len(site_rows),
        "ess_approx_min": None if ess_values.size == 0 else float(np.min(ess_values)),
        "ess_approx_median": None if ess_values.size == 0 else float(np.median(ess_values)),
        "ess_approx_max": None if ess_values.size == 0 else float(np.max(ess_values)),
        "sites": site_rows,
    }
