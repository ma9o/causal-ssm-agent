"""Prod-like MPGibbs inference check on the nonlinear DEMO-like fixture.

This script intentionally reuses ``scratchpad/demo_like_synthetic.py`` so the
stress model stays identical to the benchmarking notebooks: three latent states,
mixed Gaussian/count/gamma observations, transition inputs, Hill saturation, and
multiplicative nonlinear drift terms.

It runs the production ``fit(..., method="marginal_particle_gibbs")`` path for
the requested latent smoothers and writes a JSON trace artifact with per-sample
acceptance, movement, label probabilities, derived label ESS, complete log
posterior history, particle ESS, backward-selection entropy, reference-path hit
rates, latent-path ESS approximations, and aMALA proposal diagnostics.

Example:
    uv run python scripts/check_mpg_smoothers_demo_like.py
    uv run python scripts/check_mpg_smoothers_demo_like.py --support-mode both --T 24

The default point-support mode is the nonlinear smoother smoke check. Interval
mode is an explicit caveat probe for the same fixture and currently exercises
the backend's affine-dynamics restriction before the smoother is reached.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "apps/data-pipeline/pyproject.toml").exists():
            return candidate
    raise RuntimeError("Could not locate repository root")


REPO_ROOT = _repo_root()
for import_path in (
    REPO_ROOT / "apps/data-pipeline/src",
    REPO_ROOT / "apps/data-pipeline",
    REPO_ROOT / "scratchpad",
):
    path = str(import_path)
    if path not in sys.path:
        sys.path.insert(0, path)


def _configure_jax_cache() -> None:
    cache_dir = REPO_ROOT / "scratchpad" / ".jax_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(cache_dir)
    os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "true"
    os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
    os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "0"


_configure_jax_cache()

import jax
import jax.numpy as jnp
import numpy as np
from demo_like_synthetic import (
    RECOVERY_TARGETS,
    TRUE_HILL_BY_SITE,
    TRUE_MULTIPLICATIVE_BY_SITE,
    build_demo_like_synthetic_model,
    simulate_demo_like_synthetic_data,
)

from nof1_causal_lab.models.ssm.inference import fit
from nof1_causal_lab.models.ssm.inference.trajectory_mcmc.marginal_particle_gibbs.diagnostics import (
    MPGIBBS_DIAGNOSTIC_METRIC_VALUES,
    MPGibbsDiagnosticMetric,
)

EXPECTED_SMOOTHER_SELECTION = {
    "plain": "blocked_backward_sampling",
    "amala": "augmented_target_backward_sampling",
}
LATENT_PATH_MIXING_METRIC = "latent_path_mixing"
SCRIPT_DIAGNOSTIC_METRIC_VALUES = (
    *MPGIBBS_DIAGNOSTIC_METRIC_VALUES,
    LATENT_PATH_MIXING_METRIC,
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return np.asarray(jax.device_get(value)).tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _finite_summary(name: str, values: Any) -> dict[str, Any]:
    array = np.asarray(jax.device_get(values))
    finite = np.isfinite(array)
    if not bool(np.all(finite)):
        bad_count = int(array.size - int(np.count_nonzero(finite)))
        raise AssertionError(f"{name} contains {bad_count} non-finite values")
    if array.size == 0:
        return {"shape": list(array.shape), "mean": None, "std": None}
    return {
        "shape": list(array.shape),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _autocorrelation_ess_1d(draws: np.ndarray) -> float:
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


def _lag1_autocorrelation_1d(draws: np.ndarray) -> float | None:
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


def _scalar_posterior_ess(
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
        ess = _autocorrelation_ess_1d(draws)
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


def _block_summaries(values: Any, *, block_size: int) -> dict[str, Any]:
    array = np.asarray(jax.device_get(values), dtype=np.float64)
    if array.ndim < 3:
        raise AssertionError(f"expected chains x samples x time values, got {array.shape}")
    time_count = int(array.shape[-1])
    rows = []
    for block_start in range(0, time_count, block_size):
        block_end = min(block_start + block_size, time_count)
        block_values = array[..., block_start:block_end]
        rows.append(
            {
                "block_index": block_start // block_size,
                "time_start": block_start,
                "time_end_exclusive": block_end,
                "min": float(np.min(block_values)),
                "mean": float(np.mean(block_values)),
                "max": float(np.max(block_values)),
            }
        )
    if not rows:
        return {"block_count": 0, "blocks": []}
    worst = min(rows, key=lambda row: row["min"])
    return {
        "block_count": len(rows),
        "worst_block_by_min": worst["block_index"],
        "blocks": rows,
    }


def _particle_smoother_health(
    extra_fields: dict[str, Any],
    *,
    block_size: int,
    selected_metrics: frozenset[str],
) -> dict[str, Any]:
    health: dict[str, Any] = {
        "enabled": bool(
            selected_metrics
            & {
                MPGibbsDiagnosticMetric.PARTICLE_FILTER.value,
                MPGibbsDiagnosticMetric.BACKWARD_SELECTION.value,
                MPGibbsDiagnosticMetric.PARTICLE_IDENTITY.value,
            }
        )
    }
    if MPGibbsDiagnosticMetric.PARTICLE_FILTER.value in selected_metrics:
        health["forward_particle_ess"] = {
            "summary": _finite_summary(
                "forward_particle_ess_by_t",
                extra_fields["forward_particle_ess_by_t"],
            ),
            "by_block": _block_summaries(
                extra_fields["forward_particle_ess_by_t"],
                block_size=block_size,
            ),
        }
        health["forward_log_weight_range"] = {
            "summary": _finite_summary(
                "forward_log_weight_range_by_t",
                extra_fields["forward_log_weight_range_by_t"],
            ),
            "by_block": _block_summaries(
                extra_fields["forward_log_weight_range_by_t"],
                block_size=block_size,
            ),
        }
        health["forward_log_weight_variance"] = {
            "summary": _finite_summary(
                "forward_log_weight_variance_by_t",
                extra_fields["forward_log_weight_variance_by_t"],
            ),
            "by_block": _block_summaries(
                extra_fields["forward_log_weight_variance_by_t"],
                block_size=block_size,
            ),
        }
    if MPGibbsDiagnosticMetric.BACKWARD_SELECTION.value in selected_metrics:
        health["backward_selection_ess"] = {
            "summary": _finite_summary(
                "backward_selection_ess_by_t",
                extra_fields["backward_selection_ess_by_t"],
            ),
            "by_block": _block_summaries(
                extra_fields["backward_selection_ess_by_t"],
                block_size=block_size,
            ),
        }
        health["backward_selection_entropy"] = {
            "summary": _finite_summary(
                "backward_selection_entropy_by_t",
                extra_fields["backward_selection_entropy_by_t"],
            ),
            "by_block": _block_summaries(
                extra_fields["backward_selection_entropy_by_t"],
                block_size=block_size,
            ),
        }
        health["backward_selection_max_prob"] = _finite_summary(
            "backward_selection_max_prob_by_t",
            extra_fields["backward_selection_max_prob_by_t"],
        )
    if MPGibbsDiagnosticMetric.PARTICLE_IDENTITY.value in selected_metrics:
        selected_by_t = np.asarray(
            jax.device_get(extra_fields["selected_particle_per_t"]),
            dtype=np.float64,
        )
        reference_hit_by_t = np.mean(selected_by_t == 0.0, axis=(0, 1))
        health["reference_path_hit_rate"] = _finite_summary(
            "reference_path_hit_rate",
            extra_fields["reference_path_hit_rate"],
        )
        health["reference_path_hit_rate_by_t"] = reference_hit_by_t.tolist()
        health["selected_particle_unique_count"] = _finite_summary(
            "selected_particle_unique_count",
            extra_fields["selected_particle_unique_count"],
        )
    return health


def _latent_path_mixing(
    result,
    *,
    max_sites: int,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {
            "enabled": False,
            "note": f"Enable {LATENT_PATH_MIXING_METRIC!r} to retain latent paths and compute latent ESS.",
        }
    latent_paths = result.diagnostics.get("latent_paths")
    if latent_paths is None:
        return {
            "enabled": False,
            "note": "Run with retained latent paths to compute latent-path ESS.",
        }
    paths = np.asarray(jax.device_get(latent_paths), dtype=np.float64)
    if paths.ndim != 4:
        raise AssertionError(
            f"expected latent paths as chains x samples x time x dim, got {paths.shape}"
        )
    _chains, _samples, time_count, latent_dim = paths.shape
    candidate_times = tuple(dict.fromkeys((0, time_count // 2, time_count - 1)))
    site_rows: dict[str, Any] = {}
    for time_idx in candidate_times:
        for dim_idx in range(latent_dim):
            if len(site_rows) >= max_sites:
                break
            site_key = f"t{time_idx}_dim{dim_idx}"
            chain_draws = paths[:, :, time_idx, dim_idx]
            ess_by_chain = [_autocorrelation_ess_1d(draws) for draws in chain_draws]
            lag1_by_chain = [_lag1_autocorrelation_1d(draws) for draws in chain_draws]
            finite_lag1 = [value for value in lag1_by_chain if value is not None]
            site_rows[site_key] = {
                "time_index": time_idx,
                "latent_dim": dim_idx,
                "ess_approx_min_chain": float(np.min(ess_by_chain)),
                "ess_approx_mean_chain": float(np.mean(ess_by_chain)),
                "lag1_autocorr_mean_chain": (
                    None if not finite_lag1 else float(np.mean(finite_lag1))
                ),
                "mean": float(np.mean(chain_draws)),
                "std": float(np.std(chain_draws)),
            }
        if len(site_rows) >= max_sites:
            break
    ess_values = np.asarray(
        [row["ess_approx_min_chain"] for row in site_rows.values()],
        dtype=np.float64,
    )
    return {
        "enabled": True,
        "scope": "retained public latent paths at first/middle/final times",
        "site_count": len(site_rows),
        "ess_approx_min": None if ess_values.size == 0 else float(np.min(ess_values)),
        "ess_approx_median": None if ess_values.size == 0 else float(np.median(ess_values)),
        "ess_approx_max": None if ess_values.size == 0 else float(np.max(ess_values)),
        "sites": site_rows,
    }


def _label_probability_diagnostics(label_log_probs: jnp.ndarray) -> dict[str, Any]:
    label_probs = jax.nn.softmax(label_log_probs, axis=-1)
    label_ess = 1.0 / jnp.sum(label_probs * label_probs, axis=-1)
    label_entropy = -jnp.sum(
        jnp.where(label_probs > 0.0, label_probs * jnp.log(label_probs), 0.0),
        axis=-1,
    )
    label_max_prob = jnp.max(label_probs, axis=-1)
    return {
        "final_label_probs": label_probs,
        "final_label_ess": label_ess,
        "final_label_entropy": label_entropy,
        "final_label_max_prob": label_max_prob,
    }


def _per_step_trace(
    result,
    *,
    smoother: str,
    selected_metrics: frozenset[str],
) -> dict[str, Any]:
    extra_fields = result.diagnostics["mcmc"].get_extra_fields(group_by_chain=True)
    complete_lp = jnp.asarray(result.diagnostics["chain_complete_log_posterior_history"])
    final_label_log_probs = jnp.asarray(extra_fields["final_label_log_probs"])
    label_diagnostics = _label_probability_diagnostics(final_label_log_probs)
    fields = {
        "parameter_accept_prob": extra_fields["parameter_accept_prob"],
        "latent_accept_prob": extra_fields["latent_accept_prob"],
        "selected_parameter_label": extra_fields["selected_parameter_label"],
        "selected_particle": extra_fields["selected_particle"],
        "latent_move_rms": extra_fields["latent_move_rms"],
        "latent_move_max_abs": extra_fields["latent_move_max_abs"],
        "latent_move_rms_per_t": extra_fields["latent_move_rms_per_t"],
        "complete_log_posterior": complete_lp,
        "final_label_log_probs": final_label_log_probs,
        **label_diagnostics,
    }
    if MPGibbsDiagnosticMetric.PARTICLE_IDENTITY.value in selected_metrics:
        fields["selected_particle_per_t"] = extra_fields["selected_particle_per_t"]
        fields["reference_path_hit_rate"] = extra_fields["reference_path_hit_rate"]
        fields["selected_particle_unique_count"] = extra_fields["selected_particle_unique_count"]
    if MPGibbsDiagnosticMetric.PARAMETER_MOVEMENT.value in selected_metrics:
        fields["parameter_jump_rms"] = extra_fields["parameter_jump_rms"]
    if MPGibbsDiagnosticMetric.PARTICLE_FILTER.value in selected_metrics:
        fields["forward_particle_ess_by_t"] = extra_fields["forward_particle_ess_by_t"]
        fields["forward_log_weight_range_by_t"] = extra_fields["forward_log_weight_range_by_t"]
        fields["forward_log_weight_variance_by_t"] = extra_fields[
            "forward_log_weight_variance_by_t"
        ]
    if MPGibbsDiagnosticMetric.BACKWARD_SELECTION.value in selected_metrics:
        fields["backward_selection_ess_by_t"] = extra_fields["backward_selection_ess_by_t"]
        fields["backward_selection_entropy_by_t"] = extra_fields["backward_selection_entropy_by_t"]
        fields["backward_selection_max_prob_by_t"] = extra_fields[
            "backward_selection_max_prob_by_t"
        ]
    if smoother == "amala":
        fields["amala_grad_norm_mean"] = extra_fields["amala_grad_norm_mean"]
        fields["amala_grad_norm_max"] = extra_fields["amala_grad_norm_max"]
    if smoother == "amala" and MPGibbsDiagnosticMetric.AMALA_PROPOSAL.value in selected_metrics:
        fields["amala_grad_clip_fraction"] = extra_fields["amala_grad_clip_fraction"]
        fields["amala_drift_norm_mean"] = extra_fields["amala_drift_norm_mean"]
        fields["amala_drift_norm_max"] = extra_fields["amala_drift_norm_max"]
        fields["amala_auxiliary_noise_norm_mean"] = extra_fields["amala_auxiliary_noise_norm_mean"]
        fields["amala_auxiliary_noise_norm_max"] = extra_fields["amala_auxiliary_noise_norm_max"]
        fields["amala_drift_to_auxiliary_noise_ratio_mean"] = extra_fields[
            "amala_drift_to_auxiliary_noise_ratio_mean"
        ]
        fields["amala_proposal_displacement_norm_mean"] = extra_fields[
            "amala_proposal_displacement_norm_mean"
        ]
        fields["amala_proposal_displacement_norm_max"] = extra_fields[
            "amala_proposal_displacement_norm_max"
        ]
        fields["amala_auxiliary_correction_variance"] = extra_fields[
            "amala_auxiliary_correction_variance"
        ]
        fields["amala_auxiliary_correction_max_abs"] = extra_fields[
            "amala_auxiliary_correction_max_abs"
        ]

    summaries = {
        name: _finite_summary(f"per_step[{name}]", values) for name, values in fields.items()
    }
    return {
        "scope": "post_warmup_retained_samples",
        "shape_convention": "chains x samples for scalar fields; chains x samples x ... otherwise",
        "fields": {name: _json_ready(values) for name, values in fields.items()},
        "summaries": summaries,
    }


def _check_label_log_probs(extra_fields: dict[str, Any]) -> dict[str, Any]:
    label_log_probs = jnp.asarray(extra_fields["final_label_log_probs"])
    log_norm = jax.scipy.special.logsumexp(label_log_probs, axis=-1)
    max_abs_error = float(jnp.max(jnp.abs(log_norm)))
    if not np.isfinite(max_abs_error) or max_abs_error > 5e-5:
        raise AssertionError(
            f"final_label_log_probs are not normalized; max |logsumexp|={max_abs_error:.3g}"
        )
    selected_label = jnp.asarray(extra_fields["selected_parameter_label"])
    return {
        "final_label_log_probs": _finite_summary(
            "final_label_log_probs",
            label_log_probs,
        ),
        "final_label_log_probs_max_abs_log_norm_error": max_abs_error,
        "selected_parameter_label": _finite_summary(
            "selected_parameter_label",
            selected_label,
        ),
    }


def _check_result(
    result,
    *,
    smoother: str,
    elapsed_seconds: float,
    ess_max_sites: int,
    selected_metrics: frozenset[str],
) -> dict[str, Any]:
    diagnostics = result.diagnostics["marginal_particle_gibbs"]
    if diagnostics["latent_smoother"] != smoother:
        raise AssertionError(
            f"expected smoother {smoother!r}, got {diagnostics['latent_smoother']!r}"
        )
    expected_selection = EXPECTED_SMOOTHER_SELECTION[smoother]
    if diagnostics["latent_smoother_selection"] != expected_selection:
        raise AssertionError(
            "unexpected smoother selection for "
            f"{smoother!r}: {diagnostics['latent_smoother_selection']!r}"
        )

    sample_summaries = {
        name: _finite_summary(f"samples[{name}]", values)
        for name, values in result.get_samples().items()
    }
    extra_fields = result.diagnostics["mcmc"].get_extra_fields(group_by_chain=True)
    extra_summaries = {
        "parameter_accept_prob": _finite_summary(
            "parameter_accept_prob",
            extra_fields["parameter_accept_prob"],
        ),
        "latent_accept_prob": _finite_summary(
            "latent_accept_prob",
            extra_fields["latent_accept_prob"],
        ),
        "latent_move_rms": _finite_summary(
            "latent_move_rms",
            extra_fields["latent_move_rms"],
        ),
        **_check_label_log_probs(extra_fields),
    }
    if MPGibbsDiagnosticMetric.PARAMETER_MOVEMENT.value in selected_metrics:
        extra_summaries["parameter_jump_rms"] = _finite_summary(
            "parameter_jump_rms",
            extra_fields["parameter_jump_rms"],
        )
    if MPGibbsDiagnosticMetric.PARTICLE_FILTER.value in selected_metrics:
        extra_summaries["forward_particle_ess_by_t"] = _finite_summary(
            "forward_particle_ess_by_t",
            extra_fields["forward_particle_ess_by_t"],
        )
    if MPGibbsDiagnosticMetric.BACKWARD_SELECTION.value in selected_metrics:
        extra_summaries["backward_selection_ess_by_t"] = _finite_summary(
            "backward_selection_ess_by_t",
            extra_fields["backward_selection_ess_by_t"],
        )
    if MPGibbsDiagnosticMetric.PARTICLE_IDENTITY.value in selected_metrics:
        extra_summaries["reference_path_hit_rate"] = _finite_summary(
            "reference_path_hit_rate",
            extra_fields["reference_path_hit_rate"],
        )
        extra_summaries["selected_particle_unique_count"] = _finite_summary(
            "selected_particle_unique_count",
            extra_fields["selected_particle_unique_count"],
        )
    if smoother == "amala":
        extra_summaries["amala_grad_norm_mean"] = _finite_summary(
            "amala_grad_norm_mean",
            extra_fields["amala_grad_norm_mean"],
        )
        extra_summaries["amala_grad_norm_max"] = _finite_summary(
            "amala_grad_norm_max",
            extra_fields["amala_grad_norm_max"],
        )
        if diagnostics["amala_grad_norm_max"] < diagnostics["amala_grad_norm_mean"]:
            raise AssertionError("amala_grad_norm_max is smaller than amala_grad_norm_mean")
    if smoother == "amala" and MPGibbsDiagnosticMetric.AMALA_PROPOSAL.value in selected_metrics:
        extra_summaries["amala_grad_clip_fraction"] = _finite_summary(
            "amala_grad_clip_fraction",
            extra_fields["amala_grad_clip_fraction"],
        )
        extra_summaries["amala_auxiliary_correction_variance"] = _finite_summary(
            "amala_auxiliary_correction_variance",
            extra_fields["amala_auxiliary_correction_variance"],
        )

    diagnostic_summary = {
        "latent_kernel": diagnostics["latent_kernel"],
        "latent_smoother": diagnostics["latent_smoother"],
        "latent_smoother_selection": diagnostics["latent_smoother_selection"],
        "parameter_kernel": diagnostics["parameter_kernel"],
        "parameter_accept_rate": diagnostics["parameter_accept_rate"],
        "latent_update_fraction": diagnostics["latent_update_fraction"],
        "latent_move_rms_mean": diagnostics.get("latent_move_rms_mean"),
        "mcmc_phase_seconds": diagnostics["mcmc_phase_seconds"],
    }
    if MPGibbsDiagnosticMetric.PARAMETER_MOVEMENT.value in selected_metrics:
        diagnostic_summary["parameter_jump_rms_mean"] = diagnostics.get("parameter_jump_rms_mean")
    if MPGibbsDiagnosticMetric.PARTICLE_IDENTITY.value in selected_metrics:
        diagnostic_summary["reference_path_hit_rate_mean"] = diagnostics.get(
            "reference_path_hit_rate_mean"
        )
        diagnostic_summary["selected_particle_unique_count_mean"] = diagnostics.get(
            "selected_particle_unique_count_mean"
        )
    if MPGibbsDiagnosticMetric.PARTICLE_FILTER.value in selected_metrics:
        diagnostic_summary["forward_particle_ess_min"] = diagnostics.get("forward_particle_ess_min")
        diagnostic_summary["forward_particle_ess_mean"] = diagnostics.get(
            "forward_particle_ess_mean"
        )
        diagnostic_summary["forward_log_weight_range_max"] = diagnostics.get(
            "forward_log_weight_range_max"
        )
        diagnostic_summary["forward_log_weight_variance_mean"] = diagnostics.get(
            "forward_log_weight_variance_mean"
        )
    if MPGibbsDiagnosticMetric.BACKWARD_SELECTION.value in selected_metrics:
        diagnostic_summary["backward_selection_ess_min"] = diagnostics.get(
            "backward_selection_ess_min"
        )
        diagnostic_summary["backward_selection_ess_mean"] = diagnostics.get(
            "backward_selection_ess_mean"
        )
        diagnostic_summary["backward_selection_entropy_mean"] = diagnostics.get(
            "backward_selection_entropy_mean"
        )
        diagnostic_summary["backward_selection_max_prob_mean"] = diagnostics.get(
            "backward_selection_max_prob_mean"
        )
    if smoother == "amala":
        diagnostic_summary.update(
            {
                "amala_grad_norm_mean": diagnostics.get("amala_grad_norm_mean"),
                "amala_grad_norm_max": diagnostics.get("amala_grad_norm_max"),
            }
        )
    if smoother == "amala" and MPGibbsDiagnosticMetric.AMALA_PROPOSAL.value in selected_metrics:
        diagnostic_summary.update(
            {
                "amala_grad_clip_fraction_mean": diagnostics.get("amala_grad_clip_fraction_mean"),
                "amala_drift_norm_mean": diagnostics.get("amala_drift_norm_mean"),
                "amala_auxiliary_noise_norm_mean": diagnostics.get(
                    "amala_auxiliary_noise_norm_mean"
                ),
                "amala_drift_to_auxiliary_noise_ratio_mean": diagnostics.get(
                    "amala_drift_to_auxiliary_noise_ratio_mean"
                ),
                "amala_proposal_displacement_norm_mean": diagnostics.get(
                    "amala_proposal_displacement_norm_mean"
                ),
                "amala_auxiliary_correction_variance_mean": diagnostics.get(
                    "amala_auxiliary_correction_variance_mean"
                ),
                "amala_auxiliary_correction_max_abs": diagnostics.get(
                    "amala_auxiliary_correction_max_abs"
                ),
            }
        )

    return {
        "diagnostics": diagnostic_summary,
        "per_step_trace": _per_step_trace(
            result,
            smoother=smoother,
            selected_metrics=selected_metrics,
        ),
        "particle_smoother_health": _particle_smoother_health(
            extra_fields,
            block_size=int(diagnostics["latent_block_size"]),
            selected_metrics=selected_metrics,
        ),
        "posterior_ess": _scalar_posterior_ess(
            result,
            max_sites=ess_max_sites,
            elapsed_seconds=elapsed_seconds,
        ),
        "latent_path_mixing": _latent_path_mixing(
            result,
            max_sites=ess_max_sites,
            enabled=LATENT_PATH_MIXING_METRIC in selected_metrics,
        ),
        "sample_summaries": sample_summaries,
        "extra_summaries": extra_summaries,
    }


def _run_one(args, *, smoother: str, support_mode: str, seed: int) -> dict[str, Any]:
    selected_metrics = _selected_diagnostic_metrics(args)
    production_metrics = _production_diagnostic_metrics(selected_metrics)
    retain_latent_paths = args.retain_latent_paths or (
        LATENT_PATH_MIXING_METRIC in selected_metrics
    )
    include_interval_support = support_mode == "interval"
    setup_started = time.perf_counter()
    data = simulate_demo_like_synthetic_data(T=args.T, seed=args.data_seed)
    model = build_demo_like_synthetic_model(
        data,
        include_interval_support=include_interval_support,
    )
    setup_seconds = time.perf_counter() - setup_started

    fit_started = time.perf_counter()
    result = fit(
        model,
        observations=data.observations,
        times=data.times,
        method="marginal_particle_gibbs",
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        seed=seed,
        n_particles=args.n_particles,
        n_parameter_particles=args.n_parameter_particles,
        latent_block_size=args.latent_block_size,
        latent_smoother=smoother,
        amala_q_scale=args.amala_q_scale,
        amala_kappa=args.amala_kappa,
        amala_grad_clip=args.amala_grad_clip,
        param_step_size=args.param_step_size,
        parameter_proposal=args.parameter_proposal,
        adaptation_scheme="simple",
        init_method=args.init_method,
        latent_init_method="predictive",
        auto_preconditioner_method="none",
        init_scale=0.0,
        retain_latent_paths=retain_latent_paths,
        compute_latent_posterior_summary=not args.skip_latent_summary,
        diagnostic_metrics_all=args.all_metrics,
        diagnostic_metrics=production_metrics,
        pathfinder_num_elbo_samples=args.pathfinder_num_elbo_samples,
        pathfinder_maxiter=args.pathfinder_maxiter,
        n_pathfinder_starts=args.n_pathfinder_starts,
        pathfinder_init_scale=args.pathfinder_init_scale,
        reparam=None,
    )
    fit_seconds = time.perf_counter() - fit_started

    check_started = time.perf_counter()
    checked = _check_result(
        result,
        smoother=smoother,
        elapsed_seconds=fit_seconds,
        ess_max_sites=args.ess_max_sites,
        selected_metrics=selected_metrics,
    )
    check_seconds = time.perf_counter() - check_started
    checked["elapsed_seconds"] = float(setup_seconds + fit_seconds + check_seconds)
    checked["pipeline_steps"] = {
        "fixture_and_model_setup_seconds": float(setup_seconds),
        "production_fit_seconds": float(fit_seconds),
        "trace_validation_seconds": float(check_seconds),
    }
    checked["support_mode"] = support_mode
    checked["seed"] = int(seed)
    return checked


def _parse_smoothers(raw: str) -> tuple[str, ...]:
    smoothers = tuple(item.strip() for item in raw.split(",") if item.strip())
    allowed = set(EXPECTED_SMOOTHER_SELECTION)
    unknown = sorted(set(smoothers) - allowed)
    if unknown:
        raise ValueError(f"unknown smoothers: {unknown}; allowed: {sorted(allowed)}")
    if not smoothers:
        raise ValueError("at least one smoother is required")
    return smoothers


def _support_modes(raw: str) -> tuple[str, ...]:
    if raw == "both":
        return ("point", "interval")
    return (raw,)


def _selected_diagnostic_metrics(args: argparse.Namespace) -> frozenset[str]:
    if args.all_metrics:
        return frozenset(SCRIPT_DIAGNOSTIC_METRIC_VALUES)
    return frozenset(args.metric or ())


def _production_diagnostic_metrics(selected_metrics: frozenset[str]) -> tuple[str, ...]:
    return tuple(
        metric for metric in MPGIBBS_DIAGNOSTIC_METRIC_VALUES if metric in selected_metrics
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run MPGibbs plain CSMC and Particle-aMALA on the nonlinear "
            "DEMO-like scratchpad fixture."
        )
    )
    parser.add_argument("--T", type=int, default=32)
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--num-chains", type=int, default=1)
    parser.add_argument("--n-particles", type=int, default=8)
    parser.add_argument("--n-parameter-particles", type=int, default=2)
    parser.add_argument("--latent-block-size", type=int, default=16)
    parser.add_argument("--data-seed", type=int, default=71)
    parser.add_argument("--seed", type=int, default=1009)
    parser.add_argument("--param-step-size", type=float, default=0.001)
    parser.add_argument(
        "--parameter-proposal",
        choices=("random_walk", "pseudo_langevin"),
        default="random_walk",
    )
    parser.add_argument("--amala-q-scale", type=float, default=0.25)
    parser.add_argument("--amala-kappa", type=float, default=0.25)
    parser.add_argument("--amala-grad-clip", type=float, default=100.0)
    parser.add_argument(
        "--init-method",
        choices=("random", "pathfinder"),
        default="random",
        help=(
            "Production init method. Defaults to random to keep the check cheap; "
            "use pathfinder for a closer Stage 5-style initialization probe."
        ),
    )
    parser.add_argument("--pathfinder-num-elbo-samples", type=int, default=4)
    parser.add_argument("--pathfinder-maxiter", type=int, default=5)
    parser.add_argument("--n-pathfinder-starts", type=int, default=2)
    parser.add_argument("--pathfinder-init-scale", type=float, default=0.1)
    parser.add_argument("--ess-max-sites", type=int, default=24)
    parser.add_argument("--smoothers", default="plain,amala")
    parser.add_argument(
        "--all-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable every optional high-memory diagnostic metric group.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        choices=SCRIPT_DIAGNOSTIC_METRIC_VALUES,
        default=None,
        help=(
            "Optional diagnostic metric group to collect when --no-all-metrics is used. "
            "May be repeated."
        ),
    )
    parser.add_argument(
        "--support-mode",
        choices=("point", "interval", "both"),
        default="point",
        help=(
            "Observation-support mode. 'point' is the nonlinear smoother smoke "
            "check; 'interval' and 'both' probe the current interval-support caveat."
        ),
    )
    parser.add_argument(
        "--retain-latent-paths",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--skip-latent-summary", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "scratchpad" / "mpg_smoothers_demo_like_check.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.T < 8:
        raise ValueError("T must be at least 8 so the nonlinear fixture has sparse events")
    smoothers = _parse_smoothers(args.smoothers)
    support_modes = _support_modes(args.support_mode)
    selected_metrics = _selected_diagnostic_metrics(args)
    if not TRUE_HILL_BY_SITE or not TRUE_MULTIPLICATIVE_BY_SITE:
        raise AssertionError("demo-like fixture is not exercising nonlinear dynamics")

    runs: dict[str, Any] = {}
    for support_idx, support_mode in enumerate(support_modes):
        for smoother_idx, smoother in enumerate(smoothers):
            run_key = f"{support_mode}:{smoother}"
            print(f"running {run_key}...", flush=True)
            runs[run_key] = _run_one(
                args,
                smoother=smoother,
                support_mode=support_mode,
                seed=args.seed + 100 * support_idx + smoother_idx,
            )

    payload = {
        "artifact": "mpg_smoothers_demo_like_check",
        "target": "nonlinear_demo_like_synthetic",
        "config": {
            "T": args.T,
            "num_warmup": args.num_warmup,
            "num_samples": args.num_samples,
            "num_chains": args.num_chains,
            "n_particles": args.n_particles,
            "n_parameter_particles": args.n_parameter_particles,
            "latent_block_size": args.latent_block_size,
            "amala_q_scale": args.amala_q_scale,
            "amala_kappa": args.amala_kappa,
            "all_metrics": args.all_metrics,
            "diagnostic_metrics": sorted(selected_metrics),
            "retain_latent_paths": bool(
                args.retain_latent_paths or LATENT_PATH_MIXING_METRIC in selected_metrics
            ),
            "support_mode": args.support_mode,
            "smoothers": smoothers,
        },
        "nonlinear_fixture": {
            "hill_site_count": len(TRUE_HILL_BY_SITE),
            "multiplicative_site_count": len(TRUE_MULTIPLICATIVE_BY_SITE),
        },
        "runs": runs,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_ready(payload), indent=2) + "\n")
    print(f"wrote {args.output}", flush=True)
    print("all requested MPGibbs smoother checks passed", flush=True)


if __name__ == "__main__":
    main()
