"""Prior Predictive Validation for Stage 4.

Validates proposed priors by sampling from the prior predictive distribution
and checking for domain violations (NaN/Inf, constraint violations, extreme
values, scale plausibility).
"""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
import networkx as nx
import numpy as np
import polars as pl
from pydantic import ValidationError

from causal_ssm_agent.artifacts.model_spec import DistributionFamily, ModelSpec
from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.models.compilation_errors import AggregatedCompileError
from causal_ssm_agent.workers.schemas_prior import (
    PriorPathologyCertificate,
    PriorProposal,
    PriorRepairScope,
    PriorValidationResult,
)

logger = get_prefect_logger(__name__)
_RECOVERABLE_MODEL_BUILD_ERRORS = (
    AggregatedCompileError,
    AttributeError,
    KeyError,
    RuntimeError,
    TypeError,
    ValidationError,
    ValueError,
)
_RECOVERABLE_PRIOR_SAMPLING_ERRORS = (
    ArithmeticError,
    AttributeError,
    FloatingPointError,
    OverflowError,
    RuntimeError,
    TypeError,
    ValueError,
)

PriorValidationSeverity = Literal["error", "warning"]
PriorFailureStage = Literal[
    "compiled_parameters",
    "latent_dynamics",
    "observation_mean",
    "observation_sample",
    "support_violation",
    "model_build",
    "prior_sampling",
    "unknown",
]


def _pp_result(
    *,
    parameter: str,
    is_valid: bool,
    code: str,
    issue: str | None,
    suggested_adjustment: str | None = None,
    severity: PriorValidationSeverity = "error",
    related_parameters: list[str] | None = None,
    supporting_codes: list[str] | None = None,
    repair_scope: PriorRepairScope | None = None,
    failure_stage: PriorFailureStage | None = None,
    bad_sample_sites: list[str] | None = None,
    bad_manifest_names: list[str] | None = None,
    failing_draw_indices: list[int] | None = None,
    first_bad_time_index: int | None = None,
    pathology_certificate: PriorPathologyCertificate | None = None,
) -> PriorValidationResult:
    """Build a typed prior-predictive diagnostic."""
    return PriorValidationResult(
        parameter=parameter,
        is_valid=is_valid,
        code=code,
        origin="prior_predictive",
        severity=severity,
        issue=issue,
        suggested_adjustment=suggested_adjustment,
        related_parameters=related_parameters or ([parameter] if parameter else []),
        supporting_codes=supporting_codes or [],
        repair_scope=repair_scope,
        failure_stage=failure_stage,
        bad_sample_sites=bad_sample_sites or [],
        bad_manifest_names=bad_manifest_names or [],
        failing_draw_indices=failing_draw_indices or [],
        first_bad_time_index=first_bad_time_index,
        pathology_certificate=pathology_certificate,
    )


def _artifact_compile_diagnostics(compiled_ssm: dict | None) -> list[PriorValidationResult]:
    """Return typed compile diagnostics attached to a compiled artifact."""
    if compiled_ssm is None:
        logger.debug("No compiled SSM artifact; skipping compile diagnostics")
        return []

    diagnostics = compiled_ssm.get("compile_diagnostics") or []
    typed: list[PriorValidationResult] = []
    for diagnostic in diagnostics:
        if isinstance(diagnostic, PriorValidationResult):
            typed.append(diagnostic)
        else:
            typed.append(PriorValidationResult.model_validate(diagnostic))
    return typed


def _supporting_compile_context(
    compiled_ssm: dict | None,
    *,
    construct_names: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Return supporting compile diagnostics relevant to a later PP failure."""
    diagnostics = [
        d for d in _artifact_compile_diagnostics(compiled_ssm) if d.severity == "warning"
    ]
    if not diagnostics:
        return [], []

    relevant = diagnostics
    if construct_names:
        construct_set = set(construct_names)
        filtered = [
            diagnostic
            for diagnostic in diagnostics
            if diagnostic.parameter == "drift_offdiag"
            or any(name in diagnostic.parameter for name in construct_set)
        ]
        if filtered:
            relevant = filtered

    related_parameters = list(
        dict.fromkeys(
            str(parameter)
            for diagnostic in relevant
            for parameter in (diagnostic.related_parameters or [diagnostic.parameter])
            if parameter
        )
    )
    supporting_codes = list(
        dict.fromkeys(str(diagnostic.code) for diagnostic in relevant if diagnostic.code)
    )
    return related_parameters, supporting_codes


def compute_data_stats(data_for_model: pl.DataFrame) -> dict[str, dict]:
    """Compute per-indicator mean, std, min, max from raw data."""
    stats = {}
    for row in (
        data_for_model.group_by("indicator")
        .agg(
            [
                pl.col("value").cast(pl.Float64, strict=False).mean().alias("mean"),
                pl.col("value").cast(pl.Float64, strict=False).std().alias("std"),
                pl.col("value").cast(pl.Float64, strict=False).min().alias("min"),
                pl.col("value").cast(pl.Float64, strict=False).max().alias("max"),
            ]
        )
        .iter_rows(named=True)
    ):
        stats[row["indicator"]] = {
            "mean": row["mean"],
            "std": row["std"],
            "min": row["min"],
            "max": row["max"],
        }
    return stats


def _ordered_latent_sccs(
    causal_spec: dict | None,
    latent_names: list[str],
) -> list[tuple[str, ...]]:
    """Return latent SCCs in estimation order for deterministic repair scoping."""
    if causal_spec is None or not latent_names:
        return []

    from causal_ssm_agent.utils.causal_spec import get_estimation_edges, get_estimation_state_order

    latent_name_set = set(latent_names)
    construct_order = [
        name for name in get_estimation_state_order(causal_spec) if name in latent_name_set
    ]
    if not construct_order:
        return []

    graph = nx.DiGraph()
    graph.add_nodes_from(construct_order)
    for edge in get_estimation_edges(causal_spec):
        cause = edge.get("cause")
        effect = edge.get("effect")
        if cause in latent_name_set and effect in latent_name_set:
            graph.add_edge(cause, effect)

    order_lookup = {name: idx for idx, name in enumerate(construct_order)}
    components = sorted(
        nx.strongly_connected_components(graph),
        key=lambda members: min(order_lookup[name] for name in members),
    )
    return [
        tuple(name for name in construct_order if name in component) for component in components
    ]


def _infer_dynamics_repair_scope(
    drift_samples: np.ndarray,
    unstable_indices: list[int],
    *,
    compiled_ssm: dict | None,
    causal_spec: dict | None,
) -> PriorRepairScope | None:
    """Bound global drift instability to the smallest SCC-level repair scope."""
    if not unstable_indices or compiled_ssm is None or causal_spec is None:
        logger.debug("Skipping dynamics repair-scope attribution (missing inputs)")
        return None

    from causal_ssm_agent.models.ssm_compiler import deserialize_ssm_spec

    try:
        spec_payload = compiled_ssm.get("spec")
        if not isinstance(spec_payload, dict):
            return None
        ssm_spec = deserialize_ssm_spec(spec_payload)
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning(
            "Skipping dynamics repair-scope attribution (invalid compiled spec): %s", exc
        )
        return None

    latent_names = list(ssm_spec.latent_names or [])
    if not latent_names:
        return None

    sccs = _ordered_latent_sccs(causal_spec, latent_names)
    if not sccs:
        return None

    latent_index = {name: idx for idx, name in enumerate(latent_names)}
    implicated_sccs: list[tuple[str, ...]] = []
    for scc in sccs:
        scc_indices = [latent_index[name] for name in scc]
        for sample_idx in unstable_indices:
            submatrix = drift_samples[sample_idx][np.ix_(scc_indices, scc_indices)]
            max_real = float(np.max(np.real(np.linalg.eigvals(submatrix))))
            if max_real >= 0:
                implicated_sccs.append(scc)
                break

    if not implicated_sccs:
        implicated_sccs = sccs

    construct_names = list(dict.fromkeys(name for scc in implicated_sccs for name in scc))
    if not construct_names:
        return None
    return PriorRepairScope(kind="dynamics_scc", construct_names=construct_names)


def _check_nan_inf(
    samples: dict[str, jnp.ndarray],
    *,
    compiled_ssm: dict | None = None,
    manifest_names: list[str] | None = None,
) -> PriorValidationResult | None:
    """Check for NaN or Inf in any sample site."""
    from causal_ssm_agent.models.ssm.constants import INTERNAL_DIAGNOSTIC_SITES

    def _draw_indices(mask: np.ndarray) -> list[int]:
        if mask.ndim == 0:
            return [0] if bool(mask) else []
        leading = mask.reshape(mask.shape[0], -1).any(axis=1)
        return [int(idx) for idx in np.flatnonzero(leading)]

    def _n_draws(arr: np.ndarray) -> int:
        return 1 if arr.ndim == 0 else max(1, int(arr.shape[0]))

    def _stage_rank(stage: str) -> int:
        ordering = {
            "compiled_parameters": 0,
            "observation_mean": 1,
            "observation_sample": 2,
            "unknown": 99,
        }
        return ordering.get(stage, 99)

    bad_sites = []
    bad_draw_indices: set[int] = set()
    bad_manifest_names: set[str] = set()
    first_bad_time_index: int | None = None
    failure_stage = "unknown"

    for name, values in samples.items():
        if name in INTERNAL_DIAGNOSTIC_SITES:
            continue
        arr = np.asarray(values)
        mask = ~np.isfinite(arr)
        if name == "observations":
            observation_mask = samples.get("observations_mask")
            if observation_mask is not None:
                active_observation_mask = np.asarray(observation_mask, dtype=bool)
                if active_observation_mask.shape == arr.shape:
                    mask = mask & active_observation_mask
        if np.any(mask):
            bad_sites.append(name)
            bad_draw_indices.update(_draw_indices(mask))

            if name == "observations":
                candidate_stage = "observation_sample"
                if mask.ndim >= 3 and manifest_names:
                    manifest_mask = mask.any(axis=(0, 1))
                    for manifest_idx in np.flatnonzero(manifest_mask):
                        if manifest_idx < len(manifest_names):
                            bad_manifest_names.add(manifest_names[int(manifest_idx)])
                    time_mask = mask.any(axis=(0, 2))
                    bad_time_indices = np.flatnonzero(time_mask)
                    if bad_time_indices.size > 0:
                        candidate_time = int(bad_time_indices[0])
                        first_bad_time_index = (
                            candidate_time
                            if first_bad_time_index is None
                            else min(first_bad_time_index, candidate_time)
                        )
            else:
                candidate_stage = "compiled_parameters"

            if _stage_rank(candidate_stage) < _stage_rank(failure_stage):
                failure_stage = candidate_stage

    if bad_sites:
        related_parameters, supporting_codes = _supporting_compile_context(compiled_ssm)
        n_draws = max(_n_draws(np.asarray(samples[site_name])) for site_name in bad_sites)
        certificate = PriorPathologyCertificate(
            kind="nonfinite_samples",
            primary_score=len(bad_draw_indices) / max(1, n_draws),
            secondary_score=float(len(bad_manifest_names)) if bad_manifest_names else None,
        )
        return _pp_result(
            parameter="prior_predictive",
            is_valid=False,
            code="prior_predictive_nonfinite_samples",
            issue=f"NaN/Inf detected in sample sites: {', '.join(bad_sites)}",
            suggested_adjustment="Check for degenerate priors or numerical overflow",
            related_parameters=related_parameters,
            supporting_codes=supporting_codes,
            failure_stage=failure_stage,
            bad_sample_sites=bad_sites,
            bad_manifest_names=sorted(bad_manifest_names),
            failing_draw_indices=sorted(bad_draw_indices),
            first_bad_time_index=first_bad_time_index,
            pathology_certificate=certificate,
        )
    return None


def _dummy_values_for_distribution(distribution: DistributionFamily, n_rows: int) -> list[float]:
    """Construct support-compatible dummy observations for model validation."""
    if distribution.is_discrete:
        return [float(i % 2) for i in range(n_rows)]
    return [distribution.support_interior_point] * n_rows


def _make_support_compatible_dummy_wide_data(
    model_spec: ModelSpec,
    n_rows: int = 10,
) -> pl.DataFrame:
    """Build minimal wide data that satisfy each likelihood family's support."""
    cols: dict[str, pl.Series] = {"time": pl.Series("time", list(range(n_rows)), dtype=pl.Float64)}
    for lik in model_spec.likelihoods:
        cols[lik.variable] = pl.Series(
            lik.variable,
            _dummy_values_for_distribution(lik.distribution, n_rows),
            dtype=pl.Float64,
        )
    return pl.DataFrame(cols)


def _check_constraint_violations(
    samples: dict[str, jnp.ndarray],
    threshold: float = 0.05,
) -> list[PriorValidationResult]:
    """Check for constraint violations in sampled parameters.

    Positive-constrained sites should not have negative values.

    Args:
        samples: Dict of sample site name to array of samples.
        threshold: Fraction of violations above which to flag a failure.
            Default 5% to tolerate minor numerical rounding near the boundary.
    """
    results = []
    positive_sites = ["diffusion_diag_free", "manifest_var_diag_free", "t0_var_diag_free"]

    for site_name in positive_sites:
        if site_name not in samples:
            continue
        arr = np.asarray(samples[site_name])
        n_total = arr.size
        if n_total == 0:
            continue
        n_violations = int(np.sum(arr < 0))
        violation_rate = n_violations / n_total
        if violation_rate > threshold:
            results.append(
                _pp_result(
                    parameter=site_name,
                    is_valid=False,
                    code="constraint_violation",
                    issue=(
                        f"Constraint violation: {violation_rate:.1%} of {site_name} samples "
                        f"are negative (should be positive)"
                    ),
                    suggested_adjustment="Use a positive-constrained prior family",
                    failure_stage="support_violation",
                )
            )

    return results


def _check_extreme_values(
    samples: dict[str, jnp.ndarray],
    threshold: float = 0.10,
    extreme_cutoff: float = 1e6,
) -> list[PriorValidationResult]:
    """Check for extreme parameter values indicating priors too wide."""
    results = []
    # Check parameter sites (not deterministic outputs like drift, diffusion)
    param_sites = [k for k in samples if k.endswith("_free")]
    for site_name in param_sites:
        arr = np.asarray(samples[site_name])
        n_total = arr.size
        if n_total == 0:
            continue
        n_extreme = int(np.sum(np.abs(arr) > extreme_cutoff))
        extreme_rate = n_extreme / n_total
        if extreme_rate > threshold:
            results.append(
                _pp_result(
                    parameter=site_name,
                    is_valid=False,
                    code="extreme_values",
                    issue=(
                        f"Extreme values: {extreme_rate:.1%} of {site_name} samples "
                        f"have |value| > {extreme_cutoff:.0e}"
                    ),
                    suggested_adjustment="Tighten the prior (reduce sigma)",
                    failure_stage="compiled_parameters",
                )
            )

    return results


def _check_scale_plausibility(
    samples: dict[str, jnp.ndarray],
    data_stats: dict[str, dict],
    manifest_names: list[str],
    *,
    compiled_ssm: dict | None = None,
    causal_spec: dict | None = None,
    n_subsample: int = 50,
    ratio_threshold: float = 100.0,
) -> list[PriorValidationResult]:
    """Check implied observation scale vs data scale.

    This diagnostic is defined on sampled observations. A successful prior
    predictive run should always provide manifest draws, so missing or malformed
    observation arrays are treated as harness errors rather than silently
    approximating scale from latent stationary covariance.
    """
    from causal_ssm_agent.models.ssm.discretization import solve_lyapunov

    results = []

    observations = samples.get("observations")
    observation_mask = samples.get("observations_mask")
    if observations is None:
        return [
            _pp_result(
                parameter="prior_predictive",
                is_valid=False,
                code="prior_predictive_missing_observations",
                issue="Prior predictive samples are missing `observations` after sampling",
                suggested_adjustment=(
                    "Treat this as a harness error: prior predictive sampling must "
                    "populate manifest observation draws before scale validation"
                ),
                failure_stage="observation_sample",
                bad_sample_sites=["observations"],
            )
        ]

    obs = np.asarray(observations)
    if obs.ndim != 3 or obs.shape[2] != len(manifest_names):
        return [
            _pp_result(
                parameter="prior_predictive",
                is_valid=False,
                code="prior_predictive_malformed_observations",
                issue=(
                    "Prior predictive samples contain malformed `observations` with "
                    f"shape {obs.shape}; expected (draw, time, manifest={len(manifest_names)})"
                ),
                suggested_adjustment=(
                    "Treat this as a harness error: prior predictive sampling must "
                    "emit a manifest-aligned observation tensor"
                ),
                failure_stage="observation_sample",
                bad_sample_sites=["observations"],
            )
        ]

    if "drift" in samples and "diffusion" in samples:
        drift_samples = np.asarray(samples["drift"])
        diffusion_samples = np.asarray(samples["diffusion"])
        n_total = drift_samples.shape[0]
        idx = np.random.default_rng(42).choice(
            n_total,
            size=min(n_subsample, n_total),
            replace=False,
        )
        n_unstable = 0
        unstable_indices: list[int] = []
        for i in idx:
            drift_i = jnp.array(drift_samples[i])
            diff_i = jnp.array(diffusion_samples[i])
            diff_cov_i = diff_i @ diff_i.T

            eigvals = jnp.linalg.eigvals(drift_i)
            max_real = float(jnp.max(jnp.real(eigvals)))
            if max_real >= 0:
                logger.debug(
                    "Unstable drift draw %d (max real eigenvalue=%.4f, eigenvalue range=[%.4f, %.4f])",
                    i,
                    max_real,
                    float(jnp.min(jnp.real(eigvals))),
                    max_real,
                )
                n_unstable += 1
                unstable_indices.append(int(i))
                continue

            try:
                sigma_inf = solve_lyapunov(drift_i, diff_cov_i)
                sigma_inf_np = np.asarray(sigma_inf)
                if np.any(np.isnan(sigma_inf_np)) or np.any(np.diag(sigma_inf_np) < 0):
                    n_unstable += 1
                    unstable_indices.append(int(i))
                    continue
            except (ValueError, RuntimeError, FloatingPointError, ArithmeticError) as exc:
                logger.info("Prior draw %d unstable (Lyapunov solver failed): %s", i, exc)
                n_unstable += 1
                unstable_indices.append(int(i))
                continue

        n_draws = len(idx)
        if n_unstable > n_draws * 0.5:
            sorted_unstable_indices = sorted(unstable_indices)
            repair_scope = _infer_dynamics_repair_scope(
                drift_samples,
                sorted_unstable_indices,
                compiled_ssm=compiled_ssm,
                causal_spec=causal_spec,
            )
            related_parameters, supporting_codes = _supporting_compile_context(
                compiled_ssm,
                construct_names=list(repair_scope.construct_names) if repair_scope else None,
            )
            results.append(
                _pp_result(
                    parameter="dynamics_stability",
                    is_valid=False,
                    code="dynamics_stability",
                    issue=(
                        f"Unstable dynamics: {n_unstable}/{n_draws} prior draws have "
                        f"unstable drift (Lyapunov solver failed)"
                    ),
                    suggested_adjustment="Tighten drift_diag prior toward more negative values",
                    related_parameters=related_parameters,
                    supporting_codes=supporting_codes,
                    repair_scope=repair_scope,
                    failure_stage="latent_dynamics",
                    failing_draw_indices=sorted_unstable_indices,
                    pathology_certificate=PriorPathologyCertificate(
                        kind="dynamics_stability",
                        primary_score=n_unstable / max(1, n_draws),
                    ),
                )
            )

    mask = np.asarray(observation_mask, dtype=bool) if observation_mask is not None else None
    manifest_draw_stds: list[list[float]] = [[] for _ in manifest_names]
    for draw_idx in range(obs.shape[0]):
        draw_obs = obs[draw_idx]
        draw_mask = mask[draw_idx] if mask is not None and mask.shape == obs.shape else None
        for manifest_idx in range(len(manifest_names)):
            values = draw_obs[:, manifest_idx]
            if draw_mask is not None:
                values = values[draw_mask[:, manifest_idx]]
            else:
                values = values[np.isfinite(values)]
            values = values[np.isfinite(values)]
            if values.size >= 2:
                manifest_draw_stds[manifest_idx].append(float(np.std(values)))

    median_implied = np.asarray(
        [float(np.median(stds)) if stds else np.nan for stds in manifest_draw_stds]
    )

    for j, name in enumerate(manifest_names):
        if j >= len(median_implied):
            break
        if not np.isfinite(median_implied[j]):
            continue
        if name not in data_stats or data_stats[name]["std"] is None:
            continue

        data_std = data_stats[name]["std"]
        if data_std == 0 or data_std is None:
            continue

        ratio = float(median_implied[j]) / data_std
        if ratio > ratio_threshold or ratio < 1.0 / ratio_threshold:
            results.append(
                _pp_result(
                    parameter=f"scale_{name}",
                    is_valid=False,
                    code="scale_mismatch",
                    issue=(
                        f"Scale mismatch for {name}: implied std "
                        f"({median_implied[j]:.2g}) vs data std ({data_std:.2g}), "
                        f"ratio={ratio:.1g}"
                    ),
                    suggested_adjustment=("Adjust diffusion/drift priors to match data scale"),
                    failure_stage="observation_sample",
                )
            )

    return results


def _check_lagged_response_plausibility(
    samples: dict[str, jnp.ndarray],
    compiled_ssm: dict | None,
    causal_spec: dict | None,
    *,
    n_subsample: int = 50,
    weak_response_cutoff: float = 0.02,
    weak_response_fraction: float = 0.9,
) -> list[PriorValidationResult]:
    """Warn when the full-system one-lag response is near-zero for a declared lagged edge.

    This is intentionally a warning-only heuristic. It uses the compiled drift
    draws and the full transition matrix ``exp(A * dt)`` instead of treating a
    single off-diagonal drift mean as the edge timescale.
    """
    if compiled_ssm is None or causal_spec is None or "drift" not in samples:
        logger.debug("Skipping lagged-response plausibility check (missing inputs)")
        return []

    from causal_ssm_agent.models.ssm_compiler import (
        deserialize_edge_lag_days,
        deserialize_ssm_spec,
    )
    from causal_ssm_agent.utils.causal_spec import get_estimation_edges

    try:
        spec_payload = compiled_ssm.get("spec")
        if not isinstance(spec_payload, dict):
            return []
        edge_lag_days = deserialize_edge_lag_days(compiled_ssm.get("edge_lag_days"))
        ssm_spec = deserialize_ssm_spec(spec_payload)
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning(
            "Skipping lagged-response plausibility check (invalid compiled spec): %s", exc
        )
        return []

    latent_names = list(ssm_spec.latent_names or [])
    if not latent_names:
        return []

    drift_samples = np.asarray(samples["drift"])
    if drift_samples.ndim != 3 or drift_samples.shape[1] != drift_samples.shape[2]:
        return []

    latent_index = {name: idx for idx, name in enumerate(latent_names)}
    try:
        edges = get_estimation_edges(causal_spec)
    except (ValueError, KeyError, TypeError) as exc:
        logger.info(
            "Skipping lagged-response plausibility check (invalid estimation edges): %s", exc
        )
        return []

    lagged_edges = [
        edge
        for edge in edges
        if bool(edge.get("lagged", True))
        and edge.get("cause") in latent_index
        and edge.get("effect") in latent_index
    ]
    if not lagged_edges:
        return []

    sample_count = drift_samples.shape[0]
    sample_idx = np.random.default_rng(42).choice(
        sample_count,
        size=min(n_subsample, sample_count),
        replace=False,
    )

    import jax.scipy.linalg as jla

    results: list[PriorValidationResult] = []
    for edge in lagged_edges:
        cause = str(edge["cause"])
        effect = str(edge["effect"])
        effect_idx = latent_index[effect]
        cause_idx = latent_index[cause]
        lag_days = edge_lag_days.get((effect_idx, cause_idx))
        if lag_days is None:
            logger.warning(
                "Skipping lagged-response plausibility check for %s->%s (missing edge lag metadata)",
                cause,
                effect,
            )
            continue
        responses: list[float] = []
        for idx in sample_idx:
            drift = jnp.asarray(drift_samples[idx], dtype=jnp.float64)
            transition = np.asarray(jla.expm(drift * lag_days))
            responses.append(float(transition[effect_idx, cause_idx]))

        if not responses:
            continue

        abs_responses = np.abs(np.asarray(responses))
        weak_fraction = float(np.mean(abs_responses < weak_response_cutoff))
        median_abs = float(np.median(abs_responses))
        if weak_fraction < weak_response_fraction:
            continue

        results.append(
            _pp_result(
                parameter=f"beta_{cause}_{effect}",
                is_valid=True,
                code="lagged_response_weak",
                severity="warning",
                issue=(
                    f"Across prior draws, the full-system one-lag response for {cause}->{effect} "
                    f"at {lag_days:.1f}d is usually very small "
                    f"(median |response|={median_abs:.3f}; {weak_fraction:.0%} of draws "
                    f"are below {weak_response_cutoff:.2f})."
                ),
                suggested_adjustment=(
                    "Confirm that a near-zero one-lag effect is substantively intended. "
                    "If not, strengthen the daily-scale prior or author it on the source "
                    "study interval with `reference_interval_days`."
                ),
            )
        )

    return results


def validate_prior_predictive(
    model_spec: ModelSpec | dict,
    priors: dict[str, PriorProposal] | dict[str, dict],
    data_for_model: pl.DataFrame | None = None,
    data_stats: dict[str, dict] | None = None,
    n_samples: int = 500,
    constraint_tolerance: float = 0.05,
    causal_spec: dict | None = None,
    compiled_ssm: dict | None = None,
) -> tuple[bool, list[PriorValidationResult], dict]:
    """Validate priors via prior predictive sampling.

    Checks for:
    1. Model builds successfully
    2. No NaN/Inf in samples
    3. Constraint violations (positive params < 0, etc.)
    4. Extreme values (|param| > 1e6)
    5. Scale plausibility vs data (if data_for_model provided)
    6. Non-fatal lagged-response plausibility warnings using the full transition
       matrix over the model lag interval

    Args:
        model_spec: Model specification
        priors: Prior proposals for each parameter
        data_for_model: Raw timestamped data (optional, for scale plausibility check)
        data_stats: Optional precomputed per-indicator stats for scale checks
        n_samples: Number of prior predictive samples
        constraint_tolerance: Fraction of positive-constraint violations to
            tolerate before flagging failure (default 5%).
        causal_spec: CausalSpec dict for DAG-constrained masks
        compiled_ssm: Optional precompiled artifact to reuse within a Stage 4
            validation pass and avoid recompiling identical inputs.

    Returns:
        Tuple of (is_valid, validation results, raw prior predictive samples).
        Unpack ``simulate_predictive_observations()`` to generate per-variable
        observation samples and their effective emission mask for visualization.
    """
    from causal_ssm_agent.models.predictive_simulation import (
        PredictiveObservationMeanOverflow,
    )
    from causal_ssm_agent.models.ssm_builder import (
        prepare_model_runtime,
        prepare_wide_model_runtime,
    )
    from causal_ssm_agent.models.ssm_compilation_common import dump_prior_payloads
    from causal_ssm_agent.models.ssm_compiler import (
        compile_ssm_artifact,
        make_builder_from_compiled_artifact,
    )

    priors_dict = dump_prior_payloads(priors)

    # Parse model_spec for manifest names
    if isinstance(model_spec, dict):
        spec_obj = ModelSpec.model_validate(model_spec)
    else:
        spec_obj = model_spec

    manifest_names = [lik.variable for lik in spec_obj.likelihoods]

    # 1. Build model
    try:
        artifact = compiled_ssm or compile_ssm_artifact(
            model_spec, priors_dict, causal_spec=causal_spec
        )
        if data_for_model is not None and not data_for_model.is_empty():
            builder = prepare_model_runtime(data_for_model, compiled_ssm=artifact).builder
        else:
            # No raw data: create dummy observations inside each family's support
            X_wide = _make_support_compatible_dummy_wide_data(spec_obj)
            builder = prepare_wide_model_runtime(
                X_wide,
                builder=make_builder_from_compiled_artifact(artifact),
            ).builder
    except _RECOVERABLE_MODEL_BUILD_ERRORS as e:
        return (
            False,
            [
                _pp_result(
                    parameter="model_build",
                    is_valid=False,
                    code="model_build",
                    issue=f"Model build failed: {e}",
                    suggested_adjustment="Fix model_spec or priors to enable model construction",
                    failure_stage="model_build",
                )
            ],
            {},
        )

    # 2. Sample prior predictive
    try:
        samples = builder.sample_prior_predictive(samples=n_samples)
    except PredictiveObservationMeanOverflow as exc:
        related_parameters, supporting_codes = _supporting_compile_context(artifact)
        certificate = PriorPathologyCertificate(
            kind="nonfinite_samples",
            primary_score=len(exc.failing_draw_indices) / max(1, n_samples),
            secondary_score=float(len(exc.bad_manifest_names)) if exc.bad_manifest_names else None,
        )
        return (
            False,
            [
                _pp_result(
                    parameter="prior_predictive",
                    is_valid=False,
                    code="prior_predictive_observation_mean_overflow",
                    issue=str(exc),
                    suggested_adjustment=(
                        "Tighten priors that drive these log-link observation means so the "
                        "predictive response stays finite before sampling."
                    ),
                    related_parameters=related_parameters,
                    supporting_codes=supporting_codes,
                    failure_stage="observation_mean",
                    bad_sample_sites=["observations"],
                    bad_manifest_names=list(exc.bad_manifest_names),
                    failing_draw_indices=list(exc.failing_draw_indices),
                    first_bad_time_index=exc.first_bad_time_index,
                    pathology_certificate=certificate,
                )
            ],
            {},
        )
    except _RECOVERABLE_PRIOR_SAMPLING_ERRORS as e:
        return (
            False,
            [
                _pp_result(
                    parameter="prior_sampling",
                    is_valid=False,
                    code="prior_sampling",
                    issue=f"Prior predictive sampling failed: {e}",
                    suggested_adjustment="Check priors for numerical issues",
                    failure_stage="prior_sampling",
                )
            ],
            {},
        )

    # 3. Run checks
    results: list[PriorValidationResult] = []

    # Check 1: NaN/Inf
    nan_result = _check_nan_inf(
        samples,
        compiled_ssm=artifact,
        manifest_names=manifest_names,
    )
    if nan_result is not None:
        results.append(nan_result)

    # Check 2: Constraint violations
    results.extend(_check_constraint_violations(samples, threshold=constraint_tolerance))

    # Check 3: Extreme values
    results.extend(_check_extreme_values(samples))

    # Check 4: Scale plausibility
    scale_reference_stats = data_stats
    if (
        scale_reference_stats is None
        and data_for_model is not None
        and not data_for_model.is_empty()
    ):
        scale_reference_stats = compute_data_stats(data_for_model)
    if scale_reference_stats:
        results.extend(
            _check_scale_plausibility(
                samples,
                scale_reference_stats,
                manifest_names,
                compiled_ssm=artifact,
                causal_spec=causal_spec,
            )
        )

    # Check 5: full-system lagged response plausibility (warning-only)
    results.extend(_check_lagged_response_plausibility(samples, artifact, causal_spec))

    is_valid = all(r.is_valid for r in results)

    # If no issues found, add passing results per parameter
    if not results:
        for param_name in priors_dict:
            results.append(
                _pp_result(
                    parameter=param_name,
                    is_valid=True,
                    code="ok",
                    issue=None,
                    suggested_adjustment=None,
                )
            )
        is_valid = True

    return is_valid, results, samples


def format_validation_report(
    is_valid: bool,
    results: list[PriorValidationResult],
) -> str:
    """Format validation results as a human-readable report."""
    lines = []

    if is_valid:
        lines.append("Prior predictive validation PASSED")
    else:
        lines.append("Prior predictive validation FAILED")

    failed = [r for r in results if not r.is_valid]
    if failed:
        lines.append("")
        for r in failed:
            lines.append(f"- {r.parameter}: {r.issue}")

    return "\n".join(lines)


def format_parameter_feedback(
    parameter_name: str,
    results: list[PriorValidationResult],
    prior: dict | None = None,
    data_stats: dict[str, dict] | None = None,
) -> str:
    """Format per-parameter validation feedback for LLM re-elicitation.

    Creates a structured message that tells the LLM what went wrong with
    a specific parameter's prior and provides context for revision.

    Args:
        parameter_name: Name of the parameter to generate feedback for
        results: All validation results from the prior predictive check
        prior: The previous prior proposal dict (for showing what was tried)
        data_stats: Per-indicator data statistics (for scale context)

    Returns:
        Formatted feedback string for inclusion in re-elicitation prompt
    """
    from causal_ssm_agent.models.ssm_compilation_common import GLOBAL_FAILURE_SITES

    # Find results relevant to this parameter
    # Global failures (affect all parameters) are always included
    param_lower = parameter_name.lower()
    relevant = [
        r
        for r in results
        if not r.is_valid
        and (
            r.parameter == parameter_name
            or param_lower in r.parameter.lower()
            or r.parameter.lower().startswith("scale_")  # scale mismatch affects all
            or r.parameter in GLOBAL_FAILURE_SITES
        )
    ]

    if not relevant:
        return ""

    lines = []

    # Show what was previously proposed
    if prior:
        dist = prior.get("distribution", "Unknown")
        params = prior.get("params", {})
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        lines.append(f"Your previous prior for {parameter_name} was {dist}({params_str}).")

    lines.append("Prior predictive validation FAILED:")
    for r in relevant:
        lines.append(f"- {r.issue}")
        if r.suggested_adjustment:
            lines.append(f"  Suggested: {r.suggested_adjustment}")

    # Add data scale context only for scale-mismatch failures, scoped to the
    # specific indicators that triggered the mismatch.  Global failures
    # (model_build, prior_sampling) and constraint/extreme-value checks don't
    # benefit from a full data-stats dump.
    if data_stats:
        scale_indicators = {
            r.parameter.removeprefix("scale_")
            for r in relevant
            if r.parameter.lower().startswith("scale_")
        }
        if scale_indicators:
            scale_lines = []
            for indicator in sorted(scale_indicators):
                if indicator in data_stats:
                    stats = data_stats[indicator]
                    std = stats.get("std")
                    mean = stats.get("mean")
                    if std is not None and mean is not None:
                        scale_lines.append(f"  {indicator}: mean={mean:.2g}, std={std:.2g}")
            if scale_lines:
                lines.append("")
                lines.append("Data scale reference:")
                lines.extend(scale_lines)

    lines.append("")
    lines.append("Please revise your prior to address the issues above.")

    return "\n".join(lines)


def format_parameter_warnings(
    parameter_name: str,
    results: list[PriorValidationResult],
) -> str:
    """Format non-fatal warnings relevant to one parameter."""
    param_lower = parameter_name.lower()
    relevant = [
        r
        for r in results
        if r.severity == "warning"
        and (
            r.parameter == parameter_name
            or param_lower in r.parameter.lower()
            or r.parameter.lower() in param_lower
        )
    ]
    if not relevant:
        return ""

    lines: list[str] = []
    for result in relevant:
        if result.issue:
            lines.append(f"- {result.issue}")
        if result.suggested_adjustment:
            lines.append(f"  Suggested: {result.suggested_adjustment}")
    return "\n".join(lines)


def get_failed_parameters(
    results: list[PriorValidationResult],
    parameter_names: list[str],
    causal_spec: dict | None = None,
) -> list[str]:
    """Extract parameter names that contributed to validation failure.

    Maps validation result parameter names (which may be SSM site names like
    'drift_diag_free' or 'scale_mood') back to ModelSpec parameter names.

    When ``causal_spec`` is provided, scale mismatch failures are targeted
    to the construct whose indicator triggered the mismatch rather than
    re-eliciting all parameters.

    Args:
        results: Validation results from prior predictive check
        parameter_names: All ModelSpec parameter names
        causal_spec: Optional CausalSpec dict for targeted re-elicitation

    Returns:
        List of ModelSpec parameter names that need re-elicitation
    """
    failed_results = [r for r in results if not r.is_valid]
    if not failed_results:
        return []

    from causal_ssm_agent.models.ssm_compilation_common import (
        GLOBAL_FAILURE_SITES,
        NUISANCE_SITES,
        SITE_TO_KEYWORDS,
    )

    # Check for global failures that affect all parameters
    if any(r.parameter in GLOBAL_FAILURE_SITES for r in failed_results):
        return list(parameter_names)

    # Build indicator→construct lookup from causal_spec
    indicator_to_construct: dict[str, str] = {}
    if causal_spec:
        from causal_ssm_agent.utils.causal_spec import get_indicators

        for ind in get_indicators(causal_spec):
            ind_name = ind.get("name") if isinstance(ind, dict) else ind.name
            construct = ind.get("construct_name") if isinstance(ind, dict) else ind.construct_name
            if ind_name and construct:
                indicator_to_construct[ind_name] = construct

    failed_params = set()
    for r in failed_results:
        result_param = r.parameter.lower()

        # Skip nuisance sites — they can't be re-elicited
        if result_param in NUISANCE_SITES:
            logger.info(
                "Skipping nuisance site '%s' in failed parameter mapping "
                "(not in ModelSpec, uses fixed default prior)",
                r.parameter,
            )
            continue

        # Direct match
        for param_name in parameter_names:
            if param_name.lower() in result_param or result_param in param_name.lower():
                failed_params.add(param_name)
                continue

        # Keyword-based match via SSM site names
        for site_prefix, keywords in SITE_TO_KEYWORDS.items():
            if site_prefix in result_param:
                for param_name in parameter_names:
                    if any(kw in param_name.lower() for kw in keywords):
                        failed_params.add(param_name)

        # Scale mismatch (scale_<indicator>) -> targeted or blanket
        if result_param.startswith("scale_"):
            indicator_name = r.parameter.removeprefix("scale_")
            construct = indicator_to_construct.get(indicator_name)
            if construct:
                # Only re-elicit parameters whose name contains the construct
                for param_name in parameter_names:
                    if construct in param_name.lower():
                        failed_params.add(param_name)
            else:
                # No causal_spec or no match → fall back to all
                failed_params.update(parameter_names)

    return list(failed_params) if failed_params else list(parameter_names)
