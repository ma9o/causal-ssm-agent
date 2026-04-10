"""Local structural identifiability diagnostics via output sensitivity."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import jax.numpy as jnp
import jax.random as random
import numpy as np

from causal_ssm_agent.artifacts.model_spec import DistributionFamily
from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.models.ssm.inference.targets.base import NUMERICAL_EPSILON
from causal_ssm_agent.models.ssm.parameterization import sample_prior_unconstrained
from causal_ssm_agent.models.ssm.structure_runtime import SSMStructureRuntime

from .context import ParametricIdContext, get_stage4b_sweep_context
from .results import OutputSensitivityResult, OutputSensitivityUnsupportedError

if TYPE_CHECKING:
    from causal_ssm_agent.models.ssm.model import SSMModel, SSMSpec

logger = get_prefect_logger(__name__)

_SCALAR_PARAMETER_INDEX_RE = re.compile(r"^(?P<site>.+)\[(?P<index>\d+)\]$")


def _observation_semantic_mask(
    spec: SSMSpec,
    times: jnp.ndarray,
    observation_support,
) -> np.ndarray | None:
    """Return the support-aware emission mask aligned to the model time grid."""
    from causal_ssm_agent.models.ssm.inference.targets.trajectory_observations import (
        compile_observation_operator,
    )

    observation_operator = compile_observation_operator(observation_support)
    if not observation_operator.requires_interval_summary_handling:
        return None

    _, semantic_mask = observation_operator.project_response_trajectory(
        jnp.zeros((times.shape[0], spec.n_manifest), dtype=jnp.float64)
    )
    return np.asarray(semantic_mask > 0.5)


def _build_sensitivity_output_mask(
    observations: jnp.ndarray | None,
    *,
    semantic_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    """Return a feature mask aligned to the emitted-observation moment summary."""
    if observations is None and semantic_mask is None:
        return None

    if observations is None:
        obs_mask = np.asarray(semantic_mask, dtype=bool)
    else:
        obs_mask = ~np.isnan(np.asarray(observations))
        if semantic_mask is not None:
            obs_mask = obs_mask & np.asarray(semantic_mask, dtype=bool)
    mean_mask = obs_mask.reshape(-1)
    tri_i, tri_j = np.tril_indices(obs_mask.shape[1])
    same_cov_mask = (obs_mask[:, :, None] & obs_mask[:, None, :])[:, tri_i, tri_j].reshape(-1)
    if obs_mask.shape[0] <= 1:
        lag_cov_mask = np.zeros((0,), dtype=bool)
    else:
        lag_cov_mask = (obs_mask[1:, :, None] & obs_mask[:-1, None, :]).reshape(-1)
    return np.concatenate([mean_mask, same_cov_mask, lag_cov_mask])


def _spectral_svd_from_gram(S: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute singular values and right singular vectors via the P x P Gram matrix."""
    gram = S.T @ S
    eigvals, eigvecs = jnp.linalg.eigh(gram)
    eigvals = jnp.clip(eigvals, a_min=0.0)
    order = jnp.arange(eigvals.shape[0] - 1, -1, -1)
    singular_values = jnp.sqrt(eigvals[order])
    right_singular_vectors = eigvecs[:, order]
    return singular_values, right_singular_vectors


def _normalized_direction_status(value: float) -> str:
    """Bucket a normalized singular value into the Stage 4b severity bands."""
    if value > 10.0:
        return "pass"
    if value > 1.0:
        return "warn"
    return "fail"


def _validate_output_sensitivity_supported(model: SSMModel) -> None:
    """Validate preconditions for the observation-space sensitivity map."""
    observation_support = getattr(model, "observation_support", None)
    if observation_support is None or not observation_support.requires_interval_summary_handling:
        return

    manifest_names = _axis_names(
        model.spec.manifest_names,
        expected=model.spec.n_manifest,
        prefix="manifest",
    )
    unsupported_interval_families = {
        DistributionFamily.ORDERED_LOGISTIC,
        DistributionFamily.CATEGORICAL,
    }
    unsupported_manifests = [
        manifest_names[idx]
        for idx, (support_kind, dist) in enumerate(
            zip(observation_support.support_kinds, model.spec.manifest_dists, strict=False)
        )
        if support_kind == "interval" and dist in unsupported_interval_families
    ]
    if unsupported_manifests:
        unsupported = ", ".join(unsupported_manifests)
        raise OutputSensitivityUnsupportedError(
            "interval-summary sensitivity requires observation families with a "
            f"mean-parameter likelihood; unsupported interval manifests: {unsupported}"
        )


def _split_scalar_parameter_name(parameter: str) -> tuple[str, int]:
    """Split ``site_name[idx]`` strings into their site name and flat index."""
    match = _SCALAR_PARAMETER_INDEX_RE.fullmatch(parameter)
    if match is None:
        return parameter, 0
    return match.group("site"), int(match.group("index"))


def _axis_names(
    names: list[str] | None,
    *,
    expected: int,
    prefix: str,
) -> list[str]:
    """Return axis names with deterministic fallbacks when metadata is incomplete."""
    resolved = [str(name) for name in (names or []) if name]
    if len(resolved) >= expected:
        return resolved[:expected]
    return resolved + [f"{prefix}_{idx}" for idx in range(len(resolved), expected)]


def _binding_index_for_model(model: SSMModel) -> dict[tuple[str, int], str]:
    """Index compiler parameter bindings by sample site and flat index."""
    binding_index: dict[tuple[str, int], str] = {}
    for binding in list(getattr(model, "parameter_bindings", []) or []):
        if not isinstance(binding, dict):
            continue
        site_name = binding.get("site_name")
        flat_index = binding.get("flat_index")
        parameter = binding.get("parameter")
        if not isinstance(site_name, str) or not isinstance(flat_index, int):
            continue
        if not isinstance(parameter, str) or not parameter:
            continue
        binding_index[(site_name, flat_index)] = parameter
    return binding_index


def _fallback_interpretable_parameter_name(
    spec: SSMSpec,
    site_name: str,
    flat_index: int,
    *,
    structure_runtime: SSMStructureRuntime,
) -> str:
    """Resolve a best-effort semantic alias for one scalar sample-site element."""
    latent_names = _axis_names(spec.latent_names, expected=spec.n_latent, prefix="latent")
    manifest_names = _axis_names(spec.manifest_names, expected=spec.n_manifest, prefix="manifest")

    if site_name == "drift_diag_free" and flat_index < structure_runtime.n_drift_diag:
        latent_idx = structure_runtime.drift_diag_positions[flat_index]
        return f"rho_{latent_names[latent_idx]}"
    if site_name == "drift_offdiag_free" and flat_index < structure_runtime.n_drift_offdiag:
        effect_idx, cause_idx = structure_runtime.offdiag_positions[flat_index]
        return f"beta_{latent_names[cause_idx]}_{latent_names[effect_idx]}"
    if site_name == "diffusion_diag_free" and flat_index < structure_runtime.n_diffusion_diag:
        latent_idx = structure_runtime.diffusion_diag_positions[flat_index]
        return f"sigma_{latent_names[latent_idx]}"
    if site_name == "diffusion_lower_free" and flat_index < structure_runtime.n_diffusion_lower:
        row, col = structure_runtime.diffusion_lower_positions[flat_index]
        return f"cor_{latent_names[col]}_{latent_names[row]}"
    if site_name == "cint_free" and flat_index < structure_runtime.n_cint:
        latent_idx = structure_runtime.cint_free_positions[flat_index]
        return f"cint_{latent_names[latent_idx]}"
    if site_name == "lambda_free" and flat_index < structure_runtime.n_lambda_free:
        manifest_idx, latent_idx = structure_runtime.lambda_free_positions[flat_index]
        return f"lambda_{manifest_names[manifest_idx]}_{latent_names[latent_idx]}"
    if site_name == "manifest_means_free" and flat_index < structure_runtime.n_manifest_means:
        manifest_idx = structure_runtime.manifest_means_free_positions[flat_index]
        return f"manifest_mean_{manifest_names[manifest_idx]}"
    if site_name == "manifest_var_diag_free" and flat_index < structure_runtime.n_manifest_var_diag:
        manifest_idx = structure_runtime.manifest_var_free_positions[flat_index]
        return f"obs_sd_{manifest_names[manifest_idx]}"
    if site_name == "t0_means_free" and flat_index < structure_runtime.n_t0_means:
        latent_idx = structure_runtime.t0_means_free_positions[flat_index]
        return f"t0_mean_{latent_names[latent_idx]}"
    if site_name == "t0_var_diag_free" and flat_index < structure_runtime.n_t0_diag:
        latent_idx = structure_runtime.t0_diag_free_positions[flat_index]
        return f"t0_sd_{latent_names[latent_idx]}"
    if site_name == "t0_var_lower_free" and flat_index < structure_runtime.n_t0_correlation:
        row, col = structure_runtime.t0_correlation_positions[flat_index]
        return f"cor0_{latent_names[col]}_{latent_names[row]}"
    return site_name if flat_index == 0 else f"{site_name}[{flat_index}]"


def _interpretable_parameter_name_map(
    model: SSMModel,
    scalar_names: list[str],
) -> dict[str, str]:
    """Resolve semantic display names for all scalar parameters."""
    binding_index = _binding_index_for_model(model)
    structure_runtime = getattr(model, "_structure_runtime", None)
    if not isinstance(structure_runtime, SSMStructureRuntime):
        structure_runtime = SSMStructureRuntime(model.spec)

    resolved: dict[str, str] = {}
    for scalar_name in scalar_names:
        site_name, flat_index = _split_scalar_parameter_name(scalar_name)
        interpretable = binding_index.get((site_name, flat_index))
        if interpretable is None:
            interpretable = _fallback_interpretable_parameter_name(
                model.spec,
                site_name,
                flat_index,
                structure_runtime=structure_runtime,
            )
        resolved[scalar_name] = interpretable
    return resolved


def output_sensitivity_analysis(
    model: SSMModel,
    times: jnp.ndarray,
    observations: jnp.ndarray | None = None,
    n_draws: int = 8,
    seed: int = 42,
    sweep_context: ParametricIdContext | None = None,
) -> OutputSensitivityResult:
    """Pre-inference parametric identifiability via output sensitivity analysis."""
    _validate_output_sensitivity_supported(model)
    rng_key = random.PRNGKey(seed)
    context = sweep_context or get_stage4b_sweep_context(model)

    n_parameters = context.flat_dim
    scalar_names = context.scalar_names
    prior_state = model.get_prior_runtime_bundle().prior_state
    semantic_mask = _observation_semantic_mask(
        context.spec,
        times,
        getattr(model, "observation_support", None),
    )

    prior_z, rng_key = sample_prior_unconstrained(
        rng_key,
        context.registry,
        prior_state,
        n_samples=n_draws,
    )
    prior_std_draws = min(64, max(32, n_draws * 4))
    prior_z_std, rng_key = sample_prior_unconstrained(
        rng_key,
        context.registry,
        prior_state,
        n_samples=prior_std_draws,
    )
    prior_std = jnp.std(prior_z_std, axis=0)
    prior_std = jnp.maximum(prior_std, NUMERICAL_EPSILON)

    output_mask = _build_sensitivity_output_mask(observations, semantic_mask=semantic_mask)
    if output_mask is None:
        n_observations = int(context.predict_moments_fn(prior_z[0], times).shape[0])
    else:
        n_observations = int(output_mask.sum())

    def _per_param_effective_sv(V, sv):
        weight_threshold = 0.1
        effective = jnp.full(n_parameters, float(jnp.max(sv)))
        for param_idx in range(n_parameters):
            significant = jnp.abs(V[param_idx, :]) > weight_threshold
            if jnp.any(significant):
                effective = effective.at[param_idx].set(
                    jnp.min(jnp.where(significant, sv[: V.shape[1]], jnp.inf))
                )
        return effective

    all_sv = []
    all_col_norms = []
    all_effective_sv = []
    all_norm_sv = []
    all_norm_effective_sv = []
    all_norm_right_vectors = []
    skipped_nonfinite_draws = 0

    for draw_idx in range(n_draws):
        z_0 = prior_z[draw_idx]
        S = context.jacobian_fn(z_0, times)
        if output_mask is not None:
            S = S[output_mask]
        if not bool(jnp.all(jnp.isfinite(S))):
            skipped_nonfinite_draws += 1
            continue

        sv, V = _spectral_svd_from_gram(S)
        col_norms = jnp.linalg.norm(S, axis=0)
        if not bool(jnp.all(jnp.isfinite(sv))) or not bool(jnp.all(jnp.isfinite(col_norms))):
            skipped_nonfinite_draws += 1
            continue
        all_sv.append(sv)
        all_col_norms.append(col_norms)
        all_effective_sv.append(_per_param_effective_sv(V, sv))

        row_scales = context.row_scales_fn(z_0, times)
        if output_mask is not None:
            row_scales = row_scales[output_mask]
        row_scales = jnp.maximum(row_scales, NUMERICAL_EPSILON)
        if not bool(jnp.all(jnp.isfinite(row_scales))):
            skipped_nonfinite_draws += 1
            all_sv.pop()
            all_col_norms.pop()
            all_effective_sv.pop()
            continue
        S_norm = (prior_std[None, :] / row_scales[:, None]) * S
        sv_n, V_n = _spectral_svd_from_gram(S_norm)
        if not bool(jnp.all(jnp.isfinite(S_norm))) or not bool(jnp.all(jnp.isfinite(sv_n))):
            skipped_nonfinite_draws += 1
            all_sv.pop()
            all_col_norms.pop()
            all_effective_sv.pop()
            continue
        all_norm_sv.append(sv_n)
        all_norm_effective_sv.append(_per_param_effective_sv(V_n, sv_n))
        all_norm_right_vectors.append(V_n)

    if not all_sv:
        raise RuntimeError(
            "output sensitivity analysis produced no finite prior draws after screening"
        )
    if skipped_nonfinite_draws:
        logger.warning(
            "Output sensitivity analysis skipped %d/%d non-finite prior draws",
            skipped_nonfinite_draws,
            n_draws,
        )

    sv_matrix = jnp.stack(all_sv)
    col_norm_matrix = jnp.stack(all_col_norms)
    eff_sv_matrix = jnp.stack(all_effective_sv)
    norm_sv_matrix = jnp.stack(all_norm_sv)
    norm_eff_sv_matrix = jnp.stack(all_norm_effective_sv)

    median_sv = jnp.median(sv_matrix, axis=0)
    median_col_norms = jnp.median(col_norm_matrix, axis=0)
    median_eff_sv = jnp.median(eff_sv_matrix, axis=0)
    median_norm_eff_sv = jnp.median(norm_eff_sv_matrix, axis=0)

    sv_max = float(jnp.max(median_sv))
    median_norm_sv = jnp.median(norm_sv_matrix, axis=0)
    deficiency_count = int(jnp.sum(median_norm_sv < 1.0))

    interpretable_names = _interpretable_parameter_name_map(model, scalar_names)
    representative_idx = int(
        jnp.argmin(jnp.sum(jnp.abs(norm_sv_matrix - median_norm_sv[None, :]), axis=1))
    )
    representative_norm_v = np.asarray(all_norm_right_vectors[representative_idx], dtype=float)

    weak_directions = []
    highlighted_indices = [
        idx
        for idx, value in sorted(
            enumerate(np.asarray(median_norm_sv, dtype=float)),
            key=lambda item: item[1],
        )
        if _normalized_direction_status(float(value)) != "pass"
    ]
    for direction_idx in highlighted_indices:
        loadings = representative_norm_v[:, direction_idx].copy()
        top_indices = np.argsort(np.abs(loadings))[::-1][: min(15, loadings.shape[0])]
        if top_indices.size > 0 and loadings[top_indices[0]] < 0:
            loadings *= -1.0

        top_loadings = []
        for param_idx in top_indices:
            scalar_name = scalar_names[param_idx]
            loading = float(loadings[param_idx])
            top_loadings.append(
                {
                    "parameter": scalar_name,
                    "interpretable_parameter": interpretable_names[scalar_name],
                    "loading": loading,
                    "abs_loading": float(abs(loading)),
                }
            )

        normalized_sv_k = float(median_norm_sv[direction_idx])
        weak_directions.append(
            {
                "index": direction_idx + 1,
                "singular_value": float(median_sv[direction_idx]),
                "normalized_singular_value": normalized_sv_k,
                "status": _normalized_direction_status(normalized_sv_k),
                "top_loadings": top_loadings,
            }
        )

    per_param = []
    for param_idx, scalar_name in enumerate(scalar_names):
        norm_k = float(median_col_norms[param_idx])
        eff_sv_k = float(median_eff_sv[param_idx])
        norm_eff_sv_k = float(median_norm_eff_sv[param_idx])

        if eff_sv_k > 1e-3 * sv_max:
            sv_status = "pass"
        elif eff_sv_k > 1e-6 * sv_max:
            sv_status = "warn"
        else:
            sv_status = "fail"

        if norm_eff_sv_k > 10:
            norm_sv_status = "pass"
        elif norm_eff_sv_k > 1:
            norm_sv_status = "warn"
        else:
            norm_sv_status = "fail"

        per_param.append(
            {
                "parameter": scalar_name,
                "interpretable_parameter": interpretable_names[scalar_name],
                "sensitivity_norm": norm_k,
                "effective_sv": eff_sv_k,
                "sv_status": sv_status,
                "normalized_effective_sv": norm_eff_sv_k,
                "normalized_sv_status": norm_sv_status,
                "identifiable": sv_status != "fail",
            }
        )

    return OutputSensitivityResult(
        singular_values=[float(value) for value in median_sv],
        normalized_singular_values=[float(value) for value in median_norm_sv],
        deficiency_count=deficiency_count,
        weak_directions=weak_directions,
        per_parameter=per_param,
        n_draws=len(all_sv),
        n_observations=n_observations,
        n_parameters=n_parameters,
    )
