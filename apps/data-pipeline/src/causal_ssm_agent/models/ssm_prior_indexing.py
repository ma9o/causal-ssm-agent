"""Prior index construction for semantic parameter names -> SSM prior slots."""

from __future__ import annotations

from typing import TYPE_CHECKING

from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.models.compilation_errors import AggregatedCompileError
from causal_ssm_agent.models.ssm.parameter_names import (
    resolve_initial_state_correlation_bindings,
)
from causal_ssm_agent.models.ssm_compilation_common import PriorIndexMaps, split_compound_name
from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily, ModelSpec, ParameterRole

if TYPE_CHECKING:
    from causal_ssm_agent.models.ssm.model import SSMSpec

logger = get_prefect_logger("causal_ssm_agent.models.ssm_compilation")


class PriorIndexingError(AggregatedCompileError):
    """Aggregate independent structural binding failures for strict causal specs."""

    header = "Prior index binding failed"


def build_prior_index_maps(
    ssm_spec: SSMSpec | None,
    model_spec: ModelSpec | dict | None,
    *,
    causal_spec: dict | None = None,
) -> PriorIndexMaps:
    """Build parameter-name -> (SSMPriors field, flat index) maps."""
    offdiag_index: dict[str, tuple[str, int]] = {}
    lambda_index: dict[str, tuple[str, int]] = {}
    diag_index: dict[str, tuple[str, int]] = {}
    diffusion_diag_index: dict[str, tuple[str, int]] = {}
    diffusion_offdiag_index: dict[str, tuple[str, int]] = {}
    t0_offdiag_index: dict[str, tuple[str, int]] = {}
    t0_mean_index: dict[str, tuple[str, int]] = {}
    t0_sd_index: dict[str, tuple[str, int]] = {}
    manifest_var_index: dict[str, tuple[str, int]] = {}
    observation_site_index: dict[str, tuple[str, int]] = {}

    if ssm_spec is None or not model_spec:
        return (
            offdiag_index,
            lambda_index,
            diag_index,
            diffusion_diag_index,
            diffusion_offdiag_index,
            t0_offdiag_index,
            t0_mean_index,
            t0_sd_index,
            manifest_var_index,
            observation_site_index,
        )

    if isinstance(model_spec, dict):
        spec_obj = ModelSpec.model_validate(model_spec)
    elif isinstance(model_spec, ModelSpec):
        spec_obj = model_spec
    else:
        return (
            offdiag_index,
            lambda_index,
            diag_index,
            diffusion_diag_index,
            diffusion_offdiag_index,
            t0_offdiag_index,
            t0_mean_index,
            t0_sd_index,
            manifest_var_index,
            observation_site_index,
        )

    latent_names = ssm_spec.latent_names or []
    latent_idx_map = {name: idx for idx, name in enumerate(latent_names)}
    latent_name_set = set(latent_idx_map)
    strict_structure = causal_spec is not None
    errors: list[str] = []

    for parameter in spec_obj.parameters:
        if parameter.role != ParameterRole.AR_COEFFICIENT:
            continue
        construct = parameter.name.removeprefix("rho_").removeprefix("ar_")
        if construct in latent_idx_map:
            diag_index[parameter.name] = ("drift_diag", latent_idx_map[construct])
        elif strict_structure:
            errors.append(
                "AR parameter does not reference a construct in causal_spec: "
                f"{parameter.name!r} not in {sorted(latent_idx_map)}"
            )

    for parameter in spec_obj.parameters:
        if parameter.role != ParameterRole.RESIDUAL_SD:
            continue
        construct = parameter.name.removeprefix("sigma_")
        if construct in latent_idx_map:
            diffusion_diag_index[parameter.name] = ("diffusion_diag", latent_idx_map[construct])
        elif strict_structure:
            errors.append(
                "RESIDUAL_SD parameter does not reference a construct in causal_spec: "
                f"{parameter.name!r} not in {sorted(latent_idx_map)}"
            )

    for parameter in spec_obj.parameters:
        if parameter.role != ParameterRole.INITIAL_STATE_MEAN:
            continue
        construct = parameter.name.removeprefix("t0_mean_")
        if construct in latent_idx_map:
            t0_mean_index[parameter.name] = ("t0_means", latent_idx_map[construct])
        elif strict_structure:
            errors.append(
                "INITIAL_STATE_MEAN parameter does not reference a construct in causal_spec: "
                f"{parameter.name!r} not in {sorted(latent_idx_map)}"
            )

    for parameter in spec_obj.parameters:
        if parameter.role != ParameterRole.INITIAL_STATE_SD:
            continue
        construct = parameter.name.removeprefix("t0_sd_")
        if construct in latent_idx_map:
            t0_sd_index[parameter.name] = ("t0_var_diag", latent_idx_map[construct])
        elif strict_structure:
            errors.append(
                "INITIAL_STATE_SD parameter does not reference a construct in causal_spec: "
                f"{parameter.name!r} not in {sorted(latent_idx_map)}"
            )

    if ssm_spec.drift_mask is not None:
        positions: list[tuple[int, int]] = []
        for effect_idx in range(ssm_spec.n_latent):
            for cause_idx in range(ssm_spec.n_latent):
                if effect_idx != cause_idx and ssm_spec.drift_mask[effect_idx, cause_idx]:
                    positions.append((effect_idx, cause_idx))

        for parameter in spec_obj.parameters:
            if parameter.role != ParameterRole.FIXED_EFFECT:
                continue
            compound = parameter.name.removeprefix("beta_")
            result = split_compound_name(compound, latent_name_set, latent_name_set)
            if result is None:
                message = (
                    "Could not parse FIXED_EFFECT parameter "
                    f"{parameter.name!r} into (cause, effect) from known latents {sorted(latent_name_set)}"
                )
                if strict_structure:
                    errors.append(message)
                    continue
                logger.warning("%s", message)
                continue
            cause_name, effect_name = result
            position = (latent_idx_map[effect_name], latent_idx_map[cause_name])
            if position in positions:
                offdiag_index[parameter.name] = ("drift_offdiag", positions.index(position))
            elif strict_structure:
                errors.append(
                    "FIXED_EFFECT parameter does not correspond to an edge in causal_spec: "
                    f"{parameter.name!r}"
                )

    if ssm_spec.lambda_mask is not None:
        manifest_names = ssm_spec.manifest_names or []
        manifest_idx_map = {name: idx for idx, name in enumerate(manifest_names)}
        manifest_name_set = set(manifest_idx_map)

        positions: list[tuple[int, int]] = []
        for manifest_idx in range(ssm_spec.n_manifest):
            for latent_idx in range(ssm_spec.n_latent):
                if ssm_spec.lambda_mask[manifest_idx, latent_idx]:
                    positions.append((manifest_idx, latent_idx))

        for parameter in spec_obj.parameters:
            if parameter.role != ParameterRole.LOADING:
                continue
            compound = parameter.name.removeprefix("lambda_")
            result = split_compound_name(compound, manifest_name_set, latent_name_set)
            if result is None:
                message = (
                    "Could not parse LOADING parameter "
                    f"{parameter.name!r} into (indicator, construct) from known manifests "
                    f"{sorted(manifest_name_set)} / latents {sorted(latent_name_set)}"
                )
                if strict_structure:
                    errors.append(message)
                    continue
                logger.warning("%s", message)
                continue
            indicator_name, construct_name = result
            position = (manifest_idx_map[indicator_name], latent_idx_map[construct_name])
            if position in positions:
                lambda_index[parameter.name] = ("lambda_free", positions.index(position))
            elif strict_structure:
                errors.append(
                    "LOADING parameter does not correspond to a free loading in causal_spec: "
                    f"{parameter.name!r}"
                )

    manifest_names = ssm_spec.manifest_names or []
    manifest_idx_map = {name: idx for idx, name in enumerate(manifest_names)}
    if not (not isinstance(ssm_spec.manifest_var, str) and ssm_spec.manifest_var_mask is None):
        if ssm_spec.manifest_var_mask is None:
            free_manifest_indices = list(range(ssm_spec.n_manifest))
        else:
            free_manifest_indices = [
                idx for idx, is_free in enumerate(ssm_spec.manifest_var_mask) if bool(is_free)
            ]
        free_manifest_lookup = {
            manifest_idx: flat_idx for flat_idx, manifest_idx in enumerate(free_manifest_indices)
        }
        for parameter in spec_obj.parameters:
            if parameter.role != ParameterRole.MEASUREMENT_ERROR_SD:
                continue
            indicator_name = parameter.name.removeprefix("obs_sd_")
            manifest_idx = manifest_idx_map.get(indicator_name)
            if manifest_idx is None:
                message = (
                    "Could not parse MEASUREMENT_ERROR_SD parameter "
                    f"{parameter.name!r} into a known manifest from {sorted(manifest_idx_map)}"
                )
                if strict_structure:
                    errors.append(message)
                    continue
                logger.warning("%s", message)
                continue
            flat_idx = free_manifest_lookup.get(manifest_idx)
            if flat_idx is None:
                if strict_structure:
                    errors.append(
                        "MEASUREMENT_ERROR_SD parameter does not correspond to a free manifest "
                        f"noise term in causal_spec: {parameter.name!r}"
                    )
                continue
            manifest_var_index[parameter.name] = ("manifest_var_diag", flat_idx)

    available_observation_sites: set[str] = set()
    manifest_dists = ssm_spec.manifest_dists or [ssm_spec.manifest_dist] * ssm_spec.n_manifest
    manifest_dist_set = set(manifest_dists)
    if DistributionFamily.STUDENT_T in manifest_dist_set:
        available_observation_sites.add("obs_df")
    if DistributionFamily.GAMMA in manifest_dist_set:
        available_observation_sites.add("obs_shape")
    if DistributionFamily.NEGATIVE_BINOMIAL in manifest_dist_set:
        available_observation_sites.add("obs_r")
    if DistributionFamily.BETA in manifest_dist_set:
        available_observation_sites.add("obs_concentration")

    if ssm_spec.manifest_level_counts is not None:
        level_counts_list = list(ssm_spec.manifest_level_counts)
        max_levels = max(level_counts_list) if level_counts_list else 0
        max_cutpoints = max(max_levels - 1, 0)
        if DistributionFamily.ORDERED_LOGISTIC in manifest_dist_set and max_cutpoints > 0:
            available_observation_sites.add("obs_ordered_base")
            if max_cutpoints > 1:
                available_observation_sites.add("obs_ordered_gaps")
        if DistributionFamily.CATEGORICAL in manifest_dist_set and max_cutpoints > 0:
            available_observation_sites.update({"obs_cat_intercepts", "obs_cat_slopes"})

    for parameter in spec_obj.parameters:
        if parameter.role not in {
            ParameterRole.OBSERVATION_HYPERPARAMETER,
            ParameterRole.OBSERVATION_HYPERPARAMETER_POSITIVE,
        }:
            continue
        if parameter.name in available_observation_sites:
            observation_site_index[parameter.name] = (parameter.name, 0)
        elif strict_structure:
            errors.append(
                "Observation hyperparameter does not correspond to an active compiled "
                f"observation site: {parameter.name!r}"
            )

    if ssm_spec.diffusion == "free":
        lower_positions: list[tuple[int, int]] = []
        for i in range(ssm_spec.n_latent):
            for j in range(i):
                lower_positions.append((i, j))

        for parameter in spec_obj.parameters:
            if parameter.role != ParameterRole.CORRELATION:
                continue
            compound = parameter.name.removeprefix("cor_")
            result = split_compound_name(compound, latent_name_set, latent_name_set)
            if result is None:
                message = (
                    "Could not parse CORRELATION parameter "
                    f"{parameter.name!r} into (state1, state2) from known latents {sorted(latent_name_set)}"
                )
                if strict_structure:
                    errors.append(message)
                    continue
                logger.warning("%s", message)
                continue
            state1_name, state2_name = result
            idx1 = latent_idx_map[state1_name]
            idx2 = latent_idx_map[state2_name]
            position = (max(idx1, idx2), min(idx1, idx2))
            if position in lower_positions:
                diffusion_offdiag_index[parameter.name] = (
                    "diffusion_offdiag",
                    lower_positions.index(position),
                )
            elif strict_structure:
                errors.append(
                    "CORRELATION parameter does not correspond to a modeled latent pair: "
                    f"{parameter.name!r}"
                )

    if ssm_spec.t0_var != "diag":
        try:
            bindings = resolve_initial_state_correlation_bindings(latent_names, spec_obj)
        except ValueError as exc:
            if strict_structure:
                errors.append(str(exc))
            logger.warning("%s", exc)
            bindings = []

        modeled_pairs = {
            (row_idx, col_idx)
            for row_idx in range(ssm_spec.n_latent)
            for col_idx in range(row_idx)
            if ssm_spec.t0_correlation_mask is None
            or bool(ssm_spec.t0_correlation_mask[row_idx, col_idx])
        }
        retained_bindings = []
        for binding in bindings:
            position = (binding.row, binding.col)
            if position in modeled_pairs:
                retained_bindings.append(binding)
            elif strict_structure:
                errors.append(
                    "INITIAL_STATE_CORRELATION parameter does not correspond to a modeled "
                    f"initial-state pair: {binding.parameter_name!r}"
                )
        for dense_index, binding in enumerate(retained_bindings):
            t0_offdiag_index[binding.parameter_name] = (
                "t0_var_offdiag",
                dense_index,
            )

    if errors:
        raise PriorIndexingError(errors)

    return (
        offdiag_index,
        lambda_index,
        diag_index,
        diffusion_diag_index,
        diffusion_offdiag_index,
        t0_offdiag_index,
        t0_mean_index,
        t0_sd_index,
        manifest_var_index,
        observation_site_index,
    )
