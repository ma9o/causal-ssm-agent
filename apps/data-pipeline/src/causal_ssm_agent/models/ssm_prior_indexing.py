"""Prior index construction for semantic parameter names -> SSM prior slots."""

from __future__ import annotations

from typing import TYPE_CHECKING

from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.models.ssm.parameter_names import (
    resolve_initial_state_correlation_bindings,
)
from causal_ssm_agent.models.ssm_compilation_common import PriorIndexMaps, split_compound_name
from causal_ssm_agent.orchestrator.schemas_model import ModelSpec, ParameterRole

if TYPE_CHECKING:
    from causal_ssm_agent.models.ssm.model import SSMSpec

logger = get_prefect_logger("causal_ssm_agent.models.ssm_compilation")


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

    if ssm_spec is None or not model_spec:
        return (
            offdiag_index,
            lambda_index,
            diag_index,
            diffusion_diag_index,
            diffusion_offdiag_index,
            t0_offdiag_index,
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
        )

    latent_names = ssm_spec.latent_names or []
    latent_idx_map = {name: idx for idx, name in enumerate(latent_names)}
    latent_name_set = set(latent_idx_map)
    strict_structure = causal_spec is not None

    for parameter in spec_obj.parameters:
        if parameter.role != ParameterRole.AR_COEFFICIENT:
            continue
        construct = parameter.name.removeprefix("rho_").removeprefix("ar_")
        if construct in latent_idx_map:
            diag_index[parameter.name] = ("drift_diag", latent_idx_map[construct])
        elif strict_structure:
            raise ValueError(
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
            raise ValueError(
                "RESIDUAL_SD parameter does not reference a construct in causal_spec: "
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
                    raise ValueError(message)
                logger.warning("%s", message)
                continue
            cause_name, effect_name = result
            position = (latent_idx_map[effect_name], latent_idx_map[cause_name])
            if position in positions:
                offdiag_index[parameter.name] = ("drift_offdiag", positions.index(position))
            elif strict_structure:
                raise ValueError(
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
                    raise ValueError(message)
                logger.warning("%s", message)
                continue
            indicator_name, construct_name = result
            position = (manifest_idx_map[indicator_name], latent_idx_map[construct_name])
            if position in positions:
                lambda_index[parameter.name] = ("lambda_free", positions.index(position))
            elif strict_structure:
                raise ValueError(
                    "LOADING parameter does not correspond to a free loading in causal_spec: "
                    f"{parameter.name!r}"
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
                    raise ValueError(message)
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
                raise ValueError(
                    "CORRELATION parameter does not correspond to a modeled latent pair: "
                    f"{parameter.name!r}"
                )

    if ssm_spec.t0_var != "diag":
        try:
            bindings = resolve_initial_state_correlation_bindings(latent_names, spec_obj)
        except ValueError as exc:
            if strict_structure:
                raise
            logger.warning("%s", exc)
            bindings = []

        modeled_pairs = {
            (row_idx, col_idx)
            for row_idx in range(ssm_spec.n_latent)
            for col_idx in range(row_idx)
            if ssm_spec.t0_correlation_mask is None or bool(ssm_spec.t0_correlation_mask[row_idx, col_idx])
        }
        retained_bindings = []
        for binding in bindings:
            position = (binding.row, binding.col)
            if position in modeled_pairs:
                retained_bindings.append(binding)
            elif strict_structure:
                raise ValueError(
                    "INITIAL_STATE_CORRELATION parameter does not correspond to a modeled "
                    f"initial-state pair: {binding.parameter_name!r}"
                )
        for dense_index, binding in enumerate(retained_bindings):
            t0_offdiag_index[binding.parameter_name] = (
                "t0_var_offdiag",
                dense_index,
            )

    return (
        offdiag_index,
        lambda_index,
        diag_index,
        diffusion_diag_index,
        diffusion_offdiag_index,
        t0_offdiag_index,
    )
