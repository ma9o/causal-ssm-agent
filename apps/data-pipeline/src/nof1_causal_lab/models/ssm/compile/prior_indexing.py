"""Semantic prior binding for model-spec parameters -> SSM sample sites."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.model_spec import ModelSpec, ParameterRole
from nof1_causal_lab.flows import get_prefect_logger
from nof1_causal_lab.models.compilation_errors import AggregatedCompileError
from nof1_causal_lab.models.ssm.compile.common import axis_names_with_fallback
from nof1_causal_lab.models.ssm.parameter_layout import SSMParameterLayout
from nof1_causal_lab.models.ssm.parameter_names import (
    resolve_initial_state_correlation_bindings,
    split_compound_name,
)
from nof1_causal_lab.models.ssm.parameterization import build_site_registry
from nof1_causal_lab.models.ssm.structure.sites import (
    PriorAuthoringTransform,
    SemanticBinding,
    SiteDescriptor,
    SiteKind,
    site_size,
)

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.model import SSMSpec

logger = get_prefect_logger("nof1_causal_lab.models.ssm.compile.inputs")


class PriorIndexingError(AggregatedCompileError):
    """Aggregate independent structural binding failures for strict causal specs."""

    header = "Prior binding failed"


@dataclass(frozen=True)
class SemanticBindingRegistry:
    """Named replacement for the old positional prior-index tuple."""

    bindings: tuple[SemanticBinding, ...] = field(default_factory=tuple)

    @property
    def by_parameter(self) -> dict[str, SemanticBinding]:
        return {binding.parameter_name: binding for binding in self.bindings}

    def get(self, parameter: str) -> SemanticBinding | None:
        return self.by_parameter.get(parameter)

    def by_site_kind(self, site_kind: SiteKind) -> dict[str, SemanticBinding]:
        return {
            binding.parameter_name: binding
            for binding in self.bindings
            if binding.site_kind == site_kind
        }


def empty_prior_bindings() -> SemanticBindingRegistry:
    """Return an empty semantic binding registry for spec-only code paths."""
    return SemanticBindingRegistry(())


def _model_spec_obj(model_spec: ModelSpec | dict) -> ModelSpec:
    if isinstance(model_spec, dict):
        return ModelSpec.model_validate(model_spec)
    if isinstance(model_spec, ModelSpec):
        return model_spec
    raise TypeError(
        "build_semantic_prior_bindings() requires model_spec to be a ModelSpec or dict, "
        f"got {type(model_spec).__name__}."
    )


def _site_by_prior_field(
    sites: list[SiteDescriptor],
    prior_field: str,
) -> SiteDescriptor | None:
    matches = [site for site in sites if site.priors_field == prior_field]
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(
            f"Prior field {prior_field!r} maps to multiple active sample sites: "
            f"{[site.name for site in matches]}"
        )
    return matches[0]


def _binding_for_site(
    *,
    parameter: str,
    site: SiteDescriptor,
    flat_index: int,
    transform: PriorAuthoringTransform = PriorAuthoringTransform.IDENTITY,
    construct_names: tuple[str, ...] = (),
    indicator_names: tuple[str, ...] = (),
    component_index: int | None = None,
    effect_idx: int | None = None,
    cause_idx: int | None = None,
) -> SemanticBinding:
    return SemanticBinding(
        parameter_name=parameter,
        site_name=site.name,
        prior_field=site.priors_field or site.name,
        flat_index=flat_index,
        site_kind=site.site_kind,
        transform=transform,
        construct_names=construct_names,
        indicator_names=indicator_names,
        component_index=component_index,
        effect_idx=effect_idx,
        cause_idx=cause_idx,
    )


def _add_site_binding(
    bindings: dict[str, SemanticBinding],
    parameter_name: str,
    binding: SemanticBinding,
) -> None:
    existing = bindings.get(parameter_name)
    if existing is not None:
        raise ValueError(
            f"Parameter {parameter_name!r} maps to multiple compiled sample sites: "
            f"{existing.site_name}[{existing.flat_index}] and "
            f"{binding.site_name}[{binding.flat_index}]"
        )
    bindings[parameter_name] = (
        binding
        if binding.parameter_name == parameter_name
        else replace(binding, parameter_name=parameter_name)
    )


def _component_binding_candidates(
    ssm_spec: SSMSpec,
    active_sites: list[SiteDescriptor],
) -> dict[str, SemanticBinding]:
    """Build component-owned semantic binding candidates for dynamics sites."""
    from nof1_causal_lab.models.ssm.dynamics.composite import (
        iter_component_semantic_bindings,
    )

    sites_by_name = {site.name: site for site in active_sites}
    latent_names = axis_names_with_fallback(
        ssm_spec.latent_names,
        expected=ssm_spec.n_latent,
        prefix="latent",
    )
    candidates: dict[str, SemanticBinding] = {}

    def _put(name: str, binding: SemanticBinding) -> None:
        if name in candidates:
            raise ValueError(f"Component semantic parameter name {name!r} is ambiguous.")
        candidates[name] = binding

    for binding in iter_component_semantic_bindings(
        ssm_spec.drift_spec,
        latent_names=tuple(latent_names),
    ):
        site = sites_by_name.get(binding.site_name)
        if site is None:
            continue
        prior_field = binding.prior_field or site.priors_field or site.name
        _put(
            binding.parameter_name,
            replace(binding, site_kind=site.site_kind, prior_field=prior_field),
        )

    return candidates


def build_semantic_prior_bindings(
    ssm_spec: SSMSpec,
    model_spec: ModelSpec | dict,
    *,
    causal_spec: dict | None = None,
) -> SemanticBindingRegistry:
    """Build parameter-name -> compiled sample-site bindings."""
    spec_obj = _model_spec_obj(model_spec)
    latent_names = axis_names_with_fallback(
        ssm_spec.latent_names,
        expected=ssm_spec.n_latent,
        prefix="latent",
    )
    manifest_names = axis_names_with_fallback(
        ssm_spec.manifest_names,
        expected=ssm_spec.n_manifest,
        prefix="manifest",
    )
    latent_idx_map = {name: idx for idx, name in enumerate(latent_names)}
    latent_name_set = set(latent_idx_map)
    input_names = ssm_spec.input_names or []
    input_idx_map = {name: idx for idx, name in enumerate(input_names)}
    input_name_set = set(input_idx_map)
    strict_structure = causal_spec is not None
    errors: list[str] = []
    parameter_layout = SSMParameterLayout.from_spec(ssm_spec)
    active_sites = build_site_registry(ssm_spec)
    sites_by_name = {site.name: site for site in active_sites}
    component_candidates = _component_binding_candidates(ssm_spec, active_sites)
    bindings: dict[str, SemanticBinding] = {}

    def _site_for_field(prior_field: str) -> SiteDescriptor | None:
        return _site_by_prior_field(active_sites, prior_field)

    def _add(
        parameter_name: str,
        site: SiteDescriptor | None,
        flat_index: int | None,
        *,
        transform: PriorAuthoringTransform = PriorAuthoringTransform.IDENTITY,
        construct_names: tuple[str, ...] = (),
        indicator_names: tuple[str, ...] = (),
        effect_idx: int | None = None,
        cause_idx: int | None = None,
        error_message: str | None = None,
    ) -> None:
        if site is None or flat_index is None:
            if strict_structure and error_message:
                errors.append(error_message)
            return
        _add_site_binding(
            bindings,
            parameter_name,
            _binding_for_site(
                parameter=parameter_name,
                site=site,
                flat_index=flat_index,
                transform=transform,
                construct_names=construct_names,
                indicator_names=indicator_names,
                effect_idx=effect_idx,
                cause_idx=cause_idx,
            ),
        )

    for parameter in spec_obj.parameters:
        if parameter.role != ParameterRole.AR_COEFFICIENT:
            continue
        construct = parameter.name.removeprefix("rho_").removeprefix("ar_")
        latent_idx = latent_idx_map.get(construct)
        flat_idx = (
            parameter_layout.drift_base_decay_index.get(latent_idx)
            if latent_idx is not None
            else None
        )
        if flat_idx is not None:
            _add(
                parameter.name,
                _site_for_field("drift_base_decay"),
                flat_idx,
                transform=PriorAuthoringTransform.DT_PERSISTENCE_TO_CT_DECAY,
                construct_names=(construct,),
            )
            continue
        candidate = component_candidates.get(f"rho_{construct}")
        if candidate is not None:
            _add_site_binding(bindings, parameter.name, candidate)
        elif strict_structure:
            errors.append(
                "AR parameter does not correspond to a free dynamics decay term in "
                f"causal_spec: {parameter.name!r}"
            )

    for parameter in spec_obj.parameters:
        if parameter.role != ParameterRole.RESIDUAL_SD:
            continue
        construct = parameter.name.removeprefix("sigma_")
        latent_idx = latent_idx_map.get(construct)
        flat_idx = (
            parameter_layout.diffusion_diag_index.get(latent_idx)
            if latent_idx is not None
            else None
        )
        _add(
            parameter.name,
            _site_for_field("diffusion_diag"),
            flat_idx,
            construct_names=(construct,),
            error_message=(
                "RESIDUAL_SD parameter does not correspond to a free diffusion "
                f"diagonal term in causal_spec: {parameter.name!r}"
            ),
        )

    for parameter in spec_obj.parameters:
        if parameter.role != ParameterRole.INITIAL_STATE_MEAN:
            continue
        construct = parameter.name.removeprefix("t0_mean_")
        latent_idx = latent_idx_map.get(construct)
        flat_idx = parameter_layout.t0_means_free_index.get(latent_idx)
        _add(
            parameter.name,
            _site_for_field("t0_means"),
            flat_idx,
            construct_names=(construct,),
            error_message=(
                "INITIAL_STATE_MEAN parameter does not correspond to a free initial-state "
                f"mean in causal_spec: {parameter.name!r}"
            ),
        )

    for parameter in spec_obj.parameters:
        if parameter.role != ParameterRole.INITIAL_STATE_SD:
            continue
        construct = parameter.name.removeprefix("t0_sd_")
        latent_idx = latent_idx_map.get(construct)
        flat_idx = parameter_layout.t0_diag_free_index.get(latent_idx)
        _add(
            parameter.name,
            _site_for_field("t0_var_diag"),
            flat_idx,
            construct_names=(construct,),
            error_message=(
                "INITIAL_STATE_SD parameter does not correspond to a free initial-state "
                f"standard deviation in causal_spec: {parameter.name!r}"
            ),
        )

    for parameter in spec_obj.parameters:
        if parameter.role != ParameterRole.STATE_INTERCEPT:
            continue
        construct = parameter.name.removeprefix("cint_")
        latent_idx = latent_idx_map.get(construct)
        flat_idx = (
            parameter_layout.cint_free_index.get(latent_idx) if latent_idx is not None else None
        )
        if flat_idx is not None:
            _add(
                parameter.name,
                _site_for_field("cint"),
                flat_idx,
                construct_names=(construct,),
            )
            continue
        candidate = component_candidates.get(f"cint_{construct}")
        if candidate is not None:
            _add_site_binding(bindings, parameter.name, candidate)
        elif strict_structure:
            errors.append(
                "STATE_INTERCEPT parameter does not correspond to a free continuous-time "
                f"intercept in causal_spec: {parameter.name!r}"
            )

    for parameter in spec_obj.parameters:
        if parameter.role != ParameterRole.FIXED_EFFECT:
            continue
        compound = parameter.name.removeprefix("beta_")
        result = split_compound_name(compound, latent_name_set | input_name_set, latent_name_set)
        if result is None:
            message = (
                "Could not parse FIXED_EFFECT parameter "
                f"{parameter.name!r} into (cause, effect) from known causes "
                f"{sorted(latent_name_set | input_name_set)} and latent effects "
                f"{sorted(latent_name_set)}"
            )
            if strict_structure:
                errors.append(message)
            else:
                logger.warning("%s", message)
            continue
        cause_name, effect_name = result
        effect_idx = latent_idx_map[effect_name]
        if cause_name in input_idx_map:
            position = (effect_idx, input_idx_map[cause_name])
            _add(
                parameter.name,
                _site_for_field("input_effect"),
                parameter_layout.input_effect_index.get(position),
                transform=PriorAuthoringTransform.DT_EFFECT_TO_CT_RATE,
                construct_names=(cause_name, effect_name),
                effect_idx=effect_idx,
                error_message=(
                    "FIXED_EFFECT parameter does not correspond to a known-input edge "
                    f"in causal_spec: {parameter.name!r}"
                ),
            )
            continue
        cause_idx = latent_idx_map[cause_name]
        position = (effect_idx, cause_idx)
        flat_idx = parameter_layout.offdiag_index.get(position)
        if flat_idx is not None:
            _add(
                parameter.name,
                _site_for_field("drift_offdiag"),
                flat_idx,
                transform=PriorAuthoringTransform.DT_EFFECT_TO_CT_RATE,
                construct_names=(cause_name, effect_name),
                effect_idx=effect_idx,
                cause_idx=cause_idx,
            )
            continue
        candidate = component_candidates.get(parameter.name)
        if candidate is not None:
            _add_site_binding(bindings, parameter.name, candidate)
        elif strict_structure:
            errors.append(
                "FIXED_EFFECT parameter does not correspond to an edge in causal_spec: "
                f"{parameter.name!r}"
            )

    manifest_idx_map = {name: idx for idx, name in enumerate(manifest_names)}
    manifest_name_set = set(manifest_idx_map)
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
            else:
                logger.warning("%s", message)
            continue
        indicator_name, construct_name = result
        position = (manifest_idx_map[indicator_name], latent_idx_map[construct_name])
        _add(
            parameter.name,
            _site_for_field("lambda_free"),
            parameter_layout.lambda_free_index.get(position),
            construct_names=(construct_name,),
            indicator_names=(indicator_name,),
            error_message=(
                "LOADING parameter does not correspond to a free loading in causal_spec: "
                f"{parameter.name!r}"
            ),
        )

    for parameter in spec_obj.parameters:
        if parameter.role != ParameterRole.OBSERVATION_INTERCEPT:
            continue
        indicator_name = parameter.name.removeprefix("manifest_mean_")
        manifest_idx = manifest_idx_map.get(indicator_name)
        flat_idx = (
            parameter_layout.manifest_means_free_index.get(manifest_idx)
            if manifest_idx is not None
            else None
        )
        _add(
            parameter.name,
            _site_for_field("manifest_means"),
            flat_idx,
            indicator_names=(indicator_name,),
            error_message=(
                "OBSERVATION_INTERCEPT parameter does not correspond to a free manifest "
                f"intercept in causal_spec: {parameter.name!r}"
            ),
        )

    for parameter in spec_obj.parameters:
        if parameter.role != ParameterRole.MEASUREMENT_ERROR_SD:
            continue
        indicator_name = parameter.name.removeprefix("obs_sd_")
        manifest_idx = manifest_idx_map.get(indicator_name)
        flat_idx = (
            parameter_layout.manifest_var_free_index.get(manifest_idx)
            if manifest_idx is not None
            else None
        )
        _add(
            parameter.name,
            _site_for_field("manifest_var_diag"),
            flat_idx,
            indicator_names=(indicator_name,),
            error_message=(
                "MEASUREMENT_ERROR_SD parameter does not correspond to a free manifest "
                f"noise term in causal_spec: {parameter.name!r}"
            ),
        )

    for parameter in spec_obj.parameters:
        if parameter.role != ParameterRole.STATIC_STATE_SD:
            continue
        factor_idx = parameter_layout.static_factor_name_index.get(parameter.name)
        flat_idx = (
            parameter_layout.static_state_sd_free_index.get(factor_idx)
            if factor_idx is not None
            else None
        )
        _add(
            parameter.name,
            _site_for_field("static_state_sd"),
            flat_idx,
            construct_names=tuple(getattr(parameter, "construct_names", ()) or ()),
            error_message=(
                "STATIC_STATE_SD parameter does not correspond to a free compiled "
                f"baseline-factor scale: {parameter.name!r}"
            ),
        )

    for parameter in spec_obj.parameters:
        if parameter.role not in {
            ParameterRole.OBSERVATION_HYPERPARAMETER,
            ParameterRole.OBSERVATION_HYPERPARAMETER_POSITIVE,
        }:
            continue
        site = sites_by_name.get(parameter.name)
        if site is not None:
            _add(
                parameter.name,
                site,
                0,
                transform=PriorAuthoringTransform.SITE_WIDE,
            )
        elif strict_structure:
            errors.append(
                "Observation hyperparameter does not correspond to an active compiled "
                f"observation site: {parameter.name!r}"
            )

    for parameter in spec_obj.parameters:
        if parameter.role not in {
            ParameterRole.DYNAMICS_PARAMETER,
            ParameterRole.DYNAMICS_PARAMETER_POSITIVE,
        }:
            continue
        candidate = component_candidates.get(parameter.name)
        if candidate is not None:
            _add_site_binding(bindings, parameter.name, candidate)
            continue
        site = sites_by_name.get(parameter.name)
        if site is not None and site.assembly_group in {"dynamics", "drift", "cint"}:
            _add(
                parameter.name,
                site,
                0,
                transform=PriorAuthoringTransform.SITE_WIDE
                if site_size(site.shape) > 1
                else PriorAuthoringTransform.IDENTITY,
            )
            continue
        errors.append(
            f"Dynamics parameter {parameter.name!r} does not correspond to a component-owned "
            "dynamics sample site."
        )

    if parameter_layout.n_diffusion_lower > 0:
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
                else:
                    logger.warning("%s", message)
                continue
            state1_name, state2_name = result
            idx1 = latent_idx_map[state1_name]
            idx2 = latent_idx_map[state2_name]
            position = (max(idx1, idx2), min(idx1, idx2))
            _add(
                parameter.name,
                _site_for_field("diffusion_offdiag"),
                parameter_layout.diffusion_lower_index.get(position),
                construct_names=(state1_name, state2_name),
                error_message=(
                    "CORRELATION parameter does not correspond to a modeled latent pair: "
                    f"{parameter.name!r}"
                ),
            )

    if parameter_layout.n_t0_correlation > 0:
        try:
            t0_bindings = resolve_initial_state_correlation_bindings(latent_names, spec_obj)
        except ValueError as exc:
            if strict_structure:
                errors.append(str(exc))
            logger.warning("%s", exc)
            t0_bindings = []

        for binding in t0_bindings:
            position = (binding.row, binding.col)
            _add(
                binding.parameter_name,
                _site_for_field("t0_var_offdiag"),
                parameter_layout.t0_correlation_index.get(position),
                transform=PriorAuthoringTransform.INITIAL_STATE_CORRELATION,
                construct_names=(latent_names[binding.col], latent_names[binding.row]),
                error_message=(
                    "INITIAL_STATE_CORRELATION parameter does not correspond to a modeled "
                    f"initial-state pair: {binding.parameter_name!r}"
                ),
            )

    if errors:
        raise PriorIndexingError(errors)

    return SemanticBindingRegistry(tuple(bindings[name] for name in sorted(bindings)))


def check_backward_closure(
    ssm_spec: SSMSpec,
    bindings: SemanticBindingRegistry,
) -> list[str]:
    """Check that every non-likelihood free runtime site scalar has one semantic owner."""
    active_sites = build_site_registry(ssm_spec)
    bound_counts: dict[str, int] = {}
    for binding in bindings.bindings:
        if binding.transform == PriorAuthoringTransform.SITE_WIDE:
            site = next(site for site in active_sites if site.name == binding.site_name)
            bound_counts[binding.site_name] = bound_counts.get(binding.site_name, 0) + site_size(
                site.shape
            )
        else:
            bound_counts[binding.site_name] = bound_counts.get(binding.site_name, 0) + 1

    violations: list[str] = []
    for site in active_sites:
        if site.assembly_group == "likelihood":
            continue
        n_free = site_size(site.shape)
        n_bound = bound_counts.get(site.name, 0)
        if n_free != n_bound:
            violations.append(
                f"Backward closure violation in {site.name}: {n_free} free site(s), "
                f"{n_bound} bound parameter(s)"
            )
    return violations


__all__ = [
    "PriorAuthoringTransform",
    "PriorIndexingError",
    "SemanticBinding",
    "SemanticBindingRegistry",
    "build_semantic_prior_bindings",
    "check_backward_closure",
    "empty_prior_bindings",
]
