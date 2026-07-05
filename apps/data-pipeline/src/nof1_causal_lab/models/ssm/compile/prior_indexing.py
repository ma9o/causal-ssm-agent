"""Semantic prior binding for model-spec parameters -> SSM sample sites."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from nof1_causal_lab.artifacts.model_spec import ModelSpec, ParameterRole
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
    site_size,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from nof1_causal_lab.models.ssm.model import SSMSpec

logger = logging.getLogger("nof1_causal_lab.models.ssm.compile.inputs")


class PriorIndexingError(AggregatedCompileError):
    """Aggregate independent structural binding failures for strict causal specs."""

    header = "Prior index binding failed"


@dataclass(frozen=True)
class SemanticBindingRegistry:
    """Named replacement for the old positional prior-index tuple."""

    bindings: tuple[SemanticBinding, ...] = field(default_factory=tuple)

    @property
    def by_parameter(self) -> dict[str, SemanticBinding]:
        return {binding.parameter_name: binding for binding in self.bindings}

    def get(self, parameter: str) -> SemanticBinding | None:
        return self.by_parameter.get(parameter)


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
    from nof1_causal_lab.models.ssm.dynamics.spec import (
        iter_dynamics_semantic_bindings,
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

    for binding in iter_dynamics_semantic_bindings(
        ssm_spec.dynamics_spec,
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


# ---------------------------------------------------------------------------
# Role dispatch
#
# Every parameter role with a single-axis lookup (rho_*, sigma_*, t0_mean_*,
# t0_sd_*, cint_*, manifest_mean_*, obs_sd_*) follows the same shape: strip
# prefix(es), look up the construct in an axis index map, look up the position
# in a parameter_layout index, attach a binding with optional transform. The
# only variation is (prefixes, axis, prior_field, layout attr, transform,
# error message, component-fallback). _SimpleAxisRule encodes that variation.
#
# Roles that branch on cause type (FIXED_EFFECT), parse compound names
# (LOADING, CORRELATION), or use direct site lookups (STATIC_STATE_SD,
# OBSERVATION_HYPERPARAMETER*, DYNAMICS_PARAMETER*) get dedicated handlers.
# INITIAL_STATE_CORRELATION runs as a block step after the per-parameter pass
# because it has its own resolver that handles many parameters at once.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SimpleAxisRule:
    """Declarative rule for the single-axis prior-binding pattern."""

    prefixes: tuple[str, ...]
    axis_kind: str  # "latent" or "manifest"
    prior_field: str | None
    layout_index_attr: str | None
    transform: PriorAuthoringTransform = PriorAuthoringTransform.IDENTITY
    component_fallback_pattern: str | None = None
    error_template: str | None = None


@dataclass
class _BindingContext:
    """Mutable state plus helpers shared by every role handler."""

    spec_obj: ModelSpec
    ssm_spec: SSMSpec
    latent_names: list[str]
    manifest_names: list[str]
    latent_idx_map: dict[str, int]
    manifest_idx_map: dict[str, int]
    input_idx_map: dict[str, int]
    latent_name_set: set[str]
    input_name_set: set[str]
    manifest_name_set: set[str]
    parameter_layout: SSMParameterLayout
    active_sites: list[SiteDescriptor]
    sites_by_name: dict[str, SiteDescriptor]
    component_candidates: dict[str, SemanticBinding]
    bindings: dict[str, SemanticBinding] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    strict_structure: bool = False

    def site_for_field(self, prior_field: str) -> SiteDescriptor | None:
        return _site_by_prior_field(self.active_sites, prior_field)

    def add(
        self,
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
            if self.strict_structure and error_message:
                self.errors.append(error_message)
            return
        _add_site_binding(
            self.bindings,
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

    def add_site_binding(self, parameter_name: str, binding: SemanticBinding) -> None:
        _add_site_binding(self.bindings, parameter_name, binding)


def _apply_simple_axis_rule(
    rule: _SimpleAxisRule,
    parameter: Any,
    ctx: _BindingContext,
) -> None:
    construct = parameter.name
    for prefix in rule.prefixes:
        construct = construct.removeprefix(prefix)
    if rule.component_fallback_pattern is not None:
        candidate = ctx.component_candidates.get(
            rule.component_fallback_pattern.format(construct=construct)
        )
        if candidate is not None:
            ctx.add_site_binding(parameter.name, candidate)
            return
    if rule.prior_field is None or rule.layout_index_attr is None:
        if ctx.strict_structure and rule.error_template is not None:
            ctx.errors.append(rule.error_template.format(param=parameter.name))
        return

    axis_idx_map = ctx.latent_idx_map if rule.axis_kind == "latent" else ctx.manifest_idx_map
    axis_idx = axis_idx_map.get(construct)
    layout_index = getattr(ctx.parameter_layout, rule.layout_index_attr)
    flat_idx = layout_index.get(axis_idx) if axis_idx is not None else None
    if flat_idx is not None:
        ctx.add(
            parameter.name,
            ctx.site_for_field(rule.prior_field),
            flat_idx,
            transform=rule.transform,
            construct_names=(construct,),
        )
        return
    if ctx.strict_structure and rule.error_template is not None:
        ctx.errors.append(rule.error_template.format(param=parameter.name))


def _handle_fixed_effect(parameter: Any, ctx: _BindingContext) -> None:
    compound = parameter.name.removeprefix("beta_")
    result = split_compound_name(
        compound,
        ctx.latent_name_set | ctx.input_name_set,
        ctx.latent_name_set,
    )
    if result is None:
        message = (
            "Could not parse FIXED_EFFECT parameter "
            f"{parameter.name!r} into (cause, effect) from known causes "
            f"{sorted(ctx.latent_name_set | ctx.input_name_set)} and latent effects "
            f"{sorted(ctx.latent_name_set)}"
        )
        if ctx.strict_structure:
            ctx.errors.append(message)
        else:
            logger.warning("%s", message)
        return
    cause_name, effect_name = result
    effect_idx = ctx.latent_idx_map[effect_name]
    if cause_name in ctx.input_idx_map:
        position = (effect_idx, ctx.input_idx_map[cause_name])
        ctx.add(
            parameter.name,
            ctx.site_for_field("input_effect"),
            ctx.parameter_layout.input_effect_index.get(position),
            transform=PriorAuthoringTransform.DT_EFFECT_TO_CT_RATE,
            construct_names=(cause_name, effect_name),
            effect_idx=effect_idx,
            error_message=(
                "FIXED_EFFECT parameter does not correspond to a known-input edge "
                f"in causal_spec: {parameter.name!r}"
            ),
        )
        return
    candidate = ctx.component_candidates.get(parameter.name)
    if candidate is not None:
        ctx.add_site_binding(parameter.name, candidate)
    elif ctx.strict_structure:
        ctx.errors.append(
            "FIXED_EFFECT parameter does not correspond to an edge in causal_spec: "
            f"{parameter.name!r}"
        )


def _handle_loading(parameter: Any, ctx: _BindingContext) -> None:
    compound = parameter.name.removeprefix("lambda_")
    result = split_compound_name(compound, ctx.manifest_name_set, ctx.latent_name_set)
    if result is None:
        message = (
            "Could not parse LOADING parameter "
            f"{parameter.name!r} into (indicator, construct) from known manifests "
            f"{sorted(ctx.manifest_name_set)} / latents {sorted(ctx.latent_name_set)}"
        )
        if ctx.strict_structure:
            ctx.errors.append(message)
        else:
            logger.warning("%s", message)
        return
    indicator_name, construct_name = result
    position = (
        ctx.manifest_idx_map[indicator_name],
        ctx.latent_idx_map[construct_name],
    )
    ctx.add(
        parameter.name,
        ctx.site_for_field("lambda_free"),
        ctx.parameter_layout.lambda_free_index.get(position),
        construct_names=(construct_name,),
        indicator_names=(indicator_name,),
        error_message=(
            "LOADING parameter does not correspond to a free loading in causal_spec: "
            f"{parameter.name!r}"
        ),
    )


def _handle_static_state_sd(parameter: Any, ctx: _BindingContext) -> None:
    factor_idx = ctx.parameter_layout.static_factor_name_index.get(parameter.name)
    flat_idx = (
        ctx.parameter_layout.static_state_sd_free_index.get(factor_idx)
        if factor_idx is not None
        else None
    )
    ctx.add(
        parameter.name,
        ctx.site_for_field("static_state_sd"),
        flat_idx,
        construct_names=tuple(getattr(parameter, "construct_names", ()) or ()),
        error_message=(
            "STATIC_STATE_SD parameter does not correspond to a free compiled "
            f"baseline-factor scale: {parameter.name!r}"
        ),
    )


def _handle_observation_hyperparameter(parameter: Any, ctx: _BindingContext) -> None:
    site = ctx.sites_by_name.get(parameter.name)
    if site is not None:
        ctx.add(
            parameter.name,
            site,
            0,
            transform=PriorAuthoringTransform.SITE_WIDE,
        )
    elif ctx.strict_structure:
        ctx.errors.append(
            "Observation hyperparameter does not correspond to an active compiled "
            f"observation site: {parameter.name!r}"
        )


def _handle_dynamics_parameter(parameter: Any, ctx: _BindingContext) -> None:
    candidate = ctx.component_candidates.get(parameter.name)
    if candidate is not None:
        ctx.add_site_binding(parameter.name, candidate)
        return
    site = ctx.sites_by_name.get(parameter.name)
    if site is not None and site.assembly_group == "dynamics":
        ctx.add(
            parameter.name,
            site,
            0,
            transform=(
                PriorAuthoringTransform.SITE_WIDE
                if site_size(site.shape) > 1
                else PriorAuthoringTransform.IDENTITY
            ),
        )
        return
    ctx.errors.append(
        f"Dynamics parameter {parameter.name!r} does not correspond to a component-owned "
        "dynamics sample site."
    )


def _handle_correlation(parameter: Any, ctx: _BindingContext) -> None:
    if ctx.parameter_layout.n_diffusion_lower <= 0:
        return
    compound = parameter.name.removeprefix("cor_")
    result = split_compound_name(compound, ctx.latent_name_set, ctx.latent_name_set)
    if result is None:
        message = (
            "Could not parse CORRELATION parameter "
            f"{parameter.name!r} into (state1, state2) from known latents "
            f"{sorted(ctx.latent_name_set)}"
        )
        if ctx.strict_structure:
            ctx.errors.append(message)
        else:
            logger.warning("%s", message)
        return
    state1_name, state2_name = result
    idx1 = ctx.latent_idx_map[state1_name]
    idx2 = ctx.latent_idx_map[state2_name]
    position = (max(idx1, idx2), min(idx1, idx2))
    ctx.add(
        parameter.name,
        ctx.site_for_field("diffusion_offdiag"),
        ctx.parameter_layout.diffusion_lower_index.get(position),
        construct_names=(state1_name, state2_name),
        error_message=(
            "CORRELATION parameter does not correspond to a modeled latent pair: "
            f"{parameter.name!r}"
        ),
    )


def _handle_initial_state_correlation_block(ctx: _BindingContext) -> None:
    if ctx.parameter_layout.n_t0_correlation <= 0:
        return
    try:
        t0_bindings = resolve_initial_state_correlation_bindings(ctx.latent_names, ctx.spec_obj)
    except ValueError as exc:
        if ctx.strict_structure:
            ctx.errors.append(str(exc))
        logger.warning("%s", exc)
        t0_bindings = []
    for binding in t0_bindings:
        position = (binding.row, binding.col)
        ctx.add(
            binding.parameter_name,
            ctx.site_for_field("t0_var_offdiag"),
            ctx.parameter_layout.t0_correlation_index.get(position),
            transform=PriorAuthoringTransform.INITIAL_STATE_CORRELATION,
            construct_names=(
                ctx.latent_names[binding.col],
                ctx.latent_names[binding.row],
            ),
            error_message=(
                "INITIAL_STATE_CORRELATION parameter does not correspond to a modeled "
                f"initial-state pair: {binding.parameter_name!r}"
            ),
        )


_SIMPLE_AXIS_RULES: dict[ParameterRole, _SimpleAxisRule] = {
    ParameterRole.AR_COEFFICIENT: _SimpleAxisRule(
        prefixes=("rho_", "ar_"),
        axis_kind="latent",
        prior_field=None,
        layout_index_attr=None,
        transform=PriorAuthoringTransform.DT_PERSISTENCE_TO_CT_DECAY,
        component_fallback_pattern="rho_{construct}",
        error_template=(
            "AR parameter does not correspond to a free dynamics decay term in "
            "causal_spec: {param!r}"
        ),
    ),
    ParameterRole.RESIDUAL_SD: _SimpleAxisRule(
        prefixes=("sigma_",),
        axis_kind="latent",
        prior_field="diffusion_diag",
        layout_index_attr="diffusion_diag_index",
        error_template=(
            "RESIDUAL_SD parameter does not correspond to a free diffusion "
            "diagonal term in causal_spec: {param!r}"
        ),
    ),
    ParameterRole.INITIAL_STATE_MEAN: _SimpleAxisRule(
        prefixes=("t0_mean_",),
        axis_kind="latent",
        prior_field="t0_means",
        layout_index_attr="t0_means_free_index",
        error_template=(
            "INITIAL_STATE_MEAN parameter does not correspond to a free initial-state "
            "mean in causal_spec: {param!r}"
        ),
    ),
    ParameterRole.INITIAL_STATE_SD: _SimpleAxisRule(
        prefixes=("t0_sd_",),
        axis_kind="latent",
        prior_field="t0_var_diag",
        layout_index_attr="t0_diag_free_index",
        error_template=(
            "INITIAL_STATE_SD parameter does not correspond to a free initial-state "
            "standard deviation in causal_spec: {param!r}"
        ),
    ),
    ParameterRole.STATE_INTERCEPT: _SimpleAxisRule(
        prefixes=("cint_",),
        axis_kind="latent",
        prior_field=None,
        layout_index_attr=None,
        component_fallback_pattern="cint_{construct}",
        error_template=(
            "STATE_INTERCEPT parameter does not correspond to a free continuous-time "
            "intercept in causal_spec: {param!r}"
        ),
    ),
    ParameterRole.OBSERVATION_INTERCEPT: _SimpleAxisRule(
        prefixes=("manifest_mean_",),
        axis_kind="manifest",
        prior_field="manifest_means",
        layout_index_attr="manifest_means_free_index",
        error_template=(
            "OBSERVATION_INTERCEPT parameter does not correspond to a free manifest "
            "intercept in causal_spec: {param!r}"
        ),
    ),
    ParameterRole.MEASUREMENT_ERROR_SD: _SimpleAxisRule(
        prefixes=("obs_sd_",),
        axis_kind="manifest",
        prior_field="manifest_var_diag",
        layout_index_attr="manifest_var_free_index",
        error_template=(
            "MEASUREMENT_ERROR_SD parameter does not correspond to a free manifest "
            "noise term in causal_spec: {param!r}"
        ),
    ),
}


_SPECIAL_HANDLERS: dict[ParameterRole, Callable[[Any, _BindingContext], None]] = {
    ParameterRole.FIXED_EFFECT: _handle_fixed_effect,
    ParameterRole.LOADING: _handle_loading,
    ParameterRole.STATIC_STATE_SD: _handle_static_state_sd,
    ParameterRole.OBSERVATION_HYPERPARAMETER: _handle_observation_hyperparameter,
    ParameterRole.OBSERVATION_HYPERPARAMETER_POSITIVE: _handle_observation_hyperparameter,
    ParameterRole.DYNAMICS_PARAMETER: _handle_dynamics_parameter,
    ParameterRole.DYNAMICS_PARAMETER_POSITIVE: _handle_dynamics_parameter,
    ParameterRole.CORRELATION: _handle_correlation,
}


# Role groups, processed in this order. Within a group, parameters are visited
# in source order — preserving the original ordering of error messages when a
# group covers multiple ParameterRole members (DYNAMICS_*, OBS_HYPER_*).
_ROLE_PROCESSING_GROUPS: tuple[tuple[ParameterRole, ...], ...] = (
    (ParameterRole.AR_COEFFICIENT,),
    (ParameterRole.RESIDUAL_SD,),
    (ParameterRole.INITIAL_STATE_MEAN,),
    (ParameterRole.INITIAL_STATE_SD,),
    (ParameterRole.STATE_INTERCEPT,),
    (ParameterRole.FIXED_EFFECT,),
    (ParameterRole.LOADING,),
    (ParameterRole.OBSERVATION_INTERCEPT,),
    (ParameterRole.MEASUREMENT_ERROR_SD,),
    (ParameterRole.STATIC_STATE_SD,),
    (
        ParameterRole.OBSERVATION_HYPERPARAMETER,
        ParameterRole.OBSERVATION_HYPERPARAMETER_POSITIVE,
    ),
    (
        ParameterRole.DYNAMICS_PARAMETER,
        ParameterRole.DYNAMICS_PARAMETER_POSITIVE,
    ),
    (ParameterRole.CORRELATION,),
)


def _dispatch_parameter(parameter: Any, ctx: _BindingContext) -> None:
    rule = _SIMPLE_AXIS_RULES.get(parameter.role)
    if rule is not None:
        _apply_simple_axis_rule(rule, parameter, ctx)
        return
    handler = _SPECIAL_HANDLERS.get(parameter.role)
    if handler is not None:
        handler(parameter, ctx)


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
    manifest_idx_map = {name: idx for idx, name in enumerate(manifest_names)}
    input_names = ssm_spec.input_names or []
    input_idx_map = {name: idx for idx, name in enumerate(input_names)}
    parameter_layout = SSMParameterLayout.from_spec(ssm_spec)
    active_sites = build_site_registry(ssm_spec)
    sites_by_name = {site.name: site for site in active_sites}
    component_candidates = _component_binding_candidates(ssm_spec, active_sites)

    ctx = _BindingContext(
        spec_obj=spec_obj,
        ssm_spec=ssm_spec,
        latent_names=latent_names,
        manifest_names=manifest_names,
        latent_idx_map=latent_idx_map,
        manifest_idx_map=manifest_idx_map,
        input_idx_map=input_idx_map,
        latent_name_set=set(latent_idx_map),
        input_name_set=set(input_idx_map),
        manifest_name_set=set(manifest_idx_map),
        parameter_layout=parameter_layout,
        active_sites=active_sites,
        sites_by_name=sites_by_name,
        component_candidates=component_candidates,
        strict_structure=causal_spec is not None,
    )

    for role_group in _ROLE_PROCESSING_GROUPS:
        role_set = frozenset(role_group)
        for parameter in spec_obj.parameters:
            if parameter.role not in role_set:
                continue
            _dispatch_parameter(parameter, ctx)

    _handle_initial_state_correlation_block(ctx)

    if ctx.errors:
        raise PriorIndexingError(ctx.errors)

    return SemanticBindingRegistry(tuple(ctx.bindings[name] for name in sorted(ctx.bindings)))


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
