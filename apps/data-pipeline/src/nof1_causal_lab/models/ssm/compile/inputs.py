"""Public pure-compilation entry points for executable SSM inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.models.ssm.compile.common import (
    normalize_prior_params,
)
from nof1_causal_lab.models.ssm.compile.prior_compilation import (
    bind_parameters,
    collect_compile_diagnostics,
    compile_priors,
)
from nof1_causal_lab.models.ssm.compile.prior_indexing import (
    SemanticBindingRegistry,
    build_semantic_prior_bindings,
    check_backward_closure,
    empty_prior_bindings,
)
from nof1_causal_lab.models.ssm.compile.spec_translation import (
    build_structural_support_from_causal_spec,
    get_construct_dt_days,
    get_estimation_latent_layout,
    translate_spec,
)
from nof1_causal_lab.models.ssm.parameter_names import split_compound_name

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.model import SSMSpec
    from nof1_causal_lab.models.ssm.priors import PriorRegistry
    from nof1_causal_lab.workers.schemas_prior import PriorValidationResult


def _require_explicit_causal_structure(ssm_spec: SSMSpec, *, causal_spec: dict | None) -> None:
    """Reject implicit structural degrees of freedom on causal-spec code paths."""
    if causal_spec is None:
        return

    required_block_fields = (
        "drift_spec",
        "diffusion_block.diffusion_chol_mask",
        "diffusion_block.diffusion_chol_template",
        "lambda_block.mask",
        "lambda_block.template",
        "manifest_means_block.mask",
        "manifest_means_block.template",
        "manifest_chol_block.diag_mask",
        "manifest_chol_block.template",
        "t0_means_block.mask",
        "t0_means_block.template",
        "t0_chol_block.diag_mask",
        "t0_chol_block.correlation_mask",
        "t0_chol_block.template",
        "input_effect_block.mask",
        "input_effect_block.template",
        "static_state_sd_block.mask",
        "static_state_sd_block.template",
        "static_factor_loadings",
    )

    def _resolve_field(path: str):
        value = ssm_spec
        for part in path.split("."):
            value = getattr(value, part)
        return value

    missing_fields = [
        field_name for field_name in required_block_fields if _resolve_field(field_name) is None
    ]

    if not missing_fields:
        return

    raise ValueError(
        "Causal-spec compilation requires an explicit compiled structure on SSMSpec. "
        f"Missing block fields: {', '.join(missing_fields)}. Compile from ModelSpec + CausalSpec so "
        "translate_spec() can derive the full structural payload, or supply an already "
        "translated SSMSpec with explicit block masks and templates."
    )


def _attach_compile_binding_provenance(
    diagnostics: list[PriorValidationResult],
    bindings: list[dict[str, object]],
) -> list[PriorValidationResult]:
    """Attach direct-writer parameter provenance to compile diagnostics when possible."""
    binding_index: dict[tuple[str, int], list[str]] = {}
    for binding in bindings:
        site_name = binding.get("site_name")
        flat_index = binding.get("flat_index")
        parameter = binding.get("parameter")
        if not isinstance(site_name, str) or not isinstance(flat_index, int):
            continue
        if not isinstance(parameter, str) or not parameter:
            continue
        binding_index.setdefault((site_name, flat_index), []).append(parameter)

    for diagnostic in diagnostics:
        if diagnostic.compiled_site_name is None or diagnostic.compiled_flat_index is None:
            continue
        related_parameters = binding_index.get(
            (diagnostic.compiled_site_name, diagnostic.compiled_flat_index)
        )
        if related_parameters:
            diagnostic.related_parameters = related_parameters

    return diagnostics


def compile_ssm_inputs_from_model_spec(
    model_spec: ModelSpec | dict,
    priors: dict[str, dict] | None = None,
    *,
    causal_spec: dict | None = None,
) -> tuple[
    SSMSpec,
    PriorRegistry,
    list[dict[str, object]],
    list[PriorValidationResult],
    dict[tuple[int, int], float],
]:
    """Compile executable SSM inputs from a validated semantic model spec surface."""
    resolved_model_spec = (
        ModelSpec.model_validate(model_spec) if isinstance(model_spec, dict) else model_spec
    )
    if resolved_model_spec is None:
        raise ValueError("compile_ssm_inputs_from_model_spec() requires model_spec")

    ssm_spec, edge_lag_days = translate_spec(resolved_model_spec, causal_spec)
    _require_explicit_causal_structure(ssm_spec, causal_spec=causal_spec)

    prior_registry, index_maps, diagnostics = compile_priors(
        priors or {},
        resolved_model_spec,
        ssm_spec,
        edge_lag_days=edge_lag_days,
        causal_spec=causal_spec,
    )

    if causal_spec is not None:
        backward_gaps = check_backward_closure(ssm_spec, index_maps)
        if backward_gaps:
            from nof1_causal_lab.models.ssm.compile.prior_indexing import PriorIndexingError

            raise PriorIndexingError(backward_gaps)

    bindings = bind_parameters(index_maps, ssm_spec)
    diagnostics = _attach_compile_binding_provenance(diagnostics, bindings)
    return ssm_spec, prior_registry, bindings, diagnostics, edge_lag_days


def compile_ssm_inputs_from_spec(
    ssm_spec: SSMSpec,
    *,
    priors: dict[str, dict] | None = None,
    prior_registry: PriorRegistry | None = None,
    model_spec: ModelSpec | dict | None = None,
    causal_spec: dict | None = None,
    edge_lag_days: dict[tuple[int, int], float] | None = None,
) -> tuple[
    SSMSpec,
    PriorRegistry,
    list[dict[str, object]],
    list[PriorValidationResult],
    dict[tuple[int, int], float],
]:
    """Finalize executable SSM inputs from an explicit translated SSMSpec surface."""
    from nof1_causal_lab.models.ssm.parameterization import build_site_registry
    from nof1_causal_lab.models.ssm.priors import default_prior_registry_for_sites

    resolved_model_spec = (
        ModelSpec.model_validate(model_spec) if isinstance(model_spec, dict) else model_spec
    )

    resolved_edge_lag_days = {} if edge_lag_days is None else dict(edge_lag_days)
    _require_explicit_causal_structure(ssm_spec, causal_spec=causal_spec)
    raw_priors = priors or {}

    if resolved_model_spec is None:
        if raw_priors:
            raise ValueError(
                "compile_ssm_inputs_from_spec() requires model_spec to compile semantic prior "
                "proposals from a direct SSMSpec."
            )
        resolved_prior_registry = prior_registry or default_prior_registry_for_sites(
            build_site_registry(ssm_spec)
        )
        index_maps = empty_prior_bindings()
        diagnostics = collect_compile_diagnostics(
            ssm_spec,
            edge_lag_days=resolved_edge_lag_days,
            raw_priors=raw_priors,
            prior_registry=resolved_prior_registry,
        )
        bindings: list[dict[str, object]] = []
        diagnostics = _attach_compile_binding_provenance(diagnostics, bindings)
        return ssm_spec, resolved_prior_registry, bindings, diagnostics, resolved_edge_lag_days

    if prior_registry is None:
        prior_registry, index_maps, diagnostics = compile_priors(
            raw_priors,
            resolved_model_spec,
            ssm_spec,
            edge_lag_days=resolved_edge_lag_days,
            causal_spec=causal_spec,
        )
    else:
        index_maps = build_semantic_prior_bindings(
            ssm_spec,
            resolved_model_spec,
            causal_spec=causal_spec,
        )
        diagnostics = collect_compile_diagnostics(
            ssm_spec,
            edge_lag_days=resolved_edge_lag_days,
            raw_priors=raw_priors,
            prior_registry=prior_registry,
        )

    if causal_spec is not None:
        backward_gaps = check_backward_closure(ssm_spec, index_maps)
        if backward_gaps:
            from nof1_causal_lab.models.ssm.compile.prior_indexing import PriorIndexingError

            raise PriorIndexingError(backward_gaps)

    bindings = bind_parameters(index_maps, ssm_spec)
    diagnostics = _attach_compile_binding_provenance(diagnostics, bindings)
    return ssm_spec, prior_registry, bindings, diagnostics, resolved_edge_lag_days


__all__ = [
    "SemanticBindingRegistry",
    "bind_parameters",
    "build_structural_support_from_causal_spec",
    "build_semantic_prior_bindings",
    "compile_priors",
    "compile_ssm_inputs_from_model_spec",
    "compile_ssm_inputs_from_spec",
    "get_construct_dt_days",
    "get_estimation_latent_layout",
    "normalize_prior_params",
    "split_compound_name",
    "translate_spec",
]
