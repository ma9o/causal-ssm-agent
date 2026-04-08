"""Public pure-compilation surface for turning ModelSpec + priors into SSM inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from causal_ssm_agent.models.ssm_compilation_common import (
    PriorIndexMaps,
    normalize_prior_params,
    split_compound_name,
)
from causal_ssm_agent.models.ssm_prior_compilation import (
    bind_parameters,
    collect_compile_diagnostics,
    compile_priors,
)
from causal_ssm_agent.models.ssm_prior_indexing import build_prior_index_maps
from causal_ssm_agent.models.ssm_spec_translation import (
    build_masks_from_causal_spec,
    get_construct_dt_days,
    get_estimation_latent_layout,
    translate_spec,
)
from causal_ssm_agent.orchestrator.schemas_model import ModelSpec

if TYPE_CHECKING:
    from causal_ssm_agent.models.ssm.model import SSMPriors, SSMSpec
    from causal_ssm_agent.workers.schemas_prior import PriorValidationResult


def _require_explicit_causal_structure(ssm_spec: SSMSpec, *, causal_spec: dict | None) -> None:
    """Reject implicit structural degrees of freedom on causal-spec code paths."""
    if causal_spec is None:
        return

    required_structural_masks = (
        "drift_diag_mask",
        "drift_offdiag_mask",
        "cint_mask",
        "lambda_mask",
        "diffusion_chol_mask",
        "manifest_means_mask",
        "manifest_chol_diag_mask",
        "t0_means_mask",
        "t0_chol_diag_mask",
        "t0_correlation_mask",
    )
    required_matrix_templates = (
        "drift",
        "cint",
        "lambda_mat",
        "diffusion_chol",
        "manifest_means",
        "manifest_chol",
        "t0_means",
        "t0_chol",
    )

    missing_masks = [
        field_name
        for field_name in required_structural_masks
        if getattr(ssm_spec, field_name) is None
    ]
    missing_templates = [
        field_name
        for field_name in required_matrix_templates
        if getattr(ssm_spec, field_name) is None
    ]

    if not missing_masks and not missing_templates:
        return

    rendered_parts: list[str] = []
    if missing_masks:
        rendered_parts.append(f"masks: {', '.join(missing_masks)}")
    if missing_templates:
        rendered_parts.append(f"templates: {', '.join(missing_templates)}")
    raise ValueError(
        "Causal-spec compilation requires an explicit compiled structure on SSMSpec. "
        f"Missing {'; '.join(rendered_parts)}. Compile from ModelSpec + CausalSpec so "
        "translate_spec() can derive the full structural payload, or supply an already "
        "translated SSMSpec with explicit masks and matrix templates."
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


def compile_ssm_inputs(
    model_spec: ModelSpec | dict | None = None,
    priors: dict[str, dict] | None = None,
    *,
    ssm_spec: SSMSpec | None = None,
    ssm_priors: SSMPriors | None = None,
    causal_spec: dict | None = None,
) -> tuple[
    SSMSpec,
    SSMPriors,
    list[dict[str, object]],
    list[PriorValidationResult],
    dict[tuple[int, int], float],
]:
    """Resolve executable SSM inputs plus structured compiler diagnostics."""
    resolved_model_spec = (
        ModelSpec.model_validate(model_spec) if isinstance(model_spec, dict) else model_spec
    )

    edge_lag_days: dict[tuple[int, int], float] = {}
    if ssm_spec is None:
        if resolved_model_spec is None:
            raise ValueError("Cannot compile SSM inputs without model_spec or ssm_spec")
        ssm_spec, edge_lag_days = translate_spec(resolved_model_spec, causal_spec)
    _require_explicit_causal_structure(ssm_spec, causal_spec=causal_spec)

    index_maps = None
    if ssm_priors is None:
        ssm_priors, index_maps, diagnostics = compile_priors(
            priors or {},
            resolved_model_spec,
            ssm_spec,
            edge_lag_days=edge_lag_days,
            causal_spec=causal_spec,
        )
    else:
        diagnostics = collect_compile_diagnostics(
            ssm_spec,
            edge_lag_days=edge_lag_days,
            raw_priors=priors or {},
        )

    bindings = bind_parameters(
        resolved_model_spec,
        ssm_spec,
        index_maps=index_maps,
        causal_spec=causal_spec,
    )
    diagnostics = _attach_compile_binding_provenance(diagnostics, bindings)
    return ssm_spec, ssm_priors, bindings, diagnostics, edge_lag_days


__all__ = [
    "PriorIndexMaps",
    "bind_parameters",
    "build_masks_from_causal_spec",
    "build_prior_index_maps",
    "compile_priors",
    "compile_ssm_inputs",
    "get_construct_dt_days",
    "get_estimation_latent_layout",
    "normalize_prior_params",
    "split_compound_name",
    "translate_spec",
]
