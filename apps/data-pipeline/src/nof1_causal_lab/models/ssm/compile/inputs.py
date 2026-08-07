"""Public pure-compilation entry points for executable SSM inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.models.ssm.compile.common import (
    normalize_prior_params,
)
from nof1_causal_lab.models.ssm.compile.prior_compilation import (
    bind_parameters,
    compile_priors,
)
from nof1_causal_lab.models.ssm.compile.prior_indexing import (
    SemanticBindingRegistry,
    build_semantic_prior_bindings,
    check_backward_closure,
)
from nof1_causal_lab.models.ssm.compile.spec_translation import (
    build_structural_support_from_plan,
    get_construct_dt_days,
    get_structural_latent_layout,
    translate_spec,
)
from nof1_causal_lab.models.ssm.parameter_names import split_compound_name
from nof1_causal_lab.utils.structural_plan import get_manifest_indicators

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.prior import PriorPlan, PriorValidationResult
    from nof1_causal_lab.artifacts.statistical_model_spec import StatisticalModelSpec
    from nof1_causal_lab.artifacts.structural_plan import StructuralPlan
    from nof1_causal_lab.models.ssm.compile.contracts import CompiledParameterBinding
    from nof1_causal_lab.models.ssm.model import SSMSpec
    from nof1_causal_lab.models.ssm.priors import PriorRegistry


def _require_explicit_causal_structure(
    ssm_spec: SSMSpec,
    *,
    structural_plan: StructuralPlan | None,
) -> None:
    """Reject implicit structural degrees of freedom on causal-design code paths."""
    if structural_plan is None:
        return

    required_block_fields = (
        "dynamics_spec",
        "diffusion_block.diffusion_chol_support",
        "diffusion_block.diffusion_chol_template",
        "lambda_block.free_support",
        "lambda_block.template",
        "manifest_means_block.free_support",
        "manifest_means_block.template",
        "manifest_chol_block.diag_support",
        "manifest_chol_block.template",
        "t0_means_block.free_support",
        "t0_means_block.template",
        "t0_chol_block.diag_support",
        "t0_chol_block.correlation_support",
        "t0_chol_block.template",
        "input_effect_block.free_support",
        "input_effect_block.template",
        "static_state_sd_block.free_support",
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
        "StructuralPlan compilation requires an explicit compiled structure on SSMSpec. "
        f"Missing block fields: {', '.join(missing_fields)}. Compile from "
        "StatisticalModelSpec + StructuralPlan so "
        "translate_spec() can derive the full structural payload, or supply an already "
        "translated SSMSpec with explicit block supports and templates."
    )


def _attach_compile_binding_provenance(
    diagnostics: list[PriorValidationResult],
    bindings: list[CompiledParameterBinding],
) -> list[PriorValidationResult]:
    """Attach direct-writer parameter provenance to compile diagnostics when possible."""
    binding_index: dict[tuple[str, int], list[str]] = {}
    for binding in bindings:
        binding_index.setdefault((binding.site_name, binding.flat_index), []).append(
            binding.parameter
        )

    for diagnostic in diagnostics:
        if diagnostic.compiled_site_name is None or diagnostic.compiled_flat_index is None:
            continue
        related_parameters = binding_index.get(
            (diagnostic.compiled_site_name, diagnostic.compiled_flat_index)
        )
        if related_parameters:
            diagnostic.related_parameters = related_parameters

    return diagnostics


def _order_likelihoods_by_structural_plan(
    statistical_model_spec: StatisticalModelSpec,
    structural_plan: StructuralPlan,
) -> StatisticalModelSpec:
    """Canonicalize manifest array order to the StructuralPlan contract."""
    plan_order = [str(indicator["name"]) for indicator in get_manifest_indicators(structural_plan)]
    likelihood_by_variable = {
        likelihood.variable: likelihood for likelihood in statistical_model_spec.likelihoods
    }
    authored_names = [likelihood.variable for likelihood in statistical_model_spec.likelihoods]
    if len(likelihood_by_variable) != len(authored_names):
        raise ValueError("StatisticalModelSpec contains duplicate likelihood variables")
    if set(authored_names) != set(plan_order):
        raise ValueError(
            "StatisticalModelSpec likelihoods do not exactly cover StructuralPlan manifests: "
            f"missing={sorted(set(plan_order) - set(authored_names))}, "
            f"unplanned={sorted(set(authored_names) - set(plan_order))}."
        )
    return statistical_model_spec.model_copy(
        update={"likelihoods": [likelihood_by_variable[variable] for variable in plan_order]}
    )


def compile_ssm_inputs_from_statistical_model_spec(
    statistical_model_spec: StatisticalModelSpec,
    prior_plan: PriorPlan,
    *,
    structural_plan: StructuralPlan | None = None,
) -> tuple[
    SSMSpec,
    PriorRegistry,
    list[CompiledParameterBinding],
    list[PriorValidationResult],
    dict[tuple[int, int], float],
]:
    """Compile executable SSM inputs from a validated semantic statistical model spec surface."""
    ordered_statistical_model_spec = statistical_model_spec
    if structural_plan is not None:
        ordered_statistical_model_spec = _order_likelihoods_by_structural_plan(
            ordered_statistical_model_spec,
            structural_plan,
        )
    ssm_spec, edge_lag_days = translate_spec(
        ordered_statistical_model_spec,
        structural_plan,
    )
    _require_explicit_causal_structure(ssm_spec, structural_plan=structural_plan)

    expected_parameters = {
        parameter.name for parameter in ordered_statistical_model_spec.parameters
    }
    planned_parameters = set(prior_plan.priors)
    if planned_parameters != expected_parameters:
        raise ValueError(
            "PriorPlan must exactly cover StatisticalModelSpec parameters: "
            f"missing={sorted(expected_parameters - planned_parameters)}, "
            f"unknown={sorted(planned_parameters - expected_parameters)}."
        )

    prior_registry, index_maps, diagnostics = compile_priors(
        prior_plan.compiler_payloads(),
        ordered_statistical_model_spec,
        ssm_spec,
        edge_lag_days=edge_lag_days,
        structural_plan=structural_plan,
    )

    if structural_plan is not None:
        backward_gaps = check_backward_closure(ssm_spec, index_maps)
        if backward_gaps:
            from nof1_causal_lab.models.ssm.compile.prior_indexing import PriorIndexingError

            raise PriorIndexingError(backward_gaps)

    bindings = bind_parameters(index_maps)
    diagnostics = _attach_compile_binding_provenance(diagnostics, bindings)
    return ssm_spec, prior_registry, bindings, diagnostics, edge_lag_days


__all__ = [
    "SemanticBindingRegistry",
    "bind_parameters",
    "build_structural_support_from_plan",
    "build_semantic_prior_bindings",
    "compile_priors",
    "compile_ssm_inputs_from_statistical_model_spec",
    "get_construct_dt_days",
    "get_structural_latent_layout",
    "normalize_prior_params",
    "split_compound_name",
    "translate_spec",
]
