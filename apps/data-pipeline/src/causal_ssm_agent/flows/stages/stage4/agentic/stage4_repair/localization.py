"""Diagnostic evidence normalization and failure localization for Stage 4 repair."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .helpers import (
    _certificate_order_key,
    _compile_diagnostics_for_supporting_codes,
    _find_block_for_parameter,
    _first_diagnostic_reason,
    _parameter_construct_names,
    _validator_scope_construct_names,
)
from .types import (
    _DRIFT_RELATED_CODES,
    RepairReasons,
    Stage4FailureEvidence,
    Stage4FailureLocalization,
)

if TYPE_CHECKING:
    from causal_ssm_agent.flows.stages.stage4.agentic.stage4_orchestrator import Stage4Plan
    from causal_ssm_agent.flows.stages.stage4.assembly import AssemblyValidation
    from causal_ssm_agent.workers.schemas_prior import (
        PriorPathologyCertificate,
        PriorRepairScope,
        PriorValidationResult,
    )


def build_stage4_failure_evidence(
    plan: Stage4Plan,
    validation: AssemblyValidation,
) -> Stage4FailureEvidence:
    """Build the normalized evidence surface for a failed PP validation."""
    from causal_ssm_agent.models.ssm_compilation_common import GLOBAL_FAILURE_SITES

    failed_diagnostics = tuple(
        result for result in validation.prior_predictive_diagnostics if not result.is_valid
    )
    diagnostic_codes = tuple(
        sorted(dict.fromkeys(result.code for result in failed_diagnostics if result.code))
    )
    supporting_codes = tuple(
        sorted({code for result in failed_diagnostics for code in result.supporting_codes if code})
    )
    supporting_compile_diagnostics = tuple(
        _compile_diagnostics_for_supporting_codes(
            validation,
            supporting_codes=set(supporting_codes),
        )
    )
    return Stage4FailureEvidence(
        topology=plan.repair_topology,
        failed_diagnostics=failed_diagnostics,
        supporting_compile_diagnostics=supporting_compile_diagnostics,
        diagnostic_codes=diagnostic_codes,
        supporting_codes=supporting_codes,
        global_failure_sites=frozenset(GLOBAL_FAILURE_SITES),
    )


def _diagnostic_parameter_names(
    plan: Stage4Plan,
    diagnostics: tuple[PriorValidationResult, ...],
    *,
    model_spec: dict[str, object] | None = None,
) -> tuple[str, ...]:
    """Return authored parameter names referenced by diagnostics in sorted order."""
    from causal_ssm_agent.models.prior_predictive import resolve_scale_target_parameters

    indicator_to_construct = {
        indicator_name: construct_name
        for construct_name, indicator_names in plan.repair_topology.indicator_names_by_construct.items()
        for indicator_name in indicator_names
    }
    return tuple(
        sorted(
            dict.fromkeys(
                parameter_name
                for result in diagnostics
                for parameter_name in (

                        tuple(result.related_parameters or ())
                        or (
                            tuple(
                                resolve_scale_target_parameters(
                                    result.parameter.removeprefix("scale_"),
                                    model_spec,
                                    indicator_to_construct=indicator_to_construct,
                                )
                            )
                            if result.parameter.startswith("scale_")
                            else ()
                        )
                        or ((result.parameter,) if result.parameter else ())

                )
                if parameter_name and _find_block_for_parameter(plan, parameter_name) is not None
            )
        )
    )


def _validator_scope_from_failure_evidence(
    evidence: Stage4FailureEvidence,
) -> PriorRepairScope | None:
    """Return the first validator-owned repair scope from failed diagnostics."""
    return next(
        (
            result.repair_scope
            for result in evidence.failed_diagnostics
            if result.repair_scope is not None
        ),
        None,
    )


def _pathology_certificate_from_failure_evidence(
    evidence: Stage4FailureEvidence,
) -> PriorPathologyCertificate | None:
    """Return the dominant pathology certificate for same-scope retry gating."""
    failed_certificates = [
        result.pathology_certificate
        for result in evidence.failed_diagnostics
        if result.pathology_certificate is not None
    ]
    supporting_certificates = [
        diagnostic.pathology_certificate
        for diagnostic in evidence.supporting_compile_diagnostics
        if diagnostic.pathology_certificate is not None
    ]
    if failed_certificates:
        return max(failed_certificates, key=_certificate_order_key)
    if supporting_certificates:
        # Same-scope retry gating should prefer the certificate for the actual
        # failed PP pathology, not the supporting compile warning. Compile-side
        # certificates are only a fallback when PP has no comparable metric.
        return max(supporting_certificates, key=_certificate_order_key)
    return None


def _construct_names_from_failure_evidence(
    evidence: Stage4FailureEvidence,
    *,
    direct_parameters: tuple[str, ...],
    supporting_parameters: tuple[str, ...],
    validator_repair_scope: PriorRepairScope | None,
) -> tuple[str, ...]:
    """Return construct hints synthesized from authored parameters and validator scope."""
    construct_names = tuple(
        dict.fromkeys(
            [
                *_parameter_construct_names(evidence.topology, direct_parameters),
                *_parameter_construct_names(evidence.topology, supporting_parameters),
                *_validator_scope_construct_names(validator_repair_scope),
            ]
        )
    )
    if construct_names:
        return construct_names
    return tuple(
        dict.fromkeys(
            name
            for result in evidence.failed_diagnostics
            for parameter_name in (result.related_parameters or [])
            for name in evidence.topology.parameter_construct_names.get(parameter_name, ())
        )
    )


def _issues_text_from_failure_evidence(evidence: Stage4FailureEvidence) -> str:
    """Return the normalized lower-cased issue text across failed diagnostics."""
    return " ".join(result.issue or "" for result in evidence.failed_diagnostics).lower()


def _has_global_failure(evidence: Stage4FailureEvidence) -> bool:
    """Whether any failed diagnostic is a whole-system global failure."""
    return any(
        result.parameter in evidence.global_failure_sites for result in evidence.failed_diagnostics
    )


def _is_support_issue(result: PriorValidationResult) -> bool:
    """Whether the diagnostic describes a likelihood support violation."""
    issue = (result.issue or "").lower()
    return "support check" in issue or "outside support" in issue


def _reasons_from_failure_evidence(evidence: Stage4FailureEvidence) -> RepairReasons:
    """Extract user-facing repair reasons from the normalized evidence surface."""
    return RepairReasons(
        default=_first_diagnostic_reason(evidence.all_reason_diagnostics),
        support=_first_diagnostic_reason(
            evidence.all_reason_diagnostics,
            predicate=_is_support_issue,
        ),
        drift=_first_diagnostic_reason(
            evidence.all_reason_diagnostics,
            predicate=lambda result: (
                result.repair_scope is not None
                or result.code in _DRIFT_RELATED_CODES
                or any(
                    parameter_name.startswith("beta_")
                    for parameter_name in (
                        result.related_parameters
                        or ([result.parameter] if result.parameter else [])
                    )
                )
            ),
        ),
        validator=_first_diagnostic_reason(
            evidence.failed_diagnostics,
            predicate=lambda result: result.repair_scope is not None,
        ),
        global_=_first_diagnostic_reason(
            evidence.failed_diagnostics,
            predicate=lambda result: result.parameter in evidence.global_failure_sites,
        ),
    )


def _localize_prior_failure(
    plan: Stage4Plan,
    validation: AssemblyValidation,
) -> Stage4FailureLocalization:
    """Localize a failed PP validation into deterministic structural evidence."""
    evidence = build_stage4_failure_evidence(plan, validation)
    direct_parameters = _diagnostic_parameter_names(
        plan,
        evidence.failed_diagnostics,
        model_spec=validation.normalized_model_spec,
    )
    supporting_parameters = _diagnostic_parameter_names(
        plan,
        evidence.supporting_compile_diagnostics,
        model_spec=validation.normalized_model_spec,
    )
    validator_repair_scope = _validator_scope_from_failure_evidence(evidence)
    construct_names = _construct_names_from_failure_evidence(
        evidence,
        direct_parameters=direct_parameters,
        supporting_parameters=supporting_parameters,
        validator_repair_scope=validator_repair_scope,
    )
    failure_family = (
        evidence.diagnostic_codes,
        evidence.supporting_codes,
        tuple(sorted(construct_names)),
    )
    return Stage4FailureLocalization(
        failure_family=failure_family,
        diagnostic_codes=evidence.diagnostic_codes,
        direct_parameters=direct_parameters,
        supporting_parameters=supporting_parameters,
        construct_names=construct_names,
        validator_repair_scope=validator_repair_scope,
        pathology_certificate=_pathology_certificate_from_failure_evidence(evidence),
        has_global_failure=_has_global_failure(evidence),
        issues_text=_issues_text_from_failure_evidence(evidence),
        reasons=_reasons_from_failure_evidence(evidence),
    )
