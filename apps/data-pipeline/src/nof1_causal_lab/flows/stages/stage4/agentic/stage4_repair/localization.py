"""Diagnostic evidence normalization and failure localization for Stage 4 repair."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .helpers import (
    _certificate_order_key,
    _compile_diagnostics_for_supporting_codes,
    _find_block_for_parameter,
    _first_diagnostic_reason,
    _parameter_construct_names,
    _validator_scope_block_hints,
    _validator_scope_construct_names,
    _validator_scope_identity,
    _validator_scope_parameter_names,
)
from .types import (
    _DRIFT_RELATED_CODES,
    RepairReasons,
    Stage4FailureEvidence,
    Stage4FailureLocalization,
)

if TYPE_CHECKING:
    from nof1_causal_lab.flows.stages.stage4.agentic.stage4_orchestrator import Stage4Plan
    from nof1_causal_lab.flows.stages.stage4.assembly import AssemblyValidation
    from nof1_causal_lab.workers.schemas_prior import (
        PriorPathologyCertificate,
        PriorRepairScope,
        PriorValidationResult,
    )


def build_stage4_failure_evidence(
    plan: Stage4Plan,
    validation: AssemblyValidation,
) -> Stage4FailureEvidence:
    """Build the normalized evidence surface for a failed PP validation."""
    from nof1_causal_lab.models.ssm_compilation_common import GLOBAL_FAILURE_SITES

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
    manifest_names = _diagnostic_manifest_names(
        plan,
        (*failed_diagnostics, *supporting_compile_diagnostics),
    )
    return Stage4FailureEvidence(
        topology=plan.repair_topology,
        failed_diagnostics=failed_diagnostics,
        supporting_compile_diagnostics=supporting_compile_diagnostics,
        diagnostic_codes=diagnostic_codes,
        supporting_codes=supporting_codes,
        manifest_names=manifest_names,
        global_failure_sites=frozenset(GLOBAL_FAILURE_SITES),
    )


def _diagnostic_manifest_names(
    plan: Stage4Plan,
    diagnostics: tuple[PriorValidationResult, ...],
) -> tuple[str, ...]:
    """Return manifest names referenced by diagnostics in deterministic order."""
    topology = plan.repair_topology
    known_manifest_names = {
        *topology.indicator_to_decision_block_id,
        *topology.indicator_to_measurement_block_id,
    }
    manifest_names: set[str] = set()
    for result in diagnostics:
        manifest_names.update(
            name
            for name in (result.bad_manifest_names or ())
            if isinstance(name, str) and name in known_manifest_names
        )
        for token in (
            *(result.related_parameters or ()),
            *((result.parameter,) if isinstance(result.parameter, str) else ()),
        ):
            if not isinstance(token, str):
                continue
            if token in known_manifest_names:
                manifest_names.add(token)
                continue
            if token.startswith("scale_"):
                candidate = token.removeprefix("scale_")
                if candidate in known_manifest_names:
                    manifest_names.add(candidate)
    return tuple(
        sorted(
            manifest_names,
            key=lambda name: topology.get_indicator_owner_block_id(name) or name,
        )
    )


def _authored_parameter_names_from_tokens(
    plan: Stage4Plan,
    parameter_tokens: tuple[str, ...],
    *,
    model_spec: dict[str, object] | None = None,
    context: str,
) -> tuple[str, ...]:
    """Resolve diagnostic or validator tokens onto authored Stage 4 parameters."""
    from nof1_causal_lab.models.prior_predictive import resolve_scale_target_parameters

    indicator_to_construct = {
        indicator_name: construct_name
        for construct_name, indicator_names in plan.repair_topology.indicator_names_by_construct.items()
        for indicator_name in indicator_names
    }
    authored_parameter_names: list[str] = []
    unresolved_scale_tokens: list[str] = []
    for token in parameter_tokens:
        if not isinstance(token, str) or not token:
            continue
        if token.startswith("scale_"):
            resolved_tokens = tuple(
                resolve_scale_target_parameters(
                    token.removeprefix("scale_"),
                    model_spec,
                    indicator_to_construct=indicator_to_construct,
                )
            )
            if not resolved_tokens:
                unresolved_scale_tokens.append(token)
                continue
            retained_tokens = tuple(
                resolved_token
                for resolved_token in resolved_tokens
                if resolved_token and _find_block_for_parameter(plan, resolved_token) is not None
            )
            if not retained_tokens:
                unresolved_scale_tokens.append(token)
                continue
            authored_parameter_names.extend(retained_tokens)
            continue
        if _find_block_for_parameter(plan, token) is not None:
            authored_parameter_names.append(token)
    if unresolved_scale_tokens:
        unresolved = ", ".join(sorted(dict.fromkeys(unresolved_scale_tokens)))
        raise ValueError(
            f"Stage 4 could not resolve authored parameters for {context}: {unresolved}"
        )
    return tuple(sorted(dict.fromkeys(authored_parameter_names)))


def _diagnostic_parameter_names(
    plan: Stage4Plan,
    diagnostics: tuple[PriorValidationResult, ...],
    *,
    model_spec: dict[str, object] | None = None,
) -> tuple[str, ...]:
    """Return authored parameter names referenced by diagnostics in sorted order."""
    return _authored_parameter_names_from_tokens(
        plan,
        tuple(
            dict.fromkeys(
                token
                for result in diagnostics
                for token in (
                    *(result.related_parameters or ()),
                    *((result.parameter,) if result.parameter else ()),
                )
                if isinstance(token, str) and token
            )
        ),
        model_spec=model_spec,
        context="prior-predictive diagnostics",
    )


def _validator_scope_from_failure_evidence(
    evidence: Stage4FailureEvidence,
) -> PriorRepairScope | None:
    """Return the richest validator-owned repair scope from failed diagnostics."""
    candidate_scopes = [
        (1, result.repair_scope)
        for result in evidence.failed_diagnostics
        if result.repair_scope is not None
    ]
    candidate_scopes.extend(
        (0, result.repair_scope)
        for result in evidence.supporting_compile_diagnostics
        if result.repair_scope is not None
    )
    if not candidate_scopes:
        return None
    return max(
        candidate_scopes,
        key=lambda item: (
            item[0],
            len(_validator_scope_block_hints(item[1])),
            len(_validator_scope_parameter_names(item[1])),
            len(_validator_scope_construct_names(item[1])),
        ),
    )[1]


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
    validator_parameter_hints: tuple[str, ...],
) -> tuple[str, ...]:
    """Return construct hints synthesized from authored parameters and validator scope."""
    construct_names = tuple(
        dict.fromkeys(
            [
                *_parameter_construct_names(evidence.topology, direct_parameters),
                *_parameter_construct_names(evidence.topology, supporting_parameters),
                *_parameter_construct_names(evidence.topology, validator_parameter_hints),
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
    validator_parameter_hints = _authored_parameter_names_from_tokens(
        plan,
        _validator_scope_parameter_names(validator_repair_scope),
        model_spec=validation.normalized_model_spec,
        context="validator-owned repair scope",
    )
    construct_names = _construct_names_from_failure_evidence(
        evidence,
        direct_parameters=direct_parameters,
        supporting_parameters=supporting_parameters,
        validator_repair_scope=validator_repair_scope,
        validator_parameter_hints=validator_parameter_hints,
    )
    parameter_hints = tuple(
        dict.fromkeys([*direct_parameters, *supporting_parameters, *validator_parameter_hints])
    )
    failure_family = (
        evidence.diagnostic_codes,
        evidence.supporting_codes,
        tuple(sorted(construct_names)),
        tuple(sorted(parameter_hints)),
        tuple(sorted(evidence.manifest_names)),
        _validator_scope_identity(validator_repair_scope),
    )
    return Stage4FailureLocalization(
        failure_family=failure_family,
        diagnostic_codes=evidence.diagnostic_codes,
        direct_parameters=direct_parameters,
        supporting_parameters=supporting_parameters,
        manifest_names=evidence.manifest_names,
        construct_names=construct_names,
        validator_repair_scope=validator_repair_scope,
        validator_parameter_hints=validator_parameter_hints,
        pathology_certificate=_pathology_certificate_from_failure_evidence(evidence),
        has_global_failure=_has_global_failure(evidence),
        issues_text=_issues_text_from_failure_evidence(evidence),
        reasons=_reasons_from_failure_evidence(evidence),
    )
