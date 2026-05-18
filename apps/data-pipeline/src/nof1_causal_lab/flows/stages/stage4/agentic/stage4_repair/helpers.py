"""Shared helper functions for Stage 4 repair routing."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from .types import ResolvedRepairScope, Stage4FailureLocalization, Stage4ScopeCandidateSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from nof1_causal_lab.flows.stages.stage4.agentic.stage4_orchestrator import (
        Stage4FrontierBlock,
        Stage4Plan,
        Stage4RepairTopology,
    )
    from nof1_causal_lab.flows.stages.stage4.assembly import AssemblyValidation
    from nof1_causal_lab.workers.schemas_prior import (
        PriorPathologyCertificate,
        PriorRepairScope,
        PriorValidationResult,
    )


def _find_block_for_parameter(
    plan: Stage4Plan,
    parameter_name: str,
) -> Stage4FrontierBlock | None:
    """Map a validation parameter name back to the narrowest prompt block."""
    topology = plan.repair_topology
    if parameter_name.startswith("scale_"):
        indicator_name = parameter_name.removeprefix("scale_")
        measurement_block_id = topology.get_measurement_block_id(indicator_name)
        if measurement_block_id is not None:
            return plan.get_block(measurement_block_id)
        for construct_name, indicator_names in topology.indicator_names_by_construct.items():
            if indicator_name not in indicator_names:
                continue
            dynamics_block_id = topology.dynamics_block_id_by_construct.get(construct_name)
            if dynamics_block_id is not None:
                return plan.get_block(dynamics_block_id)
            break
    block_id = topology.get_parameter_block_id(parameter_name)
    return None if block_id is None else plan.get_block(block_id)


def _ordered_block_ids(plan: Stage4Plan, block_ids: set[str]) -> tuple[str, ...]:
    """Return block ids in deterministic plan order."""
    return tuple(block.id for block in plan.all_blocks if block.id in block_ids)


def _resolved_repair_scope(
    *,
    scope_kind: str,
    scope_rank: int,
    reason: str,
    failure_family: tuple[Any, ...],
    parameter_names: tuple[str, ...] = (),
    construct_names: tuple[str, ...] = (),
    prompt_block_hints: tuple[str, ...] = (),
    diagnostic_codes: tuple[str, ...] = (),
    pathology_certificate: PriorPathologyCertificate | None = None,
    scope_token: str | None = None,
) -> ResolvedRepairScope:
    """Build a deterministic resolved repair scope."""
    scope_parts = [*construct_names, *parameter_names]
    scope_suffix = (
        "|".join(dict.fromkeys(scope_parts)) if scope_parts else (scope_token or "global")
    )
    scope_key = f"{scope_kind}:{scope_suffix}"
    return ResolvedRepairScope(
        scope_kind=scope_kind,
        scope_rank=scope_rank,
        scope_key=scope_key,
        reason=reason,
        failure_family=failure_family,
        parameter_names=parameter_names,
        construct_names=construct_names,
        prompt_block_hints=prompt_block_hints,
        diagnostic_codes=diagnostic_codes,
        pathology_certificate=pathology_certificate,
    )


def _materialize_scope_candidate(
    localization: Stage4FailureLocalization,
    spec: Stage4ScopeCandidateSpec,
) -> ResolvedRepairScope:
    """Materialize one candidate scope spec with localization-owned metadata."""
    return _resolved_repair_scope(
        scope_kind=spec.scope_kind,
        scope_rank=spec.scope_rank,
        reason=spec.reason,
        failure_family=localization.failure_family,
        parameter_names=spec.parameter_names,
        construct_names=spec.construct_names,
        prompt_block_hints=spec.prompt_block_hints,
        diagnostic_codes=localization.diagnostic_codes,
        pathology_certificate=localization.pathology_certificate,
        scope_token=spec.scope_token,
    )


def _normalize_reason_text(text: str | None) -> str:
    """Collapse multiline validation text into one compact sentence."""
    return " ".join((text or "").split())


def _format_diagnostic_reason(result: PriorValidationResult) -> str | None:
    """Render one diagnostic into a user-facing repair reason."""
    issue = _normalize_reason_text(result.issue)
    suggested_adjustment = _normalize_reason_text(result.suggested_adjustment)
    if issue and suggested_adjustment:
        return f"{issue} Suggested fix: {suggested_adjustment}"
    return issue or suggested_adjustment or None


def _first_diagnostic_reason(
    diagnostics: Sequence[PriorValidationResult],
    *,
    predicate: Callable[[PriorValidationResult], bool] | None = None,
) -> str | None:
    """Return the first concrete reason matching the predicate."""
    for result in diagnostics:
        if predicate is not None and not predicate(result):
            continue
        reason = _format_diagnostic_reason(result)
        if reason:
            return reason
    return None


def _require_reason(*reasons: str | None, context: str) -> str:
    """Return the first concrete repair reason or raise."""
    for reason in reasons:
        if reason:
            return reason
    raise ValueError(f"Stage 4 repair classification requires a concrete reason for {context}")


def _repair_reason_from_feedback(feedback: str | None) -> str:
    """Extract a concrete repair reason from compile feedback."""
    compact = _normalize_reason_text(feedback)
    compact = re.sub(r"^(compile error:|validation errors:)\s*", "", compact, flags=re.IGNORECASE)
    return _require_reason(compact, context="compile failure")


def _feedback_mentions_identifier(text: str, identifier: str) -> bool:
    """Whether compile feedback mentions an identifier as a full token."""
    if not identifier:
        return False
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(identifier)}(?![A-Za-z0-9_])"
    return bool(re.search(pattern, text))


def _coerce_string_tuple(values: Any) -> tuple[str, ...]:
    """Normalize one string or string sequence into a deterministic tuple."""
    if isinstance(values, str):
        return (values,) if values else ()
    if not isinstance(values, (list, tuple, set, frozenset)):
        return ()
    return tuple(str(value) for value in values if isinstance(value, str) and value)


def _scope_attr(scope: Any, name: str) -> Any:
    """Read one attribute from a repair-scope object or mapping."""
    if scope is None:
        return None
    if isinstance(scope, dict):
        return scope.get(name)
    return getattr(scope, name, None)


def _validator_scope_parameter_names(repair_scope: PriorRepairScope | None) -> tuple[str, ...]:
    """Return parameter names from a validator-owned repair scope."""
    if repair_scope is None:
        return ()
    parameter_names: list[str] = []
    for attr_name in (
        "parameter_names",
        "target_parameter_names",
        "related_parameters",
        "parameters",
    ):
        parameter_names.extend(_coerce_string_tuple(_scope_attr(repair_scope, attr_name)))
    return tuple(dict.fromkeys(parameter_names))


def _validator_scope_block_hints(repair_scope: PriorRepairScope | None) -> tuple[str, ...]:
    """Return explicit block hints from a validator-owned repair scope."""
    if repair_scope is None:
        return ()
    block_hints: list[str] = []
    for attr_name in (
        "prompt_block_hints",
        "prompt_block_ids",
        "block_ids",
        "stage4_block_ids",
    ):
        block_hints.extend(_coerce_string_tuple(_scope_attr(repair_scope, attr_name)))
    return tuple(dict.fromkeys(block_hints))


def _validator_scope_identity(repair_scope: PriorRepairScope | None) -> tuple[Any, ...]:
    """Return a stable hashable identifier for one validator-owned repair scope."""
    if repair_scope is None:
        return ()
    scalar_tokens = tuple(
        value
        for attr_name in ("scope_key", "scope_kind", "kind", "name")
        if isinstance((value := _scope_attr(repair_scope, attr_name)), str) and value
    )
    return (
        scalar_tokens,
        tuple(sorted(_validator_scope_block_hints(repair_scope))),
        tuple(sorted(_validator_scope_parameter_names(repair_scope))),
        tuple(sorted(_validator_scope_construct_names(repair_scope))),
    )


def _all_dynamics_block_ids(plan: Stage4Plan) -> tuple[str, ...]:
    """Return all dynamics-prior block ids in plan order."""
    return tuple(block.id for block in plan.prior_blocks if block.kind == "dynamics_prior")


def _compile_diagnostics_for_supporting_codes(
    validation: AssemblyValidation,
    *,
    supporting_codes: set[str],
) -> list[PriorValidationResult]:
    """Return compile diagnostics referenced by a failing PP diagnostic."""
    if not supporting_codes:
        return []
    return [
        diagnostic
        for diagnostic in validation.compile_diagnostics
        if diagnostic.code in supporting_codes
    ]


def _certificate_order_key(certificate: PriorPathologyCertificate) -> tuple[float, float]:
    """Return a comparable severity key where lower means improvement."""
    secondary = (
        certificate.secondary_score if certificate.secondary_score is not None else float("inf")
    )
    return (certificate.primary_score, secondary)


def _certificate_improved(
    current: PriorPathologyCertificate | None,
    best_so_far: PriorPathologyCertificate | None,
) -> bool:
    """Whether the latest pathology certificate strictly improved at this scope."""
    if current is None or best_so_far is None:
        return False
    if current.kind != best_so_far.kind:
        return False
    return _certificate_order_key(current) < _certificate_order_key(best_so_far)


def _parameter_construct_names(
    topology: Stage4RepairTopology,
    parameter_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Return ordered construct hints implied by semantic parameter names."""
    constructs: list[str] = []
    for parameter_name in parameter_names:
        constructs.extend(topology.parameter_construct_names.get(parameter_name, ()))
    return tuple(dict.fromkeys(constructs))


def _validator_scope_construct_names(repair_scope: PriorRepairScope | None) -> tuple[str, ...]:
    """Return construct names from a validator-owned repair scope."""
    if repair_scope is None:
        return ()
    construct_names: list[str] = []
    for attr_name in (
        "construct_names",
        "target_construct_names",
        "state_names",
        "scc_construct_names",
    ):
        construct_names.extend(_coerce_string_tuple(_scope_attr(repair_scope, attr_name)))
    return tuple(dict.fromkeys(construct_names))
