"""Compile-error routing for Stage 4 repair."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .helpers import (
    _feedback_mentions_identifier,
    _find_block_for_parameter,
    _ordered_block_ids,
    _repair_reason_from_feedback,
    _resolved_repair_scope,
    _validator_scope_block_hints,
    _validator_scope_construct_names,
    _validator_scope_parameter_names,
)
from .planning import build_repair_plan

if TYPE_CHECKING:
    from nof1_causal_lab.flows.stages.stage4.agentic.stage4_orchestrator import (
        Stage4FrontierBlock,
        Stage4Plan,
    )
    from nof1_causal_lab.flows.stages.stage4.assembly import AssemblyValidation
    from nof1_causal_lab.workers.schemas_prior import PriorValidationResult

    from .types import ResolvedRepairPlan


def classify_compile_failure_route(
    plan: Stage4Plan,
    active_block: Stage4FrontierBlock,
    feedback: str | None,
    validation: AssemblyValidation | None = None,
) -> ResolvedRepairPlan:
    """Route compile failures back to the smallest matching block."""
    text = feedback or ""
    failure_family = ("compile_failure", active_block.id)
    matching_block_ids = _compile_failure_matching_block_ids(plan, text, validation=validation)
    if not matching_block_ids:
        diagnostics = (
            ()
            if validation is None
            else tuple(validation.compile_diagnostics or validation.diagnostics)
        )
        diagnostic_codes = tuple(
            diagnostic.code
            for diagnostic in diagnostics
            if isinstance(getattr(diagnostic, "code", None), str) and diagnostic.code
        )
        detail_suffix = (
            f" Structured diagnostics present: {', '.join(diagnostic_codes)}"
            if diagnostic_codes
            else ""
        )
        raise ValueError(
            "Stage 4 compile-failure routing requires concrete parameter, manifest, or "
            f"validator-owned attribution for {active_block.id!r}." + detail_suffix
        )
    scope_token = "+".join(matching_block_ids)
    return build_repair_plan(
        plan,
        _resolved_repair_scope(
            scope_kind="compile_local",
            scope_rank=0,
            reason=_repair_reason_from_feedback(feedback),
            failure_family=failure_family,
            prompt_block_hints=matching_block_ids,
            scope_token=scope_token,
        ),
        prompt_block_ids=matching_block_ids,
    )


def _compile_failure_matching_block_ids(
    plan: Stage4Plan,
    text: str,
    *,
    validation: AssemblyValidation | None = None,
) -> tuple[str, ...]:
    """Return owner block ids for structured or exact identifiers on compile failure."""
    matched_block_ids: set[str] = set()

    diagnostics = (
        ()
        if validation is None
        else tuple(validation.compile_diagnostics or validation.diagnostics)
    )
    if diagnostics:
        matched_block_ids.update(
            _compile_failure_matching_block_ids_from_diagnostics(
                plan,
                diagnostics,
            )
        )
    if matched_block_ids:
        return _ordered_block_ids(plan, matched_block_ids)

    if not text:
        return ()

    topology = plan.repair_topology

    for parameter_name in topology.parameter_to_block_id:
        if not _feedback_mentions_identifier(text, parameter_name):
            continue
        block = _find_block_for_parameter(plan, parameter_name)
        if block is not None:
            matched_block_ids.add(block.id)

    indicator_names = {
        *topology.indicator_to_decision_block_id,
        *topology.indicator_to_measurement_block_id,
    }
    for indicator_name in indicator_names:
        if not _feedback_mentions_identifier(text, indicator_name):
            continue
        block_id = topology.get_indicator_owner_block_id(indicator_name)
        if block_id is not None:
            matched_block_ids.add(block_id)

    return _ordered_block_ids(plan, matched_block_ids)


def _compile_failure_matching_block_ids_from_diagnostics(
    plan: Stage4Plan,
    diagnostics: tuple[PriorValidationResult, ...],
) -> tuple[str, ...]:
    """Return block ids implicated by structured compiler diagnostics."""
    topology = plan.repair_topology
    matched_block_ids: set[str] = set()

    for diagnostic in diagnostics:
        diagnostic_matched = False
        for parameter_name in tuple(diagnostic.related_parameters or ()) + tuple(
            [diagnostic.parameter] if diagnostic.parameter else []
        ):
            if not isinstance(parameter_name, str):
                continue
            block = _find_block_for_parameter(plan, parameter_name)
            if block is not None:
                matched_block_ids.add(block.id)
                diagnostic_matched = True

        for manifest_name in tuple(diagnostic.bad_manifest_names or ()):
            if not isinstance(manifest_name, str):
                continue
            block_id = topology.get_indicator_owner_block_id(manifest_name)
            if block_id is not None:
                matched_block_ids.add(block_id)
                diagnostic_matched = True

        repair_scope = diagnostic.repair_scope
        if repair_scope is None:
            continue

        for block_id in _validator_scope_block_hints(repair_scope):
            if plan.get_block(block_id) is not None:
                matched_block_ids.add(block_id)
                diagnostic_matched = True

        for parameter_name in _validator_scope_parameter_names(repair_scope):
            block = _find_block_for_parameter(plan, parameter_name)
            if block is not None:
                matched_block_ids.add(block.id)
                diagnostic_matched = True

        if diagnostic_matched:
            continue

        repair_construct_names = _validator_scope_construct_names(repair_scope)
        closed_scc_ids = {
            topology.get_scc_id(construct_name)
            for construct_name in repair_construct_names
            if topology.get_scc_id(construct_name) is not None
        }
        if closed_scc_ids:
            for scc_id in closed_scc_ids:
                for construct_name in topology.scc_construct_names_by_id.get(scc_id, ()):
                    dynamics_block_id = topology.dynamics_block_id_by_construct.get(construct_name)
                    if dynamics_block_id is not None:
                        matched_block_ids.add(dynamics_block_id)
                matched_block_ids.update(
                    topology.internal_effect_block_ids_by_scc_id.get(scc_id, ())
                )
            continue

        for construct_name in repair_construct_names:
            dynamics_block_id = topology.dynamics_block_id_by_construct.get(construct_name)
            if dynamics_block_id is not None:
                matched_block_ids.add(dynamics_block_id)

    return _ordered_block_ids(plan, matched_block_ids)
