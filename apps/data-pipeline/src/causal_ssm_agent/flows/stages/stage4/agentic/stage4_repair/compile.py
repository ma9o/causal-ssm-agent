"""Compile-error routing for Stage 4 repair."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .helpers import (
    _choose_compile_local_block_id,
    _feedback_mentions_identifier,
    _find_block_for_parameter,
    _ordered_block_ids,
    _repair_reason_from_feedback,
    _resolved_repair_scope,
)
from .planning import build_repair_plan
from .types import _GLOBAL_REVIEW_SCOPE_RANK, ResolvedRepairPlan

if TYPE_CHECKING:
    from causal_ssm_agent.flows.stages.stage4.agentic.stage4_orchestrator import (
        Stage4FrontierBlock,
        Stage4Plan,
    )


def classify_compile_failure_route(
    plan: Stage4Plan,
    active_block: Stage4FrontierBlock,
    feedback: str | None,
) -> ResolvedRepairPlan:
    """Route compile failures back to the smallest matching block."""
    text = feedback or ""
    failure_family = ("compile_failure", active_block.id)
    matching_block_ids = _compile_failure_matching_block_ids(plan, text)
    if matching_block_ids:
        chosen_block_id = _choose_compile_local_block_id(
            active_block=active_block,
            block_ids=matching_block_ids,
        )
        return build_repair_plan(
            plan,
            _resolved_repair_scope(
                scope_kind="compile_local",
                scope_rank=0,
                reason=_repair_reason_from_feedback(feedback),
                failure_family=failure_family,
                scope_token=chosen_block_id,
            ),
            prompt_block_ids=(chosen_block_id,),
            requires_barrier_validation=False,
        )
    if active_block.kind == "global_prior_review":
        return build_repair_plan(
            plan,
            _resolved_repair_scope(
                scope_kind="global_prior_review",
                scope_rank=_GLOBAL_REVIEW_SCOPE_RANK,
                reason=_repair_reason_from_feedback(feedback),
                failure_family=failure_family,
                scope_token="prior_system",
            ),
            requires_barrier_validation=False,
        )
    return build_repair_plan(
        plan,
        _resolved_repair_scope(
            scope_kind="compile_active_block",
            scope_rank=0,
            reason=_repair_reason_from_feedback(feedback),
            failure_family=failure_family,
            scope_token=active_block.id,
        ),
        prompt_block_ids=(active_block.id,),
        requires_barrier_validation=False,
    )


def _compile_failure_matching_block_ids(
    plan: Stage4Plan,
    text: str,
) -> tuple[str, ...]:
    """Return owner block ids for exact identifiers mentioned in compile feedback."""
    if not text:
        return ()

    topology = plan.repair_topology
    matched_block_ids: set[str] = set()

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
