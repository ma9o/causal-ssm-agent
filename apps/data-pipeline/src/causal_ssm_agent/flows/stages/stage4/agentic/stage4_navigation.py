"""Stage 4 cursor, phase, and repair-campaign navigation helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .stage4_block_specs import get_stage4_block_phase
from .stage4_state import (
    Stage4BlockCursor,
    Stage4DoneCursor,
    Stage4ModelSpecLockPendingCursor,
    Stage4RepairBarrierCursor,
    Stage4RepairCampaignState,
    Stage4Runtime,
)

if TYPE_CHECKING:
    from causal_ssm_agent.workers.schemas_prior import PriorPathologyCertificate

    from .stage4_orchestrator import Stage4FrontierBlock, Stage4Plan
    from .stage4_repair import ResolvedRepairPlan


def block_is_accepted(runtime: Stage4Runtime, block_id: str) -> bool:
    """Whether a block is currently accepted in runtime state."""
    return runtime.block_status.get(block_id) == "accepted"


def _phase_for_block_kind(kind: str) -> str:
    """Project one authored block kind onto the public Stage 4 phase labels."""
    return get_stage4_block_phase(kind)


def _set_block_cursor(
    runtime: Stage4Runtime,
    block: Stage4FrontierBlock,
) -> None:
    """Move the execution cursor onto one promptable Stage 4 block."""
    runtime.cursor = Stage4BlockCursor(block_id=block.id)


def _set_model_spec_lock_cursor(
    runtime: Stage4Runtime,
    *,
    reason: str,
) -> None:
    """Move the execution cursor into model-spec-lock pending state."""
    runtime.cursor = Stage4ModelSpecLockPendingCursor(reason=reason)


def _set_repair_barrier_cursor(
    runtime: Stage4Runtime,
    *,
    reason: str,
    scope_block_ids: tuple[str, ...],
) -> None:
    """Move the execution cursor into repair-barrier pending state."""
    runtime.cursor = Stage4RepairBarrierCursor(
        reason=reason,
        scope_block_ids=scope_block_ids,
    )


def _set_done_cursor(runtime: Stage4Runtime) -> None:
    """Move the execution cursor to the terminal Stage 4 state."""
    runtime.cursor = Stage4DoneCursor()


def _scope_phase(
    plan: Stage4Plan,
    block_ids: tuple[str, ...],
) -> str:
    """Infer the public phase label for a structural repair scope."""
    for block_id in block_ids:
        block = plan.get_block(block_id)
        if block is not None:
            return _phase_for_block_kind(block.kind)
    return "prior_blocks"


def get_active_plan_block(
    plan: Stage4Plan,
    runtime: Stage4Runtime,
) -> Stage4FrontierBlock | None:
    """Return the current active authored block from explicit runtime state."""
    cursor = runtime.cursor
    if isinstance(cursor, Stage4BlockCursor):
        return plan.get_block(cursor.block_id)
    if isinstance(
        cursor,
        (Stage4ModelSpecLockPendingCursor, Stage4RepairBarrierCursor, Stage4DoneCursor),
    ):
        return None
    return None


def get_active_prompt_block(
    plan: Stage4Plan,
    runtime: Stage4Runtime,
) -> Stage4FrontierBlock | None:
    """Return the current prompt block, including repair-scope prompt overrides."""
    block = get_active_plan_block(plan, runtime)
    if block is None:
        return None
    campaign = runtime.repair_campaign
    if campaign is None:
        return block
    return campaign.prompt_blocks_by_id.get(block.id, block)


def pending_repair_campaign_block_ids(
    campaign: Stage4RepairCampaignState,
) -> tuple[str, ...]:
    """Return the ordered pending block ids for an active repair campaign."""
    return tuple(
        block_id
        for block_id in campaign.scope_block_ids
        if block_id not in campaign.completed_block_ids
    )


def get_stage4_phase(
    runtime: Stage4Runtime,
    *,
    plan: Stage4Plan | None = None,
) -> str:
    """Return the current Stage 4 runtime phase."""
    cursor = runtime.cursor
    if isinstance(cursor, Stage4DoneCursor):
        return "done"
    if isinstance(cursor, Stage4BlockCursor):
        if plan is None:
            raise ValueError("Stage 4 phase derivation for block cursors requires the plan")
        block = plan.get_block(cursor.block_id)
        if block is None:
            raise ValueError(f"Unknown Stage 4 block id {cursor.block_id!r}")
        return _phase_for_block_kind(block.kind)
    if isinstance(cursor, Stage4ModelSpecLockPendingCursor):
        return "model_decisions"
    if isinstance(cursor, Stage4RepairBarrierCursor):
        if plan is None:
            raise ValueError("Stage 4 phase derivation for repair barriers requires the plan")
        return _scope_phase(plan, cursor.scope_block_ids)
    raise ValueError(f"Unknown Stage 4 cursor {cursor!r}")


def next_pending_block(
    blocks: tuple[Stage4FrontierBlock, ...],
    runtime: Stage4Runtime,
    *,
    skip_id: str | None = None,
) -> Stage4FrontierBlock | None:
    """Return the next pending block in deterministic order."""
    for block in blocks:
        if block.id == skip_id:
            continue
        if runtime.block_status.get(block.id) in {"pending", "reopened"}:
            return block
    return None


def _activate_model_phase(plan: Stage4Plan, runtime: Stage4Runtime) -> None:
    """Set runtime to the next pending model-decision block, if any."""
    next_block = next_pending_block(plan.model_blocks, runtime)
    if next_block is None:
        _set_model_spec_lock_cursor(
            runtime,
            reason="model decisions accepted; awaiting model_spec lock",
        )
        return
    _set_block_cursor(runtime, next_block)


def activate_review_phase(plan: Stage4Plan, runtime: Stage4Runtime) -> None:
    """Set runtime to the compact global-review block, if pending."""
    review_block = plan.review_block
    if review_block is None or block_is_accepted(runtime, review_block.id):
        _activate_prior_phase(plan, runtime)
        return
    _set_block_cursor(runtime, review_block)


def active_prior_parameter_names(runtime: Stage4Runtime) -> set[str] | None:
    """Return the locked active prior surface, or ``None`` before model lock."""
    model_spec = runtime.accepted.model_spec
    if not isinstance(model_spec, dict):
        return None
    return {
        str(parameter["name"])
        for parameter in (model_spec.get("parameters") or [])
        if isinstance(parameter, dict) and isinstance(parameter.get("name"), str)
    }


def _sync_prior_block_activity(plan: Stage4Plan, runtime: Stage4Runtime) -> None:
    """Activate only the prior blocks whose parameters survive the locked likelihood choices."""
    active_parameter_names = active_prior_parameter_names(runtime)
    if active_parameter_names is None:
        return
    for block in plan.prior_blocks:
        should_activate = bool(set(block.parameter_names) & active_parameter_names)
        current_status = runtime.block_status.get(block.id)
        if should_activate:
            if current_status == "inactive":
                runtime.block_status[block.id] = "pending"
            continue
        if current_status != "accepted":
            runtime.block_status[block.id] = "inactive"


def _activate_prior_phase(plan: Stage4Plan, runtime: Stage4Runtime) -> None:
    """Set runtime to the next pending prior block, or mark Stage 4 done."""
    _sync_prior_block_activity(plan, runtime)
    next_block = next_pending_block(plan.prior_blocks, runtime)
    if next_block is None:
        _set_done_cursor(runtime)
        return
    _set_block_cursor(runtime, next_block)


def _finish_stage4(plan: Stage4Plan, runtime: Stage4Runtime) -> None:
    """Terminate Stage 4 after the final accepted review block."""
    del plan
    _set_done_cursor(runtime)


_POST_ACCEPTANCE_NAVIGATION_BY_PHASE = {
    "model_decisions": _activate_model_phase,
    "global_review": _activate_prior_phase,
    "prior_blocks": _activate_prior_phase,
    "global_prior_review": _finish_stage4,
}


def _mark_blocks_reopened(
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    block_ids: tuple[str, ...],
) -> None:
    """Move runtime back to one or more reopened blocks."""
    if not block_ids:
        return

    blocks: list[Stage4FrontierBlock] = []
    for block_id in block_ids:
        block = plan.get_block(block_id)
        if block is None:
            raise ValueError(f"Unknown Stage 4 block id {block_id!r}")
        blocks.append(block)

    if any(_phase_for_block_kind(block.kind) == "model_decisions" for block in blocks) and (
        plan.review_block is not None
    ):
        runtime.block_status[plan.review_block.id] = "pending"

    for block_id in block_ids:
        runtime.block_status[block_id] = "reopened"

    _set_block_cursor(runtime, blocks[0])


def _advance_after_block_acceptance(
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    block: Stage4FrontierBlock,
) -> None:
    """Advance runtime after a block has been accepted."""
    phase = _phase_for_block_kind(block.kind)
    advance = _POST_ACCEPTANCE_NAVIGATION_BY_PHASE.get(phase)
    if advance is None:
        raise ValueError(f"Unsupported Stage 4 post-acceptance phase {phase!r}")
    advance(plan, runtime)


def _clear_repair_campaign(runtime: Stage4Runtime) -> None:
    """Clear any active structural repair campaign."""
    runtime.repair_campaign = None


def _merge_best_certificate(
    current: PriorPathologyCertificate | None,
    candidate: PriorPathologyCertificate | None,
) -> PriorPathologyCertificate | None:
    """Keep the best pathology certificate seen at the current repair scope."""
    if current is None:
        return candidate
    if candidate is None:
        return current
    if current.kind != candidate.kind:
        return candidate

    current_secondary = (
        current.secondary_score if current.secondary_score is not None else float("inf")
    )
    candidate_secondary = (
        candidate.secondary_score if candidate.secondary_score is not None else float("inf")
    )
    current_key = (current.primary_score, current_secondary)
    candidate_key = (candidate.primary_score, candidate_secondary)
    return candidate if candidate_key < current_key else current


def _apply_repair_campaign_progress(
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    *,
    failure_family_key: tuple[Any, ...],
    scope_kind: str,
    scope_key: str,
    scope_rank: int,
    prompt_blocks: tuple[Stage4FrontierBlock, ...],
    completed_block_ids: frozenset[str],
    requires_barrier_validation: bool,
    attempts_at_scope: int,
    best_certificate: PriorPathologyCertificate | None,
) -> None:
    """Persist repair-campaign state and move the execution cursor accordingly."""
    scope_block_ids = tuple(block.id for block in prompt_blocks)
    runtime.repair_campaign = Stage4RepairCampaignState(
        failure_family_key=failure_family_key,
        scope_kind=scope_kind,
        scope_key=scope_key,
        scope_rank=scope_rank,
        scope_block_ids=scope_block_ids,
        prompt_blocks_by_id={block.id: block for block in prompt_blocks},
        completed_block_ids=completed_block_ids,
        requires_barrier_validation=requires_barrier_validation,
        attempts_at_scope=attempts_at_scope,
        best_certificate=best_certificate,
    )
    pending_block_ids = pending_repair_campaign_block_ids(runtime.repair_campaign)
    if not pending_block_ids:
        if not requires_barrier_validation:
            _clear_repair_campaign(runtime)
            return
        _set_repair_barrier_cursor(
            runtime,
            reason=f"repair barrier pending for `{scope_key}`",
            scope_block_ids=scope_block_ids,
        )
        return

    next_block = plan.get_block(pending_block_ids[0])
    if next_block is None:
        raise ValueError(f"Unknown Stage 4 block id {pending_block_ids[0]!r}")
    _set_block_cursor(runtime, next_block)


def _start_repair_campaign(
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    repair_plan: ResolvedRepairPlan,
    *,
    accepted_block_id: str | None,
) -> None:
    """Start or widen one deterministic structural repair campaign."""
    current = runtime.repair_campaign
    attempts_at_scope = 1
    best_certificate = repair_plan.scope.pathology_certificate
    if (
        current is not None
        and current.failure_family_key == repair_plan.scope.failure_family
        and current.scope_key == repair_plan.scope.scope_key
    ):
        attempts_at_scope = current.attempts_at_scope + 1
        best_certificate = _merge_best_certificate(
            current.best_certificate,
            repair_plan.scope.pathology_certificate,
        )

    completed_block_ids = frozenset(
        ()
        if accepted_block_id is None or accepted_block_id not in repair_plan.block_ids
        else (accepted_block_id,)
    )

    if (
        any(
            (block := plan.get_block(block_id)) is not None and block.kind == "indicator_decision"
            for block_id in repair_plan.block_ids
        )
        and plan.review_block is not None
    ):
        runtime.block_status[plan.review_block.id] = "pending"

    for block_id in repair_plan.block_ids:
        if block_id == accepted_block_id:
            continue
        runtime.block_status[block_id] = "reopened"

    _apply_repair_campaign_progress(
        plan,
        runtime,
        failure_family_key=repair_plan.scope.failure_family,
        scope_kind=repair_plan.scope.scope_kind,
        scope_key=repair_plan.scope.scope_key,
        scope_rank=repair_plan.scope.scope_rank,
        prompt_blocks=repair_plan.prompt_blocks,
        completed_block_ids=completed_block_ids,
        requires_barrier_validation=repair_plan.requires_barrier_validation,
        attempts_at_scope=attempts_at_scope,
        best_certificate=best_certificate,
    )


def _advance_repair_campaign_after_acceptance(
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    accepted_block_id: str,
) -> bool:
    """Advance the active repair campaign after one block is accepted."""
    campaign = runtime.repair_campaign
    pending_block_ids = () if campaign is None else pending_repair_campaign_block_ids(campaign)
    if campaign is None or accepted_block_id not in pending_block_ids:
        return False

    completed_block_ids = frozenset((*campaign.completed_block_ids, accepted_block_id))
    if (
        len(completed_block_ids) == len(campaign.scope_block_ids)
        and not campaign.requires_barrier_validation
    ):
        _clear_repair_campaign(runtime)
        accepted_block = plan.get_block(accepted_block_id)
        if accepted_block is None:
            raise ValueError(f"Unknown Stage 4 block id {accepted_block_id!r}")
        _advance_after_block_acceptance(plan, runtime, accepted_block)
        return True

    _apply_repair_campaign_progress(
        plan,
        runtime,
        failure_family_key=campaign.failure_family_key,
        scope_kind=campaign.scope_kind,
        scope_key=campaign.scope_key,
        scope_rank=campaign.scope_rank,
        prompt_blocks=tuple(
            campaign.prompt_blocks_by_id[block_id] for block_id in campaign.scope_block_ids
        ),
        completed_block_ids=completed_block_ids,
        requires_barrier_validation=campaign.requires_barrier_validation,
        attempts_at_scope=campaign.attempts_at_scope,
        best_certificate=campaign.best_certificate,
    )
    return True


def apply_stage4_block_acceptance(
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    block_id: str,
) -> None:
    """Advance Stage 4 after one block has been accepted."""
    if _advance_repair_campaign_after_acceptance(plan, runtime, block_id):
        return
    accepted_block = plan.get_block(block_id)
    if accepted_block is None:
        raise ValueError(f"Unknown Stage 4 block id {block_id!r}")
    _advance_after_block_acceptance(plan, runtime, accepted_block)


def apply_stage4_repair_plan(
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    repair_plan: ResolvedRepairPlan,
    *,
    accepted_block_id: str | None,
) -> None:
    """Apply a typed repair plan to runtime cursor and campaign state."""
    if repair_plan.uses_repair_campaign:
        _start_repair_campaign(
            plan,
            runtime,
            repair_plan,
            accepted_block_id=accepted_block_id,
        )
        return
    _clear_repair_campaign(runtime)
    _mark_blocks_reopened(plan, runtime, repair_plan.block_ids)


def apply_stage4_barrier_validation_success(
    plan: Stage4Plan,
    runtime: Stage4Runtime,
) -> None:
    """Advance Stage 4 after a successful repair-campaign barrier validation."""
    _clear_repair_campaign(runtime)
    _activate_prior_phase(plan, runtime)


def make_stage4_runtime(plan: Stage4Plan) -> Stage4Runtime:
    """Create the mutable runtime for a new Stage 4 plan execution."""
    runtime = Stage4Runtime(
        block_status={block.id: "pending" for block in plan.all_blocks},
    )
    if plan.model_blocks:
        _set_block_cursor(runtime, plan.model_blocks[0])
    else:
        _set_model_spec_lock_cursor(
            runtime,
            reason="awaiting automatic model_spec lock",
        )
    if plan.prior_review_block is not None:
        prior_review_id = plan.prior_review_block_id
        assert prior_review_id is not None
        runtime.block_status[prior_review_id] = "inactive"
    return runtime
