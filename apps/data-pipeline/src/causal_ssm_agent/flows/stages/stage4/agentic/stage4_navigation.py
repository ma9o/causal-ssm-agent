"""Stage 4 selectors, wait-state navigation, and repair-campaign helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .stage4_block_specs import get_stage4_block_phase
from .stage4_state import Stage4RepairCampaignState, Stage4Runtime

if TYPE_CHECKING:
    from causal_ssm_agent.workers.schemas_prior import PriorPathologyCertificate

    from .stage4_orchestrator import Stage4FrontierBlock, Stage4Plan
    from .stage4_repair import ResolvedRepairPlan


def block_is_accepted(runtime: Stage4Runtime, block_id: str) -> bool:
    """Whether a block is currently accepted in runtime state."""
    return runtime.domain.block_status.get(block_id) == "accepted"


def _phase_for_block_kind(kind: str) -> str:
    """Project one authored block kind onto the public Stage 4 phase labels."""
    return get_stage4_block_phase(kind)


def _set_block_cursor(
    runtime: Stage4Runtime,
    block: Stage4FrontierBlock,
) -> None:
    """Move the persisted wait-state onto one promptable Stage 4 block."""
    runtime.domain.done = False
    runtime.domain.active_block_id = block.id


def _clear_active_block(runtime: Stage4Runtime) -> None:
    """Clear the current promptable block while deterministic settling continues."""
    runtime.domain.active_block_id = None


def _set_done_cursor(runtime: Stage4Runtime) -> None:
    """Move the runtime to the terminal Stage 4 state."""
    runtime.domain.done = True
    runtime.domain.active_block_id = None


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
    """Return the current active authored block from persisted wait-state."""
    if runtime.domain.done or runtime.domain.active_block_id is None:
        return None
    return plan.get_block(runtime.domain.active_block_id)


def get_active_prompt_block(
    plan: Stage4Plan,
    runtime: Stage4Runtime,
) -> Stage4FrontierBlock | None:
    """Return the current prompt block, including repair-scope prompt overrides."""
    block = get_active_plan_block(plan, runtime)
    if block is None:
        return None
    campaign = runtime.domain.repair_campaign
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


def current_initialization_policy(runtime: Stage4Runtime) -> str | None:
    """Return the latest draft-or-accepted initialization policy."""
    return runtime.domain.draft_model.initialization_policy or (
        runtime.domain.accepted.model_spec or {}
    ).get("initialization_policy")


def current_equilibrium_forcing(runtime: Stage4Runtime) -> bool | None:
    """Return the latest draft-or-accepted equilibrium-forcing decision."""
    equilibrium_forcing = runtime.domain.draft_model.equilibrium_forcing
    if equilibrium_forcing is not None:
        return equilibrium_forcing
    accepted_model_spec = runtime.domain.accepted.model_spec or {}
    value = accepted_model_spec.get("equilibrium_forcing")
    return None if value is None else bool(value)


def current_likelihood_lookup(runtime: Stage4Runtime) -> dict[str, dict[str, Any]]:
    """Return the current likelihood choice per indicator."""
    lookup: dict[str, dict[str, Any]] = {}
    for likelihood in (runtime.domain.accepted.model_spec or {}).get("likelihoods") or []:
        if not isinstance(likelihood, dict):
            continue
        variable = likelihood.get("variable")
        if isinstance(variable, str):
            lookup[variable] = likelihood
    for variable, choice in runtime.domain.draft_model.distribution_choices.items():
        lookup[variable] = choice
    return lookup


def active_prior_parameter_names(runtime: Stage4Runtime) -> set[str] | None:
    """Return the locked active prior surface, or ``None`` before model lock."""
    model_spec = runtime.domain.accepted.model_spec
    if not isinstance(model_spec, dict):
        return None
    return {
        str(parameter["name"])
        for parameter in (model_spec.get("parameters") or [])
        if isinstance(parameter, dict) and isinstance(parameter.get("name"), str)
    }


def required_prior_parameter_names(model_spec: dict[str, Any] | None) -> tuple[str, ...]:
    """Return the active non-optional parameters that still require priors."""
    required_names: list[str] = []
    for parameter in (model_spec or {}).get("parameters") or []:
        if not isinstance(parameter, dict):
            continue
        if parameter.get("role") in {"initial_state_mean", "initial_state_sd"}:
            continue
        name = parameter.get("name")
        if isinstance(name, str):
            required_names.append(name)
    return tuple(required_names)


def active_block_parameter_names(
    block: Stage4FrontierBlock,
    runtime: Stage4Runtime,
) -> tuple[str, ...]:
    """Return the active subset of one block's parameter surface."""
    active_parameter_names = active_prior_parameter_names(runtime)
    if active_parameter_names is None:
        return block.parameter_names
    return tuple(name for name in block.parameter_names if name in active_parameter_names)


def required_block_parameter_names(
    block: Stage4FrontierBlock,
    runtime: Stage4Runtime,
) -> tuple[str, ...]:
    """Return the active required parameter names for one block."""
    active_parameter_names = active_prior_parameter_names(runtime)
    candidates = (
        block.required_parameter_names if block.required_parameter_names else block.parameter_names
    )
    if active_parameter_names is None:
        return candidates
    return tuple(name for name in candidates if name in active_parameter_names)


def block_coverage_is_satisfied(
    block: Stage4FrontierBlock,
    runtime: Stage4Runtime,
) -> bool:
    """Whether the accepted prior state fully covers the block's active contract."""
    authored_parameter_names = set(runtime.domain.accepted.authored_priors)
    if block.coverage_policy == "subset_allowed":
        return False
    required_parameter_names = required_block_parameter_names(block, runtime)
    return bool(required_parameter_names) and all(
        name in authored_parameter_names for name in required_parameter_names
    )


def repair_scope_summary(runtime: Stage4Runtime) -> str | None:
    """Summarize the active repair scope for prompt-local frontier rendering."""
    campaign = runtime.domain.repair_campaign
    if campaign is None:
        return None
    pending_block_ids = pending_repair_campaign_block_ids(campaign)
    return f"{campaign.scope_key} ({len(pending_block_ids)} remaining)"


def get_stage4_phase(
    runtime: Stage4Runtime,
    *,
    plan: Stage4Plan | None = None,
) -> str:
    """Return the current Stage 4 runtime phase."""
    if runtime.domain.done:
        return "done"

    if plan is not None:
        block = get_active_plan_block(plan, runtime)
        if block is not None:
            return _phase_for_block_kind(block.kind)

    campaign = runtime.domain.repair_campaign
    if campaign is not None and plan is not None:
        return _scope_phase(plan, campaign.scope_block_ids)

    if (
        plan is not None
        and plan.review_block is not None
        and runtime.domain.block_status.get(plan.review_block.id) in {"pending", "reopened"}
    ):
        return _phase_for_block_kind(plan.review_block.kind)

    if (
        plan is not None
        and plan.prior_review_block is not None
        and runtime.domain.block_status.get(plan.prior_review_block.id) in {"pending", "reopened"}
    ):
        return _phase_for_block_kind(plan.prior_review_block.kind)

    if runtime.domain.accepted.model_spec is None:
        return "model_decisions"
    return "prior_blocks"


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
        if runtime.domain.block_status.get(block.id) in {"pending", "reopened"}:
            return block
    return None


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
    """Persist repair-campaign state and clear the promptable block until settled."""
    scope_block_ids = tuple(block.id for block in prompt_blocks)
    runtime.domain.repair_campaign = Stage4RepairCampaignState(
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
    _clear_active_block(runtime)


def _start_repair_campaign(
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    repair_plan: ResolvedRepairPlan,
    *,
    accepted_block_id: str | None,
) -> None:
    """Start or widen one deterministic structural repair campaign."""
    current = runtime.domain.repair_campaign
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
        runtime.domain.block_status[plan.review_block.id] = "pending"

    for block_id in repair_plan.block_ids:
        if block_id == accepted_block_id:
            continue
        runtime.domain.block_status[block_id] = "reopened"

    _apply_repair_campaign_progress(
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
    runtime: Stage4Runtime,
    accepted_block_id: str,
) -> bool:
    """Advance the active repair campaign after one block is accepted."""
    campaign = runtime.domain.repair_campaign
    pending_block_ids = () if campaign is None else pending_repair_campaign_block_ids(campaign)
    if campaign is None or accepted_block_id not in pending_block_ids:
        return False

    completed_block_ids = frozenset((*campaign.completed_block_ids, accepted_block_id))
    if (
        len(completed_block_ids) == len(campaign.scope_block_ids)
        and not campaign.requires_barrier_validation
    ):
        runtime.domain.repair_campaign = None
        _clear_active_block(runtime)
        return True

    _apply_repair_campaign_progress(
        runtime=runtime,
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
    """Update repair-campaign bookkeeping after one block has been accepted."""
    del plan
    if _advance_repair_campaign_after_acceptance(runtime, block_id):
        return
    _clear_active_block(runtime)


def apply_stage4_repair_plan(
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    repair_plan: ResolvedRepairPlan,
    *,
    accepted_block_id: str | None,
) -> None:
    """Apply a typed repair plan to runtime state without choosing the next block."""
    if repair_plan.uses_repair_campaign:
        _start_repair_campaign(
            plan,
            runtime,
            repair_plan,
            accepted_block_id=accepted_block_id,
        )
        return
    runtime.domain.repair_campaign = None
    reopened_model_decision = False
    for block_id in repair_plan.block_ids:
        block = plan.get_block(block_id)
        if block is None:
            raise ValueError(f"Unknown Stage 4 block id {block_id!r}")
        reopened_model_decision |= _phase_for_block_kind(block.kind) == "model_decisions"
        runtime.domain.block_status[block_id] = "reopened"
    if reopened_model_decision and plan.review_block is not None:
        runtime.domain.block_status[plan.review_block.id] = "pending"
    _clear_active_block(runtime)


def apply_stage4_barrier_validation_success(
    plan: Stage4Plan,
    runtime: Stage4Runtime,
) -> None:
    """Advance Stage 4 after a successful repair-campaign barrier validation."""
    del plan
    runtime.domain.repair_campaign = None
    _clear_active_block(runtime)


def reconcile_locked_prior_surface(
    plan: Stage4Plan,
    runtime: Stage4Runtime,
) -> None:
    """Prune stale priors and resync prior block activity after a model re-lock."""
    active_parameter_names = active_prior_parameter_names(runtime)
    if active_parameter_names is None:
        return

    runtime.domain.accepted.authored_priors = {
        name: prior
        for name, prior in runtime.domain.accepted.authored_priors.items()
        if name in active_parameter_names
    }
    required_parameter_names = set(
        required_prior_parameter_names(runtime.domain.accepted.model_spec)
    )
    if not required_parameter_names.issubset(runtime.domain.accepted.authored_priors):
        runtime.domain.accepted.resolved_priors = None

    _sync_prior_block_activity(plan, runtime)


def _sync_prior_block_activity(plan: Stage4Plan, runtime: Stage4Runtime) -> None:
    """Sync prior-block statuses to the locked active parameter surface."""
    active_parameter_names = active_prior_parameter_names(runtime)
    if active_parameter_names is None:
        return

    for block in plan.prior_blocks:
        current_status = runtime.domain.block_status.get(block.id)
        block_active_parameters = active_block_parameter_names(block, runtime)
        if not block_active_parameters:
            runtime.domain.block_status[block.id] = "inactive"
            continue
        if block_coverage_is_satisfied(block, runtime):
            runtime.domain.block_status[block.id] = "accepted"
            continue
        if current_status == "accepted":
            runtime.domain.block_status[block.id] = "reopened"
        elif current_status == "inactive":
            runtime.domain.block_status[block.id] = "pending"


def select_next_wait_block(
    plan: Stage4Plan,
    runtime: Stage4Runtime,
) -> Stage4FrontierBlock | None:
    """Return the next promptable block after deterministic settling."""
    campaign = runtime.domain.repair_campaign
    if campaign is not None:
        pending_block_ids = pending_repair_campaign_block_ids(campaign)
        if pending_block_ids:
            next_block = plan.get_block(pending_block_ids[0])
            if next_block is None:
                raise ValueError(f"Unknown Stage 4 block id {pending_block_ids[0]!r}")
            return next_block

    next_block = next_pending_block(plan.model_blocks, runtime)
    if next_block is not None:
        return next_block

    review_block = plan.review_block
    if review_block is not None and runtime.domain.block_status.get(review_block.id) in {
        "pending",
        "reopened",
    }:
        return review_block

    _sync_prior_block_activity(plan, runtime)
    next_block = next_pending_block(plan.prior_blocks, runtime)
    if next_block is not None:
        return next_block

    prior_review_block = plan.prior_review_block
    if prior_review_block is not None and runtime.domain.block_status.get(
        prior_review_block.id
    ) in {
        "pending",
        "reopened",
    }:
        return prior_review_block

    return None


def make_stage4_runtime(plan: Stage4Plan) -> Stage4Runtime:
    """Create the mutable runtime for a new Stage 4 plan execution."""
    runtime = Stage4Runtime()
    runtime.domain.block_status = {block.id: "pending" for block in plan.all_blocks}
    if plan.model_blocks:
        _set_block_cursor(runtime, plan.model_blocks[0])
    else:
        _clear_active_block(runtime)
    if plan.prior_review_block is not None:
        prior_review_id = plan.prior_review_block_id
        assert prior_review_id is not None
        runtime.domain.block_status[prior_review_id] = "inactive"
    return runtime
