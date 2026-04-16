"""Stage 4 reducer logic and accepted-state transitions."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from causal_ssm_agent.flows.stages.stage4.model_spec_decisions import (
    validate_model_spec_decisions_dict,
)
from causal_ssm_agent.models.compilation_errors import AggregatedCompileError

from .stage4_events import (
    Stage4AcceptedStatePersistedEvent,
    Stage4BarrierValidationPassedEvent,
    Stage4BlockAcceptedEvent,
    Stage4ReducerEvent,
    Stage4RepairPlannedEvent,
)
from .stage4_feedback import (
    Stage4ValidationPacket,
    build_validation_packet_for_block,
    render_stage4_validation_feedback,
    should_store_stage4_validation_packet,
)
from .stage4_navigation import (
    _set_block_cursor,
    _set_done_cursor,
    apply_stage4_barrier_validation_success,
    apply_stage4_block_acceptance,
    apply_stage4_repair_plan,
    block_is_accepted,
    get_active_plan_block,
    get_active_prompt_block,
    next_pending_block,
    pending_repair_campaign_block_ids,
    reconcile_locked_prior_surface,
    select_next_wait_block,
)
from .stage4_repair import (
    ResolvedRepairPlan,
    ResolvedRepairScope,
    Stage4PriorRepairDecision,
    _ordered_block_ids,
    build_repair_plan,
    classify_validation_outcome,
    resolve_prior_repair_decision,
)
from .stage4_submission import (
    get_stage4_block_handler,
    validate_stage4_block_coverage,
    validate_stage4_submission_payload,
)
from .stage4_text import summarize_stage4_names

if TYPE_CHECKING:
    from causal_ssm_agent.flows.stages.stage4.assembly import AssemblyValidation

    from .stage4_orchestrator import Stage4FrontierBlock, Stage4Plan
    from .stage4_skeleton import Stage4Skeleton
    from .stage4_state import Stage4DraftModel, Stage4RepairCampaignState, Stage4Runtime
    from .stage4_submission import Stage4BlockHandler
    from .stage4_types import Stage4Deps

_RECOVERABLE_STAGE4_REDUCER_ERRORS = (
    AggregatedCompileError,
    ValidationError,
    ValueError,
)


@dataclass(frozen=True)
class Stage4StepResult:
    """Reducer transition returned by a single Stage 4 step."""

    validation_packet: Stage4ValidationPacket
    stage_output: dict[str, Any] | None = None
    events: tuple[Stage4ReducerEvent, ...] = ()

    @property
    def repair_plan(self) -> ResolvedRepairPlan | None:
        """Return the repair plan routed by this update, if any."""
        for event in self.events:
            if isinstance(event, Stage4RepairPlannedEvent):
                return event.repair_plan
        return None


@dataclass(frozen=True)
class _Stage4PriorCampaignContext:
    """Reducer-owned campaign context for one prior submission."""

    campaign: Stage4RepairCampaignState | None
    pending_block_ids: tuple[str, ...]
    in_active_campaign: bool
    final_campaign_block: bool


@dataclass(frozen=True)
class _Stage4PriorSubmissionState:
    """Typed intermediate state for one prior-submission reducer pass."""

    stage_output: dict[str, Any] | None
    validation: AssemblyValidation | None
    validation_packet: Stage4ValidationPacket
    changed_parameters: tuple[str, ...]
    repair_plan: ResolvedRepairPlan | None = None


def _format_repair_campaign_feedback(
    repair_plan: ResolvedRepairPlan,
    *,
    accepted_block_id: str | None,
    next_block: Stage4FrontierBlock | None,
) -> str:
    """Render bounded repair-campaign progress for the LLM."""
    lines = [
        "REPAIR CAMPAIGN ACTIVE:",
        f"- scope: `{repair_plan.scope.scope_key}`",
        f"- reason: {repair_plan.scope.reason}",
    ]
    if accepted_block_id is not None:
        lines.append(f"- kept `{accepted_block_id}` as part of the repair scope")
    if next_block is not None:
        lines.append(f"- next repair block: `{next_block.id}` ({next_block.kind})")
    else:
        lines.append("- repair scope ready for barrier validation")
    return "\n".join(lines)


def _format_block_saved_feedback(
    block: Stage4FrontierBlock,
    next_block: Stage4FrontierBlock | None,
) -> str:
    """Acknowledge an accepted block and point to the next frontier."""
    lines = [
        "BLOCK ACCEPTED:",
        f"- saved `{block.id}`",
    ]
    if next_block is not None:
        lines.append(f"- next block: `{next_block.id}` ({next_block.kind})")
    else:
        lines.append("- no remaining blocks in this phase")
    return "\n".join(lines)


def persist_stage4_stage_output(
    runtime: Stage4Runtime,
    stage_output: dict[str, Any] | None,
) -> None:
    """Merge accepted Stage 4 output into reducer-owned state."""
    runtime.domain.accepted.apply_stage_output(stage_output)
    if stage_output is not None and "model_spec" in stage_output:
        runtime.domain.draft_model.sync_from_model_spec(runtime.domain.accepted.model_spec)
        runtime.domain.model_lock_pending = False


def _make_stage4_reopened_transitions(
    repair_plan: ResolvedRepairPlan,
    *,
    accepted_block_id: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Build reopen transition payloads for one repair plan."""
    return tuple(
        {
            "block_id": block_id,
            "status": "reopened",
            "detail_kind": "revision",
            "reason": repair_plan.scope.reason,
            "scope_kind": repair_plan.scope.scope_kind,
        }
        for block_id in repair_plan.block_ids
        if block_id != accepted_block_id
    )


def _all_model_blocks_accepted(plan: Stage4Plan, runtime: Stage4Runtime) -> bool:
    """Whether every model-decision block is accepted in runtime state."""
    return all(block_is_accepted(runtime, block.id) for block in plan.model_blocks)


def _needs_model_spec_lock(
    plan: Stage4Plan,
    runtime: Stage4Runtime,
) -> bool:
    """Whether settling must lock or re-lock the deterministic model spec."""
    return _all_model_blocks_accepted(plan, runtime) and (
        runtime.domain.accepted.model_spec is None or runtime.domain.model_lock_pending
    )


def _repair_barrier_pending(runtime: Stage4Runtime) -> bool:
    """Whether a repair campaign is waiting on deterministic barrier validation."""
    campaign = runtime.domain.repair_campaign
    return bool(campaign is not None and not pending_repair_campaign_block_ids(campaign))


_MISSING_DISTRIBUTION_CHOICE_RE = re.compile(
    r"missing distribution_choice for ambiguous indicator '([^']+)'"
)
_DISTRIBUTION_CHOICE_INDEX_RE = re.compile(r"distribution_choices\[(\d+)\]")


def _model_lock_failure_block_ids(
    plan: Stage4Plan,
    draft_model: Stage4DraftModel,
    errors: list[str],
    hint_block: Stage4FrontierBlock | None,
) -> tuple[str, ...]:
    """Return the model-decision blocks implicated by model-spec lock errors."""
    del hint_block
    block_ids: set[str] = set()
    distribution_choices = list((draft_model.distribution_choices or {}).values())
    model_configuration_block_id = next(
        (block.id for block in plan.model_blocks if block.kind == "model_configuration"),
        None,
    )

    for error in errors:
        if not isinstance(error, str):
            continue

        if model_configuration_block_id is not None and (
            "initialization_policy" in error or "equilibrium_forcing" in error
        ):
            block_ids.add(model_configuration_block_id)

        missing_distribution_match = _MISSING_DISTRIBUTION_CHOICE_RE.search(error)
        if missing_distribution_match is not None:
            indicator_name = missing_distribution_match.group(1)
            for block in plan.model_blocks:
                if block.kind == "indicator_decision" and indicator_name in block.variable_names:
                    block_ids.add(block.id)

        indexed_distribution_match = _DISTRIBUTION_CHOICE_INDEX_RE.search(error)
        if indexed_distribution_match is not None:
            index = int(indexed_distribution_match.group(1))
            if 0 <= index < len(distribution_choices):
                indicator_name = distribution_choices[index].get("variable")
                if isinstance(indicator_name, str):
                    for block in plan.model_blocks:
                        if (
                            block.kind == "indicator_decision"
                            and indicator_name in block.variable_names
                        ):
                            block_ids.add(block.id)

    if block_ids:
        return _ordered_block_ids(plan, block_ids)
    return ()


def _make_stage4_block_accepted_event(
    block: Stage4FrontierBlock,
    normalized: dict[str, Any],
    *,
    stage_output: dict[str, Any] | None = None,
) -> Stage4BlockAcceptedEvent:
    """Build the explicit reducer event for one accepted block."""
    handler = get_stage4_block_handler(block.kind)
    distribution_choice = normalized.get("distribution_choice")
    if not isinstance(distribution_choice, dict):
        distribution_choice = None
    model_configuration = normalized.get("model_configuration")
    if not isinstance(model_configuration, dict):
        model_configuration = None
    return Stage4BlockAcceptedEvent(
        block_id=block.id,
        transition_payload=handler.build_accepted_transition(block, normalized),
        distribution_choice=distribution_choice,
        model_configuration=model_configuration,
        stage_output=stage_output,
    )


def _make_stage4_repair_planned_event(
    repair_plan: ResolvedRepairPlan,
    *,
    accepted_block_event: Stage4BlockAcceptedEvent | None = None,
) -> Stage4RepairPlannedEvent:
    """Build the explicit reducer event for a routed repair plan."""
    return Stage4RepairPlannedEvent(
        repair_plan=repair_plan,
        accepted_block_id=None if accepted_block_event is None else accepted_block_event.block_id,
        accepted_transition_payload=(
            None if accepted_block_event is None else accepted_block_event.transition_payload
        ),
        distribution_choice=(
            None if accepted_block_event is None else accepted_block_event.distribution_choice
        ),
        model_configuration=(
            None if accepted_block_event is None else accepted_block_event.model_configuration
        ),
        stage_output=None if accepted_block_event is None else accepted_block_event.stage_output,
    )


def _apply_stage4_event(
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    event: Stage4ReducerEvent,
) -> tuple[dict[str, Any], ...]:
    """Apply one typed Stage 4 reducer event."""
    if isinstance(event, Stage4AcceptedStatePersistedEvent):
        persist_stage4_stage_output(runtime, event.stage_output)
        return ()

    if isinstance(event, Stage4BlockAcceptedEvent):
        _apply_event_acceptance_side_effects(
            runtime,
            distribution_choice=event.distribution_choice,
            model_configuration=event.model_configuration,
            stage_output=event.stage_output,
        )
        runtime.domain.block_status[event.block_id] = "accepted"
        transitions = () if event.transition_payload is None else (event.transition_payload,)
        apply_stage4_block_acceptance(plan, runtime, event.block_id)
        return transitions

    if isinstance(event, Stage4RepairPlannedEvent):
        _apply_event_acceptance_side_effects(
            runtime,
            distribution_choice=event.distribution_choice,
            model_configuration=event.model_configuration,
            stage_output=event.stage_output,
        )
        if event.accepted_block_id is not None:
            runtime.domain.block_status[event.accepted_block_id] = "accepted"
        transitions = list(
            _make_stage4_reopened_transitions(
                event.repair_plan,
                accepted_block_id=event.accepted_block_id,
            )
        )
        if event.accepted_transition_payload is not None:
            transitions.insert(0, event.accepted_transition_payload)
        apply_stage4_repair_plan(
            plan,
            runtime,
            event.repair_plan,
            accepted_block_id=event.accepted_block_id,
        )
        return tuple(transitions)

    if isinstance(event, Stage4BarrierValidationPassedEvent):
        apply_stage4_barrier_validation_success(plan, runtime)
        packet = event.success_packet
        if not event.success_packet.retain_for_next_prompt:
            representative_block = plan.get_block(event.representative_block_id)
            if representative_block is None:
                raise ValueError(
                    "Unknown Stage 4 representative block "
                    f"{event.representative_block_id!r} after barrier validation"
                )
            packet = build_validation_packet_for_block(
                block_id=representative_block.id,
                status="accepted",
                feedback=_format_block_saved_feedback(
                    representative_block,
                    get_active_plan_block(plan, runtime),
                ),
                retain_for_next_prompt=True,
                capture_stage_output=False,
            )
        runtime.interaction.last_validation_packet = (
            packet if should_store_stage4_validation_packet(packet) else None
        )
        return ()

    raise TypeError(f"Unsupported Stage 4 reducer event {event!r}")


def _apply_event_acceptance_side_effects(
    runtime: Stage4Runtime,
    *,
    distribution_choice: dict[str, Any] | None,
    model_configuration: dict[str, Any] | None,
    stage_output: dict[str, Any] | None,
) -> None:
    """Apply the accepted state deltas shared by block-accept and repair events."""
    draft_model = runtime.domain.draft_model
    if distribution_choice is not None:
        draft_model.distribution_choices[distribution_choice["variable"]] = distribution_choice
        runtime.domain.model_lock_pending = True
    if model_configuration is not None:
        draft_model.initialization_policy = str(model_configuration["initialization_policy"])
        draft_model.equilibrium_forcing = bool(model_configuration["equilibrium_forcing"])
        runtime.domain.model_lock_pending = True
    if stage_output is not None:
        persist_stage4_stage_output(runtime, stage_output)


def _apply_stage4_events(
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    events: tuple[Stage4ReducerEvent, ...],
) -> tuple[dict[str, Any], ...]:
    """Apply a sequence of typed Stage 4 reducer events."""
    transitions: list[dict[str, Any]] = []
    for event in events:
        transitions.extend(_apply_stage4_event(plan, runtime, event))
    return tuple(transitions)


def _apply_stage4_step_result(
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    result: Stage4StepResult,
) -> tuple[dict[str, Any], ...]:
    """Apply a reducer transition result in one place."""
    transitions = _apply_stage4_events(plan, runtime, result.events)
    if any(isinstance(event, Stage4BarrierValidationPassedEvent) for event in result.events):
        return transitions
    runtime.interaction.last_validation_packet = (
        result.validation_packet
        if should_store_stage4_validation_packet(result.validation_packet)
        else None
    )
    return transitions


def _apply_indicator_submission(
    *,
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    active_block: Stage4FrontierBlock,
    normalized: dict[str, Any],
    deps: Stage4Deps,
) -> Stage4StepResult:
    """Apply one indicator-distribution decision."""
    del deps
    feedback = _format_block_saved_feedback(
        active_block,
        next_pending_block(plan.model_blocks, runtime, skip_id=active_block.id),
    )
    return Stage4StepResult(
        validation_packet=build_validation_packet_for_block(
            block_id=active_block.id,
            status="accepted",
            feedback=feedback,
            retain_for_next_prompt=True,
            capture_stage_output=False,
        ),
        events=(_make_stage4_block_accepted_event(active_block, normalized),),
    )


def _build_prior_campaign_context(
    runtime: Stage4Runtime,
    active_block: Stage4FrontierBlock,
) -> _Stage4PriorCampaignContext:
    """Project the active repair-campaign context for one prior submission."""
    campaign = runtime.domain.repair_campaign
    pending_block_ids = () if campaign is None else pending_repair_campaign_block_ids(campaign)
    in_active_campaign = campaign is not None and active_block.id in pending_block_ids
    final_campaign_block = (
        in_active_campaign and campaign is not None and pending_block_ids == (active_block.id,)
    )
    return _Stage4PriorCampaignContext(
        campaign=campaign,
        pending_block_ids=pending_block_ids,
        in_active_campaign=in_active_campaign,
        final_campaign_block=final_campaign_block,
    )


def _ground_prior_submission(
    *,
    runtime: Stage4Runtime,
    normalized: dict[str, Any],
    deps: Stage4Deps,
    campaign_context: _Stage4PriorCampaignContext,
) -> _Stage4PriorSubmissionState:
    """Ground one prior bundle against the current accepted Stage 4 state."""
    grounding_result = deps.grounding_fn(
        {"priors": normalized["priors"]},
        deps.causal_spec,
        current=runtime.domain.accepted.as_current(),
        data_for_model=deps.data_for_model,
        indicator_audits=deps.indicator_audits,
        skip_ppc=campaign_context.campaign is not None,
    )
    stage_output = grounding_result.stage_output
    validation = stage_output.get("validation") if stage_output else None
    return _Stage4PriorSubmissionState(
        stage_output=stage_output,
        validation=validation,
        validation_packet=grounding_result.validation_packet,
        changed_parameters=tuple(normalized["priors"]),
    )


def _should_run_partial_drift_guard(
    *,
    active_block: Stage4FrontierBlock,
    state: _Stage4PriorSubmissionState,
) -> bool:
    """Whether the reducer should run the local partial-drift advisory guard."""
    return (
        state.stage_output is not None
        and state.validation is not None
        and state.validation.compile_ok
        and not state.validation.pp_checked
        and active_block.kind in {"dynamics_prior", "effect_prior"}
    )


def _apply_prior_partial_drift_guard(
    *,
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    active_block: Stage4FrontierBlock,
    deps: Stage4Deps,
    state: _Stage4PriorSubmissionState,
) -> _Stage4PriorSubmissionState:
    """Apply the local partial-drift guard for dynamics/effect prior blocks."""
    from .stage4_partial_drift import (
        validate_dynamics_block_partial_drift,
        validate_effect_block_partial_drift,
    )
    from .stage4_repair import (
        classify_compile_failure_route,
        classify_prior_failure_blocks,
    )

    if not _should_run_partial_drift_guard(active_block=active_block, state=state):
        return state

    stage_output = state.stage_output or {}
    authored_priors = stage_output.get("authored_priors")
    try:
        if active_block.kind == "dynamics_prior":
            partial_guard = validate_dynamics_block_partial_drift(
                model_spec=runtime.domain.accepted.model_spec,
                authored_priors=authored_priors,
                causal_spec=deps.causal_spec,
                active_construct_names=active_block.construct_names,
                active_parameter_names=active_block.parameter_names,
            )
        else:
            partial_guard = validate_effect_block_partial_drift(
                model_spec=runtime.domain.accepted.model_spec,
                authored_priors=authored_priors,
                causal_spec=deps.causal_spec,
                target_construct=str(active_block.payload.get("target_construct", "")),
                active_parameter_names=active_block.parameter_names,
            )
    except _RECOVERABLE_STAGE4_REDUCER_ERRORS as exc:
        feedback = f"COMPILE ERROR:\n{exc}"
        return replace(
            state,
            stage_output=None,
            validation=None,
            validation_packet=build_validation_packet_for_block(
                block_id=active_block.id,
                status="compile_error",
                feedback=feedback,
                changed_parameters=state.changed_parameters,
                retain_for_next_prompt=True,
                capture_stage_output=False,
            ),
            repair_plan=classify_compile_failure_route(plan, active_block, str(exc)),
        )

    if partial_guard is None:
        return state

    assert state.validation is not None
    partial_diagnostic, partial_feedback = partial_guard
    validation = state.validation.__class__(
        normalized_model_spec=state.validation.normalized_model_spec,
        compile_ok=state.validation.compile_ok,
        compile_error=state.validation.compile_error,
        compiled_ssm=state.validation.compiled_ssm,
        pp_checked=True,
        pp_valid=False,
        diagnostics=[partial_diagnostic],
        pp_raw_samples=state.validation.pp_raw_samples,
    )
    return replace(
        state,
        stage_output=None,
        validation=validation,
        validation_packet=build_validation_packet_for_block(
            block_id=active_block.id,
            status="partial_drift_failure",
            feedback=partial_feedback,
            validation=validation,
            changed_parameters=state.changed_parameters,
            retain_for_next_prompt=True,
            capture_stage_output=False,
        ),
        repair_plan=classify_prior_failure_blocks(
            plan,
            active_block,
            validation,
            runtime,
        ),
    )


def _classify_prior_submission_route(
    *,
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    active_block: Stage4FrontierBlock,
    state: _Stage4PriorSubmissionState,
) -> _Stage4PriorSubmissionState:
    """Classify compile and prior-predictive failures into repair routes."""
    if state.repair_plan is not None:
        return state
    validation_route = classify_validation_outcome(
        plan,
        active_block,
        state.validation,
        runtime,
        feedback=render_stage4_validation_feedback(state.validation_packet),
    )
    if validation_route.repair_plan is not None:
        return replace(
            state,
            repair_plan=validation_route.repair_plan,
        )
    return state


def _build_repair_campaign_progress_feedback(
    active_block: Stage4FrontierBlock,
    campaign_context: _Stage4PriorCampaignContext,
    plan: Stage4Plan,
) -> str:
    """Render reducer-owned repair-campaign progress feedback."""
    campaign = campaign_context.campaign
    assert campaign is not None
    next_block_id = next(
        (
            block_id
            for block_id in campaign_context.pending_block_ids
            if block_id != active_block.id
        ),
        None,
    )
    next_block = (
        None
        if next_block_id is None
        else campaign.prompt_blocks_by_id.get(next_block_id) or plan.get_block(next_block_id)
    )
    return (
        "REPAIR CAMPAIGN PROGRESS:\n"
        f"- kept `{active_block.id}` within `{campaign.scope_key}`\n"
        + (
            f"- next repair block: `{next_block.id}` ({next_block.kind})"
            if next_block is not None
            else "- barrier validation pending"
        )
    )


def _build_campaign_progress_result(
    *,
    plan: Stage4Plan,
    active_block: Stage4FrontierBlock,
    normalized: dict[str, Any],
    campaign_context: _Stage4PriorCampaignContext,
    state: _Stage4PriorSubmissionState,
) -> Stage4StepResult | None:
    """Return the in-campaign progress update when barrier validation is deferred."""
    campaign = campaign_context.campaign
    if campaign is None or not campaign_context.in_active_campaign or state.repair_plan is not None:
        return None

    if not campaign_context.final_campaign_block:
        feedback = _build_repair_campaign_progress_feedback(
            active_block,
            campaign_context,
            plan,
        )
        return Stage4StepResult(
            stage_output=state.stage_output,
            validation_packet=build_validation_packet_for_block(
                block_id=active_block.id,
                status="repair_campaign_progress",
                feedback=feedback,
                validation=state.validation,
                changed_parameters=state.changed_parameters,
                retain_for_next_prompt=True,
                capture_stage_output=state.stage_output is not None,
            ),
            events=(
                _make_stage4_block_accepted_event(
                    active_block,
                    normalized,
                    stage_output=state.stage_output,
                ),
            ),
        )

    feedback = f"REPAIR CAMPAIGN READY FOR VALIDATION:\n- completed `{campaign.scope_key}`"
    return Stage4StepResult(
        stage_output=state.stage_output,
        validation_packet=build_validation_packet_for_block(
            block_id=active_block.id,
            status="repair_campaign_ready",
            feedback=feedback,
            validation=state.validation,
            changed_parameters=state.changed_parameters,
            retain_for_next_prompt=True,
            capture_stage_output=state.stage_output is not None,
        ),
        events=(
            _make_stage4_block_accepted_event(
                active_block,
                normalized,
                stage_output=state.stage_output,
            ),
        ),
    )


def _promote_multi_block_repair_feedback(
    *,
    active_block: Stage4FrontierBlock,
    state: _Stage4PriorSubmissionState,
    repair_decision: Stage4PriorRepairDecision,
) -> _Stage4PriorSubmissionState:
    """Rewrite packet feedback when a local failure widens into a repair campaign."""
    repair_plan = repair_decision.repair_plan
    if (
        repair_plan is None
        or not repair_decision.promote_campaign_feedback
        or not repair_plan.uses_repair_campaign
    ):
        return state

    next_block_id = next(
        (
            block_id
            for block_id in repair_plan.block_ids
            if block_id != repair_decision.accepted_block_id
        ),
        None,
    )
    next_block = next(
        (block for block in repair_plan.prompt_blocks if block.id == next_block_id),
        None,
    )
    campaign_feedback = _format_repair_campaign_feedback(
        repair_plan,
        accepted_block_id=repair_decision.accepted_block_id,
        next_block=next_block,
    )
    feedback = (
        render_stage4_validation_feedback(state.validation_packet) + "\n\n" + campaign_feedback
        if state.validation_packet.status == "partial_drift_failure"
        else campaign_feedback
    )
    return replace(
        state,
        validation_packet=build_validation_packet_for_block(
            block_id=active_block.id,
            status="repair_campaign_active",
            feedback=feedback,
            validation=state.validation,
            changed_parameters=state.changed_parameters,
            retain_for_next_prompt=True,
            capture_stage_output=False,
        ),
    )


def _build_prior_submission_events(
    *,
    active_block: Stage4FrontierBlock,
    normalized: dict[str, Any],
    state: _Stage4PriorSubmissionState,
    accepted_block_id: str | None,
) -> tuple[Stage4ReducerEvent, ...]:
    """Build reducer events for the finalized prior-submission outcome."""
    accepted_event = (
        None
        if accepted_block_id is None
        else _make_stage4_block_accepted_event(
            active_block,
            normalized,
            stage_output=state.stage_output,
        )
    )
    if state.repair_plan is not None:
        return (
            _make_stage4_repair_planned_event(
                state.repair_plan,
                accepted_block_event=accepted_event,
            ),
        )
    if accepted_event is not None:
        return (accepted_event,)
    return ()


def _apply_prior_submission(
    *,
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    active_block: Stage4FrontierBlock,
    normalized: dict[str, Any],
    deps: Stage4Deps,
) -> Stage4StepResult:
    """Apply one prior-authoring block and route failures back locally."""
    campaign_context = _build_prior_campaign_context(runtime, active_block)
    state = _ground_prior_submission(
        runtime=runtime,
        normalized=normalized,
        deps=deps,
        campaign_context=campaign_context,
    )
    state = _apply_prior_partial_drift_guard(
        plan=plan,
        runtime=runtime,
        active_block=active_block,
        deps=deps,
        state=state,
    )
    state = _classify_prior_submission_route(
        plan=plan,
        runtime=runtime,
        active_block=active_block,
        state=state,
    )
    campaign_result = _build_campaign_progress_result(
        plan=plan,
        active_block=active_block,
        normalized=normalized,
        campaign_context=campaign_context,
        state=state,
    )
    if campaign_result is not None:
        return campaign_result

    repair_decision = resolve_prior_repair_decision(
        active_block=active_block,
        repair_plan=state.repair_plan,
        campaign=campaign_context.campaign,
        stage_output_present=state.stage_output is not None,
    )
    state = _promote_multi_block_repair_feedback(
        active_block=active_block,
        state=state,
        repair_decision=repair_decision,
    )
    return Stage4StepResult(
        stage_output=state.stage_output,
        validation_packet=state.validation_packet,
        events=_build_prior_submission_events(
            active_block=active_block,
            normalized=normalized,
            state=state,
            accepted_block_id=repair_decision.accepted_block_id,
        ),
    )


def _apply_model_configuration_submission(
    *,
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    active_block: Stage4FrontierBlock,
    normalized: dict[str, Any],
    deps: Stage4Deps,
) -> Stage4StepResult:
    """Accept a model-configuration block submission."""
    del plan, runtime, deps
    return Stage4StepResult(
        validation_packet=build_validation_packet_for_block(
            block_id=active_block.id,
            status="accepted_pending_priors",
            feedback=_format_block_saved_feedback(active_block, None),
            retain_for_next_prompt=True,
            capture_stage_output=False,
        ),
        events=(_make_stage4_block_accepted_event(active_block, normalized),),
    )


def _apply_global_review_submission(
    *,
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    active_block: Stage4FrontierBlock,
    normalized: dict[str, Any],
    deps: Stage4Deps,
) -> Stage4StepResult:
    """Apply the compact global-review checkpoint."""
    del deps
    if normalized["decision"] == "approve":
        feedback = _format_block_saved_feedback(
            active_block,
            next_pending_block(plan.prior_blocks, runtime),
        )
        return Stage4StepResult(
            validation_packet=build_validation_packet_for_block(
                block_id=active_block.id,
                status="accepted",
                feedback=feedback,
                retain_for_next_prompt=True,
                capture_stage_output=False,
            ),
            events=(_make_stage4_block_accepted_event(active_block, normalized),),
        )
    reopen_block_ids = normalized["reopen_block_ids"]
    feedback = (
        "MODEL REVIEW REOPENED:\n"
        f"- reopening {summarize_stage4_names(list(reopen_block_ids))}\n"
        f"- reason: {normalized['reasoning']}"
    )
    return Stage4StepResult(
        validation_packet=build_validation_packet_for_block(
            block_id=active_block.id,
            status="model_review_reopened",
            feedback=feedback,
            retain_for_next_prompt=True,
            capture_stage_output=False,
        ),
        events=(
            _make_stage4_repair_planned_event(
                build_repair_plan(
                    plan,
                    ResolvedRepairScope(
                        scope_kind="global_review",
                        scope_rank=0,
                        scope_key=f"global_review:{'|'.join(reopen_block_ids)}",
                        reason=normalized["reasoning"],
                        failure_family=("global_review", active_block.id),
                        prompt_block_hints=reopen_block_ids,
                    ),
                    prompt_block_ids=reopen_block_ids,
                )
            ),
        ),
    )


_APPLY_STAGE4_SUBMISSION_BY_PAYLOAD_KIND = {
    "model_configuration_choice": _apply_model_configuration_submission,
    "indicator_choice": _apply_indicator_submission,
    "global_review_decision": _apply_global_review_submission,
    "prior_bundle": _apply_prior_submission,
}


def build_model_spec_from_decisions(
    decisions: Stage4DraftModel,
    skeleton: Stage4Skeleton,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Materialize a ModelSpec from accepted model-decision state."""
    decisions_data = {
        "initialization_policy": decisions.initialization_policy,
        "equilibrium_forcing": decisions.equilibrium_forcing,
        "distribution_choices": list(decisions.distribution_choices.values()),
    }
    spec, errors = validate_model_spec_decisions_dict(
        decisions_data,
        resolved_likelihoods=skeleton.resolved_likelihoods,
        ambiguous_indicators=skeleton.ambiguous_indicators,
        parameters=skeleton.all_params,
    )
    if spec is None:
        return None, errors
    return spec.model_dump(mode="json"), []


def _apply_submission_by_kind(
    *,
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    active_block: Stage4FrontierBlock,
    handler: Stage4BlockHandler,
    normalized: dict[str, Any],
    deps: Stage4Deps,
) -> Stage4StepResult:
    """Apply one normalized submission to the reducer."""
    applier = _APPLY_STAGE4_SUBMISSION_BY_PAYLOAD_KIND.get(handler.submission_payload_kind)
    if applier is None:
        raise ValueError(
            f"Unsupported Stage 4 submission payload kind {handler.submission_payload_kind!r}"
        )
    return applier(
        plan=plan,
        runtime=runtime,
        active_block=active_block,
        normalized=normalized,
        deps=deps,
    )


def settle_to_wait_state(
    *,
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    deps: Stage4Deps,
    model_lock_hint_block: Stage4FrontierBlock | None = None,
) -> tuple[dict[str, Any] | None, Stage4ValidationPacket | None, tuple[dict[str, Any], ...], bool]:
    """Run deterministic Stage 4 follow-on transitions to the next wait-state."""
    transitions: list[dict[str, Any]] = []
    latest_stage_output: dict[str, Any] | None = None
    latest_packet: Stage4ValidationPacket | None = None
    changed = False

    while True:
        if runtime.domain.done:
            return latest_stage_output, latest_packet, tuple(transitions), changed

        active_block = get_active_prompt_block(plan, runtime)
        if active_block is not None:
            return latest_stage_output, latest_packet, tuple(transitions), changed

        if _repair_barrier_pending(runtime):
            barrier_transitions = _finalize_repair_campaign_if_complete(plan, runtime, deps)
            if not barrier_transitions:
                raise ValueError("Stage 4 repair barrier could not settle to a wait-state")
            transitions.extend(barrier_transitions)
            latest_packet = runtime.interaction.last_validation_packet
            changed = True
            continue

        if _needs_model_spec_lock(plan, runtime):
            lock_result = _lock_stage4_model_spec(
                plan=plan,
                runtime=runtime,
                deps=deps,
                failed_block=model_lock_hint_block,
            )
            transitions.extend(_apply_stage4_step_result(plan, runtime, lock_result))
            latest_stage_output = lock_result.stage_output
            latest_packet = (
                runtime.interaction.last_validation_packet or lock_result.validation_packet
            )
            changed = True
            if lock_result.repair_plan is None:
                reconcile_locked_prior_surface(plan, runtime)
            model_lock_hint_block = None
            continue

        next_block = select_next_wait_block(plan, runtime)
        if next_block is not None:
            _set_block_cursor(runtime, next_block)
            changed = True
            return latest_stage_output, latest_packet, tuple(transitions), changed

        accepted_validation = runtime.domain.accepted.validation
        if accepted_validation is None or not accepted_validation.is_valid:
            raise ValueError(
                "Stage 4 cannot enter the terminal state without a valid accepted validation result"
            )
        _set_done_cursor(runtime)
        changed = True
        return latest_stage_output, latest_packet, tuple(transitions), changed


def compute_stage4_validate_step_with_transitions(
    data: dict[str, Any],
    *,
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    deps: Stage4Deps,
) -> tuple[dict | None, str, tuple[dict[str, Any], ...]]:
    """Advance the reducer by one block-local Stage 4 submit-tool call."""
    active_block = get_active_prompt_block(plan, runtime)
    if active_block is None:
        return None, "VALIDATION ERRORS:\n- no active Stage 4 frontier block remains", ()

    error_feedback = validate_stage4_submission_payload(data)
    if error_feedback is not None:
        runtime.interaction.last_validation_packet = build_validation_packet_for_block(
            block_id=active_block.id,
            status="validation_error",
            feedback=error_feedback,
            retain_for_next_prompt=True,
            capture_stage_output=False,
        )
        return None, error_feedback, ()

    handler = get_stage4_block_handler(active_block.kind)
    normalized, error_feedback = handler.normalize_submission(active_block, data)
    if error_feedback is not None:
        runtime.interaction.last_validation_packet = build_validation_packet_for_block(
            block_id=active_block.id,
            status="validation_error",
            feedback=error_feedback,
            retain_for_next_prompt=True,
            capture_stage_output=False,
        )
        return None, error_feedback, ()
    assert normalized is not None
    error_feedback = validate_stage4_block_coverage(active_block, runtime, normalized)
    if error_feedback is not None:
        runtime.interaction.last_validation_packet = build_validation_packet_for_block(
            block_id=active_block.id,
            status="validation_error",
            feedback=error_feedback,
            retain_for_next_prompt=True,
            capture_stage_output=False,
        )
        return None, error_feedback, ()

    result = _apply_submission_by_kind(
        plan=plan,
        runtime=runtime,
        active_block=active_block,
        handler=handler,
        normalized=normalized,
        deps=deps,
    )

    transitions: list[dict[str, Any]] = list(_apply_stage4_step_result(plan, runtime, result))
    settled_stage_output, settled_packet, settlement_transitions, settlement_changed = (
        settle_to_wait_state(
            plan=plan,
            runtime=runtime,
            deps=deps,
            model_lock_hint_block=active_block,
        )
    )
    del settlement_changed
    transitions.extend(settlement_transitions)

    latest_packet = (
        settled_packet or runtime.interaction.last_validation_packet or result.validation_packet
    )
    assert latest_packet is not None
    return (
        settled_stage_output if settled_stage_output is not None else result.stage_output,
        render_stage4_validation_feedback(latest_packet),
        tuple(transitions),
    )


def _lock_stage4_model_spec(
    *,
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    deps: Stage4Deps,
    failed_block: Stage4FrontierBlock | None,
) -> Stage4StepResult:
    """Materialize and validate the locked model spec after model decisions."""
    model_spec, errors = build_model_spec_from_decisions(runtime.domain.draft_model, deps.skeleton)
    if model_spec is None:
        failed_block_ids = _model_lock_failure_block_ids(
            plan,
            runtime.domain.draft_model,
            errors,
            failed_block,
        )
        if not failed_block_ids:
            raise ValueError(
                "Stage 4 could not materialize the initial ModelSpec: " + "; ".join(errors)
            )
        active_failure_block_id = failed_block_ids[0]
        feedback = "VALIDATION ERRORS:\n" + "\n".join(f"- {error}" for error in errors)
        return Stage4StepResult(
            validation_packet=build_validation_packet_for_block(
                block_id=active_failure_block_id,
                status="validation_error",
                feedback=feedback,
                retain_for_next_prompt=True,
                capture_stage_output=False,
            ),
            events=(
                _make_stage4_repair_planned_event(
                    build_repair_plan(
                        plan,
                        ResolvedRepairScope(
                            scope_kind="model_spec_lock",
                            scope_rank=0,
                            scope_key=f"model_spec_lock:{'+'.join(failed_block_ids)}",
                            reason=errors[0]
                            if errors
                            else "locked model_spec could not be materialized",
                            failure_family=("model_spec_lock", failed_block_ids),
                            prompt_block_hints=failed_block_ids,
                        ),
                        prompt_block_ids=failed_block_ids,
                    )
                ),
            ),
        )

    active_parameter_names = {
        str(parameter["name"])
        for parameter in model_spec.get("parameters") or []
        if isinstance(parameter, dict) and isinstance(parameter.get("name"), str)
    }
    current_state = runtime.domain.accepted.as_current()
    authored_priors = current_state.get("authored_priors")
    if isinstance(authored_priors, dict):
        filtered_authored_priors = {
            name: prior for name, prior in authored_priors.items() if name in active_parameter_names
        }
        if filtered_authored_priors:
            current_state["authored_priors"] = filtered_authored_priors
        else:
            current_state.pop("authored_priors", None)
    current_state.pop("resolved_priors", None)

    grounding_result = deps.grounding_fn(
        {"model_spec": model_spec},
        deps.causal_spec,
        current=current_state,
        data_for_model=deps.data_for_model,
        indicator_audits=deps.indicator_audits,
    )
    stage_output = grounding_result.stage_output
    feedback = grounding_result.feedback
    validation = stage_output.get("validation") if stage_output else None
    if failed_block is None and validation is not None and not validation.compile_ok:
        raise ValueError(
            f"Stage 4 could not lock the initial ModelSpec: {validation.compile_error}"
        )
    if failed_block is None and validation is None:
        raise ValueError("Stage 4 could not validate the initial ModelSpec")
    if (
        failed_block is None
        and validation is not None
        and validation.compile_ok
        and (not validation.pp_checked or validation.pp_valid)
    ):
        return Stage4StepResult(
            stage_output=stage_output,
            validation_packet=grounding_result.validation_packet,
        )
    validation_block = (
        failed_block
        or plan.review_block
        or plan.prior_review_block
        or next(iter(plan.all_blocks), None)
    )
    if validation_block is None:
        raise ValueError("Stage 4 could not route ModelSpec lock validation without any blocks")
    validation_route = classify_validation_outcome(
        plan,
        validation_block,
        validation,
        runtime,
        feedback=feedback,
        include_prior_predictive=False,
    )
    if validation_route.repair_plan is not None:
        if failed_block is None:
            raise ValueError(
                "Stage 4 could not lock the initial ModelSpec: " + feedback.replace("\n", " ")
            )
        return Stage4StepResult(
            stage_output=stage_output,
            validation_packet=grounding_result.validation_packet,
            events=(_make_stage4_repair_planned_event(validation_route.repair_plan),),
        )
    return Stage4StepResult(
        stage_output=stage_output,
        validation_packet=grounding_result.validation_packet,
        events=(
            ()
            if stage_output is None
            else (Stage4AcceptedStatePersistedEvent(stage_output=stage_output),)
        ),
    )


def _campaign_representative_block(
    plan: Stage4Plan,
    runtime: Stage4Runtime,
) -> Stage4FrontierBlock:
    """Return a deterministic representative block for an active repair campaign."""
    campaign = runtime.domain.repair_campaign
    if campaign is None or not campaign.scope_block_ids:
        raise ValueError("Repair campaign representative requested with no active campaign")
    block = plan.get_block(campaign.scope_block_ids[0])
    if block is None:
        raise ValueError(
            f"Unknown Stage 4 block id {campaign.scope_block_ids[0]!r} in repair campaign"
        )
    return block


def _parameter_names_for_blocks(
    blocks: tuple[Stage4FrontierBlock, ...],
) -> list[str]:
    """Return ordered semantic parameter names owned by a set of Stage 4 blocks."""
    parameter_names: list[str] = []
    seen: set[str] = set()
    for block in blocks:
        for parameter_name in block.parameter_names:
            if parameter_name in seen:
                continue
            seen.add(parameter_name)
            parameter_names.append(parameter_name)
    return parameter_names


def _finalize_repair_campaign_if_complete(
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    deps: Stage4Deps,
) -> tuple[dict[str, Any], ...]:
    """Validate a completed multi-block repair campaign at the campaign barrier."""
    from causal_ssm_agent.flows.stages.stage4.assembly import (
        format_validation_feedback,
        validate_assembly,
    )

    campaign = runtime.domain.repair_campaign
    if campaign is None or pending_repair_campaign_block_ids(campaign):
        return ()
    if runtime.domain.accepted.model_spec is None or not runtime.domain.accepted.authored_priors:
        return ()

    validation = validate_assembly(
        runtime.domain.accepted.model_spec,
        runtime.domain.accepted.authored_priors,
        deps.data_for_model,
        deps.indicator_audits,
        deps.causal_spec,
    )
    runtime.domain.accepted.validation = validation
    representative_block = _campaign_representative_block(plan, runtime)
    changed_params = _parameter_names_for_blocks(
        tuple(campaign.prompt_blocks_by_id[block_id] for block_id in campaign.scope_block_ids)
    )
    feedback = format_validation_feedback(
        validation,
        runtime.domain.accepted.authored_priors,
        changed_params=changed_params,
    )
    validation_route = classify_validation_outcome(
        plan,
        representative_block,
        validation,
        runtime,
        feedback=feedback,
    )

    if validation_route.outcome == "compile_error":
        assert validation_route.repair_plan is not None
        return _apply_stage4_step_result(
            plan,
            runtime,
            Stage4StepResult(
                validation_packet=build_validation_packet_for_block(
                    block_id=representative_block.id,
                    status="compile_error",
                    feedback=feedback,
                    validation=validation,
                    changed_parameters=tuple(changed_params),
                    retain_for_next_prompt=True,
                    capture_stage_output=False,
                ),
                events=(_make_stage4_repair_planned_event(validation_route.repair_plan),),
            ),
        )

    if validation_route.outcome == "prior_predictive_failure":
        assert validation_route.repair_plan is not None
        return _apply_stage4_step_result(
            plan,
            runtime,
            Stage4StepResult(
                validation_packet=build_validation_packet_for_block(
                    block_id=representative_block.id,
                    status="prior_predictive_failure",
                    feedback=feedback,
                    validation=validation,
                    changed_parameters=tuple(changed_params),
                    state_retained=True,
                    retain_for_next_prompt=True,
                    capture_stage_output=False,
                ),
                events=(_make_stage4_repair_planned_event(validation_route.repair_plan),),
            ),
        )

    if validation_route.outcome == "sensitivity_failure":
        assert validation_route.repair_plan is not None
        return _apply_stage4_step_result(
            plan,
            runtime,
            Stage4StepResult(
                validation_packet=build_validation_packet_for_block(
                    block_id=representative_block.id,
                    status="sensitivity_failure",
                    feedback=feedback,
                    validation=validation,
                    changed_parameters=tuple(changed_params),
                    state_retained=True,
                    retain_for_next_prompt=True,
                    capture_stage_output=False,
                ),
                events=(_make_stage4_repair_planned_event(validation_route.repair_plan),),
            ),
        )

    success_packet = build_validation_packet_for_block(
        block_id=representative_block.id,
        status="accepted",
        feedback=feedback,
        validation=validation,
        changed_parameters=tuple(changed_params),
        retain_for_next_prompt=feedback != "VALID",
        capture_stage_output=True,
    )
    return _apply_stage4_events(
        plan,
        runtime,
        (
            Stage4BarrierValidationPassedEvent(
                representative_block_id=representative_block.id,
                success_packet=success_packet,
            ),
        ),
    )
