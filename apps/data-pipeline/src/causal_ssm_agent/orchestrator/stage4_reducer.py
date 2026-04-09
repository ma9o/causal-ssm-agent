"""Stage 4 reducer logic and accepted-state transitions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from causal_ssm_agent.models.compilation_errors import AggregatedCompileError
from causal_ssm_agent.orchestrator.schemas_model import validate_model_spec_decisions_dict

from .stage4_events import (
    Stage4AcceptedStatePersistedEvent,
    Stage4BarrierValidationPassedEvent,
    Stage4BlockAcceptedEvent,
    Stage4ReducerEvent,
    Stage4RepairPlannedEvent,
)
from .stage4_feedback import (
    Stage4ValidationPacket,
    Stage4ValidationStatus,
    make_stage4_validation_packet,
    render_stage4_validation_feedback,
    should_store_stage4_validation_packet,
)
from .stage4_navigation import (
    _block_is_accepted,
    _next_pending_block,
    _pending_repair_campaign_block_ids,
    apply_stage4_barrier_validation_success,
    apply_stage4_block_acceptance,
    apply_stage4_repair_plan,
    get_active_plan_block,
    get_active_prompt_block,
)
from .stage4_repair import (
    ResolvedRepairPlan,
    ResolvedRepairScope,
    Stage4PriorRepairDecision,
    build_repair_plan,
    classify_validation_outcome,
    resolve_prior_repair_decision,
)
from .stage4_state import (
    Stage4DecisionState,
    Stage4ModelSpecLockPendingCursor,
    Stage4RepairBarrierCursor,
    Stage4RepairCampaignState,
    Stage4Runtime,
)
from .stage4_submission import get_stage4_block_handler, validate_stage4_submission_payload

if TYPE_CHECKING:
    from causal_ssm_agent.flows.stages.stage4.assembly import AssemblyValidation

    from .stage4_orchestrator import Stage4FrontierBlock, Stage4Plan, Stage4Skeleton
    from .stage4_types import Stage4Deps

_RECOVERABLE_STAGE4_REDUCER_ERRORS = (
    AggregatedCompileError,
    AttributeError,
    KeyError,
    RuntimeError,
    TypeError,
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
    def feedback(self) -> str:
        """Return model-facing feedback from the authoritative validation packet."""
        return render_stage4_validation_feedback(self.validation_packet)

    @property
    def accepted_block_id(self) -> str | None:
        """Return the primary accepted block, if this update accepts one."""
        for event in self.events:
            if isinstance(event, Stage4BlockAcceptedEvent):
                return event.block_id
            if isinstance(event, Stage4RepairPlannedEvent):
                return event.accepted_block_id
        return None

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

    @property
    def feedback(self) -> str:
        """Return authoritative model-facing feedback for the current state."""
        return render_stage4_validation_feedback(self.validation_packet)


def _summarize_names(names: list[str], *, limit: int = 8) -> str:
    """Render a compact preview of names."""
    if not names:
        return "(none)"
    preview = ", ".join(f"`{name}`" for name in names[:limit])
    if len(names) <= limit:
        return preview
    return f"{preview}, ... (+{len(names) - limit} more)"


def _build_validation_packet_for_block(
    *,
    block: Stage4FrontierBlock | None,
    status: Stage4ValidationStatus,
    feedback: str,
    validation: AssemblyValidation | None = None,
    changed_parameters: tuple[str, ...] = (),
    state_retained: bool = False,
    retain_for_next_prompt: bool = True,
    capture_stage_output: bool = False,
) -> Stage4ValidationPacket:
    """Build the typed validation packet owned by the reducer."""
    return make_stage4_validation_packet(
        status=status,
        feedback=feedback,
        validation=validation,
        active_scope_id=None if block is None else block.id,
        changed_parameters=changed_parameters,
        state_retained=state_retained,
        retain_for_next_prompt=retain_for_next_prompt,
        capture_stage_output=capture_stage_output,
    )


def _format_repair_campaign_feedback(
    repair_plan: ResolvedRepairPlan,
    *,
    accepted_block_id: str | None,
    next_block: Stage4FrontierBlock | None,
) -> str:
    """Render bounded repair-campaign progress for the LLM."""
    lines = [
        "REPAIR CAMPAIGN ACTIVE:",
        f"- scope: `{repair_plan.scope_key}`",
        f"- reason: {repair_plan.reason}",
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


def _persist_stage4_stage_output(
    runtime: Stage4Runtime,
    stage_output: dict[str, Any] | None,
) -> None:
    """Merge accepted Stage 4 output into reducer-owned state."""
    runtime.accepted.apply_stage_output(stage_output)


def _serialize_stage4_transition_priors(
    block: Stage4FrontierBlock,
    priors: dict[str, Any],
) -> list[dict[str, Any]]:
    """Serialize one block's accepted priors for transition events."""
    serialized: list[dict[str, Any]] = []
    for parameter_name in block.parameter_names:
        prior = priors.get(parameter_name)
        if not isinstance(prior, dict):
            continue
        item: dict[str, Any] = {"parameter": parameter_name}
        for key in ("distribution", "params", "reasoning"):
            value = prior.get(key)
            if value is not None:
                item[key] = value
        serialized.append(item)
    return serialized


def _make_stage4_accepted_transition(
    block: Stage4FrontierBlock,
    normalized: dict[str, Any],
) -> dict[str, Any] | None:
    """Build the accepted transition payload for one Stage 4 block."""
    if block.kind == "indicator_decision":
        choice = normalized.get("distribution_choice")
        if not isinstance(choice, dict):
            return None
        return {
            "block_id": block.id,
            "status": "accepted",
            "detail_kind": "indicator_choice",
            "variable": choice.get("variable"),
            "distribution": choice.get("distribution"),
            "link": choice.get("link"),
            "reasoning": choice.get("reasoning"),
        }

    if block.kind == "global_review":
        if normalized.get("decision") != "approve":
            return None
        return {
            "block_id": block.id,
            "status": "accepted",
            "detail_kind": "review_approval",
            "reasoning": normalized.get("reasoning"),
        }

    priors = normalized.get("priors")
    if block.kind in {
        "measurement_prior",
        "observation_prior",
        "dynamics_prior",
        "effect_prior",
        "correlation_prior",
        "global_prior_review",
    } and isinstance(priors, dict):
        return {
            "block_id": block.id,
            "status": "accepted",
            "detail_kind": "prior_bundle",
            "parameter_names": list(block.parameter_names),
            "priors": _serialize_stage4_transition_priors(block, priors),
        }

    return None


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
            "reason": repair_plan.reason,
            "scope_kind": repair_plan.scope_kind,
        }
        for block_id in repair_plan.block_ids
        if block_id != accepted_block_id
    )


def _all_model_blocks_accepted(plan: Stage4Plan, runtime: Stage4Runtime) -> bool:
    """Whether every model-decision block is accepted in runtime state."""
    return all(_block_is_accepted(runtime, block.id) for block in plan.model_blocks)


def _make_stage4_block_accepted_event(
    block: Stage4FrontierBlock,
    normalized: dict[str, Any],
    *,
    stage_output: dict[str, Any] | None = None,
) -> Stage4BlockAcceptedEvent:
    """Build the explicit reducer event for one accepted block."""
    distribution_choice = normalized.get("distribution_choice")
    if not isinstance(distribution_choice, dict):
        distribution_choice = None
    return Stage4BlockAcceptedEvent(
        block_id=block.id,
        transition_payload=_make_stage4_accepted_transition(block, normalized),
        distribution_choice=distribution_choice,
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
        stage_output=None if accepted_block_event is None else accepted_block_event.stage_output,
    )


def _apply_stage4_event(
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    event: Stage4ReducerEvent,
) -> tuple[dict[str, Any], ...]:
    """Apply one typed Stage 4 reducer event."""
    if isinstance(event, Stage4AcceptedStatePersistedEvent):
        _persist_stage4_stage_output(runtime, event.stage_output)
        return ()

    if isinstance(event, Stage4BlockAcceptedEvent):
        if event.distribution_choice is not None:
            runtime.decisions.distribution_choices[event.distribution_choice["variable"]] = (
                event.distribution_choice
            )
        if event.stage_output is not None:
            _persist_stage4_stage_output(runtime, event.stage_output)
        runtime.block_status[event.block_id] = "accepted"
        transitions = () if event.transition_payload is None else (event.transition_payload,)
        apply_stage4_block_acceptance(plan, runtime, event.block_id)
        return transitions

    if isinstance(event, Stage4RepairPlannedEvent):
        if event.distribution_choice is not None:
            runtime.decisions.distribution_choices[event.distribution_choice["variable"]] = (
                event.distribution_choice
            )
        if event.stage_output is not None:
            _persist_stage4_stage_output(runtime, event.stage_output)
        if event.accepted_block_id is not None:
            runtime.block_status[event.accepted_block_id] = "accepted"
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
            packet = _build_validation_packet_for_block(
                block=representative_block,
                status="accepted",
                feedback=_format_block_saved_feedback(
                    representative_block,
                    get_active_plan_block(plan, runtime),
                ),
                retain_for_next_prompt=True,
                capture_stage_output=False,
            )
        runtime.last_validation_packet = (
            packet if should_store_stage4_validation_packet(packet) else None
        )
        return ()

    raise TypeError(f"Unsupported Stage 4 reducer event {event!r}")


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
    runtime.last_validation_packet = (
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
        _next_pending_block(plan.model_blocks, runtime, skip_id=active_block.id),
    )
    return Stage4StepResult(
        validation_packet=_build_validation_packet_for_block(
            block=active_block,
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
    campaign = runtime.repair_campaign
    pending_block_ids = () if campaign is None else _pending_repair_campaign_block_ids(campaign)
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
        current=runtime.accepted.as_current(),
        data_for_model=deps.data_for_model,
        indicator_audits=deps.indicator_audits,
        skip_ppc=bool(
            campaign_context.campaign is not None
            and campaign_context.campaign.requires_barrier_validation
        ),
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
        and getattr(state.validation, "compile_ok", True)
        and not getattr(state.validation, "pp_checked", False)
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
        _classify_compile_failure_route,
        _classify_prior_failure_blocks,
    )

    if not _should_run_partial_drift_guard(active_block=active_block, state=state):
        return state

    stage_output = state.stage_output or {}
    authored_priors = stage_output.get("authored_priors")
    try:
        if active_block.kind == "dynamics_prior":
            partial_guard = validate_dynamics_block_partial_drift(
                model_spec=runtime.accepted.model_spec,
                authored_priors=authored_priors,
                causal_spec=deps.causal_spec,
                active_construct_names=active_block.construct_names,
                active_parameter_names=active_block.parameter_names,
            )
        else:
            partial_guard = validate_effect_block_partial_drift(
                model_spec=runtime.accepted.model_spec,
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
            validation_packet=_build_validation_packet_for_block(
                block=active_block,
                status="compile_error",
                feedback=feedback,
                changed_parameters=state.changed_parameters,
                retain_for_next_prompt=True,
                capture_stage_output=False,
            ),
            repair_plan=_classify_compile_failure_route(plan, active_block, str(exc)),
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
        validation_packet=_build_validation_packet_for_block(
            block=active_block,
            status="partial_drift_failure",
            feedback=partial_feedback,
            validation=validation,
            changed_parameters=state.changed_parameters,
            retain_for_next_prompt=True,
            capture_stage_output=False,
        ),
        repair_plan=_classify_prior_failure_blocks(
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
        feedback=state.feedback,
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
    if (
        campaign is None
        or not campaign_context.in_active_campaign
        or not campaign.requires_barrier_validation
        or state.repair_plan is not None
    ):
        return None

    if not campaign_context.final_campaign_block:
        feedback = _build_repair_campaign_progress_feedback(
            active_block,
            campaign_context,
            plan,
        )
        return Stage4StepResult(
            stage_output=state.stage_output,
            validation_packet=_build_validation_packet_for_block(
                block=active_block,
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
        validation_packet=_build_validation_packet_for_block(
            block=active_block,
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
    if repair_plan is None or not repair_decision.promote_campaign_feedback:
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
        state.feedback + "\n\n" + campaign_feedback
        if state.validation_packet.status == "partial_drift_failure"
        else campaign_feedback
    )
    return replace(
        state,
        validation_packet=_build_validation_packet_for_block(
            block=active_block,
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
            _next_pending_block(plan.prior_blocks, runtime),
        )
        return Stage4StepResult(
            validation_packet=_build_validation_packet_for_block(
                block=active_block,
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
        f"- reopening {_summarize_names(list(reopen_block_ids))}\n"
        f"- reason: {normalized['reasoning']}"
    )
    return Stage4StepResult(
        validation_packet=_build_validation_packet_for_block(
            block=active_block,
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
                    requires_barrier_validation=False,
                )
            ),
        ),
    )


def _build_model_spec_from_decisions(
    decisions: Stage4DecisionState,
    skeleton: Stage4Skeleton,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Materialize a ModelSpec from accepted model-decision state."""
    decisions_data = {
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
    normalized: dict[str, Any],
    deps: Stage4Deps,
) -> Stage4StepResult:
    """Apply one normalized submission to the reducer."""
    if active_block.kind == "indicator_decision":
        return _apply_indicator_submission(
            plan=plan,
            runtime=runtime,
            active_block=active_block,
            normalized=normalized,
            deps=deps,
        )
    if active_block.kind == "global_review":
        return _apply_global_review_submission(
            plan=plan,
            runtime=runtime,
            active_block=active_block,
            normalized=normalized,
            deps=deps,
        )
    return _apply_prior_submission(
        plan=plan,
        runtime=runtime,
        active_block=active_block,
        normalized=normalized,
        deps=deps,
    )


def _compute_stage4_validate_step_with_transitions(
    data: dict[str, Any],
    *,
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    deps: Stage4Deps,
) -> tuple[dict | None, str, tuple[dict[str, Any], ...]]:
    """Advance the reducer by one block-local Stage 4 submit-tool call."""
    active_block = get_active_prompt_block(plan, runtime)
    if active_block is None:
        if isinstance(
            runtime.cursor,
            (Stage4ModelSpecLockPendingCursor, Stage4RepairBarrierCursor),
        ):
            return (
                None,
                "VALIDATION ERRORS:\n"
                f"- Stage 4 is in an internal transition: {runtime.cursor.reason}",
                (),
            )
        return None, "VALIDATION ERRORS:\n- no active Stage 4 frontier block remains", ()

    error_feedback = validate_stage4_submission_payload(data)
    if error_feedback is not None:
        runtime.last_validation_packet = _build_validation_packet_for_block(
            block=active_block,
            status="validation_error",
            feedback=error_feedback,
            retain_for_next_prompt=True,
            capture_stage_output=False,
        )
        return None, error_feedback, ()

    handler = get_stage4_block_handler(active_block.kind)
    normalized, error_feedback = handler.normalize_submission(active_block, data)
    if error_feedback is not None:
        runtime.last_validation_packet = _build_validation_packet_for_block(
            block=active_block,
            status="validation_error",
            feedback=error_feedback,
            retain_for_next_prompt=True,
            capture_stage_output=False,
        )
        return None, error_feedback, ()
    assert normalized is not None

    result = _apply_submission_by_kind(
        plan=plan,
        runtime=runtime,
        active_block=active_block,
        normalized=normalized,
        deps=deps,
    )

    transitions: list[dict[str, Any]] = list(_apply_stage4_step_result(plan, runtime, result))
    transitions.extend(
        _finalize_repair_campaign_if_complete(
            plan,
            runtime,
            deps,
        )
    )

    if (
        active_block.kind == "indicator_decision"
        and result.accepted_block_id == active_block.id
        and result.repair_plan is None
        and _all_model_blocks_accepted(plan, runtime)
    ):
        lock_result = _lock_stage4_model_spec(
            plan=plan,
            runtime=runtime,
            deps=deps,
            failed_block=active_block,
        )
        transitions.extend(_apply_stage4_step_result(plan, runtime, lock_result))
        if lock_result.repair_plan is None:
            from .stage4_navigation import _activate_review_phase

            _activate_review_phase(plan, runtime)
        return (
            lock_result.stage_output,
            render_stage4_validation_feedback(lock_result.validation_packet),
            tuple(transitions),
        )

    latest_packet = runtime.last_validation_packet or result.validation_packet
    assert latest_packet is not None
    return (
        result.stage_output,
        render_stage4_validation_feedback(latest_packet),
        tuple(transitions),
    )


def compute_stage4_validate_step(
    data: dict[str, Any],
    *,
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    deps: Stage4Deps,
) -> tuple[dict | None, str]:
    """Advance the reducer by one block-local Stage 4 submit-tool call."""
    stage_output, feedback, _transitions = _compute_stage4_validate_step_with_transitions(
        data,
        plan=plan,
        runtime=runtime,
        deps=deps,
    )
    return stage_output, feedback


def _lock_stage4_model_spec(
    *,
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    deps: Stage4Deps,
    failed_block: Stage4FrontierBlock,
) -> Stage4StepResult:
    """Materialize and validate the locked model spec after model decisions."""
    model_spec, errors = _build_model_spec_from_decisions(runtime.decisions, deps.skeleton)
    if model_spec is None:
        feedback = "VALIDATION ERRORS:\n" + "\n".join(f"- {error}" for error in errors)
        return Stage4StepResult(
            validation_packet=_build_validation_packet_for_block(
                block=failed_block,
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
                            scope_key=f"model_spec_lock:{failed_block.id}",
                            reason="locked model_spec could not be materialized",
                            failure_family=("model_spec_lock", failed_block.id),
                            prompt_block_hints=(failed_block.id,),
                        ),
                        prompt_block_ids=(failed_block.id,),
                        requires_barrier_validation=False,
                    )
                ),
            ),
        )

    grounding_result = deps.grounding_fn(
        {"model_spec": model_spec},
        deps.causal_spec,
        current=runtime.accepted.as_current(),
        data_for_model=deps.data_for_model,
        indicator_audits=deps.indicator_audits,
    )
    stage_output = grounding_result.stage_output
    feedback = grounding_result.feedback
    validation = stage_output.get("validation") if stage_output else None
    validation_route = classify_validation_outcome(
        plan,
        failed_block,
        validation,
        runtime,
        feedback=feedback,
        include_prior_predictive=False,
    )
    if validation_route.repair_plan is not None:
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
    campaign = runtime.repair_campaign
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

    campaign = runtime.repair_campaign
    if (
        campaign is None
        or not campaign.requires_barrier_validation
        or _pending_repair_campaign_block_ids(campaign)
    ):
        return ()
    if runtime.accepted.model_spec is None or not runtime.accepted.authored_priors:
        return ()

    validation = validate_assembly(
        runtime.accepted.model_spec,
        runtime.accepted.authored_priors,
        deps.data_for_model,
        deps.indicator_audits,
        deps.causal_spec,
    )
    runtime.accepted.validation = validation
    representative_block = _campaign_representative_block(plan, runtime)
    changed_params = _parameter_names_for_blocks(
        tuple(campaign.prompt_blocks_by_id[block_id] for block_id in campaign.scope_block_ids)
    )
    feedback = format_validation_feedback(
        validation,
        runtime.accepted.authored_priors,
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
                validation_packet=_build_validation_packet_for_block(
                    block=representative_block,
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
                validation_packet=_build_validation_packet_for_block(
                    block=representative_block,
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

    success_packet = _build_validation_packet_for_block(
        block=representative_block,
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
