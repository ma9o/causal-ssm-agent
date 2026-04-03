"""Stage 4: Model Specification & Prior Elicitation (Agentic).

Frontier-reduced multi-turn orchestration for Stage 4. The LLM only sees one
active decision block at a time while deterministic reducer state preserves
accepted decisions and routes retries back to the smallest current frontier.

Follows the same two-layer architecture as stages 1a/1b:
- This module contains pure orchestrator logic (framework-agnostic).
- The Prefect wrapper lives in ``flows/stages/stage4/flow.py``.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from causal_ssm_agent.orchestrator.schemas_model import (
    DistributionChoice,
    validate_model_spec_decisions_dict,
)

from .prompts.model_proposal import (
    build_stage4_system_prompt,
    build_stage4_user_prompt,
)
from .stage4_events import (
    Stage4AcceptedStatePersistedEvent,
    Stage4BarrierValidationPassedEvent,
    Stage4BlockAcceptedEvent,
    Stage4ReducerEvent,
    Stage4RepairPlannedEvent,
)
from .stage4_feedback import (
    Stage4GroundingResult,
    Stage4ScopeSnapshot,
    Stage4ValidationPacket,
    Stage4ValidationStatus,
    default_stage4_validation_packet,
    make_stage4_validation_packet,
    render_stage4_validation_feedback,
    should_store_stage4_validation_packet,
)
from .stage4_navigation import (
    _activate_prior_phase,
    _activate_review_phase,
    _block_is_accepted,
    _next_pending_block,
    _pending_repair_campaign_block_ids,
    _set_block_cursor,
    _set_done_cursor,
    _set_model_spec_lock_cursor,
    _set_repair_barrier_cursor,
    apply_stage4_barrier_validation_success,
    apply_stage4_block_acceptance,
    apply_stage4_repair_plan,
    get_active_plan_block,
    get_active_prompt_block,
    get_stage4_phase,
    make_stage4_runtime,
    project_stage4_graph,
    project_stage4_initial_state,
    project_stage4_snapshot,
)
from .stage4_orchestrator import (
    Stage4FrontierBlock,
    Stage4Plan,
    Stage4PromptScopePolicy,
    Stage4Skeleton,
    build_construct_scale_cards,
    build_distribution_cards,
    build_model_topology,
    build_prior_cards,
    build_stage4_plan,
    derive_deterministic_spec,
    get_stage4_prompt_scope_policy,
)
from .stage4_partial_drift import (
    build_effect_row_budget,
    validate_dynamics_block_partial_drift,
    validate_effect_block_partial_drift,
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
    Stage4AcceptedState,
    Stage4DecisionState,
    Stage4DoneCursor,
    Stage4ModelSpecLockPendingCursor,
    Stage4RepairBarrierCursor,
    Stage4RepairCampaignState,
    Stage4Runtime,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import polars as pl

    from causal_ssm_agent.flows.stages.stage4.assembly import AssemblyValidation
    from causal_ssm_agent.utils.llm import GenerateFn


_STAGE4_FRONTIER_PREFIX = "ACTIVE FRONTIER (machine-generated)"
_STAGE4_NAVIGATION_EXPORTS = (
    _activate_prior_phase,
    _set_block_cursor,
    _set_done_cursor,
    _set_model_spec_lock_cursor,
    _set_repair_barrier_cursor,
    make_stage4_runtime,
    project_stage4_graph,
    project_stage4_initial_state,
    project_stage4_snapshot,
)
_STAGE4_PLAN_EXPORTS = (
    build_construct_scale_cards,
    build_distribution_cards,
    build_model_topology,
    build_prior_cards,
    build_stage4_plan,
    derive_deterministic_spec,
)


@dataclass
class Stage4Result:
    """Result of the agentic Stage 4 flow."""

    model_spec: dict[str, Any]
    authored_priors: dict[str, dict]
    search_queries: dict[str, str] = field(default_factory=dict)
    validation: AssemblyValidation | None = None


@dataclass(frozen=True)
class Stage4Deps:
    """Static Stage 4 runtime dependencies shared across reducer steps."""

    skeleton: Stage4Skeleton
    causal_spec: dict[str, Any]
    data_for_model: pl.DataFrame
    indicator_audits: dict[str, dict[str, Any]]
    grounding_fn: Callable[..., Stage4GroundingResult]


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


@dataclass(frozen=True)
class Stage4Turn:
    """Structured current-turn projection for the active Stage 4 block."""

    block: Stage4FrontierBlock
    messages: list[dict[str, Any]]
    allowed_tool_names: tuple[str, ...]
    latest_feedback: str
    phase: str


@dataclass(frozen=True)
class Stage4TurnOutcome:
    """Structured outcome for one Stage 4 model turn."""

    block_id: str
    validate_submitted: bool
    submit_count: int
    latest_feedback: str | None
    next_block_id: str | None


@dataclass
class _Stage4TurnTracker:
    """Mutable tracker for explicit tool submissions inside one model turn."""

    block_id: str
    submit_count: int = 0
    latest_feedback: str | None = None
    next_block_id: str | None = None


@dataclass(frozen=True)
class Stage4BlockHandler:
    """Per-kind Stage 4 block behavior."""

    kind: str
    prompt_policy: Stage4PromptScopePolicy
    normalize_submission: Callable[
        [Stage4FrontierBlock, dict[str, Any]],
        tuple[dict[str, Any] | None, str | None],
    ]
    apply_submission: Callable[
        [Stage4Plan, Stage4Runtime, Stage4FrontierBlock, dict[str, Any], Stage4Deps],
        Stage4StepResult,
    ]
    include_prior_source_guidance: bool = False

    def allowed_tool_names(
        self,
        *,
        enable_literature: bool,
        enable_paraphrasing: bool,
    ) -> tuple[str, ...]:
        """Return runtime-enabled tools for this block kind."""
        return _enabled_block_tool_names(
            self.prompt_policy,
            enable_literature=enable_literature,
            enable_paraphrasing=enable_paraphrasing,
        )

    def render_turn(
        self,
        *,
        prompt_context: Stage4Messages,
        plan: Stage4Plan,
        runtime: Stage4Runtime,
        block: Stage4FrontierBlock,
    ) -> Stage4Turn:
        """Render the current model-facing turn for this block."""
        allowed_tool_names = self.allowed_tool_names(
            enable_literature=prompt_context.enable_literature,
            enable_paraphrasing=prompt_context.enable_paraphrasing,
        )
        latest_feedback = render_stage4_validation_feedback(runtime.last_validation_packet)
        messages = prompt_context.messages_for_block(
            block=block,
            plan=plan,
            runtime=runtime,
            handler=self,
        )
        return Stage4Turn(
            block=block,
            messages=messages,
            allowed_tool_names=allowed_tool_names,
            latest_feedback=latest_feedback,
            phase=get_stage4_phase(runtime, plan=plan),
        )


@dataclass
class Stage4Messages:
    """Prompt-local context used to render a single active Stage 4 block."""

    question: str
    causal_spec: dict[str, Any] | None = None
    model_topology: dict[str, Any] = field(default_factory=dict)
    distribution_cards: list[dict[str, Any]] = field(default_factory=list)
    loading_params: list[dict] = field(default_factory=list)
    construct_scale_cards: list[dict[str, Any]] = field(default_factory=list)
    prior_cards: list[dict[str, Any]] = field(default_factory=list)
    enable_literature: bool = False
    enable_paraphrasing: bool = False

    def _likelihood_lookup(self, runtime: Stage4Runtime) -> dict[str, dict[str, Any]]:
        """Return the current likelihood choice per indicator."""
        lookup: dict[str, dict[str, Any]] = {}
        for likelihood in (runtime.accepted.model_spec or {}).get("likelihoods") or []:
            if not isinstance(likelihood, dict):
                continue
            variable = likelihood.get("variable")
            if isinstance(variable, str):
                lookup[variable] = likelihood
        for variable, choice in runtime.decisions.distribution_choices.items():
            lookup[variable] = choice
        return lookup

    def _distribution_cards_for_runtime(
        self,
        runtime: Stage4Runtime,
    ) -> list[dict[str, Any]]:
        """Return stateful distribution cards for the current runtime."""
        cards = deepcopy(self.distribution_cards)
        likelihood_lookup = self._likelihood_lookup(runtime)
        for card in cards:
            choice = likelihood_lookup.get(card.get("variable"))
            if choice is None:
                continue
            card["selected_distribution"] = choice.get("distribution")
            card["selected_link"] = choice.get("link")
        return cards

    def _construct_scale_cards_for_runtime(
        self,
        runtime: Stage4Runtime,
    ) -> list[dict[str, Any]]:
        """Return construct cards enriched with accepted likelihood choices."""
        cards = deepcopy(self.construct_scale_cards)
        likelihood_lookup = self._likelihood_lookup(runtime)
        for card in cards:
            indicators = card.get("indicators") or []
            for indicator in indicators:
                choice = likelihood_lookup.get(indicator.get("indicator"))
                if choice is None:
                    continue
                indicator["selected_distribution"] = choice.get("distribution")
                indicator["selected_link"] = choice.get("link")
        return cards

    def _prior_cards_for_runtime(
        self,
        runtime: Stage4Runtime,
    ) -> list[dict[str, Any]]:
        """Return prior cards for the current runtime."""
        cards = deepcopy(self.prior_cards)
        accepted_priors = runtime.accepted.authored_priors
        for card in cards:
            parameter_name = card.get("parameter")
            if not isinstance(parameter_name, str):
                continue
            accepted_prior = accepted_priors.get(parameter_name)
            if accepted_prior is not None:
                card["accepted_prior"] = deepcopy(accepted_prior)
        return cards

    def _current_validation_packet(self, runtime: Stage4Runtime) -> Stage4ValidationPacket:
        """Return the latest typed validation state for prompt rendering."""
        return runtime.last_validation_packet or default_stage4_validation_packet()

    def _scope_snapshot_for_block(
        self,
        *,
        block: Stage4FrontierBlock,
        plan: Stage4Plan,
        runtime: Stage4Runtime,
        policy: Stage4PromptScopePolicy,
        submission_example: str,
        include_prior_source_guidance: bool,
    ) -> Stage4ScopeSnapshot:
        """Build the typed LLM-visible snapshot for one active Stage 4 block."""
        distribution_cards = self._distribution_cards_for_runtime(runtime)
        loading_params = deepcopy(self.loading_params)
        construct_scale_cards = self._construct_scale_cards_for_runtime(runtime)
        prior_cards = self._prior_cards_for_runtime(runtime)
        visible_distribution_cards = _visible_block_section(
            policy,
            "distribution_cards",
            _filter_cards(distribution_cards, "variable", block.variable_names),
        )
        visible_loading_params = _visible_block_section(
            policy,
            "loading_params",
            _filter_cards(loading_params, "name", block.parameter_names),
        )
        visible_construct_scale_cards = _visible_block_section(
            policy,
            "construct_scale_cards",
            _filter_cards(construct_scale_cards, "construct", block.construct_names),
        )
        visible_prior_cards = _visible_block_section(
            policy,
            "prior_cards",
            _filter_cards(prior_cards, "parameter", block.parameter_names),
        )
        active_parameter_names = _active_prior_parameter_names(runtime)
        if active_parameter_names is not None:
            visible_prior_cards = [
                card
                for card in visible_prior_cards
                if str(card.get("parameter") or "") in active_parameter_names
            ]
        latest_validation = self._current_validation_packet(runtime)
        local_parameter_names = set(block.parameter_names)
        coupled_parameter_names = tuple(
            parameter_name
            for parameter_name in latest_validation.coupled_parameters
            if parameter_name not in local_parameter_names
        )
        coupled_prior_cards = [
            card
            for card in prior_cards
            if card.get("parameter") in coupled_parameter_names and card.get("accepted_prior") is not None
        ]
        visible_parameter_names = tuple(
            card["parameter"]
            for card in visible_prior_cards
            if isinstance(card.get("parameter"), str)
        )
        return Stage4ScopeSnapshot(
            block_id=block.id,
            block_kind=block.kind,
            block_label=block.label,
            block_instructions=policy.user_task,
            frontier_status=_format_plan_status(
                plan,
                runtime,
                block,
                causal_spec=self.causal_spec,
            ),
            model_topology=_filter_model_topology(self.model_topology, block),
            distribution_cards=visible_distribution_cards,
            loading_params=visible_loading_params,
            construct_scale_cards=visible_construct_scale_cards,
            prior_cards=visible_prior_cards,
            coupled_prior_cards=coupled_prior_cards,
            submission_example=_format_submission_example(
                block,
                prior_cards=visible_prior_cards,
                fallback_submission_example=submission_example,
            ),
            include_prior_source_guidance=include_prior_source_guidance,
            latest_validation=latest_validation,
            editable_parameter_names=visible_parameter_names,
            visible_parameter_names=visible_parameter_names,
            coupled_parameter_names=coupled_parameter_names,
        )

    def _messages_for_scope(
        self,
        block: Stage4FrontierBlock,
        plan: Stage4Plan,
        runtime: Stage4Runtime,
        *,
        policy: Stage4PromptScopePolicy,
        enabled_tool_names: tuple[str, ...],
        submission_example: str,
        include_prior_source_guidance: bool,
    ) -> list[dict]:
        """Build the model-facing prompt for one active Stage 4 scope."""
        snapshot = self._scope_snapshot_for_block(
            block=block,
            plan=plan,
            runtime=runtime,
            policy=policy,
            submission_example=submission_example,
            include_prior_source_guidance=include_prior_source_guidance,
        )
        return [
            {
                "role": "system",
                "content": build_stage4_system_prompt(
                    system_task=policy.system_task,
                    guidance_section_keys=policy.guidance_section_keys,
                    parameter_guidance_prefixes=policy.parameter_guidance_prefixes,
                    enabled_tool_names=enabled_tool_names,
                ),
            },
            {
                "role": "user",
                "content": build_stage4_user_prompt(
                    question=self.question,
                    snapshot=snapshot,
                ),
            },
        ]

    def messages_for_block(
        self,
        block: Stage4FrontierBlock,
        plan: Stage4Plan,
        runtime: Stage4Runtime,
        handler: Stage4BlockHandler,
    ) -> list[dict]:
        """Build the model-facing prompt for one authored Stage 4 block."""
        enabled_tool_names = handler.allowed_tool_names(
            enable_literature=self.enable_literature,
            enable_paraphrasing=self.enable_paraphrasing,
        )
        return self._messages_for_scope(
            block,
            plan,
            runtime,
            policy=handler.prompt_policy,
            enabled_tool_names=enabled_tool_names,
            submission_example="",
            include_prior_source_guidance=handler.include_prior_source_guidance,
        )


@dataclass
class Stage4Session:
    """Single owner of the current Stage 4 turn and accepted state."""

    plan: Stage4Plan
    prompt_context: Stage4Messages
    deps: Stage4Deps
    runtime: Stage4Runtime = field(default_factory=Stage4Runtime)
    persist_runtime: Callable[[Stage4Runtime, tuple[dict[str, Any], ...]], None] | None = None
    _turn_tracker: _Stage4TurnTracker | None = field(default=None, init=False, repr=False)

    @property
    def accepted(self) -> Stage4AcceptedState:
        return self.runtime.accepted

    @property
    def search_cache(self) -> dict[str, str]:
        return self.runtime.search_cache

    @property
    def search_queries(self) -> dict[str, str]:
        return self.runtime.search_queries

    def current_block(self) -> Stage4FrontierBlock | None:
        """Return the active reducer block, if any."""
        return get_active_prompt_block(self.plan, self.runtime)

    def current_turn(self) -> Stage4Turn | None:
        """Return the active prompt/tool turn, if any."""
        block = self.current_block()
        if block is None:
            return None
        handler = get_stage4_block_handler(block.kind)
        return handler.render_turn(
            prompt_context=self.prompt_context,
            plan=self.plan,
            runtime=self.runtime,
            block=block,
        )

    def begin_turn(self, block_id: str) -> None:
        """Start tracking explicit submissions for one model turn."""
        if self._turn_tracker is not None:
            raise ValueError(
                f"Stage 4 turn tracking already active for block {self._turn_tracker.block_id!r}"
            )
        self._turn_tracker = _Stage4TurnTracker(block_id=block_id)

    def finish_turn(self, block_id: str) -> Stage4TurnOutcome:
        """Finish the active model turn and return its explicit submission outcome."""
        tracker = self._turn_tracker
        if tracker is None:
            raise ValueError("Stage 4 turn tracking was not started before finish_turn()")
        if tracker.block_id != block_id:
            raise ValueError(
                f"Stage 4 turn tracking mismatch: expected {tracker.block_id!r}, got {block_id!r}"
            )
        self._turn_tracker = None
        return Stage4TurnOutcome(
            block_id=tracker.block_id,
            validate_submitted=tracker.submit_count > 0,
            submit_count=tracker.submit_count,
            latest_feedback=tracker.latest_feedback,
            next_block_id=tracker.next_block_id,
        )

    def discard_turn(self) -> None:
        """Clear any active turn tracker after an aborted model call."""
        self._turn_tracker = None

    def submit(self, payload: dict[str, Any]) -> str:
        """Apply one block-local submission and return reducer feedback."""
        _stage_output, feedback, transitions = _compute_stage4_validate_step_with_transitions(
            payload,
            plan=self.plan,
            runtime=self.runtime,
            deps=self.deps,
        )
        if self.persist_runtime is not None:
            self.persist_runtime(self.runtime, transitions)
        if self._turn_tracker is not None:
            next_block = self.current_block()
            self._turn_tracker.submit_count += 1
            self._turn_tracker.latest_feedback = feedback
            self._turn_tracker.next_block_id = None if next_block is None else next_block.id
        return feedback

    def is_done(self) -> bool:
        """Whether Stage 4 has produced a final accepted result."""
        return (
            isinstance(self.runtime.cursor, Stage4DoneCursor)
            and self.accepted.model_spec is not None
            and bool(self.accepted.authored_priors)
        )

    def result(self) -> Stage4Result:
        """Materialize the current accepted Stage 4 result."""
        if self.accepted.model_spec is None or not self.accepted.authored_priors:
            raise ValueError("Stage 4 session has not completed a valid model_spec + priors")
        return Stage4Result(
            model_spec=self.accepted.model_spec,
            authored_priors=self.accepted.authored_priors,
            search_queries=dict(self.search_queries),
            validation=self.accepted.validation,
        )
def _summarize_names(names: list[str], *, limit: int = 8) -> str:
    """Render a compact preview of names."""
    if not names:
        return "(none)"
    preview = ", ".join(f"`{name}`" for name in names[:limit])
    if len(names) <= limit:
        return preview
    return f"{preview}, ... (+{len(names) - limit} more)"


def _enabled_block_tool_names(
    policy: Stage4PromptScopePolicy,
    *,
    enable_literature: bool,
    enable_paraphrasing: bool,
) -> tuple[str, ...]:
    """Return the tool names that are both scope-allowed and runtime-enabled."""
    enabled: list[str] = []
    for tool_name in policy.allowed_tool_names:
        if tool_name == "search_literature" and not enable_literature:
            continue
        if tool_name == "elicit_prior_gmm" and not enable_paraphrasing:
            continue
        enabled.append(tool_name)
    return tuple(enabled)


def _filter_cards(
    items: list[dict[str, Any]],
    key: str,
    wanted_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Filter a list of card dicts to those whose *key* is in *wanted_names*."""
    wanted = set(wanted_names)
    return [item for item in items if item[key] in wanted]


def _active_prior_parameter_names(runtime: Stage4Runtime) -> set[str] | None:
    """Return the locked active prior surface, or ``None`` before model lock."""
    model_spec = runtime.accepted.model_spec
    if not isinstance(model_spec, dict):
        return None
    return {
        str(parameter["name"])
        for parameter in (model_spec.get("parameters") or [])
        if isinstance(parameter, dict) and isinstance(parameter.get("name"), str)
    }


def _filter_model_topology(
    model_topology: dict[str, Any],
    block: Stage4FrontierBlock,
) -> dict[str, Any]:
    """Restrict model-topology context to the constructs relevant to this scope."""
    if not model_topology:
        return {}

    filtered_edges = model_topology.get("latent_edges") or []
    construct_names = set(block.construct_names)
    if construct_names:
        if block.kind == "effect_prior" and block.expand_neighbor_topology:
            expanded_construct_names = set(construct_names)
            for edge in filtered_edges:
                cause = edge.get("cause")
                effect = edge.get("effect")
                if cause in construct_names or effect in construct_names:
                    if isinstance(cause, str):
                        expanded_construct_names.add(cause)
                    if isinstance(effect, str):
                        expanded_construct_names.add(effect)
            construct_names = expanded_construct_names

        filtered_edges = [
            edge
            for edge in filtered_edges
            if edge.get("cause") in construct_names and edge.get("effect") in construct_names
        ]

    return {
        "model_clock": model_topology.get("model_clock"),
        "model_interval_days": model_topology.get("model_interval_days"),
        "outcome": model_topology.get("outcome"),
        "latent_edges": filtered_edges,
    }


def _visible_block_section(
    policy: Stage4PromptScopePolicy,
    section_name: str,
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return section items only when the active prompt scope explicitly allows them."""
    if section_name not in policy.visible_sections:
        return []
    return items


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
def _count_accepted_blocks(
    blocks: tuple[Stage4FrontierBlock, ...],
    runtime: Stage4Runtime,
) -> int:
    """Count accepted blocks in a deterministic block family."""
    return sum(_block_is_accepted(runtime, block.id) for block in blocks)


def _count_reachable_blocks(
    blocks: tuple[Stage4FrontierBlock, ...],
    runtime: Stage4Runtime,
) -> int:
    """Count non-inactive blocks in a deterministic block family."""
    return sum(runtime.block_status.get(block.id) != "inactive" for block in blocks)


def _format_plan_status(
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    block: Stage4FrontierBlock,
    *,
    causal_spec: dict[str, Any] | None = None,
) -> str:
    """Summarize the reducer frontier in a compact prompt-local format."""
    lines = [
        _STAGE4_FRONTIER_PREFIX,
        "",
        f"- phase: `{get_stage4_phase(runtime, plan=plan)}`",
        f"- model blocks accepted: `{_count_accepted_blocks(plan.model_blocks, runtime)}/{len(plan.model_blocks)}`",
        (
            "- global review: `"
            + (
                runtime.block_status.get(plan.review_block.id, "pending")
                if plan.review_block is not None
                else "skipped"
            )
            + "`"
        ),
        (
            "- prior blocks accepted: `"
            f"{_count_accepted_blocks(plan.prior_blocks, runtime)}/"
            f"{_count_reachable_blocks(plan.prior_blocks, runtime)}`"
        ),
        (
            "- prior-system review: `"
            + (
                runtime.block_status.get(plan.prior_review_block_id or "", "inactive")
                if plan.prior_review_block_id is not None
                else "skipped"
            )
            + "`"
        ),
        f"- model_spec locked: `{'yes' if runtime.accepted.model_spec is not None else 'no'}`",
        f"- active prompt scope: `{block.kind}`",
        f"- active scope names: {_summarize_names(list(block.variable_names or block.parameter_names))}",
    ]
    if runtime.block_status.get(block.id) == "reopened":
        lines.append("- block mode: `reopened`")
    if runtime.repair_campaign is not None:
        pending_block_ids = _pending_repair_campaign_block_ids(runtime.repair_campaign)
        lines.append(
            f"- active repair scope: `{runtime.repair_campaign.scope_key}` "
            f"({len(pending_block_ids)} remaining)"
        )
    if block.kind == "effect_prior":
        target_construct = block.payload.get("target_construct")
        if isinstance(target_construct, str):
            budget = build_effect_row_budget(
                model_spec=runtime.accepted.model_spec,
                authored_priors=runtime.accepted.authored_priors,
                causal_spec=causal_spec,
                target_construct=target_construct,
            )
            if budget is not None:
                lines.extend(
                    [
                        "- stability budget source: `compiled CT drift row` (advisory headroom guidance)",
                        (
                            f"- target row budget guidance: `{budget.diagonal_magnitude:.3f}` "
                            f"(conservative lower bound `{budget.diagonal_lower_bound:.3f}`)"
                        ),
                        (
                            f"- incoming effect mass currently used: `{budget.used_abs_mean:.3f}` "
                            f"(conservative `{budget.used_abs_upper:.3f}`) across "
                            f"`{budget.specified_incoming_edges}/{budget.total_incoming_edges}` edges"
                        ),
                        (
                            f"- remaining headroom guidance: `{budget.remaining_abs_mean:.3f}` "
                            f"(conservative `{budget.remaining_abs_upper:.3f}`)"
                        ),
                    ]
                )
    return "\n".join(lines)


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


def _validate_submission_envelope(
    data: dict[str, Any],
    block: Stage4FrontierBlock,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate the common Stage 4 submission envelope."""
    if not isinstance(data, dict):
        return None, "VALIDATION ERRORS:\n- submission must be a JSON object"

    block_id = data.get("block_id")
    block_kind = data.get("block_kind")
    proposal = data.get("proposal")

    if block_id != block.id:
        return (
            None,
            f"WRONG BLOCK:\n- active block id is `{block.id}`\n- received `{block_id}`",
        )
    if block_kind != block.kind:
        return (
            None,
            f"WRONG BLOCK KIND:\n- active block kind is `{block.kind}`\n- received `{block_kind}`",
        )
    if not isinstance(proposal, dict):
        return None, "VALIDATION ERRORS:\n- `proposal` must be an object"
    return proposal, None


def _normalize_indicator_submission(
    block: Stage4FrontierBlock,
    proposal: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate an indicator-decision proposal."""
    try:
        choice = DistributionChoice.model_validate(proposal).model_dump(mode="json")
    except Exception as exc:
        return None, f"VALIDATION ERRORS:\n- {exc}"

    variable = block.variable_names[0]
    if choice["variable"] != variable:
        return None, f"VALIDATION ERRORS:\n- proposal variable must be `{variable}`"

    item = block.payload
    allowed_distributions = (
        [item["fixed_distribution"]]
        if "fixed_distribution" in item
        else item.get("valid_distributions", [])
    )
    if choice["distribution"] not in allowed_distributions:
        return (
            None,
            "VALIDATION ERRORS:\n"
            f"- distribution `{choice['distribution']}` is invalid for `{variable}`",
        )

    allowed_links = (
        item.get("valid_links", [])
        if "fixed_distribution" in item
        else item.get("link_options", {}).get(choice["distribution"], [])
    )
    if choice["link"] not in allowed_links:
        return (
            None,
            "VALIDATION ERRORS:\n"
            f"- link `{choice['link']}` is invalid for `{variable}` with `{choice['distribution']}`",
        )
    return {"distribution_choice": choice}, None


def _normalize_prior_submission(
    block: Stage4FrontierBlock,
    proposal: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate a prior-block proposal."""
    raw_priors = proposal.get("priors")
    if not isinstance(raw_priors, dict) or not raw_priors:
        return None, "VALIDATION ERRORS:\n- `proposal.priors` must be a non-empty object"

    allowed = set(block.parameter_names)
    invalid = sorted(name for name in raw_priors if name not in allowed)
    if invalid:
        return (
            None,
            f"VALIDATION ERRORS:\n- priors outside the active block: {_summarize_names(invalid)}",
        )
    return {"priors": raw_priors}, None


def _normalize_global_review_submission(
    block: Stage4FrontierBlock,
    proposal: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate a compact global-review submission."""
    decision = proposal.get("decision")
    reasoning = proposal.get("reasoning")
    if decision not in {"approve", "reopen"}:
        return None, "VALIDATION ERRORS:\n- `proposal.decision` must be `approve` or `reopen`"
    if not isinstance(reasoning, str) or not reasoning.strip():
        return None, "VALIDATION ERRORS:\n- `proposal.reasoning` must be a non-empty string"
    reopen_block_ids = proposal.get("reopen_block_ids")
    if decision == "approve":
        if reopen_block_ids is not None:
            return (
                None,
                "VALIDATION ERRORS:\n- `reopen_block_ids` is only valid when decision=`reopen`",
            )
        return {"decision": decision, "reasoning": reasoning.strip()}, None

    if not isinstance(reopen_block_ids, list) or not reopen_block_ids:
        return (
            None,
            "VALIDATION ERRORS:\n- `proposal.reopen_block_ids` must be a non-empty list of model block ids",
        )
    if any(not isinstance(block_id, str) for block_id in reopen_block_ids):
        return None, "VALIDATION ERRORS:\n- `proposal.reopen_block_ids` must contain only strings"
    if len(set(reopen_block_ids)) != len(reopen_block_ids):
        return None, "VALIDATION ERRORS:\n- `proposal.reopen_block_ids` must not contain duplicates"

    allowed_ids_in_order = tuple(block.payload.get("reopenable_block_ids") or ())
    allowed_ids = set(allowed_ids_in_order)
    invalid_ids = [block_id for block_id in reopen_block_ids if block_id not in allowed_ids]
    if invalid_ids:
        return (
            None,
            "VALIDATION ERRORS:\n"
            f"- `reopen_block_ids` must be drawn from {_summarize_names(sorted(allowed_ids))}",
        )
    return {
        "decision": decision,
        "reasoning": reasoning.strip(),
        "reopen_block_ids": tuple(
            block_id for block_id in allowed_ids_in_order if block_id in set(reopen_block_ids)
        ),
    }, None


def _indicator_submission_example(block: Stage4FrontierBlock) -> dict[str, Any]:
    """Example payload for indicator-decision blocks."""
    payload = block.payload
    variable = block.variable_names[0]
    distribution = payload.get("fixed_distribution")
    if not isinstance(distribution, str):
        valid_distributions = payload.get("valid_distributions")
        if not isinstance(valid_distributions, list) or not valid_distributions:
            raise ValueError(f"Indicator block {block.id!r} is missing valid distributions")
        distribution = str(valid_distributions[0])

    valid_links = payload.get("valid_links")
    if isinstance(valid_links, list) and valid_links:
        link = str(valid_links[0])
    else:
        link_options = payload.get("link_options")
        if not isinstance(link_options, dict):
            raise ValueError(f"Indicator block {block.id!r} is missing link options")
        candidate_links = link_options.get(distribution)
        if not isinstance(candidate_links, list) or not candidate_links:
            raise ValueError(
                f"Indicator block {block.id!r} is missing links for distribution {distribution!r}"
            )
        link = str(candidate_links[0])

    return {
        "block_id": block.id,
        "block_kind": block.kind,
        "proposal": {
            "variable": variable,
            "distribution": distribution,
            "link": link,
            "reasoning": "Example only: choose one allowed distribution/link pair for the active indicator.",
        },
    }


def _example_prior_payload(prior_card: dict[str, Any]) -> dict[str, Any]:
    """Return one valid example prior payload for a concrete prompt-local prior card."""
    parameter = str(prior_card["parameter"])
    role = str(prior_card.get("role") or "")
    constraint = str(prior_card.get("constraint") or "")

    if role == "ar_coefficient" or constraint == "unit_interval":
        dist, params, reason = "Beta", {"alpha": 2.0, "beta": 2.0}, "unit-interval persistence prior for the active AR parameter."
    elif role == "fixed_effect":
        dist, params, reason = "Normal", {"mu": 0.0, "sigma": 0.2}, "conservative zero-centered lagged-effect prior for the active edge."
    elif role == "initial_state_mean":
        dist, params, reason = "Normal", {"mu": 0.0, "sigma": 1.0}, (
            "weakly informative latent-scale initial-state mean; do not copy "
            "raw indicator means or log-means unless the construct is explicitly identified "
            "on that observed scale."
        )
    elif role in {"residual_sd", "initial_state_sd", "static_state_sd", "measurement_error_sd"}:
        dist, params, reason = "HalfNormal", {"sigma": 1.0}, "positive scale prior for the active variance or measurement-noise parameter."
    elif role == "observation_hyperparameter_positive":
        dist, params, reason = "Gamma", {"concentration": 5.0, "rate": 1.0}, "positive observation-family hyperparameter prior."
    elif role == "observation_hyperparameter":
        dist, params, reason = "Normal", {"mu": 0.0, "sigma": 1.0}, "real-valued observation-family hyperparameter prior."
    elif role == "loading" and constraint == "negative":
        dist, params, reason = "TruncatedNormal", {"mu": -1.0, "sigma": 0.5, "lower": -5.0, "upper": 0.0}, "negative loading prior consistent with the locked indicator polarity."
    elif role == "loading":
        dist, params, reason = "HalfNormal", {"sigma": 1.0}, "positive loading prior consistent with the locked indicator polarity."
    elif role in {"correlation", "initial_state_correlation"} or constraint == "correlation":
        dist, params, reason = "TruncatedNormal", {"mu": 0.0, "sigma": 0.3, "lower": -1.0, "upper": 1.0}, "bounded correlation prior centered at zero."
    elif constraint == "positive":
        dist, params, reason = "HalfNormal", {"sigma": 1.0}, "positive scale prior for the active parameter."
    elif constraint == "negative":
        dist, params, reason = "TruncatedNormal", {"mu": -1.0, "sigma": 0.5, "lower": -5.0, "upper": 0.0}, "negative prior consistent with the active parameter constraint."
    else:
        dist, params, reason = "Normal", {"mu": 0.0, "sigma": 1.0}, "weakly informative unconstrained prior for the active parameter."

    return {
        "parameter": parameter,
        "distribution": dist,
        "params": params,
        "sources": [],
        "reasoning": f"Example only: {reason}",
    }


def _prior_submission_example(
    block: Stage4FrontierBlock,
    *,
    prior_cards: list[dict[str, Any]],
) -> dict[str, Any]:
    """Example payload for prior-authoring blocks."""
    if not prior_cards:
        raise ValueError(f"Prior block {block.id!r} is missing prompt-local prior cards")
    prior_payload = _example_prior_payload(prior_cards[0])
    parameter = str(prior_payload["parameter"])
    return {
        "block_id": block.id,
        "block_kind": block.kind,
        "proposal": {"priors": {parameter: prior_payload}},
    }


def _global_review_submission_example(block: Stage4FrontierBlock) -> dict[str, Any]:
    """Example payload for compact global-review blocks."""
    del block
    return {
        "block_id": "review:model_spec",
        "block_kind": "global_review",
        "proposal": {
            "decision": "approve",
            "reasoning": "The locked likelihoods and loading orientations are coherent for prior elicitation.",
        },
    }


def _prior_review_submission_example(
    block: Stage4FrontierBlock,
    *,
    prior_cards: list[dict[str, Any]],
) -> dict[str, Any]:
    """Example payload for whole-system prior-review work items."""
    example = _prior_submission_example(block, prior_cards=prior_cards)
    example["block_kind"] = "global_prior_review"
    example["block_id"] = block.id
    return example


def _format_submission_example(
    block: Stage4FrontierBlock,
    *,
    prior_cards: list[dict[str, Any]] | None = None,
    fallback_submission_example: str | None = None,
) -> str:
    """Render a block-local `validate_model` example payload."""
    if block.kind == "indicator_decision":
        example = _indicator_submission_example(block)
    elif block.kind == "global_review":
        example = _global_review_submission_example(block)
    elif block.kind == "global_prior_review":
        example = _prior_review_submission_example(block, prior_cards=prior_cards or [])
    elif block.kind in {
        "measurement_prior",
        "observation_prior",
        "dynamics_prior",
        "effect_prior",
        "correlation_prior",
    }:
        example = _prior_submission_example(block, prior_cards=prior_cards or [])
    elif fallback_submission_example is not None:
        return fallback_submission_example
    else:
        raise ValueError(f"Unsupported Stage 4 block kind {block.kind!r}")
    return "```json\n" + json.dumps(example, indent=2) + "\n```"


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
                item[key] = deepcopy(value)
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
        transitions = (
            ()
            if event.transition_payload is None
            else (event.transition_payload,)
        )
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
    from .stage4_repair import (
        _classify_compile_failure_route,
        _classify_prior_failure_blocks,
    )

    if not _should_run_partial_drift_guard(active_block=active_block, state=state):
        return state

    authored_priors = state.stage_output.get("authored_priors")
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
    except Exception as exc:
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
            events=(_make_stage4_block_accepted_event(active_block, normalized, stage_output=state.stage_output),),
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
        events=(_make_stage4_block_accepted_event(active_block, normalized, stage_output=state.stage_output),),
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
        return (_make_stage4_repair_planned_event(state.repair_plan, accepted_block_event=accepted_event),)
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


_BLOCK_HANDLERS: dict[str, Stage4BlockHandler] = {
    "indicator_decision": Stage4BlockHandler(
        kind="indicator_decision",
        prompt_policy=get_stage4_prompt_scope_policy("indicator_decision"),
        normalize_submission=_normalize_indicator_submission,
        apply_submission=_apply_indicator_submission,
    ),
    "global_review": Stage4BlockHandler(
        kind="global_review",
        prompt_policy=get_stage4_prompt_scope_policy("global_review"),
        normalize_submission=_normalize_global_review_submission,
        apply_submission=_apply_global_review_submission,
    ),
}
for _prior_kind in (
    "measurement_prior",
    "observation_prior",
    "dynamics_prior",
    "effect_prior",
    "correlation_prior",
    "global_prior_review",
):
    _BLOCK_HANDLERS[_prior_kind] = Stage4BlockHandler(
        kind=_prior_kind,
        prompt_policy=get_stage4_prompt_scope_policy(_prior_kind),
        normalize_submission=_normalize_prior_submission,
        apply_submission=_apply_prior_submission,
        include_prior_source_guidance=True,
    )


def get_stage4_block_handler(kind: str) -> Stage4BlockHandler:
    """Return the registered handler for a block kind."""
    handler = _BLOCK_HANDLERS.get(kind)
    if handler is None:
        raise ValueError(f"Unsupported Stage 4 block kind {kind!r}")
    return handler


def _compute_stage4_validate_step_with_transitions(
    data: dict[str, Any],
    *,
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    deps: Stage4Deps,
) -> tuple[dict | None, str, tuple[dict[str, Any], ...]]:
    """Advance the reducer by one ``validate_model`` submission.

    This function mutates ``runtime`` directly, including accepted output, so the
    reducer remains the single owner of Stage 4 execution state.
    """
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

    proposal, error_feedback = _validate_submission_envelope(data, active_block)
    if error_feedback is not None:
        runtime.last_validation_packet = _build_validation_packet_for_block(
            block=active_block,
            status="validation_error",
            feedback=error_feedback,
            retain_for_next_prompt=True,
            capture_stage_output=False,
        )
        return None, error_feedback, ()
    assert proposal is not None

    handler = get_stage4_block_handler(active_block.kind)
    normalized, error_feedback = handler.normalize_submission(active_block, proposal)
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

    result = handler.apply_submission(
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
            validation_override=(
                result.stage_output.get("validation") if result.stage_output is not None else None
            ),
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
    """Advance the reducer by one ``validate_model`` submission."""
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
            events=(
                _make_stage4_repair_planned_event(validation_route.repair_plan),
            ),
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


def _finalize_repair_campaign_if_complete(
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    deps: Stage4Deps,
    *,
    validation_override: AssemblyValidation | None = None,
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

    validation = validation_override or validate_assembly(
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


async def run_stage4(
    causal_spec: dict,
    question: str,
    data_for_model: pl.DataFrame,
    indicator_audits: dict[str, dict[str, Any]],
    generate: GenerateFn,
    *,
    enable_literature: bool = True,
    enable_paraphrasing: bool = False,
    n_paraphrases: int = 10,
    gmm_model: str | None = None,
    max_tool_turns: int = 40,
    load_checkpoint: Callable[[], Stage4Runtime | None] | None = None,
    save_checkpoint: Callable[[Stage4Runtime], None] | None = None,
    clear_checkpoint: Callable[[], None] | None = None,
    on_state_change: Callable[[Stage4Plan, Stage4Runtime, tuple[dict[str, Any], ...]], None]
    | None = None,
) -> Stage4Result:
    """Run the frontier-reduced Stage 4 flow sequentially."""
    from .stage4_agent_loop import run_stage4 as _run_stage4

    return await _run_stage4(
        causal_spec,
        question,
        data_for_model,
        indicator_audits,
        generate,
        enable_literature=enable_literature,
        enable_paraphrasing=enable_paraphrasing,
        n_paraphrases=n_paraphrases,
        gmm_model=gmm_model,
        max_tool_turns=max_tool_turns,
        load_checkpoint=load_checkpoint,
        save_checkpoint=save_checkpoint,
        clear_checkpoint=clear_checkpoint,
        on_state_change=on_state_change,
    )
