"""Stage 4: Model Specification & Prior Elicitation (Agentic).

Frontier-reduced multi-turn orchestration for Stage 4. The LLM only sees one
active decision block at a time while deterministic reducer state preserves
accepted decisions and routes retries back to the smallest current frontier.

Follows the same two-layer architecture as stages 1a/1b:
- This module contains pure orchestrator logic (framework-agnostic).
- The Prefect wrapper lives in ``flows/stages/stage4_model.py``.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from causal_ssm_agent.orchestrator.schemas_model import (
    DistributionChoice,
    LoadingConstraintChoice,
    validate_model_spec_decisions_dict,
)

from .prompts.model_proposal import (
    build_stage4_system_prompt,
    build_stage4_user_prompt,
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
from .stage4_repair import ResolvedRepairScope

if TYPE_CHECKING:
    from collections.abc import Callable

    import polars as pl

    from causal_ssm_agent.flows.stages.stage4_assembly import AssemblyValidation
    from causal_ssm_agent.utils.llm import GenerateFn
    from causal_ssm_agent.workers.schemas_prior import PriorPathologyCertificate


_STAGE4_FRONTIER_PREFIX = "ACTIVE FRONTIER (machine-generated)"


@dataclass
class Stage4Result:
    """Result of the agentic Stage 4 flow."""

    model_spec: dict[str, Any]
    authored_priors: dict[str, dict]
    search_queries: dict[str, str] = field(default_factory=dict)
    validation: AssemblyValidation | None = None


@dataclass
class Stage4AcceptedState:
    """Typed accepted Stage 4 artifacts accumulated across reducer steps."""

    model_spec: dict[str, Any] | None = None
    authored_priors: dict[str, dict[str, Any]] = field(default_factory=dict)
    validation: AssemblyValidation | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def as_current(self) -> dict[str, Any]:
        """Return the accepted state in grounding-compatible dict form."""
        current = dict(self.extras)
        if self.model_spec is not None:
            current["model_spec"] = self.model_spec
        if self.authored_priors:
            current["authored_priors"] = self.authored_priors
        if self.validation is not None:
            current["validation"] = self.validation
        return current

    def apply_stage_output(self, stage_output: dict[str, Any] | None) -> None:
        """Merge accepted stage output into typed state."""
        if stage_output is None:
            return
        if "model_spec" in stage_output:
            self.model_spec = stage_output["model_spec"]
        if "authored_priors" in stage_output:
            self.authored_priors = stage_output["authored_priors"]
        if "validation" in stage_output:
            self.validation = stage_output["validation"]
        for key, value in stage_output.items():
            if key not in {"model_spec", "authored_priors", "validation"}:
                self.extras[key] = value


@dataclass
class Stage4DecisionState:
    """Accepted model-decision deltas before the full ModelSpec is locked."""

    distribution_choices: dict[str, dict[str, Any]] = field(default_factory=dict)
    loading_constraints: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class Stage4Runtime:
    """Mutable Stage 4 reducer runtime."""

    phase: str = "model_decisions"
    active_block_id: str | None = None
    block_status: dict[str, str] = field(default_factory=dict)
    decisions: Stage4DecisionState = field(default_factory=Stage4DecisionState)
    accepted: Stage4AcceptedState = field(default_factory=Stage4AcceptedState)
    last_feedback: str | None = None
    search_cache: dict[str, str] = field(default_factory=dict)
    search_queries: dict[str, str] = field(default_factory=dict)
    repair_campaign: Stage4RepairCampaignState | None = None


@dataclass(frozen=True)
class Stage4Deps:
    """Static Stage 4 runtime dependencies shared across reducer steps."""

    skeleton: Stage4Skeleton
    causal_spec: dict[str, Any]
    data_for_model: pl.DataFrame
    indicator_audits: dict[str, dict[str, Any]]
    grounding_fn: Callable[..., tuple[dict | None, str]]


@dataclass(frozen=True)
class Stage4StepResult:
    """Reducer transition returned by a single Stage 4 step."""

    feedback: str | None = None
    stage_output: dict[str, Any] | None = None
    repair_scope: ResolvedRepairScope | None = None
    accepted_block_id: str | None = None
    distribution_choice: dict[str, Any] | None = None
    loading_constraints: tuple[dict[str, Any], ...] = ()
    persist_stage_output: bool = False


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


@dataclass(frozen=True)
class Stage4ParallelBlockResult:
    """Accepted output from one isolated first-pass Stage 4 effect worker."""

    block_id: str
    authored_priors: dict[str, dict[str, Any]]
    search_queries: dict[str, str]
    search_cache: dict[str, str]
    validation: AssemblyValidation | None = None


@dataclass
class Stage4RepairCampaignState:
    """Active bounded Stage 4 repair campaign over one structural scope."""

    failure_family_key: tuple[Any, ...]
    scope_kind: str
    scope_key: str
    scope_rank: int
    scope_block_ids: tuple[str, ...]
    remaining_block_ids: tuple[str, ...]
    attempts_at_scope: int = 1
    best_certificate: PriorPathologyCertificate | None = None


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
        latest_feedback = runtime.last_feedback or _default_stage4_feedback()
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
            phase=runtime.phase,
        )


@dataclass
class Stage4Messages:
    """Prompt-local context used to render a single active Stage 4 block."""

    question: str
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

    def _loading_constraint_lookup(self, runtime: Stage4Runtime) -> dict[str, str]:
        """Return the current loading constraint per loading parameter."""
        lookup: dict[str, str] = {}
        for parameter in (runtime.accepted.model_spec or {}).get("parameters") or []:
            if not isinstance(parameter, dict):
                continue
            if parameter.get("role") != "loading":
                continue
            name = parameter.get("name")
            constraint = parameter.get("constraint")
            if isinstance(name, str) and isinstance(constraint, str):
                lookup[name] = constraint
        for parameter_name, choice in runtime.decisions.loading_constraints.items():
            constraint = choice.get("constraint")
            if isinstance(constraint, str):
                lookup[parameter_name] = constraint
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

    def _loading_params_for_runtime(
        self,
        runtime: Stage4Runtime,
    ) -> list[dict[str, Any]]:
        """Return stateful loading cards for the current runtime."""
        loading_params = deepcopy(self.loading_params)
        loading_constraint_lookup = self._loading_constraint_lookup(runtime)
        for parameter in loading_params:
            selected_constraint = loading_constraint_lookup.get(parameter.get("name"))
            if selected_constraint is not None:
                parameter["selected_constraint"] = selected_constraint
        return loading_params

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
        """Return prior cards enriched with accepted loading constraints."""
        cards = deepcopy(self.prior_cards)
        loading_constraint_lookup = self._loading_constraint_lookup(runtime)
        for card in cards:
            if card.get("role") != "loading":
                continue
            selected_constraint = loading_constraint_lookup.get(card.get("parameter"))
            if selected_constraint is not None:
                card["constraint"] = selected_constraint
        return cards

    def messages_for_block(
        self,
        block: Stage4FrontierBlock,
        plan: Stage4Plan,
        runtime: Stage4Runtime,
        handler: Stage4BlockHandler,
    ) -> list[dict]:
        """Build the model-facing prompt for the current active block only."""
        policy = handler.prompt_policy
        enabled_tool_names = handler.allowed_tool_names(
            enable_literature=self.enable_literature,
            enable_paraphrasing=self.enable_paraphrasing,
        )
        distribution_cards = self._distribution_cards_for_runtime(runtime)
        loading_params = self._loading_params_for_runtime(runtime)
        construct_scale_cards = self._construct_scale_cards_for_runtime(runtime)
        prior_cards = self._prior_cards_for_runtime(runtime)
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
                    model_topology=_filter_model_topology(self.model_topology, block),
                    frontier_status=_format_plan_status(plan, runtime, block),
                    block_id=block.id,
                    block_kind=block.kind,
                    block_label=block.label,
                    block_instructions=policy.user_task,
                    distribution_cards=_visible_block_section(
                        policy,
                        "distribution_cards",
                        _filter_distribution_cards(distribution_cards, block),
                    ),
                    loading_params=_visible_block_section(
                        policy,
                        "loading_params",
                        _filter_loading_params(loading_params, block),
                    ),
                    construct_scale_cards=_visible_block_section(
                        policy,
                        "construct_scale_cards",
                        _filter_construct_scale_cards(construct_scale_cards, block),
                    ),
                    prior_cards=_visible_block_section(
                        policy,
                        "prior_cards",
                        _filter_prior_cards(prior_cards, block),
                    ),
                    submission_example=_format_submission_example(handler, block),
                    latest_feedback=runtime.last_feedback or _default_stage4_feedback(),
                    include_prior_source_guidance=handler.include_prior_source_guidance,
                ),
            },
        ]


@dataclass
class Stage4Session:
    """Single owner of the current Stage 4 turn and accepted state."""

    plan: Stage4Plan
    prompt_context: Stage4Messages
    deps: Stage4Deps
    runtime: Stage4Runtime = field(default_factory=Stage4Runtime)
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
        return get_active_plan_block(self.plan, self.runtime)

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
        _stage_output, feedback = compute_stage4_validate_step(
            payload,
            plan=self.plan,
            runtime=self.runtime,
            deps=self.deps,
        )
        if self._turn_tracker is not None:
            next_block = self.current_block()
            self._turn_tracker.submit_count += 1
            self._turn_tracker.latest_feedback = feedback
            self._turn_tracker.next_block_id = None if next_block is None else next_block.id
        return feedback

    def is_done(self) -> bool:
        """Whether Stage 4 has produced a final accepted result."""
        return (
            self.runtime.phase == "done"
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


def _filter_distribution_cards(
    distribution_cards: list[dict[str, Any]],
    block: Stage4FrontierBlock,
) -> list[dict[str, Any]]:
    wanted = set(block.variable_names)
    return [card for card in distribution_cards if card["variable"] in wanted]


def _filter_loading_params(
    loading_params: list[dict[str, Any]],
    block: Stage4FrontierBlock,
) -> list[dict[str, Any]]:
    wanted = set(block.parameter_names)
    return [param for param in loading_params if param["name"] in wanted]


def _filter_construct_scale_cards(
    construct_scale_cards: list[dict[str, Any]],
    block: Stage4FrontierBlock,
) -> list[dict[str, Any]]:
    wanted = set(block.construct_names)
    return [card for card in construct_scale_cards if card["construct"] in wanted]


def _filter_prior_cards(
    prior_cards: list[dict[str, Any]],
    block: Stage4FrontierBlock,
) -> list[dict[str, Any]]:
    wanted = set(block.parameter_names)
    return [card for card in prior_cards if card["parameter"] in wanted]


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
        if block.kind == "effect_prior":
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


def _default_stage4_feedback() -> str:
    """Default feedback shown before the first submission for a block."""
    return "No validator feedback yet. Submit the active block only."


def _block_is_accepted(runtime: Stage4Runtime, block_id: str) -> bool:
    """Whether a block is currently accepted in runtime state."""
    return runtime.block_status.get(block_id) == "accepted"


def get_active_plan_block(
    plan: Stage4Plan,
    runtime: Stage4Runtime,
) -> Stage4FrontierBlock | None:
    """Return the current active block from explicit runtime state."""
    if runtime.active_block_id is None:
        return None
    return plan.get_block(runtime.active_block_id)


def get_stage4_phase(runtime: Stage4Runtime) -> str:
    """Return the current Stage 4 runtime phase."""
    return runtime.phase


def _next_pending_block(
    blocks: tuple[Stage4FrontierBlock, ...],
    runtime: Stage4Runtime,
) -> Stage4FrontierBlock | None:
    """Return the next pending block in deterministic order."""
    for block in blocks:
        if not _block_is_accepted(runtime, block.id):
            return block
    return None


def _activate_model_phase(plan: Stage4Plan, runtime: Stage4Runtime) -> None:
    """Set runtime to the next pending model-decision block, if any."""
    runtime.phase = "model_decisions"
    next_block = _next_pending_block(plan.model_blocks, runtime)
    runtime.active_block_id = next_block.id if next_block is not None else None


def _activate_review_phase(plan: Stage4Plan, runtime: Stage4Runtime) -> None:
    """Set runtime to the compact global-review block, if pending."""
    review_block = plan.review_block
    if review_block is None or _block_is_accepted(runtime, review_block.id):
        _activate_prior_phase(plan, runtime)
        return
    runtime.phase = "global_review"
    runtime.active_block_id = review_block.id


def _activate_prior_phase(plan: Stage4Plan, runtime: Stage4Runtime) -> None:
    """Set runtime to the next pending prior block, or mark Stage 4 done."""
    next_block = _next_pending_block(plan.prior_blocks, runtime)
    if next_block is None:
        runtime.phase = "done"
        runtime.active_block_id = None
        return
    runtime.phase = "prior_blocks"
    runtime.active_block_id = next_block.id


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

    if any(block.kind in {"indicator_decision", "loading_decision"} for block in blocks) and (
        plan.review_block is not None
    ):
        runtime.block_status[plan.review_block.id] = "pending"

    for block_id in block_ids:
        runtime.block_status[block_id] = "reopened"

    active_block = blocks[0]
    runtime.active_block_id = active_block.id
    if active_block.kind in {"indicator_decision", "loading_decision"}:
        runtime.phase = "model_decisions"
    elif active_block.kind == "global_review":
        runtime.phase = "global_review"
    elif active_block.kind == "global_prior_review":
        runtime.phase = "global_prior_review"
    else:
        runtime.phase = "prior_blocks"


def _advance_after_block_acceptance(
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    block: Stage4FrontierBlock,
) -> None:
    """Advance runtime after a block has been accepted."""
    if block.kind in {"indicator_decision", "loading_decision"}:
        _activate_model_phase(plan, runtime)
        return
    if block.kind == "global_review":
        _activate_prior_phase(plan, runtime)
        return
    if block.kind == "global_prior_review":
        runtime.phase = "done"
        runtime.active_block_id = None
        return
    _activate_prior_phase(plan, runtime)


def _count_accepted_blocks(
    blocks: tuple[Stage4FrontierBlock, ...],
    runtime: Stage4Runtime,
) -> int:
    """Count accepted blocks in a deterministic block family."""
    return sum(_block_is_accepted(runtime, block.id) for block in blocks)


def _format_plan_status(
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    block: Stage4FrontierBlock,
) -> str:
    """Summarize the reducer frontier in a compact prompt-local format."""
    lines = [
        _STAGE4_FRONTIER_PREFIX,
        "",
        f"- phase: `{runtime.phase}`",
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
        f"- prior blocks accepted: `{_count_accepted_blocks(plan.prior_blocks, runtime)}/{len(plan.prior_blocks)}`",
        (
            "- prior-system review: `"
            + (
                runtime.block_status.get(plan.prior_review_block.id, "inactive")
                if plan.prior_review_block is not None
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
        lines.append(
            f"- active repair scope: `{runtime.repair_campaign.scope_key}` "
            f"({len(runtime.repair_campaign.remaining_block_ids)} remaining)"
        )
    return "\n".join(lines)


def _clear_repair_campaign(runtime: Stage4Runtime) -> None:
    """Clear any active structural repair campaign."""
    runtime.repair_campaign = None


def _format_repair_campaign_feedback(
    scope: ResolvedRepairScope,
    *,
    accepted_block_id: str | None,
    next_block: Stage4FrontierBlock | None,
) -> str:
    """Render bounded repair-campaign progress for the LLM."""
    lines = [
        "REPAIR CAMPAIGN ACTIVE:",
        f"- scope: `{scope.scope_key}`",
        f"- reason: {scope.reason}",
    ]
    if accepted_block_id is not None:
        lines.append(f"- kept `{accepted_block_id}` as part of the repair scope")
    if next_block is not None:
        lines.append(f"- next repair block: `{next_block.id}` ({next_block.kind})")
    else:
        lines.append("- repair scope ready for barrier validation")
    return "\n".join(lines)


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

    current_secondary = current.secondary_score if current.secondary_score is not None else float("inf")
    candidate_secondary = (
        candidate.secondary_score if candidate.secondary_score is not None else float("inf")
    )
    current_key = (current.primary_score, current_secondary)
    candidate_key = (candidate.primary_score, candidate_secondary)
    return candidate if candidate_key < current_key else current


def _set_phase_for_block(
    runtime: Stage4Runtime,
    block: Stage4FrontierBlock | None,
) -> None:
    """Project one next block back onto the Stage 4 phase machine."""
    if block is None:
        return
    if block.kind in {"indicator_decision", "loading_decision"}:
        runtime.phase = "model_decisions"
    elif block.kind == "global_review":
        runtime.phase = "global_review"
    elif block.kind == "global_prior_review":
        runtime.phase = "global_prior_review"
    else:
        runtime.phase = "prior_blocks"


def _parameter_names_for_blocks(
    plan: Stage4Plan,
    block_ids: tuple[str, ...],
) -> list[str]:
    """Return ordered semantic parameter names owned by a set of Stage 4 blocks."""
    parameter_names: list[str] = []
    seen: set[str] = set()
    for block_id in block_ids:
        block = plan.get_block(block_id)
        if block is None:
            continue
        for parameter_name in block.parameter_names:
            if parameter_name in seen:
                continue
            seen.add(parameter_name)
            parameter_names.append(parameter_name)
    return parameter_names


def _start_repair_campaign(
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    scope: ResolvedRepairScope,
    *,
    accepted_block_id: str | None,
) -> None:
    """Start or widen one deterministic structural repair campaign."""
    current = runtime.repair_campaign
    attempts_at_scope = 1
    best_certificate = scope.pathology_certificate
    if (
        current is not None
        and current.failure_family_key == scope.failure_family
        and current.scope_key == scope.scope_key
    ):
        attempts_at_scope = current.attempts_at_scope + 1
        best_certificate = _merge_best_certificate(
            current.best_certificate,
            scope.pathology_certificate,
        )

    remaining_block_ids = tuple(
        block_id for block_id in scope.block_ids if block_id != accepted_block_id
    )
    runtime.repair_campaign = Stage4RepairCampaignState(
        failure_family_key=scope.failure_family,
        scope_kind=scope.scope_kind,
        scope_key=scope.scope_key,
        scope_rank=scope.scope_rank,
        scope_block_ids=scope.block_ids,
        remaining_block_ids=remaining_block_ids,
        attempts_at_scope=attempts_at_scope,
        best_certificate=best_certificate,
    )

    if any(
        (block := plan.get_block(block_id)) is not None
        and block.kind in {"indicator_decision", "loading_decision"}
        for block_id in scope.block_ids
    ) and plan.review_block is not None:
        runtime.block_status[plan.review_block.id] = "pending"

    for block_id in scope.block_ids:
        if block_id == accepted_block_id:
            continue
        runtime.block_status[block_id] = "reopened"

    next_block_id = remaining_block_ids[0] if remaining_block_ids else None
    runtime.active_block_id = next_block_id
    next_block = None if next_block_id is None else plan.get_block(next_block_id)
    _set_phase_for_block(runtime, next_block)


def _advance_repair_campaign_after_acceptance(
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    accepted_block_id: str,
) -> bool:
    """Advance the active repair campaign after one block is accepted."""
    campaign = runtime.repair_campaign
    if campaign is None or accepted_block_id not in campaign.remaining_block_ids:
        return False

    remaining_block_ids = tuple(
        block_id for block_id in campaign.remaining_block_ids if block_id != accepted_block_id
    )
    runtime.repair_campaign = Stage4RepairCampaignState(
        failure_family_key=campaign.failure_family_key,
        scope_kind=campaign.scope_kind,
        scope_key=campaign.scope_key,
        scope_rank=campaign.scope_rank,
        scope_block_ids=campaign.scope_block_ids,
        remaining_block_ids=remaining_block_ids,
        attempts_at_scope=campaign.attempts_at_scope,
        best_certificate=campaign.best_certificate,
    )
    runtime.active_block_id = remaining_block_ids[0] if remaining_block_ids else None
    if runtime.active_block_id is None:
        return True

    next_block = plan.get_block(runtime.active_block_id)
    if next_block is None:
        raise ValueError(f"Unknown Stage 4 block id {runtime.active_block_id!r}")
    _set_phase_for_block(runtime, next_block)
    return True


def _uses_repair_campaign(scope: ResolvedRepairScope) -> bool:
    """Whether a reopened scope should be managed as a structural repair campaign."""
    return scope.scope_kind in {
        "direct_writer_blocks",
        "local_drift_motif",
        "reciprocal_pair",
        "scc_drift_subsystem",
        "validator_scope",
        "global_prior_review",
    }


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


def _normalize_loading_submission(
    block: Stage4FrontierBlock,
    proposal: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate a loading-decision proposal."""
    raw_constraints = proposal.get("loading_constraints")
    if not isinstance(raw_constraints, list) or not raw_constraints:
        return None, "VALIDATION ERRORS:\n- `proposal.loading_constraints` must be a non-empty list"

    allowed_parameters = set(block.parameter_names)
    validated: list[dict[str, Any]] = []
    for item in raw_constraints:
        try:
            constraint = LoadingConstraintChoice.model_validate(item).model_dump(mode="json")
        except Exception as exc:
            return None, f"VALIDATION ERRORS:\n- {exc}"
        if constraint["parameter"] not in allowed_parameters:
            return (
                None,
                "VALIDATION ERRORS:\n"
                f"- loading parameter `{constraint['parameter']}` is not in the active block",
            )
        validated.append(constraint)
    return {"loading_constraints": validated}, None


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
    return {
        "block_id": block.id,
        "block_kind": block.kind,
        "proposal": {
            "variable": block.variable_names[0],
            "distribution": "poisson",
            "link": "log",
            "reasoning": "Counts with nonnegative integer support and multiplicative effects.",
        },
    }


def _loading_submission_example(block: Stage4FrontierBlock) -> dict[str, Any]:
    """Example payload for loading-decision blocks."""
    return {
        "block_id": block.id,
        "block_kind": block.kind,
        "proposal": {
            "loading_constraints": [
                {
                    "parameter": name,
                    "constraint": "positive",
                    "reasoning": "Higher indicator values should move in the same direction as the construct.",
                }
                for name in block.parameter_names[:1]
            ]
        },
    }


def _prior_submission_example(block: Stage4FrontierBlock) -> dict[str, Any]:
    """Example payload for prior-authoring blocks."""
    parameter = block.parameter_names[0]
    return {
        "block_id": block.id,
        "block_kind": block.kind,
        "proposal": {
            "priors": {
                parameter: {
                    "parameter": parameter,
                    "distribution": "Normal",
                    "params": {"mu": 0.0, "sigma": 0.3},
                    "sources": [],
                    "reasoning": "Block-local prior justification.",
                }
            }
        },
    }


def _global_review_submission_example(block: Stage4FrontierBlock) -> dict[str, Any]:
    """Example payload for compact global-review blocks."""
    del block
    return {
        "block_id": "review:model_spec",
        "block_kind": "global_review",
        "proposal": {
            "decision": "approve",
            "reasoning": "The locked likelihoods and loading constraints are coherent for prior elicitation.",
        },
    }


def _format_submission_example(
    handler: Stage4BlockHandler,
    block: Stage4FrontierBlock,
) -> str:
    """Render a block-local `validate_model` example payload."""
    if handler.kind == "indicator_decision":
        example = _indicator_submission_example(block)
    elif handler.kind == "loading_decision":
        example = _loading_submission_example(block)
    elif handler.kind == "global_review":
        example = _global_review_submission_example(block)
    else:
        example = _prior_submission_example(block)
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


def _all_model_blocks_accepted(plan: Stage4Plan, runtime: Stage4Runtime) -> bool:
    """Whether every model-decision block is accepted in runtime state."""
    return all(_block_is_accepted(runtime, block.id) for block in plan.model_blocks)


def _next_pending_after(
    blocks: tuple[Stage4FrontierBlock, ...],
    runtime: Stage4Runtime,
    accepted_block_id: str,
) -> Stage4FrontierBlock | None:
    """Return the next pending block after accepting the given block."""
    for block in blocks:
        if block.id == accepted_block_id:
            continue
        if not _block_is_accepted(runtime, block.id):
            return block
    return None


def _apply_stage4_step_result(
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    result: Stage4StepResult,
) -> None:
    """Apply a reducer transition result in one place."""

    if result.distribution_choice is not None:
        choice = result.distribution_choice
        runtime.decisions.distribution_choices[choice["variable"]] = choice

    for constraint in result.loading_constraints:
        runtime.decisions.loading_constraints[constraint["parameter"]] = constraint

    if result.persist_stage_output:
        _persist_stage4_stage_output(runtime, result.stage_output)

    if result.accepted_block_id is not None:
        runtime.block_status[result.accepted_block_id] = "accepted"
    if result.repair_scope is not None and result.repair_scope.block_ids:
        if _uses_repair_campaign(result.repair_scope):
            _start_repair_campaign(
                plan,
                runtime,
                result.repair_scope,
                accepted_block_id=result.accepted_block_id,
            )
        else:
            _clear_repair_campaign(runtime)
            _mark_blocks_reopened(plan, runtime, result.repair_scope.block_ids)
    elif result.accepted_block_id is not None:
        handled_by_campaign = _advance_repair_campaign_after_acceptance(
            plan,
            runtime,
            result.accepted_block_id,
        )
        if not handled_by_campaign:
            accepted_block = plan.get_block(result.accepted_block_id)
            if accepted_block is None:
                raise ValueError(f"Unknown Stage 4 block id {result.accepted_block_id!r}")
            _advance_after_block_acceptance(plan, runtime, accepted_block)
    if result.feedback is not None:
        runtime.last_feedback = None if result.feedback == "VALID" else result.feedback


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
    return Stage4StepResult(
        feedback=_format_block_saved_feedback(
            active_block,
            _next_pending_after(plan.model_blocks, runtime, active_block.id),
        ),
        accepted_block_id=active_block.id,
        distribution_choice=normalized["distribution_choice"],
    )


def _apply_loading_submission(
    *,
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    active_block: Stage4FrontierBlock,
    normalized: dict[str, Any],
    deps: Stage4Deps,
) -> Stage4StepResult:
    """Apply one loading-constraint decision block."""
    del deps
    return Stage4StepResult(
        feedback=_format_block_saved_feedback(
            active_block,
            _next_pending_after(plan.model_blocks, runtime, active_block.id),
        ),
        accepted_block_id=active_block.id,
        loading_constraints=tuple(normalized["loading_constraints"]),
    )


def _apply_prior_submission(
    *,
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    active_block: Stage4FrontierBlock,
    normalized: dict[str, Any],
    deps: Stage4Deps,
) -> Stage4StepResult:
    """Apply one prior-authoring block and route failures back locally."""
    from .stage4_repair import (
        _classify_compile_failure_route,
        _classify_prior_failure_blocks,
    )

    stage_output, feedback = deps.grounding_fn(
        {"priors": normalized["priors"]},
        deps.causal_spec,
        current=runtime.accepted.as_current(),
        data_for_model=deps.data_for_model,
        indicator_audits=deps.indicator_audits,
    )
    validation = stage_output.get("validation") if stage_output else None
    repair_scope: ResolvedRepairScope | None = None
    campaign = runtime.repair_campaign
    in_active_campaign = campaign is not None and active_block.id in campaign.remaining_block_ids
    final_campaign_block = (
        in_active_campaign
        and campaign is not None
        and campaign.remaining_block_ids == (active_block.id,)
    )
    if validation is not None and getattr(validation, "compile_ok", True) is False:
        repair_scope = _classify_compile_failure_route(
            plan,
            active_block,
            getattr(validation, "compile_error", None) or feedback,
        )
    elif in_active_campaign:
        if not final_campaign_block:
            next_block_id = next(
                (
                    block_id
                    for block_id in campaign.remaining_block_ids
                    if block_id != active_block.id
                ),
                None,
            )
            next_block = None if next_block_id is None else plan.get_block(next_block_id)
            return Stage4StepResult(
                stage_output=stage_output,
                feedback=(
                    "REPAIR CAMPAIGN PROGRESS:\n"
                    f"- kept `{active_block.id}` within `{campaign.scope_key}`\n"
                    + (
                        f"- next repair block: `{next_block.id}` ({next_block.kind})"
                        if next_block is not None
                        else "- barrier validation pending"
                    )
                ),
                accepted_block_id=active_block.id,
                persist_stage_output=stage_output is not None,
            )
        return Stage4StepResult(
            stage_output=stage_output,
            feedback=(
                "REPAIR CAMPAIGN READY FOR VALIDATION:\n"
                f"- completed `{campaign.scope_key}`"
            ),
            accepted_block_id=active_block.id,
            persist_stage_output=stage_output is not None,
        )
    elif (
        validation is not None
        and getattr(validation, "pp_checked", False)
        and getattr(validation, "pp_valid", True) is False
    ):
        repair_scope = _classify_prior_failure_blocks(
            plan,
            active_block,
            validation,
            runtime,
        )

    widening_scope = (
        repair_scope is not None
        and campaign is not None
        and repair_scope.scope_rank > campaign.scope_rank
        and active_block.id in repair_scope.block_ids
    )
    accepted_block_id = (
        active_block.id
        if stage_output is not None
        and (
            repair_scope is None
            or active_block.id not in repair_scope.block_ids
            or (len(repair_scope.block_ids) > 1 and not widening_scope)
        )
        else None
    )

    if repair_scope is not None and repair_scope.block_ids and len(repair_scope.block_ids) > 1:
        next_block_id = next(
            (block_id for block_id in repair_scope.block_ids if block_id != accepted_block_id),
            None,
        )
        next_block = None if next_block_id is None else plan.get_block(next_block_id)
        feedback = _format_repair_campaign_feedback(
            repair_scope,
            accepted_block_id=accepted_block_id,
            next_block=next_block,
        )

    return Stage4StepResult(
        stage_output=stage_output,
        feedback=feedback,
        repair_scope=repair_scope,
        accepted_block_id=accepted_block_id,
        persist_stage_output=accepted_block_id is not None,
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
        return Stage4StepResult(
            feedback=_format_block_saved_feedback(
                active_block,
                _next_pending_block(plan.prior_blocks, runtime),
            ),
            accepted_block_id=active_block.id,
        )
    reopen_block_ids = normalized["reopen_block_ids"]
    return Stage4StepResult(
        feedback=(
            "MODEL REVIEW REOPENED:\n"
            f"- reopening {_summarize_names(list(reopen_block_ids))}\n"
            f"- reason: {normalized['reasoning']}"
        ),
        repair_scope=ResolvedRepairScope(
            scope_kind="global_review",
            scope_rank=0,
            scope_key=f"global_review:{'|'.join(reopen_block_ids)}",
            block_ids=reopen_block_ids,
            reason=normalized["reasoning"],
            failure_family=("global_review", active_block.id),
        ),
    )


def _build_model_spec_from_decisions(
    decisions: Stage4DecisionState,
    skeleton: Stage4Skeleton,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Materialize a ModelSpec from accepted model-decision state."""
    decisions_data = {
        "distribution_choices": list(decisions.distribution_choices.values()),
        "loading_constraints": list(decisions.loading_constraints.values()),
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
    "loading_decision": Stage4BlockHandler(
        kind="loading_decision",
        prompt_policy=get_stage4_prompt_scope_policy("loading_decision"),
        normalize_submission=_normalize_loading_submission,
        apply_submission=_apply_loading_submission,
    ),
    "measurement_prior": Stage4BlockHandler(
        kind="measurement_prior",
        prompt_policy=get_stage4_prompt_scope_policy("measurement_prior"),
        normalize_submission=_normalize_prior_submission,
        apply_submission=_apply_prior_submission,
        include_prior_source_guidance=True,
    ),
    "dynamics_prior": Stage4BlockHandler(
        kind="dynamics_prior",
        prompt_policy=get_stage4_prompt_scope_policy("dynamics_prior"),
        normalize_submission=_normalize_prior_submission,
        apply_submission=_apply_prior_submission,
        include_prior_source_guidance=True,
    ),
    "effect_prior": Stage4BlockHandler(
        kind="effect_prior",
        prompt_policy=get_stage4_prompt_scope_policy("effect_prior"),
        normalize_submission=_normalize_prior_submission,
        apply_submission=_apply_prior_submission,
        include_prior_source_guidance=True,
    ),
    "correlation_prior": Stage4BlockHandler(
        kind="correlation_prior",
        prompt_policy=get_stage4_prompt_scope_policy("correlation_prior"),
        normalize_submission=_normalize_prior_submission,
        apply_submission=_apply_prior_submission,
        include_prior_source_guidance=True,
    ),
    "global_review": Stage4BlockHandler(
        kind="global_review",
        prompt_policy=get_stage4_prompt_scope_policy("global_review"),
        normalize_submission=_normalize_global_review_submission,
        apply_submission=_apply_global_review_submission,
    ),
    "global_prior_review": Stage4BlockHandler(
        kind="global_prior_review",
        prompt_policy=get_stage4_prompt_scope_policy("global_prior_review"),
        normalize_submission=_normalize_prior_submission,
        apply_submission=_apply_prior_submission,
        include_prior_source_guidance=True,
    ),
}


def get_stage4_block_handler(kind: str) -> Stage4BlockHandler:
    """Return the registered handler for a block kind."""
    handler = _BLOCK_HANDLERS.get(kind)
    if handler is None:
        raise ValueError(f"Unsupported Stage 4 block kind {kind!r}")
    return handler


def compute_stage4_validate_step(
    data: dict[str, Any],
    *,
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    deps: Stage4Deps,
) -> tuple[dict | None, str]:
    """Advance the reducer by one ``validate_model`` submission.

    This function mutates ``runtime`` directly, including accepted output, so the
    reducer remains the single owner of Stage 4 execution state.
    """
    active_block = get_active_plan_block(plan, runtime)
    if active_block is None:
        return None, "VALIDATION ERRORS:\n- no active Stage 4 frontier block remains"

    proposal, error_feedback = _validate_submission_envelope(data, active_block)
    if error_feedback is not None:
        runtime.last_feedback = error_feedback
        return None, error_feedback
    assert proposal is not None

    handler = get_stage4_block_handler(active_block.kind)
    normalized, error_feedback = handler.normalize_submission(active_block, proposal)
    if error_feedback is not None:
        runtime.last_feedback = error_feedback
        return None, error_feedback
    assert normalized is not None

    result = handler.apply_submission(
        plan=plan,
        runtime=runtime,
        active_block=active_block,
        normalized=normalized,
        deps=deps,
    )
    _apply_stage4_step_result(plan, runtime, result)
    _finalize_repair_campaign_if_complete(
        plan,
        runtime,
        deps,
        validation_override=(
            result.stage_output.get("validation")
            if result.stage_output is not None
            else None
        ),
    )

    if (
        active_block.kind in {"indicator_decision", "loading_decision"}
        and result.accepted_block_id == active_block.id
        and result.repair_scope is None
        and _all_model_blocks_accepted(plan, runtime)
    ):
        lock_result = _lock_stage4_model_spec(
            plan=plan,
            runtime=runtime,
            deps=deps,
            failed_block=active_block,
        )
        _apply_stage4_step_result(plan, runtime, lock_result)
        if lock_result.repair_scope is None:
            _activate_review_phase(plan, runtime)
        assert lock_result.feedback is not None
        return lock_result.stage_output, lock_result.feedback

    if runtime.last_feedback is None and result.feedback is not None:
        return result.stage_output, result.feedback
    assert runtime.last_feedback is not None or result.feedback is not None
    return result.stage_output, runtime.last_feedback or result.feedback


def _lock_stage4_model_spec(
    *,
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    deps: Stage4Deps,
    failed_block: Stage4FrontierBlock,
) -> Stage4StepResult:
    """Materialize and validate the locked model spec after model decisions."""
    from .stage4_repair import _classify_compile_failure_route

    model_spec, errors = _build_model_spec_from_decisions(runtime.decisions, deps.skeleton)
    if model_spec is None:
        feedback = "VALIDATION ERRORS:\n" + "\n".join(f"- {error}" for error in errors)
        return Stage4StepResult(
            feedback=feedback,
            repair_scope=ResolvedRepairScope(
                scope_kind="model_spec_lock",
                scope_rank=0,
                scope_key=f"model_spec_lock:{failed_block.id}",
                block_ids=(failed_block.id,),
                reason="locked model_spec could not be materialized",
                failure_family=("model_spec_lock", failed_block.id),
            ),
        )

    stage_output, feedback = deps.grounding_fn(
        {"model_spec": model_spec},
        deps.causal_spec,
        current=runtime.accepted.as_current(),
        data_for_model=deps.data_for_model,
        indicator_audits=deps.indicator_audits,
    )
    validation = stage_output.get("validation") if stage_output else None
    if validation is not None and getattr(validation, "compile_ok", True) is False:
        return Stage4StepResult(
            stage_output=stage_output,
            feedback=feedback,
            repair_scope=_classify_compile_failure_route(
                plan,
                failed_block,
                getattr(validation, "compile_error", None) or feedback,
            ),
        )
    return Stage4StepResult(
        stage_output=stage_output,
        feedback=feedback,
        persist_stage_output=stage_output is not None,
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
) -> None:
    """Validate a completed multi-block repair campaign at the campaign barrier."""
    from causal_ssm_agent.flows.stages.stage4_assembly import (
        format_validation_feedback,
        validate_assembly,
    )

    from .stage4_repair import (
        _classify_compile_failure_route,
        _classify_prior_failure_blocks,
    )

    campaign = runtime.repair_campaign
    if campaign is None or campaign.remaining_block_ids:
        return
    if runtime.accepted.model_spec is None or not runtime.accepted.authored_priors:
        return

    validation = validation_override or validate_assembly(
        runtime.accepted.model_spec,
        runtime.accepted.authored_priors,
        deps.data_for_model,
        deps.indicator_audits,
        deps.causal_spec,
    )
    runtime.accepted.validation = validation
    representative_block = _campaign_representative_block(plan, runtime)
    changed_params = _parameter_names_for_blocks(plan, campaign.scope_block_ids)
    feedback = format_validation_feedback(
        validation,
        runtime.accepted.authored_priors,
        changed_params=changed_params,
    )

    if not validation.compile_ok:
        repair_scope = _classify_compile_failure_route(
            plan,
            representative_block,
            validation.compile_error,
        )
        _apply_stage4_step_result(
            plan,
            runtime,
            Stage4StepResult(
                feedback=feedback,
                repair_scope=repair_scope,
            ),
        )
        return

    if validation.pp_checked and not validation.pp_valid:
        repair_scope = _classify_prior_failure_blocks(
            plan,
            representative_block,
            validation,
            runtime,
        )
        _apply_stage4_step_result(
            plan,
            runtime,
            Stage4StepResult(
                feedback=feedback,
                repair_scope=repair_scope,
            ),
        )
        return

    _clear_repair_campaign(runtime)
    _activate_prior_phase(plan, runtime)
    if feedback == "VALID":
        next_block = get_active_plan_block(plan, runtime)
        runtime.last_feedback = _format_block_saved_feedback(
            representative_block,
            next_block,
        )
    else:
        runtime.last_feedback = feedback


def make_stage4_runtime(plan: Stage4Plan) -> Stage4Runtime:
    """Create the mutable runtime for a new Stage 4 plan execution."""
    runtime = Stage4Runtime(
        phase="model_decisions",
        active_block_id=plan.model_blocks[0].id if plan.model_blocks else None,
        block_status={block.id: "pending" for block in plan.all_blocks},
    )
    if plan.prior_review_block is not None:
        runtime.block_status[plan.prior_review_block.id] = "inactive"
    return runtime


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
    effect_block_concurrency: int = 1,
) -> Stage4Result:
    """Run the frontier-reduced Stage 4 flow."""
    from causal_ssm_agent.flows.stages.stage_tools import stage4_grounding

    from .stage4_parallel import (
        _build_stage4_tool_map,
        _finalize_parallel_effect_batch_if_complete,
        _merge_parallel_effect_batch_results,
        _pending_first_pass_effect_blocks,
        _run_parallel_effect_batch,
        _run_stage4_turn,
    )

    skeleton = derive_deterministic_spec(causal_spec)
    model_topology = build_model_topology(causal_spec)
    distribution_cards = build_distribution_cards(causal_spec, indicator_audits, skeleton)
    construct_scale_cards = build_construct_scale_cards(causal_spec, indicator_audits, skeleton)
    prior_cards = build_prior_cards(causal_spec, skeleton)
    plan = build_stage4_plan(causal_spec, skeleton)
    msgs = Stage4Messages(
        question=question,
        model_topology=model_topology,
        distribution_cards=distribution_cards,
        loading_params=skeleton.loading_params,
        construct_scale_cards=construct_scale_cards,
        prior_cards=prior_cards,
        enable_literature=enable_literature,
        enable_paraphrasing=enable_paraphrasing,
    )
    runtime = make_stage4_runtime(plan)

    session = Stage4Session(
        plan=plan,
        prompt_context=msgs,
        deps=Stage4Deps(
            skeleton=skeleton,
            causal_spec=causal_spec,
            data_for_model=data_for_model,
            indicator_audits=indicator_audits,
            grounding_fn=stage4_grounding,
        ),
        runtime=runtime,
    )
    tool_map = _build_stage4_tool_map(
        session,
        question=question,
        enable_literature=enable_literature,
        enable_paraphrasing=enable_paraphrasing,
        n_paraphrases=n_paraphrases,
        gmm_model=gmm_model,
        max_tool_turns=max_tool_turns,
    )

    deps = session.deps

    if not plan.model_blocks:
        initial_model_spec, errors = _build_model_spec_from_decisions(runtime.decisions, skeleton)
        if initial_model_spec is None:
            raise ValueError(
                "Stage 4 could not materialize an initial ModelSpec: " + "; ".join(errors)
            )
        stage_output, feedback = deps.grounding_fn(
            {"model_spec": initial_model_spec},
            deps.causal_spec,
            current=session.accepted.as_current(),
            data_for_model=deps.data_for_model,
            indicator_audits=deps.indicator_audits,
        )
        validation = stage_output.get("validation") if stage_output else None
        if validation is not None and getattr(validation, "compile_ok", True) is False:
            raise ValueError(
                f"Stage 4 could not lock the initial ModelSpec: {validation.compile_error}"
            )
        _persist_stage4_stage_output(session.runtime, stage_output)
        session.runtime.last_feedback = None if feedback == "VALID" else feedback
        _activate_review_phase(plan, session.runtime)

    max_outer_turns = max(1, len(plan.all_blocks)) * 10
    for _outer_turn in range(max_outer_turns):
        if session.is_done():
            break
        effect_batch = _pending_first_pass_effect_blocks(plan, session.runtime)
        if effect_batch and effect_block_concurrency > 1:
            results = await _run_parallel_effect_batch(
                blocks=effect_batch,
                plan=plan,
                prompt_context=msgs,
                deps=deps,
                runtime=session.runtime,
                generate=generate,
                question=question,
                enable_literature=enable_literature,
                enable_paraphrasing=enable_paraphrasing,
                n_paraphrases=n_paraphrases,
                gmm_model=gmm_model,
                max_tool_turns=max_tool_turns,
                max_outer_turns=max_outer_turns,
                effect_block_concurrency=effect_block_concurrency,
            )
            _merge_parallel_effect_batch_results(plan, session.runtime, results)
            _finalize_parallel_effect_batch_if_complete(
                plan,
                session.runtime,
                deps,
                merged_block_ids=tuple(result.block_id for result in results),
            )
            continue

        turn = session.current_turn()
        if turn is None:
            break
        await _run_stage4_turn(session, generate, tool_map)
    else:
        raise ValueError(
            "Stage 4 agentic flow exceeded the outer block-turn limit without converging"
        )

    if not session.is_done():
        raise ValueError("Stage 4 agentic flow did not produce a valid model_spec + priors")

    return session.result()
