"""Stage 4 prompt-context assembly and prompt-local projections."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .prompts.model_proposal import (
    build_stage4_system_prompt,
    build_stage4_user_prompt,
)
from .stage4_feedback import (
    Stage4ScopeSnapshot,
    default_stage4_validation_packet,
    render_stage4_validation_feedback,
)
from .stage4_navigation import (
    active_prior_parameter_names,
    block_is_accepted,
    get_stage4_phase,
    pending_repair_campaign_block_ids,
)
from .stage4_partial_drift import build_effect_row_budget
from .stage4_text import summarize_stage4_names

if TYPE_CHECKING:
    from .stage4_block_specs import Stage4PromptScopePolicy
    from .stage4_feedback import Stage4ValidationPacket
    from .stage4_orchestrator import Stage4FrontierBlock, Stage4Plan
    from .stage4_state import Stage4Runtime
    from .stage4_submission import Stage4BlockHandler


_STAGE4_FRONTIER_PREFIX = "ACTIVE FRONTIER (machine-generated)"


@dataclass(frozen=True)
class Stage4Turn:
    """Structured current-turn projection for the active Stage 4 block."""

    block: Stage4FrontierBlock
    messages: list[dict[str, Any]]
    allowed_tool_names: tuple[str, ...]
    required_submission_tool_name: str
    latest_feedback: str
    phase: str


def _filter_cards(
    items: list[dict[str, Any]],
    key: str,
    wanted_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Filter a list of card dicts to those whose *key* is in *wanted_names*."""
    wanted = set(wanted_names)
    return [item for item in items if item[key] in wanted]


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


def _count_accepted_blocks(
    blocks: tuple[Stage4FrontierBlock, ...],
    runtime: Stage4Runtime,
) -> int:
    """Count accepted blocks in a deterministic block family."""
    return sum(block_is_accepted(runtime, block.id) for block in blocks)


def _count_reachable_blocks(
    blocks: tuple[Stage4FrontierBlock, ...],
    runtime: Stage4Runtime,
) -> int:
    """Count non-inactive blocks in a deterministic block family."""
    return sum(runtime.block_status.get(block.id) != "inactive" for block in blocks)


def format_stage4_plan_status(
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
        "- active scope names: "
        f"{summarize_stage4_names(list(block.variable_names or block.parameter_names))}",
    ]
    if runtime.block_status.get(block.id) == "reopened":
        lines.append("- block mode: `reopened`")
    if runtime.repair_campaign is not None:
        pending_block_ids = pending_repair_campaign_block_ids(runtime.repair_campaign)
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
            variable = card.get("variable")
            if not isinstance(variable, str):
                continue
            choice = likelihood_lookup.get(variable)
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
        handler: Stage4BlockHandler,
        include_prior_source_guidance: bool,
    ) -> Stage4ScopeSnapshot:
        """Build the typed LLM-visible snapshot for one active Stage 4 block."""
        policy = handler.prompt_policy
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
        active_parameter_names = active_prior_parameter_names(runtime)
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
            if card.get("parameter") in coupled_parameter_names
            and card.get("accepted_prior") is not None
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
            frontier_status=format_stage4_plan_status(
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
            submission_example=handler.render_submission_example(
                block,
                prior_cards=visible_prior_cards,
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
        handler: Stage4BlockHandler,
        enabled_tool_names: tuple[str, ...],
        include_prior_source_guidance: bool,
    ) -> list[dict]:
        """Build the model-facing prompt for one active Stage 4 scope."""
        policy = handler.prompt_policy
        snapshot = self._scope_snapshot_for_block(
            block=block,
            plan=plan,
            runtime=runtime,
            handler=handler,
            include_prior_source_guidance=include_prior_source_guidance,
        )
        return [
            {
                "role": "system",
                "content": build_stage4_system_prompt(
                    system_task=policy.system_task,
                    guidance_section_keys=policy.guidance_section_keys,
                    parameter_guidance_prefixes=policy.parameter_guidance_prefixes,
                    submission_tool_name=handler.submission_tool_name,
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
            handler=handler,
            enabled_tool_names=enabled_tool_names,
            include_prior_source_guidance=handler.include_prior_source_guidance_for_prompt(
                enable_literature=self.enable_literature,
            ),
        )

    def render_turn(
        self,
        *,
        plan: Stage4Plan,
        runtime: Stage4Runtime,
        block: Stage4FrontierBlock,
        handler: Stage4BlockHandler,
    ) -> Stage4Turn:
        """Render the current model-facing turn for this block."""
        allowed_tool_names = handler.allowed_tool_names(
            enable_literature=self.enable_literature,
            enable_paraphrasing=self.enable_paraphrasing,
        )
        latest_feedback = render_stage4_validation_feedback(runtime.last_validation_packet)
        messages = self.messages_for_block(
            block=block,
            plan=plan,
            runtime=runtime,
            handler=handler,
        )
        return Stage4Turn(
            block=block,
            messages=messages,
            allowed_tool_names=allowed_tool_names,
            required_submission_tool_name=handler.submission_tool_name,
            latest_feedback=latest_feedback,
            phase=get_stage4_phase(runtime, plan=plan),
        )
