"""Async agent loop and checkpoint orchestration for Stage 4."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .stage4_cards import (
    build_construct_scale_cards,
    build_distribution_cards,
    build_model_topology,
    build_prior_cards,
)
from .stage4_navigation import make_stage4_runtime, pending_repair_campaign_block_ids
from .stage4_orchestrator import build_stage4_plan
from .stage4_prompt_context import Stage4Messages
from .stage4_reducer import settle_to_wait_state
from .stage4_session import Stage4Session
from .stage4_skeleton import derive_deterministic_spec
from .stage4_state import Stage4Runtime
from .stage4_types import Stage4Deps, Stage4Result

if TYPE_CHECKING:
    from collections.abc import Callable

    import polars as pl

    from causal_ssm_agent.utils.llm import GenerateFn

    from .stage4_orchestrator import Stage4Plan


def _build_stage4_tool_map(
    session: Stage4Session,
    *,
    question: str,
    enable_literature: bool,
    enable_paraphrasing: bool,
    n_paraphrases: int,
    gmm_model: str | None,
    max_tool_turns: int,
) -> dict[str, Any]:
    """Build the Stage 4 tool map for one reducer-owned session."""
    from causal_ssm_agent.flows.stages.stage4.tool_registry import (
        build_stage4_session_tool_map,
    )

    return build_stage4_session_tool_map(
        session,
        question=question,
        enable_literature=enable_literature,
        enable_paraphrasing=enable_paraphrasing,
        n_paraphrases=n_paraphrases,
        gmm_model=gmm_model,
        max_tool_turns=max_tool_turns,
    )


async def _run_stage4_turn(
    session: Stage4Session,
    generate: GenerateFn,
    tool_map: dict[str, Any],
) -> None:
    """Run one Stage 4 outer turn and require the block's submit tool."""
    turn = session.current_turn()
    if turn is None:
        raise ValueError("Stage 4 turn requested with no active block")

    allowed_tools = [tool_map[name] for name in turn.allowed_tool_names if name in tool_map]
    block_before = turn.block.id
    session.begin_turn(block_before)
    try:
        await generate(turn.messages, allowed_tools, label=f"stage-4:{block_before}")
    except Exception:
        session.discard_turn()
        raise
    outcome = session.finish_turn(block_before)
    if not outcome.submission_made:
        raise ValueError(
            "Stage 4 block "
            f"`{block_before}` did not submit `{turn.required_submission_tool_name}` "
            "before the turn ended"
        )


def _plan_block_ids(plan: Stage4Plan) -> frozenset[str]:
    """Return the full deterministic block-id set for one Stage 4 plan."""
    return frozenset(block.id for block in plan.all_blocks)


def _plan_prior_parameter_names(plan: Stage4Plan) -> frozenset[str]:
    """Return the semantic prior-parameter inventory for one Stage 4 plan."""
    return frozenset(
        parameter_name for block in plan.prior_blocks for parameter_name in block.parameter_names
    )


def _scope_key_tokens(scope_key: str) -> tuple[str, ...]:
    """Return the deterministic scope-key tokens after the scope-kind prefix."""
    _scope_kind, _separator, suffix = scope_key.partition(":")
    if not suffix or suffix == "global":
        return ()
    return tuple(token for token in suffix.split("|") if token)


def _validate_direct_writer_checkpoint_prompt_blocks(
    plan: Stage4Plan,
    runtime: Stage4Runtime,
) -> str | None:
    """Reject saved direct-writer campaigns whose prompt blocks exceed their scope."""
    campaign = runtime.domain.repair_campaign
    if campaign is None or campaign.scope_kind != "direct_writer_blocks":
        return None

    scope_tokens = set(_scope_key_tokens(campaign.scope_key))
    if not scope_tokens:
        return "checkpoint direct-writer repair scope no longer names any parameters"

    for block_id in campaign.scope_block_ids:
        plan_block = plan.get_block(block_id)
        prompt_block = campaign.prompt_blocks_by_id[block_id]
        scoped_parameter_names = tuple(
            parameter_name
            for parameter_name in plan_block.parameter_names
            if parameter_name in scope_tokens
        )
        if not scoped_parameter_names:
            return "checkpoint direct-writer repair scope no longer maps to active Stage 4 parameters"
        if scoped_parameter_names == plan_block.parameter_names:
            continue

        scoped_required_parameter_names = tuple(
            parameter_name
            for parameter_name in plan_block.required_parameter_names
            if parameter_name in scope_tokens
        )
        scoped_optional_parameter_names = tuple(
            parameter_name
            for parameter_name in plan_block.optional_parameter_names
            if parameter_name in scope_tokens
        )
        if (
            prompt_block.parameter_names != scoped_parameter_names
            or prompt_block.required_parameter_names != scoped_required_parameter_names
            or prompt_block.optional_parameter_names != scoped_optional_parameter_names
        ):
            return "checkpoint direct-writer prompt blocks no longer match the scoped repair surface"

    return None


def _validate_stage4_runtime_checkpoint(
    plan: Stage4Plan,
    runtime: Any,
) -> str | None:
    """Return an incompatibility reason when a saved Stage 4 runtime cannot resume."""
    if not isinstance(runtime, Stage4Runtime):
        return "checkpoint payload is not a Stage4Runtime"

    plan_block_ids = _plan_block_ids(plan)
    if set(runtime.domain.block_status) != plan_block_ids:
        return "checkpoint block-status keys no longer match the Stage 4 plan"

    authored_prior_names = set(runtime.domain.accepted.authored_priors)
    if not authored_prior_names.issubset(_plan_prior_parameter_names(plan)):
        return "checkpoint authored priors no longer match the Stage 4 parameter inventory"

    campaign = runtime.domain.repair_campaign
    if campaign is not None:
        scope_block_ids = set(campaign.scope_block_ids)
        if not campaign.scope_block_ids:
            return "checkpoint repair campaign has an empty scope"
        if not scope_block_ids.issubset(plan_block_ids):
            return "checkpoint repair campaign scope contains unknown Stage 4 blocks"
        if set(campaign.completed_block_ids) - scope_block_ids:
            return "checkpoint repair campaign marks completion outside its scope"
        if set(campaign.prompt_blocks_by_id) != scope_block_ids:
            return "checkpoint repair campaign prompt overrides do not match its scope"
        for block_id, prompt_block in campaign.prompt_blocks_by_id.items():
            if prompt_block.id != block_id:
                return "checkpoint repair prompt override ids are inconsistent"
        direct_writer_incompatibility = _validate_direct_writer_checkpoint_prompt_blocks(
            plan,
            runtime,
        )
        if direct_writer_incompatibility is not None:
            return direct_writer_incompatibility
        pending_block_ids = pending_repair_campaign_block_ids(campaign)
        if not pending_block_ids:
            return "checkpoint persists a completed repair campaign instead of settling it"
        if runtime.domain.done:
            return "checkpoint cannot be terminal while a repair campaign is active"
        if runtime.domain.active_block_id not in pending_block_ids:
            return "checkpoint repair campaign prompt block is outside its pending scope"

    if runtime.domain.done:
        if runtime.domain.active_block_id is not None:
            return "checkpoint marks Stage 4 done with an active block"
        if (
            runtime.domain.accepted.model_spec is None
            or not runtime.domain.accepted.authored_priors
        ):
            return "checkpoint marks Stage 4 done without a complete accepted result"
        return None

    if runtime.domain.active_block_id is None:
        return "checkpoint is not in a promptable wait-state"
    if runtime.domain.active_block_id not in plan_block_ids:
        return (
            f"checkpoint active block `{runtime.domain.active_block_id}` is not in the Stage 4 plan"
        )
    if runtime.domain.block_status.get(runtime.domain.active_block_id) not in {
        "pending",
        "reopened",
    }:
        return "checkpoint active block is not pending work"
    return None


def _load_resumable_stage4_runtime(
    plan: Stage4Plan,
    *,
    load_checkpoint: Callable[[], Stage4Runtime | None] | None,
    clear_checkpoint: Callable[[], None] | None,
) -> Stage4Runtime | None:
    """Load a saved Stage 4 runtime if it is still compatible with the current plan."""
    if load_checkpoint is None:
        return None
    runtime = load_checkpoint()
    if runtime is None:
        return None
    incompatibility = _validate_stage4_runtime_checkpoint(plan, runtime)
    if incompatibility is None:
        return runtime
    if clear_checkpoint is not None:
        clear_checkpoint()
    return None


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
    from causal_ssm_agent.flows.stages.stage4.grounding import stage4_grounding

    skeleton = derive_deterministic_spec(causal_spec)
    model_topology = build_model_topology(causal_spec)
    distribution_cards = build_distribution_cards(
        causal_spec,
        indicator_audits,
        skeleton,
    )
    construct_scale_cards = build_construct_scale_cards(
        causal_spec,
        indicator_audits,
        skeleton,
    )
    prior_cards = build_prior_cards(causal_spec, skeleton)
    plan = build_stage4_plan(causal_spec, skeleton)
    msgs = Stage4Messages(
        question=question,
        causal_spec=causal_spec,
        model_topology=model_topology,
        distribution_cards=distribution_cards,
        loading_params=skeleton.loading_params,
        construct_scale_cards=construct_scale_cards,
        prior_cards=prior_cards,
        enable_literature=enable_literature,
        enable_paraphrasing=enable_paraphrasing,
    )
    deps = Stage4Deps(
        skeleton=skeleton,
        causal_spec=causal_spec,
        data_for_model=data_for_model,
        indicator_audits=indicator_audits,
        grounding_fn=stage4_grounding,
    )
    runtime = _load_resumable_stage4_runtime(
        plan,
        load_checkpoint=load_checkpoint,
        clear_checkpoint=clear_checkpoint,
    ) or make_stage4_runtime(plan)

    def _persist(rt: Stage4Runtime, transitions: tuple[dict[str, Any], ...]) -> None:
        if save_checkpoint is not None:
            save_checkpoint(rt)
        if on_state_change is not None:
            on_state_change(plan, rt, transitions)

    persist_runtime = _persist if (save_checkpoint or on_state_change) else None

    _, _, startup_transitions, startup_changed = settle_to_wait_state(
        plan=plan,
        runtime=runtime,
        deps=deps,
    )
    if startup_changed:
        if persist_runtime is not None:
            persist_runtime(runtime, startup_transitions)
    elif on_state_change is not None:
        on_state_change(plan, runtime, ())

    session = Stage4Session(
        plan=plan,
        prompt_context=msgs,
        deps=deps,
        runtime=runtime,
        persist_runtime=persist_runtime,
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

    max_outer_turns = max(1, len(plan.all_blocks)) * 10
    for _outer_turn in range(max_outer_turns):
        if session.is_done():
            break

        turn = session.current_turn()
        if turn is None:
            raise ValueError(
                "Stage 4 stalled with no promptable wait-state "
                f"(active_block_id={session.runtime.domain.active_block_id!r}, done={session.runtime.domain.done!r})"
            )
        await _run_stage4_turn(session, generate, tool_map)
    else:
        raise ValueError(
            "Stage 4 agentic flow exceeded the outer block-turn limit without converging"
        )

    if not session.is_done():
        raise ValueError("Stage 4 agentic flow did not produce a valid model_spec + priors")

    return session.result()
