"""Async agent loop and checkpoint orchestration for Stage 4."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .stage4_feedback import should_store_stage4_validation_packet
from .stage4_navigation import _activate_review_phase, make_stage4_runtime
from .stage4_orchestrator import (
    build_construct_scale_cards,
    build_distribution_cards,
    build_model_topology,
    build_prior_cards,
    build_stage4_plan,
    derive_deterministic_spec,
)
from .stage4_prompt_context import Stage4Messages
from .stage4_reducer import _build_model_spec_from_decisions, _persist_stage4_stage_output
from .stage4_session import Stage4Session
from .stage4_state import (
    Stage4BlockCursor,
    Stage4DoneCursor,
    Stage4ModelSpecLockPendingCursor,
    Stage4RepairBarrierCursor,
    Stage4Runtime,
)
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
    from causal_ssm_agent.flows.stages.stage4.tools import (
        make_elicit_prior_gmm_tool,
        make_search_tool,
        make_submit_indicator_choice_tool,
        make_submit_model_review_tool,
        make_submit_prior_block_tool,
    )

    tool_map: dict[str, Any] = {
        "submit_indicator_choice": make_submit_indicator_choice_tool(session),
        "submit_model_review": make_submit_model_review_tool(session),
        "submit_prior_block": make_submit_prior_block_tool(session),
    }
    if enable_literature:
        tool_map["search_literature"] = make_search_tool(session)
    if enable_paraphrasing:
        tool_map["elicit_prior_gmm"] = make_elicit_prior_gmm_tool(
            question=question,
            model_name=gmm_model or "",
            n_paraphrases=n_paraphrases,
            max_tool_turns=max_tool_turns,
        )
    return tool_map


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


def _validate_stage4_runtime_checkpoint(
    plan: Stage4Plan,
    runtime: Any,
) -> str | None:
    """Return an incompatibility reason when a saved Stage 4 runtime cannot resume."""
    if not isinstance(runtime, Stage4Runtime):
        return "checkpoint payload is not a Stage4Runtime"

    plan_block_ids = _plan_block_ids(plan)
    if set(runtime.block_status) != plan_block_ids:
        return "checkpoint block-status keys no longer match the Stage 4 plan"

    authored_prior_names = set(runtime.accepted.authored_priors)
    if not authored_prior_names.issubset(_plan_prior_parameter_names(plan)):
        return "checkpoint authored priors no longer match the Stage 4 parameter inventory"

    cursor = runtime.cursor
    if isinstance(cursor, Stage4BlockCursor):
        if cursor.block_id not in plan_block_ids:
            return f"checkpoint cursor block `{cursor.block_id}` is not in the Stage 4 plan"
    elif isinstance(cursor, Stage4RepairBarrierCursor):
        if not cursor.scope_block_ids:
            return "checkpoint repair barrier cursor has an empty scope"
        if not set(cursor.scope_block_ids).issubset(plan_block_ids):
            return "checkpoint repair barrier scope contains unknown Stage 4 blocks"
    elif not isinstance(cursor, (Stage4ModelSpecLockPendingCursor, Stage4DoneCursor)):
        return f"checkpoint cursor `{cursor!r}` is unsupported"

    campaign = runtime.repair_campaign
    if campaign is None:
        if isinstance(cursor, Stage4RepairBarrierCursor):
            return "checkpoint has a repair barrier cursor without an active repair campaign"
    else:
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
        pending_block_ids = tuple(
            block_id
            for block_id in campaign.scope_block_ids
            if block_id not in campaign.completed_block_ids
        )
        if isinstance(cursor, Stage4RepairBarrierCursor):
            if not campaign.requires_barrier_validation:
                return (
                    "checkpoint repair barrier cursor is incompatible with a non-barrier campaign"
                )
            if tuple(campaign.scope_block_ids) != cursor.scope_block_ids:
                return "checkpoint repair barrier cursor scope disagrees with the repair campaign"
            if pending_block_ids:
                return "checkpoint repair barrier cursor still has pending repair blocks"
        elif isinstance(cursor, Stage4BlockCursor):
            if (
                cursor.block_id in campaign.scope_block_ids
                and cursor.block_id not in pending_block_ids
            ):
                return "checkpoint repair cursor points at a non-pending block"

    if isinstance(cursor, Stage4DoneCursor) and (
        runtime.accepted.model_spec is None or not runtime.accepted.authored_priors
    ):
        return "checkpoint marks Stage 4 done without a complete accepted result"
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

    if on_state_change is not None:
        on_state_change(plan, runtime, ())

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

    deps = session.deps

    if not plan.model_blocks and session.accepted.model_spec is None:
        initial_model_spec, errors = _build_model_spec_from_decisions(runtime.decisions, skeleton)
        if initial_model_spec is None:
            raise ValueError(
                "Stage 4 could not materialize an initial ModelSpec: " + "; ".join(errors)
            )
        grounding_result = deps.grounding_fn(
            {"model_spec": initial_model_spec},
            deps.causal_spec,
            current=session.accepted.as_current(),
            data_for_model=deps.data_for_model,
            indicator_audits=deps.indicator_audits,
        )
        stage_output = grounding_result.stage_output
        validation = stage_output.get("validation") if stage_output else None
        if validation is not None and getattr(validation, "compile_ok", True) is False:
            raise ValueError(
                f"Stage 4 could not lock the initial ModelSpec: {validation.compile_error}"
            )
        _persist_stage4_stage_output(session.runtime, stage_output)
        initial_packet = grounding_result.validation_packet
        session.runtime.last_validation_packet = (
            initial_packet if should_store_stage4_validation_packet(initial_packet) else None
        )
        _activate_review_phase(plan, session.runtime)
        if persist_runtime is not None:
            persist_runtime(session.runtime, ())

    max_outer_turns = max(1, len(plan.all_blocks)) * 10
    for _outer_turn in range(max_outer_turns):
        if session.is_done():
            break

        turn = session.current_turn()
        if turn is None:
            raise ValueError(
                f"Stage 4 stalled with no promptable execution cursor ({session.runtime.cursor!r})"
            )
        await _run_stage4_turn(session, generate, tool_map)
    else:
        raise ValueError(
            "Stage 4 agentic flow exceeded the outer block-turn limit without converging"
        )

    if not session.is_done():
        raise ValueError("Stage 4 agentic flow did not produce a valid model_spec + priors")

    if clear_checkpoint is not None:
        clear_checkpoint()
    return session.result()
