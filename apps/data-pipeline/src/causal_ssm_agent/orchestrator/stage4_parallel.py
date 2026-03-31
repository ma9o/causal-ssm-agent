"""Stage 4 parallel effect-batch execution helpers."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from .stage4_repair import (
    _classify_compile_failure_route,
    _classify_prior_failure_blocks,
)

if TYPE_CHECKING:
    from causal_ssm_agent.utils.llm import GenerateFn

    from .stage4 import (
        Stage4AcceptedState,
        Stage4Deps,
        Stage4Messages,
        Stage4ParallelBlockResult,
        Stage4Runtime,
        Stage4Session,
        Stage4TurnOutcome,
    )
    from .stage4_orchestrator import Stage4FrontierBlock, Stage4Plan


def _format_parallel_batch_saved_feedback(
    block_ids: tuple[str, ...],
    next_block: Stage4FrontierBlock | None,
) -> str:
    """Acknowledge an accepted effect batch and point to the next frontier."""
    from .stage4 import _summarize_names

    lines = [
        "EFFECT BATCH ACCEPTED:",
        f"- saved {_summarize_names(list(block_ids))}",
    ]
    if next_block is not None:
        lines.append(f"- next block: `{next_block.id}` ({next_block.kind})")
    else:
        lines.append("- no remaining blocks in this phase")
    return "\n".join(lines)


def _pending_first_pass_effect_blocks(
    plan: Stage4Plan,
    runtime: Stage4Runtime,
) -> tuple[Stage4FrontierBlock, ...]:
    """Return eligible first-pass effect blocks for parallel execution."""
    from .stage4 import _block_is_accepted

    if runtime.phase != "prior_blocks" or runtime.accepted.model_spec is None:
        return ()

    pending_effect_blocks: list[Stage4FrontierBlock] = []
    seen_pending_effect = False
    for block in plan.prior_blocks:
        if block.kind in {"measurement_prior", "dynamics_prior"}:
            if not _block_is_accepted(runtime, block.id):
                return ()
            continue

        if block.kind == "effect_prior":
            status = runtime.block_status.get(block.id, "pending")
            if status == "reopened":
                return ()
            if not _block_is_accepted(runtime, block.id):
                pending_effect_blocks.append(block)
                seen_pending_effect = True
            continue

        if seen_pending_effect:
            break
        if not _block_is_accepted(runtime, block.id):
            return ()

    if len(pending_effect_blocks) <= 1:
        return ()
    if runtime.active_block_id != pending_effect_blocks[0].id:
        return ()
    return tuple(pending_effect_blocks)


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
    """Build the Stage 4 tool map for one session."""
    from causal_ssm_agent.flows.stages.stage_tools import (
        make_elicit_prior_gmm_tool,
        make_search_tool,
    )
    from causal_ssm_agent.utils.openrouter_client import Tool

    async def _execute_validate(*, model_json: str) -> str:
        try:
            data = json.loads(model_json)
        except json.JSONDecodeError as exc:
            return f"JSON parse error: {exc}"
        return session.submit(data)

    validate_tool = Tool(
        name="validate_model",
        description="Submit one active Stage 4 frontier block for validation.",
        parameters={
            "type": "object",
            "properties": {
                "model_json": {
                    "type": "string",
                    "description": (
                        "JSON object with exactly `block_id`, `block_kind`, and `proposal`. "
                        "Submit only the current active Stage 4 block."
                    ),
                }
            },
            "required": ["model_json"],
            "additionalProperties": False,
        },
        execute=_execute_validate,
        stop_on_success=True,
        success_output=None,
    )

    tool_map: dict[str, Any] = {"validate_model": validate_tool}
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
) -> Stage4TurnOutcome:
    """Run one Stage 4 outer turn and require a validate_model submission."""
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
    if not outcome.validate_submitted:
        raise ValueError(
            f"Stage 4 block `{block_before}` did not submit `validate_model` before the turn ended"
        )
    return outcome


def _make_parallel_effect_worker_session(
    *,
    plan: Stage4Plan,
    prompt_context: Stage4Messages,
    deps: Stage4Deps,
    runtime: Stage4Runtime,
    block: Stage4FrontierBlock,
    accepted_snapshot: Stage4AcceptedState,
    search_cache_snapshot: dict[str, str],
) -> Stage4Session:
    """Build an isolated Stage 4 session for one first-pass effect block."""
    from .stage4 import Stage4Runtime, Stage4Session

    worker_runtime = Stage4Runtime(
        phase="prior_blocks",
        active_block_id=block.id,
        block_status=dict(runtime.block_status),
        decisions=deepcopy(runtime.decisions),
        accepted=deepcopy(accepted_snapshot),
        search_cache=dict(search_cache_snapshot),
    )
    return Stage4Session(
        plan=plan,
        prompt_context=prompt_context,
        deps=deps,
        runtime=worker_runtime,
    )


async def _run_parallel_effect_block(
    *,
    plan: Stage4Plan,
    prompt_context: Stage4Messages,
    deps: Stage4Deps,
    runtime: Stage4Runtime,
    block: Stage4FrontierBlock,
    accepted_snapshot: Stage4AcceptedState,
    search_cache_snapshot: dict[str, str],
    generate: GenerateFn,
    question: str,
    enable_literature: bool,
    enable_paraphrasing: bool,
    n_paraphrases: int,
    gmm_model: str | None,
    max_tool_turns: int,
    max_outer_turns: int,
) -> Stage4ParallelBlockResult:
    """Run one isolated first-pass effect block until it is accepted."""
    session = _make_parallel_effect_worker_session(
        plan=plan,
        prompt_context=prompt_context,
        deps=deps,
        runtime=runtime,
        block=block,
        accepted_snapshot=accepted_snapshot,
        search_cache_snapshot=search_cache_snapshot,
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

    for _ in range(max_outer_turns):
        if session.runtime.block_status.get(block.id) == "accepted":
            break

        current_block = session.current_block()
        if current_block is None or current_block.id != block.id:
            raise ValueError(
                "Stage 4 parallel effect worker lost block ownership before acceptance: "
                f"expected `{block.id}`, got `{None if current_block is None else current_block.id}`"
            )

        await _run_stage4_turn(session, generate, tool_map)

        if session.runtime.block_status.get(block.id) == "accepted":
            break

        current_block = session.current_block()
        if current_block is None or current_block.id != block.id:
            raise ValueError(
                "Stage 4 parallel effect worker reopened a non-local block: "
                f"expected `{block.id}`, got `{None if current_block is None else current_block.id}`"
            )
    else:
        raise ValueError(
            f"Stage 4 parallel effect block `{block.id}` exceeded the outer turn limit"
        )

    authored_priors = session.accepted.authored_priors
    block_priors = {
        name: deepcopy(authored_priors[name])
        for name in block.parameter_names
        if name in authored_priors
    }
    missing = [name for name in block.parameter_names if name not in block_priors]
    if missing:
        raise ValueError(
            f"Stage 4 parallel effect block `{block.id}` was accepted without all owned priors: "
            f"{', '.join(sorted(missing))}"
        )

    new_search_cache = {
        query: result
        for query, result in session.search_cache.items()
        if query not in search_cache_snapshot
    }
    from .stage4 import Stage4ParallelBlockResult

    return Stage4ParallelBlockResult(
        block_id=block.id,
        authored_priors=block_priors,
        search_queries=dict(session.search_queries),
        search_cache=new_search_cache,
        validation=session.accepted.validation,
    )


async def _run_parallel_effect_batch(
    *,
    blocks: tuple[Stage4FrontierBlock, ...],
    plan: Stage4Plan,
    prompt_context: Stage4Messages,
    deps: Stage4Deps,
    runtime: Stage4Runtime,
    generate: GenerateFn,
    question: str,
    enable_literature: bool,
    enable_paraphrasing: bool,
    n_paraphrases: int,
    gmm_model: str | None,
    max_tool_turns: int,
    max_outer_turns: int,
    effect_block_concurrency: int,
) -> tuple[Stage4ParallelBlockResult, ...]:
    """Run eligible first-pass effect blocks through a bounded async worker pool."""
    accepted_snapshot = deepcopy(runtime.accepted)
    search_cache_snapshot = dict(runtime.search_cache)
    queue: asyncio.Queue[Stage4FrontierBlock] = asyncio.Queue()
    for block in blocks:
        queue.put_nowait(block)

    results: list[Stage4ParallelBlockResult] = []
    failure: Exception | None = None
    failure_lock = asyncio.Lock()
    results_lock = asyncio.Lock()

    async def _worker() -> None:
        nonlocal failure
        while True:
            if failure is not None:
                return
            try:
                block = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                result = await _run_parallel_effect_block(
                    plan=plan,
                    prompt_context=prompt_context,
                    deps=deps,
                    runtime=runtime,
                    block=block,
                    accepted_snapshot=accepted_snapshot,
                    search_cache_snapshot=search_cache_snapshot,
                    generate=generate,
                    question=question,
                    enable_literature=enable_literature,
                    enable_paraphrasing=enable_paraphrasing,
                    n_paraphrases=n_paraphrases,
                    gmm_model=gmm_model,
                    max_tool_turns=max_tool_turns,
                    max_outer_turns=max_outer_turns,
                )
            except Exception as exc:
                async with failure_lock:
                    if failure is None:
                        failure = exc
                return
            finally:
                queue.task_done()

            async with results_lock:
                results.append(result)

    worker_count = min(len(blocks), max(1, effect_block_concurrency))
    await asyncio.gather(*[asyncio.create_task(_worker()) for _ in range(worker_count)])
    if failure is not None:
        raise failure
    return tuple(results)


def _merge_parallel_effect_batch_results(
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    results: tuple[Stage4ParallelBlockResult, ...],
) -> None:
    """Merge isolated effect-worker outputs into the shared reducer state."""
    from .stage4 import _activate_prior_phase, get_active_plan_block

    results_by_id = {result.block_id: result for result in results}
    merged_block_ids: list[str] = []
    latest_validation = None
    for block in plan.prior_blocks:
        result = results_by_id.get(block.id)
        if result is None:
            continue
        runtime.accepted.authored_priors.update(result.authored_priors)
        runtime.search_queries.update(result.search_queries)
        runtime.search_cache.update(result.search_cache)
        runtime.block_status[block.id] = "accepted"
        latest_validation = result.validation
        merged_block_ids.append(block.id)

    if latest_validation is not None:
        runtime.accepted.validation = latest_validation

    _activate_prior_phase(plan, runtime)
    runtime.last_feedback = _format_parallel_batch_saved_feedback(
        tuple(merged_block_ids),
        get_active_plan_block(plan, runtime),
    )


def _finalize_parallel_effect_batch_if_complete(
    plan: Stage4Plan,
    runtime: Stage4Runtime,
    deps: Stage4Deps,
    *,
    merged_block_ids: tuple[str, ...],
) -> None:
    """Run the missing full-system validation barrier after a merged effect batch."""
    from causal_ssm_agent.flows.stages.stage4_assembly import (
        format_validation_feedback,
        validate_assembly,
    )

    from .stage4 import Stage4StepResult, _apply_stage4_step_result

    if runtime.phase != "done":
        return
    if runtime.accepted.model_spec is None or not runtime.accepted.authored_priors:
        return

    validation = validate_assembly(
        runtime.accepted.model_spec,
        runtime.accepted.authored_priors,
        deps.data_for_model,
        deps.indicator_audits,
        deps.causal_spec,
    )
    runtime.accepted.validation = validation

    if not validation.compile_ok:
        fallback_block = plan.get_block(merged_block_ids[0]) if merged_block_ids else None
        if fallback_block is None:
            raise ValueError(
                "Parallel effect batch finished without a fallback block for compile repair"
            )
        repair_scope = _classify_compile_failure_route(
            plan,
            fallback_block,
            validation.compile_error,
        )
        _apply_stage4_step_result(
            plan,
            runtime,
            Stage4StepResult(
                feedback=format_validation_feedback(
                    validation,
                    runtime.accepted.authored_priors,
                    changed_params=list(merged_block_ids),
                ),
                repair_scope=repair_scope,
            ),
        )
        return

    if validation.pp_checked and not validation.pp_valid:
        fallback_block = plan.get_block(merged_block_ids[-1]) if merged_block_ids else None
        if fallback_block is None:
            raise ValueError(
                "Parallel effect batch finished without a fallback block for PP repair"
            )
        repair_scope = _classify_prior_failure_blocks(
            plan,
            fallback_block,
            validation,
            runtime,
        )
        _apply_stage4_step_result(
            plan,
            runtime,
            Stage4StepResult(
                feedback=format_validation_feedback(
                    validation,
                    runtime.accepted.authored_priors,
                    changed_params=list(merged_block_ids),
                ),
                repair_scope=repair_scope,
            ),
        )
        return

    feedback = format_validation_feedback(
        validation,
        runtime.accepted.authored_priors,
        changed_params=list(merged_block_ids),
    )
    if feedback != "VALID":
        from .stage4 import get_active_plan_block

        runtime.last_feedback = (
            _format_parallel_batch_saved_feedback(
                merged_block_ids,
                get_active_plan_block(plan, runtime),
            )
            + "\n\n"
            + feedback
        )
