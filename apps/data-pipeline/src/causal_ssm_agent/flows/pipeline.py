"""Main causal inference pipeline.

Prefect owns the execution graph for stage orchestration and monitoring.
Human-in-the-loop replay is handled by ``stage_overrides``: the replay route
can provide edited payloads for selected stages, and the pipeline skips those
stage computations while re-running downstream stages from the override point.
"""

import time
from collections.abc import Callable
from inspect import isawaitable
from pathlib import Path
from typing import Any

from prefect import flow
from prefect.artifacts import create_markdown_artifact
from prefect.events import emit_event

from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.flows.stages.contracts import INTERACTIVE_STAGES
from causal_ssm_agent.utils import storage
from causal_ssm_agent.utils.data import DATA_URI, runs_dir

logger = get_prefect_logger(__name__)

STAGE_SEQUENCE = (
    "stage-0",
    "stage-1a",
    "stage-1b",
    "stage-2",
    "stage-3",
    "stage-4",
    "stage-4b",
    "stage-5a",
    "stage-5b",
    "stage-6",
)
STAGE_INDEX = {stage_id: index for index, stage_id in enumerate(STAGE_SEQUENCE)}
QUESTION_STAGES = frozenset({"stage-1a", "stage-1b", "stage-2", "stage-4"})
STAGE_PROGRESS_EVENT_PREFIX = "causal-ssm.pipeline-stage"


def _preview(text: str, *, limit: int = 120) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3]}..."


def _stage_idx(stage_id: str) -> int:
    try:
        return STAGE_INDEX[stage_id]
    except KeyError as exc:
        raise ValueError(
            f"Unknown stage '{stage_id}'. Expected one of: {', '.join(STAGE_SEQUENCE)}"
        ) from exc


def _filter_stage_overrides(stage_overrides: dict[str, dict] | None) -> dict[str, dict]:
    supported: dict[str, dict] = {}
    for stage_id, payload in (stage_overrides or {}).items():
        if stage_id not in INTERACTIVE_STAGES:
            logger.warning("Ignoring unsupported stage override: %s", stage_id)
            continue
        supported[stage_id] = payload
    return supported


def _resolve_stage_window(
    *,
    start_stage: str | None,
    end_stage: str | None,
) -> tuple[str, int, str, int]:
    if start_stage is None:
        start_stage = STAGE_SEQUENCE[0]

    start_idx = _stage_idx(start_stage)

    resolved_end_stage = end_stage or STAGE_SEQUENCE[-1]
    end_idx = _stage_idx(resolved_end_stage)
    if end_idx < start_idx:
        raise ValueError(
            f"end_stage {resolved_end_stage} cannot come before start_stage {start_stage}"
        )

    return start_stage, start_idx, resolved_end_stage, end_idx


def _resolve_question(
    *,
    query: str | None,
    user_id: str,
    start_idx: int,
    end_idx: int,
) -> str | None:
    """Resolve the research question, materializing it to disk.

    On a fresh run the caller passes ``query`` (raw text from the web UI).
    The text is written to ``data/{user_id}/query.txt`` so that future
    resume runs can pick it up automatically without re-supplying it.

    On a resume run ``query`` is typically None; the function reads from the
    previously-materialized file instead.
    """
    query_path = storage.join(DATA_URI, user_id, "query.txt")
    requires_question = any(
        stage_id in QUESTION_STAGES for stage_id in STAGE_SEQUENCE[start_idx : end_idx + 1]
    )
    if query:
        question = query.strip()
        storage.write_text(query_path, question)
        return question
    if storage.exists(query_path):
        return storage.read_text(query_path).strip()
    if requires_question:
        raise ValueError("A query is required when running stages 1a, 1b, 2, or 4.")
    return None


async def _emit_causal_spec_artifact(stage1b_web: dict[str, Any]) -> None:
    causal_spec = stage1b_web.get("causal_spec", {})
    latent = causal_spec.get("latent", {})
    measurement = causal_spec.get("measurement", {})
    artifact = create_markdown_artifact(
        key="causal-spec",
        markdown=(
            f"## Causal Specification\n\n"
            f"- **Constructs**: {len(latent.get('constructs', []))}\n"
            f"- **Edges**: {len(latent.get('edges', []))}\n"
            f"- **Indicators**: {len(measurement.get('indicators', []))}\n"
        ),
    )
    if isawaitable(artifact):
        await artifact


def _emit_stage_progress_event(
    prefect_run_id: str,
    stage_id: str,
    status: str,
    *,
    outcome: str | None = None,
    error: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "stage_id": stage_id,
        "status": status,
    }
    if outcome is not None:
        payload["outcome"] = outcome
    if error is not None:
        payload["error"] = error
    emit_event(
        event=f"{STAGE_PROGRESS_EVENT_PREFIX}.{status}",
        resource={
            "prefect.resource.id": f"prefect.flow-run.{prefect_run_id}",
            "prefect.resource.name": prefect_run_id,
        },
        payload=payload,
    )


def _partial_pipeline_result(user_id: str, stage_id: str, state: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "final_stage": stage_id,
        "stage": state["web"],
    }


def _raise_if_restored_gate_failed(stage_id: str, state: dict[str, Any]) -> None:
    gate = state.get("gate")
    if gate is None or not gate.get("gate_failed") or gate.get("gate_overridden"):
        return
    if stage_id == "stage-1b":
        raise RuntimeError(
            "No identifiable treatment effects remain after filtering. "
            "All treatments are blocked by unobserved confounders."
        )
    if stage_id == "stage-4b":
        t_rule = gate.get("t_rule", {})
        raise RuntimeError(
            f"T-rule violated: {t_rule.get('n_free_params')} free parameters "
            f"> {t_rule.get('n_moments')} moment conditions. "
            "Model is provably non-identified. Halting pipeline."
        )


@flow(
    persist_result=True,
    result_storage=Path(".prefect-cache"),
    result_serializer="pickle",
)
async def causal_inference_pipeline(
    user_id: str = "test_user",
    inference_method: str | None = None,
    enable_literature: bool | None = None,
    override_gates: bool | None = None,
    query: str | None = None,
    stage_overrides: dict[str, dict] | None = None,
    start_stage: str | None = None,
    end_stage: str | None = None,
    openrouter_api_key: str | None = None,
):
    """Run the causal pipeline end to end.

    Args:
        user_id: User ID naming the workspace under ``data/{user_id}/``.
        inference_method: Override inference method (e.g. "auto", "svi", "nuts")
        enable_literature: Override literature search
        override_gates: Continue past stage failures instead of halting
        query: Raw query text (used by web UI). Materialized to
            ``data/{user_id}/query.txt`` so resume runs auto-resolve it.
        stage_overrides: Dict mapping editable stage ids (e.g. "stage-1a") to
            replacement payloads. The pipeline skips that stage's computation and
            resumes execution from the overridden output.
        start_stage: First stage to execute in this run. Earlier stages are
            loaded from existing artifacts in data/{user_id}/run/.
        end_stage: Final stage to execute in this run. Useful for stage-specific
            development replays such as rerunning only stage 2.
        openrouter_api_key: User-provided OpenRouter API key (BYOK). Overrides the
            default key.
    """
    import os

    if openrouter_api_key:
        os.environ["OPENROUTER_API_KEY"] = openrouter_api_key
        logger.info("Using user-provided OpenRouter API key")

    from causal_ssm_agent.flows import dag
    from causal_ssm_agent.utils.config import get_config

    config = get_config()
    supported_overrides = _filter_stage_overrides(stage_overrides)
    effective_start_stage, start_idx, effective_end_stage, end_idx = _resolve_stage_window(
        start_stage=start_stage,
        end_stage=end_stage,
    )
    question = _resolve_question(
        query=query,
        user_id=user_id,
        start_idx=start_idx,
        end_idx=end_idx,
    )

    gates_overridden = (
        override_gates if override_gates is not None else config.pipeline.override_gates
    )
    lit_enabled = (
        enable_literature
        if enable_literature is not None
        else config.stage4_prior_elicitation.literature_search.enabled
    )

    logger.info(
        "Pipeline starting: user_id=%s source=%s inference_method=%s literature=%s "
        "override_gates=%s start_stage=%s end_stage=%s stage_overrides=%s",
        user_id,
        "raw text" if query else "resume/no-query",
        inference_method or "config default",
        lit_enabled,
        gates_overridden,
        effective_start_stage,
        effective_end_stage,
        sorted(supported_overrides),
    )
    if question:
        logger.info("Question preview: %s", _preview(question))

    from prefect.context import get_run_context

    prefect_run_id = str(get_run_context().flow_run.id)

    # Ensure the run directory exists
    storage.makedirs(runs_dir(user_id))

    stage_states: dict[str, dict[str, Any]] = {}

    def _restore_stage(stage_id: str) -> dict[str, Any]:
        restored = dag.load_stage_state(user_id, stage_id, prior_states=stage_states)
        stage_states[stage_id] = restored
        _emit_stage_progress_event(prefect_run_id, stage_id, "completed")
        _raise_if_restored_gate_failed(stage_id, restored)
        return restored

    async def _run_stage(stage_id: str, runner: Callable[[], Any]) -> dict[str, Any]:
        logger.info(">>> %s starting", stage_id)
        t0 = time.monotonic()
        _emit_stage_progress_event(prefect_run_id, stage_id, "running")
        try:
            stage_state = runner()
            if isawaitable(stage_state):
                stage_state = await stage_state
        except Exception as exc:
            elapsed = time.monotonic() - t0
            logger.error(">>> %s FAILED after %.1fs: %s", stage_id, elapsed, exc)
            _emit_stage_progress_event(
                prefect_run_id,
                stage_id,
                "failed",
                error={"type": "execution_error", "message": str(exc)},
            )
            raise
        # Include outcome from stage result if available
        elapsed = time.monotonic() - t0
        web_data = stage_state.get("web", {}) if isinstance(stage_state, dict) else {}
        stage_outcome = web_data.get("outcome")
        logger.info(">>> %s completed in %.1fs (outcome=%s)", stage_id, elapsed, stage_outcome)
        _emit_stage_progress_event(prefect_run_id, stage_id, "completed", outcome=stage_outcome)
        return stage_state

    async def _maybe_finish(stage_id: str) -> dict[str, Any] | None:
        if stage_id != effective_end_stage:
            return None
        if "stage-1b" in stage_states:
            await _emit_causal_spec_artifact(stage_states["stage-1b"]["web"])
        if stage_id == "stage-6":
            logger.info("Pipeline complete: run finished successfully")
            return {**stage_states["stage-5b"]["web"], **stage_states["stage-6"]["web"]}
        logger.info("Pipeline partial run complete: stopped after %s", stage_id)
        return _partial_pipeline_result(user_id, stage_id, stage_states[stage_id])

    stage0_idx = _stage_idx("stage-0")
    if start_idx > stage0_idx:
        stage0_state = _restore_stage("stage-0")
    else:
        stage0_state = await _run_stage("stage-0", lambda: dag.stage0_flow(user_id))
        stage_states["stage-0"] = stage0_state
    partial = await _maybe_finish("stage-0")
    if partial is not None:
        return partial
    stage0_result = stage0_state["result"]

    stage1a_idx = _stage_idx("stage-1a")
    if start_idx > stage1a_idx:
        stage1a_state = _restore_stage("stage-1a")
    else:
        if question is None:
            raise ValueError("Question is required to execute stage-1a")
        stage1a_state = await _run_stage(
            "stage-1a",
            lambda: dag.stage1a_flow(
                question,
                user_id,
                override_payload=supported_overrides.get("stage-1a"),
            ),
        )
        stage_states["stage-1a"] = stage1a_state
    partial = await _maybe_finish("stage-1a")
    if partial is not None:
        return partial
    stage1a_result = stage1a_state["result"]

    stage1b_idx = _stage_idx("stage-1b")
    if start_idx > stage1b_idx:
        stage1b_state = _restore_stage("stage-1b")
    else:
        if question is None:
            raise ValueError("Question is required to execute stage-1b")
        stage1b_state = await _run_stage(
            "stage-1b",
            lambda: dag.stage1b_flow(
                question,
                stage0_result,
                stage1a_result,
                gates_overridden,
                user_id,
                override_payload=supported_overrides.get("stage-1b"),
            ),
        )
        stage_states["stage-1b"] = stage1b_state
    partial = await _maybe_finish("stage-1b")
    if partial is not None:
        return partial
    stage1b_result = stage1b_state["result"]
    stage1b_gate = stage1b_state["gate"]

    stage2_idx = _stage_idx("stage-2")
    if start_idx > stage2_idx:
        stage2_state = _restore_stage("stage-2")
    else:
        if question is None:
            raise ValueError("Question is required to execute stage-2")
        stage2_state = await _run_stage(
            "stage-2",
            lambda: dag.stage2_flow(
                question, stage0_result, stage1b_result, user_id, prefect_run_id
            ),
        )
        stage_states["stage-2"] = stage2_state
    partial = await _maybe_finish("stage-2")
    if partial is not None:
        return partial
    stage2_result = stage2_state["result"]

    stage3_idx = _stage_idx("stage-3")
    if start_idx > stage3_idx:
        stage3_state = _restore_stage("stage-3")
    else:
        stage3_state = await _run_stage(
            "stage-3",
            lambda: dag.stage3_flow(stage1b_result, stage2_result, user_id),
        )
        stage_states["stage-3"] = stage3_state
    partial = await _maybe_finish("stage-3")
    if partial is not None:
        return partial

    stage4_idx = _stage_idx("stage-4")
    if start_idx > stage4_idx:
        stage4_state = _restore_stage("stage-4")
    else:
        if question is None:
            raise ValueError("Question is required to execute stage-4")
        stage4_state = await _run_stage(
            "stage-4",
            lambda: dag.stage4_flow(
                question,
                stage1b_result,
                stage2_result,
                lit_enabled,
                user_id,
                override_payload=supported_overrides.get("stage-4"),
            ),
        )
        stage_states["stage-4"] = stage4_state
    partial = await _maybe_finish("stage-4")
    if partial is not None:
        return partial
    stage4_result = stage4_state["result"]

    stage4b_idx = _stage_idx("stage-4b")
    if start_idx > stage4b_idx:
        stage4b_state = _restore_stage("stage-4b")
    else:
        stage4b_state = dag.stage4b_flow(stage4_result, stage2_result, gates_overridden, user_id)
        stage_states["stage-4b"] = stage4b_state
    partial = await _maybe_finish("stage-4b")
    if partial is not None:
        return partial

    stage5a_idx = _stage_idx("stage-5a")
    if start_idx > stage5a_idx:
        pass  # stage-5a is best-effort preflight, no restore needed
    else:
        stage5a_state = dag.stage5a_flow(
            stage4_result,
            stage2_result,
            user_id,
        )
        stage_states["stage-5a"] = stage5a_state
    partial = await _maybe_finish("stage-5a")
    if partial is not None:
        return partial

    stage5b_idx = _stage_idx("stage-5b")
    if start_idx > stage5b_idx:
        stage5b_state = _restore_stage("stage-5b")
    else:
        stage5b_state = dag.stage5b_flow(
            stage4_result,
            stage2_result,
            inference_method,
            user_id,
        )
        stage_states["stage-5b"] = stage5b_state
    partial = await _maybe_finish("stage-5b")
    if partial is not None:
        return partial
    stage5b_result = stage5b_state["result"]

    stage6_state = dag.stage6_flow(
        stage5b_result,
        stage1a_result,
        stage1b_result,
        stage1b_gate,
        user_id,
    )
    stage_states["stage-6"] = stage6_state
    partial = await _maybe_finish("stage-6")
    if partial is not None:
        return partial

    raise AssertionError("Unreachable: pipeline did not terminate at a stage boundary")


if __name__ == "__main__":
    from prefect import serve as serve_deployments

    main_dep = causal_inference_pipeline.to_deployment(
        name="causal-inference",
        tags=["causal", "llm"],
    )
    serve_deployments(main_dep)
