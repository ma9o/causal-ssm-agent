"""Main causal inference pipeline.

Prefect owns the execution graph for stage orchestration and monitoring.
Human-in-the-loop replay is handled by ``stage_overrides``: the replay route
can provide edited payloads for selected stages, and the pipeline skips those
stage computations while re-running downstream stages from the override point.
"""

import os
import time
from inspect import isawaitable
from pathlib import Path
from typing import Any

from prefect import flow
from prefect.artifacts import create_markdown_artifact

from nof1_causal_lab.flows import get_current_flow_run_id, get_prefect_logger
from nof1_causal_lab.flows.runtime_events import emit_stage_progress_event
from nof1_causal_lab.utils import storage
from nof1_causal_lab.utils.byok_secret_store import consume_byok_secret_ref
from nof1_causal_lab.utils.data import DATA_URI, runs_dir

logger = get_prefect_logger(__name__)


def _preview(text: str, *, limit: int = 120) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3]}..."


def _filter_stage_overrides(
    stage_overrides: dict[str, dict] | None,
    *,
    overrideable_stages: frozenset[str],
) -> dict[str, dict]:
    supported: dict[str, dict] = {}
    for stage_id, payload in (stage_overrides or {}).items():
        if stage_id not in overrideable_stages:
            logger.warning("Ignoring unsupported stage override: %s", stage_id)
            continue
        supported[stage_id] = payload
    return supported


def _resolve_stage_window(
    *,
    start_stage: str | None,
    end_stage: str | None,
    execution_order: tuple[str, ...],
    stage_index: dict[str, int],
) -> tuple[str, int, str, int]:
    if start_stage is None:
        start_stage = execution_order[0]

    try:
        start_idx = stage_index[start_stage]
    except KeyError as exc:
        known = ", ".join(execution_order)
        raise ValueError(f"Unknown stage '{start_stage}'. Expected one of: {known}") from exc

    resolved_end_stage = end_stage or execution_order[-1]
    try:
        end_idx = stage_index[resolved_end_stage]
    except KeyError as exc:
        known = ", ".join(execution_order)
        raise ValueError(f"Unknown stage '{resolved_end_stage}'. Expected one of: {known}") from exc
    if end_idx < start_idx:
        raise ValueError(
            f"end_stage {resolved_end_stage} cannot come before start_stage {start_stage}"
        )

    return start_stage, start_idx, resolved_end_stage, end_idx


def _resolve_question(
    *,
    query: str | None,
    workspace_id: str,
    relevant_stage_ids: tuple[str, ...],
    question_stages: frozenset[str],
) -> str | None:
    """Resolve the research question, materializing it to disk.

    On a fresh run the caller passes ``query`` (raw text from the web UI).
    The text is written to ``data/{workspace_id}/query.txt`` so that future
    resume runs can pick it up automatically without re-supplying it.

    On a resume run ``query`` is typically None; the function reads from the
    previously-materialized file instead.
    """
    query_path = storage.join(DATA_URI, workspace_id, "query.txt")
    requires_question = any(stage_id in question_stages for stage_id in relevant_stage_ids)
    if query:
        question = query.strip()
        storage.write_text(query_path, question)
        return question
    if storage.exists(query_path):
        return storage.read_text(query_path).strip()
    if requires_question:
        raise ValueError("A query is required when running stages 1a, 1b, 2, or 4.")
    return None


async def _emit_causal_spec_artifact(stage1b: Any) -> None:
    causal_spec = stage1b.causal_spec
    latent = causal_spec.latent
    measurement = causal_spec.measurement
    artifact = create_markdown_artifact(
        key="causal-spec",
        markdown=(
            f"## Causal Specification\n\n"
            f"- **Constructs**: {len(latent.constructs)}\n"
            f"- **Edges**: {len(latent.edges)}\n"
            f"- **Indicators**: {len(measurement.indicators)}\n"
        ),
    )
    if isawaitable(artifact):
        await artifact


def _partial_pipeline_result(workspace_id: str, stage_id: str, contract: Any) -> dict[str, Any]:
    return {
        "workspace_id": workspace_id,
        "final_stage": stage_id,
        "stage": contract.model_dump(mode="json"),
    }


def _stage_fail_reason(contract: Any) -> str | None:
    if contract.outcome != "fail":
        return None
    return contract.fail_reason


@flow(
    persist_result=True,
    result_storage=Path(".prefect-cache"),
    result_serializer="pickle",
)
async def causal_inference_pipeline(
    workspace_id: str = "test_workspace",
    inference_method: str | None = None,
    enable_literature: bool | None = None,
    query: str | None = None,
    stage_overrides: dict[str, dict] | None = None,
    start_stage: str | None = None,
    end_stage: str | None = None,
    openrouter_access_mode: str | None = None,
    openrouter_secret_ref: str | None = None,
):
    """Run the causal pipeline end to end.

    Args:
        workspace_id: Workspace ID naming the workspace under ``data/{workspace_id}/``.
        inference_method: Override inference method (e.g. "pit_particle_mgrad")
        enable_literature: Override literature search
        query: Raw query text (used by web UI). Materialized to
            ``data/{workspace_id}/query.txt`` so resume runs auto-resolve it.
        stage_overrides: Dict mapping editable stage ids (e.g. "stage-1a") to
            replacement payloads. The pipeline skips that stage's computation and
            resumes execution from the overridden output.
        start_stage: First stage to execute in this run. Earlier stages are
            loaded from existing artifacts in data/{workspace_id}/run/.
        end_stage: Final stage to execute in this run. Useful for stage-specific
            development replays such as rerunning only stage 2.
        openrouter_access_mode: Effective OpenRouter access mode ("anonymous",
            "user", or "local") resolved by the web server for web-launched runs.
        openrouter_secret_ref: Single-use encrypted OpenRouter key ref created by
            the web server for production web-launched runs.
    """
    from nof1_causal_lab.flows.stage_registry import OpenRouterAccessMode

    resolved_openrouter_access_mode: OpenRouterAccessMode | None
    if openrouter_access_mode == "user":
        resolved_openrouter_access_mode = "user"
    elif openrouter_access_mode == "anonymous":
        resolved_openrouter_access_mode = "anonymous"
    elif openrouter_access_mode == "local":
        resolved_openrouter_access_mode = "local"
    elif openrouter_access_mode is None:
        resolved_openrouter_access_mode = None
    else:
        raise ValueError("openrouter_access_mode must be one of 'anonymous', 'user', or 'local'")

    if os.environ.get("DEPLOYMENT_ENV") == "production" and resolved_openrouter_access_mode not in {
        "anonymous",
        "user",
    }:
        raise ValueError("Production runs must set openrouter_access_mode to 'anonymous' or 'user'")

    openrouter_api_key: str | None = None
    if openrouter_secret_ref:
        if resolved_openrouter_access_mode not in {"user", "anonymous"}:
            raise ValueError("openrouter_access_mode must be 'user' or 'anonymous'")
        openrouter_api_key = consume_byok_secret_ref(openrouter_secret_ref)
        if openrouter_api_key is None:
            raise ValueError("Invalid or expired OpenRouter secret reference")
        logger.info(
            "Resolved %s OpenRouter API key from secret ref",
            resolved_openrouter_access_mode,
        )
    elif resolved_openrouter_access_mode == "local":
        logger.info("Using local OpenRouter credentials from the environment")
    elif resolved_openrouter_access_mode is not None:
        raise ValueError("openrouter_secret_ref is required for 'user' and 'anonymous' modes")

    from nof1_causal_lab.flows.stage_registry import (
        PipelineContext,
        get_execution_order,
        get_stage_registry,
        load_stage_state,
        run_stage_flow,
    )
    from nof1_causal_lab.utils.config import get_config

    config = get_config()
    registry = get_stage_registry()
    execution_order = get_execution_order()
    stage_index = {stage_id: idx for idx, stage_id in enumerate(execution_order)}
    question_stages = frozenset(
        stage_id for stage_id, defn in registry.items() if defn.question_required
    )
    overrideable_stages = frozenset(
        stage_id for stage_id, defn in registry.items() if defn.override_eligible
    )

    supported_overrides = _filter_stage_overrides(
        stage_overrides,
        overrideable_stages=overrideable_stages,
    )
    effective_start_stage, start_idx, effective_end_stage, end_idx = _resolve_stage_window(
        start_stage=start_stage,
        end_stage=end_stage,
        execution_order=execution_order,
        stage_index=stage_index,
    )
    question = _resolve_question(
        query=query,
        workspace_id=workspace_id,
        relevant_stage_ids=execution_order[start_idx : end_idx + 1],
        question_stages=question_stages,
    )

    lit_enabled = (
        enable_literature
        if enable_literature is not None
        else config.stage4_prior_elicitation.literature_search.enabled
    )

    logger.info(
        "Pipeline starting: workspace_id=%s source=%s access_mode=%s inference_method=%s "
        "literature=%s start_stage=%s end_stage=%s stage_overrides=%s",
        workspace_id,
        "raw text" if query else "resume/no-query",
        resolved_openrouter_access_mode or "implicit/default",
        inference_method or "config default",
        lit_enabled,
        effective_start_stage,
        effective_end_stage,
        sorted(supported_overrides),
    )
    if question:
        logger.info("Question preview: %s", _preview(question))

    prefect_run_id = get_current_flow_run_id()

    # Ensure the run directory exists
    storage.makedirs(runs_dir(workspace_id))

    ctx = PipelineContext(
        workspace_id=workspace_id,
        prefect_run_id=prefect_run_id,
        question=question,
        lit_enabled=lit_enabled,
        inference_method=inference_method,
        supported_overrides=supported_overrides,
        openrouter_api_key=openrouter_api_key,
        openrouter_access_mode=resolved_openrouter_access_mode,
    )

    stage_states: dict[str, Any] = {}

    async def _maybe_finish(stage_id: str) -> dict[str, Any] | None:
        if stage_id != effective_end_stage:
            return None
        if "stage-1b" in stage_states:
            await _emit_causal_spec_artifact(stage_states["stage-1b"])
        if stage_id == "stage-6":
            logger.info("Pipeline complete: run finished successfully")
            return {
                **stage_states["stage-5b"].model_dump(mode="json"),
                **stage_states["stage-6"].model_dump(mode="json"),
            }
        logger.info("Pipeline partial run complete: stopped after %s", stage_id)
        return _partial_pipeline_result(workspace_id, stage_id, stage_states[stage_id])

    for idx, stage_id in enumerate(execution_order):
        defn = registry[stage_id]

        if idx > end_idx:
            break

        if idx < start_idx:
            # Restore from prior run (stages before the execution window)
            if defn.skip_restore:
                continue
            restored = load_stage_state(workspace_id, stage_id)
            stage_states[stage_id] = restored
            emit_stage_progress_event(prefect_run_id, stage_id, "completed")
            fail_reason = _stage_fail_reason(restored)
            if fail_reason is not None:
                logger.info(
                    "Pipeline stopped at restored %s (fail_reason=%s)",
                    stage_id,
                    fail_reason,
                )
                return _partial_pipeline_result(workspace_id, stage_id, restored)
        else:
            # Execute this stage
            if defn.question_required and question is None:
                raise ValueError(f"Question is required to execute {stage_id}")

            logger.info(">>> %s starting", stage_id)
            t0 = time.monotonic()
            emit_stage_progress_event(prefect_run_id, stage_id, "running")
            try:
                state = await run_stage_flow(defn, ctx, stage_states)
            except Exception as exc:
                elapsed = time.monotonic() - t0
                logger.error(">>> %s FAILED after %.1fs: %s", stage_id, elapsed, exc)
                emit_stage_progress_event(
                    prefect_run_id,
                    stage_id,
                    "failed",
                    error={"type": "execution_error", "message": str(exc)},
                )
                raise
            elapsed = time.monotonic() - t0
            stage_outcome = state.outcome
            logger.info(">>> %s completed in %.1fs (outcome=%s)", stage_id, elapsed, stage_outcome)
            emit_stage_progress_event(prefect_run_id, stage_id, "completed", outcome=stage_outcome)
            stage_states[stage_id] = state
            fail_reason = _stage_fail_reason(state)
            if fail_reason is not None:
                logger.info(
                    "Pipeline stopped after %s (fail_reason=%s)",
                    stage_id,
                    fail_reason,
                )
                return _partial_pipeline_result(workspace_id, stage_id, state)

        partial = await _maybe_finish(stage_id)
        if partial is not None:
            return partial

    raise AssertionError("Unreachable: pipeline did not terminate at a stage boundary")


def build_main_deployment():
    return causal_inference_pipeline.to_deployment(
        name="causal-inference",
        tags=["causal", "llm"],
        enforce_parameter_schema=True,
    )


if __name__ == "__main__":
    from prefect import serve as serve_deployments

    serve_deployments(build_main_deployment())
