"""Main causal inference pipeline.

Prefect owns the execution graph for stage orchestration and monitoring.
Human-in-the-loop replay is handled by ``stage_overrides``: the replay route
can provide edited payloads for selected stages, and the pipeline skips those
stage computations while re-running downstream stages from the override point.
"""

from pathlib import Path
from typing import Any

from prefect import flow
from prefect.artifacts import create_markdown_artifact

from causal_ssm_agent.utils.data import load_query

from . import get_prefect_logger

logger = get_prefect_logger(__name__)

RESULT_STORAGE = Path("results")
OVERRIDABLE_STAGES = frozenset({"stage-1a", "stage-1b", "stage-4"})


def _preview(text: str, *, limit: int = 120) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3]}..."


def _get_stage_override(
    stage_overrides: dict[str, dict] | None,
    stage_id: str,
) -> dict[str, Any] | None:
    """Return a validated override payload for an editable stage."""
    if not stage_overrides:
        return None
    if stage_id not in OVERRIDABLE_STAGES:
        if stage_id in stage_overrides:
            logger.warning("Ignoring unsupported stage override: %s", stage_id)
        return None
    payload = stage_overrides.get(stage_id)
    if payload is None:
        return None
    logger.info("Using override payload for %s", stage_id)
    return payload


@flow(
    persist_result=True,
    result_storage=RESULT_STORAGE,
    result_serializer="pickle",
)
async def causal_inference_pipeline(
    query_file: str | None = None,
    user_id: str = "test_user",
    inference_method: str | None = None,
    enable_literature: bool | None = None,
    override_gates: bool | None = None,
    query: str | None = None,
    stage_overrides: dict[str, dict] | None = None,
    openrouter_api_key: str | None = None,
):
    """Run the causal pipeline end to end.

    Args:
        query_file: Filename in data/queries/ (e.g., 'procrastination-patterns')
        user_id: User subdirectory under data/raw/ (default: test_user)
        inference_method: Override inference method ("svi" or "nuts")
        enable_literature: Override literature search
        override_gates: Continue past stage failures instead of halting
        query: Raw query text (used by web UI). Takes precedence over query_file.
        stage_overrides: Dict mapping editable stage ids (e.g. "stage-1a") to
            replacement payloads. The pipeline skips that stage's computation and
            resumes execution from the overridden output.
        openrouter_api_key: User-provided OpenRouter API key (BYOK). Overrides the
            default key.
    """
    import os

    if openrouter_api_key:
        os.environ["OPENROUTER_API_KEY"] = openrouter_api_key
        logger.info("Using user-provided OpenRouter API key")

    from causal_ssm_agent.utils.config import get_config

    from . import dag

    config = get_config()

    if query:
        question = query.strip()
    elif query_file:
        question = load_query(query_file)
    else:
        raise ValueError("Either 'query' (raw text) or 'query_file' (filename) must be provided")

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
        "override_gates=%s stage_overrides=%s",
        user_id,
        "raw text" if query else query_file,
        inference_method or "config default",
        lit_enabled,
        gates_overridden,
        sorted(stage_overrides or {}),
    )
    logger.info("Question preview: %s", _preview(question))

    from prefect.context import get_run_context

    root_run_id = str(get_run_context().flow_run.id)
    run_dir = RESULT_STORAGE / root_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    stage0_state = await dag.stage0_flow(user_id, root_run_id)
    stage0_result = stage0_state["result"]

    stage1a_state = await dag.stage1a_flow(
        question,
        root_run_id,
        override_payload=_get_stage_override(stage_overrides, "stage-1a"),
    )
    stage1a_result = stage1a_state["result"]

    stage1b_state = await dag.stage1b_flow(
        question,
        stage0_result,
        stage1a_result,
        gates_overridden,
        root_run_id,
        override_payload=_get_stage_override(stage_overrides, "stage-1b"),
    )
    stage1b_result = stage1b_state["result"]
    stage1b_gate = stage1b_state["gate"]
    stage1b_web = stage1b_state["web"]

    stage2_state = await dag.stage2_flow(question, stage0_result, stage1b_result, root_run_id)
    stage2_result = stage2_state["result"]

    dag.stage3_flow(stage1b_result, stage2_result, root_run_id)

    stage4_state = await dag.stage4_flow(
        question,
        stage1b_result,
        stage2_result,
        lit_enabled,
        root_run_id,
        override_payload=_get_stage_override(stage_overrides, "stage-4"),
    )
    stage4_result = stage4_state["result"]

    dag.stage4b_flow(stage4_result, stage2_result, gates_overridden, root_run_id)

    stage5_state = dag.stage5_flow(
        stage4_result,
        stage1b_result,
        stage2_result,
        inference_method,
        root_run_id,
    )
    stage5_result = stage5_state["result"]
    stage5_web = stage5_state["web"]

    stage6_state = dag.stage6_flow(
        stage5_result,
        stage1a_result,
        stage1b_result,
        stage1b_gate,
        root_run_id,
    )
    stage6_web = stage6_state["web"]

    causal_spec = stage1b_web.get("causal_spec", {})
    latent = causal_spec.get("latent", {})
    measurement = causal_spec.get("measurement", {})
    create_markdown_artifact(
        key="causal-spec",
        markdown=(
            f"## Causal Specification\n\n"
            f"- **Constructs**: {len(latent.get('constructs', []))}\n"
            f"- **Edges**: {len(latent.get('edges', []))}\n"
            f"- **Indicators**: {len(measurement.get('indicators', []))}\n"
        ),
    )

    logger.info("Pipeline complete: run finished successfully")
    return {**stage5_web, **stage6_web}


if __name__ == "__main__":
    from prefect import serve as serve_deployments

    main_dep = causal_inference_pipeline.to_deployment(
        name="causal-inference",
        tags=["causal", "llm"],
    )
    serve_deployments(main_dep)
