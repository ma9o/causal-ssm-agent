"""Main causal inference pipeline.

Orchestrates all stages from agentic ingestion to intervention analysis
using a Hamilton DAG for dataflow + Prefect for scheduling/monitoring.

The Hamilton DAG (``dag.py``) defines pure-ish node functions whose
parameter names encode data dependencies. The Prefect flow here
creates an ``AsyncDriver``, passes inputs + optional overrides, and
requests all web-persistence nodes as ``final_vars``.

Human-in-the-loop replay is handled by Hamilton's native ``overrides``
parameter: the replay route sends back the edited stage payload as
an override dict, and Hamilton skips that node's computation while
re-running all downstream nodes with the new data.
"""

import logging
from pathlib import Path

from prefect import flow
from prefect.artifacts import create_markdown_artifact

from causal_ssm_agent.utils.data import load_query

logger = logging.getLogger(__name__)

RESULT_STORAGE = Path("results")

# Hamilton node name → web stage id
NODE_TO_STAGE: dict[str, str] = {
    "stage0": "stage-0",
    "stage1a": "stage-1a",
    "stage1b": "stage-1b",
    "stage4": "stage-4",
}

# Web stage id → Hamilton node name (for replay route)
STAGE_TO_NODE: dict[str, str] = {v: k for k, v in NODE_TO_STAGE.items()}

# All web-persistence nodes that must execute
WEB_NODES = [
    "stage0_web",
    "stage1a_web",
    "stage1b_web",
    "stage2_web",
    "stage3_web",
    "stage4_web",
    "stage4b_web",
    "stage5_web",
    "stage6_web",
]


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
    """Main causal inference pipeline.

    Args:
        query_file: Filename in data/queries/ (e.g., 'procrastination-patterns')
        user_id: User subdirectory under data/raw/ (default: test_user)
        inference_method: Override inference method ("svi" or "nuts")
        enable_literature: Override literature search
        override_gates: Continue past stage failures instead of halting
        query: Raw query text (used by web UI). Takes precedence over query_file.
        stage_overrides: Dict mapping stage ids (e.g. "stage-1a") to override
            payloads. Translated to Hamilton ``overrides`` by node name.
        openrouter_api_key: User-provided OpenRouter API key (BYOK). Overrides the default key.
    """
    import os

    from hamilton import async_driver

    if openrouter_api_key:
        os.environ["OPENROUTER_API_KEY"] = openrouter_api_key
        logger.info("Using user-provided OpenRouter API key")
    from causal_ssm_agent.utils.config import get_config

    from . import dag

    config = get_config()

    # ── Resolve parameters ──
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

    logger.info("Query source: %s", "raw text" if query else query_file)
    logger.info("Question: %s", f"{question[:100]}..." if len(question) > 100 else question)

    # ── Materialize run directory for replay artifacts ──
    from prefect.context import get_run_context

    run_dir = RESULT_STORAGE / str(get_run_context().flow_run.id)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Build Hamilton overrides from stage_overrides ──
    hamilton_overrides: dict = {}
    if stage_overrides:
        for stage_id, payload in stage_overrides.items():
            node_name = STAGE_TO_NODE.get(stage_id)
            if node_name:
                hamilton_overrides[node_name] = payload
                logger.info("Override: %s → Hamilton node '%s'", stage_id, node_name)
            else:
                logger.warning("Unknown stage_id in overrides: %s (ignored)", stage_id)

    # ── Build and execute Hamilton DAG ──
    dr = await async_driver.Builder().with_modules(dag).build()

    inputs = {
        "question": question,
        "user_id": user_id,
        "override_gates": gates_overridden,
        "enable_literature": lit_enabled,
        "inference_method": inference_method,
    }

    result = await dr.execute(
        final_vars=WEB_NODES,
        overrides=hamilton_overrides or {},
        inputs=inputs,
    )

    # ── Causal spec artifact ──
    stage1b_web = result.get("stage1b_web", {})
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

    # ── Assemble final return (stage 5 + 6 web data) ──
    stage5_web = result.get("stage5_web", {})
    stage6_web = result.get("stage6_web", {})
    return {**stage5_web, **stage6_web}


if __name__ == "__main__":
    from prefect import serve as serve_deployments

    main_dep = causal_inference_pipeline.to_deployment(
        name="causal-inference",
        tags=["causal", "llm"],
    )
    serve_deployments(main_dep)
