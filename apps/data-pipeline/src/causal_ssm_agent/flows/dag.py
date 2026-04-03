"""Stage computation functions for the causal inference pipeline.

Each function (stage0, stage1a, …, stage6) implements the core logic
for one pipeline stage. Wrapper flows and artifact persistence are
handled by the stage registry (``stage_registry.py``).
"""

from __future__ import annotations

from inspect import isawaitable
from pathlib import Path
from typing import Any

from . import get_prefect_logger
from .run_store import (
    load_parquet,
    unwrap_task_result,
)

logger = get_prefect_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 0: Agentic data ingestion
# ═══════════════════════════════════════════════════════════════════════════════


async def stage0(workspace_id: str) -> dict:
    """Agentic ingestion of raw data.

    Returns dict with web-serializable fields PLUS internal data:
    - ``_df``: Polars DataFrame (not web-serializable)
    - ``_column_descriptions``: dict mapping col -> description
    """
    from .pipeline_helpers import build_stage0_payload
    from .stages.stage0.flow import agentic_ingest

    result = await agentic_ingest(workspace_id)
    df = result.dataframe

    payload = build_stage0_payload(result)
    return {
        **payload,
        "_df": df,
        "_column_descriptions": result.column_descriptions,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1a: Latent model proposal
# ═══════════════════════════════════════════════════════════════════════════════


async def stage1a(question: str) -> dict:
    """Propose theoretical constructs and causal edges (latent model).

    Returns: {latent_model, llm_trace?}
    """
    from .stages.stage1a.flow import propose_latent_model

    return await propose_latent_model(question)


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1b: Measurement model + identifiability
# ═══════════════════════════════════════════════════════════════════════════════


async def stage1b(
    question: str,
    stage0: dict,
    stage1a: dict,
) -> dict:
    """Propose measurement model and check identifiability.

    Returns: {causal_spec, measurement_model, identifiability_status, llm_trace?}
    """
    from .pipeline_helpers import format_schema_for_llm
    from .stages.stage1b.flow import propose_measurement_with_identifiability_fix
    from .stages.stage1b.result import finalize_stage1b_result

    ingested_df = load_parquet(stage0["_df_path"])
    column_descriptions = stage0["_column_descriptions"]
    latent_model = stage1a["latent_model"]

    dataset_schema = format_schema_for_llm(ingested_df, column_descriptions)
    result = await propose_measurement_with_identifiability_fix(
        question,
        latent_model,
        [dataset_schema],
        dataset_summary=f"{ingested_df.shape[0]} rows x {ingested_df.shape[1]} columns",
    )
    return finalize_stage1b_result(result, latent_model=latent_model)


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2: Worker extraction (parallel, concurrency-limited)
# ═══════════════════════════════════════════════════════════════════════════════


async def stage2(
    question: str,
    stage0: dict,
    stage1b: dict,
    root_run_id: str | None = None,
    max_windows: int | None = None,
) -> dict:
    """Extract indicator values from data using LLM workers.

    Returns dict with:
    - ``_data_for_model``: encoded DataFrame for modeling (non-continuous types → numeric)
    - plus web-serializable worker metadata
    """
    from prefect.task_runners import ThreadPoolTaskRunner

    from causal_ssm_agent.utils.config import get_config

    from .stages.stage2.flow import materialize_stage2_outputs, stage2_extraction_flow

    config = get_config()
    causal_spec = stage1b["causal_spec"]
    raw_df_path = Path(stage0["_df_path"])
    stage2_subflow = stage2_extraction_flow.with_options(
        task_runner=ThreadPoolTaskRunner(max_workers=config.stage2_workers.max_concurrent_workers)
    )
    stage2_result = await stage2_subflow(
        raw_df_path=str(raw_df_path),
        question=question,
        causal_spec=causal_spec,
        root_run_id=root_run_id,
        max_windows=max_windows,
    )

    materialized = materialize_stage2_outputs(stage2_result, causal_spec)
    data_for_model = materialized["data_for_model"]
    worker_statuses = materialized["worker_statuses"]

    n_observations = len(data_for_model)
    n_unique_indicators = data_for_model["indicator"].n_unique() if n_observations > 0 else 0
    logger.info(
        "Extracted %d observation rows across %d indicators",
        n_observations,
        n_unique_indicators,
    )

    result = {
        "_data_for_model": data_for_model,
        "workers": worker_statuses,
    }
    if "llm_trace" in stage2_result:
        result["llm_trace"] = stage2_result["llm_trace"]
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 3: Extraction validation
# ═══════════════════════════════════════════════════════════════════════════════


async def _await_artifact(artifact: Any) -> None:
    if isawaitable(artifact):
        await artifact


async def stage3(stage1b: dict, stage2: dict) -> dict:
    """Audit extracted data: validation plus per-indicator empirical profiles.

    Returns: {is_valid, indicators, dataset_issues, outcome}
    """
    from prefect.artifacts import create_table_artifact

    from .stages.stage3.flow import derive_validation_status, validate_extraction

    causal_spec = stage1b["causal_spec"]
    data_for_model = load_parquet(stage2["_data_for_model_path"])

    validation_task = validate_extraction(causal_spec, [data_for_model])
    audit_result = unwrap_task_result(validation_task)
    outcome = "fail"
    fail_reason: str | None = "data_validation_failed"

    if audit_result:
        indicator_issues = [
            issue
            for audit in audit_result.get("indicators", {}).values()
            for issue in audit.get("validation", {}).get("issues", [])
        ]
        dataset_issues = audit_result.get("dataset_issues", [])
        all_issues = [*indicator_issues, *dataset_issues]
        status = derive_validation_status(all_issues)
        audit_result = {**audit_result, "is_valid": status["is_valid"]}
        outcome = status["outcome"]
        fail_reason = status["fail_reason"]

        if not status["is_valid"]:
            logger.warning("Stage 3 validation errors detected:")
            for issue in all_issues:
                logger.warning(
                    "    - %s: %s (%s) %s",
                    issue.get("indicator") or "dataset",
                    issue["issue_type"],
                    issue["severity"],
                    issue["message"],
                )
        elif all_issues:
            logger.warning("Stage 3 validation warnings:")
            for issue in all_issues:
                logger.warning(
                    "    - %s: %s (%s) %s",
                    issue.get("indicator") or "dataset",
                    issue["issue_type"],
                    issue["severity"],
                    issue["message"],
                )

        if all_issues:
            await _await_artifact(
                create_table_artifact(
                    key="validation-issues",
                    table=[
                        {
                            "indicator": i.get("indicator") or "dataset",
                            "type": i["issue_type"],
                            "severity": i["severity"],
                            "message": i["message"],
                        }
                        for i in all_issues
                    ],
                    description="Stage 3 extraction validation issues",
                )
            )

    report = audit_result or {
        "is_valid": False,
        "indicators": {},
        "dataset_issues": [],
    }

    return {**report, "outcome": outcome, "fail_reason": fail_reason}


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 4: Model specification + prior elicitation
# ═══════════════════════════════════════════════════════════════════════════════


async def stage4(
    question: str,
    stage1b: dict,
    stage2: dict,
    stage3: dict,
    enable_literature: bool,
    workspace_id: str | None = None,
    openrouter_api_key: str | None = None,
    root_run_id: str | None = None,
) -> dict:
    """Propose model spec, elicit priors, and return the grounded stage-4 result."""
    from .stages.stage4.flow import stage4_agentic_flow

    causal_spec = stage1b["causal_spec"]
    data_for_model = load_parquet(stage2["_data_for_model_path"])

    return await stage4_agentic_flow(
        causal_spec=causal_spec,
        question=question,
        data_for_model=data_for_model,
        indicator_audits=stage3["indicators"],
        enable_literature=enable_literature,
        workspace_id=workspace_id,
        openrouter_api_key=openrouter_api_key,
        root_run_id=root_run_id,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 4b: Parametric identifiability
# ═══════════════════════════════════════════════════════════════════════════════


def stage4b(
    stage4: dict,
    stage2: dict,
    ssm_builder: Any = None,
    root_run_id: str | None = None,
) -> dict:
    """Parametric identifiability diagnostics.

    Returns: {parametric_id, inference_structure, outcome}
    """
    from .stages.stage4b.flow import run_stage4b

    return run_stage4b(stage4, stage2, ssm_builder=ssm_builder, root_run_id=root_run_id)


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 5: Inference + diagnostics
# ═══════════════════════════════════════════════════════════════════════════════


def stage5a(
    stage4: dict,
    stage2: dict,
) -> dict:
    """SVI preflight: fast approximate fit before expensive inference."""
    from .stages.stage5a.flow import run_stage5a_preflight

    return run_stage5a_preflight(stage4, stage2)


def stage5b(
    stage4: dict,
    stage2: dict,
    inference_method: str | None,
) -> dict:
    """Fit model, run power-scaling and posterior predictive checks.

    Returns: {_fitted_artifact, power_scaling, ppc,
              inference_metadata, mcmc_diagnostics, svi_diagnostics, smc_diagnostics,
              loo_diagnostics, posterior_marginals, posterior_pairs, outcome}
    """
    from .stages.stage5b.flow import run_stage5b

    return run_stage5b(stage4, stage2, inference_method)


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 6: Intervention analysis
# ═══════════════════════════════════════════════════════════════════════════════


async def stage6(
    stage5b: dict,
    stage1b: dict,
    question: str | None = None,
) -> dict:
    """Run do-operator interventions and rank treatments.

    Returns: {intervention_results, outcome}
    """
    from .stages.stage6.flow import run_stage6

    return await run_stage6(stage5b, stage1b, question=question)
