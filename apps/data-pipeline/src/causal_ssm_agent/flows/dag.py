"""Stage computation functions for the causal inference pipeline.

Each function (stage0, stage1a, …, stage6) implements the core logic
for one pipeline stage and returns a typed contract instance.
Artifact persistence is handled by each runner (parquet, json, pickle).

Contract arguments (stage0, stage2, stage4) are present in runner signatures
because bind_inputs passes them — artifacts are loaded by workspace_id instead.
"""
# ruff: noqa: ARG001

from __future__ import annotations

from inspect import isawaitable
from pathlib import Path
from typing import Any, cast

from . import get_prefect_logger
from .run_store import (
    STAGE0_PARQUET_FILENAMES,
    STAGE2_MODEL_PARQUET_FILENAMES,
    STAGE4_COMPILED_SSM_FILENAMES,
    STAGE5B_PICKLE_FILENAMES,
    find_run_artifact,
    load_json,
    load_parquet,
    save_json,
    save_parquet,
    save_pickle,
    unwrap_task_result,
)
from .stage_contracts import (
    Stage0Contract,
    Stage1aContract,
    Stage1bContract,
    Stage2Contract,
    Stage3Contract,
    Stage4bContract,
    Stage4Contract,
    Stage5aContract,
    Stage5bContract,
    Stage6Contract,
)

logger = get_prefect_logger(__name__)


def _filter_to_contract(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    """Filter a dict to only the fields known by a contract class."""
    fields = set(cls.model_fields.keys())
    return {k: v for k, v in data.items() if k in fields}


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 0: Agentic data ingestion
# ═══════════════════════════════════════════════════════════════════════════════


async def stage0(workspace_id: str) -> Stage0Contract:
    """Agentic ingestion of raw data."""
    from .pipeline_helpers import build_stage0_payload
    from .stages.stage0.flow import agentic_ingest

    result = await agentic_ingest(workspace_id)
    save_parquet(result.dataframe, workspace_id, "stage0-raw-input.parquet")

    payload = build_stage0_payload(result)
    return Stage0Contract.model_validate(payload)


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1a: Latent model proposal
# ═══════════════════════════════════════════════════════════════════════════════


async def stage1a(question: str) -> Stage1aContract:
    """Propose theoretical constructs and causal edges (latent model)."""
    from .stages.stage1a.flow import propose_latent_model

    result = await propose_latent_model(question)
    return Stage1aContract.model_validate(result)


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1b: Measurement model + identifiability
# ═══════════════════════════════════════════════════════════════════════════════


async def stage1b(
    question: str,
    stage0: Stage0Contract,
    stage1a: Stage1aContract,
    workspace_id: str,
) -> Stage1bContract:
    """Propose measurement model and check identifiability."""
    from .pipeline_helpers import format_schema_for_llm
    from .stages.stage1b.flow import propose_measurement_with_identifiability_fix
    from .stages.stage1b.result import finalize_stage1b_result

    df_path = find_run_artifact(workspace_id, STAGE0_PARQUET_FILENAMES)
    ingested_df = load_parquet(df_path)
    column_descriptions = {c.name: c.description for c in stage0.column_descriptions}
    latent_model = stage1a.latent_model.model_dump()

    dataset_schema = format_schema_for_llm(ingested_df, column_descriptions)
    result = await propose_measurement_with_identifiability_fix(
        question,
        latent_model,
        [dataset_schema],
        dataset_summary=f"{ingested_df.shape[0]} rows x {ingested_df.shape[1]} columns",
    )
    finalized = finalize_stage1b_result(result, latent_model=latent_model)
    return Stage1bContract.model_validate(_filter_to_contract(Stage1bContract, finalized))


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2: Worker extraction (parallel, concurrency-limited)
# ═══════════════════════════════════════════════════════════════════════════════


async def stage2(
    question: str,
    stage0: Stage0Contract,
    stage1b: Stage1bContract,
    workspace_id: str,
    root_run_id: str | None = None,
    max_windows: int | None = None,
) -> Stage2Contract:
    """Extract indicator values from data using LLM workers."""
    from prefect.task_runners import ThreadPoolTaskRunner

    from causal_ssm_agent.utils.config import get_config

    from .stages.stage2.flow import materialize_stage2_outputs, stage2_extraction_flow

    config = get_config()
    causal_spec = stage1b.causal_spec.model_dump()
    raw_df_path = Path(find_run_artifact(workspace_id, STAGE0_PARQUET_FILENAMES))
    stage2_subflow = stage2_extraction_flow.with_options(
        task_runner=cast(
            "Any", ThreadPoolTaskRunner(max_workers=config.stage2_workers.max_concurrent_workers)
        )
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

    save_parquet(data_for_model, workspace_id, "stage2-model-data.parquet")

    llm_trace = stage2_result.get("llm_trace") if isinstance(stage2_result, dict) else None
    contract_data: dict[str, Any] = {"workers": worker_statuses}
    if llm_trace is not None:
        contract_data["llm_trace"] = llm_trace
    if n_observations == 0:
        contract_data["outcome"] = "fail"
        contract_data["fail_reason"] = "no_observations_extracted"
    return Stage2Contract.model_validate(contract_data)


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 3: Extraction validation
# ═══════════════════════════════════════════════════════════════════════════════


async def _await_artifact(artifact: Any) -> None:
    if isawaitable(artifact):
        await artifact


async def stage3(
    stage1b: Stage1bContract,
    stage2: Stage2Contract,
    workspace_id: str,
) -> Stage3Contract:
    """Audit extracted data: validation plus per-indicator empirical profiles."""
    from prefect.artifacts import create_table_artifact

    from .stages.stage3.flow import derive_validation_status, validate_extraction

    causal_spec = stage1b.causal_spec.model_dump()
    data_for_model_path = find_run_artifact(workspace_id, STAGE2_MODEL_PARQUET_FILENAMES)
    data_for_model = load_parquet(data_for_model_path)

    validation_task = validate_extraction(causal_spec, [data_for_model])
    audit_result = unwrap_task_result(validation_task)
    if not audit_result:
        raise RuntimeError(
            "Stage 3 validate_extraction returned an empty audit result; "
            "refusing to fabricate an is_valid=False report with empty indicators."
        )

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
    fail_reason = status["fail_reason"] if isinstance(status["fail_reason"], str) else None

    if not status["is_valid"]:
        logger.warning("Stage 3 validation errors detected:")
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

    report = {**audit_result, "outcome": outcome}
    if fail_reason is not None:
        report["fail_reason"] = fail_reason

    return Stage3Contract.model_validate(_filter_to_contract(Stage3Contract, report))


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 4: Model specification + prior elicitation
# ═══════════════════════════════════════════════════════════════════════════════


async def stage4(
    question: str,
    stage1b: Stage1bContract,
    stage2: Stage2Contract,
    stage3: Stage3Contract,
    enable_literature: bool,
    workspace_id: str,
    openrouter_api_key: str | None = None,
    root_run_id: str | None = None,
) -> Stage4Contract:
    """Propose model spec, elicit priors, and return the grounded stage-4 result."""
    from .stages.stage4.flow import stage4_agentic_flow

    causal_spec = stage1b.causal_spec.model_dump()
    data_for_model_path = find_run_artifact(workspace_id, STAGE2_MODEL_PARQUET_FILENAMES)
    data_for_model = load_parquet(data_for_model_path)

    result = await stage4_agentic_flow(
        causal_spec=causal_spec,
        question=question,
        data_for_model=data_for_model,
        indicator_audits={k: v.model_dump() for k, v in stage3.indicators.items()},
        enable_literature=enable_literature,
        workspace_id=workspace_id,
        openrouter_api_key=openrouter_api_key,
        root_run_id=root_run_id,
    )

    # Save compiled SSM artifact
    compiled_ssm = result.pop("_compiled_ssm", None)
    if compiled_ssm is not None:
        save_json(compiled_ssm, workspace_id, "stage4-compiled-ssm.json")

    # Determine outcome based on compilation success
    if compiled_ssm is not None:
        result["outcome"] = "success"
    else:
        result["outcome"] = "fail"
        result["fail_reason"] = "model_compile_failed"

    return Stage4Contract.model_validate(_filter_to_contract(Stage4Contract, result))


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 4b: Parametric identifiability
# ═══════════════════════════════════════════════════════════════════════════════


def _load_compiled_ssm(workspace_id: str) -> dict[str, Any] | None:
    """Load the compiled SSM artifact, or None if not found."""
    try:
        path = find_run_artifact(workspace_id, STAGE4_COMPILED_SSM_FILENAMES)
        return load_json(path)
    except FileNotFoundError:
        return None


def _load_data_for_model_path(workspace_id: str) -> str:
    """Resolve the data-for-model parquet path."""
    return find_run_artifact(workspace_id, STAGE2_MODEL_PARQUET_FILENAMES)


def stage4b(
    stage4: Stage4Contract,
    stage2: Stage2Contract,
    workspace_id: str,
    ssm_builder: Any = None,
    root_run_id: str | None = None,
) -> Stage4bContract:
    """Parametric identifiability diagnostics."""
    from .stages.stage4b.flow import run_stage4b

    compiled_ssm = _load_compiled_ssm(workspace_id)
    data_for_model_path = _load_data_for_model_path(workspace_id)

    # Bridge to internal flow that expects dicts
    result = run_stage4b(
        {"_compiled_ssm": compiled_ssm},
        {"_data_for_model_path": data_for_model_path},
        ssm_builder=ssm_builder,
        root_run_id=root_run_id,
    )

    return Stage4bContract.model_validate(_filter_to_contract(Stage4bContract, result))


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 5: Inference + diagnostics
# ═══════════════════════════════════════════════════════════════════════════════


def stage5a(
    stage4: Stage4Contract,
    stage2: Stage2Contract,
    workspace_id: str,
) -> Stage5aContract:
    """SVI preflight: fast approximate fit before expensive inference."""
    from .stages.stage5a.flow import run_stage5a_preflight

    compiled_ssm = _load_compiled_ssm(workspace_id)
    data_for_model_path = _load_data_for_model_path(workspace_id)

    result = run_stage5a_preflight(
        {"_compiled_ssm": compiled_ssm},
        {"_data_for_model_path": data_for_model_path},
        workspace_id,
    )

    return Stage5aContract.model_validate(_filter_to_contract(Stage5aContract, result))


def stage5b(
    stage4: Stage4Contract,
    stage2: Stage2Contract,
    workspace_id: str,
    inference_method: str | None = None,
) -> Stage5bContract:
    """Fit model, run power-scaling and posterior predictive checks."""
    from .stages.stage5b.flow import run_stage5b

    compiled_ssm = _load_compiled_ssm(workspace_id)
    data_for_model_path = _load_data_for_model_path(workspace_id)

    result = run_stage5b(
        {"_compiled_ssm": compiled_ssm},
        {"_data_for_model_path": data_for_model_path},
        inference_method,
        workspace_id,
    )

    # Save fitted artifact
    fitted_artifact = result.pop("_fitted_artifact", None)
    if fitted_artifact is not None:
        save_pickle(fitted_artifact, workspace_id, "stage5b-fitted-result.pkl")

    return Stage5bContract.model_validate(_filter_to_contract(Stage5bContract, result))


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 6: Intervention analysis
# ═══════════════════════════════════════════════════════════════════════════════


def _derive_identified_treatments(causal_spec: Any) -> list[str]:
    """Derive identified treatments from a CausalSpec contract."""
    from causal_ssm_agent.utils.causal_spec import get_estimable_treatments

    causal_spec_dict = (
        causal_spec.model_dump() if hasattr(causal_spec, "model_dump") else causal_spec
    )
    all_treatments = list(get_estimable_treatments(causal_spec_dict))
    non_id: dict[str, Any] = {}
    if hasattr(causal_spec, "identifiability") and causal_spec.identifiability:
        non_id = causal_spec.identifiability.non_identifiable_treatments or {}
    elif isinstance(causal_spec_dict, dict):
        non_id = (causal_spec_dict.get("identifiability") or {}).get(
            "non_identifiable_treatments", {}
        ) or {}
    return [t for t in all_treatments if t not in non_id]


async def stage6(
    stage5b: Stage5bContract,
    stage1b: Stage1bContract,
    workspace_id: str,
    question: str | None = None,
) -> Stage6Contract:
    """Run do-operator interventions and rank treatments."""
    from .stages.stage6.flow import run_stage6

    fitted_result_path = find_run_artifact(workspace_id, STAGE5B_PICKLE_FILENAMES)
    identified_treatments = _derive_identified_treatments(stage1b.causal_spec)

    # Bridge to internal flow that expects dicts
    stage5b_dict = {
        **stage5b.model_dump(),
        "_fitted_result_path": fitted_result_path,
    }
    stage1b_dict = {
        "causal_spec": stage1b.causal_spec.model_dump(),
        "_identified_treatments": identified_treatments,
    }
    result = await run_stage6(stage5b_dict, stage1b_dict, question=question)

    return Stage6Contract.model_validate(_filter_to_contract(Stage6Contract, result))
