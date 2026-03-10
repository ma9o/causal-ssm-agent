"""Stage helpers and wrapper flows for the causal inference pipeline.

This module keeps the stage-level computation, stage result persistence, and
user-facing stage subflows in one place. The top-level pipeline flow invokes
the stage wrapper flows so the UI can track canonical stage executions instead
of persistence-only plumbing tasks.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import cloudpickle
from prefect import flow

from . import get_prefect_logger

logger = get_prefect_logger(__name__)

RESULT_STORAGE = Path("results")


def _run_dir(run_id: str) -> Path:
    path = RESULT_STORAGE / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_parquet(df: Any, run_id: str, filename: str) -> str:
    path = _run_dir(run_id) / filename
    df.write_parquet(path)
    return str(path)


def _load_parquet(path: str) -> Any:
    import polars as pl

    return pl.read_parquet(path)


def _save_pickle(value: Any, run_id: str, filename: str) -> str:
    path = _run_dir(run_id) / filename
    with path.open("wb") as f:
        cloudpickle.dump(value, f)
    return str(path)


def _load_pickle(path: str) -> Any:
    with Path(path).open("rb") as f:
        return cloudpickle.load(f)


def _validation_issue_counts(report: dict[str, Any]) -> tuple[int, int]:
    issues = report.get("issues", []) or []
    error_count = sum(1 for issue in issues if issue.get("severity") == "error")
    warning_count = sum(1 for issue in issues if issue.get("severity") == "warning")
    return error_count, warning_count


def _public_stage_payload(stage_result: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in stage_result.items() if not k.startswith("_")}


def _public_web(stage_id: str, stage_result: dict[str, Any], run_id: str) -> dict[str, Any]:
    return _persist_stage_payload(stage_id, _public_stage_payload(stage_result), run_id)


def _build_web_payload(source: dict[str, Any], **field_map: str) -> dict[str, Any]:
    return {target: source.get(source_key) for target, source_key in field_map.items()}


def _persist_stage_payload(stage_id: str, web_data: dict[str, Any], run_id: str) -> dict[str, Any]:
    from .stages import persist_web_result

    return persist_web_result(stage_id, web_data, run_id)


def _stage_state(
    result: dict[str, Any],
    web: dict[str, Any],
    gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = {"result": result, "web": web}
    if gate is not None:
        state["gate"] = gate
    return state


def _raise_if_gate_failed(gate_result: dict[str, Any], message: str) -> None:
    if gate_result["gate_failed"] and not gate_result["gate_overridden"]:
        raise RuntimeError(message)


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 0: Agentic data ingestion
# ═══════════════════════════════════════════════════════════════════════════════


async def stage0(user_id: str) -> dict:
    """Agentic ingestion of raw data.

    Returns dict with web-serializable fields PLUS internal data:
    - ``_df``: Polars DataFrame (not web-serializable)
    - ``_column_descriptions``: dict mapping col -> description
    """
    from .pipeline_helpers import build_stage0_payload
    from .stages import agentic_ingest

    result = await agentic_ingest(user_id)
    df = result.dataframe

    payload = build_stage0_payload(result, df)
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

    Returns: {latent_model, outcome_name, treatments, llm_trace?}
    """
    from .stages import propose_latent_model

    return await propose_latent_model(question)


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1b: Measurement model + identifiability
# ═══════════════════════════════════════════════════════════════════════════════


async def stage1b(question: str, stage0: dict, stage1a: dict) -> dict:
    """Propose measurement model and check identifiability.

    Returns: {causal_spec, measurement_model, identifiability_status, llm_trace?}
    """
    from .pipeline_helpers import format_schema_for_llm
    from .stages import propose_measurement_with_identifiability_fix

    ingested_df = _load_parquet(stage0["_df_path"])
    column_descriptions = stage0["_column_descriptions"]
    latent_model = stage1a["latent_model"]

    dataset_schema = format_schema_for_llm(ingested_df, column_descriptions)
    return await propose_measurement_with_identifiability_fix(
        question,
        latent_model,
        [dataset_schema],
        dataset_summary=f"{ingested_df.shape[0]} rows x {ingested_df.shape[1]} columns",
    )


def stage1b_gate(stage1a: dict, stage1b: dict, override_gates: bool) -> dict:
    """Filter treatments by identifiability and check gate.

    Returns: {treatments, gate_failed, gate_overridden, web_outcome}
    """
    treatments = list(stage1a.get("treatments", []))
    outcome = stage1a.get("outcome_name", "")
    causal_spec = stage1b.get("causal_spec", {})
    identifiability = causal_spec.get("identifiability", {}) or {}
    non_identifiable = identifiability.get("non_identifiable_treatments", {})

    gate_failed = False
    if non_identifiable:
        logger.warning("NON-IDENTIFIABLE TREATMENT EFFECTS (excluded from analysis):")
        for treatment in sorted(non_identifiable.keys()):
            details = non_identifiable[treatment]
            blockers = details.get("confounders", []) if isinstance(details, dict) else []
            notes = details.get("notes") if isinstance(details, dict) else None
            if blockers:
                logger.warning(
                    "  - %s → %s (blocked by: %s)", treatment, outcome, ", ".join(blockers)
                )
            elif notes:
                logger.warning("  - %s → %s (%s)", treatment, outcome, notes)
            else:
                logger.warning("  - %s → %s", treatment, outcome)
        treatments = [t for t in treatments if t not in non_identifiable]
        logger.info("Continuing with %d identifiable treatments", len(treatments))
        if not treatments:
            gate_failed = True

    gate_overridden = override_gates and gate_failed
    if gate_overridden:
        logger.warning("GATE 1b OVERRIDDEN: No identifiable treatments, continuing with empty list")

    web_outcome = "fail" if non_identifiable else "success"

    return {
        "treatments": treatments,
        "gate_failed": gate_failed,
        "gate_overridden": gate_overridden,
        "web_outcome": web_outcome,
        "non_identifiable": non_identifiable,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2: Worker extraction (parallel, concurrency-limited)
# ═══════════════════════════════════════════════════════════════════════════════


async def stage2(question: str, stage0: dict, stage1b: dict) -> dict:
    """Extract indicator values from data using LLM workers.

    Returns dict with:
    - ``_raw_data``: long-format Polars DataFrame
    - ``_data_for_model``: aggregated DataFrame for modeling
    - ``_worker_statuses``: per-worker status list
    - plus web-serializable fields
    """
    import polars as pl
    from prefect.task_runners import ThreadPoolTaskRunner

    from causal_ssm_agent.utils.aggregations import (
        aggregate_worker_measurements,
        flatten_aggregated_data,
    )
    from causal_ssm_agent.utils.config import get_config

    from .stages import stage2_extraction_flow

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
    )

    # Reconstruct raw_data DataFrame from worker results
    raw_data_dicts = stage2_result.get("raw_data", [])
    if raw_data_dicts:
        raw_data = pl.DataFrame(
            raw_data_dicts,
            schema={"indicator": pl.Utf8, "value": pl.Utf8, "timestamp": pl.Utf8},
        )
    else:
        raw_data = pl.DataFrame(
            schema={"indicator": pl.Utf8, "value": pl.Utf8, "timestamp": pl.Utf8}
        )

    n_observations = len(raw_data)
    n_unique_indicators = raw_data["indicator"].n_unique() if n_observations > 0 else 0
    logger.info(
        "Extracted %d observations across %d indicators", n_observations, n_unique_indicators
    )

    # Aggregate to pipeline-level granularity
    worker_dfs = [raw_data]
    aggregated_result = aggregate_worker_measurements(worker_dfs, causal_spec)
    if aggregated_result:
        data_for_model = flatten_aggregated_data(aggregated_result)
        logger.info(
            "  Aggregated to %d observations across %s granularities",
            len(data_for_model),
            list(aggregated_result.keys()),
        )
    else:
        data_for_model = raw_data
        logger.info("  No aggregation applied (using raw data)")

    # Build web payload
    sample_rows = raw_data.head(20).to_dicts() if n_observations > 0 else []
    per_ind_counts = (
        dict(raw_data.group_by("indicator").len().iter_rows()) if n_observations > 0 else {}
    )
    combined_extractions_sample = [
        {
            "indicator": str(row.get("indicator", "")),
            "value": row.get("value"),
            "timestamp": (str(row.get("timestamp")) if row.get("timestamp") is not None else None),
        }
        for row in sample_rows
    ]

    worker_statuses = stage2_result.get("worker_statuses", [])

    return {
        "_raw_data": raw_data,
        "_data_for_model": data_for_model,
        "_worker_statuses": worker_statuses,
        "workers": worker_statuses,
        "combined_extractions_sample": combined_extractions_sample,
        "per_indicator_counts": per_ind_counts,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 3: Extraction validation
# ═══════════════════════════════════════════════════════════════════════════════


def stage3(stage1b: dict, stage2: dict) -> dict:
    """Validate semantic properties of extracted data.

    Returns: {validation_report, outcome}
    """
    from prefect.artifacts import create_table_artifact

    from .stages import validate_extraction

    causal_spec = stage1b["causal_spec"]
    raw_data = _load_parquet(stage2["_raw_data_path"])

    validation_task = validate_extraction(causal_spec, [raw_data])
    validation_report = (
        validation_task.result() if hasattr(validation_task, "result") else validation_task
    )

    if validation_report:
        issues = validation_report.get("issues", [])
        if not validation_report.get("is_valid", True):
            logger.warning("Stage 3 validation errors detected:")
            for issue in issues:
                logger.warning(
                    "    - %s: %s (%s) %s",
                    issue["indicator"],
                    issue["issue_type"],
                    issue["severity"],
                    issue["message"],
                )
        elif issues:
            logger.warning("Stage 3 validation warnings:")
            for issue in issues:
                logger.warning(
                    "    - %s: %s (%s) %s",
                    issue["indicator"],
                    issue["issue_type"],
                    issue["severity"],
                    issue["message"],
                )

        if issues:
            create_table_artifact(
                key="validation-issues",
                table=[
                    {
                        "indicator": i["indicator"],
                        "type": i["issue_type"],
                        "severity": i["severity"],
                        "message": i["message"],
                    }
                    for i in issues
                ],
                description="Stage 3 extraction validation issues",
            )

    report = validation_report or {
        "is_valid": False,
        "issues": [],
        "per_indicator_health": [],
    }
    if not report.get("is_valid", True):
        outcome = "fail"
    elif any(i.get("severity") in ("warning", "error") for i in report.get("issues", [])):
        outcome = "warn"
    else:
        outcome = "success"

    return {"validation_report": report, "outcome": outcome}


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 4: Model specification + prior elicitation
# ═══════════════════════════════════════════════════════════════════════════════


async def stage4(
    question: str,
    stage1b: dict,
    stage2: dict,
    enable_literature: bool,
) -> dict:
    """Propose model spec, elicit priors, validate.

    Returns: {model_spec, priors, validation, model_info, causal_spec,
              prior_predictive_samples, llm_trace?}
    """
    from .stages import stage4_orchestrated_flow

    causal_spec = stage1b["causal_spec"]
    data_for_model = _load_parquet(stage2["_data_for_model_path"])

    return await stage4_orchestrated_flow(
        causal_spec=causal_spec,
        question=question,
        raw_data=data_for_model,
        enable_literature=enable_literature,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SSM builder (shared between stage 4b, 5)
# ═══════════════════════════════════════════════════════════════════════════════


def ssm_builder(stage4: dict, stage1b: dict, stage2: dict) -> Any:
    """Pre-build SSMModelBuilder once for downstream stages."""
    from causal_ssm_agent.models.ssm_builder import build_ssm_builder

    try:
        return build_ssm_builder(
            model_spec=stage4["model_spec"],
            priors=stage4["priors"],
            raw_data=_load_parquet(stage2["_data_for_model_path"]),
            causal_spec=stage1b["causal_spec"],
        )
    except Exception:
        logger.warning(
            "Pre-building SSM builder failed; stages will build their own", exc_info=True
        )
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 4b: Parametric identifiability
# ═══════════════════════════════════════════════════════════════════════════════


def stage4b(stage4: dict, stage2: dict, ssm_builder: Any = None) -> dict:
    """Parametric identifiability diagnostics.

    Returns: {parametric_id, rb_partition, ...stage4 passthrough}
    """
    from .stages import stage4b_parametric_id_flow

    return stage4b_parametric_id_flow(
        stage4,
        raw_data=_load_parquet(stage2["_data_for_model_path"]),
        builder=ssm_builder,
    )


def stage4b_gate(stage4b: dict, override_gates: bool) -> dict:
    """Check parametric identifiability gate."""
    param_id = stage4b.get("parametric_id", {})
    gate_failed = False
    gate_overridden = False
    t_rule: dict = {}

    if param_id.get("checked", False):
        t_rule = param_id.get("t_rule", {})
        if not t_rule.get("satisfies", True):
            gate_failed = True
            if override_gates:
                logger.warning(
                    "GATE 4b OVERRIDDEN: T-rule violated (%s free params > %s moments), continuing",
                    t_rule.get("n_free_params"),
                    t_rule.get("n_moments"),
                )
                gate_overridden = True
        summary = param_id.get("summary", {})
        if summary.get("structural_issues"):
            logger.warning(
                "STRUCTURAL non-identifiability detected — some parameters unconstrained"
            )
        elif summary.get("boundary_issues"):
            logger.warning("Boundary identifiability issues at some prior draws")
        else:
            logger.info("Parametric identifiability OK")
        weak = summary.get("weak_params", [])
        if weak:
            logger.info("  Weak parameters (low contraction): %s", weak)
    else:
        logger.info("  Skipped: %s", param_id.get("error", "unknown"))

    # Compute outcome
    if gate_failed:
        outcome = "fail"
    elif param_id.get("checked", False):
        summary = param_id.get("summary", {})
        has_issues = (
            summary.get("structural_issues")
            or summary.get("boundary_issues")
            or summary.get("weak_params")
        )
        outcome = "warn" if has_issues else "success"
    else:
        outcome = "success"

    return {
        "gate_failed": gate_failed,
        "gate_overridden": gate_overridden,
        "outcome": outcome,
        "t_rule": t_rule,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 5: Inference + diagnostics
# ═══════════════════════════════════════════════════════════════════════════════


def stage5(
    stage4: dict,
    stage1b: dict,
    stage2: dict,
    inference_method: str | None,
) -> dict:
    """Fit model, run power-scaling and posterior predictive checks.

    Returns: {ps_result, ppc_result, fitted_result, inference_metadata,
              mcmc_diagnostics, svi_diagnostics, loo_diagnostics,
              posterior_marginals, posterior_pairs, ps_list, outcome}
    """
    from causal_ssm_agent.utils.config import get_config

    from .stages import fit_model, run_power_scaling, run_ppc

    config = get_config()
    data_for_model = _load_parquet(stage2["_data_for_model_path"])
    causal_spec = stage1b["causal_spec"]

    sampler_config = (
        config.inference.to_sampler_config(method_override=inference_method)
        if inference_method
        else None
    )

    if config.inference.gpu:
        from .gpu_inference import run_stage5_gpu

        logger.info("Dispatching to Modal (%s GPU)...", config.inference.gpu)
        gpu_result = run_stage5_gpu(
            stage4_result=stage4,
            raw_data=data_for_model,
            sampler_config=sampler_config,
            treatments=[],  # filled by stage6
            outcome="",
            causal_spec=causal_spec,
            gpu=config.inference.gpu,
        )
        ps_result = gpu_result["ps_result"]
        ppc_result = gpu_result.get("ppc_result", {"checked": False})
        fitted_result = gpu_result
        mcmc_diagnostics = gpu_result.get("mcmc_diagnostics")
        svi_diagnostics = gpu_result.get("svi_diagnostics")
        loo_diagnostics = gpu_result.get("loo_diagnostics")
        posterior_marginals = gpu_result.get("posterior_marginals")
        posterior_pairs = gpu_result.get("posterior_pairs")
        inf_method = (
            gpu_result.get("mcmc_diagnostics", {}).get("method", "unknown")
            if gpu_result.get("mcmc_diagnostics")
            else "unknown"
        )
    else:
        fitted = fit_model(stage4, data_for_model, sampler_config=sampler_config, builder=None)
        fitted_result = fitted.result() if hasattr(fitted, "result") else fitted

        power_scaling = run_power_scaling(fitted_result, data_for_model)
        ps_result = power_scaling.result() if hasattr(power_scaling, "result") else power_scaling

        ppc_task = run_ppc(fitted_result, data_for_model)
        ppc_result = ppc_task.result() if hasattr(ppc_task, "result") else ppc_task

        mcmc_diagnostics = fitted_result.get("mcmc_diagnostics")
        svi_diagnostics = fitted_result.get("svi_diagnostics")
        loo_diagnostics = fitted_result.get("loo_diagnostics")
        posterior_marginals = fitted_result.get("posterior_marginals")
        posterior_pairs = fitted_result.get("posterior_pairs")
        inf_method = fitted_result.get("inference_type", "unknown")

    # Log power-scaling results
    logger.info("--- Power-Scaling Sensitivity ---")
    if ps_result.get("checked", False):
        diagnosis = ps_result.get("diagnosis", {})
        prior_dominated = [k for k, v in diagnosis.items() if v == "prior_dominated"]
        conflicts = [k for k, v in diagnosis.items() if v == "prior_data_conflict"]
        if prior_dominated:
            logger.warning("  Prior-dominated parameters: %s", prior_dominated)
        if conflicts:
            logger.warning("  Prior-data conflicts: %s", conflicts)
        if not prior_dominated and not conflicts:
            logger.info("  All parameters well-identified")
    else:
        logger.info("  Skipped: %s", ps_result.get("error", "unknown"))

    # Log PPC results
    logger.info("--- Posterior Predictive Checks ---")
    if ppc_result.get("checked", False):
        ppc_warnings = ppc_result.get("per_variable_warnings", [])
        if ppc_warnings:
            logger.warning("  %d warning(s):", len(ppc_warnings))
            for w in ppc_warnings:
                logger.warning("    - %s: %s", w["variable"], w["message"])
        else:
            logger.info("  All checks passed")
    else:
        logger.info("  Skipped: %s", ppc_result.get("error", "unknown"))

    # Reshape power-scaling into per-param list for web
    ps_list = []
    if ps_result.get("checked", False):
        diag = ps_result.get("diagnosis", {})
        prior_s = ps_result.get("prior_sensitivity", {})
        lik_s = ps_result.get("likelihood_sensitivity", {})
        psis_k = ps_result.get("psis_k_hat", {})
        for param in diag:
            entry = {
                "parameter": param,
                "diagnosis": diag[param],
                "prior_sensitivity": prior_s.get(param, 0.0),
                "likelihood_sensitivity": lik_s.get(param, 0.0),
            }
            if param in psis_k:
                entry["psis_k_hat"] = psis_k[param]
            ps_list.append(entry)

    has_ppc_warnings = bool(ppc_result.get("per_variable_warnings"))
    has_ps_issues = any(
        e["diagnosis"] in ("prior_dominated", "prior_data_conflict") for e in ps_list
    )
    outcome = "warn" if (has_ppc_warnings or has_ps_issues) else "success"

    return {
        "_fitted_result": fitted_result,
        "ps_result": ps_result,
        "ppc_result": ppc_result,
        "ps_list": ps_list,
        "inference_metadata": {
            "method": inf_method,
            "n_samples": 10000,
            "duration_seconds": 0.0,
        },
        "mcmc_diagnostics": mcmc_diagnostics,
        "svi_diagnostics": svi_diagnostics,
        "loo_diagnostics": loo_diagnostics,
        "posterior_marginals": posterior_marginals,
        "posterior_pairs": posterior_pairs,
        "outcome": outcome,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 6: Intervention analysis
# ═══════════════════════════════════════════════════════════════════════════════


def stage6(
    stage5: dict,
    stage1a: dict,
    stage1b: dict,
    stage1b_gate: dict,
) -> dict:
    """Run do-operator interventions and rank treatments.

    Returns: {intervention_results, outcome}
    """
    from prefect.artifacts import create_table_artifact

    from .stages import run_interventions

    fitted_result = _load_pickle(stage5["_fitted_result_path"])
    treatments = stage1b_gate["treatments"]
    outcome_name = stage1a.get("outcome_name", "")
    causal_spec = stage1b["causal_spec"]
    ppc_result = stage5["ppc_result"]
    ps_result = stage5["ps_result"]

    logger.info("=== Stage 6: Treatment Effects ===")
    logger.info("Estimating effects of %d treatments on %s", len(treatments), outcome_name)

    results = run_interventions(
        fitted_result, treatments, outcome_name, causal_spec, ppc_result, ps_result=ps_result
    )
    intervention_results = results.result() if hasattr(results, "result") else results

    # Log ranked results
    if intervention_results:
        logger.info("%-5s %-30s %10s %8s %4s", "Rank", "Treatment", "Effect", "P(>0)", "ID")
        logger.info("-" * 59)
        for rank, entry in enumerate(intervention_results, 1):
            name = entry["treatment"]
            effect = entry.get("effect_size")
            prob = entry.get("prob_positive")
            ident = "yes" if entry.get("identifiable", True) else "NO"
            if effect is not None:
                prob_str = f"{prob:.2f}" if prob is not None else ""
                line = f"{rank:<5} {name:<30} {effect:>+10.4f} {prob_str:>8} {ident:>4}"
                if entry.get("prior_sensitivity_warning"):
                    line += "  *"
                logger.info(line)
            else:
                warning = entry.get("warning", "no estimate")
                logger.info("%-5d %-30s %10s %8s %4s  (%s)", rank, name, "—", "", ident, warning)

        create_table_artifact(
            key="treatment-ranking",
            table=[
                {
                    "rank": i + 1,
                    "treatment": r["treatment"],
                    "effect": (
                        f"{r['effect_size']:+.4f}" if r.get("effect_size") is not None else "---"
                    ),
                    "P(>0)": (
                        f"{r['prob_positive']:.2f}" if r.get("prob_positive") is not None else ""
                    ),
                    "identifiable": "yes" if r.get("identifiable", True) else "NO",
                }
                for i, r in enumerate(intervention_results)
            ],
            description="Final treatment effect ranking",
        )

    has_warnings = any(
        r.get("ppc_warnings") or r.get("prior_sensitivity_warning") for r in intervention_results
    )

    return {
        "intervention_results": intervention_results,
        "outcome": "warn" if has_warnings else "success",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# User-facing stage wrapper flows
# ═══════════════════════════════════════════════════════════════════════════════


@flow(name="stage-0-flow", persist_result=False)
async def stage0_flow(user_id: str, run_id: str) -> dict:
    logger.info("Stage 0 starting: ingesting raw input for user_id=%s", user_id)
    stage0_result = await stage0(user_id)
    raw_df = stage0_result.pop("_df")
    stage0_result["_df_path"] = _save_parquet(raw_df, run_id, "stage0-raw-input.parquet")
    web = _public_web("stage-0", stage0_result, run_id)
    date_range = web.get("date_range", {})
    logger.info(
        "Stage 0 complete: source=%s records=%d columns=%d date_range=%s..%s",
        web.get("source_label", "unknown"),
        web.get("n_records", 0),
        web.get("n_columns", 0),
        date_range.get("start") or "?",
        date_range.get("end") or "?",
    )
    return _stage_state(stage0_result, web)


@flow(name="stage-1a-flow", persist_result=False)
async def stage1a_flow(
    question: str,
    run_id: str,
    override_payload: dict[str, Any] | None = None,
) -> dict:
    logger.info("Stage 1a starting: proposing latent model")
    stage1a_result = override_payload if override_payload is not None else await stage1a(question)
    web = _public_web("stage-1a", stage1a_result, run_id)
    latent_model = web.get("latent_model", {})
    logger.info(
        "Stage 1a complete: constructs=%d edges=%d treatments=%d outcome=%s",
        len(latent_model.get("constructs", [])),
        len(latent_model.get("edges", [])),
        len(web.get("treatments", [])),
        web.get("outcome_name", "") or "unknown",
    )
    return _stage_state(stage1a_result, web)


@flow(name="stage-1b-flow", persist_result=False)
async def stage1b_flow(
    question: str,
    stage0_result: dict,
    stage1a_result: dict,
    override_gates: bool,
    run_id: str,
    override_payload: dict[str, Any] | None = None,
) -> dict:
    logger.info("Stage 1b starting: proposing measurement model and checking identifiability")
    stage1b_result = (
        override_payload
        if override_payload is not None
        else await stage1b(question, stage0_result, stage1a_result)
    )
    stage1b_gate_result = stage1b_gate(stage1a_result, stage1b_result, override_gates)
    web_data = _build_web_payload(
        stage1b_result,
        causal_spec="causal_spec",
        llm_trace="llm_trace",
    ) | {
        "outcome": stage1b_gate_result["web_outcome"],
    }
    if stage1b_gate_result["gate_overridden"]:
        web_data["gate_overridden"] = {
            "reason": "No identifiable treatments remain — all blocked by unobserved confounders"
        }
    web = _persist_stage_payload("stage-1b", web_data, run_id)
    _raise_if_gate_failed(
        stage1b_gate_result,
        "No identifiable treatment effects remain after filtering. "
        "All treatments are blocked by unobserved confounders.",
    )

    causal_spec = stage1b_result.get("causal_spec", {})
    latent = causal_spec.get("latent", {})
    measurement = causal_spec.get("measurement", {})
    non_identifiable = stage1b_gate_result.get("non_identifiable", {})
    logger.info(
        "Stage 1b complete: constructs=%d indicators=%d identifiable_treatments=%d filtered_out=%d outcome=%s",
        len(latent.get("constructs", [])),
        len(measurement.get("indicators", [])),
        len(stage1b_gate_result.get("treatments", [])),
        len(non_identifiable),
        stage1b_gate_result.get("web_outcome", "success"),
    )
    if stage1b_gate_result["gate_overridden"]:
        logger.warning("Stage 1b gate overridden: continuing with no identifiable treatments")
    return _stage_state(stage1b_result, web, gate=stage1b_gate_result)


@flow(name="stage-2-flow", persist_result=False)
async def stage2_flow(
    question: str,
    stage0_result: dict,
    stage1b_result: dict,
    run_id: str,
) -> dict:
    logger.info("Stage 2 starting: extracting measurements from raw data")
    stage2_result = await stage2(question, stage0_result, stage1b_result)
    raw_data = stage2_result.pop("_raw_data")
    data_for_model = stage2_result.pop("_data_for_model")
    stage2_result["_raw_data_path"] = _save_parquet(raw_data, run_id, "stage2-raw-data.parquet")
    stage2_result["_data_for_model_path"] = _save_parquet(
        data_for_model, run_id, "stage2-model-data.parquet"
    )
    web = _public_stage_payload(stage2_result) | {
        "outcome": "success" if len(raw_data) > 0 else "fail"
    }
    web = _persist_stage_payload("stage-2", web, run_id)
    worker_statuses = stage2_result.get("_worker_statuses", [])
    worker_counts: dict[str, int] = {}
    for worker in worker_statuses:
        status = str(worker.get("status", "unknown"))
        worker_counts[status] = worker_counts.get(status, 0) + 1
    logger.info(
        "Stage 2 complete: extracted_rows=%d modeled_rows=%d workers=%d worker_statuses=%s outcome=%s",
        len(raw_data),
        len(data_for_model),
        len(worker_statuses),
        worker_counts,
        web.get("outcome", "success"),
    )
    return _stage_state(stage2_result, web)


@flow(name="stage-3-flow", persist_result=False)
def stage3_flow(stage1b_result: dict, stage2_result: dict, run_id: str) -> dict:
    logger.info("Stage 3 starting: validating extracted measurements")
    stage3_result = stage3(stage1b_result, stage2_result)
    web = _persist_stage_payload(
        "stage-3",
        _build_web_payload(
            stage3_result,
            outcome="outcome",
            validation_report="validation_report",
        ),
        run_id,
    )
    report = web.get("validation_report", {})
    error_count, warning_count = _validation_issue_counts(report)
    logger.info(
        "Stage 3 complete: is_valid=%s issues=%d errors=%d warnings=%d outcome=%s",
        report.get("is_valid", False),
        len(report.get("issues", []) or []),
        error_count,
        warning_count,
        web.get("outcome", "success"),
    )
    return _stage_state(stage3_result, web)


@flow(name="stage-4-flow", persist_result=False)
async def stage4_flow(
    question: str,
    stage1b_result: dict,
    stage2_result: dict,
    enable_literature: bool,
    run_id: str,
    override_payload: dict[str, Any] | None = None,
) -> dict:
    from prefect.artifacts import create_markdown_artifact

    logger.info("Stage 4 starting: building model specification and priors")
    if override_payload is None:
        stage4_result = await stage4(question, stage1b_result, stage2_result, enable_literature)
    else:
        stage4_result = dict(override_payload)
        stage4_result.setdefault("causal_spec", stage1b_result["causal_spec"])
    model_spec = stage4_result.get("model_spec", {})
    validation = stage4_result.get("validation", {})
    model_info = stage4_result.get("model_info", {})
    artifact_result = create_markdown_artifact(
        key="model-spec",
        markdown=(
            f"## Model Specification\n\n"
            f"- **Parameters**: {len(model_spec.get('parameters', []))}\n"
            f"- **Priors valid**: {validation.get('is_valid', 'unknown')}\n"
            f"- **Model built**: {model_info.get('model_built', 'unknown')}\n"
        ),
    )
    if inspect.isawaitable(artifact_result):
        await artifact_result
    web = _persist_stage_payload(
        "stage-4",
        _build_web_payload(
            stage4_result,
            model_spec="model_spec",
            priors="priors",
            llm_trace="llm_trace",
            prior_predictive_samples="prior_predictive_samples",
        ),
        run_id,
    )
    logger.info(
        "Stage 4 complete: parameters=%d likelihoods=%d priors=%d validation_ok=%s model_built=%s",
        len(model_spec.get("parameters", [])),
        len(model_spec.get("likelihoods", [])),
        len(stage4_result.get("priors", {})),
        validation.get("is_valid", False),
        model_info.get("model_built", False),
    )
    return _stage_state(stage4_result, web)


@flow(name="stage-4b-flow", persist_result=False)
def stage4b_flow(
    stage4_result: dict,
    stage2_result: dict,
    override_gates: bool,
    run_id: str,
) -> dict:
    logger.info("Stage 4b starting: checking parametric identifiability")
    stage4b_result = stage4b(stage4_result, stage2_result, None)
    stage4b_gate_result = stage4b_gate(stage4b_result, override_gates)
    web_data = _build_web_payload(stage4b_result, parametric_id="parametric_id") | {
        "outcome": stage4b_gate_result["outcome"]
    }
    if stage4b_gate_result["gate_overridden"]:
        t_rule = stage4b_gate_result["t_rule"]
        web_data["gate_overridden"] = {
            "reason": (
                f"T-rule violated: {t_rule.get('n_free_params')} free parameters "
                f"> {t_rule.get('n_moments')} moment conditions"
            )
        }
    web = _persist_stage_payload("stage-4b", web_data, run_id)
    t_rule = stage4b_gate_result["t_rule"]
    _raise_if_gate_failed(
        stage4b_gate_result,
        f"T-rule violated: {t_rule.get('n_free_params')} free parameters "
        f"> {t_rule.get('n_moments')} moment conditions. "
        "Model is provably non-identified. Halting pipeline.",
    )

    parametric_id = stage4b_result.get("parametric_id", {})
    summary = parametric_id.get("summary", {})
    t_rule = parametric_id.get("t_rule", {})
    logger.info(
        "Stage 4b complete: checked=%s t_rule=%s(%s/%s) structural_issues=%d boundary_issues=%d weak_params=%d outcome=%s",
        parametric_id.get("checked", False),
        "pass" if t_rule.get("satisfies", True) else "fail",
        t_rule.get("n_free_params", "?"),
        t_rule.get("n_moments", "?"),
        len(summary.get("structural_issues", []) or []),
        len(summary.get("boundary_issues", []) or []),
        len(summary.get("weak_params", []) or []),
        stage4b_gate_result.get("outcome", "success"),
    )
    if stage4b_gate_result["gate_overridden"]:
        logger.warning("Stage 4b gate overridden: continuing despite T-rule violation")
    return _stage_state(stage4b_result, web, gate=stage4b_gate_result)


@flow(name="stage-5-flow", persist_result=False)
def stage5_flow(
    stage4_result: dict,
    stage1b_result: dict,
    stage2_result: dict,
    inference_method: str | None,
    run_id: str,
) -> dict:
    logger.info("Stage 5 starting: fitting model and running diagnostics")
    stage5_result = stage5(stage4_result, stage1b_result, stage2_result, inference_method)
    fitted_result = stage5_result.pop("_fitted_result")
    stage5_result["_fitted_result_path"] = _save_pickle(
        fitted_result, run_id, "stage5-fitted-result.pkl"
    )
    web = _persist_stage_payload(
        "stage-5",
        _build_web_payload(
            stage5_result,
            outcome="outcome",
            power_scaling="ps_list",
            ppc="ppc_result",
            inference_metadata="inference_metadata",
            mcmc_diagnostics="mcmc_diagnostics",
            svi_diagnostics="svi_diagnostics",
            loo_diagnostics="loo_diagnostics",
            posterior_marginals="posterior_marginals",
            posterior_pairs="posterior_pairs",
        ),
        run_id,
    )
    ps_list = web.get("power_scaling", [])
    ps_issues = sum(
        1
        for entry in ps_list
        if entry.get("diagnosis") in {"prior_dominated", "prior_data_conflict"}
    )
    ppc_warnings = len(web.get("ppc", {}).get("per_variable_warnings", []) or [])
    logger.info(
        "Stage 5 complete: method=%s power_scaling_issues=%d ppc_warnings=%d outcome=%s",
        web.get("inference_metadata", {}).get("method", "unknown"),
        ps_issues,
        ppc_warnings,
        web.get("outcome", "success"),
    )
    return _stage_state(stage5_result, web)


@flow(name="stage-6-flow", persist_result=False)
def stage6_flow(
    stage5_result: dict,
    stage1a_result: dict,
    stage1b_result: dict,
    stage1b_gate_result: dict,
    run_id: str,
) -> dict:
    logger.info("Stage 6 starting: estimating intervention effects")
    stage6_result = stage6(stage5_result, stage1a_result, stage1b_result, stage1b_gate_result)
    web = _persist_stage_payload(
        "stage-6",
        _build_web_payload(
            stage6_result,
            outcome="outcome",
            intervention_results="intervention_results",
        ),
        run_id,
    )
    intervention_results = web.get("intervention_results", [])
    warning_count = sum(
        1
        for result in intervention_results
        if result.get("warning")
        or result.get("ppc_warnings")
        or result.get("prior_sensitivity_warning")
    )
    logger.info(
        "Stage 6 complete: treatments_ranked=%d warnings=%d outcome=%s",
        len(intervention_results),
        warning_count,
        web.get("outcome", "success"),
    )
    return _stage_state(stage6_result, web)
