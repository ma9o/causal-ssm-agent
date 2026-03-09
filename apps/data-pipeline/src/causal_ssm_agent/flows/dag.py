"""Hamilton DAG definition for the causal inference pipeline.

Each function is a Hamilton node. The function name IS the node name, and
function parameters whose names match other node names are DAG edges
(Hamilton resolves them automatically).

Node naming convention:
- ``stageN``: main computation node for stage N
- ``stageN_web``: web persistence side-effect node
- ``stageN_gate``: gate/filtering logic that can halt the pipeline

Overridable nodes (human-in-the-loop via replay):
- ``stage1a``: Latent model proposal
- ``stage1b``: Measurement model + identifiability
- ``stage4``: Model specification + prior elicitation
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


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

    result = await agentic_ingest.fn(user_id)
    df = result.dataframe

    payload = build_stage0_payload(result, df)
    return {
        **payload,
        "_df": df,
        "_column_descriptions": result.column_descriptions,
    }


def stage0_web(stage0: dict) -> dict:
    """Persist stage 0 result to web layer."""
    from .stages import persist_web_result

    web = {k: v for k, v in stage0.items() if not k.startswith("_")}
    persist_web_result("stage-0", web)
    return web


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1a: Latent model proposal
# ═══════════════════════════════════════════════════════════════════════════════


async def stage1a(question: str) -> dict:
    """Propose theoretical constructs and causal edges (latent model).

    Returns: {latent_model, outcome_name, treatments, llm_trace?}
    """
    from .stages import propose_latent_model

    return await propose_latent_model.fn(question)


def stage1a_web(stage1a: dict) -> dict:
    """Persist stage 1a result to web layer."""
    from .stages import persist_web_result

    persist_web_result("stage-1a", stage1a)
    return stage1a


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1b: Measurement model + identifiability
# ═══════════════════════════════════════════════════════════════════════════════


async def stage1b(question: str, stage0: dict, stage1a: dict) -> dict:
    """Propose measurement model and check identifiability.

    Returns: {causal_spec, measurement_model, identifiability_status, llm_trace?}
    """
    from .pipeline_helpers import format_schema_for_llm
    from .stages import propose_measurement_with_identifiability_fix

    ingested_df = stage0["_df"]
    column_descriptions = stage0["_column_descriptions"]
    latent_model = stage1a["latent_model"]

    dataset_schema = format_schema_for_llm(ingested_df, column_descriptions)
    return await propose_measurement_with_identifiability_fix.fn(
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


def stage1b_web(stage1b: dict, stage1b_gate: dict) -> dict:
    """Persist stage 1b result to web layer (before potential halt)."""
    from .stages import persist_web_result

    web_data: dict = {
        "causal_spec": stage1b["causal_spec"],
        "llm_trace": stage1b.get("llm_trace"),
        "outcome": stage1b_gate["web_outcome"],
    }
    if stage1b_gate["gate_overridden"]:
        web_data["gate_overridden"] = {
            "reason": "No identifiable treatments remain — all blocked by unobserved confounders"
        }
    persist_web_result("stage-1b", web_data)

    # Halt pipeline if gate failed and not overridden
    if stage1b_gate["gate_failed"] and not stage1b_gate["gate_overridden"]:
        raise RuntimeError(
            "No identifiable treatment effects remain after filtering. "
            "All treatments are blocked by unobserved confounders."
        )

    return web_data


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

    from causal_ssm_agent.utils.aggregations import (
        aggregate_worker_measurements,
        flatten_aggregated_data,
    )

    from .stages import stage2_extraction_flow

    ingested_df = stage0["_df"]
    causal_spec = stage1b["causal_spec"]

    stage2_result = await stage2_extraction_flow(
        raw_df=ingested_df,
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


def stage2_web(stage2: dict) -> dict:
    """Persist stage 2 result to web layer."""
    from .stages import persist_web_result

    web = {k: v for k, v in stage2.items() if not k.startswith("_")}
    n_observations = len(stage2.get("_raw_data", []))
    web["outcome"] = "success" if n_observations > 0 else "fail"
    persist_web_result("stage-2", web)
    return web


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
    raw_data = stage2["_raw_data"]

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


def stage3_web(stage3: dict) -> dict:
    """Persist stage 3 result to web layer."""
    from .stages import persist_web_result

    persist_web_result(
        "stage-3", {"outcome": stage3["outcome"], "validation_report": stage3["validation_report"]}
    )
    return stage3


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
    data_for_model = stage2["_data_for_model"]

    return await stage4_orchestrated_flow(
        causal_spec=causal_spec,
        question=question,
        raw_data=data_for_model,
        enable_literature=enable_literature,
    )


def stage4_web(stage4: dict) -> dict:
    """Persist stage 4 result to web layer."""
    from prefect.artifacts import create_markdown_artifact

    from .stages import persist_web_result

    model_spec = stage4.get("model_spec", {})
    validation = stage4.get("validation", {})
    model_info = stage4.get("model_info", {})

    create_markdown_artifact(
        key="model-spec",
        markdown=(
            f"## Model Specification\n\n"
            f"- **Parameters**: {len(model_spec.get('parameters', []))}\n"
            f"- **Priors valid**: {validation.get('is_valid', 'unknown')}\n"
            f"- **Model built**: {model_info.get('model_built', 'unknown')}\n"
        ),
    )

    persist_web_result(
        "stage-4",
        {
            "model_spec": model_spec,
            "priors": stage4.get("priors", {}),
            "llm_trace": stage4.get("llm_trace"),
            "prior_predictive_samples": stage4.get("prior_predictive_samples"),
        },
    )
    return stage4


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
            raw_data=stage2["_data_for_model"],
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


def stage4b(stage4: dict, stage2: dict, ssm_builder: Any) -> dict:
    """Parametric identifiability diagnostics.

    Returns: {parametric_id, rb_partition, ...stage4 passthrough}
    """
    from .stages import stage4b_parametric_id_flow

    return stage4b_parametric_id_flow(
        stage4, raw_data=stage2["_data_for_model"], builder=ssm_builder
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


def stage4b_web(stage4b: dict, stage4b_gate: dict) -> dict:
    """Persist stage 4b result to web layer."""
    from .stages import persist_web_result

    web_data: dict = {
        "outcome": stage4b_gate["outcome"],
        "parametric_id": stage4b.get("parametric_id", {}),
    }
    if stage4b_gate["gate_overridden"]:
        t_rule = stage4b_gate["t_rule"]
        web_data["gate_overridden"] = {
            "reason": (
                f"T-rule violated: {t_rule.get('n_free_params')} free parameters "
                f"> {t_rule.get('n_moments')} moment conditions"
            )
        }
    persist_web_result("stage-4b", web_data)

    # Halt pipeline if gate failed and not overridden
    if stage4b_gate["gate_failed"] and not stage4b_gate["gate_overridden"]:
        t_rule = stage4b_gate["t_rule"]
        raise RuntimeError(
            f"T-rule violated: {t_rule.get('n_free_params')} free parameters "
            f"> {t_rule.get('n_moments')} moment conditions. "
            "Model is provably non-identified. Halting pipeline."
        )

    return web_data


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 5: Inference + diagnostics
# ═══════════════════════════════════════════════════════════════════════════════


def stage5(
    stage4: dict,
    stage1b: dict,
    stage2: dict,
    ssm_builder: Any,
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
    data_for_model = stage2["_data_for_model"]
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
        fitted = fit_model(
            stage4, data_for_model, sampler_config=sampler_config, builder=ssm_builder
        )
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


def stage5_web(stage5: dict) -> dict:
    """Persist stage 5 result to web layer."""
    from .stages import persist_web_result

    web_data = {
        "outcome": stage5["outcome"],
        "power_scaling": stage5["ps_list"],
        "ppc": stage5["ppc_result"],
        "inference_metadata": stage5["inference_metadata"],
        "mcmc_diagnostics": stage5["mcmc_diagnostics"],
        "svi_diagnostics": stage5["svi_diagnostics"],
        "loo_diagnostics": stage5["loo_diagnostics"],
        "posterior_marginals": stage5["posterior_marginals"],
        "posterior_pairs": stage5["posterior_pairs"],
    }
    persist_web_result("stage-5", web_data)
    return web_data


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

    fitted_result = stage5["_fitted_result"]
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


def stage6_web(stage6: dict) -> dict:
    """Persist stage 6 result to web layer."""
    from .stages import persist_web_result

    web_data = {
        "outcome": stage6["outcome"],
        "intervention_results": stage6["intervention_results"],
    }
    persist_web_result("stage-6", web_data)
    return web_data
