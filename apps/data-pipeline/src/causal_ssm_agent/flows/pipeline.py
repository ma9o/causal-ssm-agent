"""Main causal inference pipeline.

Orchestrates all stages from structure proposal to intervention analysis.

Two-stage specification following Anderson & Gerbing (1988):
- Stage 1a: Latent model (theory-driven, no data)
- Stage 1b: Measurement model (data-driven operationalization)
"""

import logging
import math
from pathlib import Path

from prefect import flow
from prefect.artifacts import create_markdown_artifact, create_table_artifact
from prefect.utilities.annotations import unmapped

from causal_ssm_agent.utils.aggregations import flatten_aggregated_data
from causal_ssm_agent.utils.data import get_sample_chunks, load_query

from .stages import (
    # Stage 3
    aggregate_measurements,
    # Stage 1b
    combine_worker_results,
    # Stage 5
    fit_model,
    load_orchestrator_chunks,
    # Stage 2
    load_worker_chunks,
    # Web persistence
    persist_web_result,
    populate_indicators,
    # Stage 0
    preprocess_raw_input,
    # Stage 1a
    propose_latent_model,
    propose_measurement_with_identifiability_fix,
    run_interventions,
    run_power_scaling,
    run_ppc,
    # Stage 4
    stage4_orchestrated_flow,
    # Stage 4b
    stage4b_parametric_id_flow,
    validate_extraction,
)

logger = logging.getLogger(__name__)

RESULT_STORAGE = Path("results")


def _coerce_sample_value(value: object) -> str | int | float | bool | None:
    """Coerce stringified extraction values back to JSON-friendly scalars."""
    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, str):
        stripped = value.strip()
        lower = stripped.lower()
        if lower == "true":
            return True
        if lower == "false":
            return False
        if stripped and (stripped.isdigit() or (stripped[0] == "-" and stripped[1:].isdigit())):
            return int(stripped)
        try:
            parsed = float(stripped)
            return parsed if math.isfinite(parsed) else stripped
        except ValueError:
            return stripped

    return str(value)


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
):
    """
    Main causal inference pipeline.

    Automatically identifies the outcome from the question and estimates
    effects of all potential treatments, ranking them by effect size.

    Args:
        query_file: Filename in data/queries/ (e.g., 'procrastination-patterns')
        user_id: User subdirectory under data/raw/ (default: test_user)
        inference_method: Override inference method ("svi" or "nuts", default from config)
        enable_literature: Override literature search (default from config)
        override_gates: Continue past stage failures instead of halting (default from config)
        query: Raw query text (used by web UI). If provided, takes precedence over query_file.
    """
    # Resolve effective override_gates setting
    from causal_ssm_agent.utils.config import get_config

    config = get_config()
    gates_overridden = (
        override_gates if override_gates is not None else config.pipeline.override_gates
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 0: Preprocess raw input and load question
    # ══════════════════════════════════════════════════════════════════════════
    if query:
        question = query.strip()
    elif query_file:
        question = load_query(query_file)
    else:
        raise ValueError("Either 'query' (raw text) or 'query_file' (filename) must be provided")
    logger.info("Query source: %s", "raw text" if query else query_file)
    logger.info("Question: %s", f"{question[:100]}..." if len(question) > 100 else question)

    logger.info("=== Stage 0: Preprocess (user: %s) ===", user_id)
    preprocess_result = preprocess_raw_input(user_id)
    lines = preprocess_result["lines"]

    persist_web_result(
        "stage-0",
        {
            "n_records": preprocess_result["n_records"],
            "date_range": preprocess_result["date_range"],
            "sample": preprocess_result["sample"],
        },
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 1a: Propose latent model (theory only, no data)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=== Stage 1a: Latent Model ===")
    stage1a_result = await propose_latent_model(question)
    latent_model = stage1a_result["latent_model"]
    outcome = stage1a_result["outcome_name"]
    treatments = stage1a_result["treatments"]
    n_constructs = len(latent_model["constructs"])
    n_edges = len(latent_model["edges"])
    logger.info("Proposed %d constructs with %d causal edges", n_constructs, n_edges)

    if not outcome:
        raise ValueError("No outcome identified in latent model (missing is_outcome=true)")
    logger.info("Outcome variable: %s", outcome)

    logger.info("Potential treatments: %d constructs with paths to %s", len(treatments), outcome)
    for t in treatments[:5]:
        logger.info("  - %s", t)
    if len(treatments) > 5:
        logger.info("  ... and %d more", len(treatments) - 5)

    persist_web_result("stage-1a", stage1a_result)

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 1b: Propose measurement model (with identifiability check)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=== Stage 1b: Measurement Model with Identifiability ===")
    orchestrator_chunks = load_orchestrator_chunks(lines)
    logger.info("Loaded %d orchestrator chunks", len(orchestrator_chunks))

    # Propose measurements and check identifiability
    stage1b_result = await propose_measurement_with_identifiability_fix(
        question,
        latent_model,
        orchestrator_chunks[: get_sample_chunks()],
    )

    causal_spec = stage1b_result["causal_spec"]
    measurement_model = stage1b_result["measurement_model"]
    identifiability_status = stage1b_result["identifiability_status"]

    n_indicators = len(measurement_model["indicators"])
    logger.info("Final model has %d indicators", n_indicators)

    # Hard gate: filter out non-identifiable treatments
    non_identifiable = identifiability_status.get("non_identifiable_treatments", {})
    gate_1b_failed = False
    if non_identifiable:
        logger.warning("NON-IDENTIFIABLE TREATMENT EFFECTS (excluded from analysis):")
        for treatment in sorted(non_identifiable.keys()):
            details = non_identifiable[treatment]
            blockers = details.get("confounders", []) if isinstance(details, dict) else []
            notes = details.get("notes") if isinstance(details, dict) else None
            if blockers:
                logger.warning("  - %s → %s (blocked by: %s)", treatment, outcome, ", ".join(blockers))
            elif notes:
                logger.warning("  - %s → %s (%s)", treatment, outcome, notes)
            else:
                logger.warning("  - %s → %s", treatment, outcome)
        treatments = [t for t in treatments if t not in non_identifiable]
        logger.info("Continuing with %d identifiable treatments", len(treatments))
        if not treatments:
            gate_1b_failed = True
            if gates_overridden:
                logger.warning(
                    "GATE 1b OVERRIDDEN: No identifiable treatments, continuing with empty list"
                )

    gate_1b_overridden = gates_overridden and gate_1b_failed

    create_markdown_artifact(
        key="causal-spec",
        markdown=f"## Causal Specification\n\n"
        f"- **Constructs**: {n_constructs}\n"
        f"- **Edges**: {n_edges}\n"
        f"- **Indicators**: {n_indicators}\n"
        f"- **Non-identifiable treatments**: "
        f"{list(non_identifiable.keys()) if non_identifiable else 'none'}\n",
    )

    # Persist web data BEFORE potential halt so frontend can display gate failure
    stage1b_web_data: dict = {
        "causal_spec": causal_spec,
        "llm_trace": stage1b_result.get("llm_trace"),
        "outcome": "fail" if non_identifiable else "success",
    }
    if gate_1b_overridden:
        stage1b_web_data["gate_overridden"] = {
            "reason": "No identifiable treatments remain — all blocked by unobserved confounders"
        }
    persist_web_result("stage-1b", stage1b_web_data)

    if gate_1b_failed and not gates_overridden:
        raise RuntimeError(
            "No identifiable treatment effects remain after filtering. "
            "All treatments are blocked by unobserved confounders."
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 2: Parallel indicator population (worker chunk size)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=== Stage 2: Worker Extraction ===")
    worker_chunks = load_worker_chunks(lines)
    logger.info("Loaded %d worker chunks", len(worker_chunks))

    worker_results = populate_indicators.map(
        worker_chunks,
        question=unmapped(question),
        causal_spec=unmapped(causal_spec),
    )
    resolved_worker_results = [
        wr.result() if hasattr(wr, "result") else wr for wr in worker_results
    ]

    # Combine raw worker results
    raw_data = combine_worker_results(resolved_worker_results)  # ty: ignore[no-matching-overload]
    raw_data_result = raw_data.result() if hasattr(raw_data, "result") else raw_data
    n_observations = len(raw_data_result)
    n_unique_indicators = raw_data_result["indicator"].n_unique() if n_observations > 0 else 0
    logger.info("  Combined %d observations across %d indicators", n_observations, n_unique_indicators)

    # Aggregate to pipeline-level aggregation window
    aggregated = aggregate_measurements(causal_spec, resolved_worker_results)  # ty: ignore[no-matching-overload]
    aggregated_result = aggregated.result() if hasattr(aggregated, "result") else aggregated
    if aggregated_result:
        data_for_model = flatten_aggregated_data(aggregated_result)
        n_agg = len(data_for_model)
        logger.info(
            "  Aggregated to %d observations across %s granularities",
            n_agg, list(aggregated_result.keys()),
        )
    else:
        data_for_model = raw_data_result
        logger.info("  No aggregation applied (using raw data)")

    # Persist stage-2 web data
    sample_rows = raw_data_result.head(20).to_dicts() if n_observations > 0 else []
    per_ind_counts = (
        dict(raw_data_result.group_by("indicator").len().iter_rows()) if n_observations > 0 else {}
    )
    worker_statuses = []
    for i, chunk in enumerate(worker_chunks):
        wr = resolved_worker_results[i] if i < len(resolved_worker_results) else None
        output = getattr(wr, "output", None)
        dataframe = getattr(wr, "dataframe", None)
        n_extractions = (
            len(output.extractions)
            if output is not None and hasattr(output, "extractions")
            else len(dataframe)
            if dataframe is not None
            else 0
        )
        worker_statuses.append(
            {
                "worker_id": i,
                "status": "completed",
                "n_extractions": n_extractions,
                "chunk_size": chunk.count("\n") + 1 if chunk else 0,
            }
        )

    combined_extractions_sample = []
    for row in sample_rows:
        combined_extractions_sample.append(
            {
                "indicator": str(row.get("indicator", "")),
                "value": _coerce_sample_value(row.get("value")),
                "timestamp": str(row.get("timestamp"))
                if row.get("timestamp") is not None
                else None,
            }
        )

    stage2_outcome = "warn" if any(w["status"] == "failed" for w in worker_statuses) else "success"
    persist_web_result(
        "stage-2",
        {
            "outcome": stage2_outcome,
            "workers": worker_statuses,
            "combined_extractions_sample": combined_extractions_sample,
            "per_indicator_counts": per_ind_counts,
        },
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 3: Validate Extraction
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=== Stage 3: Extraction Validation ===")
    validation_task = validate_extraction(causal_spec, resolved_worker_results)  # ty: ignore[no-matching-overload]
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
                    issue["indicator"], issue["issue_type"], issue["severity"], issue["message"],
                )
        elif issues:
            logger.warning("Stage 3 validation warnings:")
            for issue in issues:
                logger.warning(
                    "    - %s: %s (%s) %s",
                    issue["indicator"], issue["issue_type"], issue["severity"], issue["message"],
                )

    if validation_report and validation_report.get("issues"):
        create_table_artifact(
            key="validation-issues",
            table=[
                {
                    "indicator": i["indicator"],
                    "type": i["issue_type"],
                    "severity": i["severity"],
                    "message": i["message"],
                }
                for i in validation_report["issues"]
            ],
            description="Stage 3 extraction validation issues",
        )

    # Persist web data
    validation_report_web = validation_report or {
        "is_valid": False,
        "issues": [],
        "per_indicator_health": [],
    }
    if not validation_report_web.get("is_valid", True):
        stage3_outcome = "fail"
    elif any(
        i.get("severity") in ("warning", "error") for i in validation_report_web.get("issues", [])
    ):
        stage3_outcome = "warn"
    else:
        stage3_outcome = "success"
    persist_web_result(
        "stage-3", {"outcome": stage3_outcome, "validation_report": validation_report_web}
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 4: Model Specification (Orchestrator-Worker Architecture)
    # ══════════════════════════════════════════════════════════════════════════
    lit_enabled = (
        enable_literature
        if enable_literature is not None
        else config.stage4_prior_elicitation.literature_search.enabled
    )

    logger.info("=== Stage 4: Model Specification ===")
    stage4_result = await stage4_orchestrated_flow(
        causal_spec=causal_spec,
        question=question,
        raw_data=data_for_model,
        enable_literature=lit_enabled,
    )

    model_spec = stage4_result.get("model_spec", {})
    logger.info("Parameters: %d total", len(model_spec.get("parameters", [])))

    # Report validation issues
    validation = stage4_result.get("validation", {})
    if not validation.get("is_valid", True):
        issues = validation.get("issues", [])
        logger.warning("Stage 4 prior validation failed (%d issues):", len(issues))
        for issue in issues:
            if isinstance(issue, dict):
                logger.warning("    - %s: %s", issue.get("parameter"), issue.get("issue"))
            else:
                logger.warning("    - %s", issue)

    model_info = stage4_result.get("model_info", {})
    if not model_info.get("model_built", True):
        logger.warning("Stage 4 model build failed: %s", model_info.get("error"))

    create_markdown_artifact(
        key="model-spec",
        markdown=f"## Model Specification\n\n"
        f"- **Parameters**: {len(model_spec.get('parameters', []))}\n"
        f"- **Priors valid**: {validation.get('is_valid', 'unknown')}\n"
        f"- **Model built**: {model_info.get('model_built', 'unknown')}\n",
    )

    persist_web_result(
        "stage-4",
        {
            "model_spec": model_spec,
            "priors": list(stage4_result.get("priors", {}).values()),
            "llm_trace": stage4_result.get("llm_trace"),
            "prior_predictive_samples": stage4_result.get("prior_predictive_samples"),
        },
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Pre-build SSMModelBuilder once for downstream stages
    # ══════════════════════════════════════════════════════════════════════════
    from causal_ssm_agent.models.ssm_builder import build_ssm_builder

    try:
        builder = build_ssm_builder(
            model_spec=stage4_result["model_spec"],
            priors=stage4_result["priors"],
            raw_data=data_for_model,
            causal_spec=stage4_result.get("causal_spec"),
        )
    except Exception:
        logger.warning("Pre-building SSM builder failed; stages will build their own", exc_info=True)
        builder = None

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 4b: Parametric Identifiability Diagnostics
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=== Stage 4b: Parametric Identifiability ===")
    stage4_result = stage4b_parametric_id_flow(
        stage4_result, raw_data=data_for_model, builder=builder
    )

    gate_4b_failed = False
    gate_4b_overridden = False
    param_id = stage4_result.get("parametric_id", {})
    if param_id.get("checked", False):
        t_rule = param_id.get("t_rule", {})
        if not t_rule.get("satisfies", True):
            gate_4b_failed = True
            if gates_overridden:
                logger.warning(
                    "GATE 4b OVERRIDDEN: T-rule violated (%s free params > %s moments), continuing",
                    t_rule.get("n_free_params"), t_rule.get("n_moments"),
                )
                gate_4b_overridden = True
        summary = param_id.get("summary", {})
        if summary.get("structural_issues"):
            logger.warning("STRUCTURAL non-identifiability detected — some parameters unconstrained")
        elif summary.get("boundary_issues"):
            logger.warning("Boundary identifiability issues at some prior draws")
        else:
            logger.info("Parametric identifiability OK")
        weak = summary.get("weak_params", [])
        if weak:
            logger.info("  Weak parameters (low contraction): %s", weak)
    else:
        logger.info("  Skipped: %s", param_id.get("error", "unknown"))

    # Persist web data BEFORE potential halt so frontend can display gate failure
    if gate_4b_failed:
        stage4b_outcome = "fail"
    elif param_id.get("checked", False):
        summary = param_id.get("summary", {})
        has_issues = (
            summary.get("structural_issues")
            or summary.get("boundary_issues")
            or summary.get("weak_params")
        )
        stage4b_outcome = "warn" if has_issues else "success"
    else:
        stage4b_outcome = "success"
    stage4b_web_data: dict = {
        "outcome": stage4b_outcome,
        "parametric_id": stage4_result.get("parametric_id", {}),
    }
    if gate_4b_overridden:
        stage4b_web_data["gate_overridden"] = {
            "reason": (
                f"T-rule violated: {t_rule.get('n_free_params')} free parameters "
                f"> {t_rule.get('n_moments')} moment conditions"
            )
        }
    persist_web_result("stage-4b", stage4b_web_data)

    if gate_4b_failed and not gates_overridden:
        raise RuntimeError(
            f"T-rule violated: {t_rule.get('n_free_params')} free parameters "
            f"> {t_rule.get('n_moments')} moment conditions. "
            "Model is provably non-identified. Halting pipeline."
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 5: Fit and diagnose / Stage 6: Treatment effects
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=== Stage 5: Inference ===")
    logger.info("Estimating effects of %d treatments on %s", len(treatments), outcome)
    sampler_config = (
        config.inference.to_sampler_config(method_override=inference_method)
        if inference_method
        else None
    )

    if config.inference.gpu:
        # ── GPU path: dispatch all stage 5 tasks to Modal ──
        from causal_ssm_agent.flows.gpu_inference import run_stage5_gpu

        logger.info("Dispatching to Modal (%s GPU)...", config.inference.gpu)
        gpu_result = run_stage5_gpu(
            stage4_result=stage4_result,
            raw_data=data_for_model,
            sampler_config=sampler_config,
            treatments=treatments,
            outcome=outcome,
            causal_spec=causal_spec,
            gpu=config.inference.gpu,
        )
        ps_result = gpu_result["ps_result"]
        ppc_result = gpu_result.get("ppc_result", {"checked": False})
        intervention_results = gpu_result["intervention_results"]
    else:
        # ── Local path: run stage 5 tasks via Prefect ──
        fitted = fit_model(
            stage4_result, data_for_model, sampler_config=sampler_config, builder=builder
        )

        # Post-fit power-scaling sensitivity diagnostic
        power_scaling = run_power_scaling(fitted, data_for_model)
        ps_result = power_scaling.result() if hasattr(power_scaling, "result") else power_scaling  # ty: ignore[call-non-callable]

        # Posterior predictive checks
        ppc_task = run_ppc(fitted, data_for_model)
        ppc_result = ppc_task.result() if hasattr(ppc_task, "result") else ppc_task  # ty: ignore[call-non-callable]

        # Run interventions for all treatments (with PPC + power-scaling warnings)
        results = run_interventions(
            fitted, treatments, outcome, causal_spec, ppc_result, ps_result=ps_result
        )
        intervention_results = results.result() if hasattr(results, "result") else results  # ty: ignore[call-non-callable]

    # Log power-scaling results (shared by both paths)
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

    # Log ranked results table
    logger.info("=== Treatment Ranking by Effect on %s ===", outcome)
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
                logger.info(
                    "%-5d %-30s %10s %8s %4s  (%s)", rank, name, "—", "", ident, warning
                )

        # Log prior-sensitivity footnotes
        ps_entries = [e for e in intervention_results if e.get("prior_sensitivity_warning")]
        if ps_entries:
            logger.warning("  * Prior-dominated effects:")
            for e in ps_entries:
                logger.warning("    %s: %s", e["treatment"], e["prior_sensitivity_warning"])

    if intervention_results:
        create_table_artifact(
            key="treatment-ranking",
            table=[
                {
                    "rank": i + 1,
                    "treatment": r["treatment"],
                    "effect": (
                        f"{r['effect_size']:+.4f}" if r.get("effect_size") is not None else "—"
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

    # Assemble serializable stage-5 data for the webapp
    # Power-scaling: reshape from {param -> value} dicts to per-param list
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

    # Inference metadata
    inf_meta = {
        "method": (
            fitted.result().get("inference_type", "unknown")
            if hasattr(fitted, "result")
            else fitted.get("inference_type", "unknown")
        )
        if not config.inference.gpu
        else gpu_result.get("mcmc_diagnostics", {}).get("method", "unknown")
        if config.inference.gpu
        else "unknown",
        "n_samples": 10000,
        "duration_seconds": 0.0,
    }

    # MCMC / SVI / LOO / posterior diagnostics
    mcmc_diagnostics = None
    svi_diagnostics = None
    loo_diagnostics = None
    posterior_marginals = None
    posterior_pairs = None
    if config.inference.gpu:
        mcmc_diagnostics = gpu_result.get("mcmc_diagnostics")
        svi_diagnostics = gpu_result.get("svi_diagnostics")
        loo_diagnostics = gpu_result.get("loo_diagnostics")
        posterior_marginals = gpu_result.get("posterior_marginals")
        posterior_pairs = gpu_result.get("posterior_pairs")
    else:
        fitted_res = fitted.result() if hasattr(fitted, "result") else fitted
        mcmc_diagnostics = fitted_res.get("mcmc_diagnostics")
        svi_diagnostics = fitted_res.get("svi_diagnostics")
        loo_diagnostics = fitted_res.get("loo_diagnostics")
        posterior_marginals = fitted_res.get("posterior_marginals")
        posterior_pairs = fitted_res.get("posterior_pairs")

    has_ppc_warnings = bool(ppc_result.get("per_variable_warnings"))
    has_ps_issues = any(
        e["diagnosis"] in ("prior_dominated", "prior_data_conflict") for e in ps_list
    )
    stage5_outcome = "warn" if (has_ppc_warnings or has_ps_issues) else "success"

    stage5_data = {
        "outcome": stage5_outcome,
        "power_scaling": ps_list,
        "ppc": ppc_result,
        "inference_metadata": inf_meta,
        "mcmc_diagnostics": mcmc_diagnostics,
        "svi_diagnostics": svi_diagnostics,
        "loo_diagnostics": loo_diagnostics,
        "posterior_marginals": posterior_marginals,
        "posterior_pairs": posterior_pairs,
    }
    persist_web_result("stage-5", stage5_data)

    stage6_has_warnings = any(
        r.get("ppc_warnings") or r.get("prior_sensitivity_warning")
        for r in intervention_results
    )
    stage6_data = {
        "outcome": "warn" if stage6_has_warnings else "success",
        "intervention_results": intervention_results,
    }
    persist_web_result("stage-6", stage6_data)

    return {**stage5_data, **stage6_data}


if __name__ == "__main__":
    # Serve the flow for UI access
    causal_inference_pipeline.serve(
        name="causal-inference",
        tags=["causal", "llm"],
    )
