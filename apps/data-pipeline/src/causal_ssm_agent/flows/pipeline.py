"""Main causal inference pipeline.

Orchestrates all stages from agentic ingestion to intervention analysis.

Two-stage specification following Anderson & Gerbing (1988):
- Stage 1a: Latent model (theory-driven, no data)
- Stage 1b: Measurement model (data-driven operationalization)
"""

import logging
from pathlib import Path

import polars as pl
from prefect import flow
from prefect.artifacts import create_markdown_artifact, create_table_artifact

from causal_ssm_agent.utils.causal_spec import get_indicators
from causal_ssm_agent.utils.data import load_query

from .stages import (
    # Stage 0
    agentic_ingest,
    # Stage 3
    fit_model,
    # Web persistence
    persist_web_result,
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


# ---------------------------------------------------------------------------
# Helpers for bridging ingested DataFrame → downstream pipeline
# ---------------------------------------------------------------------------


def format_schema_for_llm(df: pl.DataFrame, column_descriptions: dict[str, str]) -> str:
    """Format a DataFrame schema and sample for LLM consumption.

    Used by Stage 1b so the LLM can see what columns are available
    when proposing the measurement model.
    """
    lines = ["## Dataset Schema\n"]
    lines.append("| Column | Type | Description |")
    lines.append("|--------|------|-------------|")
    for col in df.columns:
        dtype = str(df.schema[col])
        desc = column_descriptions.get(col, "")
        lines.append(f"| {col} | {dtype} | {desc} |")

    lines.append("\n## Sample Data (first 10 rows)\n")
    lines.append(str(df.head(10)))

    lines.append("\n## Summary\n")
    lines.append(f"- Total rows: {len(df)}")
    lines.append(f"- Total columns: {len(df.columns)}")

    # Basic stats for numeric columns
    numeric_cols = [c for c in df.columns if df.schema[c].is_numeric()]
    if numeric_cols:
        lines.append("\n## Numeric Column Statistics\n")
        lines.append(str(df.select(numeric_cols).describe()))

    return "\n".join(lines)


def map_columns_to_indicators(df: pl.DataFrame, causal_spec: dict) -> pl.DataFrame:
    """Map ingested DataFrame columns to the long-format indicator representation.

    Takes a wide-format DataFrame (one column per variable) and melts it to the
    long format (indicator, value, timestamp) expected by Stage 3+.

    The measurement model's indicator names must match DataFrame column names.

    Args:
        df: Wide-format ingested DataFrame.
        causal_spec: CausalSpec dict with measurement model indicators.

    Returns:
        Long-format DataFrame with columns: indicator, value, timestamp.
    """
    indicators = get_indicators(causal_spec)
    indicator_names = [ind["name"] for ind in indicators]

    # Find which indicator names exist as columns
    available = [name for name in indicator_names if name in df.columns]
    if not available:
        raise ValueError(
            f"No indicator columns found in DataFrame. "
            f"Expected: {indicator_names}, got: {df.columns}"
        )

    # Detect timestamp column
    time_col = None
    for candidate in ("timestamp", "date", "time", "datetime", "time_bucket"):
        if candidate in df.columns:
            time_col = candidate
            break

    # If no obvious time column, look for datetime-typed columns
    if time_col is None:
        for col in df.columns:
            if df.schema[col] in (pl.Date, pl.Datetime):
                time_col = col
                break

    # Build the long-format DataFrame
    if time_col:
        # Melt with timestamp
        long_df = df.select([time_col, *available]).unpivot(
            index=time_col,
            on=available,
            variable_name="indicator",
            value_name="value",
        )
        # Rename time column to "timestamp"
        if time_col != "timestamp":
            long_df = long_df.rename({time_col: "timestamp"})
        # Ensure string types for compatibility with downstream
        long_df = long_df.with_columns(
            pl.col("timestamp").cast(pl.Utf8),
            pl.col("value").cast(pl.Utf8),
        )
    else:
        # No timestamp — melt without it
        long_df = df.select(available).unpivot(
            on=available,
            variable_name="indicator",
            value_name="value",
        )
        long_df = long_df.with_columns(
            pl.lit(None).alias("timestamp"),
            pl.col("value").cast(pl.Utf8),
        )

    # Drop rows where value is null
    long_df = long_df.drop_nulls(subset=["value"])

    return long_df


def _compute_date_range(df: pl.DataFrame) -> dict[str, str]:
    """Compute date range from a DataFrame's time-like columns."""
    for candidate in ("timestamp", "date", "time", "datetime"):
        if candidate in df.columns:
            col = df[candidate]
            # Try to parse as date
            if col.dtype in (pl.Date, pl.Datetime):
                start = col.min()
                end = col.max()
                if start is not None and end is not None:
                    return {
                        "start": str(start)[:10],
                        "end": str(end)[:10],
                    }
            elif col.dtype == pl.Utf8:
                try:
                    parsed = col.str.to_datetime(strict=False).drop_nulls()
                    if len(parsed) > 0:
                        return {
                            "start": str(parsed.min())[:10],
                            "end": str(parsed.max())[:10],
                        }
                except Exception:
                    pass
    return {"start": "", "end": ""}


def _sample_rows(df: pl.DataFrame, n: int = 15) -> list[dict[str, str | None]]:
    """Sample rows from a DataFrame for web display."""
    if df.is_empty():
        return []
    total = len(df)
    if total <= n:
        sample = df
    else:
        step = (total - 1) / (n - 1)
        indices = [round(i * step) for i in range(n)]
        sample = df[indices]
    # Convert all values to strings for JSON serialization
    rows = []
    for row_dict in sample.to_dicts():
        rows.append({k: (str(v) if v is not None else None) for k, v in row_dict.items()})
    return rows


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

    # ── Materialize run directory for replay artifacts ──
    from prefect.context import get_run_context
    run_dir = RESULT_STORAGE / str(get_run_context().flow_run.id)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 0: Agentic data ingestion
    # ══════════════════════════════════════════════════════════════════════════
    if query:
        question = query.strip()
    elif query_file:
        question = load_query(query_file)
    else:
        raise ValueError("Either 'query' (raw text) or 'query_file' (filename) must be provided")
    logger.info("Query source: %s", "raw text" if query else query_file)
    logger.info("Question: %s", f"{question[:100]}..." if len(question) > 100 else question)

    logger.info("=== Stage 0: Agentic Ingestion (user: %s) ===", user_id)
    ingestion_result = await agentic_ingest(user_id)
    ingested_df = ingestion_result.dataframe
    column_descriptions = ingestion_result.column_descriptions

    logger.info("Ingested: %d rows x %d columns", ingested_df.shape[0], ingested_df.shape[1])

    persist_web_result(
        "stage-0",
        {
            "source_label": ingestion_result.source_label,
            "n_records": ingested_df.shape[0],
            "n_columns": ingested_df.shape[1],
            "date_range": _compute_date_range(ingested_df),
            "sample": _sample_rows(ingested_df),
            "column_descriptions": [
                {"name": col, "dtype": str(ingested_df.schema[col]), "description": desc}
                for col, desc in column_descriptions.items()
            ],
            "llm_trace": ingestion_result.llm_trace,
        },
    )

    # ── Materialize the full DataFrame for replay ──
    ingested_df.write_parquet(run_dir / "ingested_df.parquet")

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

    # Format the ingested data schema for the LLM
    dataset_schema = format_schema_for_llm(ingested_df, column_descriptions)

    # Pass schema as a single "chunk" for Stage 1b
    stage1b_result = await propose_measurement_with_identifiability_fix(
        question,
        latent_model,
        data_sample=[dataset_schema],
        dataset_summary=f"{ingested_df.shape[0]} rows x {ingested_df.shape[1]} columns",
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
    # Stage 2: Column-to-indicator mapping (replaces worker extraction)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=== Stage 2: Column-to-Indicator Mapping ===")
    raw_data = map_columns_to_indicators(ingested_df, causal_spec)
    n_observations = len(raw_data)
    n_unique_indicators = raw_data["indicator"].n_unique() if n_observations > 0 else 0
    logger.info("Mapped %d observations across %d indicators", n_observations, n_unique_indicators)

    # Aggregate to pipeline-level aggregation window
    # Wrap raw_data into a minimal WorkerResult-like structure for aggregate_measurements
    from causal_ssm_agent.utils.aggregations import (
        aggregate_worker_measurements,
        flatten_aggregated_data,
    )

    worker_dfs = [raw_data]
    aggregated_result = aggregate_worker_measurements(worker_dfs, causal_spec)
    if aggregated_result:
        data_for_model = flatten_aggregated_data(aggregated_result)
        n_agg = len(data_for_model)
        logger.info(
            "  Aggregated to %d observations across %s granularities",
            n_agg, list(aggregated_result.keys()),
        )
    else:
        data_for_model = raw_data
        logger.info("  No aggregation applied (using raw data)")

    # Persist stage-2 web data
    sample_rows = raw_data.head(20).to_dicts() if n_observations > 0 else []
    per_ind_counts = (
        dict(raw_data.group_by("indicator").len().iter_rows()) if n_observations > 0 else {}
    )

    combined_extractions_sample = []
    for row in sample_rows:
        combined_extractions_sample.append(
            {
                "indicator": str(row.get("indicator", "")),
                "value": row.get("value"),
                "timestamp": str(row.get("timestamp"))
                if row.get("timestamp") is not None
                else None,
            }
        )

    persist_web_result(
        "stage-2",
        {
            "outcome": "success",
            "workers": [],
            "combined_extractions_sample": combined_extractions_sample,
            "per_indicator_counts": per_ind_counts,
        },
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 3: Validate Extraction
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=== Stage 3: Extraction Validation ===")
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
