"""Stage computation functions for the causal inference pipeline.

Each function (stage0, stage1a, …, stage6) implements the core logic
for one pipeline stage. Wrapper flows and artifact persistence are
handled by the stage registry (``stage_registry.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import get_prefect_logger
from .run_store import (
    load_parquet,
    load_pickle,
    unwrap_task_result,
)

logger = get_prefect_logger(__name__)


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

    ingested_df = load_parquet(stage0["_df_path"])
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


async def stage2(
    question: str, stage0: dict, stage1b: dict, root_run_id: str | None = None
) -> dict:
    """Extract indicator values from data using LLM workers.

    Returns dict with:
    - ``_raw_data``: long-format Polars DataFrame (tick-level, already at model_clock resolution)
    - ``_data_for_model``: encoded DataFrame for modeling (non-continuous types → numeric)
    - ``_worker_statuses``: per-worker status list
    - plus web-serializable fields
    """
    import polars as pl
    from prefect.task_runners import ThreadPoolTaskRunner

    from causal_ssm_agent.utils.aggregations import _encode_non_continuous
    from causal_ssm_agent.utils.causal_spec import get_indicator_dtypes, get_indicators
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
        root_run_id=root_run_id,
    )

    # Reconstruct raw_data DataFrame from worker results
    # Tick-based extraction: timestamp column contains tick start times
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
        "Extracted %d tick-level observations across %d indicators",
        n_observations,
        n_unique_indicators,
    )

    # Encode non-continuous types and prepare for modeling
    # Data is already at model_clock resolution — no aggregation needed
    if n_observations > 0:
        dtype_lookup = get_indicator_dtypes(causal_spec)
        ordinal_levels_lookup: dict[str, list[str]] = {
            ind["name"]: ind["ordinal_levels"]
            for ind in get_indicators(causal_spec)
            if ind.get("ordinal_levels")
        }
        data_for_model = _encode_non_continuous(raw_data, dtype_lookup, ordinal_levels_lookup)
        data_for_model = (
            data_for_model.with_columns(
                pl.col("value").cast(pl.Float64, strict=False).alias("value"),
                pl.col("timestamp")
                .str.replace(r"[Zz]$", "")
                .str.replace(r"[+-]\d{2}:\d{2}$", "")
                .str.to_datetime(strict=False)
                .alias("time_bucket"),
            )
            .drop("timestamp")
            .drop_nulls(subset=["time_bucket", "value"])
        )
        data_for_model = data_for_model.sort("indicator", "time_bucket")
    else:
        data_for_model = raw_data

    logger.info("  Data for model: %d observations", len(data_for_model))

    # Build web payload
    sample_rows = raw_data.head(20).to_dicts() if n_observations > 0 else []
    per_ind_counts = (
        dict(raw_data.group_by("indicator").len().iter_rows()) if n_observations > 0 else {}
    )
    combined_extractions_sample = [
        {
            "indicator": str(row.get("indicator", "")),
            "value": row.get("value"),
            "tick": row.get("timestamp"),
        }
        for row in sample_rows
    ]

    worker_statuses = stage2_result.get("worker_statuses", [])

    result = {
        "_raw_data": raw_data,
        "_data_for_model": data_for_model,
        "_worker_statuses": worker_statuses,
        "workers": worker_statuses,
        "combined_extractions_sample": combined_extractions_sample,
        "per_indicator_counts": per_ind_counts,
    }
    if "llm_trace" in stage2_result:
        result["llm_trace"] = stage2_result["llm_trace"]
    return result


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
    raw_data = load_parquet(stage2["_raw_data_path"])

    validation_task = validate_extraction(causal_spec, [raw_data])
    validation_report = unwrap_task_result(validation_task)

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
              prior_predictive_samples, _compiled_ssm, llm_trace?}
    """
    from .stages import stage4_orchestrated_flow

    causal_spec = stage1b["causal_spec"]
    data_for_model = load_parquet(stage2["_data_for_model_path"])

    return await stage4_orchestrated_flow(
        causal_spec=causal_spec,
        question=question,
        raw_data=data_for_model,
        enable_literature=enable_literature,
    )


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
        raw_data=load_parquet(stage2["_data_for_model_path"]),
        builder=ssm_builder,
    )


def stage4b_gate(stage4b: dict, override_gates: bool) -> dict:
    """Check parametric identifiability gate."""
    param_id = stage4b.get("parametric_id") or {}
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


def stage5a(
    stage4: dict,
    stage2: dict,
) -> dict:
    """SVI preflight: fast approximate fit before expensive inference.

    Runs the same fit_model task as stage5b but with method="svi" forced.
    Produces SVI diagnostics (ELBO curve) and posterior marginals for
    quick sanity-checking before committing to laplace_em / SMC.
    """
    from .stages import fit_model

    data_for_model = load_parquet(stage2["_data_for_model_path"])

    svi_config = {"method": "svi", "num_steps": 5000, "num_samples": 500}

    fitted = fit_model(stage4, data_for_model, sampler_config=svi_config, builder=None)
    fitted_result = unwrap_task_result(fitted)

    if not fitted_result.get("fitted", False):
        return {
            "inference_metadata": {"method": "svi", "n_samples": 0, "duration_seconds": 0.0},
            "svi_diagnostics": None,
            "posterior_marginals": None,
            "posterior_pairs": None,
            "outcome": "fail",
        }

    return {
        "inference_metadata": {
            "method": "svi",
            "n_samples": 500,
            "duration_seconds": 0.0,
        },
        "svi_diagnostics": fitted_result.get("svi_diagnostics"),
        "posterior_marginals": fitted_result.get("posterior_marginals"),
        "posterior_pairs": fitted_result.get("posterior_pairs"),
        "outcome": "success",
    }


def stage5b(
    stage4: dict,
    stage2: dict,
    inference_method: str | None,
) -> dict:
    """Fit model, run power-scaling and posterior predictive checks.

    Returns: {_fitted_artifact, _ps_result, _ppc_result, power_scaling, ppc,
              inference_metadata, mcmc_diagnostics, svi_diagnostics,
              loo_diagnostics, posterior_marginals, posterior_pairs, outcome}
    """
    from causal_ssm_agent.models.ssm.inference import FittedArtifact
    from causal_ssm_agent.utils.config import get_config

    from .stages import fit_model, run_power_scaling, run_ppc

    config = get_config()
    data_for_model = load_parquet(stage2["_data_for_model_path"])

    sampler_config = config.inference.to_sampler_config(method_override=inference_method)

    fitted = fit_model(stage4, data_for_model, sampler_config=sampler_config, builder=None)
    fitted_result = unwrap_task_result(fitted)

    power_scaling = run_power_scaling(fitted_result)
    ps_result = unwrap_task_result(power_scaling)

    ppc_task = run_ppc(fitted_result)
    ppc_result = unwrap_task_result(ppc_task)

    mcmc_diagnostics = fitted_result.get("mcmc_diagnostics")
    svi_diagnostics = fitted_result.get("svi_diagnostics")
    loo_diagnostics = fitted_result.get("loo_diagnostics")
    posterior_marginals = fitted_result.get("posterior_marginals")
    posterior_pairs = fitted_result.get("posterior_pairs")
    inf_method = fitted_result.get("inference_type", "unknown")

    # Build FittedArtifact — the only shape persisted and consumed by stage 6
    fitted_artifact = FittedArtifact(
        result=fitted_result.get("result"),
        builder=fitted_result.get("builder"),
        times=fitted_result.get("times"),
    )

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
        "_fitted_artifact": fitted_artifact,
        "_ps_result": ps_result,
        "_ppc_result": ppc_result,
        "power_scaling": ps_list,
        "ppc": ppc_result,
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
    stage5b: dict,
    stage1a: dict,
    stage1b: dict,
    stage1b_gate: dict,
) -> dict:
    """Run do-operator interventions and rank treatments.

    Returns: {intervention_results, outcome}
    """
    from prefect.artifacts import create_table_artifact

    from .stages import run_interventions

    fitted_artifact = load_pickle(stage5b["_fitted_result_path"])
    treatments = stage1b_gate["treatments"]
    outcome_name = stage1a.get("outcome_name", "")
    causal_spec = stage1b["causal_spec"]
    ppc_result = stage5b["_ppc_result"]
    ps_result = stage5b["_ps_result"]

    logger.info("=== Stage 6: Treatment Effects ===")
    logger.info("Estimating effects of %d treatments on %s", len(treatments), outcome_name)

    results = run_interventions(
        fitted_artifact,
        treatments,
        outcome_name,
        causal_spec,
        ppc_result,
        ps_result=ps_result,
    )
    intervention_results = unwrap_task_result(results)

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


# Re-export load_stage_state from stage_registry for backward compatibility
# (tests and other code may import it from here)
from .stage_registry import load_stage_state  # noqa: E402, F401
