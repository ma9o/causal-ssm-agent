"""Stage computation functions for the causal inference pipeline.

Each function (stage0, stage1a, …, stage6) implements the core logic
for one pipeline stage. Wrapper flows and artifact persistence are
handled by the stage registry (``stage_registry.py``).
"""

from __future__ import annotations

import json
from inspect import isawaitable
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


async def stage0(workspace_id: str) -> dict:
    """Agentic ingestion of raw data.

    Returns dict with web-serializable fields PLUS internal data:
    - ``_df``: Polars DataFrame (not web-serializable)
    - ``_column_descriptions``: dict mapping col -> description
    """
    from .pipeline_helpers import build_stage0_payload
    from .stages import agentic_ingest

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
    from causal_ssm_agent.utils.causal_spec import get_all_treatments, get_outcome_name

    latent_model = stage1a.get("latent_model", {})
    treatments = list(get_all_treatments(latent_model))
    outcome = get_outcome_name(latent_model) or ""
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
    question: str,
    stage0: dict,
    stage1b: dict,
    root_run_id: str | None = None,
    max_windows: int | None = None,
) -> dict:
    """Extract indicator values from data using LLM workers.

    Returns dict with:
    - ``_data_for_model``: encoded DataFrame for modeling (non-continuous types → numeric)
    - ``_worker_statuses``: per-worker status list
    - plus web-serializable worker metadata
    """
    from prefect.task_runners import ThreadPoolTaskRunner

    from causal_ssm_agent.utils.config import get_config

    from .stages import stage2_extraction_flow
    from .stages.stage2_extract import materialize_stage2_outputs

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
        "_worker_statuses": worker_statuses,
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

    from .stages import validate_extraction

    causal_spec = stage1b["causal_spec"]
    data_for_model = load_parquet(stage2["_data_for_model_path"])

    validation_task = validate_extraction(causal_spec, [data_for_model])
    audit_result = unwrap_task_result(validation_task)

    if audit_result:
        indicator_issues = [
            issue
            for audit in audit_result.get("indicators", {}).values()
            for issue in audit.get("validation", {}).get("issues", [])
        ]
        dataset_issues = audit_result.get("dataset_issues", [])
        all_issues = [*indicator_issues, *dataset_issues]
        if not audit_result.get("is_valid", True):
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
    if not report.get("is_valid", True):
        outcome = "fail"
    elif any(
        issue.get("severity") in ("warning", "error")
        for audit in report.get("indicators", {}).values()
        for issue in audit.get("validation", {}).get("issues", [])
    ) or any(i.get("severity") in ("warning", "error") for i in report.get("dataset_issues", [])):
        outcome = "warn"
    else:
        outcome = "success"

    return {**report, "outcome": outcome}


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 4: Model specification + prior elicitation
# ═══════════════════════════════════════════════════════════════════════════════


async def stage4(
    question: str,
    stage1b: dict,
    stage2: dict,
    stage3: dict,
    enable_literature: bool,
) -> dict:
    """Propose model spec, elicit priors, and return the grounded stage-4 result."""
    from .stages import stage4_agentic_flow

    causal_spec = stage1b["causal_spec"]
    data_for_model = load_parquet(stage2["_data_for_model_path"])

    return await stage4_agentic_flow(
        causal_spec=causal_spec,
        question=question,
        data_for_model=data_for_model,
        indicator_audits=stage3["indicators"],
        enable_literature=enable_literature,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 4b: Parametric identifiability
# ═══════════════════════════════════════════════════════════════════════════════


def stage4b(stage4: dict, stage2: dict, ssm_builder: Any = None) -> dict:
    """Parametric identifiability diagnostics.

    Returns: {parametric_id, inference_structure, ...stage4 passthrough}
    """
    from .stages import stage4b_parametric_id_flow

    return stage4b_parametric_id_flow(
        stage4,
        data_for_model=load_parquet(stage2["_data_for_model_path"]),
        builder=ssm_builder,
    )


def stage4b_gate(stage4b: dict, _override_gates: bool) -> dict:
    """Summarize Stage 4b diagnostics without hard-gating the pipeline."""
    param_id = stage4b.get("parametric_id") or {}
    gate_failed = False
    gate_overridden = False
    t_rule: dict = {}

    if param_id.get("checked", False):
        t_rule = param_id.get("t_rule", {})
        if not t_rule.get("satisfies", True):
            logger.warning(
                "Stage 4b warning: T-rule screen failed (%s free params > conservative lower-bound %s moments), continuing",
                t_rule.get("n_free_params"),
                t_rule.get("n_moments"),
            )
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
    if param_id.get("checked", False):
        summary = param_id.get("summary", {})
        has_issues = (
            not t_rule.get("satisfies", True)
            or summary.get("structural_issues")
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
            "inference_metadata": {
                "method": "svi",
                "n_samples": 0,
                "duration_seconds": float(fitted_result.get("duration_seconds", 0.0)),
            },
            "svi_diagnostics": None,
            "posterior_marginals": None,
            "posterior_pairs": None,
            "outcome": "fail",
        }

    return {
        "inference_metadata": {
            "method": "svi",
            "n_samples": int(fitted_result.get("n_samples", 0)),
            "duration_seconds": float(fitted_result.get("duration_seconds", 0.0)),
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

    Returns: {_fitted_artifact, power_scaling, ppc,
              inference_metadata, mcmc_diagnostics, svi_diagnostics, smc_diagnostics,
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
    inf_method = fitted_result.get("inference_type") or sampler_config.get("method", "unknown")

    if not fitted_result.get("fitted", False):
        ps_result = {"checked": False, "error": fitted_result.get("error", "Model not fitted")}
        ppc_result = {"checked": False, "per_variable_warnings": []}
        fitted_artifact = FittedArtifact(
            result=fitted_result.get("result"),
            builder=fitted_result.get("builder"),
            times=fitted_result.get("times"),
            observation_support=getattr(fitted_result.get("runtime"), "observation_support", None),
            ppc_result=ppc_result,
            power_scaling_result=ps_result,
        )
        return {
            "_fitted_artifact": fitted_artifact,
            "power_scaling": [],
            "ppc": ppc_result,
            "inference_metadata": {
                "method": inf_method,
                "n_samples": int(fitted_result.get("n_samples", 0)),
                "duration_seconds": float(fitted_result.get("duration_seconds", 0.0)),
            },
            "mcmc_diagnostics": fitted_result.get("mcmc_diagnostics"),
            "svi_diagnostics": fitted_result.get("svi_diagnostics"),
            "smc_diagnostics": fitted_result.get("smc_diagnostics"),
            "loo_diagnostics": fitted_result.get("loo_diagnostics"),
            "posterior_marginals": fitted_result.get("posterior_marginals"),
            "posterior_pairs": fitted_result.get("posterior_pairs"),
            "outcome": "fail",
        }

    power_scaling = run_power_scaling(fitted_result)
    ps_result = unwrap_task_result(power_scaling)

    ppc_task = run_ppc(fitted_result)
    ppc_result = unwrap_task_result(ppc_task)

    mcmc_diagnostics = fitted_result.get("mcmc_diagnostics")
    svi_diagnostics = fitted_result.get("svi_diagnostics")
    smc_diagnostics = fitted_result.get("smc_diagnostics")
    loo_diagnostics = fitted_result.get("loo_diagnostics")
    posterior_marginals = fitted_result.get("posterior_marginals")
    posterior_pairs = fitted_result.get("posterior_pairs")

    # Build FittedArtifact — the only shape persisted and consumed by stage 6
    fitted_artifact = FittedArtifact(
        result=fitted_result.get("result"),
        builder=fitted_result.get("builder"),
        times=fitted_result.get("times"),
        observation_support=getattr(fitted_result.get("runtime"), "observation_support", None),
        ppc_result=ppc_result,
        power_scaling_result=ps_result,
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
        "power_scaling": ps_list,
        "ppc": ppc_result,
        "inference_metadata": {
            "method": inf_method,
            "n_samples": int(fitted_result.get("n_samples", 0)),
            "duration_seconds": float(fitted_result.get("duration_seconds", 0.0)),
        },
        "mcmc_diagnostics": mcmc_diagnostics,
        "svi_diagnostics": svi_diagnostics,
        "smc_diagnostics": smc_diagnostics,
        "loo_diagnostics": loo_diagnostics,
        "posterior_marginals": posterior_marginals,
        "posterior_pairs": posterior_pairs,
        "outcome": outcome,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 6: Intervention analysis
# ═══════════════════════════════════════════════════════════════════════════════


async def stage6(
    stage5b: dict,
    stage1a: dict,
    stage1b: dict,
    stage1b_gate: dict,
    question: str | None = None,
) -> dict:
    """Run do-operator interventions and rank treatments.

    Returns: {intervention_results, outcome}
    """
    from prefect.artifacts import create_table_artifact

    from causal_ssm_agent.utils.config import get_config
    from causal_ssm_agent.utils.llm import LLMStageContext

    from .stages import run_interventions

    def _first_assistant_summary(trace: Any) -> str | None:
        messages = getattr(trace, "messages", None) or []
        for message in messages:
            if getattr(message, "role", None) != "assistant":
                continue
            content = (getattr(message, "content", "") or "").strip()
            if content:
                return content
        return None

    from causal_ssm_agent.utils.causal_spec import get_outcome_name

    fitted_artifact = load_pickle(stage5b["_fitted_result_path"])
    treatments = stage1b_gate["treatments"]
    outcome_name = get_outcome_name(stage1a.get("latent_model", {})) or ""
    causal_spec = stage1b["causal_spec"]

    logger.info("=== Stage 6: Treatment Effects ===")
    logger.info("Estimating effects of %d treatments on %s", len(treatments), outcome_name)

    results = run_interventions(
        fitted_artifact,
        treatments,
        outcome_name,
        causal_spec,
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

        await _await_artifact(
            create_table_artifact(
                key="treatment-ranking",
                table=[
                    {
                        "rank": i + 1,
                        "treatment": r["treatment"],
                        "effect": (
                            f"{r['effect_size']:+.4f}"
                            if r.get("effect_size") is not None
                            else "---"
                        ),
                        "P(>0)": (
                            f"{r['prob_positive']:.2f}"
                            if r.get("prob_positive") is not None
                            else ""
                        ),
                        "identifiable": "yes" if r.get("identifiable", True) else "NO",
                    }
                    for i, r in enumerate(intervention_results)
                ],
                description="Final treatment effect ranking",
            )
        )

    has_warnings = any(
        r.get("ppc_warnings") or r.get("prior_sensitivity_warning") for r in intervention_results
    )

    top_results = [
        {
            "treatment": entry.get("treatment"),
            "effect_size": entry.get("effect_size"),
            "prob_positive": entry.get("prob_positive"),
            "identifiable": entry.get("identifiable", True),
            "prior_sensitivity_warning": entry.get("prior_sensitivity_warning"),
            "ppc_warning_variables": [
                warning.get("variable")
                for warning in (entry.get("ppc_warnings") or [])
                if isinstance(warning, dict) and warning.get("variable")
            ],
        }
        for entry in intervention_results[:5]
    ]
    power_scaling_issues = [
        {
            "parameter": item.get("parameter"),
            "diagnosis": item.get("diagnosis"),
            "prior_sensitivity": item.get("prior_sensitivity"),
            "likelihood_sensitivity": item.get("likelihood_sensitivity"),
        }
        for item in stage5b.get("power_scaling", [])
        if item.get("diagnosis") in {"prior_dominated", "prior_data_conflict"}
    ][:5]
    ppc_warnings = [
        {
            "variable": warning.get("variable"),
            "issue_type": warning.get("issue_type"),
            "severity": warning.get("severity"),
            "message": warning.get("message"),
        }
        for warning in stage5b.get("ppc", {}).get("per_variable_warnings", [])
    ][:5]

    commentary_input = {
        "question": question,
        "outcome": outcome_name,
        "identifiable_treatments": treatments,
        "excluded_non_identifiable_treatments": sorted(
            stage1b_gate.get("non_identifiable", {}).keys()
        ),
        "top_ranked_effects": top_results,
        "power_scaling_issues": power_scaling_issues,
        "ppc_warnings": ppc_warnings,
        "follow_up_capabilities": {
            "get_model_info": "Inspect variables, measurement, identifiability, diagnostics, and baseline effects.",
            "simulate_intervention": "Run Pearl rung-2 intervention simulations on the fitted generative model.",
            "simulate_counterfactual": "Run Pearl rung-3 counterfactual simulations conditioned on an observed history window.",
        },
    }

    commentary_messages = [
        {
            "role": "system",
            "content": (
                "You are writing the opening commentary for Stage 6 of a causal state-space "
                "analysis. Comment on the treatment-effect results for a technical user. "
                "Be concise and grounded. Do not invent certainty. Mention the strongest "
                "effects, note warnings or identifiability limits, and end by stating that "
                "follow-up chat can inspect model details or run Pearl rung 2 and rung 3 "
                "simulations. Return plain Markdown only."
            ),
        },
        {
            "role": "user",
            "content": (
                "Comment the results of Stage 6.\n\n"
                f"{json.dumps(commentary_input, indent=2, sort_keys=True)}"
            ),
        },
    ]

    async with LLMStageContext("stage-6") as ctx:
        generate = ctx.make_generate(get_config().stage0_ingestion.model)
        await generate(commentary_messages, label="comment-results")
        result = {
            "intervention_results": intervention_results,
            "outcome": "warn" if has_warnings else "success",
        }
        final_summary = _first_assistant_summary(ctx.trace_capture.get("trace"))
        if final_summary:
            result["final_summary"] = final_summary
        return ctx.finalize(result)
