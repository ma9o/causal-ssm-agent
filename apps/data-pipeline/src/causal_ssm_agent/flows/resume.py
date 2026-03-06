"""Resume pipeline from a specific stage using saved state.

Used for in-place replay after the user refines an LLM stage output
via the interactive refinement panel.
"""

import json
import logging
from pathlib import Path

import polars as pl
from prefect import flow

from causal_ssm_agent.flows.pipeline import (
    RESULT_STORAGE,
    format_schema_for_llm,
    map_columns_to_indicators,
)

from .stages import (
    fit_model,
    persist_web_result,
    propose_measurement_with_identifiability_fix,
    run_interventions,
    run_power_scaling,
    run_ppc,
    stage4_orchestrated_flow,
    stage4b_parametric_id_flow,
    validate_extraction,
)

logger = logging.getLogger(__name__)


# The linear stage ordering used to determine "downstream" stages
STAGE_ORDER = [
    "stage-0",
    "stage-1a",
    "stage-1b",
    "stage-2",
    "stage-3",
    "stage-4",
    "stage-4b",
    "stage-5",
    "stage-6",
]


def _load_stage_json(run_dir: Path, stage_id: str) -> dict:
    """Load a persisted stage JSON from the run directory."""
    path = run_dir / f"{stage_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _clear_downstream(run_dir: Path, start_from: str) -> None:
    """Delete stage JSONs for all stages downstream of (and including) start_from."""
    idx = STAGE_ORDER.index(start_from)
    for stage_id in STAGE_ORDER[idx:]:
        path = run_dir / f"{stage_id}.json"
        if path.exists():
            path.unlink()
            logger.info("Cleared %s", path.name)


@flow(
    persist_result=True,
    result_storage=RESULT_STORAGE,
    result_serializer="pickle",
)
async def resume_pipeline(
    original_run_id: str,
    start_from: str,
    inference_method: str | None = None,
    override_gates: bool | None = None,
):
    """Resume a pipeline run from a specific stage.

    Loads persisted state from a previous run and re-executes
    from ``start_from`` onward.  The caller is expected to have
    already overwritten the stage JSON that was refined.

    Args:
        original_run_id: The Prefect flow run ID of the original run.
        start_from: Stage ID to resume from (e.g. "stage-1b").
        inference_method: Override inference method.
        override_gates: Continue past stage failures.
    """
    from causal_ssm_agent.utils.config import get_config

    config = get_config()
    gates_overridden = (
        override_gates if override_gates is not None else config.pipeline.override_gates
    )

    run_dir = RESULT_STORAGE / original_run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # ── Load base artifacts ──
    question = (run_dir / "question.txt").read_text(encoding="utf-8")
    ingested_df = pl.read_parquet(run_dir / "ingested_df.parquet")

    stage0_data = _load_stage_json(run_dir, "stage-0")
    column_descriptions = {
        cd["name"]: cd["description"] for cd in stage0_data["column_descriptions"]
    }

    # ── Get the new run directory (this resume flow has its own flow run ID) ──
    from prefect.context import get_run_context

    new_run_dir = RESULT_STORAGE / str(get_run_context().flow_run.id)
    new_run_dir.mkdir(parents=True, exist_ok=True)

    # Copy base artifacts to new run
    ingested_df.write_parquet(new_run_dir / "ingested_df.parquet")
    (new_run_dir / "question.txt").write_text(question, encoding="utf-8")

    # ── Reconstruct state from completed stages ──
    start_idx = STAGE_ORDER.index(start_from)

    # Always need stage 0 data — copy it
    persist_web_result("stage-0", stage0_data)

    # Load stage 1a if completed and we're starting after it
    latent_model = None
    outcome = None
    treatments = None
    if start_idx > STAGE_ORDER.index("stage-1a"):
        stage1a_data = _load_stage_json(run_dir, "stage-1a")
        latent_model = stage1a_data["latent_model"]
        outcome = stage1a_data["outcome_name"]
        treatments = stage1a_data["treatments"]
        persist_web_result("stage-1a", stage1a_data)

    # Load stage 1b if completed and we're starting after it
    causal_spec = None
    if start_idx > STAGE_ORDER.index("stage-1b"):
        stage1b_data = _load_stage_json(run_dir, "stage-1b")
        causal_spec = stage1b_data["causal_spec"]
        persist_web_result("stage-1b", stage1b_data)

    # ══════════════════════════════════════════════════════════════════════
    # Run from start_from onward
    # ══════════════════════════════════════════════════════════════════════

    # ── Stage 1b ──
    if start_from == "stage-1b" or (start_idx <= STAGE_ORDER.index("stage-1b") and latent_model):
        if latent_model is None:
            stage1a_data = _load_stage_json(run_dir, "stage-1a")
            latent_model = stage1a_data["latent_model"]
            outcome = stage1a_data["outcome_name"]
            treatments = stage1a_data["treatments"]
            persist_web_result("stage-1a", stage1a_data)

        dataset_schema = format_schema_for_llm(ingested_df, column_descriptions)
        stage1b_result = await propose_measurement_with_identifiability_fix(
            question,
            latent_model,
            data_sample=[dataset_schema],
            dataset_summary=f"{ingested_df.shape[0]} rows x {ingested_df.shape[1]} columns",
        )
        causal_spec = stage1b_result["causal_spec"]

        non_identifiable = stage1b_result.get("identifiability_status", {}).get(
            "non_identifiable_treatments", {}
        )
        gate_1b_failed = False
        if non_identifiable:
            treatments = [t for t in treatments if t not in non_identifiable]
            if not treatments:
                gate_1b_failed = True

        stage1b_web_data: dict = {
            "causal_spec": causal_spec,
            "llm_trace": stage1b_result.get("llm_trace"),
            "outcome": "fail" if non_identifiable else "success",
        }
        if gates_overridden and gate_1b_failed:
            stage1b_web_data["gate_overridden"] = {
                "reason": "No identifiable treatments remain — all blocked by unobserved confounders"
            }
        persist_web_result("stage-1b", stage1b_web_data)

        if gate_1b_failed and not gates_overridden:
            raise RuntimeError("No identifiable treatment effects remain after filtering.")

    # ── Stage 2: Column-to-indicator mapping ──
    if start_idx <= STAGE_ORDER.index("stage-2"):
        assert causal_spec is not None
        raw_data = map_columns_to_indicators(ingested_df, causal_spec)

        from causal_ssm_agent.utils.aggregations import (
            aggregate_worker_measurements,
            flatten_aggregated_data,
        )

        worker_dfs = [raw_data]
        aggregated_result = aggregate_worker_measurements(worker_dfs, causal_spec)
        data_for_model = (
            flatten_aggregated_data(aggregated_result) if aggregated_result else raw_data
        )

        sample_rows = raw_data.head(20).to_dicts() if len(raw_data) > 0 else []
        per_ind_counts = (
            dict(raw_data.group_by("indicator").len().iter_rows()) if len(raw_data) > 0 else {}
        )
        combined_extractions_sample = [
            {
                "indicator": str(row.get("indicator", "")),
                "value": row.get("value"),
                "timestamp": str(row.get("timestamp"))
                if row.get("timestamp") is not None
                else None,
            }
            for row in sample_rows
        ]
        persist_web_result(
            "stage-2",
            {
                "outcome": "success",
                "workers": [],
                "combined_extractions_sample": combined_extractions_sample,
                "per_indicator_counts": per_ind_counts,
            },
        )
    else:
        # Need to reconstruct data_for_model from saved state
        assert causal_spec is not None
        raw_data = map_columns_to_indicators(ingested_df, causal_spec)
        from causal_ssm_agent.utils.aggregations import (
            aggregate_worker_measurements,
            flatten_aggregated_data,
        )

        worker_dfs = [raw_data]
        aggregated_result = aggregate_worker_measurements(worker_dfs, causal_spec)
        data_for_model = (
            flatten_aggregated_data(aggregated_result) if aggregated_result else raw_data
        )

    # ── Stage 3: Validation ──
    if start_idx <= STAGE_ORDER.index("stage-3"):
        assert causal_spec is not None
        validation_report = validate_extraction(causal_spec, [raw_data])
        validation_report_web = validation_report or {
            "is_valid": False,
            "issues": [],
            "per_indicator_health": [],
        }
        if not validation_report_web.get("is_valid", True):
            stage3_outcome = "fail"
        elif any(
            i.get("severity") in ("warning", "error")
            for i in validation_report_web.get("issues", [])
        ):
            stage3_outcome = "warn"
        else:
            stage3_outcome = "success"
        persist_web_result(
            "stage-3", {"outcome": stage3_outcome, "validation_report": validation_report_web}
        )

    # ── Stage 4: Model Specification ──
    if start_idx <= STAGE_ORDER.index("stage-4"):
        assert causal_spec is not None
        lit_enabled = config.stage4_prior_elicitation.literature_search.enabled

        stage4_result = await stage4_orchestrated_flow(
            causal_spec=causal_spec,
            question=question,
            raw_data=data_for_model,
            enable_literature=lit_enabled,
        )
        model_spec = stage4_result.get("model_spec", {})
        persist_web_result(
            "stage-4",
            {
                "model_spec": model_spec,
                "priors": list(stage4_result.get("priors", {}).values()),
                "llm_trace": stage4_result.get("llm_trace"),
                "prior_predictive_samples": stage4_result.get("prior_predictive_samples"),
            },
        )
    else:
        # Load stage 4 result from saved JSON and reconstruct minimal state
        stage4_web = _load_stage_json(run_dir, "stage-4")
        priors_dict = {p["parameter"]: p for p in stage4_web.get("priors", [])}
        stage4_result = {
            "model_spec": stage4_web["model_spec"],
            "priors": priors_dict,
            "causal_spec": causal_spec,
            "llm_trace": stage4_web.get("llm_trace"),
        }
        persist_web_result("stage-4", stage4_web)

    # ── Pre-build SSM builder ──
    from causal_ssm_agent.models.ssm_builder import build_ssm_builder

    try:
        builder = build_ssm_builder(
            model_spec=stage4_result["model_spec"],
            priors=stage4_result["priors"],
            raw_data=data_for_model,
            causal_spec=stage4_result.get("causal_spec"),
        )
    except Exception:
        logger.warning("Pre-building SSM builder failed", exc_info=True)
        builder = None

    # ── Stage 4b: Parametric Identifiability ──
    if start_idx <= STAGE_ORDER.index("stage-4b"):
        stage4_result = stage4b_parametric_id_flow(
            stage4_result, raw_data=data_for_model, builder=builder
        )
        param_id = stage4_result.get("parametric_id", {})
        gate_4b_failed = False
        gate_4b_overridden = False
        if param_id.get("checked", False):
            t_rule = param_id.get("t_rule", {})
            if not t_rule.get("satisfies", True):
                gate_4b_failed = True
                if gates_overridden:
                    gate_4b_overridden = True

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
                "reason": f"T-rule violated: {t_rule.get('n_free_params')} free parameters "
                f"> {t_rule.get('n_moments')} moment conditions"
            }
        persist_web_result("stage-4b", stage4b_web_data)

        if gate_4b_failed and not gates_overridden:
            raise RuntimeError("T-rule violated. Model is provably non-identified.")

    # ── Stages 5 & 6: Inference and Treatment Effects ──
    assert treatments is not None and outcome is not None
    sampler_config = (
        config.inference.to_sampler_config(method_override=inference_method)
        if inference_method
        else None
    )

    if config.inference.gpu:
        from causal_ssm_agent.flows.gpu_inference import run_stage5_gpu

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
        fitted = fit_model(
            stage4_result, data_for_model, sampler_config=sampler_config, builder=builder
        )
        power_scaling = run_power_scaling(fitted, data_for_model)
        ps_result = power_scaling.result() if hasattr(power_scaling, "result") else power_scaling
        ppc_task = run_ppc(fitted, data_for_model)
        ppc_result = ppc_task.result() if hasattr(ppc_task, "result") else ppc_task
        results = run_interventions(
            fitted, treatments, outcome, causal_spec, ppc_result, ps_result=ps_result
        )
        intervention_results = results.result() if hasattr(results, "result") else results

    # ── Persist stage 5 ──
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

    inf_meta = {
        "method": (
            fitted.result().get("inference_type", "unknown")
            if not config.inference.gpu and hasattr(fitted, "result")
            else fitted.get("inference_type", "unknown")
            if not config.inference.gpu
            else gpu_result.get("mcmc_diagnostics", {}).get("method", "unknown")
        ),
        "n_samples": 10000,
        "duration_seconds": 0.0,
    }

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
        r.get("ppc_warnings") or r.get("prior_sensitivity_warning") for r in intervention_results
    )
    stage6_data = {
        "outcome": "warn" if stage6_has_warnings else "success",
        "intervention_results": intervention_results,
    }
    persist_web_result("stage-6", stage6_data)

    return {**stage5_data, **stage6_data}
