"""Stage 5b orchestration."""

from __future__ import annotations

from typing import Any

from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.flows.run_store import load_parquet, unwrap_task_result
from causal_ssm_agent.models.ssm.inference import FittedArtifact

logger = get_prefect_logger(__name__)


def _build_failed_fit_result(
    fitted_result: dict[str, Any],
    *,
    inf_method: str,
) -> dict[str, Any]:
    ppc_result = {"checked": False, "per_variable_warnings": []}
    ps_result = {"checked": False, "error": fitted_result.get("error", "Model not fitted")}
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
        "fail_reason": "model_fit_failed",
    }


def _build_power_scaling_entries(ps_result: dict[str, Any]) -> list[dict[str, Any]]:
    ps_list: list[dict[str, Any]] = []
    if not ps_result.get("checked", False):
        return ps_list

    diagnosis = ps_result.get("diagnosis", {})
    prior_sensitivity = ps_result.get("prior_sensitivity", {})
    likelihood_sensitivity = ps_result.get("likelihood_sensitivity", {})
    psis_k_hat = ps_result.get("psis_k_hat", {})
    for parameter in diagnosis:
        entry = {
            "parameter": parameter,
            "diagnosis": diagnosis[parameter],
            "prior_sensitivity": prior_sensitivity.get(parameter, 0.0),
            "likelihood_sensitivity": likelihood_sensitivity.get(parameter, 0.0),
        }
        if parameter in psis_k_hat:
            entry["psis_k_hat"] = psis_k_hat[parameter]
        ps_list.append(entry)
    return ps_list


def _log_power_scaling(ps_result: dict[str, Any]) -> None:
    logger.info("--- Power-Scaling Sensitivity ---")
    if not ps_result.get("checked", False):
        logger.info("  Skipped: %s", ps_result.get("error", "unknown"))
        return

    diagnosis = ps_result.get("diagnosis", {})
    prior_dominated = [name for name, verdict in diagnosis.items() if verdict == "prior_dominated"]
    conflicts = [name for name, verdict in diagnosis.items() if verdict == "prior_data_conflict"]
    if prior_dominated:
        logger.warning("  Prior-dominated parameters: %s", prior_dominated)
    if conflicts:
        logger.warning("  Prior-data conflicts: %s", conflicts)
    if not prior_dominated and not conflicts:
        logger.info("  All parameters well-identified")


def _log_ppc(ppc_result: dict[str, Any]) -> None:
    logger.info("--- Posterior Predictive Checks ---")
    if not ppc_result.get("checked", False):
        logger.info("  Skipped: %s", ppc_result.get("error", "unknown"))
        return

    warnings = ppc_result.get("per_variable_warnings", [])
    if warnings:
        logger.warning("  %d warning(s):", len(warnings))
        for warning in warnings:
            logger.warning("    - %s: %s", warning["variable"], warning["message"])
        return

    logger.info("  All checks passed")


def run_stage5b(
    stage4: dict,
    stage2: dict,
    inference_method: str | None,
    workspace_id: str,
) -> dict[str, Any]:
    """Fit the model, run diagnostics, and shape the Stage 5b payload."""
    from causal_ssm_agent.utils.config import get_config

    config = get_config()
    data_for_model = load_parquet(stage2["_data_for_model_path"])
    sampler_config = config.inference.to_sampler_config(method_override=inference_method)

    return run_stage5b_with_data(
        compiled_ssm=stage4.get("_compiled_ssm"),
        data_for_model=data_for_model,
        sampler_config=sampler_config,
        workspace_id=workspace_id,
        compute_loo_diagnostics=config.inference.compute_loo_diagnostics,
    )


def run_stage5b_with_data(
    *,
    compiled_ssm: dict | None,
    data_for_model: Any,
    sampler_config: dict[str, Any],
    workspace_id: str,
    compute_loo_diagnostics: bool,
) -> dict[str, Any]:
    """Fit the model from materialized Stage 4/2 artifacts and shape Stage 5b."""
    from .fit import fit_model, run_power_scaling, run_ppc

    fitted = fit_model(
        compiled_ssm,
        data_for_model,
        sampler_config=sampler_config,
        workspace_id=workspace_id,
        wait_for_compile_cache=True,
        compute_loo_diagnostics=compute_loo_diagnostics,
    )
    fitted_result = unwrap_task_result(fitted)
    inf_method = fitted_result.get("inference_type") or sampler_config.get("method", "unknown")

    if not fitted_result.get("fitted", False):
        return _build_failed_fit_result(fitted_result, inf_method=inf_method)

    power_scaling = run_power_scaling(fitted_result)
    ps_result = unwrap_task_result(power_scaling)

    ppc_task = run_ppc(fitted_result)
    ppc_result = unwrap_task_result(ppc_task)

    fitted_artifact = FittedArtifact(
        result=fitted_result.get("result"),
        builder=fitted_result.get("builder"),
        times=fitted_result.get("times"),
        observation_support=getattr(fitted_result.get("runtime"), "observation_support", None),
        ppc_result=ppc_result,
        power_scaling_result=ps_result,
    )

    _log_power_scaling(ps_result)
    _log_ppc(ppc_result)

    power_scaling_entries = _build_power_scaling_entries(ps_result)
    has_ppc_warnings = bool(ppc_result.get("per_variable_warnings"))
    has_ps_issues = any(
        entry["diagnosis"] in ("prior_dominated", "prior_data_conflict")
        for entry in power_scaling_entries
    )
    outcome = "warn" if (has_ppc_warnings or has_ps_issues) else "success"

    return {
        "_fitted_artifact": fitted_artifact,
        "power_scaling": power_scaling_entries,
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
        "outcome": outcome,
    }
