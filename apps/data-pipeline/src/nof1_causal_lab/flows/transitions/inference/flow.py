"""posterior orchestration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from nof1_causal_lab.machine.errors import ModelFitError
from nof1_causal_lab.models.ssm.inference import FittedArtifact

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.compile.contracts import CompiledSSMArtifact
    from nof1_causal_lab.sampler_config import SamplerConfig


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


def build_sampler_config(inference_method: str | None) -> SamplerConfig:
    """Resolve the sampler configuration from config + optional override."""
    from nof1_causal_lab.utils.config import get_config

    config = get_config()
    sampler_config = config.inference.to_sampler_config(method_override=inference_method)
    if sampler_config.get("method") in {
        "aux_kalman_mcmc",
        "pit_particle_mgrad",
        "marginal_particle_gibbs",
    }:
        sampler_config["retain_latent_paths"] = True
    return sampler_config


def run_inference_with_data(
    *,
    compiled_ssm: CompiledSSMArtifact | None,
    data_for_model: Any,
    sampler_config: SamplerConfig,
    workspace_id: str,
    compute_loo_diagnostics: bool,
) -> dict[str, Any]:
    """Fit the model from materialized model-spec/2 artifacts and shape posterior."""
    from .fit import fit_model, run_ppc

    fitted_result = fit_model(
        compiled_ssm,
        data_for_model,
        sampler_config=sampler_config,
        workspace_id=workspace_id,
        wait_for_compile_cache=True,
        compute_loo_diagnostics=compute_loo_diagnostics,
    )
    inf_method = fitted_result.get("inference_type") or sampler_config.get("method", "unknown")

    if not fitted_result.get("fitted", False):
        raise ModelFitError(
            fitted_result.get("error") or "model fit failed",
            transition_id="posterior",
            diagnostics={
                "inference_metadata": {
                    "method": inf_method,
                    "n_samples": int(fitted_result.get("n_samples", 0)),
                    "duration_seconds": float(fitted_result.get("duration_seconds", 0.0)),
                },
                "mcmc_diagnostics": fitted_result.get("mcmc_diagnostics"),
                "smc_diagnostics": fitted_result.get("smc_diagnostics"),
            },
        )

    ppc_result = run_ppc(fitted_result)

    fitted_artifact = FittedArtifact(
        result=fitted_result.get("result"),
        spec=fitted_result.get("spec"),
        times=fitted_result.get("times"),
        observation_support=getattr(fitted_result.get("runtime"), "observation_support", None),
        ppc_result=ppc_result,
    )

    _log_ppc(ppc_result)

    return {
        "_fitted_artifact": fitted_artifact,
        "ppc": ppc_result,
        "inference_metadata": {
            "method": inf_method,
            "n_samples": int(fitted_result.get("n_samples", 0)),
            "duration_seconds": float(fitted_result.get("duration_seconds", 0.0)),
        },
        "mcmc_diagnostics": fitted_result.get("mcmc_diagnostics"),
        "smc_diagnostics": fitted_result.get("smc_diagnostics"),
        "loo_diagnostics": fitted_result.get("loo_diagnostics"),
        "posterior_marginals": fitted_result.get("posterior_marginals"),
        "posterior_pairs": fitted_result.get("posterior_pairs"),
    }
