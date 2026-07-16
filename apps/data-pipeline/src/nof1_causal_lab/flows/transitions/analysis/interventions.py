"""analysis intervention task."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from nof1_causal_lab.models.causal_proofs import CertifiedCausalAnalysis


def run_interventions(
    analysis: CertifiedCausalAnalysis,
) -> list[dict]:
    """Run interventions only after identification and engine proofs are joined."""
    from nof1_causal_lab.models.ssm.counterfactual import compute_interventions
    from nof1_causal_lab.models.ssm.dynamics import posterior_dynamics_from_result

    fitted_artifact = analysis.posterior.artifact
    treatments = analysis.treatments
    outcome = analysis.outcome
    logger.info(
        "Running interventions: treatments=%d outcome=%s fitted=%s",
        len(treatments),
        outcome or "unknown",
        True,
    )

    result = fitted_artifact.result
    spec = fitted_artifact.spec
    samples = result.get_samples()

    latent_names = spec.latent_names
    if latent_names is None:
        latent_names = spec.manifest_names or []

    manifest_names = spec.manifest_names or []

    posterior_dynamics = posterior_dynamics_from_result(spec, result)
    lambda_draws = samples.get("lambda")
    lambda_mean = None
    if lambda_draws is not None:
        lambda_mean = lambda_draws.mean(axis=0) if lambda_draws.ndim == 3 else lambda_draws
    results = compute_interventions(
        param_samples=posterior_dynamics.param_samples,
        vector_field=posterior_dynamics.vector_field,
        treatments=treatments,
        outcome=outcome,
        latent_names=latent_names,
        causal_design=analysis.causal_design.model_dump(mode="json"),
        manifest_names=manifest_names,
        times=fitted_artifact.times,
        lambda_mean=lambda_mean,
    )
    logger.info("Interventions complete: ranked_treatments=%d", len(results))
    return results
