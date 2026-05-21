"""Stage 6 intervention task."""

from __future__ import annotations

from typing import TYPE_CHECKING

from prefect import task

from nof1_causal_lab.flows import get_prefect_logger

logger = get_prefect_logger(__name__)

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.inference import FittedArtifact


@task(result_serializer="json")
def run_interventions(
    fitted_artifact: FittedArtifact,
    treatments: list[str],
    outcome: str,
    causal_spec: dict | None = None,
) -> list[dict]:
    """Run do-operator interventions and rank treatments by effect size."""
    from nof1_causal_lab.models.ssm.counterfactual import compute_interventions
    from nof1_causal_lab.models.ssm.dynamics import posterior_dynamics_from_result

    logger.info(
        "Running interventions: treatments=%d outcome=%s fitted=%s",
        len(treatments),
        outcome or "unknown",
        fitted_artifact.result is not None,
    )

    if fitted_artifact.result is None or fitted_artifact.spec is None:
        return [{"treatment": t} for t in treatments]

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
        causal_spec=causal_spec,
        manifest_names=manifest_names,
        times=fitted_artifact.times,
        lambda_mean=lambda_mean,
    )
    logger.info("Interventions complete: ranked_treatments=%d", len(results))
    return results
