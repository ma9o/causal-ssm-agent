"""Stage 6 intervention task."""

from __future__ import annotations

from typing import TYPE_CHECKING

from prefect import task

from causal_ssm_agent.flows import get_prefect_logger

logger = get_prefect_logger(__name__)

if TYPE_CHECKING:
    from causal_ssm_agent.models.ssm.inference import FittedArtifact


@task(result_serializer="json")
def run_interventions(
    fitted_artifact: FittedArtifact,
    treatments: list[str],
    outcome: str,
    causal_spec: dict | None = None,
) -> list[dict]:
    """Run do-operator interventions and rank treatments by effect size."""
    from causal_ssm_agent.models.ssm.counterfactual import compute_interventions

    logger.info(
        "Running interventions: treatments=%d outcome=%s fitted=%s",
        len(treatments),
        outcome or "unknown",
        fitted_artifact.result is not None,
    )

    if fitted_artifact.result is None or fitted_artifact.builder is None:
        return [{"treatment": t} for t in treatments]

    builder = fitted_artifact.builder
    result = fitted_artifact.result
    samples = result.get_samples()
    spec = builder._spec

    latent_names = spec.latent_names
    if latent_names is None:
        latent_names = spec.manifest_names or []

    manifest_names = spec.manifest_names or []

    results = compute_interventions(
        samples=samples,
        treatments=treatments,
        outcome=outcome,
        latent_names=latent_names,
        causal_spec=causal_spec,
        manifest_names=manifest_names,
        times=fitted_artifact.times,
    )
    logger.info("Interventions complete: ranked_treatments=%d", len(results))
    return results
