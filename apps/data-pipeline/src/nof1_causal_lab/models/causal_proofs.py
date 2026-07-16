"""Nominal evidence required before emitting numeric causal results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.causal_design import CausalDesign
    from nof1_causal_lab.models.ssm.inference.types import FittedArtifact


@dataclass(frozen=True, slots=True)
class CausalDesignRef:
    """Workspace-local identity of the causal design supporting a claim."""

    workspace_id: str
    version: int

    def __post_init__(self) -> None:
        if not self.workspace_id:
            raise ValueError("workspace_id must not be empty")
        if self.version < 1:
            raise ValueError("causal design version must be positive")


@dataclass(frozen=True, slots=True)
class PosteriorProvenance:
    """Artifact lineage binding a posterior to its compiled causal design."""

    causal_design: CausalDesignRef
    compiled_ssm_version: int
    panel_version: int

    def __post_init__(self) -> None:
        if self.compiled_ssm_version < 1:
            raise ValueError("compiled SSM version must be positive")
        if self.panel_version < 1:
            raise ValueError("panel version must be positive")


@dataclass(frozen=True, slots=True)
class IdentifiedEstimand:
    """Positive identification evidence for one treatment/outcome estimand."""

    causal_design: CausalDesignRef
    treatment: str
    outcome: str
    method: str
    estimand: str


@dataclass(frozen=True, slots=True)
class ReportablePosterior:
    """A persisted posterior certified as production particle-MCMC output."""

    artifact: FittedArtifact

    def __post_init__(self) -> None:
        _validate_reportable_artifact(self.artifact)


@dataclass(frozen=True, slots=True)
class CertifiedCausalAnalysis:
    """Identification and particle-posterior evidence joined by provenance."""

    causal_design: CausalDesign
    causal_design_ref: CausalDesignRef
    estimands: tuple[IdentifiedEstimand, ...]
    posterior: ReportablePosterior

    def __post_init__(self) -> None:
        if not self.estimands:
            raise ValueError("at least one identified estimand is required")
        posterior_design = self.posterior.artifact.provenance.causal_design
        if posterior_design != self.causal_design_ref:
            raise ValueError(
                "causal design payload and posterior provenance reference different designs"
            )
        outcomes = {estimand.outcome for estimand in self.estimands}
        if len(outcomes) != 1:
            raise ValueError("all identified estimands must target the same outcome")
        treatments = [estimand.treatment for estimand in self.estimands]
        if len(treatments) != len(set(treatments)):
            raise ValueError("identified estimands must not contain duplicate treatments")
        status = self.causal_design.identifiability
        for estimand in self.estimands:
            if estimand.causal_design != self.causal_design_ref:
                raise ValueError(
                    "identification evidence and posterior provenance reference different "
                    "causal designs"
                )
            details = (
                status.identifiable_treatments.get(estimand.treatment)
                if status is not None
                else None
            )
            if details is None:
                raise ValueError(f"effect of {estimand.treatment!r} is not identified")
            if details.method != estimand.method or details.estimand != estimand.estimand:
                raise ValueError("identification evidence does not match the causal design")

    @property
    def treatments(self) -> list[str]:
        return [estimand.treatment for estimand in self.estimands]

    @property
    def outcome(self) -> str:
        return self.estimands[0].outcome


def certify_identified_estimand(
    causal_design: CausalDesign,
    *,
    causal_design_ref: CausalDesignRef,
    treatment: str,
    outcome: str,
) -> IdentifiedEstimand:
    """Validate and materialize identification evidence for one estimand."""
    declared_outcomes = {
        construct.name for construct in causal_design.latent.constructs if construct.is_outcome
    }
    if outcome not in declared_outcomes:
        raise ValueError(f"{outcome!r} is not the declared outcome in the causal design")
    construct_names = {construct.name for construct in causal_design.latent.constructs}
    if treatment not in construct_names:
        raise ValueError(f"{treatment!r} is not a construct in the causal design")
    status = causal_design.identifiability
    details = status.identifiable_treatments.get(treatment) if status is not None else None
    if details is None:
        raise ValueError(f"effect of {treatment!r} on {outcome!r} is not identified")
    return IdentifiedEstimand(
        causal_design=causal_design_ref,
        treatment=treatment,
        outcome=outcome,
        method=details.method,
        estimand=details.estimand,
    )


def _validate_reportable_artifact(artifact: FittedArtifact) -> None:
    from nof1_causal_lab.models.ssm.inference.types import ParticleMCMCPosterior

    if not isinstance(artifact.result, ParticleMCMCPosterior):
        raise TypeError("reporting requires a ParticleMCMCPosterior")
    evidence = artifact.result.evidence
    if evidence.engine != "marginal_particle_gibbs":
        raise ValueError("posterior was not produced by marginalized Particle Gibbs")
    if evidence.latent_transition != "euler_maruyama":
        raise ValueError("posterior did not target the nonlinear Euler-Maruyama transition")
    samples = artifact.result.get_samples()
    if not samples:
        raise ValueError("posterior contains no retained samples")
    draw_counts = {int(values.shape[0]) for values in samples.values()}
    if len(draw_counts) != 1 or next(iter(draw_counts)) < 1:
        raise ValueError("posterior sample sites must share a positive draw dimension")


def certify_reportable_posterior(artifact: FittedArtifact) -> ReportablePosterior:
    """Validate particle-engine evidence and non-empty retained posterior draws."""
    return ReportablePosterior(artifact=artifact)
