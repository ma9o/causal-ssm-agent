"""Proof-carrying boundaries for numeric causal analysis."""

from dataclasses import replace
from typing import Any

import jax.numpy as jnp
import pytest

from nof1_causal_lab.artifacts import (
    CausalDesign,
    CausalEdge,
    Construct,
    IdentifiabilityStatus,
    IdentifiedTreatmentStatus,
    LatentStructure,
    MeasurementStructure,
    Role,
    TemporalStatus,
)
from nof1_causal_lab.models.causal_proofs import (
    CausalDesignRef,
    CertifiedCausalAnalysis,
    PosteriorProvenance,
    certify_identified_estimand,
    certify_reportable_posterior,
)
from nof1_causal_lab.models.ssm.inference.types import (
    FittedArtifact,
    ParticleMCMCPosterior,
    WarmupProposal,
)
from tests.ssm_spec_fixtures import block_ssm_spec, full_dense_matrix_dynamics_spec


def _design() -> CausalDesign:
    return CausalDesign(
        latent=LatentStructure(
            constructs=[
                Construct(
                    name="treatment",
                    description="Treatment",
                    role=Role.EXOGENOUS,
                    temporal_status=TemporalStatus.TIME_VARYING,
                ),
                Construct(
                    name="outcome",
                    description="Outcome",
                    role=Role.ENDOGENOUS,
                    is_outcome=True,
                    temporal_status=TemporalStatus.TIME_VARYING,
                ),
            ],
            edges=[CausalEdge(cause="treatment", effect="outcome", description="Test edge")],
        ),
        measurement=MeasurementStructure(indicators=[], model_clock="1d"),
        identifiability=IdentifiabilityStatus(
            identifiable_treatments={
                "treatment": IdentifiedTreatmentStatus(
                    method="do_calculus",
                    estimand="E[outcome | do(treatment)]",
                )
            }
        ),
    )


def _artifact(
    *,
    workspace_id: str = "workspace",
    causal_design_version: int = 1,
) -> FittedArtifact:
    return FittedArtifact(
        result=ParticleMCMCPosterior(_samples={"vf_0_decay": jnp.ones((2, 2), dtype=jnp.float32)}),
        spec=block_ssm_spec(
            n_latent=2,
            n_manifest=0,
            dynamics_spec=full_dense_matrix_dynamics_spec(2),
            latent_names=["treatment", "outcome"],
            manifest_names=[],
        ),
        times=jnp.array([0.0, 1.0], dtype=jnp.float32),
        provenance=PosteriorProvenance(
            causal_design=CausalDesignRef(
                workspace_id=workspace_id,
                version=causal_design_version,
            ),
            compiled_ssm_version=3,
            panel_version=4,
        ),
    )


def test_identification_proof_is_estimand_specific() -> None:
    design = _design()
    design_ref = CausalDesignRef(workspace_id="workspace", version=1)

    proof = certify_identified_estimand(
        design,
        causal_design_ref=design_ref,
        treatment="treatment",
        outcome="outcome",
    )

    assert proof.treatment == "treatment"
    assert proof.outcome == "outcome"
    assert proof.method == "do_calculus"
    assert proof.estimand == "E[outcome | do(treatment)]"


def test_identification_proof_rejects_unidentified_treatment() -> None:
    with pytest.raises(ValueError, match="is not identified"):
        certify_identified_estimand(
            _design(),
            causal_design_ref=CausalDesignRef(workspace_id="workspace", version=1),
            treatment="outcome",
            outcome="outcome",
        )


def test_reportable_posterior_rejects_warmup_from_untyped_boundary() -> None:
    artifact = _artifact()
    untyped_warmup: Any = WarmupProposal(
        _samples={"vf_0_decay": jnp.ones((2, 2), dtype=jnp.float32)}
    )
    artifact = replace(artifact, result=untyped_warmup)

    with pytest.raises(TypeError, match="requires a ParticleMCMCPosterior"):
        certify_reportable_posterior(artifact)


def test_reportable_posterior_rejects_empty_particle_draws() -> None:
    artifact = _artifact()
    artifact = replace(artifact, result=ParticleMCMCPosterior(_samples={}))

    with pytest.raises(ValueError, match="no retained samples"):
        certify_reportable_posterior(artifact)


def test_causal_analysis_rejects_cross_design_evidence() -> None:
    design = _design()
    estimand = certify_identified_estimand(
        design,
        causal_design_ref=CausalDesignRef(workspace_id="workspace", version=2),
        treatment="treatment",
        outcome="outcome",
    )

    with pytest.raises(ValueError, match="different designs"):
        CertifiedCausalAnalysis(
            causal_design=design,
            causal_design_ref=CausalDesignRef(workspace_id="workspace", version=2),
            estimands=(estimand,),
            posterior=certify_reportable_posterior(_artifact(causal_design_version=1)),
        )


def test_causal_analysis_joins_matching_proofs() -> None:
    design = _design()
    design_ref = CausalDesignRef(workspace_id="workspace", version=1)
    analysis = CertifiedCausalAnalysis(
        causal_design=design,
        causal_design_ref=design_ref,
        estimands=(
            certify_identified_estimand(
                design,
                causal_design_ref=design_ref,
                treatment="treatment",
                outcome="outcome",
            ),
        ),
        posterior=certify_reportable_posterior(_artifact()),
    )

    assert analysis.treatments == ["treatment"]
    assert analysis.outcome == "outcome"
