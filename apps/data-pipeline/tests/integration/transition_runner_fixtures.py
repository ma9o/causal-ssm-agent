"""Small canonical artifacts for fixture-backed stage-runner tests."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import polars as pl

from nof1_causal_lab.machine.artifact_files import (
    json_filename,
    parquet_filename,
)
from nof1_causal_lab.machine.artifacts import ArtifactVersionInfo, EpisodeState

if TYPE_CHECKING:
    from nof1_causal_lab.machine.store import ArtifactStore


def panel_frame(n_days: int = 20) -> pl.DataFrame:
    start = datetime(2024, 1, 1)
    rows: list[dict[str, Any]] = []
    for i in range(n_days):
        support_start = start + timedelta(days=i)
        support_end = support_start + timedelta(days=1)
        anchor = support_end
        rows.extend(
            [
                {
                    "indicator": "stress_score",
                    "value": float(1 + (i % 5)),
                    "anchor_time": anchor.isoformat(),
                    "support_kind": "interval",
                    "summary_operator": "mean",
                    "anchor_policy": "support_end",
                    "observation_window": "1d",
                    "support_start": support_start.isoformat(),
                    "support_end": support_end.isoformat(),
                },
                {
                    "indicator": "sleep_score",
                    "value": float(8 - (i % 4)),
                    "anchor_time": anchor.isoformat(),
                    "support_kind": "interval",
                    "summary_operator": "mean",
                    "anchor_policy": "support_end",
                    "observation_window": "1d",
                    "support_start": support_start.isoformat(),
                    "support_end": support_end.isoformat(),
                },
            ]
        )
    return pl.DataFrame(rows)


def statistical_model_spec() -> dict[str, Any]:
    return {
        "likelihoods": [
            {
                "variable": "stress_score",
                "distribution": "gaussian",
                "link": "identity",
                "reasoning": "Continuous stress score.",
            },
            {
                "variable": "sleep_score",
                "distribution": "gaussian",
                "link": "identity",
                "reasoning": "Continuous sleep score.",
            },
        ],
        "parameters": [
            {
                "name": "rho_Stress",
                "role": "ar_coefficient",
                "constraint": "unit_interval",
                "description": "Daily persistence of Stress.",
            }
        ],
    }


def authored_priors() -> dict[str, Any]:
    return {
        "rho_Stress": {
            "parameter": "rho_Stress",
            "distribution": "Normal",
            "params": {"mu": 0.5, "sigma": 0.2},
            "sources": [],
            "reasoning": "Weakly informative persistence prior for the fixture.",
        }
    }


def stage4_report() -> dict[str, Any]:
    priors = authored_priors()
    return {
        "statistical_model_spec": statistical_model_spec(),
        "authored_priors": priors,
        "resolved_priors": list(priors.values()),
        "prior_predictive_samples": {"stress_score": [0.1, 0.2], "sleep_score": [0.0, 0.1]},
    }


def compiled_ssm() -> dict[str, Any]:
    from nof1_causal_lab.models.ssm.compile.artifact import serialize_ssm_spec
    from nof1_causal_lab.models.ssm.compile.contracts import (
        CompiledPriorSemantics,
        CompiledSSMArtifact,
    )
    from tests.ssm_spec_fixtures import block_ssm_spec, full_dense_matrix_dynamics_spec

    spec = block_ssm_spec(
        n_latent=2,
        n_manifest=2,
        dynamics_spec=full_dense_matrix_dynamics_spec(2),
        latent_names=["Stress", "Sleep"],
        manifest_names=["stress_score", "sleep_score"],
    )
    artifact = CompiledSSMArtifact(
        schema_version=1,
        spec=serialize_ssm_spec(spec),
        edge_lag_days=[],
        compiled_prior_semantics=CompiledPriorSemantics(
            schema_version=5,
            site_registry=[],
            prior_state={},
        ),
        parameter_bindings=[],
        compile_diagnostics=[],
    )
    return artifact.model_dump(mode="json")


def state_from(*infos: ArtifactVersionInfo) -> EpisodeState:
    return EpisodeState().with_versions(list(infos))


def seed_panel(
    store: ArtifactStore,
    *,
    question_version: int = 1,
    raw_data_version: int = 1,
    measurement_structure_version: int = 1,
) -> ArtifactVersionInfo:
    return store.write_version(
        "panel",
        provenance="computed",
        derived_from={
            "question": question_version,
            "raw_data": raw_data_version,
            "measurement_structure": measurement_structure_version,
        },
        produced_by="run:measurements",
        parquet_files={parquet_filename("panel", "panel"): panel_frame()},
    )


def seed_compiled_ssm(
    store: ArtifactStore,
    *,
    causal_design_version: int = 1,
    statistical_model_spec_version: int = 1,
) -> ArtifactVersionInfo:
    return store.write_version(
        "compiled_ssm",
        provenance="computed",
        derived_from={
            "statistical_model_spec": statistical_model_spec_version,
            "causal_design": causal_design_version,
        },
        produced_by="derive:compiled_ssm",
        json_files={
            json_filename("compiled_ssm", "compiled_ssm"): compiled_ssm(),
            json_filename("compiled_ssm", "report"): stage4_report(),
        },
    )
