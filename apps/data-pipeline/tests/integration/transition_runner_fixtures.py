"""Small canonical artifacts for fixture-backed stage-runner tests."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import polars as pl

from nof1_causal_lab.flows.transitions.measurement_structure.assemble import build_causal_design
from nof1_causal_lab.machine.artifact_files import (
    json_filename,
    parquet_filename,
    pickle_filename,
)
from nof1_causal_lab.machine.artifacts import ArtifactVersionInfo, EpisodeState

if TYPE_CHECKING:
    from nof1_causal_lab.machine.store import ArtifactStore


def latent_structure() -> dict[str, Any]:
    return {
        "constructs": [
            {
                "name": "Stress",
                "description": "Daily stress level.",
                "role": "endogenous",
                "is_outcome": False,
                "temporal_status": "time_varying",
            },
            {
                "name": "Sleep",
                "description": "Sleep quality.",
                "role": "endogenous",
                "is_outcome": True,
                "temporal_status": "time_varying",
            },
        ],
        "edges": [
            {
                "cause": "Stress",
                "effect": "Sleep",
                "description": "Stress affects next-day sleep.",
                "lagged": True,
            }
        ],
    }


def measurement_structure(*, extraction_mode: str = "computed") -> dict[str, Any]:
    return {
        "model_clock": "1d",
        "indicators": [
            {
                "name": "stress_score",
                "construct_name": "Stress",
                "construct_polarity": "positive",
                "how_to_measure": "Average the daily stress score.",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
                "source_columns": ["stress_score"],
                "extraction_mode": extraction_mode,
            },
            {
                "name": "sleep_score",
                "construct_name": "Sleep",
                "construct_polarity": "positive",
                "how_to_measure": "Average the daily sleep score.",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
                "source_columns": ["sleep_score"],
                "extraction_mode": extraction_mode,
            },
        ],
    }


def identifiability_status() -> dict[str, Any]:
    return {
        "identifiable_treatments": {
            "Stress": {
                "method": "do_calculus",
                "estimand": "P(Sleep | do(Stress))",
                "marginalized_confounders": [],
                "instruments": [],
            }
        },
        "non_identifiable_treatments": {},
    }


def causal_design(*, extraction_mode: str = "computed") -> dict[str, Any]:
    return build_causal_design(
        latent_structure(),
        measurement_structure(extraction_mode=extraction_mode),
        identifiability_status(),
    )


def identification_report() -> dict[str, Any]:
    return {
        "outcome_name": "Sleep",
        "estimable_treatments": ["Stress"],
        "non_identifiable_treatments": {},
    }


def raw_dataframe(n_days: int = 20) -> pl.DataFrame:
    start = datetime(2024, 1, 1, 8)
    return pl.DataFrame(
        {
            "timestamp": [(start + timedelta(days=i)).isoformat() for i in range(n_days)],
            "stress_score": [float(1 + (i % 5)) for i in range(n_days)],
            "sleep_score": [float(8 - (i % 4)) for i in range(n_days)],
        }
    )


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
    return {"spec": {"fixture": "compiled"}, "compile_diagnostics": []}


def posterior_diagnostics() -> dict[str, Any]:
    return {
        "ppc": {
            "per_variable_warnings": [],
            "checked": True,
            "overlays": [],
            "test_stats": [],
        },
        "inference_metadata": {
            "method": "marginal_particle_gibbs",
            "n_samples": 4,
            "duration_seconds": 0.01,
        },
        "mcmc_diagnostics": None,
        "smc_diagnostics": None,
        "loo_diagnostics": None,
        "posterior_marginals": None,
        "posterior_pairs": None,
    }


def state_from(*infos: ArtifactVersionInfo) -> EpisodeState:
    return EpisodeState().with_versions(list(infos))


def seed_question(store: ArtifactStore) -> ArtifactVersionInfo:
    return store.write_version(
        "question",
        provenance="human",
        derived_from={},
        produced_by=None,
        json_files={json_filename("question", "question"): {"text": "Does stress affect sleep?"}},
    )


def seed_raw_data(store: ArtifactStore) -> ArtifactVersionInfo:
    return store.write_version(
        "raw_data",
        provenance="computed",
        derived_from={},
        produced_by="run:raw_data",
        json_files={
            json_filename("raw_data", "profile"): {
                "column_descriptions": [
                    {"name": "timestamp", "description": "Observation timestamp."},
                    {"name": "stress_score", "description": "Daily stress score."},
                    {"name": "sleep_score", "description": "Daily sleep score."},
                ]
            }
        },
        parquet_files={parquet_filename("raw_data", "raw"): raw_dataframe()},
    )


def seed_latent_structure(
    store: ArtifactStore, *, question_version: int = 1
) -> ArtifactVersionInfo:
    return store.write_version(
        "latent_structure",
        provenance="computed",
        derived_from={"question": question_version},
        produced_by="run:latent_structure",
        json_files={
            json_filename("latent_structure", "latent_structure"): {
                "latent_structure": latent_structure()
            }
        },
    )


def seed_measurement_structure(
    store: ArtifactStore,
    *,
    question_version: int = 1,
    raw_data_version: int = 1,
    latent_structure_version: int = 1,
) -> ArtifactVersionInfo:
    return store.write_version(
        "measurement_structure",
        provenance="computed",
        derived_from={
            "question": question_version,
            "raw_data": raw_data_version,
            "latent_structure": latent_structure_version,
        },
        produced_by="run:measurement_structure",
        json_files={
            json_filename("measurement_structure", "measurement_structure"): {
                "measurement_structure": measurement_structure()
            }
        },
    )


def seed_causal_design(
    store: ArtifactStore,
    *,
    latent_structure_version: int = 1,
    measurement_structure_version: int = 1,
) -> ArtifactVersionInfo:
    return store.write_version(
        "causal_design",
        provenance="computed",
        derived_from={
            "latent_structure": latent_structure_version,
            "measurement_structure": measurement_structure_version,
        },
        produced_by="derive:causal_design",
        json_files={
            json_filename("causal_design", "causal_design"): {"causal_design": causal_design()}
        },
    )


def seed_identification_report(
    store: ArtifactStore,
    *,
    causal_design_version: int = 1,
) -> ArtifactVersionInfo:
    return store.write_version(
        "identification_report",
        provenance="computed",
        derived_from={"causal_design": causal_design_version},
        produced_by="derive:identification_report",
        json_files={
            json_filename("identification_report", "identification_report"): (
                identification_report()
            )
        },
    )


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


def seed_posterior(
    store: ArtifactStore,
    *,
    compiled_ssm_version: int = 1,
    panel_version: int = 1,
    fitted_artifact: Any = None,
) -> ArtifactVersionInfo:
    return store.write_version(
        "posterior",
        provenance="computed",
        derived_from={
            "compiled_ssm": compiled_ssm_version,
            "panel": panel_version,
        },
        produced_by="run:posterior",
        json_files={json_filename("posterior", "diagnostics"): posterior_diagnostics()},
        pickle_files={pickle_filename("posterior", "fitted"): fitted_artifact},
    )
