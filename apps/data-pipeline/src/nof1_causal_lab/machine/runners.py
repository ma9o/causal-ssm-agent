"""Transition execution against the versioned artifact store.

Each runner receives explicit input pins selected by the machine before
execution. It must read exactly those versions, write new artifact versions with
the same pins in ``derived_from``, and return effects for the workflow to apply.
Heavy transitions can be routed to Modal, but routing is infra-only: it cannot
change the pinned versions or the derivation cascade applied to the result.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any

from nof1_causal_lab.machine.artifact_files import json_filename, parquet_filename, pickle_filename
from nof1_causal_lab.machine.artifacts import (  # noqa: TC001 (pydantic field annotations)
    ArtifactId,
    ArtifactVersionInfo,
    EpisodeState,
)
from nof1_causal_lab.machine.derivations import complete_derivation_cascade
from nof1_causal_lab.machine.graph import transition_spec
from nof1_causal_lab.machine.moves import (
    ExecOptions,
    TransitionEffects,
    input_pins,
    run_retractions,
)
from nof1_causal_lab.machine.store import ArtifactStore

if TYPE_CHECKING:
    import polars as pl
    from pydantic import BaseModel


def _filter_to_contract(cls: type[BaseModel], data: dict[str, Any]) -> dict[str, Any]:
    fields = set(cls.model_fields.keys())
    return {key: value for key, value in data.items() if key in fields}


def _question_text(store: ArtifactStore, pins: dict[ArtifactId, int]) -> str:
    return store.read_json_file(
        "question", pins["question"], json_filename("question", "question")
    )["text"]


def _causal_design_dict(store: ArtifactStore, pins: dict[ArtifactId, int]) -> dict[str, Any]:
    payload = store.read_json_file(
        "causal_design", pins["causal_design"], json_filename("causal_design", "causal_design")
    )
    return payload["causal_design"]


def _measurement_structure_dict(
    store: ArtifactStore,
    pins: dict[ArtifactId, int],
) -> dict[str, Any]:
    payload = store.read_json_file(
        "measurement_structure",
        pins["measurement_structure"],
        json_filename("measurement_structure", "measurement_structure"),
    )
    return payload["measurement_structure"]


def _panel_df(store: ArtifactStore, pins: dict[ArtifactId, int]) -> pl.DataFrame:
    return store.read_parquet_file("panel", pins["panel"], parquet_filename("panel", "panel"))


async def _run_latent_structure(
    workspace_id: str,
    store: ArtifactStore,
    pins: dict[ArtifactId, int],
    options: ExecOptions,
) -> list[ArtifactVersionInfo]:
    from nof1_causal_lab.flows.transitions.latent_structure.flow import propose_latent_structure

    del workspace_id, options
    payload = await propose_latent_structure(_question_text(store, pins))
    info = store.write_version(
        "latent_structure",
        provenance="computed",
        derived_from=pins,
        produced_by="run:latent_structure",
        json_files={json_filename("latent_structure", "latent_structure"): payload},
    )
    return [info]


async def _run_measurement_structure(
    workspace_id: str,
    store: ArtifactStore,
    pins: dict[ArtifactId, int],
    options: ExecOptions,
) -> list[ArtifactVersionInfo]:
    from nof1_causal_lab.artifacts.measurement_structure import MeasurementStructure
    from nof1_causal_lab.flows.pipeline_helpers import format_schema_for_llm
    from nof1_causal_lab.flows.transitions.measurement_structure.flow import (
        propose_measurement_structure,
    )

    del workspace_id, options
    question = _question_text(store, pins)
    profile = store.read_json_file(
        "raw_data", pins["raw_data"], json_filename("raw_data", "profile")
    )
    raw_df = store.read_parquet_file(
        "raw_data", pins["raw_data"], parquet_filename("raw_data", "raw")
    )
    latent_structure_payload = store.read_json_file(
        "latent_structure",
        pins["latent_structure"],
        json_filename("latent_structure", "latent_structure"),
    )
    latent_structure = latent_structure_payload["latent_structure"]

    column_descriptions = {
        column["name"]: column["description"] for column in profile.get("column_descriptions", [])
    }
    dataset_schema = format_schema_for_llm(raw_df, column_descriptions)
    result = await propose_measurement_structure(
        question,
        latent_structure,
        [dataset_schema],
        dataset_summary=f"{raw_df.shape[0]} rows x {raw_df.shape[1]} columns",
    )
    measurement_structure = MeasurementStructure.model_validate(
        result["measurement_structure"]
    ).model_dump(mode="json")
    info = store.write_version(
        "measurement_structure",
        provenance="computed",
        derived_from=pins,
        produced_by="run:measurement_structure",
        json_files={
            json_filename("measurement_structure", "measurement_structure"): {
                "measurement_structure": measurement_structure
            }
        },
    )
    return [info]


async def _run_measurements(
    workspace_id: str,
    store: ArtifactStore,
    pins: dict[ArtifactId, int],
    options: ExecOptions,
) -> list[ArtifactVersionInfo]:
    from nof1_causal_lab.flows.artifact_contracts import MeasurementsContract
    from nof1_causal_lab.flows.transitions.extraction.flow import run_extraction
    from nof1_causal_lab.flows.transitions.extraction.materialization import (
        materialize_extraction_outputs,
    )

    question = _question_text(store, pins)
    raw_df = store.read_parquet_file(
        "raw_data", pins["raw_data"], parquet_filename("raw_data", "raw")
    )
    measurement_structure = _measurement_structure_dict(store, pins)

    result = await run_extraction(
        raw_df,
        question,
        measurement_structure,
        workspace_id=workspace_id,
        max_windows=options.max_windows,
    )
    materialized = materialize_extraction_outputs(result, measurement_structure)
    panel = materialized["data_for_model"]
    worker_statuses = materialized["worker_statuses"]

    report: dict[str, Any] = {"workers": worker_statuses}
    llm_trace = result.get("llm_trace")
    if llm_trace is not None:
        report["llm_trace"] = llm_trace
    report = _filter_to_contract(MeasurementsContract, report)

    produced = [
        store.write_version(
            "measurements",
            provenance="computed",
            derived_from=pins,
            produced_by="run:measurements",
            json_files={json_filename("measurements", "measurements"): report},
        )
    ]
    if len(panel) > 0:
        produced.append(
            store.write_version(
                "panel",
                provenance="computed",
                derived_from=pins,
                produced_by="run:measurements",
                parquet_files={parquet_filename("panel", "panel"): panel},
            )
        )
    return produced


async def _run_posterior(
    workspace_id: str,
    store: ArtifactStore,
    pins: dict[ArtifactId, int],
    options: ExecOptions,
) -> list[ArtifactVersionInfo]:
    from nof1_causal_lab.flows.artifact_contracts import PosteriorContract
    from nof1_causal_lab.flows.transitions.inference.flow import (
        build_sampler_config,
        run_inference_with_data,
    )
    from nof1_causal_lab.utils.config import get_config

    compiled_ssm = store.read_json_file(
        "compiled_ssm", pins["compiled_ssm"], json_filename("compiled_ssm", "compiled_ssm")
    )
    panel = _panel_df(store, pins)

    result = await asyncio.to_thread(
        run_inference_with_data,
        compiled_ssm=compiled_ssm,
        data_for_model=panel,
        sampler_config=build_sampler_config(options.inference_method),
        workspace_id=workspace_id,
        compute_loo_diagnostics=get_config().inference.compute_loo_diagnostics,
    )

    fitted_artifact = result.pop("_fitted_artifact", None)
    payload = _filter_to_contract(PosteriorContract, result)
    info = store.write_version(
        "posterior",
        provenance="computed",
        derived_from=pins,
        produced_by="run:posterior",
        json_files={json_filename("posterior", "diagnostics"): payload},
        pickle_files={pickle_filename("posterior", "fitted"): fitted_artifact},
    )
    return [info]


async def _run_baseline_report(
    workspace_id: str,
    store: ArtifactStore,
    pins: dict[ArtifactId, int],
    options: ExecOptions,
) -> list[ArtifactVersionInfo]:
    from nof1_causal_lab.flows.artifact_contracts import BaselineReportContract
    from nof1_causal_lab.flows.transitions.analysis.flow import run_analysis

    del workspace_id, options
    diagnostics = store.read_json_file(
        "posterior", pins["posterior"], json_filename("posterior", "diagnostics")
    )
    causal_design = _causal_design_dict(store, pins)
    identification_report = store.read_json_file(
        "identification_report",
        pins["identification_report"],
        json_filename("identification_report", "identification_report"),
    )

    posterior_dict = {
        **diagnostics,
        "_fitted_result_path": store.file_path(
            "posterior", pins["posterior"], pickle_filename("posterior", "fitted")
        ),
    }
    measurement_dict = {
        "causal_design": causal_design,
        "_identified_treatments": identification_report["estimable_treatments"],
    }
    result = await run_analysis(posterior_dict, measurement_dict)
    payload = _filter_to_contract(BaselineReportContract, result)

    info = store.write_version(
        "baseline_report",
        provenance="computed",
        derived_from=pins,
        produced_by="run:baseline_report",
        json_files={json_filename("baseline_report", "baseline_report"): payload},
    )
    return [info]


_TRANSITION_RUNNERS = {
    "latent_structure": _run_latent_structure,
    "measurement_structure": _run_measurement_structure,
    "measurements": _run_measurements,
    "posterior": _run_posterior,
    "baseline_report": _run_baseline_report,
}

_TEMPORAL_ONLY_TRANSITIONS = frozenset({"raw_data", "statistical_model_spec"})

_MODAL_TRANSITIONS = frozenset({"posterior"})


async def execute_transition_locally(
    workspace_id: str,
    artifact_id: ArtifactId,
    pins: dict[ArtifactId, int],
    state: EpisodeState,
    options: ExecOptions,
) -> TransitionEffects:
    """Run a transition on this process against pinned input versions."""
    from nof1_causal_lab.flows.runtime_events import emit_transition_event

    store = ArtifactStore(workspace_id)
    if artifact_id in _TEMPORAL_ONLY_TRANSITIONS:
        raise RuntimeError(f"{artifact_id} is implemented only as a Temporal child workflow")
    runner = _TRANSITION_RUNNERS[artifact_id]
    spec = transition_spec(artifact_id)
    emit_transition_event(workspace_id, artifact_id, "running")
    try:
        produced = await runner(workspace_id, store, pins, options)
        retracted = run_retractions(state, spec, produced)
        effects = complete_derivation_cascade(store, state, produced, retracted)
    except Exception as exc:
        emit_transition_event(
            workspace_id,
            artifact_id,
            "failed",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        raise
    emit_transition_event(workspace_id, artifact_id, "completed")
    return effects


async def execute_transition(
    workspace_id: str,
    artifact_id: ArtifactId,
    state: EpisodeState,
    options: ExecOptions,
) -> TransitionEffects:
    """Run a transition, routing heavy transitions to Modal in production."""
    spec = transition_spec(artifact_id)
    pins = input_pins(state, spec)
    if artifact_id in _TEMPORAL_ONLY_TRANSITIONS:
        raise RuntimeError(f"{artifact_id} is implemented only as a Temporal child workflow")
    if os.environ.get("DEPLOYMENT_ENV") == "production" and artifact_id in _MODAL_TRANSITIONS:
        from nof1_causal_lab.flows.modal_runners import run_transition_on_modal

        return await run_transition_on_modal(workspace_id, artifact_id, pins, state, options)
    return await execute_transition_locally(workspace_id, artifact_id, pins, state, options)
