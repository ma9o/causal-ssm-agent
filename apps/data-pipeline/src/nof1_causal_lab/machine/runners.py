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
from nof1_causal_lab.machine.errors import ModelCompileError
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


async def _run_stage0(
    workspace_id: str,
    store: ArtifactStore,
    pins: dict[ArtifactId, int],
    options: ExecOptions,
) -> list[ArtifactVersionInfo]:
    from nof1_causal_lab.flows.pipeline_helpers import build_stage0_payload
    from nof1_causal_lab.flows.stages.stage0.flow import agentic_ingest

    del options
    result = await agentic_ingest(workspace_id)
    payload = build_stage0_payload(result)
    info = store.write_version(
        "raw_data",
        provenance="computed",
        derived_from=pins,
        produced_by="stage-0",
        json_files={json_filename("raw_data", "profile"): payload},
        parquet_files={parquet_filename("raw_data", "raw"): result.dataframe},
    )
    return [info]


async def _run_stage1a(
    workspace_id: str,
    store: ArtifactStore,
    pins: dict[ArtifactId, int],
    options: ExecOptions,
) -> list[ArtifactVersionInfo]:
    from nof1_causal_lab.flows.stages.stage1a.flow import propose_latent_structure

    del workspace_id, options
    payload = await propose_latent_structure(_question_text(store, pins))
    info = store.write_version(
        "latent_structure",
        provenance="computed",
        derived_from=pins,
        produced_by="stage-1a",
        json_files={json_filename("latent_structure", "latent_structure"): payload},
    )
    return [info]


async def _run_stage1b(
    workspace_id: str,
    store: ArtifactStore,
    pins: dict[ArtifactId, int],
    options: ExecOptions,
) -> list[ArtifactVersionInfo]:
    from nof1_causal_lab.artifacts.measurement_structure import MeasurementStructure
    from nof1_causal_lab.flows.pipeline_helpers import format_schema_for_llm
    from nof1_causal_lab.flows.stages.stage1b.flow import propose_measurement_structure

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
        produced_by="stage-1b",
        json_files={
            json_filename("measurement_structure", "measurement_structure"): {
                "measurement_structure": measurement_structure
            }
        },
    )
    return [info]


async def _run_stage2(
    workspace_id: str,
    store: ArtifactStore,
    pins: dict[ArtifactId, int],
    options: ExecOptions,
) -> list[ArtifactVersionInfo]:
    from nof1_causal_lab.flows.stage_contracts import Stage2Contract
    from nof1_causal_lab.flows.stages.stage2.flow import run_stage2_extraction
    from nof1_causal_lab.flows.stages.stage2.materialization import materialize_stage2_outputs

    question = _question_text(store, pins)
    raw_df = store.read_parquet_file(
        "raw_data", pins["raw_data"], parquet_filename("raw_data", "raw")
    )
    measurement_structure = _measurement_structure_dict(store, pins)

    result = await run_stage2_extraction(
        raw_df,
        question,
        measurement_structure,
        workspace_id=workspace_id,
        max_windows=options.max_windows,
    )
    materialized = materialize_stage2_outputs(result, measurement_structure)
    panel = materialized["data_for_model"]
    worker_statuses = materialized["worker_statuses"]

    report: dict[str, Any] = {"workers": worker_statuses}
    llm_trace = result.get("llm_trace")
    if llm_trace is not None:
        report["llm_trace"] = llm_trace
    report = _filter_to_contract(Stage2Contract, report)

    produced = [
        store.write_version(
            "measurements",
            provenance="computed",
            derived_from=pins,
            produced_by="stage-2",
            json_files={json_filename("measurements", "measurements"): report},
        )
    ]
    if len(panel) > 0:
        produced.append(
            store.write_version(
                "panel",
                provenance="computed",
                derived_from=pins,
                produced_by="stage-2",
                parquet_files={parquet_filename("panel", "panel"): panel},
            )
        )
    return produced


async def _run_stage4(
    workspace_id: str,
    store: ArtifactStore,
    pins: dict[ArtifactId, int],
    options: ExecOptions,
) -> list[ArtifactVersionInfo]:
    from nof1_causal_lab.flows.stage_contracts import Stage4Contract
    from nof1_causal_lab.flows.stages.stage4.flow import stage4_agentic_flow
    from nof1_causal_lab.utils.config import get_config

    question = _question_text(store, pins)
    causal_design = _causal_design_dict(store, pins)
    panel = _panel_df(store, pins)
    validation_report = store.read_json_file(
        "validation_report",
        pins["validation_report"],
        json_filename("validation_report", "validation_report"),
    )

    lit_enabled = (
        options.enable_literature
        if options.enable_literature is not None
        else get_config().stage4_prior_elicitation.literature_search.enabled
    )
    result = await stage4_agentic_flow(
        causal_design=causal_design,
        question=question,
        data_for_model=panel,
        indicator_audits=validation_report.get("indicators", {}),
        enable_literature=lit_enabled,
        workspace_id=workspace_id,
    )

    compiled_ssm = result.pop("_compiled_ssm", None)
    report = _filter_to_contract(Stage4Contract, result)
    if compiled_ssm is None:
        raise ModelCompileError(
            "stage-4 produced no compilable SSM from the proposed spec",
            stage_id="stage-4",
            diagnostics={"report": report},
        )

    info = store.write_version(
        "statistical_model_spec",
        provenance="computed",
        derived_from=pins,
        produced_by="stage-4",
        json_files={json_filename("statistical_model_spec", "statistical_model_spec"): report},
    )
    return [info]


async def _run_stage5b(
    workspace_id: str,
    store: ArtifactStore,
    pins: dict[ArtifactId, int],
    options: ExecOptions,
) -> list[ArtifactVersionInfo]:
    from nof1_causal_lab.flows.stage_contracts import Stage5bContract
    from nof1_causal_lab.flows.stages.stage5b.flow import (
        build_sampler_config,
        run_stage5b_with_data,
    )
    from nof1_causal_lab.utils.config import get_config

    compiled_ssm = store.read_json_file(
        "compiled_ssm", pins["compiled_ssm"], json_filename("compiled_ssm", "compiled_ssm")
    )
    panel = _panel_df(store, pins)

    result = await asyncio.to_thread(
        run_stage5b_with_data,
        compiled_ssm=compiled_ssm,
        data_for_model=panel,
        sampler_config=build_sampler_config(options.inference_method),
        workspace_id=workspace_id,
        compute_loo_diagnostics=get_config().inference.compute_loo_diagnostics,
    )

    fitted_artifact = result.pop("_fitted_artifact", None)
    payload = _filter_to_contract(Stage5bContract, result)
    info = store.write_version(
        "posterior",
        provenance="computed",
        derived_from=pins,
        produced_by="stage-5b",
        json_files={json_filename("posterior", "diagnostics"): payload},
        pickle_files={pickle_filename("posterior", "fitted"): fitted_artifact},
    )
    return [info]


async def _run_stage6(
    workspace_id: str,
    store: ArtifactStore,
    pins: dict[ArtifactId, int],
    options: ExecOptions,
) -> list[ArtifactVersionInfo]:
    from nof1_causal_lab.flows.stage_contracts import Stage6Contract
    from nof1_causal_lab.flows.stages.stage6.flow import run_stage6

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

    stage5b_dict = {
        **diagnostics,
        "_fitted_result_path": store.file_path(
            "posterior", pins["posterior"], pickle_filename("posterior", "fitted")
        ),
    }
    stage1b_dict = {
        "causal_design": causal_design,
        "_identified_treatments": identification_report["estimable_treatments"],
    }
    result = await run_stage6(stage5b_dict, stage1b_dict)
    payload = _filter_to_contract(Stage6Contract, result)

    info = store.write_version(
        "baseline_report",
        provenance="computed",
        derived_from=pins,
        produced_by="stage-6",
        json_files={json_filename("baseline_report", "baseline_report"): payload},
    )
    return [info]


_TRANSITION_RUNNERS = {
    "raw_data": _run_stage0,
    "latent_structure": _run_stage1a,
    "measurement_structure": _run_stage1b,
    "measurements": _run_stage2,
    "statistical_model_spec": _run_stage4,
    "posterior": _run_stage5b,
    "baseline_report": _run_stage6,
}

_MODAL_TRANSITIONS = frozenset({"statistical_model_spec", "posterior"})


async def execute_transition_locally(
    workspace_id: str,
    artifact_id: ArtifactId,
    pins: dict[ArtifactId, int],
    state: EpisodeState,
    options: ExecOptions,
) -> TransitionEffects:
    """Run a transition on this process against pinned input versions."""
    from nof1_causal_lab.flows.runtime_events import emit_stage_progress_event

    store = ArtifactStore(workspace_id)
    spec = transition_spec(artifact_id)
    runner = _TRANSITION_RUNNERS[artifact_id]
    emit_stage_progress_event(workspace_id, spec.runner_id, "running")
    try:
        produced = await runner(workspace_id, store, pins, options)
        retracted = run_retractions(state, spec, produced)
        effects = complete_derivation_cascade(store, state, produced, retracted)
    except Exception as exc:
        emit_stage_progress_event(
            workspace_id,
            spec.runner_id,
            "failed",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        raise
    emit_stage_progress_event(workspace_id, spec.runner_id, "completed")
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
    if os.environ.get("DEPLOYMENT_ENV") == "production" and artifact_id in _MODAL_TRANSITIONS:
        from nof1_causal_lab.flows.modal_runners import run_transition_on_modal

        return await run_transition_on_modal(workspace_id, artifact_id, pins, state, options)
    return await execute_transition_locally(workspace_id, artifact_id, pins, state, options)
