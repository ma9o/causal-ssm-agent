"""Stage execution against the versioned artifact store.

``execute_stage`` is the single entry point the Temporal activity calls:
it loads the *pinned* input versions from the store (never "latest on
disk" — the derived_from stamp must be honest), runs the stage logic,
writes the produced artifact versions, and returns the version stamps for
the workflow to install.

Execution failure raises the typed exceptions in :mod:`machine.errors`;
negative findings withhold optional artifacts instead of raising.

In production, stage-4 and stage-5b route to Modal (see
:mod:`nof1_causal_lab.flows.modal_runners`); both paths run this module's
``execute_stage_locally`` — Modal just runs it on remote compute against
the same R2-backed store.
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
from nof1_causal_lab.machine.errors import ModelCompileError
from nof1_causal_lab.machine.graph import stage_spec
from nof1_causal_lab.machine.moves import ExecOptions, TransitionEffects, input_pins
from nof1_causal_lab.machine.store import ArtifactStore

if TYPE_CHECKING:
    import polars as pl
    from pydantic import BaseModel


def _filter_to_contract(cls: type[BaseModel], data: dict[str, Any]) -> dict[str, Any]:
    """Filter a dict to only the fields known by a contract class."""
    fields = set(cls.model_fields.keys())
    return {k: v for k, v in data.items() if k in fields}


def _question_text(store: ArtifactStore, pins: dict[ArtifactId, int]) -> str:
    return store.read_json_file(
        "question", pins["question"], json_filename("question", "question")
    )["text"]


def _causal_design_dict(store: ArtifactStore, pins: dict[ArtifactId, int]) -> dict[str, Any]:
    payload = store.read_json_file(
        "causal_design", pins["causal_design"], json_filename("causal_design", "causal_design")
    )
    return payload["causal_design"]


def _model_data_df(store: ArtifactStore, pins: dict[ArtifactId, int]) -> pl.DataFrame:
    return store.read_parquet_file(
        "model_data", pins["model_data"], parquet_filename("model_data", "model_data")
    )


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
    from nof1_causal_lab.flows.pipeline_helpers import format_schema_for_llm
    from nof1_causal_lab.flows.stage_contracts import Stage1bContract
    from nof1_causal_lab.flows.stages.stage1b.flow import (
        propose_measurement_with_identifiability_fix,
    )
    from nof1_causal_lab.flows.stages.stage1b.result import split_stage1b_result

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
    result = await propose_measurement_with_identifiability_fix(
        question,
        latent_structure,
        [dataset_schema],
        dataset_summary=f"{raw_df.shape[0]} rows x {raw_df.shape[1]} columns",
    )
    artifacts = split_stage1b_result(result, latent_structure=latent_structure)

    spec_payload = _filter_to_contract(Stage1bContract, artifacts.causal_design_payload)
    produced = [
        store.write_version(
            "causal_design",
            provenance="computed",
            derived_from=pins,
            produced_by="stage-1b",
            json_files={json_filename("causal_design", "causal_design"): spec_payload},
        )
    ]
    if artifacts.identification_report is not None:
        produced.append(
            store.write_version(
                "identification_report",
                provenance="computed",
                derived_from=pins,
                produced_by="stage-1b",
                json_files={
                    json_filename("identification_report", "identification_report"): (
                        artifacts.identification_report
                    )
                },
            )
        )
    return produced


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
    causal_design = _causal_design_dict(store, pins)

    result = await run_stage2_extraction(
        raw_df,
        question,
        causal_design,
        workspace_id=workspace_id,
        max_windows=options.max_windows,
    )
    materialized = materialize_stage2_outputs(result, causal_design)
    data_for_model = materialized["data_for_model"]
    worker_statuses = materialized["worker_statuses"]

    report: dict[str, Any] = {"workers": worker_statuses}
    llm_trace = result.get("llm_trace")
    if llm_trace is not None:
        report["llm_trace"] = llm_trace
    report = _filter_to_contract(Stage2Contract, report)

    produced = [
        store.write_version(
            "extraction_report",
            provenance="computed",
            derived_from=pins,
            produced_by="stage-2",
            json_files={json_filename("extraction_report", "extraction_report"): report},
        )
    ]
    if len(data_for_model) > 0:
        produced.append(
            store.write_version(
                "model_data",
                provenance="computed",
                derived_from=pins,
                produced_by="stage-2",
                parquet_files={parquet_filename("model_data", "model_data"): data_for_model},
            )
        )
    return produced


async def _run_stage3(
    workspace_id: str,
    store: ArtifactStore,
    pins: dict[ArtifactId, int],
    options: ExecOptions,
) -> list[ArtifactVersionInfo]:
    from nof1_causal_lab.flows.stage_contracts import Stage3Contract
    from nof1_causal_lab.flows.stages.stage3.flow import (
        derive_validation_status,
        validate_extraction,
    )

    del workspace_id, options
    causal_design = _causal_design_dict(store, pins)
    data_for_model = _model_data_df(store, pins)

    audit_result = validate_extraction(causal_design, [data_for_model])
    if not audit_result:
        raise RuntimeError(
            "Stage 3 validate_extraction returned an empty audit result; "
            "refusing to fabricate an is_valid=False report with empty indicators."
        )

    indicator_issues = [
        issue
        for audit in audit_result.get("indicators", {}).values()
        for issue in audit.get("validation", {}).get("issues", [])
    ]
    dataset_issues = audit_result.get("dataset_issues", [])
    all_issues = [*indicator_issues, *dataset_issues]
    status = derive_validation_status(all_issues)
    report = _filter_to_contract(Stage3Contract, {**audit_result, "is_valid": status["is_valid"]})

    info = store.write_version(
        "validation_report",
        provenance="computed",
        derived_from=pins,
        produced_by="stage-3",
        json_files={json_filename("validation_report", "validation_report"): report},
    )
    return [info]


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
    data_for_model = _model_data_df(store, pins)
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
        data_for_model=data_for_model,
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
        "compiled_ssm",
        provenance="computed",
        derived_from=pins,
        produced_by="stage-4",
        json_files={
            json_filename("compiled_ssm", "compiled_ssm"): compiled_ssm,
            json_filename("compiled_ssm", "report"): report,
        },
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
    data_for_model = _model_data_df(store, pins)

    result = await asyncio.to_thread(
        run_stage5b_with_data,
        compiled_ssm=compiled_ssm,
        data_for_model=data_for_model,
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
        "baseline_ranking",
        provenance="computed",
        derived_from=pins,
        produced_by="stage-6",
        json_files={json_filename("baseline_ranking", "baseline_ranking"): payload},
    )
    return [info]


_STAGE_RUNNERS = {
    "stage-0": _run_stage0,
    "stage-1a": _run_stage1a,
    "stage-1b": _run_stage1b,
    "stage-2": _run_stage2,
    "stage-3": _run_stage3,
    "stage-4": _run_stage4,
    "stage-5b": _run_stage5b,
    "stage-6": _run_stage6,
}

_MODAL_STAGES = frozenset({"stage-4", "stage-5b"})


async def execute_stage_locally(
    workspace_id: str,
    stage_id: str,
    pins: dict[ArtifactId, int],
    options: ExecOptions,
) -> TransitionEffects:
    """Run a stage on this process against the pinned input versions."""
    from nof1_causal_lab.flows.runtime_events import emit_stage_progress_event

    store = ArtifactStore(workspace_id)
    runner = _STAGE_RUNNERS[stage_id]
    emit_stage_progress_event(workspace_id, stage_id, "running")
    try:
        produced = await runner(workspace_id, store, pins, options)
    except Exception as exc:
        emit_stage_progress_event(
            workspace_id,
            stage_id,
            "failed",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        raise
    emit_stage_progress_event(workspace_id, stage_id, "completed")
    return TransitionEffects(produced=produced)


async def execute_stage(
    workspace_id: str,
    stage_id: str,
    state: EpisodeState,
    options: ExecOptions,
) -> TransitionEffects:
    """Run a stage, routing heavy stages to Modal in production."""
    pins = input_pins(state, stage_spec(stage_id))
    if os.environ.get("DEPLOYMENT_ENV") == "production" and stage_id in _MODAL_STAGES:
        from nof1_causal_lab.flows.modal_runners import run_stage_on_modal

        return await run_stage_on_modal(workspace_id, stage_id, pins, options)
    return await execute_stage_locally(workspace_id, stage_id, pins, options)
