"""Temporal activities for the raw-data ingestion transition."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from temporalio import activity

from nof1_causal_lab.machine.artifact_files import json_filename, parquet_filename
from nof1_causal_lab.machine.derivations import complete_computed_transition
from nof1_causal_lab.machine.graph import transition_spec
from nof1_causal_lab.machine.moves import TransitionEffects, input_pins
from nof1_causal_lab.machine.store import ArtifactStore
from nof1_causal_lab.machine.temporal.activity_errors import (
    as_non_retryable_application_error,
)
from nof1_causal_lab.machine.temporal.latent_structure_activities import _llm_backend_config
from nof1_causal_lab.machine.temporal.messages import (
    SingleLLMTransitionFinalizeInput,
    SingleLLMTransitionPlan,
    SingleLLMTransitionWorkflowInput,
)
from nof1_causal_lab.utils import data as data_module
from nof1_causal_lab.utils import storage


def _raw_data_root(workspace_id: str, run_id: str) -> str:
    return storage.join(data_module.scratch_run_dir(workspace_id, run_id), "ingestion")


def _write_raw_data_json(path: str, value: Any) -> None:
    storage.write_text(path, json.dumps(value))


def _read_raw_data_json(path: str) -> Any:
    return storage.read_json(path)


def _read_raw_data_artifact_frame(path: str):
    import polars as pl

    with storage.open_file(path, "rb") as file:
        return pl.read_ipc(file)


@activity.defn
async def plan_raw_data_activity(
    input: SingleLLMTransitionWorkflowInput,
) -> SingleLLMTransitionPlan:
    from nof1_causal_lab.flows.transitions.ingestion.flow import (
        _find_raw_input,
        _prepare_raw_input,
    )
    from nof1_causal_lab.utils.config import get_config

    spec = transition_spec("raw_data")
    pins = input_pins(input.state, spec)
    run_id = f"seq-{input.seq:06d}"
    root = _raw_data_root(input.workspace_id, run_id)
    upload_dir = storage.join(root, "upload")
    extract_dir = storage.join(root, "input")
    storage.rm_tree(upload_dir)
    storage.rm_tree(extract_dir)
    storage.makedirs(upload_dir)
    storage.makedirs(extract_dir)

    raw_storage_path = _find_raw_input(input.workspace_id)
    raw_name = raw_storage_path.rsplit("/", 1)[-1]
    if storage.is_remote():
        local_raw = Path(upload_dir) / raw_name
        storage.get_fs().get(raw_storage_path, str(local_raw))
    else:
        local_raw = Path(raw_storage_path)
    _prepare_raw_input(local_raw, Path(extract_dir))

    context_ref = storage.join(root, "context.json")
    _write_raw_data_json(
        context_ref,
        {
            "extract_dir": extract_dir,
            "dataframe_ref": storage.join(root, "latest-dataframe.ipc"),
        },
    )

    config = get_config()
    max_tool_turns = config.ingestion.max_tool_turns
    return SingleLLMTransitionPlan(
        workspace_id=input.workspace_id,
        run_id=run_id,
        context_ref=context_ref,
        pins=pins,
        llm=_llm_backend_config(config.ingestion.llm, config.llm, max_tool_turns),
        max_tool_turns=max_tool_turns,
    )


@activity.defn
async def finalize_raw_data_activity(input: SingleLLMTransitionFinalizeInput) -> TransitionEffects:
    from nof1_causal_lab.flows.pipeline_helpers import build_raw_data_payload
    from nof1_causal_lab.flows.transitions.ingestion.flow import IngestionResult

    try:
        if input.result_ref is None:
            raise RuntimeError("raw-data subroutine completed without a result ref")
        result = _read_raw_data_json(input.result_ref)
        dataframe = _read_raw_data_artifact_frame(result["dataframe_ref"])
        ingestion_result = IngestionResult(
            dataframe=dataframe,
            column_descriptions=dict(result["column_descriptions"]),
        )
        payload = build_raw_data_payload(ingestion_result)

        store = ArtifactStore(input.workspace_id)
        produced = [
            store.write_version(
                "raw_data",
                provenance="computed",
                derived_from=input.pins,
                produced_by="run:raw_data",
                json_files={json_filename("raw_data", "profile"): payload},
                parquet_files={parquet_filename("raw_data", "raw"): dataframe},
            )
        ]
        return complete_computed_transition(store, input.state, "raw_data", produced)
    except Exception as exc:
        raise as_non_retryable_application_error(exc) from exc


RAW_DATA_ACTIVITIES = [
    plan_raw_data_activity,
    finalize_raw_data_activity,
]
