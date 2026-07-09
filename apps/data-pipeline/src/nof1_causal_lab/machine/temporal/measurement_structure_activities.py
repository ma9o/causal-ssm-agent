"""Temporal activities for the measurement-structure transition."""

from __future__ import annotations

import json
from typing import Any

from temporalio import activity
from temporalio.exceptions import ApplicationError

from nof1_causal_lab.artifacts.measurement_structure import MeasurementStructure
from nof1_causal_lab.machine.artifact_files import json_filename, parquet_filename
from nof1_causal_lab.machine.derivations import complete_derivation_cascade
from nof1_causal_lab.machine.errors import TransitionExecutionError
from nof1_causal_lab.machine.graph import transition_spec
from nof1_causal_lab.machine.moves import TransitionEffects, input_pins, run_retractions
from nof1_causal_lab.machine.store import ArtifactStore
from nof1_causal_lab.machine.temporal.latent_structure_activities import (
    _llm_backend_config,
)
from nof1_causal_lab.machine.temporal.messages import (
    SingleLLMTransitionFinalizeInput,
    SingleLLMTransitionPlan,
    SingleLLMTransitionWorkflowInput,
)
from nof1_causal_lab.utils import data as data_module
from nof1_causal_lab.utils import storage


def _write_measurement_structure_json(path: str, value: Any) -> None:
    storage.write_text(path, json.dumps(value))


def _read_measurement_structure_json(path: str) -> Any:
    return storage.read_json(path)


def _measurement_structure_transition_failure(exc: Exception) -> ApplicationError:
    if isinstance(exc, TransitionExecutionError):
        return ApplicationError(
            str(exc),
            exc.diagnostics,
            type=type(exc).__name__,
            non_retryable=True,
        )
    return ApplicationError(str(exc), type=type(exc).__name__, non_retryable=True)


@activity.defn
async def plan_measurement_structure_activity(
    input: SingleLLMTransitionWorkflowInput,
) -> SingleLLMTransitionPlan:
    from nof1_causal_lab.flows.pipeline_helpers import format_schema_for_llm
    from nof1_causal_lab.flows.transitions.measurement_structure.prompting import (
        build_measurement_structure_user_prompt,
        templates,
    )
    from nof1_causal_lab.utils.config import get_config

    store = ArtifactStore(input.workspace_id)
    spec = transition_spec("measurement_structure")
    pins = input_pins(input.state, spec)
    run_id = f"seq-{input.seq:06d}"

    question = store.read_json_file(
        "question",
        pins["question"],
        json_filename("question", "question"),
    )["text"]
    profile = store.read_json_file(
        "raw_data",
        pins["raw_data"],
        json_filename("raw_data", "profile"),
    )
    raw_df = store.read_parquet_file(
        "raw_data",
        pins["raw_data"],
        parquet_filename("raw_data", "raw"),
    )
    latent_payload = store.read_json_file(
        "latent_structure",
        pins["latent_structure"],
        json_filename("latent_structure", "latent_structure"),
    )
    latent_structure = latent_payload["latent_structure"]
    column_descriptions = {
        column["name"]: column["description"] for column in profile.get("column_descriptions", [])
    }
    dataset_schema = format_schema_for_llm(raw_df, column_descriptions)
    dataset_summary = f"{raw_df.shape[0]} rows x {raw_df.shape[1]} columns"
    context_ref = storage.join(
        data_module.runs_dir(input.workspace_id),
        "temporal-llm",
        run_id,
        "measurement-structure",
        "context.json",
    )
    _write_measurement_structure_json(
        context_ref,
        {
            "system_prompt": templates.SYSTEM,
            "user_messages": [
                build_measurement_structure_user_prompt(
                    question,
                    latent_structure,
                    [dataset_schema],
                    dataset_summary,
                ),
                templates.REVIEW,
            ],
            "latent_structure": latent_structure,
        },
    )

    config = get_config()
    max_tool_turns = config.structure_proposal.measurement_max_tool_turns
    return SingleLLMTransitionPlan(
        workspace_id=input.workspace_id,
        run_id=run_id,
        context_ref=context_ref,
        pins=pins,
        llm=_llm_backend_config(config.structure_proposal.llm, config.llm, max_tool_turns),
        max_tool_turns=max_tool_turns,
    )


@activity.defn
async def finalize_measurement_structure_activity(
    input: SingleLLMTransitionFinalizeInput,
) -> TransitionEffects:
    from nof1_causal_lab.flows.transitions.measurement_structure.contracts import (
        MeasurementStructureContract,
    )

    try:
        if input.result_ref is None:
            raise RuntimeError("measurement-structure subroutine completed without a result ref")
        payload = _read_measurement_structure_json(input.result_ref)
        measurement_structure = MeasurementStructure.model_validate(
            payload["measurement_structure"]
        ).model_dump(mode="json")
        report = {
            "measurement_structure": measurement_structure,
            "llm_trace_ref": input.trace_ref,
        }
        fields = set(MeasurementStructureContract.model_fields.keys())
        report = {key: value for key, value in report.items() if key in fields}

        store = ArtifactStore(input.workspace_id)
        produced = [
            store.write_version(
                "measurement_structure",
                provenance="computed",
                derived_from=input.pins,
                produced_by="run:measurement_structure",
                json_files={
                    json_filename("measurement_structure", "measurement_structure"): report
                },
            )
        ]
        spec = transition_spec("measurement_structure")
        retracted = run_retractions(input.state, spec, produced)
        return complete_derivation_cascade(store, input.state, produced, retracted)
    except Exception as exc:
        raise _measurement_structure_transition_failure(exc) from exc


MEASUREMENT_STRUCTURE_ACTIVITIES = [
    plan_measurement_structure_activity,
    finalize_measurement_structure_activity,
]
