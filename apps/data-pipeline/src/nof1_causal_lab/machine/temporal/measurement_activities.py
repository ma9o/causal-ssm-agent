"""Temporal activities for measurements extraction.

The workflow layer owns the durable control flow. These activities own all I/O:
artifact reads/writes, transient run files, OpenRouter calls, validation, and
runtime event emission.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, cast

from temporalio import activity
from temporalio.exceptions import ApplicationError

from nof1_causal_lab.machine.artifact_files import json_filename, parquet_filename
from nof1_causal_lab.machine.derivations import complete_derivation_cascade
from nof1_causal_lab.machine.errors import TransitionExecutionError
from nof1_causal_lab.machine.graph import transition_spec
from nof1_causal_lab.machine.moves import TransitionEffects, input_pins, run_retractions
from nof1_causal_lab.machine.store import ArtifactStore
from nof1_causal_lab.machine.temporal.messages import (
    ExtractionChunkFinalizeInput,
    ExtractionChunkResult,
    ExtractionProgressEventInput,
    LLMBackendConfig,
    MeasurementChunkRef,
    MeasurementsFinalizeInput,
    MeasurementsPlan,
    MeasurementsWorkflowInput,
    OpenRouterCallInput,
    OpenRouterCallResult,
    ToolCallSummary,
    TransitionRuntimeEventInput,
)
from nof1_causal_lab.utils import data as data_module
from nof1_causal_lab.utils import storage


def _run_root(workspace_id: str, run_id: str) -> str:
    return storage.join(data_module.scratch_run_dir(workspace_id, run_id), "extraction")


def _write_json(path: str, value: Any) -> None:
    storage.write_text(path, json.dumps(value))


def _read_json(path: str) -> Any:
    return storage.read_json(path)


def _first_config_value(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _filter_measurements_contract(cls: Any, data: dict[str, Any]) -> dict[str, Any]:
    fields = set(cls.model_fields.keys())
    return {key: value for key, value in data.items() if key in fields}


def _transition_failure(exc: Exception) -> ApplicationError:
    if isinstance(exc, TransitionExecutionError):
        return ApplicationError(
            str(exc),
            exc.diagnostics,
            type=type(exc).__name__,
            non_retryable=True,
        )
    return ApplicationError(str(exc), type=type(exc).__name__, non_retryable=True)


@activity.defn
async def emit_transition_runtime_event_activity(input: TransitionRuntimeEventInput) -> None:
    from nof1_causal_lab.flows.runtime_events import emit_transition_event

    emit_transition_event(
        input.workspace_id,
        input.transition_id,
        input.status,
        error=input.error.model_dump(mode="json") if input.error is not None else None,
    )


@activity.defn
async def emit_extraction_progress_event_activity(input: ExtractionProgressEventInput) -> None:
    from nof1_causal_lab.flows.runtime_events import (
        emit_extraction_plan_event,
        emit_extraction_snapshot_event,
        emit_extraction_worker_event,
    )

    if input.kind == "plan":
        if input.total_workers is None:
            raise ValueError("plan events require total_workers")
        emit_extraction_plan_event(
            input.workspace_id,
            total_workers=input.total_workers,
            max_concurrent_workers=input.max_concurrent_workers,
            max_rpm=input.max_rpm,
        )
        return

    if input.kind == "worker":
        if input.worker_id is None or input.state is None or input.n_windows is None:
            raise ValueError("worker events require worker_id, state, and n_windows")
        emit_extraction_worker_event(
            input.workspace_id,
            worker_id=input.worker_id,
            state=input.state,
            n_windows=input.n_windows,
            n_extractions=input.n_extractions,
            n_llm_calls=input.n_llm_calls,
            error=input.error,
        )
        return

    if input.kind == "snapshot":
        if input.snapshot is None:
            raise ValueError("snapshot events require snapshot")
        emit_extraction_snapshot_event(
            input.workspace_id,
            snapshot=input.snapshot.model_dump(mode="json"),
        )
        return

    raise ValueError(f"unknown extraction progress event kind {input.kind!r}")


@activity.defn
async def plan_measurements_activity(input: MeasurementsWorkflowInput) -> MeasurementsPlan:

    from nof1_causal_lab.flows.transitions.extraction.planning import prepare_semantic_chunks
    from nof1_causal_lab.utils.aggregations import compute_indicators
    from nof1_causal_lab.utils.config import get_config

    store = ArtifactStore(input.workspace_id)
    spec = transition_spec("measurements")
    pins = input_pins(input.state, spec)
    run_id = f"seq-{input.seq:06d}"
    root = _run_root(input.workspace_id, run_id)

    question = store.read_json_file(
        "question",
        pins["question"],
        json_filename("question", "question"),
    )["text"]
    raw_df = store.read_parquet_file(
        "raw_data",
        pins["raw_data"],
        parquet_filename("raw_data", "raw"),
    )
    measurement_payload = store.read_json_file(
        "measurement_structure",
        pins["measurement_structure"],
        json_filename("measurement_structure", "measurement_structure"),
    )
    measurement_structure = measurement_payload["measurement_structure"]

    config = get_config()
    extraction_workers = config.extraction_workers
    model_clock = measurement_structure.get("model_clock", "1d")
    time_col = "timestamp"
    all_indicators = list(measurement_structure.get("indicators", []))
    computed_inds = [i for i in all_indicators if i.get("extraction_mode") == "computed"]
    semantic_inds = [
        i for i in all_indicators if i.get("extraction_mode", "semantic") == "semantic"
    ]

    computed_dicts: list[dict[str, Any]] = []
    if computed_inds:
        computed_df = compute_indicators(raw_df, computed_inds, model_clock, time_col)
        computed_dicts = computed_df.to_dicts()

    chunks: list[MeasurementChunkRef] = []
    if semantic_inds:
        chunk_texts, chunk_window_starts, chunk_contexts = prepare_semantic_chunks(
            raw_df=raw_df,
            semantic_inds=semantic_inds,
            measurement_structure=measurement_structure,
            model_clock=model_clock,
            time_col=time_col,
            windows_per_chunk=extraction_workers.windows_per_chunk,
            max_events_per_window=extraction_workers.max_events_per_window,
            max_windows=input.options.max_windows,
        )
        for worker_id, (chunk_text, window_starts, chunk_context) in enumerate(
            zip(chunk_texts, chunk_window_starts, chunk_contexts, strict=True)
        ):
            spec_ref = storage.join(root, "chunks", f"worker-{worker_id:06d}.json")
            _write_json(
                spec_ref,
                {
                    "worker_id": worker_id,
                    "question": question,
                    "window_text": chunk_text,
                    "window_starts": window_starts,
                    "measurement_structure": chunk_context,
                },
            )
            chunks.append(
                MeasurementChunkRef(
                    worker_id=worker_id,
                    n_windows=len(window_starts),
                    spec_ref=spec_ref,
                )
            )

    extraction_llm = replace(
        extraction_workers.llm,
        timeout=extraction_workers.worker_timeout,
    )
    embedded_defaults = config.llm.embedded
    llm = LLMBackendConfig(
        harness="none",
        model=extraction_llm.model,
        max_tokens=_first_config_value(extraction_llm.max_tokens, embedded_defaults.max_tokens),
        timeout=_first_config_value(extraction_llm.timeout, embedded_defaults.timeout),
        reasoning_effort=_first_config_value(
            extraction_llm.reasoning_effort,
            embedded_defaults.reasoning_effort,
        ),
    )

    plan_ref = storage.join(root, "plan.json")
    _write_json(
        plan_ref,
        {
            "workspace_id": input.workspace_id,
            "run_id": run_id,
            "pins": pins,
            "question": question,
            "measurement_structure": measurement_structure,
            "computed_dicts": computed_dicts,
            "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
        },
    )

    return MeasurementsPlan(
        workspace_id=input.workspace_id,
        run_id=run_id,
        plan_ref=plan_ref,
        pins=pins,
        chunks=chunks,
        max_concurrent_workers=extraction_workers.max_concurrent_workers,
        max_rpm=extraction_workers.max_rpm,
        max_tool_turns=extraction_workers.max_tool_turns,
        llm=llm,
    )


@activity.defn
async def call_openrouter_activity(input: OpenRouterCallInput) -> OpenRouterCallResult:
    if storage.exists(input.call_ref):
        return OpenRouterCallResult.model_validate(_read_json(input.call_ref)["result"])

    from nof1_causal_lab.utils.openrouter_client import GenerateConfig, Tool, call_model

    async def _unused_tool(**kwargs: str) -> str:
        del kwargs
        return ""

    conversation = _read_json(input.conversation_ref)
    messages = list(conversation["messages"])
    tools = [
        Tool(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
            execute=_unused_tool,
            stop_on_success=tool.kind == "terminal",
            success_output=tool.success_output,
        )
        for tool in input.tools
    ] or None
    output = await call_model(
        input.llm.model,
        messages,
        tools=tools,
        config=GenerateConfig(
            max_tokens=input.llm.max_tokens,
            timeout=input.llm.timeout,
            reasoning_effort=input.llm.reasoning_effort,
        ),
        log_label=input.log_label,
    )

    _write_json(input.assistant_ref, output)

    next_messages = [*messages, output["message"]]
    _write_json(input.next_conversation_ref, {"messages": next_messages})

    tool_calls = [
        ToolCallSummary(
            index=index,
            id=str(tool_call.get("id", "")),
            name=str((tool_call.get("function") or {}).get("name") or tool_call.get("name", "")),
        )
        for index, tool_call in enumerate(output["message"].get("tool_calls") or [])
    ]
    result = OpenRouterCallResult(
        conversation_ref=input.next_conversation_ref,
        assistant_ref=input.assistant_ref,
        model=output["model"],
        stop_reason=output.get("stop_reason"),
        time=float(output.get("time") or 0.0),
        usage=output.get("usage"),
        completion_preview=str(output.get("completion") or "")[:240],
        tool_calls=tool_calls,
    )
    _write_json(input.call_ref, {"result": result.model_dump(mode="json")})
    return result


@activity.defn
async def finalize_extraction_chunk_activity(
    input: ExtractionChunkFinalizeInput,
) -> ExtractionChunkResult:
    from nof1_causal_lab.workers.schemas import WorkerOutput

    data = _read_json(input.result_ref)
    output = WorkerOutput.model_validate(data)
    dataframe = output.to_dataframe()

    result_ref = storage.join(
        _run_root(input.workspace_id, input.run_id),
        "results",
        f"worker-{input.worker_id:06d}.json",
    )
    _write_json(
        result_ref,
        {
            "dataframe": dataframe.to_dicts(),
            "n_extractions": len(output.extractions),
            "status": "completed",
        },
    )
    return ExtractionChunkResult(
        worker_id=input.worker_id,
        status="completed",
        n_extractions=len(output.extractions),
        n_windows=input.n_windows,
        n_llm_calls=input.n_llm_calls,
        result_ref=result_ref,
    )


@activity.defn
async def finalize_measurements_activity(input: MeasurementsFinalizeInput) -> TransitionEffects:
    import polars as pl

    from nof1_causal_lab.flows.artifact_contracts import MeasurementsContract
    from nof1_causal_lab.flows.transitions.extraction.materialization import (
        materialize_extraction_outputs,
    )
    from nof1_causal_lab.utils.data import ObservationRecord, annotate_observation_rows

    try:
        plan = _read_json(input.plan_ref)
        measurement_structure = plan["measurement_structure"]
        computed_dicts = list(plan.get("computed_dicts") or [])
        chunk_specs = list(plan.get("chunks") or [])
        results_by_worker = {result.worker_id: result for result in input.chunk_results}

        semantic_dicts: list[dict[str, Any]] = []
        worker_statuses: list[dict[str, Any]] = []

        for chunk_spec in chunk_specs:
            worker_id = int(chunk_spec["worker_id"])
            result = results_by_worker[worker_id]
            status: dict[str, Any] = {
                "worker_id": worker_id,
                "status": result.status,
                "n_extractions": result.n_extractions,
                "n_windows": result.n_windows,
            }
            if result.error is not None:
                status["error"] = result.error
            worker_statuses.append(status)

            if result.status == "completed" and result.result_ref is not None:
                chunk_payload = _read_json(result.result_ref)
                semantic_dicts.extend(chunk_payload.get("dataframe") or [])

        all_dicts = computed_dicts + semantic_dicts
        observation_rows = cast(
            "list[ObservationRecord]",
            annotate_observation_rows(pl.DataFrame(all_dicts), measurement_structure).to_dicts()
            if all_dicts
            else [],
        )
        extraction_result: dict[str, Any] = {
            "observation_rows": observation_rows,
            "worker_statuses": worker_statuses,
            "n_total_extractions": len(computed_dicts)
            + sum(
                result.n_extractions
                for result in input.chunk_results
                if result.status == "completed"
            ),
        }
        materialized = materialize_extraction_outputs(extraction_result, measurement_structure)
        panel = materialized["data_for_model"]
        report: dict[str, Any] = {"workers": materialized["worker_statuses"]}
        report = _filter_measurements_contract(MeasurementsContract, report)

        store = ArtifactStore(input.workspace_id)
        produced = [
            store.write_version(
                "measurements",
                provenance="computed",
                derived_from=input.pins,
                produced_by="run:measurements",
                json_files={json_filename("measurements", "measurements"): report},
            )
        ]
        if len(panel) > 0:
            produced.append(
                store.write_version(
                    "panel",
                    provenance="computed",
                    derived_from=input.pins,
                    produced_by="run:measurements",
                    parquet_files={parquet_filename("panel", "panel"): panel},
                )
            )

        spec = transition_spec("measurements")
        retracted = run_retractions(input.state, spec, produced)
        return complete_derivation_cascade(store, input.state, produced, retracted)
    except Exception as exc:
        raise _transition_failure(exc) from exc


MEASUREMENT_ACTIVITIES = [
    emit_transition_runtime_event_activity,
    emit_extraction_progress_event_activity,
    plan_measurements_activity,
    call_openrouter_activity,
    finalize_extraction_chunk_activity,
    finalize_measurements_activity,
]
