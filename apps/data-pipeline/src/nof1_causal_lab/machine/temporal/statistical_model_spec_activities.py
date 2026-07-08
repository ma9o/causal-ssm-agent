"""Temporal activities for the statistical-model-spec transition."""

from __future__ import annotations

import json
import pickle
import re
from typing import Any

from temporalio import activity
from temporalio.exceptions import ApplicationError

from nof1_causal_lab.machine.artifact_files import json_filename, parquet_filename
from nof1_causal_lab.machine.derivations import complete_derivation_cascade
from nof1_causal_lab.machine.errors import ModelCompileError, TransitionExecutionError
from nof1_causal_lab.machine.graph import transition_spec
from nof1_causal_lab.machine.moves import TransitionEffects, input_pins, run_retractions
from nof1_causal_lab.machine.store import ArtifactStore
from nof1_causal_lab.machine.temporal.latent_structure_activities import _llm_backend_config
from nof1_causal_lab.machine.temporal.messages import (
    StatisticalModelSpecAttemptFinalizeInput,
    StatisticalModelSpecAttemptPlan,
    StatisticalModelSpecAttemptPlanInput,
    StatisticalModelSpecAttemptResult,
    StatisticalModelSpecFailedEventInput,
    StatisticalModelSpecFinalizeInput,
    StatisticalModelSpecPlan,
    StatisticalModelSpecWorkflowInput,
)
from nof1_causal_lab.utils import data as data_module
from nof1_causal_lab.utils import storage


def _model_spec_root(workspace_id: str, run_id: str) -> str:
    return storage.join(data_module.runs_dir(workspace_id), "temporal-model-spec", run_id)


def _write_model_spec_json(path: str, value: Any) -> None:
    storage.write_text(path, json.dumps(value))


def _read_model_spec_json(path: str) -> Any:
    return storage.read_json(path)


def _write_model_spec_pickle(path: str, value: Any) -> None:
    with storage.open_file(path, "wb") as file:
        pickle.dump(value, file)


def _read_model_spec_pickle(path: str) -> Any:
    with storage.open_file(path, "rb") as file:
        return pickle.load(file)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-").lower() or "construct"


def _filter_model_spec_contract(cls: Any, data: dict[str, Any]) -> dict[str, Any]:
    fields = set(cls.model_fields.keys())
    return {key: value for key, value in data.items() if key in fields}


def _model_spec_transition_failure(exc: Exception) -> ApplicationError:
    if isinstance(exc, (TransitionExecutionError, ModelCompileError)):
        return ApplicationError(
            str(exc),
            getattr(exc, "diagnostics", None),
            type=type(exc).__name__,
            non_retryable=True,
        )
    return ApplicationError(str(exc), type=type(exc).__name__, non_retryable=True)


@activity.defn
async def plan_statistical_model_spec_activity(
    input: StatisticalModelSpecWorkflowInput,
) -> StatisticalModelSpecPlan:
    from nof1_causal_lab.flows.runtime_events import emit_model_spec_admission_event
    from nof1_causal_lab.flows.transitions.model_spec.agentic.construct_flow import (
        _MAX_ATTEMPTS_PER_CONSTRUCT,
        ConstructBuildState,
        _admission_plan_payload,
    )
    from nof1_causal_lab.models.ssm.construct_admission import build_construct_order
    from nof1_causal_lab.utils.config import get_config, get_secret

    store = ArtifactStore(input.workspace_id)
    spec = transition_spec("statistical_model_spec")
    pins = input_pins(input.state, spec)
    run_id = f"seq-{input.seq:06d}"
    root = _model_spec_root(input.workspace_id, run_id)

    question = store.read_json_file(
        "question",
        pins["question"],
        json_filename("question", "question"),
    )["text"]
    causal_design_payload = store.read_json_file(
        "causal_design",
        pins["causal_design"],
        json_filename("causal_design", "causal_design"),
    )
    causal_design = causal_design_payload["causal_design"]
    data_for_model = store.read_parquet_file(
        "panel",
        pins["panel"],
        parquet_filename("panel", "panel"),
    )
    validation_report = store.read_json_file(
        "validation_report",
        pins["validation_report"],
        json_filename("validation_report", "validation_report"),
    )

    config = get_config()
    requested_literature = (
        input.options.enable_literature
        if input.options.enable_literature is not None
        else config.prior_elicitation.literature_search.enabled
    )
    enable_literature = bool(requested_literature and get_secret("EXA_API_KEY"))
    order = build_construct_order(causal_design)
    emit_model_spec_admission_event(
        input.workspace_id,
        "plan",
        _admission_plan_payload(causal_design, order),
    )

    state = ConstructBuildState(
        causal_design=causal_design,
        data_for_model=data_for_model,
        order=order,
        workspace_id=input.workspace_id,
    )
    state_ref = storage.join(root, "construct-state.pkl")
    context_ref = storage.join(root, "context.json")
    _write_model_spec_pickle(state_ref, state)
    _write_model_spec_json(
        context_ref,
        {
            "question": question,
            "causal_design": causal_design,
            "indicator_audits": validation_report.get("indicators", {}),
            "enable_literature": enable_literature,
        },
    )

    max_tool_turns = config.prior_elicitation.max_tool_turns
    return StatisticalModelSpecPlan(
        workspace_id=input.workspace_id,
        run_id=run_id,
        state_ref=state_ref,
        context_ref=context_ref,
        pins=pins,
        order=order,
        llm=_llm_backend_config(config.prior_elicitation.llm, config.llm, max_tool_turns),
        max_tool_turns=max_tool_turns,
        max_attempts_per_construct=_MAX_ATTEMPTS_PER_CONSTRUCT,
    )


@activity.defn
async def plan_statistical_model_spec_attempt_activity(
    input: StatisticalModelSpecAttemptPlanInput,
) -> StatisticalModelSpecAttemptPlan:
    from nof1_causal_lab.flows.runtime_events import emit_model_spec_admission_event
    from nof1_causal_lab.flows.transitions.model_spec.agentic.construct_flow import (
        SUBMIT_CONSTRUCT_SCHEMA,
    )
    from nof1_causal_lab.flows.transitions.model_spec.agentic.construct_prompt import (
        build_construct_messages,
    )

    state = _read_model_spec_pickle(input.state_ref)
    construct = state.current_construct
    if construct is None:
        raise ValueError(
            "model-spec construct planning requested after all constructs were admitted"
        )

    state.attempt = input.attempt
    state.submission_made = False
    _write_model_spec_pickle(input.state_ref, state)

    metadata = _read_model_spec_json(input.context_ref)
    emit_model_spec_admission_event(
        input.workspace_id,
        "construct_started",
        {"construct": construct, "attempt": input.attempt},
    )
    system_prompt, user_prompt = build_construct_messages(
        state=state,
        construct=construct,
        question=metadata["question"],
        causal_design=metadata["causal_design"],
        indicator_audits=metadata["indicator_audits"],
    )
    subroutine_id = f"model-spec-{_slug(construct)}-attempt-{input.attempt:03d}"
    attempt_context_ref = storage.join(
        data_module.runs_dir(input.workspace_id),
        "temporal-llm",
        input.run_id,
        subroutine_id,
        "context.json",
    )
    _write_model_spec_json(
        attempt_context_ref,
        {
            "system_prompt": system_prompt,
            "user_messages": [user_prompt],
            "state_ref": input.state_ref,
            "enable_literature": metadata["enable_literature"],
            "submit_construct_schema": SUBMIT_CONSTRUCT_SCHEMA,
        },
    )
    return StatisticalModelSpecAttemptPlan(
        context_ref=attempt_context_ref,
        construct_name=construct,
        attempt=input.attempt,
        subroutine_id=subroutine_id,
    )


@activity.defn
async def finalize_statistical_model_spec_attempt_activity(
    input: StatisticalModelSpecAttemptFinalizeInput,
) -> StatisticalModelSpecAttemptResult:
    state = _read_model_spec_pickle(input.state_ref)
    if not state.submission_made:
        raise ValueError(
            f"model-spec construct `{input.construct_name}` did not call submit_construct before "
            "the turn ended."
        )
    admitted = state.current_construct != input.construct_name
    outcome = state.last_report.outcome if state.last_report is not None else "no report"
    return StatisticalModelSpecAttemptResult(
        construct_name=input.construct_name,
        attempt=input.attempt,
        admitted=admitted,
        outcome=outcome,
    )


@activity.defn
async def finalize_statistical_model_spec_activity(
    input: StatisticalModelSpecFinalizeInput,
) -> TransitionEffects:
    from nof1_causal_lab.flows.artifact_contracts import StatisticalModelSpecContract
    from nof1_causal_lab.flows.runtime_events import emit_model_spec_admission_event
    from nof1_causal_lab.flows.transitions.model_spec.assembly import (
        materialize_model_spec_result,
    )
    from nof1_causal_lab.utils.llm import LLMTrace, _merge_trace

    try:
        state = _read_model_spec_pickle(input.state_ref)
        if state.current_construct is not None:
            raise ValueError(f"model-spec construct `{state.current_construct}` is not admitted")
        emit_model_spec_admission_event(input.workspace_id, "done", {})

        trace = LLMTrace()
        for trace_ref in input.trace_refs:
            trace = _merge_trace(trace, LLMTrace.model_validate(_read_model_spec_json(trace_ref)))

        metadata = _read_model_spec_json(input.context_ref)
        statistical_model_spec = state.admission.statistical_model_spec().model_dump(mode="json")
        materialized = materialize_model_spec_result(
            statistical_model_spec=statistical_model_spec,
            authored_priors=dict(state.admission.priors),
            data_for_model=state.data_for_model,
            indicator_audits=metadata["indicator_audits"],
            causal_design=state.causal_design,
            validation=None,
            search_queries=dict(state.search_queries),
            skip_ppc=True,
        )
        if trace.messages or trace.usage.input_tokens or trace.usage.output_tokens:
            materialized["llm_trace"] = trace.model_dump(mode="json")

        compiled_ssm = materialized.pop("_compiled_ssm", None)
        report = _filter_model_spec_contract(StatisticalModelSpecContract, materialized)
        if compiled_ssm is None:
            raise ModelCompileError(
                "statistical_model_spec produced no compilable SSM from the proposed spec",
                transition_id="statistical_model_spec",
                diagnostics={"report": report},
            )

        store = ArtifactStore(input.workspace_id)
        produced = [
            store.write_version(
                "statistical_model_spec",
                provenance="computed",
                derived_from=input.pins,
                produced_by="run:statistical_model_spec",
                json_files={
                    json_filename("statistical_model_spec", "statistical_model_spec"): report
                },
            )
        ]
        spec = transition_spec("statistical_model_spec")
        retracted = run_retractions(input.state, spec, produced)
        return complete_derivation_cascade(store, input.state, produced, retracted)
    except Exception as exc:
        raise _model_spec_transition_failure(exc) from exc


@activity.defn
async def emit_model_spec_failed_event_activity(
    input: StatisticalModelSpecFailedEventInput,
) -> None:
    from nof1_causal_lab.flows.runtime_events import emit_model_spec_admission_event

    emit_model_spec_admission_event(
        input.workspace_id,
        "failed",
        {"construct": input.construct_name, "message": input.message},
    )


STATISTICAL_MODEL_SPEC_ACTIVITIES = [
    plan_statistical_model_spec_activity,
    plan_statistical_model_spec_attempt_activity,
    finalize_statistical_model_spec_attempt_activity,
    finalize_statistical_model_spec_activity,
    emit_model_spec_failed_event_activity,
]
