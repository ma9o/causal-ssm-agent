"""Temporal activities for the statistical-model-spec transition."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from temporalio import activity
from temporalio.exceptions import ApplicationError

from nof1_causal_lab.artifacts.structural_plan import StructuralPlan
from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001
from nof1_causal_lab.machine.artifact_files import json_filename, parquet_filename
from nof1_causal_lab.machine.derivations import complete_derivation_cascade
from nof1_causal_lab.machine.errors import ModelCompileError, TransitionExecutionError
from nof1_causal_lab.machine.graph import transition_spec
from nof1_causal_lab.machine.moves import TransitionEffects, input_pins, run_retractions
from nof1_causal_lab.machine.store import ArtifactStore
from nof1_causal_lab.machine.temporal.latent_structure_activities import _llm_backend_config
from nof1_causal_lab.machine.temporal.llm_subroutine_storage import subroutine_root
from nof1_causal_lab.machine.temporal.messages import (
    StatisticalModelSpecAdmissionUnit,
    StatisticalModelSpecAttemptFinalizeInput,
    StatisticalModelSpecAttemptPlan,
    StatisticalModelSpecAttemptPlanInput,
    StatisticalModelSpecAttemptResult,
    StatisticalModelSpecBarrierInput,
    StatisticalModelSpecBarrierResult,
    StatisticalModelSpecFailedEventInput,
    StatisticalModelSpecFinalizeInput,
    StatisticalModelSpecFrontierMergeInput,
    StatisticalModelSpecFrontierMergeResult,
    StatisticalModelSpecPlan,
    StatisticalModelSpecWorkflowInput,
)
from nof1_causal_lab.machine.temporal.model_spec_checkpoints import (
    ModelSpecRebaseSummary,
    ModelSpecSubmissionResult,
    latest_failed_model_spec_checkpoint_ref,
    read_model_spec_checkpoint,
    rebase_accepted_constructs,
    restore_construct_state,
    write_initial_model_spec_checkpoint,
    write_merged_model_spec_checkpoint,
)
from nof1_causal_lab.utils import data as data_module
from nof1_causal_lab.utils import storage

if TYPE_CHECKING:
    from nof1_causal_lab.machine.artifacts import ArtifactId


def _model_spec_root(workspace_id: str, run_id: str) -> str:
    return data_module.scratch_run_dir(workspace_id, run_id)


def _write_model_spec_json(path: str, value: Any) -> None:
    storage.write_text(path, json.dumps(value))


def _read_model_spec_json(path: str) -> Any:
    return storage.read_json(path)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-").lower() or "construct"


def _filter_model_spec_contract(cls: Any, data: UncheckedJsonObject) -> UncheckedJsonObject:
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


def _barrier_reopen_constructs(units: list[Any], failed_constructs: list[str]) -> set[str]:
    """Return failed SCC suffixes plus every downstream admission unit."""
    unit_by_id = {unit.unit_id: unit for unit in units}
    unit_by_construct = {construct: unit for unit in units for construct in unit.constructs}
    successors: dict[str, set[str]] = {unit.unit_id: set() for unit in units}
    for unit in units:
        for predecessor in unit.predecessors:
            successors[predecessor].add(unit.unit_id)

    reopen: set[str] = set()
    for construct in failed_constructs:
        unit = unit_by_construct[construct]
        failed_index = unit.constructs.index(construct)
        reopen.update(unit.constructs[failed_index:])
        pending = list(successors[unit.unit_id])
        seen_units: set[str] = set()
        while pending:
            unit_id = pending.pop()
            if unit_id in seen_units:
                continue
            seen_units.add(unit_id)
            descendant = unit_by_id[unit_id]
            reopen.update(descendant.constructs)
            pending.extend(successors[unit_id])
    return reopen


def _load_model_spec_inputs(
    workspace_id: str,
    pins: dict[ArtifactId, int],
) -> tuple[str, StructuralPlan, Any, UncheckedJsonObject]:
    store = ArtifactStore(workspace_id)
    question = store.read_json_file(
        "question",
        pins["question"],
        json_filename("question", "question"),
    )["text"]
    structural_plan = StructuralPlan.model_validate(
        store.read_json_file(
            "structural_plan",
            pins["structural_plan"],
            json_filename("structural_plan", "structural_plan"),
        )["structural_plan"]
    )
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
    return question, structural_plan, data_for_model, validation_report


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
    from nof1_causal_lab.models.ssm.construct_admission import (
        build_construct_order,
        build_construct_units,
    )
    from nof1_causal_lab.utils.config import get_config, get_secret

    spec = transition_spec("statistical_model_spec")
    pins = input_pins(input.state, spec)
    run_id = f"seq-{input.seq:06d}"
    root = _model_spec_root(input.workspace_id, run_id)
    question, structural_plan, data_for_model, validation_report = _load_model_spec_inputs(
        input.workspace_id,
        pins,
    )

    config = get_config()
    requested_literature = (
        input.options.enable_literature
        if input.options.enable_literature is not None
        else config.prior_elicitation.literature_search.enabled
    )
    enable_literature = bool(requested_literature and get_secret("EXA_API_KEY"))
    order = build_construct_order(structural_plan)
    units = build_construct_units(structural_plan)
    source_ref = latest_failed_model_spec_checkpoint_ref(input.workspace_id)
    source = read_model_spec_checkpoint(input.workspace_id, source_ref) if source_ref else None
    rebase: ModelSpecRebaseSummary | None = None
    if source is None:
        state = ConstructBuildState(
            structural_plan=structural_plan,
            data_for_model=data_for_model,
            order=order,
            workspace_id=None,
        )
        accepted_constructs = []
        search_queries: dict[str, str] = {}
        search_cache: dict[str, str] = {}
        repair_feedback: dict[str, str] = {}
    elif source.input_pins == pins:
        assert source_ref is not None
        state = restore_construct_state(
            source,
            structural_plan=structural_plan,
            data_for_model=data_for_model,
            workspace_id=None,
        )
        accepted_constructs = list(source.accepted_constructs)
        search_queries = dict(source.search_queries)
        search_cache = dict(source.search_cache)
        repair_feedback = dict(source.repair_feedback)
        rebase = ModelSpecRebaseSummary(
            source_checkpoint_ref=source_ref,
            pins_changed=False,
            retained_constructs=[item.construct_name for item in accepted_constructs],
        )
    else:
        assert source_ref is not None
        state, accepted_constructs, reopened, reason = rebase_accepted_constructs(
            source,
            structural_plan=structural_plan,
            data_for_model=data_for_model,
        )
        search_queries = dict(state.search_queries)
        search_cache = dict(state.search_cache)
        repair_feedback = {}
        rebase = ModelSpecRebaseSummary(
            source_checkpoint_ref=source_ref,
            pins_changed=True,
            retained_constructs=[item.construct_name for item in accepted_constructs],
            reopened_construct=reopened,
            reason=reason,
        )
    checkpoint_ref = write_initial_model_spec_checkpoint(
        workspace_id=input.workspace_id,
        run_id=run_id,
        seq=input.seq,
        pins=pins,
        accepted_constructs=accepted_constructs,
        search_queries=search_queries,
        search_cache=search_cache,
        repair_feedback=repair_feedback,
        parent_ref=source_ref,
        rebase=rebase,
    )
    emit_model_spec_admission_event(
        input.workspace_id,
        "plan",
        _admission_plan_payload(structural_plan, order),
    )
    if rebase is not None:
        emit_model_spec_admission_event(
            input.workspace_id,
            "resumed",
            {"checkpoint_ref": checkpoint_ref, **rebase.model_dump(mode="json")},
        )
    context_ref = storage.join(root, "context.json")
    _write_model_spec_json(
        context_ref,
        {
            "question": question,
            "structural_plan": structural_plan.model_dump(mode="json"),
            "indicator_audits": validation_report.get("indicators", {}),
            "enable_literature": enable_literature,
        },
    )

    max_tool_turns = config.prior_elicitation.max_tool_turns
    return StatisticalModelSpecPlan(
        workspace_id=input.workspace_id,
        run_id=run_id,
        checkpoint_ref=checkpoint_ref,
        context_ref=context_ref,
        pins=pins,
        units=[
            StatisticalModelSpecAdmissionUnit(
                unit_id=unit.unit_id,
                constructs=list(unit.constructs),
                predecessors=list(unit.predecessors),
            )
            for unit in units
        ],
        accepted_constructs=[item.construct_name for item in accepted_constructs],
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

    checkpoint = read_model_spec_checkpoint(input.workspace_id, input.checkpoint_ref)
    _question, structural_plan, data_for_model, validation_report = _load_model_spec_inputs(
        input.workspace_id,
        checkpoint.input_pins,
    )
    state = restore_construct_state(
        checkpoint,
        structural_plan=structural_plan,
        data_for_model=data_for_model,
        workspace_id=None,
        target_construct=input.construct_name,
    )
    construct = state.current_construct
    if construct is None:
        raise ValueError(
            "model-spec construct planning requested after all constructs were admitted"
        )

    state.attempt = input.attempt
    state.submission_made = False

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
        structural_plan=structural_plan,
        validation_report=validation_report,
    )
    subroutine_id = f"model-spec-{_slug(construct)}-attempt-{input.attempt:03d}"
    attempt_root = subroutine_root(input.workspace_id, input.run_id, subroutine_id)
    attempt_context_ref = storage.join(attempt_root, "context.json")
    result_ref = storage.join(attempt_root, "attempt-result.json")
    search_state_ref = storage.join(attempt_root, "search-state.json")
    _write_model_spec_json(
        search_state_ref,
        {
            "search_queries": checkpoint.search_queries,
            "search_cache": checkpoint.search_cache,
        },
    )
    _write_model_spec_json(
        attempt_context_ref,
        {
            "system_prompt": system_prompt,
            "user_messages": [user_prompt],
            "workspace_id": input.workspace_id,
            "checkpoint_ref": input.checkpoint_ref,
            "attempt": input.attempt,
            "attempt_result_ref": result_ref,
            "search_state_ref": search_state_ref,
            "enable_literature": metadata["enable_literature"],
            "submit_construct_schema": SUBMIT_CONSTRUCT_SCHEMA,
        },
    )
    return StatisticalModelSpecAttemptPlan(
        context_ref=attempt_context_ref,
        result_ref=result_ref,
        construct_name=construct,
        attempt=input.attempt,
        subroutine_id=subroutine_id,
    )


@activity.defn
async def merge_statistical_model_spec_frontier_activity(
    input: StatisticalModelSpecFrontierMergeInput,
) -> StatisticalModelSpecFrontierMergeResult:
    """Join completed branches into the current master through one deterministic write."""
    parent = read_model_spec_checkpoint(input.workspace_id, input.checkpoint_ref)
    accepted_by_name = {item.construct_name: item for item in parent.accepted_constructs}
    search_queries = dict(parent.search_queries)
    search_cache = dict(parent.search_cache)
    additions: list[str] = []

    def _merge_mapping(target: dict[str, str], source: dict[str, str], label: str) -> None:
        for key, value in source.items():
            existing = target.get(key)
            if existing is not None and existing != value:
                raise ValueError(f"Conflicting {label} value for {key!r} across frontier branches")
            target[key] = value

    def _merge_branch_mapping(
        target: dict[str, str],
        branch_values: dict[str, str],
        base_values: dict[str, str],
        label: str,
    ) -> None:
        for key, base_value in base_values.items():
            if branch_values.get(key) != base_value:
                raise ValueError(f"Branch changed inherited {label} value for {key!r}")
        _merge_mapping(
            target,
            {key: value for key, value in branch_values.items() if key not in base_values},
            label,
        )

    order_index = {name: index for index, name in enumerate(input.construct_order)}
    branch_refs = sorted(input.branch_checkpoint_refs)
    for branch_ref in branch_refs:
        branch = read_model_spec_checkpoint(input.workspace_id, branch_ref)
        if branch.parent_ref is None:
            raise ValueError(f"Frontier branch {branch_ref!r} has no parent checkpoint")
        branch_base = read_model_spec_checkpoint(input.workspace_id, branch.parent_ref)
        if branch.input_pins != parent.input_pins:
            raise ValueError(f"Frontier branch {branch_ref!r} has different input pins")
        if branch_base.input_pins != parent.input_pins:
            raise ValueError(f"Frontier branch base {branch.parent_ref!r} has different input pins")
        branch_base_by_name = {
            item.construct_name: item for item in branch_base.accepted_constructs
        }
        branch_by_name = {item.construct_name: item for item in branch.accepted_constructs}
        for name, base_item in branch_base_by_name.items():
            if accepted_by_name.get(name) != base_item:
                raise ValueError(
                    f"Frontier branch {branch_ref!r} was based on incompatible construct {name!r}"
                )
            if branch_by_name.get(name) != base_item:
                raise ValueError(
                    f"Frontier branch {branch_ref!r} changed inherited construct {name!r}"
                )
        branch_additions = [
            item
            for item in branch.accepted_constructs
            if item.construct_name not in branch_base_by_name
        ]
        if len(branch_additions) != 1:
            raise ValueError(
                f"Frontier branch {branch_ref!r} must add exactly one construct; "
                f"found {len(branch_additions)}"
            )
        added = branch_additions[0]
        if added.construct_name in accepted_by_name:
            raise ValueError(
                f"Frontier branch {branch_ref!r} repeats accepted construct "
                f"{added.construct_name!r}"
            )
        accepted_by_name[added.construct_name] = added
        additions.append(added.construct_name)
        _merge_branch_mapping(
            search_queries,
            branch.search_queries,
            branch_base.search_queries,
            "search query",
        )
        _merge_branch_mapping(
            search_cache,
            branch.search_cache,
            branch_base.search_cache,
            "search cache",
        )

    accepted_constructs = sorted(
        accepted_by_name.values(),
        key=lambda item: order_index[item.construct_name],
    )
    checkpoint_id = "frontier:" + ",".join(sorted(additions, key=order_index.__getitem__))
    next_ref = write_merged_model_spec_checkpoint(
        parent_ref=input.checkpoint_ref,
        parent=parent,
        checkpoint_id=checkpoint_id,
        accepted_constructs=accepted_constructs,
        search_queries=search_queries,
        search_cache=search_cache,
        repair_feedback={
            name: feedback
            for name, feedback in parent.repair_feedback.items()
            if name not in additions
        },
    )
    return StatisticalModelSpecFrontierMergeResult(
        checkpoint_ref=next_ref,
        accepted_constructs=[item.construct_name for item in accepted_constructs],
    )


@activity.defn
async def validate_statistical_model_spec_barrier_activity(
    input: StatisticalModelSpecBarrierInput,
) -> StatisticalModelSpecBarrierResult:
    """Run the exact full-model barrier and reopen only implicated dependency regions."""
    from nof1_causal_lab.flows.runtime_events import emit_model_spec_admission_event
    from nof1_causal_lab.flows.transitions.model_spec.agentic.construct_flow import (
        _acceptance_map,
        _check_result_payload,
        _closed_loop_target,
        _design_for_state,
        _timing_payload,
        render_admission_feedback,
    )
    from nof1_causal_lab.models.ssm.construct_admission import (
        build_construct_units,
        validate_full_admission_state,
    )

    checkpoint = read_model_spec_checkpoint(input.workspace_id, input.checkpoint_ref)
    _question, structural_plan, data_for_model, _validation_report = _load_model_spec_inputs(
        input.workspace_id,
        checkpoint.input_pins,
    )
    state = restore_construct_state(
        checkpoint,
        structural_plan=structural_plan,
        data_for_model=data_for_model,
        workspace_id=None,
    )
    accepted_by_name = {item.construct_name: item for item in checkpoint.accepted_constructs}
    if set(accepted_by_name) != set(input.construct_order):
        missing = sorted(set(input.construct_order) - set(accepted_by_name))
        raise ValueError(
            "Full-model barrier requires every construct; missing: " + ", ".join(missing)
        )

    try:
        targets = tuple(
            _closed_loop_target(
                state.admitted_contributions[name], structural_plan, state.admission.priors
            )
            for name in input.construct_order
        )
        design = _design_for_state(
            state.admission,
            structural_plan,
            data_for_model,
            n_draws=state.n_draws,
            seed=state.seed,
        )
        validation = validate_full_admission_state(
            state.admission,
            targets,
            structural_plan,
            design,
            accepted={
                name: _acceptance_map(accepted_by_name[name].accept)
                for name in input.construct_order
            },
        )
    except (ArithmeticError, AssertionError, LookupError, TypeError, ValueError) as exc:
        raise ApplicationError(
            str(exc),
            {
                "transition_id": "statistical_model_spec",
                "checkpoint_ref": input.checkpoint_ref,
                "report": {
                    "statistical_model_spec": state.admission.statistical_model_spec().model_dump(
                        mode="json"
                    ),
                    "authored_priors": dict(state.admission.priors),
                },
            },
            type="ModelCompileError",
            non_retryable=True,
        ) from exc
    reports_by_name = {report.name: report for report in validation.reports}
    failed = [report for report in validation.reports if not report.admitted]
    units = build_construct_units(structural_plan)
    reopen = _barrier_reopen_constructs(units, [report.name for report in failed])
    emit_model_spec_admission_event(
        input.workspace_id,
        "barrier_report",
        {
            "passed": not failed,
            "failed_constructs": [report.name for report in failed],
            "reopened_constructs": [name for name in input.construct_order if name in reopen],
            "timings": [_timing_payload(timing) for timing in validation.timings],
            "reports": [
                {
                    "name": report.name,
                    "outcome": report.outcome,
                    "admitted": report.admitted,
                    "results": [_check_result_payload(result) for result in report.results],
                    "timings": [_timing_payload(timing) for timing in report.timings],
                }
                for report in validation.reports
            ],
        },
    )

    updated: list[Any] = []
    for name in input.construct_order:
        if name in reopen:
            continue
        saved = accepted_by_name[name]
        report = reports_by_name[name]
        updated.append(
            saved.model_copy(
                update={
                    "annotations": list(report.annotations),
                    "results": [_check_result_payload(result) for result in report.results],
                    "outcome": report.outcome,
                    "feedback": render_admission_feedback(report),
                }
            )
        )

    repair_feedback = {report.name: render_admission_feedback(report) for report in failed}
    checkpoint_id = (
        "barrier:passed"
        if not failed
        else "barrier:repair:" + ",".join(report.name for report in failed)
    )
    next_ref = write_merged_model_spec_checkpoint(
        parent_ref=input.checkpoint_ref,
        parent=checkpoint,
        checkpoint_id=checkpoint_id,
        accepted_constructs=updated,
        search_queries=dict(checkpoint.search_queries),
        search_cache=dict(checkpoint.search_cache),
        repair_feedback=repair_feedback,
        full_model_validated=not failed,
    )
    return StatisticalModelSpecBarrierResult(
        passed=not failed,
        checkpoint_ref=next_ref,
        accepted_constructs=[item.construct_name for item in updated],
        reopened_constructs=[name for name in input.construct_order if name in reopen],
    )


@activity.defn
async def finalize_statistical_model_spec_attempt_activity(
    input: StatisticalModelSpecAttemptFinalizeInput,
) -> StatisticalModelSpecAttemptResult:
    if not storage.exists(input.result_ref):
        raise ValueError(
            f"model-spec construct `{input.construct_name}` did not call submit_construct before "
            "the turn ended."
        )
    submission = ModelSpecSubmissionResult.model_validate(_read_model_spec_json(input.result_ref))
    return StatisticalModelSpecAttemptResult(
        construct_name=input.construct_name,
        attempt=input.attempt,
        admitted=submission.admitted,
        outcome=submission.outcome,
        checkpoint_ref=submission.checkpoint_ref,
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

    try:
        checkpoint = read_model_spec_checkpoint(input.workspace_id, input.checkpoint_ref)
        _question, structural_plan, data_for_model, _validation_report = _load_model_spec_inputs(
            input.workspace_id,
            checkpoint.input_pins,
        )
        state = restore_construct_state(
            checkpoint,
            structural_plan=structural_plan,
            data_for_model=data_for_model,
            workspace_id=None,
        )
        from nof1_causal_lab.models.ssm.construct_admission import build_construct_order

        missing = sorted(
            set(build_construct_order(structural_plan))
            - {item.construct_name for item in checkpoint.accepted_constructs}
        )
        if missing:
            raise ValueError("model-spec constructs are not admitted: " + ", ".join(missing))
        if not checkpoint.full_model_validated:
            raise ValueError("model-spec full-model barrier has not passed")
        emit_model_spec_admission_event(input.workspace_id, "done", {})

        metadata = _read_model_spec_json(input.context_ref)
        statistical_model_spec = state.admission.statistical_model_spec().model_dump(mode="json")
        materialized = materialize_model_spec_result(
            statistical_model_spec=statistical_model_spec,
            authored_priors=dict(state.admission.priors),
            data_for_model=state.data_for_model,
            indicator_audits=metadata["indicator_audits"],
            structural_plan=state.structural_plan,
            validation=None,
            search_queries=dict(state.search_queries),
            skip_ppc=True,
        )
        materialized["prior_predictive_diagnostics"] = [
            result
            for accepted_construct in checkpoint.accepted_constructs
            for result in accepted_construct.results
        ]

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
        {
            "construct": input.construct_name,
            "message": input.message,
            "checkpoint_ref": input.checkpoint_ref,
        },
    )


STATISTICAL_MODEL_SPEC_ACTIVITIES = [
    plan_statistical_model_spec_activity,
    plan_statistical_model_spec_attempt_activity,
    merge_statistical_model_spec_frontier_activity,
    validate_statistical_model_spec_barrier_activity,
    finalize_statistical_model_spec_attempt_activity,
    finalize_statistical_model_spec_activity,
    emit_model_spec_failed_event_activity,
]
