"""Checkpointed driver for Codex-authored model-spec proposals.

The notebook owns only orchestration and presentation. Every proposal is still
validated by :class:`ConstructBuildState`, which invokes the production compiler,
exact nonlinear prior-predictive sampler, and C1--C5 reachability battery. Exact
admission evaluations are content-addressed in the workspace cache, and accepted
checkpoint state is restored through the production reducer without rerunning checks.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from nof1_causal_lab.flows.transitions.model_spec.agentic.construct_flow import (
    ConstructBuildState,
    _acceptance_map,
    _check_result_payload,
    _closed_loop_target,
    _design_for_state,
)
from nof1_causal_lab.flows.transitions.model_spec.agentic.construct_prompt import (
    build_construct_messages,
)
from nof1_causal_lab.machine.temporal.model_spec_checkpoints import (
    AcceptedConstructCheckpoint,
    ModelSpecAdmissionEvaluation,
    ModelSpecCheckpoint,
    model_spec_admission_evaluation_key,
    model_spec_admission_evaluation_path,
    read_model_spec_admission_evaluation,
    restore_construct_state,
    write_model_spec_admission_evaluation,
)
from nof1_causal_lab.models.ssm.construct_admission import (
    AdmissionReport,
    AdmissionTiming,
    FullAdmissionValidation,
    build_construct_order,
    validate_full_admission_state,
)
from nof1_causal_lab.models.ssm.reachability import CheckResult
from nof1_causal_lab.utils import storage

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import polars as pl

    from nof1_causal_lab.artifacts.structural_plan import StructuralPlan


_SUBMISSION_ERRORS = (
    ArithmeticError,
    AssertionError,
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)

type ConstructPayload = Mapping[str, Any]
type ValidationReportPayload = dict[str, Any]


@dataclass(frozen=True)
class AuthoredAttempt:
    """One notebook-authored call to the production ``submit_construct`` seam."""

    construct: str
    attempt: int
    payload: ConstructPayload
    feedback: str
    report: AdmissionReport | None
    coupled_results: tuple[CheckResult, ...]
    admitted: bool
    error_type: str | None = None
    cache_hit: bool = False


@dataclass
class WorkbenchRun:
    """Replayable result of the authored proposal list."""

    state: ConstructBuildState
    attempts: list[AuthoredAttempt] = field(default_factory=list)
    accepted_payloads: dict[str, ConstructPayload] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return self.state.current_construct is None


def run_authored_proposals(
    *,
    cache_workspace_id: str,
    structural_plan: StructuralPlan,
    data_for_model: pl.DataFrame,
    proposals: Sequence[ConstructPayload],
    n_draws: int,
    seed: int,
) -> WorkbenchRun:
    """Replay authored proposals through cached production reducer checkpoints.

    Every reactive run starts from an immutable empty checkpoint. Unchanged
    submissions reuse content-addressed admission evaluations, while misses run
    ``submit_construct`` and persist its compact semantic result. Accepted
    contributions are restored through ``restore_construct_state`` before the
    next submission, so notebook reactivity never mutates a prior run's state.
    """
    structural_plan_payload = json.dumps(
        structural_plan.model_dump(mode="json"),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    panel_payload = data_for_model.serialize(format="binary")
    if not isinstance(panel_payload, bytes):
        raise TypeError("Binary Polars serialization must return bytes")
    input_identity = {
        "kind": "prior-specification-workbench-v1",
        "structural_plan_sha256": hashlib.sha256(structural_plan_payload).hexdigest(),
        "panel_sha256": hashlib.sha256(panel_payload).hexdigest(),
    }
    checkpoint = ModelSpecCheckpoint(
        workspace_id=cache_workspace_id,
        run_id="prior-specification-workbench",
        seq=0,
        checkpoint_index=0,
        input_pins={},
        created_at="1970-01-01T00:00:00+00:00",
    )
    order = build_construct_order(structural_plan)
    first_construct = order[0] if order else None
    state = restore_construct_state(
        checkpoint,
        structural_plan=structural_plan,
        data_for_model=data_for_model,
        workspace_id=None,
        target_construct=first_construct,
    )
    state.n_draws = n_draws
    state.seed = seed
    run = WorkbenchRun(state=state)
    attempts_by_construct: dict[str, int] = {}

    for payload in proposals:
        construct = str(payload["construct"])
        attempts_by_construct[construct] = attempts_by_construct.get(construct, 0) + 1
        attempt = attempts_by_construct[construct]
        accepted_names = {item.construct_name for item in checkpoint.accepted_constructs}
        expected = next((name for name in order if name not in accepted_names), None)
        state = restore_construct_state(
            checkpoint,
            structural_plan=structural_plan,
            data_for_model=data_for_model,
            workspace_id=None,
            target_construct=expected,
        )
        state.n_draws = n_draws
        state.seed = seed
        state.attempt = attempt
        state.submission_made = False
        indicators = list(payload["indicators"])
        priors = dict(payload["priors"])
        accept = list(payload.get("accept") or [])
        report: AdmissionReport | None = None
        coupled: tuple[CheckResult, ...] = ()
        admitted = False
        error_type: str | None = None
        cache_hit = False

        if construct != expected:
            feedback = state.submit_construct(
                construct=construct,
                indicators=indicators,
                priors=priors,
                accept=accept,
            )
        else:
            evaluation_key = model_spec_admission_evaluation_key(
                input_identity=input_identity,
                accepted_constructs=checkpoint.accepted_constructs,
                ancestor_constructs=set(state.admitted_contributions),
                construct_name=construct,
                indicators=indicators,
                priors=priors,
                accept=accept,
                n_draws=n_draws,
                seed=seed,
            )
            evaluation_path = model_spec_admission_evaluation_path(
                cache_workspace_id,
                evaluation_key,
            )
            cache_hit = storage.exists(evaluation_path)
            if cache_hit:
                evaluation = read_model_spec_admission_evaluation(evaluation_path)
                feedback = evaluation.feedback
                admitted = evaluation.admitted
                error_type = "CachedSubmissionError" if evaluation.error is not None else None
                if evaluation.results:
                    report = _report_from_cached_evaluation(evaluation)
            else:
                try:
                    feedback = state.submit_construct(
                        construct=construct,
                        indicators=indicators,
                        priors=priors,
                        accept=accept,
                    )
                except _SUBMISSION_ERRORS as exc:
                    feedback = str(exc)
                    error_type = type(exc).__name__
                    evaluation = ModelSpecAdmissionEvaluation(
                        evaluation_key=evaluation_key,
                        construct_name=construct,
                        admitted=False,
                        outcome=feedback,
                        feedback=feedback,
                        error=feedback,
                    )
                else:
                    report = state.last_report
                    coupled = tuple(state.last_coupled_results)
                    admitted = state.current_construct != construct
                    outcome = (
                        report.outcome
                        if report is not None and report.name == construct
                        else feedback
                    )
                    annotations = (
                        list(report.annotations)
                        if report is not None and report.name == construct
                        else []
                    )
                    results = (
                        [_check_result_payload(result) for result in (*report.results, *coupled)]
                        if report is not None and report.name == construct
                        else []
                    )
                    evaluation = ModelSpecAdmissionEvaluation(
                        evaluation_key=evaluation_key,
                        construct_name=construct,
                        admitted=admitted,
                        outcome=outcome,
                        feedback=feedback,
                        annotations=annotations,
                        results=results,
                    )
                write_model_spec_admission_evaluation(evaluation_path, evaluation)

            if admitted:
                accepted = AcceptedConstructCheckpoint(
                    submission_id=f"workbench:{evaluation_key}",
                    construct_name=construct,
                    indicators=indicators,
                    priors=priors,
                    accept=accept,
                    annotations=evaluation.annotations,
                    results=evaluation.results,
                    outcome=evaluation.outcome,
                    feedback=evaluation.feedback,
                )
                checkpoint = checkpoint.model_copy(
                    update={
                        "checkpoint_index": checkpoint.checkpoint_index + 1,
                        "accepted_constructs": [*checkpoint.accepted_constructs, accepted],
                    }
                )

        run.attempts.append(
            AuthoredAttempt(
                construct=construct,
                attempt=attempt,
                payload=payload,
                feedback=feedback,
                report=report,
                coupled_results=tuple(coupled),
                admitted=admitted,
                error_type=error_type,
                cache_hit=cache_hit,
            )
        )
        if admitted:
            run.accepted_payloads[construct] = payload

    accepted_names = {item.construct_name for item in checkpoint.accepted_constructs}
    expected = next((name for name in order if name not in accepted_names), None)
    run.state = restore_construct_state(
        checkpoint,
        structural_plan=structural_plan,
        data_for_model=data_for_model,
        workspace_id=None,
        target_construct=expected,
    )
    run.state.n_draws = n_draws
    run.state.seed = seed
    return run


def _report_from_cached_evaluation(
    evaluation: ModelSpecAdmissionEvaluation,
) -> AdmissionReport:
    """Hydrate the compact semantic report stored by the production cache."""
    results = tuple(
        CheckResult(
            check=str(payload["check"]),
            target=str(payload["target"]),
            value=str(payload["value"]),
            band=str(payload["band"]),
            passed=bool(payload["passed"]),
            note=str(payload["note"]),
            diagnosis=tuple(str(item) for item in payload.get("diagnosis", ())),
        )
        for payload in evaluation.results
    )
    return AdmissionReport(
        name=evaluation.construct_name,
        results=results,
        timings=(
            AdmissionTiming(
                phase="admission_evaluation_cache",
                label="Cached exact admission evaluation",
                duration_ms=0.0,
            ),
        ),
        outcome=evaluation.outcome,
        annotations=tuple(evaluation.annotations),
        admitted=evaluation.admitted,
    )


def next_construct_prompt(
    *,
    run: WorkbenchRun,
    question: str,
    structural_plan: StructuralPlan,
    validation_report: ValidationReportPayload,
) -> tuple[str, str] | None:
    """Render the production prompt for the next proposal Codex should author."""
    construct = run.state.current_construct
    if construct is None:
        return None
    return build_construct_messages(
        state=run.state,
        construct=construct,
        question=question,
        structural_plan=structural_plan,
        validation_report=validation_report,
    )


def validate_full_model(
    *,
    run: WorkbenchRun,
    structural_plan: StructuralPlan,
    data_for_model: pl.DataFrame,
) -> FullAdmissionValidation:
    """Run the production full-model barrier on a complete authored state."""
    if not run.complete:
        raise ValueError(
            "Full-model validation requires every construct; next active construct is "
            f"{run.state.current_construct!r}."
        )
    order = build_construct_order(structural_plan)
    targets = tuple(
        _closed_loop_target(
            run.state.admitted_contributions[name],
            structural_plan,
            run.state.admission.priors,
        )
        for name in order
    )
    design = _design_for_state(
        run.state.admission,
        structural_plan,
        data_for_model,
        n_draws=run.state.n_draws,
        seed=run.state.seed,
    )
    accepted = {
        name: _acceptance_map(payload.get("accept"))
        for name, payload in run.accepted_payloads.items()
    }
    return validate_full_admission_state(
        run.state.admission,
        targets,
        structural_plan,
        design,
        accepted=accepted,
    )


__all__ = [
    "AuthoredAttempt",
    "WorkbenchRun",
    "next_construct_prompt",
    "run_authored_proposals",
    "validate_full_model",
]
