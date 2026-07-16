"""Storage-free driver for Codex-authored model-spec proposals.

The notebook owns only orchestration and presentation. Every proposal is still
validated by :class:`ConstructBuildState`, which invokes the production compiler,
exact nonlinear prior-predictive sampler, and C1--C5 reachability battery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from nof1_causal_lab.flows.transitions.model_spec.agentic.construct_flow import (
    ConstructBuildState,
    _acceptance_map,
    _closed_loop_target,
    _design_for_state,
)
from nof1_causal_lab.flows.transitions.model_spec.agentic.construct_prompt import (
    build_construct_messages,
)
from nof1_causal_lab.models.ssm.construct_admission import (
    AdmissionReport,
    FullAdmissionValidation,
    build_construct_order,
    validate_full_admission_state,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import polars as pl

    from nof1_causal_lab.models.ssm.reachability import CheckResult


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


@dataclass(frozen=True)
class AuthoredAttempt:
    """One notebook-authored call to the production ``submit_construct`` seam."""

    construct: str
    attempt: int
    payload: Mapping[str, Any]
    feedback: str
    report: AdmissionReport | None
    coupled_results: tuple[CheckResult, ...]
    admitted: bool
    error_type: str | None = None


@dataclass
class WorkbenchRun:
    """Replayable result of the authored proposal list."""

    state: ConstructBuildState
    attempts: list[AuthoredAttempt] = field(default_factory=list)
    accepted_payloads: dict[str, Mapping[str, Any]] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return self.state.current_construct is None


def run_authored_proposals(
    *,
    causal_design: dict[str, Any],
    data_for_model: pl.DataFrame,
    proposals: Sequence[Mapping[str, Any]],
    n_draws: int,
    seed: int,
) -> WorkbenchRun:
    """Replay authored proposals from a fresh production construct state.

    Replaying from scratch makes marimo reactivity safe: editing an earlier
    proposal deterministically recomputes every downstream admission instead of
    mutating a long-lived notebook object twice.
    """
    state = ConstructBuildState(
        causal_design=causal_design,
        data_for_model=data_for_model,
        order=build_construct_order(causal_design),
        n_draws=n_draws,
        seed=seed,
        workspace_id=None,
    )
    run = WorkbenchRun(state=state)
    attempts_by_construct: dict[str, int] = {}

    for payload in proposals:
        construct = str(payload["construct"])
        attempts_by_construct[construct] = attempts_by_construct.get(construct, 0) + 1
        attempt = attempts_by_construct[construct]
        state.attempt = attempt
        state.submission_made = False
        previous_report = state.last_report
        previous_coupled = state.last_coupled_results
        previous_construct = state.current_construct
        error_type: str | None = None
        try:
            feedback = state.submit_construct(
                construct=construct,
                indicators=list(payload["indicators"]),
                priors=dict(payload["priors"]),
                accept=list(payload.get("accept") or []),
            )
        except _SUBMISSION_ERRORS as exc:
            feedback = str(exc)
            error_type = type(exc).__name__

        report = state.last_report if state.last_report is not previous_report else None
        coupled = (
            state.last_coupled_results if state.last_coupled_results is not previous_coupled else ()
        )
        admitted = previous_construct == construct and state.current_construct != construct
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
            )
        )
        if admitted:
            run.accepted_payloads[construct] = payload

    return run


def next_construct_prompt(
    *,
    run: WorkbenchRun,
    question: str,
    causal_design: dict[str, Any],
    validation_report: dict[str, Any],
) -> tuple[str, str] | None:
    """Render the production prompt for the next proposal Codex should author."""
    construct = run.state.current_construct
    if construct is None:
        return None
    return build_construct_messages(
        state=run.state,
        construct=construct,
        question=question,
        causal_design=causal_design,
        validation_report=validation_report,
    )


def validate_full_model(
    *,
    run: WorkbenchRun,
    causal_design: dict[str, Any],
    data_for_model: pl.DataFrame,
) -> FullAdmissionValidation:
    """Run the production full-model barrier on a complete authored state."""
    if not run.complete:
        raise ValueError(
            "Full-model validation requires every construct; next active construct is "
            f"{run.state.current_construct!r}."
        )
    order = build_construct_order(causal_design)
    targets = tuple(
        _closed_loop_target(
            run.state.admitted_contributions[name],
            causal_design,
            run.state.admission.priors,
        )
        for name in order
    )
    design = _design_for_state(
        run.state.admission,
        causal_design,
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
        causal_design,
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
