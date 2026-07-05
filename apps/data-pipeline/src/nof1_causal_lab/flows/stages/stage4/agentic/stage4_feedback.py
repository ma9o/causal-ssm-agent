"""Typed Stage 4 validation and prompt-snapshot abstractions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from nof1_causal_lab.flows.stages.stage4.assembly import AssemblyValidation
    from nof1_causal_lab.workers.schemas_prior import PriorValidationResult


Stage4ValidationStatus = Literal[
    "idle",
    "accepted",
    "accepted_pending_priors",
    "compile_error",
    "prior_predictive_failure",
    "partial_drift_failure",
    "repair_campaign_active",
    "repair_campaign_progress",
    "repair_campaign_ready",
    "model_review_reopened",
    "validation_error",
    "update_rejected",
    "info",
]


@dataclass(frozen=True)
class Stage4ValidationPacket:
    """Structured validation state persisted by the Stage 4 reducer."""

    status: Stage4ValidationStatus
    summary: str
    model_feedback: str
    active_scope_id: str | None = None
    failing_parameters: tuple[str, ...] = field(default_factory=tuple)
    coupled_parameters: tuple[str, ...] = field(default_factory=tuple)
    global_failure_sites: tuple[str, ...] = field(default_factory=tuple)
    diagnostic_codes: tuple[str, ...] = field(default_factory=tuple)
    changed_parameters: tuple[str, ...] = field(default_factory=tuple)
    state_retained: bool = False
    retain_for_next_prompt: bool = True
    capture_stage_output: bool = False


@dataclass(frozen=True)
class Stage4GroundingResult:
    """Typed result returned by Stage 4 grounding."""

    stage_output: dict[str, Any] | None
    validation_packet: Stage4ValidationPacket

    @property
    def feedback(self) -> str:
        """Return the model-facing validator feedback string."""
        return self.validation_packet.model_feedback


def make_stage4_validation_packet(
    *,
    status: Stage4ValidationStatus,
    feedback: str,
    validation: AssemblyValidation | None = None,
    active_scope_id: str | None = None,
    changed_parameters: tuple[str, ...] = (),
    state_retained: bool = False,
    retain_for_next_prompt: bool = True,
    capture_stage_output: bool = False,
) -> Stage4ValidationPacket:
    """Create a typed validation packet without inferring control flow from text."""
    diagnostics = tuple(validation.diagnostics) if validation is not None else ()
    failing_parameters, coupled_parameters, global_failure_sites = _collect_failure_context(
        diagnostics
    )
    diagnostic_codes = tuple(
        sorted(
            {
                diagnostic.code
                for diagnostic in diagnostics
                if not diagnostic.is_valid and diagnostic.code
            }
        )
    )
    return Stage4ValidationPacket(
        status=status,
        summary=_validation_summary(
            status,
            failing_parameters=failing_parameters,
            coupled_parameters=coupled_parameters,
            global_failure_sites=global_failure_sites,
        ),
        model_feedback=feedback,
        active_scope_id=active_scope_id,
        failing_parameters=failing_parameters,
        coupled_parameters=coupled_parameters,
        global_failure_sites=global_failure_sites,
        diagnostic_codes=diagnostic_codes,
        changed_parameters=changed_parameters,
        state_retained=state_retained,
        retain_for_next_prompt=retain_for_next_prompt,
        capture_stage_output=capture_stage_output,
    )


def make_stage4_grounding_result(
    *,
    stage_output: dict[str, Any] | None,
    status: Stage4ValidationStatus,
    feedback: str,
    validation: AssemblyValidation | None = None,
    active_scope_id: str | None = None,
    changed_parameters: tuple[str, ...] = (),
    state_retained: bool = False,
    retain_for_next_prompt: bool = True,
    capture_stage_output: bool = False,
) -> Stage4GroundingResult:
    """Construct a typed Stage 4 grounding result."""
    return Stage4GroundingResult(
        stage_output=stage_output,
        validation_packet=make_stage4_validation_packet(
            status=status,
            feedback=feedback,
            validation=validation,
            active_scope_id=active_scope_id,
            changed_parameters=changed_parameters,
            state_retained=state_retained,
            retain_for_next_prompt=retain_for_next_prompt,
            capture_stage_output=capture_stage_output,
        ),
    )


def _collect_failure_context(
    diagnostics: tuple[PriorValidationResult, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Extract failing, coupled, and global parameter hints from diagnostics."""
    from nof1_causal_lab.models.ssm.compile.common import GLOBAL_FAILURE_SITES

    failing_parameters: set[str] = set()
    coupled_parameters: set[str] = set()
    global_failure_sites: set[str] = set()

    for diagnostic in diagnostics:
        if diagnostic.is_valid:
            continue
        parameter_name = diagnostic.parameter
        if parameter_name in GLOBAL_FAILURE_SITES:
            global_failure_sites.add(parameter_name)
        else:
            failing_parameters.add(parameter_name)
        for related_parameter in diagnostic.related_parameters:
            if not isinstance(related_parameter, str):
                continue
            if related_parameter in GLOBAL_FAILURE_SITES:
                global_failure_sites.add(related_parameter)
                continue
            if related_parameter not in failing_parameters:
                coupled_parameters.add(related_parameter)

    coupled_parameters.difference_update(failing_parameters)
    return (
        tuple(sorted(failing_parameters)),
        tuple(sorted(coupled_parameters)),
        tuple(sorted(global_failure_sites)),
    )


def _validation_summary(
    status: Stage4ValidationStatus,
    *,
    failing_parameters: tuple[str, ...],
    coupled_parameters: tuple[str, ...],
    global_failure_sites: tuple[str, ...],
) -> str:
    """Produce a compact reducer and prompt summary."""
    if status == "idle":
        return "No validator feedback yet."
    if status == "compile_error":
        return "Compile failed for the active submission."
    if status == "partial_drift_failure":
        return "Partial drift stability failed for the active prior edit."
    if status == "prior_predictive_failure":
        if global_failure_sites:
            return "Prior predictive validation failed at global sites."
        if failing_parameters:
            return "Prior predictive validation failed for active or related parameters."
        return "Prior predictive validation failed."
    if status == "accepted_pending_priors":
        return "Current state was accepted, but the full prior inventory is still incomplete."
    if status == "repair_campaign_active":
        return "The reducer widened this failure into a multi-block repair campaign."
    if status == "repair_campaign_progress":
        return "The active repair campaign recorded local progress and remains open."
    if status == "repair_campaign_ready":
        return "The active repair campaign is ready for barrier validation."
    if status == "model_review_reopened":
        return "Global review reopened earlier model-form decisions."
    if status == "validation_error":
        return "The submission shape did not satisfy the active block contract."
    if status == "update_rejected":
        return "The submission was rejected before model validation."
    if status == "accepted":
        return "The submission was accepted."
    if coupled_parameters:
        return "Validator guidance references coupled parameters outside the active block."
    return "Validator returned guidance for the active block."
