"""Stage 4 grounding."""

from __future__ import annotations

from typing import Any

from causal_ssm_agent.flows.stages.stage4.agentic.stage4_feedback import (
    Stage4GroundingResult,
    make_stage4_grounding_result,
)
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_text import summarize_stage4_names

# ---------------------------------------------------------------------------
# Stage 4: Model grounding (model spec + priors, unified)
# ---------------------------------------------------------------------------


def stage4_grounding(
    data: dict,
    causal_spec: dict,
    current: dict | None = None,
    data_for_model: Any = None,
    indicator_audits: dict[str, dict[str, Any]] | None = None,
    *,
    skip_ppc: bool = False,
) -> Stage4GroundingResult:
    """Ground stage 4 proposals: validate, compile, optionally run prior predictive.

    ``data`` contains the proposed changes — any combination of:
    - ``model_spec``: complete ModelSpec dict
    - ``priors``: partial dict mapping parameter names to prior proposals

    The function merges proposals with ``current`` (existing stage-4 state),
    then validates schemas, compiles the model, and — when real priors and
    ``data_for_model`` are available — runs prior predictive checks.

    Checks (applied in order):
    1. Schema + domain validation for any submitted fields
    2. Compile (default priors if none available, real priors otherwise)
    3. Prior predictive (only when real priors + data_for_model present and skip_ppc is False)
    """
    from causal_ssm_agent.models.ssm_compiler import resolve_prior_proposals

    from .assembly import (
        format_prior_proposal_errors,
        format_validation_feedback,
        merge_priors,
        partition_prior_proposals,
        validate_assembly,
    )

    state = dict(current or {})
    output: dict = {}

    new_model_spec = data.get("model_spec")
    new_priors = data.get("priors")
    changed_parameters: tuple[str, ...] = ()

    if new_model_spec is None and new_priors is None:
        return make_stage4_grounding_result(
            stage_output=None,
            status="validation_error",
            feedback="VALIDATION ERRORS:\n- data must contain 'model_spec' and/or 'priors'",
            retain_for_next_prompt=True,
            capture_stage_output=False,
        )

    if new_model_spec is not None and new_priors is not None:
        return make_stage4_grounding_result(
            stage_output=None,
            status="update_rejected",
            feedback=(
                "UPDATE TOO BROAD:\n"
                "- submit model decisions and priors in separate calls\n"
                "- lock the model spec first, then add priors in later calls\n\n"
                "Previously accepted state is retained. Resubmit only the fields you changed."
            ),
            state_retained=True,
            retain_for_next_prompt=True,
            capture_stage_output=False,
        )

    if new_model_spec is not None and new_model_spec == state.get("model_spec"):
        return make_stage4_grounding_result(
            stage_output=None,
            status="update_rejected",
            feedback=_format_redundant_stage4_update_feedback(
                "model decisions",
                _collect_model_spec_targets(new_model_spec),
            ),
            state_retained=True,
            retain_for_next_prompt=True,
            capture_stage_output=False,
        )

    if isinstance(new_priors, dict):
        redundant_priors = _find_redundant_prior_updates(new_priors, state.get("authored_priors"))
        if redundant_priors:
            redundant_prior_names = set(redundant_priors)
            new_priors = {
                name: prior
                for name, prior in new_priors.items()
                if name not in redundant_prior_names
            }
        if not new_priors:
            return make_stage4_grounding_result(
                stage_output=None,
                status="update_rejected",
                feedback=_format_redundant_stage4_update_feedback("priors", redundant_priors),
                state_retained=True,
                retain_for_next_prompt=True,
                capture_stage_output=False,
            )
        changed_parameters = tuple(new_priors)

    # --- Merge model_spec ---
    if new_model_spec is not None:
        state["model_spec"] = new_model_spec
        output["model_spec"] = new_model_spec

    # --- Validate & merge priors ---
    if new_priors is not None:
        validated_priors, prior_errors = partition_prior_proposals(new_priors)
        if validated_priors:
            state["authored_priors"] = merge_priors(state.get("authored_priors"), validated_priors)
            output["authored_priors"] = state["authored_priors"]
        if prior_errors:
            return make_stage4_grounding_result(
                stage_output=output or None,
                status="validation_error",
                feedback=format_prior_proposal_errors(prior_errors),
                changed_parameters=tuple(validated_priors),
                retain_for_next_prompt=True,
                capture_stage_output=bool(output),
            )

    model_spec = state.get("model_spec")
    if model_spec is None:
        guidance = "Previously accepted state is retained. Submit only the missing model fields."
        message = "COMPILE ERROR:\nNo model_spec available — submit model_spec first"
        feedback = f"{message}\n\n{guidance}" if output else message
        return make_stage4_grounding_result(
            stage_output=output or None,
            status="compile_error",
            feedback=feedback,
            state_retained=bool(output),
            retain_for_next_prompt=True,
            capture_stage_output=False,
        )

    authored_priors = state.get("authored_priors")
    required_priors = _required_prior_names(model_spec)
    missing_priors = [name for name in required_priors if name not in (authored_priors or {})]

    # First lock in the model spec, then accumulate priors incrementally.
    validation = validate_assembly(
        model_spec,
        authored_priors if not missing_priors else None,
        data_for_model,
        indicator_audits,
        causal_spec,
        skip_ppc=skip_ppc,
    )

    if new_model_spec is not None and validation.normalized_model_spec is not None:
        output["model_spec"] = validation.normalized_model_spec
    output["validation"] = validation

    if not validation.compile_ok:
        return make_stage4_grounding_result(
            stage_output=output or None,
            status="compile_error",
            feedback=_with_stateful_retry_guidance(f"COMPILE ERROR:\n{validation.compile_error}"),
            validation=validation,
            changed_parameters=changed_parameters,
            state_retained=bool(output),
            retain_for_next_prompt=True,
            capture_stage_output=False,
        )

    if missing_priors:
        return make_stage4_grounding_result(
            stage_output=output,
            status="accepted_pending_priors",
            feedback=_format_missing_priors_feedback(missing_priors),
            validation=validation,
            changed_parameters=changed_parameters,
            retain_for_next_prompt=True,
            capture_stage_output=True,
        )

    if validation.compiled_ssm is None:
        raise ValueError("Stage 4 grounding requires compiled_ssm before resolving priors")
    output["resolved_priors"] = resolve_prior_proposals(
        validation.compiled_ssm,
        authored_priors=authored_priors or {},
    )

    if validation.is_valid:
        feedback = format_validation_feedback(
            validation,
            authored_priors or {},
        )
        return make_stage4_grounding_result(
            stage_output=output,
            status="accepted",
            feedback=feedback,
            validation=validation,
            changed_parameters=changed_parameters,
            retain_for_next_prompt=feedback != "VALID",
            capture_stage_output=True,
        )

    # Grounding returns a scope-free feedback view — the state-machine reducer
    # post-filters to an active-block subset via ``focus_parameters`` if it
    # wants a narrower LLM-facing rendering.
    feedback = format_validation_feedback(
        validation,
        authored_priors or {},
    )
    failure_status = (
        "sensitivity_failure" if validation.has_sensitivity_failure else "prior_predictive_failure"
    )
    return make_stage4_grounding_result(
        stage_output=output,
        status=failure_status,
        feedback=_with_stateful_retry_guidance(feedback),
        validation=validation,
        changed_parameters=changed_parameters,
        state_retained=True,
        retain_for_next_prompt=True,
        capture_stage_output=False,
    )


def should_capture_stage4_output(result: Stage4GroundingResult) -> bool:
    """Return whether a Stage 4 tool result should become the new accepted state.

    Accepted state includes:
    - structurally valid model decisions waiting on more priors
    - schema-valid prior subsets merged into the current accepted state
    - fully valid model + prior submissions

    Rejected compile attempts and prior-predictive failures must not overwrite
    the last accepted state, or the final materialized Stage 4 payload can drift
    away from what the tool actually accepted.
    """
    return result.stage_output is not None and result.validation_packet.capture_stage_output


def _required_prior_names(model_spec: dict | None) -> list[str]:
    """Return the parameter names that still need priors."""
    optional_roles = {"initial_state_mean", "initial_state_sd"}
    names: list[str] = []
    for parameter in (model_spec or {}).get("parameters") or []:
        if not isinstance(parameter, dict):
            continue
        if parameter.get("role") in optional_roles:
            continue
        if isinstance(parameter.get("name"), str):
            names.append(parameter["name"])
    return names


def _collect_model_spec_targets(model_spec: dict | None) -> list[str]:
    """Collect human-readable model decision targets from a full model spec."""
    targets: list[str] = []
    seen: set[str] = set()

    for likelihood in (model_spec or {}).get("likelihoods") or []:
        variable = likelihood.get("variable") if isinstance(likelihood, dict) else None
        if isinstance(variable, str) and variable not in seen:
            seen.add(variable)
            targets.append(variable)

    for parameter in (model_spec or {}).get("parameters") or []:
        if not isinstance(parameter, dict):
            continue
        if parameter.get("role") != "loading":
            continue
        name = parameter.get("name")
        if isinstance(name, str) and name not in seen:
            seen.add(name)
            targets.append(name)

    return targets


def _format_missing_priors_feedback(missing_priors: list[str]) -> str:
    """Guide the LLM to submit only the unresolved priors."""
    return (
        "MODEL STATE SAVED:\n"
        "- missing priors for "
        f"{len(missing_priors)} parameters: {summarize_stage4_names(missing_priors)}\n"
        "- model decisions are already locked; do not send distribution_choices again\n"
        "- your next submit tool call must contain only `priors`\n"
        "- do not resend unchanged fields\n"
        "- submit only the missing priors or any corrections"
    )


def _find_redundant_prior_updates(
    submitted_priors: dict[str, dict] | None,
    current_priors: dict[str, dict] | None,
) -> list[str]:
    """Return submitted prior names that exactly match accepted state."""
    redundant: list[str] = []
    current = current_priors or {}
    for name, prior in (submitted_priors or {}).items():
        if isinstance(name, str) and name in current and current[name] == prior:
            redundant.append(name)
    return sorted(redundant)


def _format_redundant_stage4_update_feedback(kind: str, names: list[str]) -> str:
    """Tell the LLM not to resend already accepted stage-4 fields."""
    summary = summarize_stage4_names(names)
    if kind == "model decisions":
        return (
            f"REDUNDANT {kind.upper()} UPDATE:\n"
            f"- already accepted and unchanged: {summary}\n"
            "- model-decision phase is closed; do not send distribution_choices again\n"
            "- your next submit tool call must contain only `priors`\n"
            "- do not resend unchanged fields\n\n"
            "Previously accepted state is retained. Resubmit only changed or missing priors."
        )
    return (
        f"REDUNDANT {kind.upper()} UPDATE:\n"
        f"- already accepted and unchanged: {summary}\n"
        "- do not resend unchanged fields\n\n"
        "Previously accepted state is retained. Resubmit only the fields you changed."
    )


def _with_stateful_retry_guidance(feedback: str) -> str:
    """Remind the LLM that accepted stage-4 state is preserved across retries."""
    return f"{feedback}\n\nPreviously accepted state is retained. Resubmit only the fields you changed."
