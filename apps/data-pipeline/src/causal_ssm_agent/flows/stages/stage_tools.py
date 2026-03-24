"""Grounding functions for interactive stage tools.

These are the single source of truth for validation + derived computation.
Used by both pipeline tools (via make_stage_tool) and the refinement tool_server.

Each grounding function returns (stage_output, feedback):
- stage_output: dict with derived fields on success, None on failure
- feedback: string the LLM sees ("VALID", "VALIDATION ERRORS: ...", etc.)

Grounding functions anchor LLM proposals to reality — they validate structure,
check domain constraints, and derive fields the LLM cannot be trusted to compute.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from causal_ssm_agent.flows import get_prefect_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_prefect_logger(__name__)
MAX_STAGE4_PRIOR_BATCH_SIZE = 8


# ---------------------------------------------------------------------------
# Stage 1a: Latent model grounding
# ---------------------------------------------------------------------------


def stage1a_grounding(data: dict) -> tuple[dict | None, str]:
    """Validate latent model.

    Returns:
        (stage_output, feedback)
        stage_output = {"latent_model": data}
    """
    from causal_ssm_agent.orchestrator.schemas import validate_latent_model

    _result, errors = validate_latent_model(data)
    if errors:
        return None, "VALIDATION ERRORS:\n" + "\n".join(f"- {e}" for e in errors)

    return {"latent_model": data}, "VALID"


# ---------------------------------------------------------------------------
# Stage 1b: Measurement model grounding
# ---------------------------------------------------------------------------


def stage1b_grounding(data: dict, latent_model: dict) -> tuple[dict | None, str]:
    """Validate measurement model, check identifiability, build CausalSpec.

    Three outcomes:
    1. Schema/compiler failure → (None, "VALIDATION ERRORS: ...")
    2. Valid but not identifiable → (stage_output, "IDENTIFIABILITY ISSUES: ...")
    3. Valid and identifiable → (stage_output, "VALID")

    stage_output is set on cases 2 and 3 (whenever structurally valid).
    """
    from causal_ssm_agent.models.ssm_compiler import validate_measurement_model_for_compilation
    from causal_ssm_agent.orchestrator.agents import build_causal_spec
    from causal_ssm_agent.orchestrator.schemas import LatentModel
    from causal_ssm_agent.utils.identifiability import check_identifiability

    latent = LatentModel.model_validate(latent_model)
    validated, errors = validate_measurement_model_for_compilation(data, latent)
    if errors:
        return None, "VALIDATION ERRORS:\n" + "\n".join(f"- {e}" for e in errors)

    assert validated is not None
    measurement = validated.model_dump(mode="json")

    id_result = check_identifiability(latent_model, measurement)
    id_status = {
        "identifiable_treatments": id_result.get("identifiable_treatments", {}),
        "non_identifiable_treatments": id_result.get("non_identifiable_treatments", {}),
    }
    if "graph_info" in id_result:
        id_status["graph_info"] = id_result["graph_info"]

    causal_spec = build_causal_spec(latent_model, measurement, id_status)
    stage_output: dict[str, Any] = {"causal_spec": causal_spec}
    if "graph_info" in id_result:
        stage_output["graph_info"] = id_result["graph_info"]

    if id_result.get("non_identifiable_treatments"):
        feedback = _format_identifiability_feedback(id_result, latent_model)
        return stage_output, feedback

    return stage_output, "VALID"


def _format_identifiability_feedback(id_result: dict, latent_model: dict) -> str:
    """Rich feedback when model is valid but not fully identifiable."""
    lines = [
        "Structure is VALID but causal effects are NOT fully identifiable.",
        "",
        "Non-identifiable effects:",
    ]
    non_id = id_result.get("non_identifiable_treatments", {})
    construct_names = {c["name"] for c in latent_model.get("constructs", [])}

    all_confounders: set[str] = set()
    for treatment, info in sorted(non_id.items()):
        if not isinstance(info, dict):
            lines.append(f"  - {treatment}: {info}")
            continue
        confounders = info.get("confounders", [])
        notes = info.get("notes", "")
        if confounders:
            lines.append(f"  - {treatment}: blocked by {', '.join(confounders)}")
            all_confounders.update(c for c in confounders if c in construct_names)
        elif notes:
            lines.append(f"  - {treatment}: {notes}")

    if all_confounders:
        lines.extend(
            [
                "",
                "To fix: add proxy indicators for the blocking confounders and resubmit",
                "the COMPLETE measurement model (all existing indicators + new proxies).",
                f"Confounders needing proxies: {', '.join(sorted(all_confounders))}",
                "",
                "A proxy is an observable variable in the dataset that correlates with",
                "the unobserved confounder. Add it as a new indicator with the confounder",
                "as its construct_name.",
                "",
                "If no suitable proxy exists in the data, proceed — those effects will",
                "remain non-identifiable and be flagged in downstream analysis.",
            ]
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage 4: Model grounding (model spec + priors, unified)
# ---------------------------------------------------------------------------


def stage4_grounding(
    data: dict,
    causal_spec: dict,
    current: dict | None = None,
    data_for_model: Any = None,
    indicator_audits: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict | None, str]:
    """Ground stage 4 proposals: validate, compile, optionally run prior predictive.

    ``data`` contains the proposed changes — any combination of:
    - ``model_spec``: complete ModelSpec dict
    - ``priors``: partial dict mapping parameter names to prior proposals

    The function merges proposals with ``current`` (existing stage-4 state),
    then validates schemas, compiles the model, and — when real priors and
    ``data_for_model`` are available — runs prior predictive checks.

    Gates (applied in order):
    1. Schema + domain validation for any submitted fields
    2. Compile (default priors if none available, real priors otherwise)
    3. Prior predictive (only when real priors + data_for_model present)
    """
    from causal_ssm_agent.models.ssm_compiler import resolve_prior_proposals

    from .stage4_assembly import (
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

    if new_model_spec is None and new_priors is None:
        return None, "VALIDATION ERRORS:\n- data must contain 'model_spec' and/or 'priors'"

    if new_model_spec is not None and new_priors is not None:
        return (
            None,
            "UPDATE TOO BROAD:\n"
            "- submit model decisions and priors in separate calls\n"
            "- lock the model spec first, then add priors in later calls\n\n"
            "Previously accepted state is retained. Resubmit only the fields you changed.",
        )

    if new_model_spec is not None and new_model_spec == state.get("model_spec"):
        return (
            None,
            _format_redundant_stage4_update_feedback(
                "model decisions",
                _collect_model_spec_targets(new_model_spec),
            ),
        )

    if new_priors is not None:
        prior_names = sorted(name for name in new_priors if isinstance(name, str))
        if len(prior_names) > MAX_STAGE4_PRIOR_BATCH_SIZE:
            return None, _format_prior_batch_limit_feedback(prior_names)

        redundant_priors = _find_redundant_prior_updates(new_priors, state.get("authored_priors"))
        if redundant_priors:
            return None, _format_redundant_stage4_update_feedback("priors", redundant_priors)

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
            return output or None, format_prior_proposal_errors(prior_errors)

    model_spec = state.get("model_spec")
    if model_spec is None:
        guidance = "Previously accepted state is retained. Submit only the missing model fields."
        message = "COMPILE ERROR:\nNo model_spec available — submit model_spec first"
        return output or None, f"{message}\n\n{guidance}" if output else message

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
    )

    if new_model_spec is not None and validation.normalized_model_spec is not None:
        output["model_spec"] = validation.normalized_model_spec
    output["validation"] = validation

    if not validation.compile_ok:
        return output or None, _with_stateful_retry_guidance(
            f"COMPILE ERROR:\n{validation.compile_error}"
        )

    if missing_priors:
        return output, _format_missing_priors_feedback(missing_priors)

    output["resolved_priors"] = resolve_prior_proposals(
        validation.compiled_ssm,
        authored_priors=authored_priors or {},
    )

    if validation.is_valid:
        return output, "VALID"

    feedback = format_validation_feedback(
        validation,
        authored_priors or {},
        changed_params=list(new_priors) if new_priors else list(authored_priors or {}),
    )
    return output, _with_stateful_retry_guidance(feedback)


def _required_prior_names(model_spec: dict | None) -> list[str]:
    """Return the parameter names that still need priors."""
    names: list[str] = []
    for parameter in (model_spec or {}).get("parameters") or []:
        if isinstance(parameter, dict) and isinstance(parameter.get("name"), str):
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


def _summarize_names(names: list[str], *, limit: int = 8) -> str:
    """Render a compact preview of missing or updated parameter names."""
    if not names:
        return "(none)"
    preview = ", ".join(f"`{name}`" for name in names[:limit])
    if len(names) <= limit:
        return preview
    return f"{preview}, ... (+{len(names) - limit} more)"


def _format_missing_priors_feedback(missing_priors: list[str]) -> str:
    """Guide the LLM to submit only the unresolved priors."""
    return (
        "MODEL STATE SAVED:\n"
        f"- missing priors for {len(missing_priors)} parameters: {_summarize_names(missing_priors)}\n"
        f"- submit priors in small batches (max {MAX_STAGE4_PRIOR_BATCH_SIZE} per call)\n"
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


def _format_prior_batch_limit_feedback(prior_names: list[str]) -> str:
    """Tell the LLM to split a large prior submission into smaller batches."""
    return (
        "PRIOR UPDATE TOO LARGE:\n"
        f"- submitted {len(prior_names)} priors; max is {MAX_STAGE4_PRIOR_BATCH_SIZE} per call\n"
        f"- split this update into smaller batches: {_summarize_names(prior_names)}\n\n"
        "Previously accepted state is retained. Resubmit only the fields you changed."
    )


def _format_redundant_stage4_update_feedback(kind: str, names: list[str]) -> str:
    """Tell the LLM not to resend already accepted stage-4 fields."""
    summary = _summarize_names(names)
    return (
        f"REDUNDANT {kind.upper()} UPDATE:\n"
        f"- already accepted and unchanged: {summary}\n"
        "- do not resend unchanged fields\n\n"
        "Previously accepted state is retained. Resubmit only the fields you changed."
    )


def _with_stateful_retry_guidance(feedback: str) -> str:
    """Remind the LLM that accepted stage-4 state is preserved across retries."""
    return f"{feedback}\n\nPreviously accepted state is retained. Resubmit only the fields you changed."


# ---------------------------------------------------------------------------
# Search: literature retrieval (not a grounding function)
# ---------------------------------------------------------------------------


async def search_literature(query: str) -> str:
    """Search Exa for empirical literature, return formatted results.

    Thin wrapper over search_parameter_literature + format_literature_for_parameter.
    This is a retrieval helper, not a grounding function — it returns
    formatted text for the LLM, not a (stage_output, feedback) tuple.

    Args:
        query: Search query string

    Returns:
        Formatted literature string (or empty-result message)
    """
    from causal_ssm_agent.workers.prior_research import search_parameter_literature
    from causal_ssm_agent.workers.prompts.prior_research import format_literature_for_parameter

    sources = await search_parameter_literature(query)
    if not sources:
        return "No relevant literature found for this query."
    return format_literature_for_parameter(sources)


def make_search_tool(search_captures: dict[str, str]) -> Any:
    """Create a search_literature Tool for pipeline use.

    Captures the actual query the LLM uses for each parameter into
    ``search_captures``, keyed by parameter name.  This is process
    provenance recorded on the Stage 4 contract (not on the model spec).

    Args:
        search_captures: Mutable dict that accumulates
            ``{parameter_name: query}`` entries as the LLM calls the tool.

    Returns:
        Tool object
    """
    from causal_ssm_agent.utils.litellm_client import Tool

    async def _execute(*, query: str, parameter_name: str) -> str:
        search_captures[parameter_name] = query
        return await search_literature(query)

    return Tool(
        name="search_literature",
        description="Search for empirical literature about effect sizes for model parameters.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for empirical literature about effect sizes.",
                },
                "parameter_name": {
                    "type": "string",
                    "description": "Name of the parameter this search is for (e.g. 'beta_stress_sleep').",
                },
            },
            "required": ["query", "parameter_name"],
            "additionalProperties": False,
        },
        execute=_execute,
    )


# ---------------------------------------------------------------------------
# Tool factory for pipeline use
# ---------------------------------------------------------------------------


def make_stage_tool(
    name: str,
    description: str,
    param_name: str,
    param_description: str,
    compute_fn: Callable[[dict], tuple[dict | None, str]],
    success_feedback: str = "VALID",
) -> tuple[Any, dict]:
    """Create a fat tool for pipeline use wrapping a compute function.

    The returned Tool calls compute_fn on the parsed JSON input.
    On success (stage_output is not None), the capture dict is updated.
    The tool returns the feedback string to the LLM.

    Args:
        name: Tool name
        description: Tool description
        param_name: Name of the JSON string parameter
        param_description: Description of the parameter
        compute_fn: (data_dict) -> (stage_output | None, feedback_str)
        success_feedback: The feedback string that triggers stop_on_success

    Returns:
        (Tool, capture_dict)
    """
    from causal_ssm_agent.utils.litellm_client import Tool

    capture: dict = {}

    async def _execute(**kwargs: str) -> str:
        try:
            data = json.loads(kwargs[param_name])
        except json.JSONDecodeError as e:
            logger.warning("[%s] JSON parse error: %s", name, e)
            return f"JSON parse error: {e}"

        t0 = time.monotonic()
        stage_output, feedback = compute_fn(data)
        elapsed = time.monotonic() - t0
        is_success = feedback == success_feedback

        if stage_output is not None:
            capture.update(stage_output)
        if is_success:
            logger.info("[%s] grounding passed (%.1fs)", name, elapsed)
        else:
            preview = feedback[:200].replace("\n", " ")
            logger.info("[%s] grounding rejected (%.1fs): %s", name, elapsed, preview)

        return feedback

    return Tool(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": {
                param_name: {"type": "string", "description": param_description},
            },
            "required": [param_name],
            "additionalProperties": False,
        },
        execute=_execute,
        stop_on_success=True,
        success_output=success_feedback,
    ), capture


# ---------------------------------------------------------------------------
# Stage 4 agentic grounding (decisions-merge wrapper)
# ---------------------------------------------------------------------------


def _agentic_stage4_grounding(
    data: dict,
    causal_spec: dict,
    current: dict | None,
    data_for_model: Any,
    indicator_audits: dict[str, dict[str, Any]] | None,
    *,
    resolved_likelihoods: list[dict],
    ambiguous_indicators: list[dict],
    all_params: list[dict],
) -> tuple[dict | None, str]:
    """Agentic grounding for Stage 4: handles decisions-merge then delegates.

    On the first call the LLM typically submits ``distribution_choices`` +
    ``loading_constraints`` + ``priors``.  This wrapper merges the decisions
    with the pre-computed skeleton to produce a full ``ModelSpec``, then
    delegates to :func:`stage4_grounding` for compile + prior-predictive
    validation.

    On subsequent calls (prior refinement only) the data contains just
    ``priors`` and the wrapper delegates directly.
    """
    if ("distribution_choices" in data or "loading_constraints" in data) and "priors" in data:
        return (
            None,
            "UPDATE TOO BROAD:\n"
            "- submit model decisions and priors in separate calls\n"
            "- validate model decisions first, then add priors in later calls\n\n"
            "Previously accepted state is retained. Resubmit only the fields you changed.",
        )

    if "distribution_choices" in data or "loading_constraints" in data:
        from causal_ssm_agent.orchestrator.schemas_model import (
            validate_model_spec_decisions_dict,
        )

        redundant_decisions = _find_redundant_decision_updates(data, current=current)
        if redundant_decisions:
            return None, _format_redundant_stage4_update_feedback(
                "model decisions",
                redundant_decisions,
            )

        decisions_data = _merge_stage4_decision_updates(
            data,
            current=current,
            ambiguous_indicators=ambiguous_indicators,
            all_params=all_params,
        )
        model_spec_result, errors = validate_model_spec_decisions_dict(
            decisions_data,
            resolved_likelihoods=resolved_likelihoods,
            ambiguous_indicators=ambiguous_indicators,
            parameters=all_params,
        )
        if errors:
            return None, "VALIDATION ERRORS:\n" + "\n".join(f"- {e}" for e in errors)

        merged_data: dict[str, Any] = {
            "model_spec": model_spec_result.model_dump(mode="json"),
        }
        if "priors" in data:
            merged_data["priors"] = data["priors"]
        return stage4_grounding(
            merged_data,
            causal_spec,
            current=current,
            data_for_model=data_for_model,
            indicator_audits=indicator_audits,
        )

    # No decisions — delegate directly (model_spec and/or priors only)
    return stage4_grounding(
        data,
        causal_spec,
        current=current,
        data_for_model=data_for_model,
        indicator_audits=indicator_audits,
    )


def _find_redundant_decision_updates(
    data: dict[str, Any],
    *,
    current: dict | None,
) -> list[str]:
    """Return decision targets whose accepted values were resubmitted unchanged."""
    current_model_spec = (current or {}).get("model_spec") or {}

    current_likelihoods: dict[str, dict[str, Any]] = {}
    for likelihood in current_model_spec.get("likelihoods") or []:
        if not isinstance(likelihood, dict):
            continue
        variable = likelihood.get("variable")
        if isinstance(variable, str):
            current_likelihoods[variable] = likelihood

    current_loadings: dict[str, dict[str, Any]] = {}
    for parameter in current_model_spec.get("parameters") or []:
        if not isinstance(parameter, dict):
            continue
        name = parameter.get("name")
        if isinstance(name, str):
            current_loadings[name] = parameter

    redundant: list[str] = []
    for choice in data.get("distribution_choices") or []:
        if not isinstance(choice, dict):
            continue
        variable = choice.get("variable")
        accepted = current_likelihoods.get(variable) if isinstance(variable, str) else None
        if (
            accepted is not None
            and accepted.get("distribution") == choice.get("distribution")
            and accepted.get("link") == choice.get("link")
        ):
            redundant.append(variable)

    for constraint in data.get("loading_constraints") or []:
        if not isinstance(constraint, dict):
            continue
        parameter = constraint.get("parameter")
        accepted = current_loadings.get(parameter) if isinstance(parameter, str) else None
        if accepted is not None and accepted.get("constraint") == constraint.get("constraint"):
            redundant.append(parameter)

    return sorted(name for name in redundant if isinstance(name, str))


def _merge_stage4_decision_updates(
    data: dict[str, Any],
    *,
    current: dict | None,
    ambiguous_indicators: list[dict],
    all_params: list[dict],
) -> dict[str, list[dict[str, Any]]]:
    """Merge decision deltas with the currently accepted stage-4 model spec."""
    current_model_spec = (current or {}).get("model_spec") or {}

    ambiguous_vars = {
        indicator["variable"]
        for indicator in ambiguous_indicators
        if isinstance(indicator, dict) and isinstance(indicator.get("variable"), str)
    }
    current_dist_choices: dict[str, dict[str, Any]] = {}
    for likelihood in current_model_spec.get("likelihoods") or []:
        if not isinstance(likelihood, dict):
            continue
        variable = likelihood.get("variable")
        if isinstance(variable, str) and variable in ambiguous_vars:
            current_dist_choices[variable] = {
                "variable": variable,
                "distribution": likelihood.get("distribution"),
                "link": likelihood.get("link"),
                "reasoning": likelihood.get("reasoning")
                or "Retained from previously accepted state.",
            }
    for choice in data.get("distribution_choices") or []:
        if isinstance(choice, dict) and isinstance(choice.get("variable"), str):
            current_dist_choices[choice["variable"]] = choice

    loading_param_names = {
        parameter["name"]
        for parameter in all_params
        if isinstance(parameter, dict)
        and parameter.get("role") == "loading"
        and isinstance(parameter.get("name"), str)
    }
    current_loading_constraints: dict[str, dict[str, Any]] = {}
    for parameter in current_model_spec.get("parameters") or []:
        if not isinstance(parameter, dict):
            continue
        name = parameter.get("name")
        if isinstance(name, str) and name in loading_param_names:
            current_loading_constraints[name] = {
                "parameter": name,
                "constraint": parameter.get("constraint"),
                "reasoning": "Retained from previously accepted state.",
            }
    for constraint in data.get("loading_constraints") or []:
        if isinstance(constraint, dict) and isinstance(constraint.get("parameter"), str):
            current_loading_constraints[constraint["parameter"]] = constraint

    return {
        "distribution_choices": list(current_dist_choices.values()),
        "loading_constraints": list(current_loading_constraints.values()),
    }


# ---------------------------------------------------------------------------
# GMM elicitation tool factory
# ---------------------------------------------------------------------------


def make_elicit_prior_gmm_tool(
    question: str,
    model_name: str,
    n_paraphrases: int = 10,
) -> Any:
    """Create an ``elicit_prior_gmm`` tool for the agentic Stage 4 flow.

    The tool runs N paraphrased LLM calls for a single parameter and
    aggregates them via GMM, returning a formatted summary to the outer
    agentic conversation.
    """
    from causal_ssm_agent.utils.litellm_client import Tool

    async def _execute(
        *,
        parameter_name: str,
        parameter_role: str,
        parameter_constraint: str,
        context: str,
    ) -> str:
        from causal_ssm_agent.workers.prior_research import run_gmm_elicitation

        return await run_gmm_elicitation(
            parameter_name=parameter_name,
            parameter_role=parameter_role,
            parameter_constraint=parameter_constraint,
            context=context,
            question=question,
            model_name=model_name,
            n_paraphrases=n_paraphrases,
        )

    return Tool(
        name="elicit_prior_gmm",
        description=(
            "Run robust paraphrased prior elicitation with GMM aggregation "
            "for a single parameter. Returns an aggregated prior estimate."
        ),
        parameters={
            "type": "object",
            "properties": {
                "parameter_name": {
                    "type": "string",
                    "description": "Name of the parameter (e.g. 'beta_stress_depression')",
                },
                "parameter_role": {
                    "type": "string",
                    "description": "Role: fixed_effect, ar_coefficient, residual_sd, loading",
                },
                "parameter_constraint": {
                    "type": "string",
                    "description": "Constraint: none, positive, unit_interval, correlation",
                },
                "context": {
                    "type": "string",
                    "description": "What this parameter represents, for literature grounding",
                },
            },
            "required": ["parameter_name", "parameter_role", "parameter_constraint", "context"],
            "additionalProperties": False,
        },
        execute=_execute,
    )
