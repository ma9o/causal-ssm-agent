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


# ---------------------------------------------------------------------------
# Stage 1a: Latent model grounding
# ---------------------------------------------------------------------------


def stage1a_grounding(data: dict) -> tuple[dict | None, str]:
    """Validate latent model and derive outcome_name + treatments.

    Returns:
        (stage_output, feedback)
        stage_output = {"latent_model": data, "outcome_name": str, "treatments": [...]}
    """
    from causal_ssm_agent.orchestrator.schemas import validate_latent_model
    from causal_ssm_agent.utils.causal_spec import get_all_treatments, get_outcome_name

    _result, errors = validate_latent_model(data)
    if errors:
        return None, "VALIDATION ERRORS:\n" + "\n".join(f"- {e}" for e in errors)

    outcome = get_outcome_name(data) or ""
    treatments = get_all_treatments(data)

    return {
        "latent_model": data,
        "outcome_name": outcome,
        "treatments": treatments,
    }, "VALID"


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
    raw_data: Any = None,
    indicator_audits: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict | None, str]:
    """Ground stage 4 proposals: validate, compile, optionally run prior predictive.

    ``data`` contains the proposed changes — any combination of:
    - ``model_spec``: complete ModelSpec dict
    - ``priors``: partial dict mapping parameter names to prior proposals

    The function merges proposals with ``current`` (existing stage-4 state),
    then validates schemas, compiles the model, and — when real priors and
    ``raw_data`` are available — runs prior predictive checks.

    Gates (applied in order):
    1. Schema + domain validation for any submitted fields
    2. Compile (default priors if none available, real priors otherwise)
    3. Prior predictive (only when real priors + raw_data present)
    """
    state = dict(current or {})
    output: dict = {}

    new_model_spec = data.get("model_spec")
    new_priors = data.get("priors")

    if new_model_spec is None and new_priors is None:
        return None, "VALIDATION ERRORS:\n- data must contain 'model_spec' and/or 'priors'"

    # --- Merge model_spec ---
    if new_model_spec is not None:
        state["model_spec"] = new_model_spec

    # --- Validate & merge priors ---
    if new_priors is not None:
        from .stage4_assembly import merge_priors, validate_prior_proposals

        try:
            validated_priors = validate_prior_proposals(new_priors)
        except ValueError as exc:
            return None, str(exc)
        state["priors"] = merge_priors(state.get("priors"), validated_priors)
        output["priors"] = state["priors"]  # full merged set

    # --- Compile + Prior Predictive validation ---
    model_spec = state.get("model_spec")
    if model_spec is None:
        return None, "COMPILE ERROR:\nNo model_spec available — submit model_spec first"

    priors = state.get("priors")

    from .stage4_assembly import format_validation_feedback, validate_assembly

    validation = validate_assembly(model_spec, priors, raw_data, indicator_audits, causal_spec)
    if new_model_spec is not None and validation.normalized_model_spec is not None:
        output["model_spec"] = validation.normalized_model_spec
    if validation.is_valid:
        output["validation"] = validation
        return output, "VALID"

    feedback = format_validation_feedback(
        validation,
        priors or {},
        changed_params=list(new_priors) if new_priors else list(priors or {}),
    )
    return None, feedback


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
    capture_failures: bool = False,
    failed_params_fn: Callable[[dict], list[str]] | None = None,
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

        if stage_output is not None:
            capture.update(stage_output)
            logger.info("[%s] grounding passed (%.1fs)", name, elapsed)
        else:
            if capture_failures:
                retries = capture.setdefault("validation_retries", [])
                retries.append(
                    {
                        "attempt": len(retries) + 1,
                        "failed_params": (
                            failed_params_fn(data)
                            if failed_params_fn is not None
                            else sorted((data.get("priors") or {}).keys())
                        ),
                        "feedback": feedback,
                    }
                )
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
    raw_data: Any,
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
    if "distribution_choices" in data:
        from causal_ssm_agent.orchestrator.schemas_model import (
            validate_model_spec_decisions_dict,
        )

        decisions_data = {
            "distribution_choices": data.get("distribution_choices", []),
            "loading_constraints": data.get("loading_constraints", []),
        }
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
            raw_data=raw_data,
            indicator_audits=indicator_audits,
        )

    # No decisions — delegate directly (model_spec and/or priors only)
    return stage4_grounding(
        data,
        causal_spec,
        current=current,
        raw_data=raw_data,
        indicator_audits=indicator_audits,
    )


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
