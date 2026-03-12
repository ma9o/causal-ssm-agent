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
    from causal_ssm_agent.utils.causal_spec import get_outcome_name
    from causal_ssm_agent.utils.effects import get_all_treatments

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
    from causal_ssm_agent.orchestrator.schemas_model import validate_model_spec_dict
    from causal_ssm_agent.utils.causal_spec import get_indicators

    state = dict(current or {})
    output: dict = {}

    new_model_spec = data.get("model_spec")
    new_priors = data.get("priors")

    if new_model_spec is None and new_priors is None:
        return None, "VALIDATION ERRORS:\n- data must contain 'model_spec' and/or 'priors'"

    # --- Validate & merge model_spec ---
    if new_model_spec is not None:
        indicators = get_indicators(causal_spec)
        _, errors = validate_model_spec_dict(new_model_spec, indicators=indicators or None)
        if errors:
            return None, "VALIDATION ERRORS:\n" + "\n".join(f"- {e}" for e in errors)
        state["model_spec"] = new_model_spec
        output["model_spec"] = new_model_spec

    # --- Validate & merge priors ---
    if new_priors is not None:
        from causal_ssm_agent.workers.schemas_prior import PriorProposal

        for name, prior in new_priors.items():
            try:
                PriorProposal.model_validate(prior)
            except Exception as e:
                return None, f"SCHEMA ERRORS for prior '{name}':\n- {e}"
        state["priors"] = {**state.get("priors", {}), **new_priors}
        output["priors"] = state["priors"]  # full merged set

    # --- Compile ---
    model_spec = state.get("model_spec")
    if model_spec is None:
        return None, "COMPILE ERROR:\nNo model_spec available — submit model_spec first"

    priors = state.get("priors")
    if priors:
        # Real priors available — compile with them
        from causal_ssm_agent.models.ssm_compiler import compile_ssm_artifact

        try:
            compile_ssm_artifact(model_spec, priors, causal_spec=causal_spec)
        except (ValueError, Exception) as e:
            return None, f"COMPILE ERROR:\n{e}"
    else:
        # No priors yet — trial compile with defaults
        from causal_ssm_agent.models.ssm_compiler import trial_compile_model_spec

        compile_error = trial_compile_model_spec(model_spec, causal_spec)
        if compile_error:
            return None, f"COMPILE ERROR:\n{compile_error}"

    # --- Prior predictive (only with real priors + data) ---
    if priors and raw_data is not None:
        from causal_ssm_agent.models.prior_predictive import (
            format_parameter_feedback,
            validate_prior_predictive,
        )

        is_valid, results, _ = validate_prior_predictive(
            model_spec, priors, raw_data, causal_spec=causal_spec
        )
        if not is_valid:
            changed = list(new_priors) if new_priors else list(priors)
            parts = []
            for param_name in changed:
                fb = format_parameter_feedback(
                    parameter_name=param_name,
                    results=results,
                    prior=priors.get(param_name),
                )
                if fb:
                    parts.append(fb)
            return None, "\n\n".join(parts) if parts else "PRIOR PREDICTIVE CHECK FAILED"

    return output, "VALID"


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


def make_search_tool() -> Any:
    """Create a search_literature Tool for pipeline use.

    Unlike make_stage_tool, search tools are retrieval-only — no capture dict,
    no stop_on_success. The LLM uses search results to inform its next
    validate_model call.

    Returns:
        Tool object
    """
    from causal_ssm_agent.utils.litellm_client import Tool

    async def _execute(*, query: str) -> str:
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
            },
            "required": ["query"],
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

        if stage_output is not None:
            capture.update(stage_output)
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
