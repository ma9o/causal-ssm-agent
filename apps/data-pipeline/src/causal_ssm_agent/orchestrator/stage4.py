"""Stage 4: Model Specification & Prior Elicitation (Agentic).

Single multi-turn LLM conversation that proposes distribution choices for
ambiguous indicators and priors for all parameters, using tools to search
literature or run robust GMM aggregation as needed.

Follows the same two-layer architecture as stages 1a/1b:
- This module contains pure orchestrator logic (framework-agnostic).
- The Prefect wrapper lives in ``flows/stages/stage4_model.py``.
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .prompts.model_proposal import (
    AGENTIC_SYSTEM,
    AGENTIC_USER,
    format_construct_scale_cards,
    format_distribution_cards,
    format_loading_params,
    format_model_topology,
    format_prior_cards,
)
from .stage4_orchestrator import (
    build_construct_scale_cards,
    build_distribution_cards,
    build_model_topology,
    build_prior_cards,
    derive_deterministic_spec,
)

if TYPE_CHECKING:
    import polars as pl

    from causal_ssm_agent.flows.stages.stage4_assembly import AssemblyValidation
    from causal_ssm_agent.utils.llm import GenerateFn


@dataclass
class Stage4Result:
    """Result of the agentic Stage 4 flow."""

    model_spec: dict[str, Any]
    priors: dict[str, dict]
    search_queries: dict[str, str] = field(default_factory=dict)
    validation: AssemblyValidation | None = None
    validation_retries: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Stage4Messages:
    """Message builders for the agentic Stage 4 prompts."""

    question: str
    model_topology: dict[str, Any] = field(default_factory=dict)
    distribution_cards: list[dict[str, Any]] = field(default_factory=list)
    loading_params: list[dict] = field(default_factory=list)
    construct_scale_cards: list[dict[str, Any]] = field(default_factory=list)
    prior_cards: list[dict[str, Any]] = field(default_factory=list)

    def proposal_messages(self) -> list[dict]:
        """Build messages for the agentic model-spec + prior proposal."""
        return [
            {"role": "system", "content": AGENTIC_SYSTEM},
            {
                "role": "user",
                "content": AGENTIC_USER.format(
                    question=self.question,
                    model_topology=format_model_topology(self.model_topology),
                    distribution_cards=format_distribution_cards(self.distribution_cards),
                    loading_params=format_loading_params(self.loading_params),
                    construct_scale_cards=format_construct_scale_cards(self.construct_scale_cards),
                    prior_cards=format_prior_cards(self.prior_cards),
                ),
            },
        ]


def _unique_in_order(values: list[str]) -> list[str]:
    """Drop duplicates while preserving the authored order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _submitted_retry_targets(submission: dict[str, Any]) -> list[str]:
    """Collect authored stage-4 targets from a tool submission."""
    targets: list[str] = []

    priors = submission.get("priors")
    if isinstance(priors, dict):
        targets.extend(name for name in priors if isinstance(name, str))

    for choice in submission.get("distribution_choices") or []:
        if isinstance(choice, dict) and isinstance(choice.get("variable"), str):
            targets.append(choice["variable"])

    for constraint in submission.get("loading_constraints") or []:
        if isinstance(constraint, dict) and isinstance(constraint.get("parameter"), str):
            targets.append(constraint["parameter"])

    return _unique_in_order(targets)


def _distribution_choice_signature(submission: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    """Build a comparison-friendly view of likelihood family decisions."""
    signature: dict[str, tuple[Any, Any]] = {}
    for choice in submission.get("distribution_choices") or []:
        if not isinstance(choice, dict):
            continue
        variable = choice.get("variable")
        if isinstance(variable, str):
            signature[variable] = (choice.get("distribution"), choice.get("link"))
    return signature


def _loading_constraint_signature(submission: dict[str, Any]) -> dict[str, Any]:
    """Build a comparison-friendly view of loading sign constraints."""
    signature: dict[str, Any] = {}
    for constraint in submission.get("loading_constraints") or []:
        if not isinstance(constraint, dict):
            continue
        parameter = constraint.get("parameter")
        if isinstance(parameter, str):
            signature[parameter] = constraint.get("constraint")
    return signature


def _changed_retry_targets(
    submission: dict[str, Any],
    previous_submission: dict[str, Any] | None,
) -> list[str]:
    """Return the authored targets that changed since the previous attempt."""
    if previous_submission is None:
        return _submitted_retry_targets(submission)

    changed: list[str] = []

    current_priors = submission.get("priors") if isinstance(submission.get("priors"), dict) else {}
    previous_priors = (
        previous_submission.get("priors")
        if isinstance(previous_submission.get("priors"), dict)
        else {}
    )
    for name in current_priors:
        if previous_priors.get(name) != current_priors[name]:
            changed.append(name)

    current_choices = _distribution_choice_signature(submission)
    previous_choices = _distribution_choice_signature(previous_submission)
    for variable, signature in current_choices.items():
        if previous_choices.get(variable) != signature:
            changed.append(variable)

    current_constraints = _loading_constraint_signature(submission)
    previous_constraints = _loading_constraint_signature(previous_submission)
    for parameter, constraint in current_constraints.items():
        if previous_constraints.get(parameter) != constraint:
            changed.append(parameter)

    return _unique_in_order(changed)


def _extract_schema_retry_targets(feedback: str) -> list[str]:
    """Extract prior names from schema-validation failures."""
    return _unique_in_order(re.findall(r"SCHEMA ERRORS for prior '([^']+)'", feedback))


def _extract_observation_support_targets(feedback: str) -> list[str]:
    """Extract indicator names from observation-support failures."""
    return _unique_in_order(re.findall(r"'([^']+)' uses [A-Za-z_]+ emission", feedback))


def _extract_nan_inf_sample_targets(feedback: str) -> list[str]:
    """Extract named sample sites from NaN/Inf failures when they are specific."""
    match = re.search(r"NaN/Inf detected in sample sites:\s*([^\n]+)", feedback)
    if match is None:
        return []

    ignored = {"observations"}
    sample_sites = [site.strip() for site in match.group(1).split(",")]
    return _unique_in_order([site for site in sample_sites if site and site.lower() not in ignored])


def _infer_validation_retry_targets(
    feedback: str,
    submission: dict[str, Any],
    previous_submission: dict[str, Any] | None,
) -> list[str]:
    """Infer the smallest actionable retry targets from stage-4 feedback."""
    for extractor in (
        _extract_schema_retry_targets,
        _extract_observation_support_targets,
        _extract_nan_inf_sample_targets,
    ):
        targets = extractor(feedback)
        if targets:
            return targets

    changed_targets = _changed_retry_targets(submission, previous_submission)
    if changed_targets:
        return changed_targets

    return _submitted_retry_targets(submission)


async def run_stage4(
    causal_spec: dict,
    question: str,
    raw_data: pl.DataFrame,
    indicator_audits: dict[str, dict[str, Any]],
    generate: GenerateFn,
    *,
    enable_literature: bool = True,
    enable_paraphrasing: bool = False,
    n_paraphrases: int = 10,
    gmm_model: str | None = None,
) -> Stage4Result:
    """Run the agentic Stage 4 flow: propose model spec + priors in one conversation.

    This is the core logic, decoupled from any framework. The caller provides
    a ``generate`` function that handles LLM calls.

    Args:
        causal_spec: Full CausalSpec dict
        question: Research question
        raw_data: Raw timestamped data
        indicator_audits: Stage 3 per-indicator empirical profiles + validations
        generate: Async function (messages, tools) -> completion
        enable_literature: Whether to offer the search_literature tool
        enable_paraphrasing: Whether to offer the elicit_prior_gmm tool
        n_paraphrases: Number of paraphrases for GMM tool
        gmm_model: Model name for inner GMM paraphrase calls

    Returns:
        Stage4Result with model_spec and priors
    """
    from causal_ssm_agent.flows.stages.stage_tools import (
        _agentic_stage4_grounding,
        make_elicit_prior_gmm_tool,
        make_search_tool,
        make_stage_tool,
    )

    # 1. Pre-compute deterministic spec from CausalSpec
    skeleton = derive_deterministic_spec(causal_spec)
    all_params = skeleton.all_params
    model_topology = build_model_topology(causal_spec)
    distribution_cards = build_distribution_cards(causal_spec, indicator_audits, skeleton)
    construct_scale_cards = build_construct_scale_cards(causal_spec, indicator_audits, skeleton)
    prior_cards = build_prior_cards(skeleton)

    # 2. Build messages
    msgs = Stage4Messages(
        question=question,
        model_topology=model_topology,
        distribution_cards=distribution_cards,
        loading_params=skeleton.loading_params,
        construct_scale_cards=construct_scale_cards,
        prior_cards=prior_cards,
    )

    # 3. Build tools
    retry_state: dict[str, Any] = {
        "last_failed_targets": [],
        "previous_submission": None,
    }

    def _compute_with_retry_attribution(data: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
        previous_submission = retry_state["previous_submission"]
        stage_output, feedback = _agentic_stage4_grounding(
            data,
            causal_spec,
            current=capture,
            raw_data=raw_data,
            indicator_audits=indicator_audits,
            resolved_likelihoods=skeleton.resolved_likelihoods,
            ambiguous_indicators=skeleton.ambiguous_indicators,
            all_params=all_params,
        )
        retry_state["last_failed_targets"] = (
            _infer_validation_retry_targets(feedback, data, previous_submission)
            if stage_output is None
            else []
        )
        retry_state["previous_submission"] = deepcopy(data)
        return stage_output, feedback

    validate_tool, capture = make_stage_tool(
        name="validate_model",
        description="Submit model specification decisions and prior proposals for validation.",
        param_name="model_json",
        param_description=(
            "JSON with distribution_choices, loading_constraints, and priors. "
            "Or just priors to update after a previous successful submission."
        ),
        compute_fn=_compute_with_retry_attribution,
        capture_failures=True,
        failed_params_fn=lambda _data: list(retry_state["last_failed_targets"]),
    )

    search_captures: dict[str, str] = {}
    tools = [validate_tool]
    if enable_literature:
        tools.append(make_search_tool(search_captures))
    if enable_paraphrasing:
        tools.append(
            make_elicit_prior_gmm_tool(
                question=question,
                model_name=gmm_model or "",
                n_paraphrases=n_paraphrases,
            )
        )

    # 4. Single multi-turn conversation
    await generate(msgs.proposal_messages(), tools)

    # 5. Extract results from capture
    model_spec = capture.get("model_spec")
    priors = capture.get("priors")
    if not model_spec or not priors:
        raise ValueError("Stage 4 agentic flow did not produce a valid model_spec + priors")

    return Stage4Result(
        model_spec=model_spec,
        priors=priors,
        search_queries=search_captures,
        validation=capture.get("validation"),
        validation_retries=list(capture.get("validation_retries", []) or []),
    )
