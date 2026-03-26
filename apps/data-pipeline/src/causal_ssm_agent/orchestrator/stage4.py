"""Stage 4: Model Specification & Prior Elicitation (Agentic).

Single multi-turn LLM conversation that proposes distribution choices for
ambiguous indicators and priors for all parameters, using tools to search
literature or run robust GMM aggregation as needed.

Follows the same two-layer architecture as stages 1a/1b:
- This module contains pure orchestrator logic (framework-agnostic).
- The Prefect wrapper lives in ``flows/stages/stage4_model.py``.
"""

from __future__ import annotations

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
    authored_priors: dict[str, dict]
    search_queries: dict[str, str] = field(default_factory=dict)
    validation: AssemblyValidation | None = None


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


async def run_stage4(
    causal_spec: dict,
    question: str,
    data_for_model: pl.DataFrame,
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
        data_for_model: Numerically encoded observation data
        indicator_audits: Stage 3 per-indicator empirical profiles + validations
        generate: Async function (messages, tools) -> completion
        enable_literature: Whether to offer the search_literature tool
        enable_paraphrasing: Whether to offer the elicit_prior_gmm tool
        n_paraphrases: Number of paraphrases for GMM tool
        gmm_model: Model name for inner GMM paraphrase calls

    Returns:
        Stage4Result with model_spec and authored priors
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
    prior_cards = build_prior_cards(causal_spec, skeleton)

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
    validate_tool, capture = make_stage_tool(
        name="validate_model",
        description="Submit model specification decisions and prior proposals for validation.",
        param_name="model_json",
        param_description=(
            "Stateful JSON with either distribution_choices/loading_constraints or priors. "
            "Do not mix model decisions with priors in the same call. "
            "After any rejected validation, resubmit only the fields you changed."
        ),
        compute_fn=lambda data: _agentic_stage4_grounding(
            data,
            causal_spec,
            current=capture,
            data_for_model=data_for_model,
            indicator_audits=indicator_audits,
            resolved_likelihoods=skeleton.resolved_likelihoods,
            ambiguous_indicators=skeleton.ambiguous_indicators,
            all_params=all_params,
        ),
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
    authored_priors = capture.get("authored_priors")
    if not model_spec or not authored_priors:
        raise ValueError("Stage 4 agentic flow did not produce a valid model_spec + priors")

    return Stage4Result(
        model_spec=model_spec,
        authored_priors=authored_priors,
        search_queries=search_captures,
        validation=capture.get("validation"),
    )
