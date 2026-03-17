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
    format_ambiguous_indicators,
    format_full_causal_spec,
    format_loading_params,
    format_parameters,
    format_resolved_likelihoods,
)
from .stage4_orchestrator import derive_deterministic_spec

if TYPE_CHECKING:
    import polars as pl

    from causal_ssm_agent.flows.stages.stage4_assembly import AssemblyValidation
    from causal_ssm_agent.utils.llm import GenerateFn


@dataclass
class Stage4Result:
    """Result of the agentic Stage 4 flow."""

    model_spec: dict[str, Any]
    priors: dict[str, dict]
    validation: AssemblyValidation | None = None


@dataclass
class Stage4Messages:
    """Message builders for the agentic Stage 4 prompts."""

    question: str
    causal_spec: dict[str, Any] = field(default_factory=dict)
    resolved_likelihoods: list[dict] = field(default_factory=list)
    ambiguous_indicators: list[dict] = field(default_factory=list)
    all_params: list[dict] = field(default_factory=list)
    loading_params: list[dict] = field(default_factory=list)
    data_summary: str = ""

    def proposal_messages(self) -> list[dict]:
        """Build messages for the agentic model-spec + prior proposal."""
        return [
            {"role": "system", "content": AGENTIC_SYSTEM},
            {
                "role": "user",
                "content": AGENTIC_USER.format(
                    question=self.question,
                    full_causal_model=format_full_causal_spec(self.causal_spec),
                    resolved_likelihoods=format_resolved_likelihoods(self.resolved_likelihoods),
                    ambiguous_indicators=format_ambiguous_indicators(self.ambiguous_indicators),
                    parameters=format_parameters(self.all_params),
                    loading_params=format_loading_params(self.loading_params),
                    data_summary=self.data_summary,
                ),
            },
        ]


def build_raw_data_summary(raw_data: pl.DataFrame) -> str:
    """Build a summary of data for the agentic prompt.

    Args:
        raw_data: DataFrame with columns: indicator, value, and either
            timestamp (raw) or time_bucket (aggregated).

    Returns:
        Text summary of the data
    """
    import polars as _pl

    if raw_data.is_empty():
        return "No data available."

    time_col = "time_bucket" if "time_bucket" in raw_data.columns else "timestamp"
    lines = [f"Data Summary (observations, time column: {time_col}):"]

    n_obs = len(raw_data)
    lines.append(f"  Total observations: {n_obs}")

    indicator_stats = (
        raw_data.group_by("indicator")
        .agg(
            [
                _pl.col("value").cast(_pl.Float64, strict=False).count().alias("n_obs"),
                _pl.col("value").cast(_pl.Float64, strict=False).mean().alias("mean"),
                _pl.col("value").cast(_pl.Float64, strict=False).std().alias("std"),
            ]
        )
        .sort("indicator")
    )

    lines.append("  Per indicator:")
    for row in indicator_stats.iter_rows(named=True):
        mean_str = f"{row['mean']:.2f}" if row["mean"] is not None else "N/A"
        std_str = f"{row['std']:.2f}" if row["std"] is not None else "N/A"
        lines.append(f"    {row['indicator']}: n={row['n_obs']}, mean={mean_str}, std={std_str}")

    return "\n".join(lines)


async def run_stage4(
    causal_spec: dict,
    question: str,
    raw_data: pl.DataFrame,
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
    resolved, ambiguous, parameters, loading_params = derive_deterministic_spec(causal_spec)
    all_params = parameters + [
        {k: v for k, v in lp.items() if k not in ("indicator", "construct")}
        for lp in loading_params
    ]

    # 2. Build messages
    msgs = Stage4Messages(
        question=question,
        causal_spec=causal_spec,
        resolved_likelihoods=resolved,
        ambiguous_indicators=ambiguous,
        all_params=all_params,
        loading_params=loading_params,
        data_summary=build_raw_data_summary(raw_data),
    )

    # 3. Build tools
    validate_tool, capture = make_stage_tool(
        name="validate_model",
        description="Submit model specification decisions and prior proposals for validation.",
        param_name="model_json",
        param_description=(
            "JSON with distribution_choices, loading_constraints, and priors. "
            "Or just priors to update after a previous successful submission."
        ),
        compute_fn=lambda data: _agentic_stage4_grounding(
            data,
            causal_spec,
            current=capture,
            raw_data=raw_data,
            resolved_likelihoods=resolved,
            ambiguous_indicators=ambiguous,
            all_params=all_params,
        ),
    )

    tools = [validate_tool]
    if enable_literature:
        tools.append(make_search_tool())
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

    return Stage4Result(model_spec=model_spec, priors=priors)
