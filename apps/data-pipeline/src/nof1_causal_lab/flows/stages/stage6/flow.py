"""Stage 6 orchestration."""

from __future__ import annotations

import json
from inspect import isawaitable
from typing import Any

from nof1_causal_lab.flows.llm_stage_runtime import (
    LLMStageRuntimeConfig,
    attach_trace,
    open_llm_stage,
)
from nof1_causal_lab.flows.run_store import load_pickle, unwrap_task_result

from .interventions import run_interventions


async def _await_artifact(artifact: Any) -> None:
    if isawaitable(artifact):
        await artifact


def _first_assistant_summary(trace: Any) -> str | None:
    messages = getattr(trace, "messages", None) or []
    for message in messages:
        if getattr(message, "role", None) != "assistant":
            continue
        content = (getattr(message, "content", "") or "").strip()
        if content:
            return content
    return None


def _draws_stats(draws: list[float] | None) -> tuple[float | None, float | None]:
    if not draws:
        return None, None
    return sum(draws) / len(draws), sum(1 for draw in draws if draw > 0) / len(draws)


async def run_stage6(
    stage5b: dict,
    stage1b: dict,
    question: str | None = None,
) -> dict[str, Any]:
    """Run interventions and synthesize the Stage 6 commentary payload."""
    from prefect.artifacts import create_table_artifact

    from nof1_causal_lab.flows import get_prefect_logger
    from nof1_causal_lab.utils.causal_spec import get_outcome_name
    from nof1_causal_lab.utils.config import get_config

    logger = get_prefect_logger(__name__)
    fitted_artifact = load_pickle(stage5b["_fitted_result_path"])
    treatments = stage1b["_identified_treatments"]
    causal_spec = stage1b["causal_spec"]
    outcome_name = get_outcome_name(causal_spec) or ""

    logger.info("=== Stage 6: Treatment Effects ===")
    logger.info("Estimating effects of %d treatments on %s", len(treatments), outcome_name)

    results = run_interventions(
        fitted_artifact,
        treatments,
        outcome_name,
        causal_spec,
    )
    intervention_results = unwrap_task_result(results)

    if intervention_results:
        logger.info("%-5s %-30s %10s %8s", "Rank", "Treatment", "Effect", "P(>0)")
        logger.info("-" * 55)
        for rank, entry in enumerate(intervention_results, 1):
            name = entry["treatment"]
            effect, prob = _draws_stats(entry.get("posterior_draws"))
            if effect is not None:
                logger.info("%d     %-30s %+10.4f %8.2f", rank, name, effect, prob)
            else:
                logger.info("%-5d %-30s %10s", rank, name, "—")

        await _await_artifact(
            create_table_artifact(
                key="treatment-ranking",
                table=[
                    {
                        "rank": index + 1,
                        "treatment": result["treatment"],
                        "effect": (
                            f"{effect:+.4f}"
                            if (effect := _draws_stats(result.get("posterior_draws"))[0])
                            is not None
                            else "---"
                        ),
                        "P(>0)": (
                            f"{prob:.2f}"
                            if (prob := _draws_stats(result.get("posterior_draws"))[1]) is not None
                            else ""
                        ),
                    }
                    for index, result in enumerate(intervention_results)
                ],
                description="Final treatment effect ranking",
            )
        )

    ppc_warnings = [
        {
            "variable": warning.get("variable"),
            "issue_type": warning.get("issue_type"),
            "severity": warning.get("severity"),
            "message": warning.get("message"),
        }
        for warning in stage5b.get("ppc", {}).get("per_variable_warnings", [])
    ][:5]
    has_warnings = bool(ppc_warnings)

    top_results = [
        {
            "treatment": entry.get("treatment"),
            "effect_size": _draws_stats(entry.get("posterior_draws"))[0],
            "prob_positive": _draws_stats(entry.get("posterior_draws"))[1],
        }
        for entry in intervention_results[:5]
    ]

    commentary_input = {
        "question": question,
        "outcome": outcome_name,
        "identifiable_treatments": treatments,
        "excluded_non_identifiable_treatments": sorted(
            stage1b.get("causal_spec", {})
            .get("identifiability", {})
            .get("non_identifiable_treatments", {})
            .keys()
        ),
        "top_ranked_effects": top_results,
        "ppc_warnings": ppc_warnings,
        "follow_up_capabilities": {
            "get_model_info": "Inspect variables, measurement, identifiability, diagnostics, and baseline effects.",
            "simulate_intervention": "Run Pearl rung-2 intervention simulations on the fitted generative model.",
            "simulate_counterfactual": "Run Pearl rung-3 counterfactual simulations conditioned on an observed history window.",
        },
    }

    system_prompt = (
        "You are writing the opening commentary for Stage 6 of a causal state-space "
        "analysis. Comment on the treatment-effect results for a technical user. "
        "Be concise and grounded. Do not invent certainty. Mention the strongest "
        "effects, note warnings or identifiability limits, and end by stating that "
        "follow-up chat can inspect model details or run Pearl rung 2 and rung 3 "
        "simulations. Return plain Markdown only."
    )
    user_prompt = (
        "Comment the results of Stage 6.\n\n"
        f"{json.dumps(commentary_input, indent=2, sort_keys=True)}"
    )

    cfg = get_config()
    runtime_config = LLMStageRuntimeConfig(
        stage_id="stage-6",
        stage_llm=cfg.stage6_commentary.llm,
        llm_defaults=cfg.llm,
    )
    async with (
        open_llm_stage(
            config=runtime_config,
            openrouter_api_key=None,
            logger=logger,
        ) as factory,
        factory.open(
            system_prompt=system_prompt,
            log_label="comment-results",
        ) as session,
    ):
        await session.turn(user_prompt)

    result: dict[str, Any] = {
        "intervention_results": intervention_results,
        "outcome": "warn" if has_warnings else "success",
    }
    final_summary = _first_assistant_summary(factory.accumulated_trace)
    if final_summary:
        result["final_summary"] = final_summary
    attach_trace(result, factory.accumulated_trace)
    return result
