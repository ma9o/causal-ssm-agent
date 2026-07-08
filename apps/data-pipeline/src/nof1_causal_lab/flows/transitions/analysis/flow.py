"""analysis orchestration."""

from __future__ import annotations

import json
import logging
from typing import Any

from nof1_causal_lab.flows.llm_transition_runtime import (
    LLMTransitionRuntimeConfig,
    attach_trace,
    open_llm_transition,
)
from nof1_causal_lab.flows.run_store import load_pickle

from .interventions import run_interventions


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


async def run_analysis(
    posterior: dict,
    measurement_payload: dict,
) -> dict[str, Any]:
    """Run interventions and synthesize the analysis commentary payload."""
    from nof1_causal_lab.utils.causal_design import get_outcome_name
    from nof1_causal_lab.utils.config import get_config

    logger = logging.getLogger(__name__)
    fitted_artifact = load_pickle(posterior["_fitted_result_path"])
    treatments = measurement_payload["_identified_treatments"]
    causal_design = measurement_payload["causal_design"]
    outcome_name = get_outcome_name(causal_design) or ""

    logger.info("=== analysis: Treatment Effects ===")
    logger.info("Estimating effects of %d treatments on %s", len(treatments), outcome_name)

    intervention_results = run_interventions(
        fitted_artifact,
        treatments,
        outcome_name,
        causal_design,
    )

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

    ppc_warnings = [
        {
            "variable": warning.get("variable"),
            "issue_type": warning.get("issue_type"),
            "severity": warning.get("severity"),
            "message": warning.get("message"),
        }
        for warning in posterior.get("ppc", {}).get("per_variable_warnings", [])
    ][:5]

    top_results = [
        {
            "treatment": entry.get("treatment"),
            "effect_size": _draws_stats(entry.get("posterior_draws"))[0],
            "prob_positive": _draws_stats(entry.get("posterior_draws"))[1],
        }
        for entry in intervention_results[:5]
    ]

    commentary_input = {
        "outcome": outcome_name,
        "identifiable_treatments": treatments,
        "excluded_non_identifiable_treatments": sorted(
            measurement_payload.get("causal_design", {})
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
        "You are writing the opening commentary for analysis of a causal state-space "
        "analysis. Comment on the treatment-effect results for a technical user. "
        "Be concise and grounded. Do not invent certainty. Mention the strongest "
        "effects, note warnings or identifiability limits, and end by stating that "
        "follow-up chat can inspect model details or run Pearl rung 2 and rung 3 "
        "simulations. Return plain Markdown only."
    )
    user_prompt = (
        "Comment the results of analysis.\n\n"
        f"{json.dumps(commentary_input, indent=2, sort_keys=True)}"
    )

    cfg = get_config()
    runtime_config = LLMTransitionRuntimeConfig(
        context_id="analysis",
        profile_llm=cfg.analysis_commentary.llm,
        llm_defaults=cfg.llm,
    )
    async with (
        open_llm_transition(
            config=runtime_config,
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
    }
    final_summary = _first_assistant_summary(factory.accumulated_trace)
    if final_summary:
        result["final_summary"] = final_summary
    attach_trace(result, factory.accumulated_trace)
    return result
