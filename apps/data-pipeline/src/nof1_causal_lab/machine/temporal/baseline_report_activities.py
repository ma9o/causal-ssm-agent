"""Temporal activities for the baseline-report transition."""

from __future__ import annotations

import json
import logging
from typing import Any

from temporalio import activity
from temporalio.exceptions import ApplicationError

from nof1_causal_lab.machine.artifact_files import json_filename, pickle_filename
from nof1_causal_lab.machine.derivations import complete_derivation_cascade
from nof1_causal_lab.machine.errors import TransitionExecutionError
from nof1_causal_lab.machine.graph import transition_spec
from nof1_causal_lab.machine.moves import TransitionEffects, input_pins, run_retractions
from nof1_causal_lab.machine.store import ArtifactStore
from nof1_causal_lab.machine.temporal.latent_structure_activities import _llm_backend_config
from nof1_causal_lab.machine.temporal.messages import (
    SingleLLMTransitionFinalizeInput,
    SingleLLMTransitionPlan,
    SingleLLMTransitionWorkflowInput,
)
from nof1_causal_lab.utils import data as data_module
from nof1_causal_lab.utils import storage

logger = logging.getLogger(__name__)


def _write_baseline_json(path: str, value: Any) -> None:
    storage.write_text(path, json.dumps(value))


def _read_baseline_json(path: str) -> Any:
    return storage.read_json(path)


def _baseline_transition_failure(exc: Exception) -> ApplicationError:
    if isinstance(exc, TransitionExecutionError):
        return ApplicationError(
            str(exc),
            exc.diagnostics,
            type=type(exc).__name__,
            non_retryable=True,
        )
    return ApplicationError(str(exc), type=type(exc).__name__, non_retryable=True)


def _baseline_draws_stats(draws: list[float] | None) -> tuple[float | None, float | None]:
    if not draws:
        return None, None
    return sum(draws) / len(draws), sum(1 for draw in draws if draw > 0) / len(draws)


def _first_baseline_assistant_summary(trace: dict[str, Any]) -> str | None:
    for message in trace.get("messages", []):
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content") or "").strip()
        if content:
            return content
    return None


@activity.defn
async def plan_baseline_report_activity(
    input: SingleLLMTransitionWorkflowInput,
) -> SingleLLMTransitionPlan:
    from nof1_causal_lab.flows.run_store import load_pickle
    from nof1_causal_lab.flows.transitions.analysis.interventions import run_interventions
    from nof1_causal_lab.utils.causal_design import get_outcome_name
    from nof1_causal_lab.utils.config import get_config

    store = ArtifactStore(input.workspace_id)
    spec = transition_spec("baseline_report")
    pins = input_pins(input.state, spec)
    run_id = f"seq-{input.seq:06d}"

    diagnostics = store.read_json_file(
        "posterior",
        pins["posterior"],
        json_filename("posterior", "diagnostics"),
    )
    causal_design_payload = store.read_json_file(
        "causal_design",
        pins["causal_design"],
        json_filename("causal_design", "causal_design"),
    )
    causal_design = causal_design_payload["causal_design"]
    identification_report = store.read_json_file(
        "identification_report",
        pins["identification_report"],
        json_filename("identification_report", "identification_report"),
    )
    fitted_artifact = load_pickle(
        store.file_path("posterior", pins["posterior"], pickle_filename("posterior", "fitted"))
    )
    treatments = identification_report["estimable_treatments"]
    outcome_name = get_outcome_name(causal_design) or ""

    logger.info("=== analysis: Treatment Effects ===")
    logger.info("Estimating effects of %d treatments on %s", len(treatments), outcome_name)
    intervention_results = run_interventions(
        fitted_artifact,
        treatments,
        outcome_name,
        causal_design,
    )

    ppc_warnings = [
        {
            "variable": warning.get("variable"),
            "issue_type": warning.get("issue_type"),
            "severity": warning.get("severity"),
            "message": warning.get("message"),
        }
        for warning in diagnostics.get("ppc", {}).get("per_variable_warnings", [])
    ][:5]
    top_results = [
        {
            "treatment": entry.get("treatment"),
            "effect_size": _baseline_draws_stats(entry.get("posterior_draws"))[0],
            "prob_positive": _baseline_draws_stats(entry.get("posterior_draws"))[1],
        }
        for entry in intervention_results[:5]
    ]
    commentary_input = {
        "outcome": outcome_name,
        "identifiable_treatments": treatments,
        "excluded_non_identifiable_treatments": sorted(
            causal_design.get("identifiability", {}).get("non_identifiable_treatments", {}).keys()
        ),
        "top_ranked_effects": top_results,
        "ppc_warnings": ppc_warnings,
        "follow_up_capabilities": {
            "get_model_info": (
                "Inspect variables, measurement, identifiability, diagnostics, and baseline effects."
            ),
            "simulate_intervention": (
                "Run Pearl rung-2 intervention simulations on the fitted generative model."
            ),
            "simulate_counterfactual": (
                "Run Pearl rung-3 counterfactual simulations conditioned on an observed history window."
            ),
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
    context_ref = storage.join(
        data_module.runs_dir(input.workspace_id),
        "temporal-llm",
        run_id,
        "baseline-report",
        "context.json",
    )
    _write_baseline_json(
        context_ref,
        {
            "system_prompt": system_prompt,
            "user_messages": [user_prompt],
            "intervention_results": intervention_results,
        },
    )

    config = get_config()
    return SingleLLMTransitionPlan(
        workspace_id=input.workspace_id,
        run_id=run_id,
        context_ref=context_ref,
        pins=pins,
        llm=_llm_backend_config(config.analysis_commentary.llm, config.llm, None),
        max_tool_turns=1,
    )


@activity.defn
async def finalize_baseline_report_activity(
    input: SingleLLMTransitionFinalizeInput,
) -> TransitionEffects:
    from nof1_causal_lab.flows.transitions.analysis.contracts import BaselineReportContract

    try:
        context = _read_baseline_json(input.context_ref)
        trace = _read_baseline_json(input.trace_ref)
        payload: dict[str, Any] = {
            "intervention_results": context["intervention_results"],
            "llm_trace": trace,
        }
        final_summary = _first_baseline_assistant_summary(trace)
        if final_summary:
            payload["final_summary"] = final_summary
        fields = set(BaselineReportContract.model_fields.keys())
        payload = {key: value for key, value in payload.items() if key in fields}

        store = ArtifactStore(input.workspace_id)
        produced = [
            store.write_version(
                "baseline_report",
                provenance="computed",
                derived_from=input.pins,
                produced_by="run:baseline_report",
                json_files={json_filename("baseline_report", "baseline_report"): payload},
            )
        ]
        spec = transition_spec("baseline_report")
        retracted = run_retractions(input.state, spec, produced)
        return complete_derivation_cascade(store, input.state, produced, retracted)
    except Exception as exc:
        raise _baseline_transition_failure(exc) from exc


BASELINE_REPORT_ACTIVITIES = [
    plan_baseline_report_activity,
    finalize_baseline_report_activity,
]
