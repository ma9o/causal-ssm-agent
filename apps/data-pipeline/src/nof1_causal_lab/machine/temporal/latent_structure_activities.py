"""Temporal activities for the latent-structure transition."""

from __future__ import annotations

import json
from typing import Any

from temporalio import activity
from temporalio.exceptions import ApplicationError

from nof1_causal_lab.machine.artifact_files import json_filename
from nof1_causal_lab.machine.derivations import complete_derivation_cascade
from nof1_causal_lab.machine.errors import TransitionExecutionError
from nof1_causal_lab.machine.graph import transition_spec
from nof1_causal_lab.machine.moves import TransitionEffects, input_pins, run_retractions
from nof1_causal_lab.machine.store import ArtifactStore
from nof1_causal_lab.machine.temporal.messages import (
    LLMBackendConfig,
    SingleLLMTransitionFinalizeInput,
    SingleLLMTransitionPlan,
    SingleLLMTransitionWorkflowInput,
)
from nof1_causal_lab.utils import data as data_module
from nof1_causal_lab.utils import storage


def _write_latent_json(path: str, value: Any) -> None:
    storage.write_text(path, json.dumps(value))


def _read_latent_json(path: str) -> Any:
    return storage.read_json(path)


def _first_latent_config_value(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _latent_transition_failure(exc: Exception) -> ApplicationError:
    if isinstance(exc, TransitionExecutionError):
        return ApplicationError(
            str(exc),
            exc.diagnostics,
            type=type(exc).__name__,
            non_retryable=True,
        )
    return ApplicationError(str(exc), type=type(exc).__name__, non_retryable=True)


def _llm_backend_config(
    profile_llm: Any,
    defaults: Any,
    max_tool_turns: int | None,
) -> LLMBackendConfig:
    if profile_llm.harness == "none":
        embedded = defaults.embedded
        return LLMBackendConfig(
            harness="none",
            model=profile_llm.model,
            max_tokens=_first_latent_config_value(profile_llm.max_tokens, embedded.max_tokens),
            timeout=_first_latent_config_value(profile_llm.timeout, embedded.timeout),
            reasoning_effort=_first_latent_config_value(
                profile_llm.reasoning_effort,
                embedded.reasoning_effort,
            ),
        )

    if profile_llm.harness == "claude-code":
        claude = defaults.claude_code
        return LLMBackendConfig(
            harness="claude-code",
            model=profile_llm.model,
            bin=_first_latent_config_value(profile_llm.bin, claude.bin),
            effort=_first_latent_config_value(profile_llm.effort, claude.effort),
            max_turns=_first_latent_config_value(
                profile_llm.max_turns, max_tool_turns, claude.max_turns
            ),
            max_budget_usd=_first_latent_config_value(
                profile_llm.max_budget_usd,
                claude.max_budget_usd,
            ),
            fallback_model=_first_latent_config_value(
                profile_llm.fallback_model,
                claude.fallback_model,
            ),
        )

    if profile_llm.harness == "codex":
        codex = defaults.codex
        return LLMBackendConfig(
            harness="codex",
            model=profile_llm.model,
            bin=_first_latent_config_value(profile_llm.bin, codex.bin),
            reasoning_effort=_first_latent_config_value(
                profile_llm.reasoning_effort,
                codex.reasoning_effort,
            ),
            service_tier=_first_latent_config_value(profile_llm.service_tier, codex.service_tier),
            timeout=profile_llm.timeout,
        )

    raise ValueError(f"unknown LLM harness {profile_llm.harness!r}")


@activity.defn
async def plan_latent_structure_activity(
    input: SingleLLMTransitionWorkflowInput,
) -> SingleLLMTransitionPlan:
    from nof1_causal_lab.flows.transitions.latent_structure.prompting import templates
    from nof1_causal_lab.utils.config import get_config

    store = ArtifactStore(input.workspace_id)
    spec = transition_spec("latent_structure")
    pins = input_pins(input.state, spec)
    run_id = f"seq-{input.seq:06d}"
    question = store.read_json_file(
        "question",
        pins["question"],
        json_filename("question", "question"),
    )["text"]
    context_ref = storage.join(
        data_module.runs_dir(input.workspace_id),
        "temporal-llm",
        run_id,
        "latent-structure",
        "context.json",
    )
    _write_latent_json(
        context_ref,
        {
            "system_prompt": templates.SYSTEM,
            "user_messages": [
                templates.USER.format(question=question),
                templates.REVIEW,
            ],
        },
    )

    config = get_config()
    max_tool_turns = config.structure_proposal.latent_max_tool_turns
    return SingleLLMTransitionPlan(
        workspace_id=input.workspace_id,
        run_id=run_id,
        context_ref=context_ref,
        pins=pins,
        llm=_llm_backend_config(config.structure_proposal.llm, config.llm, max_tool_turns),
        max_tool_turns=max_tool_turns,
    )


@activity.defn
async def finalize_latent_structure_activity(
    input: SingleLLMTransitionFinalizeInput,
) -> TransitionEffects:
    from nof1_causal_lab.flows.transitions.latent_structure.contracts import (
        LatentStructureContract,
    )

    try:
        if input.result_ref is None:
            raise RuntimeError("latent-structure subroutine completed without a result ref")
        payload = _read_latent_json(input.result_ref)
        payload["llm_trace"] = _read_latent_json(input.trace_ref)
        fields = set(LatentStructureContract.model_fields.keys())
        payload = {key: value for key, value in payload.items() if key in fields}

        store = ArtifactStore(input.workspace_id)
        produced = [
            store.write_version(
                "latent_structure",
                provenance="computed",
                derived_from=input.pins,
                produced_by="run:latent_structure",
                json_files={json_filename("latent_structure", "latent_structure"): payload},
            )
        ]
        spec = transition_spec("latent_structure")
        retracted = run_retractions(input.state, spec, produced)
        return complete_derivation_cascade(store, input.state, produced, retracted)
    except Exception as exc:
        raise _latent_transition_failure(exc) from exc


LATENT_STRUCTURE_ACTIVITIES = [
    plan_latent_structure_activity,
    finalize_latent_structure_activity,
]
