"""Stage 4 megaprompt mode: single-session, whole-task agent loop.

This module implements the alternative Stage 4 execution path that runs
when :attr:`Stage4Config.state_machine_enabled` is ``False``. Instead of
prompting the LLM one frontier block at a time, it opens a single agent
session with every submit tool exposed at once and lets the model submit
decisions and priors in any order.

The action space matches the state-machine mode exactly — the same submit
tools (``submit_model_configuration``, ``submit_indicator_choice``,
``submit_prior_block``, and the optional ``search_literature`` /
``elicit_prior_gmm``) — and the same validation checks gate every
submission (schema validation, model compilation, prior-predictive
checks, output-Jacobian sensitivity). The only thing the megaprompt mode
removes is the reducer's block-cursor: tool calls are dispatched based
on their payload, not on a current active block.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from causal_ssm_agent.flows.stages.stage4.model_spec_decisions import (
    ModelConfigurationChoice,
)

from .stage4_cards import (
    build_construct_scale_cards,
    build_distribution_cards,
    build_model_topology,
    build_prior_cards,
)
from .stage4_orchestrator import build_stage4_plan
from .stage4_skeleton import derive_deterministic_spec
from .stage4_state import Stage4AcceptedArtifacts, Stage4DraftModel
from .stage4_submission import _normalize_indicator_submission
from .stage4_types import Stage4Deps, Stage4Result

logger = logging.getLogger(__name__)
# Per-turn diagnostics are at INFO level; bypass the root logger's default
# WARNING threshold so they propagate to Prefect's flow-run logger.
logger.setLevel(logging.INFO)

if TYPE_CHECKING:
    import polars as pl

    from causal_ssm_agent.utils.agent_session import StageSessionFactory


_OPTIONAL_PRIOR_ROLES = frozenset({"initial_state_mean", "initial_state_sd"})


@dataclass
class Stage4MegapromptState:
    """Mutable accumulated state for one Stage 4 megaprompt session."""

    draft_model: Stage4DraftModel = field(default_factory=Stage4DraftModel)
    accepted: Stage4AcceptedArtifacts = field(default_factory=Stage4AcceptedArtifacts)
    search_cache: dict[str, str] = field(default_factory=dict)
    search_queries: dict[str, str] = field(default_factory=dict)
    last_feedback: str = ""
    tool_call_count: int = 0

    def is_done(self) -> bool:
        """Whether the accumulated state satisfies the Stage 4 done predicate.

        Mirrors the state-machine invariant: the stage-machine only reaches
        ``runtime.domain.done`` after every prior block is accepted, which
        implicitly guarantees that every required prior is authored before
        the terminal validation is consulted. Without the coverage check
        below, ``validation.is_valid`` can be trivially ``True`` for a
        partial prior set — :func:`stage4_grounding` passes ``None`` to
        :func:`validate_assembly` when priors are missing, which skips the
        prior-predictive pass and leaves the default ``pp_valid=True``.
        """
        validation = self.accepted.validation
        if (
            self.accepted.model_spec is None
            or not self.accepted.authored_priors
            or validation is None
            or not validation.is_valid
        ):
            return False
        required = _required_prior_names_from_spec(self.accepted.model_spec)
        return all(name in self.accepted.authored_priors for name in required)


def _required_prior_names_from_spec(model_spec: dict[str, Any] | None) -> tuple[str, ...]:
    """Return the parameter names that still need a prior for a locked model_spec."""
    if not isinstance(model_spec, dict):
        return ()
    names: list[str] = []
    for parameter in model_spec.get("parameters") or []:
        if not isinstance(parameter, dict):
            continue
        if parameter.get("role") in _OPTIONAL_PRIOR_ROLES:
            continue
        name = parameter.get("name")
        if isinstance(name, str):
            names.append(name)
    return tuple(names)


def _optional_prior_names_from_spec(model_spec: dict[str, Any] | None) -> tuple[str, ...]:
    """Return the optional prior parameter names for a locked model_spec."""
    if not isinstance(model_spec, dict):
        return ()
    names: list[str] = []
    for parameter in model_spec.get("parameters") or []:
        if not isinstance(parameter, dict):
            continue
        if parameter.get("role") not in _OPTIONAL_PRIOR_ROLES:
            continue
        name = parameter.get("name")
        if isinstance(name, str):
            names.append(name)
    return tuple(names)


def _parameter_inventory_from_skeleton(skeleton: Any) -> tuple[str, ...]:
    """Return the parameter-name inventory implied by the deterministic skeleton."""
    names: list[str] = []
    for parameter in skeleton.all_params:
        if isinstance(parameter, dict):
            name = parameter.get("name")
            if isinstance(name, str):
                names.append(name)
    return tuple(names)


def _model_decisions_complete(
    state: Stage4MegapromptState,
    ambiguous_variables: tuple[str, ...],
) -> bool:
    """Whether every model decision has been submitted at least once."""
    decisions = state.draft_model
    if (
        decisions.initialization_policy is None
        or decisions.observation_intercept_policy is None
        or decisions.equilibrium_forcing is None
    ):
        return False
    return all(variable in decisions.distribution_choices for variable in ambiguous_variables)


def _lock_model_spec_if_ready(
    state: Stage4MegapromptState,
    *,
    deps: Stage4Deps,
    ambiguous_variables: tuple[str, ...],
) -> str | None:
    """Build and ground the model spec when every model decision is present.

    Returns a compact lock-status line for inclusion in tool feedback, or
    ``None`` when the lock is still pending.
    """
    from .stage4_reducer import build_model_spec_from_decisions

    if not _model_decisions_complete(state, ambiguous_variables):
        return None
    model_spec, errors = build_model_spec_from_decisions(state.draft_model, deps.skeleton)
    if model_spec is None:
        return "MODEL SPEC LOCK ERROR:\n" + "\n".join(f"- {error}" for error in errors)

    current = state.accepted.as_current()
    existing_priors = current.get("authored_priors")
    active_parameter_names = {
        str(parameter["name"])
        for parameter in model_spec.get("parameters") or []
        if isinstance(parameter, dict) and isinstance(parameter.get("name"), str)
    }
    if isinstance(existing_priors, dict):
        filtered = {
            name: prior for name, prior in existing_priors.items() if name in active_parameter_names
        }
        if filtered:
            current["authored_priors"] = filtered
        else:
            current.pop("authored_priors", None)
    current.pop("resolved_priors", None)

    result = deps.grounding_fn(
        {"model_spec": model_spec},
        deps.causal_spec,
        current=current,
        data_for_model=deps.data_for_model,
        indicator_audits=deps.indicator_audits,
    )
    state.accepted.apply_stage_output(result.stage_output)
    # grounding only emits ``authored_priors`` in the output when priors are
    # submitted in the same call, so a model-spec-only re-lock does not
    # propagate the inventory filter back to accepted state. Reconcile it
    # explicitly — the state machine covers this via
    # ``reconcile_locked_prior_surface``.
    if state.accepted.authored_priors:
        state.accepted.authored_priors = {
            name: prior
            for name, prior in state.accepted.authored_priors.items()
            if name in active_parameter_names
        }
    state.accepted.resolved_priors = None
    validation = None if result.stage_output is None else result.stage_output.get("validation")
    if validation is not None and not validation.compile_ok:
        compile_error = (validation.compile_error or "(no detail)").strip()
        return (
            "MODEL SPEC LOCK: compile failed — the combined model spec cannot be "
            "built. Fix the indicator choice or configuration that causes this "
            "error and resubmit; do not keep cycling configurations when the "
            "message points at an indicator.\n\nCOMPILE ERROR:\n" + compile_error
        )
    return "MODEL SPEC LOCKED: proceed with `submit_prior_block` for the missing priors."


def _apply_indicator_choice(
    state: Stage4MegapromptState,
    *,
    deps: Stage4Deps,
    plan_block_by_variable: dict[str, Any],
    ambiguous_variables: tuple[str, ...],
    variable: str,
    distribution: str,
    link: str,
    reasoning: str,
) -> str:
    """Validate and apply one indicator-likelihood choice."""
    state.tool_call_count += 1
    block = plan_block_by_variable.get(variable)
    if block is None:
        return (
            f"VALIDATION ERRORS:\n- `{variable}` is not an ambiguous indicator. "
            "Allowed variables: "
            + (", ".join(f"`{name}`" for name in ambiguous_variables) or "(none)")
        )
    normalized, error = _normalize_indicator_submission(
        block,
        {
            "variable": variable,
            "distribution": distribution,
            "link": link,
            "reasoning": reasoning,
        },
    )
    if error is not None:
        return error
    assert normalized is not None
    choice = normalized["distribution_choice"]
    state.draft_model.distribution_choices[choice["variable"]] = choice
    status_line = _lock_model_spec_if_ready(
        state,
        deps=deps,
        ambiguous_variables=ambiguous_variables,
    )
    accepted_line = (
        f"ACCEPTED indicator choice: `{variable}` → `{choice['distribution']}` / `{choice['link']}`"
    )
    return accepted_line if status_line is None else f"{accepted_line}\n\n{status_line}"


def _apply_model_configuration(
    state: Stage4MegapromptState,
    *,
    deps: Stage4Deps,
    ambiguous_variables: tuple[str, ...],
    initialization_policy: str,
    observation_intercept_policy: str,
    equilibrium_forcing: bool,
    reasoning: str,
) -> str:
    """Validate and apply the global model configuration."""
    state.tool_call_count += 1
    try:
        config = ModelConfigurationChoice.model_validate(
            {
                "initialization_policy": initialization_policy,
                "observation_intercept_policy": observation_intercept_policy,
                "equilibrium_forcing": equilibrium_forcing,
                "reasoning": reasoning,
            }
        ).model_dump(mode="json")
    except ValidationError as exc:
        return f"VALIDATION ERRORS:\n- {exc}"
    state.draft_model.initialization_policy = str(config["initialization_policy"])
    state.draft_model.observation_intercept_policy = str(config["observation_intercept_policy"])
    state.draft_model.equilibrium_forcing = bool(config["equilibrium_forcing"])
    status_line = _lock_model_spec_if_ready(
        state,
        deps=deps,
        ambiguous_variables=ambiguous_variables,
    )
    accepted_line = (
        "ACCEPTED model configuration: "
        f"init=`{config['initialization_policy']}`, "
        f"obs_intercepts=`{config['observation_intercept_policy']}`, "
        f"equilibrium_forcing=`{str(bool(config['equilibrium_forcing'])).lower()}`"
    )
    return accepted_line if status_line is None else f"{accepted_line}\n\n{status_line}"


def _apply_prior_block(
    state: Stage4MegapromptState,
    *,
    deps: Stage4Deps,
    parameter_inventory: tuple[str, ...],
    priors: dict[str, dict[str, Any]],
) -> str:
    """Validate and apply one prior-block submission."""
    state.tool_call_count += 1
    if not isinstance(priors, dict) or not priors:
        return "VALIDATION ERRORS:\n- `priors` must be a non-empty object"

    inventory = set(parameter_inventory)
    if state.accepted.model_spec is not None:
        locked_inventory = {
            str(parameter.get("name"))
            for parameter in state.accepted.model_spec.get("parameters") or []
            if isinstance(parameter, dict) and isinstance(parameter.get("name"), str)
        }
        inventory = locked_inventory or inventory
    invalid = sorted(name for name in priors if name not in inventory)
    if invalid:
        preview = ", ".join(f"`{name}`" for name in invalid[:20])
        if len(invalid) > 20:
            preview += f", … ({len(invalid) - 20} more)"
        return "VALIDATION ERRORS:\n- priors outside the parameter inventory: " + preview

    result = deps.grounding_fn(
        {"priors": priors},
        deps.causal_spec,
        current=state.accepted.as_current(),
        data_for_model=deps.data_for_model,
        indicator_audits=deps.indicator_audits,
    )
    state.accepted.apply_stage_output(result.stage_output)
    return result.feedback


def _make_megaprompt_tools(
    state: Stage4MegapromptState,
    *,
    deps: Stage4Deps,
    plan_block_by_variable: dict[str, Any],
    ambiguous_variables: tuple[str, ...],
    parameter_inventory: tuple[str, ...],
    enable_literature: bool,
    enable_paraphrasing: bool,
    question: str,
    paraphrase_session_factory: StageSessionFactory,
    n_paraphrases: int,
) -> list[Any]:
    """Build the megaprompt tool list without state-machine gating."""
    from causal_ssm_agent.utils.openrouter_client import Tool

    async def _submit_model_configuration(
        *,
        initialization_policy: str,
        observation_intercept_policy: str,
        equilibrium_forcing: bool,
        reasoning: str,
    ) -> str:
        feedback = _apply_model_configuration(
            state,
            deps=deps,
            ambiguous_variables=ambiguous_variables,
            initialization_policy=initialization_policy,
            observation_intercept_policy=observation_intercept_policy,
            equilibrium_forcing=equilibrium_forcing,
            reasoning=reasoning,
        )
        state.last_feedback = feedback
        return feedback

    async def _submit_indicator_choice(
        *, variable: str, distribution: str, link: str, reasoning: str
    ) -> str:
        feedback = _apply_indicator_choice(
            state,
            deps=deps,
            plan_block_by_variable=plan_block_by_variable,
            ambiguous_variables=ambiguous_variables,
            variable=variable,
            distribution=distribution,
            link=link,
            reasoning=reasoning,
        )
        state.last_feedback = feedback
        return feedback

    async def _submit_prior_block(*, priors: dict[str, dict[str, Any]]) -> str:
        feedback = _apply_prior_block(
            state,
            deps=deps,
            parameter_inventory=parameter_inventory,
            priors=priors,
        )
        state.last_feedback = feedback
        return feedback

    tools: list[Any] = [
        Tool(
            name="submit_model_configuration",
            description=(
                "Submit the global initialization, observation-intercept, and "
                "equilibrium-forcing decision."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "initialization_policy": {
                        "type": "string",
                        "enum": ["stationary", "free"],
                        "description": "Global initial-state policy for retained dynamic states.",
                    },
                    "observation_intercept_policy": {
                        "type": "string",
                        "enum": ["fixed", "free"],
                        "description": (
                            "Whether eligible manifest intercepts remain free or are fixed."
                        ),
                    },
                    "equilibrium_forcing": {
                        "type": "boolean",
                        "description": (
                            "Whether eligible dynamic states may have a continuous-time intercept."
                        ),
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Short justification for the global model configuration.",
                    },
                },
                "required": [
                    "initialization_policy",
                    "observation_intercept_policy",
                    "equilibrium_forcing",
                    "reasoning",
                ],
                "additionalProperties": False,
            },
            execute=_submit_model_configuration,
        ),
        Tool(
            name="submit_indicator_choice",
            description=("Submit one distribution/link choice for an ambiguous indicator."),
            parameters={
                "type": "object",
                "properties": {
                    "variable": {
                        "type": "string",
                        "description": "Ambiguous indicator variable name.",
                    },
                    "distribution": {
                        "type": "string",
                        "description": "Chosen distribution for the indicator.",
                    },
                    "link": {
                        "type": "string",
                        "description": "Chosen link function for the indicator.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Short justification for the indicator choice.",
                    },
                },
                "required": ["variable", "distribution", "link", "reasoning"],
                "additionalProperties": False,
            },
            execute=_submit_indicator_choice,
        ),
        Tool(
            name="submit_prior_block",
            description=(
                "Submit prior proposals keyed by parameter name for any subset of the "
                "parameter inventory; call multiple times to cover all required parameters."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "priors": {
                        "type": "object",
                        "description": "Prior proposals keyed by parameter name.",
                    }
                },
                "required": ["priors"],
                "additionalProperties": False,
            },
            execute=_submit_prior_block,
        ),
    ]
    if enable_literature:
        from causal_ssm_agent.flows.stages.stage4.tools import search_literature

        async def _search_literature(*, query: str, parameter_name: str) -> str:
            state.search_queries[parameter_name] = query
            cached = state.search_cache.get(query)
            if cached is not None:
                return cached
            result = await search_literature(query)
            state.search_cache[query] = result
            return result

        tools.append(
            Tool(
                name="search_literature",
                description=(
                    "Search for empirical literature about effect sizes for model parameters."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for empirical literature about effect sizes.",
                        },
                        "parameter_name": {
                            "type": "string",
                            "description": (
                                "Name of the parameter this search is for "
                                "(e.g. 'beta_stress_sleep')."
                            ),
                        },
                    },
                    "required": ["query", "parameter_name"],
                    "additionalProperties": False,
                },
                execute=_search_literature,
            )
        )
    if enable_paraphrasing:
        from causal_ssm_agent.flows.stages.stage4.tools import make_elicit_prior_gmm_tool

        tools.append(
            make_elicit_prior_gmm_tool(
                question=question,
                paraphrase_session_factory=paraphrase_session_factory,
                n_paraphrases=n_paraphrases,
            )
        )
    return tools


async def run_stage4_megaprompt(
    causal_spec: dict,
    question: str,
    data_for_model: pl.DataFrame,
    indicator_audits: dict[str, dict[str, Any]],
    session_factory: StageSessionFactory,
    *,
    paraphrase_session_factory: StageSessionFactory | None = None,
    enable_literature: bool = True,
    enable_paraphrasing: bool = False,
    n_paraphrases: int = 10,
    max_tool_turns: int = 40,
    max_outer_turns: int = 8,
) -> Stage4Result:
    """Run the Stage 4 flow with a single megaprompt agent session.

    The action space matches :func:`run_stage4` — same submit tools, same
    per-submission validation — but the harness exposes every tool at once
    and lets the model choose the order. The loop stops as soon as the
    accumulated state satisfies the same "valid model_spec + priors"
    predicate the state-machine mode uses.
    """
    del max_tool_turns  # inherited from StageSessionFactory initialization
    from causal_ssm_agent.flows.stages.stage4.grounding import stage4_grounding

    from .prompts.megaprompt import (
        build_stage4_megaprompt_system_prompt,
        build_stage4_megaprompt_user_prompt,
    )

    skeleton = derive_deterministic_spec(causal_spec)
    plan = build_stage4_plan(causal_spec, skeleton)
    model_topology = build_model_topology(causal_spec)
    distribution_cards = build_distribution_cards(causal_spec, indicator_audits, skeleton)
    construct_scale_cards = build_construct_scale_cards(causal_spec, indicator_audits, skeleton)
    prior_cards = build_prior_cards(causal_spec, skeleton)

    ambiguous_variables = tuple(str(item["variable"]) for item in skeleton.ambiguous_indicators)
    plan_block_by_variable = {
        block.variable_names[0]: block
        for block in plan.model_blocks
        if block.kind == "indicator_decision" and block.variable_names
    }
    parameter_inventory = _parameter_inventory_from_skeleton(skeleton)

    configuration_block = next(
        (block for block in plan.model_blocks if block.kind == "model_configuration"),
        None,
    )
    centerable_construct_names: tuple[str, ...] = ()
    baseline_factor_names: tuple[str, ...] = ()
    if configuration_block is not None:
        centerable_construct_names = tuple(
            configuration_block.payload.get("centerable_construct_names") or ()
        )
        baseline_factor_names = tuple(
            configuration_block.payload.get("baseline_factor_names") or ()
        )

    deps = Stage4Deps(
        skeleton=skeleton,
        causal_spec=causal_spec,
        data_for_model=data_for_model,
        indicator_audits=indicator_audits,
        grounding_fn=stage4_grounding,
    )
    state = Stage4MegapromptState()
    system_prompt = build_stage4_megaprompt_system_prompt(
        enable_literature=enable_literature,
        enable_paraphrasing=enable_paraphrasing,
    )
    tools = _make_megaprompt_tools(
        state,
        deps=deps,
        plan_block_by_variable=plan_block_by_variable,
        ambiguous_variables=ambiguous_variables,
        parameter_inventory=parameter_inventory,
        enable_literature=enable_literature,
        enable_paraphrasing=enable_paraphrasing,
        question=question,
        paraphrase_session_factory=paraphrase_session_factory or session_factory,
        n_paraphrases=n_paraphrases,
    )

    async with session_factory.open(
        system_prompt=system_prompt,
        tools=tools,
        log_label="stage-4:megaprompt",
    ) as agent_session:
        for _outer_turn in range(max(1, max_outer_turns)):
            required_prior_names = (
                _required_prior_names_from_spec(state.accepted.model_spec)
                if state.accepted.model_spec is not None
                else parameter_inventory
            )
            optional_prior_names = _optional_prior_names_from_spec(state.accepted.model_spec)
            user_prompt = build_stage4_megaprompt_user_prompt(
                question=question,
                model_topology=model_topology,
                distribution_cards=distribution_cards,
                loading_params=skeleton.loading_params,
                construct_scale_cards=construct_scale_cards,
                prior_cards=prior_cards,
                ambiguous_indicators=skeleton.ambiguous_indicators,
                distribution_choices=state.draft_model.distribution_choices,
                initialization_policy=state.draft_model.initialization_policy,
                observation_intercept_policy=state.draft_model.observation_intercept_policy,
                equilibrium_forcing=state.draft_model.equilibrium_forcing,
                centerable_construct_names=centerable_construct_names,
                baseline_factor_names=baseline_factor_names,
                required_prior_names=required_prior_names,
                optional_prior_names=optional_prior_names,
                authored_priors=state.accepted.authored_priors,
                model_spec_locked=state.accepted.model_spec is not None,
                latest_feedback=state.last_feedback,
                include_prior_source_guidance=enable_literature,
            )
            calls_before = state.tool_call_count
            turn_result = await agent_session.turn(user_prompt)
            tool_calls = list(getattr(turn_result, "tool_calls_fired", ()) or ())
            completion = (getattr(turn_result, "completion", "") or "").strip()
            logger.info(
                "stage-4:megaprompt turn %d: tool_calls=%d (fired=%s) completion_preview=%r",
                _outer_turn + 1,
                state.tool_call_count - calls_before,
                tool_calls,
                completion[:400],
            )
            if state.is_done():
                break
            if state.tool_call_count == calls_before:
                # The model made no tool calls this turn; there is no point
                # in re-prompting with the same state. Fall through and raise
                # below so the caller sees a deterministic failure instead of
                # a silent infinite loop.
                break

    if not state.is_done():
        missing = [
            name
            for name in _required_prior_names_from_spec(state.accepted.model_spec)
            if name not in state.accepted.authored_priors
        ]
        raise ValueError(
            "Stage 4 megaprompt flow did not produce a valid model_spec + priors "
            f"(model_spec_locked={state.accepted.model_spec is not None!r}, "
            f"tool_call_count={state.tool_call_count!r}, "
            f"missing_priors={len(missing)!r}, "
            f"last_feedback={state.last_feedback!r})"
        )

    validation = state.accepted.validation
    assert state.accepted.model_spec is not None
    assert validation is not None
    return Stage4Result(
        model_spec=state.accepted.model_spec,
        authored_priors=state.accepted.authored_priors,
        search_queries=dict(state.search_queries),
        validation=validation,
    )
