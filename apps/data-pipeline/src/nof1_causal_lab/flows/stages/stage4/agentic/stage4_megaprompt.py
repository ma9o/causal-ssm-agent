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
    checks). The only thing the megaprompt mode
removes is the reducer's block-cursor: tool calls are dispatched based
on their payload, not on a current active block.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .stage4_core import (
    apply_prior_subset,
    lock_model_spec,
    validate_and_store_indicator_choice,
    validate_and_store_model_configuration,
)
from .stage4_state import Stage4AcceptedArtifacts, Stage4DraftModel

if TYPE_CHECKING:
    from .stage4_types import Stage4Deps

MEGAPROMPT_CHECKPOINT_VERSION = 1

logger = logging.getLogger(__name__)
# Per-turn diagnostics are at INFO level; bypass the root logger's default
# WARNING threshold so they propagate to Prefect's flow-run logger.
logger.setLevel(logging.INFO)

if TYPE_CHECKING:
    from collections.abc import Callable

    from nof1_causal_lab.utils.agent_session import StageSessionFactory


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


def serialize_stage4_megaprompt_state(state: Stage4MegapromptState) -> dict[str, Any]:
    """Render the megaprompt state as a JSON-safe dict for checkpointing.

    Persists only the *inputs* of the stage (decisions, model_spec,
    authored priors, search caches). ``validation`` is a derived
    artifact and is deliberately excluded — on resume it is recomputed
    from the retained inputs via :func:`validate_assembly`. Keeping it
    out of the checkpoint avoids the trap where a validator rule change
    (e.g. scale_mismatch was blocking, now warns) leaves a stale verdict
    on disk that contradicts the current rules.
    """
    return {
        "version": MEGAPROMPT_CHECKPOINT_VERSION,
        "draft_model": {
            "distribution_choices": dict(state.draft_model.distribution_choices),
            "initialization_policy": state.draft_model.initialization_policy,
            "observation_intercept_policy": state.draft_model.observation_intercept_policy,
            "equilibrium_forcing": state.draft_model.equilibrium_forcing,
        },
        "accepted": {
            "model_spec": state.accepted.model_spec,
            "authored_priors": dict(state.accepted.authored_priors),
            "resolved_priors": state.accepted.resolved_priors,
        },
        "search_cache": dict(state.search_cache),
        "search_queries": dict(state.search_queries),
        "last_feedback": state.last_feedback,
        "tool_call_count": state.tool_call_count,
    }


def deserialize_stage4_megaprompt_state(payload: dict[str, Any]) -> Stage4MegapromptState:
    """Reconstruct :class:`Stage4MegapromptState` from a checkpoint payload.

    ``accepted.validation`` is left as ``None`` by design — the caller
    recomputes it from the retained inputs on resume. An unknown
    ``version`` raises ``ValueError`` so callers can fall back to a
    fresh state rather than silently importing a schema from the future.
    """
    version = payload.get("version")
    if version != MEGAPROMPT_CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported Stage 4 megaprompt checkpoint version {version!r}; "
            f"expected {MEGAPROMPT_CHECKPOINT_VERSION}"
        )
    draft = payload.get("draft_model") or {}
    accepted_payload = payload.get("accepted") or {}
    return Stage4MegapromptState(
        draft_model=Stage4DraftModel(
            distribution_choices=dict(draft.get("distribution_choices") or {}),
            initialization_policy=draft.get("initialization_policy"),
            observation_intercept_policy=draft.get("observation_intercept_policy"),
            equilibrium_forcing=draft.get("equilibrium_forcing"),
        ),
        accepted=Stage4AcceptedArtifacts(
            model_spec=accepted_payload.get("model_spec"),
            authored_priors=dict(accepted_payload.get("authored_priors") or {}),
            resolved_priors=accepted_payload.get("resolved_priors"),
            validation=None,
        ),
        search_cache=dict(payload.get("search_cache") or {}),
        search_queries=dict(payload.get("search_queries") or {}),
        last_feedback=str(payload.get("last_feedback") or ""),
        tool_call_count=int(payload.get("tool_call_count") or 0),
    )


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
    """Eagerly lock the model spec via the shared core, once all decisions are in.

    Returns a compact lock-status line for inclusion in tool feedback,
    or ``None`` when the lock is still pending. Thin wrapper around
    :func:`stage4_core.lock_model_spec` so the megaprompt keeps its
    "lock eagerly after every decision" semantics while the state
    machine continues to lock lazily from its settle loop.
    """
    if not _model_decisions_complete(state, ambiguous_variables):
        return None
    result, errors = lock_model_spec(state.draft_model, state.accepted, deps)
    if result is None:
        return "MODEL SPEC LOCK ERROR:\n" + "\n".join(f"- {error}" for error in errors)
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
    """Dispatch an indicator-choice submission to the scope-free core."""
    state.tool_call_count += 1
    block = plan_block_by_variable.get(variable)
    if block is None:
        return (
            f"VALIDATION ERRORS:\n- `{variable}` is not an ambiguous indicator. "
            "Allowed variables: "
            + (", ".join(f"`{name}`" for name in ambiguous_variables) or "(none)")
        )
    error = validate_and_store_indicator_choice(
        state.draft_model,
        block,
        variable=variable,
        distribution=distribution,
        link=link,
        reasoning=reasoning,
    )
    if error is not None:
        return error
    choice = state.draft_model.distribution_choices[variable]
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
    """Dispatch a model-configuration submission to the scope-free core."""
    state.tool_call_count += 1
    error = validate_and_store_model_configuration(
        state.draft_model,
        initialization_policy=initialization_policy,
        observation_intercept_policy=observation_intercept_policy,
        equilibrium_forcing=equilibrium_forcing,
        reasoning=reasoning,
    )
    if error is not None:
        return error
    status_line = _lock_model_spec_if_ready(
        state,
        deps=deps,
        ambiguous_variables=ambiguous_variables,
    )
    accepted_line = (
        "ACCEPTED model configuration: "
        f"init=`{state.draft_model.initialization_policy}`, "
        f"obs_intercepts=`{state.draft_model.observation_intercept_policy}`, "
        f"equilibrium_forcing=`{str(bool(state.draft_model.equilibrium_forcing)).lower()}`"
    )
    return accepted_line if status_line is None else f"{accepted_line}\n\n{status_line}"


def _apply_prior_block(
    state: Stage4MegapromptState,
    *,
    deps: Stage4Deps,
    parameter_inventory: tuple[str, ...],
    priors: dict[str, dict[str, Any]],
) -> str:
    """Dispatch a prior-subset submission to the scope-free core.

    In megaprompt mode we pass ``allowed_parameter_names=None`` — the core
    grounds against the full accepted ``model_spec`` inventory rather than
    restricting to any active block.
    """
    state.tool_call_count += 1
    # Locked-inventory resolution: prefer the model_spec's inventory when
    # available, fall back to the skeleton's if the spec hasn't locked yet.
    if state.accepted.model_spec is not None:
        locked_inventory = {
            str(parameter.get("name"))
            for parameter in state.accepted.model_spec.get("parameters") or []
            if isinstance(parameter, dict) and isinstance(parameter.get("name"), str)
        }
        allowed = (
            frozenset(locked_inventory) if locked_inventory else frozenset(parameter_inventory)
        )
    else:
        allowed = frozenset(parameter_inventory)
    outcome = apply_prior_subset(
        state.accepted,
        deps,
        priors=priors,
        allowed_parameter_names=allowed,
    )
    if isinstance(outcome, str):
        return outcome
    return outcome.feedback


class Stage4MegapromptSessionAdapter:
    """Tool-surface adapter that makes :class:`Stage4MegapromptState` look
    like a :class:`Stage4Session` to the shared tool factories.

    The adapter exposes the same ``submit_*`` method signatures the
    state-machine session offers, so the single shared ``make_submit_*_tool``
    factories in ``stage4/tools.py`` can build megaprompt tools too. It
    also proxies ``search_cache`` / ``search_queries`` for the literature
    tool. Everything scope-free: no active-block cursor, no coverage
    enforcement, no reducer events — just direct dispatch to the
    scope-free apply helpers above.
    """

    def __init__(
        self,
        state: Stage4MegapromptState,
        *,
        deps: Stage4Deps,
        plan_block_by_variable: dict[str, Any],
        ambiguous_variables: tuple[str, ...],
        parameter_inventory: tuple[str, ...],
        save_checkpoint: Callable[[Stage4MegapromptState], None] | None = None,
    ) -> None:
        self._state = state
        self._deps = deps
        self._plan_block_by_variable = plan_block_by_variable
        self._ambiguous_variables = ambiguous_variables
        self._parameter_inventory = parameter_inventory
        self._save_checkpoint = save_checkpoint

    @property
    def search_cache(self) -> dict[str, str]:
        return self._state.search_cache

    @property
    def search_queries(self) -> dict[str, str]:
        return self._state.search_queries

    def _persist(self) -> None:
        if self._save_checkpoint is None:
            return
        try:
            self._save_checkpoint(self._state)
        except Exception as exc:  # noqa: BLE001 — checkpoint failures must not crash the run
            logger.warning(
                "stage-4:megaprompt checkpoint save failed (%s: %s) — continuing without persist",
                type(exc).__name__,
                exc,
            )

    def submit_model_configuration(
        self,
        *,
        initialization_policy: str,
        observation_intercept_policy: str,
        equilibrium_forcing: bool,
        reasoning: str,
    ) -> str:
        feedback = _apply_model_configuration(
            self._state,
            deps=self._deps,
            ambiguous_variables=self._ambiguous_variables,
            initialization_policy=initialization_policy,
            observation_intercept_policy=observation_intercept_policy,
            equilibrium_forcing=equilibrium_forcing,
            reasoning=reasoning,
        )
        self._state.last_feedback = feedback
        self._persist()
        return feedback

    def submit_indicator_choice(
        self,
        *,
        variable: str,
        distribution: str,
        link: str,
        reasoning: str,
    ) -> str:
        feedback = _apply_indicator_choice(
            self._state,
            deps=self._deps,
            plan_block_by_variable=self._plan_block_by_variable,
            ambiguous_variables=self._ambiguous_variables,
            variable=variable,
            distribution=distribution,
            link=link,
            reasoning=reasoning,
        )
        self._state.last_feedback = feedback
        self._persist()
        return feedback

    def submit_prior_block(self, *, priors: dict[str, dict[str, Any]]) -> str:
        feedback = _apply_prior_block(
            self._state,
            deps=self._deps,
            parameter_inventory=self._parameter_inventory,
            priors=priors,
        )
        self._state.last_feedback = feedback
        self._persist()
        return feedback


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
    save_checkpoint: Callable[[Stage4MegapromptState], None] | None = None,
) -> list[Any]:
    """Build the megaprompt tool list by reusing the shared tool factories.

    We wrap the megaprompt state in an adapter that matches the
    state-machine session's submit_* method shape, then build tools with
    ``stop_on_success=False`` so the long-running single session doesn't
    terminate after each accepted submission. ``save_checkpoint`` is
    plumbed into the adapter so every successful submit call overwrites
    the on-disk checkpoint with the latest accepted state.
    """
    from nof1_causal_lab.flows.stages.stage4.tools import (
        make_elicit_prior_gmm_tool,
        make_search_tool,
        make_submit_indicator_choice_tool,
        make_submit_model_configuration_tool,
        make_submit_prior_block_tool,
    )

    adapter = Stage4MegapromptSessionAdapter(
        state,
        deps=deps,
        plan_block_by_variable=plan_block_by_variable,
        ambiguous_variables=ambiguous_variables,
        parameter_inventory=parameter_inventory,
        save_checkpoint=save_checkpoint,
    )
    tools: list[Any] = [
        make_submit_model_configuration_tool(adapter, stop_on_success=False),
        make_submit_indicator_choice_tool(adapter, stop_on_success=False),
        make_submit_prior_block_tool(adapter, stop_on_success=False),
    ]
    if enable_literature:
        tools.append(make_search_tool(adapter))
    if enable_paraphrasing:
        tools.append(
            make_elicit_prior_gmm_tool(
                question=question,
                paraphrase_session_factory=paraphrase_session_factory,
                n_paraphrases=n_paraphrases,
            )
        )
    return tools
