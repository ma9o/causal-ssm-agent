"""Stage 4 scope-free core: shared apply primitives for every entry path.

The state-machine reducer and the megaprompt flow are two adapters over
the same underlying pipeline: validate → normalize → (optionally) ground
against the accepted state → mutate the accepted store → return the
validator's feedback. This module houses that shared pipeline.

Design rules:

* No function here looks at ``runtime.domain.active_block_id`` or any
  other state-machine cursor. The caller always supplies the target
  block (for indicator/config submissions that need block-shape
  validation) or an explicit parameter-inventory allowlist (for prior
  submissions). That keeps the core truly scope-free; scope is a
  property of the *caller*.
* Callers that want a narrower view of the returned feedback filter it
  themselves, typically by calling :func:`format_validation_feedback`
  on the resulting validation with a ``focus_parameters`` argument.
  This module never narrows feedback.
* Callers that want to emit state-machine events, route repair
  campaigns, or enforce per-block coverage policies do so on top of
  the primitives here — none of that logic belongs in the core.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from causal_ssm_agent.flows.stages.stage4.model_spec_decisions import (
    ModelConfigurationChoice,
)

from .stage4_submission import _normalize_indicator_submission

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .stage4_feedback import Stage4GroundingResult
    from .stage4_orchestrator import Stage4FrontierBlock
    from .stage4_state import Stage4AcceptedArtifacts, Stage4DraftModel
    from .stage4_types import Stage4Deps


def validate_and_store_indicator_choice(
    draft_model: Stage4DraftModel,
    target_block: Stage4FrontierBlock,
    *,
    variable: str,
    distribution: str,
    link: str,
    reasoning: str,
) -> str | None:
    """Validate an indicator submission and mutate ``draft_model`` in place.

    Returns ``None`` on success, or a pre-formatted ``VALIDATION ERRORS:``
    string when the payload is malformed or the distribution/link choice
    is not valid for the target block's ambiguous indicator. The mutation
    is idempotent — resubmitting the same choice overwrites the slot.
    """
    normalized, error = _normalize_indicator_submission(
        target_block,
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
    draft_model.distribution_choices[choice["variable"]] = choice
    return None


def validate_and_store_model_configuration(
    draft_model: Stage4DraftModel,
    *,
    initialization_policy: str,
    observation_intercept_policy: str,
    equilibrium_forcing: bool,
    reasoning: str,
) -> str | None:
    """Validate a model-configuration submission and mutate ``draft_model``.

    Returns ``None`` on success, or a ``VALIDATION ERRORS:`` string when
    the payload fails Pydantic schema validation against
    :class:`ModelConfigurationChoice`.
    """
    try:
        choice = ModelConfigurationChoice.model_validate(
            {
                "initialization_policy": initialization_policy,
                "observation_intercept_policy": observation_intercept_policy,
                "equilibrium_forcing": equilibrium_forcing,
                "reasoning": reasoning,
            }
        ).model_dump(mode="json")
    except ValidationError as exc:
        return f"VALIDATION ERRORS:\n- {exc}"
    draft_model.initialization_policy = str(choice["initialization_policy"])
    draft_model.observation_intercept_policy = str(choice["observation_intercept_policy"])
    draft_model.equilibrium_forcing = bool(choice["equilibrium_forcing"])
    return None


def lock_model_spec(
    draft_model: Stage4DraftModel,
    accepted: Stage4AcceptedArtifacts,
    deps: Stage4Deps,
) -> tuple[Stage4GroundingResult | None, tuple[str, ...]]:
    """Build a locked ``ModelSpec`` from the accepted decisions and ground it.

    Returns ``(grounding_result, ())`` on success, or ``(None, errors)``
    when :func:`build_model_spec_from_decisions` cannot materialize a
    spec yet (the caller typically surfaces those errors to the LLM).

    Callers are responsible for deciding *when* to lock — the megaprompt
    locks eagerly after every draft-model mutation, while the
    state-machine reducer locks from its ``settle_to_wait_state`` loop.
    ``accepted`` is mutated in place with ``apply_stage_output`` so the
    new model_spec (and any upstream-filtered ``authored_priors``) lands
    in the shared store.
    """
    from .stage4_reducer import build_model_spec_from_decisions

    model_spec, errors = build_model_spec_from_decisions(draft_model, deps.skeleton)
    if model_spec is None:
        return None, tuple(errors)

    current = accepted.as_current()
    active_parameter_names = {
        str(parameter["name"])
        for parameter in model_spec.get("parameters") or []
        if isinstance(parameter, dict) and isinstance(parameter.get("name"), str)
    }
    existing_priors = current.get("authored_priors")
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
    accepted.apply_stage_output(result.stage_output)
    # grounding emits ``authored_priors`` in its output only when priors are
    # part of the submission — so a pure model-spec lock needs to reconcile
    # the accepted priors against the new inventory explicitly.
    if accepted.authored_priors:
        accepted.authored_priors = {
            name: prior
            for name, prior in accepted.authored_priors.items()
            if name in active_parameter_names
        }
    accepted.resolved_priors = None
    return result, ()


def apply_prior_subset(
    accepted: Stage4AcceptedArtifacts,
    deps: Stage4Deps,
    *,
    priors: dict[str, dict[str, Any]],
    allowed_parameter_names: Iterable[str] | None = None,
    skip_ppc: bool = False,
) -> Stage4GroundingResult | str:
    """Apply a prior-subset submission against the accepted Stage 4 state.

    ``allowed_parameter_names`` is the write pre-filter: the state-machine
    passes the active block's parameter names to reject out-of-block
    submissions; the megaprompt passes ``None`` to accept any parameter
    in the locked model spec's inventory (grounding enforces that
    second-level check downstream). Callers that pass ``None`` here get
    the full scope-free behavior.

    Returns the :class:`Stage4GroundingResult` from the shared grounding
    pipeline on success, or a pre-formatted error string if the payload
    fails the payload-shape / inventory pre-filter. Successful results
    have already been merged into ``accepted``.
    """
    if not isinstance(priors, dict) or not priors:
        return "VALIDATION ERRORS:\n- `priors` must be a non-empty object"

    if allowed_parameter_names is not None:
        allowed = set(allowed_parameter_names)
        invalid = sorted(name for name in priors if name not in allowed)
        if invalid:
            from .stage4_text import summarize_stage4_names

            return (
                "VALIDATION ERRORS:\n- priors outside the parameter inventory: "
                + summarize_stage4_names(invalid)
            )

    result = deps.grounding_fn(
        {"priors": priors},
        deps.causal_spec,
        current=accepted.as_current(),
        data_for_model=deps.data_for_model,
        indicator_audits=deps.indicator_audits,
        skip_ppc=skip_ppc,
    )
    accepted.apply_stage_output(result.stage_output)
    return result
