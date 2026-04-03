"""Stage 4 block-local submission policies and validators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from causal_ssm_agent.orchestrator.schemas_model import DistributionChoice

from .stage4_orchestrator import (
    Stage4FrontierBlock,
    Stage4PromptScopePolicy,
    get_stage4_prompt_scope_policy,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _summarize_names(names: list[str], *, limit: int = 8) -> str:
    """Render a compact preview of names."""
    if not names:
        return "(none)"
    preview = ", ".join(f"`{name}`" for name in names[:limit])
    if len(names) <= limit:
        return preview
    return f"{preview}, ... (+{len(names) - limit} more)"


def _enabled_block_tool_names(
    policy: Stage4PromptScopePolicy,
    *,
    enable_literature: bool,
    enable_paraphrasing: bool,
) -> tuple[str, ...]:
    """Return the tool names that are both scope-allowed and runtime-enabled."""
    enabled: list[str] = []
    for tool_name in policy.allowed_tool_names:
        if tool_name == "search_literature" and not enable_literature:
            continue
        if tool_name == "elicit_prior_gmm" and not enable_paraphrasing:
            continue
        enabled.append(tool_name)
    return tuple(enabled)


@dataclass(frozen=True)
class Stage4BlockHandler:
    """Per-kind Stage 4 prompt and submission behavior."""

    kind: str
    prompt_policy: Stage4PromptScopePolicy
    normalize_submission: Callable[
        [Stage4FrontierBlock, dict[str, Any]],
        tuple[dict[str, Any] | None, str | None],
    ]
    include_prior_source_guidance: bool = False

    def allowed_tool_names(
        self,
        *,
        enable_literature: bool,
        enable_paraphrasing: bool,
    ) -> tuple[str, ...]:
        """Return runtime-enabled tools for this block kind."""
        return _enabled_block_tool_names(
            self.prompt_policy,
            enable_literature=enable_literature,
            enable_paraphrasing=enable_paraphrasing,
        )

    def include_prior_source_guidance_for_prompt(
        self,
        *,
        enable_literature: bool,
    ) -> bool:
        """Whether the prompt should mention authored literature-source payloads."""
        return self.include_prior_source_guidance and enable_literature


def validate_submission_envelope(
    data: dict[str, Any],
    block: Stage4FrontierBlock,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate the common Stage 4 submission envelope."""
    if not isinstance(data, dict):
        return None, "VALIDATION ERRORS:\n- submission must be a JSON object"

    block_id = data.get("block_id")
    block_kind = data.get("block_kind")
    proposal = data.get("proposal")

    if block_id != block.id:
        return (
            None,
            f"WRONG BLOCK:\n- active block id is `{block.id}`\n- received `{block_id}`",
        )
    if block_kind != block.kind:
        return (
            None,
            f"WRONG BLOCK KIND:\n- active block kind is `{block.kind}`\n- received `{block_kind}`",
        )
    if not isinstance(proposal, dict):
        return None, "VALIDATION ERRORS:\n- `proposal` must be an object"
    return proposal, None


def _normalize_indicator_submission(
    block: Stage4FrontierBlock,
    proposal: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate an indicator-decision proposal."""
    try:
        choice = DistributionChoice.model_validate(proposal).model_dump(mode="json")
    except Exception as exc:
        return None, f"VALIDATION ERRORS:\n- {exc}"

    variable = block.variable_names[0]
    if choice["variable"] != variable:
        return None, f"VALIDATION ERRORS:\n- proposal variable must be `{variable}`"

    item = block.payload
    allowed_distributions = (
        [item["fixed_distribution"]]
        if "fixed_distribution" in item
        else item.get("valid_distributions", [])
    )
    if choice["distribution"] not in allowed_distributions:
        return (
            None,
            "VALIDATION ERRORS:\n"
            f"- distribution `{choice['distribution']}` is invalid for `{variable}`",
        )

    allowed_links = (
        item.get("valid_links", [])
        if "fixed_distribution" in item
        else item.get("link_options", {}).get(choice["distribution"], [])
    )
    if choice["link"] not in allowed_links:
        return (
            None,
            "VALIDATION ERRORS:\n"
            f"- link `{choice['link']}` is invalid for `{variable}` with `{choice['distribution']}`",
        )
    return {"distribution_choice": choice}, None


def _normalize_prior_submission(
    block: Stage4FrontierBlock,
    proposal: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate a prior-block proposal."""
    raw_priors = proposal.get("priors")
    if not isinstance(raw_priors, dict) or not raw_priors:
        return None, "VALIDATION ERRORS:\n- `proposal.priors` must be a non-empty object"

    allowed = set(block.parameter_names)
    invalid = sorted(name for name in raw_priors if name not in allowed)
    if invalid:
        return (
            None,
            f"VALIDATION ERRORS:\n- priors outside the active block: {_summarize_names(invalid)}",
        )
    return {"priors": raw_priors}, None


def _normalize_global_review_submission(
    block: Stage4FrontierBlock,
    proposal: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate a compact global-review submission."""
    decision = proposal.get("decision")
    reasoning = proposal.get("reasoning")
    if decision not in {"approve", "reopen"}:
        return None, "VALIDATION ERRORS:\n- `proposal.decision` must be `approve` or `reopen`"
    if not isinstance(reasoning, str) or not reasoning.strip():
        return None, "VALIDATION ERRORS:\n- `proposal.reasoning` must be a non-empty string"
    reopen_block_ids = proposal.get("reopen_block_ids")
    if decision == "approve":
        if reopen_block_ids is not None:
            return (
                None,
                "VALIDATION ERRORS:\n- `reopen_block_ids` is only valid when decision=`reopen`",
            )
        return {"decision": decision, "reasoning": reasoning.strip()}, None

    if not isinstance(reopen_block_ids, list) or not reopen_block_ids:
        return (
            None,
            "VALIDATION ERRORS:\n- `proposal.reopen_block_ids` must be a non-empty list of model block ids",
        )
    if any(not isinstance(block_id, str) for block_id in reopen_block_ids):
        return None, "VALIDATION ERRORS:\n- `proposal.reopen_block_ids` must contain only strings"
    if len(set(reopen_block_ids)) != len(reopen_block_ids):
        return None, "VALIDATION ERRORS:\n- `proposal.reopen_block_ids` must not contain duplicates"

    allowed_ids_in_order = tuple(block.payload.get("reopenable_block_ids") or ())
    allowed_ids = set(allowed_ids_in_order)
    invalid_ids = [block_id for block_id in reopen_block_ids if block_id not in allowed_ids]
    if invalid_ids:
        return (
            None,
            "VALIDATION ERRORS:\n"
            f"- `reopen_block_ids` must be drawn from {_summarize_names(sorted(allowed_ids))}",
        )
    return {
        "decision": decision,
        "reasoning": reasoning.strip(),
        "reopen_block_ids": tuple(
            block_id for block_id in allowed_ids_in_order if block_id in set(reopen_block_ids)
        ),
    }, None


_BLOCK_HANDLERS: dict[str, Stage4BlockHandler] = {
    "indicator_decision": Stage4BlockHandler(
        kind="indicator_decision",
        prompt_policy=get_stage4_prompt_scope_policy("indicator_decision"),
        normalize_submission=_normalize_indicator_submission,
    ),
    "global_review": Stage4BlockHandler(
        kind="global_review",
        prompt_policy=get_stage4_prompt_scope_policy("global_review"),
        normalize_submission=_normalize_global_review_submission,
    ),
}
for _prior_kind in (
    "measurement_prior",
    "observation_prior",
    "dynamics_prior",
    "effect_prior",
    "correlation_prior",
    "global_prior_review",
):
    _BLOCK_HANDLERS[_prior_kind] = Stage4BlockHandler(
        kind=_prior_kind,
        prompt_policy=get_stage4_prompt_scope_policy(_prior_kind),
        normalize_submission=_normalize_prior_submission,
        include_prior_source_guidance=True,
    )


def get_stage4_block_handler(kind: str) -> Stage4BlockHandler:
    """Return the registered handler for a block kind."""
    handler = _BLOCK_HANDLERS.get(kind)
    if handler is None:
        raise ValueError(f"Unsupported Stage 4 block kind {kind!r}")
    return handler
