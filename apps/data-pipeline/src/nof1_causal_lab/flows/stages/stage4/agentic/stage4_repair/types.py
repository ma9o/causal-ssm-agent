"""Typed Stage 4 repair models and shared constants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

    from nof1_causal_lab.flows.stages.stage4.agentic.stage4_orchestrator import (
        Stage4FrontierBlock,
        Stage4Plan,
        Stage4RepairTopology,
    )
    from nof1_causal_lab.workers.schemas_prior import (
        PriorPathologyCertificate,
        PriorRepairScope,
        PriorValidationResult,
    )


_MAX_SCOPE_ATTEMPTS = 2
_MAX_SCOPE_ATTEMPTS_BY_BLOCK_ID = {
    "review:prior_system": 5,
}
_GLOBAL_REVIEW_SCOPE_RANK = 3
_VALIDATOR_SCOPE_RANK = 2
_DRIFT_RELATED_CODES = frozenset(
    {
        "dt_ct_approximation_warning",
        "partial_drift_stability",
    }
)


def _max_scope_attempts_for_block_ids(block_ids: tuple[str, ...]) -> int:
    """Return the configured retry budget for one scope's prompt blocks."""
    return max(
        (
            _MAX_SCOPE_ATTEMPTS_BY_BLOCK_ID.get(block_id, _MAX_SCOPE_ATTEMPTS)
            for block_id in block_ids
        ),
        default=_MAX_SCOPE_ATTEMPTS,
    )


Stage4ValidationOutcome = Literal[
    "accepted",
    "compile_error",
    "prior_predictive_failure",
]


@dataclass(frozen=True)
class RepairReasons:
    """Candidate repair reasons extracted from diagnostic evidence."""

    default: str | None
    support: str | None
    drift: str | None
    validator: str | None
    global_: str | None


@dataclass(frozen=True)
class Stage4FailureLocalization:
    """Localized evidence for one Stage 4 prior-validation failure family."""

    failure_family: tuple[Any, ...]
    diagnostic_codes: tuple[str, ...]
    direct_parameters: tuple[str, ...]
    supporting_parameters: tuple[str, ...]
    manifest_names: tuple[str, ...]
    construct_names: tuple[str, ...]
    validator_repair_scope: PriorRepairScope | None
    validator_parameter_hints: tuple[str, ...]
    pathology_certificate: PriorPathologyCertificate | None
    has_global_failure: bool
    issues_text: str
    reasons: RepairReasons

    @property
    def parameter_hints(self) -> tuple[str, ...]:
        """Return deterministic seed parameters for scope synthesis."""
        return tuple(
            dict.fromkeys(
                [
                    *self.direct_parameters,
                    *self.supporting_parameters,
                    *self.validator_parameter_hints,
                ]
            )
        )


@dataclass(frozen=True)
class Stage4FailureEvidence:
    """Normalized diagnostic evidence for one failed prior-predictive validation."""

    topology: Stage4RepairTopology
    failed_diagnostics: tuple[PriorValidationResult, ...]
    supporting_compile_diagnostics: tuple[PriorValidationResult, ...]
    diagnostic_codes: tuple[str, ...]
    supporting_codes: tuple[str, ...]
    manifest_names: tuple[str, ...]
    global_failure_sites: frozenset[str]

    @property
    def all_reason_diagnostics(self) -> tuple[PriorValidationResult, ...]:
        """Return diagnostics eligible to contribute user-facing repair reasons."""
        return (*self.failed_diagnostics, *self.supporting_compile_diagnostics)


@dataclass(frozen=True)
class ResolvedRepairScope:
    """Deterministic structural repair scope independent of prompt blocks."""

    scope_kind: str
    scope_rank: int
    scope_key: str
    reason: str
    failure_family: tuple[Any, ...]
    parameter_names: tuple[str, ...] = ()
    construct_names: tuple[str, ...] = ()
    prompt_block_hints: tuple[str, ...] = ()
    diagnostic_codes: tuple[str, ...] = ()
    pathology_certificate: PriorPathologyCertificate | None = None


@dataclass(frozen=True)
class Stage4RepairScopeStrategy:
    """Strategy for projecting one structural repair scope into prompt execution."""

    scope_kind: str
    resolve_prompt_block_ids: Callable[[Stage4Plan, ResolvedRepairScope], tuple[str, ...]]
    project_prompt_block: Callable[
        [Stage4Plan, Stage4FrontierBlock, ResolvedRepairScope],
        Stage4FrontierBlock | None,
    ]
    uses_repair_campaign: bool = False


@dataclass(frozen=True)
class Stage4ScopeCandidateSpec:
    """Candidate structural scope emitted before prompt-block projection."""

    scope_kind: str
    scope_rank: int
    reason: str
    parameter_names: tuple[str, ...] = ()
    construct_names: tuple[str, ...] = ()
    prompt_block_hints: tuple[str, ...] = ()
    scope_token: str | None = None


@dataclass(frozen=True)
class Stage4ScopeCandidateStrategy:
    """Strategy for emitting candidate repair scopes from localized evidence."""

    name: str
    build_specs: Callable[
        [Stage4Plan, Stage4FailureLocalization],
        tuple[Stage4ScopeCandidateSpec, ...],
    ]
    stop_after_match: bool = False


@dataclass(frozen=True)
class ResolvedRepairPlan:
    """Prompt execution plan for one resolved structural repair scope."""

    scope: ResolvedRepairScope
    prompt_blocks: tuple[Stage4FrontierBlock, ...]
    uses_repair_campaign: bool = False

    @property
    def block_ids(self) -> tuple[str, ...]:
        return tuple(block.id for block in self.prompt_blocks)


@dataclass(frozen=True)
class Stage4ValidationOutcomeDecision:
    """Typed classification for one validation outcome."""

    outcome: Stage4ValidationOutcome
    repair_plan: ResolvedRepairPlan | None = None


@dataclass(frozen=True)
class Stage4PriorRepairDecision:
    """Typed reducer decision for a prior submission after repair routing."""

    repair_plan: ResolvedRepairPlan | None
    accepted_block_id: str | None
    route_kind: Literal["accepted", "repair_single", "repair_multi", "rejected"]

    @property
    def promote_campaign_feedback(self) -> bool:
        """Whether reducer feedback should surface campaign-wide routing."""
        return self.route_kind == "repair_multi"
