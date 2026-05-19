"""Stage 4 repair routing package."""

from .compile import classify_compile_failure_route
from .helpers import (
    _find_block_for_parameter,
    _materialize_scope_candidate,
    _ordered_block_ids,
    _resolved_repair_scope,
)
from .localization import _localize_prior_failure, build_stage4_failure_evidence
from .planning import build_repair_plan, get_stage4_repair_scope_strategy
from .prior import classify_prior_failure_blocks
from .routing import classify_validation_outcome, resolve_prior_repair_decision
from .types import (
    RepairReasons,
    ResolvedRepairPlan,
    ResolvedRepairScope,
    Stage4FailureEvidence,
    Stage4FailureLocalization,
    Stage4PriorRepairDecision,
    Stage4RepairScopeStrategy,
    Stage4ScopeCandidateSpec,
    Stage4ScopeCandidateStrategy,
    Stage4ValidationOutcome,
    Stage4ValidationOutcomeDecision,
)

__all__ = [
    "RepairReasons",
    "ResolvedRepairPlan",
    "ResolvedRepairScope",
    "Stage4FailureEvidence",
    "Stage4FailureLocalization",
    "Stage4PriorRepairDecision",
    "Stage4RepairScopeStrategy",
    "Stage4ScopeCandidateSpec",
    "Stage4ScopeCandidateStrategy",
    "Stage4ValidationOutcome",
    "Stage4ValidationOutcomeDecision",
    "_find_block_for_parameter",
    "_localize_prior_failure",
    "_materialize_scope_candidate",
    "_ordered_block_ids",
    "_resolved_repair_scope",
    "build_repair_plan",
    "build_stage4_failure_evidence",
    "classify_compile_failure_route",
    "classify_prior_failure_blocks",
    "classify_validation_outcome",
    "get_stage4_repair_scope_strategy",
    "resolve_prior_repair_decision",
]
