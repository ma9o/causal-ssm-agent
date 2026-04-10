"""Stage 4 execution state types.

Cursor variants, runtime state, and accepted-state containers shared across
stage4 orchestration modules.  Extracted from ``stage4.py`` so that
``stage4_repair.py`` can reference cursor and runtime types without a circular
import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from causal_ssm_agent.flows.stages.stage4.assembly import AssemblyValidation
    from causal_ssm_agent.workers.schemas_prior import PriorPathologyCertificate

    from .stage4_feedback import Stage4ValidationPacket
    from .stage4_orchestrator import Stage4FrontierBlock


# ---------------------------------------------------------------------------
# Accepted / decision state
# ---------------------------------------------------------------------------


@dataclass
class Stage4AcceptedState:
    """Typed accepted Stage 4 artifacts accumulated across reducer steps."""

    model_spec: dict[str, Any] | None = None
    authored_priors: dict[str, dict[str, Any]] = field(default_factory=dict)
    validation: AssemblyValidation | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def as_current(self) -> dict[str, Any]:
        """Return the accepted state in grounding-compatible dict form."""
        current = dict(self.extras)
        if self.model_spec is not None:
            current["model_spec"] = self.model_spec
        if self.authored_priors:
            current["authored_priors"] = self.authored_priors
        if self.validation is not None:
            current["validation"] = self.validation
        return current

    def apply_stage_output(self, stage_output: dict[str, Any] | None) -> None:
        """Merge accepted stage output into typed state."""
        if stage_output is None:
            return
        if "model_spec" in stage_output:
            self.model_spec = stage_output["model_spec"]
        if "authored_priors" in stage_output:
            self.authored_priors = stage_output["authored_priors"]
        if "validation" in stage_output:
            self.validation = stage_output["validation"]
        for key, value in stage_output.items():
            if key not in {"model_spec", "authored_priors", "validation"}:
                self.extras[key] = value


@dataclass
class Stage4DecisionState:
    """Accepted model-decision deltas before the full ModelSpec is locked."""

    distribution_choices: dict[str, dict[str, Any]] = field(default_factory=dict)
    initialization_policy: str | None = None
    equilibrium_forcing: bool | None = None


# ---------------------------------------------------------------------------
# Cursor variants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage4BlockCursor:
    """Promptable execution cursor anchored to one authored Stage 4 block."""

    block_id: str


@dataclass(frozen=True)
class Stage4ModelSpecLockPendingCursor:
    """Internal state waiting to lock the deterministic model spec."""

    reason: str


@dataclass(frozen=True)
class Stage4RepairBarrierCursor:
    """Internal state waiting for structural repair barrier validation."""

    reason: str
    scope_block_ids: tuple[str, ...]


@dataclass(frozen=True)
class Stage4DoneCursor:
    """Terminal Stage 4 cursor."""


Stage4ExecutionCursor = (
    Stage4BlockCursor
    | Stage4ModelSpecLockPendingCursor
    | Stage4RepairBarrierCursor
    | Stage4DoneCursor
)


# ---------------------------------------------------------------------------
# Repair campaign
# ---------------------------------------------------------------------------


@dataclass
class Stage4RepairCampaignState:
    """Active bounded Stage 4 repair campaign over one structural scope."""

    failure_family_key: tuple[Any, ...]
    scope_kind: str
    scope_key: str
    scope_rank: int
    scope_block_ids: tuple[str, ...]
    prompt_blocks_by_id: dict[str, Stage4FrontierBlock] = field(default_factory=dict)
    completed_block_ids: frozenset[str] = field(default_factory=frozenset)
    requires_barrier_validation: bool = False
    attempts_at_scope: int = 1
    best_certificate: PriorPathologyCertificate | None = None


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


@dataclass
class Stage4Runtime:
    """Mutable Stage 4 reducer runtime."""

    cursor: Stage4ExecutionCursor = field(
        default_factory=lambda: Stage4ModelSpecLockPendingCursor(
            reason="awaiting initial Stage 4 block activation",
        )
    )
    block_status: dict[str, str] = field(default_factory=dict)
    decisions: Stage4DecisionState = field(default_factory=Stage4DecisionState)
    accepted: Stage4AcceptedState = field(default_factory=Stage4AcceptedState)
    last_validation_packet: Stage4ValidationPacket | None = None
    search_cache: dict[str, str] = field(default_factory=dict)
    search_queries: dict[str, str] = field(default_factory=dict)
    repair_campaign: Stage4RepairCampaignState | None = None
