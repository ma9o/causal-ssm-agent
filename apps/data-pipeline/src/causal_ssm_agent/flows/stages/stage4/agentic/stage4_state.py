"""Stage 4 execution state types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from causal_ssm_agent.flows.stages.stage4.assembly import AssemblyValidation
    from causal_ssm_agent.workers.schemas_prior import PriorPathologyCertificate

    from .stage4_feedback import Stage4ValidationPacket
    from .stage4_orchestrator import Stage4FrontierBlock


@dataclass
class Stage4AcceptedArtifacts:
    """Typed accepted Stage 4 artifacts accumulated across reducer steps."""

    model_spec: dict[str, Any] | None = None
    authored_priors: dict[str, dict[str, Any]] = field(default_factory=dict)
    resolved_priors: list[dict[str, Any]] | None = None
    validation: AssemblyValidation | None = None

    def as_current(self) -> dict[str, Any]:
        """Return the accepted state in grounding-compatible dict form."""
        current: dict[str, Any] = {}
        if self.model_spec is not None:
            current["model_spec"] = self.model_spec
        if self.authored_priors:
            current["authored_priors"] = self.authored_priors
        if self.resolved_priors is not None:
            current["resolved_priors"] = self.resolved_priors
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
        if "resolved_priors" in stage_output:
            self.resolved_priors = stage_output["resolved_priors"]
        if "validation" in stage_output:
            self.validation = stage_output["validation"]


@dataclass
class Stage4DraftModel:
    """Accepted model-decision deltas before the full ModelSpec is locked."""

    distribution_choices: dict[str, dict[str, Any]] = field(default_factory=dict)
    initialization_policy: str | None = None
    equilibrium_forcing: bool | None = None

    def sync_from_model_spec(self, model_spec: dict[str, Any] | None) -> None:
        """Refresh the draft decisions from one accepted locked model spec."""
        if not isinstance(model_spec, dict):
            return
        self.initialization_policy = (
            None
            if model_spec.get("initialization_policy") is None
            else str(model_spec.get("initialization_policy"))
        )
        equilibrium_forcing = model_spec.get("equilibrium_forcing")
        self.equilibrium_forcing = (
            None if equilibrium_forcing is None else bool(equilibrium_forcing)
        )
        distribution_choices: dict[str, dict[str, Any]] = {}
        for likelihood in model_spec.get("likelihoods") or []:
            if not isinstance(likelihood, dict) or not isinstance(likelihood.get("variable"), str):
                continue
            choice = dict(likelihood)
            choice.setdefault("reasoning", "Accepted locked likelihood choice.")
            distribution_choices[str(likelihood["variable"])] = choice
        self.distribution_choices = distribution_choices


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
    attempts_at_scope: int = 1
    best_certificate: PriorPathologyCertificate | None = None


@dataclass
class Stage4DomainState:
    """Reducer-owned domain state that determines the next promptable block."""

    active_block_id: str | None = None
    done: bool = False
    model_lock_pending: bool = False
    block_status: dict[str, str] = field(default_factory=dict)
    draft_model: Stage4DraftModel = field(default_factory=Stage4DraftModel)
    accepted: Stage4AcceptedArtifacts = field(default_factory=Stage4AcceptedArtifacts)
    repair_campaign: Stage4RepairCampaignState | None = None


@dataclass
class Stage4InteractionState:
    """Prompt/session state that should not drive reducer transitions."""

    last_validation_packet: Stage4ValidationPacket | None = None
    search_cache: dict[str, str] = field(default_factory=dict)
    search_queries: dict[str, str] = field(default_factory=dict)


@dataclass
class Stage4Runtime:
    """Mutable Stage 4 runtime split into domain and interaction concerns."""

    domain: Stage4DomainState = field(default_factory=Stage4DomainState)
    interaction: Stage4InteractionState = field(default_factory=Stage4InteractionState)
