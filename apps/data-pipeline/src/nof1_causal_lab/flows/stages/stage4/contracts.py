"""Stage 4 contracts and tool metadata."""

from __future__ import annotations

from causal_ssm_agent.artifacts.model_spec import ModelSpec  # noqa: TC001
from causal_ssm_agent.flows.contracts_base import LLMStageContract
from causal_ssm_agent.flows.stages.stage4.tool_registry import build_stage4_public_tool_contracts
from causal_ssm_agent.workers.schemas_prior import PriorProposal  # noqa: TC001

STAGE_ID = "stage-4"
IS_INTERACTIVE_STAGE = True
STAGE4_TOOL_CONTRACTS = build_stage4_public_tool_contracts()


class Stage4Contract(LLMStageContract):
    model_spec: ModelSpec
    authored_priors: dict[str, PriorProposal]
    resolved_priors: list[PriorProposal]
    search_queries: dict[str, str] | None = None
    validation_warnings: list[str] | None = None
    prior_predictive_samples: dict[str, list[float]] | None = None

    def summary_message(self) -> str:
        return (
            f"Stage 4 summary: parameters={len(self.model_spec.parameters)} "
            f"likelihoods={len(self.model_spec.likelihoods)} "
            f"authored_priors={len(self.authored_priors)} "
            f"resolved_priors={len(self.resolved_priors)} "
            f"warnings={len(self.validation_warnings or [])} "
            f"prior_predictive_channels={len(self.prior_predictive_samples or {})}"
        )
