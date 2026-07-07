"""Stage 4 contracts and tool metadata."""

from __future__ import annotations

from nof1_causal_lab.artifacts.model_spec import ModelSpec  # noqa: TC001
from nof1_causal_lab.flows.contracts_base import LLMStageContract
from nof1_causal_lab.flows.stages.stage4.tool_registry import build_stage4_public_tool_contracts
from nof1_causal_lab.workers.schemas_prior import PriorProposal  # noqa: TC001

IS_INTERACTIVE_STAGE = True
STAGE4_TOOL_CONTRACTS = build_stage4_public_tool_contracts()


class Stage4Contract(LLMStageContract):
    model_spec: ModelSpec
    authored_priors: dict[str, PriorProposal]
    resolved_priors: list[PriorProposal]
    search_queries: dict[str, str] | None = None
    validation_warnings: list[str] | None = None
    prior_predictive_samples: dict[str, list[float]] | None = None
