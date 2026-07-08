"""model-spec contracts and tool metadata."""

from __future__ import annotations

from nof1_causal_lab.artifacts.statistical_model_spec import StatisticalModelSpec  # noqa: TC001
from nof1_causal_lab.flows.contracts_base import LLMArtifactContract
from nof1_causal_lab.flows.transitions.model_spec.tool_registry import (
    build_model_spec_public_tool_contracts,
)
from nof1_causal_lab.workers.schemas_prior import PriorProposal  # noqa: TC001

IS_INTERACTIVE_CONTEXT = True
MODEL_SPEC_TOOL_CONTRACTS = build_model_spec_public_tool_contracts()


class StatisticalModelSpecContract(LLMArtifactContract):
    statistical_model_spec: StatisticalModelSpec
    authored_priors: dict[str, PriorProposal]
    resolved_priors: list[PriorProposal]
    search_queries: dict[str, str] | None = None
    validation_warnings: list[str] | None = None
    prior_predictive_samples: dict[str, list[float]] | None = None
