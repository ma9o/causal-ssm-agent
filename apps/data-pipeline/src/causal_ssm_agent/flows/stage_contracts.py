"""Thin aggregation layer for persisted stage contracts and tool metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from causal_ssm_agent.flows.contracts_base import (
    BaseStageContract,
    LLMStageContract,
    StageId,
    ToolContract,
)
from causal_ssm_agent.flows.stage_tools import INTERACTIVE_STAGES, STAGE_TOOLS
from causal_ssm_agent.flows.stages.inference_contracts import InferenceMetadataContract
from causal_ssm_agent.flows.stages.stage0.contracts import (
    Stage0ColumnDescriptionContract,
    Stage0Contract,
)
from causal_ssm_agent.flows.stages.stage1a.contracts import Stage1aContract
from causal_ssm_agent.flows.stages.stage1b.contracts import Stage1bContract
from causal_ssm_agent.flows.stages.stage2.contracts import (
    ObservationRecordContract,
    Stage2Contract,
    WorkerStatusContract,
)
from causal_ssm_agent.flows.stages.stage3.contracts import (
    IndicatorAuditContract,
    IndicatorEmpiricalProfileContract,
    IndicatorValidationContract,
    Stage3Contract,
    ValidationIssueContract,
)
from causal_ssm_agent.flows.stages.stage4.contracts import Stage4Contract
from causal_ssm_agent.flows.stages.stage4b.contracts import Stage4bContract
from causal_ssm_agent.flows.stages.stage5a.contracts import Stage5aContract
from causal_ssm_agent.flows.stages.stage5b.contracts import (
    PowerScalingResultContract,
    PPCResultContract,
    Stage5bContract,
)
from causal_ssm_agent.flows.stages.stage6.contracts import (
    EXPORTED_TOOL_RESULT_MODELS,
    CounterfactualEvidenceResultContract,
    EffectSummaryContract,
    EffectTrajectoryPointContract,
    SavedScenarioContract,
    SimulateCounterfactualInput,
    SimulateCounterfactualResultContract,
    SimulateCounterfactualToolResultContract,
    SimulateInterventionInput,
    SimulateInterventionResultContract,
    SimulateInterventionToolResultContract,
    Stage6Contract,
    Stage6VisualizationContract,
    ToolErrorContract,
    TreatmentEffectContract,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

STAGE_CONTRACTS: dict[StageId, type[BaseStageContract]] = {
    "stage-0": Stage0Contract,
    "stage-1a": Stage1aContract,
    "stage-1b": Stage1bContract,
    "stage-2": Stage2Contract,
    "stage-3": Stage3Contract,
    "stage-4": Stage4Contract,
    "stage-4b": Stage4bContract,
    "stage-5a": Stage5aContract,
    "stage-5b": Stage5bContract,
    "stage-6": Stage6Contract,
}


def _validate_stage_model(stage_id: str, data: dict[str, Any]) -> BaseModel:
    if stage_id not in STAGE_CONTRACTS:
        known = ", ".join(sorted(STAGE_CONTRACTS.keys()))
        raise ValueError(f"Unknown stage_id '{stage_id}'. Expected one of: {known}")
    sid = cast("StageId", stage_id)
    return STAGE_CONTRACTS[sid].model_validate(data)


def validate_stage_payload(stage_id: str, data: dict[str, Any]) -> dict[str, Any]:
    return _validate_stage_model(stage_id, data).model_dump(mode="json")


__all__ = [
    "BaseStageContract",
    "CounterfactualEvidenceResultContract",
    "EffectSummaryContract",
    "EffectTrajectoryPointContract",
    "EXPORTED_TOOL_RESULT_MODELS",
    "IndicatorAuditContract",
    "IndicatorEmpiricalProfileContract",
    "IndicatorValidationContract",
    "InferenceMetadataContract",
    "INTERACTIVE_STAGES",
    "LLMStageContract",
    "ObservationRecordContract",
    "PPCResultContract",
    "PowerScalingResultContract",
    "SavedScenarioContract",
    "SimulateCounterfactualInput",
    "SimulateCounterfactualResultContract",
    "SimulateCounterfactualToolResultContract",
    "SimulateInterventionInput",
    "SimulateInterventionResultContract",
    "SimulateInterventionToolResultContract",
    "Stage0ColumnDescriptionContract",
    "Stage0Contract",
    "Stage1aContract",
    "Stage1bContract",
    "Stage2Contract",
    "Stage3Contract",
    "Stage4Contract",
    "Stage4bContract",
    "Stage5aContract",
    "Stage5bContract",
    "Stage6Contract",
    "Stage6VisualizationContract",
    "STAGE_CONTRACTS",
    "STAGE_TOOLS",
    "StageId",
    "ToolContract",
    "ToolErrorContract",
    "TreatmentEffectContract",
    "ValidationIssueContract",
    "WorkerStatusContract",
    "validate_stage_payload",
]
