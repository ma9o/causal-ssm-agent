"""Thin aggregation layer for persisted artifact contracts and tool metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from nof1_causal_lab.flows.context_tools import CONTEXT_TOOLS, INTERACTIVE_CONTEXTS
from nof1_causal_lab.flows.contracts_base import (
    ArtifactContractId,
    BaseArtifactContract,
    LLMArtifactContract,
    ToolContract,
)
from nof1_causal_lab.flows.transitions.analysis.contracts import (
    EXPORTED_TOOL_RESULT_MODELS,
    BaselineReportContract,
    BaselineReportVisualizationContract,
    EffectSummaryContract,
    EffectTrajectoryPointContract,
    SavedScenarioContract,
    ScenarioStartResultContract,
    SimulateScenarioInput,
    SimulateScenarioResultContract,
    SimulateScenarioToolResultContract,
    ToolErrorContract,
    TreatmentEffectContract,
)
from nof1_causal_lab.flows.transitions.extraction.contracts import (
    MeasurementsContract,
    WorkerStatusContract,
)
from nof1_causal_lab.flows.transitions.inference.contracts import (
    PosteriorContract,
    PPCResultContract,
)
from nof1_causal_lab.flows.transitions.inference_contracts import InferenceMetadataContract
from nof1_causal_lab.flows.transitions.ingestion.contracts import (
    RawDataColumnDescriptionContract,
    RawDataContract,
)
from nof1_causal_lab.flows.transitions.latent_structure.contracts import LatentStructureContract
from nof1_causal_lab.flows.transitions.measurement_structure.contracts import (
    MeasurementStructureContract,
)
from nof1_causal_lab.flows.transitions.model_spec.contracts import StatisticalModelSpecContract
from nof1_causal_lab.flows.transitions.validation.contracts import (
    IndicatorAuditContract,
    IndicatorEmpiricalProfileContract,
    IndicatorValidationContract,
    ValidationIssueContract,
    ValidationReportContract,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

ARTIFACT_CONTRACTS: dict[ArtifactContractId, type[BaseArtifactContract]] = {
    "raw_data": RawDataContract,
    "latent_structure": LatentStructureContract,
    "measurement_structure": MeasurementStructureContract,
    "measurements": MeasurementsContract,
    "validation_report": ValidationReportContract,
    "statistical_model_spec": StatisticalModelSpecContract,
    "posterior": PosteriorContract,
    "baseline_report": BaselineReportContract,
}


def _validate_artifact_model(artifact_id: str, data: dict[str, Any]) -> BaseModel:
    if artifact_id not in ARTIFACT_CONTRACTS:
        known = ", ".join(sorted(ARTIFACT_CONTRACTS.keys()))
        raise ValueError(f"Unknown artifact_id '{artifact_id}'. Expected one of: {known}")
    aid = cast("ArtifactContractId", artifact_id)
    return ARTIFACT_CONTRACTS[aid].model_validate(data)


def validate_artifact_payload(artifact_id: str, data: dict[str, Any]) -> dict[str, Any]:
    return _validate_artifact_model(artifact_id, data).model_dump(mode="json")


__all__ = [
    "ARTIFACT_CONTRACTS",
    "ArtifactContractId",
    "BaseArtifactContract",
    "CONTEXT_TOOLS",
    "EffectSummaryContract",
    "EffectTrajectoryPointContract",
    "EXPORTED_TOOL_RESULT_MODELS",
    "IndicatorAuditContract",
    "IndicatorEmpiricalProfileContract",
    "IndicatorValidationContract",
    "InferenceMetadataContract",
    "INTERACTIVE_CONTEXTS",
    "LLMArtifactContract",
    "PPCResultContract",
    "SavedScenarioContract",
    "ScenarioStartResultContract",
    "SimulateScenarioInput",
    "SimulateScenarioResultContract",
    "SimulateScenarioToolResultContract",
    "RawDataColumnDescriptionContract",
    "RawDataContract",
    "LatentStructureContract",
    "MeasurementStructureContract",
    "MeasurementsContract",
    "ValidationReportContract",
    "StatisticalModelSpecContract",
    "PosteriorContract",
    "BaselineReportContract",
    "BaselineReportVisualizationContract",
    "ToolContract",
    "ToolErrorContract",
    "TreatmentEffectContract",
    "ValidationIssueContract",
    "WorkerStatusContract",
    "validate_artifact_payload",
]
