"""Artifact-contract validation helpers owned by the test suite."""

from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict

from nof1_causal_lab.flows.contracts_base import BaseArtifactContract, LLMArtifactContract
from nof1_causal_lab.flows.transitions.analysis.contracts import BaselineReportContract
from nof1_causal_lab.flows.transitions.extraction.contracts import MeasurementsContract
from nof1_causal_lab.flows.transitions.inference.contracts import PosteriorContract
from nof1_causal_lab.flows.transitions.latent_structure.contracts import LatentStructureContract
from nof1_causal_lab.flows.transitions.measurement_structure.contracts import (
    MeasurementStructureContract,
)
from nof1_causal_lab.flows.transitions.model_spec.contracts import StatisticalModelSpecContract
from nof1_causal_lab.flows.transitions.validation.contracts import ValidationReportContract

ArtifactContractId = Literal[
    "raw_data",
    "latent_structure",
    "measurement_structure",
    "measurements",
    "validation_report",
    "statistical_model_spec",
    "posterior",
    "baseline_report",
]


class RawDataColumnDescriptionContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str


class RawDataContract(LLMArtifactContract):
    column_descriptions: list[RawDataColumnDescriptionContract]


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


def validate_artifact_payload(artifact_id: str, data: dict[str, Any]) -> dict[str, Any]:
    if artifact_id not in ARTIFACT_CONTRACTS:
        known = ", ".join(sorted(ARTIFACT_CONTRACTS))
        raise ValueError(f"Unknown artifact_id '{artifact_id}'. Expected one of: {known}")
    contract_id = cast("ArtifactContractId", artifact_id)
    return ARTIFACT_CONTRACTS[contract_id].model_validate(data).model_dump(mode="json")
