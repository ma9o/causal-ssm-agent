"""Artifact-contract catalog owned by schema and lineage tooling."""

from typing import Literal

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
