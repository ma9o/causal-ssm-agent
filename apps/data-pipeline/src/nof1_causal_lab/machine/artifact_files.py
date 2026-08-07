"""Canonical file layout for versioned machine artifacts.

The artifact graph names semantic dependencies such as ``panel`` or
``posterior``. This module is the single map from those artifact ids to the
files inside ``store/{artifact_id}/v{N}/``. UI projections, fixture seeders,
stage runners, and tool contexts should refer to this map instead of spelling
filenames independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nof1_causal_lab.machine.artifacts import ArtifactId


@dataclass(frozen=True)
class ArtifactFileSpec:
    json: dict[str, str] = field(default_factory=dict)
    parquet: dict[str, str] = field(default_factory=dict)
    pickle: dict[str, str] = field(default_factory=dict)

    def all_filenames(self) -> frozenset[str]:
        return frozenset([*self.json.values(), *self.parquet.values(), *self.pickle.values()])


ARTIFACT_FILE_SPECS: dict[ArtifactId, ArtifactFileSpec] = {
    "question": ArtifactFileSpec(json={"question": "question.json"}),
    "raw_data": ArtifactFileSpec(
        json={"profile": "profile.json"},
        parquet={"raw": "raw.parquet"},
    ),
    "latent_structure": ArtifactFileSpec(json={"latent_structure": "latent-structure.json"}),
    "measurement_structure": ArtifactFileSpec(
        json={"measurement_structure": "measurement_structure.json"}
    ),
    "causal_design": ArtifactFileSpec(json={"causal_design": "causal_design.json"}),
    "structural_plan": ArtifactFileSpec(json={"structural_plan": "structural-plan.json"}),
    "identification_report": ArtifactFileSpec(
        json={"identification_report": "identification_report.json"}
    ),
    "measurements": ArtifactFileSpec(json={"measurements": "measurements.json"}),
    "panel": ArtifactFileSpec(parquet={"panel": "panel.parquet"}),
    "validation_report": ArtifactFileSpec(json={"validation_report": "validation_report.json"}),
    "statistical_model_spec": ArtifactFileSpec(
        json={"statistical_model_spec": "statistical_model_spec.json"}
    ),
    "compiled_ssm": ArtifactFileSpec(
        json={"compiled_ssm": "compiled-ssm.json", "report": "report.json"}
    ),
    "posterior": ArtifactFileSpec(
        json={"diagnostics": "diagnostics.json"},
        pickle={"fitted": "fitted.pkl"},
    ),
    "baseline_report": ArtifactFileSpec(json={"baseline_report": "baseline_report.json"}),
    "saved_scenarios": ArtifactFileSpec(json={"saved_scenarios": "saved_scenarios.json"}),
}


def artifact_file_spec(artifact_id: ArtifactId) -> ArtifactFileSpec:
    return ARTIFACT_FILE_SPECS[artifact_id]


def json_filename(artifact_id: ArtifactId, key: str) -> str:
    return artifact_file_spec(artifact_id).json[key]


def parquet_filename(artifact_id: ArtifactId, key: str) -> str:
    return artifact_file_spec(artifact_id).parquet[key]


def pickle_filename(artifact_id: ArtifactId, key: str) -> str:
    return artifact_file_spec(artifact_id).pickle[key]


def is_declared_artifact_file(artifact_id: ArtifactId, filename: str) -> bool:
    return filename in artifact_file_spec(artifact_id).all_filenames()
