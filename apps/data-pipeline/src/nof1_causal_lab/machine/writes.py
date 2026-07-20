"""Write-move executors: schema-validated artifact writes.

Writes are the only way a caller bypasses a delegated transition. They are not
raw file writes: each executor validates the public payload shape, stamps
human/LLM provenance, pins any existing contextual inputs declared by the
artifact graph, and then runs the same derivation cascade as a computed
transition result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nof1_causal_lab.machine.artifact_files import json_filename
from nof1_causal_lab.machine.derivations import complete_derivation_cascade
from nof1_causal_lab.machine.errors import ArtifactWriteRejected
from nof1_causal_lab.machine.graph import ROOTS, transition_spec
from nof1_causal_lab.machine.moves import TransitionEffects, write_pins
from nof1_causal_lab.machine.store import ArtifactStore

if TYPE_CHECKING:
    from pydantic import BaseModel

    from nof1_causal_lab.machine.artifacts import (
        ArtifactId,
        ArtifactVersionInfo,
        EpisodeState,
        Provenance,
    )


def _validated(artifact_id: ArtifactId, model_cls: type[BaseModel], payload: dict) -> dict:
    try:
        return model_cls.model_validate(payload).model_dump(mode="json")
    except Exception as exc:
        raise ArtifactWriteRejected(str(exc), artifact_id=artifact_id) from exc


def _write_question(
    store: ArtifactStore,
    payload: dict[str, Any],
    provenance: Provenance,
) -> ArtifactVersionInfo:
    text = payload.get("text", "")
    if not isinstance(text, str) or not text.strip():
        raise ArtifactWriteRejected(
            "question payload must be {'text': <non-empty string>}", artifact_id="question"
        )
    return store.write_version(
        "question",
        provenance=provenance,
        derived_from={},
        produced_by=None,
        json_files={json_filename("question", "question"): {"text": text.strip()}},
    )


def _write_saved_scenarios(
    store: ArtifactStore,
    state: EpisodeState,
    payload: dict[str, Any],
    provenance: Provenance,
) -> ArtifactVersionInfo:
    from nof1_causal_lab.flows.transitions.analysis.contracts import SavedScenarioContract

    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list):
        raise ArtifactWriteRejected(
            "saved_scenarios payload must be {'scenarios': [...]}",
            artifact_id="saved_scenarios",
        )
    validated = [
        _validated("saved_scenarios", SavedScenarioContract, scenario) for scenario in scenarios
    ]
    roots = {root.artifact_id: root for root in ROOTS}
    return store.write_version(
        "saved_scenarios",
        provenance=provenance,
        derived_from=write_pins(state, roots["saved_scenarios"].write_pins),
        produced_by=None,
        json_files={json_filename("saved_scenarios", "saved_scenarios"): {"scenarios": validated}},
    )


_CONTRACT_WRITES: dict[ArtifactId, tuple[type[BaseModel] | str, str]] = {
    "latent_structure": (
        "LatentStructureContract",
        json_filename("latent_structure", "latent_structure"),
    ),
    "measurement_structure": (
        "MeasurementStructureContract",
        json_filename("measurement_structure", "measurement_structure"),
    ),
    "statistical_model_spec": (
        "StatisticalModelSpecContract",
        json_filename("statistical_model_spec", "statistical_model_spec"),
    ),
    "baseline_report": (
        "BaselineReportContract",
        json_filename("baseline_report", "baseline_report"),
    ),
}


def _contract_class(name: str) -> type[BaseModel]:
    from nof1_causal_lab.flows import artifact_contracts

    return getattr(artifact_contracts, name)


def _write_contract_artifact(
    store: ArtifactStore,
    state: EpisodeState,
    artifact_id: ArtifactId,
    payload: dict[str, Any],
    provenance: Provenance,
) -> ArtifactVersionInfo:
    contract_ref, filename = _CONTRACT_WRITES[artifact_id]
    contract = _contract_class(contract_ref) if isinstance(contract_ref, str) else contract_ref
    validated = _validated(artifact_id, contract, payload)
    pins = write_pins(state, transition_spec(artifact_id).consumes)
    return store.write_version(
        artifact_id,
        provenance=provenance,
        derived_from=pins,
        produced_by=None,
        json_files={filename: validated},
    )


def execute_write(
    workspace_id: str,
    artifact_id: ArtifactId,
    payload: dict[str, Any],
    provenance: Provenance,
    state: EpisodeState,
) -> TransitionEffects:
    """Validate, persist, and cascade a write move.

    A cascade failure removes versions written during this failed move before
    re-raising, so rejected writes do not become current and do not leave
    listable orphan versions.
    """
    store = ArtifactStore(workspace_id)
    if artifact_id == "question":
        info = _write_question(store, payload, provenance)
        return complete_derivation_cascade(store, state, [info])
    if artifact_id == "saved_scenarios":
        info = _write_saved_scenarios(store, state, payload, provenance)
        return complete_derivation_cascade(store, state, [info])
    if artifact_id in _CONTRACT_WRITES:
        info = _write_contract_artifact(store, state, artifact_id, payload, provenance)
        return complete_derivation_cascade(store, state, [info])
    raise ArtifactWriteRejected(
        f"artifact '{artifact_id}' has no write executor",
        artifact_id=artifact_id,
    )
