"""Write-move executors: schema-validated artifact writes.

A ``write`` is how state enters the machine from outside a stage run —
the user's question, an edited causal design, saved scenarios. Payloads are
validated against the artifact's contract (the old override adapters'
coercion logic, kept as the schema boundary it always was), stamped with
the caller's provenance, and journaled like any other transition.

Writing ``causal_design`` fans out: the positive ``identification_report`` is
recomputed from the spec's explicit identifiability status in the same write
— otherwise an edited spec would leave downstream enabledness keyed to
superseded findings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic import BaseModel


from nof1_causal_lab.machine.artifact_files import json_filename
from nof1_causal_lab.machine.artifacts import (  # noqa: TC001 (pydantic field annotations)
    ArtifactId,
    Provenance,
)
from nof1_causal_lab.machine.errors import ArtifactWriteRejected
from nof1_causal_lab.machine.moves import TransitionEffects
from nof1_causal_lab.machine.store import ArtifactStore


def _validated(artifact_id: ArtifactId, model_cls: type[BaseModel], payload: dict) -> dict:
    try:
        return model_cls.model_validate(payload).model_dump(mode="json")
    except Exception as exc:
        raise ArtifactWriteRejected(str(exc), artifact_id=artifact_id) from exc


def _write_question(
    store: ArtifactStore, payload: dict[str, Any], provenance: Provenance
) -> TransitionEffects:
    text = payload.get("text", "")
    if not isinstance(text, str) or not text.strip():
        raise ArtifactWriteRejected(
            "question payload must be {'text': <non-empty string>}", artifact_id="question"
        )
    info = store.write_version(
        "question",
        provenance=provenance,
        derived_from={},
        produced_by=None,
        json_files={json_filename("question", "question"): {"text": text.strip()}},
    )
    return TransitionEffects(produced=[info])


def _write_causal_design(
    store: ArtifactStore, payload: dict[str, Any], provenance: Provenance
) -> TransitionEffects:
    from nof1_causal_lab.flows.stage_contracts import Stage1bContract
    from nof1_causal_lab.flows.stages.stage1b.contracts import IdentificationReportContract
    from nof1_causal_lab.flows.stages.stage1b.result import derive_identification_report

    validated = _validated("causal_design", Stage1bContract, payload)
    spec_info = store.write_version(
        "causal_design",
        provenance=provenance,
        derived_from={},
        produced_by=None,
        json_files={json_filename("causal_design", "causal_design"): validated},
    )
    report = derive_identification_report(validated["causal_design"])
    produced = [spec_info]
    retracted: list[ArtifactId] = []
    if report is not None:
        produced.append(
            store.write_version(
                "identification_report",
                provenance=provenance,
                derived_from={"causal_design": spec_info.version},
                produced_by=None,
                json_files={
                    json_filename("identification_report", "identification_report"): (
                        _validated("identification_report", IdentificationReportContract, report)
                    )
                },
            )
        )
    else:
        retracted.append("identification_report")

    return TransitionEffects(produced=produced, retracted=retracted)


def _write_saved_scenarios(
    store: ArtifactStore, payload: dict[str, Any], provenance: Provenance
) -> TransitionEffects:
    from nof1_causal_lab.flows.stages.stage6.contracts import SavedScenarioContract

    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list):
        raise ArtifactWriteRejected(
            "saved_scenarios payload must be {'scenarios': [...]}",
            artifact_id="saved_scenarios",
        )
    validated = [
        _validated("saved_scenarios", SavedScenarioContract, scenario) for scenario in scenarios
    ]
    # Pin the posterior version the scenarios were simulated against: when
    # the posterior moves, staleness surfaces through the same provenance
    # mechanism as everything else.
    derived_from: dict[ArtifactId, int] = {}
    posterior_versions = store.list_versions("posterior")
    if posterior_versions:
        derived_from["posterior"] = posterior_versions[-1]
    info = store.write_version(
        "saved_scenarios",
        provenance=provenance,
        derived_from=derived_from,
        produced_by=None,
        json_files={json_filename("saved_scenarios", "saved_scenarios"): {"scenarios": validated}},
    )
    return TransitionEffects(produced=[info])


_CONTRACT_WRITES: dict[ArtifactId, tuple[str, str]] = {
    # artifact -> (contract import name, payload filename). ``identification_report``
    # is absent by design: it is a derived milestone of ``causal_design`` (see
    # ``_write_causal_design``), recomputed on every spec creation, never written
    # directly.
    "latent_structure": (
        "Stage1aContract",
        json_filename("latent_structure", "latent_structure"),
    ),
    "extraction_report": (
        "Stage2Contract",
        json_filename("extraction_report", "extraction_report"),
    ),
    "validation_report": (
        "Stage3Contract",
        json_filename("validation_report", "validation_report"),
    ),
    "baseline_ranking": ("Stage6Contract", json_filename("baseline_ranking", "baseline_ranking")),
}


def _contract_class(name: str) -> type[BaseModel]:
    from nof1_causal_lab.flows import stage_contracts
    from nof1_causal_lab.flows.stages.stage1b import contracts as stage1b_contracts

    if hasattr(stage_contracts, name):
        return getattr(stage_contracts, name)
    return getattr(stage1b_contracts, name)


def execute_write(
    workspace_id: str,
    artifact_id: ArtifactId,
    payload: dict[str, Any],
    provenance: Provenance,
) -> TransitionEffects:
    """Validate and persist a write move; raises ArtifactWriteRejected."""
    store = ArtifactStore(workspace_id)
    if artifact_id == "question":
        return _write_question(store, payload, provenance)
    if artifact_id == "causal_design":
        return _write_causal_design(store, payload, provenance)
    if artifact_id == "saved_scenarios":
        return _write_saved_scenarios(store, payload, provenance)
    if artifact_id in _CONTRACT_WRITES:
        contract_name, filename = _CONTRACT_WRITES[artifact_id]
        validated = _validated(artifact_id, _contract_class(contract_name), payload)
        info = store.write_version(
            artifact_id,
            provenance=provenance,
            derived_from={},
            produced_by=None,
            json_files={filename: validated},
        )
        return TransitionEffects(produced=[info])
    raise ArtifactWriteRejected(
        f"artifact '{artifact_id}' has no write executor (binary payloads are "
        "produced by stages, not written directly)",
        artifact_id=artifact_id,
    )
