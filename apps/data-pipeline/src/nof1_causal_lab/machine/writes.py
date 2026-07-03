"""Write-move executors: schema-validated artifact writes.

A ``write`` is how state enters the machine from outside a stage run —
the user's question, an edited causal spec, saved scenarios. Payloads are
validated against the artifact's contract (the old override adapters'
coercion logic, kept as the schema boundary it always was), stamped with
the caller's provenance, and journaled like any other transition.

Writing ``causal_spec`` fans out: identification is pure computation over
the spec, so the derived ``identification_report`` / ``estimands``
artifacts are recomputed in the same write — otherwise an edited spec
would leave downstream enabledness keyed to superseded findings.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.machine.artifacts import (  # noqa: TC001 (pydantic field annotations)
    ArtifactId,
    ArtifactVersionInfo,
    Provenance,
)
from nof1_causal_lab.machine.errors import ArtifactWriteRejected
from nof1_causal_lab.machine.store import ArtifactStore


class WriteResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    produced: list[ArtifactVersionInfo] = Field(default_factory=list)
    # Optional artifacts withheld by a fan-out derivation (e.g. an edited
    # causal_spec under which nothing is estimable retracts estimands).
    retracted: list[ArtifactId] = Field(default_factory=list)


def _validated(artifact_id: ArtifactId, model_cls: type[BaseModel], payload: dict) -> dict:
    try:
        return model_cls.model_validate(payload).model_dump(mode="json")
    except Exception as exc:
        raise ArtifactWriteRejected(str(exc), artifact_id=artifact_id) from exc


def _write_question(
    store: ArtifactStore, payload: dict[str, Any], provenance: Provenance
) -> WriteResult:
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
        json_files={"question.json": {"text": text.strip()}},
    )
    return WriteResult(produced=[info])


def _write_causal_spec(
    store: ArtifactStore, payload: dict[str, Any], provenance: Provenance
) -> WriteResult:
    from nof1_causal_lab.flows.stage_contracts import Stage1bContract
    from nof1_causal_lab.flows.stages.stage1b.contracts import (
        EstimandsContract,
        IdentificationReportContract,
    )
    from nof1_causal_lab.flows.stages.stage1b.result import derive_identification_artifacts

    validated = _validated("causal_spec", Stage1bContract, payload)
    spec_info = store.write_version(
        "causal_spec",
        provenance=provenance,
        derived_from={},
        produced_by=None,
        json_files={"causal_spec.json": validated},
    )
    report, estimands = derive_identification_artifacts(validated["causal_spec"])
    produced = [
        spec_info,
        store.write_version(
            "identification_report",
            provenance=provenance,
            derived_from={"causal_spec": spec_info.version},
            produced_by=None,
            json_files={
                "identification_report.json": _validated(
                    "identification_report", IdentificationReportContract, report
                )
            },
        ),
    ]
    retracted: list[ArtifactId] = []
    if estimands is not None:
        produced.append(
            store.write_version(
                "estimands",
                provenance=provenance,
                derived_from={"causal_spec": spec_info.version},
                produced_by=None,
                json_files={
                    "estimands.json": _validated("estimands", EstimandsContract, estimands)
                },
            )
        )
    else:
        retracted.append("estimands")

    from nof1_causal_lab.flows.stage_persistence import persist_validated_web_result

    persist_validated_web_result("stage-1b", validated, store.workspace_id)
    return WriteResult(produced=produced, retracted=retracted)


def _write_saved_scenarios(
    store: ArtifactStore, payload: dict[str, Any], provenance: Provenance
) -> WriteResult:
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
        json_files={"saved_scenarios.json": {"scenarios": validated}},
    )
    return WriteResult(produced=[info])


_CONTRACT_WRITES: dict[ArtifactId, tuple[str, str]] = {
    # artifact -> (contract import name, payload filename)
    "constructs": ("Stage1aContract", "constructs.json"),
    "identification_report": ("IdentificationReportContract", "identification_report.json"),
    "estimands": ("EstimandsContract", "estimands.json"),
    "extraction_report": ("Stage2Contract", "extraction_report.json"),
    "validation_report": ("Stage3Contract", "validation_report.json"),
    "baseline_ranking": ("Stage6Contract", "baseline_ranking.json"),
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
) -> WriteResult:
    """Validate and persist a write move; raises ArtifactWriteRejected."""
    store = ArtifactStore(workspace_id)
    if artifact_id == "question":
        return _write_question(store, payload, provenance)
    if artifact_id == "causal_spec":
        return _write_causal_spec(store, payload, provenance)
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
        return WriteResult(produced=[info])
    raise ArtifactWriteRejected(
        f"artifact '{artifact_id}' has no write executor (binary payloads are "
        "produced by stages, not written directly)",
        artifact_id=artifact_id,
    )
