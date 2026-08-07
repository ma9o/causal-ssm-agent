"""Deterministic derivation cascade for machine-maintained artifacts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001
from nof1_causal_lab.machine.artifact_files import json_filename, parquet_filename
from nof1_causal_lab.machine.graph import Derivation, topological_derivation_order, transition_spec
from nof1_causal_lab.machine.moves import (
    RetractedArtifact,
    TransitionEffects,
    apply_transition,
    is_stale,
    run_retractions,
)

if TYPE_CHECKING:
    import polars as pl

    from nof1_causal_lab.artifacts.structural_plan import StructuralPlan
    from nof1_causal_lab.machine.artifacts import ArtifactId, ArtifactVersionInfo, EpisodeState
    from nof1_causal_lab.machine.store import ArtifactStore


def complete_computed_transition(
    store: ArtifactStore,
    state: EpisodeState,
    transition_id: ArtifactId,
    produced: list[ArtifactVersionInfo],
) -> TransitionEffects:
    """Apply optional-output retractions and the derivation cascade for one run."""
    retracted = run_retractions(state, transition_spec(transition_id), produced)
    return complete_derivation_cascade(store, state, produced, retracted)


def complete_derivation_cascade(
    store: ArtifactStore,
    state: EpisodeState,
    produced: list[ArtifactVersionInfo],
    retracted: list[RetractedArtifact] | None = None,
) -> TransitionEffects:
    """Apply initial effects to a temporary state, then derive reachable nodes."""
    all_produced = list(produced)
    all_retracted = list(retracted or [])
    next_state = apply_transition(state, all_produced, all_retracted)
    affected: set[ArtifactId] = {info.artifact_id for info in produced} | {
        item.artifact_id for item in all_retracted
    }

    try:
        for spec in topological_derivation_order():
            parents = _current_parent_versions(next_state, spec)
            if parents is None:
                retraction = _retract_current(
                    next_state,
                    spec.produces,
                    reason_ref=f"{spec.produces}.parents_absent",
                )
                if retraction is not None:
                    all_retracted.append(retraction)
                    next_state = next_state.without([spec.produces])
                    affected.add(spec.produces)
                continue

            stale_parents = [parent for parent in spec.from_ if is_stale(next_state, parent)]
            if stale_parents:
                retraction = _retract_current(
                    next_state,
                    spec.produces,
                    reason_ref=f"{spec.produces}.parents_stale.{stale_parents[0]}",
                )
                if retraction is not None:
                    all_retracted.append(retraction)
                    next_state = next_state.without([spec.produces])
                    affected.add(spec.produces)
                continue

            if not affected.intersection(spec.from_):
                continue

            info = _derive_one(store, spec, parents)
            if info is None:
                retraction = _retract_current(
                    next_state,
                    spec.produces,
                    reason_ref=_empty_finding_reason(spec.produces),
                )
                if retraction is not None:
                    all_retracted.append(retraction)
                    next_state = next_state.without([spec.produces])
                    affected.add(spec.produces)
                continue

            all_produced.append(info)
            next_state = next_state.with_versions([info])
            affected.add(spec.produces)
    except Exception:
        for info in reversed(all_produced):
            store.delete_version(info.artifact_id, info.version)
        raise

    return TransitionEffects(produced=all_produced, retracted=all_retracted)


def _retract_current(
    state: EpisodeState,
    artifact_id: ArtifactId,
    *,
    reason_ref: str,
) -> RetractedArtifact | None:
    if not state.has(artifact_id):
        return None
    return RetractedArtifact(artifact_id=artifact_id, reason_ref=reason_ref)


def _current_parent_versions(
    state: EpisodeState,
    spec: Derivation,
) -> dict[ArtifactId, int] | None:
    pins: dict[ArtifactId, int] = {}
    for parent in spec.from_:
        info = state.get(parent)
        if info is None:
            return None
        pins[parent] = info.version
    return pins


def _derive_one(
    store: ArtifactStore,
    spec: Derivation,
    pins: dict[ArtifactId, int],
) -> ArtifactVersionInfo | None:
    if spec.produces == "causal_design":
        return _derive_causal_design(store, pins)
    if spec.produces == "structural_plan":
        return _derive_structural_plan(store, pins)
    if spec.produces == "identification_report":
        return _derive_identification_report(store, pins)
    if spec.produces == "validation_report":
        return _derive_validation_report(store, pins)
    if spec.produces == "compiled_ssm":
        return _derive_compiled_ssm(store, pins)
    raise AssertionError(f"No derivation body for {spec.produces}")


def _read_latent_structure(store: ArtifactStore, version: int) -> UncheckedJsonObject:
    payload = store.read_json_file(
        "latent_structure",
        version,
        json_filename("latent_structure", "latent_structure"),
    )
    return payload["latent_structure"]


def _read_measurement_structure_contract(
    store: ArtifactStore,
    version: int,
) -> UncheckedJsonObject:
    return store.read_json_file(
        "measurement_structure",
        version,
        json_filename("measurement_structure", "measurement_structure"),
    )


def _read_causal_design(store: ArtifactStore, version: int) -> UncheckedJsonObject:
    payload = store.read_json_file(
        "causal_design",
        version,
        json_filename("causal_design", "causal_design"),
    )
    return payload["causal_design"]


def _read_structural_plan(store: ArtifactStore, version: int) -> StructuralPlan:
    from nof1_causal_lab.artifacts.structural_plan import StructuralPlan

    payload = store.read_json_file(
        "structural_plan",
        version,
        json_filename("structural_plan", "structural_plan"),
    )
    return StructuralPlan.model_validate(payload["structural_plan"])


def _read_panel(store: ArtifactStore, version: int) -> pl.DataFrame:
    return store.read_parquet_file("panel", version, parquet_filename("panel", "panel"))


def _derive_causal_design(
    store: ArtifactStore,
    pins: dict[ArtifactId, int],
) -> ArtifactVersionInfo:
    from nof1_causal_lab.flows.transitions.measurement_structure.assemble import build_causal_design
    from nof1_causal_lab.utils.identifiability import check_identifiability

    latent_structure = _read_latent_structure(store, pins["latent_structure"])
    measurement_contract = _read_measurement_structure_contract(
        store,
        pins["measurement_structure"],
    )
    measurement_structure = measurement_contract["measurement_structure"]
    known_inputs = measurement_contract["known_inputs"]
    scientific_only_constructs = measurement_contract["scientific_only_constructs"]
    id_result = check_identifiability(latent_structure, measurement_structure)
    id_status = {
        "identifiable_treatments": id_result.get("identifiable_treatments", {}),
        "non_identifiable_treatments": id_result.get("non_identifiable_treatments", {}),
    }
    causal_design = build_causal_design(
        latent_structure,
        measurement_structure,
        id_status,
        known_inputs=known_inputs,
        scientific_only_constructs=scientific_only_constructs,
    )
    return store.write_version(
        "causal_design",
        provenance="computed",
        derived_from=pins,
        produced_by="derive:causal_design",
        json_files={
            json_filename("causal_design", "causal_design"): {
                "causal_design": causal_design.model_dump(mode="json")
            }
        },
    )


def _derive_structural_plan(
    store: ArtifactStore,
    pins: dict[ArtifactId, int],
) -> ArtifactVersionInfo:
    from nof1_causal_lab.artifacts.causal_design import CausalDesign
    from nof1_causal_lab.models.structural import build_structural_plan

    causal_design = CausalDesign.model_validate(_read_causal_design(store, pins["causal_design"]))
    structural_plan = build_structural_plan(causal_design)
    return store.write_version(
        "structural_plan",
        provenance="computed",
        derived_from=pins,
        produced_by="derive:structural_plan",
        json_files={
            json_filename("structural_plan", "structural_plan"): {
                "structural_plan": structural_plan.model_dump(mode="json")
            }
        },
    )


def _derive_identification_report(
    store: ArtifactStore,
    pins: dict[ArtifactId, int],
) -> ArtifactVersionInfo | None:
    from nof1_causal_lab.flows.transitions.measurement_structure.contracts import (
        IdentificationReportContract,
    )
    from nof1_causal_lab.flows.transitions.measurement_structure.identification import (
        derive_identification_report,
    )

    causal_design = _read_causal_design(store, pins["causal_design"])
    report = derive_identification_report(causal_design)
    if report is None:
        return None
    validated = IdentificationReportContract.model_validate(report).model_dump(mode="json")
    return store.write_version(
        "identification_report",
        provenance="computed",
        derived_from=pins,
        produced_by="derive:identification_report",
        json_files={json_filename("identification_report", "identification_report"): validated},
    )


def _derive_validation_report(
    store: ArtifactStore,
    pins: dict[ArtifactId, int],
) -> ArtifactVersionInfo:
    from nof1_causal_lab.flows.artifact_contracts import ValidationReportContract
    from nof1_causal_lab.flows.transitions.validation.flow import (
        derive_validation_status,
        validate_extraction,
    )

    causal_design = _read_causal_design(store, pins["causal_design"])
    panel = _read_panel(store, pins["panel"])
    audit_result = validate_extraction(causal_design, [panel])
    if not audit_result:
        raise RuntimeError(
            "validation_report derivation returned an empty audit result; "
            "refusing to fabricate an is_valid=False report with empty indicators."
        )

    indicator_issues = [
        issue
        for audit in audit_result.get("indicators", {}).values()
        for issue in audit.get("validation", {}).get("issues", [])
    ]
    dataset_issues = audit_result.get("dataset_issues", [])
    status = derive_validation_status([*indicator_issues, *dataset_issues])
    payload = ValidationReportContract.model_validate(
        {**audit_result, "is_valid": status["is_valid"]}
    ).model_dump(mode="json")
    return store.write_version(
        "validation_report",
        provenance="computed",
        derived_from=pins,
        produced_by="derive:validation_report",
        json_files={json_filename("validation_report", "validation_report"): payload},
    )


def _derive_compiled_ssm(
    store: ArtifactStore,
    pins: dict[ArtifactId, int],
) -> ArtifactVersionInfo:
    from nof1_causal_lab.artifacts.statistical_model_spec import StatisticalModelSpec
    from nof1_causal_lab.models.ssm.compile.artifact import compile_ssm_artifact
    from nof1_causal_lab.workers.prior_research import build_prior_plan_from_payloads

    structural_plan = _read_structural_plan(store, pins["structural_plan"])
    report = store.read_json_file(
        "statistical_model_spec",
        pins["statistical_model_spec"],
        json_filename("statistical_model_spec", "statistical_model_spec"),
    )
    statistical_model_spec = StatisticalModelSpec.model_validate(report["statistical_model_spec"])
    compiled_ssm = compile_ssm_artifact(
        statistical_model_spec,
        build_prior_plan_from_payloads(
            statistical_model_spec,
            report["authored_priors"],
        ),
        structural_plan=structural_plan,
    )
    return store.write_version(
        "compiled_ssm",
        provenance="computed",
        derived_from=pins,
        produced_by="derive:compiled_ssm",
        json_files={
            json_filename("compiled_ssm", "compiled_ssm"): compiled_ssm.model_dump(mode="json"),
            json_filename("compiled_ssm", "report"): report,
        },
    )


def _empty_finding_reason(artifact_id: ArtifactId) -> str:
    if artifact_id == "identification_report":
        return "causal_design.identifiability.identifiable_treatments"
    raise AssertionError(f"Unexpected optional derivation with empty finding: {artifact_id}")
