"""Immutable checkpoints for the Temporal statistical-model-spec workflow.

Checkpoints are execution sidecars, not public episode artifacts.  They contain
only the semantic state needed to reconstruct the construct-admission reducer;
large and executable inputs are always reloaded from their pinned artifact
versions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.artifacts.structural_plan import StructuralPlan
from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001
from nof1_causal_lab.machine.artifacts import ArtifactId  # noqa: TC001
from nof1_causal_lab.machine.moves import RunArtifact
from nof1_causal_lab.machine.store import EpisodeJournal, utc_now_iso
from nof1_causal_lab.utils import data as data_module
from nof1_causal_lab.utils import storage

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

CHECKPOINT_REF_PREFIX = "model-spec-checkpoint:"
CHECKPOINT_SCHEMA_VERSION = 3
ADMISSION_EVALUATION_SCHEMA_VERSION = 1
ADMISSION_ENGINE_VERSION = 2


class AcceptedConstructCheckpoint(BaseModel):
    """One construct submission that passed the admission battery."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    submission_id: str
    construct_name: str
    indicators: list[UncheckedJsonObject] = Field(default_factory=list)
    priors: dict[str, UncheckedJsonObject] = Field(default_factory=dict)
    accept: list[dict[str, str]] = Field(default_factory=list)
    annotations: list[str] = Field(default_factory=list)
    results: list[UncheckedJsonObject] = Field(default_factory=list)
    outcome: str
    feedback: str


class ModelSpecRebaseSummary(BaseModel):
    """How a new run reused a previous run's accepted dependency-closed set."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_checkpoint_ref: str
    pins_changed: bool
    retained_constructs: list[str] = Field(default_factory=list)
    reopened_construct: str | None = None
    reason: str | None = None


class ModelSpecCheckpoint(BaseModel):
    """Immutable accepted state for one point in a Stage 4 run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[3] = CHECKPOINT_SCHEMA_VERSION
    workspace_id: str
    run_id: str
    seq: int
    checkpoint_index: int
    parent_ref: str | None = None
    input_pins: dict[ArtifactId, int]
    accepted_constructs: list[AcceptedConstructCheckpoint] = Field(default_factory=list)
    search_queries: dict[str, str] = Field(default_factory=dict)
    search_cache: dict[str, str] = Field(default_factory=dict)
    repair_feedback: dict[str, str] = Field(default_factory=dict)
    full_model_validated: bool = False
    rebase: ModelSpecRebaseSummary | None = None
    created_at: str


class ModelSpecSubmissionResult(BaseModel):
    """Idempotent result of one ``submit_construct`` request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    submission_id: str
    construct_name: str
    admitted: bool
    outcome: str
    feedback: str
    checkpoint_ref: str | None = None
    error: str | None = None


class ModelSpecAdmissionEvaluation(BaseModel):
    """Content-addressed result of one exact construct evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = ADMISSION_EVALUATION_SCHEMA_VERSION
    evaluation_key: str
    construct_name: str
    admitted: bool
    outcome: str
    feedback: str
    annotations: list[str] = Field(default_factory=list)
    results: list[UncheckedJsonObject] = Field(default_factory=list)
    error: str | None = None


def model_spec_admission_evaluation_key(
    *,
    input_identity: Mapping[str, Any],
    accepted_constructs: Sequence[AcceptedConstructCheckpoint],
    ancestor_constructs: set[str],
    construct_name: str,
    indicators: list[UncheckedJsonObject],
    priors: dict[str, UncheckedJsonObject],
    accept: list[dict[str, str]],
    n_draws: int,
    seed: int,
) -> str:
    """Fingerprint every semantic input to the exact admission evaluation."""
    accepted = [
        {
            "construct_name": item.construct_name,
            "indicators": item.indicators,
            "priors": item.priors,
        }
        for item in accepted_constructs
        if item.construct_name in ancestor_constructs
    ]
    payload = {
        "schema_version": ADMISSION_EVALUATION_SCHEMA_VERSION,
        "engine_version": ADMISSION_ENGINE_VERSION,
        "input_identity": input_identity,
        "accepted_ancestors": accepted,
        "proposal": {
            "construct_name": construct_name,
            "indicators": indicators,
            "priors": priors,
            "accept": sorted(
                accept,
                key=lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")),
            ),
        },
        "n_draws": n_draws,
        "seed": seed,
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def model_spec_admission_evaluation_path(workspace_id: str, evaluation_key: str) -> str:
    return storage.join(
        data_module.cache_dir(workspace_id),
        "admission-evaluations",
        f"v{ADMISSION_EVALUATION_SCHEMA_VERSION}",
        f"{evaluation_key}.json",
    )


def read_model_spec_admission_evaluation(path: str) -> ModelSpecAdmissionEvaluation:
    return ModelSpecAdmissionEvaluation.model_validate(storage.read_json(path))


def write_model_spec_admission_evaluation(
    path: str,
    evaluation: ModelSpecAdmissionEvaluation,
) -> None:
    """Write an immutable evaluation, rejecting semantic hash collisions."""
    if storage.exists(path):
        existing = read_model_spec_admission_evaluation(path)
        if existing != evaluation:
            raise ValueError(f"Admission evaluation path collision at {path}")
        return
    storage.write_text(path, evaluation.model_dump_json())


def _checkpoint_dir(workspace_id: str, run_id: str) -> str:
    return storage.join(
        data_module.scratch_run_dir(workspace_id, run_id),
        "checkpoints",
    )


def _checkpoint_ref(run_id: str, checkpoint_id: str) -> str:
    if "/" in run_id or not run_id:
        raise ValueError(f"Invalid model-spec checkpoint run id: {run_id!r}")
    if "/" in checkpoint_id or not checkpoint_id:
        raise ValueError(f"Invalid model-spec checkpoint id: {checkpoint_id!r}")
    return f"{CHECKPOINT_REF_PREFIX}{run_id}/{checkpoint_id}"


def _parse_checkpoint_ref(ref: str) -> tuple[str, str]:
    if not ref.startswith(CHECKPOINT_REF_PREFIX):
        raise ValueError(f"Not a model-spec checkpoint ref: {ref!r}")
    parts = ref[len(CHECKPOINT_REF_PREFIX) :].split("/")
    if len(parts) != 2 or not all(parts) or any(part in {".", ".."} for part in parts):
        raise ValueError(f"Invalid model-spec checkpoint ref: {ref!r}")
    return parts[0], parts[1]


def _checkpoint_path(workspace_id: str, ref: str) -> str:
    run_id, checkpoint_id = _parse_checkpoint_ref(ref)
    return storage.join(_checkpoint_dir(workspace_id, run_id), checkpoint_id)


def read_model_spec_checkpoint(workspace_id: str, ref: str) -> ModelSpecCheckpoint:
    return ModelSpecCheckpoint.model_validate(
        storage.read_json(_checkpoint_path(workspace_id, ref))
    )


def _write_checkpoint(path: str, checkpoint: ModelSpecCheckpoint) -> str:
    ref = _checkpoint_ref(checkpoint.run_id, path.rsplit("/", 1)[-1])
    if storage.exists(path):
        existing = ModelSpecCheckpoint.model_validate(storage.read_json(path))
        expected = checkpoint.model_copy(update={"created_at": existing.created_at})
        if existing != expected:
            raise ValueError(f"Checkpoint path collision at {path}")
        return ref
    storage.write_text(path, checkpoint.model_dump_json())
    return ref


def write_initial_model_spec_checkpoint(
    *,
    workspace_id: str,
    run_id: str,
    seq: int,
    pins: dict[ArtifactId, int],
    accepted_constructs: list[AcceptedConstructCheckpoint],
    search_queries: dict[str, str],
    search_cache: dict[str, str],
    repair_feedback: dict[str, str] | None,
    parent_ref: str | None,
    rebase: ModelSpecRebaseSummary | None,
) -> str:
    path = storage.join(_checkpoint_dir(workspace_id, run_id), "checkpoint-000000.json")
    checkpoint = ModelSpecCheckpoint(
        workspace_id=workspace_id,
        run_id=run_id,
        seq=seq,
        checkpoint_index=0,
        parent_ref=parent_ref,
        input_pins=pins,
        accepted_constructs=accepted_constructs,
        search_queries=search_queries,
        search_cache=search_cache,
        repair_feedback=repair_feedback or {},
        full_model_validated=False,
        rebase=rebase,
        created_at=utc_now_iso(),
    )
    return _write_checkpoint(path, checkpoint)


def accepted_checkpoint_path(
    checkpoint: ModelSpecCheckpoint,
    submission_id: str,
) -> str:
    digest = hashlib.sha256(submission_id.encode()).hexdigest()[:12]
    return storage.join(
        _checkpoint_dir(checkpoint.workspace_id, checkpoint.run_id),
        f"checkpoint-{checkpoint.checkpoint_index + 1:06d}-{digest}.json",
    )


def existing_accepted_checkpoint_ref(
    checkpoint: ModelSpecCheckpoint,
    submission_id: str,
) -> str | None:
    path = accepted_checkpoint_path(checkpoint, submission_id)
    return (
        _checkpoint_ref(checkpoint.run_id, path.rsplit("/", 1)[-1])
        if storage.exists(path)
        else None
    )


def write_accepted_model_spec_checkpoint(
    *,
    parent_ref: str,
    parent: ModelSpecCheckpoint,
    accepted: AcceptedConstructCheckpoint,
    search_queries: dict[str, str],
    search_cache: dict[str, str],
) -> str:
    path = accepted_checkpoint_path(parent, accepted.submission_id)
    checkpoint = ModelSpecCheckpoint(
        workspace_id=parent.workspace_id,
        run_id=parent.run_id,
        seq=parent.seq,
        checkpoint_index=parent.checkpoint_index + 1,
        parent_ref=parent_ref,
        input_pins=parent.input_pins,
        accepted_constructs=[*parent.accepted_constructs, accepted],
        search_queries=search_queries,
        search_cache=search_cache,
        repair_feedback={
            name: feedback
            for name, feedback in parent.repair_feedback.items()
            if name != accepted.construct_name
        },
        full_model_validated=False,
        created_at=utc_now_iso(),
    )
    return _write_checkpoint(path, checkpoint)


def write_merged_model_spec_checkpoint(
    *,
    parent_ref: str,
    parent: ModelSpecCheckpoint,
    checkpoint_id: str,
    accepted_constructs: list[AcceptedConstructCheckpoint],
    search_queries: dict[str, str],
    search_cache: dict[str, str],
    repair_feedback: dict[str, str] | None = None,
    full_model_validated: bool = False,
) -> str:
    """Write one deterministic single-writer checkpoint after a frontier join."""
    path = accepted_checkpoint_path(parent, checkpoint_id)
    checkpoint = ModelSpecCheckpoint(
        workspace_id=parent.workspace_id,
        run_id=parent.run_id,
        seq=parent.seq,
        checkpoint_index=parent.checkpoint_index + 1,
        parent_ref=parent_ref,
        input_pins=parent.input_pins,
        accepted_constructs=accepted_constructs,
        search_queries=search_queries,
        search_cache=search_cache,
        repair_feedback=repair_feedback or {},
        full_model_validated=full_model_validated,
        created_at=utc_now_iso(),
    )
    return _write_checkpoint(path, checkpoint)


def latest_failed_model_spec_checkpoint_ref(workspace_id: str) -> str | None:
    """Latest resumable checkpoint advertised by a raised Stage 4 move."""
    for record in reversed(EpisodeJournal(workspace_id).read_all()):
        if not (
            isinstance(record.move, RunArtifact)
            and record.move.artifact_id == "statistical_model_spec"
        ):
            continue
        if record.status == "applied":
            return None
        if record.status == "raised" and record.resume is not None:
            return _checkpoint_ref(record.resume.run_id, record.resume.checkpoint_id)
    return None


def restore_construct_state(
    checkpoint: ModelSpecCheckpoint,
    *,
    structural_plan: StructuralPlan,
    data_for_model: Any,
    workspace_id: str | None,
    target_construct: str | None = None,
):
    """Reconstruct reducer state without rerunning accepted admission checks."""
    from nof1_causal_lab.flows.transitions.model_spec.agentic.construct_flow import (
        ConstructBuildState,
        contribution_from_payload,
    )
    from nof1_causal_lab.models.ssm.construct_admission import (
        build_construct_order,
        build_construct_units,
        trial_admission_state,
    )

    global_order = build_construct_order(structural_plan)
    accepted_by_name = {saved.construct_name: saved for saved in checkpoint.accepted_constructs}
    if len(accepted_by_name) != len(checkpoint.accepted_constructs):
        raise ValueError("Checkpoint contains duplicate accepted constructs")
    unknown = set(accepted_by_name) - set(global_order)
    if unknown:
        raise ValueError(
            "Checkpoint contains constructs absent from the current structural plan: "
            + ", ".join(sorted(unknown))
        )
    if target_construct is not None:
        if target_construct not in global_order:
            raise ValueError(f"Unknown model-spec target construct {target_construct!r}")
        if target_construct in accepted_by_name:
            raise ValueError(
                f"Model-spec target construct {target_construct!r} is already accepted"
            )
        units = build_construct_units(structural_plan)
        unit_by_id = {unit.unit_id: unit for unit in units}
        target_unit = next(unit for unit in units if target_construct in unit.constructs)
        required_units: set[str] = set()
        pending = list(target_unit.predecessors)
        while pending:
            unit_id = pending.pop()
            if unit_id in required_units:
                continue
            required_units.add(unit_id)
            pending.extend(unit_by_id[unit_id].predecessors)
        required_constructs = {
            construct for unit_id in required_units for construct in unit_by_id[unit_id].constructs
        }
        required_constructs.update(
            construct for construct in target_unit.constructs if construct in accepted_by_name
        )
        accepted_order = [
            name
            for name in global_order
            if name in accepted_by_name and name in required_constructs
        ]
    else:
        accepted_order = [name for name in global_order if name in accepted_by_name]
    order = [*accepted_order, *([target_construct] if target_construct is not None else [])]
    state = ConstructBuildState(
        structural_plan=structural_plan,
        data_for_model=data_for_model,
        order=order,
        workspace_id=workspace_id,
    )
    state.search_queries = dict(checkpoint.search_queries)
    state.search_cache = dict(checkpoint.search_cache)
    for construct_name in accepted_order:
        saved = accepted_by_name[construct_name]
        if state.current_construct != construct_name:
            raise AssertionError(
                f"Scoped checkpoint restore expected {state.current_construct!r}, "
                f"found {construct_name!r}"
            )
        inventory = state.parameter_inventory_for(construct_name)
        contribution = contribution_from_payload(
            structural_plan,
            {
                "construct": saved.construct_name,
                "indicators": saved.indicators,
                "priors": saved.priors,
            },
            inventory.catalog,
        )
        trial = trial_admission_state(state.admission, contribution)
        state.admission = replace(
            trial,
            annotations=(*state.admission.annotations, *saved.annotations),
        )
        state.admitted_contributions[saved.construct_name] = contribution
        state.cursor += 1
    if target_construct is not None:
        state.last_tool_feedback = checkpoint.repair_feedback.get(target_construct)
    return state


def load_checkpoint_construct_state(
    workspace_id: str,
    checkpoint_ref: str,
    *,
    emit_workspace_id: str | None,
    target_construct: str,
):
    """Load pinned inputs and reconstruct one checkpoint's reducer state."""
    from nof1_causal_lab.machine.artifact_files import json_filename, parquet_filename
    from nof1_causal_lab.machine.store import ArtifactStore

    checkpoint = read_model_spec_checkpoint(workspace_id, checkpoint_ref)
    store = ArtifactStore(workspace_id)
    structural_plan = StructuralPlan.model_validate(
        store.read_json_file(
            "structural_plan",
            checkpoint.input_pins["structural_plan"],
            json_filename("structural_plan", "structural_plan"),
        )["structural_plan"]
    )
    data_for_model = store.read_parquet_file(
        "panel",
        checkpoint.input_pins["panel"],
        parquet_filename("panel", "panel"),
    )
    state = restore_construct_state(
        checkpoint,
        structural_plan=structural_plan,
        data_for_model=data_for_model,
        workspace_id=emit_workspace_id,
        target_construct=target_construct,
    )
    return checkpoint, state


def rebase_accepted_constructs(
    source: ModelSpecCheckpoint,
    *,
    structural_plan: StructuralPlan,
    data_for_model: Any,
) -> tuple[Any, list[AcceptedConstructCheckpoint], str | None, str | None]:
    """Replay saved units and invalidate only a failed unit and its descendants."""
    from nof1_causal_lab.models.ssm.construct_admission import (
        build_construct_order,
        build_construct_units,
    )

    order = build_construct_order(structural_plan)
    units = build_construct_units(structural_plan)
    unit_by_id = {unit.unit_id: unit for unit in units}
    unit_by_construct = {construct: unit.unit_id for unit in units for construct in unit.constructs}
    successors: dict[str, set[str]] = {unit.unit_id: set() for unit in units}
    for unit in units:
        for predecessor in unit.predecessors:
            successors[predecessor].add(unit.unit_id)

    def _descendant_units(unit_id: str) -> set[str]:
        pending = [unit_id]
        found: set[str] = set()
        while pending:
            current = pending.pop()
            if current in found:
                continue
            found.add(current)
            pending.extend(successors[current])
        return found

    saved_by_name = {saved.construct_name: saved for saved in source.accepted_constructs}
    retained: list[AcceptedConstructCheckpoint] = []
    reopened: str | None = None
    reason: str | None = None
    invalid_units: set[str] = set()
    for construct_name in order:
        saved = saved_by_name.get(construct_name)
        unit_id = unit_by_construct[construct_name]
        if unit_id in invalid_units:
            continue
        if saved is None:
            reopened = reopened or construct_name
            reason = reason or f"No saved contribution exists for {construct_name!r}"
            invalid_units.update(_descendant_units(unit_id))
            continue
        retained_names = {item.construct_name for item in retained}
        missing_predecessors = [
            name
            for predecessor in unit_by_id[unit_id].predecessors
            for name in unit_by_id[predecessor].constructs
            if name not in retained_names
        ]
        if missing_predecessors:
            reopened = reopened or construct_name
            reason = reason or (
                f"Saved contribution {construct_name!r} is missing retained predecessors: "
                + ", ".join(missing_predecessors)
            )
            invalid_units.update(_descendant_units(unit_id))
            continue
        scoped = source.model_copy(
            update={
                "accepted_constructs": list(retained),
                "repair_feedback": {},
            }
        )
        state = restore_construct_state(
            scoped,
            structural_plan=structural_plan,
            data_for_model=data_for_model,
            workspace_id=None,
            target_construct=construct_name,
        )
        state.attempt = 0
        state.submission_made = False
        try:
            feedback = state.submit_construct(
                construct=saved.construct_name,
                indicators=saved.indicators,
                priors=saved.priors,
                accept=saved.accept,
            )
        except (
            ArithmeticError,
            AssertionError,
            AttributeError,
            LookupError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            reopened = reopened or construct_name
            reason = reason or str(exc)
            invalid_units.update(_descendant_units(unit_id))
            continue
        if state.current_construct == saved.construct_name:
            reopened = reopened or construct_name
            reason = reason or feedback
            invalid_units.update(_descendant_units(unit_id))
            continue
        report = state.last_report
        if report is None or report.name != saved.construct_name or not report.admitted:
            raise AssertionError(
                f"Rebase admitted {saved.construct_name!r} without an admission report"
            )
        retained.append(
            saved.model_copy(
                update={
                    "annotations": list(report.annotations),
                    "outcome": report.outcome,
                    "feedback": feedback,
                }
            )
        )
    retained_checkpoint = source.model_copy(
        update={
            "accepted_constructs": retained,
            "repair_feedback": {},
        }
    )
    state = restore_construct_state(
        retained_checkpoint,
        structural_plan=structural_plan,
        data_for_model=data_for_model,
        workspace_id=None,
    )
    return state, retained, reopened, reason


__all__ = [
    "AcceptedConstructCheckpoint",
    "ModelSpecAdmissionEvaluation",
    "ModelSpecCheckpoint",
    "ModelSpecRebaseSummary",
    "ModelSpecSubmissionResult",
    "existing_accepted_checkpoint_ref",
    "latest_failed_model_spec_checkpoint_ref",
    "load_checkpoint_construct_state",
    "model_spec_admission_evaluation_key",
    "model_spec_admission_evaluation_path",
    "read_model_spec_admission_evaluation",
    "read_model_spec_checkpoint",
    "rebase_accepted_constructs",
    "restore_construct_state",
    "write_accepted_model_spec_checkpoint",
    "write_model_spec_admission_evaluation",
    "write_initial_model_spec_checkpoint",
]
