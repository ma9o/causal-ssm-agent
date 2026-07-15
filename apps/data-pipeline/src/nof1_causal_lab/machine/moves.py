"""Moves, legality, transition application, staleness, freshness.

The machine mutates only through artifact-named moves: ``run(artifact)`` for a
transition's primary output, or ``write(artifact)`` for roots and writable
judgment artifacts. Run legality is existence-only over declared inputs. Write
legality is schema/provenance-bound and the write executor owns any derivation
cascade that must complete before the move can become current.

Freshness is a derived query over version pins, not a stored flag. Produced and
written artifacts can be stale; derived artifacts are maintained by the cascade
and are either current or absent on the public surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.machine.artifacts import (
    ARTIFACT_IDS,
    ArtifactId,
    ArtifactVersionInfo,
    Provenance,
)
from nof1_causal_lab.machine.graph import (
    ARTIFACT_GRAPH,
    DERIVATIONS,
    WRITABLE_ARTIFACTS,
    transition_spec,
)

if TYPE_CHECKING:
    from nof1_causal_lab.machine.artifacts import EpisodeState
    from nof1_causal_lab.machine.graph import Transition


class RunArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["run"] = "run"
    artifact_id: ArtifactId


class WriteArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["write"] = "write"
    artifact_id: ArtifactId
    provenance: Provenance = "human"


Move = Annotated[RunArtifact | WriteArtifact, Field(discriminator="kind")]


class RetractedArtifact(BaseModel):
    """A current artifact removed by a move, with the finding that caused it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_id: ArtifactId
    reason_ref: str


class TransitionEffects(BaseModel):
    """What an executed move did to the store: the workflow installs this."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    produced: list[ArtifactVersionInfo] = Field(default_factory=list)
    retracted: list[RetractedArtifact] = Field(default_factory=list)


class ExecOptions(BaseModel):
    """Per-move execution parameters (infra, not domain state)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    inference_method: str | None = None
    enable_literature: bool | None = None
    max_windows: int | None = None


def legal_moves(state: EpisodeState) -> list[Move]:
    """The full machine-legal move set at ``state``.

    This deliberately does not rank or hide moves by usefulness. The action
    hierarchy computes affordances separately so the core has one legality
    model: run guards check input existence, and writes are offered for every
    declared writable artifact.
    """
    moves: list[Move] = []
    for spec in ARTIFACT_GRAPH:
        if all(state.has(artifact) for artifact in spec.consumes):
            moves.append(RunArtifact(artifact_id=spec.transition_id))
    moves.extend(WriteArtifact(artifact_id=aid) for aid in WRITABLE_ARTIFACTS)
    return moves


def validate_move(state: EpisodeState, move: Move) -> str | None:
    """Return a rejection reason, or None if the move is legal at ``state``."""
    if isinstance(move, WriteArtifact):
        if move.provenance == "computed":
            return "write moves must declare provenance 'human' or 'llm'"
        if move.artifact_id not in WRITABLE_ARTIFACTS:
            return f"artifact '{move.artifact_id}' is not writable"
        return None
    try:
        spec = transition_spec(move.artifact_id)
    except KeyError as exc:
        return str(exc)
    missing = [artifact for artifact in spec.consumes if not state.has(artifact)]
    if missing:
        return f"{move.artifact_id} requires artifacts that do not exist: {', '.join(missing)}"
    return None


def input_pins(state: EpisodeState, spec: Transition) -> dict[ArtifactId, int]:
    """The exact input versions a run of ``spec`` at ``state`` consumes."""
    pins: dict[ArtifactId, int] = {}
    for artifact in spec.consumes:
        info = state.get(artifact)
        if info is None:
            raise ValueError(f"{spec.transition_id} input '{artifact}' does not exist")
        pins[artifact] = info.version
    return pins


def write_pins(state: EpisodeState, artifact_ids: tuple[ArtifactId, ...]) -> dict[ArtifactId, int]:
    """Pin existing inputs for write moves.

    Writes stay existence-free, so absent pins are omitted. Existing pins still
    make a hand-authored judgment artifact stale when the context it was written
    against moves.
    """
    pins: dict[ArtifactId, int] = {}
    for artifact in artifact_ids:
        info = state.get(artifact)
        if info is not None:
            pins[artifact] = info.version
    return pins


def apply_transition(
    state: EpisodeState,
    produced: list[ArtifactVersionInfo],
    retracted: list[RetractedArtifact] | None = None,
) -> EpisodeState:
    """Install produced versions and retractions into a new state."""
    next_state = state.with_versions(produced)
    if retracted:
        next_state = next_state.without([item.artifact_id for item in retracted])
    return next_state


def run_retractions(
    state: EpisodeState,
    spec: Transition,
    produced: list[ArtifactVersionInfo],
) -> list[RetractedArtifact]:
    """Optional co-outputs to retract after a successful run of ``spec``."""
    produced_ids = {info.artifact_id for info in produced}
    return [
        RetractedArtifact(
            artifact_id=artifact,
            reason_ref=f"{spec.transition_id}.produces_optional.{artifact}",
        )
        for artifact in spec.produces_optional
        if artifact not in produced_ids and state.has(artifact)
    ]


_DERIVED_ARTIFACTS = frozenset(spec.produces for spec in DERIVATIONS)


def is_stale(state: EpisodeState, artifact_id: ArtifactId) -> bool:
    """Whether an artifact's provenance chain references superseded versions.

    Derived artifacts are never stale. If their parents change, the move that
    changed the parents also recomputes or retracts the derivation.
    """
    if artifact_id in _DERIVED_ARTIFACTS:
        return False
    return _staleness(state, artifact_id, frozenset())


def _staleness(
    state: EpisodeState, artifact_id: ArtifactId, visiting: frozenset[ArtifactId]
) -> bool:
    info = state.get(artifact_id)
    if info is None or artifact_id in visiting:
        return False
    marked = visiting | {artifact_id}
    for input_id, pinned in info.derived_from.items():
        current = state.get(input_id)
        if current is None or current.version != pinned:
            return True
        if _staleness(state, input_id, marked):
            return True
    return False


class ArtifactStatus(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_id: ArtifactId
    exists: bool
    stale: bool
    version: int | None = None
    provenance: Provenance | None = None
    produced_by: str | None = None


def freshness_report(state: EpisodeState) -> list[ArtifactStatus]:
    """Per-artifact existence/staleness — the navigator's and UI's state view."""
    report: list[ArtifactStatus] = []
    for artifact_id in ARTIFACT_IDS:
        info = state.get(artifact_id)
        report.append(
            ArtifactStatus(
                artifact_id=artifact_id,
                exists=info is not None,
                stale=is_stale(state, artifact_id),
                version=info.version if info else None,
                provenance=info.provenance if info else None,
                produced_by=info.produced_by if info else None,
            )
        )
    return report
