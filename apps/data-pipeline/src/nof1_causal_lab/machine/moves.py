"""Moves, legality, transition application, staleness, freshness.

The whole machine in five lines:

- state: versioned artifact store, provenance-stamped (:mod:`artifacts`)
- transitions: ``run(stage)`` — enabled iff all consumed artifacts exist;
  ``write(artifact)`` — schema-validated, any artifact
- derived, never stored: staleness and enabledness
- ``run`` raises typed exceptions (state unchanged, attempt journaled)
- reported numeric results require a fresh provenance chain (enforced at
  the query plane on every serve, not at a one-shot publish transition)
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
from nof1_causal_lab.machine.graph import ARTIFACT_GRAPH, stage_spec

if TYPE_CHECKING:
    from nof1_causal_lab.machine.artifacts import EpisodeState
    from nof1_causal_lab.machine.graph import StageSpec


class RunStage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["run"] = "run"
    stage_id: str


class WriteArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["write"] = "write"
    artifact_id: ArtifactId
    provenance: Provenance = "human"


Move = Annotated[RunStage | WriteArtifact, Field(discriminator="kind")]


class TransitionEffects(BaseModel):
    """What an executed move did to the store: the workflow installs this."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    produced: list[ArtifactVersionInfo] = Field(default_factory=list)
    retracted: list[ArtifactId] = Field(default_factory=list)


class ExecOptions(BaseModel):
    """Per-move execution parameters (infra, not domain state).

    ``openrouter_secret_ref`` is a single-use encrypted key reference —
    the key itself is resolved activity-side and never enters workflow
    history or the artifact store.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    openrouter_access_mode: str | None = None
    openrouter_secret_ref: str | None = None
    # Resolved key for Modal transit only — never set on options that cross
    # the workflow/update boundary (those carry the single-use ref instead).
    openrouter_api_key: str | None = None
    inference_method: str | None = None
    enable_literature: bool | None = None
    max_windows: int | None = None


def legal_moves(state: EpisodeState) -> list[Move]:
    """The full affordance set at ``state``.

    ``run`` moves require all consumed artifacts to exist — a pure existence
    check, no content predicates. Every artifact is writable; content is
    gated by schema validation at execution time, and ``write`` provenance
    is whatever the caller declares (``computed`` is reserved for stages).
    """
    moves: list[Move] = []
    for spec in ARTIFACT_GRAPH:
        if all(state.has(artifact) for artifact in spec.consumes):
            moves.append(RunStage(stage_id=spec.stage_id))
    moves.extend(WriteArtifact(artifact_id=aid) for aid in ARTIFACT_IDS)
    return moves


def validate_move(state: EpisodeState, move: Move) -> str | None:
    """Return a rejection reason, or None if the move is legal at ``state``."""
    if isinstance(move, WriteArtifact):
        if move.provenance == "computed":
            return "write moves must declare provenance 'human' or 'llm'"
        return None
    try:
        spec = stage_spec(move.stage_id)
    except KeyError as exc:
        return str(exc)
    missing = [artifact for artifact in spec.consumes if not state.has(artifact)]
    if missing:
        return f"{move.stage_id} requires artifacts that do not exist: {', '.join(missing)}"
    return None


def input_pins(state: EpisodeState, spec: StageSpec) -> dict[ArtifactId, int]:
    """The exact input versions a run of ``spec`` at ``state`` consumes."""
    pins: dict[ArtifactId, int] = {}
    for artifact in spec.consumes:
        info = state.get(artifact)
        if info is None:
            raise ValueError(f"{spec.stage_id} input '{artifact}' does not exist")
        pins[artifact] = info.version
    return pins


def apply_transition(
    state: EpisodeState,
    produced: list[ArtifactVersionInfo],
    retracted: list[ArtifactId] | None = None,
) -> EpisodeState:
    """Install produced versions (and retractions) into a new state."""
    next_state = state.with_versions(produced)
    if retracted:
        next_state = next_state.without(retracted)
    return next_state


def run_retractions(
    state: EpisodeState,
    spec: StageSpec,
    produced: list[ArtifactVersionInfo],
) -> list[ArtifactId]:
    """Optional artifacts to retract after a successful run of ``spec``.

    An optional artifact that the previous run produced but this run withheld
    is a *changed negative finding* — it must leave ``current`` so downstream
    enabledness reflects it.
    """
    produced_ids = {info.artifact_id for info in produced}
    return [
        artifact
        for artifact in spec.produces_optional
        if artifact not in produced_ids and state.has(artifact)
    ]


def is_stale(state: EpisodeState, artifact_id: ArtifactId) -> bool:
    """Whether an artifact's provenance chain references superseded versions.

    An artifact is stale iff any pinned input is absent, has moved past the
    pinned version, or is itself (transitively) stale. Root artifacts (empty
    ``derived_from``) are never stale. Absent artifacts are not stale — they
    are absent.
    """
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


def is_fresh(state: EpisodeState, artifact_id: ArtifactId) -> bool:
    """Exists and its whole provenance chain pins current versions.

    This is the point-of-claim gate: the query plane must refuse (or
    hard-flag) serving numeric results derived from a non-fresh chain.
    """
    return state.has(artifact_id) and not is_stale(state, artifact_id)


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
