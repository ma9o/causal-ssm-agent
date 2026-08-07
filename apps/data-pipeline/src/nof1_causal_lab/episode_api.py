"""Episode facade: HTTP surface over the state machine.

Mounted into the tool server so the web app has a single Python backend.
Reads derive current state by replaying the append-only transition log and
serve timeline/events from their execution logs without touching Temporal.
Moves go through the episode workflow's ``propose`` update, which validates,
executes, and records outcomes durably.

The ``auto`` endpoint is the default navigation policy — run enabled
transitions in dependency order while their outputs are missing or stale —
giving the web "run the pipeline" parity as one background driver. An
LLM navigator replaces this policy by calling ``moves`` directly.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.flows.runtime_events import read_events
from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001
from nof1_causal_lab.machine.artifacts import (  # noqa: TC001 (FastAPI runtime annotation)
    ArtifactId,
    ArtifactVersionInfo,
)
from nof1_causal_lab.machine.graph import ARTIFACT_GRAPH, topological_transition_order
from nof1_causal_lab.utils.llm import LLMTrace

if TYPE_CHECKING:
    from nof1_causal_lab.machine.artifacts import EpisodeState
    from nof1_causal_lab.machine.graph import Transition
from nof1_causal_lab.machine.moves import (
    ExecOptions,
    Move,
    RunArtifact,
    WriteArtifact,
    freshness_report,
    is_stale,
    legal_moves,
    validate_move,
)
from nof1_causal_lab.machine.store import (
    EpisodeJournal,
    TransitionRecord,
    derive_current_state,
    read_episode_trace,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/episodes")

capabilities_router = APIRouter(prefix="/api")
workspaces_router = APIRouter(prefix="/api")
uploads_router = APIRouter(prefix="/api")


class CapabilitiesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    moves_enabled: bool


class WorkspaceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    href: str
    question: str | None = None
    workspaceId: str


class WorkspaceList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspaces: list[WorkspaceEntry]


class UploadResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str


class ArtifactEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    artifact_id: ArtifactId
    version: int
    meta: ArtifactVersionInfo
    payload: UncheckedJsonObject
    binary_files: list[str]


def moves_enabled() -> bool:
    """Whether this facade deployment serves the move plane.

    A read-only facade (the hosted viewer's backend) runs the same read
    endpoints against a published store with no Temporal attached; the
    viewer reads this capability instead of being built as a fork.
    """
    return os.environ.get("EPISODE_FACADE_READ_ONLY") != "1"


def _require_moves_enabled() -> None:
    if not moves_enabled():
        raise HTTPException(403, "This facade is read-only: the move plane is not deployed here")


def _safe_workspace_id(value: str) -> str:
    workspace_id = value.strip()
    if (
        not workspace_id
        or len(workspace_id) > 200
        or re.fullmatch(r"[A-Za-z0-9_-]+", workspace_id) is None
    ):
        raise HTTPException(400, "Invalid workspaceId format")
    return workspace_id


@capabilities_router.get("/capabilities", response_model=CapabilitiesResponse)
def get_capabilities() -> CapabilitiesResponse:
    """Whether this deployment serves the move plane.

    `moves_enabled` is `false` on the hosted read-only viewer backend, where
    every `POST` (moves, auto-run, start-episode) returns 403 and only the read
    endpoints are live.
    """
    return CapabilitiesResponse(moves_enabled=moves_enabled())


def _workspace_question(workspace_id: str) -> str | None:
    from nof1_causal_lab.machine.artifact_files import json_filename
    from nof1_causal_lab.machine.store import ArtifactStore

    info = derive_current_state(workspace_id).get("question")
    if info is None:
        return None
    payload = ArtifactStore(workspace_id).read_json_file(
        "question", info.version, json_filename("question", "question")
    )
    text = payload.get("text")
    return text.strip() if isinstance(text, str) and text.strip() else None


@workspaces_router.get("/workspaces", response_model=WorkspaceList)
def list_workspaces() -> WorkspaceList:
    """Published/local workspaces visible through this facade."""
    from nof1_causal_lab.utils import data as data_module
    from nof1_causal_lab.utils import storage

    workspaces: list[WorkspaceEntry] = []
    for entry in sorted(storage.listdir(data_module.data_root())):
        workspace_id = entry.rstrip("/").rsplit("/", 1)[-1]
        if not workspace_id or workspace_id.startswith("."):
            continue
        workspaces.append(
            WorkspaceEntry(
                href=f"/analysis/{workspace_id}",
                question=_workspace_question(workspace_id),
                workspaceId=workspace_id,
            )
        )
    return WorkspaceList(workspaces=workspaces)


@uploads_router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: Annotated[UploadFile, File()],
    workspace_id: Annotated[str, Form(alias="workspaceId")],
) -> UploadResponse:
    """Stage one raw input file for the raw_data transition."""
    from nof1_causal_lab.utils import data as data_module
    from nof1_causal_lab.utils import storage

    _require_moves_enabled()
    safe_workspace_id = _safe_workspace_id(workspace_id)
    filename = (file.filename or "").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if not filename:
        raise HTTPException(400, "Invalid file name")

    upload_dir = data_module.input_dir(safe_workspace_id)
    storage.makedirs(upload_dir)
    path = storage.join(upload_dir, filename)
    with storage.open_file(path, "wb") as handle:
        handle.write(await file.read())
    return UploadResponse(path=f"{safe_workspace_id}/input/{filename}")


machine_router = APIRouter(prefix="/api")


def machine_description() -> UncheckedJsonObject:
    """The static shape of the episode machine, independent of any workspace.

    Aggregates the artifact graph, its roots, and the action/context hierarchy
    into the single payload an agent reads once to orient. The route
    `GET /api/machine` returns exactly this; it never touches Temporal or a
    workspace store.
    """
    from nof1_causal_lab.machine.artifacts import ARTIFACT_IDS
    from nof1_causal_lab.machine.graph import (
        ARTIFACT_GRAPH,
        DERIVATIONS,
        ROOTS,
        topological_artifact_order,
        topological_transition_order,
    )
    from nof1_causal_lab.machine.hierarchy import describe_actions, describe_contexts

    return {
        "artifact_ids": list(ARTIFACT_IDS),
        "topological_artifact_order": topological_artifact_order(),
        "topological_transition_order": topological_transition_order(),
        "contexts": describe_contexts(),
        "actions": describe_actions(),
        "roots": [
            {"artifact_id": root.artifact_id, "write_pins": list(root.write_pins)} for root in ROOTS
        ],
        "transitions": [
            {
                "transition_id": spec.transition_id,
                "consumes": list(spec.consumes),
                "produces": [spec.produces],
                "produces_optional": list(spec.produces_optional),
                "creation_class": spec.creation_class,
                "writable": spec.writable,
            }
            for spec in ARTIFACT_GRAPH
        ],
        "derivations": [
            {
                "produces": spec.produces,
                "from": list(spec.from_),
                "optional": spec.optional,
            }
            for spec in DERIVATIONS
        ],
    }


@machine_router.get("/machine")
def get_machine() -> UncheckedJsonObject:
    """The static artifact graph and action hierarchy — read once to orient.

    Each transition entry declares what it `consumes`, `produces`, and
    optionally co-produces (`produces_optional`), plus its **creation
    class**: `deterministic` (pure compute, no credentials), `batch_llm` (bulk
    LLM compute on the service's ambient key — you trigger it with a `run` move,
    you never supply a key), or `judgment` (proposal work you can author yourself
    by writing the produced artifact directly — these are flagged `writable`).
    """
    return machine_description()


# ---------------------------------------------------------------------------
# Temporal client plumbing (moves only; reads never touch Temporal)
# ---------------------------------------------------------------------------

_client_lock = asyncio.Lock()
_client: Any = None


async def _get_client():
    global _client
    async with _client_lock:
        if _client is None:
            from nof1_causal_lab.machine.temporal.client import connect_client

            _client = await connect_client()
        return _client


async def _episode_handle(workspace_id: str):
    """Start-or-attach the entity workflow for a workspace."""
    from temporalio.common import WorkflowIDConflictPolicy

    from nof1_causal_lab.machine.temporal.client import (
        EPISODE_TASK_QUEUE,
        episode_workflow_id,
    )
    from nof1_causal_lab.machine.temporal.messages import EpisodeInit
    from nof1_causal_lab.machine.temporal.workflow import EpisodeWorkflow

    client = await _get_client()
    # Reconstruct the resume seed from committed transition effects. On a fresh
    # start this rehydrates a workflow Temporal lost (empty for a new episode);
    # on attach (USE_EXISTING) the live workflow keeps its own state and the
    # seed is ignored.
    journal = EpisodeJournal(workspace_id)
    return await client.start_workflow(
        EpisodeWorkflow.run,
        EpisodeInit(
            workspace_id=workspace_id,
            initial_state=derive_current_state(workspace_id),
            initial_seq=journal.latest_seq(),
        ),
        id=episode_workflow_id(workspace_id),
        task_queue=EPISODE_TASK_QUEUE,
        id_conflict_policy=WorkflowIDConflictPolicy.USE_EXISTING,
    )


async def _propose(workspace_id: str, request_body: MoveBody) -> UncheckedJsonObject:
    from nof1_causal_lab.machine.temporal.messages import MoveRequest
    from nof1_causal_lab.machine.temporal.workflow import EpisodeWorkflow

    handle = await _episode_handle(workspace_id)
    outcome = await handle.execute_update(
        EpisodeWorkflow.propose,
        MoveRequest(
            move=request_body.move,
            payload=request_body.payload,
            options=request_body.options,
        ),
    )
    return outcome.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Request/response bodies
# ---------------------------------------------------------------------------


class StartEpisodeBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    question: str | None = None


class MoveBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    move: Move
    payload: UncheckedJsonObject | None = None
    options: ExecOptions = Field(default_factory=ExecOptions)


class AutoRunBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    options: ExecOptions = Field(default_factory=ExecOptions)


# ---------------------------------------------------------------------------
# Reads: transition-log-backed (no Temporal dependency)
# ---------------------------------------------------------------------------


def _episode_status(workspace_id: str) -> UncheckedJsonObject:
    journal = EpisodeJournal(workspace_id)
    state = derive_current_state(workspace_id)
    records = journal.read_all()
    return {
        "workspace_id": workspace_id,
        "seq": records[-1].seq if records else 0,
        "state": state.model_dump(mode="json"),
        "artifacts": [status.model_dump(mode="json") for status in freshness_report(state)],
        "legal": [move.model_dump(mode="json") for move in legal_moves(state)],
        "auto_running": workspace_id in _AUTO_DRIVERS,
    }


@router.get("/{workspace_id}")
def get_episode(workspace_id: str) -> UncheckedJsonObject:
    """Current episode state: the single read to poll while navigating.

    Returns per-artifact freshness (existence, staleness, version, provenance),
    the `legal` moves available right now, and `auto_running` — whether the
    background driver is active. Replayed from the append-only transition log,
    so it works even against a published read-only store.
    """
    return _episode_status(workspace_id)


class TimelineResponse(BaseModel):
    """Typed transition journal returned by the episode read plane."""

    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    transitions: list[TransitionRecord]


@router.get("/{workspace_id}/timeline", response_model=TimelineResponse)
def get_timeline(workspace_id: str) -> TimelineResponse:
    """The transition journal: every move attempt in order.

    Each record is `applied` (state advanced), `rejected` (illegal move, state
    unchanged), or `raised` (the transition ran but threw — the record carries the
    typed error). Re-running after a `raised`/`rejected` is just proposing the
    move again.
    """
    records = EpisodeJournal(workspace_id).read_all()
    return TimelineResponse(workspace_id=workspace_id, transitions=records)


@router.get("/{workspace_id}/events")
def get_events(workspace_id: str, after: str | None = None) -> UncheckedJsonObject:
    """Fine-grained telemetry (e.g. extraction worker fan-out, transition progress).

    Pass the last-seen event id as `after` to page forward; omit it for the full
    stream. This is finer-grained than the timeline, which records only whole
    move outcomes.
    """
    return {
        "workspace_id": workspace_id,
        "events": read_events(workspace_id, after=after),
    }


@router.get("/{workspace_id}/artifacts/{artifact_id}", response_model=ArtifactEnvelope)
def get_artifact(
    workspace_id: str, artifact_id: ArtifactId, version: int | None = None
) -> ArtifactEnvelope:
    """One artifact version: meta + inline JSON payloads.

    Defaults to the episode's *current* version from replayed applied transition
    effects. Binary payload files (parquet, pickle) are listed by name, never
    inlined.
    """
    from nof1_causal_lab.machine.store import ArtifactStore
    from nof1_causal_lab.utils import storage

    store = ArtifactStore(workspace_id)
    if version is None:
        info = derive_current_state(workspace_id).get(artifact_id)
        if info is None:
            raise HTTPException(
                404, f"No current '{artifact_id}' artifact for workspace {workspace_id}"
            )
        version = info.version

    version_dir = store.version_dir(artifact_id, version)
    if not storage.exists(storage.join(version_dir, "meta.json")):
        raise HTTPException(404, f"{artifact_id} v{version} does not exist for {workspace_id}")

    payload: UncheckedJsonObject = {}
    binary_files: list[str] = []
    for entry in storage.listdir(version_dir):
        name = entry.rstrip("/").rsplit("/", 1)[-1]
        if name == "meta.json":
            continue
        if name.endswith(".json"):
            payload[name] = store.read_json_file(artifact_id, version, name)
        else:
            binary_files.append(name)

    return ArtifactEnvelope(
        workspace_id=workspace_id,
        artifact_id=artifact_id,
        version=version,
        meta=store.read_meta(artifact_id, version),
        payload=payload,
        binary_files=sorted(binary_files),
    )


class TransitionTraceIndex(BaseModel):
    """Promoted traces of the applied transition that produced an artifact version."""

    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    artifact_id: ArtifactId
    version: int
    seq: int
    trace_ids: list[str]


@router.get("/{workspace_id}/artifacts/{artifact_id}/traces", response_model=TransitionTraceIndex)
def get_artifact_traces(
    workspace_id: str, artifact_id: ArtifactId, version: int | None = None
) -> TransitionTraceIndex:
    """Traces of the applied transition that produced an artifact version.

    Defaults to the episode's current version. The join runs over the
    transition journal, so it works against a published read-only store.
    """
    if version is None:
        info = derive_current_state(workspace_id).get(artifact_id)
        if info is None:
            raise HTTPException(
                404, f"No current '{artifact_id}' artifact for workspace {workspace_id}"
            )
        version = info.version
    for record in reversed(EpisodeJournal(workspace_id).read_all()):
        if record.status != "applied":
            continue
        if any(
            item.artifact_id == artifact_id and item.version == version for item in record.produced
        ):
            return TransitionTraceIndex(
                workspace_id=workspace_id,
                artifact_id=artifact_id,
                version=version,
                seq=record.seq,
                trace_ids=record.trace_ids,
            )
    raise HTTPException(404, f"No applied transition produced {artifact_id} v{version}")


@router.get("/{workspace_id}/traces/{seq}/{subroutine_id}", response_model=LLMTrace)
def get_trace(workspace_id: str, seq: int, subroutine_id: str) -> LLMTrace:
    """One promoted LLM trace from ``episode/traces/{seq:06d}/{subroutine_id}.json``."""
    if "/" in subroutine_id or ".." in subroutine_id:
        raise HTTPException(400, f"Invalid subroutine id {subroutine_id!r}")
    try:
        return LLMTrace.model_validate(read_episode_trace(workspace_id, seq, subroutine_id))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/{workspace_id}/artifacts/{artifact_id}/files/{filename}")
def get_artifact_file(
    workspace_id: str,
    artifact_id: ArtifactId,
    filename: str,
    version: int | None = None,
) -> Response:
    """One declared payload file from an artifact version.

    Defaults to the episode's current version. Unlike the JSON artifact
    endpoint, this serves binary files as bytes and refuses undeclared
    filenames so callers cannot browse arbitrary workspace paths.
    """
    from nof1_causal_lab.machine.artifact_files import is_declared_artifact_file
    from nof1_causal_lab.machine.store import ArtifactStore
    from nof1_causal_lab.utils import storage

    if "/" in filename or not is_declared_artifact_file(artifact_id, filename):
        raise HTTPException(404, f"{filename} is not a declared file for {artifact_id}")

    store = ArtifactStore(workspace_id)
    if version is None:
        info = derive_current_state(workspace_id).get(artifact_id)
        if info is None:
            raise HTTPException(
                404, f"No current '{artifact_id}' artifact for workspace {workspace_id}"
            )
        version = info.version

    path = store.file_path(artifact_id, version, filename)
    if not storage.exists(path):
        raise HTTPException(404, f"{artifact_id} v{version}/{filename} does not exist")

    with storage.open_file(path, "rb") as handle:
        data = handle.read()
    return Response(content=data, media_type="application/octet-stream")


# ---------------------------------------------------------------------------
# Moves
# ---------------------------------------------------------------------------


@router.post("")
async def start_episode(body: StartEpisodeBody) -> UncheckedJsonObject:
    """Ensure the episode workflow exists; optionally seed the `question` root.

    Idempotent: attaches to an existing episode or starts a fresh one. Passing
    `question` writes the `question` root artifact with `human` provenance.
    Upload raw data at `POST /api/upload` before running the `raw_data`
    transition. Returns the same shape as
    `GET /api/episodes/{id}`.
    """
    _require_moves_enabled()
    await _episode_handle(body.workspace_id)
    outcome = None
    if body.question is not None:
        outcome = await _propose(
            body.workspace_id,
            MoveBody(
                move=WriteArtifact(artifact_id="question", provenance="human"),
                payload={"text": body.question},
            ),
        )
    return {"ok": True, "outcome": outcome, **_episode_status(body.workspace_id)}


@router.post("/{workspace_id}/moves")
async def propose_move(workspace_id: str, body: MoveBody) -> UncheckedJsonObject:
    """Propose one move; blocks until it is applied, rejected, or raises.

    Two kinds:

    - Run a transition: `{"move": {"kind": "run", "artifact_id": "latent_structure"}}`.
    - Author a judgment artifact directly (skip the in-service stage):
      `{"move": {"kind": "write", "artifact_id": "latent_structure", "provenance":
      "llm"}, "payload": {...}}`. The payload is schema-validated against that
      artifact's contract, journaled, and provenance-stamped; the write becomes a
      new provenance root and marks everything downstream stale until re-run.

    The synchronous outcome is the same record the timeline stores. Long transitions
    (statistical model specification, posterior — minutes to hours) can outlive a client timeout; for
    those prefer `POST /api/episodes/{workspace_id}/auto` plus polling.
    """
    _require_moves_enabled()
    return await _propose(workspace_id, body)


# ---------------------------------------------------------------------------
# Default navigation policy (auto-run)
# ---------------------------------------------------------------------------

_AUTO_DRIVERS: dict[str, asyncio.Task[None]] = {}


def _needs_run(state: EpisodeState, spec: Transition) -> bool:
    """Missing required outputs, or any existing output gone stale.

    An *absent optional* output with a fresh report is a standing negative
    finding, not a reason to rerun — otherwise the driver would loop on
    transitions whose finding was legitimately empty.
    """
    if not state.has(spec.produces):
        return True
    return any(is_stale(state, artifact) for artifact in spec.all_produces if state.has(artifact))


def _next_auto_move(state: EpisodeState) -> RunArtifact | None:
    specs = {spec.transition_id: spec for spec in ARTIFACT_GRAPH}
    for artifact_id in topological_transition_order():
        spec = specs[artifact_id]
        move = RunArtifact(artifact_id=artifact_id)
        if validate_move(state, move) is None and _needs_run(state, spec):
            return move
    return None


async def _auto_drive(workspace_id: str, options: ExecOptions) -> None:
    try:
        while True:
            state = derive_current_state(workspace_id)
            move = _next_auto_move(state)
            if move is None:
                logger.info("auto-run %s: quiescent", workspace_id)
                return
            logger.info("auto-run %s: %s", workspace_id, move.artifact_id)
            outcome = await _propose(workspace_id, MoveBody(move=move, options=options))
            if outcome["status"] != "applied":
                logger.warning(
                    "auto-run %s stopped: %s %s (%s)",
                    workspace_id,
                    move.artifact_id,
                    outcome["status"],
                    outcome.get("error_type") or outcome.get("reason"),
                )
                return
    finally:
        _AUTO_DRIVERS.pop(workspace_id, None)


@router.post("/{workspace_id}/auto")
async def auto_run(workspace_id: str, body: AutoRunBody) -> UncheckedJsonObject:
    """Start the default navigation policy in the background.

    Runs enabled stages in dependency order while their outputs are missing or
    stale, stopping when quiescent or when a move fails. Returns immediately;
    follow progress with `GET /api/episodes/{workspace_id}` (`auto_running`) and
    the timeline. An LLM navigator replaces this policy by proposing `moves`
    itself. 409 if a driver is already active for this workspace.
    """
    _require_moves_enabled()
    if workspace_id in _AUTO_DRIVERS:
        raise HTTPException(409, f"auto-run already active for {workspace_id}")
    await _episode_handle(workspace_id)  # fail fast if Temporal is down
    _AUTO_DRIVERS[workspace_id] = asyncio.create_task(_auto_drive(workspace_id, body.options))
    return {"ok": True, "auto_running": True, "workspace_id": workspace_id}
