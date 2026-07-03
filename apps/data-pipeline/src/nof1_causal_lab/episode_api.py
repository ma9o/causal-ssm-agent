"""Episode facade: HTTP surface over the state machine.

Mounted into the tool server so the web app has a single Python backend.
CQRS split: reads (status, timeline, events) come straight from the
journal read model and never touch Temporal; moves go through the
episode workflow's ``propose`` update, which validates, executes, and
journals durably.

The ``auto`` endpoint is the default navigation policy — run enabled
stages in dependency order while their outputs are missing or stale —
giving the web "run the pipeline" parity as one background driver. An
LLM navigator replaces this policy by calling ``moves`` directly.

BYOK caveat: an ``openrouter_secret_ref`` is single-use, so it authorizes
exactly one move; auto-run across LLM stages needs ambient credentials
(``openrouter_access_mode: local``).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.flows.runtime_events import read_events
from nof1_causal_lab.machine.graph import ARTIFACT_GRAPH, topological_stage_order

if TYPE_CHECKING:
    from nof1_causal_lab.machine.artifacts import EpisodeState
    from nof1_causal_lab.machine.graph import StageSpec
from nof1_causal_lab.machine.moves import (
    ExecOptions,
    Move,
    RunStage,
    freshness_report,
    is_stale,
    legal_moves,
    validate_move,
)
from nof1_causal_lab.machine.store import EpisodeJournal

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/episodes")


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
    return await client.start_workflow(
        EpisodeWorkflow.run,
        EpisodeInit(workspace_id=workspace_id),
        id=episode_workflow_id(workspace_id),
        task_queue=EPISODE_TASK_QUEUE,
        id_conflict_policy=WorkflowIDConflictPolicy.USE_EXISTING,
    )


async def _propose(workspace_id: str, request_body: MoveBody) -> dict[str, Any]:
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
    payload: dict[str, Any] | None = None
    options: ExecOptions = Field(default_factory=ExecOptions)


class AutoRunBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    options: ExecOptions = Field(default_factory=ExecOptions)


# ---------------------------------------------------------------------------
# Reads: journal-backed (no Temporal dependency)
# ---------------------------------------------------------------------------


def _episode_status(workspace_id: str) -> dict[str, Any]:
    journal = EpisodeJournal(workspace_id)
    state = journal.latest_state()
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
def get_episode(workspace_id: str) -> dict[str, Any]:
    return _episode_status(workspace_id)


@router.get("/{workspace_id}/timeline")
def get_timeline(workspace_id: str) -> dict[str, Any]:
    records = EpisodeJournal(workspace_id).read_all()
    return {
        "workspace_id": workspace_id,
        "transitions": [record.model_dump(mode="json") for record in records],
    }


@router.get("/{workspace_id}/events")
def get_events(workspace_id: str, after: str | None = None) -> dict[str, Any]:
    return {
        "workspace_id": workspace_id,
        "events": read_events(workspace_id, after=after),
    }


# ---------------------------------------------------------------------------
# Moves
# ---------------------------------------------------------------------------


@router.post("")
async def start_episode(body: StartEpisodeBody) -> dict[str, Any]:
    """Ensure the episode workflow exists; optionally write the question."""
    await _episode_handle(body.workspace_id)
    outcome = None
    if body.question is not None:
        outcome = await _propose(
            body.workspace_id,
            MoveBody(
                move={"kind": "write", "artifact_id": "question", "provenance": "human"},
                payload={"text": body.question},
            ),
        )
    return {"ok": True, "outcome": outcome, **_episode_status(body.workspace_id)}


@router.post("/{workspace_id}/moves")
async def propose_move(workspace_id: str, body: MoveBody) -> dict[str, Any]:
    return await _propose(workspace_id, body)


# ---------------------------------------------------------------------------
# Default navigation policy (auto-run)
# ---------------------------------------------------------------------------

_AUTO_DRIVERS: dict[str, asyncio.Task] = {}


def _needs_run(state: EpisodeState, spec: StageSpec) -> bool:
    """Missing required outputs, or any existing output gone stale.

    An *absent optional* output with a fresh report is a standing negative
    finding, not a reason to rerun — otherwise the driver would loop on
    stages like 1b/2 whose finding was legitimately empty.
    """
    if any(not state.has(artifact) for artifact in spec.produces):
        return True
    return any(is_stale(state, artifact) for artifact in spec.all_produces if state.has(artifact))


def _next_auto_move(state: EpisodeState) -> RunStage | None:
    specs = {spec.stage_id: spec for spec in ARTIFACT_GRAPH}
    for stage_id in topological_stage_order():
        spec = specs[stage_id]
        move = RunStage(stage_id=stage_id)
        if validate_move(state, move) is None and _needs_run(state, spec):
            return move
    return None


async def _auto_drive(workspace_id: str, options: ExecOptions) -> None:
    try:
        while True:
            state = EpisodeJournal(workspace_id).latest_state()
            move = _next_auto_move(state)
            if move is None:
                logger.info("auto-run %s: quiescent", workspace_id)
                return
            logger.info("auto-run %s: %s", workspace_id, move.stage_id)
            outcome = await _propose(workspace_id, MoveBody(move=move, options=options))
            if outcome["status"] != "applied":
                logger.warning(
                    "auto-run %s stopped: %s %s (%s)",
                    workspace_id,
                    move.stage_id,
                    outcome["status"],
                    outcome.get("error_type") or outcome.get("reason"),
                )
                return
    finally:
        _AUTO_DRIVERS.pop(workspace_id, None)


@router.post("/{workspace_id}/auto")
async def auto_run(workspace_id: str, body: AutoRunBody) -> dict[str, Any]:
    """Run enabled stages in dependency order until quiescent (background)."""
    if workspace_id in _AUTO_DRIVERS:
        raise HTTPException(409, f"auto-run already active for {workspace_id}")
    await _episode_handle(workspace_id)  # fail fast if Temporal is down
    _AUTO_DRIVERS[workspace_id] = asyncio.create_task(_auto_drive(workspace_id, body.options))
    return {"ok": True, "auto_running": True, "workspace_id": workspace_id}
