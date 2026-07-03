"""Temporal shell: the thin durable layer around the pure machine.

One entity workflow per episode holds the ``EpisodeState`` and exposes
``propose`` (update) / queries; activities execute stage runs and writes
against the artifact store and project every transition attempt into the
episode journal. All validation delegates to the pure functions in
:mod:`nof1_causal_lab.machine.moves` — the workflow adds durability, not
semantics.
"""

from nof1_causal_lab.machine.temporal.client import (
    EPISODE_TASK_QUEUE,
    connect_client,
    episode_workflow_id,
)
from nof1_causal_lab.machine.temporal.messages import (
    EpisodeInit,
    MoveOutcome,
    MoveRequest,
)
from nof1_causal_lab.machine.temporal.workflow import EpisodeWorkflow

__all__ = [
    "EPISODE_TASK_QUEUE",
    "EpisodeInit",
    "EpisodeWorkflow",
    "MoveOutcome",
    "MoveRequest",
    "connect_client",
    "episode_workflow_id",
]
