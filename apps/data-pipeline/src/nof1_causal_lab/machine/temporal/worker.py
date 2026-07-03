"""Episode worker entrypoint.

Run with::

    uv run python -m nof1_causal_lab.machine.temporal.worker
"""

from __future__ import annotations

import asyncio
import logging

from temporalio.worker import Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

from nof1_causal_lab.machine.temporal.activities import ALL_ACTIVITIES
from nof1_causal_lab.machine.temporal.client import EPISODE_TASK_QUEUE, connect_client
from nof1_causal_lab.machine.temporal.workflow import EpisodeWorkflow

logger = logging.getLogger(__name__)


def episode_workflow_runner() -> SandboxedWorkflowRunner:
    """Sandbox runner with the package passed through.

    The package root configures the JAX persistent cache at import time;
    re-importing jaxlib inside the sandbox aborts the process. Passing the
    package through is safe here because workflow determinism is carried
    by construction (the workflow only calls the pure machine functions).
    """
    return SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules(
            "nof1_causal_lab", "pydantic"
        )
    )


def build_worker(client, task_queue: str = EPISODE_TASK_QUEUE) -> Worker:
    return Worker(
        client,
        task_queue=task_queue,
        workflows=[EpisodeWorkflow],
        activities=ALL_ACTIVITIES,
        workflow_runner=episode_workflow_runner(),
    )


async def run_worker() -> None:
    client = await connect_client()
    worker = build_worker(client)
    logger.info("Episode worker started on task queue %s", EPISODE_TASK_QUEUE)
    await worker.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_worker())
