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
from nof1_causal_lab.machine.temporal.client import (
    EPISODE_TASK_QUEUE,
    HARNESS_CLAUDE_TASK_QUEUE,
    HARNESS_CODEX_TASK_QUEUE,
    HARNESS_PI_TASK_QUEUE,
    MODEL_SPEC_SIMULATION_TASK_QUEUE,
    OPENROUTER_TASK_QUEUE,
    connect_client,
)
from nof1_causal_lab.machine.temporal.llm_subroutine_activities import (
    execute_harness_tool_request_activity,
    execute_llm_tool_calls_activity,
    run_harness_turn_activity,
)
from nof1_causal_lab.machine.temporal.llm_subroutine_workflow import LLMSubroutineWorkflow
from nof1_causal_lab.machine.temporal.llm_transition_workflow import SingleLLMTransitionWorkflow
from nof1_causal_lab.machine.temporal.measurement_activities import call_openrouter_activity
from nof1_causal_lab.machine.temporal.measurement_workflow import (
    ExtractionChunkWorkflow,
    MeasurementsWorkflow,
)
from nof1_causal_lab.machine.temporal.statistical_model_spec_activities import (
    plan_statistical_model_spec_activity,
    validate_statistical_model_spec_barrier_activity,
)
from nof1_causal_lab.machine.temporal.statistical_model_spec_workflow import (
    StatisticalModelSpecWorkflow,
)
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
        workflows=[
            EpisodeWorkflow,
            SingleLLMTransitionWorkflow,
            MeasurementsWorkflow,
            StatisticalModelSpecWorkflow,
            ExtractionChunkWorkflow,
            LLMSubroutineWorkflow,
        ],
        activities=ALL_ACTIVITIES,
        workflow_runner=episode_workflow_runner(),
    )


def build_openrouter_worker(client, task_queue: str = OPENROUTER_TASK_QUEUE) -> Worker:
    from nof1_causal_lab.utils.config import get_config

    max_rpm = get_config().extraction_workers.max_rpm
    return Worker(
        client,
        task_queue=task_queue,
        activities=[call_openrouter_activity],
        max_task_queue_activities_per_second=(max_rpm / 60) if max_rpm else None,
    )


def build_harness_worker(
    client,
    task_queue: str,
) -> Worker:
    return Worker(
        client,
        task_queue=task_queue,
        activities=[run_harness_turn_activity],
    )


def build_model_spec_simulation_worker(
    client,
    task_queue: str = MODEL_SPEC_SIMULATION_TASK_QUEUE,
) -> Worker:
    """Serialize exact Stage 4 simulations without limiting harness turns."""
    return Worker(
        client,
        task_queue=task_queue,
        activities=[
            execute_harness_tool_request_activity,
            execute_llm_tool_calls_activity,
            plan_statistical_model_spec_activity,
            validate_statistical_model_spec_barrier_activity,
        ],
        max_concurrent_activities=1,
    )


async def run_worker() -> None:
    client = await connect_client()
    episode_worker = build_worker(client)
    openrouter_worker = build_openrouter_worker(client)
    claude_worker = build_harness_worker(
        client,
        HARNESS_CLAUDE_TASK_QUEUE,
    )
    codex_worker = build_harness_worker(
        client,
        HARNESS_CODEX_TASK_QUEUE,
    )
    pi_worker = build_harness_worker(
        client,
        HARNESS_PI_TASK_QUEUE,
    )
    model_spec_simulation_worker = build_model_spec_simulation_worker(client)
    logger.info("Episode worker started on task queue %s", EPISODE_TASK_QUEUE)
    logger.info("OpenRouter worker started on task queue %s", OPENROUTER_TASK_QUEUE)
    logger.info("Claude harness worker started on task queue %s", HARNESS_CLAUDE_TASK_QUEUE)
    logger.info("Codex harness worker started on task queue %s", HARNESS_CODEX_TASK_QUEUE)
    logger.info("Pi harness worker started on task queue %s", HARNESS_PI_TASK_QUEUE)
    logger.info(
        "Model-spec simulation worker started on task queue %s",
        MODEL_SPEC_SIMULATION_TASK_QUEUE,
    )
    await asyncio.gather(
        episode_worker.run(),
        openrouter_worker.run(),
        claude_worker.run(),
        codex_worker.run(),
        pi_worker.run(),
        model_spec_simulation_worker.run(),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_worker())
