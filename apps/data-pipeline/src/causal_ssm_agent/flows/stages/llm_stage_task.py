"""Shared Prefect task factory for LLM-backed pipeline stages."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from prefect import task
from prefect.cache_policies import INPUTS

from causal_ssm_agent.utils.llm import LLMStageContext

LLMOrchestrator = Callable[..., Awaitable[Any]]
ModelNameGetter = Callable[[], str]
PayloadBuilder = Callable[[Any], dict[str, Any]]


def make_llm_stage_task(
    *,
    stage_id: str,
    orchestrator_fn: LLMOrchestrator,
    model_name_getter: ModelNameGetter,
    payload_builder: PayloadBuilder,
    task_options: dict[str, Any] | None = None,
):
    """Build a Prefect task wrapper for an LLM-backed stage.

    The wrapper is responsible for:
    - opening the ``LLMStageContext``
    - constructing the stage-scoped ``generate`` function
    - calling the orchestrator
    - finalizing the web payload with the captured LLM trace
    """

    options: dict[str, Any] = {
        "cache_policy": INPUTS,
        "persist_result": True,
    }
    if task_options:
        options.update(task_options)

    @task(**options)
    async def _run(*args: Any, **kwargs: Any) -> dict[str, Any]:
        async with LLMStageContext(stage_id) as ctx:
            generate = ctx.make_generate(model_name_getter())
            result = await orchestrator_fn(*args, generate=generate, **kwargs)
            return ctx.finalize(payload_builder(result))

    return _run
