"""Shared Prefect task factory for LLM-backed pipeline stages."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from prefect import task
from prefect.cache_policies import INPUTS

from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.utils.agent_session import StageSessionFactory
from causal_ssm_agent.utils.config import get_config
from causal_ssm_agent.utils.openrouter_client import use_openrouter_api_key

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from causal_ssm_agent.utils.config import LLMDefaults, StageLLMConfig

LLMOrchestrator = "Callable[..., Awaitable[Any]]"
StageLLMGetter = "Callable[[], StageLLMConfig]"
PayloadBuilder = "Callable[[Any], dict[str, Any]]"
MaxToolTurnsGetter = "Callable[[], int]"

logger = get_prefect_logger(__name__)


def make_llm_stage_task(
    *,
    stage_id: str,
    orchestrator_fn: Callable[..., Awaitable[Any]],
    stage_llm_getter: Callable[[], StageLLMConfig],
    payload_builder: Callable[[Any], dict[str, Any]],
    max_tool_turns_getter: Callable[[], int] | None = None,
    llm_defaults_getter: Callable[[], LLMDefaults] | None = None,
    task_options: dict[str, Any] | None = None,
):
    """Build a Prefect task wrapper for an LLM-backed stage.

    The wrapper:
    - opens a :class:`StageSessionFactory` bound to the stage's
      :class:`StageLLMConfig` and global :class:`LLMDefaults`,
    - passes the factory to ``orchestrator_fn`` as ``session_factory=``,
    - attaches the factory's accumulated LLM trace to the payload.
    """
    options: dict[str, Any] = {
        "cache_policy": INPUTS,
        "persist_result": True,
    }
    if task_options:
        options.update(task_options)

    _llm_defaults_getter = llm_defaults_getter or (lambda: get_config().llm)

    @task(**options)
    async def _run(*args: Any, **kwargs: Any) -> dict[str, Any]:
        openrouter_api_key = kwargs.pop("openrouter_api_key", None)
        with use_openrouter_api_key(openrouter_api_key):
            started_at = time.monotonic()
            logger.info("[%s] starting", stage_id)
            stage_llm = stage_llm_getter()
            llm_defaults = _llm_defaults_getter()
            max_turns = max_tool_turns_getter() if max_tool_turns_getter else None
            factory = StageSessionFactory(
                stage_llm,
                llm_defaults,
                stage_id=stage_id,
                max_tool_turns=max_turns,
            )
            result = await orchestrator_fn(*args, session_factory=factory, **kwargs)
            payload = payload_builder(result)
            trace = factory.accumulated_trace
            if trace.messages or trace.usage.input_tokens or trace.usage.output_tokens:
                payload["llm_trace"] = trace.model_dump(mode="json")
            elapsed = time.monotonic() - started_at
            logger.info(
                "[%s] completed in %.1fs (harness=%s, model=%s)",
                stage_id,
                elapsed,
                stage_llm.harness,
                stage_llm.model,
            )
            return payload

    return _run
