"""Shared Prefect task factory for LLM-backed pipeline stages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from prefect import task
from prefect.cache_policies import INPUTS

from nof1_causal_lab.flows import get_prefect_logger
from nof1_causal_lab.utils.config import get_config

from .llm_stage_runtime import LLMStageRuntimeConfig, attach_trace, open_llm_stage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from nof1_causal_lab.utils.config import LLMDefaults, StageLLMConfig

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
        stage_llm = stage_llm_getter()
        llm_defaults = _llm_defaults_getter()
        max_turns = max_tool_turns_getter() if max_tool_turns_getter else None
        runtime_config = LLMStageRuntimeConfig(
            stage_id=stage_id,
            stage_llm=stage_llm,
            llm_defaults=llm_defaults,
            max_tool_turns=max_turns,
        )
        async with open_llm_stage(
            config=runtime_config,
            openrouter_api_key=openrouter_api_key,
            logger=logger,
        ) as factory:
            result = await orchestrator_fn(*args, session_factory=factory, **kwargs)
            payload = payload_builder(result)
            attach_trace(payload, factory.accumulated_trace)
            return payload

    return _run
