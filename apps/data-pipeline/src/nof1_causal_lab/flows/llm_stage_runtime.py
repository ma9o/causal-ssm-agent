"""Shared lifecycle helpers for LLM-backed stages."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from nof1_causal_lab.utils.agent_session import StageSessionFactory

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from nof1_causal_lab.utils.config import LLMDefaults, StageLLMConfig
    from nof1_causal_lab.utils.llm import LLMTrace


@dataclass(frozen=True)
class LLMStageRuntimeConfig:
    """Factory settings for one LLM-backed stage."""

    stage_id: str
    stage_llm: StageLLMConfig
    llm_defaults: LLMDefaults
    max_tool_turns: int | None = None


def build_stage_session_factory(config: LLMStageRuntimeConfig) -> StageSessionFactory:
    """Create the stage-scoped session factory used throughout one LLM stage."""
    return StageSessionFactory(
        config.stage_llm,
        config.llm_defaults,
        stage_id=config.stage_id,
        max_tool_turns=config.max_tool_turns,
    )


def trace_has_content(trace: LLMTrace) -> bool:
    return bool(trace.messages or trace.usage.input_tokens or trace.usage.output_tokens)


def attach_trace(payload: dict[str, Any], trace: LLMTrace) -> dict[str, Any]:
    """Attach a serialized trace when the stage generated one."""
    if trace_has_content(trace):
        payload["llm_trace"] = trace.model_dump(mode="json")
    return payload


@asynccontextmanager
async def open_llm_stage(
    *,
    config: LLMStageRuntimeConfig,
    logger: logging.Logger,
) -> AsyncIterator[StageSessionFactory]:
    """Open the stage session factory for one stage run."""
    started_at = time.monotonic()
    logger.info("[%s] starting", config.stage_id)
    factory = build_stage_session_factory(config)
    try:
        yield factory
    except Exception as exc:
        elapsed = time.monotonic() - started_at
        trace_messages = len(factory.accumulated_trace.messages)
        if trace_messages:
            logger.error(
                "[%s] failed after %.1fs with %d trace messages: %s",
                config.stage_id,
                elapsed,
                trace_messages,
                exc,
            )
        else:
            logger.error("[%s] failed after %.1fs: %s", config.stage_id, elapsed, exc)
        raise
    elapsed = time.monotonic() - started_at
    logger.info(
        "[%s] completed in %.1fs (harness=%s, model=%s)",
        config.stage_id,
        elapsed,
        config.stage_llm.harness,
        config.stage_llm.model,
    )


def make_llm_stage_runner(
    *,
    stage_id: str,
    orchestrator_fn: Callable[..., Awaitable[Any]],
    stage_llm_getter: Callable[[], StageLLMConfig],
    payload_builder: Callable[[Any], dict[str, Any]],
    max_tool_turns_getter: Callable[[], int] | None = None,
    llm_defaults_getter: Callable[[], LLMDefaults] | None = None,
):
    """Build a plain async runner for an LLM-backed stage.

    The wrapper opens a :class:`StageSessionFactory` bound to the stage's
    LLM config, passes it to ``orchestrator_fn`` as ``session_factory=``,
    and attaches the accumulated LLM trace to the payload. Retries live in
    the Temporal activity policy, not here.
    """
    from nof1_causal_lab.utils.config import get_config

    _llm_defaults_getter = llm_defaults_getter or (lambda: get_config().llm)
    logger = logging.getLogger(f"nof1_causal_lab.flows.{stage_id}")

    async def _run(*args: Any, **kwargs: Any) -> dict[str, Any]:
        runtime_config = LLMStageRuntimeConfig(
            stage_id=stage_id,
            stage_llm=stage_llm_getter(),
            llm_defaults=_llm_defaults_getter(),
            max_tool_turns=max_tool_turns_getter() if max_tool_turns_getter else None,
        )
        async with open_llm_stage(
            config=runtime_config,
            logger=logger,
        ) as factory:
            result = await orchestrator_fn(*args, session_factory=factory, **kwargs)
            payload = payload_builder(result)
            attach_trace(payload, factory.accumulated_trace)
            return payload

    return _run
