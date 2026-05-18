"""Shared lifecycle helpers for LLM-backed stages."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from causal_ssm_agent.utils.agent_session import StageSessionFactory
from causal_ssm_agent.utils.openrouter_client import use_openrouter_api_key

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from logging import Logger

    from causal_ssm_agent.utils.config import LLMDefaults, StageLLMConfig
    from causal_ssm_agent.utils.llm import LLMTrace


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
    openrouter_api_key: str | None,
    logger: Logger,
) -> AsyncIterator[StageSessionFactory]:
    """Open the OpenRouter context and stage session factory for one stage run."""
    started_at = time.monotonic()
    with use_openrouter_api_key(openrouter_api_key):
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
