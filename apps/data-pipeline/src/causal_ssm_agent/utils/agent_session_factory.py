"""Backend-dispatching factory for :class:`AgentSession` context managers.

Stages call :func:`open_session` with their :class:`StageLLMConfig` (plus
the global :class:`LLMDefaults`) and a tool list. The factory picks the
backend, merges per-stage overrides with global defaults, and yields a
live :class:`AgentSession` — the stage doesn't need to know whether it
got an embedded OpenRouter session or a harness subprocess.

Per-stage overrides (set on ``StageLLMConfig``) take precedence over
the matching global default; ``None`` means "inherit". Fields that are
invalid for a given harness are rejected by :func:`validate_config` at
load time, so dispatch here can trust the shape.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from causal_ssm_agent.utils.agent_session_embedded import open_embedded_session
from causal_ssm_agent.utils.harness.claude import open_claude_harness_session
from causal_ssm_agent.utils.harness.codex import open_codex_harness_session
from causal_ssm_agent.utils.llm import DEFAULT_MAX_TOOL_LOOP_TURNS
from causal_ssm_agent.utils.openrouter_client import GenerateConfig

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from causal_ssm_agent.utils.agent_session import AgentSession
    from causal_ssm_agent.utils.config import LLMDefaults, StageLLMConfig
    from causal_ssm_agent.utils.openrouter_client import Tool


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


@asynccontextmanager
async def open_session(
    stage_llm: StageLLMConfig,
    llm_defaults: LLMDefaults,
    *,
    system_prompt: str | None,
    tools: list[Tool],
    log_label: str | None = None,
    max_tool_turns: int | None = None,
) -> AsyncIterator[AgentSession]:
    """Open an :class:`AgentSession` against the configured backend.

    ``stage_llm.harness`` picks between ``"none"`` (embedded OpenRouter),
    ``"claude-code"`` (``claude -p``), and ``"codex"`` (``codex exec``).
    Per-stage overrides on ``stage_llm`` trump the matching global
    default in ``llm_defaults``.
    """
    harness = stage_llm.harness
    label = log_label

    if harness == "none":
        embedded = llm_defaults.embedded
        config = GenerateConfig(
            max_tokens=_first_not_none(stage_llm.max_tokens, embedded.max_tokens),
            timeout=_first_not_none(stage_llm.timeout, embedded.timeout),
            reasoning_effort=_first_not_none(stage_llm.reasoning_effort, embedded.reasoning_effort),
        )
        async with open_embedded_session(
            model_name=stage_llm.model,
            system_prompt=system_prompt,
            tools=tools,
            config=config,
            log_label=label,
            max_tool_turns=max_tool_turns or DEFAULT_MAX_TOOL_LOOP_TURNS,
        ) as session:
            yield session
        return

    if harness == "claude-code":
        defaults = llm_defaults.claude_code
        async with open_claude_harness_session(
            tools=tools,
            system_prompt=system_prompt,
            model=stage_llm.model,
            bin=_first_not_none(stage_llm.bin, defaults.bin),
            effort=_first_not_none(stage_llm.effort, defaults.effort),
            max_turns=_first_not_none(stage_llm.max_turns, max_tool_turns, defaults.max_turns),
            max_budget_usd=_first_not_none(stage_llm.max_budget_usd, defaults.max_budget_usd),
            fallback_model=_first_not_none(stage_llm.fallback_model, defaults.fallback_model),
            log_label=label,
        ) as session:
            yield session
        return

    if harness == "codex":
        defaults = llm_defaults.codex
        async with open_codex_harness_session(
            tools=tools,
            model=stage_llm.model,
            bin=_first_not_none(stage_llm.bin, defaults.bin),
            reasoning_effort=_first_not_none(stage_llm.reasoning_effort, defaults.reasoning_effort),
            log_label=label,
        ) as session:
            yield session
        return

    raise ValueError(f"Unknown harness {harness!r} in stage config")
