"""Embedded (OpenRouter-backed) AgentSession implementation.

Wraps the same primitives as :func:`multi_turn_generate` — ``call_model``,
``execute_tools``, the tool-repair + timeout-retry loop — but exposes
them as a stateful session so callers can drive turns individually and
decide follow-up prompts based on per-turn results.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from nof1_causal_lab.utils.agent_session import AgentResult, TurnResult
from nof1_causal_lab.utils.llm import (
    DEFAULT_MAX_TOOL_LOOP_TURNS,
    _build_trace,
    _call_model_with_tool_repair,
    _combine_log_label,
    _run_tool_loop,
    _terminal_tool_success,
    get_generate_config,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from nof1_causal_lab.utils.openrouter_client import GenerateConfig, Tool


class EmbeddedSession:
    """AgentSession backed by OpenRouter via ``call_model``/``execute_tools``."""

    def __init__(
        self,
        *,
        model_name: str,
        system_prompt: str | None,
        tools: list[Tool],
        config: GenerateConfig | None = None,
        log_label: str | None = None,
        max_tool_turns: int = DEFAULT_MAX_TOOL_LOOP_TURNS,
    ) -> None:
        self._model_name = model_name
        self._tools = list(tools)
        self._config = config or get_generate_config()
        self._log_label = log_label
        self._max_tool_turns = max_tool_turns

        self._messages: list[dict[str, Any]] = []
        if system_prompt is not None:
            self._messages.append({"role": "system", "content": system_prompt})
        self._trace_messages: list[dict[str, Any]] = list(self._messages)

        self._last_output: dict[str, Any] | None = None
        self._last_nonempty: str = ""
        self._terminal_tool: tuple[str, str] | None = None
        self._turn_index: int = 0

    async def turn(self, user_message: str) -> TurnResult:
        self._turn_index += 1
        user_msg: dict[str, Any] = {"role": "user", "content": user_message}
        self._messages.append(user_msg)
        self._trace_messages.append(user_msg)

        turn_label = f"turn-{self._turn_index}"

        if self._tools:
            self._messages, output = await _run_tool_loop(
                self._messages,
                self._trace_messages,
                self._model_name,
                self._tools,
                self._config,
                label=turn_label,
                log_label=self._log_label,
                max_turns=self._max_tool_turns,
            )
        else:
            output = await _call_model_with_tool_repair(
                self._messages,
                self._model_name,
                tools=None,
                config=self._config,
                log_label=_combine_log_label(self._log_label, turn_label, "llm"),
                trace_messages=self._trace_messages,
            )
            self._messages.append(output["message"])
            self._trace_messages.append(output["message"])

        self._last_output = output
        completion = output.get("completion", "")
        if completion and completion.strip():
            self._last_nonempty = completion

        tail_tool_messages: list[dict[str, Any]] = []
        for msg in reversed(self._trace_messages):
            if msg.get("role") == "tool":
                tail_tool_messages.insert(0, msg)
                continue
            break

        tool_calls_fired = [str(msg["name"]) for msg in tail_tool_messages]
        terminal: tuple[str, str] | None = None
        if tail_tool_messages:
            terminal = _terminal_tool_success(tail_tool_messages, self._tools)
        if terminal is not None:
            self._terminal_tool = terminal

        return TurnResult(
            completion=completion,
            terminal_tool_name=terminal[0] if terminal else None,
            terminal_tool_output=terminal[1] if terminal else None,
            tool_calls_fired=tool_calls_fired,
        )

    @property
    def result(self) -> AgentResult:
        if self._last_output is None:
            raise RuntimeError("result requested before any turn was executed")
        trace = _build_trace(self._trace_messages, self._last_output)
        return AgentResult(
            completion=self._last_nonempty,
            trace=trace,
            terminal_tool_name=self._terminal_tool[0] if self._terminal_tool else None,
            terminal_tool_output=self._terminal_tool[1] if self._terminal_tool else None,
        )

    async def aclose(self) -> None:
        # Embedded sessions hold no external resources; this is a no-op
        # to keep lifecycle symmetric with harness-backed sessions.
        return None


@asynccontextmanager
async def open_embedded_session(
    *,
    model_name: str,
    system_prompt: str | None = None,
    tools: list[Tool] | None = None,
    config: GenerateConfig | None = None,
    log_label: str | None = None,
    max_tool_turns: int = DEFAULT_MAX_TOOL_LOOP_TURNS,
) -> AsyncIterator[EmbeddedSession]:
    """Open an embedded ``AgentSession`` scoped to an ``async with`` block."""
    from nof1_causal_lab.utils.config import ensure_harness_prereqs

    ensure_harness_prereqs("none")
    session = EmbeddedSession(
        model_name=model_name,
        system_prompt=system_prompt,
        tools=tools or [],
        config=config,
        log_label=log_label,
        max_tool_turns=max_tool_turns,
    )
    try:
        yield session
    finally:
        await session.aclose()
