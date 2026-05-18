"""Agent-driven multi-turn conversation primitives.

An ``AgentSession`` is a live, stateful conversation with a fixed system
prompt and tool set, opened against a backend. Callers drive the session
turn by turn; the backend handles the inner tool-use loop and returns
each turn's assistant output.

Two backends are supported:

* Embedded: calls OpenRouter directly via ``call_model``/``execute_tools``
  (see :mod:`nof1_causal_lab.utils.agent_session_embedded`).
* Harness: spawns an external agent CLI such as ``claude -p`` or
  ``codex exec`` and exposes tools over an in-process MCP server
  (see :mod:`nof1_causal_lab.utils.harness`).

Stages don't talk to those modules directly: they receive a
:class:`StageSessionFactory` that already knows the stage's
:class:`StageLLMConfig` and accumulates LLM traces across every session
it opens. ``StageSessionFactory.open(...)`` returns an async context
manager yielding a live :class:`AgentSession`.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from nof1_causal_lab.utils.llm import LLMTrace, _merge_trace

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from nof1_causal_lab.utils.config import LLMDefaults, StageLLMConfig
    from nof1_causal_lab.utils.openrouter_client import Tool


@dataclass
class TurnResult:
    """Outcome of one ``AgentSession.turn`` call.

    A turn sends one user message and runs the backend's inner tool loop
    until the model produces a final text message or a tool marked
    ``stop_on_success`` fires and returns its success sentinel.
    """

    completion: str
    terminal_tool_name: str | None = None
    terminal_tool_output: str | None = None
    tool_calls_fired: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    """Cumulative result of a completed session."""

    completion: str
    trace: LLMTrace
    terminal_tool_name: str | None = None
    terminal_tool_output: str | None = None


class AgentSession(Protocol):
    """A live multi-turn conversation with a fixed tool set."""

    async def turn(self, user_message: str) -> TurnResult: ...

    @property
    def result(self) -> AgentResult: ...


class StageSessionFactory:
    """Opens :class:`AgentSession` instances for a stage, accumulating traces.

    Stages receive an instance of this class instead of a plain generate
    function; each ``.open(...)`` call yields a fresh session bound to
    the configured backend. After every session closes, its
    :class:`LLMTrace` is merged into :attr:`accumulated_trace` so the
    outer stage wrapper can attach one combined trace to its output
    payload.
    """

    def __init__(
        self,
        stage_llm: StageLLMConfig,
        llm_defaults: LLMDefaults,
        *,
        stage_id: str,
        max_tool_turns: int | None = None,
    ) -> None:
        self._stage_llm = stage_llm
        self._llm_defaults = llm_defaults
        self._stage_id = stage_id
        self._max_tool_turns = max_tool_turns
        self.accumulated_trace: LLMTrace = LLMTrace()

    @asynccontextmanager
    async def open(
        self,
        *,
        system_prompt: str | None = None,
        tools: list[Tool] | None = None,
        log_label: str | None = None,
    ) -> AsyncIterator[AgentSession]:
        """Open a backend-appropriate :class:`AgentSession`."""
        from nof1_causal_lab.utils.agent_session_factory import open_session

        async with open_session(
            self._stage_llm,
            self._llm_defaults,
            system_prompt=system_prompt,
            tools=tools or [],
            log_label=log_label or self._stage_id,
            max_tool_turns=self._max_tool_turns,
        ) as session:
            try:
                yield session
            finally:
                try:
                    this_trace = session.result.trace
                except RuntimeError:
                    # No turns were executed inside the block.
                    this_trace = None
                if this_trace is not None:
                    self.accumulated_trace = _merge_trace(self.accumulated_trace, this_trace)
