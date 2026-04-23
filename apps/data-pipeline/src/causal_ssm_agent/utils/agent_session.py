"""Agent-driven multi-turn conversation primitives.

An ``AgentSession`` is a live, stateful conversation with a fixed system
prompt and tool set, opened against a backend. Callers drive the session
turn by turn; the backend handles the inner tool-use loop and returns
each turn's assistant output.

Two backends are anticipated:

* Embedded: calls OpenRouter directly via ``call_model``/``execute_tools``
  (see :mod:`causal_ssm_agent.utils.agent_session_embedded`).
* Harness: spawns an external agent CLI such as ``claude -p`` or
  ``codex exec`` and exposes tools over an in-process MCP server.

The protocol below fixes the calling shape so stages don't depend on
backend specifics. Today's :func:`multi_turn_generate` function still
works as a static seed + follow-ups wrapper; new callers that need
dynamic follow-ups or the harness backend should open a session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from causal_ssm_agent.utils.llm import LLMTrace


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
