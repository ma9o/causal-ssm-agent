"""Shared turn and aggregate result values for harness conversations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nof1_causal_lab.utils.llm import LLMTrace


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
