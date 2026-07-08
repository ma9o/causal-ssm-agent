"""Shared cross-cutting test helpers (LLM/session fakes, async runners)."""

import asyncio
from typing import Any


def run_async(coro):
    """Run an async coroutine synchronously in tests."""
    return asyncio.run(coro)


def invalid_dict_payload(value: object) -> Any:
    return value


def make_mock_session_factory(responses: list[str]):
    """Create a mock ``ScopedSessionFactory``-shaped object for tests.

    When ``.open(..., tools=[...])`` is entered and ``.turn()`` is called,
    the mock consumes the next canned response, invokes the first tool
    with it (so ``make_context_tool`` capture dicts populate the same way
    they do in the real tool loop), and returns it as the turn completion.

    ``accumulated_trace`` is present but empty — stages that attach a
    trace to their payload skip the attach when the trace is empty.
    """
    from contextlib import asynccontextmanager

    from nof1_causal_lab.utils.agent_session import AgentResult, TurnResult
    from nof1_causal_lab.utils.llm import LLMTrace

    call_count = [0]

    class _MockSession:
        def __init__(self, tools):
            self._tools = tools or []
            self._last_completion = ""

        async def turn(self, user_message: str) -> TurnResult:
            idx = min(call_count[0], len(responses) - 1)
            call_count[0] += 1
            response = responses[idx]

            if self._tools:
                tool = self._tools[0]
                props = tool.parameters.get("properties", {})
                required = tool.parameters.get("required", [])
                param_name = required[0] if required else next(iter(props), None)
                if param_name:
                    await tool(**{param_name: response})
                else:
                    await tool(response)

            self._last_completion = response
            return TurnResult(completion=response)

        @property
        def result(self) -> AgentResult:
            return AgentResult(completion=self._last_completion, trace=LLMTrace())

    class _MockFactory:
        accumulated_trace = LLMTrace()

        @asynccontextmanager
        async def open(self, *, system_prompt=None, tools=None, log_label=None):
            yield _MockSession(tools)

    return _MockFactory()
