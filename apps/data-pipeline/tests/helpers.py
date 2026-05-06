"""Shared test helpers for LLM/session fakes.

These are utilities that can be imported directly into test modules.
For fixtures, see conftest.py.
"""

from typing import Any


def invalid_dict_payload(value: object) -> Any:
    return value


def make_mock_session_factory(responses: list[str]):
    """Create a mock ``StageSessionFactory``-shaped object for tests.

    When ``.open(..., tools=[...])`` is entered and ``.turn()`` is called,
    the mock consumes the next canned response, invokes the first tool
    with it (so ``make_stage_tool`` capture dicts populate the same way
    they do in the real tool loop), and returns it as the turn completion.

    ``accumulated_trace`` is present but empty — stages that attach a
    trace to their payload skip the attach when the trace is empty.
    """
    from contextlib import asynccontextmanager

    from causal_ssm_agent.utils.agent_session import AgentResult, TurnResult
    from causal_ssm_agent.utils.llm import LLMTrace

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


def make_session_factory_from_handler(handler):
    """Build a mock session factory from a per-turn handler.

    ``handler(tools, user_message) -> completion`` is called each time
    the session's ``turn(user_message)`` runs; the handler can dispatch
    to any tool in ``tools`` and return a completion string.

    Useful when a test needs to invoke multiple tools per turn (e.g.
    Stage 0 ingestion tests that exercise a tool sequence).
    """
    from contextlib import asynccontextmanager

    from causal_ssm_agent.utils.agent_session import AgentResult, TurnResult
    from causal_ssm_agent.utils.llm import LLMTrace

    class _Session:
        def __init__(self, tools):
            self._tools = tools or []
            self._completion = ""

        async def turn(self, user_message: str) -> TurnResult:
            completion = await handler(self._tools, user_message)
            self._completion = completion or ""
            return TurnResult(completion=self._completion)

        @property
        def result(self) -> AgentResult:
            return AgentResult(completion=self._completion, trace=LLMTrace())

    class _Factory:
        accumulated_trace = LLMTrace()

        @asynccontextmanager
        async def open(self, *, system_prompt=None, tools=None, log_label=None):
            yield _Session(tools)

    return _Factory()

