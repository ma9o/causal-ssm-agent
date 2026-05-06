"""Tests for the embedded AgentSession wrapper.

Mirrors the behavioral coverage of ``multi_turn_generate`` in
``test_llm_utils.py`` but exercises the stateful turn-by-turn interface.
"""

import json

import pytest

from causal_ssm_agent.utils.agent_session_embedded import open_embedded_session
from causal_ssm_agent.utils.llm import LLMTrace
from tests.agent._support import make_worker_tool as _make_worker_tool
from tests.agent._support import valid_worker_output_json as _valid_worker_output_json
from tests.helpers import run_async as _run


def _tool_call_message(call_id: str, tool_name: str, arguments: dict) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "name": tool_name,
                "arguments": json.dumps(arguments),
            }
        ],
    }


def _tool_call_response(call_id: str, tool_name: str, arguments: dict) -> dict:
    return {
        "message": _tool_call_message(call_id, tool_name, arguments),
        "completion": "",
        "usage": None,
        "model": "test-model",
        "time": 0.1,
        "stop_reason": "tool_calls",
    }


def _text_response(text: str) -> dict:
    return {
        "message": {"role": "assistant", "content": text},
        "completion": text,
        "usage": None,
        "model": "test-model",
        "time": 0.1,
        "stop_reason": "stop",
    }


class TestEmbeddedSessionTurns:
    def test_single_turn_with_terminal_tool(self, monkeypatch):
        call_count = 0
        tool, capture = _make_worker_tool()

        async def fake_call_model(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            return _tool_call_response(
                "call_1",
                "validate_extractions",
                {"output_json": _valid_worker_output_json()},
            )

        monkeypatch.setattr("causal_ssm_agent.utils.llm.call_model", fake_call_model)

        async def scenario():
            async with open_embedded_session(
                model_name="test-model",
                system_prompt=None,
                tools=[tool],
            ) as session:
                result = await session.turn("Extract sleep hours")
                return result, session.result

        turn_result, final = _run(scenario())

        assert call_count == 1
        assert turn_result.terminal_tool_name == "validate_extractions"
        assert turn_result.terminal_tool_output == "VALID"
        assert turn_result.tool_calls_fired == ["validate_extractions"]
        assert capture["output"]["extractions"][0]["indicator"] == "sleep_hours"
        assert final.terminal_tool_name == "validate_extractions"

    def test_follow_up_turn_reuses_tools(self, monkeypatch):
        call_count = 0
        seen_tool_presence: list[bool] = []
        tool, capture = _make_worker_tool()

        async def fake_call_model(*_args, **kwargs):
            nonlocal call_count
            call_count += 1
            tools = kwargs.get("tools")
            seen_tool_presence.append(tools is not None)

            if call_count == 1:
                return _tool_call_response(
                    "call_1",
                    "validate_extractions",
                    {"output_json": _valid_worker_output_json()},
                )
            if call_count == 2:
                return _text_response("reviewed")
            raise AssertionError("unexpected call")

        monkeypatch.setattr("causal_ssm_agent.utils.llm.call_model", fake_call_model)

        async def scenario():
            async with open_embedded_session(
                model_name="test-model",
                system_prompt=None,
                tools=[tool],
            ) as session:
                await session.turn("Extract sleep hours")
                review = await session.turn("Review the extraction.")
                return review, session.result

        review_result, final = _run(scenario())

        assert call_count == 2
        assert seen_tool_presence == [True, True]
        assert review_result.completion == "reviewed"
        assert final.completion == "reviewed"
        assert capture["output"]["extractions"][0]["indicator"] == "sleep_hours"

    def test_result_includes_cumulative_trace(self, monkeypatch):
        tool, _ = _make_worker_tool()
        responses = iter(
            [
                _tool_call_response(
                    "call_1",
                    "validate_extractions",
                    {"output_json": _valid_worker_output_json()},
                ),
                _text_response("all good"),
            ]
        )

        async def fake_call_model(*_args, **_kwargs):
            return next(responses)

        monkeypatch.setattr("causal_ssm_agent.utils.llm.call_model", fake_call_model)

        async def scenario():
            async with open_embedded_session(
                model_name="test-model",
                system_prompt="sys",
                tools=[tool],
            ) as session:
                await session.turn("Extract sleep hours")
                await session.turn("Review the extraction.")
                return session.result

        final = _run(scenario())

        assert isinstance(final.trace, LLMTrace)
        roles = [msg.role for msg in final.trace.messages]
        # system, user1, assistant(tool_call), tool, user2, assistant(final)
        assert roles == ["system", "user", "assistant", "tool", "user", "assistant"]
        assert final.trace.messages[0].content == "sys"
        assert final.trace.messages[-1].content == "all good"

    def test_result_before_any_turn_raises(self):
        async def scenario():
            async with open_embedded_session(
                model_name="test-model",
                system_prompt=None,
                tools=[],
            ) as session:
                return session.result

        with pytest.raises(RuntimeError, match="before any turn"):
            _run(scenario())

    def test_tool_repair_after_call_model_error(self, monkeypatch):
        call_count = 0
        seen_retry_prompt: dict[str, str] = {}
        tool, capture = _make_worker_tool()

        async def fake_call_model(*args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("MALFORMED_FUNCTION_CALL: invalid tool payload")

            messages = args[1]
            seen_retry_prompt["content"] = messages[-1]["content"]
            return _tool_call_response(
                "call_1",
                "validate_extractions",
                {"output_json": _valid_worker_output_json()},
            )

        monkeypatch.setattr("causal_ssm_agent.utils.llm.call_model", fake_call_model)

        async def scenario():
            async with open_embedded_session(
                model_name="test-model",
                system_prompt=None,
                tools=[tool],
            ) as session:
                return await session.turn("Extract sleep hours")

        turn_result = _run(scenario())

        assert call_count == 2
        assert turn_result.terminal_tool_name == "validate_extractions"
        assert "MALFORMED_FUNCTION_CALL" in seen_retry_prompt["content"]
        assert capture["output"]["extractions"][0]["indicator"] == "sleep_hours"

    def test_timeout_retry_without_repair_prompt(self, monkeypatch):
        call_count = 0
        seen_second_messages: list[dict] = []
        tool, capture = _make_worker_tool()

        async def fake_call_model(*args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("call_model timed out after 120s")
            seen_second_messages.extend(args[1])
            return _tool_call_response(
                "call_1",
                "validate_extractions",
                {"output_json": _valid_worker_output_json()},
            )

        monkeypatch.setattr("causal_ssm_agent.utils.llm.call_model", fake_call_model)

        async def scenario():
            async with open_embedded_session(
                model_name="test-model",
                system_prompt=None,
                tools=[tool],
            ) as session:
                return await session.turn("Extract sleep hours")

        turn_result = _run(scenario())

        assert call_count == 2
        assert [m["role"] for m in seen_second_messages] == ["user"]
        assert turn_result.terminal_tool_name == "validate_extractions"
        assert capture["output"]["extractions"][0]["indicator"] == "sleep_hours"

    def test_max_tool_turns_exceeded_raises(self, monkeypatch):
        from causal_ssm_agent.utils.openrouter_client import Tool

        call_count = 0

        async def _retry() -> str:
            return "try again"

        retry_tool = Tool(
            name="retry_tool",
            description="Never terminal; forces the loop to keep going.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            execute=_retry,
        )

        async def fake_call_model(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            return _tool_call_response(f"call_{call_count}", "retry_tool", {})

        monkeypatch.setattr("causal_ssm_agent.utils.llm.call_model", fake_call_model)

        async def scenario():
            async with open_embedded_session(
                model_name="test-model",
                system_prompt=None,
                tools=[retry_tool],
                max_tool_turns=3,
            ) as session:
                await session.turn("Go")

        with pytest.raises(RuntimeError, match="exceeded 3 turns"):
            _run(scenario())

        assert call_count == 3
