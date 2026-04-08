"""Tests for utils/llm.py pure utility functions.

Covers: parse_json_response, _validate_json_and_format, attach_trace,
        dict_messages_to_chat, calculate, parse_date.
"""

import asyncio
import json
import logging
from typing import Any, cast

import pytest

from causal_ssm_agent.utils.llm import (
    LLMTrace,
    TraceMessage,
    _validate_json_and_format,
    attach_trace,
    make_generate_fn,
    make_validation_tool,
    multi_turn_generate,
    parse_json_response,
)
from causal_ssm_agent.workers.schemas import validate_worker_output
from tests.helpers import _run


def _require_mapping(value: object) -> dict[str, Any]:
    assert isinstance(value, dict)
    return {str(key): item for key, item in value.items()}


# =============================================================================
# parse_json_response
# =============================================================================


class TestParseJsonResponse:
    def test_plain_json(self):
        result = parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_in_markdown_block(self):
        content = '```json\n{"key": "value"}\n```'
        result = parse_json_response(content)
        assert result == {"key": "value"}

    def test_json_in_generic_code_block(self):
        content = '```\n{"key": "value"}\n```'
        result = parse_json_response(content)
        assert result == {"key": "value"}

    def test_nested_json(self):
        content = '{"a": {"b": [1, 2, 3]}}'
        result = parse_json_response(content)
        assert result == {"a": {"b": [1, 2, 3]}}

    def test_surrounding_whitespace(self):
        content = '  \n  {"key": 42}  \n  '
        result = parse_json_response(content)
        assert result == {"key": 42}

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Failed to parse"):
            parse_json_response("not json at all")

    def test_markdown_with_surrounding_text(self):
        content = 'Here is the JSON:\n```json\n{"x": 1}\n```\nDone!'
        result = parse_json_response(content)
        assert result == {"x": 1}


# =============================================================================
# _validate_json_and_format
# =============================================================================


class TestValidateJsonAndFormat:
    def test_valid_returns_valid_string(self):
        def validate(data):
            return data, []

        result = _validate_json_and_format('{"key": "value"}', validate)
        assert result == "VALID"

    def test_errors_returned_as_string(self):
        def validate(data):
            return None, ["Field 'x' is required", "Field 'y' must be positive"]

        result = _validate_json_and_format('{"key": "value"}', validate)
        assert "VALIDATION ERRORS" in result
        assert "Field 'x' is required" in result
        assert "Field 'y' must be positive" in result

    def test_invalid_json_returns_parse_error(self):
        def validate(data):
            return data, []

        result = _validate_json_and_format("not json", validate)
        assert "JSON parse error" in result

    def test_capture_stores_observation_rows(self):
        def validate(data):
            return "validated_result", []

        capture = {}
        _validate_json_and_format(
            '{"key": "value"}',
            validate,
            capture=capture,
            capture_key="test",
            capture_result=False,
        )
        assert capture["test"] == {"key": "value"}

    def test_capture_stores_result(self):
        def validate(data):
            return "validated_result", []

        capture = {}
        _validate_json_and_format(
            '{"key": "value"}',
            validate,
            capture=capture,
            capture_key="test",
            capture_result=True,
        )
        assert capture["test"] == "validated_result"

    def test_no_capture_on_errors(self):
        def validate(data):
            return None, ["error"]

        capture = {}
        _validate_json_and_format(
            '{"key": "value"}',
            validate,
            capture=capture,
            capture_key="test",
        )
        assert "test" not in capture


def _worker_schema():
    return {
        "latent": {
            "constructs": [
                {"name": "stress", "role": "exogenous"},
                {"name": "sleep", "role": "endogenous", "is_outcome": True},
            ],
            "edges": [{"cause": "stress", "effect": "sleep"}],
        },
        "measurement": {
            "model_clock": "1d",
            "indicators": [
                {
                    "name": "sleep_hours",
                    "construct_name": "sleep",
                    "measurement_dtype": "continuous",
                    "aggregation": "last",
                    "how_to_measure": "Read sleep hours directly from the rows",
                }
            ],
        },
    }


def _valid_worker_output_json() -> str:
    return json.dumps(
        {
            "extractions": [
                {
                    "indicator": "sleep_hours",
                    "value": 7.5,
                    "window_start": "2024-01-01T00:00:00Z",
                }
            ]
        }
    )


def _make_worker_tool(schema=None):
    """Create a worker extraction validation tool for tests."""
    if schema is None:
        schema = _worker_schema()
    return make_validation_tool(
        name="validate_extractions",
        description="Validate worker extraction output JSON.",
        param_name="output_json",
        param_description="The JSON string containing the worker output.",
        validator=lambda data: validate_worker_output(data, schema),
        capture_key="output",
    )


class TestWorkerValidationTools:
    def test_validate_worker_tool_stops_on_valid_output(self):
        tool, _capture = _make_worker_tool()
        assert tool.stop_on_success is True
        assert tool.success_output == "VALID"

    def test_multi_turn_generate_stops_after_valid_terminal_tool(self, monkeypatch):
        call_count = 0
        tool, capture = _make_worker_tool()

        async def fake_call_model(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "name": "validate_extractions",
                            "arguments": json.dumps({"output_json": _valid_worker_output_json()}),
                        }
                    ],
                },
                "completion": "",
                "usage": None,
                "model": "test-model",
                "time": 0.1,
                "stop_reason": "tool_calls",
            }

        monkeypatch.setattr("causal_ssm_agent.utils.llm.call_model", fake_call_model)

        result = _run(
            multi_turn_generate(
                messages=[{"role": "user", "content": "Extract sleep hours"}],
                model_name="test-model",
                tools=[tool],
            )
        )

        assert result == ""
        assert call_count == 1
        assert capture["output"]["extractions"][0]["indicator"] == "sleep_hours"

    def test_multi_turn_generate_retries_tool_turn_after_call_model_error(self, monkeypatch):
        call_count = 0
        seen_retry_prompt = {}
        tool, capture = _make_worker_tool()

        async def fake_call_model(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            messages = args[1]
            tools = kwargs.get("tools")
            if call_count == 1:
                assert tools is not None
                raise RuntimeError("MALFORMED_FUNCTION_CALL: invalid tool payload")

            seen_retry_prompt["content"] = messages[-1]["content"]
            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "name": "validate_extractions",
                            "arguments": json.dumps({"output_json": _valid_worker_output_json()}),
                        }
                    ],
                },
                "completion": "",
                "usage": None,
                "model": "test-model",
                "time": 0.1,
                "stop_reason": "tool_calls",
            }

        monkeypatch.setattr("causal_ssm_agent.utils.llm.call_model", fake_call_model)

        result = _run(
            multi_turn_generate(
                messages=[{"role": "user", "content": "Extract sleep hours"}],
                model_name="test-model",
                tools=[tool],
            )
        )

        assert result == ""
        assert call_count == 2
        assert "MALFORMED_FUNCTION_CALL" in seen_retry_prompt["content"]
        assert "validate_extractions" in seen_retry_prompt["content"]
        assert capture["output"]["extractions"][0]["indicator"] == "sleep_hours"

    def test_multi_turn_generate_retries_timeout_without_repair_prompt(self, monkeypatch):
        call_count = 0
        seen_second_messages: list[dict] = []
        tool, capture = _make_worker_tool()

        async def fake_call_model(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            messages = args[1]
            if call_count == 1:
                raise TimeoutError("call_model timed out after 120s")
            seen_second_messages.extend(messages)
            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "name": "validate_extractions",
                            "arguments": json.dumps({"output_json": _valid_worker_output_json()}),
                        }
                    ],
                },
                "completion": "",
                "usage": None,
                "model": "test-model",
                "time": 0.1,
                "stop_reason": "tool_calls",
            }

        monkeypatch.setattr("causal_ssm_agent.utils.llm.call_model", fake_call_model)

        result = _run(
            multi_turn_generate(
                messages=[{"role": "user", "content": "Extract sleep hours"}],
                model_name="test-model",
                tools=[tool],
            )
        )

        assert result == ""
        assert call_count == 2
        assert [message["role"] for message in seen_second_messages] == ["user"]
        assert capture["output"]["extractions"][0]["indicator"] == "sleep_hours"

    @pytest.mark.parametrize(
        "tool_result",
        [
            "JSON parse error: Expecting ',' delimiter",
            "VALIDATION ERRORS:\n- block_id is required",
        ],
    )
    def test_multi_turn_generate_continues_after_recoverable_terminal_tool_feedback(
        self,
        monkeypatch,
        tool_result: str,
    ):
        from causal_ssm_agent.utils.openrouter_client import Tool

        call_count = 0
        execute_count = 0

        async def _validate() -> str:
            nonlocal execute_count
            execute_count += 1
            if execute_count == 1:
                return tool_result
            return "BLOCK ACCEPTED"

        tool = Tool(
            name="validate_model",
            description="Stage 4 validation tool.",
            parameters={"type": "object", "properties": {}, "required": []},
            execute=_validate,
            stop_on_success=True,
            success_output=None,
        )

        async def fake_call_model(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            messages = args[1]

            if call_count == 1:
                assert [message["role"] for message in messages] == ["user"]
            elif call_count == 2:
                assert [message["role"] for message in messages] == ["user", "assistant", "tool"]
                assert messages[-1]["content"] == tool_result
            else:
                raise AssertionError("unexpected call")

            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call_{call_count}",
                            "name": "validate_model",
                            "arguments": "{}",
                        }
                    ],
                },
                "completion": "",
                "usage": None,
                "model": "test-model",
                "time": 0.1,
                "stop_reason": "tool_calls",
            }

        monkeypatch.setattr("causal_ssm_agent.utils.llm.call_model", fake_call_model)

        result = _run(
            multi_turn_generate(
                messages=[{"role": "user", "content": "Submit the active Stage 4 block"}],
                model_name="test-model",
                tools=[tool],
                max_tool_turns=3,
            )
        )

        assert result == ""
        assert call_count == 2
        assert execute_count == 2

    def test_multi_turn_generate_follow_up_gets_same_tools(self, monkeypatch):
        """Follow-up turns receive the same validation tool as the initial turn."""
        call_count = 0
        tool, capture = _make_worker_tool()

        async def fake_call_model(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            tools = kwargs.get("tools")

            if call_count == 1:
                # Initial turn: tool present
                assert tools is not None
                return {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "name": "validate_extractions",
                                "arguments": json.dumps(
                                    {"output_json": _valid_worker_output_json()}
                                ),
                            }
                        ],
                    },
                    "completion": "",
                    "usage": None,
                    "model": "test-model",
                    "time": 0.1,
                    "stop_reason": "tool_calls",
                }

            if call_count == 2:
                # Follow-up turn: same tool should be present
                assert tools is not None, "follow-up turn must receive the same tools"
                return {
                    "message": {
                        "role": "assistant",
                        "content": "confirmed",
                    },
                    "completion": "confirmed",
                    "usage": None,
                    "model": "test-model",
                    "time": 0.1,
                    "stop_reason": "stop",
                }

            raise AssertionError("unexpected call")

        monkeypatch.setattr("causal_ssm_agent.utils.llm.call_model", fake_call_model)

        result = _run(
            multi_turn_generate(
                messages=[{"role": "user", "content": "Extract sleep hours"}],
                model_name="test-model",
                tools=[tool],
                follow_ups=["Review the extraction."],
            )
        )

        assert result == "confirmed"
        assert call_count == 2
        assert capture["output"]["extractions"][0]["indicator"] == "sleep_hours"

    def test_multi_turn_generate_follow_up_tools_opt_out(self, monkeypatch):
        """Passing follow_up_tools=[] explicitly disables tools on follow-ups."""
        call_count = 0
        tool, _capture = _make_worker_tool()

        async def fake_call_model(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            tools = kwargs.get("tools")

            if call_count == 1:
                assert tools is not None
                return {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "name": "validate_extractions",
                                "arguments": json.dumps(
                                    {"output_json": _valid_worker_output_json()}
                                ),
                            }
                        ],
                    },
                    "completion": "",
                    "usage": None,
                    "model": "test-model",
                    "time": 0.1,
                    "stop_reason": "tool_calls",
                }

            if call_count == 2:
                # Explicit opt-out: no tools on follow-up
                assert tools is None
                return {
                    "message": {
                        "role": "assistant",
                        "content": "no-tool review",
                    },
                    "completion": "no-tool review",
                    "usage": None,
                    "model": "test-model",
                    "time": 0.1,
                    "stop_reason": "stop",
                }

            raise AssertionError("unexpected call")

        monkeypatch.setattr("causal_ssm_agent.utils.llm.call_model", fake_call_model)

        result = _run(
            multi_turn_generate(
                messages=[{"role": "user", "content": "Extract sleep hours"}],
                model_name="test-model",
                tools=[tool],
                follow_up_tools=[],
                follow_ups=["Review the extraction."],
            )
        )

        assert result == "no-tool review"
        assert call_count == 2

    def test_multi_turn_generate_respects_max_tool_turns(self, monkeypatch):
        from causal_ssm_agent.utils.openrouter_client import Tool

        call_count = 0

        async def _retry_tool() -> str:
            return "try again"

        tool = Tool(
            name="retry_tool",
            description="Never terminal; forces the loop to keep going.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            execute=_retry_tool,
        )

        async def fake_call_model(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call_{call_count}",
                            "name": "retry_tool",
                            "arguments": "{}",
                        }
                    ],
                },
                "completion": "",
                "usage": None,
                "model": "test-model",
                "time": 0.1,
                "stop_reason": "tool_calls",
            }

        monkeypatch.setattr("causal_ssm_agent.utils.llm.call_model", fake_call_model)

        with pytest.raises(RuntimeError, match="exceeded 3 turns"):
            _run(
                multi_turn_generate(
                    messages=[{"role": "user", "content": "Extract sleep hours"}],
                    model_name="test-model",
                    tools=[tool],
                    max_tool_turns=3,
                )
            )

        assert call_count == 3

    def test_multi_turn_generate_rewrites_context_but_preserves_trace(self, monkeypatch):
        from causal_ssm_agent.utils.openrouter_client import Tool

        call_count = 0
        rewrite_inputs: list[list[str]] = []
        trace_capture: dict[str, object] = {}

        def rewrite_messages(messages: list[dict]) -> list[dict]:
            rewrite_inputs.append([str(message["role"]) for message in messages])
            return [{"role": "user", "content": f"compact-{len(rewrite_inputs)}"}]

        async def _retry_tool() -> str:
            return "keep going"

        tool = Tool(
            name="retry_tool",
            description="Non-terminal tool used to force another model turn.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            execute=_retry_tool,
        )

        async def fake_call_model(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            messages = args[1]

            assert messages == [{"role": "user", "content": f"compact-{call_count}"}]

            if call_count == 1:
                return {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "name": "retry_tool",
                                "arguments": "{}",
                            }
                        ],
                    },
                    "completion": "",
                    "usage": None,
                    "model": "test-model",
                    "time": 0.1,
                    "stop_reason": "tool_calls",
                }

            if call_count == 2:
                return {
                    "message": {
                        "role": "assistant",
                        "content": "done",
                    },
                    "completion": "done",
                    "usage": None,
                    "model": "test-model",
                    "time": 0.1,
                    "stop_reason": "stop",
                }

            raise AssertionError("unexpected call")

        monkeypatch.setattr("causal_ssm_agent.utils.llm.call_model", fake_call_model)

        result = _run(
            multi_turn_generate(
                messages=[{"role": "user", "content": "original prompt"}],
                model_name="test-model",
                tools=[tool],
                trace_capture=trace_capture,
                rewrite_messages=rewrite_messages,
            )
        )

        assert result == "done"
        assert call_count == 2
        assert rewrite_inputs == [["user"], ["user", "assistant", "tool"]]

        trace = trace_capture["trace"]
        assert isinstance(trace, LLMTrace)
        assert [message.role for message in trace.messages] == [
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
        assert trace.messages[0].content == "original prompt"
        assert trace.messages[2].tool_result == "keep going"
        assert all("compact-" not in message.content for message in trace.messages)

    def test_multi_turn_generate_rewrites_available_tools_each_turn(self, monkeypatch):
        from causal_ssm_agent.utils.openrouter_client import Tool

        call_count = 0
        seen_tool_names: list[list[str]] = []

        async def _validate() -> str:
            return "keep going"

        async def _search() -> str:
            return "done"

        validate_tool = Tool(
            name="validate_model",
            description="validate",
            parameters={"type": "object", "properties": {}, "required": []},
            execute=_validate,
        )
        search_tool = Tool(
            name="search_literature",
            description="search",
            parameters={"type": "object", "properties": {}, "required": []},
            execute=_search,
        )

        def rewrite_tools(tools: list[Tool]) -> list[Tool]:
            if call_count == 0:
                return [tool for tool in tools if tool.name == "validate_model"]
            return [tool for tool in tools if tool.name == "search_literature"]

        async def fake_call_model(*args, **kwargs):
            nonlocal call_count
            tools = kwargs.get("tools") or []
            seen_tool_names.append([tool.name for tool in tools])
            call_count += 1

            if call_count == 1:
                return {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "name": "validate_model",
                                "arguments": "{}",
                            }
                        ],
                    },
                    "completion": "",
                    "usage": None,
                    "model": "test-model",
                    "time": 0.1,
                    "stop_reason": "tool_calls",
                }

            if call_count == 2:
                return {
                    "message": {
                        "role": "assistant",
                        "content": "done",
                    },
                    "completion": "done",
                    "usage": None,
                    "model": "test-model",
                    "time": 0.1,
                    "stop_reason": "stop",
                }

            raise AssertionError("unexpected call")

        monkeypatch.setattr("causal_ssm_agent.utils.llm.call_model", fake_call_model)

        result = _run(
            multi_turn_generate(
                messages=[{"role": "user", "content": "original prompt"}],
                model_name="test-model",
                tools=[validate_tool, search_tool],
                rewrite_tools=rewrite_tools,
            )
        )

        assert result == "done"
        assert seen_tool_names == [["validate_model"], ["search_literature"]]


# =============================================================================
# make_generate_fn
# =============================================================================


class TestMakeGenerateFn:
    def test_single_turn_capture_preserves_original_context_when_rewriting(self, monkeypatch):
        trace_capture: dict[str, object] = {}
        rewrite_inputs: list[list[str]] = []

        def rewrite_messages(messages: list[dict]) -> list[dict]:
            rewrite_inputs.append([str(message["role"]) for message in messages])
            return [{"role": "user", "content": "compact-context"}]

        async def fake_call_model(*args, **kwargs):
            messages = args[1]
            assert messages == [{"role": "user", "content": "compact-context"}]
            return {
                "message": {
                    "role": "assistant",
                    "content": "summary",
                    "reasoning": "hidden rationale",
                },
                "completion": "summary",
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 7,
                    "reasoning_tokens": 3,
                },
                "model": "test-model",
                "time": 0.2,
                "stop_reason": "stop",
            }

        monkeypatch.setattr("causal_ssm_agent.utils.llm.call_model", fake_call_model)

        generate = make_generate_fn("test-model", trace_capture=trace_capture)
        result = _run(
            generate(
                [{"role": "user", "content": "original prompt"}],
                label="single-turn",
                rewrite_messages=rewrite_messages,
            )
        )

        assert result == "summary"
        assert rewrite_inputs == [["user"]]

        trace = trace_capture["trace"]
        assert isinstance(trace, LLMTrace)
        assert [message.role for message in trace.messages] == ["user", "assistant"]
        assert trace.messages[0].content == "original prompt"
        assert trace.messages[1].content == "summary"
        assert trace.messages[1].reasoning == "hidden rationale"
        assert trace.usage.input_tokens == 11
        assert trace.usage.output_tokens == 7
        assert trace.usage.reasoning_tokens == 3
        assert all("compact-context" not in message.content for message in trace.messages)


# =============================================================================
# attach_trace
# =============================================================================


class TestAttachTrace:
    def test_attaches_trace(self):
        trace = LLMTrace(messages=[TraceMessage(role="user", content="hello")])
        capture = {"trace": trace}
        output = {}
        attach_trace(output, capture)
        assert "llm_trace" in output
        assert output["llm_trace"]["messages"][0]["content"] == "hello"

    def test_no_trace_no_op(self):
        capture = {}
        output = {}
        attach_trace(output, capture)
        assert "llm_trace" not in output


class _FakeChatCompletions:
    def __init__(self, response: dict[str, object], seen: dict[str, object]):
        self._response = response
        self._seen = seen

    async def create(self, **kwargs):
        self._seen["kwargs"] = kwargs
        return self._response


class _FakeOpenRouterClient:
    def __init__(self, response: dict[str, object], seen: dict[str, object]):
        self.chat = type(
            "_FakeChatNamespace",
            (),
            {"completions": _FakeChatCompletions(response, seen)},
        )()


class TestOpenRouterClient:
    def test_call_model_enforces_local_timeout(self, monkeypatch):
        from causal_ssm_agent.utils import openrouter_client

        seen: dict[str, object] = {}

        class _SlowChatCompletions:
            async def create(self, **kwargs):
                seen["kwargs"] = kwargs
                await asyncio.sleep(0.05)
                return {"choices": []}

        slow_client = type(
            "_SlowOpenRouterClient",
            (),
            {
                "chat": type(
                    "_SlowChatNamespace",
                    (),
                    {"completions": _SlowChatCompletions()},
                )()
            },
        )()

        monkeypatch.setattr(
            openrouter_client,
            "_get_openrouter_client",
            lambda _api_key=None: slow_client,
        )

        timeout_seconds = cast("Any", 0.01)
        with pytest.raises(TimeoutError, match=r"call_model timed out after 0\.01s"):
            _run(
                openrouter_client.call_model(
                    "test-model",
                    [{"role": "user", "content": "hello"}],
                    config=openrouter_client.GenerateConfig(timeout=timeout_seconds),
                )
            )

        assert _require_mapping(seen["kwargs"])["timeout"] == 0.01

    def test_call_model_logs_completion_tool_calls_and_reasoning(self, monkeypatch, caplog):
        from causal_ssm_agent.utils import openrouter_client

        seen: dict[str, object] = {}
        response = {
            "model": "test-model",
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "final answer",
                        "reasoning": "thought process",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "validate_extractions",
                                    "arguments": {"ok": True},
                                },
                            }
                        ],
                    },
                }
            ],
        }

        monkeypatch.setattr(
            openrouter_client,
            "_get_openrouter_client",
            lambda _api_key=None: _FakeOpenRouterClient(response, seen),
        )

        with caplog.at_level(logging.INFO):
            _run(
                openrouter_client.call_model(
                    "test-model",
                    [{"role": "user", "content": "hello"}],
                    config=openrouter_client.GenerateConfig(),
                    log_label="stage2 chunk=1",
                )
            )

        assert _require_mapping(seen["kwargs"])["model"] == "test-model"
        assert "call_model completion:\nfinal answer" in caplog.text
        assert '"name": "validate_extractions"' in caplog.text
        assert "call_model reasoning:\nthought process" in caplog.text

    def test_call_model_logs_completion_without_label(self, monkeypatch, caplog):
        from causal_ssm_agent.utils import openrouter_client

        seen: dict[str, object] = {}
        response = {
            "model": "test-model",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "unlabeled completion",
                    },
                }
            ],
        }

        monkeypatch.setattr(
            openrouter_client,
            "_get_openrouter_client",
            lambda _api_key=None: _FakeOpenRouterClient(response, seen),
        )

        with caplog.at_level(logging.INFO):
            _run(
                openrouter_client.call_model(
                    "test-model",
                    [{"role": "user", "content": "hello"}],
                    config=openrouter_client.GenerateConfig(),
                )
            )

        assert _require_mapping(seen["kwargs"])["messages"] == [
            {"role": "user", "content": "hello"}
        ]
        assert "call_model completion:\nunlabeled completion" in caplog.text

    def test_call_model_strips_repo_openrouter_prefix(self, monkeypatch):
        from causal_ssm_agent.utils import openrouter_client

        seen: dict[str, object] = {}
        response = {
            "model": "anthropic/claude-sonnet-4",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "ok",
                    },
                }
            ],
        }

        monkeypatch.setattr(
            openrouter_client,
            "_get_openrouter_client",
            lambda _api_key=None: _FakeOpenRouterClient(response, seen),
        )

        _run(
            openrouter_client.call_model(
                "openrouter/anthropic/claude-sonnet-4",
                [{"role": "user", "content": "hello"}],
                config=openrouter_client.GenerateConfig(),
            )
        )

        assert _require_mapping(seen["kwargs"])["model"] == "anthropic/claude-sonnet-4"

    def test_call_model_uses_request_local_openrouter_key(self, monkeypatch):
        from causal_ssm_agent.utils import openrouter_client

        seen: dict[str, object] = {}
        monkeypatch.setattr(openrouter_client, "_openrouter_clients", {})

        def fake_build_client(api_key: str | None):
            seen["api_key"] = api_key
            response = {
                "model": "test-model",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": "ok",
                        },
                    }
                ],
            }
            return _FakeOpenRouterClient(response, seen)

        monkeypatch.setattr(openrouter_client, "_build_openrouter_client", fake_build_client)

        with openrouter_client.use_openrouter_api_key("user-key"):
            _run(
                openrouter_client.call_model(
                    "test-model",
                    [{"role": "user", "content": "hello"}],
                    config=openrouter_client.GenerateConfig(),
                )
            )

        assert seen["api_key"] == "user-key"
        assert _require_mapping(seen["kwargs"])["model"] == "test-model"

    def test_call_model_uses_reasoning_config_in_extra_body(self, monkeypatch):
        from causal_ssm_agent.utils import openrouter_client

        seen: dict[str, object] = {}
        response = {
            "model": "test-model",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "ok",
                    },
                }
            ],
        }

        monkeypatch.setattr(
            openrouter_client,
            "_get_openrouter_client",
            lambda _api_key=None: _FakeOpenRouterClient(response, seen),
        )

        _run(
            openrouter_client.call_model(
                "test-model",
                [{"role": "user", "content": "hello"}],
                config=openrouter_client.GenerateConfig(reasoning_effort="high"),
            )
        )

        assert _require_mapping(seen["kwargs"])["extra_body"] == {
            "provider": {"sort": "throughput"},
            "reasoning": {"effort": "high"},
        }

    def test_use_openrouter_api_key_none_preserves_current_request_local_key(self):
        from causal_ssm_agent.utils import openrouter_client

        with (
            openrouter_client.use_openrouter_api_key("user-key"),
            openrouter_client.use_openrouter_api_key(None),
        ):
            assert openrouter_client.get_openrouter_api_key() == "user-key"


# =============================================================================
# dict_messages_to_chat
# =============================================================================


class TestDictMessagesToChat:
    def test_unknown_role_skipped(self):
        from causal_ssm_agent.utils.llm import dict_messages_to_chat

        msgs = dict_messages_to_chat([{"role": "unknown", "content": "test"}])
        assert len(msgs) == 0

    def test_empty_list(self):
        from causal_ssm_agent.utils.llm import dict_messages_to_chat

        msgs = dict_messages_to_chat([])
        assert len(msgs) == 0

    def test_preserves_reasoning_blocks(self):
        from causal_ssm_agent.utils.llm import dict_messages_to_chat

        msgs = dict_messages_to_chat(
            [
                {
                    "role": "assistant",
                    "content": "tool call pending",
                    "reasoning": "thinking",
                    "reasoning_details": [{"type": "reasoning.text", "text": "thinking"}],
                }
            ]
        )

        assert msgs[0]["reasoning"] == "thinking"
        assert msgs[0]["reasoning_details"] == [{"type": "reasoning.text", "text": "thinking"}]


class TestOpenRouterKeyContext:
    def test_llm_stage_task_uses_request_local_key_without_explicit_override(self, monkeypatch):
        from causal_ssm_agent.flows.llm_stage_task import make_llm_stage_task
        from causal_ssm_agent.utils import openrouter_client

        class _FakeLLMStageContext:
            def __init__(self, *_args, **_kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def make_generate(self, _model_name, **_kwargs):
                async def _generate(_messages):
                    return {"content": "ok"}

                return _generate

            def finalize(self, result):
                return result

        async def orchestrator_fn(*, generate):
            _ = await generate([{"role": "user", "content": "hello"}])
            return {"api_key": openrouter_client.get_openrouter_api_key()}

        monkeypatch.setattr(
            "causal_ssm_agent.flows.llm_stage_task.LLMStageContext",
            _FakeLLMStageContext,
        )

        task = make_llm_stage_task(
            stage_id="test-stage",
            orchestrator_fn=orchestrator_fn,
            payload_builder=lambda result: result,
            model_name_getter=lambda: "test-model",
            task_options={"cache_policy": None},
        )

        with openrouter_client.use_openrouter_api_key("user-key"):
            result = _run(task())

        assert result == {"api_key": "user-key"}

    def test_llm_stage_task_forwards_stage_max_tool_turns(self, monkeypatch):
        from causal_ssm_agent.flows.llm_stage_task import make_llm_stage_task

        captured: dict[str, object] = {}

        class _FakeLLMStageContext:
            def __init__(self, *_args, **_kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def make_generate(self, _model_name, **kwargs):
                captured["make_generate_kwargs"] = kwargs

                async def _generate(_messages):
                    return {"content": "ok"}

                return _generate

            def finalize(self, result):
                return result

        async def orchestrator_fn(*, generate):
            _ = await generate([{"role": "user", "content": "hello"}])
            return {"ok": True}

        monkeypatch.setattr(
            "causal_ssm_agent.flows.llm_stage_task.LLMStageContext",
            _FakeLLMStageContext,
        )

        task = make_llm_stage_task(
            stage_id="test-stage-turn-cap",
            orchestrator_fn=orchestrator_fn,
            payload_builder=lambda result: result,
            model_name_getter=lambda: "test-model",
            max_tool_turns_getter=lambda: 77,
            task_options={"cache_policy": None},
        )

        result = _run(task())

        assert result == {"ok": True}
        assert _require_mapping(captured["make_generate_kwargs"])["max_tool_turns"] == 77
