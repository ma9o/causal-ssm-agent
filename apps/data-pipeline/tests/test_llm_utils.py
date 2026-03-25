"""Tests for utils/llm.py pure utility functions.

Covers: parse_json_response, _validate_json_and_format, attach_trace,
        dict_messages_to_chat, calculate, parse_date.
"""

import json
import logging

import pytest

from causal_ssm_agent.utils.llm import (
    LLMTrace,
    TraceMessage,
    _validate_json_and_format,
    attach_trace,
    make_validation_tool,
    multi_turn_generate,
    parse_json_response,
)
from causal_ssm_agent.workers.schemas import validate_worker_output
from tests.helpers import _run

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

    def test_empty_object(self):
        result = parse_json_response("{}")
        assert result == {}


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

        assert seen["kwargs"]["model"] == "test-model"
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

        assert seen["kwargs"]["messages"] == [{"role": "user", "content": "hello"}]
        assert "call_model completion:\nunlabeled completion" in caplog.text

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
        assert seen["kwargs"]["model"] == "test-model"

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

        assert seen["kwargs"]["extra_body"] == {"reasoning": {"effort": "high"}}

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
        from causal_ssm_agent.flows.stages.llm_stage_task import make_llm_stage_task
        from causal_ssm_agent.utils import openrouter_client

        class _FakeLLMStageContext:
            def __init__(self, *_args, **_kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def make_generate(self, _model_name):
                async def _generate(_messages):
                    return {"content": "ok"}

                return _generate

            def finalize(self, result):
                return result

        async def orchestrator_fn(*, generate):
            _ = await generate([{"role": "user", "content": "hello"}])
            return {"api_key": openrouter_client.get_openrouter_api_key()}

        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.llm_stage_task.LLMStageContext",
            _FakeLLMStageContext,
        )

        task = make_llm_stage_task(
            stage_id="test-stage",
            orchestrator_fn=orchestrator_fn,
            payload_builder=lambda result: result,
            model_name_getter=lambda: "test-model",
        )

        with openrouter_client.use_openrouter_api_key("user-key"):
            result = _run(task())

        assert result == {"api_key": "user-key"}
