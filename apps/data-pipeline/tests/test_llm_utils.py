"""Tests for utils/llm.py pure utility functions.

Covers: parse_json_response, _validate_json_and_format, attach_trace,
        dict_messages_to_chat, calculate, parse_date.
"""

import asyncio
import logging

import litellm
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

    def test_capture_stores_raw_data(self):
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
            "indicators": [
                {
                    "name": "sleep_hours",
                    "construct_name": "sleep",
                    "measurement_dtype": "continuous",
                    "how_to_measure": "Read sleep hours directly from the rows",
                }
            ]
        },
    }


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
                            "arguments": '{"output_json":"{\\"extractions\\":[{\\"indicator\\":\\"sleep_hours\\",\\"value\\":7.5,\\"timestamp\\":\\"2024-01-01T00:00:00Z\\"}]}"}',
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
                            "arguments": '{"output_json":"{\\"extractions\\":[{\\"indicator\\":\\"sleep_hours\\",\\"value\\":7.5,\\"timestamp\\":\\"2024-01-01T00:00:00Z\\"}]}"}',
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
                                "arguments": '{"output_json":"{\\"extractions\\":[{\\"indicator\\":\\"sleep_hours\\",\\"value\\":7.5,\\"timestamp\\":\\"2024-01-01T00:00:00Z\\"}]}"}',
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
                                "arguments": '{"output_json":"{\\"extractions\\":[{\\"indicator\\":\\"sleep_hours\\",\\"value\\":7.5,\\"timestamp\\":\\"2024-01-01T00:00:00Z\\"}]}"}',
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

    def test_none_trace_no_op(self):
        capture = {"trace": None}
        output = {}
        attach_trace(output, capture)
        assert "llm_trace" not in output


class TestLiteLLMAsyncLoggingPatch:
    def test_skips_async_success_logging_without_callbacks(self, monkeypatch):
        from causal_ssm_agent.utils import litellm_client

        called = False

        async def fake_original(*args, **kwargs):
            nonlocal called
            called = True

        class DummyLoggingObj:
            dynamic_async_success_callbacks = None

        monkeypatch.setattr(litellm_client, "_ORIGINAL_CLIENT_ASYNC_LOGGING_HELPER", fake_original)
        monkeypatch.setattr(litellm, "_async_success_callback", [])

        _run(
            litellm_client._quiet_client_async_logging_helper(
                logging_obj=DummyLoggingObj(),
                result=None,
                start_time=None,
                end_time=None,
                is_completion_with_fallbacks=False,
            )
        )

        assert called is False

    def test_preserves_async_success_logging_when_callbacks_exist(self, monkeypatch):
        from causal_ssm_agent.utils import litellm_client

        seen = {}

        async def fake_original(*args, **kwargs):
            seen["called"] = True
            seen["kwargs"] = kwargs

        class DummyLoggingObj:
            dynamic_async_success_callbacks = ("callback",)

        monkeypatch.setattr(litellm_client, "_ORIGINAL_CLIENT_ASYNC_LOGGING_HELPER", fake_original)
        monkeypatch.setattr(litellm, "_async_success_callback", [])

        _run(
            litellm_client._quiet_client_async_logging_helper(
                logging_obj=DummyLoggingObj(),
                result="ok",
                start_time=1,
                end_time=2,
                is_completion_with_fallbacks=False,
            )
        )

        assert seen["called"] is True
        assert seen["kwargs"]["result"] == "ok"


class TestVerboseResponseLogging:
    def test_call_model_logs_completion_tool_calls_and_reasoning(self, monkeypatch, caplog):
        from causal_ssm_agent.utils import litellm_client

        async def fake_acompletion(**kwargs):
            return {
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

        monkeypatch.setattr(litellm_client, "acompletion", fake_acompletion)

        with caplog.at_level(logging.INFO):
            _run(
                litellm_client.call_model(
                    "test-model",
                    [{"role": "user", "content": "hello"}],
                    config=litellm_client.GenerateConfig(
                        verbose_logging=True,
                        log_reasoning=True,
                        log_output_char_limit=1000,
                    ),
                    log_label="stage2 chunk=1",
                )
            )

        assert "call_model completion:\nfinal answer" in caplog.text
        assert '"name": "validate_extractions"' in caplog.text
        assert "call_model reasoning:\nthought process" in caplog.text

    def test_call_model_logs_completion_without_label(self, monkeypatch, caplog):
        from causal_ssm_agent.utils import litellm_client

        async def fake_acompletion(**kwargs):
            return {
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

        monkeypatch.setattr(litellm_client, "acompletion", fake_acompletion)

        with caplog.at_level(logging.INFO):
            _run(
                litellm_client.call_model(
                    "test-model",
                    [{"role": "user", "content": "hello"}],
                    config=litellm_client.GenerateConfig(verbose_logging=True),
                )
            )

        assert "call_model completion:\nunlabeled completion" in caplog.text

    def test_call_model_verbose_logs_respect_char_limit(self, monkeypatch, caplog):
        from causal_ssm_agent.utils import litellm_client

        async def fake_acompletion(**kwargs):
            return {
                "model": "test-model",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": "abcdefghijklmnopqrstuvwxyz",
                        },
                    }
                ],
            }

        monkeypatch.setattr(litellm_client, "acompletion", fake_acompletion)

        with caplog.at_level(logging.INFO):
            _run(
                litellm_client.call_model(
                    "test-model",
                    [{"role": "user", "content": "hello"}],
                    config=litellm_client.GenerateConfig(
                        verbose_logging=True,
                        log_output_char_limit=10,
                    ),
                    log_label="stage2 chunk=2",
                )
            )

        assert "abcdefghij" in caplog.text
        assert "[truncated 16 chars]" in caplog.text


# =============================================================================
# dict_messages_to_chat
# =============================================================================


class TestDictMessagesToChat:
    def test_system_message(self):
        from causal_ssm_agent.utils.llm import dict_messages_to_chat

        msgs = dict_messages_to_chat([{"role": "system", "content": "Be helpful"}])
        assert len(msgs) == 1
        assert msgs[0] == {"role": "system", "content": "Be helpful"}

    def test_user_message(self):
        from causal_ssm_agent.utils.llm import dict_messages_to_chat

        msgs = dict_messages_to_chat([{"role": "user", "content": "Hello"}])
        assert len(msgs) == 1
        assert msgs[0] == {"role": "user", "content": "Hello"}

    def test_mixed_messages(self):
        from causal_ssm_agent.utils.llm import dict_messages_to_chat

        msgs = dict_messages_to_chat(
            [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "User message"},
            ]
        )
        assert len(msgs) == 2

    def test_unknown_role_skipped(self):
        from causal_ssm_agent.utils.llm import dict_messages_to_chat

        msgs = dict_messages_to_chat([{"role": "unknown", "content": "test"}])
        assert len(msgs) == 0

    def test_empty_list(self):
        from causal_ssm_agent.utils.llm import dict_messages_to_chat

        msgs = dict_messages_to_chat([])
        assert len(msgs) == 0


# =============================================================================
# calculate (tool function)
# =============================================================================


def _run(coro):
    """Run an async function synchronously for testing."""
    return asyncio.run(coro)


class TestCalculate:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from causal_ssm_agent.utils.llm import calculate

        self.calc = calculate()

    def test_addition(self):
        assert _run(self.calc("2 + 3")) == "5"

    def test_multiplication(self):
        assert _run(self.calc("4 * 5")) == "20"

    def test_division(self):
        assert _run(self.calc("10 / 4")) == "2.5"

    def test_floor_division(self):
        assert _run(self.calc("10 // 3")) == "3"

    def test_modulo(self):
        assert _run(self.calc("10 % 3")) == "1"

    def test_exponent(self):
        assert _run(self.calc("2 ** 8")) == "256"

    def test_negative(self):
        assert _run(self.calc("-5 + 3")) == "-2"

    def test_parentheses(self):
        assert _run(self.calc("(10 + 5) * 2")) == "30"

    def test_complex_expression(self):
        assert _run(self.calc("2 + 3 * 4")) == "14"

    def test_division_by_zero(self):
        result = _run(self.calc("1 / 0"))
        assert "Error" in result

    def test_invalid_expression(self):
        result = _run(self.calc("import os"))
        assert "Error" in result

    def test_function_call_rejected(self):
        result = _run(self.calc("__import__('os')"))
        assert "Error" in result

    def test_float_result(self):
        assert _run(self.calc("1.5 + 2.5")) == "4.0"


# =============================================================================
# parse_date (tool function)
# =============================================================================


class TestParseDate:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from causal_ssm_agent.utils.llm import parse_date

        self.parse = parse_date()

    def test_iso_date(self):
        result = _run(self.parse("2024-03-15"))
        assert "March" in result
        assert "15" in result
        assert "2024" in result

    def test_iso_datetime(self):
        result = _run(self.parse("2024-03-15T10:30:00"))
        assert "March" in result
        assert "15" in result

    def test_iso_datetime_with_z(self):
        result = _run(self.parse("2024-03-15T10:30:00Z"))
        assert "March" in result

    def test_slash_date_ymd(self):
        result = _run(self.parse("2024/03/15"))
        assert "March" in result

    def test_unparseable_date(self):
        result = _run(self.parse("not-a-date"))
        assert "Could not parse" in result

    def test_whitespace_stripped(self):
        result = _run(self.parse("  2024-03-15  "))
        assert "March" in result

    def test_day_of_week_included(self):
        result = _run(self.parse("2024-03-15"))
        assert "Friday" in result
