"""Tests for utils/llm.py pure utility functions.

Covers: parse_json_response, _validate_json_and_format, attach_trace,
        dict_messages_to_chat.
"""

import pytest

from causal_ssm_agent.utils.llm import (
    LLMTrace,
    TraceMessage,
    _validate_json_and_format,
    attach_trace,
    parse_json_response,
)

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


# =============================================================================
# dict_messages_to_chat
# =============================================================================


class TestDictMessagesToChat:
    def test_system_message(self):
        from inspect_ai.model import ChatMessageSystem

        from causal_ssm_agent.utils.llm import dict_messages_to_chat

        msgs = dict_messages_to_chat([{"role": "system", "content": "Be helpful"}])
        assert len(msgs) == 1
        assert isinstance(msgs[0], ChatMessageSystem)
        assert msgs[0].content == "Be helpful"

    def test_user_message(self):
        from inspect_ai.model import ChatMessageUser

        from causal_ssm_agent.utils.llm import dict_messages_to_chat

        msgs = dict_messages_to_chat([{"role": "user", "content": "Hello"}])
        assert len(msgs) == 1
        assert isinstance(msgs[0], ChatMessageUser)

    def test_mixed_messages(self):
        from causal_ssm_agent.utils.llm import dict_messages_to_chat

        msgs = dict_messages_to_chat([
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User message"},
        ])
        assert len(msgs) == 2

    def test_unknown_role_skipped(self):
        from causal_ssm_agent.utils.llm import dict_messages_to_chat

        msgs = dict_messages_to_chat([{"role": "unknown", "content": "test"}])
        assert len(msgs) == 0

    def test_empty_list(self):
        from causal_ssm_agent.utils.llm import dict_messages_to_chat

        msgs = dict_messages_to_chat([])
        assert len(msgs) == 0
