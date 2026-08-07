"""Tests for the Claude and Codex stream-json parsers.

Fixture events follow the documented shapes from the Claude Code CLI
reference (``--output-format stream-json``) and the Codex CLI
``exec --json`` reference. Real-world event payloads carry additional
metadata fields; the parsers record those in ``raw_events`` without
failing the primary trace reconstruction.
"""

import json
from typing import Any

import pytest

from nof1_causal_lab.utils.harness.stream_json import (
    ClaudeStreamState,
    CodexStreamState,
    PiStreamState,
    apply_claude_event,
    apply_codex_event,
    apply_pi_event,
    finalize_codex_trace,
    finalize_trace,
    format_claude_event_for_log,
    format_codex_event_for_log,
    format_pi_event_for_log,
)


def _parse_stream(lines, state, apply_event, label):
    for idx, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{label} stream line {idx} is not valid JSON: {line[:200]!r}"
            ) from exc
        if not isinstance(event, dict):
            raise ValueError(f"{label} stream line {idx} is JSON but not an object: {line[:200]!r}")
        apply_event(state, event)
    return state


def parse_claude_stream(lines):
    return _parse_stream(lines, ClaudeStreamState(), apply_claude_event, "claude")


def parse_codex_stream(lines):
    return _parse_stream(lines, CodexStreamState(), apply_codex_event, "codex")


def parse_pi_stream(lines):
    return _parse_stream(lines, PiStreamState(), apply_pi_event, "pi")


def _pi_events_tool_loop() -> list[dict[str, Any]]:
    return [
        {"type": "session", "version": 3, "id": "pi-session"},
        {
            "type": "message_end",
            "message": {"role": "user", "content": "Validate this", "timestamp": 1000},
        },
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "call-1",
                        "name": "validate_model",
                        "arguments": {"payload": "x"},
                    }
                ],
                "model": "gpt-5.4-mini",
                "usage": {"input": 20, "output": 5, "reasoning": 3},
                "stopReason": "toolUse",
                "timestamp": 1500,
            },
        },
        {
            "type": "tool_execution_end",
            "toolCallId": "call-1",
            "toolName": "validate_model",
            "result": {"content": [{"type": "text", "text": "VALID"}], "details": {}},
            "isError": False,
        },
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Done."}],
                "model": "gpt-5.4-mini",
                "usage": {"input": 8, "output": 2, "reasoning": 1},
                "stopReason": "stop",
                "timestamp": 2200,
            },
        },
        {"type": "nof1.turn_timing", "duration_seconds": 1.2},
    ]


class TestPiParser:
    def test_reconstructs_tool_loop_and_usage(self):
        state = parse_pi_stream([json.dumps(event) for event in _pi_events_tool_loop()])

        assert state.session_id == "pi-session"
        assert state.model == "gpt-5.4-mini"
        assert state.final_text == "Done."
        assert [message.role for message in state.messages] == [
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
        tool_calls = state.messages[1].tool_calls
        assert tool_calls is not None
        assert tool_calls[0]["function"]["name"] == "validate_model"
        assert state.messages[2].tool_result == "VALID"
        assert state.usage.input_tokens == 28
        assert state.usage.output_tokens == 7
        assert state.usage.reasoning_tokens == 4
        assert state.total_time_seconds == 1.2

    def test_incremental_apply_and_log_format(self):
        from nof1_causal_lab.utils.harness.stream_json import PiStreamState

        state = PiStreamState()
        for event in _pi_events_tool_loop():
            apply_pi_event(state, event)

        assert state.final_text == "Done."
        formatted = format_pi_event_for_log(_pi_events_tool_loop()[4])
        assert formatted == "pi message: Done. [in=8 out=2 reasoning=1]"

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="pi stream line 0 is not valid JSON"):
            parse_pi_stream(["not-json"])


def _claude_events_simple() -> list[dict[str, Any]]:
    return [
        {
            "type": "system",
            "subtype": "init",
            "session_id": "sess-1",
            "model": "claude-sonnet-4-6",
            "tools": [],
            "mcp_servers": [],
        },
        {
            "type": "user",
            "message": {"role": "user", "content": "Hello there"},
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hi back!"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 12, "output_tokens": 5},
            },
        },
        {
            "type": "result",
            "subtype": "success",
            "result": "Hi back!",
            "duration_ms": 1250,
            "session_id": "sess-1",
            "usage": {"input_tokens": 12, "output_tokens": 5},
        },
    ]


def _claude_events_tool_loop() -> list[dict[str, Any]]:
    return [
        {"type": "system", "subtype": "init", "session_id": "s2", "model": "claude-opus-4-7"},
        {"type": "user", "message": {"role": "user", "content": "Validate this"}},
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Submitting..."},
                    {
                        "type": "tool_use",
                        "id": "toolu_abc",
                        "name": "validate_model",
                        "input": {"payload": '{"x":1}'},
                    },
                ],
                "usage": {"input_tokens": 10, "output_tokens": 4},
            },
        },
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_abc",
                        "content": "VALID",
                        "is_error": False,
                    }
                ],
            },
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Done."}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 3, "output_tokens": 2},
            },
        },
        {
            "type": "result",
            "subtype": "success",
            "result": "Done.",
            "duration_ms": 2500,
        },
    ]


class TestClaudeParser:
    def test_captures_session_id_and_model(self):
        state = parse_claude_stream([json.dumps(e) for e in _claude_events_simple()])
        assert state.session_id == "sess-1"
        assert state.model == "claude-sonnet-4-6"

    def test_user_and_assistant_messages(self):
        state = parse_claude_stream([json.dumps(e) for e in _claude_events_simple()])
        roles = [m.role for m in state.messages]
        assert roles == ["user", "assistant"]
        assert state.messages[0].content == "Hello there"
        assert state.messages[1].content == "Hi back!"

    def test_final_text_and_duration_from_result(self):
        state = parse_claude_stream([json.dumps(e) for e in _claude_events_simple()])
        assert state.final_text == "Hi back!"
        assert state.total_time_seconds == 1.25

    def test_tool_use_and_tool_result_blocks(self):
        state = parse_claude_stream([json.dumps(e) for e in _claude_events_tool_loop()])
        roles = [m.role for m in state.messages]
        # user -> assistant(with tool_call) -> tool -> assistant(final)
        assert roles == ["user", "assistant", "tool", "assistant"]

        assistant_with_tool = state.messages[1]
        assert assistant_with_tool.content == "Submitting..."
        assert assistant_with_tool.tool_calls is not None
        call = assistant_with_tool.tool_calls[0]
        assert call["id"] == "toolu_abc"
        assert call["function"]["name"] == "validate_model"
        assert json.loads(call["function"]["arguments"]) == {"payload": '{"x":1}'}

        tool_msg = state.messages[2]
        assert tool_msg.tool_call_id == "toolu_abc"
        assert tool_msg.content == "VALID"
        assert tool_msg.tool_is_error is False

    def test_usage_accumulates_across_assistant_turns(self):
        state = parse_claude_stream([json.dumps(e) for e in _claude_events_tool_loop()])
        # result event doesn't supply usage in the fixture, so accumulated
        # assistant usage stands.
        assert state.usage.input_tokens == 13
        assert state.usage.output_tokens == 6

    def test_finalize_produces_llm_trace(self):
        state = parse_claude_stream([json.dumps(e) for e in _claude_events_simple()])
        trace = finalize_trace(state)
        assert trace.model == "claude-sonnet-4-6"
        assert trace.total_time_seconds == 1.25
        assert [m.role for m in trace.messages] == ["user", "assistant"]
        assert trace.usage.input_tokens == 12
        assert trace.usage.output_tokens == 5

    def test_invalid_json_lines_raise(self):
        lines = [
            "not json",
            json.dumps({"type": "system", "subtype": "init", "session_id": "s3", "model": "x"}),
        ]
        with pytest.raises(ValueError, match="not valid JSON"):
            parse_claude_stream(lines)

    def test_non_object_json_raises(self):
        with pytest.raises(ValueError, match="not an object"):
            parse_claude_stream(["[1, 2, 3]"])

    def test_unknown_event_type_is_recorded_not_erroring(self):
        state = parse_claude_stream([json.dumps({"type": "nonsense_event", "payload": 1})])
        assert state.raw_events == [{"type": "nonsense_event", "payload": 1}]
        assert state.messages == []

    def test_apply_event_incrementally(self):
        from nof1_causal_lab.utils.harness.stream_json import ClaudeStreamState

        state = ClaudeStreamState()
        for event in _claude_events_simple():
            apply_claude_event(state, event)
        assert state.session_id == "sess-1"
        assert state.final_text == "Hi back!"

    def test_log_formatter_keeps_full_message_text(self):
        long_text = "A" * 300 + "\n" + "B" * 300
        event = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": long_text}],
            },
        }

        formatted = format_claude_event_for_log(event)

        assert formatted == f"claude message: {long_text}"
        assert "…" not in formatted

    def test_log_formatter_keeps_full_tool_result_text(self):
        long_text = "result-" * 80
        event = {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_abc",
                        "content": long_text,
                        "is_error": False,
                    }
                ],
            },
        }

        formatted = format_claude_event_for_log(event)

        assert formatted == f"claude tool result -> {long_text}"
        assert "…" not in formatted


def _codex_events_simple() -> list[dict[str, Any]]:
    return [
        {"type": "thread.started", "thread_id": "0199a213-81c0-7800-8aa1-bbab2a035a53"},
        {"type": "agent_message", "message": {"role": "user", "content": "Fix it"}},
        {"type": "agent_message", "message": {"role": "assistant", "content": "Done."}},
        {
            "type": "thread.completed",
            "usage": {"input_tokens": 7, "output_tokens": 3},
            "duration_ms": 800,
            "stop_reason": "complete",
        },
    ]


def _codex_events_tool_loop() -> list[dict[str, Any]]:
    return [
        {"type": "thread.started", "thread_id": "tid-2"},
        {
            "type": "tool_call",
            "call_id": "call_1",
            "name": "submit_model",
            "arguments": {"model": "x"},
        },
        {
            "type": "tool_result",
            "call_id": "call_1",
            "output": "ok",
            "is_error": False,
        },
        {"type": "agent_message", "message": {"role": "assistant", "content": "Submitted"}},
        {"type": "thread.completed", "duration_ms": 1500},
    ]


class TestCodexParser:
    def test_captures_thread_id(self):
        state = parse_codex_stream([json.dumps(e) for e in _codex_events_simple()])
        assert state.thread_id == "0199a213-81c0-7800-8aa1-bbab2a035a53"

    def test_assistant_and_user_messages(self):
        state = parse_codex_stream([json.dumps(e) for e in _codex_events_simple()])
        roles = [m.role for m in state.messages]
        assert roles == ["user", "assistant"]
        assert state.messages[1].content == "Done."
        assert state.final_text == "Done."

    def test_usage_and_duration_from_completed(self):
        state = parse_codex_stream([json.dumps(e) for e in _codex_events_simple()])
        assert state.usage.input_tokens == 7
        assert state.usage.output_tokens == 3
        assert state.total_time_seconds == 0.8
        assert state.stop_reason == "complete"

    def test_tool_call_and_result(self):
        state = parse_codex_stream([json.dumps(e) for e in _codex_events_tool_loop()])
        roles = [m.role for m in state.messages]
        assert roles == ["assistant", "tool", "assistant"]

        call_msg = state.messages[0]
        assert call_msg.tool_calls is not None
        assert call_msg.tool_calls[0]["id"] == "call_1"
        assert call_msg.tool_calls[0]["function"]["name"] == "submit_model"

        tool_msg = state.messages[1]
        assert tool_msg.tool_call_id == "call_1"
        assert tool_msg.content == "ok"

    def test_finalize_produces_llm_trace(self):
        state = parse_codex_stream([json.dumps(e) for e in _codex_events_simple()])
        trace = finalize_codex_trace(state)
        assert trace.total_time_seconds == 0.8
        assert trace.usage.input_tokens == 7

    def test_apply_event_incrementally(self):
        from nof1_causal_lab.utils.harness.stream_json import CodexStreamState

        state = CodexStreamState()
        for event in _codex_events_simple():
            apply_codex_event(state, event)
        assert state.thread_id == "0199a213-81c0-7800-8aa1-bbab2a035a53"

    def test_log_formatter_keeps_full_message_text(self):
        long_text = "A" * 300 + "\n" + "B" * 300
        event = {
            "type": "agent_message",
            "message": {"role": "assistant", "content": long_text},
        }

        formatted = format_codex_event_for_log(event)

        assert formatted == f"codex message: {long_text}"
        assert "…" not in formatted

    def test_log_formatter_keeps_full_tool_result_text(self):
        long_text = "payload-" * 80
        event = {
            "type": "tool_result",
            "name": "submit_model",
            "output": long_text,
            "is_error": False,
        }

        formatted = format_codex_event_for_log(event)

        assert formatted == f"codex tool result: submit_model -> {long_text}"
        assert "…" not in formatted
