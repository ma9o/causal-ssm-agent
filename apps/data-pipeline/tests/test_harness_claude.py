"""Tests for the Claude-backed AgentSession.

Unit tests stub out ``asyncio.create_subprocess_exec`` with a fake
process that emits canned stream-json, so the session can be exercised
without a real ``claude`` binary. An integration test that actually
spawns Claude is skipped unless the binary is on PATH AND the
``CAUSAL_SSM_RUN_CLAUDE_HARNESS`` env var is set.
"""

import asyncio
import json
import os
import shutil
from types import SimpleNamespace

import pytest

from causal_ssm_agent.utils.harness.claude import (
    ClaudeHarnessSession,
    build_claude_argv,
    build_mcp_config_json,
    open_claude_harness_session,
)
from causal_ssm_agent.utils.openrouter_client import Tool
from tests.helpers import _run


class _FakeStdout:
    def __init__(self, lines: list[bytes]):
        self._lines = list(lines)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._lines:
            raise StopAsyncIteration
        return self._lines.pop(0)


class _FakeStderr:
    def __init__(self, text: bytes = b""):
        self._text = text

    async def read(self) -> bytes:
        return self._text


class _FakeProcess:
    def __init__(
        self,
        *,
        lines: list[bytes],
        returncode: int = 0,
        stderr: bytes = b"",
    ):
        self.stdout = _FakeStdout(lines)
        self.stderr = _FakeStderr(stderr)
        self.returncode = returncode
        self._waited = False

    async def wait(self) -> int:
        self._waited = True
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9


def _jsonl(events: list[dict]) -> list[bytes]:
    return [(json.dumps(e) + "\n").encode() for e in events]


def _make_terminal_tool() -> Tool:
    async def _execute(payload: str) -> str:
        return "VALID"

    return Tool(
        name="validate_model",
        description="Validate the submitted model.",
        parameters={
            "type": "object",
            "properties": {"payload": {"type": "string"}},
            "required": ["payload"],
            "additionalProperties": False,
        },
        execute=_execute,
        stop_on_success=True,
        success_output="VALID",
    )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestBuildMcpConfigJson:
    def test_shape(self):
        raw = build_mcp_config_json("http://127.0.0.1:1234/mcp")
        parsed = json.loads(raw)
        assert "mcpServers" in parsed
        entry = parsed["mcpServers"]["pipeline-tools"]
        assert entry["type"] == "http"
        assert entry["url"] == "http://127.0.0.1:1234/mcp"

    def test_custom_server_name(self):
        raw = build_mcp_config_json("http://x", server_name="custom")
        parsed = json.loads(raw)
        assert "custom" in parsed["mcpServers"]


class TestBuildClaudeArgv:
    def test_first_turn_uses_session_id_not_resume(self):
        argv = build_claude_argv(
            bin="claude",
            user_message="hello",
            session_id="abc123",
            resume=False,
            mcp_config_path="/tmp/mcp.json",
            system_prompt="You are a test.",
            model="claude-sonnet-4-6",
            effort="high",
            max_turns=20,
            max_budget_usd=None,
            fallback_model=None,
        )
        assert "--session-id" in argv
        assert "--resume" not in argv
        assert argv[argv.index("--session-id") + 1] == "abc123"
        assert argv[argv.index("-p") + 1] == "hello"
        assert argv[argv.index("--model") + 1] == "claude-sonnet-4-6"
        assert argv[argv.index("--effort") + 1] == "high"
        assert argv[argv.index("--max-turns") + 1] == "20"
        assert "--bare" not in argv  # subscription auth needs OAuth/keychain
        assert "--disable-slash-commands" in argv
        assert "--strict-mcp-config" in argv
        assert argv[argv.index("--permission-mode") + 1] == "bypassPermissions"
        assert argv[argv.index("--output-format") + 1] == "stream-json"

    def test_follow_up_turn_uses_resume(self):
        argv = build_claude_argv(
            bin="claude",
            user_message="review",
            session_id="abc123",
            resume=True,
            mcp_config_path="/tmp/mcp.json",
            system_prompt="ignored on resume",
            model="claude-sonnet-4-6",
            effort=None,
            max_turns=None,
            max_budget_usd=None,
            fallback_model=None,
        )
        assert "--resume" in argv
        assert "--session-id" not in argv
        assert argv[argv.index("--resume") + 1] == "abc123"

    def test_optional_flags_omitted_when_none(self):
        argv = build_claude_argv(
            bin="claude",
            user_message="hi",
            session_id="s",
            resume=False,
            mcp_config_path="/tmp/mcp.json",
            system_prompt=None,
            model="sonnet",
            effort=None,
            max_turns=None,
            max_budget_usd=None,
            fallback_model=None,
        )
        assert "--effort" not in argv
        assert "--max-turns" not in argv
        assert "--max-budget-usd" not in argv
        assert "--fallback-model" not in argv
        assert "--append-system-prompt" not in argv

    def test_max_budget_and_fallback_model_pass_through(self):
        argv = build_claude_argv(
            bin="claude",
            user_message="hi",
            session_id="s",
            resume=False,
            mcp_config_path="/tmp/mcp.json",
            system_prompt=None,
            model="opus",
            effort=None,
            max_turns=None,
            max_budget_usd=2.50,
            fallback_model="sonnet",
        )
        assert argv[argv.index("--max-budget-usd") + 1] == "2.5"
        assert argv[argv.index("--fallback-model") + 1] == "sonnet"


# ---------------------------------------------------------------------------
# Session turn() with mocked subprocess
# ---------------------------------------------------------------------------


def _patch_subprocess(monkeypatch, process_factory):
    """Replace asyncio.create_subprocess_exec with ``process_factory(argv)``."""
    captured = {"invocations": []}

    async def fake_create_subprocess_exec(*args, **_kwargs):
        captured["invocations"].append(list(args))
        return process_factory(list(args))

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    return captured


class TestSessionTurn:
    def test_single_turn_accumulates_trace_and_returns_completion(self, monkeypatch, tmp_path):
        events = [
            {
                "type": "system",
                "subtype": "init",
                "session_id": "s-ignored",
                "model": "claude-sonnet-4-6",
            },
            {"type": "user", "message": {"role": "user", "content": "Prompt"}},
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "All good"}],
                    "usage": {"input_tokens": 9, "output_tokens": 3},
                },
            },
            {
                "type": "result",
                "subtype": "success",
                "result": "All good",
                "duration_ms": 1500,
            },
        ]

        captured = _patch_subprocess(monkeypatch, lambda _argv: _FakeProcess(lines=_jsonl(events)))

        mcp_config = tmp_path / "mcp.json"
        mcp_config.write_text(build_mcp_config_json("http://127.0.0.1:0/mcp"))

        session = ClaudeHarnessSession(
            tools=[_make_terminal_tool()],
            mcp_config_path=mcp_config,
            system_prompt="sys",
            model="claude-sonnet-4-6",
        )

        result = _run(session.turn("Prompt"))

        assert result.completion == "All good"
        assert result.tool_calls_fired == []
        assert result.terminal_tool_name is None
        assert len(captured["invocations"]) == 1
        first_argv = captured["invocations"][0]
        assert "--session-id" in first_argv
        assert session.session_id == first_argv[first_argv.index("--session-id") + 1]

        agent_result = session.result
        assert agent_result.completion == "All good"
        assert [m.role for m in agent_result.trace.messages] == ["user", "assistant"]
        assert agent_result.trace.usage.input_tokens == 9

    def test_follow_up_turn_uses_resume_flag(self, monkeypatch, tmp_path):
        turn1 = [
            {
                "type": "system",
                "subtype": "init",
                "session_id": "ignored",
                "model": "claude",
            },
            {"type": "user", "message": {"role": "user", "content": "Hi"}},
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hello"}],
                },
            },
            {
                "type": "result",
                "subtype": "success",
                "result": "Hello",
                "duration_ms": 100,
            },
        ]
        turn2 = [
            {"type": "user", "message": {"role": "user", "content": "Bye"}},
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Bye!"}],
                },
            },
            {
                "type": "result",
                "subtype": "success",
                "result": "Bye!",
                "duration_ms": 100,
            },
        ]
        script = [turn1, turn2]

        def factory(_argv):
            return _FakeProcess(lines=_jsonl(script.pop(0)))

        captured = _patch_subprocess(monkeypatch, factory)

        mcp_config = tmp_path / "mcp.json"
        mcp_config.write_text(build_mcp_config_json("http://127.0.0.1:0/mcp"))
        session = ClaudeHarnessSession(
            tools=[],
            mcp_config_path=mcp_config,
            system_prompt=None,
            model="sonnet",
        )

        _run(session.turn("Hi"))
        _run(session.turn("Bye"))

        assert len(captured["invocations"]) == 2
        first, second = captured["invocations"]
        assert "--session-id" in first
        assert "--resume" in second
        assert second[second.index("--resume") + 1] == session.session_id

    def test_terminal_tool_detected_from_tool_result(self, monkeypatch, tmp_path):
        events = [
            {
                "type": "system",
                "subtype": "init",
                "session_id": "s",
                "model": "claude",
            },
            {"type": "user", "message": {"role": "user", "content": "Submit"}},
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Submitting"},
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "mcp__pipeline-tools__validate_model",
                            "input": {"payload": "x"},
                        },
                    ],
                },
            },
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
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
                    "content": [{"type": "text", "text": "Accepted."}],
                },
            },
            {
                "type": "result",
                "subtype": "success",
                "result": "Accepted.",
                "duration_ms": 900,
            },
        ]

        _patch_subprocess(monkeypatch, lambda _argv: _FakeProcess(lines=_jsonl(events)))

        mcp_config = tmp_path / "mcp.json"
        mcp_config.write_text(build_mcp_config_json("http://127.0.0.1:0/mcp"))
        session = ClaudeHarnessSession(
            tools=[_make_terminal_tool()],
            mcp_config_path=mcp_config,
            system_prompt=None,
            model="sonnet",
        )

        result = _run(session.turn("Submit"))

        assert result.terminal_tool_name == "validate_model"
        assert result.terminal_tool_output == "VALID"
        assert result.tool_calls_fired == ["mcp__pipeline-tools__validate_model"]

        agent_result = session.result
        assert agent_result.terminal_tool_name == "validate_model"

    def test_non_zero_exit_raises(self, monkeypatch, tmp_path):
        _patch_subprocess(
            monkeypatch,
            lambda _argv: _FakeProcess(lines=[], returncode=2, stderr=b"auth failed"),
        )

        mcp_config = tmp_path / "mcp.json"
        mcp_config.write_text(build_mcp_config_json("http://127.0.0.1:0/mcp"))
        session = ClaudeHarnessSession(
            tools=[],
            mcp_config_path=mcp_config,
            system_prompt=None,
            model="sonnet",
        )

        with pytest.raises(RuntimeError, match="claude exited with status 2"):
            _run(session.turn("Hi"))


# ---------------------------------------------------------------------------
# Integration (requires claude CLI)
# ---------------------------------------------------------------------------


_CLAUDE_AVAILABLE = shutil.which("claude") is not None and bool(
    os.getenv("CAUSAL_SSM_RUN_CLAUDE_HARNESS")
)


@pytest.mark.skipif(
    not _CLAUDE_AVAILABLE,
    reason="claude binary or CAUSAL_SSM_RUN_CLAUDE_HARNESS not set",
)
class TestClaudeIntegration:
    def test_round_trip(self):
        async def scenario():
            async with open_claude_harness_session(
                tools=[],
                system_prompt=None,
                model="sonnet",
                effort="low",
                max_turns=2,
            ) as session:
                result = await session.turn("Respond with exactly the word 'ok' and nothing else.")
                return result, session.result

        turn_result, agent_result = _run(scenario())
        assert "ok" in turn_result.completion.lower()
        assert agent_result.trace.messages


# Keep a reference to SimpleNamespace to stop ruff from flagging unused import
_ = SimpleNamespace
