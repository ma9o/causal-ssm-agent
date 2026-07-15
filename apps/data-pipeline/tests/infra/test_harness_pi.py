"""Tests for the Pi-backed AgentSession."""

import json

import pytest

from nof1_causal_lab.utils.harness.pi import (
    PiHarnessSession,
    build_pi_argv,
    build_pi_extension,
)
from tests.helpers import run_async as _run
from tests.infra.harness_fakes import FakeProcess, jsonl, make_terminal_tool, patch_subprocess


def _terminal_tool():
    return make_terminal_tool(name="submit_model", description="Submit the model.")


def test_extension_registers_only_supplied_tools():
    extension = build_pi_extension([_terminal_tool()], "http://127.0.0.1:1234/token")

    assert '"name": "submit_model"' in extension
    assert "pi.registerTool" in extension
    assert "http://127.0.0.1:1234/token" in extension
    assert "registerCommand" not in extension


def test_argv_disables_builtin_and_discovered_capabilities(tmp_path):
    argv = build_pi_argv(
        bin="pi",
        user_message="hello",
        provider="openai-codex",
        model="gpt-5.4-mini",
        thinking="high",
        system_prompt="system",
        extension_path=tmp_path / "tools.ts",
        session_id="00000000-0000-4000-8000-000000000001",
        session_dir=tmp_path / "sessions",
        tool_names=["submit_model"],
    )

    assert argv[argv.index("--provider") + 1] == "openai-codex"
    assert argv[argv.index("--model") + 1] == "gpt-5.4-mini"
    assert argv[argv.index("--thinking") + 1] == "high"
    assert argv[argv.index("--tools") + 1] == "submit_model"
    assert "--no-builtin-tools" in argv
    assert "--no-extensions" in argv
    assert "--no-skills" in argv
    assert "--no-prompt-templates" in argv
    assert "--no-context-files" in argv
    assert argv[-1] == "hello"


def test_turn_parses_tool_and_terminal_result(monkeypatch, tmp_path):
    events = [
        {"type": "session", "id": "session-1"},
        {
            "type": "tool_execution_start",
            "toolCallId": "call-1",
            "toolName": "submit_model",
            "args": {},
        },
        {
            "type": "tool_execution_end",
            "toolCallId": "call-1",
            "toolName": "submit_model",
            "result": {"content": [{"type": "text", "text": "VALID"}], "details": {}},
            "isError": False,
        },
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Accepted"}],
                "model": "gpt-5.4-mini",
                "usage": {"input": 10, "output": 3, "reasoning": 1},
                "stopReason": "stop",
            },
        },
        {"type": "agent_end", "messages": []},
    ]
    captured = patch_subprocess(
        monkeypatch,
        lambda _argv: FakeProcess(lines=jsonl(events)),
    )
    (tmp_path / "pipeline-tools.ts").write_text("")
    session = PiHarnessSession(
        tools=[_terminal_tool()],
        scratch_dir=tmp_path,
        system_prompt="system",
        provider="openai-codex",
        model="gpt-5.4-mini",
        thinking="high",
    )

    result = _run(session.turn("submit"))

    assert result.completion == "Accepted"
    assert result.tool_calls_fired == ["submit_model"]
    assert result.terminal_tool_name == "submit_model"
    assert result.terminal_tool_output == "VALID"
    argv = captured["invocations"][0]
    assert argv[argv.index("--session-id") + 1] == session.session_id


def test_non_zero_exit_raises(monkeypatch, tmp_path):
    patch_subprocess(
        monkeypatch,
        lambda _argv: FakeProcess(lines=[], returncode=1, stderr=b"nope"),
    )
    (tmp_path / "pipeline-tools.ts").write_text("")
    session = PiHarnessSession(
        tools=[],
        scratch_dir=tmp_path,
        system_prompt=None,
        provider="openai-codex",
        model="gpt-5.4-mini",
        thinking="high",
    )

    with pytest.raises(RuntimeError, match="pi exited with status 1"):
        _run(session.turn("hello"))


def test_restored_session_jsonl_is_persisted(tmp_path):
    session_id = "00000000-0000-4000-8000-000000000002"
    content = (
        json.dumps(
            {"type": "session", "version": 3, "id": session_id, "timestamp": "now", "cwd": "/tmp"}
        )
        + "\n"
    )
    (tmp_path / "pipeline-tools.ts").write_text("")
    session = PiHarnessSession(
        tools=[],
        scratch_dir=tmp_path,
        system_prompt=None,
        provider="openai-codex",
        model="gpt-5.4-mini",
        thinking="high",
        initial_session_jsonl=content,
        session_id=session_id,
    )

    assert session.session_jsonl == content
