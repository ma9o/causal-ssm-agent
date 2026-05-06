"""Tests for the Codex-backed AgentSession.

Unit tests stub out ``asyncio.create_subprocess_exec`` with a fake
process that emits canned Codex ``--json`` events. Integration against
a real ``codex`` binary is gated on presence + ``CAUSAL_SSM_RUN_CODEX_HARNESS``.
"""

import os
import shutil

import pytest

from causal_ssm_agent.utils.harness.codex import (
    CodexHarnessSession,
    build_codex_argv,
    build_codex_mcp_toml,
    open_codex_harness_session,
)
from tests.helpers import run_async as _run
from tests.infra.harness_fakes import FakeProcess as _FakeProcess
from tests.infra.harness_fakes import jsonl as _jsonl
from tests.infra.harness_fakes import make_terminal_tool
from tests.infra.harness_fakes import patch_subprocess as _patch_subprocess


def _make_terminal_tool():
    return make_terminal_tool(
        name="submit_model",
        description="Submit a model for validation.",
    )


class TestBuildCodexMcpToml:
    def test_default_server_name(self):
        toml = build_codex_mcp_toml("http://127.0.0.1:9999/mcp")
        assert "[mcp_servers.pipeline-tools]" in toml
        assert 'url = "http://127.0.0.1:9999/mcp"' in toml

    def test_custom_server_name(self):
        toml = build_codex_mcp_toml("http://x", server_name="custom")
        assert "[mcp_servers.custom]" in toml


class TestBuildCodexArgv:
    def test_first_turn_omits_resume_subcommand(self):
        argv = build_codex_argv(
            bin="codex",
            user_message="hello",
            thread_id=None,
            model="gpt-5.4",
            reasoning_effort="high",
        )
        # exec is first subcommand, "resume" must not appear between exec and flags.
        assert argv[0] == "codex"
        assert argv[1] == "exec"
        assert "resume" not in argv
        assert "--json" in argv
        assert argv[argv.index("-m") + 1] == "gpt-5.4"
        assert "--dangerously-bypass-approvals-and-sandbox" in argv
        assert "--sandbox" not in argv
        # user message is last
        assert argv[-1] == "hello"
        # reasoning_effort passes through -c
        assert any(arg == "model_reasoning_effort=high" for arg in argv)

    def test_follow_up_turn_uses_resume_subcommand_with_thread_id(self):
        argv = build_codex_argv(
            bin="codex",
            user_message="continue",
            thread_id="tid-123",
            model="gpt-5.4",
            reasoning_effort=None,
        )
        assert argv[0:4] == ["codex", "exec", "resume", "tid-123"]
        assert argv[-1] == "continue"

    def test_optional_flags_omitted(self):
        argv = build_codex_argv(
            bin="codex",
            user_message="hi",
            thread_id=None,
            model="m",
            reasoning_effort=None,
        )
        # No stray reasoning_effort key=value entries.
        assert not any("model_reasoning_effort" in arg for arg in argv)
        assert "-C" not in argv

    def test_cwd_and_extra_config_pass_through(self):
        argv = build_codex_argv(
            bin="codex",
            user_message="hi",
            thread_id=None,
            model="m",
            reasoning_effort=None,
            cwd="/tmp/work",
            extra_config=[("key1", "val1"), ("key2", "val2")],
        )
        assert argv[argv.index("-C") + 1] == "/tmp/work"
        assert "key1=val1" in argv
        assert "key2=val2" in argv


class TestSessionTurn:
    def test_thread_id_parsed_from_first_turn(self, monkeypatch, tmp_path):
        events = [
            {"type": "thread.started", "thread_id": "t-abc"},
            {
                "type": "agent_message",
                "message": {"role": "assistant", "content": "Done."},
            },
            {"type": "thread.completed", "duration_ms": 400},
        ]
        _patch_subprocess(monkeypatch, lambda _argv: _FakeProcess(lines=_jsonl(events)))

        session = CodexHarnessSession(
            tools=[],
            codex_home=tmp_path,
            model="gpt-5.4",
        )
        result = _run(session.turn("hi"))

        assert session.thread_id == "t-abc"
        assert result.completion == "Done."

    def test_follow_up_uses_resume_subcommand(self, monkeypatch, tmp_path):
        turn1 = [
            {"type": "thread.started", "thread_id": "tid-X"},
            {"type": "agent_message", "message": {"role": "assistant", "content": "One"}},
            {"type": "thread.completed"},
        ]
        turn2 = [
            {"type": "agent_message", "message": {"role": "assistant", "content": "Two"}},
            {"type": "thread.completed"},
        ]
        script = [turn1, turn2]

        captured = _patch_subprocess(
            monkeypatch, lambda _argv: _FakeProcess(lines=_jsonl(script.pop(0)))
        )

        session = CodexHarnessSession(tools=[], codex_home=tmp_path, model="m")
        _run(session.turn("a"))
        _run(session.turn("b"))

        first, second = captured["invocations"]
        first_inner = first[first.index("codex") :]
        second_inner = second[second.index("codex") :]
        assert "resume" not in first_inner
        assert second_inner[2] == "resume"
        assert second_inner[3] == "tid-X"

    def test_codex_home_env_exported(self, monkeypatch, tmp_path):
        events = [
            {"type": "thread.started", "thread_id": "t"},
            {"type": "thread.completed"},
        ]
        captured = _patch_subprocess(monkeypatch, lambda _argv: _FakeProcess(lines=_jsonl(events)))

        session = CodexHarnessSession(tools=[], codex_home=tmp_path, model="m")
        _run(session.turn("hi"))

        assert captured["envs"][0]["CODEX_HOME"] == str(tmp_path)

    def test_terminal_tool_detected_from_tool_result(self, monkeypatch, tmp_path):
        events = [
            {"type": "thread.started", "thread_id": "t"},
            {
                "type": "tool_call",
                "call_id": "c1",
                "name": "submit_model",
                "arguments": {},
            },
            {"type": "tool_result", "call_id": "c1", "output": "VALID"},
            {
                "type": "agent_message",
                "message": {"role": "assistant", "content": "Accepted"},
            },
            {"type": "thread.completed"},
        ]
        _patch_subprocess(monkeypatch, lambda _argv: _FakeProcess(lines=_jsonl(events)))

        session = CodexHarnessSession(
            tools=[_make_terminal_tool()],
            codex_home=tmp_path,
            model="m",
        )
        result = _run(session.turn("submit"))

        assert result.terminal_tool_name == "submit_model"
        assert result.terminal_tool_output == "VALID"
        assert result.tool_calls_fired == ["submit_model"]

    def test_non_zero_exit_raises(self, monkeypatch, tmp_path):
        _patch_subprocess(
            monkeypatch,
            lambda _argv: _FakeProcess(lines=[], returncode=1, stderr=b"nope"),
        )
        session = CodexHarnessSession(tools=[], codex_home=tmp_path, model="m")
        with pytest.raises(RuntimeError, match="codex exited with status 1"):
            _run(session.turn("hi"))


_CODEX_AVAILABLE = shutil.which("codex") is not None and bool(
    os.getenv("CAUSAL_SSM_RUN_CODEX_HARNESS")
)


@pytest.mark.skipif(
    not _CODEX_AVAILABLE,
    reason="codex binary or CAUSAL_SSM_RUN_CODEX_HARNESS not set",
)
class TestCodexIntegration:
    def test_round_trip(self):
        async def scenario():
            async with open_codex_harness_session(
                tools=[],
                model="gpt-5.4",
                reasoning_effort="low",
            ) as session:
                result = await session.turn("Respond with exactly the word 'ok' and nothing else.")
                return result, session.result

        turn_result, agent_result = _run(scenario())
        assert "ok" in turn_result.completion.lower()
        assert agent_result.trace.messages
