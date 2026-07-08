"""Tests for :func:`open_session` — the backend-dispatching factory.

Verifies:
- ``harness: none`` opens an :class:`EmbeddedSession` with config merged from
  per-context overrides + :class:`EmbeddedLLMDefaults`.
- ``harness: claude-code`` and ``harness: codex`` call into the harness
  openers with the correct merged kwargs.
- ``max_tool_turns`` passed at call-time overrides per-context and global
  defaults in the appropriate spot for each backend.
- Unknown harness raises.
"""

import asyncio

import pytest

from nof1_causal_lab.utils.agent_session_embedded import EmbeddedSession
from nof1_causal_lab.utils.agent_session_factory import open_session
from nof1_causal_lab.utils.config import (
    ClaudeCodeDefaults,
    CodexDefaults,
    EmbeddedLLMDefaults,
    LLMDefaults,
    LLMProfileConfig,
)
from tests.helpers import run_async as _run


@pytest.fixture(autouse=True)
def _embedded_harness_prereqs(monkeypatch):
    from nof1_causal_lab.utils.config import _reset_verified_harnesses_for_testing

    _reset_verified_harnesses_for_testing()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    yield
    _reset_verified_harnesses_for_testing()


def _defaults() -> LLMDefaults:
    return LLMDefaults(
        embedded=EmbeddedLLMDefaults(max_tokens=12345, timeout=60, reasoning_effort="low"),
        claude_code=ClaudeCodeDefaults(
            bin="claude",
            effort="medium",
            max_turns=22,
            max_budget_usd=None,
            fallback_model=None,
        ),
        codex=CodexDefaults(bin="codex", reasoning_effort="low"),
    )


class TestEmbeddedDispatch:
    def test_opens_embedded_session_with_merged_config(self):
        stage = LLMProfileConfig(harness="none", model="openrouter/test-model")

        async def scenario():
            async with open_session(
                stage,
                _defaults(),
                system_prompt="sys",
                tools=[],
            ) as session:
                return session

        session = _run(scenario())
        assert isinstance(session, EmbeddedSession)
        # pylint: disable=protected-access  — checking merge behavior
        cfg = session._config  # type: ignore[attr-defined]
        assert cfg.max_tokens == 12345
        assert cfg.timeout == 60
        assert cfg.reasoning_effort == "low"
        assert cfg.max_tool_output is None

    def test_per_stage_override_takes_precedence(self):
        stage = LLMProfileConfig(
            harness="none",
            model="openrouter/test-model",
            max_tokens=999,
            timeout=30,
            reasoning_effort="high",
        )

        async def scenario():
            async with open_session(
                stage,
                _defaults(),
                system_prompt=None,
                tools=[],
            ) as session:
                return session

        session = _run(scenario())
        cfg = session._config  # type: ignore[attr-defined]
        assert cfg.max_tokens == 999
        assert cfg.timeout == 30
        assert cfg.reasoning_effort == "high"


class TestHarnessDispatch:
    def test_claude_code_dispatches_with_merged_kwargs(self, monkeypatch):
        captured: dict = {}

        # Replace the opener with a stub that records its kwargs. The
        # real opener is an async context manager; wrap the stub.
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def stub_open_claude_harness_session(**kwargs):
            captured.update(kwargs)
            yield "claude-stub"

        monkeypatch.setattr(
            "nof1_causal_lab.utils.agent_session_factory.open_claude_harness_session",
            stub_open_claude_harness_session,
        )

        stage = LLMProfileConfig(
            harness="claude-code",
            model="claude-sonnet-4-6",
            effort="xhigh",  # per-context override
            max_budget_usd=5.0,  # per-context override
        )

        async def scenario():
            async with open_session(
                stage,
                _defaults(),
                system_prompt="You are a test.",
                tools=[],
                max_tool_turns=99,
            ) as session:
                return session

        session = _run(scenario())
        assert session == "claude-stub"
        # per-context override wins
        assert captured["effort"] == "xhigh"
        # max_tool_turns from call wins over per-context max_turns (None) and global
        assert captured["max_turns"] == 99
        assert captured["max_budget_usd"] == 5.0
        # bin inherits global default (no per-context override)
        assert captured["bin"] == "claude"
        # model is passed through
        assert captured["model"] == "claude-sonnet-4-6"

    def test_codex_dispatches_with_merged_kwargs(self, monkeypatch):
        captured: dict = {}
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def stub_open_codex_harness_session(**kwargs):
            captured.update(kwargs)
            yield "codex-stub"

        monkeypatch.setattr(
            "nof1_causal_lab.utils.agent_session_factory.open_codex_harness_session",
            stub_open_codex_harness_session,
        )

        stage = LLMProfileConfig(
            harness="codex",
            model="gpt-5.4",
            reasoning_effort="high",  # per-context override
        )

        async def scenario():
            async with open_session(
                stage,
                _defaults(),
                system_prompt=None,
                tools=[],
            ) as session:
                return session

        session = _run(scenario())
        assert session == "codex-stub"
        assert captured["model"] == "gpt-5.4"
        assert captured["reasoning_effort"] == "high"
        assert captured["bin"] == "codex"


class TestUnknownHarness:
    def test_raises(self):
        stage = LLMProfileConfig(harness="anthropic", model="x")

        async def scenario():
            async with open_session(stage, _defaults(), system_prompt=None, tools=[]) as _session:
                pass

        with pytest.raises(ValueError, match="Unknown harness"):
            _run(scenario())


# Silence unused-imports complaints from ruff if any of the above are
# moved into parametrized forms later.
_ = asyncio
