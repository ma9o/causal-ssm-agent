"""Shared fixtures for pipeline integration tests."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import pytest

from nof1_causal_lab.machine.store import ArtifactStore
from nof1_causal_lab.utils.llm import LLMTrace

ScriptedTurnHandler = Callable[[list[Any], str], Awaitable[str | None]]


@pytest.fixture
def integration_workspace(monkeypatch, tmp_path) -> str:
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "DATA_URI", str(tmp_path / "data"))
    return "fixture_workspace"


@pytest.fixture
def artifact_store(integration_workspace: str) -> ArtifactStore:
    return ArtifactStore(integration_workspace)


@pytest.fixture
def install_scripted_stage_factory(monkeypatch):
    def _install(handler: ScriptedTurnHandler) -> None:
        from nof1_causal_lab.flows import llm_stage_runtime

        class _ScriptedSession:
            def __init__(self, tools: list[Any] | None) -> None:
                self._tools = tools or []
                self._completion = ""

            async def turn(self, user_message: str):
                from nof1_causal_lab.utils.agent_session import TurnResult

                completion = await handler(self._tools, user_message)
                self._completion = completion or ""
                return TurnResult(completion=self._completion)

            @property
            def result(self):
                from nof1_causal_lab.utils.agent_session import AgentResult

                return AgentResult(completion=self._completion, trace=LLMTrace())

        class _ScriptedFactory:
            def __init__(self) -> None:
                self.accumulated_trace = LLMTrace()

            @asynccontextmanager
            async def open(
                self,
                *,
                system_prompt: str | None = None,
                tools: list[Any] | None = None,
                log_label: str | None = None,
            ):
                del system_prompt, log_label
                yield _ScriptedSession(tools)

        monkeypatch.setattr(
            llm_stage_runtime,
            "build_stage_session_factory",
            lambda _config: _ScriptedFactory(),
        )

    return _install
