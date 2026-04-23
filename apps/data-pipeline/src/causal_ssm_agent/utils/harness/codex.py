"""AgentSession backed by the ``codex exec`` subprocess.

Codex's non-interactive mode is structured around threads rather than
a caller-provided session ID: the first ``codex exec`` invocation
emits a ``thread.started`` event with a ``thread_id`` we parse from
stream-json, and subsequent turns use ``codex exec resume <thread_id>``
to continue the conversation.

Tool exposure uses the same in-process streamable-HTTP MCP server as
the Claude backend. Codex reads MCP server definitions from a TOML
config file in ``$CODEX_HOME`` (default ``~/.codex``); we write a
per-session ``config.toml`` into a scratch directory and point Codex
at it by exporting ``CODEX_HOME`` for the child process.

Pipeline-fixed flags: ``--sandbox read-only`` (our only side-effect
channel is the MCP tools we expose), ``--ask-for-approval never`` for
non-interactive use, ``--json`` for event streaming, and
``--skip-git-repo-check`` so the pipeline can run outside a repo.

Integration status: the ``codex`` CLI's MCP config key names for HTTP
transport are not fully documented; the TOML we write matches
``codex mcp add --url`` documented behavior. If Codex refuses to load
the MCP server, verify the generated config against
``codex mcp list`` or fall back to pre-populating ``~/.codex/config.toml``
by hand and skipping the CODEX_HOME override.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from causal_ssm_agent.utils.agent_session import AgentResult, TurnResult
from causal_ssm_agent.utils.harness.mcp_server import serve_tools_http
from causal_ssm_agent.utils.harness.stream_json import (
    CodexStreamState,
    apply_codex_event,
    finalize_codex_trace,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from causal_ssm_agent.utils.openrouter_client import Tool

logger = logging.getLogger(__name__)

MCP_SERVER_NAME = "pipeline-tools"
_DEFAULT_TIMEOUT_SECONDS = 900


def build_codex_mcp_toml(url: str, server_name: str = MCP_SERVER_NAME) -> str:
    """Render a ``config.toml`` snippet pointing Codex at our HTTP MCP server."""
    return f'[mcp_servers.{server_name}]\nurl = "{url}"\n'


def build_codex_argv(
    *,
    bin: str,
    user_message: str,
    thread_id: str | None,
    model: str,
    reasoning_effort: str | None,
    cwd: str | Path | None = None,
    extra_config: list[tuple[str, str]] | None = None,
) -> list[str]:
    """Build the argv for one ``codex exec`` (or ``codex exec resume``) call.

    When ``thread_id`` is provided, use the ``resume`` subcommand; on a
    fresh session, the ``thread_id`` comes back in the first event.
    """
    argv: list[str] = [str(bin), "exec"]
    if thread_id is not None:
        argv.extend(["resume", thread_id])
    argv.extend(
        [
            "--json",
            "--sandbox",
            "read-only",
            "--ask-for-approval",
            "never",
            "--skip-git-repo-check",
            "-m",
            model,
        ]
    )
    if reasoning_effort is not None:
        argv.extend(["-c", f"model_reasoning_effort={reasoning_effort}"])
    if cwd is not None:
        argv.extend(["--cd", str(cwd)])
    for key, value in extra_config or []:
        argv.extend(["-c", f"{key}={value}"])
    argv.append(user_message)
    return argv


class CodexHarnessSession:
    """:class:`AgentSession` implemented over ``codex exec`` subprocesses."""

    def __init__(
        self,
        *,
        tools: list[Tool],
        codex_home: Path,
        model: str,
        bin: str = "codex",
        reasoning_effort: str | None = None,
        cwd: str | Path | None = None,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        log_label: str | None = None,
    ) -> None:
        self._tools = list(tools)
        self._tool_stop_map = {
            t.name: (t.success_output if t.stop_on_success else None)
            for t in tools
            if t.stop_on_success
        }
        self._codex_home = codex_home
        self._model = model
        self._bin = bin
        self._reasoning_effort = reasoning_effort
        self._cwd = cwd
        self._timeout_seconds = timeout_seconds
        self._log_label = log_label

        self._state = CodexStreamState()
        self._turn_index = 0
        self._terminal_tool: tuple[str, str] | None = None

    @property
    def thread_id(self) -> str | None:
        return self._state.thread_id

    async def turn(self, user_message: str) -> TurnResult:
        self._turn_index += 1
        pre_event_count = len(self._state.raw_events)

        argv = build_codex_argv(
            bin=self._bin,
            user_message=user_message,
            thread_id=self._state.thread_id if self._turn_index > 1 else None,
            model=self._model,
            reasoning_effort=self._reasoning_effort,
            cwd=self._cwd,
        )

        env = dict(os.environ)
        env["CODEX_HOME"] = str(self._codex_home)

        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        try:
            await asyncio.wait_for(self._drain_stdout(proc), timeout=self._timeout_seconds)
            await asyncio.wait_for(proc.wait(), timeout=self._timeout_seconds)
        except TimeoutError:
            proc.kill()
            with contextlib.suppress(ProcessLookupError):
                await proc.wait()
            raise

        if proc.returncode != 0:
            stderr_text = ""
            if proc.stderr is not None:
                stderr_bytes = await proc.stderr.read()
                stderr_text = stderr_bytes.decode("utf-8", errors="replace")
            raise RuntimeError(f"codex exited with status {proc.returncode}: {stderr_text[:500]}")

        turn_events = self._state.raw_events[pre_event_count:]
        return self._build_turn_result(turn_events)

    async def _drain_stdout(self, proc: asyncio.subprocess.Process) -> None:
        if proc.stdout is None:
            return
        async for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("[%s] non-JSON codex line: %r", self._log_label, line[:200])
                continue
            if isinstance(event, dict):
                apply_codex_event(self._state, event)

    def _build_turn_result(self, turn_events: list[dict]) -> TurnResult:
        tool_calls_fired: list[str] = []
        terminal: tuple[str, str] | None = None
        for event in turn_events:
            etype = event.get("type")
            if etype == "tool_call":
                name = str(event.get("name") or "")
                if name:
                    tool_calls_fired.append(name)
            elif etype == "tool_result":
                result_raw = event.get("output") or event.get("result") or ""
                result_text = result_raw if isinstance(result_raw, str) else json.dumps(result_raw)
                is_error = bool(event.get("is_error", False))
                matched = self._match_terminal(result_text, is_error=is_error)
                if matched is not None:
                    terminal = matched
        if terminal is not None:
            self._terminal_tool = terminal

        return TurnResult(
            completion=self._state.final_text,
            terminal_tool_name=terminal[0] if terminal else None,
            terminal_tool_output=terminal[1] if terminal else None,
            tool_calls_fired=tool_calls_fired,
        )

    def _match_terminal(self, result_text: str, *, is_error: bool) -> tuple[str, str] | None:
        for tool_name, success_output in self._tool_stop_map.items():
            if success_output is None:
                if not is_error:
                    return tool_name, result_text
            elif result_text.strip() == success_output:
                return tool_name, result_text
        return None

    @property
    def result(self) -> AgentResult:
        trace = finalize_codex_trace(self._state)
        return AgentResult(
            completion=self._state.final_text,
            trace=trace,
            terminal_tool_name=self._terminal_tool[0] if self._terminal_tool else None,
            terminal_tool_output=self._terminal_tool[1] if self._terminal_tool else None,
        )

    async def aclose(self) -> None:
        return None


@asynccontextmanager
async def open_codex_harness_session(
    *,
    tools: list[Tool],
    model: str,
    bin: str = "codex",
    reasoning_effort: str | None = None,
    cwd: str | Path | None = None,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    log_label: str | None = None,
) -> AsyncIterator[CodexHarnessSession]:
    """Open a Codex-backed agent session scoped to an ``async with`` block.

    Starts an in-process MCP server for ``tools``, writes a
    per-invocation ``config.toml`` in a scratch ``CODEX_HOME`` that
    points Codex at it, then yields a :class:`CodexHarnessSession`.
    The scratch directory and MCP server are cleaned up on exit.
    """
    async with serve_tools_http(tools, name=MCP_SERVER_NAME) as mcp_url:
        mcp_toml = build_codex_mcp_toml(mcp_url)
        with tempfile.TemporaryDirectory(prefix="codex-home-") as tmpdir:
            codex_home = Path(tmpdir)
            (codex_home / "config.toml").write_text(mcp_toml)
            session = CodexHarnessSession(
                tools=tools,
                codex_home=codex_home,
                model=model,
                bin=bin,
                reasoning_effort=reasoning_effort,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                log_label=log_label,
            )
            try:
                yield session
            finally:
                await session.aclose()
