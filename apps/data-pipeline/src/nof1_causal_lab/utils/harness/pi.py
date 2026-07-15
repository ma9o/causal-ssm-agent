"""AgentSession backed by Pi's non-interactive JSON mode."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

from nof1_causal_lab.utils.agent_session import AgentResult, TurnResult
from nof1_causal_lab.utils.harness.pi_tool_bridge import serve_pi_tools_http
from nof1_causal_lab.utils.harness.stream_json import (
    PiStreamState,
    apply_pi_event,
    finalize_pi_trace,
    format_pi_event_for_log,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from nof1_causal_lab.utils.openrouter_client import Tool

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_DEFAULT_TIMEOUT_SECONDS = 1800


def build_pi_extension(tools: list[Tool], bridge_url: str) -> str:
    """Render a Pi extension registering exactly the supplied pipeline tools."""
    definitions = [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        for tool in tools
    ]
    return f"""import type {{ ExtensionAPI }} from \"@earendil-works/pi-coding-agent\";

const bridgeUrl = {json.dumps(bridge_url)};
const definitions = {json.dumps(definitions)};

export default function (pi: ExtensionAPI) {{
  for (const definition of definitions) {{
    pi.registerTool({{
      name: definition.name,
      label: definition.name,
      description: definition.description,
      parameters: definition.parameters as any,
      async execute(_toolCallId, parameters, signal) {{
        const response = await fetch(bridgeUrl, {{
          method: \"POST\",
          headers: {{ \"content-type\": \"application/json\" }},
          body: JSON.stringify({{ name: definition.name, arguments: parameters }}),
          signal,
        }});
        const payload = await response.json();
        if (!response.ok) {{
          throw new Error(payload.error ?? `tool bridge returned ${{response.status}}`);
        }}
        if (payload.is_error) {{
          throw new Error(payload.output);
        }}
        return {{ content: [{{ type: \"text\", text: payload.output }}], details: {{}} }};
      }},
    }});
  }}
}}
"""


def build_pi_argv(
    *,
    bin: str,
    user_message: str,
    provider: str,
    model: str,
    thinking: str,
    system_prompt: str | None,
    extension_path: Path,
    session_id: str,
    session_dir: Path,
    tool_names: list[str],
) -> list[str]:
    """Build one deterministic, non-interactive Pi invocation."""
    argv = [
        bin,
        "--mode",
        "json",
        "--print",
        "--provider",
        provider,
        "--model",
        model,
        "--thinking",
        thinking,
        "--system-prompt",
        system_prompt or "",
        "--no-builtin-tools",
        "--no-extensions",
        "--extension",
        str(extension_path),
        "--no-skills",
        "--no-prompt-templates",
        "--no-themes",
        "--no-context-files",
        "--no-approve",
        "--session-id",
        session_id,
        "--session-dir",
        str(session_dir),
    ]
    if tool_names:
        argv.extend(["--tools", ",".join(tool_names)])
    else:
        argv.append("--no-tools")
    argv.append(user_message)
    return argv


class PiHarnessSession:
    """:class:`AgentSession` implemented over ``pi --mode json`` subprocesses."""

    def __init__(
        self,
        *,
        tools: list[Tool],
        scratch_dir: Path,
        system_prompt: str | None,
        provider: str,
        model: str,
        thinking: str,
        bin: str = "pi",
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        log_label: str | None = None,
        initial_events: list[dict[str, Any]] | None = None,
        initial_session_jsonl: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self._tools = list(tools)
        self._tool_stop_map = {
            tool.name: tool.success_output if tool.stop_on_success else None
            for tool in tools
            if tool.stop_on_success
        }
        self._scratch_dir = scratch_dir
        self._system_prompt = system_prompt
        self._provider = provider
        self._model = model
        self._thinking = thinking
        self._bin = bin
        self._timeout_seconds = timeout_seconds
        self._log_label = log_label
        self._session_id = session_id or str(uuid.uuid4())
        self._session_dir = scratch_dir / "sessions"
        self._session_dir.mkdir()
        if initial_session_jsonl is not None:
            (self._session_dir / f"restored_{self._session_id}.jsonl").write_text(
                initial_session_jsonl
            )

        self._state = PiStreamState()
        for event in initial_events or []:
            apply_pi_event(self._state, event)
        self._terminal_tool: tuple[str, str] | None = None

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def session_jsonl(self) -> str:
        files = list(self._session_dir.glob("*.jsonl"))
        if len(files) != 1:
            raise RuntimeError(f"expected one Pi session file, found {len(files)}")
        return files[0].read_text()

    @property
    def raw_events(self) -> list[dict[str, Any]]:
        return list(self._state.raw_events)

    async def turn(self, user_message: str) -> TurnResult:
        pre_event_count = len(self._state.raw_events)
        started = perf_counter()
        argv = build_pi_argv(
            bin=self._bin,
            user_message=user_message,
            provider=self._provider,
            model=self._model,
            thinking=self._thinking,
            system_prompt=self._system_prompt,
            extension_path=self._scratch_dir / "pipeline-tools.ts",
            session_id=self._session_id,
            session_dir=self._session_dir,
            tool_names=[tool.name for tool in self._tools],
        )
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=dict(os.environ),
            cwd="/tmp",
        )
        stderr_bytes = bytearray()

        async def _drain_stderr() -> None:
            if proc.stderr is None:
                return
            while chunk := await proc.stderr.read(65536):
                stderr_bytes.extend(chunk)
                for line in chunk.split(b"\n"):
                    text = line.decode(errors="replace").strip()
                    if text:
                        logger.info("[%s] pi stderr: %s", self._log_label, text)

        try:
            await asyncio.wait_for(
                asyncio.gather(self._drain_stdout(proc), _drain_stderr()),
                timeout=self._timeout_seconds,
            )
            await asyncio.wait_for(proc.wait(), timeout=self._timeout_seconds)
        except TimeoutError:
            proc.kill()
            with contextlib.suppress(ProcessLookupError):
                await proc.wait()
            raise
        if proc.returncode != 0:
            stderr = stderr_bytes.decode(errors="replace")
            raise RuntimeError(f"pi exited with status {proc.returncode}: {stderr}")
        apply_pi_event(
            self._state,
            {"type": "nof1.turn_timing", "duration_seconds": perf_counter() - started},
        )
        return self._build_turn_result(self._state.raw_events[pre_event_count:])

    async def _drain_stdout(self, proc: asyncio.subprocess.Process) -> None:
        if proc.stdout is None:
            return
        buffer = bytearray()
        while chunk := await proc.stdout.read(65536):
            buffer.extend(chunk)
            while (newline := buffer.find(b"\n")) >= 0:
                self._handle_line(bytes(buffer[:newline]))
                del buffer[: newline + 1]
        if buffer:
            self._handle_line(bytes(buffer))

    def _handle_line(self, raw: bytes) -> None:
        line = raw.decode(errors="replace").strip()
        if not line:
            return
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Pi emitted non-JSON on stdout: {line[:200]!r}") from exc
        if not isinstance(event, dict):
            raise RuntimeError(f"Pi emitted non-object JSON on stdout: {line[:200]!r}")
        log_line = format_pi_event_for_log(event)
        if log_line is not None:
            logger.info("[%s] %s", self._log_label, log_line)
        apply_pi_event(self._state, event)

    def _build_turn_result(self, events: list[dict[str, Any]]) -> TurnResult:
        tool_calls_fired: list[str] = []
        terminal: tuple[str, str] | None = None
        for event in events:
            if event.get("type") == "tool_execution_start":
                name = str(event.get("toolName") or "")
                if name:
                    tool_calls_fired.append(name)
            if event.get("type") != "tool_execution_end":
                continue
            name = str(event.get("toolName") or "")
            if name not in self._tool_stop_map or bool(event.get("isError", False)):
                continue
            output = self._result_text(event.get("result"))
            success_output = self._tool_stop_map[name]
            if success_output is None or output.strip() == success_output:
                terminal = (name, output)
        if terminal is not None:
            self._terminal_tool = terminal
        return TurnResult(
            completion=self._state.final_text,
            terminal_tool_name=terminal[0] if terminal else None,
            terminal_tool_output=terminal[1] if terminal else None,
            tool_calls_fired=tool_calls_fired,
        )

    @staticmethod
    def _result_text(result: Any) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, dict) and isinstance(result.get("content"), list):
            return "".join(
                str(block.get("text") or "")
                for block in result["content"]
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return json.dumps(result) if result is not None else ""

    @property
    def result(self) -> AgentResult:
        return AgentResult(
            completion=self._state.final_text,
            trace=finalize_pi_trace(self._state),
            terminal_tool_name=self._terminal_tool[0] if self._terminal_tool else None,
            terminal_tool_output=self._terminal_tool[1] if self._terminal_tool else None,
        )

    async def aclose(self) -> None:
        return None


@asynccontextmanager
async def open_pi_harness_session(
    *,
    tools: list[Tool],
    system_prompt: str | None = None,
    provider: str = "openai-codex",
    model: str,
    thinking: str = "high",
    bin: str = "pi",
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    log_label: str | None = None,
    initial_events: list[dict[str, Any]] | None = None,
    initial_session_jsonl: str | None = None,
    session_id: str | None = None,
) -> AsyncIterator[PiHarnessSession]:
    """Open a Pi session with only the supplied pipeline tools enabled."""
    from nof1_causal_lab.utils.config import ensure_harness_prereqs

    ensure_harness_prereqs("pi")
    async with serve_pi_tools_http(tools) as bridge_url:
        with tempfile.TemporaryDirectory(prefix="pi-harness-") as tmpdir:
            scratch_dir = Path(tmpdir)
            (scratch_dir / "pipeline-tools.ts").write_text(build_pi_extension(tools, bridge_url))
            session = PiHarnessSession(
                tools=tools,
                scratch_dir=scratch_dir,
                system_prompt=system_prompt,
                provider=provider,
                model=model,
                thinking=thinking,
                bin=bin,
                timeout_seconds=timeout_seconds,
                log_label=log_label,
                initial_events=initial_events,
                initial_session_jsonl=initial_session_jsonl,
                session_id=session_id,
            )
            try:
                yield session
            finally:
                await session.aclose()
