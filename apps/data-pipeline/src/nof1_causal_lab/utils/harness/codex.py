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

Subscription auth (ChatGPT Plus/Pro/Team) lives in
``~/.codex/auth.json``. Because we override ``CODEX_HOME`` for the
MCP config, we also symlink the user's ``auth.json`` into the scratch
directory so subscription credentials survive the override. The
prerequisite validator refuses to start any Codex
stage when ``~/.codex/auth.json`` is missing.

Pipeline-fixed flags: ``--json`` for event streaming,
``--skip-git-repo-check`` so the pipeline can run outside a repo, and
``--dangerously-bypass-approvals-and-sandbox``. We still wrap the child
process in an OS-level sandbox (see ``build_codex_sandbox_profile``)
because codex-cli 0.123.0 still cancels localhost MCP HTTP tool calls in
non-interactive mode under stricter documented configs like
``sandbox_mode = "workspace-write"``,
``approval_policy = "never"``, and
``sandbox_workspace_write.network_access = true``.

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

from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001
from nof1_causal_lab.utils.agent_session import AgentResult, TurnResult
from nof1_causal_lab.utils.harness.mcp_server import serve_tools_http
from nof1_causal_lab.utils.harness.stream_json import (
    CodexStreamState,
    apply_codex_event,
    finalize_codex_trace,
    format_codex_event_for_log,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from nof1_causal_lab.utils.openrouter_client import Tool

logger = logging.getLogger(__name__)
# Stream events from the codex subprocess are at INFO level; make sure they
# clear the root logger's default WARNING threshold so they propagate up to
# the Prefect APILogHandler attached to ``nof1_causal_lab``.
logger.setLevel(logging.INFO)

MCP_SERVER_NAME = "pipeline-tools"
_DEFAULT_TIMEOUT_SECONDS = 1800


def build_codex_mcp_toml(
    url: str,
    server_name: str = MCP_SERVER_NAME,
    *,
    developer_instructions: str | None = None,
    trusted_project_paths: tuple[Path, ...] = (),
) -> str:
    """Render a ``config.toml`` snippet pointing Codex at our HTTP MCP server.

    - Uses the documented ``enabled_tools`` allowlist so Codex only sees
      the MCP tools the stage intentionally exposes.
    - Optionally marks the scratch working directory ``trust_level =
      "trusted"`` for Codex's project-trust machinery. This does not clear
      non-interactive localhost MCP approval failures in codex-cli 0.123.0,
      but it also does not broaden filesystem access beyond the external
      sandbox.
    - When provided, ``developer_instructions`` are added via Codex's
      config-level developer message so backend callers can preserve their
      stage-owned system prompt without replacing Codex's bundled base
      instructions.
    """
    lines: list[str] = []
    if developer_instructions is not None:
        lines.append(f"developer_instructions = {json.dumps(developer_instructions)}")
        lines.append("")
    lines.extend([f"[mcp_servers.{server_name}]", f'url = "{url}"'])
    for path in trusted_project_paths:
        lines.append("")
        # Path must be absolute for codex's project-key lookup.
        lines.append(f'[projects."{Path(path).resolve()}"]')
        lines.append('trust_level = "trusted"')
    return "\n".join(lines) + "\n"


def build_codex_argv(
    *,
    bin: str,
    user_message: str,
    thread_id: str | None,
    model: str,
    reasoning_effort: str | None,
    service_tier: str | None,
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
            # codex exec 0.123.0 still cancels localhost MCP HTTP tool
            # calls in non-interactive mode under the documented stricter
            # configs we tried, including:
            # - ``approval_policy="never"``
            # - ``--sandbox workspace-write``
            # - ``sandbox_workspace_write.network_access=true``
            # - ``features.shell_tool=false``
            # Project trust also did not clear the failure, and the
            # undocumented ``mcp_servers.<id>.tools.<tool>.approval_mode``
            # stanza is not surfaced by ``codex mcp get``. The only
            # configuration that actually lets the MCP calls through is
            # ``--dangerously-bypass-approvals-and-sandbox``. We pair it
            # with an OS-level sandbox-exec wrapper so codex still cannot
            # read or write outside its scratch CODEX_HOME.
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "-m",
            model,
        ]
    )
    if reasoning_effort is not None:
        argv.extend(["-c", f"model_reasoning_effort={reasoning_effort}"])
    if service_tier is not None:
        argv.extend(["-c", f"service_tier={json.dumps(service_tier)}"])
    # ``-C`` is only valid on ``codex exec``; the ``resume`` subcommand
    # inherits the cwd set on the original session and rejects it.
    if cwd is not None and thread_id is None:
        argv.extend(["-C", str(cwd)])
    for key, value in extra_config or []:
        argv.extend(["-c", f"{key}={value}"])
    argv.append(user_message)
    return argv


def build_codex_sandbox_profile(
    writable_root: Path,
    *,
    protected_paths: tuple[Path, ...] = (),
) -> str:
    """Render a macOS sandbox-exec (SBPL) profile for wrapping ``codex exec``.

    Because codex 0.123 requires ``--dangerously-bypass-approvals-and-sandbox``
    for non-interactive MCP tool calls, we apply the sandbox externally.
    The profile starts from the default-allow baseline (so codex can load
    system libraries, hit the network, and read public system files) and
    then layers on two narrowings:

    * writes are denied everywhere except ``writable_root`` and the usual
      tmp locations — codex cannot modify any file outside its scratch
      ``CODEX_HOME``;
    * every path in ``protected_paths`` is additionally denied for both
      read and write — typically the pipeline repo root, so codex cannot
      peek at project source, workspace run artifacts, or scratchpad
      notes either. Only our MCP tools can reveal pipeline state.
    """
    writable = str(Path(writable_root).resolve())
    lines: list[str] = [
        "(version 1)",
        "(allow default)",
        "",
        ";; Deny all writes by default; re-allow only the scratch + tmp roots.",
        "(deny file-write*)",
        "(allow file-write*",
        f'  (subpath "{writable}")',
        '  (subpath "/private/tmp")',
        '  (subpath "/private/var/folders")',
        '  (subpath "/tmp")',
        '  (subpath "/var/folders")',
        '  (literal "/dev/null")',
        '  (literal "/dev/dtracehelper")',
        '  (literal "/dev/tty"))',
    ]
    for path in protected_paths:
        resolved = str(Path(path).resolve())
        lines.extend(
            [
                "",
                f';; Block codex from reading or writing "{resolved}" and everything under it.',
                f'(deny file-read* (subpath "{resolved}"))',
                f'(deny file-write* (subpath "{resolved}"))',
            ]
        )
    # ``writable_root`` may sit under a protected_paths ancestor (e.g. TMPDIR
    # on macOS resolves inside ``/private/var/folders``); re-allow it last so
    # those denies cannot shadow codex's own scratch.
    lines.extend(
        [
            "",
            ";; Scratch CODEX_HOME stays readable/writable even if it nests under a protected path.",
            f'(allow file-read* (subpath "{writable}"))',
            f'(allow file-write* (subpath "{writable}"))',
        ]
    )
    return "\n".join(lines) + "\n"


def _detect_project_root(start: Path | None = None) -> Path | None:
    """Return the nearest ancestor of ``start`` containing a ``.git`` directory."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def build_sandboxed_argv(
    inner_argv: list[str],
    *,
    writable_root: Path,
    profile_path: Path,
    protected_paths: tuple[Path, ...] | None = None,
) -> list[str]:
    """Wrap ``inner_argv`` in a ``sandbox-exec -f <profile>`` invocation.

    ``protected_paths`` defaults to the detected git-repo root so codex
    cannot observe or mutate the surrounding project tree.
    """
    if protected_paths is None:
        detected = _detect_project_root()
        protected_paths = (detected,) if detected is not None else ()
    profile_path.write_text(
        build_codex_sandbox_profile(writable_root, protected_paths=protected_paths)
    )
    return ["/usr/bin/sandbox-exec", "-f", str(profile_path), *inner_argv]


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
        service_tier: str | None = None,
        cwd: str | Path | None = None,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        log_label: str | None = None,
        initial_events: list[UncheckedJsonObject] | None = None,
        turn_index: int = 0,
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
        self._service_tier = service_tier
        self._cwd = cwd
        self._timeout_seconds = timeout_seconds
        self._log_label = log_label

        self._state = CodexStreamState()
        for event in initial_events or []:
            apply_codex_event(self._state, event)
        self._turn_index = turn_index
        self._terminal_tool: tuple[str, str] | None = None

    @property
    def thread_id(self) -> str | None:
        return self._state.thread_id

    @property
    def raw_events(self) -> list[UncheckedJsonObject]:
        return list(self._state.raw_events)

    async def turn(self, user_message: str) -> TurnResult:
        self._turn_index += 1
        pre_event_count = len(self._state.raw_events)

        inner_argv = build_codex_argv(
            bin=self._bin,
            user_message=user_message,
            thread_id=self._state.thread_id if self._turn_index > 1 else None,
            model=self._model,
            reasoning_effort=self._reasoning_effort,
            service_tier=self._service_tier,
            cwd=self._cwd,
        )
        argv = build_sandboxed_argv(
            inner_argv,
            writable_root=self._codex_home,
            profile_path=self._codex_home / "sandbox.sb",
        )

        env = dict(os.environ)
        env["CODEX_HOME"] = str(self._codex_home)

        proc = await asyncio.create_subprocess_exec(
            *argv,
            # Codex reads additional instructions from stdin when stdin is
            # open; inheriting the parent's stdin (e.g. a Prefect worker
            # with an open pipe) hangs the subprocess on read() forever.
            # Close stdin immediately so codex uses the argv prompt only.
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            # Spawn in the scratch CODEX_HOME rather than the parent's cwd.
            # The sandbox profile denies reads under the project root, and
            # codex fails with "Operation not permitted" just stat'ing its
            # own cwd if that cwd happens to sit inside the denied tree
            # (e.g. the Prefect worker runs from apps/data-pipeline).
            cwd=str(self._codex_home),
        )

        stderr_bytes = bytearray()

        async def _drain_stderr() -> None:
            if proc.stderr is None:
                return
            while True:
                chunk = await proc.stderr.read(65536)
                if not chunk:
                    break
                stderr_bytes.extend(chunk)
                for line in chunk.split(b"\n"):
                    text = line.decode("utf-8", errors="replace").strip()
                    if text:
                        logger.info("[%s] codex stderr: %s", self._log_label, text)

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
            stderr_text = stderr_bytes.decode("utf-8", errors="replace")
            raise RuntimeError(f"codex exited with status {proc.returncode}: {stderr_text}")

        turn_events = self._state.raw_events[pre_event_count:]
        return self._build_turn_result(turn_events)

    async def _drain_stdout(self, proc: asyncio.subprocess.Process) -> None:
        if proc.stdout is None:
            return
        # Codex can emit single JSON events (e.g. large reasoning or agent
        # messages) that exceed asyncio's default 64 KiB readline limit.
        # Read raw bytes and split on newlines ourselves to accept arbitrarily
        # long lines without raising ``Separator is found, but chunk is
        # longer than limit``.
        buffer = bytearray()
        while True:
            chunk = await proc.stdout.read(65536)
            if not chunk:
                break
            buffer.extend(chunk)
            while True:
                newline_index = buffer.find(b"\n")
                if newline_index < 0:
                    break
                raw = bytes(buffer[:newline_index])
                del buffer[: newline_index + 1]
                self._handle_codex_line(raw)
        if buffer:
            self._handle_codex_line(bytes(buffer))

    def _handle_codex_line(self, raw: bytes) -> None:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            return
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"[{self._log_label}] codex emitted non-JSON on stdout: {line[:200]!r}"
            ) from exc
        if not isinstance(event, dict):
            raise RuntimeError(
                f"[{self._log_label}] codex emitted non-object JSON on stdout: {line[:200]!r}"
            )
        log_line = format_codex_event_for_log(event)
        if log_line is not None:
            logger.info("[%s] %s", self._log_label, log_line)
        apply_codex_event(self._state, event)

    def _build_turn_result(self, turn_events: list[UncheckedJsonObject]) -> TurnResult:
        tool_calls_fired: list[str] = []
        terminal: tuple[str, str] | None = None

        def _unwrap(event: UncheckedJsonObject) -> UncheckedJsonObject:
            # Codex 0.121 nests items inside `item.completed`; older/prototype
            # schemas put tool_call/tool_result at the top level.
            if event.get("type") == "item.completed" and isinstance(event.get("item"), dict):
                return event["item"]
            return event

        for raw in turn_events:
            event = _unwrap(raw)
            etype = event.get("type")
            if etype in {"tool_call", "mcp_tool_call"}:
                name = str(event.get("name") or event.get("tool") or "")
                if name:
                    tool_calls_fired.append(name)
            elif etype in {"tool_result", "mcp_tool_result"}:
                result_raw = event.get("output") or event.get("result") or event.get("text") or ""
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


def _link_codex_auth(codex_home: Path) -> None:
    """Copy ``~/.codex/auth.json`` into a scratch CODEX_HOME.

    Subscription auth (ChatGPT Plus/Pro/Team) lives in that file.
    Without this, setting CODEX_HOME to a fresh directory would drop
    the user out of their session. ``ensure_harness_prereqs`` ensures
    the file is present before any Codex stage runs.

    We copy (rather than symlink) so codex can refresh the access
    token in place: the copy sits inside the writable scratch subpath,
    while a symlink would resolve to ``~/.codex/auth.json`` outside the
    sandbox-exec write-allowlist and fail with ``Operation not permitted``.
    """
    user_auth = Path("~/.codex/auth.json").expanduser()
    if not user_auth.exists():
        return
    (codex_home / "auth.json").write_bytes(user_auth.read_bytes())


def _persist_codex_auth(codex_home: Path) -> None:
    """Write a rotated ``auth.json`` back to ``~/.codex``.

    Subscription refresh tokens rotate on use: after codex refreshes
    in-session, the scratch copy is the only holder of the live refresh
    token. Discarding it with the scratch dir strands the whole token
    family — the next session copies the consumed token and dies with
    "refresh token was already used", forcing an interactive re-login.
    """
    scratch_auth = codex_home / "auth.json"
    user_auth = Path("~/.codex/auth.json").expanduser()
    if not scratch_auth.exists() or not user_auth.exists():
        return
    rotated = scratch_auth.read_bytes()
    if rotated != user_auth.read_bytes():
        user_auth.write_bytes(rotated)


@asynccontextmanager
async def open_codex_harness_session(
    *,
    tools: list[Tool],
    system_prompt: str | None = None,
    model: str,
    bin: str = "codex",
    reasoning_effort: str | None = None,
    service_tier: str | None = None,
    cwd: str | Path | None = None,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    log_label: str | None = None,
    initial_events: list[UncheckedJsonObject] | None = None,
    turn_index: int = 0,
) -> AsyncIterator[CodexHarnessSession]:
    """Open a Codex-backed agent session scoped to an ``async with`` block.

    Starts an in-process MCP server for ``tools``, writes a
    per-invocation ``config.toml`` in a scratch ``CODEX_HOME`` (with
    ``auth.json`` symlinked from the user's ``~/.codex`` so subscription
    auth survives), then yields a :class:`CodexHarnessSession`. The
    scratch directory and MCP server are cleaned up on exit.
    """
    from nof1_causal_lab.utils.config import ensure_harness_prereqs

    ensure_harness_prereqs("codex")
    async with serve_tools_http(tools, name=MCP_SERVER_NAME) as mcp_url:
        with tempfile.TemporaryDirectory(prefix="codex-home-") as tmpdir:
            codex_home = Path(tmpdir)
            mcp_toml = build_codex_mcp_toml(
                mcp_url,
                developer_instructions=system_prompt,
                trusted_project_paths=(codex_home,),
            )
            (codex_home / "config.toml").write_text(mcp_toml)
            _link_codex_auth(codex_home)
            session = CodexHarnessSession(
                tools=tools,
                codex_home=codex_home,
                model=model,
                bin=bin,
                reasoning_effort=reasoning_effort,
                service_tier=service_tier,
                # Default the agent's working root to the scratch CODEX_HOME
                # so it has nothing to auto-explore (the real repo would
                # otherwise invite hours of autonomous shell commands). Our
                # MCP tools remain the only way to act on the prompt.
                cwd=cwd if cwd is not None else codex_home,
                timeout_seconds=timeout_seconds,
                log_label=log_label,
                initial_events=initial_events,
                turn_index=turn_index,
            )
            try:
                yield session
            finally:
                await session.aclose()
                _persist_codex_auth(codex_home)
