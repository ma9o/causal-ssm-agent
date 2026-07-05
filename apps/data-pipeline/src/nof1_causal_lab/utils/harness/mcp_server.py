"""In-process MCP server that exposes pipeline tools to an external harness.

The pipeline keeps its tool implementations as :class:`Tool` objects from
:mod:`nof1_causal_lab.utils.openrouter_client`. When a stage runs against
a harness backend, those same tool callables need to be reachable from a
subprocess (``claude -p`` / ``codex exec``) — Model Context Protocol
(MCP) provides the transport.

:func:`serve_tools_http` runs a streamable-HTTP MCP server on localhost
as an asyncio task inside the pipeline process. The yielded URL is passed
to the harness via ``--mcp-config``. On context-manager exit the server
shuts down and any in-flight sessions are cancelled.

Design notes:
- **Stateless sessions.** Each call from the harness opens a fresh MCP
  session; we do not retain per-session state. The tool handlers
  themselves close over pipeline-side state (same as the embedded path).
- **Random port by default.** Multiple concurrent harness invocations
  inside one pipeline process each get their own server on a free port.
- **Tool surface mirrors the embedded path.** The same ``Tool.execute``
  callables are invoked; we just wrap them in the MCP response shape.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
import traceback
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import mcp.types as mcp_types
import uvicorn
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from nof1_causal_lab.utils.openrouter_client import Tool

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _find_free_port(host: str = "127.0.0.1") -> int:
    """Ask the OS for an unused TCP port on ``host``."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def build_mcp_server(tools: list[Tool], *, name: str = "pipeline-tools") -> Server:
    """Build a low-level MCP :class:`Server` exposing the given tool list.

    ``Tool.parameters`` is already a JSON schema, so we pass it through
    as ``inputSchema``. ``Tool.execute`` is called directly; its return
    value is wrapped in a :class:`mcp_types.TextContent`.
    """
    server: Server = Server(name)
    tool_map = {t.name: t for t in tools}

    @server.list_tools()
    async def _list_tools() -> list[mcp_types.Tool]:
        return [
            mcp_types.Tool(
                name=t.name,
                description=t.description,
                inputSchema=t.parameters,
            )
            for t in tools
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict | None) -> list[mcp_types.TextContent]:
        logger.info("MCP call_tool name=%s args_keys=%s", name, list((arguments or {}).keys()))
        tool = tool_map.get(name)
        if tool is None:
            logger.warning("MCP call_tool unknown tool name=%s", name)
            return [mcp_types.TextContent(type="text", text=f"Unknown tool: {name}")]
        try:
            result = await tool.execute(**(arguments or {}))
        except Exception as exc:  # noqa: BLE001 — surface to harness as tool error
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            logger.warning("MCP call_tool %s raised %s: %s", name, type(exc).__name__, exc)
            return [
                mcp_types.TextContent(
                    type="text",
                    text=f"Tool execution failed: {type(exc).__name__}: {exc}\n{tb}",
                )
            ]
        text = str(result)
        logger.info("MCP call_tool %s -> %d chars", name, len(text))
        return [mcp_types.TextContent(type="text", text=text)]

    return server


@asynccontextmanager
async def serve_tools_http(
    tools: list[Tool],
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
    name: str = "pipeline-tools",
) -> AsyncIterator[str]:
    """Serve ``tools`` over streamable-HTTP MCP; yield the base URL.

    Example::

        async with serve_tools_http(my_tools) as url:
            # url == "http://127.0.0.1:54321/mcp"
            # pass url to the harness via --mcp-config
            ...

    On exit, the uvicorn task is cancelled and any in-flight sessions
    are torn down. The port is discovered automatically if not supplied.
    """
    resolved_port = port if port is not None else _find_free_port(host)
    server = build_mcp_server(tools, name=name)

    session_manager = StreamableHTTPSessionManager(
        app=server,
        stateless=True,
        json_response=False,
    )

    # Raw ASGI app, no router: a Starlette Mount("/mcp") answers POST /mcp
    # with a 307 to /mcp/, and codex's MCP client treats any non-2xx
    # initialize response as a failed handshake ("connection closed").
    # The server exists for exactly one client on an ephemeral port, so
    # every path dispatches straight to the session manager.
    async def _mcp_asgi(scope, receive, send) -> None:
        await session_manager.handle_request(scope, receive, send)

    config = uvicorn.Config(
        _mcp_asgi,
        host=host,
        port=resolved_port,
        log_level="warning",
        access_log=False,
    )
    uvicorn_server = uvicorn.Server(config)

    async with session_manager.run():
        serve_task = asyncio.create_task(uvicorn_server.serve(), name=f"mcp-server-{resolved_port}")
        try:
            # Wait for uvicorn to finish startup before yielding the URL.
            while not uvicorn_server.started and not serve_task.done():
                await asyncio.sleep(0.01)
            if serve_task.done():
                # Bubble up startup failure.
                serve_task.result()
                raise RuntimeError("MCP uvicorn server exited during startup")
            yield f"http://{host}:{resolved_port}/mcp"
        finally:
            uvicorn_server.should_exit = True
            try:
                await asyncio.wait_for(serve_task, timeout=5.0)
            except TimeoutError:
                serve_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await serve_task
