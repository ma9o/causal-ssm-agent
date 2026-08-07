"""End-to-end tests for the in-process MCP server that wraps pipeline tools.

Spins the server up on a random localhost port, connects to it with the
MCP streamable-HTTP client, and verifies that (a) the tool list is the
expected shape and (b) tool calls dispatch to the underlying
``Tool.execute`` callable and return its result.
"""

from contextlib import asynccontextmanager

import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from nof1_causal_lab.utils.harness.mcp_server import serve_tools_http
from nof1_causal_lab.utils.harness.networking import find_free_port
from nof1_causal_lab.utils.openrouter_client import Tool
from tests.helpers import run_async as _run


@asynccontextmanager
async def _client(url):
    async with (
        streamablehttp_client(url) as (read, write, _sid),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        yield session


def _make_echo_tool() -> Tool:
    async def _execute(message: str) -> str:
        return f"echo: {message}"

    return Tool(
        name="echo",
        description="Echo the input message.",
        parameters={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
            "additionalProperties": False,
        },
        execute=_execute,
    )


def _make_failing_tool() -> Tool:
    async def _execute() -> str:
        raise RuntimeError("boom")

    return Tool(
        name="fail",
        description="Always raises.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        execute=_execute,
    )


class TestMCPServer:
    def test_initialize_post_hits_exact_url_without_redirect(self):
        """codex's MCP client POSTs initialize to the configured URL verbatim
        and treats a 307 (Starlette Mount slash-redirect) as a failed
        handshake; the SDK client used elsewhere follows redirects and would
        mask a regression."""
        import httpx

        init = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "probe", "version": "0"},
            },
        }

        async def scenario():
            async with (
                serve_tools_http([_make_echo_tool()]) as url,
                httpx.AsyncClient(follow_redirects=False) as client,
            ):
                return await client.post(
                    url,
                    json=init,
                    headers={"Accept": "application/json, text/event-stream"},
                )

        response = _run(scenario())
        assert response.status_code == 200

    def test_lists_registered_tools(self):
        async def scenario():
            async with serve_tools_http([_make_echo_tool()]) as url, _client(url) as session:
                return await session.list_tools()

        listing = _run(scenario())
        assert [t.name for t in listing.tools] == ["echo"]
        schema = listing.tools[0].inputSchema
        assert schema["properties"]["message"]["type"] == "string"
        assert schema["required"] == ["message"]

    def test_call_tool_dispatches_to_execute(self):
        async def scenario():
            async with serve_tools_http([_make_echo_tool()]) as url, _client(url) as session:
                return await session.call_tool("echo", arguments={"message": "hi"})

        result = _run(scenario())
        assert result.isError is False
        text_parts = [c.text for c in result.content if getattr(c, "type", None) == "text"]
        assert text_parts == ["echo: hi"]

    def test_unknown_tool_returns_error_text(self):
        async def scenario():
            async with serve_tools_http([_make_echo_tool()]) as url, _client(url) as session:
                return await session.call_tool("does_not_exist", arguments={})

        result = _run(scenario())
        text_parts = [c.text for c in result.content if getattr(c, "type", None) == "text"]
        assert any("Unknown tool" in t for t in text_parts)

    def test_tool_exception_is_reported(self):
        async def scenario():
            async with serve_tools_http([_make_failing_tool()]) as url, _client(url) as session:
                return await session.call_tool("fail", arguments={})

        result = _run(scenario())
        text_parts = [c.text for c in result.content if getattr(c, "type", None) == "text"]
        assert any("boom" in t for t in text_parts)

    def test_concurrent_servers_use_distinct_ports(self):
        async def scenario():
            async with (
                serve_tools_http([_make_echo_tool()]) as url_a,
                serve_tools_http([_make_echo_tool()]) as url_b,
            ):
                return url_a, url_b

        url_a, url_b = _run(scenario())
        assert url_a != url_b

    def test_find_free_port_returns_bindable_port(self):
        port = find_free_port()
        assert 1024 < port < 65536

    @pytest.mark.timeout(10)
    def test_server_shuts_down_cleanly(self):
        async def scenario():
            async with serve_tools_http([_make_echo_tool()]) as url:
                port = int(url.rsplit(":", 1)[1].split("/", 1)[0])
            # After exit the port should be reusable; bind to verify.
            import socket

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("127.0.0.1", port))
            return True

        assert _run(scenario()) is True
