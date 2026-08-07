"""Integration tests for the tokenized Pi tool bridge."""

from __future__ import annotations

import socket

import httpx
import pytest

from nof1_causal_lab.utils.harness.pi_tool_bridge import serve_pi_tools_http
from nof1_causal_lab.utils.openrouter_client import Tool
from tests.helpers import run_async


def _echo_tool() -> Tool:
    async def execute(message: str) -> str:
        return f"echo: {message}"

    return Tool(
        name="echo",
        description="Echo one message.",
        parameters={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
            "additionalProperties": False,
        },
        execute=execute,
    )


def _failing_tool() -> Tool:
    async def execute() -> str:
        raise RuntimeError("pi boom")

    return Tool(
        name="fail",
        description="Always fail.",
        parameters={"type": "object", "properties": {}, "required": []},
        execute=execute,
    )


@pytest.mark.timeout(10)
def test_pi_bridge_dispatches_and_isolates_the_tokenized_route() -> None:
    async def scenario():
        async with (
            serve_pi_tools_http([_echo_tool()]) as url,
            httpx.AsyncClient() as client,
        ):
            response = await client.post(
                url,
                json={"name": "echo", "arguments": {"message": "hello"}},
            )
            hidden_route = await client.post(url.rsplit("/", 1)[0], json={})
            return response, hidden_route

    response, hidden_route = run_async(scenario())

    assert response.status_code == 200
    assert response.json() == {"output": "echo: hello", "is_error": False}
    assert hidden_route.status_code == 404


@pytest.mark.timeout(10)
def test_pi_bridge_returns_tool_failures_and_releases_its_port() -> None:
    async def scenario():
        async with (
            serve_pi_tools_http([_failing_tool()]) as url,
            httpx.AsyncClient() as client,
        ):
            port = int(url.split(":", 2)[2].split("/", 1)[0])
            response = await client.post(url, json={"name": "fail", "arguments": {}})
        return port, response

    port, response = run_async(scenario())

    assert response.status_code == 200
    assert response.json()["is_error"] is True
    assert "pi boom" in response.json()["output"]
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
