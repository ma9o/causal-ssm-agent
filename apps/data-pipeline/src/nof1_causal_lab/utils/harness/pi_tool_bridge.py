"""Narrow localhost bridge from Pi custom tools to pipeline ``Tool`` objects."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import secrets
import socket
import traceback
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import uvicorn

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from nof1_causal_lab.utils.openrouter_client import Tool

logger = logging.getLogger(__name__)


def _find_pi_bridge_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


async def _request_body(receive) -> bytes:
    chunks: list[bytes] = []
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            break
        if message["type"] != "http.request":
            continue
        chunks.append(message.get("body", b""))
        if not message.get("more_body", False):
            break
    return b"".join(chunks)


async def _send_json(send, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


@asynccontextmanager
async def serve_pi_tools_http(
    tools: list[Tool],
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
) -> AsyncIterator[str]:
    """Expose exactly ``tools`` through a tokenized request/response endpoint."""
    resolved_port = port if port is not None else _find_pi_bridge_port(host)
    token = secrets.token_urlsafe(32)
    path = f"/{token}"
    tool_map = {tool.name: tool for tool in tools}

    async def _asgi(scope, receive, send) -> None:
        if scope["type"] != "http" or scope.get("method") != "POST" or scope.get("path") != path:
            await _send_json(send, 404, {"error": "not found"})
            return
        try:
            request = json.loads((await _request_body(receive)).decode())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            await _send_json(send, 400, {"error": f"invalid request: {exc}"})
            return
        if not isinstance(request, dict):
            await _send_json(send, 400, {"error": "request must be an object"})
            return
        name = request.get("name")
        arguments = request.get("arguments") or {}
        tool = tool_map.get(name)
        if tool is None or not isinstance(arguments, dict):
            await _send_json(send, 400, {"error": "unknown tool or invalid arguments"})
            return
        logger.info("Pi tool call name=%s args_keys=%s", name, list(arguments))
        try:
            result = await tool.execute(**arguments)
        except Exception as exc:  # noqa: BLE001 - tool failures return to the model
            detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            output = f"Tool execution failed: {type(exc).__name__}: {exc}\n{detail}"
            logger.warning("Pi tool call %s raised %s: %s", name, type(exc).__name__, exc)
            await _send_json(send, 200, {"output": output, "is_error": True})
            return
        output = str(result)
        logger.info("Pi tool call %s -> %d chars", name, len(output))
        await _send_json(send, 200, {"output": output, "is_error": False})

    server = uvicorn.Server(
        uvicorn.Config(
            _asgi,
            host=host,
            port=resolved_port,
            log_level="warning",
            access_log=False,
            lifespan="off",
        )
    )
    task = asyncio.create_task(server.serve(), name=f"pi-tool-bridge-{resolved_port}")
    try:
        while not server.started and not task.done():
            await asyncio.sleep(0.01)
        if task.done():
            task.result()
            raise RuntimeError("Pi tool bridge exited during startup")
        yield f"http://{host}:{resolved_port}{path}"
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except TimeoutError:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
