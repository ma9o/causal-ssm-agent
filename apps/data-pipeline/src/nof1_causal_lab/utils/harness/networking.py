"""Local networking helpers shared by harness transports."""

from __future__ import annotations

import asyncio
import contextlib
import socket
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

DEFAULT_SERVER_STARTUP_TIMEOUT_SECONDS = 10.0
DEFAULT_SERVER_SHUTDOWN_TIMEOUT_SECONDS = 5.0


class UvicornServer(Protocol):
    """Minimal lifecycle surface exposed by ``uvicorn.Server``."""

    started: bool
    should_exit: bool

    async def serve(self) -> None: ...


def find_free_port(host: str = "127.0.0.1") -> int:
    """Ask the OS for an unused TCP port on ``host``."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


@asynccontextmanager
async def run_uvicorn_server(
    server: UvicornServer,
    *,
    task_name: str,
    startup_error: str,
    startup_timeout_seconds: float = DEFAULT_SERVER_STARTUP_TIMEOUT_SECONDS,
    shutdown_timeout_seconds: float = DEFAULT_SERVER_SHUTDOWN_TIMEOUT_SECONDS,
) -> AsyncIterator[None]:
    """Run one Uvicorn server with bounded startup and shutdown."""
    task = asyncio.create_task(server.serve(), name=task_name)
    try:
        try:
            async with asyncio.timeout(startup_timeout_seconds):
                while not server.started and not task.done():
                    await asyncio.sleep(0.01)
        except TimeoutError as exc:
            raise RuntimeError(
                f"{startup_error}: startup timed out after {startup_timeout_seconds:g}s"
            ) from exc
        if task.done():
            task.result()
            raise RuntimeError(startup_error)
        yield
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=shutdown_timeout_seconds)
        except TimeoutError:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
