"""Tests for bounded local harness server lifecycle support."""

from __future__ import annotations

import asyncio

import pytest

from nof1_causal_lab.utils.harness.networking import run_uvicorn_server
from tests.helpers import run_async


class _NeverStartingServer:
    started = False
    should_exit = False

    async def serve(self) -> None:
        while not self.should_exit:
            await asyncio.sleep(0)


def test_uvicorn_lifecycle_bounds_startup_and_stops_the_task() -> None:
    server = _NeverStartingServer()

    async def scenario() -> None:
        async with run_uvicorn_server(
            server,
            task_name="never-starts",
            startup_error="test server failed",
            startup_timeout_seconds=0.01,
            shutdown_timeout_seconds=0.1,
        ):
            pytest.fail("server context must not be entered")

    with pytest.raises(RuntimeError, match="startup timed out"):
        run_async(scenario())
    assert server.should_exit is True
