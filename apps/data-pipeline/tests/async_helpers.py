"""Async test helpers."""

import asyncio


def run_async(coro):
    """Run an async coroutine synchronously in tests."""
    return asyncio.run(coro)
